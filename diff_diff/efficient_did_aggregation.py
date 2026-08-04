"""
Aggregation mixin for the EfficientDiD estimator (Chen, Sant'Anna & Xie 2025).

Extracted from ``diff_diff/efficient_did.py`` with the M-023 post-fit
``aggregate()`` migration so the aggregation methods are importable by BOTH
the estimator (``efficient_did.py``) and the results module
(``efficient_did_results.py``) without an import cycle — ``efficient_did.py``
imports ``efficient_did_results.py``, so the results module can never import
the estimator.  This mirrors the CallawaySantAnna layout
(``staggered_aggregation.py``).

Contents:

- ``_cluster_aggregate`` / ``_compute_se_from_eif`` — module-level variance
  helpers (moved verbatim; ``efficient_did.py`` re-imports both).
- ``_EfficientAggregationMixin`` — the six estimator methods that compute
  the overall / event-study / group aggregations from the per-(g,t) EIF
  dict plus O(n_units) bookkeeping.  ``EfficientDiD`` inherits it for the
  fit-time path, and ``EfficientDiDResults._aggregate_compute`` runs the
  same methods post-fit on a throwaway ``_EDiDKitAggregator`` host built
  from the retained :class:`~diff_diff.aggregation.AggregationKit` — which
  is what keeps ``aggregate()`` off an ``_estimator_ref``.

The numerical content of every function in this module is byte-identical to
its pre-extraction form, with ONE additive exception recorded in the M-023
ledger notes: ``_aggregate_by_group`` records a per-row ``df_used`` key (the
``self._survey_df`` value at that row's ``safe_inference`` call) so the
post-fit group relay can publish exact per-row df provenance.
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from diff_diff.utils import safe_inference

if TYPE_CHECKING:
    from diff_diff.survey import ResolvedSurveyDesign


def _cluster_aggregate(
    eif_mat: np.ndarray,
    cluster_indices: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Sum EIF values within clusters and center.

    Parameters
    ----------
    eif_mat : ndarray, shape (n_units,) or (n_units, k)
        EIF values — 1-D for a single estimand, 2-D for multiple.
    cluster_indices : ndarray, shape (n_units,)
        Integer cluster assignment per unit.
    n_clusters : int
        Number of unique clusters.

    Returns
    -------
    ndarray, shape (n_clusters,) or (n_clusters, k)
        Centered cluster-level sums.
    """
    if eif_mat.ndim == 1:
        sums = np.bincount(cluster_indices, weights=eif_mat, minlength=n_clusters).astype(float)
    else:
        sums = np.column_stack(
            [
                np.bincount(cluster_indices, weights=eif_mat[:, j], minlength=n_clusters)
                for j in range(eif_mat.shape[1])
            ]
        ).astype(float)
    return sums - sums.mean(axis=0)


def _compute_se_from_eif(
    eif: np.ndarray,
    n_units: int,
    cluster_indices: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
) -> float:
    """SE from EIF values, optionally with cluster-robust correction.

    Without clusters: ``sqrt(mean(EIF^2) / n)``.
    With clusters: Liang-Zeger sandwich — aggregate EIF within clusters,
    center, and apply G/(G-1) small-sample correction.
    """
    if cluster_indices is not None and n_clusters is not None:
        centered = _cluster_aggregate(eif, cluster_indices, n_clusters)
        correction = n_clusters / (n_clusters - 1) if n_clusters > 1 else 1.0
        var = correction * np.sum(centered**2) / (n_units**2)
        return float(np.sqrt(max(var, 0.0)))
    return float(np.sqrt(np.mean(eif**2) / n_units))


class _EfficientAggregationMixin:
    """EIF-based aggregation methods shared by fit-time and post-fit paths.

    Not intended for standalone use.  A host class must expose exactly the
    five attributes declared below — this is the ``_EDiDKitAggregator``
    contract (``efficient_did_results.py``): the post-fit path constructs a
    fresh throwaway host per ``aggregate()`` call, so the ONE mutation these
    methods perform (``_compute_survey_eif_se`` writes ``self._survey_df``
    when a degenerate replicate design drops replicates) lands on the
    throwaway, never on the retained kit — preserving the aggregate()
    immutability contract.
    """

    # Typed host-attribute contract (mypy attr-defined; the
    # staggered_aggregation.py precedent).  Values are read-only here
    # except _survey_df (see class docstring).
    alpha: float
    anticipation: int
    _survey_df: Optional[float]
    _unit_resolved_survey: Optional["ResolvedSurveyDesign"]
    _unit_level_weights: Optional[np.ndarray]

    # -- Survey SE helpers ----------------------------------------------------

    def _compute_survey_eif_se(self, eif_vals: np.ndarray) -> float:
        """Compute SE from EIF scores using Taylor Series Linearization.

        Uses the pre-built unit-level ``_unit_resolved_survey`` constructed
        once in ``fit()`` (or carried by the post-fit aggregation kit),
        ensuring consistent unit-level arrays and avoiding repeated
        subsetting of panel-level survey data.
        """
        # Built once in fit() before any call lands here (see docstring).
        assert self._unit_resolved_survey is not None
        if self._unit_resolved_survey.uses_replicate_variance:
            from diff_diff.survey import compute_replicate_if_variance

            # Score-scale IFs to match TSL bread: psi = w * eif / sum(w)
            w = self._unit_resolved_survey.weights
            psi_scaled = w * eif_vals / w.sum()
            variance, n_valid = compute_replicate_if_variance(
                psi_scaled, self._unit_resolved_survey
            )
            # Update survey df to reflect effective replicate count
            if n_valid < self._unit_resolved_survey.n_replicates:
                self._survey_df = n_valid - 1 if n_valid > 1 else None
            return float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan

        from diff_diff.survey import compute_survey_vcov

        X_ones = np.ones((len(eif_vals), 1))
        vcov = compute_survey_vcov(X_ones, eif_vals, self._unit_resolved_survey)
        return float(np.sqrt(np.abs(vcov[0, 0])))

    def _eif_se(
        self,
        eif_vals: np.ndarray,
        n_units: int,
        cluster_indices: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
    ) -> float:
        """Compute SE from aggregated EIF scores.

        Dispatches to survey TSL when ``_unit_resolved_survey`` is set
        (during fit, or via the post-fit kit), otherwise uses
        cluster-robust or standard formula.
        """
        if self._unit_resolved_survey is not None:
            return self._compute_survey_eif_se(eif_vals)
        return _compute_se_from_eif(eif_vals, n_units, cluster_indices, n_clusters)

    # -- Aggregation helpers --------------------------------------------------

    def _compute_wif_contribution(
        self,
        keepers: List[Tuple],
        effects: np.ndarray,
        unit_cohorts: np.ndarray,
        cohort_fractions: Dict[float, float],
        n_units: int,
        unit_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute weight influence function correction (O(1) scale, matching EIF).

        This accounts for uncertainty in cohort-size aggregation weights.
        Matches R's ``did`` package WIF formula (staggered_aggregation.py:282-309),
        adapted to EDiD's EIF scale.

        Parameters
        ----------
        keepers : list of (g, t) tuples
            Post-treatment group-time pairs included in aggregation.
        effects : ndarray, shape (n_keepers,)
            ATT estimates for each keeper.
        unit_cohorts : ndarray, shape (n_units,)
            Cohort assignment for each unit (0 = never-treated).
        cohort_fractions : dict
            ``{cohort: n_cohort / n}`` for each cohort.
        n_units : int
            Total number of units.
        unit_weights : ndarray, shape (n_units,), optional
            Survey weights at the unit level.  When provided, uses the
            survey-weighted WIF formula: IF_i(p_g) = (w_i * 1{G_i=g} - pg_k).

        Returns
        -------
        ndarray, shape (n_units,)
            WIF contribution at O(1) scale, additive with ``agg_eif``.
        """
        groups_for_keepers = np.array([g for (g, t) in keepers])
        pg_keepers = np.array([cohort_fractions.get(g, 0.0) for g, t in keepers])
        sum_pg = pg_keepers.sum()
        if sum_pg == 0:
            return np.zeros(n_units)

        indicator = (unit_cohorts[:, None] == groups_for_keepers[None, :]).astype(float)

        if unit_weights is not None:
            # Survey-weighted WIF (matches staggered_aggregation.py:392-401):
            # IF_i(p_g) = (w_i * 1{G_i=g} - pg_k), NOT (1{G_i=g} - pg_k)
            weighted_indicator = indicator * unit_weights[:, None]
            indicator_diff = weighted_indicator - pg_keepers
            indicator_sum = np.sum(indicator_diff, axis=1)
        else:
            indicator_diff = indicator - pg_keepers
            indicator_sum = np.sum(indicator_diff, axis=1)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if1 = indicator_diff / sum_pg
            if2 = np.outer(indicator_sum, pg_keepers) / sum_pg**2
            wif_matrix = if1 - if2
            wif_contrib = wif_matrix @ effects
        return wif_contrib  # O(1) scale, same as agg_eif

    def _aggregate_overall(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray],
        n_units: int,
        cohort_fractions: Dict[float, float],
        unit_cohorts: np.ndarray,
        cluster_indices: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute overall ATT with WIF-adjusted SE.

        Parameters
        ----------
        group_time_effects : dict
            Group-time ATT estimates.
        eif_by_gt : dict
            Per-unit EIF values for each (g, t).
        n_units : int
            Total number of units.
        cohort_fractions : dict
            Cohort size fractions.
        unit_cohorts : ndarray, shape (n_units,)
            Cohort assignment for each unit.
        """
        # Filter to post-treatment effects
        keepers = [
            (g, t)
            for (g, t) in group_time_effects
            if t >= g - self.anticipation and np.isfinite(group_time_effects[(g, t)]["effect"])
        ]
        if not keepers:
            return np.nan, np.nan

        # Cohort-size weights
        pg = np.array([cohort_fractions.get(g, 0.0) for (g, _) in keepers])
        total_pg = pg.sum()
        if total_pg == 0:
            return np.nan, np.nan
        w = pg / total_pg

        effects = np.array([group_time_effects[gt]["effect"] for gt in keepers])
        overall_att = float(np.sum(w * effects))

        # Aggregate EIF
        agg_eif = np.zeros(n_units)
        for k, gt in enumerate(keepers):
            agg_eif += w[k] * eif_by_gt[gt]

        # WIF correction: accounts for uncertainty in cohort-size weights
        wif = self._compute_wif_contribution(
            keepers,
            effects,
            unit_cohorts,
            cohort_fractions,
            n_units,
            unit_weights=self._unit_level_weights,
        )
        # Compute SE: survey path uses score-level psi to avoid double-weighting
        # (compute_survey_vcov applies w_i internally, which would double-weight
        # the survey-weighted WIF term). Dispatch replicate vs TSL.
        if self._unit_resolved_survey is not None:
            uw = self._unit_level_weights
            # Set together with _unit_resolved_survey in fit().
            assert uw is not None
            total_w = float(np.sum(uw))
            psi_total = uw * agg_eif / total_w + wif / total_w

            if (
                hasattr(self._unit_resolved_survey, "uses_replicate_variance")
                and self._unit_resolved_survey.uses_replicate_variance
            ):
                from diff_diff.survey import compute_replicate_if_variance

                variance, _ = compute_replicate_if_variance(psi_total, self._unit_resolved_survey)
            else:
                from diff_diff.survey import compute_survey_if_variance

                variance = compute_survey_if_variance(psi_total, self._unit_resolved_survey)
            se = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan
        else:
            agg_eif_total = agg_eif + wif
            se = self._eif_se(agg_eif_total, n_units, cluster_indices, n_clusters)

        return overall_att, se

    def _aggregate_event_study(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray],
        n_units: int,
        cohort_fractions: Dict[float, float],
        treatment_groups: List[Any],
        time_periods: List[Any],
        balance_e: Optional[int] = None,
        unit_cohorts: Optional[np.ndarray] = None,
        cluster_indices: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate ATT(g,t) by relative time e = t - g.

        Parameters
        ----------
        group_time_effects : dict
            Group-time ATT estimates.
        eif_by_gt : dict
            Per-unit EIF values for each (g, t).
        n_units : int
            Total number of units.
        cohort_fractions : dict
            Cohort size fractions.
        treatment_groups : list
            Treatment cohort identifiers.
        time_periods : list
            All time periods.
        balance_e : int, optional
            Balance event study at this relative period.
        unit_cohorts : ndarray, optional
            Cohort assignment for each unit (for WIF correction).
        """
        # Organize by relative time
        effects_by_e: Dict[int, List[Tuple[Tuple[Any, Any], float, float]]] = {}
        for (g, t), data in group_time_effects.items():
            if not np.isfinite(data["effect"]):
                continue
            e = int(t - g)
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append(((g, t), data["effect"], cohort_fractions.get(g, 0.0)))

        # Balance if requested
        if balance_e is not None:
            groups_at_e = {gt[0] for gt, _, _ in effects_by_e.get(balance_e, [])}
            balanced: Dict[int, List[Tuple[Tuple[Any, Any], float, float]]] = {}
            for (g, t), data in group_time_effects.items():
                if not np.isfinite(data["effect"]):
                    continue
                if g in groups_at_e:
                    e = int(t - g)
                    if e not in balanced:
                        balanced[e] = []
                    balanced[e].append(((g, t), data["effect"], cohort_fractions.get(g, 0.0)))
            effects_by_e = balanced

        if balance_e is not None and not effects_by_e:
            warnings.warn(
                f"balance_e={balance_e}: no cohort has a finite effect at the "
                "anchor horizon. Event study will be empty.",
                UserWarning,
                stacklevel=2,
            )

        result: Dict[int, Dict[str, Any]] = {}
        for e, elist in sorted(effects_by_e.items()):
            gt_pairs = [x[0] for x in elist]
            effs = np.array([x[1] for x in elist])
            pgs = np.array([x[2] for x in elist])
            total_pg = pgs.sum()
            w = pgs / total_pg if total_pg > 0 else np.ones(len(pgs)) / len(pgs)

            agg_eff = float(np.sum(w * effs))

            # Aggregate EIF
            agg_eif = np.zeros(n_units)
            for k, gt in enumerate(gt_pairs):
                agg_eif += w[k] * eif_by_gt[gt]

            # WIF correction for event-study aggregation
            wif_e = np.zeros(n_units)
            if unit_cohorts is not None:
                es_keepers = [(g, t) for (g, t) in gt_pairs]
                es_effects = effs
                wif_e = self._compute_wif_contribution(
                    es_keepers,
                    es_effects,
                    unit_cohorts,
                    cohort_fractions,
                    n_units,
                    unit_weights=self._unit_level_weights,
                )

            if self._unit_resolved_survey is not None:
                uw = self._unit_level_weights
                # Set together with _unit_resolved_survey in fit().
                assert uw is not None
                total_w = float(np.sum(uw))
                psi_total = uw * agg_eif / total_w + wif_e / total_w

                if (
                    hasattr(self._unit_resolved_survey, "uses_replicate_variance")
                    and self._unit_resolved_survey.uses_replicate_variance
                ):
                    from diff_diff.survey import compute_replicate_if_variance

                    variance, _ = compute_replicate_if_variance(
                        psi_total, self._unit_resolved_survey
                    )
                else:
                    from diff_diff.survey import compute_survey_if_variance

                    variance = compute_survey_if_variance(psi_total, self._unit_resolved_survey)
                agg_se = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan
            else:
                agg_eif = agg_eif + wif_e
                agg_se = self._eif_se(agg_eif, n_units, cluster_indices, n_clusters)

            t_stat, p_val, ci = safe_inference(
                agg_eff, agg_se, alpha=self.alpha, df=self._survey_df
            )
            result[e] = {
                "effect": agg_eff,
                "se": agg_se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_groups": len(elist),
            }

        return result

    def _aggregate_by_group(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray],
        n_units: int,
        cohort_fractions: Dict[float, float],
        treatment_groups: List[Any],
        unit_cohorts: Optional[np.ndarray] = None,
        cluster_indices: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        """Aggregate ATT(g,t) by treatment cohort.

        Parameters
        ----------
        group_time_effects : dict
            Group-time ATT estimates.
        eif_by_gt : dict
            Per-unit EIF values for each (g, t).
        n_units : int
            Total number of units.
        cohort_fractions : dict
            Cohort size fractions.
        treatment_groups : list
            Treatment cohort identifiers.
        unit_cohorts : ndarray, optional
            Cohort assignment for each unit (unused — group aggregation
            uses equal weights, not cohort-size weights).

        Notes
        -----
        Each row dict records ``df_used`` — the ``self._survey_df`` value at
        that row's ``safe_inference`` call (M-023: exact per-row df
        provenance for the post-fit group relay).  In every constructible
        fit all rows share one value (replicate tightening completes during
        the per-cell estimation loop), but capture-at-use is exact by
        construction and robust to any future path that could diverge.
        The key is additive to the public row-dict schema.
        """
        result: Dict[Any, Dict[str, Any]] = {}
        for g in treatment_groups:
            g_gts = [
                (gg, t)
                for (gg, t) in group_time_effects
                if gg == g
                and t >= g - self.anticipation
                and np.isfinite(group_time_effects[(gg, t)]["effect"])
            ]
            if not g_gts:
                continue

            effs = np.array([group_time_effects[gt]["effect"] for gt in g_gts])
            w = np.ones(len(effs)) / len(effs)
            agg_eff = float(np.sum(w * effs))

            agg_eif = np.zeros(n_units)
            for k, gt in enumerate(g_gts):
                agg_eif += w[k] * eif_by_gt[gt]
            agg_se = self._eif_se(agg_eif, n_units, cluster_indices, n_clusters)

            df_used = self._survey_df
            t_stat, p_val, ci = safe_inference(agg_eff, agg_se, alpha=self.alpha, df=df_used)
            result[g] = {
                "effect": agg_eff,
                "se": agg_se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_periods": len(g_gts),
                "df_used": df_used,
            }

        return result
