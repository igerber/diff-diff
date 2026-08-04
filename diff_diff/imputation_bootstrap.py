"""
Bootstrap inference methods for the Imputation DiD estimator.

This module contains ImputationDiDBootstrapMixin, which provides multiplier
bootstrap inference. Extracted from imputation.py for module size management.
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats as _compute_effect_bootstrap_stats,
)
from diff_diff.bootstrap_utils import (
    generate_bootstrap_weights_batch as _generate_bootstrap_weights_batch,
)
from diff_diff.bootstrap_utils import (
    generate_survey_multiplier_weights_batch as _generate_survey_multiplier_weights_batch,
)
from diff_diff.imputation_aggregation import _compute_target_weights  # noqa: F401 (shared helper)
from diff_diff.imputation_results import ImputationBootstrapResults

__all__ = [
    "ImputationDiDBootstrapMixin",
]


class ImputationDiDBootstrapMixin:
    """Mixin providing bootstrap inference methods for ImputationDiD."""

    # Type hints for attributes accessed from the main class
    n_bootstrap: int
    bootstrap_weights: str
    alpha: float
    seed: Optional[int]
    anticipation: int
    horizon_max: Optional[int]

    if TYPE_CHECKING:

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
            proj_cache: Optional[Dict[Any, Any]] = None,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

        @staticmethod
        def _build_cohort_rel_times(
            df: pd.DataFrame,
            first_treat: str,
        ) -> Dict[Any, Set[int]]: ...

        @staticmethod
        def _compute_balanced_cohort_mask(
            df_treated: pd.DataFrame,
            first_treat: str,
            all_horizons: List[int],
            balance_e: int,
            cohort_rel_times: Dict[Any, Set[int]],
        ) -> np.ndarray: ...

    def _precompute_bootstrap_psi(
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
        kept_cov_mask: Optional[np.ndarray],
        overall_weights: np.ndarray,
        event_study_effects: Optional[Dict[int, Dict[str, Any]]],
        group_effects: Optional[Dict[Any, Dict[str, Any]]],
        treatment_groups: List[Any],
        tau_hat: np.ndarray,
        balance_e: Optional[int],
        survey_weights_0: Optional[np.ndarray] = None,
        survey_weights_1: Optional[np.ndarray] = None,
        proj_cache: Optional[Dict[Any, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Pre-compute cluster-level influence function sums for each bootstrap target.

        For each aggregation target (overall, per-horizon, per-group), computes
        psi_i = sum_t v_it * epsilon_tilde_it for each cluster. The multiplier
        bootstrap then perturbs these psi sums with multiplier weights
        (rademacher/mammen/webb; configurable via ``bootstrap_weights``).

        The per-target v_untreated computation reuses the cached, target-invariant
        untreated-projection factorization via ``proj_cache`` (shared with the
        analytical path of the same ``fit()``), so the design + factorization is
        built once rather than once per target.
        """
        result: Dict[str, Any] = {}

        common = dict(
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
            survey_weights_0=survey_weights_0,
            proj_cache=proj_cache,
        )

        # Overall ATT
        overall_psi, cluster_ids, _ = self._compute_cluster_psi_sums(
            **common, weights=overall_weights
        )
        result["overall"] = (overall_psi, cluster_ids)

        # Event study: per-horizon weights
        if event_study_effects:
            result["event_study"] = {}
            df_1 = df.loc[omega_1_mask]
            rel_times = df_1["_rel_time"].values

            all_horizons = sorted(set(int(h) for h in rel_times if np.isfinite(h)))
            if self.horizon_max is not None:
                all_horizons = [h for h in all_horizons if abs(h) <= self.horizon_max]

            # Balanced cohort mask (same logic as _aggregate_event_study)
            balanced_mask = None
            if balance_e is not None:
                cohort_rel_times = self._build_cohort_rel_times(df, first_treat)
                balanced_mask = self._compute_balanced_cohort_mask(
                    df_1, first_treat, all_horizons, balance_e, cohort_rel_times
                )

            ref_period = -1 - self.anticipation

            for h in event_study_effects:
                if event_study_effects[h].get("n_obs", 0) == 0:
                    continue
                if h == ref_period:
                    continue
                if not np.isfinite(event_study_effects[h].get("effect", np.nan)):
                    continue

                # Skip pre-period horizons — their SEs come from Test 1
                # lead regression, not bootstrap
                if h < -self.anticipation:
                    continue

                h_mask = rel_times == h
                if balanced_mask is not None:
                    h_mask = h_mask & balanced_mask

                # When survey weights are provided, build weights proportional
                # to treated-observation survey weights (matching the analytical
                # path in _aggregate_event_study).  Otherwise use equal weights.
                if survey_weights_1 is not None:
                    finite_target = np.isfinite(tau_hat) & h_mask
                    n_valid_h = int(finite_target.sum())
                    if n_valid_h == 0:
                        continue
                    treated_sw = survey_weights_1
                    sw_h = treated_sw[h_mask]
                    finite_in_h = np.isfinite(tau_hat[h_mask])
                    sw_finite = sw_h[finite_in_h]
                    weights_h = np.zeros(len(tau_hat))
                    if sw_finite.sum() > 0:
                        h_indices = np.where(h_mask)[0]
                        finite_indices = h_indices[finite_in_h]
                        weights_h[finite_indices] = sw_finite / sw_finite.sum()
                else:
                    weights_h, n_valid_h = _compute_target_weights(tau_hat, h_mask)
                    if n_valid_h == 0:
                        continue

                psi_h, _, _ = self._compute_cluster_psi_sums(**common, weights=weights_h)
                result["event_study"][h] = psi_h

        # Group effects: per-group weights
        if group_effects:
            result["group"] = {}
            df_1 = df.loc[omega_1_mask]
            cohorts = df_1[first_treat].values

            for g in group_effects:
                if group_effects[g].get("n_obs", 0) == 0:
                    continue
                if not np.isfinite(group_effects[g].get("effect", np.nan)):
                    continue
                g_mask = cohorts == g

                # When survey weights are provided, build weights proportional
                # to treated-observation survey weights (matching the analytical
                # path in _aggregate_group).  Otherwise use equal weights.
                if survey_weights_1 is not None:
                    finite_target = np.isfinite(tau_hat) & g_mask
                    n_valid_g = int(finite_target.sum())
                    if n_valid_g == 0:
                        continue
                    treated_sw = survey_weights_1
                    sw_g = treated_sw[g_mask]
                    finite_in_g = np.isfinite(tau_hat[g_mask])
                    sw_finite = sw_g[finite_in_g]
                    weights_g = np.zeros(len(tau_hat))
                    if sw_finite.sum() > 0:
                        g_indices = np.where(g_mask)[0]
                        finite_indices = g_indices[finite_in_g]
                        weights_g[finite_indices] = sw_finite / sw_finite.sum()
                else:
                    weights_g, n_valid_g = _compute_target_weights(tau_hat, g_mask)
                    if n_valid_g == 0:
                        continue

                psi_g, _, _ = self._compute_cluster_psi_sums(**common, weights=weights_g)
                result["group"][g] = psi_g

        return result

    def _build_nan_bootstrap_results(
        self,
        original_event_study: Optional[Dict[int, Dict[str, Any]]],
        original_group: Optional[Dict[Any, Dict[str, Any]]],
    ) -> ImputationBootstrapResults:
        """Build an all-NaN ImputationBootstrapResults for degenerate-design
        bootstrap paths (n_clusters<2 or n_psu<2).

        Per-horizon and per-group dicts are populated with NaN entries keyed
        by the same horizons/groups as the analytical originals so that the
        downstream post-bootstrap override loop in :meth:`ImputationDiD.fit`
        iterates over them and propagates NaN to ``event_study_effects[h]["se"]``
        / ``group_effects[g]["se"]`` (rather than silently no-oping by
        finding ``None``).
        """
        n_nan = float("nan")
        ci_nan: Tuple[float, float] = (n_nan, n_nan)

        es_ses: Optional[Dict[int, float]] = None
        es_cis: Optional[Dict[int, Tuple[float, float]]] = None
        es_ps: Optional[Dict[int, float]] = None
        if original_event_study:
            es_ses = {h: n_nan for h in original_event_study}
            es_cis = {h: ci_nan for h in original_event_study}
            es_ps = {h: n_nan for h in original_event_study}

        g_ses: Optional[Dict[Any, float]] = None
        g_cis: Optional[Dict[Any, Tuple[float, float]]] = None
        g_ps: Optional[Dict[Any, float]] = None
        if original_group:
            g_ses = {g: n_nan for g in original_group}
            g_cis = {g: ci_nan for g in original_group}
            g_ps = {g: n_nan for g in original_group}

        return ImputationBootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weights,
            alpha=self.alpha,
            overall_att_se=n_nan,
            overall_att_ci=ci_nan,
            overall_att_p_value=n_nan,
            event_study_ses=es_ses,
            event_study_cis=es_cis,
            event_study_p_values=es_ps,
            group_ses=g_ses,
            group_cis=g_cis,
            group_p_values=g_ps,
            bootstrap_distribution=None,
        )

    def _run_bootstrap(
        self,
        original_att: float,
        original_event_study: Optional[Dict[int, Dict[str, Any]]],
        original_group: Optional[Dict[Any, Dict[str, Any]]],
        psi_data: Dict[str, Any],
        resolved_survey: Optional[Any] = None,
    ) -> ImputationBootstrapResults:
        """
        Run multiplier bootstrap on pre-computed influence function sums.

        Uses T_b = sum_i w_b_i * psi_i where w_b_i are multiplier weights
        (rademacher/mammen/webb; configurable via ``bootstrap_weights``)
        and psi_i are cluster-level influence function sums from Theorem 3.
        SE = std(T_b, ddof=1).

        When ``resolved_survey`` carries PSU/strata/FPC structure, weights are
        generated via ``generate_survey_multiplier_weights_batch`` so the
        bootstrap variance respects the survey design (stratification and FPC
        scaling).
        """
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        overall_psi, cluster_ids = psi_data["overall"]
        n_clusters = len(cluster_ids)

        # Determine whether to use survey-aware bootstrap weights
        _use_survey_bootstrap = resolved_survey is not None and (
            resolved_survey.strata is not None
            or resolved_survey.psu is not None
            or resolved_survey.fpc is not None
        )

        # Fail-closed when the analytical-cluster path has fewer than 2
        # independent clusters. Without this guard, the multiplier bootstrap
        # SE collapses to ~0 from BLAS roundoff (NOT NaN), and downstream
        # zero-SE checks miss the degenerate-design case.
        if not _use_survey_bootstrap and n_clusters < 2:
            warnings.warn(
                f"Bootstrap with n_clusters={n_clusters} (<2 independent "
                "clusters) produces degenerate variance; returning NaN SE.",
                UserWarning,
                stacklevel=3,
            )
            return self._build_nan_bootstrap_results(original_event_study, original_group)

        # Generate ALL weights upfront: shape (n_bootstrap, n_clusters)
        if _use_survey_bootstrap:
            psu_weights, psu_ids = _generate_survey_multiplier_weights_batch(
                self.n_bootstrap, resolved_survey, self.bootstrap_weights, rng
            )
            # Fail-closed when the survey-PSU path has fewer than 2 PSUs.
            # Same BLAS-roundoff failure mode as the non-survey cluster path
            # above; same NaN-propagation contract.
            if len(psu_ids) < 2:
                warnings.warn(
                    f"Survey-PSU bootstrap with n_psu={len(psu_ids)} (<2 "
                    "independent PSUs) produces degenerate variance from "
                    "BLAS roundoff; returning NaN SE.",
                    UserWarning,
                    stacklevel=3,
                )
                return self._build_nan_bootstrap_results(original_event_study, original_group)
            # Reindex PSU weights to match cluster_ids ordering.
            # cluster_ids are unique PSU values from _compute_cluster_psi_sums;
            # psu_ids are unique PSU values from the survey weight generator.
            # Build a map from psu_id -> column index in psu_weights.
            psu_id_to_col = {int(p): c for c, p in enumerate(psu_ids)}
            cluster_to_psu_col = np.array([psu_id_to_col[int(cid)] for cid in cluster_ids])
            all_weights = psu_weights[:, cluster_to_psu_col]
        else:
            all_weights = _generate_bootstrap_weights_batch(
                self.n_bootstrap, n_clusters, self.bootstrap_weights, rng
            )

        # Overall ATT bootstrap draws
        boot_overall = np.dot(all_weights, overall_psi)  # (n_bootstrap,)

        # Event study: loop over horizons
        boot_event_study: Optional[Dict[int, np.ndarray]] = None
        if original_event_study and "event_study" in psi_data:
            boot_event_study = {}
            for h, psi_h in psi_data["event_study"].items():
                boot_event_study[h] = np.dot(all_weights, psi_h)

        # Group effects: loop over groups
        boot_group: Optional[Dict[Any, np.ndarray]] = None
        if original_group and "group" in psi_data:
            boot_group = {}
            for g, psi_g in psi_data["group"].items():
                boot_group[g] = np.dot(all_weights, psi_g)

        # --- Inference (percentile bootstrap, matching CS/SA convention) ---
        # Shift perturbation-centered draws to effect-centered draws.
        # The multiplier bootstrap produces T_b = sum w_b_i * psi_i centered at 0.
        # CS adds the original effect back (L411 of staggered_bootstrap.py).
        # We do the same here so percentile CIs and empirical p-values work correctly.
        boot_overall_shifted = boot_overall + original_att

        overall_se, overall_ci, overall_p = _compute_effect_bootstrap_stats(
            original_att,
            boot_overall_shifted,
            alpha=self.alpha,
            context="ImputationDiD overall ATT",
        )

        event_study_ses = None
        event_study_cis = None
        event_study_p_values = None
        if boot_event_study and original_event_study:
            event_study_ses = {}
            event_study_cis = {}
            event_study_p_values = {}
            for h in boot_event_study:
                orig_eff = original_event_study[h]["effect"]
                shifted_h = boot_event_study[h] + orig_eff
                se_h, ci_h, p_h = _compute_effect_bootstrap_stats(
                    orig_eff,
                    shifted_h,
                    alpha=self.alpha,
                    context=f"ImputationDiD event study (h={h})",
                )
                event_study_ses[h] = se_h
                event_study_cis[h] = ci_h
                event_study_p_values[h] = p_h

        group_ses = None
        group_cis = None
        group_p_values = None
        if boot_group and original_group:
            group_ses = {}
            group_cis = {}
            group_p_values = {}
            for g in boot_group:
                orig_eff = original_group[g]["effect"]
                shifted_g = boot_group[g] + orig_eff
                se_g, ci_g, p_g = _compute_effect_bootstrap_stats(
                    orig_eff,
                    shifted_g,
                    alpha=self.alpha,
                    context=f"ImputationDiD group effect (g={g})",
                )
                group_ses[g] = se_g
                group_cis[g] = ci_g
                group_p_values[g] = p_g

        return ImputationBootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weights,
            alpha=self.alpha,
            overall_att_se=overall_se,
            overall_att_ci=overall_ci,
            overall_att_p_value=overall_p,
            event_study_ses=event_study_ses,
            event_study_cis=event_study_cis,
            event_study_p_values=event_study_p_values,
            group_ses=group_ses,
            group_cis=group_cis,
            group_p_values=group_p_values,
            bootstrap_distribution=boot_overall_shifted,
        )
