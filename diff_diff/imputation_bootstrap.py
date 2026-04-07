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
from diff_diff.imputation_results import ImputationBootstrapResults

__all__ = [
    "ImputationDiDBootstrapMixin",
]


def _compute_target_weights(
    tau_hat: np.ndarray,
    target_mask: np.ndarray,
) -> "tuple[np.ndarray, int]":
    """
    Equal weights for finite tau_hat observations within target_mask.

    Used by both aggregation and bootstrap paths to avoid weight logic
    duplication.

    Parameters
    ----------
    tau_hat : np.ndarray
        Per-observation treatment effects (may contain NaN).
    target_mask : np.ndarray
        Boolean mask selecting the target subset within tau_hat.

    Returns
    -------
    weights : np.ndarray
        Weight array (same length as tau_hat). 1/n_valid for finite
        observations in target_mask, 0 elsewhere.
    n_valid : int
        Number of finite observations in the target subset.
    """
    finite_target = np.isfinite(tau_hat) & target_mask
    n_valid = int(finite_target.sum())
    weights = np.zeros(len(tau_hat))
    if n_valid > 0:
        weights[np.where(finite_target)[0]] = 1.0 / n_valid
    return weights, n_valid


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
        ) -> Tuple[np.ndarray, np.ndarray]: ...

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
    ) -> Dict[str, Any]:
        """
        Pre-compute cluster-level influence function sums for each bootstrap target.

        For each aggregation target (overall, per-horizon, per-group), computes
        psi_i = sum_t v_it * epsilon_tilde_it for each cluster. The multiplier
        bootstrap then perturbs these psi sums with multiplier weights
        (rademacher/mammen/webb; configurable via ``bootstrap_weights``).

        Computational cost scales with the number of aggregation targets, since
        each target requires its own v_untreated computation (weight-dependent).
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
        )

        # Overall ATT
        overall_psi, cluster_ids = self._compute_cluster_psi_sums(**common, weights=overall_weights)
        result["overall"] = (overall_psi, cluster_ids)

        # Event study: per-horizon weights
        if event_study_effects:
            result["event_study"] = {}
            df_1 = df.loc[omega_1_mask]
            rel_times = df_1["_rel_time"].values

            # Balanced cohort mask (same logic as _aggregate_event_study)
            balanced_mask = None
            if balance_e is not None:
                all_horizons = sorted(set(int(h) for h in rel_times if np.isfinite(h)))
                if self.horizon_max is not None:
                    all_horizons = [h for h in all_horizons if abs(h) <= self.horizon_max]
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

                psi_h, _ = self._compute_cluster_psi_sums(**common, weights=weights_h)
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

                psi_g, _ = self._compute_cluster_psi_sums(**common, weights=weights_g)
                result["group"][g] = psi_g

        return result

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

        # Generate ALL weights upfront: shape (n_bootstrap, n_clusters)
        if _use_survey_bootstrap:
            psu_weights, psu_ids = _generate_survey_multiplier_weights_batch(
                self.n_bootstrap, resolved_survey, self.bootstrap_weights, rng
            )
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
