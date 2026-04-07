"""
Bootstrap inference methods for the Two-Stage DiD estimator.

This module contains TwoStageDiDBootstrapMixin, which provides multiplier
bootstrap inference on the GMM influence function. Extracted from two_stage.py
for module size management.
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse.linalg import factorized as sparse_factorized

from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats as _compute_effect_bootstrap_stats,
)
from diff_diff.bootstrap_utils import (
    generate_bootstrap_weights_batch as _generate_bootstrap_weights_batch,
)
from diff_diff.bootstrap_utils import (
    generate_survey_multiplier_weights_batch as _generate_survey_multiplier_weights_batch,
)
from diff_diff.linalg import solve_ols
from diff_diff.two_stage_results import TwoStageBootstrapResults

# Maximum number of elements before falling back to per-column sparse aggregation.
# Keep in sync with two_stage.py.
_SPARSE_DENSE_THRESHOLD = 10_000_000

__all__ = [
    "TwoStageDiDBootstrapMixin",
]


class TwoStageDiDBootstrapMixin:
    """Mixin providing bootstrap inference methods for TwoStageDiD."""

    # Type hints for attributes accessed from the main class
    n_bootstrap: int
    bootstrap_weights: str
    alpha: float
    seed: Optional[int]
    horizon_max: Optional[int]

    if TYPE_CHECKING:
        from scipy import sparse

        def _build_fe_design(
            self,
            df: pd.DataFrame,
            unit: str,
            time: str,
            covariates: Optional[List[str]],
            omega_0_mask: pd.Series,
        ) -> Tuple["sparse.csr_matrix", "sparse.csr_matrix", Dict[Any, int], Dict[Any, int]]: ...

        @staticmethod
        def _compute_gmm_scores(
            c_by_cluster: np.ndarray,
            gamma_hat: np.ndarray,
            s2_by_cluster: np.ndarray,
        ) -> np.ndarray: ...

    def _compute_cluster_S_scores(
        self,
        df: pd.DataFrame,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        delta_hat: Optional[np.ndarray],
        kept_cov_mask: Optional[np.ndarray],
        X_2: np.ndarray,
        eps_2: np.ndarray,
        cluster_ids: np.ndarray,
        survey_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-cluster S_g scores for bootstrap.

        Returns
        -------
        S : np.ndarray, shape (G, k)
            Per-cluster influence scores.
        bread : np.ndarray, shape (k, k)
            (X'_2 X_2)^{-1}.
        unique_clusters : np.ndarray
            Unique cluster identifiers.
        """
        n = len(df)
        k = X_2.shape[1]

        cov_list = covariates
        if covariates and kept_cov_mask is not None and not np.all(kept_cov_mask):
            cov_list = [c for c, k_ in zip(covariates, kept_cov_mask) if k_]

        X_1_sparse, X_10_sparse, _, _ = self._build_fe_design(
            df, unit, time, cov_list, omega_0_mask
        )
        p = X_1_sparse.shape[1]

        # Reconstruct Y and compute eps_10
        alpha_i = df[unit].map(unit_fe).values
        beta_t = df[time].map(time_fe).values
        alpha_i = np.where(pd.isna(alpha_i), 0.0, alpha_i).astype(float)
        beta_t = np.where(pd.isna(beta_t), 0.0, beta_t).astype(float)
        fitted_1 = alpha_i + beta_t
        if delta_hat is not None and cov_list:
            if kept_cov_mask is not None and not np.all(kept_cov_mask):
                fitted_1 = fitted_1 + np.dot(df[cov_list].values, delta_hat[kept_cov_mask])
            else:
                fitted_1 = fitted_1 + np.dot(df[cov_list].values, delta_hat)

        y_tilde = df["_y_tilde"].values
        y_vals = y_tilde + fitted_1

        eps_10 = np.empty(n)
        omega_0 = omega_0_mask.values
        eps_10[omega_0] = y_vals[omega_0] - fitted_1[omega_0]
        eps_10[~omega_0] = y_vals[~omega_0]

        # gamma_hat — with survey weights, both cross-products need W
        if survey_weights is not None:
            XtX_10 = X_10_sparse.T @ X_10_sparse.multiply(survey_weights[:, None])
            Xt1_X2 = X_1_sparse.T @ (X_2 * survey_weights[:, None])
        else:
            XtX_10 = X_10_sparse.T @ X_10_sparse
            Xt1_X2 = X_1_sparse.T @ X_2

        try:
            solve_XtX = sparse_factorized(XtX_10.tocsc())
            if Xt1_X2.ndim == 1:
                gamma_hat = solve_XtX(Xt1_X2).reshape(-1, 1)
            else:
                gamma_hat = np.column_stack(
                    [solve_XtX(Xt1_X2[:, j]) for j in range(Xt1_X2.shape[1])]
                )
        except RuntimeError:
            gamma_hat = np.linalg.lstsq(XtX_10.toarray(), Xt1_X2, rcond=None)[0]
            if gamma_hat.ndim == 1:
                gamma_hat = gamma_hat.reshape(-1, 1)

        # Per-cluster aggregation — survey weights multiply eps_10 before sparse multiply
        if survey_weights is not None:
            weighted_eps_10 = survey_weights * eps_10
        else:
            weighted_eps_10 = eps_10
        weighted_X10 = X_10_sparse.multiply(weighted_eps_10[:, None])
        unique_clusters, cluster_indices = np.unique(cluster_ids, return_inverse=True)
        G = len(unique_clusters)

        n_elements = weighted_X10.shape[0] * weighted_X10.shape[1]
        c_by_cluster = np.zeros((G, p))
        if n_elements > _SPARSE_DENSE_THRESHOLD:
            # Per-column path: limits peak memory for large FE matrices
            weighted_X10_csc = weighted_X10.tocsc()
            for j_col in range(p):
                col_data = weighted_X10_csc.getcol(j_col).toarray().ravel()
                np.add.at(c_by_cluster[:, j_col], cluster_indices, col_data)
        else:
            # Dense path: faster for moderate-size matrices
            weighted_X10_dense = weighted_X10.toarray()
            for j_col in range(p):
                np.add.at(c_by_cluster[:, j_col], cluster_indices, weighted_X10_dense[:, j_col])

        if survey_weights is not None:
            weighted_eps_2 = survey_weights * eps_2
        else:
            weighted_eps_2 = eps_2
        weighted_X2 = X_2 * weighted_eps_2[:, None]
        s2_by_cluster = np.zeros((G, k))
        for j_col in range(k):
            np.add.at(s2_by_cluster[:, j_col], cluster_indices, weighted_X2[:, j_col])

        S = self._compute_gmm_scores(c_by_cluster, gamma_hat, s2_by_cluster)

        # Bread — (X'_2 W X_2)^{-1} with survey weights
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            if survey_weights is not None:
                XtX_2 = X_2.T @ (X_2 * survey_weights[:, None])
            else:
                XtX_2 = np.dot(X_2.T, X_2)
        try:
            bread = np.linalg.solve(XtX_2, np.eye(k))
        except np.linalg.LinAlgError:
            bread = np.linalg.lstsq(XtX_2, np.eye(k), rcond=None)[0]

        return S, bread, unique_clusters

    def _run_bootstrap(
        self,
        df: pd.DataFrame,
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
        treatment_groups: List[Any],
        ref_period: int,
        balance_e: Optional[int],
        original_att: float,
        original_event_study: Optional[Dict[int, Dict[str, Any]]],
        original_group: Optional[Dict[Any, Dict[str, Any]]],
        aggregate: Optional[str],
        resolved_survey: Optional[Any] = None,
    ) -> Optional[TwoStageBootstrapResults]:
        """Run multiplier bootstrap on GMM influence function."""
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        y_tilde = df["_y_tilde"].values.copy()  # .copy() to avoid mutating df column
        n = len(df)
        cluster_ids = df[cluster_var].values

        # Extract survey weights for S-score computation and Stage-2 WLS
        survey_weights: Optional[np.ndarray] = None
        survey_weight_type: str = "pweight"
        if resolved_survey is not None:
            survey_weights = resolved_survey.weights
            survey_weight_type = resolved_survey.weight_type

        # Handle NaN y_tilde (from unidentified FEs) — matches _stage2_static logic
        nan_mask = ~np.isfinite(y_tilde)
        if nan_mask.any():
            y_tilde[nan_mask] = 0.0

        # --- Static specification bootstrap ---
        D = omega_1_mask.values.astype(float)  # .astype() already creates a copy
        D[nan_mask] = 0.0  # Exclude NaN y_tilde obs from bootstrap estimation

        # Degenerate case: all treated obs have NaN y_tilde
        if D.sum() == 0:
            return None

        X_2_static = D.reshape(-1, 1)
        coef_static = solve_ols(
            X_2_static, y_tilde, return_vcov=False,
            weights=survey_weights, weight_type=survey_weight_type,
        )[0]
        eps_2_static = y_tilde - np.dot(X_2_static, coef_static)

        S_static, bread_static, unique_clusters = self._compute_cluster_S_scores(
            df=df,
            unit=unit,
            time=time,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            delta_hat=delta_hat,
            kept_cov_mask=kept_cov_mask,
            X_2=X_2_static,
            eps_2=eps_2_static,
            cluster_ids=cluster_ids,
            survey_weights=survey_weights,
        )

        n_clusters = len(unique_clusters)

        # Generate bootstrap weights — PSU-level when survey design is present
        _use_survey_bootstrap = resolved_survey is not None and (
            resolved_survey.strata is not None
            or resolved_survey.psu is not None
            or resolved_survey.fpc is not None
        )

        if _use_survey_bootstrap:
            psu_weights, psu_ids = _generate_survey_multiplier_weights_batch(
                self.n_bootstrap, resolved_survey, self.bootstrap_weights, rng
            )
            # Map unique_clusters (PSU values) to PSU weight columns.
            # When survey+PSU is active, cluster_var == "_survey_cluster" so
            # unique_clusters are the PSU ids used in S-score aggregation.
            psu_id_to_col = {int(p): c for c, p in enumerate(psu_ids)}
            cluster_to_psu_col = np.array([psu_id_to_col[int(cl)] for cl in unique_clusters])
            all_weights = psu_weights[:, cluster_to_psu_col]
        else:
            all_weights = _generate_bootstrap_weights_batch(
                self.n_bootstrap, n_clusters, self.bootstrap_weights, rng
            )

        # T_b = bread @ (sum_g w_bg * S_g) = bread @ (W @ S)'  per boot
        # IF_b = bread @ S_g for each cluster, then perturb
        # boot_coef = all_weights @ S_static @ bread_static.T  -> (B, k)
        # For static (k=1): boot_att = all_weights @ S_static @ bread_static.T
        boot_att_vec = np.dot(all_weights, S_static)  # (B, 1)
        boot_att_vec = np.dot(boot_att_vec, bread_static.T)  # (B, 1)
        boot_overall = boot_att_vec[:, 0]

        boot_overall_shifted = boot_overall + original_att
        overall_se, overall_ci, overall_p = _compute_effect_bootstrap_stats(
            original_att,
            boot_overall_shifted,
            alpha=self.alpha,
            context="TwoStageDiD overall ATT",
        )

        # --- Event study bootstrap ---
        event_study_ses = None
        event_study_cis = None
        event_study_p_values = None

        if original_event_study and aggregate in ("event_study", "all"):
            # Recompute S scores for event study specification
            rel_times = df["_rel_time"].values
            treated_rel = rel_times[omega_1_mask.values]
            all_horizons = sorted(set(int(h) for h in treated_rel if np.isfinite(h)))
            if self.horizon_max is not None:
                all_horizons = [h for h in all_horizons if abs(h) <= self.horizon_max]

            if balance_e is not None:
                cohort_rel_times = self._build_cohort_rel_times(df, first_treat)
                balanced_cohorts = set()
                if all_horizons:
                    max_h = max(all_horizons)
                    required_range = set(range(-balance_e, max_h + 1))
                    for g, horizons in cohort_rel_times.items():
                        if required_range.issubset(horizons):
                            balanced_cohorts.add(g)
                if not balanced_cohorts:
                    all_horizons = []  # No qualifying cohorts -> skip event study bootstrap
                else:
                    balance_mask = df[first_treat].isin(balanced_cohorts).values
            else:
                balance_mask = np.ones(n, dtype=bool)

            est_horizons = [h for h in all_horizons if h != ref_period]

            # Filter out Prop 5 horizons (same logic as _stage2_event_study)
            has_never_treated = df["_never_treated"].any()
            h_bar_boot = np.inf
            if not has_never_treated and len(treatment_groups) > 1:
                h_bar_boot = max(treatment_groups) - min(treatment_groups)
            if h_bar_boot < np.inf:
                est_horizons = [h for h in est_horizons if h < h_bar_boot]

            if est_horizons:
                horizon_to_col = {h: j for j, h in enumerate(est_horizons)}
                k_es = len(est_horizons)
                X_2_es = np.zeros((n, k_es))
                for i in range(n):
                    if not balance_mask[i]:
                        continue
                    if nan_mask[i]:
                        continue  # NaN y_tilde -> exclude from bootstrap event study
                    h = rel_times[i]
                    if np.isfinite(h):
                        h_int = int(h)
                        if h_int in horizon_to_col:
                            X_2_es[i, horizon_to_col[h_int]] = 1.0

                coef_es = solve_ols(
                    X_2_es, y_tilde, return_vcov=False,
                    weights=survey_weights, weight_type=survey_weight_type,
                )[0]
                eps_2_es = y_tilde - np.dot(X_2_es, coef_es)

                S_es, bread_es, _ = self._compute_cluster_S_scores(
                    df=df,
                    unit=unit,
                    time=time,
                    covariates=covariates,
                    omega_0_mask=omega_0_mask,
                    unit_fe=unit_fe,
                    time_fe=time_fe,
                    delta_hat=delta_hat,
                    kept_cov_mask=kept_cov_mask,
                    X_2=X_2_es,
                    eps_2=eps_2_es,
                    cluster_ids=cluster_ids,
                    survey_weights=survey_weights,
                )

                # boot_coef_es: (B, k_es)
                boot_coef_es = np.dot(np.dot(all_weights, S_es), bread_es.T)

                event_study_ses = {}
                event_study_cis = {}
                event_study_p_values = {}
                for h in original_event_study:
                    if original_event_study[h].get("n_obs", 0) == 0:
                        continue
                    if np.isnan(original_event_study[h]["effect"]):
                        continue  # Skip Prop 5 and other NaN-effect horizons
                    if h not in horizon_to_col:
                        continue
                    j = horizon_to_col[h]
                    orig_eff = original_event_study[h]["effect"]
                    boot_h = boot_coef_es[:, j]
                    shifted_h = boot_h + orig_eff
                    se_h, ci_h, p_h = _compute_effect_bootstrap_stats(
                        orig_eff,
                        shifted_h,
                        alpha=self.alpha,
                        context=f"TwoStageDiD event study (h={h})",
                    )
                    event_study_ses[h] = se_h
                    event_study_cis[h] = ci_h
                    event_study_p_values[h] = p_h

        # --- Group bootstrap ---
        group_ses = None
        group_cis = None
        group_p_values = None

        if original_group and aggregate in ("group", "all"):
            group_to_col = {g: j for j, g in enumerate(treatment_groups)}
            k_grp = len(treatment_groups)
            X_2_grp = np.zeros((n, k_grp))
            ft_vals = df[first_treat].values
            treated_mask = omega_1_mask.values
            for i in range(n):
                if treated_mask[i]:
                    if nan_mask[i]:
                        continue  # NaN y_tilde -> exclude from group bootstrap
                    g = ft_vals[i]
                    if g in group_to_col:
                        X_2_grp[i, group_to_col[g]] = 1.0

            coef_grp = solve_ols(
                X_2_grp, y_tilde, return_vcov=False,
                weights=survey_weights, weight_type=survey_weight_type,
            )[0]
            eps_2_grp = y_tilde - np.dot(X_2_grp, coef_grp)

            S_grp, bread_grp, _ = self._compute_cluster_S_scores(
                df=df,
                unit=unit,
                time=time,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                delta_hat=delta_hat,
                kept_cov_mask=kept_cov_mask,
                X_2=X_2_grp,
                eps_2=eps_2_grp,
                cluster_ids=cluster_ids,
                survey_weights=survey_weights,
            )

            boot_coef_grp = np.dot(np.dot(all_weights, S_grp), bread_grp.T)

            group_ses = {}
            group_cis = {}
            group_p_values = {}
            for g in original_group:
                if g not in group_to_col:
                    continue
                j = group_to_col[g]
                orig_eff = original_group[g]["effect"]
                boot_g = boot_coef_grp[:, j]
                shifted_g = boot_g + orig_eff
                se_g, ci_g, p_g = _compute_effect_bootstrap_stats(
                    orig_eff,
                    shifted_g,
                    alpha=self.alpha,
                    context=f"TwoStageDiD group effect (g={g})",
                )
                group_ses[g] = se_g
                group_cis[g] = ci_g
                group_p_values[g] = p_g

        return TwoStageBootstrapResults(
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

    # =========================================================================
    # Utility
    # =========================================================================

    @staticmethod
    def _build_cohort_rel_times(
        df: pd.DataFrame,
        first_treat: str,
    ) -> Dict[Any, Set[int]]:
        """Build mapping of cohort -> set of observed relative times."""
        treated_mask = ~df["_never_treated"]
        treated_df = df.loc[treated_mask]
        result: Dict[Any, Set[int]] = {}
        ft_vals = treated_df[first_treat].values
        rt_vals = treated_df["_rel_time"].values
        for i in range(len(treated_df)):
            h = rt_vals[i]
            if np.isfinite(h):
                result.setdefault(ft_vals[i], set()).add(int(h))
        return result
