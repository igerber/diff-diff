"""Stage-2 aggregation + GMM variance engine for the Gardner two-stage estimator.

Extracted verbatim from ``diff_diff/two_stage.py`` (and, for the two shared
static helpers, ``two_stage_bootstrap.py``) for the M-022/M-119 post-fit
``aggregate()`` migration: ``two_stage.py`` imports ``two_stage_results.py``
and ``two_stage_bootstrap.py`` (which imports ``two_stage_results.py`` too),
so the results module can import neither -- the shared machinery lives here,
an import-leaf module both sides can reach (the
``efficient_did_aggregation.py`` / ``imputation_aggregation.py`` precedent).

Contents:

- module helpers ``_SPARSE_DENSE_THRESHOLD``, ``_LSMRUnconvergedError`` and
  ``_lsmr_certified_normal_solve`` (``two_stage.py`` re-imports all three --
  ``spillover.py`` and the bootstrap module's lazy imports keep working);
- :class:`_TwoStageAggregationMixin` -- the three Stage-2 aggregation levels
  (static / event-study / group), the joint GMM sandwich they recompute
  through, the Stage-1 helpers the replicate replay refits with, and the
  replicate-weight inference override replay. Inherited by ``TwoStageDiD``
  (fit-time behavior byte-identical) and hosted post-fit by the throwaway
  ``_TwoStageKitAggregator`` (``two_stage_results.py``).
"""

import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import factorized as sparse_factorized

from diff_diff.linalg import _rank_guarded_inv, solve_ols
from diff_diff.utils import _iterative_fe_solve, demean_by_groups, safe_inference

# Maximum number of elements before falling back to per-column sparse aggregation.
# 10M float64 elements ≈ 80 MB peak allocation. Above this, per-column .getcol()
# trades throughput for bounded memory. Keep in sync with two_stage_bootstrap.py.
_SPARSE_DENSE_THRESHOLD = 10_000_000


class _LSMRUnconvergedError(RuntimeError):
    """LSMR could not certify the Stage-1 normal-equation solve; the
    variance boundary converts this to NaN inference (fail-closed)."""


def _lsmr_certified_normal_solve(
    gram_csc, rhs: np.ndarray, context: str = "TwoStageDiD GMM sandwich"
) -> np.ndarray:
    """Least-squares solve of the (possibly singular) sparse Stage-1 Gram
    system ``gram @ out = rhs`` via per-column LSMR — no dense
    materialization of the ``(p_1, p_1)`` normal matrix (`O((U+T+K)^2)`
    OOM risk on large panels; the pattern the ImputationDiD LSMR fix
    closed, ported here after the consumer-invariance analysis).

    OUTPUT-PRESERVING despite the min-norm ambiguity on singular systems:
    least-squares solutions differ only by a ``null(X'X) = null(X_10)``
    component (weighted: ``null(X'WX) = null(W^{1/2}X_10)`` — zero-weight
    rows are inert in every weighted consumer because Psi/score/residual
    contributions carry the same ``W`` factor), and EVERY ``gamma_hat``
    consumer is an ``X_10``-range functional. One ``theta_exact`` consumer
    (the bootstrap exact-residual helper's ``X_1_sparse @ theta_exact``)
    evaluates theta on TREATED rows where a ``null(X_10)`` component would
    NOT annihilate — parity there holds for a second reason: both dense
    ``lstsq`` (SVD) and LSMR return the MIN-NORM least-squares solution, so
    the two solvers agree on the whole vector (to iterative tolerance), not
    just on range functionals; the fit-level singular-design parity test
    locks this at the V/SE level. The remaining consumers are the
    ``X_10``-range functionals — ``Psi_stage1 = X_10 @ gamma_hat``, the GMM
    score correction ``c_g' gamma_hat`` with ``c_g = X_{10,g}' eps_{10,g}``
    in ``rowspace(X_10)``, and Stage-1 residuals ``y - X_10 theta`` — so
    every null component annihilates. Locked by the singular-system parity
    test against a dense-lstsq oracle.

    CONVERGENCE IS VALIDATED (fail-closed): ``istop`` in ``{0, 1, 2, 4, 5}``
    certifies a solution / least-squares solution within tolerance (4/5 are
    the machine-precision analogues of 1/2 per SciPy); anything else gets
    ONE retry with an uncapped condition limit, then raises
    :class:`_LSMRUnconvergedError` — converted to NaN inference at the
    variance boundary rather than feeding an unverified solution into the
    GMM sandwich.
    """
    import scipy.sparse.linalg as spla

    _certified = (0, 1, 2, 4, 5)
    rhs_2d = np.atleast_2d(np.asarray(rhs, dtype=np.float64))
    if rhs_2d.shape[0] == 1 and np.asarray(rhs).ndim == 1:
        rhs_2d = rhs_2d.T
    dim = gram_csc.shape[0]
    out = np.empty((dim, rhs_2d.shape[1]))
    for j in range(rhs_2d.shape[1]):
        result = spla.lsmr(gram_csc, rhs_2d[:, j], atol=1e-14, btol=1e-14)
        z, istop = result[0], int(result[1])
        if istop not in _certified or not np.all(np.isfinite(z)):
            result = spla.lsmr(
                gram_csc,
                rhs_2d[:, j],
                atol=1e-14,
                btol=1e-14,
                conlim=1e16,
                maxiter=max(50 * dim, 10_000),
            )
            z, istop = result[0], int(result[1])
            if istop not in _certified or not np.all(np.isfinite(z)):
                warnings.warn(
                    f"{context}: the LSMR fallback solve of the "
                    f"Stage-1 normal equations did not converge (istop={istop}); "
                    "the affected variance is reported as NaN rather than from "
                    "an unverified solution.",
                    UserWarning,
                    stacklevel=3,
                )
                raise _LSMRUnconvergedError(f"LSMR uncertified (istop={istop})")
        out[:, j] = z
    return out


class _TwoStageAggregationMixin:
    """Shared Stage-2/GMM methods (moved verbatim from ``TwoStageDiD``).

    HOST-ATTRIBUTE CONTRACT -- the complete ``self.`` surface the moved
    methods read (typed class-level declarations for ``mypy diff_diff`` at
    zero errors on both hosts). Zero methods WRITE to ``self`` -- the
    post-fit throwaway host exists for estimator-mutation isolation only.
    """

    alpha: float
    pretrends: bool
    horizon_max: Optional[int]
    rank_deficient_action: str

    def _iterative_fe(
        self,
        y: np.ndarray,
        unit_vals: np.ndarray,
        time_vals: np.ndarray,
        idx: pd.Index,
        max_iter: int = 10_000,
        tol: float = 1e-10,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[Any, float], Dict[Any, float]]:
        """
        Estimate unit and time FE via iterative alternating projection.

        Thin wrapper over the shared bincount solver
        (``diff_diff.utils._iterative_fe_solve``): factorize unit/time once,
        solve on integer codes, map the level arrays back to dicts.

        Parameters
        ----------
        idx : pd.Index
            Unused; retained for call-site stability.
        weights : np.ndarray, optional
            Survey weights (weighted group means ``sum(w*x)/sum(w)``). A
            unit/period whose observations ALL carry zero weight has no
            identifying contribution and gets ``NaN`` FE (its key is kept so
            the rank-condition membership check still sees the group).

        Returns
        -------
        unit_fe : dict
            Mapping from unit -> unit fixed effect.
        time_fe : dict
            Mapping from time -> time fixed effect.
        """
        unit_codes, unit_uniques = pd.factorize(unit_vals, sort=False)
        time_codes, time_uniques = pd.factorize(time_vals, sort=False)
        if (unit_codes < 0).any() or (time_codes < 0).any():
            raise ValueError(
                "TwoStageDiD: unit or time column contains NaN. Drop or "
                "impute missing group keys before fitting."
            )
        unit_fe_arr, time_fe_arr = _iterative_fe_solve(
            np.asarray(y, dtype=np.float64),
            unit_codes.astype(np.intp, copy=False),
            time_codes.astype(np.intp, copy=False),
            len(unit_uniques),
            len(time_uniques),
            weights=weights,
            max_iter=max_iter,
            tol=tol,
            method_name="TwoStageDiD iterative FE solver",
        )
        unit_fe = dict(zip(unit_uniques, unit_fe_arr))
        time_fe = dict(zip(time_uniques, time_fe_arr))
        return unit_fe, time_fe

    def _fit_untreated_model(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[
        Dict[Any, float], Dict[Any, float], float, Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """
        Stage 1: Estimate unit + time FE on untreated observations.

        Parameters
        ----------
        weights : np.ndarray, optional
            Full-panel survey weights (same length as df). The untreated subset
            is extracted internally via omega_0_mask. When None, unweighted.

        Returns
        -------
        unit_fe, time_fe, grand_mean, delta_hat, kept_cov_mask
        """
        df_0 = df.loc[omega_0_mask]
        w_0 = weights[omega_0_mask.values] if weights is not None else None

        if covariates is None or len(covariates) == 0:
            y = df_0[outcome].values.copy()
            unit_fe, time_fe = self._iterative_fe(
                y, df_0[unit].values, df_0[time].values, df_0.index, weights=w_0
            )
            return unit_fe, time_fe, 0.0, None, None

        else:
            y = df_0[outcome].values.copy()
            X_raw = df_0[covariates].values.copy()
            units = df_0[unit].values
            times = df_0[time].values

            # Within-transform Y and all X columns through the shared MAP
            # engine (factorize-once + bincount + optional Rust kernel), one
            # dispatch for every column. within_transform pins [unit, time];
            # [time, unit] here preserves the historical time-then-unit sweep
            # order of the per-estimator loops.
            narrow = df_0[[outcome, *covariates, time, unit]].copy()
            demeaned, _ = demean_by_groups(
                narrow,
                [outcome, *covariates],
                [time, unit],
                inplace=True,
                weights=w_0,
                max_iter=10_000,
                tol=1e-10,
            )
            y_dm = demeaned[outcome].to_numpy(dtype=np.float64)
            X_dm = demeaned[covariates].to_numpy(dtype=np.float64)

            result = solve_ols(
                X_dm,
                y_dm,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
                column_names=covariates,
                weights=w_0,
            )
            delta_hat = result[0]
            kept_cov_mask = np.isfinite(delta_hat)
            delta_hat_clean = np.where(np.isfinite(delta_hat), delta_hat, 0.0)

            y_adj = y - np.dot(X_raw, delta_hat_clean)
            unit_fe, time_fe = self._iterative_fe(y_adj, units, times, df_0.index, weights=w_0)

            return unit_fe, time_fe, 0.0, delta_hat_clean, kept_cov_mask

    def _residualize(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute residualized outcome y_tilde for ALL observations.

        y_tilde_i = y_i - mu_hat_i - eta_hat_t [- X_i @ delta_hat]
        """
        alpha_i = df[unit].map(unit_fe).values
        beta_t = df[time].map(time_fe).values

        # Handle missing FE (NaN for units/periods not in untreated sample)
        alpha_i = np.where(pd.isna(alpha_i), np.nan, alpha_i).astype(float)
        beta_t = np.where(pd.isna(beta_t), np.nan, beta_t).astype(float)

        y_hat = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            y_hat = y_hat + np.dot(df[covariates].values, delta_hat)

        y_tilde = df[outcome].values - y_hat
        return y_tilde

    @staticmethod
    def _mask_nan_ytilde(y_tilde, warn: bool = True):
        """Mask non-finite y_tilde values and warn if any found.

        Returns the boolean mask of non-finite values. Modifies y_tilde in-place
        (sets NaN values to 0.0). ``warn=False`` suppresses the UserWarning -
        used ONLY by the replicate-refit closures, where zero-weight replicate
        designs (JK1/BRR) make NaN FE for zeroed-out PSUs expected mechanics
        (the main-fit warning still fires once; per-replicate repeats would
        emit up to ~3x n_replicates copies of the same message).
        """
        nan_mask = ~np.isfinite(y_tilde)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            if warn:
                warnings.warn(
                    f"{n_nan} observation(s) have non-finite imputed outcomes "
                    f"(y_tilde) from unidentified fixed effects. These "
                    f"observations are excluded from ATT estimation.",
                    UserWarning,
                    stacklevel=3,
                )
            y_tilde[nan_mask] = 0.0
        return nan_mask

    def _stage2_static(
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
        survey_weights: Optional[np.ndarray] = None,
        survey_weight_type: str = "pweight",
        resolved_survey=None,
        score_pad_mask: Optional[np.ndarray] = None,
        cluster_ids_full: Optional[np.ndarray] = None,
        warn_nan: bool = True,
    ) -> Tuple[float, float]:
        """
        Static (simple ATT) Stage 2: OLS of y_tilde on D_it.

        Returns (att, se).
        """
        y_tilde = df["_y_tilde"].values.copy()
        nan_mask = self._mask_nan_ytilde(y_tilde, warn=warn_nan)

        D = omega_1_mask.values.astype(float)
        # Zero out treatment indicator for NaN y_tilde obs (don't count in ATT)
        D[nan_mask] = 0.0

        # X_2: treatment indicator (no intercept)
        X_2 = D.reshape(-1, 1)

        # Avoid degenerate case where all treated obs have NaN y_tilde
        if D.sum() == 0:
            return np.nan, np.nan

        # Stage 2 OLS for point estimate (discard naive SE)
        coef, residuals, _ = solve_ols(
            X_2,
            y_tilde,
            return_vcov=False,
            weights=survey_weights,
            weight_type=survey_weight_type,
        )
        att = float(coef[0])

        # GMM sandwich variance
        # An uncertified LSMR Stage-1 fallback solve fails closed:
        # NaN vcov -> NaN SE/t/p/CI (the helper already warned).
        try:
            V = self._compute_gmm_variance(
                df=df,
                unit=unit,
                time=time,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                delta_hat=delta_hat,
                kept_cov_mask=kept_cov_mask,
                X_2=X_2,
                cluster_ids=df[cluster_var].values,
                survey_weights=survey_weights,
                resolved_survey=resolved_survey,
                score_pad_mask=score_pad_mask,
                cluster_ids_full=cluster_ids_full,
            )
        except _LSMRUnconvergedError:
            V = np.full((X_2.shape[1], X_2.shape[1]), np.nan)

        se = float(np.sqrt(max(V[0, 0], 0.0)))
        return att, se

    def _stage2_event_study(
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
        treatment_groups: List[Any],
        ref_period: int,
        balance_e: Optional[int],
        kept_cov_mask: Optional[np.ndarray],
        survey_weights: Optional[np.ndarray] = None,
        survey_weight_type: str = "pweight",
        survey_df: Optional[int] = None,
        resolved_survey=None,
        score_pad_mask: Optional[np.ndarray] = None,
        cluster_ids_full: Optional[np.ndarray] = None,
        warn_nan: bool = True,
    ) -> Tuple[Dict[int, Dict[str, Any]], Optional[np.ndarray], Optional[List[int]]]:
        """Event study Stage 2: OLS of y_tilde on relative-time dummies.

        Returns ``(effects, vcov, vcov_index)``: the per-horizon effects
        dict, the full GMM variance-covariance matrix over the ESTIMATED
        horizon coefficients, and the horizon labels ordering its
        rows/columns. The reference period and Proposition-5 horizons are
        never regression columns, so they appear in ``effects`` but not in
        ``vcov_index``; all-filtered horizons (n_obs == 0) ARE columns,
        with NaN-filled rows/columns from the rank guard. ``(dict, None,
        None)`` on the degenerate early returns that fit no Stage-2
        regression.
        """
        y_tilde = df["_y_tilde"].values.copy()
        nan_mask = self._mask_nan_ytilde(y_tilde, warn=warn_nan)
        rel_times = df["_rel_time"].values
        n = len(df)

        # Get all horizons — include pre-periods when pretrends=True
        if self.pretrends:
            evt_rel = rel_times[~df["_never_treated"].values]
        else:
            evt_rel = rel_times[omega_1_mask.values]
        all_horizons = sorted(set(int(h) for h in evt_rel if np.isfinite(h)))

        # Apply horizon_max filter
        if self.horizon_max is not None:
            all_horizons = [h for h in all_horizons if abs(h) <= self.horizon_max]

        # Apply balance_e filter
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
                warnings.warn(
                    f"No cohorts satisfy balance_e={balance_e} requirement. "
                    "Event study results will contain only the reference period. "
                    "Consider reducing balance_e.",
                    UserWarning,
                    stacklevel=2,
                )
                return (
                    {
                        ref_period: {
                            "effect": 0.0,
                            "se": 0.0,
                            "t_stat": np.nan,
                            "p_value": np.nan,
                            "conf_int": (0.0, 0.0),
                            "n_obs": 0,
                        }
                    },
                    None,
                    None,
                )
            balance_mask = df[first_treat].isin(balanced_cohorts).values
        else:
            balance_mask = np.ones(n, dtype=bool)

        # Check Proposition 5: no never-treated units
        has_never_treated = df["_never_treated"].any()
        h_bar = np.inf
        if not has_never_treated and len(treatment_groups) > 1:
            h_bar = max(treatment_groups) - min(treatment_groups)

        # Identify Prop 5 horizons and compute their actual treated obs counts.
        # Treated obs have NaN y_tilde at these horizons (counterfactual
        # unidentified), but actual_n counts them to distinguish from truly
        # empty horizons. rel_times is NaN for untreated/never-treated obs
        # (line ~653), so (rel_times == h) is False for them.
        prop5_horizons = []
        prop5_effects: Dict[int, Dict[str, Any]] = {}
        if h_bar < np.inf:
            for h in all_horizons:
                if h == ref_period:
                    continue
                if h >= h_bar:
                    actual_n = int(np.sum((rel_times == h) & omega_1_mask.values & balance_mask))
                    if actual_n > 0:
                        prop5_horizons.append(h)
                        prop5_effects[h] = {
                            "effect": np.nan,
                            "se": np.nan,
                            "t_stat": np.nan,
                            "p_value": np.nan,
                            "conf_int": (np.nan, np.nan),
                            "n_obs": actual_n,
                        }

        # Remove reference period AND Prop 5 horizons from estimation
        prop5_set = set(prop5_horizons)
        est_horizons = [h for h in all_horizons if h != ref_period and h not in prop5_set]

        if len(est_horizons) == 0:
            # No horizons to estimate — return the reference row PLUS any
            # Proposition-5 rows (local-review fix, 2(b) PR-3b): when EVERY
            # non-reference horizon is Prop-5-unidentified, the rows must
            # still surface as all-NaN with n_obs > 0 and the consolidated
            # warning, exactly as on the normal path below — dropping them
            # here reported real treated horizons as absent instead of
            # unidentified (contra REGISTRY Prop-5 contract).
            if prop5_horizons:
                warnings.warn(
                    f"Horizons {prop5_horizons} are not identified without "
                    f"never-treated units (Proposition 5). Set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
            ref_only: Dict[int, Dict[str, Any]] = {
                ref_period: {
                    "effect": 0.0,
                    "se": 0.0,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (0.0, 0.0),
                    "n_obs": 0,
                }
            }
            ref_only.update(prop5_effects)
            return (ref_only, None, None)

        # Build Stage 2 design: one column per horizon (no intercept)
        # Never-treated obs get all-zero rows (undefined relative time -> NaN)
        # With no intercept, they contribute zero to X'_2 X_2 and X'_2 y_tilde
        horizon_to_col = {h: j for j, h in enumerate(est_horizons)}
        k = len(est_horizons)
        X_2 = np.zeros((n, k))

        for i in range(n):
            if not balance_mask[i]:
                continue
            if nan_mask[i]:
                continue  # NaN y_tilde -> don't include in event study
            h = rel_times[i]
            if np.isfinite(h):
                h_int = int(h)
                if h_int in horizon_to_col:
                    X_2[i, horizon_to_col[h_int]] = 1.0

        # Stage 2 OLS
        coef, residuals, _ = solve_ols(
            X_2,
            y_tilde,
            return_vcov=False,
            weights=survey_weights,
            weight_type=survey_weight_type,
        )

        # GMM variance for full coefficient vector
        # An uncertified LSMR Stage-1 fallback solve fails closed:
        # NaN vcov -> NaN SE/t/p/CI (the helper already warned).
        try:
            V = self._compute_gmm_variance(
                df=df,
                unit=unit,
                time=time,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                delta_hat=delta_hat,
                kept_cov_mask=kept_cov_mask,
                X_2=X_2,
                cluster_ids=df[cluster_var].values,
                survey_weights=survey_weights,
                resolved_survey=resolved_survey,
                score_pad_mask=score_pad_mask,
                cluster_ids_full=cluster_ids_full,
            )
        except _LSMRUnconvergedError:
            V = np.full((X_2.shape[1], X_2.shape[1]), np.nan)

        # Build results dict
        event_study_effects: Dict[int, Dict[str, Any]] = {}

        # Reference period marker
        event_study_effects[ref_period] = {
            "effect": 0.0,
            "se": 0.0,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_int": (0.0, 0.0),
            "n_obs": 0,
        }

        for h in est_horizons:
            j = horizon_to_col[h]
            n_obs = int(np.sum(X_2[:, j]))

            if n_obs == 0:
                event_study_effects[h] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": 0,
                }
                continue

            effect = float(coef[j])
            se = float(np.sqrt(max(V[j, j], 0.0)))

            t_stat, p_val, ci = safe_inference(effect, se, alpha=self.alpha, df=survey_df)

            event_study_effects[h] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_obs": n_obs,
            }

        # Add Proposition 5 entries (unidentified horizons with n_obs > 0)
        event_study_effects.update(prop5_effects)

        if prop5_horizons:
            warnings.warn(
                f"Horizons {prop5_horizons} are not identified without "
                f"never-treated units (Proposition 5). Set to NaN.",
                UserWarning,
                stacklevel=2,
            )

        return event_study_effects, V, [int(h) for h in est_horizons]

    def _stage2_group(
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
        treatment_groups: List[Any],
        kept_cov_mask: Optional[np.ndarray],
        survey_weights: Optional[np.ndarray] = None,
        survey_weight_type: str = "pweight",
        survey_df: Optional[int] = None,
        resolved_survey=None,
        score_pad_mask: Optional[np.ndarray] = None,
        cluster_ids_full: Optional[np.ndarray] = None,
        warn_nan: bool = True,
    ) -> Dict[Any, Dict[str, Any]]:
        """Group (cohort) Stage 2: OLS of y_tilde on cohort dummies."""
        y_tilde = df["_y_tilde"].values.copy()
        nan_mask = self._mask_nan_ytilde(y_tilde, warn=warn_nan)
        n = len(df)

        # Build Stage 2 design: one column per cohort (no intercept)
        group_to_col = {g: j for j, g in enumerate(treatment_groups)}
        k = len(treatment_groups)
        X_2 = np.zeros((n, k))

        ft_vals = df[first_treat].values
        treated_mask = omega_1_mask.values
        for i in range(n):
            if treated_mask[i] and not nan_mask[i]:
                g = ft_vals[i]
                if g in group_to_col:
                    X_2[i, group_to_col[g]] = 1.0

        # Stage 2 OLS
        coef, residuals, _ = solve_ols(
            X_2,
            y_tilde,
            return_vcov=False,
            weights=survey_weights,
            weight_type=survey_weight_type,
        )

        # GMM variance
        # An uncertified LSMR Stage-1 fallback solve fails closed:
        # NaN vcov -> NaN SE/t/p/CI (the helper already warned).
        try:
            V = self._compute_gmm_variance(
                df=df,
                unit=unit,
                time=time,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                delta_hat=delta_hat,
                kept_cov_mask=kept_cov_mask,
                X_2=X_2,
                cluster_ids=df[cluster_var].values,
                survey_weights=survey_weights,
                resolved_survey=resolved_survey,
                score_pad_mask=score_pad_mask,
                cluster_ids_full=cluster_ids_full,
            )
        except _LSMRUnconvergedError:
            V = np.full((X_2.shape[1], X_2.shape[1]), np.nan)

        group_effects: Dict[Any, Dict[str, Any]] = {}
        for g in treatment_groups:
            j = group_to_col[g]
            n_obs = int(np.sum(X_2[:, j]))

            if n_obs == 0:
                group_effects[g] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": 0,
                }
                continue

            effect = float(coef[j])
            se = float(np.sqrt(max(V[j, j], 0.0)))

            t_stat, p_val, ci = safe_inference(effect, se, alpha=self.alpha, df=survey_df)

            group_effects[g] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_obs": n_obs,
            }

        return group_effects

    @staticmethod
    def _compute_gmm_scores(
        c_by_cluster: np.ndarray,
        gamma_hat: np.ndarray,
        s2_by_cluster: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-cluster GMM scores S_g = gamma_hat' c_g - X'_{2g} eps_{2g}.

        Handles NaN/overflow from rank-deficient FE by wrapping in errstate
        and replacing non-finite values with 0.

        Parameters
        ----------
        c_by_cluster : np.ndarray, shape (G, p)
            Per-cluster Stage 1 scores.
        gamma_hat : np.ndarray, shape (p, k)
            Cross-moment correction matrix.
        s2_by_cluster : np.ndarray, shape (G, k)
            Per-cluster Stage 2 scores.

        Returns
        -------
        np.ndarray, shape (G, k)
            Per-cluster influence scores.
        """
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            correction = np.dot(c_by_cluster, gamma_hat)
        np.nan_to_num(correction, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return correction - s2_by_cluster

    def _compute_gmm_variance(
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
        cluster_ids: np.ndarray,
        survey_weights: Optional[np.ndarray] = None,
        resolved_survey=None,
        score_pad_mask: Optional[np.ndarray] = None,
        cluster_ids_full: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute GMM sandwich variance (Butts & Gardner 2022).

        Matches the R `did2s` source code implementation: uses the GLOBAL
        Hessian inverse (not per-cluster) and NO finite-sample adjustments.

        The per-observation influence function is:
            IF_i = (X'_2 X_2)^{-1} [gamma_hat' x_{10i} eps_{10i} - x_{2i} eps_{2i}]

        where gamma_hat = (X'_{10} X_{10})^{-1} (X'_1 X_2) uses the GLOBAL
        cross-moment.

        The cluster-robust variance is:
            V = (X'_2 X_2)^{-1} (sum_g S_g S'_g) (X'_2 X_2)^{-1}
            S_g = gamma_hat' c_g - X'_{2g} eps_{2g}
            c_g = X'_{10g} eps_{10g}

        With survey weights W (diagonal):
            Bread: (X'_2 W X_2)^{-1}
            gamma_hat: (X'_{10} W X_{10})^{-1} (X'_1 W X_2)
            c_g = sum_{i in g} w_i * x_{10i} * eps_{10i}
            s2_g = sum_{i in g} w_i * x_{2i} * eps_{2i}

        Parameters
        ----------
        X_2 : np.ndarray, shape (n, k)
            Stage 2 design matrix (treatment indicators). The Stage-2 residual
            ``eps_2`` is re-solved internally from the *exact* Stage-1 residuals
            (see the exact-residual note below), so it is not a parameter.
        cluster_ids : np.ndarray, shape (n,)
            Cluster identifiers, fit-sample length. Used for the per-cluster
            stage-1 / stage-2 score aggregation (OLS path).
        survey_weights : np.ndarray, optional
            Survey weights of shape (n,). When None, unweighted (identical
            to current code).
        resolved_survey : ResolvedSurveyDesign, optional
            Resolved survey design. Under Wave E.3 parity (PR #482 SpilloverDiD
            precedent) the design retains full-domain `n_psu` / `n_strata` /
            `df_survey` / `strata` / `fpc` / `psu` arrays even when the
            always-treated drop removes rows from the OLS sample. The
            zero-padded per-cluster scores expand onto the full-domain PSU
            list before stratified-meat dispatch. R `survey::svyrecvar(subset())`
            convention (Lumley 2010 §2.5); mirrors `imputation.py:2175-2183`
            (PreTrendsImputation) and `prep.py:1401-1432` (DCDH cell variance).
        score_pad_mask : np.ndarray of shape (n_full,), bool, optional
            Wave E.3 parity zero-pad mask. When supplied, indicates which
            FULL-DOMAIN rows are present in the fit sample (True = kept
            for OLS). Requires `n == int(np.sum(score_pad_mask))`. Co-supplied
            with `cluster_ids_full`. Per-cluster stage-1 / stage-2 score
            aggregates computed at fit-length are expanded onto the
            full-domain unique-PSU list; PSUs absent from the fit sample
            (e.g. PSUs containing only always-treated rows) get zero score
            rows but still count toward `G_full` for `n_psu` / `df_survey`.
            None (default) → no padding, exact pre-PR behavior.
        cluster_ids_full : np.ndarray of shape (n_full,), optional
            Full-domain PSU labels. Co-supplied with `score_pad_mask`. Must
            share the same length. Provides the full-domain unique-PSU list
            used both for score zero-pad expansion and for downstream
            strata/FPC `obs_idx` lookups against the full-domain
            `resolved_survey.strata` / `.fpc` arrays. None (default) → no
            padding, exact pre-PR behavior.

        Returns
        -------
        np.ndarray, shape (k, k)
            Variance-covariance matrix.
        """
        n = len(df)
        k = X_2.shape[1]

        # Exclude rank-deficient covariates
        cov_list = covariates
        if covariates and kept_cov_mask is not None and not np.all(kept_cov_mask):
            cov_list = [c for c, k_ in zip(covariates, kept_cov_mask) if k_]

        # Build sparse FE design matrices X_1 (all obs) and X_10 (untreated only)
        X_1_sparse, X_10_sparse, unit_to_idx, time_to_idx = self._build_fe_design(
            df, unit, time, cov_list, omega_0_mask
        )

        p = X_1_sparse.shape[1]

        # eps_10 = Y - X_10 @ gamma_hat
        # Untreated: stage 1 residual (Y - fitted). Treated: Y (X_10 rows = 0).
        # Reconstruct Y from y_tilde: Y = y_tilde + fitted_stage1. Because
        # y_tilde = Y - fitted_1, the iterative FE in fitted_1 cancel exactly, so
        # y_vals == Y (independent of the iterative solver's tolerance).
        alpha_i = df[unit].map(unit_fe).values
        beta_t = df[time].map(time_fe).values
        # Identification mask: obs whose unit AND time FE are both identified by the
        # untreated Stage-1 fit. Rank-deficient / Proposition-5 obs (NaN FE) keep the
        # iterative-residual behavior; only identified obs get the exact residuals.
        identified = np.isfinite(np.asarray(alpha_i, dtype=float)) & np.isfinite(
            np.asarray(beta_t, dtype=float)
        )
        alpha_i = np.where(pd.isna(alpha_i), 0.0, alpha_i).astype(float)
        beta_t = np.where(pd.isna(beta_t), 0.0, beta_t).astype(float)
        fitted_1 = alpha_i + beta_t
        if delta_hat is not None and cov_list:
            if kept_cov_mask is not None and not np.all(kept_cov_mask):
                fitted_1 = fitted_1 + np.dot(df[cov_list].values, delta_hat[kept_cov_mask])
            else:
                fitted_1 = fitted_1 + np.dot(df[cov_list].values, delta_hat)

        y_tilde = df["_y_tilde"].values
        y_vals = y_tilde + fitted_1  # reconstruct Y
        y_vals_clean = np.nan_to_num(y_vals, nan=0.0)

        omega_0 = omega_0_mask.values

        # 1. gamma_hat = (X'_{10} W X_{10})^{-1} (X'_1 W X_2)  [p x k]
        # With survey weights, both cross-products need W. We reuse the SAME
        # factorization of (X'_{10} W X_{10}) to also solve the exact Stage-1 FE
        # coefficients theta_exact (see exact-residual note below).
        if survey_weights is not None:
            XtWX_10 = X_10_sparse.T @ X_10_sparse.multiply(survey_weights[:, None])
            Xt1_WX2 = X_1_sparse.T @ (X_2 * survey_weights[:, None])
            rhs_fe = X_10_sparse.T @ (survey_weights * y_vals_clean)
        else:
            XtWX_10 = X_10_sparse.T @ X_10_sparse  # (p x p) sparse
            Xt1_WX2 = X_1_sparse.T @ X_2  # (p x k) dense
            rhs_fe = X_10_sparse.T @ y_vals_clean  # (p,) X'_{10} W Y

        try:
            solve_XtX = sparse_factorized(XtWX_10.tocsc())
            if Xt1_WX2.ndim == 1:
                gamma_hat = solve_XtX(Xt1_WX2).reshape(-1, 1)
            else:
                gamma_hat = np.column_stack(
                    [solve_XtX(Xt1_WX2[:, j]) for j in range(Xt1_WX2.shape[1])]
                )
            theta_exact = np.asarray(solve_XtX(np.asarray(rhs_fe).ravel())).ravel()
        except RuntimeError as exc:
            # Singular matrix — fall back to certified sparse LSMR. Silent-failure
            # audit axis C: emit a UserWarning on fallback instead of swallowing.
            warnings.warn(
                "TwoStageDiD GMM sandwich: sparse factorization of "
                f"(X'_{{10}} W X_{{10}}) failed ({type(exc).__name__}); falling "
                "back to sparse LSMR. This may indicate a rank-deficient or "
                "near-singular Stage 1 design matrix and SE estimates may be "
                "less reliable.",
                UserWarning,
                stacklevel=2,
            )
            XtWX_10_csc = XtWX_10.tocsc()
            gamma_hat = _lsmr_certified_normal_solve(XtWX_10_csc, Xt1_WX2)
            theta_exact = _lsmr_certified_normal_solve(
                XtWX_10_csc, np.asarray(rhs_fe).ravel()
            ).ravel()

        # Exact Stage-1 / Stage-2 residuals. The point-estimate path uses the
        # iterative alternating-projection FE solver (`_iterative_fe`), which
        # converges only to ~1e-7 on unbalanced untreated panels; that error is
        # negligible for the ATT but perturbs the variance by ~1% relative to the
        # analytical GMM sandwich. The variance therefore re-solves the Stage-1 FE
        # EXACTLY using the sparse normal equations already factorized for gamma_hat
        # (theta_exact), matching R `did2s` to ~1e-7 and mirroring ImputationDiD's
        # exact-sparse variance path. The shared `_exact_gmm_residuals` helper is
        # used by BOTH this analytical path and the multiplier bootstrap
        # (`_compute_cluster_S_scores`) so the influence function is single-sourced.
        eps_10, eps_2 = self._exact_gmm_residuals(
            X_1_sparse,
            theta_exact,
            y_vals_clean,
            identified,
            omega_0,
            y_tilde,
            X_2,
            survey_weights,
        )

        # 2. Per-cluster Stage 1 scores: c_g = sum_{i in g} w_i * x_{10i} * eps_{10i}
        # Only untreated obs have non-zero X_10 rows
        # With survey weights: multiply eps_10 by survey_weights before sparse multiply
        if survey_weights is not None:
            weighted_eps_10 = survey_weights * eps_10
        else:
            weighted_eps_10 = eps_10
        weighted_X10 = X_10_sparse.multiply(weighted_eps_10[:, None])  # sparse element-wise

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

        # 3. Per-cluster Stage 2 scores: s2_g = sum_{i in g} w_i * x_{2i} * eps_{2i}
        if survey_weights is not None:
            weighted_eps_2 = survey_weights * eps_2
        else:
            weighted_eps_2 = eps_2
        weighted_X2 = X_2 * weighted_eps_2[:, None]  # (n x k) dense
        s2_by_cluster = np.zeros((G, k))
        for j_col in range(k):
            np.add.at(s2_by_cluster[:, j_col], cluster_indices, weighted_X2[:, j_col])

        # Wave E.3 parity (PR #482 SpilloverDiD precedent): when the caller
        # supplies `score_pad_mask` + `cluster_ids_full`, expand per-cluster
        # stage-1 / stage-2 score aggregates onto the FULL-DOMAIN unique-PSU
        # list. PSUs absent from the fit sample (those containing only
        # always-treated rows) get zero score rows but still count toward
        # `G_full` for `n_psu` / `df_survey` accounting. Mirrors R
        # `survey::svyrecvar(subset())` (Lumley 2010 §2.5) and the in-library
        # convention at `imputation.py:2175-2183` (PreTrendsImputation) and
        # `prep.py:1401-1432` (DCDH cell variance). Downstream strata / FPC
        # lookups use `cluster_ids_for_lookup` so the obs_idx applies to the
        # full-domain `resolved_survey.strata` / `.fpc` arrays.
        if score_pad_mask is not None:
            if cluster_ids_full is None:
                raise ValueError(
                    "_compute_gmm_variance: score_pad_mask requires "
                    "cluster_ids_full to be co-supplied (Wave E.3 parity "
                    "contract — score zero-pad expansion needs the "
                    "full-domain PSU labels to align with resolved_survey)."
                )
            if resolved_survey is None:
                raise ValueError(
                    "_compute_gmm_variance: score_pad_mask requires "
                    "resolved_survey to be co-supplied (Wave E.3 parity "
                    "contract — zero-pad only meaningful under a survey "
                    "design that retains full-domain dimensions)."
                )
            n_full = int(len(score_pad_mask))
            if int(len(cluster_ids_full)) != n_full:
                raise ValueError(
                    "_compute_gmm_variance: score_pad_mask and "
                    "cluster_ids_full must share the FULL-DOMAIN length; "
                    f"got len(score_pad_mask)={n_full}, "
                    f"len(cluster_ids_full)={int(len(cluster_ids_full))}."
                )
            if int(np.sum(score_pad_mask)) != n:
                raise ValueError(
                    "_compute_gmm_variance: int(np.sum(score_pad_mask)) "
                    f"({int(np.sum(score_pad_mask))}) must equal the "
                    f"fit-sample length n ({n}) so the score expansion "
                    "is well-defined."
                )
            unique_clusters_full = np.unique(cluster_ids_full)
            G_full = int(len(unique_clusters_full))
            # Map fit-sample unique_clusters into positions in
            # unique_clusters_full via searchsorted (both arrays sorted by
            # np.unique). Verify the mapping is exact — otherwise the fit
            # sample contains PSU labels absent from the full domain (a
            # contract violation that should never occur under the upstream
            # `_inject_cluster_as_psu` invariant).
            fit_to_full_idx = np.searchsorted(unique_clusters_full, unique_clusters)
            if not np.array_equal(
                unique_clusters_full[fit_to_full_idx], np.asarray(unique_clusters)
            ):
                raise ValueError(
                    "_compute_gmm_variance: fit-sample unique cluster "
                    "labels are not a subset of full-domain cluster labels "
                    "(Wave E.3 parity invariant violated). This should be "
                    "impossible under `_inject_cluster_as_psu` — please "
                    "file an issue with a minimal reproducer."
                )
            c_by_cluster_full = np.zeros((G_full, p))
            s2_by_cluster_full = np.zeros((G_full, k))
            c_by_cluster_full[fit_to_full_idx] = c_by_cluster
            s2_by_cluster_full[fit_to_full_idx] = s2_by_cluster
            c_by_cluster = c_by_cluster_full
            s2_by_cluster = s2_by_cluster_full
            unique_clusters = unique_clusters_full
            G = G_full
            cluster_ids_for_lookup = np.asarray(cluster_ids_full)
        else:
            cluster_ids_for_lookup = cluster_ids

        # 4. S_g = gamma_hat' c_g - X'_{2g} eps_{2g}
        S = self._compute_gmm_scores(c_by_cluster, gamma_hat, s2_by_cluster)

        # 5. Meat: sum_g S_g S'_g = S' S
        _use_stratified_meat = resolved_survey is not None and (
            resolved_survey.strata is not None or resolved_survey.fpc is not None
        )
        if _use_stratified_meat:
            from diff_diff.survey import _compute_stratified_meat_from_psu_scores

            # Build PSU→stratum and PSU→FPC mappings from observation-level arrays.
            # cluster_ids_for_lookup is full-domain length under Wave E.3 parity
            # (score_pad_mask path) and fit-sample length otherwise; either way it
            # aligns with `resolved_survey.strata` / `resolved_survey.fpc` so the
            # obs_idx lookup resolves to the correct stratum / FPC value.
            G_meat = len(unique_clusters)

            # Strata: synthesize single stratum when strata is None (unstratified FPC)
            if resolved_survey.strata is not None:
                psu_strata = np.empty(G_meat, dtype=resolved_survey.strata.dtype)
                for idx, c in enumerate(unique_clusters):
                    obs_idx = np.where(cluster_ids_for_lookup == c)[0][0]
                    psu_strata[idx] = resolved_survey.strata[obs_idx]
            else:
                psu_strata = np.zeros(G_meat, dtype=int)

            # FPC: map observation-level FPC to PSU level
            psu_fpc = None
            if resolved_survey.fpc is not None:
                psu_fpc = np.empty(G_meat, dtype=np.float64)
                for idx, c in enumerate(unique_clusters):
                    obs_idx = np.where(cluster_ids_for_lookup == c)[0][0]
                    psu_fpc[idx] = resolved_survey.fpc[obs_idx]

            # Unstratified single-PSU: variance is unidentified (matches
            # _compute_stratified_psu_meat at survey.py:1225 which returns
            # zero meat with no variance_computed flag for n_psu < 2).
            # Under Wave E.3 parity, G_meat = G_full (post zero-pad), so the
            # gate fires on the full-domain PSU count, not the fit-sample.
            if resolved_survey.strata is None and G_meat < 2:
                return np.full((k, k), np.nan)

            # Reorder S rows to match unique_clusters ordering
            # S is built using np.add.at with cluster_indices from pd.factorize,
            # which uses the same order as unique_clusters from the data.
            meat, _var_computed, _legit_zero = _compute_stratified_meat_from_psu_scores(
                psu_scores=S,
                psu_strata=psu_strata,
                fpc_per_psu=psu_fpc,
                lonely_psu=resolved_survey.lonely_psu,
            )
            # If no variance was computed and no legitimate zeros, variance
            # is unidentified — return NaN VCV so caller gets NaN SE.
            if not _var_computed and _legit_zero == 0:
                return np.full((k, k), np.nan)
        else:
            with np.errstate(invalid="ignore", over="ignore"):
                meat = S.T @ S  # (k x k)

        # 6. Bread: (X'_2 W X_2)^{-1}
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            if survey_weights is not None:
                XtWX_2 = X_2.T @ (X_2 * survey_weights[:, None])
            else:
                XtWX_2 = X_2.T @ X_2
        # np.linalg.solve only raises on an *exactly* singular Gram; a *near*-
        # singular X_2'WX_2 would otherwise flow a garbage inverse (~1e13)
        # straight into the SE. `_rank_guarded_inv` truncates redundant
        # directions on the equilibrated Gram -> finite SE on the identified
        # subspace (NaN only at rank 0), matching the covariate IF rank-guard.
        # Sibling of finding #17 (axis A): the prior fallback fired only on an
        # exactly-singular matrix. X_2 is the Stage-2 indicator design (not user
        # covariates), so the diagnostic guidance points at that layer.
        bread, n_dropped, _, dropped = _rank_guarded_inv(XtWX_2, return_dropped=True)
        if n_dropped:
            warnings.warn(
                "Rank-deficient second-stage design matrix X_2'WX_2 in "
                "TwoStageDiD TSL variance; rank-reducing to a finite SE on the "
                f"identified subspace ({n_dropped} redundant direction(s) "
                "dropped, NaN if rank 0). The Stage-2 design is built from "
                "treatment, event-time, or group indicators, so this typically "
                "indicates a zero-weight or all-zero indicator column "
                "(e.g. an aggregation path with no qualifying observations).",
                UserWarning,
                stacklevel=2,
            )

        # 7. V = bread @ meat @ bread
        V = bread @ meat @ bread
        # A dropped (unidentified) Stage-2 coefficient is zero-filled in `bread`,
        # which would report se=0 for that named coefficient; NaN its row/col in
        # the FINAL vcov so per-coefficient SE extraction yields NaN (not 0).
        if dropped.any():
            V[dropped, :] = np.nan
            V[:, dropped] = np.nan
        return V

    def _build_fe_design(
        self,
        df: pd.DataFrame,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict[Any, int], Dict[Any, int]]:
        """
        Build sparse FE design matrices X_1 (all obs) and X_10 (untreated rows only).

        Column layout: [intercept, unit_1, ..., unit_{U-1}, time_1, ..., time_{T-1},
        cov_1, ..., cov_C] (drop first unit and first time for identification, with an
        intercept). The intercept makes the column space span the constant (the grand
        mean); the prior intercept-free layout silently omitted the grand mean from the
        FE span, which biased the GMM-sandwich residuals when re-solved exactly. With
        the intercept this is the standard full-rank two-way FE (matches fixest / R
        ``did2s``).

        X_10 is identical to X_1 except that rows for treated observations are zeroed out.

        Returns
        -------
        X_1_sparse : sparse.csr_matrix, shape (n, p)
        X_10_sparse : sparse.csr_matrix, shape (n, p)
        unit_to_idx : dict
        time_to_idx : dict
        """
        n = len(df)
        unit_vals = df[unit].values
        time_vals = df[time].values
        omega_0 = omega_0_mask.values

        all_units = np.unique(unit_vals)
        all_times = np.unique(time_vals)
        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        n_units = len(all_units)
        n_times = len(all_times)
        n_cov = len(covariates) if covariates else 0
        # [intercept, unit_1..unit_{U-1}, time_1..time_{T-1}] — the intercept (col 0)
        # makes the column space span the constant / grand mean (see docstring).
        n_fe_cols = 1 + (n_units - 1) + (n_times - 1)

        def _build_rows(mask=None):
            """Build sparse matrix for given observation mask."""
            all_rows = np.arange(n)

            # Intercept (col 0): 1 for every (masked) row.
            i_rows = all_rows if mask is None else all_rows[mask]
            i_cols = np.zeros(len(i_rows), dtype=int)

            # Unit dummies (drop first) at cols 1..n_units-1
            u_indices = np.array([unit_to_idx[u] for u in unit_vals])
            u_mask = u_indices > 0
            if mask is not None:
                u_mask = u_mask & mask

            u_rows = all_rows[u_mask]
            u_cols = u_indices[u_mask]  # 1..n_units-1 (intercept occupies col 0)

            # Time dummies (drop first) at cols n_units..n_units+n_times-2
            t_indices = np.array([time_to_idx[t] for t in time_vals])
            t_mask = t_indices > 0
            if mask is not None:
                t_mask = t_mask & mask

            t_rows = all_rows[t_mask]
            t_cols = n_units + t_indices[t_mask] - 1

            rows = np.concatenate([i_rows, u_rows, t_rows])
            cols = np.concatenate([i_cols, u_cols, t_cols])
            data = np.ones(len(rows))

            A_fe = sparse.csr_matrix((data, (rows, cols)), shape=(n, n_fe_cols))

            if n_cov > 0:
                cov_data = df[covariates].values.copy()
                if mask is not None:
                    cov_data[~mask] = 0.0
                A_cov = sparse.csr_matrix(cov_data)
                A = sparse.hstack([A_fe, A_cov], format="csr")
            else:
                A = A_fe

            return A

        X_1 = _build_rows(mask=None)
        X_10 = _build_rows(mask=omega_0)

        return X_1, X_10, unit_to_idx, time_to_idx

    @staticmethod
    def _exact_gmm_residuals(
        X_1_sparse,
        theta_exact: np.ndarray,
        y_vals_clean: np.ndarray,
        identified: np.ndarray,
        omega_0: np.ndarray,
        y_tilde: np.ndarray,
        X_2: np.ndarray,
        survey_weights: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact Stage-1 / Stage-2 residuals for the GMM influence function.

        Given the EXACT Stage-1 FE coefficients ``theta_exact`` (solved from the
        same ``(X'_{10} W X_{10})`` factorization used for ``gamma_hat``), return the
        exact Stage-1 residual ``eps_10`` (untreated rows) and the exact Stage-2
        residual ``eps_2``. **Shared** by the analytical GMM variance
        (``TwoStageDiD._compute_gmm_variance``) and the multiplier bootstrap
        (``_compute_cluster_S_scores``) so both build the per-cluster influence
        score ``S_g = gamma_hat' c_g - X'_{2g} eps_{2g}`` from the same exact
        residuals. The iterative alternating-projection FE used for the point
        estimate is only ~1e-7-accurate on unbalanced untreated panels, which
        perturbs the variance ~1% relative to the analytical sandwich; obs whose FE
        are unidentified (rank-deficient / Proposition-5) fall back to the iterative
        residual ``y_tilde`` so those edge cases are unchanged.
        """
        n = X_1_sparse.shape[0]
        fitted_exact = np.asarray(X_1_sparse @ theta_exact).ravel()
        y_tilde_exact = y_vals_clean - fitted_exact
        use_exact = identified & np.isfinite(y_tilde_exact)
        y_tilde_use = np.where(use_exact, y_tilde_exact, y_tilde)
        eps_10 = np.empty(n)
        eps_10[omega_0] = y_tilde_use[omega_0]  # exact Stage-1 residual (untreated)
        eps_10[~omega_0] = y_vals_clean[~omega_0]  # x_{10i} = 0, so value is inert
        # Exact Stage-2 residual: re-solve delta on the exact residualized outcome
        # (X_2 already has NaN-y_tilde rows zeroed by the caller, so masked obs
        # contribute nothing to the normal equations).
        y_tilde_s2 = np.where(np.isfinite(y_tilde_use), y_tilde_use, 0.0)
        if survey_weights is not None:
            XtWX2 = X_2.T @ (X_2 * survey_weights[:, None])
            XtWy2 = X_2.T @ (survey_weights * y_tilde_s2)
        else:
            XtWX2 = X_2.T @ X_2
            XtWy2 = X_2.T @ y_tilde_s2
        try:
            delta_2 = np.linalg.solve(XtWX2, XtWy2)
        except np.linalg.LinAlgError:
            # Silent-failure audit convention: warn before the dense fallback.
            warnings.warn(
                "TwoStageDiD GMM sandwich: Stage-2 design (X'_2 W X_2) is "
                "singular; falling back to dense lstsq for the exact-residual "
                "re-solve. This may indicate collinear treatment/horizon "
                "indicators and SE estimates may be less reliable.",
                UserWarning,
                stacklevel=2,
            )
            delta_2 = np.linalg.lstsq(XtWX2, XtWy2, rcond=None)[0]
        eps_2 = y_tilde_s2 - X_2 @ delta_2
        return eps_10, eps_2

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

    def _replay_replicate_inference(
        self,
        *,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        omega_1_mask: pd.Series,
        cluster_var: str,
        treatment_groups: List[Any],
        ref_period: int,
        balance_e: Optional[int],
        keep_mask: pd.Series,
        resolved_survey: Any,
        overall_att: float,
        event_study_effects: Optional[Dict[int, Dict[str, Any]]],
        group_effects: Optional[Dict[Any, Dict[str, Any]]],
        survey_df_seed: Optional[int],
    ) -> Tuple[np.ndarray, int, Optional[int]]:
        """Replicate-weight inference override for the aggregation surfaces.

        Extracted verbatim from ``fit()``'s replicate block so the post-fit
        ``aggregate()`` path can replay it from the kit payload. The stacked
        layout is ``[overall, es..., grp...]`` built from whichever family
        dicts are non-None. ``compute_replicate_refit_variance`` validates
        replicates JOINTLY (all-finite rows), so inference is LEVEL-MATCHED:
        post-fit ``aggregate('event_study')`` replays ``[overall, ES]`` and
        reproduces ``fit(aggregate='event_study')`` exactly; a
        ``fit(aggregate='all')`` surface is NOT the equivalence target when
        a replicate NaNs on exactly one family's targets.

        Overrides ``se``/``t_stat``/``p_value``/``conf_int`` IN PLACE on the
        passed dicts and returns ``(vcov, n_valid, survey_df_final)``. The
        fit-only tail -- the overall-row quintet and the
        ``survey_metadata.df_survey`` write -- deliberately stays in
        ``fit()``: this method must never mutate results-owned metadata
        (post-fit ``aggregate()`` calls it on kit refs and a throwaway
        host). Cost: one full per-replicate two-stage refit pass per call
        (R refits + R frame copies), the same work the fit-time path does.
        """
        from diff_diff.survey import compute_replicate_refit_variance

        # Derive keys from actual outputs (excludes filtered/Prop5 horizons)
        _es_effects_ts = event_study_effects or {}
        _grp_effects_ts = group_effects or {}
        _sorted_es_periods_ts = sorted(
            e for e in _es_effects_ts.keys() if np.isfinite(_es_effects_ts[e]["effect"])
        )
        _sorted_groups_ts = sorted(
            g for g in _grp_effects_ts.keys() if np.isfinite(_grp_effects_ts[g]["effect"])
        )
        _n_es_ts = len(_sorted_es_periods_ts)
        _n_grp_ts = len(_sorted_groups_ts)

        # Build full-sample estimate from actual outputs
        _full_est_ts = [overall_att]
        _full_est_ts.extend([_es_effects_ts[e]["effect"] for e in _sorted_es_periods_ts])
        _full_est_ts.extend([_grp_effects_ts[g]["effect"] for g in _sorted_groups_ts])

        def _refit_ts(w_r):
            # Wave E.3 parity (PR #482 SpilloverDiD precedent): the main fit
            # path keeps `resolved_survey` at full-domain length but subsets
            # `survey_weights` for stage-1 / stage-2 OLS arithmetic via
            # `keep_mask` (always-treated drop). The replicate refit
            # callback receives a FULL-DOMAIN replicate weight `w_r`
            # (sourced from `resolved_survey.replicate_weights` which is
            # also full-domain) and must apply the SAME `keep_mask`
            # subsetting before threading through stage-1 / stage-2,
            # otherwise `solve_ols` rejects the length mismatch
            # (full-domain w_r vs post-drop df) and the ValueError is
            # swallowed by `compute_replicate_refit_variance` →
            # NaN replicate inference.
            w_r_fit = np.asarray(w_r)[keep_mask.values]
            ufe_r, tfe_r, gm_r, delta_r, kcm_r = self._fit_untreated_model(
                df,
                outcome,
                unit,
                time,
                covariates,
                omega_0_mask,
                weights=w_r_fit,
            )
            y_tilde_r = self._residualize(
                df,
                outcome,
                unit,
                time,
                covariates,
                ufe_r,
                tfe_r,
                gm_r,
                delta_r,
            )
            df_tmp = df.copy()
            df_tmp["_y_tilde"] = y_tilde_r
            results = []

            att_r, _ = self._stage2_static(
                df=df_tmp,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                unit_fe=ufe_r,
                time_fe=tfe_r,
                grand_mean=gm_r,
                delta_hat=delta_r,
                cluster_var=cluster_var,
                kept_cov_mask=kcm_r,
                survey_weights=w_r_fit,
                survey_weight_type="pweight",
                warn_nan=False,
            )
            results.append(att_r)

            if _sorted_es_periods_ts:
                # Replicate refits only need the point effects; the
                # per-replicate V is irrelevant to the refit variance.
                es_r, _, _ = self._stage2_event_study(
                    df=df_tmp,
                    unit=unit,
                    time=time,
                    first_treat=first_treat,
                    covariates=covariates,
                    omega_0_mask=omega_0_mask,
                    omega_1_mask=omega_1_mask,
                    unit_fe=ufe_r,
                    time_fe=tfe_r,
                    grand_mean=gm_r,
                    delta_hat=delta_r,
                    cluster_var=cluster_var,
                    treatment_groups=treatment_groups,
                    ref_period=ref_period,
                    balance_e=balance_e,
                    kept_cov_mask=kcm_r,
                    survey_weights=w_r_fit,
                    survey_weight_type="pweight",
                    survey_df=None,
                    warn_nan=False,
                )
                for e in _sorted_es_periods_ts:
                    results.append(es_r[e]["effect"] if e in es_r else np.nan)

            if _sorted_groups_ts:
                grp_r = self._stage2_group(
                    df=df_tmp,
                    unit=unit,
                    time=time,
                    first_treat=first_treat,
                    covariates=covariates,
                    omega_0_mask=omega_0_mask,
                    omega_1_mask=omega_1_mask,
                    unit_fe=ufe_r,
                    time_fe=tfe_r,
                    grand_mean=gm_r,
                    delta_hat=delta_r,
                    cluster_var=cluster_var,
                    treatment_groups=treatment_groups,
                    kept_cov_mask=kcm_r,
                    survey_weights=w_r_fit,
                    survey_weight_type="pweight",
                    survey_df=None,
                    warn_nan=False,
                )
                for g in _sorted_groups_ts:
                    results.append(grp_r[g]["effect"] if g in grp_r else np.nan)

            return np.array(results)

        _vcov_rep_ts, _n_valid_rep_ts = compute_replicate_refit_variance(
            _refit_ts, np.array(_full_est_ts), resolved_survey
        )

        # Override df if replicates were dropped
        # Replicate-refit path is only reached with a resolved design.
        assert resolved_survey is not None
        survey_df_final = survey_df_seed
        if _n_valid_rep_ts < resolved_survey.n_replicates:
            survey_df_final = _n_valid_rep_ts - 1 if _n_valid_rep_ts > 1 else 0

        # Override event-study SEs (only for identified effects)
        for i, e in enumerate(_sorted_es_periods_ts):
            if event_study_effects is not None and e in event_study_effects:
                se_e = float(np.sqrt(max(_vcov_rep_ts[1 + i, 1 + i], 0.0)))
                eff_e = event_study_effects[e]["effect"]
                t_e, p_e, ci_e = safe_inference(eff_e, se_e, alpha=self.alpha, df=survey_df_final)
                event_study_effects[e]["se"] = se_e
                event_study_effects[e]["t_stat"] = t_e
                event_study_effects[e]["p_value"] = p_e
                event_study_effects[e]["conf_int"] = ci_e

        # Override group SEs (only for identified effects)
        for j, g in enumerate(_sorted_groups_ts):
            if group_effects is not None and g in group_effects:
                se_g = float(np.sqrt(max(_vcov_rep_ts[1 + _n_es_ts + j, 1 + _n_es_ts + j], 0.0)))
                eff_g = group_effects[g]["effect"]
                t_g, p_g, ci_g = safe_inference(eff_g, se_g, alpha=self.alpha, df=survey_df_final)
                group_effects[g]["se"] = se_g
                group_effects[g]["t_stat"] = t_g
                group_effects[g]["p_value"] = p_g
                group_effects[g]["conf_int"] = ci_g

        return _vcov_rep_ts, _n_valid_rep_ts, survey_df_final
