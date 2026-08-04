"""Aggregation + Theorem-3 variance engine for the BJS imputation estimator.

Extracted verbatim from ``diff_diff/imputation.py`` for the M-021/M-118
post-fit ``aggregate()`` migration: ``diff_diff/imputation.py`` imports
``imputation_results.py`` (and ``imputation_bootstrap.py`` imports it too),
so the results module can import neither -- the shared machinery lives here,
an import-leaf module both sides can reach (the
``efficient_did_aggregation.py`` precedent).

Contents:

- module helpers ``_compute_target_weights`` (lifted from
  ``imputation_bootstrap.py`` -- the bootstrap module re-imports it; moving
  it here keeps the module graph acyclic), ``_UntreatedProjection``,
  ``_LSMRUnconvergedError`` and ``_lsmr_minnorm_normal_solve``;
- :class:`_ImputationAggregationMixin` -- the event-study / group
  aggregators, the Theorem-3 conservative-variance stack they recompute
  through, the pretrends lead regression, and the replicate-weight
  inference override replay. Inherited by ``ImputationDiD`` (fit-time
  behavior byte-identical) and hosted post-fit by the throwaway
  ``_ImputationKitAggregator`` (``imputation_results.py``).
"""

import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import factorized as sparse_factorized

from diff_diff.linalg import solve_ols
from diff_diff.utils import (
    _iterative_fe_solve,
    absorbed_fe_cr1_k_increment,
    absorbed_fe_rank,
    demean_by_groups,
    pre_demean_norms,
    resolve_tail_df,
    safe_inference,
    snap_absorbed_regressors,
)


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


class _UntreatedProjection(NamedTuple):
    """Cached, target-invariant pieces of the untreated imputation projection
    ``v_untreated = -A_0 (A_0' [W] A_0)^{-1} A_1' w`` (BJS 2024 Theorem 3).

    Within a single ``fit()`` the untreated design (``df_0``/``df_1``, covariates,
    survey weights) is identical across every estimand target (overall ATT, each
    event-study horizon, each group, and the bootstrap precompute) -- only the
    treated aggregation ``weights`` (the RHS ``A_1' w``) vary. So ``A_0``, ``A_1``
    and the factorization of ``A_0'[W]A_0`` are built once and reused across
    targets (factorize-once / solve-many), mirroring the TwoStageDiD GMM-sandwich
    ``sparse_factorized`` pattern.
    """

    A_0: sparse.csr_matrix
    A_1: sparse.csr_matrix
    # solver(rhs) -> z; None when the factorization was exactly singular (the
    # solve path then routes to the sparse LSMR least-squares fallback).
    solver: Optional[Callable[[np.ndarray], np.ndarray]]
    A0tA0_csc: sparse.csc_matrix  # retained for the LSMR fallback
    survey_weights_0: Optional[np.ndarray]
    singular: bool


class _LSMRUnconvergedError(RuntimeError):
    """LSMR failed to certify a solution on the singular-variance fallback.

    Raised (not returned as NaN) so the variance boundary can fail closed:
    a NaN vector would be laundered into zeros by the missing-FE
    ``nan_to_num`` in the psi product — producing a finite, WRONG variance —
    whereas this exception is caught in ``_compute_conservative_variance``
    and converted to a NaN SE (the all-or-nothing NaN inference convention).
    """


def _lsmr_minnorm_normal_solve(A0tA0_csc, rhs: np.ndarray) -> np.ndarray:
    """Least-squares solve of the (possibly singular) normal equations
    ``(A_0'[W]A_0) z = rhs`` WITHOUT densifying the sparse matrix.

    Replaces the previous ``np.linalg.lstsq(A0tA0.toarray(), ...)`` fallback,
    whose dense materialization scales ``O((U+T+K)^2)`` — an OOM risk on
    large panels (the TODO row this resolves). ``scipy.sparse.linalg.lsmr``
    handles singular symmetric systems, converging to the minimum-norm
    least-squares solution (the same solution family as ``lstsq``'s
    pseudo-inverse solution).

    Solver choice cannot change the estimator output: any two least-squares
    solutions differ by a ``null(A_0'[W]A_0) = null(sqrt(W) A_0)`` component,
    which the downstream projection ``v_untreated = -[W_0] A_0 z``
    annihilates (unweighted: ``null = null(A_0)`` so ``A_0 z`` is invariant;
    weighted: the weight multiplication zeroes exactly the rows where the
    null component can be nonzero). Locked by the singular-system parity
    test against a dense-lstsq oracle.

    CONVERGENCE IS VALIDATED (fail-closed): ``istop`` in ``{0, 1, 2, 4, 5}``
    means LSMR certified an (approximate) solution / least-squares solution
    within ``atol``/``btol`` (4 and 5 are the machine-precision analogues of
    1 and 2 per SciPy's documentation); anything else (condition-limit stop,
    max-iteration exhaustion) gets ONE retry with an uncapped condition
    limit and a generous iteration budget, and if still uncertified raises
    :class:`_LSMRUnconvergedError` — caught at the variance boundary and
    converted to a NaN SE — rather than feeding a finite-but-unverified
    solution into the Theorem 3 weights.
    """
    import scipy.sparse.linalg as spla

    _certified = (0, 1, 2, 4, 5)
    result = spla.lsmr(A0tA0_csc, rhs, atol=1e-14, btol=1e-14)
    z, istop = result[0], int(result[1])
    if istop not in _certified or not np.all(np.isfinite(z)):
        dim = A0tA0_csc.shape[0]
        result = spla.lsmr(
            A0tA0_csc, rhs, atol=1e-14, btol=1e-14, conlim=1e16, maxiter=max(50 * dim, 10_000)
        )
        z, istop = result[0], int(result[1])
        if istop not in _certified or not np.all(np.isfinite(z)):
            warnings.warn(
                "ImputationDiD variance: the LSMR fallback solve of "
                f"(A_0'[W]A_0) z = rhs did not converge (istop={istop}); "
                "the affected variance is reported as NaN rather than from "
                "an unverified solution.",
                UserWarning,
                stacklevel=3,
            )
            raise _LSMRUnconvergedError(f"LSMR uncertified (istop={istop})")
    return z


class _ImputationAggregationMixin:
    """Shared aggregation/variance methods (moved verbatim from ``ImputationDiD``).

    HOST-ATTRIBUTE CONTRACT -- the complete ``self.`` surface the moved
    methods read (typed class-level declarations, not docstring prose:
    ``mypy diff_diff`` at zero errors needs the attributes declared on the
    mixin for both hosts). Zero methods WRITE to ``self`` -- the post-fit
    throwaway host exists for estimator-mutation isolation only.
    """

    alpha: float
    anticipation: int
    horizon_max: Optional[int]
    pretrends: bool
    aux_partition: str
    leave_one_out: bool
    rank_deficient_action: str
    df_convention: str

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
        Estimate unit and time FE via iterative alternating projection (Gauss-Seidel).

        Thin wrapper over the shared bincount solver
        (``diff_diff.utils._iterative_fe_solve``): factorize unit/time once,
        solve on integer codes, map the level arrays back to dicts.
        Converges to the exact (W)LS solution for balanced and unbalanced
        panels; balanced panels converge in 1-2 iterations.

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
                "ImputationDiD: unit or time column contains NaN. Drop or "
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
            method_name="ImputationDiD iterative FE solver",
        )
        unit_fe = dict(zip(unit_uniques, unit_fe_arr))
        time_fe = dict(zip(time_uniques, time_fe_arr))
        return unit_fe, time_fe

    @staticmethod
    def _compute_balanced_cohort_mask(
        df_treated: pd.DataFrame,
        first_treat: str,
        all_horizons: List[int],
        balance_e: int,
        cohort_rel_times: Dict[Any, Set[int]],
    ) -> np.ndarray:
        """Compute boolean mask selecting treated obs from balanced cohorts.

        A cohort is 'balanced' if it has observations at every relative time
        in [-balance_e, max(all_horizons)].

        Parameters
        ----------
        df_treated : pd.DataFrame
            Post-treatment observations (Omega_1).
        first_treat : str
            Column name for cohort identifier.
        all_horizons : list of int
            Post-treatment horizons in the event study.
        balance_e : int
            Number of pre-treatment periods to require.
        cohort_rel_times : dict
            Maps each cohort value to the set of all observed relative times
            (including pre-treatment) from the full panel. Built by
            _build_cohort_rel_times().
        """
        if not all_horizons:
            return np.ones(len(df_treated), dtype=bool)

        max_h = max(all_horizons)
        required_range = set(range(-balance_e, max_h + 1))

        balanced_cohorts = set()
        for g, horizons in cohort_rel_times.items():
            if required_range.issubset(horizons):
                balanced_cohorts.add(g)

        return df_treated[first_treat].isin(balanced_cohorts).values

    @staticmethod
    def _build_cohort_rel_times(
        df: pd.DataFrame,
        first_treat: str,
    ) -> Dict[Any, Set[int]]:
        """Build mapping of cohort -> set of observed relative times from full panel.

        Precondition: df must have '_never_treated' and '_rel_time' columns
        (set by fit() before any aggregation calls).
        """
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
        Step 1: Estimate unit + time FE on untreated observations.

        Uses iterative alternating projection (Gauss-Seidel) to compute exact
        OLS fixed effects for both balanced and unbalanced panels. For balanced
        panels, converges in 1-2 iterations (identical to one-pass demeaning).

        Parameters
        ----------
        weights : np.ndarray, optional
            Full-panel survey weights (same length as df). The untreated subset
            is extracted internally via omega_0_mask. When None, unweighted.

        Returns
        -------
        unit_fe : dict
            Unit fixed effects {unit_id: alpha_i}.
        time_fe : dict
            Time fixed effects {time_period: beta_t}.
        grand_mean : float
            Grand mean (0.0 — absorbed into iterative FE).
        delta_hat : np.ndarray or None
            Covariate coefficients (if covariates provided).
        kept_cov_mask : np.ndarray or None
            Boolean mask of shape (n_covariates,) indicating which covariates
            have finite coefficients. None if no covariates.
        """
        df_0 = df.loc[omega_0_mask]
        w_0 = weights[omega_0_mask.values] if weights is not None else None

        if covariates is None or len(covariates) == 0:
            # No covariates: estimate FE via iterative alternating projection
            # (exact OLS for both balanced and unbalanced panels)
            y = df_0[outcome].values.copy()
            unit_fe, time_fe = self._iterative_fe(
                y, df_0[unit].values, df_0[time].values, df_0.index, weights=w_0
            )
            # grand_mean = 0: iterative FE absorb the intercept
            return unit_fe, time_fe, 0.0, None, None

        else:
            # With covariates: iteratively demean Y and X, OLS for delta,
            # then recover FE from covariate-adjusted outcome
            y = df_0[outcome].values.copy()
            X_raw = df_0[covariates].values.copy()
            units = df_0[unit].values
            times = df_0[time].values

            # Step A: within-transform Y and all X columns through the shared
            # MAP engine (factorize-once + bincount + optional Rust kernel),
            # one dispatch for every column. within_transform pins
            # [unit, time]; [time, unit] here preserves the historical
            # time-then-unit sweep order of the per-estimator loops.
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

            # Step B: OLS for covariate coefficients on demeaned data
            result = solve_ols(
                X_dm,
                y_dm,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
                column_names=covariates,
                weights=w_0,
            )
            delta_hat = result[0]

            # Mask of covariates with finite coefficients (before cleaning)
            # Used to exclude rank-deficient covariates from variance design matrices
            kept_cov_mask = np.isfinite(delta_hat)

            # Replace NaN coefficients with 0 for adjustment
            # (rank-deficient covariates are dropped)
            delta_hat_clean = np.where(np.isfinite(delta_hat), delta_hat, 0.0)

            # Step C: Recover FE from covariate-adjusted outcome using iterative FE
            y_adj = y - np.dot(X_raw, delta_hat_clean)
            unit_fe, time_fe = self._iterative_fe(y_adj, units, times, df_0.index, weights=w_0)

            # grand_mean = 0: iterative FE absorb the intercept
            return unit_fe, time_fe, 0.0, delta_hat_clean, kept_cov_mask

    def _impute_treatment_effects(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_1_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 2: Impute Y(0) for treated observations and compute tau_hat.

        Returns
        -------
        tau_hat : np.ndarray
            Imputed treatment effects for each treated observation.
        y_hat_0 : np.ndarray
            Imputed counterfactual Y(0).
        """
        df_1 = df.loc[omega_1_mask]

        # Look up unit and time FE
        alpha_i = df_1[unit].map(unit_fe).values
        beta_t = df_1[time].map(time_fe).values

        # Handle missing FE (set to NaN)
        alpha_i = np.where(pd.isna(alpha_i), np.nan, alpha_i).astype(float)
        beta_t = np.where(pd.isna(beta_t), np.nan, beta_t).astype(float)

        y_hat_0 = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            X_1 = df_1[covariates].values
            y_hat_0 = y_hat_0 + np.dot(X_1, delta_hat)

        tau_hat = df_1[outcome].values - y_hat_0

        return tau_hat, y_hat_0

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
        proj_cache: Optional[Dict[Any, _UntreatedProjection]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute cluster-level influence function sums (Theorem 3).

        psi_i = sum_t v_it * epsilon_tilde_it, summed within each cluster.

        Returns
        -------
        cluster_psi_sums : np.ndarray
            Array of cluster-level psi sums.
        cluster_ids_unique : np.ndarray
            Unique cluster identifiers (matching order of psi sums).
        """
        df_0 = df.loc[omega_0_mask]
        df_1 = df.loc[omega_1_mask]

        # ---- Compute v_it for treated observations ----
        v_treated = weights.copy()

        # ---- Compute v_it for untreated observations ----
        # Exact two-way-FE imputation projection
        # v_untreated = -A_0 (A_0' [W] A_0)^{-1} A_1' w_treated  (Theorem 3 / the
        # implied weights of Supplementary Proposition A3), used for BOTH the
        # FE-only and the covariate case. The earlier FE-only closed form
        # -(w_i/n0_i + w_t/n0_t - w/N_0) is exact only for a *balanced* untreated
        # panel; Omega_0 is generically unbalanced in staggered designs (treated
        # observations are removed), which biased the analytical SE downward
        # (~27% on the parity panel). The projection matches R `didimputation`
        # exactly -- see tests/test_methodology_imputation.py::TestImputationDiDParityR.
        # Build the target-invariant projection design + factorization once per
        # fit() (cached in proj_cache), then solve only the target-specific RHS.
        # survey_weights is DELIBERATELY excluded from the key: the cache is a
        # fit-LOCAL dict, and within one fit() survey_weights is a single fixed
        # object, so the masks deterministically map to one sw_0 =
        # survey_weights[omega_0_mask]. The masks + covariates + kept_cov_mask
        # therefore FULLY identify the design (sw_0 itself is a fresh-sliced array
        # per call -- keying on its id() would miss every time and balloon the
        # cache to 1+H+G full A_0/A_1/factorization entries). id()-keys are safe:
        # the masks are fit() locals alive for the whole fit and the cache is a
        # fit-local dict, so no cross-fit leak / id reuse.
        cov_list = covariates if covariates is not None else []
        ctx: Optional[_UntreatedProjection] = None
        if proj_cache is not None:
            key = (
                id(omega_0_mask),
                id(omega_1_mask),
                tuple(cov_list),
                kept_cov_mask.tobytes() if kept_cov_mask is not None else None,
            )
            ctx = proj_cache.get(key)
        if ctx is None:
            ctx = self._build_untreated_projection(
                df_0,
                df_1,
                unit,
                time,
                cov_list,
                kept_cov_mask=kept_cov_mask,
                survey_weights_0=survey_weights_0,
            )
            if proj_cache is not None:
                proj_cache[key] = ctx
        v_untreated = self._solve_untreated_v(ctx, weights)

        # ---- Compute auxiliary model residuals (Equation 8) ----
        epsilon_treated = self._compute_auxiliary_residuals_treated(
            df_1,
            outcome,
            unit,
            time,
            first_treat,
            covariates,
            unit_fe,
            time_fe,
            grand_mean,
            delta_hat,
            v_treated,
        )
        epsilon_untreated = self._compute_residuals_untreated(
            df_0, outcome, unit, time, covariates, unit_fe, time_fe, grand_mean, delta_hat
        )

        # ---- psi_it = v_it * epsilon_tilde_it ----
        v_all = np.empty(len(df))
        v_all[omega_1_mask.values] = v_treated
        v_all[omega_0_mask.values] = v_untreated

        eps_all = np.empty(len(df))
        eps_all[omega_1_mask.values] = epsilon_treated
        eps_all[omega_0_mask.values] = epsilon_untreated

        ve_product = v_all * eps_all
        # NaN eps from missing FE (rank condition violation). Zero their variance
        # contribution — matches R's did_imputation which drops unimputable obs.
        np.nan_to_num(ve_product, copy=False, nan=0.0)

        # Sum within clusters
        cluster_ids = df[cluster_var].values
        ve_series = pd.Series(ve_product, index=df.index)
        cluster_sums = ve_series.groupby(cluster_ids).sum()

        return cluster_sums.values, cluster_sums.index.values, ve_product

    def _compute_conservative_variance(
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
        survey_weights: Optional[np.ndarray] = None,
        resolved_survey=None,
        proj_cache: Optional[Dict[Any, _UntreatedProjection]] = None,
    ) -> float:
        """
        Compute conservative clustered variance (Theorem 3, Equation 7).

        Parameters
        ----------
        weights : np.ndarray
            Aggregation weights w_it for treated observations.
            Shape: (n_treated,), must sum to 1.
        survey_weights : np.ndarray, optional
            Full-panel survey weights. When provided, they enter the untreated
            v_it WLS projection (weighted normal equations plus the left
            per-observation weight factor) and the design-based variance path.
        resolved_survey : ResolvedSurveyDesign, optional
            When provided, uses design-based variance via
            ``compute_survey_if_variance()`` (supports strata, PSU, FPC).

        Returns
        -------
        float
            Standard error.
        """
        sw_0 = survey_weights[omega_0_mask.values] if survey_weights is not None else None
        try:
            cluster_psi_sums, _, ve_product = self._compute_cluster_psi_sums(
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
                weights=weights,
                cluster_var=cluster_var,
                kept_cov_mask=kept_cov_mask,
                survey_weights_0=sw_0,
                proj_cache=proj_cache,
            )
        except _LSMRUnconvergedError:
            # Solver failure is GLOBAL (the untreated projection is invalid),
            # unlike per-observation missing-FE NaNs — fail the whole SE
            # closed instead of letting nan_to_num launder it to zeros.
            return np.nan

        if resolved_survey is not None:
            # Design-based variance with strata/PSU/FPC support
            from diff_diff.survey import compute_survey_if_variance

            variance = compute_survey_if_variance(ve_product, resolved_survey)
            if np.isnan(variance):
                return np.nan
            return np.sqrt(max(variance, 0.0))

        sigma_sq = float((cluster_psi_sums**2).sum())
        return np.sqrt(max(sigma_sq, 0.0))

    def _build_untreated_projection(
        self,
        df_0: pd.DataFrame,
        df_1: pd.DataFrame,
        unit: str,
        time: str,
        covariates: List[str],
        kept_cov_mask: Optional[np.ndarray] = None,
        survey_weights_0: Optional[np.ndarray] = None,
    ) -> _UntreatedProjection:
        """
        Build the target-INVARIANT pieces of the exact imputation projection
        ``v_untreated = -A_0 (A_0' [W] A_0)^{-1} A_1' w_treated`` and factorize the
        normal-equations matrix once. The result is cached per ``fit()`` (see
        ``_compute_cluster_psi_sums``) and reused across all estimand targets;
        only the target-specific RHS ``A_1' w`` is solved per target in
        ``_solve_untreated_v``.

        This is the GENERAL path -- used for both the FE-only and the covariate
        cases (an empty ``covariates`` list builds a pure two-way-FE design;
        ``n_cov == 0`` is the FE-only path). When survey_weights_0 is provided,
        uses the weighted normal equations ``A_0' W A_0`` (the per-observation
        survey weight is reapplied to the solved v in ``_solve_untreated_v``).

        Uses scipy.sparse for FE dummy columns to reduce memory from O(N*(U+T))
        to O(N) for the FE portion. An exactly singular ``A_0'[W]A_0`` makes
        ``sparse_factorized`` raise ``RuntimeError``; we emit a UserWarning (once
        per fit) and record ``singular=True`` so the solve routes to the sparse
        LSMR least-squares fallback (no dense materialization; see
        :func:`_lsmr_minnorm_normal_solve`).
        """
        # Exclude rank-deficient covariates from design matrices
        if kept_cov_mask is not None and not np.all(kept_cov_mask):
            covariates = [c for c, k in zip(covariates, kept_cov_mask) if k]

        units_0 = df_0[unit].values
        times_0 = df_0[time].values
        units_1 = df_1[unit].values
        times_1 = df_1[time].values

        all_units = np.unique(np.concatenate([units_0, units_1]))
        all_times = np.unique(np.concatenate([times_0, times_1]))
        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        n_units = len(all_units)
        n_times = len(all_times)
        n_cov = len(covariates)
        # Two-way FE design = all unit dummies (their sum spans the intercept) +
        # time dummies dropping the first (identification). Dropping the first
        # unit dummy too -- with no intercept column -- would omit the baseline
        # level dimension and project onto a space one rank short of the true
        # two-way-FE span, biasing the imputation weights (and hence the SE).
        n_fe_cols = n_units + (n_times - 1)

        def _build_A_sparse(df_sub, unit_vals, time_vals):
            n = len(df_sub)

            # Unit dummies — keep ALL (together they span the intercept).
            u_indices = np.array([unit_to_idx[u] for u in unit_vals])
            u_rows = np.arange(n)
            u_cols = u_indices

            # Time dummies (drop first) — vectorized
            t_indices = np.array([time_to_idx[t] for t in time_vals])
            t_mask = t_indices > 0
            t_rows = np.arange(n)[t_mask]
            t_cols = n_units + (t_indices[t_mask] - 1)

            rows = np.concatenate([u_rows, t_rows])
            cols = np.concatenate([u_cols, t_cols])
            data = np.ones(len(rows))

            A_fe = sparse.csr_matrix((data, (rows, cols)), shape=(n, n_fe_cols))

            # Covariates (dense, typically few columns)
            if n_cov > 0:
                A_cov = sparse.csr_matrix(df_sub[covariates].values)
                A = sparse.hstack([A_fe, A_cov], format="csr")
            else:
                A = A_fe

            return A

        A_0 = _build_A_sparse(df_0, units_0, times_0)
        A_1 = _build_A_sparse(df_1, units_1, times_1)

        # Form (A_0' [W] A_0). When survey weights present, use the weighted
        # normal equations A_0' W A_0.
        if survey_weights_0 is not None:
            A0tA0_sparse = A_0.T @ A_0.multiply(survey_weights_0[:, None])
        else:
            A0tA0_sparse = A_0.T @ A_0  # stays sparse
        A0tA0_csc = A0tA0_sparse.tocsc()

        # Factorize once (factorize-once / solve-many). An exactly singular
        # matrix makes sparse_factorized raise RuntimeError -- the same condition
        # that previously surfaced as spsolve's MatrixRankWarning -> non-finite
        # solution. Warn once and fall back to the sparse LSMR least-squares
        # solve per target (no dense materialization). (The factorized path is
        # bit-identical to the prior per-target spsolve for a single dense
        # RHS -- both use the SuperLU simple driver with the same defaults.)
        try:
            solver: Optional[Callable[[np.ndarray], np.ndarray]] = sparse_factorized(A0tA0_csc)
            singular = False
        except RuntimeError as exc:
            # Silent-failure audit axis C: emit a UserWarning on fallback instead
            # of swallowing the error. Keep the "sparse LSMR" substring (asserted
            # by tests).
            warnings.warn(
                "ImputationDiD variance: sparse factorization of (A_0' [W] A_0) "
                f"failed ({type(exc).__name__}); falling back to a sparse LSMR "
                "least-squares solve (no dense materialization). This may "
                "indicate a rank-deficient or near-singular normal-equations "
                "matrix and variance estimates may be less reliable.",
                UserWarning,
                stacklevel=2,
            )
            solver = None
            singular = True

        return _UntreatedProjection(
            A_0=A_0,
            A_1=A_1,
            solver=solver,
            A0tA0_csc=A0tA0_csc,
            survey_weights_0=survey_weights_0,
            singular=singular,
        )

    def _solve_untreated_v(self, ctx: _UntreatedProjection, weights: np.ndarray) -> np.ndarray:
        """
        Solve the target-SPECIFIC RHS of the untreated imputation projection using
        the cached design + factorization in ``ctx``:
        ``v_untreated = -[W_0] A_0 (A_0'[W]A_0)^{-1} A_1' w_treated``.
        """
        A1_w = ctx.A_1.T @ weights  # (p,)

        if ctx.singular:
            # Factorization was singular at build time (warned once already).
            z = _lsmr_minnorm_normal_solve(ctx.A0tA0_csc, A1_w)
        else:
            assert ctx.solver is not None
            z = ctx.solver(A1_w)
            if not np.all(np.isfinite(z)):
                # Defensive, target-specific: a non-finite solve on an otherwise
                # factorizable matrix routes this RHS to the LSMR fallback. Warn per
                # target (silent-failure audit axis C) -- distinct from the
                # once-per-fit build-time singular warning.
                warnings.warn(
                    "ImputationDiD variance: sparse solve of (A_0' [W] A_0) z = "
                    "A_1' w returned a non-finite solution; falling back to a "
                    "sparse LSMR least-squares solve for this target. Variance "
                    "estimates may be less reliable.",
                    UserWarning,
                    stacklevel=2,
                )
                z = _lsmr_minnorm_normal_solve(ctx.A0tA0_csc, A1_w)

        # v_untreated = -[W_0] A_0 z (WLS projection requires per-obs weight)
        v_untreated = -(ctx.A_0 @ z)
        if ctx.survey_weights_0 is not None:
            v_untreated = v_untreated * ctx.survey_weights_0
        return v_untreated

    def _compute_auxiliary_residuals_treated(
        self,
        df_1: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
        v_treated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute auxiliary residuals for treated obs (Theorem 3, Equation 8).

        Implements the paper's *unit-clustered* group aggregator (Borusyak,
        Jaravel & Spiess 2024, eq. 8, p. 3272), which minimizes the excess
        variance of the conservative estimator under a within-group
        constant-effect auxiliary model (Supplementary Appendix A.8):

            tau_tilde_g = sum_i (sum_{t in G_g,i} v_it)(sum_{t in G_g,i} v_it * tau_hat_it)
                          ----------------------------------------------------------------
                                          sum_i (sum_{t in G_g,i} v_it)^2

        i.e. for each unit i form the within-unit weight sum a_{i,g} and the
        within-unit weighted-effect sum b_{i,g} over the unit's observations in
        group g, then combine across units. At the default cohort x event-time
        partition (<=1 obs/unit/group) this reduces to sum(v^2 * tau_hat) /
        sum(v^2) -- the form the R `didimputation` package implements -- and
        equals the naive observation-level mean sum(v * tau_hat) / sum(v) only
        when within-group weights are uniform. Under coarser `cohort` / `horizon`
        partitions (a unit contributes several observations to a group) or
        non-uniform v_it (e.g. survey weights) the two genuinely differ.

        epsilon_tilde_it = Y_it - alpha_i - beta_t [- X'delta] - tau_tilde_g
        """
        n_1 = len(df_1)

        # Compute base residuals (Y - Y_hat(0) = tau_hat)
        # NaN for missing FE (consistent with _impute_treatment_effects)
        alpha_i = df_1[unit].map(unit_fe).values.astype(float)  # NaN for missing
        beta_t = df_1[time].map(time_fe).values.astype(float)  # NaN for missing
        y_hat_0 = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            y_hat_0 = y_hat_0 + np.dot(df_1[covariates].values, delta_hat)

        tau_hat = df_1[outcome].values - y_hat_0

        # Partition Omega_1 into groups G_g
        if self.aux_partition == "cohort_horizon":
            group_keys = list(zip(df_1[first_treat].values, df_1["_rel_time"].values))
        elif self.aux_partition == "cohort":
            group_keys = list(df_1[first_treat].values)
        elif self.aux_partition == "horizon":
            group_keys = list(df_1["_rel_time"].values)
        else:
            group_keys = list(range(n_1))  # each obs is its own group

        # Factorize group keys to integer codes (robust to tuple-valued keys).
        group_codes = pd.factorize(pd.Series(group_keys), sort=False)[0]
        gc_series = pd.Series(group_codes, index=df_1.index)
        tau_series = pd.Series(tau_hat, index=df_1.index)

        # Unit-clustered Equation 8. Only v_it != 0 observations contribute: a
        # zero-weight row adds exactly 0 to both a_{i,g} and b_{i,g}, so dropping
        # it is exact for finite tau_hat AND avoids letting an unimputable row
        # (NaN tau_hat, which always carries v_it == 0 by construction in
        # _compute_target_weights) poison its whole group via 0 * NaN = NaN. The
        # previous observation-level pandas sum relied on skipna to drop them.
        contrib = (v_treated != 0.0) & np.isfinite(tau_hat)
        loo_factor: Optional[pd.Series] = None
        n_single_loo = 0
        if contrib.any():
            inner = pd.DataFrame(
                {
                    "g": group_codes[contrib],
                    "u": df_1[unit].values[contrib],
                    "v": v_treated[contrib],
                    "vt": v_treated[contrib] * tau_hat[contrib],
                }
            )
            # Per (group, unit): a_{i,g} = sum v_it, b_{i,g} = sum v_it * tau_hat
            per_unit = inner.groupby(["g", "u"], sort=False).agg(a=("v", "sum"), b=("vt", "sum"))
            # Per group: numerator sum_i a*b, denominator sum_i a^2
            per_group = (
                per_unit.assign(ab=per_unit["a"] * per_unit["b"], a2=per_unit["a"] ** 2)
                .groupby(level="g")
                .agg(num=("ab", "sum"), den=("a2", "sum"))
            )
            den_ok = per_group["den"].abs() >= 1e-15
            tau_tilde_map = (per_group["num"] / per_group["den"]).where(den_ok)
            # BJS 2024 App. A.9 leave-one-out refinement: rescale each treated
            # residual by 1/(1 - v_ig^2 / sum_j v_jg^2) (== the direct-LOO tau_tilde
            # exactly, at the per-unit cluster sum). Reuses a_{i,g} = per_unit['a']
            # and sum_j v_jg^2 = per_group['den']; applied to epsilon_treated below.
            if self.leave_one_out:
                loo_factor, n_single_loo = self._leave_one_out_factor(per_unit, per_group)
        else:
            tau_tilde_map = pd.Series(dtype=float)

        tau_tilde_per_obs = gc_series.map(tau_tilde_map)

        # Groups with no contributing (v_it != 0, finite tau_hat) observations --
        # e.g. off-target horizons in an event-study SE -- are a variance no-op
        # (psi_g = sum_t v_it * eps_tilde_it = 0 there regardless of tau_tilde_g),
        # so fall back to the unweighted group mean of tau_hat for a finite value.
        if tau_tilde_per_obs.isna().any():
            simple_means = tau_series.groupby(gc_series).mean()
            tau_tilde_per_obs = tau_tilde_per_obs.fillna(gc_series.map(simple_means))

        tau_tilde = tau_tilde_per_obs.values

        # Auxiliary residuals
        epsilon_treated = tau_hat - tau_tilde

        # Leave-one-out rescale (BJS 2024 App. A.9): map each treated obs to its
        # (group, unit) factor and inflate the residual. Non-contributing rows
        # (v_it == 0, psi == 0 anyway) and single-positive-weight-unit groups
        # (LOO undefined, fn. 51) keep factor 1.0.
        if self.leave_one_out and loo_factor is not None:
            obs_index = pd.MultiIndex.from_arrays(
                [group_codes, df_1[unit].values], names=["g", "u"]
            )
            factor_per_obs = loo_factor.reindex(obs_index).to_numpy(dtype=float)
            factor_per_obs = np.where(np.isfinite(factor_per_obs), factor_per_obs, 1.0)
            epsilon_treated = epsilon_treated * factor_per_obs
            if n_single_loo > 0:
                warnings.warn(
                    f"leave_one_out=True: {n_single_loo} auxiliary group(s) have a single "
                    f"positive-weight unit, where the leave-one-out variance is undefined "
                    f"(Borusyak, Jaravel & Spiess 2024, Supp. App. A.9 fn. 51); those groups "
                    f"keep the non-leave-out residual. A coarser aux_partition reduces "
                    f"singleton groups.",
                    UserWarning,
                    stacklevel=2,
                )

        return epsilon_treated

    def _compute_residuals_untreated(
        self,
        df_0: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute Step 1 residuals for untreated observations."""
        # Preserve NaN for any missing FE, symmetric with the treated path in
        # _compute_auxiliary_residuals_treated. On valid data this is inert --
        # every untreated observation's unit and period appear in the Step 1 FE
        # dicts (the dicts are estimated FROM Omega_0) -- but it stops a missing
        # FE from silently becoming a 0 residual, which would mask a rank-
        # condition logic error. Any NaN is zeroed downstream in the variance
        # product (np.nan_to_num), exactly like the treated path.
        alpha_i = df_0[unit].map(unit_fe).values.astype(float)
        beta_t = df_0[time].map(time_fe).values.astype(float)
        y_hat = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            y_hat = y_hat + np.dot(df_0[covariates].values, delta_hat)

        return df_0[outcome].values - y_hat

    def _aggregate_event_study(
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
        treatment_groups: List[Any],
        balance_e: Optional[int] = None,
        kept_cov_mask: Optional[np.ndarray] = None,
        survey_weights: Optional[np.ndarray] = None,
        survey_df: Optional[int] = None,
        resolved_survey=None,
        proj_cache: Optional[Dict[Any, _UntreatedProjection]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate treatment effects by event-study horizon."""
        df_1 = df.loc[omega_1_mask]
        tau_hat = df["_tau_hat"].loc[omega_1_mask].values
        rel_times = df_1["_rel_time"].values

        # Get all horizons
        all_horizons = sorted(set(int(h) for h in rel_times if np.isfinite(h)))

        # Apply horizon_max filter
        if self.horizon_max is not None:
            all_horizons = [h for h in all_horizons if abs(h) <= self.horizon_max]

        # Apply balance_e filter
        if balance_e is not None:
            cohort_rel_times = self._build_cohort_rel_times(df, first_treat)
            balanced_mask = pd.Series(
                self._compute_balanced_cohort_mask(
                    df_1, first_treat, all_horizons, balance_e, cohort_rel_times
                ),
                index=df_1.index,
            )
        else:
            balanced_mask = pd.Series(True, index=df_1.index)

        # Check Proposition 5: no never-treated units
        has_never_treated = df["_never_treated"].any()
        h_bar = np.inf
        if not has_never_treated and len(treatment_groups) > 1:
            h_bar = max(treatment_groups) - min(treatment_groups)

        # Reference period
        ref_period = -1 - self.anticipation

        event_study_effects: Dict[int, Dict[str, Any]] = {}

        # Add reference period marker
        event_study_effects[ref_period] = {
            "effect": 0.0,
            "se": 0.0,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_int": (0.0, 0.0),
            "n_obs": 0,
        }

        # Pre-period coefficients via BJS Test 1 lead regression
        if self.pretrends:
            df_0 = df.loc[omega_0_mask].copy()

            # Determine which cohorts' lead indicators to include.
            # balance_e restricts which cohorts contribute lead dummies,
            # but the full Omega_0 sample (including never-treated controls)
            # is kept for the within-transformed OLS (BJS Test 1, Equation 9).
            balanced_cohorts = None
            skip_preperiods = False
            if balance_e is not None:
                cohort_rel_times_0 = self._build_cohort_rel_times(df, first_treat)
                balanced_cohorts = set()
                if all_horizons:
                    max_h = max(all_horizons)
                    required_range = set(range(-balance_e, max_h + 1))
                    for g, horizons in cohort_rel_times_0.items():
                        if required_range.issubset(horizons):
                            balanced_cohorts.add(g)
                if not balanced_cohorts:
                    skip_preperiods = True  # No cohorts qualify — skip entirely

            if not skip_preperiods:
                rel_time_0 = np.where(
                    ~df_0["_never_treated"],
                    df_0[time] - df_0[first_treat],
                    np.nan,
                )

                # When balance_e is set, only include leads from balanced cohorts
                if balanced_cohorts is not None:
                    is_balanced = df_0[first_treat].isin(balanced_cohorts).values
                    rel_time_for_leads = np.where(is_balanced, rel_time_0, np.nan)
                else:
                    rel_time_for_leads = rel_time_0

                pre_rel_times = sorted(
                    set(
                        int(h)
                        for h in rel_time_for_leads
                        if np.isfinite(h) and h < -self.anticipation
                    )
                )
                pre_rel_times = [h for h in pre_rel_times if h != ref_period]
                if self.horizon_max is not None:
                    pre_rel_times = [h for h in pre_rel_times if abs(h) <= self.horizon_max]
                if pre_rel_times:
                    # Survey pretrends: pass full design (subpopulation approach)
                    _sw_0_pre = None
                    _rs_full_pre = None
                    _n_full_pre = None
                    _o0_idx_pre = None
                    if survey_weights is not None and resolved_survey is not None:
                        _sw_0_pre = survey_weights[omega_0_mask.values]
                        _rs_full_pre = resolved_survey
                        _n_full_pre = len(df)
                        _o0_idx_pre = np.where(omega_0_mask.values)[0]
                    _survey_df_pre = (
                        resolved_survey.df_survey if resolved_survey is not None else None
                    )
                    pre_effects, _, _ = self._compute_lead_coefficients(
                        df_0,
                        outcome,
                        unit,
                        time,
                        first_treat,
                        covariates,
                        cluster_var,
                        pre_rel_times,
                        alpha=self.alpha,
                        balanced_cohorts=balanced_cohorts,
                        survey_weights_0=_sw_0_pre,
                        resolved_survey_full=_rs_full_pre,
                        n_obs_full=_n_full_pre,
                        omega_0_indices=_o0_idx_pre,
                        survey_df=_survey_df_pre,
                    )
                    event_study_effects.update(pre_effects)

        # Collect horizons with Proposition 5 violations
        prop5_horizons = []

        for h in all_horizons:
            if h == ref_period:
                continue

            # Select treated obs at this horizon from balanced cohorts
            h_mask = (rel_times == h) & balanced_mask.values
            n_h = int(h_mask.sum())

            if n_h == 0:
                continue

            # Proposition 5 check
            if not has_never_treated and h >= h_bar:
                prop5_horizons.append(h)
                event_study_effects[h] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_h,
                }
                continue

            tau_h = tau_hat[h_mask]
            finite_h = np.isfinite(tau_h)
            valid_tau = tau_h[finite_h]

            if len(valid_tau) == 0:
                event_study_effects[h] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_h,
                }
                continue

            # Survey-weighted or simple mean for per-horizon effect
            if survey_weights is not None:
                treated_sw = survey_weights[omega_1_mask.values]
                sw_h = treated_sw[h_mask]
                sw_valid = sw_h[finite_h]
                effect = float(np.average(valid_tau, weights=sw_valid))
            else:
                effect = float(np.mean(valid_tau))

            # Compute SE via conservative variance with horizon-specific weights
            # When survey, aggregation weights are proportional to survey weights
            if survey_weights is not None:
                treated_sw = survey_weights[omega_1_mask.values]
                n_1 = len(tau_hat)
                weights_h = np.zeros(n_1)
                sw_h = treated_sw[h_mask]
                finite_in_h = np.isfinite(tau_h)
                sw_finite = sw_h[finite_in_h]
                # Set weights proportional to survey weights, summing to 1
                if sw_finite.sum() > 0:
                    h_indices = np.where(h_mask)[0]
                    finite_indices = h_indices[finite_in_h]
                    weights_h[finite_indices] = sw_finite / sw_finite.sum()
                n_valid = int(finite_in_h.sum())
            else:
                weights_h, n_valid = _compute_target_weights(tau_hat, h_mask)

            se = self._compute_conservative_variance(
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
                weights=weights_h,
                cluster_var=cluster_var,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
                resolved_survey=resolved_survey,
                proj_cache=proj_cache,
            )

            t_stat, p_value, conf_int = safe_inference(effect, se, alpha=self.alpha, df=survey_df)

            event_study_effects[h] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_obs": n_h,
            }

        # Proposition 5 warning
        if prop5_horizons:
            warnings.warn(
                f"Horizons {prop5_horizons} are not identified without "
                f"never-treated units (Proposition 5). Set to NaN.",
                UserWarning,
                stacklevel=3,
            )

        # Check for empty result set after filtering
        real_effects = [
            h for h, v in event_study_effects.items() if h != ref_period and v.get("n_obs", 0) > 0
        ]
        if len(real_effects) == 0:
            filter_info = []
            if balance_e is not None:
                filter_info.append(f"balance_e={balance_e}")
            if self.horizon_max is not None:
                filter_info.append(f"horizon_max={self.horizon_max}")
            filter_str = " and ".join(filter_info) if filter_info else "filters"
            warnings.warn(
                f"Event study aggregation produced no horizons with observations "
                f"after applying {filter_str}. The result contains only the "
                f"reference period marker. Consider relaxing filter parameters.",
                UserWarning,
                stacklevel=3,
            )

        return event_study_effects

    def _aggregate_group(
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
        treatment_groups: List[Any],
        kept_cov_mask: Optional[np.ndarray] = None,
        survey_weights: Optional[np.ndarray] = None,
        survey_df: Optional[int] = None,
        resolved_survey=None,
        proj_cache: Optional[Dict[Any, _UntreatedProjection]] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        """Aggregate treatment effects by cohort."""
        df_1 = df.loc[omega_1_mask]
        tau_hat = df["_tau_hat"].loc[omega_1_mask].values
        cohorts = df_1[first_treat].values

        group_effects: Dict[Any, Dict[str, Any]] = {}

        for g in treatment_groups:
            g_mask = cohorts == g
            n_g = int(g_mask.sum())

            if n_g == 0:
                continue

            tau_g = tau_hat[g_mask]
            finite_g = np.isfinite(tau_g)
            valid_tau = tau_g[finite_g]

            if len(valid_tau) == 0:
                group_effects[g] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_g,
                }
                continue

            # Survey-weighted or simple mean for per-group effect
            if survey_weights is not None:
                treated_sw = survey_weights[omega_1_mask.values]
                sw_g = treated_sw[g_mask]
                sw_valid = sw_g[finite_g]
                effect = float(np.average(valid_tau, weights=sw_valid))
            else:
                effect = float(np.mean(valid_tau))

            # Compute SE with group-specific weights
            # When survey, aggregation weights proportional to survey weights
            if survey_weights is not None:
                treated_sw = survey_weights[omega_1_mask.values]
                n_1 = len(tau_hat)
                weights_g = np.zeros(n_1)
                sw_g = treated_sw[g_mask]
                sw_finite = sw_g[finite_g]
                if sw_finite.sum() > 0:
                    g_indices = np.where(g_mask)[0]
                    finite_indices = g_indices[finite_g]
                    weights_g[finite_indices] = sw_finite / sw_finite.sum()
            else:
                weights_g, _ = _compute_target_weights(tau_hat, g_mask)

            se = self._compute_conservative_variance(
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
                weights=weights_g,
                cluster_var=cluster_var,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
                resolved_survey=resolved_survey,
                proj_cache=proj_cache,
            )

            t_stat, p_value, conf_int = safe_inference(effect, se, alpha=self.alpha, df=survey_df)

            group_effects[g] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_obs": n_g,
                # Per-row df provenance (M-021): the exact df THIS row's
                # safe_inference received. Capture-at-use — the replicate
                # override rewrites it (with the row's se/t/p/CI) and the
                # bootstrap override clears it; the all-NaN cohort branch
                # above writes no key (consumers read via .get).
                "df_used": survey_df,
            }

        return group_effects

    def _compute_lead_coefficients(
        self,
        df_0: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        cluster_var: str,
        pre_rel_times: List[int],
        alpha: float = 0.05,
        balanced_cohorts: Optional[set] = None,
        survey_weights_0: Optional[np.ndarray] = None,
        resolved_survey_full=None,
        n_obs_full: Optional[int] = None,
        omega_0_indices: Optional[np.ndarray] = None,
        survey_df: Optional[int] = None,
    ) -> Tuple[Dict[int, Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        Compute pre-period lead coefficients via within-transformed OLS (Test 1).

        Adds lead indicator dummies W_it(h) = 1[K_it = h] to the untreated
        model and estimates their coefficients. Uses cluster-robust SEs by
        default, or design-based survey VCV when ``resolved_survey_full``
        is provided (subpopulation approach: scores zero-padded to full
        panel length to preserve PSU/strata structure).

        The full Omega_0 sample (including never-treated controls) is always
        used for within-transformation. When balanced_cohorts is provided,
        lead indicators are restricted to observations from those cohorts only.

        Returns
        -------
        effects : dict
            Per-horizon event_study_effects entries.
        gamma : ndarray
            Lead coefficient vector.
        V_gamma : ndarray
            Sub-VCV matrix for lead coefficients.
        """
        rel_time_0 = np.where(
            ~df_0["_never_treated"],
            df_0[time] - df_0[first_treat],
            np.nan,
        )

        # Build lead indicators — restrict to balanced cohorts if specified
        if balanced_cohorts is not None:
            is_balanced = df_0[first_treat].isin(balanced_cohorts).values
        else:
            is_balanced = None

        lead_cols = []
        for h in pre_rel_times:
            col_name = f"_lead_{h}"
            indicator = (rel_time_0 == h).astype(float)
            if is_balanced is not None:
                indicator = indicator * is_balanced  # zero out non-balanced cohorts
            df_0[col_name] = indicator
            lead_cols.append(col_name)

        all_x_cols = lead_cols[:]
        if covariates:
            all_x_cols.extend(covariates)

        # Within-transform through the shared MAP engine (survey-weighted when
        # present), one dispatch for outcome + leads + covariates. Demean into
        # a narrow copy: df_0's raw lead indicators must survive for the
        # per-horizon n_obs counts below. within_transform pins [unit, time];
        # [time, unit] here preserves the historical time-then-unit sweep order.
        narrow = df_0[[outcome, *all_x_cols, time, unit]].copy()
        _pre_norms = pre_demean_norms(narrow, all_x_cols, weights=survey_weights_0)
        demeaned, _ = demean_by_groups(
            narrow,
            [outcome, *all_x_cols],
            [time, unit],
            inplace=True,
            weights=survey_weights_0,
            max_iter=10_000,
            tol=1e-10,
        )
        # FE-spanned regressors demean to numerical junk, not exact zero;
        # snap them so rank handling drops them deterministically (NaN
        # coefficient for that horizon) instead of the junk direction
        # perturbing the identified lead coefficients. Lead indicators are
        # the most plausible FE-spanned regressors here: with a single
        # (balanced-restricted) cohort a lead h collapses to a calendar-time
        # dummy on Omega_0, which lies exactly in the span of the absorbed
        # time FE.
        snap_absorbed_regressors(
            demeaned,
            all_x_cols,
            _pre_norms,
            absorbed_desc="unit and time fixed effects (pretrends lead model)",
            group_vars=[time, unit],
            rank_deficient_action=self.rank_deficient_action,
            display_names={f"_lead_{h}": f"lead[{h}]" for h in pre_rel_times},
            weights=survey_weights_0,
        )
        y_dm = demeaned[outcome].to_numpy(dtype=np.float64)
        X_dm = demeaned[all_x_cols].to_numpy(dtype=np.float64)

        # OLS for point estimates + VCV. When survey VCV will replace the
        # cluster-robust VCV, skip cluster_ids to avoid errors on domains
        # with few PSUs (the cluster-robust VCV is discarded anyway).
        cluster_ids = df_0[cluster_var].values
        _ols_weights = survey_weights_0
        _ols_weight_type = "pweight" if survey_weights_0 is not None else None
        _use_survey_vcov = resolved_survey_full is not None
        # Clustered-CR1 K_reference increment (variance-conventions.md D2):
        # the demeaned lead design carries NO intercept column, so the
        # absorbed constant contributes the +1 term. Survey fits pass 0
        # (cluster_ids is None there and the survey vcov replaces CR1).
        _cr1_k_adj_imp = 0
        if not _use_survey_vcov:
            _cr1_k_adj_imp = absorbed_fe_cr1_k_increment(
                df_0,
                [time, unit],
                cluster_ids,
                has_intercept_col=False,
                weights=survey_weights_0,
            )
        try:
            result = solve_ols(
                X_dm,
                y_dm,
                weights=_ols_weights,
                weight_type=_ols_weight_type,
                cluster_ids=None if _use_survey_vcov else cluster_ids,
                cluster_k_adjustment=_cr1_k_adj_imp,
                return_vcov=True,
                rank_deficient_action=self.rank_deficient_action,
                column_names=all_x_cols,
            )
        except (IndexError, np.linalg.LinAlgError):
            # All lead columns dropped (rank deficient after demeaning)
            effects: Dict[int, Dict[str, Any]] = {}
            for h in pre_rel_times:
                n_obs = int(df_0[f"_lead_{h}"].sum())
                effects[h] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_obs,
                }
            for col in lead_cols:
                df_0.drop(columns=col, inplace=True)
            return (
                effects,
                np.full(len(pre_rel_times), np.nan),
                np.full((len(pre_rel_times), len(pre_rel_times)), np.nan),
            )

        coefficients = result[0]
        vcov = result[2]
        assert vcov is not None

        # Replace cluster-robust VCV with survey design-based VCV.
        # Use the FULL survey design (subpopulation approach): zero-pad
        # the Omega_0 scores back to full-panel length so PSU/strata
        # structure is preserved for variance estimation.
        if resolved_survey_full is not None:
            from diff_diff.survey import compute_survey_vcov

            # Use residuals from solve_ols (safe for rank-deficient fits).
            residuals_0 = result[1]

            # Reduce to kept (finite-coefficient) columns for VCV
            kept_mask = np.isfinite(coefficients)
            if np.all(kept_mask):
                X_for_vcov = X_dm
                res_for_vcov = residuals_0
            else:
                X_for_vcov = X_dm[:, kept_mask]
                res_for_vcov = residuals_0

            # Zero-pad to full panel length (subpopulation approach):
            # observations outside Omega_0 contribute zero to the score,
            # but preserve PSU/strata structure for design-based variance.
            # The survey full-design path always supplies the full obs count.
            assert n_obs_full is not None
            n_full_obs = n_obs_full
            k_vcov = X_for_vcov.shape[1]
            X_full = np.zeros((n_full_obs, k_vcov), dtype=np.float64)
            res_full = np.zeros(n_full_obs, dtype=np.float64)
            X_full[omega_0_indices] = X_for_vcov
            res_full[omega_0_indices] = res_for_vcov

            vcov_kept = compute_survey_vcov(X_full, res_full, resolved_survey_full)

            if not np.all(kept_mask):
                # Expand back: NaN rows/cols for dropped columns
                n_coef = len(coefficients)
                vcov = np.full((n_coef, n_coef), np.nan)
                kept_idx = np.where(kept_mask)[0]
                vcov[np.ix_(kept_idx, kept_idx)] = vcov_kept
            else:
                vcov = vcov_kept

        n_leads = len(lead_cols)
        gamma = coefficients[:n_leads]
        V_gamma = vcov[:n_leads, :n_leads]

        # Resolve the per-lead tail df: the full-design survey df keeps
        # precedence on survey fits; otherwise resolve through the
        # df_convention knob (3.9 fix: previously df=None → silent
        # normal-theory z on plain clustered fits). Residual df = n −
        # k_kept − absorbed [time, unit] rank (the demeaned lead design
        # carries no intercept column); the "cluster" G counts
        # positive-weight clusters on the SAME raw cluster ids the CR1
        # vcov partitioned on. NOTE this resolution serves BOTH callers —
        # the fit-time pretrends event-study path and the post-fit public
        # ``pretrend_test()`` (which consumes only gamma/V_gamma; its
        # joint Wald F denominator is knob-independent).
        if _use_survey_vcov:
            _df = survey_df
        else:
            _k_kept_imp = int(np.count_nonzero(np.isfinite(coefficients)))
            _df_res_imp = float(
                len(df_0)
                - _k_kept_imp
                - absorbed_fe_rank(
                    df_0,
                    [time, unit],
                    has_intercept_col=False,
                    weights=survey_weights_0,
                )
            )
            from diff_diff.linalg import effective_cluster_count

            _df = resolve_tail_df(
                self.df_convention,
                residual_df=_df_res_imp,
                n_clusters=effective_cluster_count(cluster_ids, survey_weights_0),
            )

        # Build per-horizon effects
        effects = {}
        for j, h in enumerate(pre_rel_times):
            effect = float(gamma[j])
            se = float(np.sqrt(max(V_gamma[j, j], 0.0)))
            # n_obs from the lead indicator (respects balanced_cohorts restriction)
            n_obs = int(df_0[f"_lead_{h}"].sum())
            t_stat, p_value, conf_int = safe_inference(effect, se, alpha=alpha, df=_df)
            effects[h] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_obs": n_obs,
            }

        # Clean up temporary columns
        for col in lead_cols:
            df_0.drop(columns=col, inplace=True)

        return effects, gamma, V_gamma

    @staticmethod
    def _leave_one_out_factor(
        per_unit: pd.DataFrame, per_group: pd.DataFrame
    ) -> Tuple[pd.Series, int]:
        """Per-(group, unit) leave-one-out residual-rescale factor (BJS 2024 A.9).

        ``factor_{g,i} = 1 / (1 - v_ig**2 / sum_j v_jg**2)`` with
        ``v_ig = per_unit['a']`` and ``sum_j v_jg**2 = per_group['den']``. This
        rescale of ``epsilon_tilde_it`` reproduces the direct leave-one-out
        aggregate ``tau_tilde_it^LO`` exactly at the per-unit cluster sum
        ``psi_i = sum_t v_it * epsilon_tilde_it`` (App. A.9). A group with a
        single positive-weight unit has ``v_ig**2 == sum_j v_jg**2`` so the
        factor diverges (LOO undefined, App. A.9 fn. 51); those groups fall back
        to ``1.0`` (non-LOO). A genuinely unit-dominated but >=2-unit group keeps
        its large finite factor -- that is the paper's intended inflation.

        Returns
        -------
        (factor : pd.Series indexed like ``per_unit`` (g, u), n_single_unit_groups : int)
        """
        a = per_unit["a"].to_numpy(dtype=float)
        sq = a**2
        g_level = per_unit.index.get_level_values("g")
        u_level = per_unit.index.get_level_values("u")
        den = per_group["den"].reindex(g_level).to_numpy(dtype=float)  # D_g per (g,u)

        # A group is "singleton" for LOO (App. A.9 fn. 51) when fewer than two
        # units carry positive squared weight -- covers a true 1-unit group AND
        # the effective-singleton case (>=2 rows, only one with a_ig != 0).
        pos_per_group = pd.Series(sq > 0.0, index=per_unit.index).groupby(level="g").sum()
        single_groups = pos_per_group.index[pos_per_group < 2]
        is_single = pd.Index(g_level).isin(single_groups)

        den_ok = np.abs(den) >= 1e-15
        # factor = D_g / (D_g - v_ig^2) = D_g / sum_{j!=i} v_jg^2. Compute the
        # leave-one-out denominator as the sum of the OTHER units' squared
        # weights -- NOT as D_g - v_ig^2 after forming the ratio: for a genuinely
        # dominated (but >=2-unit) group the subtraction loses precision (and can
        # cancel to 0/negative) in float64 -- a finite-but-wrong or silently
        # non-LOO factor. The fast subtraction is accurate away from the
        # near-cancellation boundary; wherever the leave-one-out mass is a tiny
        # fraction of D (relative loss of >~1e-6), recompute it exactly as the
        # drop-then-sum of the OTHER units' squared weights. At most one unit per
        # group can be that dominant, so the recompute stays O(units).
        other_mass = den - sq
        suspect = (~is_single) & den_ok & (other_mass <= 1e-6 * den)
        if suspect.any():
            sq_series = pd.Series(sq, index=per_unit.index)
            for pos in np.nonzero(suspect)[0]:
                grp = sq_series.xs(g_level[pos], level="g")
                other_mass[pos] = float(grp.drop(u_level[pos]).sum())
        # Fall back to non-LOO (factor 1.0) only where LOO is genuinely undefined:
        # a singleton group (fn. 51), a degenerate den, or no other positive mass.
        fallback = is_single | ~den_ok | (other_mass <= 0.0)
        factor = np.where(fallback, 1.0, den / np.where(fallback, 1.0, other_mass))
        return pd.Series(factor, index=per_unit.index), int(len(single_groups))

    def _replicate_override_aggregates(
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
        resolved_survey: Any,
        overall_att: float,
        event_study_effects: Optional[Dict[int, Dict[str, Any]]],
        group_effects: Optional[Dict[Any, Dict[str, Any]]],
        balance_e: Optional[int],
        survey_df_seed: Optional[int],
    ) -> Tuple[np.ndarray, int, Optional[int]]:
        """Replicate-weight inference override for the aggregation surfaces.

        Extracted verbatim from ``fit()``'s replicate block so the post-fit
        ``aggregate()`` path can replay it from the kit payload. The stacked
        layout is ``[overall, es..., grp...]`` built from whichever family
        dicts are non-None. ``compute_replicate_refit_variance`` validates
        replicates JOINTLY (all-finite rows, ``survey.py``), so inference is
        LEVEL-MATCHED: post-fit ``aggregate('event_study')`` replays
        ``[overall, ES]`` and reproduces ``fit(aggregate='event_study')``
        exactly; a ``fit(aggregate='all')`` surface is NOT the equivalence
        target when a replicate NaNs on exactly one family's targets (its
        joint stack drops that replicate for every row).

        Overrides ``se``/``t_stat``/``p_value``/``conf_int`` (and, on group
        rows, the ``df_used`` provenance key) IN PLACE on the passed dicts
        and returns ``(vcov, n_valid, survey_df_final)``. The fit-only tail
        -- the overall-row quintet and the ``survey_metadata.df_survey``
        write -- deliberately stays in ``fit()``: this method must never
        mutate results-owned metadata (post-fit ``aggregate()`` calls it on
        kit refs and a throwaway host). Cost: one full per-replicate refit
        pass per call (R refits), the same work the fit-time path does.
        """
        from diff_diff.survey import compute_replicate_refit_variance

        _rel_times_treated = df.loc[omega_1_mask, "_rel_time"].values
        _cohorts_treated = df.loc[omega_1_mask, first_treat].values

        # Derive keys from actual outputs (excludes filtered/Prop5/ref)
        _es_effects = event_study_effects or {}
        _grp_effects = group_effects or {}
        _sorted_rel_times = sorted(
            e
            for e in _es_effects.keys()
            if np.isfinite(_es_effects[e]["effect"]) and _es_effects[e].get("n_obs", 1) > 0
        )
        _sorted_groups = sorted(
            g for g in _grp_effects.keys() if np.isfinite(_grp_effects[g]["effect"])
        )
        _n_es = len(_sorted_rel_times)

        # Pre-compute balanced cohort mask for balance_e
        _balanced_mask_treated = None
        if balance_e is not None and _sorted_rel_times:
            df_1 = df.loc[omega_1_mask]
            rel_times_all = df_1["_rel_time"].values
            all_horizons_full = sorted(set(int(h) for h in rel_times_all if np.isfinite(h)))
            if self.horizon_max is not None:
                all_horizons_full = [h for h in all_horizons_full if abs(h) <= self.horizon_max]
            cohort_rel_times = self._build_cohort_rel_times(df, first_treat)
            _balanced_mask_treated = self._compute_balanced_cohort_mask(
                df_1, first_treat, all_horizons_full, balance_e, cohort_rel_times
            )

        # Single vectorized refit: [overall, es_e0..., grp_g0...]
        def _refit_imp(w_r):
            ufe_r, tfe_r, gm_r, delta_r, _ = self._fit_untreated_model(
                df,
                outcome,
                unit,
                time,
                covariates,
                omega_0_mask,
                weights=w_r,
            )
            tau_r, _ = self._impute_treatment_effects(
                df,
                outcome,
                unit,
                time,
                covariates,
                omega_1_mask,
                ufe_r,
                tfe_r,
                gm_r,
                delta_r,
            )
            fin = np.isfinite(tau_r)
            treated_w = w_r[omega_1_mask.values]
            results = []
            # [0] Overall ATT
            tw_fin = treated_w[fin]
            tw_sum = np.sum(tw_fin)
            results.append(float(np.sum(tau_r[fin] * tw_fin) / tw_sum) if tw_sum > 0 else np.nan)
            # [1..n_es] Event-study (identified only)
            for e in _sorted_rel_times:
                mask_e = fin & (_rel_times_treated == e)
                if _balanced_mask_treated is not None:
                    mask_e = mask_e & _balanced_mask_treated
                tw_e = treated_w[mask_e]
                s = np.sum(tw_e)
                results.append(float(np.sum(tau_r[mask_e] * tw_e) / s) if s > 0 else np.nan)
            # [n_es+1..] Group (identified only)
            for g in _sorted_groups:
                mask_g = fin & (_cohorts_treated == g)
                tw_g = treated_w[mask_g]
                s = np.sum(tw_g)
                results.append(float(np.sum(tau_r[mask_g] * tw_g) / s) if s > 0 else np.nan)
            return np.array(results)

        # Build full-sample estimate from actual effects
        _full_est = [overall_att]
        _full_est.extend([_es_effects[e]["effect"] for e in _sorted_rel_times])
        _full_est.extend([_grp_effects[g]["effect"] for g in _sorted_groups])

        _vcov_rep_imp, _n_valid_rep_imp = compute_replicate_refit_variance(
            _refit_imp, np.array(_full_est), resolved_survey
        )

        # Override df if replicates were dropped
        # Replicate-refit path is only reached with a resolved design.
        assert resolved_survey is not None
        survey_df_final = survey_df_seed
        if _n_valid_rep_imp < resolved_survey.n_replicates:
            survey_df_final = _n_valid_rep_imp - 1 if _n_valid_rep_imp > 1 else 0

        # Override event-study SEs from vcov diagonal
        for i, e in enumerate(_sorted_rel_times):
            if event_study_effects is not None and e in event_study_effects:
                se_e = float(np.sqrt(max(_vcov_rep_imp[1 + i, 1 + i], 0.0)))
                eff_e = event_study_effects[e]["effect"]
                t_e, p_e, ci_e = safe_inference(eff_e, se_e, alpha=self.alpha, df=survey_df_final)
                event_study_effects[e]["se"] = se_e
                event_study_effects[e]["t_stat"] = t_e
                event_study_effects[e]["p_value"] = p_e
                event_study_effects[e]["conf_int"] = ci_e

        # Override group SEs from vcov diagonal
        for j, g in enumerate(_sorted_groups):
            if group_effects is not None and g in group_effects:
                se_g = float(np.sqrt(max(_vcov_rep_imp[1 + _n_es + j, 1 + _n_es + j], 0.0)))
                eff_g = group_effects[g]["effect"]
                t_g, p_g, ci_g = safe_inference(eff_g, se_g, alpha=self.alpha, df=survey_df_final)
                group_effects[g]["se"] = se_g
                group_effects[g]["t_stat"] = t_g
                group_effects[g]["p_value"] = p_g
                group_effects[g]["conf_int"] = ci_g
                # the override rewrote se/t/p/CI under the final df --
                # the per-row provenance key follows the rewrite
                group_effects[g]["df_used"] = survey_df_final

        return _vcov_rep_imp, _n_valid_rep_imp, survey_df_final
