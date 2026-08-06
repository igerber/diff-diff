"""
Classic Synthetic Control Method (SCM) estimator.

Implements Abadie, Diamond & Hainmueller (2010), "Synthetic Control Methods for
Comparative Case Studies: Estimating the Effect of California's Tobacco Control
Program," *JASA* 105(490):493-505. The method originates in Abadie &
Gardeazabal (2003).

A single treated unit's counterfactual is built as a convex combination of
"donor" (never-treated) units. Donor weights ``W*(V)`` solve a simplex-constrained
weighted least-squares fit of the treated unit's predictors; the predictor-importance
matrix ``V`` (diagonal, PSD) is chosen data-driven by minimizing pre-period outcome MSPE
("nested"), by out-of-sample cross-validation ("cv", split at ``v_cv_t0``), or by the
closed-form inverse-variance heuristic ("inverse_variance"), or supplied by the user
("custom"). The treatment-effect path is the gap ``α̂_1t = Y_1t − Σ_j w_j · Y_jt`` over
the post periods.

Distinct from :class:`~diff_diff.SyntheticDiD` (Arkhangelsky et al. 2021), which adds
time weights and ridge regularization: classic SCM uses **donor weights only** and a
level-matching estimator, plus the outer ``V`` search SyntheticDiD has no analog for.

Inference: classic SCM has **no analytical standard error**, so
``se``/``t_stat``/``p_value``/``conf_int`` are always NaN; ``att`` (mean post-period
gap) is the reported estimate. Significance comes from in-space placebo permutation
inference (ADH 2010 §2.4) via ``SyntheticControlResults.in_space_placebo()`` — a
separate ``placebo_p_value`` field, distinct from the NaN analytical ``p_value``.

Numerics provenance: the standardization divisor and the inner/outer optimization
scheme are NOT specified in ADH (2010) — they are pinned from the R ``Synth`` package
source / Abadie-Gardeazabal (2003) App. B. See ``docs/methodology/REGISTRY.md``
§SyntheticControl for the deviation/Note labels.
"""

import itertools
import warnings
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from diff_diff._base import BaseEstimator
from diff_diff.synthetic_control_results import (
    SyntheticControlResults,
    _SyntheticControlFitSnapshot,
)
from diff_diff.utils import _sc_weight_fw, safe_inference, warn_if_not_converged

__all__ = ["SyntheticControl", "synthetic_control"]

# Aggregation operators allowed for predictor / special-predictor rows. Restricted to
# LINEAR combinations of pre-period values, matching ADH (2010) §2.3
# (`Ȳ_i^{K_m} = Σ_s k_s Y_is`): mean (k_s = 1/T0) and sum (k_s = 1). Non-linear
# aggregations (e.g. median) are intentionally NOT supported.
_OP_FUNCS: Dict[str, Any] = {"mean": np.mean, "sum": np.sum}

# Interpretability floor: donor weights below this are dropped from the reported
# ``donor_weights`` dict (mirrors prep.py's 1e-6 sparsification). The full weight
# vector is still used for the gap path / ATT.
_MIN_REPORT_WEIGHT = 1e-6


@dataclass
class _PredictorSpec:
    """Normalized predictor specification (one matrix row of X1/X0)."""

    label: str
    kind: str  # "predictor_avg" | "special" | "lag"
    var: str
    periods: List[Any]
    op: str  # "mean" | "sum" | "identity"


def _softmax(theta: np.ndarray) -> np.ndarray:
    """Map an unconstrained vector to the unit simplex (v >= 0, sum v = 1)."""
    e = np.exp(theta - np.max(theta))
    return e / np.sum(e)


class SyntheticControl(BaseEstimator):
    """
    Classic Synthetic Control Method estimator (Abadie-Diamond-Hainmueller 2010).

    Parameters
    ----------
    v_method : {"nested", "custom", "cv", "inverse_variance"}, default "nested"
        How the predictor-importance matrix V is chosen. ``"nested"`` selects V
        data-driven by minimizing the pre-period outcome MSPE of ``W*(V)``
        (ADH 2010 §2.3). ``"custom"`` uses the user-supplied ``custom_v`` and
        skips the outer search. ``"cv"`` selects V by out-of-sample
        cross-validation (ADH 2015 §; Abadie 2021 Eq. 9): the pre-period is split
        at ``v_cv_t0`` into a training and a validation window; V is chosen to
        minimize the validation-window outcome MSPE of the training-fit weights,
        then the final weights are re-estimated on the validation-window
        predictors. ``"inverse_variance"`` uses the closed-form
        ``v_h = 1/Var(X_{h·})`` (Abadie 2021 §3.2(a); variance over donors+treated),
        applied to the RAW predictors so the effective objective is the
        unit-variance-rescaled ``Σ_h diff_h²/Var_h`` — no search, deterministic. Note
        this rescaling *is* what ``standardize="std"`` does, so the ``standardize``
        setting does not compose with it (equivalent to uniform V on standardized
        predictors); applying ``1/Var`` on already-standardized rows would
        double-rescale to ``Σ_h diff_h²/Var_h²``.
    custom_v : array-like, optional
        Diagonal of V (length = number of predictors). Required iff
        ``v_method="custom"``; must be None for every other ``v_method``
        (``nested`` / ``cv`` / ``inverse_variance``). Must be finite and
        non-negative; trace-normalized internally.
    optimizer_options : dict, optional
        Extra options merged into every ``scipy.optimize.minimize`` call in the
        outer V search (e.g. ``maxiter``, ``xatol``, ``fatol``).
    n_starts : int, default 4
        Number of starting points for the multistart outer V search.
    inner_max_iter : int, default 10000
        Max iterations for the inner Frank-Wolfe simplex solve.
    inner_min_decrease : float, default 1e-5
        Inner-solve convergence scale (matches the SDID/Frank-Wolfe precedent in
        ``prep.py``). The Frank-Wolfe stop threshold is
        ``(inner_min_decrease * max(||b||, 1e-12))**2`` where ``b`` is the
        V^½-scaled treated predictor vector — scale-aware so convergence is
        meaningful at any data magnitude. 1e-5 reproduces R ``Synth``'s donor
        weights to ~1e-4 on the Basque benchmark while still signalling
        convergence; tighter values (e.g. 1e-6) can exhaust ``inner_max_iter``.
    standardize : {"std", "none"}, default "std"
        Predictor standardization. ``"std"`` divides each predictor row by its
        standard deviation across donors+treated (ddof=1), matching R ``Synth``.
        ``"none"`` is a deviation from R (see REGISTRY).
    alpha : float, default 0.05
        Significance level recorded for downstream (placebo) inference.
    seed : int, optional
        Seed for the multistart random (Dirichlet) starting points.
    v_cv_t0 : int, optional
        Training/validation split index for ``v_method="cv"`` only (positional
        into the pre-periods: training = ``pre[:v_cv_t0]``, validation =
        ``pre[v_cv_t0:]``). Must leave at least 1 training and 1 validation
        pre-period. Default None → ``len(pre_periods) // 2`` (Abadie 2021's
        ``t0 = T0/2``). Must be None unless ``v_method="cv"``.
    """

    def __init__(
        self,
        v_method: str = "nested",
        custom_v: Optional[Any] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
        n_starts: int = 4,
        inner_max_iter: int = 10000,
        inner_min_decrease: float = 1e-5,
        standardize: str = "std",
        alpha: float = 0.05,
        seed: Optional[int] = None,
        v_cv_t0: Optional[int] = None,
    ):
        self.v_method = v_method
        self.custom_v = custom_v
        self.optimizer_options = optimizer_options
        self.n_starts = n_starts
        self.inner_max_iter = inner_max_iter
        self.inner_min_decrease = inner_min_decrease
        self.standardize = standardize
        self.alpha = alpha
        self.seed = seed
        self.v_cv_t0 = v_cv_t0

        self._validate_config()

        # Internal state
        self.results_: Optional[SyntheticControlResults] = None
        self.is_fitted_: bool = False

    # =========================================================================
    # sklearn-like API
    # =========================================================================

    def _validate_config(self) -> None:
        """Validate hyperparameters; shared by ``__init__`` and ``set_params``."""
        if self.v_method not in ("nested", "custom", "cv", "inverse_variance"):
            raise ValueError(
                "v_method must be one of ('nested', 'custom', 'cv', 'inverse_variance'), "
                f"got {self.v_method!r}"
            )
        if self.standardize not in ("std", "none"):
            raise ValueError(
                f"standardize must be one of ('std', 'none'), got {self.standardize!r}"
            )
        # custom_v cross-field rules — fail-closed, never silently ignore.
        if self.v_method == "custom" and self.custom_v is None:
            raise ValueError("custom_v is required when v_method='custom'.")
        if self.v_method != "custom" and self.custom_v is not None:
            raise ValueError(
                f"custom_v must be None when v_method={self.v_method!r} "
                "(it would be silently ignored otherwise)."
            )
        if self.custom_v is not None:
            v = np.asarray(self.custom_v, dtype=float).ravel()
            if v.size == 0:
                raise ValueError("custom_v must be non-empty.")
            if not np.all(np.isfinite(v)):
                raise ValueError("custom_v must be finite.")
            if np.any(v < 0):
                raise ValueError("custom_v must be non-negative.")
            if v.sum() <= 0:
                raise ValueError("custom_v must have a positive sum.")
        if not isinstance(self.n_starts, (int, np.integer)) or self.n_starts < 1:
            raise ValueError(f"n_starts must be a positive integer, got {self.n_starts!r}")
        if not isinstance(self.inner_max_iter, (int, np.integer)) or self.inner_max_iter < 1:
            raise ValueError(
                f"inner_max_iter must be a positive integer, got {self.inner_max_iter!r}"
            )
        if not (np.isfinite(self.inner_min_decrease) and self.inner_min_decrease > 0):
            raise ValueError(
                f"inner_min_decrease must be a positive float, got {self.inner_min_decrease!r}"
            )
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha!r}")
        # v_cv_t0 is meaningful ONLY for v_method="cv" — fail closed (never silently
        # ignore). The full range check (1 <= t0 <= n_pre-1) needs the fitted pre-period
        # count and is enforced in fit(); here we validate type/positivity + the
        # cross-field rule. (bool is an int subclass but is rejected by the < 1 path only
        # for False; treat True/False as invalid via the explicit bool guard.)
        if self.v_cv_t0 is not None:
            if self.v_method != "cv":
                raise ValueError(
                    "v_cv_t0 is only valid when v_method='cv' "
                    f"(got v_method={self.v_method!r}); leave v_cv_t0=None otherwise."
                )
            if isinstance(self.v_cv_t0, bool) or not isinstance(self.v_cv_t0, (int, np.integer)):
                raise ValueError(
                    f"v_cv_t0 must be a positive integer or None, got {self.v_cv_t0!r}"
                )
            if self.v_cv_t0 < 1:
                raise ValueError(f"v_cv_t0 must be >= 1, got {self.v_cv_t0!r}")

    # get_params/set_params come from BaseEstimator.

    # =========================================================================
    # fit
    # =========================================================================

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        *,
        post_periods: Optional[List[Any]] = None,
        treated_unit: Optional[Any] = None,
        predictors: Optional[List[str]] = None,
        predictors_op: str = "mean",
        predictor_window: Optional[List[Any]] = None,
        special_predictors: Optional[List[Tuple[str, List[Any], str]]] = None,
        pre_period_outcomes: Optional[Any] = None,
        donor_pool: Optional[List[Any]] = None,
        survey_design: Any = None,
    ) -> SyntheticControlResults:
        """
        Fit the classic synthetic control model.

        Parameters
        ----------
        data : pandas.DataFrame
            Balanced panel.
        outcome, treatment, unit, time : str
            Column names. ``treatment`` is the ABSORBING treatment indicator
            (0/1): 1 for the treated unit in its treated periods, 0 otherwise.
        post_periods : list, optional
            Explicit post-treatment period values. If None, inferred from the
            treated unit's treatment column (the D==1 periods).
        treated_unit : Any, optional
            Identifier of the treated unit. If None, inferred as the single
            ever-treated unit.
        predictors : list of str, optional
            Columns averaged over ``predictor_window`` (using ``predictors_op``)
            to form predictor rows.
        predictors_op : {"mean", "sum"}, default "mean"
            Aggregation operator for ``predictors`` (linear combinations only, per
            ADH 2010 §2.3).
        predictor_window : list, optional
            Pre-periods over which ``predictors`` are averaged. Defaults to all
            pre periods. Must be a non-empty subset of the pre periods.
        special_predictors : list of (var, periods, op), optional
            Per-variable special predictors, each averaged over its own periods
            with its own operator (mirrors R ``Synth`` ``special.predictors``).
        pre_period_outcomes : "all" or list, optional
            Use individual pre-period outcomes as predictor rows ("all" = every
            pre period). When no predictor arguments are given at all, defaults
            to all pre-period outcomes.
        donor_pool : list, optional
            Explicit donor unit identifiers (must be never-treated). Defaults to
            all never-treated units.
        survey_design : optional
            Not yet supported — raises ``NotImplementedError`` if provided.

        Returns
        -------
        SyntheticControlResults
        """
        if survey_design is not None:
            raise NotImplementedError(
                "SyntheticControl does not yet support survey_design. "
                "Survey integration is planned for a later release."
            )

        # --- column validation ---
        required = [outcome, treatment, unit, time]
        if predictors:
            required += list(predictors)
        if special_predictors:
            required += [sp[0] for sp in special_predictors]
        missing = [c for c in dict.fromkeys(required) if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Reject missing structural values up front: a NaN treatment status would be
        # silently dropped by groupby(...).max() (a partially-missing donor history
        # could be misclassified as never-treated), and NaN unit/time break the panel
        # structure. Done BEFORE classification so donor-pool composition is honest.
        for col in (treatment, unit, time):
            if bool(data[col].isna().to_numpy().any()):
                raise ValueError(
                    f"Column {col!r} contains missing (NaN) values; treatment, unit, and "
                    "time must be fully observed (a missing treatment status would "
                    "silently change donor classification)."
                )

        # Validate the treatment indicator on the FULL input BEFORE classifying
        # units: a non-{0,1} history would otherwise be silently dropped from both
        # the treated and never-treated sets by _resolve_treated_and_donors, quietly
        # changing the donor pool / weights / ATT.
        _check_binary(data[treatment].to_numpy(), treatment)

        # --- treated unit + donor pool ---
        treated_id, donor_ids = _resolve_treated_and_donors(
            data, treatment, unit, treated_unit, donor_pool
        )

        # --- restrict to treated + donors; all_periods derived AFTER donor filter ---
        keep_ids = [treated_id] + donor_ids
        sub = cast(pd.DataFrame, data[data[unit].isin(keep_ids)])
        all_periods = sorted(sub[time].unique())

        # balanced panel check (on the analysis subset)
        _check_balanced(sub, unit, time, keep_ids, all_periods)

        # --- pre/post period resolution + canonicalization + cross-checks ---
        pre_periods, post_periods = _resolve_periods(
            sub, time, treatment, unit, treated_id, all_periods, post_periods, self.v_method
        )

        # --- predictor specs (validations 1, 5, 7) ---
        specs = _validate_predictors(
            predictors,
            predictors_op,
            predictor_window,
            special_predictors,
            pre_period_outcomes,
            outcome,
            pre_periods,
        )

        # --- pivots, predictor matrices, outcome matrices ---
        needed_vars = {outcome} | {s.var for s in specs}
        pivots = {
            v: sub.pivot(index=time, columns=unit, values=v).reindex(index=all_periods)
            for v in needed_vars
        }
        Y = pivots[outcome]
        Z1 = Y.loc[pre_periods, treated_id].to_numpy(dtype=float)
        Z0 = Y.loc[pre_periods, donor_ids].to_numpy(dtype=float)  # (n_pre, J)

        # Fail closed on non-finite outcomes for the treated/donor panel: NaN/inf in
        # the outcome would silently propagate into the gap path / ATT (and into Z for
        # the nested objective).
        outcome_block = Y.loc[all_periods, [treated_id] + donor_ids].to_numpy(dtype=float)
        if not np.all(np.isfinite(outcome_block)):
            raise ValueError(
                f"Outcome {outcome!r} contains non-finite (NaN/inf) values for the "
                "treated unit or a donor over the analysis periods; synthetic control "
                "requires a complete outcome panel."
            )

        X1, X0, labels = _build_predictor_matrix(pivots, specs, treated_id, donor_ids)
        # Fail closed on non-finite predictor cells (e.g. a covariate that is only
        # observed on a subset of the pre periods averaged over the full pre window).
        bad = [
            labels[i]
            for i in range(len(labels))
            if not (np.isfinite(X1[i]) and np.all(np.isfinite(X0[i, :])))
        ]
        if bad:
            raise ValueError(
                f"Predictor(s) {bad} have non-finite (NaN/inf) values for the treated "
                "unit or a donor over their selected periods. Restrict predictor_window / "
                "special_predictors periods to where the variable is observed."
            )
        # Standardized predictors feed the nested / custom paths. inverse_variance applies
        # 1/Var to the RAW predictors (standardization would double-rescale to 1/Var²); cv
        # re-standardizes each window separately inside _outer_solve_V_cv (ADH runs a
        # separate dataprep per window). Both bind X1s/X0s to the raw matrices here so they
        # never trigger _standardize's (irrelevant) full-pre zero-variance-row warning on a
        # path that does not use the full-pre standardized matrices.
        k = X1.shape[0]
        J = len(donor_ids)
        if self.v_method in ("inverse_variance", "cv"):
            X1s, X0s = X1, X0
        else:
            X1s, X0s, _ = _standardize(X1, X0, self.standardize)

        # --- solve for V and donor weights ---
        # mspe_v is the OUTER-objective value; it is populated only when a nested V
        # search actually ran (None on the custom and single-donor paths).
        mspe_v: Optional[float] = None
        # ``outer_converged`` tracks whether the nested V search reached an optimum
        # (trivially True when there is no outer search: custom V or a single donor).
        outer_converged = True
        # The RESOLVED cv train/validation split actually used (None unless v_method="cv");
        # surfaced on the Results object as public, pickle-surviving metadata.
        resolved_v_cv_t0: Optional[int] = None
        # Resolve + validate the cv split and the cv predictor precondition UP FRONT, before
        # the single-donor / solve dispatch, so an invalid v_cv_t0 or predictor configuration
        # fails closed and v_cv_t0 is surfaced consistently even on the degenerate J==1 path
        # (where the V search is skipped — the single donor forces w=[1] regardless).
        if self.v_method == "cv":
            n_pre = len(pre_periods)
            if n_pre < 2:
                raise ValueError(
                    f"v_method='cv' requires at least 2 pre-treatment periods (got {n_pre}) "
                    "to form a training and a validation window."
                )
            t0 = self.v_cv_t0 if self.v_cv_t0 is not None else n_pre // 2
            if not (1 <= t0 <= n_pre - 1):
                raise ValueError(
                    f"v_cv_t0={t0} is out of range; it must leave >=1 training and >=1 "
                    f"validation pre-period (1 <= v_cv_t0 <= {n_pre - 1})."
                )
            _, _, all_spanning = _cv_window_status(specs, pre_periods, t0)
            if not all_spanning:
                # PRECONDITION (identification): faithful per-window re-aggregation needs
                # every predictor measurable on BOTH windows — the training-window fit
                # (steps 2-3) AND the validation-window step-4 refit. A predictor present
                # in only one window cannot be re-aggregated on the other, so its V weight
                # would be unidentified. The default per-period outcome lags are
                # single-period and live in one window only, so they violate this.
                raise ValueError(
                    "v_method='cv' requires every predictor to span BOTH the training "
                    "window (pre[:v_cv_t0]) and the validation window (pre[v_cv_t0:]) so it "
                    "can be re-aggregated on each (ADH 2015 cross-validation re-measures "
                    "each predictor per window). At least one predictor is observed in only "
                    "one window. The default per-period outcome lags are single-period and "
                    "cannot span; pass covariate `predictors` or multi-period "
                    "`special_predictors` that cover both windows (or adjust `v_cv_t0`)."
                )
            if not _cv_windows_have_variation(
                pivots, specs, treated_id, donor_ids, pre_periods, t0
            ):
                # IDENTIFICATION (fail-closed): if a window's re-aggregated predictors are
                # constant ACROSS DONORS (all donor columns identical), X0·W is constant in
                # W, so the inner solve's objective is flat and returns arbitrary weights
                # reported as converged. Raise rather than return an arbitrary ATT. (Donor
                # distinguishability identifies W; treated-vs-donor variation does not.)
                raise ValueError(
                    "v_method='cv': a cross-validation window (pre[:v_cv_t0] or "
                    "pre[v_cv_t0:]) has no variation ACROSS DONORS in ANY predictor after "
                    "re-aggregation — every donor has identical predictors in that window, "
                    "so X0·W is constant in W and the weight solve is unidentified (any "
                    "donor weights fit it equally, even when the treated unit differs). "
                    "Adjust `v_cv_t0`, the `predictors`/`special_predictors`, or the donor pool."
                )
            resolved_v_cv_t0 = t0
        if self.v_method == "custom":
            v = self._prepare_custom_v(k)
            w, converged = _inner_solve_W(X1s, X0s, v, self.inner_max_iter, self.inner_min_decrease)
        elif J == 1:
            # Degenerate: a single donor forces w = [1.0], so the predictor-importance V is
            # UNIDENTIFIED — every V yields the same synthetic — for EVERY v_method, cv and
            # inverse_variance included (their selected / closed-form V would be inert). We
            # report a uniform v_weights and leave mspe_v None, and warn. The donor weights /
            # gap path / ATT do not depend on V here, so they are unaffected. This is the
            # documented single-donor contract (REGISTRY §SyntheticControl).
            warnings.warn(
                "Only one donor unit is available; the synthetic control is that "
                "single donor (w = 1) and the V search is skipped (V is unidentified with "
                "one donor), so v_weights is uniform regardless of v_method. SCM is "
                "degenerate with a single donor.",
                UserWarning,
                stacklevel=2,
            )
            v = np.ones(k) / k
            w = np.array([1.0])
            converged = True
        elif self.v_method == "inverse_variance":
            # Closed-form V = 1/Var(X) (Abadie 2021 §3.2(a)); no outer search, so
            # mspe_v stays None and outer_converged stays True. Apply V to the RAW
            # (UNstandardized) predictors: inverse-variance weighting IS the unit-variance
            # rescaling, so the effective objective is Σ_h diff_h²/Var_h (the paper's
            # selector). Using the standardized X1s/X0s here would double-rescale to
            # Σ_h diff_h²/Var_h² — hence the standardize pre-scaling is intentionally
            # bypassed on this branch (equivalent to uniform V on standardized predictors).
            v = _inverse_variance_v(X1, X0)
            w, converged = _inner_solve_W(X1, X0, v, self.inner_max_iter, self.inner_min_decrease)
        elif self.v_method == "cv":
            # Out-of-sample CV V-selection (ADH 2015; Abadie 2021 t0=T0/2). The split and
            # the predictor precondition were resolved/validated up front (above), so
            # resolved_v_cv_t0 is a valid int here.
            assert resolved_v_cv_t0 is not None
            # The 6th return (structural-infeasible flag) is for the placebo/LOO path;
            # fit() enforces the fully-spanning + donor-variation precondition up front
            # (raising ValueError), so the headline solve never returns it -> ignore here.
            v, w, converged, mspe_v, outer_converged, _ = _outer_solve_V_cv(
                pivots,
                specs,
                treated_id,
                donor_ids,
                Z1,
                Z0,
                pre_periods,
                resolved_v_cv_t0,
                self.n_starts,
                self.seed,
                self.optimizer_options,
                self.inner_max_iter,
                self.inner_min_decrease,
                self.standardize,
            )
        else:
            v, w, converged, mspe_v, outer_converged = _outer_solve_V(
                X1,
                X0,
                X1s,
                X0s,
                Z1,
                Z0,
                self.n_starts,
                self.seed,
                self.optimizer_options,
                self.inner_max_iter,
                self.inner_min_decrease,
            )

        # --- gap path, fit diagnostics, ATT ---
        gap_path = _compute_gap_path(Y, w, treated_id, donor_ids, all_periods)
        pre_gaps = np.array([gap_path[p] for p in pre_periods], dtype=float)
        post_gaps = np.array([gap_path[p] for p in post_periods], dtype=float)
        pre_rmspe = float(np.sqrt(np.mean(pre_gaps**2)))
        att = float(np.mean(post_gaps))

        # Treated unit's post/pre RMSPE ratio — the in-space placebo test statistic
        # (ADH 2010 §2.4). Cheap to compute now; the placebo reference distribution
        # is built on demand by SyntheticControlResults.in_space_placebo().
        treated_scale = float(np.max(np.abs(Z1))) if Z1.size else 0.0
        rmspe_ratio = _rmspe_ratio(pre_gaps, post_gaps, treated_scale)

        # Poor-fit warning (REGISTRY contract: warn when pre-RMSPE exceeds the SD of
        # the treated unit's pre-period outcomes). This includes a FLAT treated pre-path
        # (pre_sd == 0): any non-trivial RMSPE then means the synthetic cannot reproduce
        # a constant series. A scale-aware absolute floor (`_fit_tol`) guards against a
        # spurious warning on a near-perfect flat fit (RMSPE ~ roundoff).
        pre_sd = float(np.std(Z1, ddof=1)) if Z1.size > 1 else 0.0
        _fit_tol = 1e-8 * max(float(np.max(np.abs(Z1))) if Z1.size else 0.0, 1.0)
        if pre_rmspe > pre_sd and pre_rmspe > _fit_tol:
            warnings.warn(
                f"Pre-treatment fit is poor: RMSPE ({pre_rmspe:.4f}) exceeds the "
                f"standard deviation of treated pre-treatment outcomes ({pre_sd:.4f}). "
                f"The synthetic control may not adequately reproduce the treated unit's "
                f"pre-treatment trajectory; consider a different donor pool or predictors.",
                UserWarning,
                stacklevel=2,
            )

        # Inner-solve non-convergence warning — fires on nested / custom / k==1 paths.
        warn_if_not_converged(
            converged,
            "Synthetic control inner weight solver (Frank-Wolfe)",
            self.inner_max_iter,
            stacklevel=2,
        )

        # No analytical SE: all inference fields NaN.
        t_stat, p_value, conf_int = safe_inference(att, np.nan, self.alpha)

        # --- reporting structures ---
        donor_weights = {donor_ids[j]: float(w[j]) for j in range(J) if w[j] > _MIN_REPORT_WEIGHT}
        v_weights = {labels[i]: float(v[i]) for i in range(k)}
        # Predictor-balance basis: under cv the reported donor_weights come from the
        # ADH-2015 step-4 refit on the VALIDATION-window re-aggregated predictors, so the
        # balance table is reported on that same basis — making the public surface
        # internally consistent (v_weights + this balance reproduce donor_weights). Every
        # other V method fits on the full-pre predictors and reports balance there.
        if self.v_method == "cv":
            assert resolved_v_cv_t0 is not None
            bal_specs = cast(
                List[_PredictorSpec],
                _truncate_specs_to_window(specs, set(pre_periods[resolved_v_cv_t0:])),
            )
            Xb1, Xb0, _ = _build_predictor_matrix(pivots, bal_specs, treated_id, donor_ids)
        else:
            Xb1, Xb0 = X1, X0
        synthetic_X = Xb0 @ w
        donor_mean = Xb0.mean(axis=1)
        predictor_balance = pd.DataFrame(
            {
                "predictor": labels,
                "treated": Xb1,
                "synthetic": synthetic_X,
                "donor_mean": donor_mean,
            }
        )

        results = SyntheticControlResults(
            att=att,
            se=np.nan,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=int(len(sub)),
            n_donors=J,
            n_pre_periods=len(pre_periods),
            n_post_periods=len(post_periods),
            donor_weights=donor_weights,
            v_weights=v_weights,
            predictor_balance=predictor_balance,
            gap_path=gap_path,
            pre_rmspe=pre_rmspe,
            treated_unit=treated_id,
            pre_periods=list(pre_periods),
            post_periods=list(post_periods),
            v_method=self.v_method,
            standardize=self.standardize,
            alpha=self.alpha,
            mspe_v=mspe_v,
            v_cv_t0=resolved_v_cv_t0,
            rmspe_ratio=rmspe_ratio,
        )
        # Retain the panel state needed to refit each donor as a pseudo-treated
        # unit for in-space placebo inference (ADH 2010 §2.4). Stored as a plain
        # attribute (not a dataclass field) and excluded from pickling via
        # SyntheticControlResults.__getstate__ (it holds the full panel).
        # COPY all caller-owned mutable inputs (the custom_v array, the
        # optimizer_options dict, the specs list) so post-fit mutation of the
        # estimator's inputs cannot silently change in_space_placebo() output on an
        # already-returned results object. (pivots are freshly pivoted here, and the
        # period/id lists are re-listed below, so those are not caller-aliased.)
        #
        # Capture the EXACT (X1s, X0s, V) triple that produced the treated unit's final
        # donor weights, so the ADH-2015 §4 regression_weights() / sparse_synthetic_control()
        # diagnostics can hold V FIXED (unlike leave_one_out / in_time_placebo, which
        # re-search V). For nested / custom / inverse_variance the inner solve ran on X1s/X0s
        # (standardized or raw per the branch above), so those locals are exactly right. For
        # cv the final W is solved on the VALIDATION-window standardized predictors inside
        # _outer_solve_V_cv; reuse the balance matrices (Xb1/Xb0, the identical
        # validation-window re-aggregation) standardized the same way (== the solver's
        # X1s_va/X0s_va, SC _outer_solve_V_cv). The J==1 short-circuit never reaches the cv
        # solve, so guard on J > 1 (the diagnostics fail closed on J < 2 anyway).
        if self.v_method == "cv" and J > 1:
            fit_X1s, fit_X0s, _ = _standardize(Xb1, Xb0, self.standardize)
        else:
            fit_X1s, fit_X0s = X1s, X0s
        results._fit_snapshot = _SyntheticControlFitSnapshot(
            pivots=pivots,
            specs=list(specs),
            outcome=outcome,
            all_periods=list(all_periods),
            pre_periods=list(pre_periods),
            post_periods=list(post_periods),
            donor_ids=list(donor_ids),
            # Freeze the reportably-weighted support (donor_weights keys, in donor_ids
            # order) so leave_one_out() is immune to post-fit mutation of donor_weights.
            weighted_donor_ids=[d for d in donor_ids if d in donor_weights],
            treated_id=treated_id,
            standardize=self.standardize,
            v_method=self.v_method,
            custom_v=(
                None if self.custom_v is None else np.array(self.custom_v, dtype=float, copy=True)
            ),
            n_starts=self.n_starts,
            seed=self.seed,
            optimizer_options=(
                None if self.optimizer_options is None else dict(self.optimizer_options)
            ),
            inner_max_iter=self.inner_max_iter,
            inner_min_decrease=self.inner_min_decrease,
            v_cv_t0=self.v_cv_t0,
            # The V-fixed (X1s, X0s, V) triple captured just above (copied so post-fit
            # mutation cannot change the diagnostics' output).
            fit_X1s=np.array(fit_X1s, dtype=float, copy=True),
            fit_X0s=np.array(fit_X0s, dtype=float, copy=True),
            fit_v=np.array(v, dtype=float, copy=True),
        )
        # Persist whether the treated unit's own fit reached a valid optimum — both
        # the inner Frank-Wolfe weight solve AND (on the nested path) the outer V
        # search. in_space_placebo() fails closed otherwise: ranking a statistic from
        # a truncated / under-optimized treated fit would not be a valid ADH 2010
        # §2.4 permutation (and an under-optimized V is anti-conservative).
        results._fit_converged = bool(converged and outer_converged)

        self.results_ = results
        self.is_fitted_ = True
        return results

    def _prepare_custom_v(self, k: int) -> np.ndarray:
        """Validate ``custom_v`` against the predictor count and trace-normalize."""
        v = np.asarray(self.custom_v, dtype=float).ravel()
        if v.shape[0] != k:
            raise ValueError(
                f"custom_v has length {v.shape[0]} but there are {k} predictors. "
                "custom_v must have exactly one entry per predictor row."
            )
        # Finiteness / non-negativity already enforced in _validate_config.
        return v / v.sum()


def synthetic_control(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit: str,
    time: str,
    **kwargs,
) -> SyntheticControlResults:
    """
    Convenience function for classic synthetic control estimation.

    .. deprecated:: 3.9
        ``synthetic_control()`` is deprecated and will be removed in 4.0
        (row M-074). Construct the estimator instead:
        ``SyntheticControl(...).fit(data, ...)``.

    Constructor-only keyword arguments (``v_method`` — ``"nested"`` / ``"custom"`` /
    ``"cv"`` / ``"inverse_variance"`` — ``custom_v``, ``v_cv_t0``, ``n_starts``,
    ``standardize``, ``alpha``, ``seed``, ``optimizer_options``,
    ``inner_max_iter``, ``inner_min_decrease``) and ``fit`` keyword arguments
    (``post_periods``, ``treated_unit``, ``predictors``, ``special_predictors``,
    ...) may both be passed via ``**kwargs``.

    Examples
    --------
    >>> from diff_diff import synthetic_control
    >>> res = synthetic_control(data, "y", "treated", "unit", "year",
    ...                         predictors=["x1", "x2"])
    >>> print(f"ATT: {res.att:.3f}, pre-RMSPE: {res.pre_rmspe:.3f}")
    """
    warnings.warn(
        "synthetic_control() is deprecated and will be removed in 4.0; "
        "construct the estimator instead: SyntheticControl(...).fit(data, ...).",
        FutureWarning,
        stacklevel=2,
    )
    ctor_keys = set(SyntheticControl().get_params().keys())
    ctor_kwargs = {k: v for k, v in kwargs.items() if k in ctor_keys}
    fit_kwargs = {k: v for k, v in kwargs.items() if k not in ctor_keys}
    estimator = SyntheticControl(**ctor_kwargs)
    return estimator.fit(data, outcome, treatment, unit, time, **fit_kwargs)


# =============================================================================
# module-private helpers
# =============================================================================


def _check_binary(arr: np.ndarray, name: str) -> None:
    """Raise ValueError if ``arr`` contains values other than 0/1."""
    uniq = np.unique(arr[~np.isnan(arr)]) if arr.dtype.kind == "f" else np.unique(arr)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(f"{name} must be binary (0 or 1). Found values: {uniq}")


def _check_balanced(
    sub: pd.DataFrame, unit: str, time: str, keep_ids: List[Any], all_periods: List[Any]
) -> None:
    """Raise ValueError if the analysis subset is not a balanced panel."""
    n_periods = len(all_periods)
    counts = sub.groupby(unit)[time].nunique()
    bad = [u for u in keep_ids if counts.get(u, 0) != n_periods]
    if bad:
        raise ValueError(
            f"Unbalanced panel: units {bad} are not observed at all {n_periods} "
            "periods. Synthetic control requires a balanced panel."
        )
    if len(sub) != len(keep_ids) * n_periods:
        raise ValueError(
            "Panel has duplicate (unit, time) rows; synthetic control requires "
            "exactly one observation per unit-period."
        )


def _resolve_treated_and_donors(
    data: pd.DataFrame,
    treatment: str,
    unit: str,
    treated_unit: Optional[Any],
    donor_pool: Optional[List[Any]],
) -> Tuple[Any, List[Any]]:
    """Identify the single treated unit and the (never-treated) donor pool."""
    ever_treated = cast(pd.Series, data.groupby(unit)[treatment].max())
    treated_units = [u for u, v in ever_treated.items() if v == 1]
    never_treated = [u for u, v in ever_treated.items() if v == 0]

    if treated_unit is not None:
        if treated_unit not in set(data[unit]):
            raise ValueError(f"treated_unit {treated_unit!r} not found in the data.")
        if treated_unit not in treated_units:
            raise ValueError(
                f"treated_unit {treated_unit!r} has no treated (D==1) periods. "
                "The treatment column must mark the treated unit's post periods."
            )
        treated_id = treated_unit
    else:
        if len(treated_units) == 0:
            raise ValueError(
                "No treated unit found (no unit has any D==1 period). Synthetic "
                "control requires exactly one treated unit."
            )
        if len(treated_units) > 1:
            raise ValueError(
                f"Found {len(treated_units)} ever-treated units {treated_units}; "
                "synthetic control requires exactly one. Pass treated_unit= to "
                "select one, and exclude the others from the donor pool."
            )
        treated_id = treated_units[0]

    if donor_pool is not None:
        donor_ids = list(dict.fromkeys(donor_pool))
        if treated_id in donor_ids:
            raise ValueError("donor_pool must not contain the treated unit.")
        missing = [u for u in donor_ids if u not in set(data[unit])]
        if missing:
            raise ValueError(f"donor_pool units not found in the data: {missing}")
        contaminated = [u for u in donor_ids if u not in set(never_treated)]
        if contaminated:
            raise ValueError(
                f"donor_pool contains ever-treated units {contaminated}; donors "
                "must be never-treated."
            )
    else:
        donor_ids = [u for u in never_treated if u != treated_id]

    if len(donor_ids) == 0:
        raise ValueError("No donor units available (the donor pool is empty).")

    return treated_id, donor_ids


def _resolve_periods(
    sub: pd.DataFrame,
    time: str,
    treatment: str,
    unit: str,
    treated_id: Any,
    all_periods: List[Any],
    post_periods: Optional[List[Any]],
    v_method: str,
) -> Tuple[List[Any], List[Any]]:
    """Resolve and validate pre/post periods (absorbing + no-anticipation)."""
    period_index = {p: i for i, p in enumerate(all_periods)}
    treated_rows = sub[sub[unit] == treated_id].set_index(time)[treatment]
    treated_d = {p: int(treated_rows.loc[p]) for p in all_periods}
    treated_active = [p for p in all_periods if treated_d[p] == 1]

    if post_periods is None:
        if not treated_active:
            raise ValueError(
                "Cannot infer post periods: the treated unit has no D==1 periods. "
                "Pass post_periods= explicitly."
            )
        post = sorted(treated_active, key=lambda p: period_index[p])
    else:
        # Canonicalize: unique + calendar-sorted (validation 3).
        seen = set()
        canon = []
        for p in post_periods:
            if p not in period_index:
                raise ValueError(f"post_periods value {p!r} is not a period in the data.")
            if p not in seen:
                seen.add(p)
                canon.append(p)
        if not canon:
            raise ValueError("post_periods must be non-empty.")
        post = sorted(canon, key=lambda p: period_index[p])

    # Contiguous suffix (absorbing assignment): post must be the last len(post)
    # periods of the calendar axis (validation 2).
    suffix = all_periods[len(all_periods) - len(post) :]
    if post != suffix:
        raise ValueError(
            "post_periods must be a contiguous suffix of the time axis (absorbing "
            f"treatment). Got {post}, expected the trailing periods {suffix}."
        )

    pre_periods = all_periods[: len(all_periods) - len(post)]

    # Absorbing-treatment cross-check vs the treated unit's D column (validation 2),
    # enforced on BOTH the inferred and explicit branches:
    #   - no anticipation: the treated unit must be untreated (D==0) in every pre period;
    #   - uninterrupted exposure: the treated unit must be treated (D==1) in every post
    #     period (an explicit suffix like [t2, t3] over a path 0,0,1,0 would otherwise
    #     average a treated period with an untreated one).
    anticipated = [p for p in pre_periods if treated_d[p] == 1]
    if anticipated:
        raise ValueError(
            f"Treated unit has D==1 in pre periods {anticipated}; this violates the "
            "no-anticipation / absorbing-treatment assumption. Redefine post_periods "
            "to begin at the first treated period."
        )
    untreated_post = [p for p in post if treated_d[p] != 1]
    if untreated_post:
        raise ValueError(
            f"Treated unit has D==0 in post periods {untreated_post}; absorbing "
            "treatment requires uninterrupted exposure (D==1) in every post period. "
            "Check post_periods against the treatment column."
        )

    if len(pre_periods) == 0:
        raise ValueError(
            "No pre-treatment periods: synthetic control cannot fit a counterfactual "
            "without at least one pre period."
        )
    if v_method == "nested" and len(pre_periods) == 1:
        warnings.warn(
            "Only one pre-treatment period: data-driven V selection (v_method='nested') "
            "is unreliable with a single pre period (the outer MSPE degenerates to one "
            "term). Consider v_method='custom' or more pre periods.",
            UserWarning,
            stacklevel=2,
        )

    return pre_periods, post


def _validate_predictors(
    predictors: Optional[List[str]],
    predictors_op: str,
    predictor_window: Optional[List[Any]],
    special_predictors: Optional[List[Tuple[str, List[Any], str]]],
    pre_period_outcomes: Optional[Any],
    outcome: str,
    pre_periods: List[Any],
) -> List[_PredictorSpec]:
    """Build the normalized, collision-checked predictor specification list.

    Canonical row ORDER: covariate/predictor averages -> special predictors ->
    per-period outcome lags (the row order matches R ``Synth::dataprep``). NOTE: the
    aggregation itself fails closed on any non-finite cell (handled in ``fit``),
    whereas R ``dataprep`` uses ``na.rm=TRUE`` — a documented deviation (see REGISTRY).
    """
    pre_set = set(pre_periods)
    pre_index = {p: i for i, p in enumerate(pre_periods)}
    specs: List[_PredictorSpec] = []
    labels: set = set()

    def _canon(periods: List[Any]) -> List[Any]:
        # Unique + calendar-sorted, so reordered/duplicated period lists are
        # equivalent (e.g. [c,b,a] == [a,b,c]; a repeated period does not
        # re-weight a "mean"). Assumes membership already checked by _check_periods.
        return sorted(dict.fromkeys(periods), key=lambda p: pre_index[p])

    def _add(label: str, kind: str, var: str, periods: List[Any], op: str) -> None:
        if label in labels:
            raise ValueError(
                f"Duplicate predictor label {label!r}. Each predictor (including "
                "per-period outcome lags) must have a unique label."
            )
        labels.add(label)
        specs.append(_PredictorSpec(label=label, kind=kind, var=var, periods=periods, op=op))

    def _check_periods(periods: List[Any], ctx: str) -> None:
        if len(periods) == 0:
            raise ValueError(f"{ctx} period list must not be empty.")
        leak = [p for p in periods if p not in pre_set]
        if leak:
            raise ValueError(
                f"{ctx} references periods {leak} outside the pre-treatment window "
                "(would leak post-treatment information)."
            )

    # 1) predictor averages
    if predictors:
        if predictors_op not in _OP_FUNCS:
            raise ValueError(
                f"predictors_op must be one of {sorted(_OP_FUNCS)}, got {predictors_op!r}"
            )
        if predictor_window is not None:
            window = list(predictor_window)
            _check_periods(window, "predictor_window")
            window = _canon(window)
        else:
            window = list(pre_periods)
        for var in predictors:
            _add(var, "predictor_avg", var, window, predictors_op)

    # 2) special predictors
    if special_predictors:
        for entry in special_predictors:
            if len(entry) != 3:
                raise ValueError(
                    "Each special predictor must be a (var, periods, op) tuple; " f"got {entry!r}."
                )
            var, periods, op = entry
            if op not in _OP_FUNCS:
                raise ValueError(
                    f"special predictor op must be one of {sorted(_OP_FUNCS)}, got {op!r}"
                )
            periods = list(periods)
            _check_periods(periods, f"special predictor {var!r}")
            periods = _canon(periods)  # reordered/duplicated -> same canonical spec
            # Full ordered period list in the label so distinct period sets sharing
            # the same endpoints/length (e.g. [2000,2002,2004] vs [2000,2003,2004])
            # do not collide — the label is also the v_weights / balance key.
            label = f"{var}@{op}[{','.join(str(p) for p in periods)}]"
            _add(label, "special", var, periods, op)

    # 3) per-period outcome lags
    if pre_period_outcomes is not None:
        if isinstance(pre_period_outcomes, str):
            if pre_period_outcomes != "all":
                raise ValueError(
                    f"pre_period_outcomes string must be 'all', got {pre_period_outcomes!r}"
                )
            lag_periods = list(pre_periods)
        else:
            lag_periods = list(pre_period_outcomes)
            _check_periods(lag_periods, "pre_period_outcomes")
            lag_periods = _canon(lag_periods)
        for p in lag_periods:
            _add(f"{outcome}_{p}", "lag", outcome, [p], "identity")

    # default: nothing specified -> use all pre-period outcomes as lag predictors
    if not specs:
        for p in pre_periods:
            _add(f"{outcome}_{p}", "lag", outcome, [p], "identity")

    return specs


def _build_predictor_matrix(
    pivots: Dict[str, pd.DataFrame],
    specs: List[_PredictorSpec],
    treated_id: Any,
    donor_ids: List[Any],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build X1 (k,) and X0 (k, J) from the normalized predictor specs."""
    k = len(specs)
    J = len(donor_ids)
    X1 = np.empty(k, dtype=float)
    X0 = np.empty((k, J), dtype=float)
    labels: List[str] = []

    for i, spec in enumerate(specs):
        pv = pivots[spec.var]
        block = pv.loc[spec.periods]  # rows: periods, cols: units
        if spec.op == "identity":
            X1[i] = float(block[treated_id].iloc[0])
            X0[i, :] = block[donor_ids].iloc[0].to_numpy(dtype=float)
        else:
            op_func = _OP_FUNCS[spec.op]
            X1[i] = float(op_func(block[treated_id].to_numpy(dtype=float)))
            X0[i, :] = op_func(block[donor_ids].to_numpy(dtype=float), axis=0)
        labels.append(spec.label)

    return X1, X0, labels


def _standardize(
    X1: np.ndarray, X0: np.ndarray, mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-standardize predictors by SD across donors+treated (ddof=1).

    Matches R ``Synth``: ``divisor <- sqrt(apply(cbind(X0, X1), 1, var))``.
    Zero-variance rows keep divisor 1.0 (with a warning).
    """
    k = X1.shape[0]
    if mode == "none":
        return X1.copy(), X0.copy(), np.ones(k)
    combined = np.column_stack([X0, X1.reshape(-1, 1)])  # (k, J+1)
    divisor = np.sqrt(np.var(combined, axis=1, ddof=1))
    zero = divisor <= 0
    if np.any(zero):
        warnings.warn(
            f"{int(np.sum(zero))} predictor row(s) have zero variance across "
            "donors+treated; their standardization divisor is set to 1.0.",
            UserWarning,
            stacklevel=2,
        )
        divisor = divisor.copy()
        divisor[zero] = 1.0
    X1s = X1 / divisor
    X0s = X0 / divisor[:, None]
    return X1s, X0s, divisor


def _inverse_variance_v(X1: np.ndarray, X0: np.ndarray) -> np.ndarray:
    """Closed-form inverse-variance predictor importance ``v_h = 1/Var(X_{h·})``.

    Abadie (2021) §3.2(a). Variance is taken over donors+treated on the
    UNSTANDARDIZED predictors ``X1``/``X0``. The caller applies this V to the RAW
    predictors (NOT the standardized ``X1s``/``X0s``): inverse-variance weighting IS
    the unit-variance rescaling, so the effective objective is ``Σ_h diff_h²/Var_h``;
    applying it to already-standardized rows would double-rescale to
    ``Σ_h diff_h²/Var_h²``. The inverse is **exact** for every strictly-positive
    variance (``1/Var_h``, no flooring of small-but-positive variances — flooring would
    distort the relative weights of rows with tiny unequal variances and break the
    documented closed form). Special handling is reserved for **non-positive** variance:
    a zero-variance predictor row (no cross-unit information) contributes 0 weight,
    matching the inverse-variance multistart seed in ``_v_starts``. If EVERY row is
    zero-variance — or, in the pathological case of a positive variance so tiny that
    ``1/Var_h`` overflows to a non-finite total — the result falls back to uniform V
    (with a ``UserWarning``). Trace-normalized to sum to 1 (``W*`` is invariant to V's
    scale; this just matches ``_prepare_custom_v``).
    """
    k = X1.shape[0]
    combined = np.column_stack([X0, X1.reshape(-1, 1)])
    row_var = np.var(combined, axis=1, ddof=1)
    # Exact 1/Var on the strictly-positive rows only (mask the division so a zero-variance
    # row neither divides by zero nor gets clipped); non-positive rows get 0 weight.
    inv = np.zeros(k)
    pos = row_var > 0
    inv[pos] = 1.0 / row_var[pos]
    total = float(np.sum(inv))
    if total <= 0 or not np.isfinite(total):
        warnings.warn(
            "inverse_variance V: no usable predictor variance across donors+treated "
            "(every row is zero-variance, or a positive variance is so tiny that 1/Var "
            "overflows), so there is no information to weight predictors; falling back to "
            "uniform V.",
            UserWarning,
            stacklevel=2,
        )
        return np.ones(k) / k
    return inv / total


def _inner_solve_W(
    X1s: np.ndarray,
    X0s: np.ndarray,
    v: np.ndarray,
    inner_max_iter: int,
    inner_min_decrease: float,
) -> Tuple[np.ndarray, bool]:
    """Solve W*(V) = argmin_W (X1-X0 W)' diag(V) (X1-X0 W) on the unit simplex.

    Folds V^½ into the predictors and delegates to the Frank-Wolfe simplex solver.
    Returns the RAW simplex weights (no sparsification) so the outer objective is
    not perturbed; reporting-level cleanup happens in ``fit``.
    """
    J = X0s.shape[1]
    if J == 1:
        return np.array([1.0]), True
    vh = np.sqrt(v)
    A = vh[:, None] * X0s  # (k, J)
    b = vh * X1s  # (k,)
    packed = np.column_stack([A, b])  # (k, J+1); last column is the target
    min_decrease = inner_min_decrease * max(float(np.linalg.norm(b)), 1e-12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*did not converge.*", category=UserWarning)
        w, converged = _sc_weight_fw(
            packed,
            zeta=0.0,
            intercept=False,
            min_decrease=min_decrease,
            max_iter=inner_max_iter,
            return_convergence=True,
        )
    return np.asarray(w, dtype=float), bool(converged)


def _v_starts(
    k: int,
    X1: np.ndarray,
    X0: np.ndarray,
    X1s: np.ndarray,
    X0s: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
    inner_max_iter: int,
    inner_min_decrease: float,
) -> Tuple[List[np.ndarray], int, int]:
    """Build a list of DISTINCT starting ``theta`` vectors for the outer V search.

    Returns ``(candidates, n_inner_solves, n_inner_nonconverged)`` — the latter two
    count the inner Frank-Wolfe solves run by the univariate-fit heuristic so the
    caller can surface aggregate intermediate non-convergence.

    Heuristic starts: uniform V; inverse-row-variance V (computed from the
    UNSTANDARDIZED predictors ``X1``/``X0`` — on the standardized rows every variance
    is 1 by construction, so it would collapse to the uniform start); univariate-fit V
    (v_i ∝ 1/MSPE_i from solving with mass concentrated on predictor i). Remaining
    starts are random Dirichlet draws. Candidates are de-duplicated so the multistart
    never runs the same Nelder-Mead seed twice; non-finite candidates are dropped
    (validation 10); uniform is always retained.
    """

    def _to_theta(v: np.ndarray) -> Optional[np.ndarray]:
        v = np.asarray(v, dtype=float)
        # Keep only finite, positive mass; reject candidates with no usable mass
        # (validation 10). Floor before log so a near-zero (or underflowed-to-0)
        # Dirichlet draw cannot yield log(0) = -inf and force an infinite retry.
        v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
        total = float(np.sum(v))
        if total <= 0:
            return None
        v = np.maximum(v / total, 1e-12)
        theta = np.log(v)
        theta = theta - np.mean(theta)
        return theta if np.all(np.isfinite(theta)) else None

    def _add_unique(t: Optional[np.ndarray], pool: List[np.ndarray]) -> None:
        # Append only DISTINCT, finite candidates so the multistart never runs the same
        # Nelder-Mead seed twice (codex: a degenerate heuristic must not waste a start).
        if t is not None and not any(np.allclose(t, e, atol=1e-9) for e in pool):
            pool.append(t)

    # Candidates are generated lazily and we stop as soon as `target` DISTINCT starts are
    # collected, so a small n_starts does not pay for heuristic starts it would only
    # discard. In particular n_starts=1 returns the uniform start without running the
    # O(k) univariate inner-solve loop below.
    target = max(n_starts, 1)
    candidates: List[np.ndarray] = [np.zeros(k)]  # uniform V

    # inverse row variance of the UNSTANDARDIZED predictors over donors+treated.
    # (On the standardized rows every variance is ~1, so this would collapse to the
    # uniform start — using the raw scales makes it a genuinely different seed.)
    if len(candidates) < target:
        combined = np.column_stack([X0, X1.reshape(-1, 1)])
        row_var = np.var(combined, axis=1, ddof=1)
        inv_var = np.where(row_var > 0, 1.0 / np.maximum(row_var, 1e-12), 0.0)
        if np.sum(inv_var) > 0:
            _add_unique(_to_theta(inv_var / np.sum(inv_var)), candidates)

    # univariate-fit start: v_i ∝ 1 / (pre-outcome MSPE of W solved with V=e_i).
    # Skipped entirely when enough candidates are already collected (saves k inner solves).
    inner_total = 0
    inner_nonconv = 0
    if len(candidates) < target:
        uni_mspe = np.empty(k)
        for i in range(k):
            e = np.zeros(k)
            e[i] = 1.0
            w_i, conv_i = _inner_solve_W(X1s, X0s, e, inner_max_iter, inner_min_decrease)
            inner_total += 1
            if not conv_i:
                # Don't trust a truncated solve: inf -> 0 inverse-MSPE weight, so this
                # predictor doesn't shape the heuristic start.
                inner_nonconv += 1
                uni_mspe[i] = np.inf
            else:
                uni_mspe[i] = float(np.mean((Z1 - Z0 @ w_i) ** 2))
        inv_mspe = np.where(uni_mspe > 0, 1.0 / np.maximum(uni_mspe, 1e-12), 0.0)
        if np.sum(inv_mspe) > 0:
            _add_unique(_to_theta(inv_mspe / np.sum(inv_mspe)), candidates)

    # random Dirichlet draws to fill the remaining slots with DISTINCT starts
    attempts = 0
    max_attempts = 20 * n_starts + 20
    while len(candidates) < target and attempts < max_attempts:
        attempts += 1
        _add_unique(_to_theta(rng.dirichlet(np.ones(k))), candidates)

    return candidates[:target], inner_total, inner_nonconv


def _outer_solve_V(
    X1: np.ndarray,
    X0: np.ndarray,
    X1s: np.ndarray,
    X0s: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    n_starts: int,
    seed: Optional[int],
    optimizer_options: Optional[Dict[str, Any]],
    inner_max_iter: int,
    inner_min_decrease: float,
) -> Tuple[np.ndarray, np.ndarray, bool, float, bool]:
    """Data-driven V selection: minimize pre-period outcome MSPE of W*(V).

    Multistart Nelder-Mead over ``theta`` (V = softmax(theta)) plus a derivative-free
    Powell polish from the best point. Returns
    ``(v_star, w_star, inner_converged, mspe, outer_converged)``.
    """
    k = X1s.shape[0]
    if k == 1:
        # Single predictor: V is fixed (no outer search), so outer convergence is
        # trivially satisfied.
        v = np.array([1.0])
        w, converged = _inner_solve_W(X1s, X0s, v, inner_max_iter, inner_min_decrease)
        return v, w, converged, float(np.mean((Z1 - Z0 @ w) ** 2)), True

    # Track inner Frank-Wolfe non-convergence across ALL intermediate evaluations so
    # the outer search cannot silently rank truncated W*(V) solves (codex). `_inner_solve_W`
    # suppresses its own per-call warning during the search; we aggregate here.
    _st = {"total": 0, "nonconv": 0}

    # Finite penalty for a non-converged evaluation: the objective is convex in w, so its
    # maximum over the simplex is attained at a single-donor vertex. Penalizing above that
    # bound guarantees a truncated W*(V) can never win the argmin, while staying FINITE
    # (np.inf would flood scipy's simplex arithmetic with RuntimeWarnings).
    _vertex_mspe = [float(np.mean((Z1 - Z0[:, j]) ** 2)) for j in range(Z0.shape[1])]
    _penalty = 10.0 * (max(_vertex_mspe) + 1.0) if _vertex_mspe else 1.0

    def objective(theta: np.ndarray) -> float:
        v = _softmax(theta)
        w, conv = _inner_solve_W(X1s, X0s, v, inner_max_iter, inner_min_decrease)
        _st["total"] += 1
        if not conv:
            # A truncated W*(V) is unusable for V ranking: in an argmin search even a
            # single non-converged evaluation could win and silently flip the selected V.
            # Penalize above the feasible objective bound so it can never be chosen (and
            # is tallied for the aggregated warning below).
            _st["nonconv"] += 1
            return _penalty
        return float(np.mean((Z1 - Z0 @ w) ** 2))

    nm_options = {"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8}
    if optimizer_options:
        nm_options.update(optimizer_options)
    # Powell uses xtol/ftol rather than Nelder-Mead's xatol/fatol; translate so the
    # same optimizer_options control both solvers without an OptimizeWarning.
    powell_options = dict(nm_options)
    if "xatol" in powell_options:
        powell_options["xtol"] = powell_options.pop("xatol")
    if "fatol" in powell_options:
        powell_options["ftol"] = powell_options.pop("fatol")

    rng = np.random.default_rng(seed)
    starts, start_total, start_nonconv = _v_starts(
        k, X1, X0, X1s, X0s, Z1, Z0, n_starts, rng, inner_max_iter, inner_min_decrease
    )
    _st["total"] += start_total
    _st["nonconv"] += start_nonconv

    # Track convergence of the SELECTED (lowest-objective) incumbent, NOT "any start
    # converged" — a non-winning start's success says nothing about whether the V we
    # actually return is a valid optimum. ``best_success`` follows ``best_x``.
    best_x: np.ndarray = starts[0]
    best_fun = np.inf
    best_success = False
    for theta0 in starts:
        res = minimize(objective, theta0, method="Nelder-Mead", options=nm_options)
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = res.x
            best_success = bool(res.success)

    # Derivative-free polish from the incumbent (best-of, mirrors R optimx). If the
    # polish improves on the incumbent it becomes the selected solution (carry its
    # success); if it does not improve but converged AT the incumbent's objective
    # level, that validates the incumbent as an optimum.
    res_p = minimize(objective, best_x, method="Powell", options=powell_options)
    if res_p.fun < best_fun:
        best_fun = float(res_p.fun)
        best_x = res_p.x
        best_success = bool(res_p.success)
    elif bool(res_p.success) and np.isclose(res_p.fun, best_fun, rtol=1e-5, atol=1e-8):
        # Powell did not improve but converged back AT the incumbent (same objective
        # within tolerance) -> the incumbent is a validated optimum. A Powell run that
        # "succeeds" at a STRICTLY WORSE objective ended elsewhere and says nothing
        # about the selected incumbent's validity, so it must NOT flip best_success
        # (which would silently admit an under-optimized V from a success=False start).
        best_success = True
    outer_converged = best_success

    # Surface a silent under-optimized V: if the SELECTED outer solution did not
    # converge (e.g. optimizer_options={"maxiter": 1}, or the lowest-objective
    # incumbent came from a truncated run), the selected V / donor weights / ATT may
    # be sub-optimal.
    if not outer_converged:
        warnings.warn(
            "Outer V-search (Nelder-Mead / Powell) did not converge; the selected "
            "predictor-importance matrix V (and the resulting donor weights / ATT) may "
            "be sub-optimal. Increase optimizer_options['maxiter'] or n_starts.",
            UserWarning,
            stacklevel=3,
        )

    # Aggregate intermediate inner Frank-Wolfe non-convergence across the whole nested
    # search (univariate starts + every objective evaluation). Non-converged objective
    # evaluations were excluded from V ranking (returned as +inf); warn on ANY such
    # occurrence — unlike a bootstrap summary, an argmin search is sensitive to even one
    # truncated solve, so no rate threshold is appropriate here.
    if _st["nonconv"] > 0:
        warnings.warn(
            f"Inner Frank-Wolfe did not converge on {_st['nonconv']} of {_st['total']} "
            f"weight solves during nested V selection (inner_max_iter={inner_max_iter}); "
            "those evaluations were excluded from V ranking, but the search space was "
            "effectively restricted, so the selected V / donor weights / ATT may be "
            "sub-optimal. Increase inner_max_iter or relax inner_min_decrease.",
            UserWarning,
            stacklevel=3,
        )

    v_star = _softmax(best_x)
    w_star, converged = _inner_solve_W(X1s, X0s, v_star, inner_max_iter, inner_min_decrease)
    mspe = float(np.mean((Z1 - Z0 @ w_star) ** 2))
    return v_star, w_star, converged, mspe, outer_converged


def _truncate_specs_to_window(
    specs: List["_PredictorSpec"], window: set
) -> List[Optional["_PredictorSpec"]]:
    """Re-aggregate each predictor spec over a sub-window of the pre-period.

    Returns a list parallel to ``specs`` whose entry ``i`` is ``specs[i]`` with its
    ``periods`` intersected with ``window`` (so the spec's op — mean/sum/identity — then
    recomputes over only those periods), or ``None`` where the spec has NO period in
    ``window``. RE-aggregating (recomputing the aggregate over the window's periods) is
    the faithful ADH-2015 cross-validation operation — a separate ``dataprep`` per
    window — as opposed to masking a fixed full-pre aggregate in/out (which cannot make
    a spanning predictor out-of-sample, since the full-pre value already bakes in the
    validation-window observations).
    """
    out: List[Optional["_PredictorSpec"]] = []
    for spec in specs:
        kept = [p for p in spec.periods if p in window]
        out.append(replace(spec, periods=kept) if kept else None)
    return out


def _cv_window_status(
    specs: List["_PredictorSpec"], pre_periods: List[Any], t0: int
) -> Tuple[bool, bool, bool]:
    """Classify cv predictor coverage at the positional split ``t0``.

    Returns ``(has_training, has_validation, all_spanning)`` over the spec set, where a
    spec is *training*-active iff any of its periods lie in ``pre[:t0]`` and
    *validation*-active iff any lie in ``pre[t0:]``:

    - ``has_training`` / ``has_validation``: at least one spec is active in that window.
    - ``all_spanning``: EVERY spec is active in BOTH windows — the cv identification
      precondition for faithful per-window re-aggregation. Each predictor can then be
      re-measured on the training window for the V-search fit AND on the validation
      window for the step-4 refit, so the cross-validated ``V*`` is fully identified and
      drives both fits with no zeroed coordinate. This holds for ADH-2015's shared
      covariate / multi-period special predictors (which span the windows) but NOT for
      the window-specific default per-period outcome lags (each lives in one window).
    """
    tr_set = set(pre_periods[:t0])
    va_set = set(pre_periods[t0:])
    has_tr = has_va = False
    all_spanning = True
    for s in specs:
        in_tr = any(p in tr_set for p in s.periods)
        in_va = any(p in va_set for p in s.periods)
        has_tr = has_tr or in_tr
        has_va = has_va or in_va
        if not (in_tr and in_va):
            all_spanning = False
    return has_tr, has_va, all_spanning


def _window_has_donor_variation(X0w: np.ndarray) -> bool:
    """True iff at least one predictor row varies ACROSS DONORS in this window.

    SCM weights are identified by the DONORS being distinguishable: ``X0·W`` is a convex
    combination of the donor columns, so if every predictor row is constant across donors
    (all donor columns identical) then ``X0·W`` is the same for every simplex ``W`` and the
    weight solve is unidentified — ``_inner_solve_W`` returns arbitrary (tie-broken) weights
    while reporting convergence, REGARDLESS of whether the treated unit differs (the treated
    unit is the matching target, not part of W's identification). So this checks donor-side
    variance only, NOT treated-vs-donor variance. ``_standardize`` merely *warns* on
    zero-variance rows, so cv needs this as an explicit fail-closed gate. A single donor
    (``J < 2``) forces ``W=[1]`` and is handled by the J==1 short-circuit, so it is treated
    as identified here (and avoids a ddof=1 variance over one column).
    """
    if X0w.shape[1] < 2:
        return True
    return bool(np.any(np.var(X0w, axis=1, ddof=1) > 0))


def _cv_windows_have_variation(
    pivots: Dict[str, pd.DataFrame],
    specs: List["_PredictorSpec"],
    treated_id: Any,
    donor_ids: List[Any],
    pre_periods: List[Any],
    t0: int,
) -> bool:
    """True iff BOTH cv windows carry identifying cross-DONOR predictor variation.

    For each window, re-aggregate the specs over it and check that at least one row varies
    across donors (see ``_window_has_donor_variation``). A window with no such variation
    leaves the weights unidentified, so the CV fit must fail closed (headline ``ValueError``;
    placebo / backdated refit dropped or marked infeasible) rather than feed arbitrary
    weights into the ATT or the placebo reference set.
    """
    for window in (set(pre_periods[:t0]), set(pre_periods[t0:])):
        wspecs = [s for s in _truncate_specs_to_window(specs, window) if s is not None]
        if not wspecs:
            return False
        _, X0w, _ = _build_predictor_matrix(pivots, wspecs, treated_id, donor_ids)
        if not _window_has_donor_variation(X0w):
            return False
    return True


def _outer_solve_V_cv(
    pivots: Dict[str, pd.DataFrame],
    specs: List["_PredictorSpec"],
    treated_id: Any,
    donor_ids: List[Any],
    Z1: np.ndarray,
    Z0: np.ndarray,
    pre_periods: List[Any],
    t0: int,
    n_starts: int,
    seed: Optional[int],
    optimizer_options: Optional[Dict[str, Any]],
    inner_max_iter: int,
    inner_min_decrease: float,
    standardize: str,
) -> Tuple[np.ndarray, np.ndarray, bool, float, bool, Optional[str]]:
    """Out-of-sample cross-validation V selection (ADH 2015 §; Abadie 2021 Eq. 9).

    Faithful per-window re-aggregation. The pre-period is split positionally at ``t0``
    into a training window ``pre[:t0]`` and a validation window ``pre[t0:]``. Each
    predictor spec is RE-AGGREGATED over each window — its op (mean/sum/identity) is
    recomputed over only the periods that fall in that window — exactly as ADH-2015's
    cross-validation re-runs ``dataprep`` separately on each window. The predictor
    dimension ``k`` is preserved: re-aggregation changes each row's VALUES per window,
    not the number of rows, so V stays ``k``-dimensional throughout.

    - Steps 2-3 (V-search): for each candidate V, fit the training weights on the
      TRAINING-window re-aggregated predictors and rank V by the held-out
      validation-window outcome MSPE ``mean((Z1[t0:] - Z0[t0:]@w)**2)`` -> ``V*``.
      Because each predictor is re-measured on the training window only, the V-search is
      genuinely out-of-sample for ALL predictor types (a spanning predictor's
      validation-window observations never enter the training fit) — the property that
      masking a fixed full-pre aggregate could not deliver.
    - Step 4 (final weights): re-estimate ``W* = W(V*)`` on the VALIDATION-window
      re-aggregated predictors (ADH-2015 "predictors from the last part of the
      pre-period"). The same ``V*`` drives both fits with no zeroed coordinate, so the
      reported ``v_weights`` (= V*) reproduce the reported ``donor_weights`` on the
      validation-window predictors.

    Each window's predictor matrix is standardized SEPARATELY (ADH runs a separate
    ``dataprep`` per window; ``V*`` is predictor-importance applied in each window's own
    scaled space). The caller (``fit`` / ``_truncate_snapshot_in_time``) enforces the
    fully-spanning precondition (every spec observed in BOTH windows) before calling, so
    every re-aggregated spec is non-empty; this guards defensively and fails closed
    (non-converged sentinel) otherwise.

    Returns ``(v_star, w_star, inner_converged, mspe_v, outer_converged, infeasible)``
    where ``mspe_v`` is the step-3 validation-MSPE selection criterion at V* (the
    validation MSPE of the TRAINING-window fit), and ``infeasible`` is ``"infeasible"``
    when the failure is STRUCTURAL (a non-spanning spec set or a re-aggregated window
    with no cross-donor variation — the weights are unidentified, not merely
    under-optimized) and ``None`` otherwise. The caller (``_placebo_fit_unit``) uses it
    to report a dropped placebo as ``status="infeasible"`` (remedied by adjusting the
    predictors / ``v_cv_t0`` / donor pool) vs ``status="failed"`` (a solver
    non-convergence, remedied by ``inner_max_iter`` / ``n_starts``) — mirroring the
    split ``in_time_placebo`` already emits.
    """
    k = len(specs)
    train_specs = _truncate_specs_to_window(specs, set(pre_periods[:t0]))
    val_specs = _truncate_specs_to_window(specs, set(pre_periods[t0:]))
    Z1_va = Z1[t0:]
    Z0_va = Z0[t0:, :]

    # Fail closed if any predictor is absent from a window (a non-fully-spanning spec set
    # — e.g. an in-time-truncated placebo snapshot). The caller enforces the
    # fully-spanning precondition up front for the headline fit and each placebo date;
    # this guards the path defensively, returning a non-converged sentinel so the caller
    # drops/raises rather than crashing on an empty predictor build.
    if any(s is None for s in train_specs) or any(s is None for s in val_specs):
        return np.ones(k) / k, np.zeros(len(donor_ids)), False, float("nan"), False, "infeasible"
    train_specs_c = cast(List["_PredictorSpec"], train_specs)
    val_specs_c = cast(List["_PredictorSpec"], val_specs)

    # Re-aggregate the predictors over each window and standardize each window SEPARATELY.
    X1_tr, X0_tr, _ = _build_predictor_matrix(pivots, train_specs_c, treated_id, donor_ids)
    X1_va, X0_va, _ = _build_predictor_matrix(pivots, val_specs_c, treated_id, donor_ids)

    # Fail closed if a window has NO cross-DONOR predictor variation: the donor columns are
    # then identical, so X0·W is constant in W, the inner solve's objective is flat, and it
    # returns arbitrary weights while reporting convergence — silently corrupting the ATT
    # (headline) or the placebo reference set. fit() raises up front for the headline; here
    # we return the non-converged sentinel so a placebo refit is dropped (_placebo_fit_unit
    # returns None) instead of entering the permutation rank. (Unit-specific: in-space
    # placebos reassign the treated role and shrink the donor pool, so donor-distinguishability
    # must be re-checked for each pseudo-treated unit, not just the headline.)
    if not _window_has_donor_variation(X0_tr) or not _window_has_donor_variation(X0_va):
        return np.ones(k) / k, np.zeros(len(donor_ids)), False, float("nan"), False, "infeasible"

    X1s_tr, X0s_tr, _ = _standardize(X1_tr, X0_tr, standardize)
    X1s_va, X0s_va, _ = _standardize(X1_va, X0_va, standardize)

    if k == 1:
        # Single predictor: V is fixed; re-aggregated in each window (guarded non-empty).
        v = np.array([1.0])
        w_tr, conv_tr = _inner_solve_W(X1s_tr, X0s_tr, v, inner_max_iter, inner_min_decrease)
        w_star, converged = _inner_solve_W(X1s_va, X0s_va, v, inner_max_iter, inner_min_decrease)
        # mspe_v is Eq. 9's validation MSPE of the TRAINING-window fit, so it is only valid
        # when that solve converged; if it truncates, fail closed (nan criterion + outer
        # non-converged) so downstream placebo/LOO diagnostics do not run off an invalid CV
        # criterion (there is no outer search here, so conv_tr IS the "search" convergence).
        mspe_v = float(np.mean((Z1_va - Z0_va @ w_tr) ** 2)) if conv_tr else float("nan")
        # conv_tr=False here is a SOLVER truncation (training fit did not converge), not a
        # structural infeasibility -> infeasible=None so the caller reports "failed".
        return v, w_star, converged, mspe_v, conv_tr, None

    _st = {"total": 0, "nonconv": 0}
    # Penalty bound on the VALIDATION window (the objective's window), so a truncated
    # training fit can never win the argmin. Finite (np.inf floods scipy).
    _vertex_mspe = [float(np.mean((Z1_va - Z0_va[:, j]) ** 2)) for j in range(Z0_va.shape[1])]
    _penalty = 10.0 * (max(_vertex_mspe) + 1.0) if _vertex_mspe else 1.0

    def objective(theta: np.ndarray) -> float:
        v = _softmax(theta)
        # Step 2: fit the training weights on the TRAINING-window re-aggregated predictors.
        w, conv = _inner_solve_W(X1s_tr, X0s_tr, v, inner_max_iter, inner_min_decrease)
        _st["total"] += 1
        if not conv:
            _st["nonconv"] += 1
            return _penalty
        # Step 3: rank V by the held-out validation-window OUTCOME MSPE.
        return float(np.mean((Z1_va - Z0_va @ w) ** 2))

    def _uniformity(theta: np.ndarray) -> float:
        # Squared distance of V=softmax(theta) from uniform; smaller = more uniform.
        v = _softmax(theta)
        return float(np.sum((v - 1.0 / k) ** 2))

    # Deterministic tie-break (Abadie 2021 fn. 7: the CV V* "need not be unique" and an
    # implementation must pick a deterministic tie-break). Among candidates whose
    # validation MSPE ties to tolerance, prefer the V closest to uniform — a principled,
    # start-order-independent choice (the densest V among equally-good optima).
    _tie_rtol, _tie_atol = 1e-9, 1e-12

    def _is_better(
        new_fun: float,
        new_x: np.ndarray,
        new_success: bool,
        cur_fun: float,
        cur_x: Optional[np.ndarray],
        cur_success: bool,
    ) -> bool:
        if cur_x is None:
            return True
        tol = _tie_atol + _tie_rtol * abs(cur_fun)
        if new_fun < cur_fun - tol:
            return True
        if abs(new_fun - cur_fun) <= tol:
            # Tie on the validation MSPE. Prefer a CONVERGED candidate over a non-converged
            # one first — a success=False candidate must not displace a converged incumbent
            # (which would spuriously flip outer_converged to False and drop the fit /
            # placebos). Among EQUALLY-converged ties, prefer the V closest to uniform (the
            # deterministic densest-optimum tie-break, Abadie 2021 fn. 7).
            if new_success != cur_success:
                return new_success
            return _uniformity(new_x) < _uniformity(cur_x) - 1e-15
        return False

    nm_options = {"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8}
    if optimizer_options:
        nm_options.update(optimizer_options)
    powell_options = dict(nm_options)
    if "xatol" in powell_options:
        powell_options["xtol"] = powell_options.pop("xatol")
    if "fatol" in powell_options:
        powell_options["ftol"] = powell_options.pop("fatol")

    rng = np.random.default_rng(seed)
    # Heuristic starts use the TRAINING-window matrices (the V-search fits there) scored
    # on the validation outcomes (the CV criterion), so the univariate seed is aligned
    # with the objective the search actually minimizes.
    starts, start_total, start_nonconv = _v_starts(
        k,
        X1_tr,
        X0_tr,
        X1s_tr,
        X0s_tr,
        Z1_va,
        Z0_va,
        n_starts,
        rng,
        inner_max_iter,
        inner_min_decrease,
    )
    _st["total"] += start_total
    _st["nonconv"] += start_nonconv

    best_x: Optional[np.ndarray] = None
    best_fun = np.inf
    best_success = False
    for theta0 in starts:
        res = minimize(objective, theta0, method="Nelder-Mead", options=nm_options)
        if _is_better(float(res.fun), res.x, bool(res.success), best_fun, best_x, best_success):
            best_fun = float(res.fun)
            best_x = res.x
            best_success = bool(res.success)

    # ``starts`` is non-empty (always includes the uniform start) and the first
    # iteration sets ``best_x`` via the ``cur_x is None`` guard, so it is non-None here.
    assert best_x is not None
    res_p = minimize(objective, best_x, method="Powell", options=powell_options)
    if _is_better(float(res_p.fun), res_p.x, bool(res_p.success), best_fun, best_x, best_success):
        best_fun = float(res_p.fun)
        best_x = res_p.x
        best_success = bool(res_p.success)
    elif bool(res_p.success) and np.isclose(res_p.fun, best_fun, rtol=1e-5, atol=1e-8):
        # Powell converged back AT the incumbent objective → validates it as an optimum
        # (mirrors _outer_solve_V; a "success" at a strictly worse point must NOT flip
        # best_success).
        best_success = True
    outer_converged = best_success

    if not outer_converged:
        warnings.warn(
            "Outer CV V-search (Nelder-Mead / Powell) did not converge; the selected "
            "predictor-importance matrix V (and the resulting donor weights / ATT) may "
            "be sub-optimal. Increase optimizer_options['maxiter'] or n_starts.",
            UserWarning,
            stacklevel=3,
        )
    if _st["nonconv"] > 0:
        warnings.warn(
            f"Inner Frank-Wolfe did not converge on {_st['nonconv']} of {_st['total']} "
            f"weight solves during CV V selection (inner_max_iter={inner_max_iter}); those "
            "evaluations were excluded from V ranking, so the selected V / donor weights / "
            "ATT may be sub-optimal. Increase inner_max_iter or relax inner_min_decrease.",
            UserWarning,
            stacklevel=3,
        )

    assert best_x is not None  # set in the NM loop above; narrows past the Powell branch
    v_star = _softmax(best_x)
    # mspe_v = step-3 selection criterion at V* (validation MSPE of the TRAINING-window fit).
    # It is Eq. 9's criterion only if the training solve converged; if it truncates (e.g.
    # inner_max_iter too small), fail closed — nan criterion + outer non-converged — so the
    # fit and downstream placebo/LOO diagnostics do not run off an invalid CV criterion.
    w_tr, conv_tr = _inner_solve_W(X1s_tr, X0s_tr, v_star, inner_max_iter, inner_min_decrease)
    if conv_tr:
        mspe_v = float(np.mean((Z1_va - Z0_va @ w_tr) ** 2))
    else:
        mspe_v = float("nan")
        outer_converged = False
    # Step 4 reported weights: refit V* on the VALIDATION-window re-aggregated predictors.
    w_star, converged = _inner_solve_W(X1s_va, X0s_va, v_star, inner_max_iter, inner_min_decrease)
    # A non-converged outer search / truncated step-4 refit is a SOLVER failure (the
    # windows spanned + varied, else we returned "infeasible" above) -> infeasible=None.
    return v_star, w_star, converged, mspe_v, outer_converged, None


def _compute_gap_path(
    Y: pd.DataFrame,
    w: np.ndarray,
    treated_id: Any,
    donor_ids: List[Any],
    all_periods: List[Any],
) -> Dict[Any, float]:
    """Period-keyed gap path ``α̂_1t = Y_1t − Σ_j w_j Y_jt`` over all periods."""
    treated_series = Y.loc[all_periods, treated_id].to_numpy(dtype=float)
    donor_block = Y.loc[all_periods, donor_ids].to_numpy(dtype=float)  # (T, J)
    synthetic = donor_block @ w
    gaps = treated_series - synthetic
    return {period: float(g) for period, g in zip(all_periods, gaps)}


# =============================================================================
# in-space placebo permutation inference (ADH 2010 §2.4) — used by
# SyntheticControlResults.in_space_placebo() via function-level import
# =============================================================================


def _mspe(gap_path: Dict[Any, float], periods: List[Any]) -> float:
    """Mean squared prediction error of ``gap_path`` over ``periods``."""
    if not periods:
        return float("nan")
    g = np.array([gap_path[p] for p in periods], dtype=float)
    return float(np.mean(g**2))


def _floored_pre_mspe(pre_gaps: np.ndarray, scale: float) -> float:
    """Pre-period MSPE with the scale-aware floor used as the RMSPE-ratio denominator.

    ``= max(mean(pre_gaps**2), 1e-8 * max(scale, 1)**2)``. Factored out of
    ``_rmspe_ratio`` so the sharp-null test inversion (Firpo & Possebom 2018) can reuse
    the SAME per-unit floored denominator the in-space placebo used — the floor scale is
    PER-UNIT (``scale`` = ``max|Z1|`` of that unit's pre-period outcomes), so this is
    what guarantees ``test_sharp_null(0) == placebo_p_value`` bit-for-bit even when the
    floor bites (a near-perfect pre-fit). The denominator is GRID-INVARIANT under a sharp
    null whose ``f`` is zero in the pre-period (the operational constant/linear families).
    """
    pre_mspe = float(np.mean(pre_gaps**2)) if pre_gaps.size else float("nan")
    floor = 1e-8 * max(float(scale), 1.0) ** 2
    return max(pre_mspe, floor)


def _rmspe_ratio(pre_gaps: np.ndarray, post_gaps: np.ndarray, scale: float) -> float:
    """Post/pre RMSPE ratio — the in-space placebo test statistic (ADH 2010 §2.4).

    Returns ``RMSPE_post / RMSPE_pre = sqrt(MSPE_post / MSPE_pre)`` (the root-scale
    ratio, matching the ``rmspe_ratio`` name). ADH 2010 §3.4 reports the *MSPE*
    ratio (the square of this); the two are monotone-equivalent on nonnegative
    values, so the permutation rank and p-value are identical either way — only the
    reported statistic's scale differs.

    The pre-period MSPE denominator is floored at a scale-aware
    ``1e-8 * max(scale, 1)**2`` (squared-outcome units) BEFORE the square root, so a
    (near-)perfect pre-treatment fit (pre-MSPE → 0) yields a large-but-FINITE ratio
    rather than ``inf``/``nan`` (which would corrupt the permutation rank).
    ``scale`` is the magnitude of the unit's pre-period outcomes. Mirrors the
    ``_fit_tol`` poor-fit guard in ``fit()``.
    """
    post_mspe = float(np.mean(post_gaps**2)) if post_gaps.size else float("nan")
    return float(np.sqrt(post_mspe / _floored_pre_mspe(pre_gaps, scale)))


# =============================================================================
# Sharp-null test inversion (Firpo & Possebom 2018, "Synthetic Control Method:
# Inference, Sensitivity Analysis and Confidence Sets," J. Causal Inference 6(2)).
# These helpers RE-RANK the gap paths the in-space placebo already computed (Eqs
# 12-13, φ=0, v=(1,...,1)); no synthetic-control refits. The benchmark f≡0 case is
# identically the existing in-space placebo permutation (Eq 5 = Eq 13 at f≡0).
# =============================================================================


def _constant_f_post(c: float, n_post: int) -> np.ndarray:
    """Constant-in-time post-period effect path f(t)=c (Firpo & Possebom Eq 15)."""
    return np.full(n_post, float(c), dtype=float)


def _linear_f_post(c_tilde: float, n_post: int) -> np.ndarray:
    """Linear-in-time post-period effect path f(t)=c̃·(t−T0) (Firpo & Possebom Eq 17).

    ``(t−T0)`` is the 1-based post-period index (1 for the first post period), so the
    path is ``c̃·[1, 2, …, n_post]`` aligned to calendar-sorted ``post_periods``.
    """
    return float(c_tilde) * np.arange(1, n_post + 1, dtype=float)


def _rmspe_f_ratio(
    gap_path: Dict[Any, float],
    post_periods: List[Any],
    f_post: np.ndarray,
    pre_denom: float,
) -> float:
    """RMSPE^f ratio under a common-effect sharp null (Firpo & Possebom Eq 12).

    Subtracts the post-period effect path ``f_post`` (length ``len(post_periods)``,
    aligned to calendar-sorted ``post_periods``) from the unit's post gaps, then divides
    by the GRID-INVARIANT floored pre-period denominator ``pre_denom`` (the pre window is
    ``f``-free for the operational constant/linear families, Eqs 15/17, so the
    denominator is unchanged across the inversion grid). Returns
    ``sqrt(mean_post((g_t − f_t)**2) / pre_denom)``.
    """
    post = np.array([gap_path[p] for p in post_periods], dtype=float)
    resid = post - np.asarray(f_post, dtype=float)
    post_mspe = float(np.mean(resid**2)) if resid.size else float("nan")
    return float(np.sqrt(post_mspe / pre_denom))


def _sharp_null_pvalue(
    treated_gap: Dict[Any, float],
    placebo_gaps: Dict[Any, Dict[Any, float]],
    post_periods: List[Any],
    f_post: np.ndarray,
    pre_denoms: Dict[Any, float],
    treated_id: Any,
) -> Tuple[float, float, int]:
    """Permutation p-value for a common-effect sharp null (Firpo & Possebom Eq 13, φ=0, v=1).

    ``p^f = (1 + #{converged placebos j : RMSPE^f_j ≥ RMSPE^f_1}) / (n_ref + 1)`` — the
    treated unit is the "+1" and ``n_ref`` is the number of converged placebos (the SAME
    reference set the in-space placebo built). ``placebo_gaps`` maps each converged donor
    to its gap path; ``pre_denoms`` maps each unit (treated + those donors) to its floored
    pre-denominator. Ties counted via ``>=`` so the p-value is conservative, matching
    ``in_space_placebo``. Returns ``(p, rmspe_f_treated, n_ref)``.
    """
    f_post = np.asarray(f_post, dtype=float)
    r1 = _rmspe_f_ratio(treated_gap, post_periods, f_post, pre_denoms[treated_id])
    n_ref = 0
    n_ge = 0
    for j, g in placebo_gaps.items():
        rj = _rmspe_f_ratio(g, post_periods, f_post, pre_denoms[j])
        n_ref += 1
        if rj >= r1:
            n_ge += 1
    p = (1 + n_ge) / (n_ref + 1)
    return p, r1, n_ref


def _invert_sharp_null(
    treated_gap: Dict[Any, float],
    placebo_gaps: Dict[Any, Dict[Any, float]],
    post_periods: List[Any],
    pre_denoms: Dict[Any, float],
    treated_id: Any,
    family: str,
    gamma: float,
    *,
    bounds: Optional[Tuple[float, float]] = None,
    n_grid: int = 200,
) -> Dict[str, Any]:
    """Invert the sharp-null RMSPE^f test over a one-parameter effect family.

    Returns the confidence set ``{ param : p^param > gamma }`` (Firpo & Possebom Eqs
    14/16/18; STRICT inequality -- ``p == gamma`` is excluded). ``family`` is ``"constant"``
    (``f(t)=param``, Eq 15) or ``"linear"`` (``f(t)=param*(t-T0)``, Eq 17).

    ``p^param`` is a piecewise-constant step function of the scalar ``param`` (each placebo's
    indicator flips only at the real roots of a quadratic in ``param``), so with
    ``bounds=None`` the EXACT set is recovered with NO shape assumption (no "centered
    interval" / monotonicity): the placebo breakpoints partition the line, ``p`` is evaluated
    once per induced interval (and the two unbounded tails), and the union of intervals with
    ``p > gamma`` is the set -- correctly handling accepted tails, disjoint components, and the
    empty / unbounded cases. With explicit ``bounds`` a fixed ``linspace(*bounds, n_grid)``
    grid is scanned instead (grid-limited membership). Each ``p`` is a pure O(J) re-ranking.

    Result dict: ``{lower, upper, contiguous, status, n_ref, point_estimate, grid}`` where
    ``status`` is ``"ran"`` / ``"empty"`` / ``"unbounded"``, ``point_estimate`` is the treated
    unit's own RMSPE-minimizing ``param`` (the ATT for constant, the no-intercept OLS slope
    for linear -- reported for reference, NOT assumed to maximize ``p``), and ``grid`` is a
    list of ``(param, p_value, in_set)`` rows for the returned table.
    """
    if family not in ("constant", "linear"):
        raise ValueError(f"family must be 'constant' or 'linear', got {family!r}")
    n_post = len(post_periods)
    # Family shape s_t and per-unit moments of A_u(param) = mean_t((g_ut - param*s_t)**2)
    # = S2*param**2 - 2*P_u*param + Q_u  (s_t = 1 constant / (t - T0) linear; S2 = mean(s**2)).
    s = (
        np.ones(n_post, dtype=float)
        if family == "constant"
        else np.arange(1, n_post + 1, dtype=float)
    )
    S2 = float(np.mean(s**2)) if n_post else 0.0

    def moments(gap_path: Dict[Any, float], unit: Any) -> Tuple[float, float, float]:
        g = np.array([gap_path[p] for p in post_periods], dtype=float)
        return float(np.mean(g * s)), float(np.mean(g**2)), pre_denoms[unit]

    P1, Q1, D1 = moments(treated_gap, treated_id)
    placebo_mom = [moments(gp, j) for j, gp in placebo_gaps.items()]
    n_ref = len(placebo_mom)
    # The treated unit's own RMSPE-minimizing param: reported as the point estimate ONLY
    # (it is NOT assumed to maximize p -- see the exact-inversion comment below).
    center = float(P1 / S2) if S2 > 0 else 0.0

    def pval(param: float) -> float:
        # p^param = (1 + #{placebos j : A_j/D_j >= A_1/D_1}) / (n_ref + 1). Comparing A/D is
        # monotone-equivalent to comparing the RMSPE ratios sqrt(A/D), so this is identical to
        # _sharp_null_pvalue / test_sharp_null (the squared form just avoids the sqrt).
        r1 = ((S2 * param - 2.0 * P1) * param + Q1) / D1
        n_ge = sum(
            1 for (Pj, Qj, Dj) in placebo_mom if ((S2 * param - 2.0 * Pj) * param + Qj) / Dj >= r1
        )
        return (1 + n_ge) / (n_ref + 1)

    def grid_rows(lo: float, hi: float) -> List[Tuple[float, float, bool]]:
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return []
        out: List[Tuple[float, float, bool]] = []
        for x in np.linspace(lo, hi, max(int(n_grid), 2)):
            p = pval(float(x))
            out.append((float(x), float(p), bool(p > gamma)))
        return out

    # ----- fixed-grid path (user-supplied bounds) -----
    if bounds is not None:
        lo_b, hi_b = float(bounds[0]), float(bounds[1])
        if hi_b <= lo_b:
            raise ValueError(f"bounds must be (lo, hi) with hi > lo, got {bounds!r}")
        rows = grid_rows(lo_b, hi_b)
        accepted = [x for x, _, ok in rows if ok]
        if not accepted:
            return {
                "lower": np.nan,
                "upper": np.nan,
                "contiguous": True,
                "status": "empty",
                "n_ref": n_ref,
                "point_estimate": center,
                "grid": rows,
            }
        lower, upper = min(accepted), max(accepted)
        contiguous = all(ok for x, _, ok in rows if lower <= x <= upper)
        return {
            "lower": lower,
            "upper": upper,
            "contiguous": contiguous,
            "status": "ran",
            "n_ref": n_ref,
            "point_estimate": center,
            "grid": rows,
        }

    # ----- exact piecewise-constant inversion (bounds=None; no shape assumption) -----
    # p^param is a step function of param: each placebo's indicator flips only at the real
    # roots of A_j(param)*D1 - A_1(param)*Dj = 0, a quadratic in param. So the EXACT set is
    # recovered with NO "centered interval" / monotonicity assumption -- collect every placebo
    # breakpoint, evaluate p once on each open interval they induce (plus the two unbounded
    # tails), and union the intervals with p > gamma. This correctly handles accepted tails,
    # disjoint components, and the empty / unbounded cases: a poor-pre-fit treated unit can
    # have its accepted region in the TAILS, not around the point estimate.
    if gamma < 1.0 / (n_ref + 1):
        # p >= 1/(J+1) everywhere (the treated ranks itself), so nothing is ever rejected
        # (Firpo & Possebom fn 8 -- the discrete granularity); the set is all of R.
        return {
            "lower": -np.inf,
            "upper": np.inf,
            "contiguous": True,
            "status": "unbounded",
            "n_ref": n_ref,
            "point_estimate": center,
            "grid": [],
        }

    def quad_roots(a: float, b: float, c: float) -> List[float]:
        # Real roots of a*x**2 + b*x + c, degenerate-safe (linear / constant fall through).
        if abs(a) <= 1e-15:
            return [] if abs(b) <= 1e-15 else [-c / b]
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return []
        sq = disc**0.5
        return [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)]

    # g_j(param) = A_j(param)*D1 - A_1(param)*Dj = a*param^2 + b*param + c; the indicator
    # 1[g_j(param) >= 0] == 1[RMSPE^param_j >= RMSPE^param_1], and breakpoints (where an
    # indicator flips) are the real roots of each g_j.
    coeffs = [
        (S2 * (D1 - Dj), -2.0 * (Pj * D1 - P1 * Dj), Qj * D1 - Q1 * Dj)
        for (Pj, Qj, Dj) in placebo_mom
    ]
    raw_breaks: List[float] = []
    for a, b, c in coeffs:
        raw_breaks.extend(quad_roots(a, b, c))
    breaks: List[float] = []
    for r in sorted(raw_breaks):
        # dedup near-equal roots so each interval midpoint stays strictly interior
        if not breaks or abs(r - breaks[-1]) > 1e-12 + 1e-9 * abs(r):
            breaks.append(r)

    def pval_break(r: float) -> float:
        # p AT a breakpoint: strict-`>` membership (Eq 14) combined with the conservative
        # tie-counting `>=` (Eqs 5/13) means a placebo that EXACTLY ties the treated at a root
        # is counted, so p can spike above gamma there even when both neighboring open
        # intervals are rejected (a tangent / co-located root => an isolated accepted point).
        # A relative tolerance registers the tie robustly in floating point.
        n_ge = 0
        for a, b, c in coeffs:
            gj = (a * r + b) * r + c
            if gj >= -(1e-9 * (abs(a) * r * r + abs(b) * abs(r) + abs(c)) + 1e-12):
                n_ge += 1
        return (1 + n_ge) / (n_ref + 1)

    if not breaks:
        # p is constant in param (e.g. every placebo shares the treated's moments + denom).
        if pval(center) > gamma:
            return {
                "lower": -np.inf,
                "upper": np.inf,
                "contiguous": True,
                "status": "unbounded",
                "n_ref": n_ref,
                "point_estimate": center,
                "grid": [],
            }
        return {
            "lower": np.nan,
            "upper": np.nan,
            "contiguous": True,
            "status": "empty",
            "n_ref": n_ref,
            "point_estimate": center,
            "grid": [],
        }

    # Atoms along the line: cell_0, b_0, cell_1, b_1, ..., b_{m-1}, cell_m (2m+1 atoms). p is
    # constant on each open cell (sampled at an interior point) and is evaluated WITH the tie
    # tolerance at each breakpoint. The set = union of accepted atoms; consecutive accepted
    # atoms share a boundary => one connected component, a rejected atom splits components
    # (so an accepted breakpoint whose neighbor cells are both rejected is a singleton).
    m = len(breaks)
    cell_pts = (
        [breaks[0] - 1.0]
        + [0.5 * (breaks[k] + breaks[k + 1]) for k in range(m - 1)]
        + [breaks[-1] + 1.0]
    )
    cell_in = [pval(x) > gamma for x in cell_pts]
    break_in = [pval_break(r) > gamma for r in breaks]
    acc = [cell_in[i // 2] if i % 2 == 0 else break_in[(i - 1) // 2] for i in range(2 * m + 1)]
    if not any(acc):
        return {
            "lower": np.nan,
            "upper": np.nan,
            "contiguous": True,
            "status": "empty",
            "n_ref": n_ref,
            "point_estimate": center,
            "grid": [],
        }

    def atom_extent(i: int) -> Tuple[float, float]:
        if i % 2 == 0:
            k = i // 2  # open cell k = (left boundary, right boundary)
            return (-np.inf if k == 0 else breaks[k - 1], np.inf if k == m else breaks[k])
        b = breaks[(i - 1) // 2]  # a breakpoint singleton
        return (b, b)

    components: List[Tuple[float, float]] = []
    i = 0
    while i < len(acc):
        if not acc[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(acc) and acc[j + 1]:
            j += 1
        components.append((atom_extent(i)[0], atom_extent(j)[1]))
        i = j + 1

    lower = min(comp[0] for comp in components)
    upper = max(comp[1] for comp in components)
    contiguous = len(components) == 1
    status = "ran" if (np.isfinite(lower) and np.isfinite(upper)) else "unbounded"
    # Display grid: the finite hull, else a padded breakpoint range (for inspection only).
    if np.isfinite(lower) and np.isfinite(upper):
        if upper <= lower:
            # A pure singleton set (lower == upper, a lone accepted breakpoint): a zero-width
            # range yields an empty grid_rows(), so emit the single accepted point explicitly
            # (with its tie-spike p) — the inspection table must reflect the non-empty set.
            grid = [(float(lower), float(pval_break(lower)), True)]
        else:
            grid = grid_rows(lower, upper)
    else:
        pad = max(1.0, breaks[-1] - breaks[0])
        grid = grid_rows(breaks[0] - pad, breaks[-1] + pad)
    return {
        "lower": float(lower),
        "upper": float(upper),
        "contiguous": bool(contiguous),
        "status": status,
        "n_ref": n_ref,
        "point_estimate": center,
        "grid": grid,
    }


def _placebo_fit_unit(
    snap: _SyntheticControlFitSnapshot,
    unit: Any,
    donor_pool: List[Any],
    n_starts: int,
) -> Tuple[Optional[Tuple[Dict[Any, float], float]], str]:
    """Refit a synthetic control for one (pseudo-)treated ``unit`` vs ``donor_pool``.

    Reuses the exact predictor build / standardization / weight solve / gap-path
    path as ``fit()`` — none of those helpers reads estimator (``self``) state, so
    the refit is driven entirely by the snapshot. The real treated unit is never in
    ``donor_pool`` (the caller passes the other ``J−1`` donors).

    Returns ``(result, status)`` where ``status`` is one of:

    - ``"ran"`` — ``result`` is ``(gap_path, rmspe_ratio)`` (a valid optimum).
    - ``"infeasible"`` — ``result`` is ``None``; the refit is STRUCTURALLY
      unidentified (non-finite cells, or — under ``v_method="cv"`` — a re-aggregated
      window with no cross-donor variation for this pseudo-treated pool). Remedied by
      adjusting the predictors / ``v_cv_t0`` / donor pool, NOT the optimizer budget.
    - ``"failed"`` — ``result`` is ``None``; the refit reached no valid optimum (a
      non-converged inner weight solve or outer V search). Remedied by a larger
      ``inner_max_iter`` / ``n_starts``.

    The caller excludes both non-``"ran"`` outcomes from the permutation reference set
    but counts them separately (``n_failed`` vs ``n_infeasible``), mirroring the split
    ``in_time_placebo`` already emits. Per-placebo ``UserWarning``s (poor fit,
    zero-variance row, non-convergence) are suppressed here; the caller surfaces an
    aggregate count.
    """
    X1, X0, _ = _build_predictor_matrix(snap.pivots, snap.specs, unit, donor_pool)
    # Belt-and-suspenders: fit() already gated non-finite outcomes over the full
    # treated+donor panel, so a donor reassigned as pseudo-treated has finite cells.
    if not (np.all(np.isfinite(X1)) and np.all(np.isfinite(X0))):
        # Non-finite predictor cells make the pool structurally unfittable (a data
        # problem, not a solver hiccup) -> "infeasible". (Defensive: fit() already gated
        # finiteness over the full treated+donor panel.)
        return None, "infeasible"
    # inverse_variance applies 1/Var to the RAW predictors; cv re-standardizes each window
    # separately inside _outer_solve_V_cv (mirrors fit()). Both skip the full-pre
    # _standardize pass (unused on those paths) and its irrelevant zero-variance warning.
    if snap.v_method in ("inverse_variance", "cv"):
        X1s, X0s = X1, X0
    else:
        X1s, X0s, _ = _standardize(X1, X0, snap.standardize)
    Y = snap.pivots[snap.outcome]
    Z1 = Y.loc[snap.pre_periods, unit].to_numpy(dtype=float)
    Z0 = Y.loc[snap.pre_periods, donor_pool].to_numpy(dtype=float)
    # ``outer_converged`` is trivially True when there is no outer V search (custom
    # V or a single-donor degenerate pool). ``infeasible_reason`` is set to
    # ``"infeasible"`` only by the cv path's structural sentinel (below); every other
    # path's non-convergence is a solver failure.
    outer_converged = True
    infeasible_reason: Optional[str] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if snap.v_method == "custom":
            # custom_v length == k (= len(specs)) is unit-agnostic, so it stays
            # valid for every placebo; it was finiteness/non-negativity-checked at
            # construction. Just trace-normalize (matches _prepare_custom_v).
            v = np.asarray(snap.custom_v, dtype=float).ravel()
            v = v / v.sum()
            w, converged = _inner_solve_W(X1s, X0s, v, snap.inner_max_iter, snap.inner_min_decrease)
        elif len(donor_pool) == 1:
            # Degenerate: a single donor forces w = [1.0]; V is irrelevant.
            w, converged = np.array([1.0]), True
        elif snap.v_method == "inverse_variance":
            # Recompute the closed-form V for THIS pseudo-treated unit's predictors,
            # applied to the RAW predictors (deterministic; no outer search; the
            # standardize pre-scaling is bypassed to avoid the 1/Var² double-rescale —
            # see fit()). outer_converged stays True.
            v = _inverse_variance_v(X1, X0)
            w, converged = _inner_solve_W(X1, X0, v, snap.inner_max_iter, snap.inner_min_decrease)
        elif snap.v_method == "cv":
            # Reproduce the FULL per-window re-aggregation CV procedure for this
            # pseudo-treated unit so placebos use the SAME estimator as the treated fit.
            # The positional t0 split and fully-spanning precondition depend only on
            # snap.specs / snap.pre_periods (NOT on which unit is treated), so a headline
            # fit that satisfied fully-spanning satisfies it for every pseudo-treated unit;
            # n_pre>=2 and t0 in range are likewise guaranteed (the headline snapshot was
            # fit()-validated, and an in-time-truncated snapshot has >=2 pre-fake periods
            # with an out-of-range pinned t0 nulled to the default).
            n_pre = len(snap.pre_periods)
            t0 = snap.v_cv_t0 if snap.v_cv_t0 is not None else n_pre // 2
            _, w, converged, _, outer_converged, infeasible_reason = _outer_solve_V_cv(
                snap.pivots,
                snap.specs,
                unit,
                donor_pool,
                Z1,
                Z0,
                snap.pre_periods,
                t0,
                n_starts,
                snap.seed,
                snap.optimizer_options,
                snap.inner_max_iter,
                snap.inner_min_decrease,
                snap.standardize,
            )
        else:
            _, w, converged, _, outer_converged = _outer_solve_V(
                X1,
                X0,
                X1s,
                X0s,
                Z1,
                Z0,
                n_starts,
                snap.seed,
                snap.optimizer_options,
                snap.inner_max_iter,
                snap.inner_min_decrease,
            )
    # Exclude a placebo whose fit is not a valid optimum — a truncated inner W OR an
    # under-optimized outer V search. An under-optimized placebo V fits the pre-period
    # worse, shrinking its RMSPE ratio and biasing the permutation p-value
    # anti-conservatively, so such placebos must not silently enter the rank. Attribute
    # a cv structural sentinel to "infeasible"; every other non-convergence to "failed".
    if not (converged and outer_converged):
        return None, ("infeasible" if infeasible_reason == "infeasible" else "failed")
    gap_path = _compute_gap_path(Y, w, unit, donor_pool, snap.all_periods)
    pre_gaps = np.array([gap_path[p] for p in snap.pre_periods], dtype=float)
    post_gaps = np.array([gap_path[p] for p in snap.post_periods], dtype=float)
    scale = float(np.max(np.abs(Z1))) if Z1.size else 0.0
    return (gap_path, _rmspe_ratio(pre_gaps, post_gaps, scale)), "ran"


# =============================================================================
# in-time (backdating) placebo (ADH 2015 §4) — used by
# SyntheticControlResults.in_time_placebo() via function-level import
# =============================================================================


def _truncate_snapshot_in_time(
    snap: _SyntheticControlFitSnapshot,
    t_f: Any,
) -> Tuple[Optional[_SyntheticControlFitSnapshot], List[str]]:
    """Build a snapshot for an in-time placebo reassigning the intervention to ``t_f``.

    Re-cuts the panel so the **pre-fake** window is the pre-periods strictly before
    ``t_f`` and the **post-fake** window is the held-out pre-periods from ``t_f``
    onward (``t_f`` is the first post-fake period). The true post-treatment periods
    are EXCLUDED entirely — ``all_periods`` is set to ``pre_fake + post_fake`` so the
    placebo refit (which fits V/W on ``pre_periods`` and measures the gap over
    ``all_periods``) never sees treatment-contaminated data.

    Predictor specs are TRUNCATED to the pre-fake window (ADH 2015 §4 "lag the
    predictors accordingly"; for our absolute-period specs this means intersecting
    each spec's ``periods`` with the pre-fake window). A spec whose window lies
    ENTIRELY in the held-out region is DROPPED (its label is collected for an
    aggregated warning), and ``custom_v`` is subset in lockstep with the surviving
    specs so its length stays ``== len(kept_specs)``.

    Returns ``(modified_snapshot, dropped_spec_labels)``, or ``(None, dropped)`` when
    the date is infeasible: fewer than 2 pre-fake periods (the weight / V solve needs
    at least 2), no post-fake period, or every predictor dropped. ``t_f`` must be an
    element of ``snap.pre_periods`` (the caller validates membership).
    """
    pre = snap.pre_periods
    idx = pre.index(t_f)  # positional split (period labels may be non-numeric)
    new_pre = list(pre[:idx])
    new_post = list(pre[idx:])  # t_f is the first post-fake period
    pre_set = set(new_pre)

    kept_specs: List[_PredictorSpec] = []
    keep_mask: List[int] = []
    dropped: List[str] = []
    for i, spec in enumerate(snap.specs):
        # Intersect with the pre-fake window (preserves the spec's canonical order).
        kept_periods = [p for p in spec.periods if p in pre_set]
        if kept_periods:
            # NEW spec object — never mutate the shared snapshot specs in place.
            kept_specs.append(replace(spec, periods=kept_periods))
            keep_mask.append(i)
        else:
            dropped.append(spec.label)

    # Feasibility: >=2 pre-fake periods, >=1 post-fake period (to measure the placebo
    # gap), and at least one surviving predictor. The >=2 pre-fake rule is DELIBERATELY
    # stricter than the base estimator's T0>=1 allowance (which warns that single-pre
    # nested-V selection is unreliable): an auto-swept placebo date with a single
    # pre-fake period is a trivially-matchable, non-credible pre-fit, so it is dropped
    # as infeasible rather than surfaced as a "ran" placebo. Documented as a Note in
    # docs/methodology/REGISTRY.md §SyntheticControl.
    if len(new_pre) < 2 or len(new_post) < 1 or not kept_specs:
        return None, dropped

    # Subset custom_v IN LOCKSTEP with the surviving specs (custom_v length must stay
    # == k = len(specs); a desync silently misaligns V on the custom path). RAVEL first:
    # fit() accepts array-like custom_v (e.g. a (1, k) row vector) and the snapshot keeps
    # its original shape, so `[keep_mask]` would otherwise index the wrong axis of a 2D
    # custom_v and raise (matches _placebo_fit_unit, which also ravels at use).
    new_custom_v = (
        None if snap.custom_v is None else np.asarray(snap.custom_v, dtype=float).ravel()[keep_mask]
    )
    # A custom V whose surviving entries sum to ~0 (all mass was on dropped specs)
    # cannot fit — _placebo_fit_unit's `v / v.sum()` would be 0/0. This is the date
    # being INFEASIBLE UNDER THE SUPPLIED custom_v, not a solver-convergence failure,
    # so return None (→ status="infeasible") rather than letting the refit "fail".
    if new_custom_v is not None and float(np.sum(new_custom_v)) <= 0.0:
        return None, dropped
    # An explicitly pinned v_cv_t0 (v_method="cv") may exceed the truncated pre-fake
    # window's valid range; null it to the //2 default for the placebo refit (the
    # placebo is a robustness re-run, not the headline — an explicit v_cv_t0 is not
    # preserved across in-time truncation). None already means "use the default".
    new_v_cv_t0 = snap.v_cv_t0
    if new_v_cv_t0 is not None and not (1 <= new_v_cv_t0 <= len(new_pre) - 1):
        new_v_cv_t0 = None
    # Under v_method="cv", in-time truncation can break the fully-spanning precondition
    # even when the headline fit satisfied it: a spec may lose its training-window (or
    # validation-window) periods to the truncation and stop spanning both windows. That is
    # a STRUCTURAL infeasibility of the date (not a solver-convergence failure) — return
    # None (→ status="infeasible") so it is not mislabeled "failed" with
    # optimizer-remediation messaging. Uses the same fully-spanning precondition the
    # headline fit enforces (over the kept, re-truncated specs at the truncated split).
    if snap.v_method == "cv":
        eff_t0 = new_v_cv_t0 if new_v_cv_t0 is not None else len(new_pre) // 2
        _, _, all_spanning = _cv_window_status(kept_specs, new_pre, eff_t0)
        if not all_spanning:
            return None, dropped
        # Truncation can also leave a window with NO cross-unit variation (its predictors
        # become constant across treated+donors) even when fully-spanning still holds —
        # another STRUCTURAL infeasibility of the date (unidentified weight solve), so mark
        # it infeasible rather than letting the refit return arbitrary "converged" weights.
        if not _cv_windows_have_variation(
            snap.pivots, kept_specs, snap.treated_id, snap.donor_ids, new_pre, eff_t0
        ):
            return None, dropped
    snap_mod = replace(
        snap,
        specs=kept_specs,
        all_periods=new_pre + new_post,
        pre_periods=new_pre,
        post_periods=new_post,
        custom_v=new_custom_v,
        v_cv_t0=new_v_cv_t0,
        # Drop the baseline fit's (X1s, X0s, V) triple: it was built on the FULL pre-period
        # predictors, so it must never be paired with these truncated specs. No in-time
        # consumer reads it today, but nulling closes the latent trap up front.
        fit_X1s=None,
        fit_X0s=None,
        fit_v=None,
    )
    return snap_mod, dropped


# =============================================================================
# ADH-2015 §4 tail diagnostics (opt-in) — used by SyntheticControlResults
# .regression_weights() / .sparse_synthetic_control() via function-level import
# =============================================================================


def _regression_weights(X1s: np.ndarray, X0s: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    """Regression-weight extrapolation diagnostic (ADH 2015 §4, journal pp. 498-499).

    Returns the implied donor weights ``W^reg = X0a'(X0a X0a')^{-1} X1a`` of the
    regression counterfactual ``B̂'X_1`` (``B̂ = (X0 X0')^{-1} X0 Y0'``), where ``X0a`` /
    ``X1a`` prepend an intercept row of ones to the predictor matrices. With the constant,
    ``ι'W^reg = 1`` (under full row rank), so the regression is ALSO a weighting estimator
    summing to one — but with UNRESTRICTED weights (can be < 0 or > 1), i.e. it
    extrapolates outside the donors' convex hull (the simplex-constrained synthetic control
    cannot). Detectable by flagging ``W^reg`` entries outside ``[0, 1]``.

    Computed as the min-norm least-squares solution ``argmin_w ||X0a w - X1a||`` =
    ``pinv(X0a) @ X1a`` (equal to the gram-inverse form under full row rank, but avoids
    squaring the condition number; ``lstsq`` returns the rank directly). **At full row rank**
    the system is exactly solvable, so ``W^reg`` is invariant to per-predictor row scaling
    (standardization only divides rows by SD): the result is identical across the
    standardized and raw predictor spaces. In the **rank-deficient** case the fit is inexact,
    and row scaling reweights the least-squares residual metric, so the min-norm ``W^reg`` is
    computed in the captured fit space and can differ across predictor spaces (still a valid
    extrapolation witness there, just space-dependent).

    Returns ``(w_reg, rank_deficient, weight_sum)``. ``rank_deficient`` is True when the
    augmented predictor matrix is not full ROW rank (``k+1 > J``, or collinear predictors);
    the min-norm ``W^reg`` is then still an informative extrapolation witness, but need not
    sum to 1 (so ``weight_sum`` may deviate from 1 exactly in that case).
    """
    J = X0s.shape[1]
    X0a = np.vstack([np.ones((1, J), dtype=float), np.asarray(X0s, dtype=float)])  # (k+1, J)
    X1a = np.concatenate([[1.0], np.asarray(X1s, dtype=float).ravel()])  # (k+1,)
    w_reg, _, rank, _ = np.linalg.lstsq(X0a, X1a, rcond=None)
    rank_deficient = bool(rank < X0a.shape[0])  # not full ROW rank (rows = k+1)
    w_reg = np.asarray(w_reg, dtype=float)
    return w_reg, rank_deficient, float(np.sum(w_reg))


def _sparse_search_size(snap: "_SyntheticControlFitSnapshot", size: int) -> Dict[str, Any]:
    """Exhaustive best size-``size`` donor subset, holding V FIXED at the baseline fit.

    Iterates every ``C(J, size)`` donor subset in deterministic ``itertools.combinations``
    order, solves the inner simplex-constrained weight problem on that subset with the
    baseline fit's FIXED ``V`` (``snap.fit_v`` — NOT re-searched, per ADH 2015 footnote 20),
    and returns the subset minimizing the pre-period OUTCOME MSPE (strict ``<`` tie-break, so
    the first subset in combination order wins ties). Non-converged subset solves are
    excluded from the argmin and counted. Only the WINNER's full gap path / ATT is
    materialized (the search itself computes only the cheap pre-MSPE per subset).

    Returns a dict always carrying ``n_subsets_evaluated`` / ``n_failed`` / ``all_failed``;
    when ``all_failed`` is False it also carries the winner's ``donor_ids`` (tuple),
    ``weights`` (dict), ``gap_path``, ``att``, ``pre_rmspe``, ``post_rmspe``, ``rmspe_ratio``.
    """
    X1s = snap.fit_X1s
    X0s = snap.fit_X0s
    v = snap.fit_v
    assert X1s is not None and X0s is not None and v is not None
    donor_ids = snap.donor_ids
    Y = snap.pivots[snap.outcome]
    pre_periods = snap.pre_periods
    post_periods = snap.post_periods

    # Pre-period treated + donor OUTCOMES for ranking (uniform-outcome MSPE — the SC pre-fit
    # metric that the reported pre_rmspe uses; the pre-period gaps equal this residual).
    Z1_pre = Y.loc[pre_periods, snap.treated_id].to_numpy(dtype=float)  # (T0,)
    Z0_pre_all = Y.loc[pre_periods, donor_ids].to_numpy(dtype=float)  # (T0, J)

    n_eval = 0
    n_failed = 0
    best_mspe = np.inf
    best: Optional[Tuple[List[int], np.ndarray]] = None
    # Suppress the per-solve non-convergence / poor-fit warnings across the (up to
    # max_subsets) inner solves; the aggregate is surfaced via n_failed instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cols in itertools.combinations(range(len(donor_ids)), size):
            n_eval += 1
            cols_arr = list(cols)
            w_sub, conv = _inner_solve_W(
                X1s, X0s[:, cols_arr], v, snap.inner_max_iter, snap.inner_min_decrease
            )
            if not conv:
                n_failed += 1
                continue
            resid = Z1_pre - Z0_pre_all[:, cols_arr] @ w_sub
            mspe = float(np.mean(resid**2))
            if mspe < best_mspe:  # strict < → first subset in combination order wins ties
                best_mspe = mspe
                best = (cols_arr, w_sub)

    if best is None:
        return {"n_subsets_evaluated": n_eval, "n_failed": n_failed, "all_failed": True}

    cols_arr, w_sub = best
    sub_ids = [donor_ids[c] for c in cols_arr]
    gap_path = _compute_gap_path(Y, w_sub, snap.treated_id, sub_ids, snap.all_periods)
    pre_gaps = np.array([gap_path[p] for p in pre_periods], dtype=float)
    post_gaps = np.array([gap_path[p] for p in post_periods], dtype=float)
    scale = float(np.max(np.abs(Z1_pre))) if Z1_pre.size else 0.0
    return {
        "donor_ids": tuple(sub_ids),
        "weights": {sub_ids[i]: float(w_sub[i]) for i in range(len(sub_ids))},
        "gap_path": gap_path,
        "att": float(np.mean(post_gaps)),
        "pre_rmspe": float(np.sqrt(_mspe(gap_path, pre_periods))),
        "post_rmspe": float(np.sqrt(_mspe(gap_path, post_periods))),
        "rmspe_ratio": _rmspe_ratio(pre_gaps, post_gaps, scale),
        "n_subsets_evaluated": n_eval,
        "n_failed": n_failed,
        "all_failed": False,
    }
