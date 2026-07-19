"""WooldridgeDiD: Extended Two-Way Fixed Effects (ETWFE) estimator.

Implements Wooldridge (2025, 2023) ETWFE, faithful to the Stata jwdid package.

References
----------
Wooldridge (2025). Two-Way Fixed Effects, the Two-Way Mundlak Regression,
  and Difference-in-Differences Estimators. Empirical Economics, 69(5), 2545-2587.
Wooldridge (2023). Simple approaches to nonlinear difference-in-differences
  with panel data. The Econometrics Journal, 26(3), C31-C66.
Friosavila (2021). jwdid: Stata module. SSC s459114.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.linalg import compute_robust_vcov, solve_logit, solve_ols, solve_poisson
from diff_diff.utils import (
    pre_demean_norms,
    safe_inference,
    snap_absorbed_regressors,
    within_transform,
)
from diff_diff.wooldridge_results import WooldridgeDiDResults

_VALID_METHODS = ("ols", "logit", "poisson")
_VALID_CONTROL_GROUPS = ("never_treated", "not_yet_treated")
_VALID_BOOTSTRAP_WEIGHTS = ("rademacher", "webb", "mammen")


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logistic_deriv(x: np.ndarray) -> np.ndarray:
    p = _logistic(x)
    return p * (1.0 - p)


def _compute_weighted_agg(
    gt_effects: Dict,
    gt_weights: Dict,
    gt_keys: List,
    gt_vcov: Optional[np.ndarray],
    alpha: float,
    df: Optional[float] = None,
) -> Dict:
    """Compute simple (overall) weighted average ATT and SE via delta method."""
    post_keys = [(g, t) for (g, t) in gt_keys if t >= g]
    w_total = sum(gt_weights.get(k, 0) for k in post_keys)
    if w_total == 0:
        att = float("nan")
        se = float("nan")
    else:
        att = (
            sum(gt_weights.get(k, 0) * gt_effects[k]["att"] for k in post_keys if k in gt_effects)
            / w_total
        )
        if gt_vcov is not None:
            w_vec = np.array(
                [gt_weights.get(k, 0) / w_total if k in post_keys else 0.0 for k in gt_keys]
            )
            var = float(w_vec @ gt_vcov @ w_vec)
            se = float(np.sqrt(max(var, 0.0)))
        else:
            se = float("nan")

    t_stat, p_value, conf_int = safe_inference(att, se, alpha=alpha, df=df)
    return {"att": att, "se": se, "t_stat": t_stat, "p_value": p_value, "conf_int": conf_int}


def _resolve_survey_for_wooldridge(survey_design, sample, cluster_ids, cluster_name):
    """Resolve survey design, inject cluster as PSU, recompute metadata.

    Shared helper for all three WooldridgeDiD sub-fitters.  Matches the
    resolution chain in DifferenceInDifferences.fit() (estimators.py:344-359).
    """
    from diff_diff.survey import (
        _inject_cluster_as_psu,
        _resolve_effective_cluster,
        _resolve_survey_for_fit,
        compute_survey_metadata,
    )

    resolved, survey_weights, survey_weight_type, survey_metadata = _resolve_survey_for_fit(
        survey_design, sample
    )
    if resolved is not None and resolved.uses_replicate_variance:
        raise NotImplementedError(
            "WooldridgeDiD does not yet support replicate-weight variance. "
            "Use TSL (strata/PSU/FPC) instead."
        )
    if resolved is not None and resolved.weight_type != "pweight":
        raise ValueError(
            f"WooldridgeDiD survey support requires weight_type='pweight', "
            f"got '{resolved.weight_type}'. The survey variance math "
            f"assumes probability weights (pweight)."
        )
    if resolved is not None:
        effective_cluster = _resolve_effective_cluster(resolved, cluster_ids, cluster_name)
        if effective_cluster is not None:
            resolved = _inject_cluster_as_psu(resolved, effective_cluster)
            if resolved.psu is not None and survey_metadata is not None:
                raw_w = (
                    sample[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(sample), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved, raw_w)
    df_inf = resolved.df_survey if resolved is not None else None
    return resolved, survey_weights, survey_weight_type, survey_metadata, df_inf


def _warn_and_fill_nan_cohort(df: pd.DataFrame, cohort: str, stacklevel: int) -> pd.DataFrame:
    """Fill NaN cohort with 0 (never-treated) and warn with the row count.

    Used by both `_filter_sample` (pre-fit) and `WooldridgeDiD.fit()` so the
    silent recategorization is surfaced on whichever entry path the caller
    hits first. See REGISTRY.md §WooldridgeDiD (axis-E silent coercion).
    """
    n_nan_cohort = int(df[cohort].isna().sum())
    if n_nan_cohort > 0:
        warnings.warn(
            f"{n_nan_cohort} row(s) have NaN cohort values; filling with 0 "
            f"and treating the corresponding units as never-treated. Pass "
            f"an explicit never-treated marker (0) if this is not intended.",
            UserWarning,
            stacklevel=stacklevel,
        )
    df[cohort] = df[cohort].fillna(0)
    return df


def _suggest_nonlinear_method(outcome_values) -> Optional[str]:
    """Outcome-fit hint for ``WooldridgeDiD(method="ols")``.

    Return ``"logit"`` when the outcome support is binary (exactly ``{0, 1}``)
    or ``"poisson"`` when it is a non-negative integer count (>2 distinct
    values), else ``None``.

    OLS/Gaussian QMLE is a valid method for any response (Wooldridge 2023
    Table 1), but for binary / count outcomes a matching nonlinear model
    (logit / Poisson) is often the *more appropriate* specification: it imposes
    parallel trends on the link/index scale rather than in levels (level-PT is
    implausible for such limited outcomes), and where that nonlinear mean holds
    Wooldridge's (2023) simulations show the linear model is both biased and
    less precise. It rests on a *different identifying assumption* — which one
    fits depends on which parallel-trends restriction holds — so this is a
    recommended comparison, not a free upgrade and not a canonical-link
    requirement.

    Pure and non-fatal: non-numeric / all-non-finite / empty inputs return
    ``None`` rather than raising, so a fit that would otherwise succeed is
    never broken by the hint.
    """
    try:
        y = np.asarray(outcome_values, dtype=float)
    except (TypeError, ValueError):
        return None
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None
    uniq = np.unique(y)
    # Binary: support is exactly {0, 1} (both present) -> Bernoulli QMLE (logit).
    if uniq.size == 2 and np.all((uniq == 0) | (uniq == 1)):
        return "logit"
    # Count: non-negative integers with >2 distinct values -> Poisson QMLE.
    if uniq.size > 2 and np.all(y >= 0) and np.all(y == np.floor(y)):
        return "poisson"
    return None


def _filter_sample(
    data: pd.DataFrame,
    unit: str,
    time: str,
    cohort: str,
    control_group: str,
    anticipation: int,
) -> pd.DataFrame:
    """Return the analysis sample following jwdid selection rules.

    All treated units keep ALL observations (pre- and post-treatment) for
    proper FE estimation. The control_group setting affects which additional
    control observations are included, AND the interaction matrix structure
    (see _build_interaction_matrix).
    """
    df = data.copy()
    df = _warn_and_fill_nan_cohort(df, cohort, stacklevel=3)

    treated_mask = df[cohort] > 0

    if control_group == "never_treated":
        control_mask = df[cohort] == 0
    else:  # not_yet_treated
        # Keep untreated-at-t observations for not-yet-treated units
        control_mask = (df[cohort] == 0) | (df[cohort] > df[time])

    return df[treated_mask | control_mask].copy()


def _build_interaction_matrix(
    data: pd.DataFrame,
    cohort: str,
    time: str,
    anticipation: int,
    control_group: str = "not_yet_treated",
    method: str = "ols",
) -> Tuple[np.ndarray, List[str], List[Tuple[Any, Any]]]:
    """Build the saturated cohort×time interaction design matrix.

    For ``not_yet_treated``: only post-treatment cells (t >= g - anticipation).
    Pre-treatment obs from treated units sit in the regression baseline alongside
    not-yet-treated controls.

    For ``never_treated`` + OLS: ALL (g, t) pairs for each treated cohort. This
    "absorbs" pre-treatment obs from treated units into their own indicators so
    they do not serve as implicit controls in the baseline. Only never-treated
    observations remain in the omitted category. Pre-treatment coefficients
    (t < g) serve as placebo/pre-trend tests.

    For ``never_treated`` + nonlinear (logit/Poisson): post-treatment cells only.
    Nonlinear paths use explicit cohort + time dummies (not within-transformation),
    so including all (g, t) cells would create exact collinearity between each
    cohort dummy and the sum of its cell indicators.

    Returns
    -------
    X_int : (n, n_cells) binary indicator matrix
    col_names : list of string labels "g{g}_t{t}"
    gt_keys : list of (g, t) tuples in same column order
    """
    groups = sorted(g for g in data[cohort].unique() if g > 0)
    times = sorted(data[time].unique())
    cohort_vals = data[cohort].values
    time_vals = data[time].values

    # OLS + never_treated: all (g,t) pairs (placebo via within-transform FE)
    # Nonlinear + never_treated: post-treatment only (avoids cohort dummy collinearity)
    # not_yet_treated: post-treatment only (always)
    include_pre = control_group == "never_treated" and method == "ols"

    cols = []
    col_names = []
    gt_keys = []

    for g in groups:
        for t in times:
            if include_pre or t >= g - anticipation:
                indicator = ((cohort_vals == g) & (time_vals == t)).astype(float)
                cols.append(indicator)
                col_names.append(f"g{g}_t{t}")
                gt_keys.append((g, t))

    if not cols:
        return np.empty((len(data), 0)), [], []
    return np.column_stack(cols), col_names, gt_keys


def _prepare_covariates(
    data: pd.DataFrame,
    exovar: Optional[List[str]],
    xtvar: Optional[List[str]],
    xgvar: Optional[List[str]],
    cohort: str,
    time: str,
    demean_covariates: bool,
    groups: List[Any],
) -> Optional[np.ndarray]:
    """Build covariate matrix following jwdid covariate type conventions.

    Returns None if no covariates, else (n, k) array.
    """
    parts = []

    if exovar:
        parts.append(data[exovar].values.astype(float))

    if xtvar:
        if demean_covariates:
            # Within-cohort×period demeaning
            grp_key = data[cohort].astype(str) + "_" + data[time].astype(str)
            tmp = data[xtvar].copy()
            for col in xtvar:
                tmp[col] = tmp[col] - tmp.groupby(grp_key)[col].transform("mean")
            parts.append(tmp.values.astype(float))
        else:
            parts.append(data[xtvar].values.astype(float))

    if xgvar:
        for g in groups:
            g_indicator = (data[cohort] == g).values.astype(float)
            for col in xgvar:
                parts.append((g_indicator * data[col].values).reshape(-1, 1))

    if not parts:
        return None
    return np.hstack([p if p.ndim == 2 else p.reshape(-1, 1) for p in parts])


class WooldridgeDiD:
    """Extended Two-Way Fixed Effects (ETWFE) DiD estimator.

    Implements the Wooldridge (2025) saturated cohort×time regression
    (*Empirical Economics* 69(5), 2545-2587; DOI
    10.1007/s00181-025-02807-z) and Wooldridge (2023) nonlinear
    extensions (logit, Poisson). Produces all four ``jwdid_estat``
    aggregation types: simple, group, calendar, event. Opt-in surfaces
    include paper W2025 Section 7 cohort-share aggregation
    (``aggregate(weights="cohort_share")``, Eqs. 7.4 + 7.6) and paper
    W2025 Section 8 heterogeneous cohort-specific linear trends
    (``cohort_trends=True``, Eq. 8.1; OLS path only).

    Parameters
    ----------
    method : {"ols", "logit", "poisson"}
        Estimation method. ``"ols"`` is the linear baseline — valid for any
        response (Wooldridge 2023) and the usual choice for continuous outcomes;
        ``"logit"`` for binary or fractional outcomes; ``"poisson"`` for count
        data. When ``method="ols"`` is used on a binary (``{0, 1}``) or
        non-negative integer-count outcome, a ``UserWarning`` notes that a
        matching nonlinear model (logit / Poisson) is often the *more
        appropriate* specification — it imposes parallel trends on the link
        scale rather than in levels, and Wooldridge's (2023) simulations show
        the linear model both biased and less precise for such outcomes when
        the nonlinear mean holds. It rests on a different identifying assumption
        than linear OLS, so it is a recommended comparison, not an automatic
        switch; suppress via ``warnings.filterwarnings``.
    control_group : {"not_yet_treated", "never_treated"}
        Which units serve as the comparison group.  "not_yet_treated" (jwdid
        default) uses all untreated observations at each time period;
        "never_treated" uses only units never treated throughout the sample.
    anticipation : int
        Number of periods before treatment onset to include as treatment cells
        (anticipation effects).  0 means no anticipation.
    demean_covariates : bool
        If True (jwdid default), ``xtvar`` covariates are demeaned within each
        cohort×period cell before entering the regression.  Set to False to
        replicate jwdid's ``xasis`` option.
    alpha : float
        Significance level for confidence intervals.
    cluster : str or None
        Column name to use for cluster-robust SEs.  Defaults to the ``unit``
        identifier passed to ``fit()``.
    n_bootstrap : int
        Number of bootstrap replications.  0 disables bootstrap.
    bootstrap_weights : {"rademacher", "webb", "mammen"}
        Bootstrap weight distribution.
    seed : int or None
        Random seed for reproducibility.
    rank_deficient_action : {"warn", "error", "silent"}
        How to handle rank-deficient design matrices.
    vcov_type : {"classical", "hc1", "hc2", "hc2_bm", "conley"}, default "hc1"
        Variance-covariance family for the analytical sandwich, OLS path only.
        ``hc1`` (default) preserves the prior bit-equal CR1 Liang-Zeger
        cluster-robust behavior via the within-transform path. ``hc2_bm``
        auto-routes to a full-dummy saturated design (intercept + treatment
        cells + unit dummies + time dummies) — FWL preserves cohort coefficients
        but NOT the hat matrix, so HC2 leverage and Bell-McCaffrey Satterthwaite
        DOF must be computed on the full FE projection (matches
        ``clubSandwich::vcovCR(lm(...), type="CR2") + coef_test()$df_Satt``).
        ``classical`` / ``hc2`` are supported via the same full-dummy route AND
        an auto-drop of the unit auto-cluster (one-way families don't compose
        with cluster_ids per the linalg validator). Explicit ``cluster="X"`` +
        one-way ``vcov_type`` raises at the validator. ``"conley"`` (Conley 1999
        spatial-HAC) threads the ``conley_*`` params through ``solve_ols`` on the
        within-transform design (``conley_lag_cutoff=0`` = within-period spatial only;
        ``>0`` adds within-unit Bartlett serial — the panel-aware path, not pooled
        cross-sectional, since ``conley_time`` / ``conley_unit`` are always supplied); the
        unit auto-cluster is dropped (an explicit ``cluster=`` enables the
        spatial+cluster product kernel) and ``survey_design=`` / ``weights`` /
        ``n_bootstrap>0`` are rejected. Conley is OLS-path-only; it routes
        through the full-dummy design when ``cohort_trends=True`` (same as the
        other full-dummy families), and its vcov flows through
        ``aggregate("group"|"calendar"|"event")``.

        ``method`` in ``{"logit","poisson"}`` + ``vcov_type != "hc1"`` is
        REJECTED at ``__init__``: the GLM QMLE sandwich path uses pseudo-
        residuals, and CR2-BM composition with QMLE on canonical-link pseudo-
        residuals needs derivation + R parity (tracked in DEFERRED.md). Survey
        designs combined with ``vcov_type != "hc1"`` raise
        ``NotImplementedError`` at ``fit()`` because the survey TSL / replicate-
        refit variance overrides the analytical sandwich.
    cohort_trends : bool, default False
        When True, adds linear ``dg_i · t`` cohort-specific trend
        interactions to the design matrix per paper W2025 Section 8 /
        Eq. 8.1. Under a heterogeneous-trends DGP this recovers ``τ``
        even when parallel trends fails (paper Section 8.3). OLS-path
        only: ``cohort_trends=True`` + ``method ∈ {"logit","poisson"}``
        raises ``NotImplementedError`` at ``__init__``. Auto-routes to
        the full-dummy design regardless of ``vcov_type`` (matching the
        absorb→fixed_effects auto-route). Each treated cohort must have
        ≥ 2 observed pre-periods in the analysis sample for ``dg_i · t``
        to be separately identified from cohort + time FE; ``fit()``
        raises ``ValueError`` otherwise. On all-eventually-treated
        panels the last cohort's trend column is dropped per paper
        Section 5.4. ``cohort_trends=True`` + ``survey_design`` raises
        ``NotImplementedError`` at ``fit()`` (deferred follow-up).
        ``cohort_trends=True`` + ``control_group="never_treated"``
        also raises ``NotImplementedError`` at ``fit()`` because the
        OLS + never_treated branch emits ALL ``(g, t)`` placebo cell
        dummies (paper Section 4.4 placebo coverage); the appended
        ``dg_i · t`` trend columns are linearly spanned by the
        per-cohort sum of those cell dummies, so the Section 8 trend
        specification is unidentified on this branch. Use
        ``control_group="not_yet_treated"`` (the default) for the
        cohort_trends surface.
    """

    def __init__(
        self,
        method: str = "ols",
        control_group: str = "not_yet_treated",
        anticipation: int = 0,
        demean_covariates: bool = True,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        vcov_type: str = "hc1",
        cohort_trends: bool = False,
        conley_coords: Optional[Tuple[str, str]] = None,
        conley_cutoff_km: Optional[float] = None,
        conley_metric: str = "haversine",
        conley_kernel: str = "bartlett",
        conley_lag_cutoff: Optional[int] = None,
    ) -> None:
        self._validate_constructor_args(
            method=method,
            control_group=control_group,
            anticipation=anticipation,
            bootstrap_weights=bootstrap_weights,
            vcov_type=vcov_type,
            cohort_trends=cohort_trends,
        )

        self.method = method
        self.control_group = control_group
        self.anticipation = anticipation
        self.demean_covariates = demean_covariates
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.vcov_type = vcov_type
        self.cohort_trends = cohort_trends
        self.conley_coords = conley_coords
        self.conley_cutoff_km = conley_cutoff_km
        self.conley_metric = conley_metric
        self.conley_kernel = conley_kernel
        self.conley_lag_cutoff = conley_lag_cutoff
        # Track whether the user explicitly opted out of the "hc1" default.
        # The auto-cluster-at-unit default in `_fit_ols` is suppressed only
        # when the user explicitly opts into a one-way family (``hc2``,
        # ``classical``). ``hc1`` and ``hc2_bm`` preserve the auto-cluster
        # (route to CR1 / CR2 Bell-McCaffrey at unit respectively). Mirrors
        # the SunAbraham PR #472 pattern at ``sun_abraham.py:572``.
        self._vcov_type_explicit = vcov_type != "hc1"

        self.is_fitted_: bool = False
        self._results: Optional[WooldridgeDiDResults] = None

    @staticmethod
    def _validate_constructor_args(
        *,
        method: str,
        control_group: str,
        anticipation: int,
        bootstrap_weights: str,
        vcov_type: str,
        cohort_trends: bool = False,
    ) -> None:
        """Shared validation for both ``__init__`` and ``set_params``.

        Catches the input-contract surface (allowed sets, ranges, and the
        ``method`` × ``vcov_type`` interaction) without depending on instance
        state, so ``set_params`` can re-run it after mutation.
        """
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
        if control_group not in _VALID_CONTROL_GROUPS:
            raise ValueError(
                f"control_group must be one of {_VALID_CONTROL_GROUPS}, got {control_group!r}"
            )
        if anticipation < 0:
            raise ValueError(f"anticipation must be >= 0, got {anticipation}")
        if bootstrap_weights not in _VALID_BOOTSTRAP_WEIGHTS:
            raise ValueError(
                f"bootstrap_weights must be one of {_VALID_BOOTSTRAP_WEIGHTS}, "
                f"got {bootstrap_weights!r}"
            )
        if vcov_type not in ("classical", "hc1", "hc2", "hc2_bm", "conley"):
            raise ValueError(
                f"vcov_type must be one of "
                f"{{'classical','hc1','hc2','hc2_bm','conley'}}; got '{vcov_type}'"
            )
        if method != "ols" and vcov_type != "hc1":
            raise NotImplementedError(
                f"WooldridgeDiD(method={method!r}, vcov_type={vcov_type!r}) is "
                "not yet supported. The logit / poisson paths use a QMLE "
                "sandwich with pseudo-residuals (probs*(1-probs) or mu_hat "
                "weights); composing HC2 leverage and Bell-McCaffrey "
                "Satterthwaite DOF with QMLE on canonical-link pseudo-"
                "residuals needs derivation + R parity against "
                "clubSandwich::vcovCR(glm(...)). Tracked in DEFERRED.md "
                "(WooldridgeDiD logit/poisson vcov_type follow-up row). "
                "Use vcov_type='hc1' (default) for non-OLS methods."
            )
        if cohort_trends and method != "ols":
            raise NotImplementedError(
                f"WooldridgeDiD(method={method!r}, cohort_trends=True) is "
                f"not supported. Paper W2025 Section 8 / Eq. 8.1 specifies "
                f"heterogeneous cohort-specific linear trends `dg_i · t` "
                f"only for the OLS path (Eq. 5.3 POLS); the paper does not "
                f"extend Section 8 to logit / Poisson. Set "
                f"cohort_trends=False or use method='ols'."
            )

    @property
    def results_(self) -> WooldridgeDiDResults:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before accessing results_")
        return self._results

    def get_params(self) -> Dict[str, Any]:
        """Return estimator parameters (sklearn-compatible)."""
        return {
            "method": self.method,
            "control_group": self.control_group,
            "anticipation": self.anticipation,
            "demean_covariates": self.demean_covariates,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            "seed": self.seed,
            "rank_deficient_action": self.rank_deficient_action,
            "vcov_type": self.vcov_type,
            "cohort_trends": self.cohort_trends,
            "conley_coords": self.conley_coords,
            "conley_cutoff_km": self.conley_cutoff_km,
            "conley_metric": self.conley_metric,
            "conley_kernel": self.conley_kernel,
            "conley_lag_cutoff": self.conley_lag_cutoff,
        }

    def set_params(self, **params: Any) -> "WooldridgeDiD":
        """Set estimator parameters (sklearn-compatible). Returns self.

        Atomic: if validation rejects the incoming combination (unknown
        parameter, invalid value, or the ``method`` × ``vcov_type``
        interaction guard fires), ``self`` is unchanged so a caller that
        catches ``ValueError`` / ``NotImplementedError`` can keep using
        the estimator with its previous configuration. Mirrors the
        ``DifferenceInDifferences.set_params`` pattern at
        ``estimators.py:995-1023``.
        """
        # First pass: validate all incoming keys are known attributes so
        # we don't partially apply a batch that ends in "Unknown parameter".
        for key in params:
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key!r}")

        # Compute pending values by overlaying ``params`` on the current
        # configuration; validate on those locals (catches invalid sets +
        # the method × vcov_type interaction) BEFORE mutating ``self``.
        pending = {
            "method": params.get("method", self.method),
            "control_group": params.get("control_group", self.control_group),
            "anticipation": params.get("anticipation", self.anticipation),
            "bootstrap_weights": params.get("bootstrap_weights", self.bootstrap_weights),
            "vcov_type": params.get("vcov_type", self.vcov_type),
            "cohort_trends": params.get("cohort_trends", self.cohort_trends),
        }
        self._validate_constructor_args(**pending)

        # All validation passed — apply mutations atomically.
        for key, value in params.items():
            setattr(self, key, value)
        # Recompute the explicit-vcov flag after any vcov_type mutation.
        self._vcov_type_explicit = self.vcov_type != "hc1"
        return self

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
        exovar: Optional[List[str]] = None,
        xtvar: Optional[List[str]] = None,
        xgvar: Optional[List[str]] = None,
        survey_design=None,
    ) -> WooldridgeDiDResults:
        """Fit the ETWFE model.  See class docstring for parameter details.

        Parameters
        ----------
        data : DataFrame with panel data (long format)
        outcome : outcome column name
        unit : unit identifier column
        time : time period column
        cohort : first treatment period (0 or NaN = never treated)
        exovar : time-invariant covariates added without interaction/demeaning
        xtvar : time-varying covariates (demeaned within cohort×period cells
                when ``demean_covariates=True``)
        xgvar : covariates interacted with each cohort indicator
        survey_design : SurveyDesign, optional
            Survey design specification for complex survey data.  Supports
            stratified, clustered, and weighted designs via Taylor Series
            Linearization (TSL).  Replicate-weight designs raise
            ``NotImplementedError``.
        """
        df = data.copy()
        df = _warn_and_fill_nan_cohort(df, cohort, stacklevel=2)

        # 0a. Validate cohort is time-invariant within unit
        cohort_per_unit = df.groupby(unit)[cohort].nunique()
        bad_units = cohort_per_unit[cohort_per_unit > 1]
        if len(bad_units) > 0:
            example = bad_units.index[0]
            raise ValueError(
                f"Cohort column '{cohort}' is not time-invariant within unit. "
                f"Unit {example!r} has {int(bad_units.iloc[0])} distinct cohort "
                f"values. The cohort column must be constant within each unit."
            )

        # 0b. Reject bootstrap for nonlinear methods (not implemented)
        if self.n_bootstrap > 0 and self.method != "ols":
            raise ValueError(
                f"Bootstrap inference is only supported for method='ols'. "
                f"Got method={self.method!r} with n_bootstrap={self.n_bootstrap}. "
                f"Set n_bootstrap=0 for analytic SEs."
            )

        # 0c. Reject bootstrap + survey (no survey-aware bootstrap variant)
        if self.n_bootstrap > 0 and survey_design is not None:
            raise ValueError(
                "Bootstrap inference is not supported with survey_design. "
                "Set n_bootstrap=0 for analytic survey SEs."
            )

        # 0d.i Reject cohort_trends=True + survey_design. The cohort_trends
        # path auto-routes to the full-dummy design (regardless of
        # vcov_type) to keep the math closure verified against PR #483's
        # R-parity goldens; the survey TSL machinery hasn't yet been
        # validated under the full-dummy + dg_i · t composition. Use
        # cohort_trends=False on survey designs (the default) or wait
        # for the deferred follow-up.
        if survey_design is not None and self.cohort_trends:
            raise NotImplementedError(
                "WooldridgeDiD(cohort_trends=True) with survey_design is "
                "not yet supported: the cohort_trends path auto-routes to "
                "a full-dummy design with `dg_i · t` interactions whose "
                "composition with the survey TSL variance has not been "
                "validated against R-parity goldens. Use "
                "cohort_trends=False (default) for survey designs, or "
                "wait for the deferred follow-up tracked in DEFERRED.md."
            )

        # 0d.ii Reject cohort_trends=True + control_group="never_treated".
        # The OLS + never_treated branch (paper W2025 Section 4.4 / library
        # ``_build_interaction_matrix`` ``include_pre=True``) emits ALL
        # ``(g, t)`` cells for each treated cohort as treatment-cell
        # indicator dummies — including pre-treatment placebo cells. For
        # any treated cohort, ``dg_i · t = Σ_t t · 1{cohort=g, time=t}``
        # is fully spanned by the existing per-cohort sum of cell dummies,
        # so the appended trend columns are unidentified. The per-cohort
        # pre-period guard above counts observed periods but doesn't
        # catch this collinearity. Fail-close pending a redesigned
        # design-matrix path that drops the placebo dummies (or restricts
        # to post-treatment cells) under the trend specification — codex
        # R9 P1 fix.
        if self.cohort_trends and self.control_group == "never_treated":
            raise NotImplementedError(
                "WooldridgeDiD(cohort_trends=True, "
                "control_group='never_treated') is not yet supported: "
                "the OLS never_treated path emits ALL (g, t) cells (paper "
                "W2025 Section 4.4 placebo coverage), and the appended "
                "dg_i · t trend columns are linearly spanned by the "
                "per-cohort sum of those cell dummies, so the Section 8 "
                "trend specification is unidentified on this branch. Use "
                "control_group='not_yet_treated' (the default) for "
                "cohort_trends=True, or wait for the deferred follow-up "
                "tracked in DEFERRED.md."
            )

        # 0d. Reject survey_design + non-hc1 analytical family. The survey-
        # design TSL (or replicate-weight refit) variance overrides the
        # analytical sandwich, so the requested HC2/HC2-BM/classical family
        # would be silently discarded. Mirrors the SunAbraham PR #472 pattern
        # at ``sun_abraham.py:688-705``. Use vcov_type='hc1' (default) for
        # survey designs. ``conley`` is excluded here so it routes to the
        # dedicated conley validator at 0f below, which raises the
        # conley-specific survey-deferral message (consistent with SunAbraham).
        if survey_design is not None and self.vcov_type not in ("hc1", "conley"):
            raise NotImplementedError(
                f"WooldridgeDiD(vcov_type={self.vcov_type!r}) with "
                "survey_design is not yet supported: the survey-design TSL "
                "(or replicate-weight refit) variance overrides the analytical "
                "sandwich, so the requested HC2/HC2-BM/classical family would "
                "be silently discarded. Use vcov_type='hc1' (default) for "
                "survey designs; the survey TSL machinery computes the "
                "design-based variance independently."
            )

        # 0e. Reject bootstrap + one-way analytical vcov_type. The multiplier
        # bootstrap is intrinsically clustered (draws per-cluster weights);
        # one-way ``vcov_type in {"hc2","classical"}`` either drops the unit
        # auto-cluster (``cluster=None`` → bootstrap has no cluster to draw
        # at) OR is rejected by the linalg validator (``cluster=X`` + one-way
        # + cluster_ids). Both fail paths produce a less-informative downstream
        # error, so reject at the estimator boundary across both. The user
        # must drop bootstrap (``n_bootstrap=0``) or pick a cluster-compatible
        # ``vcov_type`` (``hc1`` or ``hc2_bm``).
        if self.n_bootstrap > 0 and self.vcov_type in ("hc2", "classical"):
            raise ValueError(
                f"WooldridgeDiD(vcov_type={self.vcov_type!r}, "
                f"n_bootstrap={self.n_bootstrap}) is not supported: the "
                "multiplier bootstrap is intrinsically clustered, but the "
                "one-way vcov_type does not compose with cluster_ids — "
                "either the unit auto-cluster is dropped (when cluster=None) "
                "leaving the bootstrap with no cluster to draw weights at, "
                "or the linalg validator rejects one-way + cluster_ids at "
                "fit (when cluster=X). Use vcov_type='hc1' / 'hc2_bm' for "
                "the analytical sandwich, or set n_bootstrap=0."
            )

        # 0f. Conley spatial-HAC front-door validation + bootstrap incompatibility.
        # The shared validator gates coords/cutoff/unit/lag/cluster columns and
        # rejects conley + survey_design (deferred). WooldridgeDiD has no
        # `inference=` param, so pass the literal "analytical"; the OLS-only
        # restriction is already enforced by the method != "ols" guard in
        # _validate_constructor_args.
        if self.vcov_type == "conley":
            from diff_diff.conley import _validate_conley_estimator_inputs

            _validate_conley_estimator_inputs(
                estimator_name="WooldridgeDiD",
                data=data,
                unit=unit,
                conley_coords=self.conley_coords,
                conley_cutoff_km=self.conley_cutoff_km,
                conley_lag_cutoff=self.conley_lag_cutoff,
                survey_design=survey_design,
                inference="analytical",
                cluster=self.cluster,
            )
            if self.n_bootstrap > 0:
                raise ValueError(
                    "WooldridgeDiD(vcov_type='conley') is incompatible with "
                    "n_bootstrap > 0: the multiplier bootstrap overrides the "
                    "analytical Conley sandwich. Use n_bootstrap=0 for the "
                    "analytical Conley SE, or vcov_type='hc1' with the bootstrap."
                )

        # 0g. Outcome-fit hint (non-fatal): method="ols" on a binary or
        # non-negative-count outcome. For such outcomes a matching nonlinear
        # model (logit / Poisson) is often the MORE APPROPRIATE specification —
        # it imposes parallel trends on the link/index scale, not in levels
        # (Wooldridge 2023: level-PT is implausible for limited outcomes), and
        # where the nonlinear mean holds its sims show the linear model both
        # biased and less precise. A DIFFERENT identifying assumption than
        # linear OLS (which fits depends on which PT holds), so the hint is a
        # recommended comparison, not a free upgrade (see REGISTRY.md
        # §WooldridgeDiD "Nonlinear extensions"). Detection reads the FULL
        # outcome column: outcome *type* is a property of the variable, so the
        # named column is the right signal. This is also identical to the
        # estimation sample here — _filter_sample expresses the control-group
        # choice through the design matrix, NOT by dropping rows, so the column
        # and the fitted sample share the same outcome support (invariant pinned
        # by test_filter_sample_preserves_outcome_support); classifying on the
        # variable stays correct should that ever change.
        if self.method == "ols":
            suggested = _suggest_nonlinear_method(df[outcome])
            if suggested is not None:
                kind = "binary" if suggested == "logit" else "count"
                warnings.warn(
                    f"WooldridgeDiD(method='ols'): outcome {outcome!r} looks "
                    f"{kind}. For a {kind} outcome a matching nonlinear model "
                    f"(method={suggested!r}) is often the more appropriate "
                    f"specification — it imposes parallel trends on the "
                    f"link/index scale, not in levels (Wooldridge 2023: "
                    f"level-PT is implausible for limited outcomes), and where "
                    f"that nonlinear mean holds its simulations show the linear "
                    f"model both biased and less precise. It is a different "
                    f"identifying assumption than linear OLS (which fits depends "
                    f"on which parallel-trends restriction holds), so treat "
                    f"method={suggested!r} as a recommended comparison, not an "
                    f"automatic switch; suppress this hint via "
                    f"warnings.filterwarnings.",
                    UserWarning,
                    stacklevel=2,
                )

        # 1. Filter to analysis sample
        sample = _filter_sample(df, unit, time, cohort, self.control_group, self.anticipation)

        # 1b. Identification checks
        groups = sorted(g for g in sample[cohort].unique() if g > 0)
        if len(groups) == 0:
            raise ValueError(
                "No treated cohorts found in data. Ensure the cohort column "
                "contains values > 0 for treated units."
            )
        if self.control_group == "never_treated" and not (sample[cohort] == 0).any():
            raise ValueError(
                "control_group='never_treated' but no never-treated units "
                "(cohort == 0) found. Use 'not_yet_treated' or add "
                "never-treated units."
            )
        if self.control_group == "not_yet_treated":
            # Verify at least some untreated comparison observations exist
            has_untreated = (sample[cohort] == 0).any() or (
                (sample[cohort] - self.anticipation) > sample[time]
            ).any()
            if not has_untreated:
                raise ValueError(
                    "control_group='not_yet_treated' but no untreated comparison "
                    "observations exist. All units are treated at all observed "
                    "time periods. Use 'never_treated' with a never-treated group."
                )

        # 1c. Identification check for cohort_trends=True (paper W2025 Section 8 /
        # Eq. 8.1). Each treated cohort needs at least 2 distinct pre-treatment
        # periods (``t < g - anticipation``) for the cohort-specific linear trend
        # ``dg_i · t`` to be separately identified from cohort + time FE. With
        # only 1 pre-period the linear trend is observationally equivalent to
        # cohort FE on that single point.
        #
        # Counts pre-treatment periods OBSERVED FOR THIS COHORT (per-cohort
        # sample subset) rather than the global panel time set — on
        # unbalanced panels a cohort can have only one observed pre-period
        # even when the global panel has many, and the linear trend is
        # still underidentified (per codex R2 P1 fix).
        if self.cohort_trends:
            for g in groups:
                cohort_pre_times = sample.loc[
                    (sample[cohort] == g) & (sample[time] < g - self.anticipation),
                    time,
                ].unique()
                n_pre_periods = len(cohort_pre_times)
                if n_pre_periods < 2:
                    raise ValueError(
                        f"cohort_trends=True requires at least 2 pre-treatment "
                        f"periods OBSERVED FOR EACH TREATED COHORT (paper W2025 "
                        f"Section 8 / Eq. 8.1 identification). Cohort g={g} has "
                        f"only {n_pre_periods} pre-treatment period(s) observed "
                        f"in the analysis sample (t < g - anticipation = "
                        f"{g - self.anticipation}); the cohort-specific linear "
                        f"trend dg_i · t is not separately identified from "
                        f"cohort + time fixed effects on a single point. Drop "
                        f"cohort_trends=True or use a panel where each treated "
                        f"cohort has at least 2 observed pre-periods."
                    )

        # 2. Build interaction matrix
        X_int, int_col_names, gt_keys = _build_interaction_matrix(
            sample,
            cohort=cohort,
            time=time,
            anticipation=self.anticipation,
            control_group=self.control_group,
            method=self.method,
        )
        if X_int.shape[1] == 0:
            raise ValueError(
                "No valid treatment cells found. Check that treated units "
                "have post-treatment observations in the data."
            )

        # 3. Covariates
        X_cov = _prepare_covariates(
            sample,
            exovar=exovar,
            xtvar=xtvar,
            xgvar=xgvar,
            cohort=cohort,
            time=time,
            demean_covariates=self.demean_covariates,
            groups=groups,
        )

        all_regressors = int_col_names.copy()
        if X_cov is not None:
            # Build treatment × demeaned-covariate interactions (W2025 Eq. 5.3)
            # For each (g,t) cell indicator and each covariate, create the
            # moderating interaction: X_int[:, i] * x_hat[:, j]
            # This allows treatment effects to vary with covariates within cells.
            cov_names_list = list(exovar or []) + list(xtvar or []) + list(xgvar or [])
            # Compute cohort-demeaned covariates for interaction terms
            X_cov_demeaned = X_cov.copy()
            if self.demean_covariates:
                cohort_vals = sample[cohort].values
                for j in range(X_cov.shape[1]):
                    for g in groups:
                        mask = cohort_vals == g
                        if mask.any():
                            X_cov_demeaned[mask, j] -= X_cov[mask, j].mean()

            interact_cols = []
            interact_names = []
            for i, gt_name in enumerate(int_col_names):
                for j in range(X_cov_demeaned.shape[1]):
                    interact_cols.append(X_int[:, i] * X_cov_demeaned[:, j])
                    cov_label = cov_names_list[j] if j < len(cov_names_list) else f"cov{j}"
                    interact_names.append(f"{gt_name}_x_{cov_label}")

            # Cohort × covariate interactions (W2025 Eq. 5.3: D_g × X)
            # exovar/xtvar get automatic D_g × X; xgvar already has D_g × X
            cov_cols_for_dg = list(exovar or []) + list(xtvar or [])
            cohort_cov_cols = []
            cohort_cov_names = []
            if cov_cols_for_dg:
                cohort_vals_arr = sample[cohort].values
                for g in groups:
                    g_ind = (cohort_vals_arr == g).astype(float)
                    for col in cov_cols_for_dg:
                        cohort_cov_cols.append(g_ind * sample[col].values.astype(float))
                        cohort_cov_names.append(f"D{g}_x_{col}")

            # Time × covariate interactions (W2025 Eq. 5.3: f_t × X)
            # All covariates get f_t × X, drop first time for identification
            all_cov_cols = list(exovar or []) + list(xtvar or []) + list(xgvar or [])
            times_sorted = sorted(sample[time].unique())
            time_cov_cols = []
            time_cov_names = []
            time_vals_arr = sample[time].values
            for t in times_sorted[1:]:  # drop first
                t_ind = (time_vals_arr == t).astype(float)
                for col in all_cov_cols:
                    time_cov_cols.append(t_ind * sample[col].values.astype(float))
                    time_cov_names.append(f"ft{t}_x_{col}")

            # Assemble: [cell_indicators, cell×cov, D_g×X, f_t×X, raw_cov]
            blocks = [X_int]
            if interact_cols:
                blocks.append(np.column_stack(interact_cols))
                all_regressors.extend(interact_names)
            if cohort_cov_cols:
                blocks.append(np.column_stack(cohort_cov_cols))
                all_regressors.extend(cohort_cov_names)
            if time_cov_cols:
                blocks.append(np.column_stack(time_cov_cols))
                all_regressors.extend(time_cov_names)
            blocks.append(X_cov)
            for i in range(X_cov.shape[1]):
                all_regressors.append(f"_cov_{i}")
            X_design = np.hstack(blocks)
        else:
            X_design = X_int

        if self.method == "ols":
            results = self._fit_ols(
                sample,
                outcome,
                unit,
                time,
                cohort,
                X_design,
                all_regressors,
                gt_keys,
                int_col_names,
                groups,
                survey_design=survey_design,
            )
        elif self.method == "logit":
            n_cov_interact = X_cov.shape[1] if X_cov is not None else 0
            results = self._fit_logit(
                sample,
                outcome,
                unit,
                time,
                cohort,
                X_design,
                all_regressors,
                gt_keys,
                int_col_names,
                groups,
                n_cov_interact=n_cov_interact,
                survey_design=survey_design,
            )
        else:  # poisson
            n_cov_interact = X_cov.shape[1] if X_cov is not None else 0
            results = self._fit_poisson(
                sample,
                outcome,
                unit,
                time,
                cohort,
                X_design,
                all_regressors,
                gt_keys,
                int_col_names,
                groups,
                n_cov_interact=n_cov_interact,
                survey_design=survey_design,
            )

        self._results = results
        self.is_fitted_ = True
        return results

    def _count_control_units(self, sample: pd.DataFrame, unit: str, cohort: str, time: str) -> int:
        """Count control units consistent with control_group setting."""
        n_never = int(sample[sample[cohort] == 0][unit].nunique())
        if self.control_group == "not_yet_treated":
            # Also count future-treated units that contribute pre-anticipation obs
            nyt = sample[
                (sample[cohort] > 0) & (sample[time] < sample[cohort] - self.anticipation)
            ][unit].nunique()
            return n_never + int(nyt)
        return n_never

    def _fit_ols(
        self,
        sample: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
        X_design: np.ndarray,
        col_names: List[str],
        gt_keys: List[Tuple],
        int_col_names: List[str],
        groups: List[Any],
        survey_design=None,
    ) -> WooldridgeDiDResults:
        """OLS path: within-transform FE, solve_ols, cluster SE.

        Branches on ``self.vcov_type``: ``hc1`` (default) and ``conley``
        (spatial-HAC) use the within-transform path bit-equally;
        ``hc2``/``hc2_bm``/``classical`` — and any vcov_type when
        ``cohort_trends=True`` — auto-route to a full-dummy saturated design
        because FWL preserves cohort coefficients but NOT the hat matrix (HC2
        leverage and Bell-McCaffrey Satterthwaite DOF require the full FE
        projection; conley works on either design, with row-aligned coord/time/
        unit arrays threaded into ``solve_ols``). Mirrors the SunAbraham PR #472
        pattern at ``sun_abraham.py:1364``.
        """
        # Reset index so numpy positional indexing matches pandas groupby
        sample = sample.reset_index(drop=True)
        # Cluster IDs: default to unit level for hc1/hc2_bm; drop the auto-
        # cluster when the user opts into one-way ``vcov_type in {"hc2",
        # "classical"}`` explicitly (one-way families don't compose with
        # cluster_ids per the linalg validator). Explicit ``self.cluster=X``
        # always wins. Mirrors SunAbraham PR #472 at ``sun_abraham.py:792-797``.
        if self.cluster is not None:
            cluster_col: Optional[str] = self.cluster
            cluster_ids: Optional[np.ndarray] = sample[cluster_col].values
        elif self.vcov_type == "conley":
            # Conley: never auto-cluster at unit (a unit-cluster product kernel
            # would zero every between-unit spatial pair). Explicit cluster=
            # enables the combined spatial+cluster product kernel (above).
            cluster_col = None
            cluster_ids = None
        elif self.vcov_type in ("hc2", "classical") and self._vcov_type_explicit:
            cluster_col = None
            cluster_ids = None
        else:
            cluster_col = unit
            cluster_ids = sample[cluster_col].values
        # Bootstrap cluster level: user's ``self.cluster`` if set, else unit
        # (the panel's natural unit of variation). This preserves the prior
        # behavior — bootstrap matches the analytical cluster on hc1/hc2_bm
        # paths and falls back to unit when the analytical sandwich drops the
        # auto-cluster under explicit one-way. The fit() guard rejects
        # ``n_bootstrap > 0`` + one-way + ``cluster=None``, so the fallback to
        # unit only kicks in when the user explicitly set ``cluster=X``
        # (already handled above) — in that explicit-one-way + explicit-cluster
        # case the bootstrap matches the analytical cluster too.
        bootstrap_cluster_col = self.cluster if self.cluster else unit
        cluster_ids_bootstrap = sample[bootstrap_cluster_col].values

        # Resolve survey design, inject cluster as PSU only when user explicitly set cluster=
        survey_cluster_ids = cluster_ids if self.cluster else None
        resolved, survey_weights, survey_weight_type, survey_metadata, df_inf = (
            _resolve_survey_for_wooldridge(survey_design, sample, survey_cluster_ids, self.cluster)
        )

        # Branch design build on vcov_type. ``hc1`` keeps within-transform
        # (FWL preserves the CR1 cluster-robust score → bit-equal to prior
        # at atol=1e-14). The full-dummy branch handles hc2 / hc2_bm /
        # classical, which need the hat matrix on the full FE projection
        # (FWL does not preserve it). ``coef_offset`` shifts gt_effects
        # indexing to account for the intercept under full-dummy.
        #
        # ``cohort_trends=True`` (paper W2025 Section 8 / Eq. 8.1) forces
        # the full-dummy path regardless of ``vcov_type``: composing
        # ``dg_i · t`` interactions with the within-transformation yields
        # ``(dg_i − mean(dg_i)) · (t − mean(t))`` which is correct but
        # non-trivial to verify across all panel shapes; the full-dummy
        # auto-route (matching the absorb→fixed_effects pattern at
        # ``feedback_absorb_to_fixed_effects_auto_route``) keeps the math
        # closure verified on the same path already locked by PR #483's
        # HC2 / HC2-BM / classical R-parity goldens.
        use_full_dummy = self.vcov_type in ("hc2", "hc2_bm", "classical") or self.cohort_trends

        if use_full_dummy:
            # Full-dummy build: [intercept, X_design, unit_dummies,
            # time_dummies, cohort_trend_cols (if cohort_trends=True)].
            # Survey + non-hc1 was rejected at fit(), so survey_weights /
            # resolved are None here. ``coef_offset = 1`` shifts the
            # gt_effects loop to skip the intercept.
            n_obs = len(sample)
            n_units_fe = int(sample[unit].nunique())
            n_times_fe = int(sample[time].nunique())
            n_trend_cols = len(groups) if self.cohort_trends else 0
            dense_cells = n_obs * (
                1 + X_design.shape[1] + (n_units_fe - 1) + (n_times_fe - 1) + n_trend_cols
            )
            if dense_cells > 50_000_000:
                warnings.warn(
                    f"WooldridgeDiD(vcov_type={self.vcov_type!r}, "
                    f"cohort_trends={self.cohort_trends!r}) builds a "
                    f"dense full-dummy saturated design (~{dense_cells:,} "
                    "float64 cells, >50M). FWL preserves coefficients but not "
                    "the hat matrix, so HC2/HC2-BM/classical requires the full-"
                    "dummy projection (within-transform would produce a "
                    "methodologically different statistic). For very high-"
                    "cardinality panels, consider vcov_type='hc1' (within-"
                    "transform) + cohort_trends=False or reducing the panel "
                    "size.",
                    UserWarning,
                    stacklevel=3,
                )
            intercept_col = np.ones((n_obs, 1))
            unit_dummies = pd.get_dummies(
                sample[unit], prefix=f"_fe_{unit}", drop_first=True
            ).values.astype(float)
            time_dummies = pd.get_dummies(
                sample[time], prefix=f"_fe_{time}", drop_first=True
            ).values.astype(float)
            design_parts: List[np.ndarray] = [intercept_col, X_design, unit_dummies, time_dummies]
            cohort_trend_col_names: List[str] = []
            if self.cohort_trends:
                # Paper W2025 Eq. 8.1: ``dg_i · t`` for each treated cohort.
                # Never-treated cohort uses no trend (acts as control trend).
                cohort_vals = sample[cohort].values
                time_vals = sample[time].values.astype(float)
                # All-eventually-treated normalization: when there is no
                # never-treated baseline cohort (g=0 not in sample), the
                # full set of G trend columns ``dg_i · t`` is collinear
                # with the time fixed effects (paper W2025 Section 5.4:
                # "all variables in regression (5.3) involving dT_i get
                # dropped" when the last cohort serves as control). Drop
                # the LAST cohort's trend column deterministically — that
                # cohort then acts as the trend baseline, mirroring the
                # paper's all-treated normalization rule and matching the
                # ``_build_interaction_matrix`` last-cohort handling for
                # cohort × time interactions.
                has_never_treated = (sample[cohort] == 0).any()
                trend_groups = groups if has_never_treated else groups[:-1]
                trend_cols: List[np.ndarray] = []
                for g in trend_groups:
                    trend_col = (cohort_vals == g).astype(float) * time_vals
                    trend_cols.append(trend_col.reshape(-1, 1))
                    cohort_trend_col_names.append(f"trend_g{g}")
                if trend_cols:
                    design_parts.append(np.hstack(trend_cols))
            X = np.hstack(design_parts)
            y = sample[outcome].values.astype(float)
            coef_offset = 1
        else:
            # Within-transform path (hc1 default; preserves prior bit-equal
            # behavior). 4. Within-transform: absorb unit + time FE
            all_vars = [outcome] + [f"_x{i}" for i in range(X_design.shape[1])]
            tmp = sample[[unit, time]].copy()
            tmp[outcome] = sample[outcome].values
            for i in range(X_design.shape[1]):
                tmp[f"_x{i}"] = X_design[:, i]

            # Use iterative alternating projections for demeaning (exact for
            # both balanced and unbalanced panels).  Survey weights change
            # the weighted FWL projection — all columns (treatment
            # interactions + covariates) are demeaned together.
            wt_weights = survey_weights if survey_weights is not None else np.ones(len(tmp))

            # Guard: zero-weight unit/time groups cause 0/0 in within_transform
            if survey_weights is not None and np.any(survey_weights == 0):
                sw_series = pd.Series(survey_weights, index=sample.index)
                for grp_col, grp_label in [(unit, "unit"), (time, "time period")]:
                    grp_sums = sw_series.groupby(sample[grp_col]).sum()
                    zero_grps = grp_sums[grp_sums == 0].index.tolist()
                    if zero_grps:
                        raise ValueError(
                            f"Survey weights sum to zero for {grp_label}(s) "
                            f"{zero_grps[:3]}. Cannot compute weighted "
                            f"within-transformation. Remove zero-weight "
                            f"{grp_label}s or use non-zero weights."
                        )

            _w_regressors = [f"_x{i}" for i in range(X_design.shape[1])]
            _pre_norms = pre_demean_norms(tmp, _w_regressors, weights=wt_weights)
            transformed = within_transform(
                tmp,
                all_vars,
                unit=unit,
                time=time,
                suffix="_demeaned",
                weights=wt_weights,
            )
            # Snap FE-spanned design columns (e.g. a unit-constant covariate)
            # to exact zero so rank handling drops them deterministically.
            snap_absorbed_regressors(
                transformed,
                _w_regressors,
                _pre_norms,
                absorbed_desc=f"unit '{unit}' and time '{time}' fixed effects",
                group_vars=[unit, time],
                rank_deficient_action=self.rank_deficient_action,
                suffix="_demeaned",
                display_names=dict(zip(_w_regressors, col_names)),
                weights=wt_weights,
            )

            y = transformed[f"{outcome}_demeaned"].values
            X_cols = [f"_x{i}_demeaned" for i in range(X_design.shape[1])]
            X = transformed[X_cols].values
            coef_offset = 0

        # 6. Solve OLS (skip cluster-robust vcov when survey will provide TSL vcov).
        # Pass ``column_names=col_names`` only on the within-transform branch;
        # under full-dummy ``X`` has additional intercept + FE columns whose
        # names aren't in ``col_names``, and ``solve_ols`` only uses names for
        # rank-deficiency error messages (cosmetic). Omitting under full-dummy
        # keeps rank-deficiency reporting consistent with the column count.
        # Conley spatial-HAC arrays, row-aligned to X. Both the within-transform
        # and full-dummy designs are built from `sample` without dropping rows
        # (conley rejects survey/weights, so there is no weight-based drop), so
        # coordinates/time/unit read from the post-reset `sample` align to X.
        if self.vcov_type == "conley":
            assert self.conley_coords is not None  # guaranteed by _validate_conley_estimator_inputs
            _cl_coords = np.column_stack(
                [
                    sample[self.conley_coords[0]].values.astype(np.float64),
                    sample[self.conley_coords[1]].values.astype(np.float64),
                ]
            )
            _cl_time = np.asarray(sample[time].values)
            _cl_unit = sample[unit].values
        else:
            _cl_coords = _cl_time = _cl_unit = None

        coefs, resids, vcov = solve_ols(  # type: ignore[call-overload]  # mypy union-explosion limitation at this many Optional args
            X,
            y,
            cluster_ids=cluster_ids,
            return_vcov=(resolved is None),
            rank_deficient_action=self.rank_deficient_action,
            column_names=col_names if not use_full_dummy else None,
            weights=survey_weights,
            weight_type=survey_weight_type,
            vcov_type=self.vcov_type,
            conley_coords=_cl_coords,
            conley_cutoff_km=self.conley_cutoff_km,
            conley_metric=self.conley_metric,
            conley_kernel=self.conley_kernel,
            conley_time=_cl_time,
            conley_unit=_cl_unit,
            conley_lag_cutoff=self.conley_lag_cutoff,
        )

        # Extract cohort-trend coefficients (paper W2025 Eq. 8.1 ``δ_g``).
        # Trend columns live at the tail of the full-dummy design after
        # the unit + time dummies. Empty dict when cohort_trends=False
        # (matches the no-op contract under default).
        #
        # ``trend_groups`` mirrors the design-build branch above: when
        # there is no never-treated baseline cohort the last cohort's
        # trend column is dropped per paper W2025 Section 5.4's
        # all-treated normalization rule. ``cohort_trend_coefs`` only
        # surfaces the identified cohorts; the dropped cohort's slope
        # is the baseline (zero in deviation form) and is intentionally
        # absent from the dict.
        cohort_trend_coefs: Dict[Any, float] = {}
        if self.cohort_trends and use_full_dummy:
            n_units_for_trend = int(sample[unit].nunique())
            n_times_for_trend = int(sample[time].nunique())
            trend_start_idx = (
                1 + X_design.shape[1] + (n_units_for_trend - 1) + (n_times_for_trend - 1)
            )
            has_never_treated_trend = (sample[cohort] == 0).any()
            trend_groups_extract = groups if has_never_treated_trend else groups[:-1]
            for i, g in enumerate(trend_groups_extract):
                idx = trend_start_idx + i
                if idx < len(coefs):
                    cohort_trend_coefs[g] = float(coefs[idx])
                else:
                    cohort_trend_coefs[g] = float("nan")

        # Survey TSL vcov replaces cluster-robust vcov
        if resolved is not None:
            from diff_diff.survey import compute_survey_vcov

            nan_mask_ols = np.isnan(coefs)
            if np.any(nan_mask_ols):
                kept = ~nan_mask_ols
                vcov_kept = compute_survey_vcov(X[:, kept], resids, resolved)
                vcov = np.full((len(coefs), len(coefs)), np.nan)
                kept_idx = np.where(kept)[0]
                vcov[np.ix_(kept_idx, kept_idx)] = vcov_kept
            else:
                vcov = compute_survey_vcov(X, resids, resolved)

        # 7. Extract β_{g,t} and build gt_effects dict. Under full-dummy
        # (``coef_offset = 1``), treatment cells occupy columns
        # ``1..1+n_int-1`` (intercept at 0); under within-transform
        # (``coef_offset = 0``), treatment cells occupy columns
        # ``0..n_int-1``. The shift only affects this loop's indexing into
        # ``coefs`` / ``vcov`` — the (g, t) key space and the order of
        # ``gt_keys`` are identical across branches.
        #
        # First pass: collect non-dropped (g, t) cells + att + se. Per-cell
        # df_Satt is computed in a single batched call below (BM DOF section)
        # so per-cell inference fields use the Satterthwaite DOF rather than
        # df_inf=None (normal-theory).
        gt_effects: Dict[Tuple, Dict] = {}
        gt_weights: Dict[Tuple, int] = {}
        gt_coef_index_map: Dict[Tuple, int] = {}  # (g, t) -> full-coef-space index
        for idx, (g, t) in enumerate(gt_keys):
            coef_idx = idx + coef_offset
            if coef_idx >= len(coefs):
                break
            # Skip cells whose coefficient was dropped (rank deficiency)
            if np.isnan(coefs[coef_idx]):
                continue
            att = float(coefs[coef_idx])
            se = (
                float(np.sqrt(max(vcov[coef_idx, coef_idx], 0.0)))
                if vcov is not None
                else float("nan")
            )
            gt_effects[(g, t)] = {
                "att": att,
                "se": se,
                "t_stat": float("nan"),
                "p_value": float("nan"),
                "conf_int": (float("nan"), float("nan")),
            }
            gt_weights[(g, t)] = int(((sample[cohort] == g) & (sample[time] == t)).sum())
            gt_coef_index_map[(g, t)] = coef_idx

        # Extract vcov submatrix for identified β_{g,t} only (skip NaN/dropped).
        # Shift by coef_offset so the submatrix lands on treatment cells
        # under full-dummy.
        gt_keys_ordered = list(gt_effects.keys())
        if vcov is not None and gt_keys_ordered:
            orig_indices = [i + coef_offset for i, k in enumerate(gt_keys) if k in gt_effects]
            gt_vcov = vcov[np.ix_(orig_indices, orig_indices)]
        else:
            gt_vcov = None

        # 8. Bell-McCaffrey contrast DOF threading for hc2_bm. Computes (in
        # one batched call) the per-coefficient ``df_Satt`` for every
        # present ``(g, t)`` cell AND the post-period-average overall ATT
        # contrast DOF. Per-cell DOFs are applied to ``gt_effects`` inference
        # below (8a); the overall DOF is applied to the simple aggregation
        # (8b). BM artifacts (X, cluster_ids, bread, coef_index_map) are
        # also stashed on the Results object so that downstream
        # ``aggregate("group" | "calendar" | "event")`` can compute contrast-
        # specific DOFs lazily without recomputing the full-dummy fit.
        # Mirrors the SunAbraham PR #472 pattern at
        # ``sun_abraham.py:1008-1097`` and the StackedDiD PR #479 R3 fix.
        # Per ``feedback_bm_contrast_dof_fail_closed``: when DOF is
        # unavailable (helper raises or returns non-finite), affected
        # user-facing inference is NaN rather than falling back to
        # ``safe_inference(df=None)`` (silent normal-theory).
        overall_att_bm_dof: Optional[float] = None
        per_cell_bm_dof: Dict[Tuple, float] = {}
        bm_artifacts: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple, int]]] = None
        # Residual DOF for one-way ``vcov_type in {"classical","hc2"}`` paths
        # (full-dummy, no survey). Matches R's ``lm()`` / ``coef_test()`` use
        # of ``n - rank(X)`` for the t-distribution under both classical OLS
        # SE and ``sandwich::vcovHC(type="HC2")``. ``None`` on hc1 /
        # hc2_bm / surveyed paths (those use their own DOF threading or
        # df_inf). Mirrors R's t-distribution convention so per-cell +
        # aggregate p-values/CIs are not normal-theory under small samples.
        df_one_way: Optional[float] = None
        if (
            self.vcov_type in ("classical", "hc2")
            and use_full_dummy
            and resolved is None
            and vcov is not None
        ):
            n_kept = int((~np.isnan(coefs)).sum())
            df_candidate = X.shape[0] - n_kept
            df_one_way = float(df_candidate) if df_candidate > 0 else float("nan")
        if (
            self.vcov_type == "hc2_bm"
            and use_full_dummy
            and resolved is None
            and vcov is not None
            and gt_keys_ordered
            and cluster_ids is not None
        ):
            from diff_diff.linalg import _compute_cr2_bm_contrast_dof

            # Honor rank deficiency: solve_ols sets coefs[i] = NaN for dropped
            # columns. The full-design bread (X.T @ X) is singular on the
            # dropped columns, so _compute_cr2_bm_contrast_dof would
            # LinAlgError on it. Reduce X / bread / contrasts to the kept-
            # column subspace before computing BM DOF (matches the existing
            # MPD pattern at estimators.py:1860-1913 and SA's full-dummy
            # behavior). Identified (g, t) cells survive; cells whose
            # treatment-cell coefficient was dropped get per_cell_bm_dof=NaN
            # and the gt_effects loop fail-closes their inference.
            nan_mask = np.isnan(coefs)
            kept_indices = np.where(~nan_mask)[0]
            kept_set = set(int(i) for i in kept_indices.tolist())
            X_red = X[:, kept_indices]
            bread_red = X_red.T @ X_red
            # Map full-coef index → reduced-coef index for kept columns only.
            full_to_reduced: Dict[int, int] = {
                int(full_idx): red_pos for red_pos, full_idx in enumerate(kept_indices)
            }
            # Reduced coef-index map for (g, t) cells whose coefficient was
            # kept; cells with dropped coefficients are absent here and will
            # be fail-closed at gt_effects inference + aggregate() time.
            reduced_coef_idx_map: Dict[Tuple, int] = {
                k: full_to_reduced[v] for k, v in gt_coef_index_map.items() if int(v) in kept_set
            }
            n_red = X_red.shape[1]
            # Per-cell one-hot contrasts (kept cells only). Dropped cells get
            # NaN per_cell_bm_dof (caller fail-closes inference fields).
            per_cell_keys_kept = [k for k in gt_keys_ordered if k in reduced_coef_idx_map]
            per_cell_keys_dropped = [k for k in gt_keys_ordered if k not in reduced_coef_idx_map]
            # Overall ATT contrast across post-period kept cells.
            post_keys = [(g, t) for (g, t) in gt_keys_ordered if t >= g]
            post_keys_kept = [k for k in post_keys if k in reduced_coef_idx_map]
            w_total_post = sum(gt_weights.get(k, 0) for k in post_keys_kept)
            overall_contrast = np.zeros(n_red)
            if w_total_post > 0:
                for k in post_keys_kept:
                    overall_contrast[reduced_coef_idx_map[k]] = gt_weights[k] / w_total_post
            include_overall = w_total_post > 0 and bool(np.any(overall_contrast != 0))
            cols: List[np.ndarray] = []
            for k in per_cell_keys_kept:
                col = np.zeros(n_red)
                col[reduced_coef_idx_map[k]] = 1.0
                cols.append(col)
            if include_overall:
                cols.append(overall_contrast)
            if cols:
                contrasts_matrix = np.column_stack(cols)
                try:
                    dof_vec = _compute_cr2_bm_contrast_dof(
                        X_red, cluster_ids, bread_red, contrasts_matrix
                    )
                    for i, k in enumerate(per_cell_keys_kept):
                        candidate = float(dof_vec[i])
                        per_cell_bm_dof[k] = candidate if np.isfinite(candidate) else float("nan")
                    if include_overall:
                        candidate = float(dof_vec[-1])
                        overall_att_bm_dof = candidate if np.isfinite(candidate) else float("nan")
                except (ValueError, np.linalg.LinAlgError) as exc:
                    warnings.warn(
                        f"WooldridgeDiD(vcov_type='hc2_bm') analytical "
                        f"inference could not compute Bell-McCaffrey "
                        f"contrast DOF ({type(exc).__name__}: {exc}). "
                        "Affected per-cell and overall inference (t_stat / "
                        "p_value / conf_int) will be NaN to preserve the "
                        "hc2_bm contract.",
                        UserWarning,
                        stacklevel=3,
                    )
                    for k in per_cell_keys_kept:
                        per_cell_bm_dof[k] = float("nan")
                    if include_overall:
                        overall_att_bm_dof = float("nan")
            # Cells whose coefficient was dropped get NaN regardless of
            # whether the batched call succeeded.
            for k in per_cell_keys_dropped:
                per_cell_bm_dof[k] = float("nan")
            # Stash REDUCED artifacts for ``aggregate()`` so the lazy
            # contrast DOF computation operates in the same reduced
            # coefficient space and avoids the singular full-design bread.
            bm_artifacts = (X_red, cluster_ids, bread_red, reduced_coef_idx_map)

        # 8a. Apply DOF threading (or fail-closed NaN) to ``gt_effects``:
        # hc2_bm uses per-cell BM Satterthwaite DOF; classical/hc2 (one-way,
        # no survey) use the residual ``df_one_way = n - rank(X)`` so
        # p-values/CIs match R ``lm()`` / ``coef_test()`` t-distribution
        # instead of normal-theory; hc1 / surveyed paths use ``df_inf``
        # (survey df or None). Per ``feedback_bm_contrast_dof_fail_closed``:
        # NaN BM DOF emits NaN inference fields (never normal-theory).
        for (g, t), eff in gt_effects.items():
            if self.vcov_type == "hc2_bm" and use_full_dummy and resolved is None:
                cell_dof = per_cell_bm_dof.get((g, t), float("nan"))
                if np.isfinite(cell_dof):
                    t_stat, p_value, conf_int = safe_inference(
                        eff["att"], eff["se"], alpha=self.alpha, df=cell_dof
                    )
                else:
                    t_stat = float("nan")
                    p_value = float("nan")
                    conf_int = (float("nan"), float("nan"))
            elif (
                self.vcov_type in ("classical", "hc2")
                and use_full_dummy
                and resolved is None
                and df_one_way is not None
                and np.isfinite(df_one_way)
            ):
                t_stat, p_value, conf_int = safe_inference(
                    eff["att"], eff["se"], alpha=self.alpha, df=df_one_way
                )
            else:
                t_stat, p_value, conf_int = safe_inference(
                    eff["att"], eff["se"], alpha=self.alpha, df=df_inf
                )
            eff["t_stat"] = t_stat
            eff["p_value"] = p_value
            eff["conf_int"] = conf_int

        # 8b. Simple aggregation (always computed). DOF threading mirrors 8a:
        # hc2_bm uses the overall ATT BM contrast DOF (fail-closed NaN if
        # unavailable); classical/hc2 (one-way, no survey) use the residual
        # ``df_one_way``; hc1 / surveyed paths use ``df_inf`` (survey df or
        # None).
        if self.vcov_type == "hc2_bm" and use_full_dummy and resolved is None:
            if overall_att_bm_dof is not None and np.isfinite(overall_att_bm_dof):
                overall = _compute_weighted_agg(
                    gt_effects,
                    gt_weights,
                    gt_keys_ordered,
                    gt_vcov,
                    self.alpha,
                    df=overall_att_bm_dof,
                )
            else:
                # BM DOF unavailable: preserve att + se from a finite-df run
                # (so the user-facing att/se still match the sandwich), then
                # NaN-out the inference fields.
                overall = _compute_weighted_agg(
                    gt_effects,
                    gt_weights,
                    gt_keys_ordered,
                    gt_vcov,
                    self.alpha,
                    df=df_inf,
                )
                overall = {
                    "att": overall["att"],
                    "se": overall["se"],
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "conf_int": (float("nan"), float("nan")),
                }
        elif (
            self.vcov_type in ("classical", "hc2")
            and use_full_dummy
            and resolved is None
            and df_one_way is not None
            and np.isfinite(df_one_way)
        ):
            overall = _compute_weighted_agg(
                gt_effects, gt_weights, gt_keys_ordered, gt_vcov, self.alpha, df=df_one_way
            )
        else:
            overall = _compute_weighted_agg(
                gt_effects, gt_weights, gt_keys_ordered, gt_vcov, self.alpha, df=df_inf
            )

        # Metadata
        n_treated = int(sample[sample[cohort] > 0][unit].nunique())
        n_control = self._count_control_units(sample, unit, cohort, time)
        all_times = sorted(sample[time].unique().tolist())
        # Per-cohort unit counts ``N_g`` (paper Eqs. 7.4, 7.6) — needed by
        # ``aggregate(weights="cohort_share")``.
        n_g_per_cohort = {g: int(sample[sample[cohort] == g][unit].nunique()) for g in groups}

        results = WooldridgeDiDResults(
            group_time_effects=gt_effects,
            overall_att=overall["att"],
            overall_se=overall["se"],
            overall_t_stat=overall["t_stat"],
            overall_p_value=overall["p_value"],
            overall_conf_int=overall["conf_int"],
            method=self.method,
            control_group=self.control_group,
            groups=groups,
            time_periods=all_times,
            n_obs=len(sample),
            n_treated_units=n_treated,
            n_control_units=n_control,
            alpha=self.alpha,
            anticipation=self.anticipation,
            survey_metadata=survey_metadata,
            vcov_type=self.vcov_type,
            # Cluster metadata: ``None`` under survey TSL because the
            # analytical sandwich (and its cluster_ids) was overridden by
            # the survey variance; ``survey_metadata`` carries the
            # design-side stratification/PSU instead. Under non-survey, the
            # analytical cluster (default unit, dropped for explicit one-way).
            cluster_name=(None if resolved is not None else cluster_col),
            n_clusters=(
                None
                if resolved is not None
                else (int(np.unique(cluster_ids).size) if cluster_ids is not None else None)
            ),
            conley_lag_cutoff=(self.conley_lag_cutoff if self.vcov_type == "conley" else None),
            _gt_weights=gt_weights,
            _n_g_per_cohort=n_g_per_cohort,
            _gt_vcov=gt_vcov,
            _gt_keys=gt_keys_ordered,
            _df_survey=df_inf,
            _bm_per_cell_dof=per_cell_bm_dof,
            _bm_artifacts=bm_artifacts,
            _df_one_way=df_one_way,
            cohort_trend_coefs=cohort_trend_coefs,
            cohort_trends=self.cohort_trends,
        )

        # 9. Optional multiplier bootstrap (overrides analytic SE for overall ATT).
        # Clusters at ``self.cluster if self.cluster else unit`` (via the
        # ``cluster_ids_bootstrap`` variable set just below the cluster-
        # handling block at the top of _fit_ols) — i.e., the bootstrap
        # honors the user's explicit ``cluster=X`` and falls back to unit
        # only when ``cluster=None`` (the panel's natural unit of variation).
        # The fit() guard at the top rejects ``n_bootstrap > 0`` +
        # ``vcov_type in {"hc2","classical"}`` regardless of cluster, so
        # under any combination that reaches here, the bootstrap cluster
        # matches the analytical cluster on hc1/hc2_bm paths.
        if self.n_bootstrap > 0:
            rng = np.random.default_rng(self.seed)
            unique_boot_clusters = np.unique(cluster_ids_bootstrap)
            n_boot_clusters = len(unique_boot_clusters)
            post_keys = [(g, t) for (g, t) in gt_keys_ordered if t >= g]
            w_total_b = sum(gt_weights.get(k, 0) for k in post_keys)
            boot_atts: List[float] = []
            for _ in range(self.n_bootstrap):
                if self.bootstrap_weights == "rademacher":
                    cl_weights = rng.choice([-1.0, 1.0], size=n_boot_clusters)
                elif self.bootstrap_weights == "webb":
                    cl_weights = rng.choice(
                        [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)],
                        size=n_boot_clusters,
                    )
                else:  # mammen
                    phi = (1 + np.sqrt(5)) / 2
                    cl_weights = rng.choice(
                        [-(phi - 1), phi],
                        p=[phi / np.sqrt(5), (phi - 1) / np.sqrt(5)],
                        size=n_boot_clusters,
                    )
                obs_weights = cl_weights[
                    np.searchsorted(unique_boot_clusters, cluster_ids_bootstrap)
                ]
                y_boot = y + obs_weights * resids
                # Thread vcov_type for grep consistency (no-op at runtime
                # because ``return_vcov=False``). Pass the analytical
                # ``cluster_ids`` (which may be ``None`` under one-way
                # explicit + ``cluster=None`` — the fit() guard prevents
                # that combination from reaching here).
                coefs_b, _, _ = solve_ols(
                    X,
                    y_boot,
                    cluster_ids=cluster_ids,
                    return_vcov=False,
                    rank_deficient_action="silent",
                    vcov_type=self.vcov_type,
                )
                if w_total_b > 0:
                    att_b = (
                        sum(
                            gt_weights.get(k, 0) * float(coefs_b[i + coef_offset])
                            for i, k in enumerate(gt_keys)
                            if k in post_keys and i + coef_offset < len(coefs_b)
                        )
                        / w_total_b
                    )
                    boot_atts.append(att_b)
            if boot_atts:
                boot_se = float(np.std(boot_atts, ddof=1))
                t_stat_b, p_b, ci_b = safe_inference(results.overall_att, boot_se, alpha=self.alpha)
                results.overall_se = boot_se
                results.overall_t_stat = t_stat_b
                results.overall_p_value = p_b
                results.overall_conf_int = ci_b
                results._bootstrap_used = True

        return results

    def _fit_logit(
        self,
        sample: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
        X_int: np.ndarray,
        col_names: List[str],
        gt_keys: List[Tuple],
        int_col_names: List[str],
        groups: List[Any],
        n_cov_interact: int = 0,
        survey_design=None,
    ) -> WooldridgeDiDResults:
        """Logit path: cohort + time additive FEs + solve_logit + ASF ATT.

        Matches Stata jwdid method(logit): logit y [treatment_interactions]
        i.gvar i.tvar — cohort main effects + time main effects (additive),
        not cohort×time saturated group FEs.
        """
        n_int = len(int_col_names)

        # Design matrix: treatment interactions + cohort FEs + time FEs
        # This matches Stata's `i.gvar i.tvar` specification.
        cohort_dummies = pd.get_dummies(sample[cohort], drop_first=True).values.astype(float)
        time_dummies = pd.get_dummies(sample[time], drop_first=True).values.astype(float)
        X_full = np.hstack([X_int, cohort_dummies, time_dummies])

        y = sample[outcome].values.astype(float)
        if not np.all(np.isfinite(y)):
            raise ValueError("Outcome contains non-finite values (NaN/Inf).")
        if np.any(y < 0) or np.any(y > 1):
            raise ValueError(
                f"method='logit' requires outcomes in [0, 1]. "
                f"Got range [{y.min():.4f}, {y.max():.4f}]."
            )
        cluster_col = self.cluster if self.cluster else unit
        cluster_ids = sample[cluster_col].values

        # Resolve survey design, inject cluster as PSU only when user explicitly set cluster=
        survey_cluster_ids = cluster_ids if self.cluster else None
        resolved, survey_weights, survey_weight_type, survey_metadata, df_inf = (
            _resolve_survey_for_wooldridge(survey_design, sample, survey_cluster_ids, self.cluster)
        )
        _has_survey = resolved is not None

        beta, probs = solve_logit(
            X_full,
            y,
            rank_deficient_action=self.rank_deficient_action,
            weights=survey_weights,
        )
        # solve_logit prepends intercept — beta[0] is intercept, beta[1:] are X_full cols
        beta_int_cols = beta[1 : n_int + 1]  # treatment interaction coefficients

        # Handle rank-deficient designs: identify kept columns, compute vcov
        # on reduced design, then expand back
        nan_mask = np.isnan(beta)
        beta_clean = np.where(nan_mask, 0.0, beta)
        kept_beta = ~nan_mask

        # QMLE sandwich vcov
        resids = y - probs
        X_with_intercept = np.column_stack([np.ones(len(y)), X_full])

        if _has_survey:
            # X_tilde trick: transform design matrix so compute_survey_vcov
            # produces the correct QMLE sandwich for nonlinear models.
            # Bread: (X_tilde'WX_tilde)^{-1} = (X'diag(w*V)X)^{-1}
            # Scores: w*X_tilde*r_tilde = w*X*(y-mu)
            from diff_diff.survey import compute_survey_vcov

            V = probs * (1 - probs)
            sqrt_V = np.sqrt(np.clip(V, 1e-20, None))
            X_tilde = X_with_intercept * sqrt_V[:, None]
            r_tilde = resids / sqrt_V
            if np.any(nan_mask):
                X_tilde_r = X_tilde[:, kept_beta]
                vcov_reduced = compute_survey_vcov(X_tilde_r, r_tilde, resolved)
                k_full = len(beta)
                vcov_full = np.full((k_full, k_full), np.nan)
                kept_idx = np.where(kept_beta)[0]
                vcov_full[np.ix_(kept_idx, kept_idx)] = vcov_reduced
            else:
                vcov_full = compute_survey_vcov(X_tilde, r_tilde, resolved)
        else:
            # Cluster-robust QMLE sandwich (non-survey path)
            if np.any(nan_mask):
                X_reduced = X_with_intercept[:, kept_beta]
                vcov_reduced = compute_robust_vcov(
                    X_reduced,
                    resids,
                    cluster_ids=cluster_ids,
                    weights=probs * (1 - probs),
                    weight_type="aweight",
                )
                k_full = len(beta)
                vcov_full = np.full((k_full, k_full), np.nan)
                kept_idx = np.where(kept_beta)[0]
                vcov_full[np.ix_(kept_idx, kept_idx)] = vcov_reduced
            else:
                vcov_full = compute_robust_vcov(
                    X_with_intercept,
                    resids,
                    cluster_ids=cluster_ids,
                    weights=probs * (1 - probs),
                    weight_type="aweight",
                )
        beta = beta_clean

        # Survey-weighted averaging helpers for ASF computation
        def _avg(a, cell_mask):
            if survey_weights is not None:
                return float(np.average(a, weights=survey_weights[cell_mask]))
            return float(np.mean(a))

        def _avg_ax0(a, cell_mask):
            if survey_weights is not None:
                return np.average(a, weights=survey_weights[cell_mask], axis=0)
            return np.mean(a, axis=0)

        # ASF ATT(g,t) for treated units in each cell
        gt_effects: Dict[Tuple, Dict] = {}
        gt_weights: Dict[Tuple, int] = {}
        gt_grads: Dict[Tuple, np.ndarray] = {}  # store per-cell gradients for aggregate SE
        for idx, (g, t) in enumerate(gt_keys):
            if idx >= n_int:
                break
            cell_mask = (sample[cohort] == g) & (sample[time] == t)
            if cell_mask.sum() == 0:
                continue
            # Skip cells whose interaction coefficient was dropped (rank deficiency)
            # Skip cells where all survey weights are zero (non-estimable)
            if survey_weights is not None and np.sum(survey_weights[cell_mask]) == 0:
                continue
            delta = beta_int_cols[idx]
            if np.isnan(delta):
                continue
            eta_base = X_with_intercept[cell_mask] @ beta
            # Counterfactual: zero the FULL treatment block for cell (g,t).
            # This includes the scalar cell effect δ_{g,t} AND any cell ×
            # covariate interaction effects ξ_{g,t,j} * x_hat_j (W2023 Eq. 3.15).
            delta_total = np.full(cell_mask.sum(), float(delta))
            for j in range(n_cov_interact):
                coef_pos = 1 + n_int + idx * n_cov_interact + j
                if coef_pos < len(beta):
                    x_hat_j = X_with_intercept[cell_mask, coef_pos]
                    delta_total = delta_total + beta[coef_pos] * x_hat_j
            eta_0 = eta_base - delta_total
            att = _avg(_logistic(eta_base) - _logistic(eta_0), cell_mask)
            # Delta method gradient: d(ATT)/d(β)
            #   for nuisance p: mean_i[(Λ'(η_1) - Λ'(η_0)) * X_p]
            #   for cell intercept: mean_i[Λ'(η_1)]
            #   for cell × cov j: mean_i[Λ'(η_1) * x_hat_j]
            d_diff = _logistic_deriv(eta_base) - _logistic_deriv(eta_0)
            grad = _avg_ax0(X_with_intercept[cell_mask] * d_diff[:, None], cell_mask)
            grad[1 + idx] = _avg(_logistic_deriv(eta_base), cell_mask)
            for j in range(n_cov_interact):
                coef_pos = 1 + n_int + idx * n_cov_interact + j
                if coef_pos < len(beta):
                    x_hat_j = X_with_intercept[cell_mask, coef_pos]
                    grad[coef_pos] = _avg(_logistic_deriv(eta_base) * x_hat_j, cell_mask)
            # Compute SE in reduced parameter space if rank-deficient
            if np.any(nan_mask):
                grad_r = grad[kept_beta]
                se = float(np.sqrt(max(grad_r @ vcov_reduced @ grad_r, 0.0)))
            else:
                se = float(np.sqrt(max(grad @ vcov_full @ grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=df_inf)
            gt_effects[(g, t)] = {
                "att": att,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
            gt_weights[(g, t)] = int(cell_mask.sum())
            # Store gradient in reduced space for aggregate SE
            gt_grads[(g, t)] = grad[kept_beta] if np.any(nan_mask) else grad

        gt_keys_ordered = [k for k in gt_keys if k in gt_effects]
        # Use reduced vcov for all downstream SE computations
        _vcov_se = vcov_reduced if np.any(nan_mask) else vcov_full
        # ATT-level covariance: J @ vcov @ J' where J rows are per-cell gradients
        if gt_keys_ordered:
            J = np.array([gt_grads[k] for k in gt_keys_ordered])
            gt_vcov = J @ _vcov_se @ J.T
        else:
            gt_vcov = None

        # Overall SE via joint delta method: ∇β(overall_att) = Σ w_k/w_total * grad_k
        post_keys = [(g, t) for (g, t) in gt_keys_ordered if t >= g]
        w_total = sum(gt_weights.get(k, 0) for k in post_keys)
        if w_total > 0 and post_keys:
            overall_att = sum(gt_weights[k] * gt_effects[k]["att"] for k in post_keys) / w_total
            agg_grad = sum((gt_weights[k] / w_total) * gt_grads[k] for k in post_keys)
            overall_se = float(np.sqrt(max(agg_grad @ _vcov_se @ agg_grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(
                overall_att, overall_se, alpha=self.alpha, df=df_inf
            )
            overall = {
                "att": overall_att,
                "se": overall_se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
        else:
            overall = _compute_weighted_agg(
                gt_effects, gt_weights, gt_keys_ordered, None, self.alpha, df=df_inf
            )

        return WooldridgeDiDResults(
            group_time_effects=gt_effects,
            overall_att=overall["att"],
            overall_se=overall["se"],
            overall_t_stat=overall["t_stat"],
            overall_p_value=overall["p_value"],
            overall_conf_int=overall["conf_int"],
            method=self.method,
            control_group=self.control_group,
            groups=groups,
            time_periods=sorted(sample[time].unique().tolist()),
            n_obs=len(sample),
            n_treated_units=int(sample[sample[cohort] > 0][unit].nunique()),
            n_control_units=self._count_control_units(sample, unit, cohort, time),
            alpha=self.alpha,
            anticipation=self.anticipation,
            survey_metadata=survey_metadata,
            # vcov_type is locked to "hc1" on the nonlinear paths (the
            # __init__ guard rejects non-hc1 + method != "ols"). Surface
            # cluster_name / n_clusters for shared introspection contract
            # with the OLS path. Under survey TSL the analytical sandwich
            # (including its cluster_ids) is replaced — surface ``None``
            # so downstream introspection doesn't claim a unit-cluster
            # that wasn't actually applied; survey_metadata carries the
            # design-side structure instead.
            vcov_type=self.vcov_type,
            cluster_name=(None if _has_survey else cluster_col),
            n_clusters=(None if _has_survey else int(np.unique(cluster_ids).size)),
            _gt_weights=gt_weights,
            _n_g_per_cohort={g: int(sample[sample[cohort] == g][unit].nunique()) for g in groups},
            _gt_vcov=gt_vcov,
            _gt_keys=gt_keys_ordered,
            _df_survey=df_inf,
        )

    def _fit_poisson(
        self,
        sample: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
        X_int: np.ndarray,
        col_names: List[str],
        gt_keys: List[Tuple],
        int_col_names: List[str],
        groups: List[Any],
        n_cov_interact: int = 0,
        survey_design=None,
    ) -> WooldridgeDiDResults:
        """Poisson path: cohort + time additive FEs + solve_poisson + ASF ATT.

        Matches Stata jwdid method(poisson): poisson y [treatment_interactions]
        i.gvar i.tvar — cohort main effects + time main effects (additive),
        not cohort×time saturated group FEs.
        """
        n_int = len(int_col_names)

        # Design matrix: intercept + treatment interactions + cohort FEs + time FEs.
        # Matches Stata's `i.gvar i.tvar` + treatment interaction specification.
        # solve_poisson does not prepend an intercept, so we include one explicitly.
        intercept = np.ones((len(sample), 1))
        cohort_dummies = pd.get_dummies(sample[cohort], drop_first=True).values.astype(float)
        time_dummies = pd.get_dummies(sample[time], drop_first=True).values.astype(float)
        X_full = np.hstack([intercept, X_int, cohort_dummies, time_dummies])
        # Treatment interaction coefficients start at column index 1.

        y = sample[outcome].values.astype(float)
        if not np.all(np.isfinite(y)):
            raise ValueError("Outcome contains non-finite values (NaN/Inf).")
        if np.any(y < 0):
            raise ValueError(
                f"method='poisson' requires non-negative outcomes. "
                f"Got minimum value {y.min():.4f}."
            )
        cluster_col = self.cluster if self.cluster else unit
        cluster_ids = sample[cluster_col].values

        # Resolve survey design, inject cluster as PSU only when user explicitly set cluster=
        survey_cluster_ids = cluster_ids if self.cluster else None
        resolved, survey_weights, survey_weight_type, survey_metadata, df_inf = (
            _resolve_survey_for_wooldridge(survey_design, sample, survey_cluster_ids, self.cluster)
        )
        _has_survey = resolved is not None

        beta, mu_hat = solve_poisson(
            X_full,
            y,
            rank_deficient_action=self.rank_deficient_action,
            weights=survey_weights,
        )

        # Handle rank-deficient designs: compute vcov on reduced design.
        # Preserve raw interaction coefficients BEFORE zeroing NaN so the
        # NaN check in the ASF loop correctly skips dropped cells.
        nan_mask = np.isnan(beta)
        beta_int_raw = beta[1 : 1 + n_int].copy()  # before zeroing
        beta_clean = np.where(nan_mask, 0.0, beta)
        kept_beta = ~nan_mask

        # QMLE sandwich vcov
        resids = y - mu_hat

        if _has_survey:
            # X_tilde trick for nonlinear survey vcov (V = mu for Poisson)
            from diff_diff.survey import compute_survey_vcov

            sqrt_V = np.sqrt(np.clip(mu_hat, 1e-20, None))
            X_tilde = X_full * sqrt_V[:, None]
            r_tilde = resids / sqrt_V
            if np.any(nan_mask):
                X_tilde_r = X_tilde[:, kept_beta]
                vcov_reduced = compute_survey_vcov(X_tilde_r, r_tilde, resolved)
                k_full = len(beta)
                vcov_full = np.full((k_full, k_full), np.nan)
                kept_idx = np.where(kept_beta)[0]
                vcov_full[np.ix_(kept_idx, kept_idx)] = vcov_reduced
            else:
                vcov_full = compute_survey_vcov(X_tilde, r_tilde, resolved)
        else:
            # Cluster-robust QMLE sandwich (non-survey path)
            if np.any(nan_mask):
                X_reduced = X_full[:, kept_beta]
                vcov_reduced = compute_robust_vcov(
                    X_reduced,
                    resids,
                    cluster_ids=cluster_ids,
                    weights=mu_hat,
                    weight_type="aweight",
                )
                k_full = len(beta)
                vcov_full = np.full((k_full, k_full), np.nan)
                kept_idx = np.where(kept_beta)[0]
                vcov_full[np.ix_(kept_idx, kept_idx)] = vcov_reduced
            else:
                vcov_full = compute_robust_vcov(
                    X_full,
                    resids,
                    cluster_ids=cluster_ids,
                    weights=mu_hat,
                    weight_type="aweight",
                )
        beta = beta_clean

        # Treatment interaction coefficients (from cleaned beta for computation)
        beta_int = beta[1 : 1 + n_int]

        # Survey-weighted averaging helpers for ASF computation
        def _avg(a, cell_mask):
            if survey_weights is not None:
                return float(np.average(a, weights=survey_weights[cell_mask]))
            return float(np.mean(a))

        def _avg_ax0(a, cell_mask):
            if survey_weights is not None:
                return np.average(a, weights=survey_weights[cell_mask], axis=0)
            return np.mean(a, axis=0)

        # ASF ATT(g,t) for treated units in each cell.
        # eta_base = X_full @ beta already includes the treatment effect (D_{g,t}=1).
        # Counterfactual: eta_0 = eta_base - delta  (treatment switched off).
        # ATT = E[exp(η_1)] - E[exp(η_0)] = E[exp(η_base)] - E[exp(η_base - δ)]
        gt_effects: Dict[Tuple, Dict] = {}
        gt_weights: Dict[Tuple, int] = {}
        gt_grads: Dict[Tuple, np.ndarray] = {}  # per-cell gradients for aggregate SE
        for idx, (g, t) in enumerate(gt_keys):
            if idx >= n_int:
                break
            cell_mask = (sample[cohort] == g) & (sample[time] == t)
            if cell_mask.sum() == 0:
                continue
            # Skip cells whose interaction coefficient was dropped (rank deficiency).
            # Use raw coefficients (before NaN->0 zeroing) to detect dropped cells.
            if np.isnan(beta_int_raw[idx]):
                continue
            # Skip cells where all survey weights are zero (non-estimable)
            if survey_weights is not None and np.sum(survey_weights[cell_mask]) == 0:
                continue
            delta = beta_int[idx]
            if np.isnan(delta):
                continue
            eta_base = np.clip(X_full[cell_mask] @ beta, -500, 500)
            # Counterfactual: zero the FULL treatment block (W2023 Eq. 3.15)
            delta_total = np.full(cell_mask.sum(), float(delta))
            for j in range(n_cov_interact):
                coef_pos = 1 + n_int + idx * n_cov_interact + j
                if coef_pos < len(beta):
                    x_hat_j = X_full[cell_mask, coef_pos]
                    delta_total = delta_total + beta[coef_pos] * x_hat_j
            eta_0 = eta_base - delta_total
            mu_1 = np.exp(eta_base)
            mu_0 = np.exp(eta_0)
            att = _avg(mu_1 - mu_0, cell_mask)
            # Delta method gradient:
            #   for nuisance p: mean_i[(μ_1 - μ_0) * X_p]
            #   for cell intercept: mean_i[μ_1]
            #   for cell × cov j: mean_i[μ_1 * x_hat_j]
            diff_mu = mu_1 - mu_0
            grad = _avg_ax0(X_full[cell_mask] * diff_mu[:, None], cell_mask)
            grad[1 + idx] = _avg(mu_1, cell_mask)
            for j in range(n_cov_interact):
                coef_pos = 1 + n_int + idx * n_cov_interact + j
                if coef_pos < len(beta):
                    x_hat_j = X_full[cell_mask, coef_pos]
                    grad[coef_pos] = _avg(mu_1 * x_hat_j, cell_mask)
            # Compute SE in reduced parameter space if rank-deficient
            if np.any(nan_mask):
                grad_r = grad[kept_beta]
                se = float(np.sqrt(max(grad_r @ vcov_reduced @ grad_r, 0.0)))
            else:
                se = float(np.sqrt(max(grad @ vcov_full @ grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=df_inf)
            gt_effects[(g, t)] = {
                "att": att,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
            gt_weights[(g, t)] = int(cell_mask.sum())
            gt_grads[(g, t)] = grad[kept_beta] if np.any(nan_mask) else grad

        gt_keys_ordered = [k for k in gt_keys if k in gt_effects]
        _vcov_se = vcov_reduced if np.any(nan_mask) else vcov_full
        # ATT-level covariance: J @ vcov @ J' where J rows are per-cell gradients
        if gt_keys_ordered:
            J = np.array([gt_grads[k] for k in gt_keys_ordered])
            gt_vcov = J @ _vcov_se @ J.T
        else:
            gt_vcov = None

        # Overall SE via joint delta method
        post_keys = [(g, t) for (g, t) in gt_keys_ordered if t >= g]
        w_total = sum(gt_weights.get(k, 0) for k in post_keys)
        if w_total > 0 and post_keys:
            overall_att = sum(gt_weights[k] * gt_effects[k]["att"] for k in post_keys) / w_total
            agg_grad = sum((gt_weights[k] / w_total) * gt_grads[k] for k in post_keys)
            overall_se = float(np.sqrt(max(agg_grad @ _vcov_se @ agg_grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(
                overall_att, overall_se, alpha=self.alpha, df=df_inf
            )
            overall = {
                "att": overall_att,
                "se": overall_se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
        else:
            overall = _compute_weighted_agg(
                gt_effects, gt_weights, gt_keys_ordered, None, self.alpha, df=df_inf
            )

        return WooldridgeDiDResults(
            group_time_effects=gt_effects,
            overall_att=overall["att"],
            overall_se=overall["se"],
            overall_t_stat=overall["t_stat"],
            overall_p_value=overall["p_value"],
            overall_conf_int=overall["conf_int"],
            method=self.method,
            control_group=self.control_group,
            groups=groups,
            time_periods=sorted(sample[time].unique().tolist()),
            n_obs=len(sample),
            n_treated_units=int(sample[sample[cohort] > 0][unit].nunique()),
            n_control_units=self._count_control_units(sample, unit, cohort, time),
            alpha=self.alpha,
            anticipation=self.anticipation,
            survey_metadata=survey_metadata,
            # vcov_type locked to "hc1" on Poisson path (the __init__ guard
            # rejects non-hc1 + method != "ols"). Surface cluster_name /
            # n_clusters for shared introspection contract. Under survey
            # TSL the analytical sandwich is replaced — surface ``None``
            # so downstream introspection doesn't claim a unit-cluster
            # that wasn't actually applied.
            vcov_type=self.vcov_type,
            cluster_name=(None if _has_survey else cluster_col),
            n_clusters=(None if _has_survey else int(np.unique(cluster_ids).size)),
            _gt_weights=gt_weights,
            _n_g_per_cohort={g: int(sample[sample[cohort] == g][unit].nunique()) for g in groups},
            _gt_vcov=gt_vcov,
            _gt_keys=gt_keys_ordered,
            _df_survey=df_inf,
        )
