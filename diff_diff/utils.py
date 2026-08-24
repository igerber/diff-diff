"""
Utility functions for difference-in-differences estimation.
"""

import inspect
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Import Rust backend if available (from _backend to avoid circular imports)
from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_compute_noise_level,
    _rust_compute_time_weights,
    _rust_demean_map,
    _rust_sc_weight_fw,
    _rust_sc_weight_fw_weighted,
    _rust_sc_weight_fw_weighted_with_convergence,
    _rust_sc_weight_fw_with_convergence,
    _rust_sdid_unit_weights,
)
from diff_diff.linalg import _validate_cluster_k_adjustment_type
from diff_diff.linalg import compute_robust_vcov as _compute_robust_vcov_linalg
from diff_diff.linalg import solve_ols as _solve_ols_linalg

# Numerical constants for optimization algorithms
_OPTIMIZATION_MAX_ITER = 1000  # Maximum iterations for weight optimization
_OPTIMIZATION_TOL = 1e-8  # Convergence tolerance for optimization
_NUMERICAL_EPS = 1e-10  # Small constant to prevent division by zero

# Cache for critical values to avoid repeated scipy calls
_critical_value_cache: Dict[Tuple[float, Optional[float]], float] = {}


def _get_critical_value(alpha: float, df: Optional[float] = None) -> float:
    """Return cached critical value for (alpha, df) pair."""
    key = (alpha, df)
    if key not in _critical_value_cache:
        if df is not None:
            _critical_value_cache[key] = float(stats.t.ppf(1 - alpha / 2, df))
        else:
            _critical_value_cache[key] = float(stats.norm.ppf(1 - alpha / 2))
    return _critical_value_cache[key]


def validate_binary(arr: np.ndarray, name: str) -> None:
    """
    Validate that an array contains only binary values (0 or 1).

    Parameters
    ----------
    arr : np.ndarray
        Array to validate.
    name : str
        Name of the variable (for error messages).

    Raises
    ------
    ValueError
        If array contains non-binary values.
    """
    unique_values = np.unique(arr[~np.isnan(arr)])
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(f"{name} must be binary (0 or 1). " f"Found values: {unique_values}")


def validate_covariate_names(
    covariates: Optional[List[str]],
    reserved_names: Iterable[str],
    *,
    estimator: str = "estimator",
) -> None:
    """
    Validate that covariate column names do not collide with reserved
    structural term names (and are not duplicated within ``covariates``).

    Fitted coefficients are stored in a ``name -> value`` dict built by zipping
    a variable-name list -- structural term names PLUS the user covariate column
    names appended verbatim -- with the coefficient vector. A covariate whose
    name equals a reserved structural name (the intercept ``const``, the
    treatment/time indicators, the interaction term, period dummies,
    fixed-effect dummies, or an internal working column) would silently
    overwrite the structural coefficient (Python dict last-write-wins),
    corrupting the result with no error. Duplicate names within ``covariates``
    collapse to a single dict entry the same way.

    The comparison is case-sensitive: column names and dict keys are
    case-sensitive, so e.g. ``Const`` does not actually collide with ``const``
    and is allowed.

    Parameters
    ----------
    covariates : list of str or None
        User-supplied covariate column names. ``None`` or empty is a no-op.
    reserved_names : iterable of str
        Reserved structural term names this estimator builds (estimator-specific).
    estimator : str
        Estimator name, used in the error message.

    Raises
    ------
    ValueError
        If a covariate name collides with a reserved structural name, or if
        ``covariates`` contains duplicate names.
    """
    if not covariates:
        return
    reserved = set(reserved_names)
    collisions = sorted({c for c in covariates if c in reserved})
    if collisions:
        raise ValueError(
            f"{estimator}: covariate name(s) {collisions} collide with reserved "
            f"structural term name(s). These names are used internally for the "
            f"intercept, the treatment/time indicators, the interaction term, "
            f"period dummies, fixed-effect dummies, or internal working columns, "
            f"and a colliding covariate would silently overwrite the structural "
            f"coefficient. Rename the covariate column(s). Reserved names for "
            f"this fit: {sorted(reserved)}."
        )
    seen: set = set()
    duplicates = []
    for c in covariates:
        if c in seen:
            duplicates.append(c)
        seen.add(c)
    if duplicates:
        raise ValueError(
            f"{estimator}: duplicate covariate name(s) {sorted(set(duplicates))} "
            f"in `covariates`. Each covariate maps to one coefficient; duplicates "
            f"collapse to a single entry. Remove the duplicate(s)."
        )


def validate_design_term_names(
    var_names: Iterable[str],
    *,
    estimator: str = "estimator",
) -> None:
    """
    Raise if the assembled design term-name list contains duplicates.

    Backstop for :func:`validate_covariate_names`: even after the user
    covariates are cleared, a fixed-effect dummy name (``{fe}_{value}``) can
    still collide with a structural term — most notably a ``MultiPeriodDiD``
    ``period_{p}`` event-study key when a non-time fixed effect produces matching
    dummy names — or with another dummy. Such a duplicate would silently
    overwrite a coefficient when ``var_names`` is zipped into the result's
    ``coefficients`` dict (Python dict last-write-wins). This checks the FINAL
    name list (structural terms + covariates + fixed-effect dummies) right
    before the dict is built, catching collisions that depend on the data and so
    cannot be known up front.

    Parameters
    ----------
    var_names : iterable of str
        The fully assembled design-matrix column-name list.
    estimator : str
        Estimator name, used in the error message.

    Raises
    ------
    ValueError
        If any name appears more than once.
    """
    seen: set = set()
    duplicates = []
    for name in var_names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            f"{estimator}: the fitted design has duplicate term name(s) "
            f"{sorted(set(duplicates))} — a covariate or fixed-effect dummy name "
            f"collides with a structural term (intercept, treatment/time "
            f"indicators, the interaction, or period dummies) or with another "
            f"column. This would silently overwrite a coefficient in the result. "
            f"Rename the offending fixed-effect category or covariate column."
        )


def fe_dummy_names(col: pd.Series, prefix: str) -> List[str]:
    """
    Reserved fixed-effect dummy column names for the collision guard, matching
    ``pd.get_dummies(col, prefix=prefix, drop_first=True).columns`` WITHOUT
    materializing the dense ``(n x G)`` dummy matrix.

    The within-transform ``TwoWayFixedEffects`` path is specifically designed to
    avoid expanding high-cardinality fixed-effect dummies (that is its scaling
    contract), so the collision guard must reserve those names without building
    the dummy block. ``pd.get_dummies`` orders categories via
    ``pd.Categorical(col).categories`` — sorted unique values for a plain column,
    the declared category order for a ``Categorical`` — then ``drop_first=True``
    drops the first. This derivation reproduces that exactly (including
    ``Categorical`` columns with a non-default category order) at ``O(G)`` memory.

    Parameters
    ----------
    col : pandas.Series
        The fixed-effect / unit / time column.
    prefix : str
        Dummy-name prefix (the project uses ``fe`` for ``fixed_effects`` and
        ``_fe_{unit}`` / ``_fe_{time}`` for TWFE unit/time dummies).

    Returns
    -------
    list of str
        The kept (post ``drop_first``) dummy column names.
    """
    if isinstance(col.dtype, pd.CategoricalDtype):
        cats = list(col.cat.categories)
    else:
        cats = list(pd.Categorical(col).categories)
    return [f"{prefix}_{c}" for c in cats[1:]]


def build_fe_dummy_blocks(
    data: pd.DataFrame,
    fe_cols: List[str],
    prefixes: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[str]]:
    """Materialize drop-first fixed-effect dummy blocks and their column names.

    Single shared implementation of the ``pd.get_dummies(col, prefix=...,
    drop_first=True)`` design-build used by ``DifferenceInDifferences`` /
    ``MultiPeriodDiD`` (``fixed_effects=``) and the ``TwoWayFixedEffects``
    HC2/HC2-BM full-dummy path — previously three inline copies whose FE
    naming / dtype / column-order conventions could drift independently.
    Names match :func:`fe_dummy_names` (the reserved-name collision guard)
    exactly; values are the dense ``float64`` dummy matrix per FE, in
    ``get_dummies`` column order.

    Parameters
    ----------
    data : pandas.DataFrame
        Frame holding the FE columns.
    fe_cols : list of str
        Fixed-effect column names, in design order.
    prefixes : list of str, optional
        Dummy-name prefix per FE column (defaults to the column name itself;
        TWFE passes ``_fe_{col}`` to keep its internal-name convention).

    Returns
    -------
    blocks : list of ndarray
        One dense ``(n, G_j - 1)`` float64 dummy block per FE column.
    names : list of str
        The kept dummy column names across all FEs, in block order.
    """
    if prefixes is not None and len(prefixes) != len(fe_cols):
        raise ValueError(
            f"build_fe_dummy_blocks: prefixes length {len(prefixes)} does not "
            f"match fe_cols length {len(fe_cols)}; zip would silently skip "
            "trailing FE columns."
        )
    blocks: List[np.ndarray] = []
    names: List[str] = []
    for fe, prefix in zip(fe_cols, prefixes or fe_cols):
        dummies = pd.get_dummies(data[fe], prefix=prefix, drop_first=True)
        blocks.append(dummies.values.astype(np.float64))
        names.extend(dummies.columns)
    return blocks, names


def warn_if_not_converged(
    converged: bool,
    method_name: str,
    max_iter: int,
    tol: Optional[float] = None,
    stacklevel: int = 3,
) -> None:
    """Emit a UserWarning when an iterative solver exhausts max_iter without converging.

    Shared helper for axis-B silent-failure fixes (iterative loops that otherwise
    return the current iterate without signaling non-convergence).
    """
    if converged:
        return
    tol_suffix = f" (tol={tol})" if tol is not None else ""
    warnings.warn(
        f"{method_name} did not converge in {max_iter} iterations{tol_suffix}. "
        "Results may be inaccurate.",
        UserWarning,
        stacklevel=stacklevel,
    )


def compute_robust_se(
    X: np.ndarray, residuals: np.ndarray, cluster_ids: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute heteroskedasticity-robust (HC1) or cluster-robust standard errors.

    This function is a thin wrapper around the optimized implementation in
    diff_diff.linalg for backwards compatibility.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    residuals : np.ndarray
        Residuals from regression of shape (n,).
    cluster_ids : np.ndarray, optional
        Cluster identifiers for cluster-robust SEs.

    Returns
    -------
    np.ndarray
        Variance-covariance matrix of shape (k, k).
    """
    return _compute_robust_vcov_linalg(X, residuals, cluster_ids)


def compute_confidence_interval(
    estimate: float, se: float, alpha: float = 0.05, df: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute confidence interval for an estimate.

    Parameters
    ----------
    estimate : float
        Point estimate.
    se : float
        Standard error.
    alpha : float
        Significance level (default 0.05 for 95% CI).
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of confidence interval.
    """
    critical_value = _get_critical_value(alpha, df)
    lower = estimate - critical_value * se
    upper = estimate + critical_value * se

    return (lower, upper)


def compute_p_value(t_stat: float, df: Optional[int] = None, two_sided: bool = True) -> float:
    """
    Compute p-value for a t-statistic.

    Parameters
    ----------
    t_stat : float
        T-statistic.
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.
    two_sided : bool
        Whether to compute two-sided p-value (default True).

    Returns
    -------
    float
        P-value.
    """
    if df is not None:
        p_value = stats.t.sf(np.abs(t_stat), df)
    else:
        p_value = stats.norm.sf(np.abs(t_stat))

    if two_sided:
        p_value *= 2

    return float(p_value)


def safe_inference(effect, se, alpha=0.05, df=None):
    """Compute t_stat, p_value, conf_int with NaN-safe gating.

    When SE is non-finite, zero, or negative, ALL inference fields
    are set to NaN to prevent misleading statistical output.

    Accepts scalar inputs only (not numpy arrays). All existing inference
    call sites operate on scalars within loops.

    Parameters
    ----------
    effect : float
        Point estimate (treatment effect or coefficient).
    se : float
        Standard error of the estimate.
    alpha : float, optional
        Significance level for confidence interval (default 0.05).
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.

    Returns
    -------
    tuple
        (t_stat, p_value, (ci_lower, ci_upper)). All NaN when SE is
        non-finite, zero, or negative.
    """
    if not (np.isfinite(se) and se > 0):
        return np.nan, np.nan, (np.nan, np.nan)
    if df is not None and not (np.isfinite(df) and df > 0):
        # Undefined degrees of freedom: df <= 0 (e.g., rank-deficient replicate
        # design) OR non-finite (a guard-suppressed / non-physical Bell-McCaffrey
        # Satterthwaite DOF, which `_cr2_bm_dof_inner` returns as NaN for
        # high-leverage nuisance contrasts). All inference fields are NaN so a
        # coefficient whose DOF was declared unreliable never reports finite
        # t/p/CI - preserving the joint-NaN inference contract.
        return np.nan, np.nan, (np.nan, np.nan)
    t_stat = effect / se
    p_value = compute_p_value(t_stat, df=df)
    conf_int = compute_confidence_interval(effect, se, alpha, df=df)
    return t_stat, p_value, conf_int


def safe_inference_batch(effects, ses, alpha=0.05, df=None):
    """Vectorized batch inference for arrays of effects and SEs.

    Parameters
    ----------
    effects : np.ndarray
        Array of point estimates.
    ses : np.ndarray
        Array of standard errors.
    alpha : float, optional
        Significance level (default 0.05).
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.

    Returns
    -------
    t_stats : np.ndarray
    p_values : np.ndarray
    ci_lowers : np.ndarray
    ci_uppers : np.ndarray
    """
    effects = np.asarray(effects, dtype=float)
    ses = np.asarray(ses, dtype=float)
    n = len(effects)

    t_stats = np.full(n, np.nan)
    p_values = np.full(n, np.nan)
    ci_lowers = np.full(n, np.nan)
    ci_uppers = np.full(n, np.nan)

    # Undefined df (e.g., rank-deficient replicate design) → all NaN
    if df is not None and df <= 0:
        return t_stats, p_values, ci_lowers, ci_uppers

    valid = np.isfinite(ses) & (ses > 0)
    if not np.any(valid):
        return t_stats, p_values, ci_lowers, ci_uppers

    t_stats[valid] = effects[valid] / ses[valid]

    if df is not None:
        p_values[valid] = 2.0 * stats.t.sf(np.abs(t_stats[valid]), df)
    else:
        p_values[valid] = 2.0 * stats.norm.sf(np.abs(t_stats[valid]))

    crit = _get_critical_value(alpha, df)
    ci_lowers[valid] = effects[valid] - crit * ses[valid]
    ci_uppers[valid] = effects[valid] + crit * ses[valid]

    return t_stats, p_values, ci_lowers, ci_uppers


_DF_CONVENTIONS = ("residual", "cluster", "normal")


def validate_df_convention(value: str) -> None:
    """Raise ValueError unless ``value`` is a supported df_convention.

    Shared by every estimator constructor and ``set_params`` that carries the
    ``df_convention`` knob. ``LinearRegression`` keeps its own inline check
    (``diff_diff.utils`` imports ``diff_diff.linalg`` at module level, so
    linalg cannot import from here at module scope).
    """
    if value not in _DF_CONVENTIONS:
        raise ValueError(f"df_convention must be one of {_DF_CONVENTIONS}, got {value!r}")


def validate_n_bootstrap(n_bootstrap: Any) -> None:
    """Raise ValueError unless ``n_bootstrap`` is a non-negative integer.

    Shared by every estimator constructor whose ``n_bootstrap`` gates an
    optional bootstrap (promoted from ChangesInChanges' local validator;
    accepts numpy integers, rejects bool/None/float/negative). ``0`` means
    bootstrap off wherever a ``> 0`` gate exists — the zero-default
    estimators (CallawaySantAnna, SunAbraham, EfficientDiD, ImputationDiD,
    TwoStageDiD, WooldridgeDiD, ContinuousDiD, StaggeredTripleDifference,
    TripleDifference, DMLDiD)
    and every analytical lane. On the DiD/TWFE wild-bootstrap lane 0 never
    meant off (the routing consults only ``inference=``); their fit-level
    floor rejects ``n_bootstrap < 2`` under ``inference="wild_bootstrap"``.
    """
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise ValueError(f"n_bootstrap must be a non-negative integer, got '{n_bootstrap}'")
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be a non-negative integer, got '{n_bootstrap}'")


# The staggered-only TripleDifference constructor params that genuinely select
# staggered behavior (row M-013). Lives here because TWO independent consumers
# need the same boundary and must not drift apart: TripleDifference.fit(), which
# rejects them in 2x2x2 mode, and the power entry points, which reject a
# staggered-configured estimator at the front door. `bootstrap_weights`, `seed`
# and `cband` are deliberately EXCLUDED - they act only through n_bootstrap > 0,
# which 2x2x2 mode already rejects, so both surfaces accept them as inert.
STAGGERED_DDD_CTOR_PARAMS = (
    "control_group",
    "anticipation",
    "base_period",
    "n_bootstrap",
)


def staggered_ddd_ctor_defaults(estimator: Any) -> Dict[str, Any]:
    """Read :data:`STAGGERED_DDD_CTOR_PARAMS` defaults off ``estimator``'s class.

    Derived from the live ``__init__`` signature rather than hard-coded, so a
    future default change cannot make the two mode-detection boundaries
    disagree — or, worse, reclassify an otherwise-default estimator as
    staggered-configured and reject a legitimate 2x2x2 run.

    Params absent from the signature, or present with no default, are skipped:
    a param with no default has no "non-default value" to detect, and treating
    ``Parameter.empty`` as the default would flag every instance.
    """
    try:
        params = inspect.signature(type(estimator).__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return {}
    return {
        name: params[name].default
        for name in STAGGERED_DDD_CTOR_PARAMS
        if name in params and params[name].default is not inspect.Parameter.empty
    }


def staggered_ddd_ctor_offenders(estimator: Any) -> List[str]:
    """Names in :data:`STAGGERED_DDD_CTOR_PARAMS` set to a non-default value."""
    return [
        name
        for name, default in staggered_ddd_ctor_defaults(estimator).items()
        if getattr(estimator, name, default) != default
    ]


def validate_anticipation(anticipation: Any) -> int:
    """Validate ``anticipation`` and return it as a normalized Python ``int``.

    Raises ``ValueError`` unless the value is a non-negative integer; on
    success returns ``int(anticipation)`` so numpy integers are normalized at
    the assignment site (``self.anticipation = validate_anticipation(...)``)
    before any ``-1 - anticipation``-style arithmetic can overflow on an
    unsigned numpy scalar.

    An out-of-domain anticipation window does not fail loudly on its own — it
    silently changes the ESTIMAND. On the staggered DDD engine the value feeds
    both the base-period rule and the comparison-cohort threshold, so
    ``anticipation=-1`` makes the universal base period ``g`` (an
    ALREADY-TREATED period) and relaxes the not-yet-treated threshold to
    ``max(t, base) - 1``, admitting cohorts treated at the evaluation period as
    clean controls. Neither condition is observable in the output.

    ``bool`` is rejected: ``True`` would otherwise coerce to a silent
    one-period window. Numpy integers are accepted, matching
    :func:`validate_n_bootstrap`, whose shape this follows.

    Adopted at ``__init__`` by all ten anticipation-taking estimators
    (CallawaySantAnna, SunAbraham, ImputationDiD, TwoStageDiD, StackedDiD,
    ContinuousDiD, EfficientDiD, WooldridgeDiD, SpilloverDiD, DMLDiD) plus
    ``TripleDifference``, and RE-CHECKED on the fit path on all of them —
    the uniform direct-mutation defense: manual re-assignments on eight,
    EfficientDiD via its ``_validate_params`` re-run, SpilloverDiD via its
    in-fit check, TripleDifference and the deprecated
    ``StaggeredTripleDifference`` via the shared staggered engine.
    """
    if isinstance(anticipation, bool) or not isinstance(anticipation, (int, np.integer)):
        raise ValueError(
            f"anticipation must be a non-negative integer; got "
            f"{anticipation!r} (type {type(anticipation).__name__})."
        )
    if anticipation < 0:
        raise ValueError(f"anticipation must be a non-negative integer; got {anticipation!r}.")
    return int(anticipation)


def resolve_tail_df(
    df_convention: str,
    *,
    residual_df: Optional[float],
    n_clusters: Optional[int],
) -> Optional[float]:
    """Resolve the fallback-level tail df for analytical t/p/CI inference.

    The single implementation of the ``df_convention`` knob's fallback slot
    for the standalone estimators (SunAbraham aggregates, WooldridgeDiD OLS,
    StackedDiD, ImputationDiD pretrends leads, LPDiD). Survey/replicate df
    and per-coefficient Bell-McCaffrey DOF keep precedence at the CALL SITE
    under every value — callers consult this helper only when no
    higher-precedence df applies. ``LinearRegression.get_inference`` and
    MultiPeriodDiD keep their own equivalent inline ladders.

    Branch table:

    - ``"residual"`` — ``residual_df`` (the fit's ``n_eff - K_full``).
    - ``"cluster"`` — ``n_clusters - 1`` when ``n_clusters >= 2``. When
      ``n_clusters`` is not None but ``<= 1`` the cluster df is undefined:
      warn and return ``0``, the fail-closed sentinel ``safe_inference``
      turns into all-NaN inference (the ``df = 0`` twin of MultiPeriodDiD's
      inline ``_df_cluster_knob_invalid`` path; ``LinearRegression.
      get_inference`` reaches the same observable warn+NaN outcome via an
      early return). Defense-in-depth only: every in-scope clustered solve
      raises "Need at least 2 clusters" at the vcov layer before df
      resolution, and LPDiD's degenerate lanes bypass this helper entirely.
      When ``n_clusters`` is None (unclustered/conley fit) the value is
      inert and falls back to ``residual_df``.
    - ``"normal"`` — ``None`` (deliberate normal-theory z inference at the
      fallback level, on clustered and unclustered fits alike).

    Whenever the residual fallback is what would be returned, a non-finite
    or non-positive ``residual_df`` warns and returns ``None`` (normal
    theory) — mirroring ``get_inference``'s non-positive-df fallback. The
    reachable case is absorbed-FE accounting driving ``n - k - rank`` to
    zero or below while the SE is still finite (a plain ``n <= k``
    saturation NaNs at the vcov layer first and never reaches this guard).

    Never wraps ``safe_inference`` — callers pass the resolved df to their
    own module-level ``safe_inference`` binding so the variance-conventions
    audit instrumentation observes it.
    """
    validate_df_convention(df_convention)

    def _guarded_residual(df: Optional[float]) -> Optional[float]:
        if df is None:
            return None
        if not (np.isfinite(df) and df > 0):
            warnings.warn(
                f"Degrees of freedom is non-positive (df={df}). "
                "Using normal distribution instead of t-distribution for inference.",
                UserWarning,
                stacklevel=3,
            )
            return None
        return float(df)

    if df_convention == "normal":
        return None
    if df_convention == "cluster":
        if n_clusters is None:
            return _guarded_residual(residual_df)
        if n_clusters <= 1:
            warnings.warn(
                "df_convention='cluster' requires at least 2 effective "
                f"clusters; got {n_clusters}. Inference fields will be NaN.",
                UserWarning,
                stacklevel=3,
            )
            return 0.0
        return float(n_clusters - 1)
    return _guarded_residual(residual_df)


# =============================================================================
# Wild Cluster Bootstrap
# =============================================================================


@dataclass
class WildBootstrapResults:
    """
    Results from wild cluster bootstrap inference.

    Attributes
    ----------
    se : float
        Analytical cluster-robust (CR1) standard error of the coefficient. The
        wild bootstrap studentizes the test with this SE; it is not a rescaled
        bootstrap dispersion.
    p_value : float
        Wild cluster bootstrap p-value (two-tailed or equal-tailed).
    t_stat_original : float
        Studentized statistic of the original estimate, ``(coef - null) / se``.
    ci_lower : float
        Lower bound of the confidence interval (by test inversion).
    ci_upper : float
        Upper bound of the confidence interval (by test inversion).
    n_clusters : int
        Number of clusters in the data.
    n_bootstrap : int
        Number of bootstrap replications.
    weight_type : str
        Type of bootstrap weights used ("rademacher", "webb", or "mammen").
    alpha : float
        Significance level used for confidence interval.
    p_val_type : str
        Test shape used ("two-tailed" or "equal-tailed").
    bootstrap_distribution : np.ndarray, optional
        Bootstrap distribution of the studentized statistic ``t*`` evaluated at
        the null (if requested).

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008).
    Bootstrap-Based Improvements for Inference with Clustered Errors.
    The Review of Economics and Statistics, 90(3), 414-427.
    """

    se: float
    p_value: float
    t_stat_original: float
    ci_lower: float
    ci_upper: float
    n_clusters: int
    n_bootstrap: int
    weight_type: str
    alpha: float = 0.05
    p_val_type: str = "two-tailed"
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        """Generate formatted summary of bootstrap results."""
        lines = [
            "Wild Cluster Bootstrap Results",
            "=" * 40,
            f"Cluster-robust SE:   {self.se:.6f}",
            f"Bootstrap p-value:   {self.p_value:.4f}",
            f"Studentized t-stat:  {self.t_stat_original:.4f}",
            f"CI ({int((1-self.alpha)*100)}%):           [{self.ci_lower:.6f}, {self.ci_upper:.6f}]",
            f"Number of clusters:  {self.n_clusters}",
            f"Bootstrap reps:      {self.n_bootstrap}",
            f"Weight type:         {self.weight_type}",
            f"Test type:           {self.p_val_type}",
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print formatted summary to stdout."""
        print(self.summary())


def _generate_rademacher_weights(n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate Rademacher weights: +1 or -1 with probability 0.5.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of Rademacher weights.
    """
    return np.asarray(rng.choice([-1.0, 1.0], size=n_clusters))


def _generate_webb_weights(n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate Webb's 6-point distribution weights.

    Values: {-sqrt(3/2), -sqrt(2/2), -sqrt(1/2), sqrt(1/2), sqrt(2/2), sqrt(3/2)}
    with equal probabilities (1/6 each), giving E[w]=0 and Var(w)=1.0.

    This distribution is recommended for very few clusters (G < 10) as it
    provides better finite-sample properties than Rademacher weights.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of Webb weights.

    References
    ----------
    Webb, M. D. (2014). Reworking wild bootstrap based inference for
    clustered errors. Queen's Economics Department Working Paper No. 1315.

    Note: Uses equal probabilities (1/6 each) matching R's `did` package,
    which gives unit variance for consistency with other weight distributions.
    """
    values = np.array(
        [
            -np.sqrt(3 / 2),
            -np.sqrt(2 / 2),
            -np.sqrt(1 / 2),
            np.sqrt(1 / 2),
            np.sqrt(2 / 2),
            np.sqrt(3 / 2),
        ]
    )
    # Equal probabilities (1/6 each) matching R's did package, giving Var(w) = 1.0
    return np.asarray(rng.choice(values, size=n_clusters))


def _generate_mammen_weights(
    n_clusters: "Union[int, Tuple[int, ...]]", rng: np.random.Generator
) -> np.ndarray:
    """
    Generate Mammen's two-point distribution weights.

    Values: {-(sqrt(5)-1)/2, (sqrt(5)+1)/2}
    with probabilities {(sqrt(5)+1)/(2*sqrt(5)), (sqrt(5)-1)/(2*sqrt(5))}.

    This distribution satisfies E[v]=0, E[v^2]=1, E[v^3]=1, which provides
    asymptotic refinement for skewed error distributions.

    Parameters
    ----------
    n_clusters : int or tuple of int
        Number of clusters, or an output shape. A batched draw of shape
        ``(B, G)`` consumes the generator's variate stream identically to
        ``B`` sequential draws of size ``G`` (``rng.choice`` fills output
        in C order), so batching replicate draws preserves the bootstrap
        law draw-for-draw.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of Mammen weights.

    References
    ----------
    Mammen, E. (1993). Bootstrap and Wild Bootstrap for High Dimensional
    Linear Models. The Annals of Statistics, 21(1), 255-285.
    """
    sqrt5 = np.sqrt(5)
    # Values from Mammen (1993)
    val1 = -(sqrt5 - 1) / 2  # approximately -0.618
    val2 = (sqrt5 + 1) / 2  # approximately 1.618 (golden ratio)

    # Probability of val1
    p1 = (sqrt5 + 1) / (2 * sqrt5)  # approximately 0.724

    return np.asarray(rng.choice([val1, val2], size=n_clusters, p=[p1, 1 - p1]))


def _wild_weight_matrix(
    n_clusters: int,
    n_bootstrap: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the ``(B, n_clusters)`` matrix of cluster-level bootstrap weights.

    For Rademacher weights with few clusters all ``2**n_clusters`` sign-vectors
    are enumerated (deterministic) once ``n_bootstrap`` reaches the number of
    possible draws — i.e. when ``2**n_clusters <= n_bootstrap`` (and
    ``n_clusters <= 20``, a guard against pathological memory use). This matches
    the full-enumeration switch of ``fwildclusterboot::boottest`` (verified: for
    ``G=10`` boottest samples at ``B=1023`` and enumerates at ``B=1024``); the
    reported ``n_bootstrap`` is then ``2**n_clusters``. (Only ``2**(n_clusters-1)``
    of those draws have distinct ``|t*|`` — each draw and its all-signs-flipped
    mirror share ``|t*|`` — but the full set is materialized.) Enumeration removes
    RNG dependence in the few-cluster regime where the wild bootstrap matters
    most. Otherwise ``n_bootstrap`` weight vectors are sampled. Webb/Mammen always
    sample: the sign-flip enumeration symmetry is Rademacher-specific (Mammen is
    asymmetric, Webb is a 6-point law).
    """
    if weight_type == "rademacher" and n_clusters <= 20 and 2**n_clusters <= n_bootstrap:
        n_enum = 2**n_clusters
        bits = (np.arange(n_enum)[:, None] >> np.arange(n_clusters)) & 1
        return np.where(bits == 1, 1.0, -1.0)

    generators = {
        "rademacher": _generate_rademacher_weights,
        "webb": _generate_webb_weights,
        "mammen": _generate_mammen_weights,
    }
    generate = generators[weight_type]
    weights = np.empty((n_bootstrap, n_clusters))
    for b in range(n_bootstrap):
        weights[b] = generate(n_clusters, rng)
    return weights


# Per-chunk byte budget for wild_bootstrap_se's r-independent precompute
# pass (divided by a conservative x8 peak-temporary multiplier to size the
# draw-rows per chunk). Read at CALL time so tests can monkeypatch it.
_WILD_PRECOMPUTE_CHUNK_BYTES = 256 * 1024 * 1024


def wild_bootstrap_se(
    X: np.ndarray,
    y: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
    coefficient_index: int,
    n_bootstrap: int = 999,
    weight_type: str = "rademacher",
    null_hypothesis: float = 0.0,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    return_distribution: bool = False,
    p_val_type: str = "two-tailed",
    *,
    cluster_k_adjustment: int = 0,
) -> WildBootstrapResults:
    """
    Compute wild cluster bootstrap standard errors and p-values.

    Implements the Wild Cluster Restricted (WCR) bootstrap of Cameron, Gelbach,
    and Miller (2008), matching the defaults of R's ``fwildclusterboot::boottest``
    (Roodman, MacKinnon, Nielsen & Webb 2019): the null ``H0: coefficient =
    null_hypothesis`` is genuinely imposed by re-estimating the model with the
    coefficient's column dropped, the bootstrap DGP resamples the *restricted*
    residuals, and the confidence interval is obtained by **inverting the
    bootstrap test** (the set of null values not rejected at level ``alpha``) so
    that the p-value and CI are mutually consistent (``0 in CI`` iff
    ``p >= alpha``). For Rademacher weights with few clusters all
    ``2**n_clusters`` sign-vectors are enumerated (deterministic) when
    ``2**n_clusters <= n_bootstrap`` (the ``boottest`` full-enumeration trigger —
    it switches to enumeration once ``n_bootstrap`` reaches the number of
    possible draws) and ``n_clusters <= 20`` (a memory guard); the reported
    ``n_bootstrap`` is then ``2**n_clusters``. Otherwise signs are sampled.

    The reported ``se`` is the analytical cluster-robust (CR1) standard error of
    the original estimate — the studentized bootstrap drives the p-value and CI,
    not a re-scaled bootstrap dispersion.

    ``cluster_k_adjustment`` (keyword-only, default 0) applies the signed
    K_reference count to BOTH the analytical CR1 SE and the bootstrap ``corr``
    factor (see ``docs/methodology/variance-conventions.md`` D1/D2), keeping
    ``se``, ``t_stat_original`` (= effect/se) and any returned
    ``bootstrap_distribution`` on the corrected scale. The constant cancels in
    the studentized statistic, so p-values are invariant and CI endpoints move
    only within the test-inversion bisection tolerance (~1e-10) — never assert
    exact CI equality across adjustment values.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    y : np.ndarray
        Outcome vector of shape (n,).
    residuals : np.ndarray
        Retained for backward compatibility and IGNORED by the WCR
        implementation, which recomputes the original fit and the restricted
        (null-imposed) residualization internally from ``X`` and ``y``.
    cluster_ids : np.ndarray
        Cluster identifiers of shape (n,).
    coefficient_index : int
        Index of the coefficient for which to compute bootstrap inference.
        For DiD, this is typically 3 (the treatment*post interaction term).
    n_bootstrap : int, default=999
        Number of bootstrap replications. Odd numbers are recommended for
        exact p-value computation.
    weight_type : str, default="rademacher"
        Type of bootstrap weights:
        - "rademacher": +1 or -1 with equal probability (standard choice)
        - "webb": 6-point distribution (recommended for <10 clusters)
        - "mammen": Two-point distribution with skewness correction
    null_hypothesis : float, default=0.0
        Value of the null hypothesis for p-value computation.
    alpha : float, default=0.05
        Significance level for confidence interval.
    seed : int, optional
        Random seed for reproducibility. If None (default), results
        will vary between runs.
    return_distribution : bool, default=False
        If True, include the bootstrap distribution of the studentized statistic
        ``t*`` (evaluated at the null) in the results.
    p_val_type : str, default="two-tailed"
        Shape of the test (mirrors ``boottest``'s ``p_val_type``):

        - "two-tailed": test on ``|t*|``; two-tailed CI by inversion (the
          interval need not be symmetric about the estimate).
        - "equal-tailed": each tail tested at ``alpha/2``; equal-tailed CI.

    Returns
    -------
    WildBootstrapResults
        Dataclass containing bootstrap SE, p-value, confidence interval,
        and other inference results.

    Raises
    ------
    ValueError
        If weight_type is not recognized or if there are fewer than 2 clusters.

    Warns
    -----
    UserWarning
        If the number of clusters is less than 5, as bootstrap inference
        may be unreliable.

    Examples
    --------
    >>> from diff_diff.utils import wild_bootstrap_se
    >>> results = wild_bootstrap_se(
    ...     X, y, residuals, cluster_ids,
    ...     coefficient_index=3,  # ATT coefficient
    ...     n_bootstrap=999,
    ...     weight_type="rademacher",
    ...     seed=42
    ... )
    >>> print(f"Bootstrap SE: {results.se:.4f}")
    >>> print(f"Bootstrap p-value: {results.p_value:.4f}")

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008).
    Bootstrap-Based Improvements for Inference with Clustered Errors.
    The Review of Economics and Statistics, 90(3), 414-427.

    MacKinnon, J. G., & Webb, M. D. (2018). The wild bootstrap for
    few (treated) clusters. The Econometrics Journal, 21(2), 114-135.
    """
    # Validate inputs. The adjustment TYPE is checked at entry — before the
    # dropped-coefficient / saturation early returns below — so a non-int
    # cannot be silently absorbed into a degenerate NaN result on those
    # routes (the value/family contract is enforced by solve_ols itself at
    # the clustered vcov call).
    _validate_cluster_k_adjustment_type(cluster_k_adjustment)
    valid_weight_types = ["rademacher", "webb", "mammen"]
    if weight_type not in valid_weight_types:
        raise ValueError(f"weight_type must be one of {valid_weight_types}, got '{weight_type}'")
    valid_p_val_types = ["two-tailed", "equal-tailed"]
    if p_val_type not in valid_p_val_types:
        raise ValueError(f"p_val_type must be one of {valid_p_val_types}, got '{p_val_type}'")

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        raise ValueError(f"Wild cluster bootstrap requires at least 2 clusters, got {n_clusters}")

    if n_clusters < 5:
        warnings.warn(
            f"Only {n_clusters} clusters detected. Wild cluster bootstrap inference may be "
            "unreliable with fewer than 5 clusters. With Rademacher weights all "
            f"{2 ** n_clusters} sign-vectors are enumerated exactly when "
            f"n_bootstrap >= 2**n_clusters = {2 ** n_clusters}; Webb weights "
            "(weight_type='webb') improve finite-sample behaviour but are sampled, not "
            "enumerated.",
            UserWarning,
        )

    rng = np.random.default_rng(seed)
    n = X.shape[0]

    def _degenerate() -> WildBootstrapResults:
        # All-or-nothing NaN contract (feedback_bootstrap_nan_on_invalid_contract):
        # when the original fit or the bootstrap is degenerate, NaN the entire
        # (se, t_stat, p_value, ci) inference family together rather than mixing
        # analytical and bootstrap quantities on the same coefficient.
        return WildBootstrapResults(
            se=float("nan"),
            p_value=float("nan"),
            t_stat_original=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_clusters=n_clusters,
            n_bootstrap=n_bootstrap,
            weight_type=weight_type,
            alpha=alpha,
            p_val_type=p_val_type,
            bootstrap_distribution=None,
        )

    # Step 1: original fit. Establishes the analytical cluster-robust (CR1) SE
    # that studentizes the test, and the set of identified (kept) columns so the
    # bootstrap stays rank-robust (e.g. an always-treated unit dummy collinear
    # with treated*post on the full-dummy TWFE path: solve_ols drops the nuisance
    # column and reports it as NaN, while the identified ATT is retained).
    # First fit WITHOUT the cluster-robust vcov: this identifies the kept
    # (full-rank) columns and lets us reject a saturated design *before*
    # requesting the cluster sandwich. The shared CR1 small-sample adjustment
    # (n_eff-1)/(n_eff-k) divides by zero on a saturated design (n == rank), so
    # routing the degenerate case here keeps the all-or-nothing NaN contract.
    beta_hat, _, _ = _solve_ols_linalg(X, y, return_vcov=False)
    original_coef = float(beta_hat[coefficient_index])
    if not np.isfinite(original_coef):
        return _degenerate()
    kept = np.isfinite(beta_hat)
    if not bool(kept[coefficient_index]):
        return _degenerate()
    X_eff = X[:, kept]
    j_eff = int(np.sum(kept[:coefficient_index]))  # position of the coef among kept columns
    k_eff = X_eff.shape[1]
    # Fail closed on ALL THREE saturation sides, matching the kernel: the
    # visible n <= k_eff (residuals identically zero), the K_reference side
    # n <= k_eff + adj, and a negative adjustment exceeding the reduced width
    # (k_eff + adj <= 0, where the corr factor below would silently DEFLATE).
    _k_inf_eff = k_eff + cluster_k_adjustment
    if n <= k_eff or n <= _k_inf_eff or _k_inf_eff <= 0:
        return _degenerate()

    # Now the cluster-robust (CR1) vcov is well-defined; it studentizes the test.
    _, _, vcov_original = _solve_ols_linalg(
        X, y, cluster_ids=cluster_ids, return_vcov=True, cluster_k_adjustment=cluster_k_adjustment
    )
    if vcov_original is None:
        return _degenerate()
    se_a = float(np.sqrt(vcov_original[coefficient_index, coefficient_index]))
    if not np.isfinite(se_a) or se_a <= 0:
        return _degenerate()

    # Projections on the (full-rank) effective design.
    XtX_inv = np.linalg.inv(X_eff.T @ X_eff)
    a_vec = X_eff @ XtX_inv[:, j_eff]  # influence of each obs on beta_j: beta*_j = a_vec . y*
    proj = XtX_inv @ X_eff.T  # (k_eff, n) OLS projection onto coefficients

    # Restricted residualization imposing H0: regress y and x_j on X_eff \ {col j}.
    # The restricted residuals u(r) = M_{-j} y - r * M_{-j} x_j are linear in the
    # candidate null r, so the whole test can be re-evaluated at any r cheaply.
    xj = X_eff[:, j_eff]
    X_reduced = np.delete(X_eff, j_eff, axis=1)
    if X_reduced.shape[1] == 0:
        # Single-regressor design: the reduced model has no regressors, so the
        # restricted fit is identically 0 and the residuals are the variables
        # themselves (solve_ols cannot fit a zero-column design).
        fit_y_red = np.zeros(n)
        fit_xj_red = np.zeros(n)
    else:
        _, _, fit_y_red, _ = _solve_ols_linalg(X_reduced, y, return_vcov=False, return_fitted=True)
        _, _, fit_xj_red, _ = _solve_ols_linalg(
            X_reduced, xj, return_vcov=False, return_fitted=True
        )
    m_y = y - fit_y_red
    m_xj = xj - fit_xj_red

    # CR1 small-sample correction on the K_reference count (k_eff + adj).
    # NOTE: this constant cancels in |t*| vs |t0| (it scales se* and se_a
    # identically), so it affects only the reported SE, not the p-value or
    # CI. Kept for fidelity with the analytical CR1 SE.
    corr = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - _k_inf_eff))

    # Cluster membership: indicator matrix C (G, n) for fast per-cluster score sums.
    cluster_pos = {c: i for i, c in enumerate(unique_clusters)}
    cl_idx = np.array([cluster_pos[c] for c in cluster_ids])
    cluster_indicator = np.zeros((n_clusters, n))
    cluster_indicator[cl_idx, np.arange(n)] = 1.0

    # Fixed bootstrap weights, held constant across the whole test inversion so
    # that p(r) is a stable (monotone, step) function amenable to root-finding.
    weights = _wild_weight_matrix(n_clusters, n_bootstrap, weight_type, rng)
    n_boot_eff = int(weights.shape[0])

    # ---- r-independent precompute for the test inversion --------------------
    # The WCR DGP is LINEAR in the candidate null r:
    #     u_r = m_y - r * m_xj
    #     y*(r) = y - u_r * (1 - w_obs) = A + r * B,
    # with A = y - m_y*(1 - w_obs) and B = m_xj*(1 - w_obs) (both (B, n) and
    # r-independent). Everything downstream is affine in y*, so
    #     beta_j*(r)      = alpha1 + r * beta1                    ((B,) each)
    #     scores(r)       = SA + r * SB                           ((G, B))
    #     se*(r)^2 / corr = qa + r * qb + r^2 * qc                ((B,) each,
    # the PSD quadratic form sum_g (SA + r SB)^2). The CI inversion calls
    # _t_star ~O(100) times; precomputing these five (B,) vectors turns each
    # call from three (B x n)/(n x B) GEMM passes into O(B) arithmetic, and
    # the ONE precompute pass is chunked over draws so peak memory stays
    # bounded (the old code materialized fresh (B, n) intermediates on every
    # call). Not bit-identical to the per-call form (the variance is evaluated
    # through the expanded quadratic instead of sum(scores^2) at each r —
    # ~1 ULP reassociation class); the strict-inequality tie guard below
    # absorbs sub-1e-9 shifts, so the discrete outputs (p-value, CI
    # endpoints) are unchanged on the regression grid.
    # The cluster-level weights matrix is only (B, G); the (B, n)
    # observation-level expansion is built PER CHUNK below so no full
    # (B, n) array is ever materialized.
    alpha1 = np.empty(n_boot_eff, dtype=np.float64)
    beta1 = np.empty(n_boot_eff, dtype=np.float64)
    qa = np.empty(n_boot_eff, dtype=np.float64)
    qb = np.empty(n_boot_eff, dtype=np.float64)
    qc = np.empty(n_boot_eff, dtype=np.float64)
    # Conservative sizing: up to ~8 (Bc, n) float64 arrays/temporaries can
    # be live around the residual + score construction (one_minus_w_blk,
    # A_blk, B_blk, RA_blk, RB_blk, plus elementwise/GEMM temporaries), so
    # the divisor uses that peak multiplier against the byte cap below
    # (module constant, read at call time so tests can monkeypatch it to
    # force the multi-chunk branch).
    _rows_per_chunk = max(1, int(_WILD_PRECOMPUTE_CHUNK_BYTES // (8 * 8 * max(n, 1))))
    for _cs in range(0, n_boot_eff, _rows_per_chunk):
        blk = slice(_cs, min(_cs + _rows_per_chunk, n_boot_eff))
        one_minus_w_blk = 1.0 - weights[blk][:, cl_idx]  # (Bc, n)
        A_blk = y[None, :] - m_y[None, :] * one_minus_w_blk  # (Bc, n)
        B_blk = m_xj[None, :] * one_minus_w_blk  # (Bc, n)
        alpha1[blk] = A_blk @ a_vec
        beta1[blk] = B_blk @ a_vec
        # Residual-maker applied to each draw row: R = Z - (Z @ proj.T) @ X_eff.T
        RA_blk = A_blk - (A_blk @ proj.T) @ X_eff.T  # (Bc, n)
        RB_blk = B_blk - (B_blk @ proj.T) @ X_eff.T
        SA_blk = (a_vec[None, :] * RA_blk) @ cluster_indicator.T  # (Bc, G)
        SB_blk = (a_vec[None, :] * RB_blk) @ cluster_indicator.T
        qa[blk] = np.sum(SA_blk * SA_blk, axis=1)
        qb[blk] = 2.0 * np.sum(SA_blk * SB_blk, axis=1)
        qc[blk] = np.sum(SB_blk * SB_blk, axis=1)

    def _t_star(r: float) -> np.ndarray:
        """Studentized bootstrap statistics t*(r) under H0: beta_j = r."""
        # PSD quadratic form; roundoff can dip microscopically negative at an
        # interior root — clamp so sqrt is defined, then the se>0 guard NaNs
        # the degenerate draws exactly as the per-call form did.
        var_star = corr * np.maximum(qa + r * qb + r * r * qc, 0.0)
        se_star = np.sqrt(var_star)  # (B,)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (alpha1 + r * beta1 - r) / se_star
        t[~(se_star > 0)] = np.nan
        return t

    t_star = _t_star(null_hypothesis)
    finite = np.isfinite(t_star)
    n_valid = int(finite.sum())
    if n_valid < 2:
        return _degenerate()
    t_star_valid = t_star[finite]
    t0 = (original_coef - null_hypothesis) / se_a

    # Strict-inequality tail counts, matching fwildclusterboot/boottest: a
    # bootstrap statistic is counted only if it *exceeds* the observed one. In
    # the fully-enumerated few-cluster case the all-(+1) / all-(-1) sign-vectors
    # reproduce t* = +/- t0 exactly (the observed draw and its mirror); strict
    # ">" excludes those boundary ties, as boottest does. The small relative
    # guard (~1e-9) makes the exclusion robust to floating-point noise from the
    # fast-form path so a true tie never sneaks in as a strict exceedance.
    def _frac_gt(vals: np.ndarray, thresh: float) -> float:
        return float(np.mean(vals > thresh + 1e-9 * max(1.0, abs(thresh))))

    def _frac_lt(vals: np.ndarray, thresh: float) -> float:
        return float(np.mean(vals < thresh - 1e-9 * max(1.0, abs(thresh))))

    # p-value at the test null (two-tailed on |t*|, or equal-tailed).
    if p_val_type == "two-tailed":
        raw_p = _frac_gt(np.abs(t_star_valid), abs(t0))
    else:
        p_low = _frac_lt(t_star_valid, t0)
        p_up = _frac_gt(t_star_valid, t0)
        raw_p = 2.0 * min(p_low, p_up)
    # Floor the reported p-value to avoid an exact zero (a documented departure
    # from boottest, which can report p == 0) — but NEVER let the floor reach
    # the significance level. With very few valid draws 1/(n_valid+1) can exceed
    # alpha, and flooring there would flip a bootstrap-significant result (0
    # outside the inverted CI) to "non-significant", re-creating the very
    # p-vs-CI contradiction this estimator fixes. When the floor would cross
    # alpha we report the raw p-value (which is < alpha in exactly those cases),
    # so the significance verdict always agrees with the inverted CI.
    floor = 1.0 / (n_valid + 1)
    p_value = max(raw_p, floor) if floor < alpha else raw_p
    p_value = float(min(1.0, p_value))

    # ---- Confidence interval by test inversion ------------------------------
    # The CI is the set of nulls r not rejected at level alpha. The relevant
    # rejection frequency is monotonically decreasing as r moves away from the
    # point estimate, so each endpoint is found by outward bracketing + plain
    # bisection — robust to the step-function nature of a finite bootstrap
    # (unlike brentq, which assumes a continuous sign change).
    def _reject_two_tailed(r: float) -> float:
        t = _t_star(r)
        t = t[np.isfinite(t)]
        if t.size < 2:
            return 0.0
        t0_r = (original_coef - r) / se_a
        return _frac_gt(np.abs(t), abs(t0_r))

    def _tail_freq(r: float, upper: bool) -> float:
        t = _t_star(r)
        t = t[np.isfinite(t)]
        if t.size < 2:
            return 0.0
        t0_r = (original_coef - r) / se_a
        return _frac_gt(t, t0_r) if upper else _frac_lt(t, t0_r)

    def _bisect(f: Any, level: float, direction: int) -> float:
        # f(center) >= level; search outward (direction -1 lower, +1 upper) for
        # the crossing f(r) = level (f decreasing in |r - center|), then bisect.
        center = original_coef
        scale = se_a if se_a > 0 else 1.0
        step = scale
        hi = center + direction * step
        bracketed = False
        for _ in range(64):
            if f(hi) < level:
                bracketed = True
                break
            step *= 2.0
            hi = center + direction * step
        if not bracketed:
            # The test never rejects arbitrarily far out: the inverted CI is
            # genuinely unbounded on this side. Represent it with a signed
            # infinity (NOT NaN) so the (se, t, p, CI) inference family stays
            # internally consistent — 0 still lies inside an unbounded interval
            # exactly when the test fails to reject it, preserving
            # 0 ∈ CI ⟺ p ≥ alpha.
            return float(direction) * np.inf
        lo = center  # f(lo) >= level, f(hi) < level
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if f(mid) >= level:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) <= 1e-10 * max(1.0, abs(center)):
                break
        return 0.5 * (lo + hi)

    if p_val_type == "two-tailed":
        ci_lower = _bisect(_reject_two_tailed, alpha, -1)
        ci_upper = _bisect(_reject_two_tailed, alpha, +1)
    else:
        # equal-tailed: lower endpoint where the upper-tail frequency hits
        # alpha/2; upper endpoint where the lower-tail frequency hits alpha/2.
        ci_lower = _bisect(lambda r: _tail_freq(r, True), alpha / 2.0, -1)
        ci_upper = _bisect(lambda r: _tail_freq(r, False), alpha / 2.0, +1)

    return WildBootstrapResults(
        se=se_a,
        p_value=p_value,
        t_stat_original=t0,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_clusters=n_clusters,
        n_bootstrap=n_boot_eff,
        weight_type=weight_type,
        alpha=alpha,
        p_val_type=p_val_type,
        bootstrap_distribution=t_star_valid if return_distribution else None,
    )


def check_parallel_trends(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    treatment_group: str,
    pre_periods: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Perform a simple check for parallel trends assumption.

    This computes the trend (slope) in the outcome variable for both
    treatment and control groups during pre-treatment periods.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Name of outcome variable column.
    time : str
        Name of time period column.
    treatment_group : str
        Name of treatment group indicator column.
    pre_periods : list, optional
        List of pre-treatment time periods. If None, infers from data.

    Returns
    -------
    dict
        Dictionary with trend statistics and test results.
    """
    if pre_periods is None:
        # Assume treatment happens at median time period
        all_periods = sorted(data[time].unique())
        mid_point = len(all_periods) // 2
        pre_periods = all_periods[:mid_point]

    pre_data = data[data[time].isin(pre_periods)]

    # Compute trends for each group
    treated_data = pre_data[pre_data[treatment_group] == 1]
    control_data = pre_data[pre_data[treatment_group] == 0]

    # Simple linear regression for trends
    def compute_trend(group_data: pd.DataFrame) -> Tuple[float, float]:
        time_values = group_data[time].values
        outcome_values = group_data[outcome].values

        # Normalize time to start at 0
        time_norm = time_values - time_values.min()

        # Compute slope using least squares
        n = len(time_norm)
        if n < 2:
            return np.nan, np.nan

        mean_t = np.mean(time_norm)
        mean_y = np.mean(outcome_values)

        # Check for zero variance in time (all same time period)
        time_var = np.sum((time_norm - mean_t) ** 2)
        if time_var == 0:
            return np.nan, np.nan

        slope = np.sum((time_norm - mean_t) * (outcome_values - mean_y)) / time_var

        # Compute standard error of slope
        y_hat = mean_y + slope * (time_norm - mean_t)
        residuals = outcome_values - y_hat
        mse = np.sum(residuals**2) / (n - 2)
        se_slope = np.sqrt(mse / time_var)

        return slope, se_slope

    treated_slope, treated_se = compute_trend(treated_data)
    control_slope, control_se = compute_trend(control_data)

    # Test for difference in trends
    slope_diff = treated_slope - control_slope
    se_diff = np.sqrt(treated_se**2 + control_se**2)
    t_stat, p_value, _ = safe_inference(slope_diff, se_diff)

    return {
        "treated_trend": treated_slope,
        "treated_trend_se": treated_se,
        "control_trend": control_slope,
        "control_trend_se": control_se,
        "trend_difference": slope_diff,
        "trend_difference_se": se_diff,
        "t_statistic": t_stat,
        "p_value": p_value,
        "parallel_trends_plausible": p_value > 0.05 if not np.isnan(p_value) else None,
    }


def check_parallel_trends_robust(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    treatment_group: str,
    unit: Optional[str] = None,
    pre_periods: Optional[List[Any]] = None,
    n_permutations: int = 1000,
    seed: Optional[int] = None,
    wasserstein_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Perform robust parallel trends testing using distributional comparisons.

    Uses the Wasserstein (Earth Mover's) distance to compare the full
    distribution of outcome changes between treated and control groups,
    with permutation-based inference.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with repeated observations over time.
    outcome : str
        Name of outcome variable column.
    time : str
        Name of time period column.
    treatment_group : str
        Name of treatment group indicator column (0/1).
    unit : str, optional
        Name of unit identifier column. If provided, computes unit-level
        changes. Otherwise uses observation-level data.
    pre_periods : list, optional
        List of pre-treatment time periods. If None, uses first half of periods.
    n_permutations : int, default=1000
        Number of permutations for computing p-value.
    seed : int, optional
        Random seed for reproducibility.
    wasserstein_threshold : float, default=0.2
        Threshold for normalized Wasserstein distance. Values below this
        threshold (combined with p > 0.05) suggest parallel trends are plausible.

    Returns
    -------
    dict
        Dictionary containing:
        - wasserstein_distance: Wasserstein distance between group distributions
        - wasserstein_p_value: Permutation-based p-value
        - ks_statistic: Kolmogorov-Smirnov test statistic
        - ks_p_value: KS test p-value
        - mean_difference: Difference in mean changes
        - variance_ratio: Ratio of variances in changes
        - treated_changes: Array of outcome changes for treated
        - control_changes: Array of outcome changes for control
        - parallel_trends_plausible: Boolean assessment

    Examples
    --------
    >>> results = check_parallel_trends_robust(
    ...     data, outcome='sales', time='year',
    ...     treatment_group='treated', unit='firm_id'
    ... )
    >>> print(f"Wasserstein distance: {results['wasserstein_distance']:.4f}")
    >>> print(f"P-value: {results['wasserstein_p_value']:.4f}")

    Notes
    -----
    The Wasserstein distance (Earth Mover's Distance) measures the minimum
    "cost" of transforming one distribution into another. Unlike simple
    mean comparisons, it captures differences in the entire distribution
    shape, making it more robust to non-normal data and heterogeneous effects.

    A small Wasserstein distance and high p-value suggest the distributions
    of pre-treatment changes are similar, supporting the parallel trends
    assumption.
    """
    # Use local RNG to avoid affecting global random state
    rng = np.random.default_rng(seed)

    # Identify pre-treatment periods
    if pre_periods is None:
        all_periods = sorted(data[time].unique())
        mid_point = len(all_periods) // 2
        pre_periods = all_periods[:mid_point]

    pre_data = data[data[time].isin(pre_periods)].copy()

    # Compute outcome changes
    treated_changes, control_changes = _compute_outcome_changes(
        pre_data,
        outcome,
        time,
        treatment_group,
        unit,
        caller_label="check_parallel_trends_robust",
    )

    if len(treated_changes) < 2 or len(control_changes) < 2:
        return {
            "wasserstein_distance": np.nan,
            "wasserstein_p_value": np.nan,
            "ks_statistic": np.nan,
            "ks_p_value": np.nan,
            "mean_difference": np.nan,
            "variance_ratio": np.nan,
            "treated_changes": treated_changes,
            "control_changes": control_changes,
            "parallel_trends_plausible": None,
            "error": "Insufficient data for comparison",
        }

    # Compute Wasserstein distance
    wasserstein_dist = stats.wasserstein_distance(treated_changes, control_changes)

    # Permutation test for Wasserstein distance
    all_changes = np.concatenate([treated_changes, control_changes])
    n_treated = len(treated_changes)
    n_total = len(all_changes)

    permuted_distances = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        perm_treated = all_changes[perm_idx[:n_treated]]
        perm_control = all_changes[perm_idx[n_treated:]]
        permuted_distances[i] = stats.wasserstein_distance(perm_treated, perm_control)

    # P-value: proportion of permuted distances >= observed
    wasserstein_p = np.mean(permuted_distances >= wasserstein_dist)

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.ks_2samp(treated_changes, control_changes)

    # Additional summary statistics
    mean_diff = np.mean(treated_changes) - np.mean(control_changes)
    var_treated = np.var(treated_changes, ddof=1)
    var_control = np.var(control_changes, ddof=1)
    var_ratio = var_treated / var_control if var_control > 0 else np.nan

    # Normalized Wasserstein (relative to pooled std)
    pooled_std = np.std(all_changes, ddof=1)
    wasserstein_normalized = wasserstein_dist / pooled_std if pooled_std > 0 else np.nan

    # Assessment: parallel trends plausible if p-value > 0.05
    # and normalized Wasserstein is small (below threshold)
    plausible = bool(
        wasserstein_p > 0.05
        and (
            wasserstein_normalized < wasserstein_threshold
            if not np.isnan(wasserstein_normalized)
            else True
        )
    )

    return {
        "wasserstein_distance": wasserstein_dist,
        "wasserstein_normalized": wasserstein_normalized,
        "wasserstein_p_value": wasserstein_p,
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "mean_difference": mean_diff,
        "variance_ratio": var_ratio,
        "n_treated": len(treated_changes),
        "n_control": len(control_changes),
        "treated_changes": treated_changes,
        "control_changes": control_changes,
        "parallel_trends_plausible": plausible,
    }


def _compute_outcome_changes(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    treatment_group: str,
    unit: Optional[str] = None,
    caller_label: str = "parallel-trend diagnostic",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute period-to-period outcome changes for treated and control groups.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column.
    time : str
        Time period column.
    treatment_group : str
        Treatment group indicator column.
    unit : str, optional
        Unit identifier column.

    Returns
    -------
    tuple
        (treated_changes, control_changes) as numpy arrays.
    """
    if unit is not None:
        # Unit-level changes: compute change for each unit across periods
        data_sorted = data.sort_values([unit, time])
        data_sorted["_outcome_change"] = data_sorted.groupby(unit)[outcome].diff()

        # Remove NaN from first period of each unit. The first period per unit
        # has no prior observation to diff against, so n_units drops are
        # expected. Anything beyond that is a silent side-effect of gaps or
        # NaN outcomes — surface the excess via warning (axis-E drop counter).
        n_units_observed = int(data_sorted[unit].nunique())
        n_dropped = int(data_sorted["_outcome_change"].isna().sum())
        n_unexpected_drops = max(0, n_dropped - n_units_observed)
        if n_unexpected_drops > 0:
            warnings.warn(
                f"{caller_label}: dropped {n_dropped} row(s) with NaN "
                f"first-differences; {n_units_observed} are the expected "
                f"first-period-per-unit drops, and {n_unexpected_drops} are "
                f"additional NaN first-differences (e.g. NaN outcomes or "
                f"unit-period gaps upstream). Parallel-trend statistics are "
                f"computed on the remaining rows.",
                UserWarning,
                stacklevel=3,
            )
        changes_data = data_sorted.dropna(subset=["_outcome_change"])

        treated_changes = changes_data[changes_data[treatment_group] == 1]["_outcome_change"].values

        control_changes = changes_data[changes_data[treatment_group] == 0]["_outcome_change"].values
    else:
        # Aggregate changes: compute mean change per period per group
        treated_data = data[data[treatment_group] == 1]
        control_data = data[data[treatment_group] == 0]

        # Compute period means
        treated_means = treated_data.groupby(time)[outcome].mean()
        control_means = control_data.groupby(time)[outcome].mean()

        # Compute changes between consecutive periods
        treated_changes = np.diff(treated_means.values)
        control_changes = np.diff(control_means.values)

    return treated_changes.astype(float), control_changes.astype(float)


def equivalence_test_trends(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    treatment_group: str,
    unit: Optional[str] = None,
    pre_periods: Optional[List[Any]] = None,
    equivalence_margin: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Perform equivalence testing (TOST) for parallel trends.

    Tests whether the difference in trends is practically equivalent to zero
    using Two One-Sided Tests (TOST) procedure.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Name of outcome variable column.
    time : str
        Name of time period column.
    treatment_group : str
        Name of treatment group indicator column.
    unit : str, optional
        Name of unit identifier column.
    pre_periods : list, optional
        List of pre-treatment time periods.
    equivalence_margin : float, optional
        The margin for equivalence (delta). If None, uses 0.5 * pooled SD
        of outcome changes as a default.

    Returns
    -------
    dict
        Dictionary containing:
        - mean_difference: Difference in mean changes
        - equivalence_margin: The margin used
        - lower_p_value: P-value for lower bound test
        - upper_p_value: P-value for upper bound test
        - tost_p_value: Maximum of the two p-values
        - equivalent: Boolean indicating equivalence at alpha=0.05
    """
    # Get pre-treatment periods
    if pre_periods is None:
        all_periods = sorted(data[time].unique())
        mid_point = len(all_periods) // 2
        pre_periods = all_periods[:mid_point]

    pre_data = data[data[time].isin(pre_periods)].copy()

    # Compute outcome changes
    treated_changes, control_changes = _compute_outcome_changes(
        pre_data,
        outcome,
        time,
        treatment_group,
        unit,
        caller_label="equivalence_test_trends",
    )

    # Need at least 2 observations per group to compute variance
    # and at least 3 total for meaningful df calculation
    if len(treated_changes) < 2 or len(control_changes) < 2:
        return {
            "mean_difference": np.nan,
            "se_difference": np.nan,
            "equivalence_margin": np.nan,
            "lower_t_stat": np.nan,
            "upper_t_stat": np.nan,
            "lower_p_value": np.nan,
            "upper_p_value": np.nan,
            "tost_p_value": np.nan,
            "degrees_of_freedom": np.nan,
            "equivalent": None,
            "error": "Insufficient data (need at least 2 observations per group)",
        }

    # Compute statistics
    var_t = np.var(treated_changes, ddof=1)
    var_c = np.var(control_changes, ddof=1)
    n_t = len(treated_changes)
    n_c = len(control_changes)

    mean_diff = np.mean(treated_changes) - np.mean(control_changes)

    # Handle zero variance case
    if var_t == 0 and var_c == 0:
        return {
            "mean_difference": mean_diff,
            "se_difference": 0.0,
            "equivalence_margin": np.nan,
            "lower_t_stat": np.nan,
            "upper_t_stat": np.nan,
            "lower_p_value": np.nan,
            "upper_p_value": np.nan,
            "tost_p_value": np.nan,
            "degrees_of_freedom": np.nan,
            "equivalent": None,
            "error": "Zero variance in both groups - cannot perform t-test",
        }

    se_diff = np.sqrt(var_t / n_t + var_c / n_c)

    # Handle zero SE case (cannot divide by zero in t-stat calculation)
    if se_diff == 0:
        return {
            "mean_difference": mean_diff,
            "se_difference": 0.0,
            "equivalence_margin": np.nan,
            "lower_t_stat": np.nan,
            "upper_t_stat": np.nan,
            "lower_p_value": np.nan,
            "upper_p_value": np.nan,
            "tost_p_value": np.nan,
            "degrees_of_freedom": np.nan,
            "equivalent": None,
            "error": "Zero standard error - cannot perform t-test",
        }

    # Set equivalence margin if not provided
    if equivalence_margin is None:
        pooled_changes = np.concatenate([treated_changes, control_changes])
        equivalence_margin = 0.5 * np.std(pooled_changes, ddof=1)

    # Degrees of freedom (Welch-Satterthwaite approximation)
    # Guard against division by zero when one group has zero variance
    numerator = (var_t / n_t + var_c / n_c) ** 2
    denom_t = (var_t / n_t) ** 2 / (n_t - 1) if var_t > 0 else 0
    denom_c = (var_c / n_c) ** 2 / (n_c - 1) if var_c > 0 else 0
    denominator = denom_t + denom_c

    if denominator == 0:
        # Fall back to minimum of n_t-1 and n_c-1 when one variance is zero
        df = min(n_t - 1, n_c - 1)
    else:
        df = numerator / denominator

    # TOST: Two one-sided tests
    # Test 1: H0: diff <= -margin vs H1: diff > -margin
    t_lower = (mean_diff - (-equivalence_margin)) / se_diff
    p_lower = stats.t.sf(t_lower, df)

    # Test 2: H0: diff >= margin vs H1: diff < margin
    t_upper = (mean_diff - equivalence_margin) / se_diff
    p_upper = stats.t.cdf(t_upper, df)

    # TOST p-value is the maximum of the two
    tost_p = max(p_lower, p_upper)

    return {
        "mean_difference": mean_diff,
        "se_difference": se_diff,
        "equivalence_margin": equivalence_margin,
        "lower_t_stat": t_lower,
        "upper_t_stat": t_upper,
        "lower_p_value": p_lower,
        "upper_p_value": p_upper,
        "tost_p_value": tost_p,
        "degrees_of_freedom": df,
        "equivalent": bool(tost_p < 0.05),
    }


# compute_synthetic_weights and _compute_synthetic_weights_numpy removed in the
# silent-failures audit post-cleanup (finding #22). The one caller
# (`diff_diff.prep.rank_control_units`) inlines a single-pass, uncentered
# Frank-Wolfe via the shared `_sc_weight_fw` dispatcher — a ranking heuristic,
# NOT the canonical SDID/R `synthdid::sc.weight.fw` two-pass procedure
# (intercept=True, 100-iter -> sparsify -> 10000-iter). Canonical SDID unit
# weights go through `compute_sdid_unit_weights` (see `_sc_weight_fw_numpy`
# below and REGISTRY.md SDID section).


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector onto probability simplex (sum to 1, non-negative).

    Uses the algorithm from Duchi et al. (2008).

    Parameters
    ----------
    v : np.ndarray
        Vector to project.

    Returns
    -------
    np.ndarray
        Projected vector on the simplex.
    """
    n = len(v)
    if n == 0:
        return v

    # Sort in descending order
    u = np.sort(v)[::-1]

    # Find the threshold
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / np.arange(1, n + 1))[0]

    if len(rho) == 0:
        # All elements are negative or zero
        rho_val = 0
    else:
        rho_val = rho[-1]

    theta = (cssv[rho_val] - 1) / (rho_val + 1)

    return np.asarray(np.maximum(v - theta, 0))


# =============================================================================
# SDID Weight Optimization (Frank-Wolfe, matching R's synthdid)
# =============================================================================


def _sum_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to sum to 1. Fallback to uniform if sum is zero.

    Matches R's synthdid ``sum_normalize()`` helper.
    """
    s = np.sum(v)
    if s > 0:
        return v / s
    return np.ones(len(v)) / len(v)


def _compute_noise_level(Y_pre_control: np.ndarray) -> float:
    """Compute noise level from first-differences of control outcomes.

    Matches R's ``sd(apply(Y[1:N0, 1:T0], 1, diff))`` which computes
    first-differences across time for each control unit, then takes the
    pooled standard deviation.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control unit pre-treatment outcomes, shape (n_pre, n_control).

    Returns
    -------
    float
        Noise level (standard deviation of first-differences).
    """
    if HAS_RUST_BACKEND:
        return float(_rust_compute_noise_level(np.ascontiguousarray(Y_pre_control)))
    return _compute_noise_level_numpy(Y_pre_control)


def _compute_noise_level_numpy(Y_pre_control: np.ndarray) -> float:
    """Pure NumPy implementation of noise level computation."""
    if Y_pre_control.shape[0] < 2:
        return 0.0
    # R: apply(Y[1:N0, 1:T0], 1, diff) computes diff per row (unit).
    # Our matrix is (T, N) so diff along axis=0 gives (T-1, N).
    first_diffs = np.diff(Y_pre_control, axis=0)  # (T_pre-1, N_co)
    if first_diffs.size <= 1:
        return 0.0
    return float(np.std(first_diffs, ddof=1))


def _compute_regularization(
    Y_pre_control: np.ndarray,
    n_treated: int,
    n_post: int,
) -> tuple:
    """Compute auto-regularization parameters matching R's synthdid.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control unit pre-treatment outcomes, shape (n_pre, n_control).
    n_treated : int
        Number of treated units.
    n_post : int
        Number of post-treatment periods.

    Returns
    -------
    tuple
        (zeta_omega, zeta_lambda) regularization parameters.
    """
    sigma = _compute_noise_level(Y_pre_control)
    eta_omega = (n_treated * n_post) ** 0.25
    eta_lambda = 1e-6
    return eta_omega * sigma, eta_lambda * sigma


def _fw_step(
    A: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Single Frank-Wolfe step on the simplex.

    Matches R's ``fw.step()`` in synthdid's ``sc.weight.fw()``.

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (N, T0).
    x : np.ndarray
        Current weight vector of shape (T0,).
    b : np.ndarray
        Target vector of shape (N,).
    eta : float
        Regularization strength (N * zeta^2).

    Returns
    -------
    np.ndarray
        Updated weight vector on the simplex.
    """
    Ax = A @ x
    half_grad = A.T @ (Ax - b) + eta * x
    i = int(np.argmin(half_grad))
    d_x = -x.copy()
    d_x[i] += 1.0
    if np.allclose(d_x, 0.0):
        return x.copy()
    d_err = A[:, i] - Ax
    denom = d_err @ d_err + eta * (d_x @ d_x)
    if denom <= 0:
        return x.copy()
    step = -(half_grad @ d_x) / denom
    step = float(np.clip(step, 0.0, 1.0))
    return x + step * d_x


def _sc_weight_fw(
    Y: np.ndarray,
    zeta: float,
    intercept: bool = True,
    init_weights: Optional[np.ndarray] = None,
    min_decrease: float = 1e-5,
    max_iter: int = 10000,
    return_convergence: bool = False,
    reg_weights: Optional[np.ndarray] = None,
):
    """Compute synthetic control weights via Frank-Wolfe optimization.

    Matches R's ``sc.weight.fw()`` from the synthdid package. Solves::

        min_{lambda on simplex}  zeta^2 * ||lambda||^2
            + (1/N) * ||A_centered @ lambda - b_centered||^2

    With ``reg_weights`` set, solves the weighted-regularization variant
    used by SDID survey-bootstrap (PR #352)::

        min_{lambda on simplex}  zeta^2 * sum_j reg_weights[j] * lambda[j]^2
            + (1/N) * ||A_centered @ lambda - b_centered||^2

    Parameters
    ----------
    Y : np.ndarray
        Matrix of shape (N, T0+1). Last column is the target (post-period
        mean or treated pre-period mean depending on context).
    zeta : float
        Regularization strength.
    intercept : bool, default True
        If True, column-center Y before optimization.
    init_weights : np.ndarray, optional
        Initial weights. If None, starts with uniform weights.
    min_decrease : float, default 1e-5
        Convergence criterion: stop when objective decreases by less than
        ``min_decrease**2``. R uses ``1e-5 * noise_level``; the caller
        should pass the data-dependent value for best results.
    max_iter : int, default 10000
        Maximum number of iterations. Matches R's default.
    return_convergence : bool, default False
        If True, returns a tuple ``(weights, converged)`` where
        ``converged`` is ``True`` iff the min-decrease criterion fired
        rather than ``max_iter`` being reached. Dispatches to the Rust
        ``sc_weight_fw_with_convergence`` entry point when available, and
        to ``_sc_weight_fw_numpy(return_convergence=True)`` otherwise. Used
        by SDID bootstrap to surface per-draw FW non-convergence
        explicitly instead of relying on ``warnings.catch_warnings`` (the
        default Rust FW entry point is silent on non-convergence).
    reg_weights : np.ndarray, optional
        Per-coordinate regularization weights of shape ``(T0,)``. When
        set, switches to the weighted-regularization Rust kernel
        (``sc_weight_fw_weighted`` / ``_with_convergence``) which solves
        the SDID survey-bootstrap objective with ``ζ²·Σ rw·ω²`` in place
        of the uniform ``ζ²·||ω||²``. The caller is responsible for any
        column-scaling of ``Y`` to match the loss form. Default ``None``
        delegates to the unweighted kernel — preserves the legacy ABI for
        all existing callers.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, bool]
        Weights of shape (T0,) on the simplex; with
        ``return_convergence=True``, additionally the convergence flag.
    """
    Y_c = np.ascontiguousarray(Y, dtype=np.float64)
    init_c = (
        np.ascontiguousarray(init_weights, dtype=np.float64) if init_weights is not None else None
    )
    rw_c = np.ascontiguousarray(reg_weights, dtype=np.float64) if reg_weights is not None else None

    if rw_c is not None:
        # Validate reg_weights shape at the dispatcher so Rust and NumPy
        # backends share a single failure surface. The Rust
        # ``sc_weight_fw_weighted_internal`` silently falls back to the
        # unweighted kernel on a length mismatch, while the NumPy
        # implementation raises — dispatching without a shared upstream
        # check would let callers get the wrong objective on the Rust
        # path with no error (PR #355 R5 P2).
        expected_t0 = Y_c.shape[1] - 1
        if rw_c.shape != (expected_t0,):
            raise ValueError(
                f"reg_weights shape {rw_c.shape} does not match expected "
                f"({expected_t0},) — must equal Y.shape[1] - 1"
            )

    if HAS_RUST_BACKEND:
        if reg_weights is not None:
            if return_convergence:
                weights, converged = _rust_sc_weight_fw_weighted_with_convergence(
                    Y_c,
                    zeta,
                    intercept,
                    init_c,
                    min_decrease,
                    max_iter,
                    rw_c,
                )
                return np.asarray(weights), converged
            return np.asarray(
                _rust_sc_weight_fw_weighted(
                    Y_c,
                    zeta,
                    intercept,
                    init_c,
                    min_decrease,
                    max_iter,
                    rw_c,
                )
            )
        if return_convergence:
            weights, converged = _rust_sc_weight_fw_with_convergence(
                Y_c,
                zeta,
                intercept,
                init_c,
                min_decrease,
                max_iter,
            )
            return np.asarray(weights), converged
        return np.asarray(
            _rust_sc_weight_fw(
                Y_c,
                zeta,
                intercept,
                init_c,
                min_decrease,
                max_iter,
            )
        )
    return _sc_weight_fw_numpy(
        Y,
        zeta,
        intercept,
        init_weights,
        min_decrease,
        max_iter,
        return_convergence=return_convergence,
        reg_weights=reg_weights,
    )


def _sc_weight_fw_numpy(
    Y: np.ndarray,
    zeta: float,
    intercept: bool = True,
    init_weights: Optional[np.ndarray] = None,
    min_decrease: float = 1e-5,
    max_iter: int = 10000,
    return_convergence: bool = False,
    reg_weights: Optional[np.ndarray] = None,
):
    """Pure NumPy implementation of Frank-Wolfe SC weight solver.

    When ``return_convergence=True``, returns a tuple ``(weights, converged)``
    and suppresses the default ``warn_if_not_converged`` side effect — the
    caller is responsible for deciding how to surface non-convergence.

    With ``reg_weights`` set, solves the weighted-regularization variant
    (matches the Rust ``sc_weight_fw_weighted`` kernel; PR #352). The loss
    term is unchanged; only the regularization becomes
    ``ζ²·Σ_j reg_weights[j]·lam[j]²`` and the FW step uses the diag(rw)-
    weighted simplex direction norm.
    """
    T0 = Y.shape[1] - 1
    N = Y.shape[0]

    if T0 <= 0:
        lam_trivial = np.ones(max(T0, 1))
        if return_convergence:
            return lam_trivial, True
        return lam_trivial

    # Column-center if using intercept (matches R's intercept=TRUE default)
    if intercept:
        Y = Y - Y.mean(axis=0)

    A = Y[:, :T0]
    b = Y[:, T0]
    eta = N * zeta**2

    if init_weights is not None:
        lam = init_weights.copy()
    else:
        lam = np.ones(T0) / T0

    if reg_weights is not None:
        rw = np.asarray(reg_weights, dtype=np.float64)
        if rw.shape != (T0,):
            raise ValueError(
                f"reg_weights shape {rw.shape} does not match expected "
                f"({T0},) — must equal A.shape[1]"
            )
    else:
        rw = None

    vals = np.full(max_iter, np.nan)
    converged = False
    for t in range(max_iter):
        if rw is None:
            lam = _fw_step(A, lam, b, eta)
            err = Y @ np.append(lam, -1.0)
            vals[t] = zeta**2 * np.sum(lam**2) + np.sum(err**2) / N
        else:
            # Weighted FW step with diag(rw) regularization. Mirrors the
            # Rust sc_weight_fw_*_weighted derivation in rust/src/weights.rs.
            ax_minus_b = A @ lam - b
            half_grad = A.T @ ax_minus_b + eta * rw * lam
            i = int(np.argmin(half_grad))
            d = -lam.copy()
            d[i] += 1.0
            d_x_w_norm_sq = float(np.sum(rw * d * d))
            if d_x_w_norm_sq < 1e-24:
                err = ax_minus_b
                vals[t] = zeta**2 * float(np.sum(rw * lam * lam)) + float(np.sum(err**2)) / N
                if t >= 1 and vals[t - 1] - vals[t] < min_decrease**2:
                    converged = True
                    break
                continue
            d_err_sq = float(np.sum((A @ d) ** 2))
            denom = d_err_sq + eta * d_x_w_norm_sq
            if denom <= 0.0:
                err = ax_minus_b
                vals[t] = zeta**2 * float(np.sum(rw * lam * lam)) + float(np.sum(err**2)) / N
                if t >= 1 and vals[t - 1] - vals[t] < min_decrease**2:
                    converged = True
                    break
                continue
            hg_dot_dx = float(half_grad @ d)
            step = float(np.clip(-hg_dot_dx / denom, 0.0, 1.0))
            lam = lam + step * d
            err = A @ lam - b
            vals[t] = zeta**2 * float(np.sum(rw * lam * lam)) + float(np.sum(err**2)) / N
        if t >= 1 and vals[t - 1] - vals[t] < min_decrease**2:
            converged = True
            break
    if return_convergence:
        return lam, converged
    warn_if_not_converged(converged, "Frank-Wolfe SC weight solver", max_iter, min_decrease)

    return lam


def _sparsify(v: np.ndarray) -> np.ndarray:
    """Sparsify weight vector by zeroing out small entries.

    Matches R's synthdid ``sparsify_function``:
    ``v[v <= max(v)/4] = 0; v = v / sum(v)``

    Parameters
    ----------
    v : np.ndarray
        Weight vector.

    Returns
    -------
    np.ndarray
        Sparsified weight vector summing to 1.
    """
    v = v.copy()
    max_v = np.max(v)
    if max_v <= 0:
        return np.ones(len(v)) / len(v)
    v[v <= max_v / 4] = 0.0
    return _sum_normalize(v)


def compute_time_weights(
    Y_pre_control: np.ndarray,
    Y_post_control: np.ndarray,
    zeta_lambda: float,
    intercept: bool = True,
    min_decrease: float = 1e-5,
    max_iter_pre_sparsify: int = 100,
    max_iter: int = 10000,
    init_weights: Optional[np.ndarray] = None,
    return_convergence: bool = False,
):
    """Compute SDID time weights via Frank-Wolfe optimization.

    Matches R's ``synthdid::sc.weight.fw(Yc[1:N0, ], zeta=zeta.lambda,
    intercept=TRUE)`` where ``Yc`` is the collapsed-form matrix. Uses
    two-pass optimization with sparsification (same as unit weights),
    matching R's default ``sparsify=sparsify_function``.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control outcomes in pre-treatment periods, shape (n_pre, n_control).
    Y_post_control : np.ndarray
        Control outcomes in post-treatment periods, shape (n_post, n_control).
    zeta_lambda : float
        Regularization parameter for time weights.
    intercept : bool, default True
        If True, column-center the optimization matrix.
    min_decrease : float, default 1e-5
        Convergence criterion for Frank-Wolfe. R uses ``1e-5 * noise_level``.
    max_iter_pre_sparsify : int, default 100
        Iterations for first pass (before sparsification).
    max_iter : int, default 10000
        Maximum iterations for second pass (after sparsification).
        Matches R's default.
    init_weights : np.ndarray, optional
        Warm-start weights for the first Frank-Wolfe pass, shape ``(n_pre,)``.
        If None (default), the solver starts from uniform, matching the
        top-level ``synthdid_estimate(update.lambda=TRUE)`` path. When
        provided, the Rust fast-path is skipped in favor of the Python
        two-pass dispatcher so the first-pass init can be threaded
        through; this matches R's ``synthdid::bootstrap_sample`` shape
        (which passes ``weights$lambda`` as FW init per draw). Used by
        ``SyntheticDiD._bootstrap_se`` on the refit loop.
    return_convergence : bool, default False
        If True, returns a tuple ``(weights, converged)`` where ``converged``
        is the AND of the first-pass and second-pass convergence flags from
        the underlying ``_sc_weight_fw`` calls (True iff the min-decrease
        criterion fired on BOTH passes; False if either hit ``max_iter``).
        Setting this flag also forces the Python two-pass dispatcher even
        when ``init_weights`` is None, because the Rust top-level fast-path
        is silent on non-convergence. Used by SDID bootstrap to surface
        per-draw FW non-convergence explicitly; standalone callers can
        leave this at the default to preserve the legacy ABI.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, bool]
        Time weights of shape (n_pre,) on the simplex. With
        ``return_convergence=True``, additionally the two-pass convergence
        flag (as described above).
    """
    if Y_post_control.shape[0] == 0:
        raise ValueError(
            "Y_post_control has no rows. At least one post-treatment period "
            "is required for time weight computation."
        )

    # When the caller asks for convergence tracking, skip the Rust top-level
    # fast-path even if init_weights is None — that entry point bypasses the
    # Python two-pass dispatcher and is silent on FW non-convergence.
    if HAS_RUST_BACKEND and init_weights is None and not return_convergence:
        return np.asarray(
            _rust_compute_time_weights(
                np.ascontiguousarray(Y_pre_control, dtype=np.float64),
                np.ascontiguousarray(Y_post_control, dtype=np.float64),
                zeta_lambda,
                intercept,
                min_decrease,
                max_iter_pre_sparsify,
                max_iter,
            )
        )

    n_pre = Y_pre_control.shape[0]

    if n_pre <= 1:
        lam_trivial = np.ones(n_pre)
        if return_convergence:
            return lam_trivial, True
        return lam_trivial

    # Build collapsed form: (N_co, T_pre + 1), last col = per-control post mean
    post_means = np.mean(Y_post_control, axis=0)  # (N_co,)
    Y_time = np.column_stack([Y_pre_control.T, post_means])  # (N_co, T_pre+1)

    # First pass: limited iterations (matching R's max.iter.pre.sparsify).
    # init_weights is either None (uniform start) or the caller-supplied
    # warm-start; the inner _sc_weight_fw still dispatches to Rust for the
    # 100-iter run, so we only pay a Python-level dispatch overhead.
    if return_convergence:
        lam, conv1 = _sc_weight_fw(
            Y_time,
            zeta=zeta_lambda,
            intercept=intercept,
            init_weights=init_weights,
            min_decrease=min_decrease,
            max_iter=max_iter_pre_sparsify,
            return_convergence=True,
        )
    else:
        lam = _sc_weight_fw(
            Y_time,
            zeta=zeta_lambda,
            intercept=intercept,
            init_weights=init_weights,
            min_decrease=min_decrease,
            max_iter=max_iter_pre_sparsify,
        )

    # Sparsify: zero out small weights, renormalize (R's sparsify_function)
    lam = _sparsify(lam)

    # Second pass: from sparsified initialization (matching R's max.iter)
    if return_convergence:
        lam, conv2 = _sc_weight_fw(
            Y_time,
            zeta=zeta_lambda,
            intercept=intercept,
            init_weights=lam,
            min_decrease=min_decrease,
            max_iter=max_iter,
            return_convergence=True,
        )
        return lam, bool(conv1 and conv2)

    lam = _sc_weight_fw(
        Y_time,
        zeta=zeta_lambda,
        intercept=intercept,
        init_weights=lam,
        min_decrease=min_decrease,
        max_iter=max_iter,
    )

    return lam


def compute_sdid_unit_weights(
    Y_pre_control: np.ndarray,
    Y_pre_treated_mean: np.ndarray,
    zeta_omega: float,
    intercept: bool = True,
    min_decrease: float = 1e-5,
    max_iter_pre_sparsify: int = 100,
    max_iter: int = 10000,
    init_weights: Optional[np.ndarray] = None,
    return_convergence: bool = False,
):
    """Compute SDID unit weights via Frank-Wolfe with two-pass sparsification.

    Matches R's ``synthdid::sc.weight.fw(t(Yc[, 1:T0]), zeta=zeta.omega,
    intercept=TRUE)`` followed by the sparsify/re-optimize pass.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control outcomes in pre-treatment periods, shape (n_pre, n_control).
    Y_pre_treated_mean : np.ndarray
        Mean treated outcomes in pre-treatment periods, shape (n_pre,).
    zeta_omega : float
        Regularization parameter for unit weights.
    intercept : bool, default True
        If True, column-center the optimization matrix.
    min_decrease : float, default 1e-5
        Convergence criterion for Frank-Wolfe. R uses ``1e-5 * noise_level``.
    max_iter_pre_sparsify : int, default 100
        Iterations for first pass (before sparsification).
    max_iter : int, default 10000
        Iterations for second pass (after sparsification). Matches R's default.
    init_weights : np.ndarray, optional
        Warm-start weights for the first Frank-Wolfe pass, shape
        ``(n_control,)``. If None (default), the solver starts from
        uniform — matching the top-level ``synthdid_estimate(update.omega=TRUE)``
        path. When provided, the Rust fast-path is skipped in favor of the
        Python two-pass dispatcher so the first-pass init can be threaded
        through; this matches R's ``synthdid::bootstrap_sample`` shape
        (which passes ``sum_normalize(weights$omega[...])`` as FW init per
        draw). Used by ``SyntheticDiD._bootstrap_se`` on the refit loop.
    return_convergence : bool, default False
        If True, returns a tuple ``(weights, converged)`` where ``converged``
        is the AND of the first-pass and second-pass convergence flags from
        the underlying ``_sc_weight_fw`` calls (True iff the min-decrease
        criterion fired on BOTH passes; False if either hit ``max_iter``).
        Setting this flag also forces the Python two-pass dispatcher even
        when ``init_weights`` is None, because the Rust top-level fast-path
        is silent on non-convergence. Used by SDID bootstrap to surface
        per-draw FW non-convergence explicitly; standalone callers can
        leave this at the default to preserve the legacy ABI.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, bool]
        Unit weights of shape (n_control,) on the simplex. With
        ``return_convergence=True``, additionally the two-pass convergence
        flag (as described above).
    """
    n_control = Y_pre_control.shape[1]

    if n_control == 0:
        empty = np.asarray([])
        if return_convergence:
            return empty, True
        return empty
    if n_control == 1:
        singleton = np.asarray([1.0])
        if return_convergence:
            return singleton, True
        return singleton

    # When the caller asks for convergence tracking, skip the Rust top-level
    # fast-path even if init_weights is None — that entry point bypasses the
    # Python two-pass dispatcher and is silent on FW non-convergence.
    if HAS_RUST_BACKEND and init_weights is None and not return_convergence:
        return np.asarray(
            _rust_sdid_unit_weights(
                np.ascontiguousarray(Y_pre_control, dtype=np.float64),
                np.ascontiguousarray(Y_pre_treated_mean, dtype=np.float64),
                zeta_omega,
                intercept,
                min_decrease,
                max_iter_pre_sparsify,
                max_iter,
            )
        )

    # Build collapsed form: (T_pre, N_co + 1), last col = treated pre means
    Y_unit = np.column_stack([Y_pre_control, Y_pre_treated_mean.reshape(-1, 1)])

    # First pass: limited iterations. init_weights is either None (uniform
    # start) or the caller-supplied warm-start; the inner _sc_weight_fw
    # still dispatches to Rust for the 100-iter run, so we only pay a
    # Python-level dispatch overhead.
    if return_convergence:
        omega, conv1 = _sc_weight_fw(
            Y_unit,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=init_weights,
            max_iter=max_iter_pre_sparsify,
            min_decrease=min_decrease,
            return_convergence=True,
        )
    else:
        omega = _sc_weight_fw(
            Y_unit,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=init_weights,
            max_iter=max_iter_pre_sparsify,
            min_decrease=min_decrease,
        )

    # Sparsify: zero out weights <= max/4, renormalize
    omega = _sparsify(omega)

    # Second pass: from sparsified initialization
    if return_convergence:
        omega, conv2 = _sc_weight_fw(
            Y_unit,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=omega,
            max_iter=max_iter,
            min_decrease=min_decrease,
            return_convergence=True,
        )
        return omega, bool(conv1 and conv2)

    omega = _sc_weight_fw(
        Y_unit,
        zeta=zeta_omega,
        intercept=intercept,
        init_weights=omega,
        max_iter=max_iter,
        min_decrease=min_decrease,
    )

    return omega


# =============================================================================
# Survey-weighted SDID FW helpers (PR #352 — internal, called from
# SyntheticDiD._bootstrap_se on per-draw survey-weighted refits)
# =============================================================================


def compute_sdid_unit_weights_survey(
    Y_pre_control: np.ndarray,
    Y_pre_treated_mean: np.ndarray,
    rw_control: np.ndarray,
    zeta_omega: float,
    intercept: bool = True,
    min_decrease: float = 1e-5,
    max_iter_pre_sparsify: int = 100,
    max_iter: int = 10000,
    init_weights: Optional[np.ndarray] = None,
    return_convergence: bool = False,
):
    """Survey-weighted SDID unit weights via two-pass weighted Frank-Wolfe.

    Solves the weighted-FW objective (PR #352 §2.2)::

        min_{ω on simplex}
            Σ_t (Σ_i rw_control[i]·ω[i]·Y_pre_control[t,i] - Y_pre_treated_mean[t])²
            + ζ²·Σ_i rw_control[i]·ω[i]²

    Implementation: pre-scales each control column of Y_unit by
    ``rw_control`` (so the loss term picks up the per-control linear
    combination) and passes ``rw_control`` as ``reg_weights`` to
    ``_sc_weight_fw`` (so the regularization picks up the per-ω scaling).
    Two-pass sparsify-refit structure mirrors ``compute_sdid_unit_weights``.

    The returned ω is on the standard simplex. The caller (typically
    ``SyntheticDiD._bootstrap_se``) is responsible for composing
    ``ω_eff = rw_control·ω / Σ(rw_control·ω)`` for the downstream SDID
    estimator, which expects a normalized weight vector.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control outcomes in pre-treatment periods, shape (n_pre, n_control).
    Y_pre_treated_mean : np.ndarray
        Mean treated outcomes in pre-treatment periods, shape (n_pre,).
    rw_control : np.ndarray
        Per-control survey weights, shape (n_control,). Must be non-negative.
        For pweight-only bootstrap this is the constant survey weight per
        control unit; for Rao-Wu bootstrap this is the per-draw rescaled
        weight (``generate_rao_wu_weights`` output sliced to control units).
    zeta_omega : float
        Regularization parameter (already normalized by Y_scale).
    intercept : bool, default True
        Column-center the optimization matrix.
    min_decrease : float, default 1e-5
        Convergence criterion.
    max_iter_pre_sparsify : int, default 100
        First-pass iteration cap before sparsification.
    max_iter : int, default 10000
        Second-pass iteration cap.
    init_weights : np.ndarray, optional
        Warm-start weights for the first pass; shape (n_control,).
    return_convergence : bool, default False
        If True, returns ``(ω, converged)`` where converged is the AND of
        both passes' convergence flags.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, bool]
        ω on the simplex (NOT ω_eff).
    """
    n_control = Y_pre_control.shape[1]

    if rw_control.shape != (n_control,):
        raise ValueError(
            f"rw_control shape {rw_control.shape} does not match expected " f"({n_control},)"
        )

    if n_control == 0:
        empty = np.asarray([])
        return (empty, True) if return_convergence else empty
    if n_control == 1:
        singleton = np.asarray([1.0])
        return (singleton, True) if return_convergence else singleton

    # Build the column-scaled Y matrix: each control column j is multiplied by
    # rw_control[j], so A·ω in the loss equals Σ_j rw_j·ω_j·Y_j,pre.
    rw = np.ascontiguousarray(rw_control, dtype=np.float64)
    Y_scaled = np.column_stack(
        [
            Y_pre_control * rw[np.newaxis, :],
            Y_pre_treated_mean.reshape(-1, 1),
        ]
    )

    if return_convergence:
        omega, conv1 = _sc_weight_fw(
            Y_scaled,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=init_weights,
            max_iter=max_iter_pre_sparsify,
            min_decrease=min_decrease,
            return_convergence=True,
            reg_weights=rw,
        )
    else:
        omega = _sc_weight_fw(
            Y_scaled,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=init_weights,
            max_iter=max_iter_pre_sparsify,
            min_decrease=min_decrease,
            reg_weights=rw,
        )

    omega = _sparsify(omega)

    if return_convergence:
        omega, conv2 = _sc_weight_fw(
            Y_scaled,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=omega,
            max_iter=max_iter,
            min_decrease=min_decrease,
            return_convergence=True,
            reg_weights=rw,
        )
        return omega, bool(conv1 and conv2)

    return _sc_weight_fw(
        Y_scaled,
        zeta=zeta_omega,
        intercept=intercept,
        init_weights=omega,
        max_iter=max_iter,
        min_decrease=min_decrease,
        reg_weights=rw,
    )


def compute_time_weights_survey(
    Y_pre_control: np.ndarray,
    Y_post_control: np.ndarray,
    rw_control: np.ndarray,
    zeta_lambda: float,
    intercept: bool = True,
    min_decrease: float = 1e-5,
    max_iter_pre_sparsify: int = 100,
    max_iter: int = 10000,
    init_weights: Optional[np.ndarray] = None,
    return_convergence: bool = False,
):
    """Survey-weighted SDID time weights via two-pass row-weighted FW.

    Solves the WLS-style time-weight objective (PR #352 §2.2)::

        min_{λ on simplex}
            Σ_u rw_control[u]·(Σ_t λ[t]·Y_u,pre-centered[t] - Y_u,post_mean-centered)²
            + ζ²·||λ||²

    Regularization stays uniform on λ (rw is per-control, λ is per-period —
    no alignment for per-λ reg weighting). The loss term uses WLS-style
    row weights; when ``intercept=True``, the column-centering step is
    *also* survey-weighted (weighted mean across controls, weights
    ``rw_control``) so the centered loss minimizes
    ``Σ_u rw_u·(A_u·λ - b_u)²`` on the rw-centered matrix — equivalent
    to the stated weighted objective. The Rust kernel then sees the
    weighted-centered + sqrt(rw)-row-scaled matrix with
    ``intercept=False`` (no additional unweighted centering).

    The returned λ is on the standard simplex.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Shape (n_pre, n_control).
    Y_post_control : np.ndarray
        Shape (n_post, n_control).
    rw_control : np.ndarray
        Shape (n_control,), non-negative.
    zeta_lambda : float
        Regularization parameter (already normalized by Y_scale).
    Other parameters mirror ``compute_time_weights``.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, bool]
        λ on the simplex.
    """
    n_pre = Y_pre_control.shape[0]
    n_control = Y_pre_control.shape[1]

    if rw_control.shape != (n_control,):
        raise ValueError(
            f"rw_control shape {rw_control.shape} does not match expected " f"({n_control},)"
        )

    if Y_post_control.shape[0] == 0:
        raise ValueError(
            "Y_post_control has no rows. At least one post-treatment period "
            "is required for time weight computation."
        )

    if n_pre <= 1:
        lam_trivial = np.ones(n_pre)
        return (lam_trivial, True) if return_convergence else lam_trivial

    # Build collapsed form like compute_time_weights: (N_co, T_pre+1)
    post_means = np.mean(Y_post_control, axis=0)
    Y_time = np.column_stack([Y_pre_control.T, post_means])  # (N_co, T_pre+1)

    # Column-center the (N_co, T_pre+1) matrix using the SURVEY-WEIGHTED
    # mean across control units when ``intercept=True``. Plain
    # ``intercept=True`` inside the FW kernel would use an unweighted
    # column mean which does not correspond to the stated weighted-loss
    # objective once ``rw_control`` varies. Perform the weighted
    # centering here and pass ``intercept=False`` below so the kernel
    # does not re-center on the row-scaled matrix.
    rw_sum = float(np.sum(rw_control))
    if intercept and rw_sum > 0:
        col_weighted_means = (Y_time * rw_control[:, np.newaxis]).sum(axis=0) / rw_sum
        Y_time = Y_time - col_weighted_means[np.newaxis, :]

    # Row-scale by sqrt(rw): after weighted centering (if any), each
    # control unit's contribution to the loss is weighted by
    # ``rw_control[u]`` via the sqrt(rw) row scaling, which reproduces
    # ``||diag(sqrt(rw))·(A·λ - b)||²`` = ``Σ_u rw_u·(A_u·λ - b_u)²``.
    # Reg on λ stays uniform (no reg_weights).
    sqrt_rw = np.sqrt(np.maximum(rw_control, 0.0))
    Y_weighted = Y_time * sqrt_rw[:, np.newaxis]

    if return_convergence:
        lam, conv1 = _sc_weight_fw(
            Y_weighted,
            zeta=zeta_lambda,
            intercept=False,  # weighted centering already applied above
            init_weights=init_weights,
            min_decrease=min_decrease,
            max_iter=max_iter_pre_sparsify,
            return_convergence=True,
        )
    else:
        lam = _sc_weight_fw(
            Y_weighted,
            zeta=zeta_lambda,
            intercept=False,  # weighted centering already applied above
            init_weights=init_weights,
            min_decrease=min_decrease,
            max_iter=max_iter_pre_sparsify,
        )

    lam = _sparsify(lam)

    if return_convergence:
        lam, conv2 = _sc_weight_fw(
            Y_weighted,
            zeta=zeta_lambda,
            intercept=False,  # weighted centering already applied above
            init_weights=lam,
            min_decrease=min_decrease,
            max_iter=max_iter,
            return_convergence=True,
        )
        return lam, bool(conv1 and conv2)

    return _sc_weight_fw(
        Y_weighted,
        zeta=zeta_lambda,
        intercept=False,  # weighted centering already applied above
        init_weights=lam,
        min_decrease=min_decrease,
        max_iter=max_iter,
    )


def compute_sdid_estimator(
    Y_pre_control: np.ndarray,
    Y_post_control: np.ndarray,
    Y_pre_treated: np.ndarray,
    Y_post_treated: np.ndarray,
    unit_weights: np.ndarray,
    time_weights: np.ndarray,
) -> float:
    """
    Compute the Synthetic DiD estimator.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control outcomes in pre-treatment periods, shape (n_pre, n_control).
    Y_post_control : np.ndarray
        Control outcomes in post-treatment periods, shape (n_post, n_control).
    Y_pre_treated : np.ndarray
        Treated unit outcomes in pre-treatment periods, shape (n_pre,).
    Y_post_treated : np.ndarray
        Treated unit outcomes in post-treatment periods, shape (n_post,).
    unit_weights : np.ndarray
        Weights for control units, shape (n_control,).
    time_weights : np.ndarray
        Weights for pre-treatment periods, shape (n_pre,).

    Returns
    -------
    float
        The synthetic DiD treatment effect estimate.

    Notes
    -----
    The SDID estimator is:

        τ̂ = (Ȳ_treated,post - Σ_t λ_t * Y_treated,t)
            - Σ_j ω_j * (Ȳ_j,post - Σ_t λ_t * Y_j,t)

    Where:
    - ω_j are unit weights
    - λ_t are time weights
    - Ȳ denotes average over post periods
    """
    # Weighted pre-treatment averages
    weighted_pre_control = time_weights @ Y_pre_control  # shape: (n_control,)
    weighted_pre_treated = time_weights @ Y_pre_treated  # scalar

    # Post-treatment averages
    mean_post_control = np.mean(Y_post_control, axis=0)  # shape: (n_control,)
    mean_post_treated = np.mean(Y_post_treated)  # scalar

    # DiD for treated: post - weighted pre
    did_treated = mean_post_treated - weighted_pre_treated

    # Weighted DiD for controls: sum over j of omega_j * (post_j - weighted_pre_j)
    did_control = unit_weights @ (mean_post_control - weighted_pre_control)

    # SDID estimator
    tau = did_treated - did_control

    return float(tau)


def demean_by_group(
    data: pd.DataFrame,
    variables: List[str],
    group_var: str,
    inplace: bool = False,
    suffix: str = "",
    weights: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Demean variables by a grouping variable (one-way within transformation).

    For each variable, computes: x_ig - mean(x_g) where g is the group.
    When weights are provided, uses weighted group means:
    mean_g = sum(w_i * x_i) / sum(w_i) for i in group g.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variables to demean.
    variables : list of str
        Column names to demean.
    group_var : str
        Column name for the grouping variable.
    inplace : bool, default False
        If True, modifies the original columns. If False, leaves original
        columns unchanged (demeaning is still applied to return value).
    suffix : str, default ""
        Suffix to add to demeaned column names (only used when inplace=False
        and you want to keep both original and demeaned columns).
    weights : np.ndarray, optional
        Observation weights for weighted group means.

    Returns
    -------
    data : pd.DataFrame
        DataFrame with demeaned variables.
    n_effects : int
        Raw level count (``nunique - 1``). **NOT a valid degrees-of-freedom
        adjustment** on weighted fits — it counts levels carried only by
        zero-weight rows; for df adjustments use :func:`absorbed_fe_rank`.

    Examples
    --------
    >>> df, n_fe = demean_by_group(df, ['y', 'x1', 'x2'], 'unit')
    >>> # df['y'], df['x1'], df['x2'] are now demeaned by unit
    """
    if not inplace:
        data = data.copy()

    # Count fixed effects (categories - 1 for identification)
    n_effects = data[group_var].nunique() - 1

    if weights is not None:
        # Weighted demeaning: weighted_mean_g = sum(w*x) / sum(w) per group
        groups = data[group_var].values
        w = np.asarray(weights, dtype=np.float64)
        # Cache weight sums per group (invariant across variables)
        w_sum = pd.Series(w).groupby(groups).transform("sum").values
        for var in variables:
            col_name = var if not suffix else f"{var}{suffix}"
            x = data[var].values.astype(np.float64)
            wx = pd.Series(w * x).groupby(groups).transform("sum").values
            # Guard zero-total-weight groups (survey subpopulation / zero-weight
            # domain padding) so those rows pass through unchanged and stay inert
            # in WLS. For positive-weight groups this is bit-identical to wx/w_sum.
            weighted_means = np.divide(
                wx, w_sum, out=np.zeros_like(wx, dtype=np.float64), where=w_sum > 0
            )
            data[col_name] = x - weighted_means
    else:
        # Cache the groupby object for efficiency
        grouper = data.groupby(group_var, sort=False)
        for var in variables:
            col_name = var if not suffix else f"{var}{suffix}"
            group_means = grouper[var].transform("mean")
            data[col_name] = data[var] - group_means

    return data, n_effects


def _demean_map_numpy(
    x_cols: List[np.ndarray],
    codes_list: List[np.ndarray],
    n_groups_list: List[int],
    weights: Optional[np.ndarray],
    tol: float,
    max_iter: int,
) -> Tuple[List[np.ndarray], List[int]]:
    """Canonical numpy MAP engine over pre-factorized group codes.

    The reference implementation for the demeaning contract (python-canonical
    policy): the Rust kernel mirrors this exactly (sweep order, row-order
    bincount accumulation, ``max|x - x_old| < tol`` convergence per column)
    and equivalence tests compare against THIS function directly.

    Returns (demeaned columns, iterations per column; -1 = not converged).
    """
    demeaned: List[np.ndarray] = []
    iters: List[int] = []
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        # Cache per-group weight sums once (invariant across variables/iterations).
        w_sums = [
            np.bincount(codes, weights=w, minlength=n_g)
            for codes, n_g in zip(codes_list, n_groups_list)
        ]
        for x0 in x_cols:
            x = np.asarray(x0, dtype=np.float64).copy()
            it_out = -1
            for _iter in range(max_iter):
                x_old = x.copy()
                for codes, n_g, w_sum in zip(codes_list, n_groups_list, w_sums):
                    wx_sum = np.bincount(codes, weights=w * x, minlength=n_g)
                    # Guard zero-total-weight groups (survey subpopulation /
                    # zero-weight domain padding): leave such rows unchanged
                    # (mean 0) so they remain inert in the downstream WLS
                    # instead of poisoning the design with NaN/Inf.
                    means = np.divide(wx_sum, w_sum, out=np.zeros_like(wx_sum), where=w_sum > 0)
                    x = x - means[codes]
                if np.max(np.abs(x - x_old)) < tol:
                    it_out = _iter + 1
                    break
            demeaned.append(x)
            iters.append(it_out)
    else:
        counts = [
            np.bincount(codes, minlength=n_g).astype(np.float64)
            for codes, n_g in zip(codes_list, n_groups_list)
        ]
        for x0 in x_cols:
            x = np.asarray(x0, dtype=np.float64).copy()
            it_out = -1
            for _iter in range(max_iter):
                x_old = x.copy()
                for codes, n_g, cnt in zip(codes_list, n_groups_list, counts):
                    means = np.bincount(codes, weights=x, minlength=n_g) / cnt
                    x = x - means[codes]
                if np.max(np.abs(x - x_old)) < tol:
                    it_out = _iter + 1
                    break
            demeaned.append(x)
            iters.append(it_out)
    return demeaned, iters


# Opt-in column-block width for Rust demean_map kernel calls (None = single
# dispatch, the default). The kernel holds an owned copy of its input block
# plus the result, and chunking caps those transients at the block width while
# leaving per-column results bit-identical: every column's MAP loop is fully
# independent (no cross-column arithmetic), so partitioning changes neither
# values nor iteration counts. Measured on the 2.4M x 130 firm-panel workload
# (2026-07): dispatch-level transients shrink as designed, but the fit's peak
# RSS is dominated by the downstream solver phase, so end-to-end peak dropped
# only ~5-12% while wall-clock rose ~2-7% - hence OFF by default; the env
# knob remains for memory-constrained runs (a block width near the machine's
# core count measured best: each chunk is one full parallel wave). Peak-RSS
# measurements need repeated runs under matched machine state - macOS memory
# compression deflates single-run ru_maxrss readings under ambient pressure.
_DEMEAN_MAP_CHUNK_COLS: Optional[int] = None


def _resolve_demean_chunk_cols() -> Optional[int]:
    """Opt-in chunk width for the Rust kernel dispatch (None = unchunked).

    ``DIFF_DIFF_DEMEAN_CHUNK_COLS`` (positive integer) enables chunking;
    invalid or non-positive values fall back silently to the module default,
    mirroring ``DIFF_DIFF_BACKEND``'s convention. Unlike the backend switch
    this is read PER CALL, not at import - deliberate, so benchmarks and tests
    can A/B chunked vs unchunked dispatch within one process/build.
    """
    raw = os.environ.get("DIFF_DIFF_DEMEAN_CHUNK_COLS")
    if raw is None:
        return _DEMEAN_MAP_CHUNK_COLS
    try:
        value = int(raw)
    except ValueError:
        return _DEMEAN_MAP_CHUNK_COLS
    return value if value > 0 else _DEMEAN_MAP_CHUNK_COLS


def _demean_map_rust(
    x_cols: List[np.ndarray],
    codes_list: List[np.ndarray],
    n_groups_list: List[int],
    weights: Optional[np.ndarray],
    tol: float,
    max_iter: int,
) -> Optional[Tuple[List[np.ndarray], List[int]]]:
    """Marshal to the Rust ``demean_map`` kernel in column chunks.

    Returns None to signal "use the canonical numpy engine" (kernel absent,
    degenerate shapes, or a deliberate kernel-side validation error). Dtypes
    are coerced explicitly BEFORE the call - never rely on exception handling
    for dtype mismatches.

    Variables are dispatched in blocks of ``_resolve_demean_chunk_cols()``
    columns to bound peak memory (see ``_DEMEAN_MAP_CHUNK_COLS``); the codes
    matrix and weights are built once and shared across chunks. Chunking is
    exact partitioning - per-column outputs and iteration counts are identical
    to a single-call dispatch.
    """
    if _rust_demean_map is None:
        return None
    if not x_cols or x_cols[0].shape[0] == 0:
        return None
    codes_mat = np.ascontiguousarray(np.column_stack(codes_list), dtype=np.int64)
    w = None if weights is None else np.ascontiguousarray(weights, dtype=np.float64)
    n_groups = [int(g) for g in n_groups_list]
    k = len(x_cols)
    chunk = _resolve_demean_chunk_cols()
    if chunk is None:
        chunk = k  # default: single dispatch, no partitioning
    # Balanced partition into ceil(k / chunk) near-equal blocks. A naive
    # fixed-stride split leaves a small remainder block (e.g. 130 columns at
    # width 32 -> 4x32 + 2); that tail runs nearly serial across the rayon
    # pool, and one slow-converging column in it stalls the whole dispatch.
    n_chunks = -(-k // chunk)
    bounds = [round(i * k / n_chunks) for i in range(n_chunks + 1)]
    demeaned: List[np.ndarray] = []
    iters_all: List[int] = []
    for start, stop in zip(bounds, bounds[1:]):
        x_mat = np.ascontiguousarray(np.column_stack(x_cols[start:stop]), dtype=np.float64)
        try:
            out, iters = _rust_demean_map(
                x_mat,
                codes_mat,
                n_groups,
                w,
                float(tol),
                int(max_iter),
            )
        except ValueError as e:
            if "demean_map" in str(e):
                # deliberate kernel-side validation marker -> numpy fallback
                # (kernel validation is chunk-invariant, so this fires on the
                # first chunk; the whole call falls back, same contract as an
                # unchunked dispatch)
                return None
            raise
        # F-order result: per-column views are contiguous, no extra copy
        demeaned.extend(out[:, j] for j in range(out.shape[1]))
        iters_all.extend(int(i) for i in iters)
    return demeaned, iters_all


def absorbed_fe_rank(
    data: pd.DataFrame,
    group_vars: List[str],
    *,
    has_intercept_col: bool,
    weights: Optional[np.ndarray] = None,
) -> int:
    """Degrees of freedom absorbed by a set of fixed-effect dimensions.

    Returns the rank of the absorbed FE dummy space, minus one when the caller's
    design matrix already carries its own intercept column (so the shared
    constant is not counted twice).

    The rank INCLUDING an implicit intercept is::

        N == 2:  sum(levels) - C     C = connected components of the bipartite
                                     level graph (Abowd-Creecy-Kramarz)
        else:    sum(levels) - N + 1

    For a single connected component the two-dimensional form collapses to the
    historical ``sum_d (n_d - 1) + 1``, so every existing caller is unchanged on
    a connected panel. It differs — correctly — in two supported cases the old
    count over-stated:

    - **Disconnected panels** (``C > 1``): the FE are not jointly identified
      across components, so the dummy space is rank ``sum(levels) - C``.
    - **Nested / hierarchical dimensions** (e.g. ``["state", "state_year"]``):
      each parent level forms its own component, so ``C = n_parents`` and the
      absorbed rank is that of the finer dimension alone.

    ``N >= 3`` keeps the ``sum(levels) - N + 1`` form, which is exact for
    mutually independent, connected dimensions and over-states rank otherwise
    (a duplicated third dimension is the simplest counterexample). Computing the
    general N-way rank is a hypergraph problem and is deliberately not attempted
    here; see the TODO backlog row.

    Parameters
    ----------
    data : DataFrame
        The frame the design is built from. Must be pre-transform: after an
        in-place demean the group columns may hold demeaned floats.
    group_vars : list of str
        Absorbed FE columns.
    has_intercept_col : bool
        True when the caller's ``X`` already contains an intercept column.
    weights : ndarray, optional
        Observation weights. Levels and connectivity are evaluated over
        POSITIVE-weight rows only, per the REGISTRY guarantee that zero-weight
        padding is inference-invariant on the generic HC1/classical paths.

    Returns
    -------
    int
        Absorbed degrees of freedom for the caller's df adjustment.
    """
    if not group_vars:
        return 0

    codes_list, levels, _ = _factorize_fe_codes(data, group_vars, weights)
    rank = _fe_rank_from_codes(codes_list, levels)
    if rank == 0:
        return 0
    return rank - 1 if has_intercept_col else rank


def _factorize_fe_codes(
    data: pd.DataFrame,
    group_vars: List[str],
    weights: Optional[np.ndarray],
) -> Tuple[List[np.ndarray], List[int], Optional[np.ndarray]]:
    """Shared factorization for the absorbed-FE helpers.

    Applies the positive-weight row filter (REGISTRY zero-weight-padding
    guarantee), validates against NaN group keys, and returns per-dim
    integer codes with level counts plus the positive-row mask (None when no
    filtering applied) so callers can align companion arrays (cluster ids).
    """
    frame = data
    mask: Optional[np.ndarray] = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != len(data):
            raise ValueError(f"weights length ({w.shape[0]}) must match data rows ({len(data)})")
        positive = w > 0
        if not positive.any():
            return [np.empty(0, dtype=np.intp) for _ in group_vars], [0] * len(group_vars), positive
        if not positive.all():
            frame = data.loc[positive]
            mask = positive

    codes_list = []
    for g in group_vars:
        codes = pd.factorize(frame[g].values, sort=False)[0]
        if codes.size and codes.min() < 0:
            # pd.factorize assigns NaN keys code -1; a negative index would
            # otherwise surface as a cryptic sparse-graph error downstream.
            raise ValueError(
                f"Absorbed fixed-effect column '{g}' contains NaN group keys; "
                "drop or impute those rows before fitting."
            )
        codes_list.append(codes)
    levels = [int(c.max()) + 1 if c.size else 0 for c in codes_list]
    return codes_list, levels, mask


def _fe_rank_from_codes(codes_list: List[np.ndarray], levels: List[int]) -> int:
    """Rank of the FE dummy span INCLUDING the shared constant.

    N == 2: ``sum(levels) - C`` with C the connected components of the
    bipartite level graph (Abowd-Creecy-Kramarz); N == 1: ``levels``;
    N >= 3: ``sum(levels) - N + 1`` (the documented approximate form).
    """
    total_levels = sum(levels)
    if total_levels == 0:
        return 0

    n_dims = len(codes_list)
    if n_dims == 2:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        left, right = codes_list
        offset = levels[0]
        adjacency = coo_matrix(
            (
                np.ones(left.shape[0], dtype=np.int8),
                (left, right + offset),
            ),
            shape=(total_levels, total_levels),
        )
        # Weak connectivity of the directed one-way bipartite graph equals
        # undirected connectivity, and skips materializing A + A.T (~2.4x).
        n_components = int(connected_components(adjacency, directed=True, connection="weak")[0])
        return total_levels - n_components
    return total_levels - n_dims + 1


def _factorize_cluster_codes(
    cluster_ids: np.ndarray,
    n_rows: int,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Factorize the cluster array over the same positive-weight rows."""
    cids = np.asarray(cluster_ids)
    if cids.shape[0] != n_rows:
        raise ValueError(f"cluster_ids length ({cids.shape[0]}) must match data rows ({n_rows})")
    if mask is not None:
        cids = cids[mask]
    ccodes = pd.factorize(cids, sort=False)[0]
    if ccodes.size and ccodes.min() < 0:
        # "missing values" is the established cluster-validation contract
        # phrase (tests match on it); keep it in the message.
        raise ValueError(
            "cluster_ids contain missing values (NaN keys); drop or impute "
            "those rows before fitting."
        )
    return ccodes


def _nested_dims_from_codes(
    group_vars: List[str],
    codes_list: List[np.ndarray],
    levels: List[int],
    ccodes: np.ndarray,
) -> List[str]:
    """Dims whose every level maps to exactly one cluster id (O(n) scatter)."""
    nested = []
    for g, codes, n_levels in zip(group_vars, codes_list, levels):
        if n_levels == 0:
            continue
        lo = np.full(n_levels, np.iinfo(np.int64).max, dtype=np.int64)
        hi = np.full(n_levels, -1, dtype=np.int64)
        np.minimum.at(lo, codes, ccodes)
        np.maximum.at(hi, codes, ccodes)
        present = hi >= 0
        if np.array_equal(lo[present], hi[present]):
            nested.append(g)
    return nested


def cluster_nested_fe_dims(
    data: pd.DataFrame,
    group_vars: List[str],
    cluster_ids: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
) -> List[str]:
    """Absorbed FE dims nested in the cluster partition.

    A dim ``g`` is nested iff every level of ``g`` maps to exactly one
    cluster id — the reghdfe/fixest ``ssc(K.fixef = "nested")`` condition
    under which the dim's parameters are dropped from the clustered CR1
    count. Evaluated over POSITIVE-weight rows only, mirroring
    :func:`absorbed_fe_rank`. A coarser cluster (e.g. unit FE clustered by
    state) is nested; a finer or crossing cluster is not.
    """
    if not group_vars:
        return []
    codes_list, levels, mask = _factorize_fe_codes(data, group_vars, weights)
    if sum(levels) == 0:
        return []
    ccodes = _factorize_cluster_codes(cluster_ids, len(data), mask)
    return _nested_dims_from_codes(group_vars, codes_list, levels, ccodes)


def absorbed_fe_cr1_k_increment(
    data: pd.DataFrame,
    group_vars: List[str],
    cluster_ids: np.ndarray,
    *,
    has_intercept_col: bool,
    weights: Optional[np.ndarray] = None,
) -> int:
    """The clustered-CR1 ``k`` increment for absorbed fixed effects.

    Implements the K_reference bracket (see ``docs/methodology/
    variance-conventions.md`` defect D2)::

        increment = (0 if has_intercept_col else 1)
                  + rank(all absorbed dims, incl. constant)
                  - max(rank(cluster-nested dims, incl. constant), 1)

    The ``+1`` term is the rank of the ABSORBED constant, so it exists only
    when an FE set is absorbed; with no absorbed dims — or an EMPTY effective
    support (all rows zero-weight) — the increment is 0. The ``max(..., 1)``
    floor is load-bearing for the zero-nested-dim case: with no absorbed dim
    nested in the cluster the conditioning set is the shared constant alone
    (rank 1). Verified against Stata reghdfe 3.2.9 (via jwdid, ~1e-15) and
    R fixest 0.14.2 (~1e-12); see the REGISTRY absorbed-FE notes.

    Both rank terms inherit :func:`absorbed_fe_rank`'s dimensionality
    contract: exact (component-aware) for one and two dims; for THREE or
    more dims the documented D3 ``sum(levels) - N + 1`` approximation
    applies — exact for independent connected dims, an over-count for
    duplicated/nested triples (REGISTRY absorbed-FE note; TODO.md N-way
    row), so the increment is approximate on such designs.
    """
    if not group_vars:
        return 0
    codes_list, levels, mask = _factorize_fe_codes(data, group_vars, weights)
    if sum(levels) == 0:
        # Empty effective support: nothing absorbed, increment 0 (guarded so
        # the max(..., 1) floor cannot drive the formula to -1).
        return 0
    ccodes = _factorize_cluster_codes(cluster_ids, len(data), mask)
    nested_names = _nested_dims_from_codes(group_vars, codes_list, levels, ccodes)
    full_rank = _fe_rank_from_codes(codes_list, levels)
    if nested_names:
        nested_idx = [i for i, g in enumerate(group_vars) if g in nested_names]
        nested_rank = _fe_rank_from_codes(
            [codes_list[i] for i in nested_idx], [levels[i] for i in nested_idx]
        )
    else:
        nested_rank = 0
    return (0 if has_intercept_col else 1) + full_rank - max(nested_rank, 1)


def demean_by_groups(
    data: pd.DataFrame,
    variables: List[str],
    group_vars: List[str],
    inplace: bool = False,
    suffix: str = "",
    weights: Optional[np.ndarray] = None,
    max_iter: int = 10_000,
    tol: float = 1e-10,
) -> Tuple[pd.DataFrame, int]:
    """N-way within transformation via the method of alternating projections (MAP).

    Removes ``len(group_vars)`` absorbed fixed-effect dimensions by repeatedly
    demeaning each variable by each group in ``group_vars`` order until the
    iterate stops changing (``max|x - x_old| < tol``) or ``max_iter`` is reached.
    This converges to the exact (W)LS Frisch-Waugh-Lovell residualization onto the
    combined column space of all ``group_vars`` dummies, for both balanced and
    unbalanced panels and for both unweighted and survey-weighted designs (it is
    the algorithm used by R ``fixest`` / ``reghdfe`` / ``lfe``).

    Single-pass sequential demeaning (one sweep) is only the one-iteration
    approximation of this projection; it is exact only when the FE subspaces are
    orthogonal (balanced fully-crossed panels). For ``len(group_vars) == 1`` the
    projection is exact in one pass, so this delegates to :func:`demean_by_group`
    (byte-identical to the prior one-way behavior).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variables to demean.
    variables : list of str
        Column names to demean.
    group_vars : list of str
        Grouping (fixed-effect) columns to absorb. Must be non-empty.
    inplace : bool, default False
        If True, writes the demeaned values onto ``data`` (the caller must own it)
        and returns it. If False, attaches them as a consolidated block via
        ``pd.concat`` and returns a new frame (the input is not mutated).
    suffix : str, default ""
        Demeaned column naming. Empty overwrites the source columns; a non-empty
        suffix writes to ``f"{var}{suffix}"`` columns (originals preserved).
    weights : np.ndarray, optional
        Observation weights. When provided, uses weighted group means
        ``sum(w*x)/sum(w)`` per group and converges to the WLS-FWL residual.
    max_iter : int, default 10_000
        Maximum number of alternating-projection iterations per variable
        (matching the R ``fixest`` ``fixef.iter`` / ``pyfixest``
        ``fixef_maxiter`` defaults; correlated FE incidence can genuinely
        require hundreds of iterations). Emits a single ``UserWarning`` per
        call listing any variable that fails to converge.
    tol : float, default 1e-10
        Convergence tolerance on the max absolute change across the iterate.

    Returns
    -------
    data : pd.DataFrame
        DataFrame with demeaned variables.
    n_effects : int
        Raw level count ``sum_d (nunique_d - 1)`` over ``group_vars``.
        **NOT a valid degrees-of-freedom adjustment**: it over-counts the
        absorbed dummy-space rank on disconnected or nested/hierarchical
        dimensions and counts levels carried only by zero-weight rows. No
        library caller consumes it as df any more — for df adjustments use
        :func:`absorbed_fe_rank`, which computes the component-aware rank
        over positive-weight rows.

    Raises
    ------
    ValueError
        If ``group_vars`` is empty, or if any absorbed group column contains
        NaN (checked for every ``len(group_vars)``, one-way included) —
        pandas groupby silently drops NaN keys and ``pd.factorize`` codes
        them ``-1``, which would silently mis-index the group means, so
        missing group keys must be dropped or imputed by the caller.

    Notes
    -----
    For ``len(group_vars) >= 2`` each dimension is factorized once and the MAP
    sweeps compute group means via ``np.bincount`` (two O(n) passes per
    dimension). Plain bincount summation is not compensated the way pandas'
    grouped mean is, so results agree with a pandas ``groupby`` implementation
    to ~1e-10 order (drift compounds across iterations), not bit-for-bit. See
    ``docs/methodology/REGISTRY.md`` "Absorbed Fixed Effects with Survey
    Weights".

    Examples
    --------
    >>> df, n_fe = demean_by_groups(df, ['y', 'x'], ['unit', 'period'])
    """
    if not group_vars:
        raise ValueError("demean_by_groups requires at least one grouping variable.")

    # NaN group keys are rejected for EVERY N (one-way included): pandas
    # groupby silently drops NaN keys (NaN-poisoning the affected rows
    # unweighted, passing them through un-demeaned weighted) and
    # pd.factorize codes them -1, which would mis-index the last group's
    # mean in the bincount sweeps below.
    for g in group_vars:
        if pd.isna(data[g].values).any():
            raise ValueError(
                f"demean_by_groups: absorbed group column '{g}' contains NaN. "
                "Drop or impute missing group keys before absorbing this "
                "dimension."
            )

    # One dimension: the within projection is exact in a single pass. Delegate so
    # the one-way callers stay byte-identical to demean_by_group (for valid,
    # NaN-free group keys).
    if len(group_vars) == 1:
        return demean_by_group(
            data,
            variables,
            group_vars[0],
            inplace=inplace,
            suffix=suffix,
            weights=weights,
        )

    # N >= 2: method of alternating projections. Per-variable independent
    # convergence (outer loop = variable, inner = iterations), same sweep order
    # over group_vars as within_transform's historical unit-then-time loop.
    #
    # Each absorbed dimension is factorized ONCE; every MAP sweep is then two
    # O(n) passes (np.bincount group-sum + fancy-index gather) with no group
    # re-hashing — pandas groupby.transform rebuilt its hash table on every
    # (iteration x variable x dimension) call. Same precompute pattern as
    # survey._PsuScaffolding. Bincount accumulation is not compensated the way
    # pandas' grouped mean is: results agree with the prior implementation to
    # ~1e-10 order, not bit-for-bit (see REGISTRY "Absorbed Fixed Effects").
    target_cols = [var if not suffix else f"{var}{suffix}" for var in variables]
    codes_list: List[np.ndarray] = []
    n_groups_list: List[int] = []
    for g in group_vars:
        # NaN keys already rejected above, so codes are all >= 0 here.
        codes, uniques = pd.factorize(data[g].values, sort=False)
        codes_list.append(codes.astype(np.intp, copy=False))
        n_groups_list.append(len(uniques))
    n_effects = sum(n_g - 1 for n_g in n_groups_list)

    x_cols = [data[var].values for var in variables]
    result = None
    if HAS_RUST_BACKEND and _rust_demean_map is not None:
        # Rust kernel: identical sweep order, accumulation order, and
        # convergence criterion (rayon-parallel across columns). None means
        # "use the canonical numpy engine".
        result = _demean_map_rust(x_cols, codes_list, n_groups_list, weights, tol, max_iter)
    if result is None:
        result = _demean_map_numpy(x_cols, codes_list, n_groups_list, weights, tol, max_iter)
    demeaned_values, _iters = result
    non_converged_vars = [v for v, it in zip(variables, _iters) if it < 0]

    if non_converged_vars:
        warn_if_not_converged(
            False,
            f"demean_by_groups alternating projection (variables: {non_converged_vars})",
            max_iter,
            tol,
        )

    if inplace:
        for col, vals in zip(target_cols, demeaned_values):
            data[col] = vals
        return data, n_effects

    # Non-inplace: attach the demeaned columns as a single consolidated block via
    # pd.concat (no defensive copy — the demean above is read-only). Mirrors the
    # within_transform attach contract: a colliding target name (suffix="" or a
    # re-demean) is dropped first so concat replaces rather than duplicates.
    new_block = pd.DataFrame(dict(zip(target_cols, demeaned_values)), index=data.index)
    collisions = [c for c in target_cols if c in data.columns]
    if collisions:
        data = data.drop(columns=collisions)
    return pd.concat([data, new_block], axis=1), n_effects


def _iterative_fe_solve(
    y: np.ndarray,
    unit_codes: np.ndarray,
    time_codes: np.ndarray,
    n_units: int,
    n_times: int,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    *,
    method_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-way Gauss-Seidel FE solver on integer-coded factors via bincount.

    Estimates ``y = alpha_i + beta_t + u`` by alternating projections
    (beta from ``y - alpha``, alpha from ``y - beta_new``), the same
    recursion as the historical per-estimator pandas loops in ImputationDiD
    and TwoStageDiD but with each factor coded once and each sweep two O(n)
    ``np.bincount`` passes (same engine family as ``_demean_map_numpy``;
    same accumulation-order caveat vs pandas' compensated grouped mean, see
    REGISTRY "Absorbed Fixed Effects"). SpilloverDiD's
    ``_iterative_fe_subset`` is a thin wrapper over this function (its
    Omega_0-specific gates stay local; the recursion lives here).

    Zero-weight convention (per the SpilloverDiD REGISTRY contract, whose
    ``_iterative_fe_subset`` wrapper routes through here): rows with ``weights == 0`` are
    outside the WLS estimating sample, so any unit/period whose rows ALL
    carry zero weight has no identifying contribution and surfaces as
    ``NaN`` FE — never a silent finite ``0.0``. The iteration itself runs
    on the positive-weight subset, which also keeps a zero-total-weight
    group from NaN-poisoning the convergence check (the historical pandas
    loops divided 0/0 there and could never converge).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Outcome values (float64).
    unit_codes, time_codes : ndarray of shape (n,)
        Non-negative integer factor codes (``pd.factorize`` output).
    n_units, n_times : int
        Number of factor levels (lengths of the factorize uniques).
    weights : ndarray of shape (n,), optional
        Survey weights; weighted group means ``sum(w*x)/sum(w)``.
    max_iter, tol : convergence budget; warns via ``warn_if_not_converged``
        (labelled ``method_name``) when exhausted.
    method_name : str
        Caller label for the non-convergence warning (keyword-only).

    Returns
    -------
    unit_fe_arr : ndarray of shape (n_units,)
        Unit FE indexed by code; NaN for codes absent from the
        (positive-weight) estimating sample.
    time_fe_arr : ndarray of shape (n_times,)
        Time FE indexed by code; NaN likewise.
    """
    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64)
        keep = w_arr > 0
        y_sub = y[keep]
        unit_sub = unit_codes[keep]
        time_sub = time_codes[keep]
        w_sub: Optional[np.ndarray] = w_arr[keep]
    else:
        y_sub = y
        unit_sub = unit_codes
        time_sub = time_codes
        w_sub = None

    # Per-group denominators are invariant across iterations.
    if w_sub is None:
        unit_denoms = np.bincount(unit_sub, minlength=n_units).astype(np.float64)
        time_denoms = np.bincount(time_sub, minlength=n_times).astype(np.float64)
    else:
        unit_denoms = np.bincount(unit_sub, weights=w_sub, minlength=n_units)
        time_denoms = np.bincount(time_sub, weights=w_sub, minlength=n_times)
    unit_seen = unit_denoms > 0
    time_seen = time_denoms > 0
    unit_safe = np.maximum(unit_denoms, 1e-300)
    time_safe = np.maximum(time_denoms, 1e-300)

    alpha = np.zeros(n_units)
    beta = np.zeros(n_times)
    converged = False
    for _ in range(max_iter):
        resid = y_sub - alpha[unit_sub]
        wx = resid if w_sub is None else w_sub * resid
        beta_new = np.where(
            time_seen, np.bincount(time_sub, weights=wx, minlength=n_times) / time_safe, 0.0
        )

        resid = y_sub - beta_new[time_sub]
        wx = resid if w_sub is None else w_sub * resid
        alpha_new = np.where(
            unit_seen, np.bincount(unit_sub, weights=wx, minlength=n_units) / unit_safe, 0.0
        )

        max_change = max(
            float(np.max(np.abs(alpha_new - alpha))),
            float(np.max(np.abs(beta_new - beta))),
        )
        alpha = alpha_new
        beta = beta_new
        if max_change < tol:
            converged = True
            break
    warn_if_not_converged(converged, method_name, max_iter, tol)

    unit_fe_arr = np.where(unit_seen, alpha, np.nan)
    time_fe_arr = np.where(time_seen, beta, np.nan)
    return unit_fe_arr, time_fe_arr


def pre_demean_norms(
    data: pd.DataFrame,
    regressors: List[str],
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """L2 norms of regressor columns BEFORE demeaning, for
    :func:`snap_absorbed_regressors` (capture before an in-place demean
    overwrites the source values).

    When ``weights`` is given, norms are ``||sqrt(w) * x||`` — the same
    effective-sample scaling ``solve_ols`` applies in WLS, so the snap
    decision is made on the sample the solver actually sees (zero-weight
    domain rows, which the weighted demean leaves inert by design, must not
    mask a regressor that is FE-spanned on the positive-weight sample).
    """
    sw = None if weights is None else np.sqrt(np.asarray(weights, dtype=np.float64))
    out: Dict[str, float] = {}
    for v in regressors:
        x = data[v].to_numpy(dtype=np.float64)
        out[v] = float(np.linalg.norm(x if sw is None else sw * x))
    return out


def _fe_span_residual_norm(
    x_eff: np.ndarray,
    group_codes: List[np.ndarray],
    n_groups: List[int],
    sqrt_w: Optional[np.ndarray],
) -> float:
    """Exact residual norm of an (effective-sample) column projected onto the
    absorbed-FE span, via sparse LSMR.

    The MAP demean's per-iteration stopping rule bounds the last STEP, not the
    distance to the limit, so a truncated iterate of an exactly-spanned column
    can carry a structured residual orders of magnitude above the convergence
    tolerance (slow-convergence regimes: unbalanced, correlated FE incidence).
    A Krylov solve on the sparse FE incidence decides span membership exactly
    (up to fp), and because ``x_eff`` is the ALREADY-DEMEANED column (nearly
    orthogonal to the span), LSMR converges in a handful of iterations.

    Returns the achieved residual norm ``||x_eff - A @ sol||`` — an upper
    bound on the true projection residual, so an unconverged LSMR errs toward
    KEEPING the column (the pre-guard status quo), never toward over-snapping.
    """
    from scipy.sparse import csr_matrix, hstack
    from scipy.sparse.linalg import lsmr

    n = x_eff.shape[0]
    rows = np.arange(n)
    ones = np.ones(n) if sqrt_w is None else sqrt_w
    blocks = [
        csr_matrix((ones, (rows, codes)), shape=(n, n_g))
        for codes, n_g in zip(group_codes, n_groups)
    ]
    a_mat = hstack(blocks, format="csr")
    sol = lsmr(a_mat, x_eff, atol=1e-13, btol=1e-13)[0]
    return float(np.linalg.norm(x_eff - a_mat @ sol))


def snap_absorbed_regressors(
    demeaned: pd.DataFrame,
    regressors: List[str],
    pre_norms: Dict[str, float],
    absorbed_desc: str,
    group_vars: List[str],
    rank_deficient_action: str = "warn",
    suffix: str = "",
    display_names: Optional[Dict[str, str]] = None,
    rel_tol: float = 1e-10,
    weights: Optional[np.ndarray] = None,
    screen_tol: float = 1e-3,
    stacklevel: int = 3,
) -> List[str]:
    """Zero out regressors that were absorbed (spanned) by the fixed effects.

    A regressor lying exactly in the span of the absorbed FE dummies demeans
    to numerical junk rather than exact zero. Such a column must not reach the
    solver: column equilibration re-inflates it to unit norm, it passes the
    rank check as linearly independent, and its arbitrary direction perturbs
    the OTHER coefficients (measured up to ~3e-3 on ATT with a garbage
    ~1e14-scale coefficient reported for the spanned column itself). Snapping
    it to exact zero makes the downstream rank-deficiency handling drop it
    deterministically (coefficient NaN) — the documented contract for
    FE-collinear regressors. See ``docs/methodology/REGISTRY.md`` "Absorbed
    Fixed Effects".

    Detection is TWO-STAGE, because MAP truncation bounds the last iteration
    step, not the distance to the limit — a spanned column in a
    slow-convergence regime (unbalanced, correlated FE incidence) can stop
    with a structured residual far above ``rel_tol``:

    1. fast path: relative demeaned norm ``<= rel_tol`` snaps immediately
       (covers exactly-converged spanned columns);
    2. candidates with relative norm in ``(rel_tol, screen_tol]`` get an
       exact span-membership confirmation via sparse LSMR on the FE incidence
       (:func:`_fe_span_residual_norm`) and snap iff the true projection
       residual is ``<= rel_tol`` — genuinely identified low-within-variation
       regressors keep their (real) residual and are left untouched.

    Parameters
    ----------
    demeaned : pd.DataFrame
        Frame holding the demeaned columns AND the ``group_vars`` columns
        (modified in place).
    regressors : list of str
        SOURCE names of the regressors (never the outcome — an outcome spanned
        by the FEs is a zero-residual-variance situation, not a collinearity).
    pre_norms : dict
        Pre-demean L2 norm per source name (:func:`pre_demean_norms`).
    absorbed_desc : str
        Human description of the absorbed dimensions, for the warning.
    group_vars : list of str
        The absorbed FE columns (present in ``demeaned``), needed to build
        the sparse incidence for the stage-2 confirmation.
    rank_deficient_action : str, default "warn"
        The owning estimator's contract: ``"warn"`` emits the cause-specific
        ``UserWarning`` here; ``"silent"`` and ``"error"`` defer entirely to
        the downstream rank machinery (silent drop / raise respectively).
    suffix : str, default ""
        Demeaned column naming (``f"{var}{suffix}"``).
    display_names : dict, optional
        Source name -> user-facing name for the warning message.
    rel_tol : float, default 1e-10
        Snap threshold on the (confirmed) relative projection residual.
    weights : np.ndarray, optional
        WLS observation weights. When given, BOTH norms must be
        ``sqrt(w)``-weighted (pass the same weights to
        :func:`pre_demean_norms`) so the snap decision matches the effective
        sample ``solve_ols`` sees — zero-weight rows, which the weighted
        demean leaves inert by design, must not mask an FE-spanned regressor
        on the positive-weight sample.
    screen_tol : float, default 1e-3
        Stage-2 screening bound. Spanned columns whose MAP truncation
        residual exceeds this are outside the guard — but at that point the
        demeaning of every other column is equally unconverged and the
        non-convergence warning contract applies.

    Returns
    -------
    list of str
        Source names of the snapped regressors (empty if none).
    """
    sw = None if weights is None else np.sqrt(np.asarray(weights, dtype=np.float64))
    snapped: List[str] = []
    span_cache: Optional[Tuple[List[np.ndarray], List[int]]] = None
    for var in regressors:
        pre = pre_norms.get(var, 0.0)
        if pre <= 0.0:
            continue  # all-zero input column: rank handling covers it as-is
        col = f"{var}{suffix}" if suffix else var
        vals = demeaned[col].to_numpy(dtype=np.float64)
        eff = vals if sw is None else sw * vals
        eff_norm = float(np.linalg.norm(eff))
        if eff_norm <= rel_tol * pre:
            demeaned[col] = np.zeros(len(vals), dtype=np.float64)
            snapped.append(var)
            continue
        if eff_norm <= screen_tol * pre:
            # Stage 2: MAP truncation may mask an exactly-spanned column;
            # confirm with an exact projection on the FE incidence.
            if span_cache is None:
                codes_l: List[np.ndarray] = []
                sizes_l: List[int] = []
                for g in group_vars:
                    codes_g, uniques_g = pd.factorize(demeaned[g].values, sort=False)
                    codes_l.append(codes_g.astype(np.intp, copy=False))
                    sizes_l.append(len(uniques_g))
                span_cache = (codes_l, sizes_l)
            resid = _fe_span_residual_norm(eff, span_cache[0], span_cache[1], sw)
            if resid <= rel_tol * pre:
                demeaned[col] = np.zeros(len(vals), dtype=np.float64)
                snapped.append(var)
    if snapped and rank_deficient_action == "warn":
        shown = [display_names.get(v, v) if display_names else v for v in snapped]
        warnings.warn(
            f"Regressor(s) {shown} are collinear with the absorbed fixed "
            f"effects ({absorbed_desc}): their within-transformed values are "
            f"numerically zero (relative projection residual <= {rel_tol:g}), "
            "so their coefficients are not identified and will be reported "
            "as NaN.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return snapped


def within_transform(
    data: pd.DataFrame,
    variables: List[str],
    unit: str,
    time: str,
    inplace: bool = False,
    suffix: str = "_demeaned",
    weights: Optional[np.ndarray] = None,
    max_iter: int = 10_000,
    tol: float = 1e-8,
) -> pd.DataFrame:
    """
    Apply two-way within transformation to remove unit and time fixed effects.

    Computed via the method of alternating projections (a thin two-way wrapper
    over :func:`demean_by_groups` with ``group_vars=[unit, time]``): each variable
    is demeaned by unit, then by time, iterated until convergence. This is the
    exact (weighted) Frisch-Waugh-Lovell residualization for both balanced and
    unbalanced panels. The closed-form additive expression ``y_it - y_i. - y_.t +
    y_..`` is the balanced fully-crossed special case (it equals the converged
    iterate only when the unit and time FE subspaces are orthogonal). When weights
    are provided, weighted group means are used at each step.

    This is the standard fixed effects transformation for panel data that
    removes both unit-specific and time-specific effects.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing the variables to transform.
    variables : list of str
        Column names to transform.
    unit : str
        Column name for unit identifier.
    time : str
        Column name for time period identifier.
    inplace : bool, default False
        Controls how the demeaned columns are attached. If False (default), they
        are concatenated onto the input as a single block and a new frame is
        returned; the input frame is not mutated (no defensive deep copy is taken
        — the demean is read-only and ``concat`` does not mutate its inputs). If
        True, they are written onto the passed frame in place (the caller must own
        it) and that frame is returned. Independent of ``suffix``.
    suffix : str, default "_demeaned"
        Column naming, independent of ``inplace``. A non-empty suffix writes the
        demeaned values to new ``f"{var}{suffix}"`` columns (originals preserved);
        ``suffix=""`` overwrites the source columns. Assigning to an existing
        column name overwrites it rather than appending a duplicate label.
    weights : np.ndarray, optional
        Observation weights for weighted group means.
    max_iter : int, default 10_000
        Maximum number of alternating-projection iterations. Applies to BOTH the
        weighted and unweighted paths (both now iterate via the method of
        alternating projections, exact for unbalanced panels). Emits a
        ``UserWarning`` per call when any variable fails to converge within this
        budget. Balanced panels converge in ~2 iterations; correlated FE
        incidence (contiguous unit lifetimes) can require hundreds. The default
        matches ``fixest``'s ``fixef.iter`` / ``pyfixest``'s ``fixef_maxiter``.
    tol : float, default 1e-8
        Convergence tolerance on the max absolute change across the iterate.

    Returns
    -------
    pd.DataFrame
        DataFrame with within-transformed variables.

    Notes
    -----
    The within transformation removes variation that is constant within units
    (unit fixed effects) and constant within time periods (time fixed effects).
    The resulting estimates are equivalent to including unit and time dummies
    but is computationally more efficient for large panels.

    Examples
    --------
    >>> df = within_transform(df, ['y', 'x'], 'unit_id', 'year')
    >>> # df now has 'y_demeaned' and 'x_demeaned' columns
    """
    # Two-way (unit + time) within transformation is the N-way method of
    # alternating projections specialized to two FE dimensions. Delegate to the
    # shared engine so there is one MAP implementation (same per-variable
    # convergence, unit-then-time sweep order; agrees with the historical
    # pandas-groupby loop to ~1e-10 order — see the engine's Notes). ``tol`` is
    # forwarded explicitly (within_transform's default is 1e-8, not
    # demean_by_groups' 1e-10) to preserve the historical convergence budget.
    return demean_by_groups(
        data,
        variables,
        [unit, time],
        inplace=inplace,
        suffix=suffix,
        weights=weights,
        max_iter=max_iter,
        tol=tol,
    )[0]
