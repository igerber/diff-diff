"""
Staggered Difference-in-Differences estimators.

Implements modern methods for DiD with variation in treatment timing,
including the Callaway-Sant'Anna (2021) estimator.
"""

import bisect
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff._base import BaseEstimator
from diff_diff.aggregation import (
    AggregationKit,
)
from diff_diff.linalg import (
    _check_propensity_diagnostics,
    _detect_rank_deficiency,
    _equilibrated_lstsq,
    _format_dropped_columns,
    _rank_guarded_inv,
    solve_logit,
    solve_ols,
)
from diff_diff.staggered_aggregation import (
    CallawaySantAnnaAggregationMixin,
)
from diff_diff.staggered_bootstrap import (
    CallawaySantAnnaBootstrapMixin,
    CSBootstrapResults,
)

# Import from split modules
from diff_diff.staggered_results import (
    CallawaySantAnnaResults,
    GroupTimeEffect,
)
from diff_diff.utils import safe_inference, safe_inference_batch

if TYPE_CHECKING:
    from diff_diff.survey import SurveyDesign

# Re-export for backward compatibility
__all__ = [
    "CallawaySantAnna",
    "CallawaySantAnnaResults",
    "CSBootstrapResults",
    "GroupTimeEffect",
]

# Type alias for pre-computed structures
PrecomputedData = Dict[str, Any]


class _DeprecatedFitArg:
    """Sentinel for `fit(aggregate=)` / `fit(balance_e=)` (rows M-020 / M-117).

    A plain ``None`` default cannot distinguish "not passed" from "passed
    None", so a bare ``None`` default would fire the FutureWarning on EVERY
    fit. The warning must fire only when the caller actually supplies the
    argument.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<not supplied>"


_DEPRECATED_FIT_ARG = _DeprecatedFitArg()


def _linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    rank_deficient_action: str = "warn",
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit OLS regression.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features). Intercept added automatically.
    y : np.ndarray
        Outcome variable.
    rank_deficient_action : str, default "warn"
        Action when design matrix is rank-deficient:
        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning
    weights : np.ndarray, optional
        Observation weights for WLS. When None, OLS is used.

    Returns
    -------
    beta : np.ndarray
        Fitted coefficients (including intercept).
    residuals : np.ndarray
        Residuals from the fit.
    """
    n = X.shape[0]
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])

    # Use unified OLS backend (no vcov needed)
    beta, residuals, _ = solve_ols(
        X_with_intercept,
        y,
        return_vcov=False,
        rank_deficient_action=rank_deficient_action,
        weights=weights,
    )

    return beta, residuals


def _cluster_robust_se_from_per_gt_if(
    inf_info: Dict[str, Any],
    resolved_survey: "Any",
) -> Optional[float]:
    """CR1 Liang-Zeger cluster-robust SE for a single (g,t) ATT.

    Builds the per-(g,t) per-index IF vector from ``inf_info`` and routes
    through ``compute_survey_if_variance`` so that the per-cell variance
    inherits the SAME design-based machinery as the aggregate path:

        V = sum_h (1 - f_h) * (n_h / (n_h - 1)) * sum_j (psi_hj - psi_h_bar)^2

    where ``psi_hj = sum_{i in PSU j, stratum h} psi_i``. This matches
    the documented CR1 contract in REGISTRY.md (synthesized
    ``SurveyDesign(psu=cluster)`` → ``_compute_stratified_psu_meat``)
    and applies the G/(G-1) finite-sample correction, PSU centering,
    FPC, and lonely-PSU handling uniformly with overall / event-study
    inference.

    For the panel path, ``resolved_survey`` is ``resolved_survey_unit``
    (length n_units) and the IF index space is per-unit. For the RCS
    path, ``resolved_survey`` is the per-obs ``resolved_survey`` (length
    n_obs). The helper is index-space agnostic — it just requires
    ``treated_idx`` / ``control_idx`` in ``inf_info`` to be valid
    offsets into ``resolved_survey.psu``.

    Return contract (callers depend on this distinction):

    * **float SE** — finite cluster-robust variance; caller uses it.
    * **NaN** — ``compute_survey_if_variance`` returned NaN (clustered
      variance unidentified, e.g., G<2 or lonely-PSU removed all strata).
      Caller MUST propagate this NaN through to ``safe_inference`` so
      the per-cell inference surface (se / t_stat / p_value / conf_int)
      is NaN-consistent — NEVER fall back to the unit-level SE. Falling
      back would silently report a different estimator's variance under
      a clustered request (``feedback_no_silent_failures``).
    * **None** — malformed inputs or invariant violations:
      ``inf_info`` lacks required IF fields, ``resolved_survey.psu`` is
      None, index alignment cannot be verified, or
      ``compute_survey_if_variance`` returned a negative variance. In
      these cases the helper cannot evaluate the contract; caller falls
      back to the unit-level SE returned by the underlying estimation
      method (no PSU is in play, so unit-level is the documented default).
    """
    if (
        inf_info is None
        or "treated_inf" not in inf_info
        or "control_inf" not in inf_info
        or "treated_idx" not in inf_info
        or "control_idx" not in inf_info
    ):
        return None
    treated_idx = np.asarray(inf_info["treated_idx"])
    control_idx = np.asarray(inf_info["control_idx"])
    treated_inf = np.asarray(inf_info["treated_inf"])
    control_inf = np.asarray(inf_info["control_inf"])
    psu_array = getattr(resolved_survey, "psu", None)
    if psu_array is None:
        return None
    n = len(psu_array)
    if (
        treated_idx.size > 0
        and (treated_idx.max(initial=-1) >= n or treated_idx.min(initial=0) < 0)
    ) or (
        control_idx.size > 0
        and (control_idx.max(initial=-1) >= n or control_idx.min(initial=0) < 0)
    ):
        return None
    # Index arrays are unique within each cell by construction at every
    # producer (np.where on disjoint masks), so fancy += is exact — same
    # scatter contract as staggered_aggregation._combined_if_fast, without
    # np.add.at's unbuffered-ufunc overhead (runs once per (g,t) cell when
    # cluster= is set).
    psi_per_index = np.zeros(n)
    if treated_idx.size:
        psi_per_index[treated_idx] += treated_inf
    if control_idx.size:
        psi_per_index[control_idx] += control_inf
    # Route through the shared survey helper so the per-cell variance
    # gets the same G/(G-1) finite-sample correction, PSU centering,
    # FPC handling, and lonely-PSU/G<2→NaN behavior as overall +
    # event-study inference (per the documented CR1 contract).
    from diff_diff.survey import compute_survey_if_variance

    var = compute_survey_if_variance(psi_per_index, resolved_survey)
    # Return contract:
    #   float SE → use it (finite cluster-robust variance)
    #   NaN     → propagate NaN so the caller can NaN-out the inference
    #             surface rather than silently falling back to the
    #             unit-level SE (per feedback_no_silent_failures: when
    #             clustered variance is undefined — e.g., G<2, lonely-PSU
    #             removed all strata — the user-facing per-cell SE must
    #             reflect that, not silently revert to a different
    #             estimator).
    #   None    → malformed (negative variance or other invariant
    #             violation); caller falls back to the unit-level SE.
    if np.isnan(var):
        return float("nan")
    if var < 0:
        return None
    return float(np.sqrt(var))


def _safe_inv(
    A: np.ndarray,
    tracker: Optional[list] = None,
) -> np.ndarray:
    """Rank-guarded generalized inverse of a Gram matrix for analytical SE paths.

    Parameters
    ----------
    A : np.ndarray
        Square matrix to invert.
    tracker : list, optional
        When provided, one condition-number sample of ``A`` is appended each
        time ``A`` is rank-deficient (near-singular). ``CallawaySantAnna.fit()``
        initializes a list and emits a single aggregate `UserWarning` after the
        fit finishes, rather than surfacing a separate warning per fallback.
        Sibling of finding #17 in the Phase 2 silent-failures audit.

    Notes
    -----
    Delegates to :func:`~diff_diff.linalg._rank_guarded_inv`, which is the sole
    owner of the ``tracker`` append. The old ``except LinAlgError: lstsq``
    fallback only caught *exactly* singular matrices; a *near*-singular Gram
    (e.g. a constant/collinear covariate) returned a garbage inverse (~1e13)
    that flowed into the SE. The rank-guarded inverse truncates redundant
    directions (finite SE on the identified subset) and returns an all-NaN
    matrix only on true rank-0.
    """
    inv, _, _ = _rank_guarded_inv(A, tracker=tracker)
    return inv


def _nan_gt_entry(
    n_treated: int = 0,
    n_control: int = 0,
    skip_reason: Optional[str] = None,
    survey_weight_sum: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a materialized NaN group-time entry for a non-estimable (g, t) cell.

    Non-estimable cells (missing base/post period, zero treated/control, zero
    survey-weight mass, or a non-finite regression solve) are stored as a NaN
    entry in ``group_time_effects`` rather than omitted, so the (g, t) grid is
    inspectable (``to_dataframe`` / direct dict access) and the reason is
    machine-readable via ``skip_reason`` (one of ``"missing_period"``,
    ``"zero_treated_control"``, ``"zero_weight_mass"``,
    ``"non_finite_regression"``; estimable cells carry ``None``).

    The cell carries NO ``influence_func_info`` entry: every aggregation and
    bootstrap consumer finite-masks (``np.isfinite(effect)``) or filters to IF
    members before use, so the NaN cell contributes nothing to any aggregate or
    SE — aggregates stay numerically identical to the prior omit behavior, which
    matches R ``did``'s ``aggte()``. See REGISTRY.md "CallawaySantAnna" edge
    cases for the documented contract.
    """
    entry: Dict[str, Any] = {
        "effect": np.nan,
        "se": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
        "conf_int": (np.nan, np.nan),
        "n_treated": int(n_treated),
        "n_control": int(n_control),
        "skip_reason": skip_reason,
    }
    if survey_weight_sum is not None:
        entry["survey_weight_sum"] = survey_weight_sum
    return entry


class CallawaySantAnna(
    CallawaySantAnnaBootstrapMixin,
    CallawaySantAnnaAggregationMixin,
    BaseEstimator,
):
    """
    Callaway-Sant'Anna (2021) estimator for staggered Difference-in-Differences.

    This estimator handles DiD designs with variation in treatment timing
    (staggered adoption) and heterogeneous treatment effects. It avoids the
    bias of traditional two-way fixed effects (TWFE) estimators by:

    1. Computing group-time average treatment effects ATT(g,t) for each
       cohort g (units first treated in period g) and time t.
    2. Aggregating these to summary measures (overall ATT, event study, etc.)
       using appropriate weights.

    Parameters
    ----------
    control_group : str, default="never_treated"
        Which units to use as controls:
        - "never_treated": Use only never-treated units (recommended)
        - "not_yet_treated": Use never-treated and not-yet-treated units
    anticipation : int, default=0
        Number of periods before treatment where effects may occur.
        Set to > 0 if treatment effects can begin before the official
        treatment date.
    estimation_method : str, default="dr"
        Estimation method:
        - "dr": Doubly robust (recommended)
        - "ipw": Inverse probability weighting
        - "reg": Outcome regression
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional
        Column name for cluster-robust standard errors. When set, the
        influence-function aggregator clusters at the named level via a
        synthesized ``SurveyDesign(psu=cluster_col)`` threaded through the
        existing PSU-meat machinery (``_compute_stratified_psu_meat``) and
        PSU-level multiplier bootstrap. When ``None`` (default), the
        aggregator uses per-unit IF variance (Williams 2000 form). When
        ``survey_design=SurveyDesign(psu=...)`` is also provided, the
        explicit PSU takes precedence; a ``UserWarning`` fires if the bare
        ``cluster=`` partition differs from the explicit PSU partition.
    vcov_type : str, default="hc1"
        Variance family. CallawaySantAnna accepts ``{"hc1"}`` only —
        ``hc1`` means per-unit IF variance when ``cluster=None`` and CR1
        Liang-Zeger on the IF when ``cluster=X`` is set. The
        analytical-sandwich families (``classical``, ``hc2``, ``hc2_bm``)
        and spatial-HAC (``conley``) are rejected at ``__init__`` because
        CS's per-(g,t) doubly-robust / IPW / outcome-regression structure
        has no single design matrix to compute hat-matrix leverage or
        Bell-McCaffrey Satterthwaite DOF on. See REGISTRY.md "IF-based
        variance estimators vs analytical-sandwich estimators" for the
        structural taxonomy.
    n_bootstrap : int, default=0
        Number of bootstrap iterations for inference.
        If 0, uses analytical standard errors.
        Recommended: 999 or more for reliable inference.

        .. note:: Memory Usage
            Bootstrap multiplier weights are generated and consumed one
            draw-block at a time (see :mod:`diff_diff.bootstrap_chunking`), so the
            full ``(n_bootstrap, n_units)`` weight matrix is never materialized.
            The live weight intermediate is bounded by roughly
            ``max(~256 MB, 8 * n_units)`` bytes -- a block holds at least one full
            draw row -- independent of ``n_bootstrap``. Only the small bootstrap
            *output* arrays (``(n_bootstrap, n_group_time)`` and ``(n_bootstrap,)``
            per aggregation) stay fully in memory. Stratified survey designs are
            the current exception (the full PSU-weight matrix is built up front,
            but PSUs are few).

    bootstrap_weights : str, default="rademacher"
        Type of weights for multiplier bootstrap:
        - "rademacher": +1/-1 with equal probability (standard choice)
        - "mammen": Two-point distribution (asymptotically valid, matches skewness)
        - "webb": Six-point distribution (recommended when n_clusters < 20)
    seed : int, optional
        Random seed for reproducibility.
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient (linearly dependent columns):

        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning
    base_period : str, default="varying"
        Method for selecting the base (reference) period for computing
        ATT(g,t). Base periods are selected *positionally* (by the nearest
        observed period in the sorted panel), matching R ``did::att_gt`` -- so
        on gapped (non-consecutive) grids the base is the nearest observed
        period, not literal ``t-1`` / ``g-1``. The pre/post split is on the
        current period vs the cohort (``t < g`` -> pre), independent of
        anticipation; anticipation only shifts the post/universal base. Options:

        - "varying": pre-treatment (``t < g``) uses the immediately-preceding
          observed period as base; post-treatment uses the last observed
          pre-treatment period (largest observed ``p`` with
          ``p + anticipation < g``).
        - "universal": always uses that last observed pre-treatment period as
          base.

        On consecutive grids these reduce to ``t-1`` / ``g-1-anticipation``.
        Both produce identical post-treatment effects. Matches R's
        ``did::att_gt()`` on gapped panels (base selection, estimable ATT/SE
        cells, the ``"universal"`` zero reference cells, and all aggregations).
        See :func:`_select_base_period`.
    cband : bool, default=True
        Whether to compute simultaneous confidence bands (sup-t) for
        event study aggregation. Requires ``n_bootstrap > 0``.
        When True, results include ``cband_crit_value`` and per-event-time
        ``cband_conf_int`` entries controlling family-wise error rate.
    pscore_trim : float, default=0.01
        Trimming bound for propensity scores. Scores are clipped to
        ``[pscore_trim, 1 - pscore_trim]`` before weight computation
        in IPW and DR estimation. Must be in ``(0, 0.5)``.
    panel : bool, default=True
        Whether the data is a balanced/unbalanced panel (units observed
        across multiple time periods). Set to ``False`` for stationary
        repeated cross-sections where each observation has a unique unit
        ID and units do not repeat across periods. Requires that the
        cross-sectional samples are drawn from the same population in
        each period (stationarity). Uses cross-sectional DRDID
        (Sant'Anna & Zhao 2020, Section 4) with per-observation influence
        functions.
    allow_unbalanced_panel : bool, default=False
        When ``True`` and the input panel is unbalanced (some units are not
        observed in every period), route the pooled observations through the
        repeated-cross-section levels estimator (matching R
        ``did::att_gt(allow_unbalanced_panel=TRUE)`` / ``DRDID::reg_did_rc``)
        instead of within-cell panel differencing, and cluster the influence
        function by unit for the standard error. **Inert on a balanced panel**
        (results are byte-identical to the default). When ``False`` (default)
        an unbalanced panel is handled by within-cell differencing and a
        ``UserWarning`` is emitted. ATT matches R bit-for-bit; the SE matches
        up to the documented CR1 ``sqrt(G/(G-1))`` finite-sample factor.
        ``survey_design=`` combined with this flag raises ``NotImplementedError``.
    epv_threshold : float, default=10
        Events Per Variable threshold for propensity score logit.
        When the ratio of minority-class observations to predictor
        variables (excluding intercept) falls below this value, a
        warning is emitted (or ``ValueError`` raised if
        ``rank_deficient_action="error"``). Based on Peduzzi et al.
        (1996). Only applies to IPW and DR estimation methods.
        Use ``diagnose_propensity()`` for a pre-estimation check across
        all cohorts.
    pscore_fallback : str, default="error"
        Action when propensity score estimation fails entirely
        (``LinAlgError`` or ``ValueError`` from IRLS):

        - "error": Raise the exception (default). Ensures the user is
          aware of estimation failures.
        - "unconditional": Fall back to unconditional propensity
          with a warning. For IPW, this drops all covariates. For DR,
          the propensity model becomes unconditional but outcome
          regression still uses covariates.

        When ``rank_deficient_action="error"``, errors are always
        re-raised regardless of this setting.

    Attributes
    ----------
    results_ : CallawaySantAnnaResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> from diff_diff import CallawaySantAnna
    >>>
    >>> # Panel data with staggered treatment
    >>> # 'first_treat' = period when unit was first treated (0 if never treated)
    >>> data = pd.DataFrame({
    ...     'unit': [...],
    ...     'time': [...],
    ...     'outcome': [...],
    ...     'first_treat': [...]  # 0 for never-treated, else first treatment period
    ... })
    >>>
    >>> cs = CallawaySantAnna()
    >>> results = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat')
    >>>
    >>> results.print_summary()

    With event study aggregation (post-fit - no refit required):

    >>> cs = CallawaySantAnna()
    >>> results = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat')
    >>> event_study = results.aggregate('event_study')
    >>> event_study.to_dataframe()  # doctest: +SKIP

    Plotting and the sensitivity analyses (``plot_event_study``,
    ``compute_honest_did``, ``compute_pretrends_power``) still read the
    fit-time surface, so they take a fit that requested aggregation:

    >>> from diff_diff import plot_event_study
    >>> plotted = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat',
    ...                  aggregate='event_study')  # doctest: +SKIP
    >>> plot_event_study(plotted)  # doctest: +SKIP

    With covariate adjustment (conditional parallel trends):

    >>> # When parallel trends only holds conditional on covariates
    >>> cs = CallawaySantAnna(estimation_method='dr')  # doubly robust
    >>> results = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat',
    ...                  covariates=['age', 'income'])
    >>>
    >>> # DR is recommended: consistent if either outcome model
    >>> # or propensity model is correctly specified

    Notes
    -----
    The key innovation of Callaway & Sant'Anna (2021) is the disaggregated
    approach: instead of estimating a single treatment effect, they estimate
    ATT(g,t) for each cohort-time pair. This avoids the "forbidden comparison"
    problem where already-treated units act as controls.

    The ATT(g,t) is identified under parallel trends conditional on covariates:

        E[Y(0)_t - Y(0)_g-1 | G=g] = E[Y(0)_t - Y(0)_g-1 | C=1]

    where G=g indicates treatment cohort g and C=1 indicates control units.
    This uses g-1 as the base period, which applies to post-treatment (t >= g).
    With base_period="varying" (default), pre-treatment uses the immediately-
    preceding observed period as base for the consecutive comparisons useful in
    parallel trends diagnostics. Base periods are selected positionally (nearest
    observed period), matching R did::att_gt on gapped grids (see
    ``_select_base_period``).

    References
    ----------
    Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences with
    multiple time periods. Journal of Econometrics, 225(2), 200-230.
    """

    def __init__(
        self,
        control_group: str = "never_treated",
        anticipation: int = 0,
        estimation_method: str = "dr",
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        bootstrap_weights: Optional[str] = None,
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        base_period: str = "varying",
        cband: bool = True,
        pscore_trim: float = 0.01,
        panel: bool = True,
        allow_unbalanced_panel: bool = False,
        epv_threshold: float = 10,
        pscore_fallback: str = "error",
        vcov_type: str = "hc1",
    ):

        if control_group not in ["never_treated", "not_yet_treated"]:
            raise ValueError(
                f"control_group must be 'never_treated' or 'not_yet_treated', "
                f"got '{control_group}'"
            )
        if estimation_method not in ["dr", "ipw", "reg"]:
            raise ValueError(
                f"estimation_method must be 'dr', 'ipw', or 'reg', " f"got '{estimation_method}'"
            )
        if not (0 < pscore_trim < 0.5):
            raise ValueError(f"pscore_trim must be in (0, 0.5), got {pscore_trim}")
        if epv_threshold <= 0:
            raise ValueError(f"epv_threshold must be > 0, got {epv_threshold}")
        if pscore_fallback not in ["error", "unconditional"]:
            raise ValueError(
                f"pscore_fallback must be 'error' or 'unconditional', " f"got '{pscore_fallback}'"
            )

        # Default to rademacher if not specified
        if bootstrap_weights is None:
            bootstrap_weights = "rademacher"

        if bootstrap_weights not in ["rademacher", "mammen", "webb"]:
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{bootstrap_weights}'"
            )

        if rank_deficient_action not in ["warn", "error", "silent"]:
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )

        if base_period not in ["varying", "universal"]:
            raise ValueError(
                f"base_period must be 'varying' or 'universal', " f"got '{base_period}'"
            )

        # vcov_type input contract: CallawaySantAnna is permanently narrow
        # to {"hc1"} because the analytical-sandwich families (classical,
        # hc2, hc2_bm) require a single regression's hat matrix that CS's
        # per-(g,t) doubly-robust / IPW / outcome-regression structure
        # doesn't have. See REGISTRY.md "IF-based variance estimators vs
        # analytical-sandwich estimators" for the structural taxonomy.
        # Factored out so fit() can re-run it: set_params now validates
        # eagerly via the BaseEstimator probe re-init, but DIRECT attribute
        # mutation (est.vcov_type = ...) still bypasses validation until fit.
        self._validate_vcov_type(vcov_type)

        self.control_group = control_group
        self.anticipation = anticipation
        self.estimation_method = estimation_method
        self.alpha = alpha
        self.cluster = cluster
        self.vcov_type = vcov_type
        # Track whether vcov_type was explicitly set (for future symmetry
        # with SA / StackedDiD / WooldridgeDiD set_params patterns; the
        # narrow contract makes the flag a no-op today but consistency
        # avoids surprises if the contract ever broadens).
        self._vcov_type_explicit = vcov_type != "hc1"
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.base_period = base_period

        self.cband = cband
        self.pscore_trim = pscore_trim
        self.panel = panel
        # When True AND the input panel is unbalanced (some units unobserved in
        # some periods), route through the repeated-cross-section (RC) levels
        # estimator on the pooled observations — matching R
        # `did::att_gt(allow_unbalanced_panel=TRUE)` (which sets panel=FALSE ->
        # DRDID::reg_did_rc). Inert on a balanced panel (the default within-cell
        # differencing path is byte-identical). See fit() for the routing.
        self.allow_unbalanced_panel = allow_unbalanced_panel
        self.epv_threshold = epv_threshold
        self.pscore_fallback = pscore_fallback

        self.is_fitted_ = False
        self.results_: Optional[CallawaySantAnnaResults] = None

    def diagnose_propensity(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Check Events Per Variable (EPV) across all cohorts without estimation.

        Examines the data to identify cohorts where propensity score logit may
        be unreliable due to too few events per covariate. Based on Peduzzi
        et al. (1996).

        This is a raw-count heuristic: it uses total cohort/control unit
        counts without filtering for missing outcomes, zero survey weights,
        or period-specific validity. The actual fit-time EPV (stored in
        ``results.epv_diagnostics``) may be lower because ``fit()`` operates
        on the valid base/post outcome pair and the positive-weight effective
        sample. Use this method as a quick pre-check; rely on
        ``results.epv_diagnostics`` for authoritative per-cell EPV.

        Parameters
        ----------
        df, outcome, unit, time, first_treat, covariates
            Same arguments as ``fit()``.

        Returns
        -------
        pd.DataFrame
            Per-cohort EPV diagnostics with columns: group, n_treated,
            n_control, n_covariates, n_params, epv, status.
        """
        if not self.panel:
            raise NotImplementedError(
                "diagnose_propensity() is not yet supported for repeated "
                "cross-section data (panel=False). Use fit() with covariates "
                "and check results.epv_diagnostics instead."
            )
        if self.control_group == "not_yet_treated":
            raise NotImplementedError(
                "diagnose_propensity() is not yet supported for "
                "control_group='not_yet_treated' because the control set "
                "varies per (g, t) cell. Use fit() with covariates and "
                "check results.epv_diagnostics instead."
            )
        if self.estimation_method == "reg":
            return pd.DataFrame(
                columns=[
                    "group",
                    "n_treated",
                    "n_control",
                    "n_covariates",
                    "n_params",
                    "epv",
                    "status",
                ]
            )
        if not covariates:
            return pd.DataFrame(
                columns=[
                    "group",
                    "n_treated",
                    "n_control",
                    "n_covariates",
                    "n_params",
                    "epv",
                    "status",
                ]
            )

        # Normalize np.inf → 0 for never-treated encoding (same as fit())
        df = df.copy()
        _inf_mask_diag = df[first_treat].isin([np.inf, float("inf")])
        if _inf_mask_diag.any():
            n_inf_units = df.loc[_inf_mask_diag, unit].nunique()
            warnings.warn(
                f"{n_inf_units} unit(s) have first_treat=inf; recoding to 0 "
                f"(never-treated). Use first_treat=0 to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
        df[first_treat] = df[first_treat].replace([np.inf, float("inf")], 0)

        # Compute time_periods and treatment_groups (same logic as fit())
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0])
        precomputed = self._precompute_structures(
            df,
            outcome,
            unit,
            time,
            first_treat,
            covariates,
            time_periods=time_periods,
            treatment_groups=treatment_groups,
        )
        cohort_masks = precomputed["cohort_masks"]
        never_treated_mask = precomputed["never_treated_mask"]
        unit_cohorts = precomputed["unit_cohorts"]
        n_covariates = len(covariates)
        n_params = n_covariates  # predictor count, excluding intercept (Peduzzi convention)

        rows = []
        for g in sorted(cohort_masks.keys()):
            treated_mask = cohort_masks[g]
            if self.control_group == "never_treated":
                control_mask = never_treated_mask
            else:
                base_period_val = g - 1 - self.anticipation
                nyt_threshold = base_period_val + self.anticipation
                control_mask = never_treated_mask | (
                    (unit_cohorts > nyt_threshold) & (unit_cohorts != g)
                )

            n_treated = int(np.sum(treated_mask))
            n_control = int(np.sum(control_mask))
            n_events = min(n_treated, n_control)
            epv = n_events / n_params if n_params > 0 else float("inf")

            if epv >= self.epv_threshold:
                status = "ok"
            elif epv >= 2:
                status = "low"
            else:
                status = "critical"

            rows.append(
                {
                    "group": g,
                    "n_treated": n_treated,
                    "n_control": n_control,
                    "n_covariates": n_covariates,
                    "n_params": n_params,
                    "epv": round(epv, 1),
                    "status": status,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _collapse_survey_to_unit_level(resolved_survey, df, unit_col, all_units):
        """Create unit-level ResolvedSurveyDesign for panel IF-based variance.

        Survey design columns are constant within units (validated upstream).
        This extracts one row per unit, aligned to ``all_units`` ordering.
        """
        from diff_diff.survey import collapse_survey_to_unit_level

        return collapse_survey_to_unit_level(resolved_survey, df, unit_col, all_units)

    def _precompute_structures(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        time_periods: List[Any],
        treatment_groups: List[Any],
        resolved_survey=None,
    ) -> PrecomputedData:
        """
        Pre-compute data structures for efficient ATT(g,t) computation.

        This pivots data to wide format and pre-computes:
        - Outcome matrix (units x time periods)
        - Covariate matrix (units x covariates) from base period
        - Unit cohort membership masks
        - Control unit masks

        Returns
        -------
        PrecomputedData
            Dictionary with pre-computed structures.
        """
        # Get unique units and their cohort assignments
        unit_info = df.groupby(unit)[first_treat].first()
        all_units = unit_info.index.values
        unit_cohorts = unit_info.values

        # Create unit index mapping for fast lookups
        unit_to_idx = {u: i for i, u in enumerate(all_units)}

        # Pivot outcome to wide format: rows = units, columns = time periods
        outcome_wide = df.pivot(index=unit, columns=time, values=outcome)
        # Reindex to ensure all units are present (handles unbalanced panels)
        outcome_wide = outcome_wide.reindex(all_units)
        outcome_matrix = outcome_wide.values  # Shape: (n_units, n_periods)
        period_to_col = {t: i for i, t in enumerate(outcome_wide.columns)}

        # Pre-compute cohort masks (boolean arrays)
        cohort_masks = {}
        for g in treatment_groups:
            cohort_masks[g] = unit_cohorts == g

        # Never-treated mask
        # np.inf was normalized to 0 in fit(), so the np.inf check is defensive only
        never_treated_mask = (unit_cohorts == 0) | (unit_cohorts == np.inf)

        # Pre-compute covariate matrices by time period if needed
        # (covariates are retrieved from the base period of each comparison)
        covariate_by_period = None
        if covariates:
            covariate_by_period = {}
            for t in time_periods:
                period_data = df[df[time] == t].set_index(unit)
                period_cov = period_data.reindex(all_units)[covariates]
                covariate_by_period[t] = period_cov.values  # Shape: (n_units, n_covariates)

        is_balanced = not np.any(np.isnan(outcome_matrix))

        # Extract per-unit survey weights (one weight per unit)
        if resolved_survey is not None:
            sw_by_unit = (
                pd.Series(resolved_survey.weights, index=df.index).groupby(df[unit]).first()
            )
            survey_weights_arr = sw_by_unit.reindex(all_units).values
        else:
            survey_weights_arr = None

        resolved_survey_unit = (
            self._collapse_survey_to_unit_level(resolved_survey, df, unit, all_units)
            if resolved_survey is not None
            else None
        )

        return {
            "all_units": all_units,
            "unit_to_idx": unit_to_idx,
            "unit_cohorts": unit_cohorts,
            "outcome_matrix": outcome_matrix,
            "period_to_col": period_to_col,
            "observed_sorted": sorted(period_to_col),
            "cohort_masks": cohort_masks,
            "never_treated_mask": never_treated_mask,
            "covariate_by_period": covariate_by_period,
            "time_periods": time_periods,
            "is_balanced": is_balanced,
            "is_panel": True,
            "canonical_size": len(all_units),
            "survey_weights": survey_weights_arr,
            "resolved_survey": resolved_survey,
            "resolved_survey_unit": resolved_survey_unit,
            "df_survey": (
                resolved_survey_unit.df_survey if resolved_survey_unit is not None else None
            ),
        }

    def _select_base_period(self, g: Any, t: Any, observed_sorted: List) -> Optional[Any]:
        """Select the base period for cell ``(g, t)``, matching R ``did::att_gt``.

        R selects the base period by *position* in the sorted list of observed
        periods, not by literal calendar arithmetic: on gapped (non-consecutive)
        period grids the base is the nearest observed period, not ``t-1`` /
        ``g-1``. This reproduces ``did`` 2.5.1 ``compute.att_gt`` exactly and is
        byte-identical to the old ``t-1`` / ``g-1-anticipation`` rule on
        consecutive grids (where positional == calendar).

        Parameters
        ----------
        g, t : cohort and evaluation period (calendar values).
        observed_sorted : ascending list of the unique observed periods.

        Returns
        -------
        The base period value, or ``None`` when no valid earlier observed
        period exists (a non-estimable cell, materialized by callers as a
        ``missing_period`` NaN; R cannot estimate it either).

        Notes
        -----
        Following R ``compute.att_gt``, the pre/post split is on the current
        period vs the cohort (``t < g`` -> pre), **independent of
        anticipation**; anticipation only enters the post/universal base.
        - ``universal``, or post-treatment (``t >= g``): base is the last
          pre-treatment observed period, i.e. the largest observed ``p`` with
          ``p + anticipation < g``.
        - ``varying`` pre-treatment (``t < g``): base is the immediately
          preceding observed period, i.e. the largest observed ``p < t``.
        """
        if self.base_period == "universal" or t >= g:
            threshold = g - self.anticipation
        else:  # varying pre-treatment
            threshold = t
        # Largest observed period strictly below `threshold` (positional
        # neighbor). bisect_left gives the insertion index of `threshold`; the
        # element just before it is the largest observed value < threshold.
        idx = bisect.bisect_left(observed_sorted, threshold)
        return observed_sorted[idx - 1] if idx > 0 else None

    def _valid_periods_for_group(self, g: Any, time_periods: List, observed_sorted: List) -> List:
        """Evaluation periods ``t`` to attempt as ``ATT(g, t)`` for cohort ``g``.

        Centralizes the per-group period filter used by every estimation path
        (single source of truth, so the positional-base contract cannot drift):

        - ``universal``: all observed periods except the (positional) base
          reference period (which is the trivial ``ATT = 0`` cell); the base is
          the last observed pre-treatment period, matching R -- NOT literal
          ``g-1-anticipation`` (which on gapped grids would leave the real base
          in and materialize a fake zero cell).
        - ``varying``: all periods except the earliest observed one (which can
          never be a current period -- it has no earlier base).

        Cells that remain non-estimable (``_select_base_period`` returns
        ``None``) are pruned downstream as ``missing_period`` NaNs.
        """
        if self.base_period == "universal":
            universal_base = self._select_base_period(g, g, observed_sorted)
            return [t for t in time_periods if t != universal_base]
        min_period = observed_sorted[0]
        return [t for t in time_periods if t >= g - self.anticipation or t > min_period]

    def _compute_att_gt_fast(
        self,
        precomputed: PrecomputedData,
        g: Any,
        t: Any,
        covariates: Optional[List[str]],
        pscore_cache: Optional[Dict] = None,
        epv_diagnostics: Optional[Dict] = None,
    ) -> Tuple[
        Optional[float], float, int, int, Optional[Dict[str, Any]], Optional[float], Optional[str]
    ]:
        """
        Compute ATT(g,t) using pre-computed data structures (fast version).

        Uses vectorized numpy operations on pre-pivoted outcome matrix
        instead of repeated pandas filtering.

        Returns
        -------
        att_gt : float or None
        se_gt : float
        n_treated : int
        n_control : int
        inf_func_info : dict or None
        survey_weight_sum : float or None
            Sum of survey weights for treated units (for aggregation weighting).
        skip_reason : str or None
            When ``att_gt is None`` (non-estimable cell), the machine-readable
            reason (``"missing_period"`` / ``"zero_treated_control"`` /
            ``"zero_weight_mass"``) so the caller can materialize a NaN cell with
            a ``skip_reason``. ``None`` on a successful return.
        """
        period_to_col = precomputed["period_to_col"]
        outcome_matrix = precomputed["outcome_matrix"]
        cohort_masks = precomputed["cohort_masks"]
        never_treated_mask = precomputed["never_treated_mask"]
        unit_cohorts = precomputed["unit_cohorts"]
        covariate_by_period = precomputed["covariate_by_period"]

        # Base period selection: positional (sorted-index) neighbor, matching
        # R did::att_gt (see _select_base_period). Returns None for a
        # non-estimable cell (no earlier observed period).
        base_period_val = self._select_base_period(g, t, precomputed["observed_sorted"])

        if base_period_val is None or t not in period_to_col:
            return None, 0.0, 0, 0, None, None, "missing_period"

        base_col = period_to_col[base_period_val]
        post_col = period_to_col[t]

        # Get treated units mask (cohort g)
        treated_mask = cohort_masks[g]

        # Get control units mask
        if self.control_group == "never_treated":
            control_mask = never_treated_mask
        else:  # not_yet_treated
            # Not yet treated at BOTH time t and the base period:
            # Controls must be untreated at whichever is later, otherwise
            # their outcome at the base period is contaminated by treatment.
            nyt_threshold = max(t, base_period_val) + self.anticipation
            control_mask = never_treated_mask | (
                (unit_cohorts > nyt_threshold) & (unit_cohorts != g)
            )

        # Extract outcomes for base and post periods
        y_base = outcome_matrix[:, base_col]
        y_post = outcome_matrix[:, post_col]

        # Compute outcome changes (vectorized)
        outcome_change = y_post - y_base

        # Filter to units with valid data (no NaN in either period)
        valid_mask = ~(np.isnan(y_base) | np.isnan(y_post))

        # Get treated and control with valid data
        treated_valid = treated_mask & valid_mask
        control_valid = control_mask & valid_mask

        n_treated = np.sum(treated_valid)
        n_control = np.sum(control_valid)

        if n_treated == 0 or n_control == 0:
            return None, 0.0, int(n_treated), int(n_control), None, None, "zero_treated_control"

        # Extract outcome changes for treated and control
        treated_change = outcome_change[treated_valid]
        control_change = outcome_change[control_valid]

        # Extract survey weights for treated and control
        survey_w = precomputed.get("survey_weights")
        sw_treated = survey_w[treated_valid] if survey_w is not None else None
        sw_control = survey_w[control_valid] if survey_w is not None else None

        # Guard against zero effective mass after subpopulation filtering
        if sw_treated is not None and np.sum(sw_treated) <= 0:
            return None, 0.0, int(n_treated), int(n_control), None, None, "zero_weight_mass"
        if sw_control is not None and np.sum(sw_control) <= 0:
            return None, 0.0, int(n_treated), int(n_control), None, None, "zero_weight_mass"

        # Get covariates if specified (from the base period)
        X_treated = None
        X_control = None
        if covariates and covariate_by_period is not None:
            cov_matrix = covariate_by_period[base_period_val]
            X_treated = cov_matrix[treated_valid]
            X_control = cov_matrix[control_valid]

            # Check for missing values
            if np.any(np.isnan(X_treated)) or np.any(np.isnan(X_control)):
                warnings.warn(
                    f"Missing values in covariates for group {g}, time {t}. "
                    "Falling back to unconditional estimation.",
                    UserWarning,
                    stacklevel=3,
                )
                X_treated = None
                X_control = None

        # Compute cache key for propensity score reuse
        pscore_key = None
        if pscore_cache is not None and X_treated is not None:
            is_balanced = precomputed.get("is_balanced", False)
            if is_balanced and self.control_group == "never_treated":
                pscore_key = (g, base_period_val)
            else:
                pscore_key = (g, base_period_val, t)

        # Estimation method
        if self.estimation_method == "reg":
            att_gt, se_gt, inf_func = self._outcome_regression(
                treated_change,
                control_change,
                X_treated,
                X_control,
                sw_treated=sw_treated,
                sw_control=sw_control,
            )
        elif self.estimation_method == "ipw":
            sw_all = np.concatenate([sw_treated, sw_control]) if sw_treated is not None else None
            epv_diag: dict = {}
            att_gt, se_gt, inf_func = self._ipw_estimation(
                treated_change,
                control_change,
                int(n_treated),
                int(n_control),
                X_treated,
                X_control,
                pscore_cache=pscore_cache,
                pscore_key=pscore_key,
                sw_treated=sw_treated,
                sw_control=sw_control,
                sw_all=sw_all,
                context_label=f"cohort g={g}",
                epv_diagnostics_out=epv_diag,
            )
            if epv_diagnostics is not None and epv_diag:
                epv_diagnostics[(g, t)] = epv_diag
        else:  # doubly robust
            sw_all = np.concatenate([sw_treated, sw_control]) if sw_treated is not None else None
            epv_diag = {}
            att_gt, se_gt, inf_func = self._doubly_robust(
                treated_change,
                control_change,
                X_treated,
                X_control,
                pscore_cache=pscore_cache,
                pscore_key=pscore_key,
                sw_treated=sw_treated,
                sw_control=sw_control,
                sw_all=sw_all,
                context_label=f"cohort g={g}",
                epv_diagnostics_out=epv_diag,
            )
            if epv_diagnostics is not None and epv_diag:
                epv_diagnostics[(g, t)] = epv_diag

        # Package influence function info with index arrays (positions into
        # precomputed['all_units']) for O(1) downstream lookups instead of
        # O(n) Python dict lookups. Unit-LABEL arrays are deliberately NOT
        # materialized (labels are always recoverable as all_units[idx]);
        # only the precomputed-None fallbacks of the combined-IF assembly
        # and the multiplier bootstrap ever consumed them — no in-package
        # caller reaches those — so dropping them cuts two O(n_control)
        # allocations per (g,t) cell.
        # INVARIANT: treated_idx/control_idx are np.where over boolean masks,
        # so each is duplicate-free - the aggregation fast path relies on this
        # to scatter with fancy `psi[idx] += vals` (see
        # staggered_aggregation._combined_if_fast).
        n_t = int(n_treated)
        treated_positions = np.where(treated_valid)[0]
        control_positions = np.where(control_valid)[0]
        inf_func_info = {
            "treated_idx": treated_positions,
            "control_idx": control_positions,
            "treated_inf": inf_func[:n_t],
            "control_inf": inf_func[n_t:],
        }

        sw_sum = float(np.sum(sw_treated)) if sw_treated is not None else None
        return att_gt, se_gt, int(n_treated), int(n_control), inf_func_info, sw_sum, None

    def _compute_all_att_gt_vectorized(
        self,
        precomputed: PrecomputedData,
        treatment_groups: List[Any],
        time_periods: List[Any],
        min_period: Any,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Vectorized computation of all ATT(g,t) for the no-covariates regression case.

        This inlines the simple difference-in-means path from _outcome_regression()
        and eliminates per-(g,t) Python function call overhead.

        Returns
        -------
        group_time_effects : dict
            Mapping (g, t) -> effect dict.
        influence_func_info : dict
            Mapping (g, t) -> influence function info dict.
        """
        period_to_col = precomputed["period_to_col"]
        outcome_matrix = precomputed["outcome_matrix"]
        cohort_masks = precomputed["cohort_masks"]
        never_treated_mask = precomputed["never_treated_mask"]
        unit_cohorts = precomputed["unit_cohorts"]
        survey_w = precomputed.get("survey_weights")

        group_time_effects = {}
        influence_func_info = {}
        skipped_missing_period: List[Tuple] = []
        skipped_empty_cell: List[Tuple] = []

        # Collect all valid (g, t, base_col, post_col) tuples
        tasks = []
        for g in treatment_groups:
            valid_periods = self._valid_periods_for_group(
                g, time_periods, precomputed["observed_sorted"]
            )

            for t in valid_periods:
                # Base period selection: positional (sorted-index) neighbor,
                # matching R did::att_gt (see _select_base_period).
                base_period_val = self._select_base_period(g, t, precomputed["observed_sorted"])

                if base_period_val is None or t not in period_to_col:
                    skipped_missing_period.append((g, t))
                    group_time_effects[(g, t)] = _nan_gt_entry(skip_reason="missing_period")
                    continue

                tasks.append(
                    (g, t, period_to_col[base_period_val], period_to_col[t], base_period_val)
                )

        # Process all tasks
        atts = []
        ses = []
        task_keys = []

        for g, t, base_col, post_col, base_period_val in tasks:
            treated_mask = cohort_masks[g]

            if self.control_group == "never_treated":
                control_mask = never_treated_mask
            else:
                # Controls must be untreated at both t and base_period_val
                nyt_threshold = max(t, base_period_val) + self.anticipation
                control_mask = never_treated_mask | (
                    (unit_cohorts > nyt_threshold) & (unit_cohorts != g)
                )

            y_base = outcome_matrix[:, base_col]
            y_post = outcome_matrix[:, post_col]
            outcome_change = y_post - y_base
            valid_mask = ~(np.isnan(y_base) | np.isnan(y_post))

            treated_valid = treated_mask & valid_mask
            control_valid = control_mask & valid_mask

            n_treated = np.sum(treated_valid)
            n_control = np.sum(control_valid)

            if n_treated == 0 or n_control == 0:
                skipped_empty_cell.append((g, t))
                group_time_effects[(g, t)] = _nan_gt_entry(
                    n_treated=int(n_treated),
                    n_control=int(n_control),
                    skip_reason="zero_treated_control",
                )
                continue

            treated_change = outcome_change[treated_valid]
            control_change = outcome_change[control_valid]

            n_t = int(n_treated)
            n_c = int(n_control)

            # Inline no-covariates regression (difference in means)
            if survey_w is not None:
                sw_t = survey_w[treated_valid]
                sw_c = survey_w[control_valid]
                # Guard against zero effective mass
                if np.sum(sw_t) <= 0 or np.sum(sw_c) <= 0:
                    skipped_empty_cell.append((g, t))
                    group_time_effects[(g, t)] = _nan_gt_entry(
                        n_treated=n_t,
                        n_control=n_c,
                        skip_reason="zero_weight_mass",
                    )
                    continue
                sw_t_norm = sw_t / np.sum(sw_t)
                sw_c_norm = sw_c / np.sum(sw_c)
                mu_t = float(np.sum(sw_t_norm * treated_change))
                mu_c = float(np.sum(sw_c_norm * control_change))
                att = mu_t - mu_c

                # Influence function (survey-weighted)
                inf_treated = sw_t_norm * (treated_change - mu_t)
                inf_control = -sw_c_norm * (control_change - mu_c)
                # SE derived from IF: sum(IF_i^2)
                se = (
                    float(np.sqrt(np.sum(inf_treated**2) + np.sum(inf_control**2)))
                    if (n_t > 0 and n_c > 0)
                    else 0.0
                )
                sw_sum = float(np.sum(sw_t))
            else:
                mu_t = float(np.mean(treated_change))
                mu_c = float(np.mean(control_change))
                att = mu_t - mu_c

                # Influence function
                inf_treated = (treated_change - mu_t) / n_t
                inf_control = -(control_change - mu_c) / n_c
                # SE from the same IF that feeds aggregation (DRDID convention
                # sqrt(sum(phi^2)); the prior ddof=1 plug-in deviated from R's
                # reg_did_panel by O(1/n)).
                se = (
                    float(np.sqrt(np.sum(inf_treated**2) + np.sum(inf_control**2)))
                    if (n_t > 0 and n_c > 0)
                    else 0.0
                )
                sw_sum = None

            # A difference-in-means ATT is finite given n_t, n_c > 0, but enforce
            # the uniform contract anyway: a non-estimable cell is a NaN entry with
            # NO influence-function entry, skipped from batch inference and the
            # bootstrap (matches the covariate / general / RCS paths).
            if not np.isfinite(att):
                group_time_effects[(g, t)] = _nan_gt_entry(
                    n_treated=n_t,
                    n_control=n_c,
                    skip_reason="non_finite_regression",
                    survey_weight_sum=sw_sum,
                )
                continue

            gte_entry = {
                "effect": att,
                "se": se,
                # t_stat, p_value, conf_int filled by batch inference below
                "t_stat": np.nan,
                "p_value": np.nan,
                "conf_int": (np.nan, np.nan),
                "n_treated": n_t,
                "n_control": n_c,
                "skip_reason": None,
            }
            if sw_sum is not None:
                gte_entry["survey_weight_sum"] = sw_sum
            group_time_effects[(g, t)] = gte_entry

            treated_positions = np.where(treated_valid)[0]
            control_positions = np.where(control_valid)[0]
            inf_info_gt = {
                "treated_idx": treated_positions,
                "control_idx": control_positions,
                "treated_inf": inf_treated,
                "control_inf": inf_control,
            }
            influence_func_info[(g, t)] = inf_info_gt

            # Cluster-aware per-(g,t) SE: aggregate the per-(g,t) IF by
            # PSU when a survey design (explicit OR synthesized from bare
            # cluster=) provides one. Bit-equal to pre-PR when psu is None.
            rsu_for_gt = precomputed.get("resolved_survey_unit")
            if rsu_for_gt is not None and getattr(rsu_for_gt, "psu", None) is not None:
                se_cluster = _cluster_robust_se_from_per_gt_if(inf_info_gt, rsu_for_gt)
                # se_cluster is float (use), NaN (cluster-undefined, propagate),
                # or None (malformed, keep unit-level). Per the helper's
                # contract, NaN is a deliberate signal that the survey/PSU
                # variance is unidentified (e.g., G<2, lonely-PSU removed
                # all strata) — propagating NaN here causes safe_inference
                # to NaN-out the full per-cell inference surface, which is
                # the documented CS contract (feedback_no_silent_failures).
                if se_cluster is not None:
                    se = se_cluster
                    group_time_effects[(g, t)]["se"] = se

            atts.append(att)
            ses.append(se)
            task_keys.append((g, t))

        # Batch inference for all (g,t) pairs at once
        if task_keys:
            df_survey_val = precomputed.get("df_survey")
            # Guard: replicate design with undefined df → NaN inference
            if (
                df_survey_val is None
                and precomputed.get("resolved_survey_unit") is not None
                and hasattr(precomputed["resolved_survey_unit"], "uses_replicate_variance")
                and precomputed["resolved_survey_unit"].uses_replicate_variance
            ):
                df_survey_val = 0
            t_stats, p_values, ci_lowers, ci_uppers = safe_inference_batch(
                np.array(atts),
                np.array(ses),
                alpha=self.alpha,
                df=df_survey_val,
            )
            for idx, key in enumerate(task_keys):
                group_time_effects[key]["t_stat"] = float(t_stats[idx])
                group_time_effects[key]["p_value"] = float(p_values[idx])
                group_time_effects[key]["conf_int"] = (float(ci_lowers[idx]), float(ci_uppers[idx]))

        skip_info = {
            "missing_period": skipped_missing_period,
            "empty_cell": skipped_empty_cell,
        }
        return group_time_effects, influence_func_info, skip_info

    def _compute_all_att_gt_covariate_reg(
        self,
        precomputed: PrecomputedData,
        treatment_groups: List[Any],
        time_periods: List[Any],
        min_period: Any,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Optimized computation of all ATT(g,t) for the covariate regression case.

        Fits the per-cohort control outcome regression through the shared
        scale-equilibrated ``solve_ols`` (column-equilibrated SVD/gelsd; matches
        ``TripleDifference`` and R's ``lm()``/QR), reusing the per-control-group
        design across the (g, t) pairs that share it.

        Returns
        -------
        group_time_effects : dict
            Mapping (g, t) -> effect dict.
        influence_func_info : dict
            Mapping (g, t) -> influence function info dict.
        """
        period_to_col = precomputed["period_to_col"]
        outcome_matrix = precomputed["outcome_matrix"]
        cohort_masks = precomputed["cohort_masks"]
        never_treated_mask = precomputed["never_treated_mask"]
        unit_cohorts = precomputed["unit_cohorts"]
        covariate_by_period = precomputed["covariate_by_period"]
        is_balanced = precomputed["is_balanced"]

        group_time_effects = {}
        influence_func_info = {}
        atts = []
        ses = []
        task_keys = []
        n_nan_cells = 0
        skipped_missing_period: List[Tuple] = []
        skipped_empty_cell: List[Tuple] = []

        # Collect all valid (g, t) tasks with their base periods
        tasks_by_group = {}  # control_key -> list of (g, t, base_period_val, base_col, post_col)
        for g in treatment_groups:
            valid_periods = self._valid_periods_for_group(
                g, time_periods, precomputed["observed_sorted"]
            )

            for t in valid_periods:
                # Base period selection: positional (sorted-index) neighbor,
                # matching R did::att_gt (see _select_base_period).
                base_period_val = self._select_base_period(g, t, precomputed["observed_sorted"])

                if base_period_val is None or t not in period_to_col:
                    skipped_missing_period.append((g, t))
                    group_time_effects[(g, t)] = _nan_gt_entry(skip_reason="missing_period")
                    continue

                # Determine control regression grouping key.
                # For balanced panels with never_treated control, X_control depends
                # only on base_period_val (control mask is time-invariant).
                # For not_yet_treated, the control mask excludes cohort g, so include g.
                if is_balanced and self.control_group == "never_treated":
                    control_key = base_period_val
                else:
                    control_key = (g, base_period_val, t)

                tasks_by_group.setdefault(control_key, []).append(
                    (g, t, base_period_val, period_to_col[base_period_val], period_to_col[t])
                )

        # Process each group of tasks sharing the same control regression
        for control_key, tasks in tasks_by_group.items():
            # Use the first task to build X_control (same for all in the group)
            first_g, first_t, base_period_val, first_base_col, first_post_col = tasks[0]

            cov_matrix = covariate_by_period[base_period_val]

            # Build control mask (same for all tasks in this group)
            if self.control_group == "never_treated":
                control_mask = never_treated_mask
            else:
                # Controls must be untreated at both t and base_period_val
                nyt_threshold = max(first_t, base_period_val) + self.anticipation
                control_mask = never_treated_mask | (
                    (unit_cohorts > nyt_threshold) & (unit_cohorts != first_g)
                )

            # For balanced panels, valid_mask is all True so control_valid = control_mask
            if is_balanced:
                control_valid_base = control_mask
            else:
                y_base_first = outcome_matrix[:, first_base_col]
                y_post_first = outcome_matrix[:, first_post_col]
                valid_first = ~(np.isnan(y_base_first) | np.isnan(y_post_first))
                control_valid_base = control_mask & valid_first

            X_ctrl_raw = cov_matrix[control_valid_base]

            # Check for NaN in control covariates
            ctrl_has_nan = bool(np.any(np.isnan(X_ctrl_raw)))

            # Build X_ctrl with intercept
            n_c_base = int(np.sum(control_valid_base))
            if n_c_base == 0:
                skipped_empty_cell.extend((g, t) for g, t, *_ in tasks)
                # Controls are empty for this whole group; report each cell's actual
                # treated count (computed as in the per-pair loop), not a hardcoded 0.
                for g, t, _bp, base_col, post_col in tasks:
                    if is_balanced:
                        n_t_task = int(np.sum(cohort_masks[g]))
                    else:
                        valid_pair = ~(
                            np.isnan(outcome_matrix[:, base_col])
                            | np.isnan(outcome_matrix[:, post_col])
                        )
                        n_t_task = int(np.sum(cohort_masks[g] & valid_pair))
                    group_time_effects[(g, t)] = _nan_gt_entry(
                        n_treated=n_t_task, n_control=0, skip_reason="zero_treated_control"
                    )
                continue

            X_ctrl = None
            group_bread = None
            group_xbar_c = None
            group_Xc_centered = None
            if not ctrl_has_nan:
                X_ctrl = np.column_stack([np.ones(n_c_base), X_ctrl_raw])

                # One-time rank check per control group — for the informative
                # warning ONLY. The OR coefficient solve now routes through the
                # shared scale-robust solver (`solve_ols` -> equilibrated SVD),
                # which does its own rank handling, so we no longer pre-factor
                # X'X here (a large-scale covariate would have made the
                # normal-equations Cholesky / cond=1e-7 lstsq scale-sensitive).
                rank, dropped_cols, _ = _detect_rank_deficiency(X_ctrl)
                if len(dropped_cols) > 0 and self.rank_deficient_action == "warn":
                    col_info = _format_dropped_columns(dropped_cols)
                    warnings.warn(
                        f"Rank-deficient covariate design (control_key={control_key}): "
                        f"dropped columns {col_info}. Rank {rank} < {X_ctrl.shape[1]}. "
                        "Using a rank-reduced least-squares solution (dropped columns "
                        "excluded from the prediction).",
                        UserWarning,
                        stacklevel=2,
                    )

                # The IF bread is shared by every task in this group when the
                # control design is; hoist it (rank-guarded, so collinear
                # covariates get the column-drop generalized inverse
                # consistent with the DR/IPW per-cell breads). It is computed
                # in the CENTERED basis — the projection
                # x'(X'X)^{-1}x̄_t == 1/n_c + (x - x̄_c)'G^{-1}(x̄_t - x̄_c)
                # (G = centered control Gram) is the same quantity evaluated
                # offset-invariantly: the raw Gram squares the design's
                # conditioning, and a large constant covariate offset would
                # push the equilibrated-Gram rank check below its threshold,
                # silently truncating a genuine direction.
                if is_balanced and self.control_group == "never_treated":
                    group_xbar_c = X_ctrl_raw.mean(axis=0)
                    group_Xc_centered = X_ctrl_raw - group_xbar_c
                    group_bread = self._centered_or_bread(group_Xc_centered.T @ group_Xc_centered)

            # Process each (g, t) pair in this group
            for g, t, bp_val, base_col, post_col in tasks:
                treated_mask = cohort_masks[g]

                # Recompute control mask for not_yet_treated (varies by g, t)
                if self.control_group == "not_yet_treated":
                    # Controls must be untreated at both t and base period
                    nyt_threshold = max(t, bp_val) + self.anticipation
                    control_mask = never_treated_mask | (
                        (unit_cohorts > nyt_threshold) & (unit_cohorts != g)
                    )

                y_base = outcome_matrix[:, base_col]
                y_post = outcome_matrix[:, post_col]
                outcome_change = y_post - y_base

                if is_balanced:
                    valid_mask_pair = np.ones(len(y_base), dtype=bool)
                else:
                    valid_mask_pair = ~(np.isnan(y_base) | np.isnan(y_post))

                treated_valid = treated_mask & valid_mask_pair
                # For balanced + never_treated, control_valid is same as control_valid_base
                if is_balanced and self.control_group == "never_treated":
                    control_valid = control_valid_base
                else:
                    control_valid = control_mask & valid_mask_pair

                n_t = int(np.sum(treated_valid))
                n_c = int(np.sum(control_valid))

                if n_t == 0 or n_c == 0:
                    skipped_empty_cell.append((g, t))
                    group_time_effects[(g, t)] = _nan_gt_entry(
                        n_treated=n_t, n_control=n_c, skip_reason="zero_treated_control"
                    )
                    continue

                treated_change = outcome_change[treated_valid]
                control_change = outcome_change[control_valid]

                X_treated_pair = cov_matrix[treated_valid]
                X_control_pair = cov_matrix[control_valid]

                # Check for NaN in this pair's covariates
                if np.any(np.isnan(X_treated_pair)) or np.any(np.isnan(X_control_pair)):
                    # Fall back to unconditional (difference in means)
                    warnings.warn(
                        f"Missing values in covariates for group {g}, time {t}. "
                        "Falling back to unconditional estimation.",
                        UserWarning,
                        stacklevel=3,
                    )
                    mu_t_fb = float(np.mean(treated_change))
                    mu_c_fb = float(np.mean(control_change))
                    att = mu_t_fb - mu_c_fb
                    inf_treated = (treated_change - mu_t_fb) / n_t
                    inf_control = -(control_change - mu_c_fb) / n_c
                    # SE from the same IF that feeds aggregation (DRDID
                    # convention sqrt(sum(phi^2)); prior ddof=1 plug-in).
                    se = float(np.sqrt(np.sum(inf_treated**2) + np.sum(inf_control**2)))
                else:
                    # Build per-pair X_ctrl if control_valid differs from base
                    if is_balanced and self.control_group == "never_treated" and X_ctrl is not None:
                        pair_X_ctrl = X_ctrl
                        pair_n_c = n_c_base
                    else:
                        pair_X_ctrl = np.column_stack([np.ones(n_c), X_control_pair])
                        pair_n_c = n_c

                    # Solve for beta via the shared scale-robust solver
                    # (`solve_ols` -> column-equilibrated SVD/gelsd; matches
                    # TripleDifference._fit_predict_mu and R's lm()/QR). This
                    # replaces the prior cached/per-pair `cho_solve(X'X)` +
                    # `lstsq(cond=1e-7)` fallbacks, which were NOT scale-
                    # equilibrated. `rank_deficient_action="silent"` preserves
                    # this path's pre-existing never-raise behavior — the
                    # informative rank warning is emitted once per control group
                    # above.
                    with np.errstate(all="ignore"):
                        if pair_X_ctrl.shape[0] < pair_X_ctrl.shape[1]:
                            # Underdetermined control cell (fewer controls than
                            # covariate columns): solve_ols raises on n < k *before*
                            # it can rank-drop, so reproduce the documented R-style /
                            # solve_ols column-drop contract here — detect the rank-
                            # deficient columns, fit the reduced (full-column-rank)
                            # design via the equilibrated lstsq, and leave dropped
                            # coefficients NaN (zero-filled for prediction below, so
                            # a dropped covariate contributes 0). This matches the
                            # prior reduced-lstsq's no-crash handling under
                            # warn/silent AND avoids the non-unique minimum-norm
                            # extrapolation of a full n<k solve (the rank-deficiency
                            # warning already fired for this control group above).
                            pair_rank, _, pair_pivot = _detect_rank_deficiency(pair_X_ctrl)
                            kept_cols = np.sort(pair_pivot[:pair_rank])
                            beta = np.full(pair_X_ctrl.shape[1], np.nan)
                            if kept_cols.size:
                                beta[kept_cols] = _equilibrated_lstsq(
                                    pair_X_ctrl[:, kept_cols], control_change
                                )
                        else:
                            beta, _, _ = solve_ols(
                                pair_X_ctrl,
                                control_change,
                                rank_deficient_action="silent",
                                return_vcov=False,
                            )
                    # Dropped (collinear) columns come back NaN; zero them for
                    # prediction (a dropped covariate contributes 0 to the
                    # column-space projection), matching the prior reduced-lstsq.
                    beta = np.where(np.isnan(beta), 0.0, beta)

                    nan_cell = False

                    if beta is None or np.any(~np.isfinite(beta)):
                        nan_cell = True
                        n_nan_cells += 1

                    if not nan_cell:
                        X_treated_w_intercept = np.column_stack([np.ones(n_t), X_treated_pair])
                        with np.errstate(all="ignore"):
                            predicted_control = X_treated_w_intercept @ beta
                        treated_residuals = treated_change - predicted_control
                        if np.any(~np.isfinite(predicted_control)):
                            nan_cell = True
                            n_nan_cells += 1

                    if not nan_cell:
                        att = float(np.mean(treated_residuals))
                        with np.errstate(all="ignore"):
                            residuals = control_change - pair_X_ctrl @ beta
                        if np.any(~np.isfinite(residuals)):
                            nan_cell = True
                            n_nan_cells += 1

                    if nan_cell:
                        att = np.nan
                    else:
                        # DRDID::reg_did_panel influence function. The control
                        # leg carries the OLS estimation-effect term: the
                        # control residual projected through the rank-guarded
                        # bread and evaluated at the treated covariate mean
                        # (phi convention — every R 1/n factor cancels).
                        # Evaluated in the CENTERED basis,
                        # 1/n_c + (x - x̄_c)'G^{-1}(x̄_t - x̄_c), which is the
                        # same quantity as x'(X'X)^{-1}x̄_t but offset-
                        # invariant (see the group_bread hoist above); the
                        # no-covariate collapse -residuals/n_c is the leading
                        # 1/n_c term. Omitting this term left ATTs exact (it
                        # is mean-zero by the OLS normal equations) while SEs
                        # sat 4-13% from R per-cell.
                        if (
                            is_balanced
                            and self.control_group == "never_treated"
                            and group_bread is not None
                        ):
                            xbar_c = group_xbar_c
                            Xc_centered = group_Xc_centered
                            bread = group_bread
                        else:
                            xbar_c = X_control_pair.mean(axis=0)
                            Xc_centered = X_control_pair - xbar_c
                            bread = self._centered_or_bread(Xc_centered.T @ Xc_centered)
                        d_tc = X_treated_pair.mean(axis=0) - xbar_c
                        with np.errstate(all="ignore"):
                            proj_c = 1.0 / pair_n_c + Xc_centered @ (bread @ d_tc)
                            inf_treated = (treated_residuals - att) / n_t
                            inf_control = -residuals * proj_c
                        # Per-cell SE from the same IF that feeds aggregation
                        # (sqrt(sum(phi^2)), the DR convention). A rank-0
                        # CENTERED Gram is benign — _centered_or_bread maps it
                        # to a zero correction (finite SE on the identified
                        # intercept subset); the NaN guard below covers only
                        # non-finite inputs, never a silent 0.0.
                        var_psi = float(np.sum(inf_treated**2) + np.sum(inf_control**2))
                        if not np.isfinite(var_psi):
                            se = float("nan")
                        elif var_psi > 0:
                            se = float(np.sqrt(var_psi))
                        else:
                            se = 0.0

                # Non-estimable (non-finite regression) cell: materialize NaN with
                # NO influence-function entry — uniform with the missing-period /
                # zero-control / zero-weight paths, so every NaN cell is excluded
                # from aggregation and the bootstrap (the IF-membership filter and
                # the finite-mask both drop it). Skip batch inference (already NaN).
                if not np.isfinite(att):
                    group_time_effects[(g, t)] = _nan_gt_entry(
                        n_treated=n_t,
                        n_control=n_c,
                        skip_reason="non_finite_regression",
                    )
                    continue

                group_time_effects[(g, t)] = {
                    "effect": att,
                    "se": se,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_treated": n_t,
                    "n_control": n_c,
                    "skip_reason": None,
                }

                # INVARIANT: np.where over boolean masks -> duplicate-free
                # index arrays (fancy-+= scatter contract, see
                # staggered_aggregation._combined_if_fast).
                treated_positions = np.where(treated_valid)[0]
                control_positions = np.where(control_valid)[0]
                inf_info_gt = {
                    "treated_idx": treated_positions,
                    "control_idx": control_positions,
                    "treated_inf": inf_treated,
                    "control_inf": inf_control,
                }
                influence_func_info[(g, t)] = inf_info_gt

                # Cluster-aware per-(g,t) SE — see same pattern in
                # _compute_all_att_gt_vectorized.
                rsu_for_gt = precomputed.get("resolved_survey_unit")
                if rsu_for_gt is not None and getattr(rsu_for_gt, "psu", None) is not None:
                    se_cluster = _cluster_robust_se_from_per_gt_if(inf_info_gt, rsu_for_gt)
                    # Propagate NaN (cluster-undefined) per the helper's
                    # contract — see _compute_all_att_gt_vectorized site
                    # for the same pattern.
                    if se_cluster is not None:
                        se = se_cluster
                        group_time_effects[(g, t)]["se"] = se

                atts.append(att)
                ses.append(se)
                task_keys.append((g, t))

        if n_nan_cells > 0:
            warnings.warn(
                f"{n_nan_cells} group-time cell(s) have non-finite regression results "
                "(near-singular covariates). These cells are preserved with NaN inference.",
                UserWarning,
                stacklevel=2,
            )

        # Batch inference
        if task_keys:
            # Use survey df for replicate designs (propagated from precomputed)
            _ipw_dr_df = precomputed.get("df_survey") if precomputed is not None else None
            # Guard: replicate design with undefined df → NaN inference
            if (
                _ipw_dr_df is None
                and precomputed is not None
                and precomputed.get("resolved_survey_unit") is not None
                and hasattr(precomputed["resolved_survey_unit"], "uses_replicate_variance")
                and precomputed["resolved_survey_unit"].uses_replicate_variance
            ):
                _ipw_dr_df = 0
            t_stats, p_values, ci_lowers, ci_uppers = safe_inference_batch(
                np.array(atts), np.array(ses), alpha=self.alpha, df=_ipw_dr_df
            )
            for idx, key in enumerate(task_keys):
                group_time_effects[key]["t_stat"] = float(t_stats[idx])
                group_time_effects[key]["p_value"] = float(p_values[idx])
                group_time_effects[key]["conf_int"] = (float(ci_lowers[idx]), float(ci_uppers[idx]))

        skip_info = {
            "missing_period": skipped_missing_period,
            "empty_cell": skipped_empty_cell,
        }
        return group_time_effects, influence_func_info, skip_info

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        aggregate: Any = _DEPRECATED_FIT_ARG,
        balance_e: Any = _DEPRECATED_FIT_ARG,
        survey_design: Optional["SurveyDesign"] = None,
    ) -> CallawaySantAnnaResults:
        """
        Fit the Callaway-Sant'Anna estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with unit and time identifiers. For repeated
            cross-sections (``panel=False``), each observation should
            have a unique unit ID — units do not repeat across periods.
        outcome : str
            Name of outcome variable column.
        unit : str
            Name of unit identifier column.
        time : str
            Name of time period column.
        first_treat : str
            Name of column indicating when unit was first treated.
            Use 0 (or np.inf) for never-treated units.
        covariates : list, optional
            List of covariate column names for conditional parallel trends.
        aggregate : str, optional
            DEPRECATED since 3.9, removed in 4.0 (ledger row M-020). Passing
            it emits a ``FutureWarning``. Aggregate as a post-fit step
            instead - ``results.aggregate("simple" | "event_study" |
            "group")`` - which needs no refit and returns a new object,
            leaving ``results`` unchanged. Accepted values are unchanged:
            - None: Only compute ATT(g,t) (default)
            - "simple": Simple weighted average (overall ATT)
            - "event_study": Aggregate by relative time (event study)
            - "group": Aggregate by treatment cohort
            - "all": Compute all aggregations

            Fit-time aggregation remains ONLY as the temporary compatibility
            surface for consumers that still read it (``compute_honest_did``,
            ``compute_pretrends_power`` and ``plot_event_study``, tracked in
            ``TODO.md``); it is not the recommended path for new code.
        balance_e : int, optional
            DEPRECATED since 3.9, removed in 4.0 (ledger row M-117). Passing
            it emits a ``FutureWarning``; it moves onto the post-fit call as
            ``results.aggregate("event_study", balance_e=...)``. For event
            study, balance the panel at relative time e. Ensures all groups
            contribute to each relative period.
        survey_design : SurveyDesign, optional
            Survey design specification. Supports pweight with strata/PSU/FPC.
            Aggregated SEs (overall, event study, group) use design-based
            variance via compute_survey_if_variance(). All estimation methods
            (reg, ipw, dr) support covariates + survey. For repeated
            cross-sections (``panel=False``), survey weights are
            per-observation (no unit-level collapse).

        Returns
        -------
        CallawaySantAnnaResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # ---- fit(aggregate=) / fit(balance_e=) shims (rows M-020 / M-117) ----
        # Deprecated in 3.9, removed at 4.0: aggregation moves to the post-fit
        # results.aggregate(type=...). The sentinel default means the warning
        # fires ONLY when the caller supplies the argument, never on a plain
        # fit(). Routing is otherwise untouched, so the deprecated path returns
        # exactly the numbers it always did.
        _deprecated_passed = [
            name
            for name, value in (("aggregate", aggregate), ("balance_e", balance_e))
            if not isinstance(value, _DeprecatedFitArg)
        ]
        if _deprecated_passed:
            _args = " / ".join(f"{n}=" for n in _deprecated_passed)
            warnings.warn(
                f"CallawaySantAnna.fit({_args}) is deprecated and will be removed "
                "in 4.0. Fit once, then aggregate as a post-fit step: "
                "results = CallawaySantAnna().fit(...); "
                "results.aggregate('event_study') / .aggregate('group') / "
                ".aggregate('simple'). balance_e moves onto aggregate() alongside "
                "it: results.aggregate('event_study', balance_e=2).",
                FutureWarning,
                stacklevel=2,
            )
        if isinstance(aggregate, _DeprecatedFitArg):
            aggregate = None
        if isinstance(balance_e, _DeprecatedFitArg):
            balance_e = None

        # Validate pscore_trim (may have been changed via set_params)
        if not (0 < self.pscore_trim < 0.5):
            raise ValueError(f"pscore_trim must be in (0, 0.5), got {self.pscore_trim}")

        # NB: the event-study VCV and its df provenance used to be reset here,
        # because ``_aggregate_event_study`` stashed them on ``self`` and a
        # reused estimator could leak them across fits / non-ES aggregates /
        # the ES empty-result early return. The aggregator now RETURNS them
        # (``EventStudyAggregation``), so there is no cross-fit state to reset
        # and the leak is impossible by construction.

        # Tracker for _safe_inv lstsq fallbacks across all analytical SE
        # paths (PS Hessian, OR bread, event-study bread, etc.). Emit ONE
        # aggregate warning at the end of fit rather than fanning out per
        # cell. Sibling of PR #9 finding #17.
        self._safe_inv_tracker: List[float] = []

        # Re-validate vcov_type at fit-time: __init__ and set_params (via
        # the BaseEstimator probe re-init) both validate eagerly, so this
        # second layer only catches DIRECT attribute mutation
        # (est.vcov_type = ...) before it propagates to Results metadata.
        self._validate_vcov_type(self.vcov_type)

        # --- allow_unbalanced_panel routing (RC-on-panel = R's allow_unbalanced_panel) ---
        # Detect an unbalanced panel (some units unobserved in some periods).
        # Only meaningful for panel input; declared RC (panel=False) is handled
        # by the branches below via `_use_rc`.
        _is_unbalanced_panel = False
        _finite_out = None
        if self.panel and all(c in data.columns for c in (unit, time, outcome)):
            # Complete-case balance (matches R `pre_process_did`, which
            # complete-case-filters BEFORE deciding to flip to the RC path): a
            # rectangular panel with a missing / non-finite OUTCOME in a cell is
            # effectively unbalanced — the default panel path treats that cell as
            # a missing panel cell (wide-pivot valid-mask) and silently does
            # within-cell differencing. So detect unbalancedness on the
            # finite-outcome rows, not just structural (unit, time) row presence.
            _finite_out = np.isfinite(
                pd.to_numeric(data[outcome], errors="coerce").to_numpy(dtype=np.float64)
            )
            # Assess rectangularity on the complete-case frame for BOTH the cell
            # count AND the units x periods denominator — the same frame the RC
            # path estimates on (below). If a whole unit or a whole period is
            # dropped by non-finite outcomes, the remaining sample may still be
            # rectangular (balanced) and must stay inert, matching R's
            # complete-case preprocessing. Using the unfiltered unit/period counts
            # in the denominator would spuriously route such a panel to RC.
            _cc = data.loc[_finite_out]
            _is_unbalanced_panel = (
                int(_cc.drop_duplicates(subset=[unit, time]).shape[0])
                < _cc[unit].nunique() * _cc[time].nunique()
            )
        # Route through the RC levels estimator ONLY when the flag is set AND the
        # panel is actually unbalanced. Inert on a balanced panel (default
        # within-cell differencing is byte-identical). This MATCHES R: R's
        # pre_process_did recomputes `allow_unbalanced_panel` from the observed
        # balance and only flips `panel <- FALSE` when the panel is genuinely
        # unbalanced (verified: R keeps panel=TRUE on balanced data under the
        # flag), so balanced-panel inertness is the R contract, not a deviation.
        _route_as_rc = bool(self.allow_unbalanced_panel and _is_unbalanced_panel)
        # `_use_rc` selects the RC precompute / per-cell loop / obs-level counting
        # for BOTH declared RC (panel=False) and RC-routing (unbalanced + flag).
        _use_rc = (not self.panel) or _route_as_rc
        if _route_as_rc and survey_design is not None:
            raise NotImplementedError(
                "allow_unbalanced_panel=True is not yet supported together with "
                "survey_design=. The RC-on-panel path carries per-observation "
                "weights, whereas survey designs assume per-unit weights; the "
                "combination needs a per-unit weight resolution that is not "
                "implemented (deferred). Use a balanced panel, drop "
                "survey_design=, or pass panel=False for genuine repeated "
                "cross-sections."
            )
        if _is_unbalanced_panel and not self.allow_unbalanced_panel:
            warnings.warn(
                "Unbalanced panel detected (some units are unobserved in some "
                "periods). Each ATT(g,t) is estimated by within-cell panel "
                "differencing on the units observed at BOTH the base period and "
                "t — a valid but different estimand than R's repeated-cross-"
                "section handling. Pass allow_unbalanced_panel=True to match R "
                "`did::att_gt(allow_unbalanced_panel=TRUE)` (the DRDID "
                "reg_did_rc levels estimator on the pooled observations).",
                UserWarning,
                stacklevel=2,
            )
        if _route_as_rc:
            warnings.warn(
                "allow_unbalanced_panel=True: routing the unbalanced panel "
                "through the repeated-cross-section (RC) levels estimator to "
                "match R `did::att_gt(allow_unbalanced_panel=TRUE)`. The RC "
                "estimator assumes the population distribution of (Y, X, G) is "
                "stable across periods (not data-checkable); standard errors "
                "cluster the per-observation influence function by unit.",
                UserWarning,
                stacklevel=2,
            )
            # Complete-case analysis sample: DROP non-finite-outcome rows so that
            # survey/PSU resolution, cohort masses, the RC precompute, AND
            # estimation all operate on the SAME rows (matches R
            # `pre_process_did`). The RC estimator slices the outcome arrays
            # directly (no wide-pivot valid-mask), so an unfiltered NaN outcome
            # would spuriously materialize its 2x2 cells as non-estimable and
            # break the bit-exact ATT parity. Reassign `data` once, up front, so
            # every downstream consumer sees the filtered sample.
            if _finite_out is not None and not bool(_finite_out.all()):
                data = data.loc[_finite_out].reset_index(drop=True)
            # Validate panel structure BEFORE routing to the RC estimator — R's
            # pre_process_did does the same. The RC precompute reads cohort and
            # cluster PER OBSERVATION (no wide pivot), so a duplicate (unit,
            # period) row, a unit whose treatment cohort changes over time, or a
            # time-varying cluster would silently corrupt the reweighting /
            # composition. The normal panel path catches duplicates via the wide
            # pivot; the RC route must check explicitly. Fail closed.
            # Column-presence guards: `unit`/`time` are guaranteed present (the
            # unbalance detection above required them); for a missing `first_treat`
            # / `cluster` column, fall through so the canonical required-column
            # check below raises "Missing columns: ..." rather than a KeyError.
            _dups = int(data.duplicated(subset=[unit, time]).sum())
            if _dups:
                raise ValueError(
                    f"allow_unbalanced_panel=True requires at most one observation "
                    f"per ({unit}, {time}); found {_dups} duplicate row(s). Aggregate "
                    f"or de-duplicate before fitting."
                )
            if first_treat in data.columns:
                # Coerce to numeric first (like the normal fit path) so coercible
                # labels such as "3" and "3.0" for the same unit are not falsely
                # flagged as a changing cohort.
                _ft = pd.to_numeric(data[first_treat], errors="coerce")
                _ft_counts = _ft.groupby(data[unit].to_numpy()).nunique()
                if (_ft_counts > 1).any():
                    _bad = list(_ft_counts.index[_ft_counts > 1][:5])
                    raise ValueError(
                        f"allow_unbalanced_panel=True requires a time-invariant treatment "
                        f"cohort ('{first_treat}') per unit; unit(s) {_bad} have a changing "
                        f"first_treat."
                    )
            if self.cluster is not None and self.cluster in data.columns:
                _cl_counts = data.groupby(unit)[self.cluster].nunique()
                if (_cl_counts > 1).any():
                    _bad = list(_cl_counts.index[_cl_counts > 1][:5])
                    raise ValueError(
                        f"allow_unbalanced_panel=True with cluster='{self.cluster}' "
                        f"requires a time-invariant cluster per unit; unit(s) {_bad} "
                        f"have a changing cluster value."
                    )

        if not self.panel:
            warnings.warn(
                "panel=False uses repeated cross-section DRDID estimators "
                "(Sant'Anna & Zhao 2020, Section 4) which assume stationary "
                "cross-sectional sampling: the population distribution of "
                "(Y, X, G) must be stable across periods. This assumption "
                "is not data-checkable.",
                UserWarning,
                stacklevel=2,
            )

        # Validate unique unit IDs for panel=False
        if not self.panel:
            if data[unit].duplicated().any():
                raise ValueError(
                    "panel=False requires unique unit IDs (one observation per unit). "
                    "Found duplicate unit IDs. If your data is a panel, use panel=True."
                )

        # Normalize empty covariates list to None
        if covariates is not None and len(covariates) == 0:
            covariates = None

        # Resolve survey design if provided
        from diff_diff.survey import (
            SurveyDesign,
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        # Wire bare cluster= into the survey-PSU machinery BEFORE the
        # unit-constant + pweight validators below, so the synthesized
        # survey design also passes through validation (otherwise movers
        # on panel data — units crossing cluster boundaries — would
        # silently get a first-value-wins collapse via
        # _collapse_survey_to_unit_level). Mirrors estimators.py:497-516,
        # adapted for CS's IF-based variance: no solve_ols(cluster_ids=...)
        # fallback exists, so we synthesize a minimal SurveyDesign(psu=...)
        # to reach the existing PSU-meat aggregator + PSU multiplier
        # bootstrap. Both consume resolved_survey.psu, so synthesis
        # transparently activates the same code paths used when the user
        # passes SurveyDesign(psu=X) explicitly.
        effective_survey_design = survey_design  # refreshed below if synthesized
        if self.cluster is not None:
            if self.cluster not in data.columns:
                raise ValueError(f"cluster column '{self.cluster}' not found in data")
            # Pre-validate cluster NaN with a cluster-domain error message.
            # Without this, SurveyDesign.resolve() raises "PSU column ...
            # contains missing values" — wrong domain for the user-facing
            # cluster= API.
            if data[self.cluster].isna().any():
                raise ValueError(
                    f"cluster column '{self.cluster}' contains missing "
                    "values. All observations must have valid cluster "
                    "identifiers."
                )
            # Reject replicate-weight + cluster=: replicate IF variance is
            # computed by replicate reweighting (BRR / Fay / JK1 / JKn / SDR)
            # and ignores PSU/cluster entirely (survey.py:104-109 enforces
            # replicate_weights are mutually exclusive with strata/psu/fpc).
            # Honoring bare cluster= here would silently have no effect on
            # variance while populating cluster_name/n_clusters on Results
            # dishonestly. Fail-closed per feedback_no_silent_failures.
            if (
                survey_design is not None
                and getattr(survey_design, "replicate_weights", None) is not None
            ):
                raise NotImplementedError(
                    f"CallawaySantAnna(cluster={self.cluster!r}) is not "
                    "supported with replicate-weight survey designs. "
                    "Replicate-weight variance is computed by replicate "
                    "reweighting (BRR / Fay / JK1 / JKn / SDR) and ignores "
                    "PSU/cluster entirely — setting cluster= would silently "
                    "have no effect on the variance estimate. Either omit "
                    "cluster= (the replicate weights encode the design "
                    "structure implicitly) or use a non-replicate survey "
                    "design (with explicit strata/psu/fpc)."
                )
            cluster_ids = data[self.cluster].values

            if resolved_survey is None:
                # Bare cluster=, no survey: synthesize minimal
                # SurveyDesign(psu=...). Activates the same survey-PSU
                # aggregator + bootstrap machinery already used when the
                # user passes SurveyDesign(psu=X) explicitly.
                synthetic_design = SurveyDesign(psu=self.cluster, weight_type="pweight")
                (
                    resolved_survey,
                    survey_weights,
                    survey_weight_type,
                    _,
                ) = _resolve_survey_for_fit(synthetic_design, data, "analytical")
                effective_survey_design = synthetic_design
                # survey_metadata stays None — user did NOT provide a
                # survey design, so downstream consumers that check
                # `survey_metadata is not None` for "original fit used a
                # survey design" (DiagnosticReport at diagnostic_report.py:
                # 848-856 + 1150-1158, summary at staggered_results.py:
                # 235-238) must continue to see a non-survey fit. The
                # cluster-level df is carried via the dedicated
                # `df_inference` field on Results (set below after the
                # cluster-handling block).
            elif resolved_survey.psu is None:
                # User provided survey_design (weights/strata/etc.) without
                # PSU AND cluster=X. Inject cluster as PSU into the resolved
                # survey. Construct effective_survey_design with
                # psu=self.cluster (via dataclasses.replace) so the
                # downstream _validate_unit_constant_survey catches movers
                # on panel data (otherwise the validator runs on the user-
                # provided design which has no PSU column).
                from dataclasses import replace

                from diff_diff.survey import (
                    _inject_cluster_as_psu,
                    compute_survey_metadata,
                )

                # This branch is only reached with a survey design present.
                assert survey_design is not None
                effective_survey_design = replace(survey_design, psu=self.cluster)
                resolved_survey = _inject_cluster_as_psu(resolved_survey, cluster_ids)
                # Recompute survey_metadata to reflect the new effective PSU
                # (so n_psu / df_survey on Results reflect the injected
                # cluster, not the no-PSU pre-injection state).
                # resolved_survey non-None implies survey_design was passed.
                assert survey_design is not None
                raw_w = (
                    data[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(data), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)
            else:
                # User provided survey_design with psu= AND cluster=X. PSU
                # wins; _resolve_effective_cluster emits a UserWarning if
                # partitions differ. Return value intentionally discarded
                # — helper's purpose is the warning, and the effective PSU
                # is already resolved_survey.psu (no rewrite needed).
                from diff_diff.survey import _resolve_effective_cluster

                _resolve_effective_cluster(resolved_survey, cluster_ids, self.cluster)

        # allow_unbalanced_panel (RC-routing) with NO user cluster=: cluster the
        # per-observation RC influence function by the ORIGINAL unit so the SE
        # accounts for within-unit correlation — matching R's unit-level IF
        # aggregation for allow_unbalanced_panel. Synthesize SurveyDesign(psu=unit)
        # → the same PSU-meat aggregator + PSU multiplier bootstrap used for an
        # explicit cluster=. When the user set cluster=X the block above already
        # synthesized psu=X (their choice); do NOT double-cluster here.
        if _route_as_rc and self.cluster is None and resolved_survey is None:
            _unit_psu_design = SurveyDesign(psu=unit, weight_type="pweight")
            (
                resolved_survey,
                survey_weights,
                survey_weight_type,
                _,
            ) = _resolve_survey_for_fit(_unit_psu_design, data, "analytical")
            effective_survey_design = _unit_psu_design

        # Validate within-unit constancy for panel survey designs (uses
        # effective_survey_design so synthesized designs are validated too).
        # Skipped under RC-routing: the RC path is observation-level and the
        # synthesized psu=unit design is trivially unit-constant.
        if resolved_survey is not None:
            if self.panel and not _route_as_rc:
                _validate_unit_constant_survey(data, unit, effective_survey_design)
            if resolved_survey.weight_type != "pweight":
                raise ValueError(
                    f"CallawaySantAnna survey support requires weight_type='pweight', "
                    f"got '{resolved_survey.weight_type}'. The survey variance math "
                    f"assumes probability weights (pweight)."
                )
        # Note: strata/PSU/FPC are now supported — aggregated SEs use
        # compute_survey_if_variance() for design-based inference.

        # Bootstrap + survey is now supported via PSU-level multiplier bootstrap.

        # Validate inputs
        required_cols = [outcome, unit, time, first_treat]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Create working copy
        df = data.copy()

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Standardize the first_treat column name for internal use
        # This avoids hardcoding column names in internal methods
        df["first_treat"] = df[first_treat]

        # Never-treated indicator (must precede treatment_groups to exclude np.inf)
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)
        # Normalize np.inf → 0 so all downstream `> 0` checks exclude never-treated
        _inf_mask = df[first_treat] == np.inf
        if _inf_mask.any():
            n_inf_units = df.loc[_inf_mask, unit].nunique()
            warnings.warn(
                f"{n_inf_units} unit(s) have first_treat=inf; recoding to 0 "
                f"(never-treated). Use first_treat=0 to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
        df.loc[_inf_mask, first_treat] = 0

        # Identify groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0])

        if self.panel:
            # Panel (incl. an unbalanced panel routed as RC): count unique
            # UNITS — the units are real even when the aggregation runs the RC
            # estimator. Only a declared repeated cross-section (panel=False)
            # counts observations.
            unit_info = (
                df.groupby(unit)
                .agg({first_treat: "first", "_never_treated": "first"})
                .reset_index()
            )
            n_treated_units = (unit_info[first_treat] > 0).sum()
            n_control_units = (unit_info["_never_treated"]).sum()
        else:
            # RCS: count observations per cohort (no unit tracking)
            n_treated_units = int((df[first_treat] > 0).sum())
            n_control_units = int(df["_never_treated"].sum())

        if n_control_units == 0 and self.control_group == "never_treated":
            raise ValueError(
                "No never-treated units found. Check 'first_treat' column. "
                "Use control_group='not_yet_treated' if all units are eventually treated."
            )
        if n_control_units == 0 and self.control_group == "not_yet_treated":
            # With not_yet_treated, controls are units not yet treated at each
            # (g, t) pair — never-treated units are not required.
            if len(treatment_groups) < 2:
                raise ValueError(
                    "not_yet_treated control group requires at least 2 treatment "
                    "cohorts when there are no never-treated units."
                )

        # Note: CallawaySantAnna supports survey weights, strata, PSU, and FPC.
        # Per-cell SEs use IF-based variance; aggregated SEs use design-based
        # variance via compute_survey_if_variance() or PSU-level bootstrap.
        # Pre-compute data structures for efficient ATT(g,t) computation
        if not _use_rc:
            precomputed = self._precompute_structures(
                df,
                outcome,
                unit,
                time,
                first_treat,
                covariates,
                time_periods,
                treatment_groups,
                resolved_survey=resolved_survey,
            )
        else:
            precomputed = self._precompute_structures_rc(
                df,
                outcome,
                unit,
                time,
                first_treat,
                covariates,
                time_periods,
                treatment_groups,
                resolved_survey=resolved_survey,
            )

        # Recompute survey metadata from the unit-level resolved survey so
        # that n_psu and df_survey reflect the actual survey design (explicit
        # PSU/strata) rather than hard-coding n_units.
        if resolved_survey is not None and survey_metadata is not None:
            resolved_survey_unit = precomputed.get("resolved_survey_unit")
            if resolved_survey_unit is not None:
                from diff_diff.survey import compute_survey_metadata

                unit_w = resolved_survey_unit.weights
                survey_metadata = compute_survey_metadata(resolved_survey_unit, unit_w)

        # Survey df for safe_inference calls — use the unit-level resolved
        # survey df computed in _precompute_structures for consistency.
        df_survey = precomputed.get("df_survey")
        # Guard: replicate design with undefined df (rank <= 1) → NaN inference
        if (
            df_survey is None
            and resolved_survey is not None
            and hasattr(resolved_survey, "uses_replicate_variance")
            and resolved_survey.uses_replicate_variance
        ):
            df_survey = 0

        # Compute ATT(g,t) for each group-time combination
        min_period = min(time_periods)
        has_survey = resolved_survey is not None

        _skip_info = {"missing_period": [], "empty_cell": []}
        _n_skipped_other = 0

        if _use_rc:
            # --- Repeated cross-section path ---
            # No vectorized/Cholesky fast paths (panel-only optimizations).
            # Loop using _compute_att_gt_rc() for each (g,t).
            group_time_effects = {}
            influence_func_info = {}
            epv_diagnostics = (
                {} if (covariates and self.estimation_method in ("ipw", "dr")) else None
            )

            for g in treatment_groups:
                valid_periods = self._valid_periods_for_group(
                    g, time_periods, precomputed["observed_sorted"]
                )

                for t in valid_periods:
                    rc_result = self._compute_att_gt_rc(
                        precomputed,
                        g,
                        t,
                        covariates,
                        epv_diagnostics=epv_diagnostics,
                    )
                    (
                        att_gt,
                        se_gt,
                        n_treat,
                        n_ctrl,
                        inf_info,
                        sw_sum,
                        cohort_mass,
                        skip_reason,
                    ) = rc_result
                    agg_w = cohort_mass if cohort_mass is not None else n_treat

                    if att_gt is None or not np.isfinite(att_gt):
                        # Non-estimable cell: materialize NaN with NO IF entry,
                        # uniform across paths. att_gt is None -> the helper
                        # reported the reason; a non-finite att_gt (solve produced
                        # NaN/inf) -> "non_finite_regression".
                        _n_skipped_other += 1
                        group_time_effects[(g, t)] = _nan_gt_entry(
                            n_treated=n_treat,
                            n_control=n_ctrl,
                            skip_reason=(
                                skip_reason if att_gt is None else "non_finite_regression"
                            ),
                        )
                    else:
                        # Cluster-aware per-(g,t) SE on the RCS path. RC
                        # IF indices are per-obs (vs per-unit on the panel
                        # path); the corresponding PSU array is
                        # ``resolved_survey.psu`` (length n_obs), not
                        # ``resolved_survey_unit.psu``. Bit-equal to pre-PR
                        # when psu is None.
                        rs_for_gt = precomputed.get("resolved_survey") if precomputed else None
                        if (
                            rs_for_gt is not None
                            and getattr(rs_for_gt, "psu", None) is not None
                            and inf_info is not None
                        ):
                            se_cluster = _cluster_robust_se_from_per_gt_if(inf_info, rs_for_gt)
                            # Propagate NaN (cluster-undefined) per the
                            # helper's contract — see
                            # _compute_all_att_gt_vectorized for the
                            # pattern.
                            if se_cluster is not None:
                                se_gt = se_cluster

                        t_stat, p_val, ci = safe_inference(
                            att_gt,
                            se_gt,
                            alpha=self.alpha,
                            df=df_survey,
                        )

                        gte_entry = {
                            "effect": att_gt,
                            "se": se_gt,
                            "t_stat": t_stat,
                            "p_value": p_val,
                            "conf_int": ci,
                            "n_treated": n_treat,
                            "n_control": n_ctrl,
                            "agg_weight": agg_w,
                            "skip_reason": None,
                        }
                        if sw_sum is not None:
                            gte_entry["survey_weight_sum"] = sw_sum
                        group_time_effects[(g, t)] = gte_entry

                        if inf_info is not None:
                            influence_func_info[(g, t)] = inf_info

        elif covariates is None and self.estimation_method == "reg":
            # Fast vectorized path for the common no-covariates regression case
            group_time_effects, influence_func_info, _skip_info = (
                self._compute_all_att_gt_vectorized(
                    precomputed, treatment_groups, time_periods, min_period
                )
            )
            epv_diagnostics = None  # No logit in this path
        elif (
            covariates is not None
            and self.estimation_method == "reg"
            and self.rank_deficient_action != "error"
            and not has_survey  # vectorized OR path is unweighted; survey reg routes separately
        ):
            # Optimized (vectorized) covariate regression path; the OR nuisance
            # is fit through the shared scale-equilibrated solve_ols.
            group_time_effects, influence_func_info, _skip_info = (
                self._compute_all_att_gt_covariate_reg(
                    precomputed, treatment_groups, time_periods, min_period
                )
            )
            epv_diagnostics = None  # No logit in this path
        else:
            # General path: IPW, DR, rank_deficient_action="error", or edge cases
            group_time_effects = {}
            influence_func_info = {}

            # Propensity score cache for IPW/DR with covariates
            pscore_cache = {} if (covariates and self.estimation_method in ("ipw", "dr")) else None

            epv_diagnostics = (
                {} if (covariates and self.estimation_method in ("ipw", "dr")) else None
            )

            for g in treatment_groups:
                valid_periods = self._valid_periods_for_group(
                    g, time_periods, precomputed["observed_sorted"]
                )

                for t in valid_periods:
                    att_gt, se_gt, n_treat, n_ctrl, inf_info, sw_sum, skip_reason = (
                        self._compute_att_gt_fast(
                            precomputed,
                            g,
                            t,
                            covariates,
                            pscore_cache=pscore_cache,
                            epv_diagnostics=epv_diagnostics,
                        )
                    )

                    if att_gt is None or not np.isfinite(att_gt):
                        # Non-estimable cell: materialize NaN with NO IF entry,
                        # uniform across paths. att_gt is None -> the helper
                        # reported the reason; a non-finite att_gt (solve produced
                        # NaN/inf) -> "non_finite_regression".
                        _n_skipped_other += 1
                        group_time_effects[(g, t)] = _nan_gt_entry(
                            n_treated=n_treat,
                            n_control=n_ctrl,
                            skip_reason=(
                                skip_reason if att_gt is None else "non_finite_regression"
                            ),
                        )
                    else:
                        # Cluster-aware per-(g,t) SE: when a survey PSU is
                        # in play (explicit OR synthesized from bare
                        # cluster=), aggregate the per-(g,t) IF by PSU
                        # and use CR1 Liang-Zeger SE instead of the
                        # unit-level diff-of-means SE returned by OR/IPW/DR.
                        # Preserves bit-equality when psu is None.
                        rsu_for_gt = precomputed.get("resolved_survey_unit")
                        if (
                            rsu_for_gt is not None
                            and getattr(rsu_for_gt, "psu", None) is not None
                            and inf_info is not None
                        ):
                            se_cluster = _cluster_robust_se_from_per_gt_if(inf_info, rsu_for_gt)
                            # Propagate NaN (cluster-undefined) per the
                            # helper's contract — see
                            # _compute_all_att_gt_vectorized for the
                            # pattern.
                            if se_cluster is not None:
                                se_gt = se_cluster

                        t_stat, p_val, ci = safe_inference(
                            att_gt,
                            se_gt,
                            alpha=self.alpha,
                            df=df_survey,
                        )

                        gte_entry = {
                            "effect": att_gt,
                            "se": se_gt,
                            "t_stat": t_stat,
                            "p_value": p_val,
                            "conf_int": ci,
                            "n_treated": n_treat,
                            "n_control": n_ctrl,
                            "skip_reason": None,
                        }
                        if sw_sum is not None:
                            gte_entry["survey_weight_sum"] = sw_sum
                        group_time_effects[(g, t)] = gte_entry

                        if inf_info is not None:
                            influence_func_info[(g, t)] = inf_info

        if not group_time_effects or not any(
            np.isfinite(v["effect"]) for v in group_time_effects.values()
        ):
            raise ValueError(
                "Could not estimate any group-time effects. "
                "Check that data has sufficient observations."
            )

        # Consolidated EPV summary warning
        if epv_diagnostics:
            low_epv = {k: v for k, v in epv_diagnostics.items() if v.get("is_low")}
            if low_epv:
                n_affected = len(low_epv)
                n_total = len(epv_diagnostics)
                min_entry = min(low_epv.values(), key=lambda v: v["epv"])
                min_g = min(low_epv.keys(), key=lambda k: low_epv[k]["epv"])
                warnings.warn(
                    f"Low Events Per Variable (EPV) detected in propensity "
                    f"score estimation for {n_affected} of {n_total} cell(s). "
                    f"Minimum EPV = {min_entry['epv']:.1f} "
                    f"(cohort g={min_g[0]}). "
                    f"Consider estimation_method='reg' (avoids propensity "
                    f"scores) or reducing the number of covariates. "
                    f"See results.epv_summary() for details.",
                    UserWarning,
                    stacklevel=2,
                )

        # Consolidated (g,t) cell skip warning (all paths)
        _n_missing = len(_skip_info.get("missing_period", []))
        _n_empty = len(_skip_info.get("empty_cell", []))
        _n_total_skipped = _n_missing + _n_empty + _n_skipped_other
        if _n_total_skipped > 0:
            _parts = []
            if _n_missing:
                _parts.append(
                    f"{_n_missing} due to missing base/post period " f"in panel structure"
                )
            if _n_empty:
                _parts.append(f"{_n_empty} due to zero treated or control " f"observations")
            if _n_skipped_other:
                _parts.append(
                    f"{_n_skipped_other} due to insufficient data or " f"non-estimable cells"
                )
            warnings.warn(
                f"{_n_total_skipped} (group, time) cell(s) could not be "
                f"estimated: {'; '.join(_parts)}.",
                UserWarning,
                stacklevel=2,
            )

        # Universal base period: materialize each cohort's positional base as a
        # zero reference cell (att=0, se=NaN, zero influence function) so it
        # appears in the group-time table AND is weighted into the dynamic
        # event-study aggregation and the multiplier bootstrap exactly like R
        # `did` (which lists the base as a zero att_gt cell and includes it in
        # aggte(type="dynamic")). Injected once here — before every aggregation
        # and the bootstrap — so all consumers see the same cells uniformly. The
        # `group` / `simple` aggregations filter to post-treatment cells
        # (t >= g - anticipation) and so exclude the reference (t = base is the
        # last observed period with base + anticipation < g).
        if self.base_period == "universal":
            _obs_sorted = precomputed["observed_sorted"]
            # Each cohort's reference weight is the FIXED cohort mass = R's aggte
            # `pg` numerator: the survey-weighted sum (else the unit / observation
            # count) over the WHOLE cohort, read straight from the cohort column.
            # NOT a single cell's valid-treated count (which under-weights the
            # reference on unbalanced panels). Because it needs no estimable cell,
            # the reference is materialized for every cohort with a valid base --
            # matching R, which materializes the base zero cell before estimating
            # the other cells, even for a cohort whose non-reference cells are all
            # non-estimable. RCS cells carry `agg_weight` (fixed mass); panel cells
            # read `n_treated`; survey aggregation overrides both with the cohort
            # survey mass.
            _unit_cohorts = precomputed.get("unit_cohorts")
            _survey_w = precomputed.get("survey_weights")
            _is_panel = precomputed.get("is_panel", True)
            for _g in treatment_groups:
                _base = self._select_base_period(_g, _g, _obs_sorted)
                if _base is None or (_g, _base) in group_time_effects or _unit_cohorts is None:
                    continue
                _cmask = np.asarray(_unit_cohorts) == _g
                _mass = (
                    float(np.sum(np.asarray(_survey_w)[_cmask]))
                    if _survey_w is not None
                    else float(np.count_nonzero(_cmask))
                )
                if _mass <= 0:
                    continue  # cohort has no units -> nothing to weight
                _ref: Dict[str, Any] = {
                    "effect": 0.0,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_treated": int(round(_mass)),
                    "n_control": 0,
                    "skip_reason": None,
                    "is_reference": True,
                }
                if not _is_panel:
                    _ref["agg_weight"] = _mass
                if _survey_w is not None:
                    _ref["survey_weight_sum"] = _mass
                group_time_effects[(_g, _base)] = _ref
                influence_func_info[(_g, _base)] = {
                    "treated_idx": np.array([], dtype=int),
                    "control_idx": np.array([], dtype=int),
                    "treated_inf": np.array([]),
                    "control_inf": np.array([]),
                }

        # Compute overall ATT (simple aggregation)
        overall_att, overall_se, overall_effective_df = self._aggregate_simple(
            group_time_effects, influence_func_info, df, unit, precomputed
        )
        # Use per-statistic effective df from replicate aggregation if available;
        # otherwise fall back to the original df from the survey design.
        if overall_effective_df is not None:
            df_survey = overall_effective_df
            # Propagate to survey_metadata for display consistency
            if survey_metadata is not None:
                survey_metadata.df_survey = df_survey
        # Guard: replicate design with undefined df (rank <= 1) → NaN inference
        if (
            df_survey is None
            and resolved_survey is not None
            and hasattr(resolved_survey, "uses_replicate_variance")
            and resolved_survey.uses_replicate_variance
        ):
            df_survey = 0
        overall_t, overall_p, overall_ci = safe_inference(
            overall_att,
            overall_se,
            alpha=self.alpha,
            df=df_survey,
        )

        # Compute additional aggregations if requested
        event_study_effects = None
        group_effects = None

        # Aggregation outputs that used to arrive as ``self._event_study_*``
        # side channels; the aggregator is now pure and returns them.
        es_aggregation = None
        if aggregate in ["event_study", "all"]:
            es_aggregation = self._aggregate_event_study(
                group_time_effects,
                influence_func_info,
                treatment_groups,
                time_periods,
                balance_e,
                df,
                unit,
                precomputed,
            )
            event_study_effects = es_aggregation.effects

        if aggregate in ["group", "all"]:
            group_effects = self._aggregate_by_group(
                group_time_effects,
                influence_func_info,
                treatment_groups,
                precomputed=precomputed,
                df=df,
                unit=unit,
            )

        # Reject replicate-weight designs for bootstrap — replicate variance
        # is an analytical alternative, not compatible with bootstrap
        if (
            self.n_bootstrap > 0
            and resolved_survey is not None
            and hasattr(resolved_survey, "uses_replicate_variance")
            and resolved_survey.uses_replicate_variance
        ):
            raise NotImplementedError(
                "CallawaySantAnna bootstrap (n_bootstrap > 0) is not supported "
                "with replicate-weight survey designs. Replicate weights provide "
                "analytical variance; use n_bootstrap=0 instead."
            )

        # Run bootstrap inference if requested
        bootstrap_results = None
        if self.n_bootstrap > 0 and influence_func_info:
            bootstrap_results = self._run_multiplier_bootstrap(
                group_time_effects=group_time_effects,
                influence_func_info=influence_func_info,
                aggregate=aggregate,
                balance_e=balance_e,
                treatment_groups=treatment_groups,
                time_periods=time_periods,
                df=df,
                unit=unit,
                precomputed=precomputed,
                cband=self.cband,
            )

            # Update estimates with bootstrap inference
            overall_se = bootstrap_results.overall_att_se
            overall_t = safe_inference(overall_att, overall_se, alpha=self.alpha)[0]
            overall_p = bootstrap_results.overall_att_p_value
            overall_ci = bootstrap_results.overall_att_ci

            # Update group-time effects with bootstrap SEs (batched)
            gt_keys = [gt for gt in group_time_effects if gt in bootstrap_results.group_time_ses]
            if gt_keys:
                gt_effects_arr = np.array(
                    [float(group_time_effects[gt]["effect"]) for gt in gt_keys]
                )
                gt_ses_arr = np.array(
                    [float(bootstrap_results.group_time_ses[gt]) for gt in gt_keys]
                )
                gt_t_stats, _, _, _ = safe_inference_batch(
                    gt_effects_arr, gt_ses_arr, alpha=self.alpha
                )
                for idx, gt in enumerate(gt_keys):
                    group_time_effects[gt]["se"] = bootstrap_results.group_time_ses[gt]
                    group_time_effects[gt]["conf_int"] = bootstrap_results.group_time_cis[gt]
                    group_time_effects[gt]["p_value"] = bootstrap_results.group_time_p_values[gt]
                    group_time_effects[gt]["t_stat"] = float(gt_t_stats[idx])

            # Update event study effects with bootstrap SEs (batched)
            if (
                event_study_effects is not None
                and bootstrap_results.event_study_ses is not None
                and bootstrap_results.event_study_cis is not None
                and bootstrap_results.event_study_p_values is not None
            ):
                es_keys = [e for e in event_study_effects if e in bootstrap_results.event_study_ses]
                if es_keys:
                    es_effects_arr = np.array(
                        [float(event_study_effects[e]["effect"]) for e in es_keys]
                    )
                    es_ses_arr = np.array(
                        [float(bootstrap_results.event_study_ses[e]) for e in es_keys]
                    )
                    es_t_stats, _, _, _ = safe_inference_batch(
                        es_effects_arr, es_ses_arr, alpha=self.alpha
                    )
                    for idx, e in enumerate(es_keys):
                        event_study_effects[e]["se"] = bootstrap_results.event_study_ses[e]
                        event_study_effects[e]["conf_int"] = bootstrap_results.event_study_cis[e]
                        event_study_effects[e]["p_value"] = bootstrap_results.event_study_p_values[
                            e
                        ]
                        event_study_effects[e]["t_stat"] = float(es_t_stats[idx])

            # Update group effects with bootstrap SEs (batched)
            if (
                group_effects is not None
                and bootstrap_results.group_effect_ses is not None
                and bootstrap_results.group_effect_cis is not None
                and bootstrap_results.group_effect_p_values is not None
            ):
                grp_keys = [g for g in group_effects if g in bootstrap_results.group_effect_ses]
                if grp_keys:
                    grp_effects_arr = np.array(
                        [float(group_effects[g]["effect"]) for g in grp_keys]
                    )
                    grp_ses_arr = np.array(
                        [float(bootstrap_results.group_effect_ses[g]) for g in grp_keys]
                    )
                    grp_t_stats, _, _, _ = safe_inference_batch(
                        grp_effects_arr, grp_ses_arr, alpha=self.alpha
                    )
                    for idx, g in enumerate(grp_keys):
                        group_effects[g]["se"] = bootstrap_results.group_effect_ses[g]
                        group_effects[g]["conf_int"] = bootstrap_results.group_effect_cis[g]
                        group_effects[g]["p_value"] = bootstrap_results.group_effect_p_values[g]
                        group_effects[g]["t_stat"] = float(grp_t_stats[idx])
                        # Same clearing rule the ES df provenance follows below:
                        # these se/p/CI are now percentile-bootstrap values that
                        # never used the analytical df, so keeping df_used would
                        # claim a t-reference that governed nothing.
                        group_effects[g]["df_used"] = None

        # Compute simultaneous confidence band CIs if cband is available
        cband_crit_value = None
        if bootstrap_results is not None:
            cband_crit_value = bootstrap_results.cband_crit_value

        if cband_crit_value is not None and event_study_effects is not None:
            for e, eff_data in event_study_effects.items():
                se_val = eff_data["se"]
                if np.isfinite(se_val) and se_val > 0:
                    eff_data["cband_conf_int"] = (
                        eff_data["effect"] - cband_crit_value * se_val,
                        eff_data["effect"] + cband_crit_value * se_val,
                    )

        # Consolidated _safe_inv rank-guard warning (sibling of PR #9
        # finding #17). Rank-deficient PS Hessian / OR bread matrices in the
        # analytical SE paths are near-singular under a constant/collinear
        # covariate. They are now inverted by a rank-guarded generalized inverse
        # that truncates the redundant direction(s) (finite SE on the identified
        # subset). Aggregated here into ONE UserWarning so a bad design surface
        # is surfaced once rather than per cell.
        if self._safe_inv_tracker and self.rank_deficient_action != "silent":
            n_fallbacks = len(self._safe_inv_tracker)
            finite_conds = [c for c in self._safe_inv_tracker if np.isfinite(c)]
            max_cond = max(finite_conds) if finite_conds else float("inf")
            warnings.warn(
                f"Rank-deficient matrix encountered {n_fallbacks} time(s) "
                f"in analytical SE paths (propensity-score Hessian or "
                f"outcome-regression bread); dropped redundant direction(s) "
                f"via a rank-guarded inverse. Max condition number of affected "
                f"matrix: {max_cond:.2e}. Standard errors use the identified "
                f"covariate subset; consider dropping collinear covariates or "
                f"using n_bootstrap > 0.",
                UserWarning,
                stacklevel=2,
            )

        # Store results
        # Retrieve event-study VCV from aggregation mixin (Phase 7d).
        # Clear it when bootstrap overwrites event-study SEs to prevent
        # HonestDiD from mixing analytical VCV with bootstrap SEs.
        event_study_vcov = es_aggregation.vcov if es_aggregation is not None else None
        event_study_vcov_index = es_aggregation.vcov_index if es_aggregation is not None else None
        if bootstrap_results is not None and event_study_vcov is not None:
            event_study_vcov = None
            event_study_vcov_index = None
        # Same clearing rule for the ES df provenance: bootstrap replaces the
        # stored ES se/p/CI with percentile values that never used the
        # analytical df, so surfacing it would be false provenance.
        event_study_df = es_aggregation.df_used if es_aggregation is not None else None
        if bootstrap_results is not None:
            event_study_df = None

        # Resolve canonical cluster_name + n_clusters for Results metadata.
        # Canonical PSU column wins when explicit: if survey_design has
        # psu=, that's the canonical column even if bare cluster= was also
        # set (a UserWarning would have fired above if partitions differed).
        if survey_design is not None and getattr(survey_design, "psu", None) is not None:
            cluster_name_for_results: Optional[str] = survey_design.psu
        elif self.cluster is not None:
            cluster_name_for_results = self.cluster
        elif _route_as_rc:
            # RC-on-panel auto-synthesizes SurveyDesign(psu=unit); surface the
            # effective unit-level clustering so introspection / summaries don't
            # report cluster_name=None while n_clusters is populated.
            cluster_name_for_results = unit
        else:
            cluster_name_for_results = None
        n_clusters_for_results: Optional[int] = (
            int(np.unique(resolved_survey.psu).size)
            if (resolved_survey is not None and resolved_survey.psu is not None)
            else None
        )
        # df_inference: cluster-level degrees of freedom for downstream
        # inference (HonestDiD t-critical selection). Populated ONLY for
        # the bare-cluster-synthesize path (where survey_metadata is None
        # because the user did not provide a survey design). For
        # inject/conflict branches, survey_metadata is populated and
        # survey_metadata.df_survey carries the actual CS-internal df
        # (which may have been tightened by overall_effective_df recompute
        # at the aggregation step around L1995-1999). Narrowing here
        # prevents HonestDiD from reading a stale/wrong df_inference for
        # survey fits whose df was tightened post-resolve — fix for
        # CI codex P1 (PR #487 round 1).
        df_inference_for_results: Optional[int] = (
            int(resolved_survey.df_survey)
            if (
                resolved_survey is not None
                and survey_metadata is None
                and getattr(resolved_survey, "df_survey", None) is not None
            )
            else None
        )

        self.results_ = CallawaySantAnnaResults(
            group_time_effects=group_time_effects,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=len(df),
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            alpha=self.alpha,
            control_group=self.control_group,
            base_period=self.base_period,
            anticipation=self.anticipation,
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            bootstrap_results=bootstrap_results,
            cband_crit_value=cband_crit_value,
            pscore_trim=self.pscore_trim,
            survey_metadata=survey_metadata,
            event_study_vcov=event_study_vcov,
            event_study_df=event_study_df,
            event_study_vcov_index=event_study_vcov_index,
            panel=self.panel,
            allow_unbalanced_panel=self.allow_unbalanced_panel,
            used_rc_on_unbalanced_panel=_route_as_rc,
            epv_diagnostics=epv_diagnostics if epv_diagnostics else None,
            epv_threshold=self.epv_threshold,
            pscore_fallback=self.pscore_fallback,
            vcov_type=self.vcov_type,
            cluster_name=cluster_name_for_results,
            n_clusters=n_clusters_for_results,
            df_inference=df_inference_for_results,
        )

        # Attach the post-fit aggregation kit (spec section 6, rows M-020/M-117).
        # Built HERE because neither `precomputed` nor `influence_func_info`
        # survives fit() - they are locals - so the kit cannot be reconstructed
        # later. It deliberately excludes the data matrices: re-aggregation reads
        # only unit-level bookkeeping, so the source panel is never retained.
        self.results_._aggregation_kit = _build_aggregation_kit(
            self, precomputed, influence_func_info, group_time_effects
        )

        self.is_fitted_ = True
        return self.results_

    def _centered_or_bread(self, gram: np.ndarray) -> np.ndarray:
        """Rank-guarded inverse of the CENTERED control covariate Gram.

        Unlike the ``[1, X]`` breads elsewhere, the intercept direction is
        handled analytically here (the ``1/sum(W_c)`` leading term of the
        reg estimation-effect projection), so a rank-0 centered Gram — zero
        within-control covariate variation, e.g. a constant covariate as
        the only regressor — is benign: the correction on the identified
        (intercept-only) subset is exactly zero, and the projection
        collapses to ``1/sum(W_c)``. Map ``_safe_inv``'s all-NaN rank-0
        sentinel to a zero matrix so the cell keeps the finite SE on the
        identified subset (rank-guard REGISTRY contract; equals both the
        no-covariate fit and the pre-centered raw-Gram column-drop) instead
        of NaN-poisoning the IF. Partial deficiency is untouched —
        ``_safe_inv`` already returns a zero-filled column-dropped inverse,
        and the aggregate rank-guard warning still fires via the tracker.
        """
        bread = _safe_inv(gram, tracker=self._safe_inv_tracker)
        if bread.size and not np.any(np.isfinite(bread)):
            return np.zeros_like(bread)
        return bread

    def _outcome_regression(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
        X_treated: Optional[np.ndarray] = None,
        X_control: Optional[np.ndarray] = None,
        sw_treated: Optional[np.ndarray] = None,
        sw_control: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using outcome regression.

        With covariates:
        1. Regress outcome changes on covariates for control group
        2. Predict counterfactual for treated using their covariates
        3. ATT = mean(treated_change) - mean(predicted_counterfactual)

        Without covariates:
        Simple difference in means.

        Parameters
        ----------
        sw_treated, sw_control : np.ndarray, optional
            Survey weights for treated and control units.
        """
        n_t = len(treated_change)
        n_c = len(control_change)

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Covariate-adjusted outcome regression
            # Fit regression on control units: E[Delta Y | X, D=0]
            # (residuals are recomputed below from the NaN-zeroed beta so the
            # IF uses the same prediction convention as the ATT).
            beta, _ = _linear_regression(
                X_control,
                control_change,
                rank_deficient_action=self.rank_deficient_action,
                weights=sw_control,
            )

            # Zero NaN coefficients for prediction (dropped rank-deficient columns
            # contribute 0 to the column space projection, matching DR path convention)
            beta = np.where(np.isnan(beta), 0.0, beta)

            # Predict counterfactual for treated units
            X_treated_with_intercept = np.column_stack([np.ones(n_t), X_treated])
            predicted_control = np.dot(X_treated_with_intercept, beta)

            # ATT: survey-weighted mean of treated residuals
            treated_residuals = treated_change - predicted_control

            # DRDID::reg_did_panel influence function (survey weights map to
            # R's i.weights; only weight RATIOS enter every term, so the
            # normalization constant cancels and the raw sw arrays can be
            # used directly). The control leg carries the OLS estimation-
            # effect term: the (weighted) control residual projected through
            # the rank-guarded WLS bread and evaluated at the (weighted)
            # treated covariate mean. Evaluated in the CENTERED basis,
            # 1/sum(W_c) + (x - x̄_c)'G^{-1}(x̄_t - x̄_c) with G the centered
            # weighted control Gram — the same quantity as x'(X'WX)^{-1}x̄_t
            # but offset-invariant (the raw Gram squares the design's
            # conditioning; a large constant covariate offset would push the
            # equilibrated-Gram rank check below threshold and silently
            # truncate a genuine direction). The no-covariate collapse is the
            # leading -resid_c/n_c term. Omitting the estimation-effect term
            # left ATTs exact (it is mean-zero by the WLS normal equations)
            # while per-cell SEs sat 4-13% from R. phi convention: psi_R / n.
            X_c_int = np.column_stack([np.ones(n_c), X_control])
            with np.errstate(all="ignore"):
                resid_c = control_change - np.dot(X_c_int, beta)

            if sw_treated is not None:
                sw_t_sum = float(np.sum(sw_treated))
                sw_t_norm = sw_treated / sw_t_sum
                att = float(np.sum(sw_t_norm * treated_residuals))
                # sw_treated/sw_control are supplied together by all callers.
                assert sw_control is not None
                W_c = sw_control
                xbar_t = np.sum(sw_treated[:, None] * X_treated, axis=0) / sw_t_sum
                inf_treated = sw_t_norm * (treated_residuals - att)
            else:
                att = float(np.mean(treated_residuals))
                W_c = np.ones(n_c)
                xbar_t = X_treated.mean(axis=0)
                inf_treated = (treated_residuals - att) / n_t

            w_c_sum = float(np.sum(W_c))
            with np.errstate(all="ignore"):
                xbar_c = np.sum(W_c[:, None] * X_control, axis=0) / w_c_sum
                Xc_centered = X_control - xbar_c
                bread = self._centered_or_bread(Xc_centered.T @ (W_c[:, None] * Xc_centered))
                proj_c = 1.0 / w_c_sum + Xc_centered @ (bread @ (xbar_t - xbar_c))
                inf_control = -(W_c * resid_c) * proj_c
            inf_func = np.concatenate([inf_treated, inf_control])

            # Per-cell SE from the same IF that feeds aggregation
            # (sqrt(sum(phi^2)), the DR convention). A rank-0 CENTERED Gram
            # is benign — _centered_or_bread maps it to a zero correction
            # (finite SE on the identified intercept subset); the NaN guard
            # below covers only non-finite inputs.
            var_psi = float(np.sum(inf_func**2))
            if not np.isfinite(var_psi):
                se = float("nan")
            elif var_psi > 0:
                se = float(np.sqrt(var_psi))
            else:
                se = 0.0
        else:
            # Simple difference in means (no covariates)
            if sw_treated is not None:
                sw_t_norm = sw_treated / np.sum(sw_treated)
                sw_c_norm = sw_control / np.sum(sw_control)
                mu_t = float(np.sum(sw_t_norm * treated_change))
                mu_c = float(np.sum(sw_c_norm * control_change))
                att = mu_t - mu_c

                # Influence function (survey-weighted)
                inf_treated = sw_t_norm * (treated_change - mu_t)
                inf_control = -sw_c_norm * (control_change - mu_c)
                inf_func = np.concatenate([inf_treated, inf_control])

                # SE from influence function variance
                se = (
                    float(np.sqrt(np.sum(inf_treated**2) + np.sum(inf_control**2)))
                    if (n_t > 0 and n_c > 0)
                    else 0.0
                )
            else:
                mu_t = float(np.mean(treated_change))
                mu_c = float(np.mean(control_change))
                att = mu_t - mu_c

                # Influence function (for aggregation)
                inf_treated = (treated_change - mu_t) / n_t
                inf_control = -(control_change - mu_c) / n_c
                inf_func = np.concatenate([inf_treated, inf_control])

                # SE from the same IF that feeds aggregation (DRDID convention
                # sqrt(sum(phi^2)); the prior ddof=1 plug-in deviated from R's
                # reg_did_panel by O(1/n)).
                se = float(np.sqrt(np.sum(inf_func**2))) if (n_t > 0 and n_c > 0) else 0.0

        return att, se, inf_func

    def _ipw_estimation(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
        n_treated: int,
        n_control: int,
        X_treated: Optional[np.ndarray] = None,
        X_control: Optional[np.ndarray] = None,
        pscore_cache: Optional[Dict] = None,
        pscore_key: Optional[Any] = None,
        sw_treated: Optional[np.ndarray] = None,
        sw_control: Optional[np.ndarray] = None,
        sw_all: Optional[np.ndarray] = None,
        context_label: str = "",
        epv_diagnostics_out: Optional[dict] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using inverse probability weighting.

        With covariates:
        1. Estimate propensity score P(D=1|X) using logistic regression
        2. Reweight control units to match treated covariate distribution
        3. ATT = mean(treated) - weighted_mean(control)

        Without covariates:
        Simple difference in means with unconditional propensity weighting.

        Parameters
        ----------
        sw_treated, sw_control, sw_all : np.ndarray, optional
            Survey weights for treated, control, and all units.
        """
        n_t = len(treated_change)
        n_c = len(control_change)

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Covariate-adjusted IPW estimation
            ps_fallback_used = False
            # Check propensity score cache
            cached_pscore = None
            if pscore_cache is not None and pscore_key is not None:
                cached_pscore = pscore_cache.get(pscore_key)

            if cached_pscore is not None:
                # Use cached propensity scores (beta coefficients + EPV diag)
                beta_logistic, cached_diag = cached_pscore
                X_all = np.vstack([X_treated, X_control])
                X_all_with_intercept = np.column_stack([np.ones(n_t + n_c), X_all])
                z = np.dot(X_all_with_intercept, beta_logistic)
                z = np.clip(z, -500, 500)
                pscore = 1 / (1 + np.exp(-z))
                if epv_diagnostics_out is not None and cached_diag:
                    epv_diagnostics_out.update(cached_diag)
            else:
                # Stack covariates and create treatment indicator
                X_all = np.vstack([X_treated, X_control])
                D = np.concatenate([np.ones(n_t), np.zeros(n_c)])

                # Estimate propensity scores using IRLS logistic regression
                diag = {}
                try:
                    beta_logistic, pscore = solve_logit(
                        X_all,
                        D,
                        rank_deficient_action=self.rank_deficient_action,
                        weights=sw_all,
                        epv_threshold=self.epv_threshold,
                        context_label=context_label,
                        diagnostics_out=diag,
                    )
                    _check_propensity_diagnostics(pscore, self.pscore_trim)
                    # Cache the fitted coefficients (zero-fill NaN from
                    # dropped rank-deficient columns to prevent NaN
                    # propagation on cache reuse) alongside EPV diagnostics
                    if pscore_cache is not None and pscore_key is not None:
                        beta_clean = np.where(np.isfinite(beta_logistic), beta_logistic, 0.0)
                        pscore_cache[pscore_key] = (beta_clean, diag)
                except (np.linalg.LinAlgError, ValueError):
                    if self.pscore_fallback == "error" or self.rank_deficient_action == "error":
                        raise
                    # Fallback to unconditional if logistic regression fails
                    ctx = f" for {context_label}" if context_label else ""
                    warnings.warn(
                        f"Propensity score estimation failed{ctx}. "
                        f"Falling back to unconditional propensity "
                        f"(all covariates dropped for this cell). "
                        f"Consider estimation_method='reg' to avoid "
                        f"propensity scores entirely.",
                        UserWarning,
                        stacklevel=4,
                    )
                    if sw_all is not None:
                        pos = sw_all > 0
                        p_uc = float(np.average(D[pos], weights=sw_all[pos]))
                    else:
                        p_uc = n_t / (n_t + n_c)
                    pscore = np.full(len(D), p_uc)
                    ps_fallback_used = True
                if epv_diagnostics_out is not None and diag:
                    epv_diagnostics_out.update(diag)

            # Propensity scores for treated and control
            pscore_treated = pscore[:n_t]
            pscore_control = pscore[n_t:]

            # Clip propensity scores to avoid extreme weights
            pscore_control = np.clip(pscore_control, self.pscore_trim, 1 - self.pscore_trim)
            pscore_treated = np.clip(pscore_treated, self.pscore_trim, 1 - self.pscore_trim)

            if sw_treated is not None:
                # IPW weights compose with survey weights:
                # w_i = sw_i * p(X_i) / (1 - p(X_i))
                weights_control = sw_control * pscore_control / (1 - pscore_control)
                weights_control_norm = weights_control / np.sum(weights_control)

                # ATT: survey-weighted treated mean minus composite-weighted control mean
                sw_t_norm = sw_treated / np.sum(sw_treated)
                mu_t = float(np.sum(sw_t_norm * treated_change))
                att = mu_t - float(np.sum(weights_control_norm * control_change))

                # Influence function (survey-weighted)
                inf_treated = sw_t_norm * (treated_change - mu_t)
                inf_control = -weights_control_norm * (
                    control_change - np.sum(weights_control_norm * control_change)
                )
                inf_func = np.concatenate([inf_treated, inf_control])

                if not ps_fallback_used:
                    # Propensity score IF correction
                    # Accounts for estimation uncertainty in logistic regression coefficients
                    X_all_int = np.column_stack([np.ones(n_t + n_c), X_all])
                    pscore_all = np.concatenate([pscore_treated, pscore_control])

                    # PS IF correction — compute in R's psi convention, convert to phi
                    n_all_panel = n_t + n_c
                    W_ps = pscore_all * (1 - pscore_all)
                    if sw_all is not None:
                        W_ps = W_ps * sw_all
                    # R: Hessian.ps = crossprod(X * sqrt(W)) / n
                    H_psi = X_all_int.T @ (W_ps[:, None] * X_all_int) / n_all_panel
                    H_psi_inv = _safe_inv(H_psi, tracker=self._safe_inv_tracker)

                    D_all = np.concatenate([np.ones(n_t), np.zeros(n_c)])
                    score_ps = (D_all - pscore_all)[:, None] * X_all_int
                    if sw_all is not None:
                        score_ps = score_ps * sw_all[:, None]
                    # R: asy.lin.rep.ps = score.ps %*% Hessian.ps  (psi scale, O(1) per obs)
                    asy_lin_rep_psi = score_ps @ H_psi_inv

                    att_control_weighted = np.sum(weights_control_norm * control_change)
                    # R: M2 = colMeans(w.cont * (y - att) * X) / mean(w.cont)
                    # np.sum (not mean): subset sum with normalized weights matches
                    # R's full-sample colMeans/mean(w) after cancellation
                    M2 = np.sum(
                        (weights_control_norm * (control_change - att_control_weighted))[:, None]
                        * X_all_int[n_t:],
                        axis=0,
                    )

                    # psi-scale correction, convert to phi for storage
                    # Subtract: R adds PS correction to inf.control, then att = treat - control
                    inf_func = inf_func - (asy_lin_rep_psi @ M2) / n_all_panel

                # SE from influence function variance
                var_psi = np.sum(inf_func**2)
                # A rank-0 cell yields an all-NaN bread (and thus NaN inf_func);
                # propagate NaN rather than masking it as 0.0 via ``var_psi > 0``
                # (NaN > 0 is False). Partial deficiency keeps var_psi finite.
                if not np.isfinite(var_psi):
                    se = float("nan")
                elif var_psi > 0:
                    se = float(np.sqrt(var_psi))
                else:
                    se = 0.0
            else:
                # IPW weights for control units: p(X) / (1 - p(X))
                # This reweights controls to have same covariate distribution as treated
                weights_control = pscore_control / (1 - pscore_control)
                weights_control_norm = weights_control / np.sum(weights_control)

                # ATT = mean(treated) - weighted_mean(control)
                mu_t = float(np.mean(treated_change))
                eta_c = float(np.sum(weights_control_norm * control_change))
                att = mu_t - eta_c

                # Influence function (Hajek leading terms; mirrors the survey
                # branch above with unit weights)
                inf_treated = (treated_change - mu_t) / n_t
                inf_control = -weights_control_norm * (control_change - eta_c)
                inf_func = np.concatenate([inf_treated, inf_control])

                if not ps_fallback_used:
                    # Propensity-score estimation-effect correction
                    # (DRDID::std_ipw_did_panel's asy.lin.rep.ps %*% M2) —
                    # mirrors the survey branch above with sw=None. Its
                    # omission left ATTs exact (the logit score is mean-zero
                    # at the MLE) while aggregated SEs sat ~2.4% from R.
                    n_all_panel = n_t + n_c
                    X_all_int = np.column_stack([np.ones(n_all_panel), X_all])
                    pscore_all = np.concatenate([pscore_treated, pscore_control])

                    W_ps = pscore_all * (1 - pscore_all)
                    # R: Hessian.ps = crossprod(X * sqrt(W)) / n
                    H_psi = X_all_int.T @ (W_ps[:, None] * X_all_int) / n_all_panel
                    H_psi_inv = _safe_inv(H_psi, tracker=self._safe_inv_tracker)

                    D_all = np.concatenate([np.ones(n_t), np.zeros(n_c)])

                    # R: M2 = colMeans(w.cont * (y - att) * X) / mean(w.cont);
                    # the normalized-weight subset sum matches R's full-sample
                    # colMeans/mean(w) after cancellation.
                    M2 = np.sum(
                        (weights_control_norm * (control_change - eta_c))[:, None]
                        * X_all_int[n_t:],
                        axis=0,
                    )

                    # R: asy.lin.rep.ps %*% M2 with
                    # asy.lin.rep.ps = score.ps %*% Hessian.ps — evaluated
                    # fused as (D - ps) * (X @ (H^{-1} M2)), which avoids
                    # materializing the (n x k) asy.lin.rep matrix (same
                    # algebra, one O(n*k) matvec instead of an O(n*k^2)
                    # matmul). psi-scale, converted to phi via /n_all_panel.
                    # Subtract: R adds it to inf.control, and att = treat - control.
                    corr = (D_all - pscore_all) * (X_all_int @ (H_psi_inv @ M2))
                    inf_func = inf_func - corr / n_all_panel

                # SE from the same IF that feeds aggregation. The prior
                # plug-in sqrt(var_t/n_t + weighted_var_c) used a weighted
                # POPULATION variance for the control leg — never scaled by
                # an effective sample size — inflating per-cell SEs ~7x
                # versus R's std_ipw_did_panel.
                var_psi = np.sum(inf_func**2)
                if not np.isfinite(var_psi):
                    se = float("nan")
                elif var_psi > 0:
                    se = float(np.sqrt(var_psi))
                else:
                    se = 0.0
        else:
            # Unconditional IPW (reduces to difference in means)
            if sw_treated is not None:
                # Survey-weighted difference in means
                sw_t_norm = sw_treated / np.sum(sw_treated)
                sw_c_norm = sw_control / np.sum(sw_control)
                mu_t = float(np.sum(sw_t_norm * treated_change))
                mu_c = float(np.sum(sw_c_norm * control_change))
                att = mu_t - mu_c

                inf_treated = sw_t_norm * (treated_change - mu_t)
                inf_control = -sw_c_norm * (control_change - mu_c)
                inf_func = np.concatenate([inf_treated, inf_control])

                se = (
                    float(np.sqrt(np.sum(inf_treated**2) + np.sum(inf_control**2)))
                    if (n_t > 0 and n_c > 0)
                    else 0.0
                )
            else:
                mu_t = float(np.mean(treated_change))
                mu_c = float(np.mean(control_change))
                att = mu_t - mu_c

                # Influence function (for aggregation)
                inf_treated = (treated_change - mu_t) / n_t
                inf_control = -(control_change - mu_c) / n_c
                inf_func = np.concatenate([inf_treated, inf_control])

                # SE from the same IF that feeds aggregation. The prior
                # "adjusted" plug-in var_c*(1-p_treat)/(n_c*p_treat) equals
                # R's std_ipw_did_panel analytical SE only at p_treat = 0.5;
                # with a constant propensity the IF sum IS R's SE (no
                # estimated PS parameter, so no correction term).
                se = float(np.sqrt(np.sum(inf_func**2))) if (n_t > 0 and n_c > 0) else 0.0

        return att, se, inf_func

    def _doubly_robust(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
        X_treated: Optional[np.ndarray] = None,
        X_control: Optional[np.ndarray] = None,
        pscore_cache: Optional[Dict] = None,
        pscore_key: Optional[Any] = None,
        sw_treated: Optional[np.ndarray] = None,
        sw_control: Optional[np.ndarray] = None,
        sw_all: Optional[np.ndarray] = None,
        context_label: str = "",
        epv_diagnostics_out: Optional[dict] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using doubly robust estimation.

        With covariates:
        Combines outcome regression and IPW for double robustness.
        The estimator is consistent if either the outcome model OR
        the propensity model is correctly specified.

        ATT_DR = (1/n_t) * sum_i[D_i * (Y_i - m(X_i))]
               + (1/n_t) * sum_i[(1-D_i) * w_i * (m(X_i) - Y_i)]

        where m(X) is the outcome model and w_i are IPW weights.

        Without covariates:
        Reduces to simple difference in means.

        Parameters
        ----------
        sw_treated, sw_control, sw_all : np.ndarray, optional
            Survey weights for treated, control, and all units.
        """
        n_t = len(treated_change)
        n_c = len(control_change)

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Doubly robust estimation with covariates
            ps_fallback_used = False
            # Step 1: Outcome regression - fit E[Delta Y | X] on control via the
            # shared scale-robust solver (`_linear_regression` -> `solve_ols` ->
            # column-equilibrated SVD/gelsd; matches TripleDifference and R's
            # lm()/QR). This replaces the prior `cho_solve(X'X)` cache fast path,
            # which was NOT scale-equilibrated (a large-scale covariate would
            # corrupt the OR fit). `solve_ols` adds the intercept and handles
            # rank deficiency; we only need beta for prediction (m_treated,
            # m_control), so zero the NaN (dropped-column) coefficients — a
            # dropped covariate contributes 0 to the projection.
            X_control_with_intercept = np.column_stack([np.ones(n_c), X_control])
            beta, _ = _linear_regression(
                X_control,
                control_change,
                rank_deficient_action=self.rank_deficient_action,
                weights=sw_control,
            )
            beta = np.where(np.isnan(beta), 0.0, beta)

            # Predict counterfactual for both treated and control
            X_treated_with_intercept = np.column_stack([np.ones(n_t), X_treated])
            m_treated = np.dot(X_treated_with_intercept, beta)
            m_control = np.dot(X_control_with_intercept, beta)

            # Step 2: Propensity score estimation
            # Check propensity score cache
            cached_pscore = None
            if pscore_cache is not None and pscore_key is not None:
                cached_pscore = pscore_cache.get(pscore_key)

            if cached_pscore is not None:
                beta_logistic, cached_diag = cached_pscore
                X_all = np.vstack([X_treated, X_control])
                X_all_with_intercept = np.column_stack([np.ones(n_t + n_c), X_all])
                z = np.dot(X_all_with_intercept, beta_logistic)
                z = np.clip(z, -500, 500)
                pscore = 1 / (1 + np.exp(-z))
                if epv_diagnostics_out is not None and cached_diag:
                    epv_diagnostics_out.update(cached_diag)
            else:
                X_all = np.vstack([X_treated, X_control])
                D = np.concatenate([np.ones(n_t), np.zeros(n_c)])

                diag = {}
                try:
                    beta_logistic, pscore = solve_logit(
                        X_all,
                        D,
                        rank_deficient_action=self.rank_deficient_action,
                        weights=sw_all,
                        epv_threshold=self.epv_threshold,
                        context_label=context_label,
                        diagnostics_out=diag,
                    )
                    _check_propensity_diagnostics(pscore, self.pscore_trim)
                    if pscore_cache is not None and pscore_key is not None:
                        beta_clean = np.where(np.isfinite(beta_logistic), beta_logistic, 0.0)
                        pscore_cache[pscore_key] = (beta_clean, diag)
                except (np.linalg.LinAlgError, ValueError):
                    if self.pscore_fallback == "error" or self.rank_deficient_action == "error":
                        raise
                    # Fallback to unconditional if logistic regression fails
                    ctx = f" for {context_label}" if context_label else ""
                    warnings.warn(
                        f"Propensity score estimation failed{ctx}. "
                        f"Falling back to unconditional propensity "
                        f"(propensity model ignores covariates; outcome "
                        f"regression still uses them). "
                        f"Consider estimation_method='reg' to avoid "
                        f"propensity scores entirely.",
                        UserWarning,
                        stacklevel=4,
                    )
                    if sw_all is not None:
                        pos = sw_all > 0
                        p_uc = float(np.average(D[pos], weights=sw_all[pos]))
                    else:
                        p_uc = n_t / (n_t + n_c)
                    pscore = np.full(len(D), p_uc)
                    ps_fallback_used = True
                if epv_diagnostics_out is not None and diag:
                    epv_diagnostics_out.update(diag)

            pscore_control = pscore[n_t:]

            # Clip propensity scores
            pscore_control = np.clip(pscore_control, self.pscore_trim, 1 - self.pscore_trim)

            if sw_treated is not None:
                # IPW weights compose with survey weights
                weights_control = sw_control * pscore_control / (1 - pscore_control)

                # Step 3: DR ATT (survey-weighted)
                sw_t_sum = np.sum(sw_treated)
                att_treated_part = float(
                    np.sum(sw_treated * (treated_change - m_treated)) / sw_t_sum
                )
                augmentation = float(
                    np.sum(weights_control * (m_control - control_change)) / sw_t_sum
                )
                att = att_treated_part + augmentation

                # Step 4: Influence function (survey-weighted DR)
                # Start with plug-in IF, then add nuisance parameter corrections
                # (Sant'Anna & Zhao 2020, Theorem 3.1)
                psi_treated = (sw_treated / sw_t_sum) * (treated_change - m_treated - att)
                psi_control = (weights_control / sw_t_sum) * (m_control - control_change)
                inf_func = np.concatenate([psi_treated, psi_control])

                if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
                    if not ps_fallback_used:
                        # --- PS IF correction (mirrors IPW L1929-1961) ---
                        # Accounts for propensity score estimation uncertainty
                        X_all_int = np.column_stack([np.ones(n_t + n_c), X_all])
                        pscore_treated_clipped = np.clip(
                            pscore[:n_t], self.pscore_trim, 1 - self.pscore_trim
                        )
                        pscore_all = np.concatenate([pscore_treated_clipped, pscore_control])

                        # PS IF correction — psi convention, convert to phi
                        n_all_panel = n_t + n_c
                        W_ps = pscore_all * (1 - pscore_all)
                        if sw_all is not None:
                            W_ps = W_ps * sw_all
                        H_psi = X_all_int.T @ (W_ps[:, None] * X_all_int) / n_all_panel
                        H_psi_inv = _safe_inv(H_psi, tracker=self._safe_inv_tracker)

                        D_all = np.concatenate([np.ones(n_t), np.zeros(n_c)])
                        score_ps = (D_all - pscore_all)[:, None] * X_all_int
                        if sw_all is not None:
                            score_ps = score_ps * sw_all[:, None]
                        # R: asy.lin.rep.ps = score.ps %*% Hessian.ps  (psi scale)
                        asy_lin_rep_psi = score_ps @ H_psi_inv

                        dr_resid_control = m_control - control_change
                        M2_dr = np.sum(
                            ((weights_control / sw_t_sum) * dr_resid_control)[:, None]
                            * X_all_int[n_t:],
                            axis=0,
                        )
                        inf_func = inf_func + (asy_lin_rep_psi @ M2_dr) / n_all_panel

                    # --- OR IF correction ---
                    # Accounts for outcome regression estimation uncertainty
                    X_c_int = X_control_with_intercept
                    W_diag = sw_control if sw_control is not None else np.ones(n_c)
                    XtWX = X_c_int.T @ (W_diag[:, None] * X_c_int)
                    bread = _safe_inv(XtWX, tracker=self._safe_inv_tracker)

                    # M1: dATT/dbeta — gradient of DR ATT w.r.t. OR parameters
                    X_t_int = X_treated_with_intercept
                    M1 = (
                        -np.sum(sw_treated[:, None] * X_t_int, axis=0)
                        + np.sum(weights_control[:, None] * X_c_int, axis=0)
                    ) / sw_t_sum

                    # OR asymptotic linear representation (control-only)
                    resid_c = control_change - m_control
                    asy_lin_rep_or = (W_diag * resid_c)[:, None] * X_c_int @ bread
                    # Apply to control portion only (treated contribute zero)
                    inf_func[n_t:] += asy_lin_rep_or @ M1

                # Recompute SE from corrected IF
                var_psi = np.sum(inf_func**2)
                # A rank-0 cell yields an all-NaN bread (and thus NaN inf_func);
                # propagate NaN rather than masking it as 0.0 via ``var_psi > 0``
                # (NaN > 0 is False). Partial deficiency keeps var_psi finite.
                if not np.isfinite(var_psi):
                    se = float("nan")
                elif var_psi > 0:
                    se = float(np.sqrt(var_psi))
                else:
                    se = 0.0
            else:
                # IPW weights for control: p(X) / (1 - p(X))
                weights_control = pscore_control / (1 - pscore_control)

                # Step 3: Doubly robust ATT
                att_treated_part = float(np.mean(treated_change - m_treated))
                augmentation = float(np.sum(weights_control * (m_control - control_change)) / n_t)
                att = att_treated_part + augmentation

                # Step 4: Influence function with nuisance IF corrections
                psi_treated = (treated_change - m_treated - att) / n_t
                psi_control = (weights_control * (m_control - control_change)) / n_t
                inf_func = np.concatenate([psi_treated, psi_control])

                if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
                    if not ps_fallback_used:
                        # --- PS IF correction — psi convention, convert to phi ---
                        n_all_panel = n_t + n_c
                        X_all_int = np.column_stack([np.ones(n_all_panel), X_all])
                        pscore_treated_clipped = np.clip(
                            pscore[:n_t], self.pscore_trim, 1 - self.pscore_trim
                        )
                        pscore_all = np.concatenate([pscore_treated_clipped, pscore_control])

                        W_ps = pscore_all * (1 - pscore_all)
                        H_psi = X_all_int.T @ (W_ps[:, None] * X_all_int) / n_all_panel
                        H_psi_inv = _safe_inv(H_psi, tracker=self._safe_inv_tracker)

                        D_all = np.concatenate([np.ones(n_t), np.zeros(n_c)])
                        score_ps = (D_all - pscore_all)[:, None] * X_all_int
                        # R: asy.lin.rep.ps = score.ps %*% Hessian.ps  (psi scale)
                        asy_lin_rep_psi = score_ps @ H_psi_inv

                        dr_resid_control = m_control - control_change
                        M2_dr = np.sum(
                            ((weights_control / n_t) * dr_resid_control)[:, None] * X_all_int[n_t:],
                            axis=0,
                        )
                        inf_func = inf_func + (asy_lin_rep_psi @ M2_dr) / n_all_panel

                    # --- OR IF correction ---
                    X_c_int = X_control_with_intercept
                    XtX = X_c_int.T @ X_c_int
                    bread = _safe_inv(XtX, tracker=self._safe_inv_tracker)

                    X_t_int = X_treated_with_intercept
                    M1 = (
                        -np.sum(X_t_int, axis=0)
                        + np.sum(weights_control[:, None] * X_c_int, axis=0)
                    ) / n_t

                    resid_c = control_change - m_control
                    asy_lin_rep_or = resid_c[:, None] * X_c_int @ bread
                    inf_func[n_t:] += asy_lin_rep_or @ M1

                # Recompute SE from corrected IF
                var_psi = np.sum(inf_func**2)
                # A rank-0 cell yields an all-NaN bread (and thus NaN inf_func);
                # propagate NaN rather than masking it as 0.0 via ``var_psi > 0``
                # (NaN > 0 is False). Partial deficiency keeps var_psi finite.
                if not np.isfinite(var_psi):
                    se = float("nan")
                elif var_psi > 0:
                    se = float(np.sqrt(var_psi))
                else:
                    se = 0.0
        else:
            # Without covariates, DR simplifies to difference in means
            if sw_treated is not None:
                sw_t_norm = sw_treated / np.sum(sw_treated)
                sw_c_norm = sw_control / np.sum(sw_control)
                mu_t = float(np.sum(sw_t_norm * treated_change))
                mu_c = float(np.sum(sw_c_norm * control_change))
                att = mu_t - mu_c

                inf_treated = sw_t_norm * (treated_change - mu_t)
                inf_control = -sw_c_norm * (control_change - mu_c)
                inf_func = np.concatenate([inf_treated, inf_control])

                se = (
                    float(np.sqrt(np.sum(inf_treated**2) + np.sum(inf_control**2)))
                    if (n_t > 0 and n_c > 0)
                    else 0.0
                )
            else:
                mu_t = float(np.mean(treated_change))
                mu_c = float(np.mean(control_change))
                att = mu_t - mu_c

                # Influence function for the DR estimator; without covariates DR
                # reduces to difference in means, so the IF matches the vectorized
                # no-covariate regression path (_compute_all_att_gt_vectorized).
                inf_treated = (treated_change - mu_t) / n_t
                inf_control = -(control_change - mu_c) / n_c
                inf_func = np.concatenate([inf_treated, inf_control])

                # SE from the same IF that feeds aggregation (DRDID
                # reg_did_panel/drdid_panel convention sqrt(sum(phi^2))). The
                # prior ddof=1 plug-in sqrt(var_t/n_t + var_c/n_c) deviated from
                # R by O(1/n) and from the reg/ipw/DR-covariate branches; the
                # aggregation and bootstrap already consumed this same IF.
                se = (
                    float(np.sqrt(np.sum(inf_treated**2) + np.sum(inf_control**2)))
                    if (n_t > 0 and n_c > 0)
                    else 0.0
                )

        return att, se, inf_func

    # =========================================================================
    # Repeated Cross-Section (RCS) methods
    # =========================================================================

    def _precompute_structures_rc(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        time_periods: List[Any],
        treatment_groups: List[Any],
        resolved_survey=None,
    ) -> PrecomputedData:
        """
        Pre-compute observation-level structures for repeated cross-section.

        Unlike the panel path, RCS does not pivot to wide format. Each
        observation is treated independently (no within-unit differencing).

        Returns
        -------
        PrecomputedData
            Dictionary with pre-computed structures (observation-level).
        """
        n_obs = len(df)

        # Observation-level arrays (no pivot)
        obs_time = df[time].values
        obs_outcome = df[outcome].values
        unit_cohorts = df[first_treat].values

        # "all_units" key holds integer observation indices for backward
        # compatibility with aggregation code
        all_units = np.arange(n_obs)

        # Pre-compute cohort masks (boolean arrays, observation-level)
        cohort_masks = {}
        for g in treatment_groups:
            cohort_masks[g] = unit_cohorts == g

        # Never-treated mask
        never_treated_mask = (unit_cohorts == 0) | (unit_cohorts == np.inf)

        # Period-to-column mapping (identity for RCS — used for base period checks)
        period_to_col = {t: i for i, t in enumerate(sorted(time_periods))}

        # Covariates (observation-level, not per-period)
        obs_covariates = None
        if covariates:
            obs_covariates = df[covariates].values

        # Survey weights (already per-observation for RCS)
        if resolved_survey is not None:
            survey_weights_arr = resolved_survey.weights.copy()
        else:
            survey_weights_arr = None

        # For RCS, the resolved survey is already per-observation
        resolved_survey_rc = resolved_survey

        # Fixed cohort masses used as aggregation weights (R's pg = n_g / N).
        # Count UNIQUE UNITS per cohort, not observations: for a genuine repeated
        # cross-section each obs is a distinct unit, so unique-unit-count ==
        # observation-count (a no-op); but for an unbalanced PANEL routed as RC
        # (allow_unbalanced_panel), a cohort's obs-count exceeds its unit-count,
        # and R `did::aggte` weights by the fixed UNIT cohort mass — so unique
        # units is the R-correct weight on both paths.
        rcs_cohort_masses = {}
        _units_per_cohort = df.groupby(first_treat)[unit].nunique()
        for g in treatment_groups:
            rcs_cohort_masses[g] = int(_units_per_cohort.get(g, 0))

        # Per-unit aggregation cohort-mass basis for the R pg (= n_g / N over
        # UNITS, incl. never-treated). Consumed by _get_agg_cache (fast-path pg)
        # and _aggregate_event_study (survey_cohort_weights) so the point
        # estimate AND the WIF weight by fixed unit cohort mass. No-op for a true
        # RC (each obs is a distinct unit => per-unit == per-obs); the fix for an
        # unbalanced panel routed as RC (allow_unbalanced_panel).
        _first_pos = df.reset_index(drop=True).drop_duplicates(subset=[unit]).index.values
        _unit_cohort_vals = unit_cohorts[_first_pos]
        _unit_w = (
            survey_weights_arr[_first_pos]
            if survey_weights_arr is not None
            else np.ones(len(_first_pos), dtype=np.float64)
        )
        agg_cohort_masses = {
            float(cv): float(np.sum(_unit_w[_unit_cohort_vals == cv]))
            for cv in np.unique(unit_cohorts)
        }
        agg_total_weight = float(np.sum(_unit_w))
        # Observations per unit, aligned to the per-observation IF index space.
        # The WIF is a per-UNIT quantity; on a panel routed as RC each of a
        # unit's observations carries its cohort's WIF, so the unit-clustered
        # sum over-counts by this factor unless the per-obs WIF is divided by it
        # (see _combined_if_fast). 1 for every obs on a true RC (no-op).
        obs_per_unit = df.groupby(unit)[unit].transform("size").to_numpy(dtype=np.float64)

        return {
            "all_units": all_units,
            "unit_to_idx": None,  # RCS: obs indices are positions
            "unit_cohorts": unit_cohorts,
            "canonical_size": n_obs,
            "is_panel": False,
            "obs_time": obs_time,
            "obs_outcome": obs_outcome,
            "obs_covariates": obs_covariates,
            "cohort_masks": cohort_masks,
            "never_treated_mask": never_treated_mask,
            "time_periods": time_periods,
            "period_to_col": period_to_col,
            "observed_sorted": sorted(period_to_col),
            "is_balanced": False,
            "survey_weights": survey_weights_arr,
            "resolved_survey": resolved_survey,
            "resolved_survey_unit": resolved_survey_rc,
            "df_survey": (
                resolved_survey_rc.df_survey
                if resolved_survey_rc is not None and hasattr(resolved_survey_rc, "df_survey")
                else None
            ),
            "rcs_cohort_masses": rcs_cohort_masses,
            "agg_cohort_masses": agg_cohort_masses,
            "agg_total_weight": agg_total_weight,
            "obs_per_unit": obs_per_unit,
        }

    def _compute_att_gt_rc(
        self,
        precomputed: PrecomputedData,
        g: Any,
        t: Any,
        covariates: Optional[List[str]],
        epv_diagnostics: Optional[Dict] = None,
    ) -> Tuple[
        Optional[float],
        float,
        int,
        int,
        Optional[Dict[str, Any]],
        Optional[float],
        Optional[float],
        Optional[str],
    ]:
        """
        Compute ATT(g,t) for repeated cross-section data.

        For RCS, the 2x2 DiD compares outcomes across two independent
        cross-sections (periods t and base period s) rather than
        within-unit changes.

        Returns
        -------
        att_gt : float or None
        se_gt : float
        n_treated : int (treated obs at period t)
        n_control : int (control obs at period t)
        inf_func_info : dict or None
        survey_weight_sum : float or None
        cohort_mass : float or None
            Aggregation weight (survey-weighted cohort mass); ``None`` on a
            non-estimable return.
        skip_reason : str or None
            When ``att_gt is None`` (non-estimable cell), the machine-readable
            reason (``"missing_period"`` / ``"zero_treated_control"`` /
            ``"zero_weight_mass"``); ``None`` on a successful return.
        """
        cohort_masks = precomputed["cohort_masks"]
        never_treated_mask = precomputed["never_treated_mask"]
        unit_cohorts = precomputed["unit_cohorts"]
        obs_time = precomputed["obs_time"]
        obs_outcome = precomputed["obs_outcome"]
        period_to_col = precomputed["period_to_col"]

        # Base period selection: positional (sorted-index) neighbor, matching
        # R did::att_gt (see _select_base_period). Same logic as the panel path.
        base_period_val = self._select_base_period(g, t, precomputed["observed_sorted"])

        if base_period_val is None or t not in period_to_col:
            return None, 0.0, 0, 0, None, None, None, "missing_period"

        # Treated mask = cohort g
        treated_mask = cohort_masks[g]

        # Control mask (same logic as panel)
        if self.control_group == "never_treated":
            control_mask = never_treated_mask
        else:  # not_yet_treated
            nyt_threshold = max(t, base_period_val) + self.anticipation
            control_mask = never_treated_mask | (
                (unit_cohorts > nyt_threshold) & (unit_cohorts != g)
            )

        # Period masks
        at_t = obs_time == t
        at_s = obs_time == base_period_val

        # 4 groups of observations
        treated_t = treated_mask & at_t
        treated_s = treated_mask & at_s
        control_t = control_mask & at_t
        control_s = control_mask & at_s

        n_gt = int(np.sum(treated_t))
        n_gs = int(np.sum(treated_s))
        n_ct = int(np.sum(control_t))
        n_cs = int(np.sum(control_s))

        if n_gt == 0 or n_ct == 0 or n_gs == 0 or n_cs == 0:
            return None, 0.0, n_gt, n_ct, None, None, None, "zero_treated_control"

        # Extract outcomes for each group
        y_gt = obs_outcome[treated_t]
        y_gs = obs_outcome[treated_s]
        y_ct = obs_outcome[control_t]
        y_cs = obs_outcome[control_s]

        # Survey weights
        survey_w = precomputed.get("survey_weights")
        sw_gt = survey_w[treated_t] if survey_w is not None else None
        sw_gs = survey_w[treated_s] if survey_w is not None else None
        sw_ct = survey_w[control_t] if survey_w is not None else None
        sw_cs = survey_w[control_s] if survey_w is not None else None

        # Guard against zero effective mass
        if sw_gt is not None:
            if np.sum(sw_gt) <= 0 or np.sum(sw_gs) <= 0:
                return None, 0.0, n_gt, n_ct, None, None, None, "zero_weight_mass"
            if np.sum(sw_ct) <= 0 or np.sum(sw_cs) <= 0:
                return None, 0.0, n_gt, n_ct, None, None, None, "zero_weight_mass"

        # Get covariates if specified
        obs_covariates = precomputed.get("obs_covariates")
        has_covariates = covariates is not None and obs_covariates is not None

        if has_covariates:
            # The flag definition above guarantees this (mypy can't track it).
            assert obs_covariates is not None
            X_gt = obs_covariates[treated_t]
            X_gs = obs_covariates[treated_s]
            X_ct = obs_covariates[control_t]
            X_cs = obs_covariates[control_s]

            # Check for NaN in covariates
            if (
                np.any(np.isnan(X_gt))
                or np.any(np.isnan(X_gs))
                or np.any(np.isnan(X_ct))
                or np.any(np.isnan(X_cs))
            ):
                warnings.warn(
                    f"Missing values in covariates for group {g}, time {t} (RCS). "
                    "Falling back to unconditional estimation.",
                    UserWarning,
                    stacklevel=3,
                )
                has_covariates = False

        if has_covariates and self.estimation_method == "reg":
            att, se, inf_func_all, idx_all = self._outcome_regression_rc(
                y_gt,
                y_gs,
                y_ct,
                y_cs,
                X_gt,
                X_gs,
                X_ct,
                X_cs,
                sw_gt=sw_gt,
                sw_gs=sw_gs,
                sw_ct=sw_ct,
                sw_cs=sw_cs,
            )
        elif has_covariates and self.estimation_method == "ipw":
            epv_diag: dict = {}
            att, se, inf_func_all, idx_all = self._ipw_estimation_rc(
                y_gt,
                y_gs,
                y_ct,
                y_cs,
                X_gt,
                X_gs,
                X_ct,
                X_cs,
                sw_gt=sw_gt,
                sw_gs=sw_gs,
                sw_ct=sw_ct,
                sw_cs=sw_cs,
                context_label=f"cohort g={g}",
                epv_diagnostics_out=epv_diag,
            )
            if epv_diagnostics is not None and epv_diag:
                epv_diagnostics[(g, t)] = epv_diag
        elif has_covariates and self.estimation_method == "dr":
            epv_diag = {}
            att, se, inf_func_all, idx_all = self._doubly_robust_rc(
                y_gt,
                y_gs,
                y_ct,
                y_cs,
                X_gt,
                X_gs,
                X_ct,
                X_cs,
                sw_gt=sw_gt,
                sw_gs=sw_gs,
                sw_ct=sw_ct,
                sw_cs=sw_cs,
                context_label=f"cohort g={g}",
                epv_diagnostics_out=epv_diag,
            )
            if epv_diagnostics is not None and epv_diag:
                epv_diagnostics[(g, t)] = epv_diag
        else:
            # No-covariates 2x2 DiD (all methods reduce to same)
            att, se, inf_func_all, idx_all = self._rc_2x2_did(
                y_gt,
                y_gs,
                y_ct,
                y_cs,
                treated_t,
                treated_s,
                control_t,
                control_s,
                sw_gt=sw_gt,
                sw_gs=sw_gs,
                sw_ct=sw_ct,
                sw_cs=sw_cs,
            )

        # Build influence function info
        # For RCS, treated_idx/control_idx combine obs from BOTH periods.
        # INVARIANT: the two period masks are disjoint (obs_time == t vs
        # == base period), so the concatenated index arrays stay
        # duplicate-free (fancy-+= scatter contract, see
        # staggered_aggregation._combined_if_fast).
        treated_idx = np.concatenate([np.where(treated_t)[0], np.where(treated_s)[0]])
        control_idx = np.concatenate([np.where(control_t)[0], np.where(control_s)[0]])

        n_treated_combined = len(treated_idx)
        inf_func_info = {
            "treated_idx": treated_idx,
            "control_idx": control_idx,
            "treated_inf": inf_func_all[:n_treated_combined],
            "control_inf": inf_func_all[n_treated_combined:],
        }

        sw_sum = float(np.sum(sw_gt)) if sw_gt is not None else None
        # n_treated = per-cell treated count at period t (for display).
        # cohort_mass = total treated across all periods (for aggregation weights).
        cohort_mass = precomputed.get("rcs_cohort_masses", {}).get(g, n_gt)
        return att, se, n_gt, n_ct, inf_func_info, sw_sum, cohort_mass, None

    def _rc_2x2_did(
        self,
        y_gt,
        y_gs,
        y_ct,
        y_cs,
        mask_gt,
        mask_gs,
        mask_ct,
        mask_cs,
        sw_gt=None,
        sw_gs=None,
        sw_ct=None,
        sw_cs=None,
    ):
        """
        Compute the basic 2x2 DiD for RCS (no covariates).

        ATT = (mean(Y_treated_t) - mean(Y_control_t))
            - (mean(Y_treated_s) - mean(Y_control_s))

        Returns (att, se, inf_func_concat, idx_concat) where inf_func_concat
        has treated obs (both periods) first, then control obs (both periods).
        """
        n_gt = len(y_gt)
        n_gs = len(y_gs)
        n_ct = len(y_ct)
        n_cs = len(y_cs)

        if sw_gt is not None:
            sw_gt_norm = sw_gt / np.sum(sw_gt)
            sw_gs_norm = sw_gs / np.sum(sw_gs)
            sw_ct_norm = sw_ct / np.sum(sw_ct)
            sw_cs_norm = sw_cs / np.sum(sw_cs)

            mu_gt = float(np.sum(sw_gt_norm * y_gt))
            mu_gs = float(np.sum(sw_gs_norm * y_gs))
            mu_ct = float(np.sum(sw_ct_norm * y_ct))
            mu_cs = float(np.sum(sw_cs_norm * y_cs))

            att = (mu_gt - mu_ct) - (mu_gs - mu_cs)

            # Influence function for 4 groups (survey-weighted)
            inf_gt = sw_gt_norm * (y_gt - mu_gt)
            inf_ct = -sw_ct_norm * (y_ct - mu_ct)
            inf_gs = -sw_gs_norm * (y_gs - mu_gs)
            inf_cs = sw_cs_norm * (y_cs - mu_cs)
        else:
            mu_gt = float(np.mean(y_gt))
            mu_gs = float(np.mean(y_gs))
            mu_ct = float(np.mean(y_ct))
            mu_cs = float(np.mean(y_cs))

            att = (mu_gt - mu_ct) - (mu_gs - mu_cs)

            # Influence function for 4 groups
            inf_gt = (y_gt - mu_gt) / n_gt
            inf_ct = -(y_ct - mu_ct) / n_ct
            inf_gs = -(y_gs - mu_gs) / n_gs
            inf_cs = (y_cs - mu_cs) / n_cs

        # Concatenate: treated (t then s), control (t then s)
        inf_treated = np.concatenate([inf_gt, inf_gs])
        inf_control = np.concatenate([inf_ct, inf_cs])
        inf_all = np.concatenate([inf_treated, inf_control])

        # SE from influence function
        se = float(np.sqrt(np.sum(inf_all**2)))

        idx_all = np.concatenate(
            [
                np.where(mask_gt)[0],
                np.where(mask_gs)[0],
                np.where(mask_ct)[0],
                np.where(mask_cs)[0],
            ]
        )

        return att, se, inf_all, idx_all

    def _outcome_regression_rc(
        self,
        y_gt,
        y_gs,
        y_ct,
        y_cs,
        X_gt,
        X_gs,
        X_ct,
        X_cs,
        sw_gt=None,
        sw_gs=None,
        sw_ct=None,
        sw_cs=None,
    ):
        """
        Cross-sectional outcome regression for ATT(g,t).

        Matches R DRDID::reg_did_rc (Sant'Anna & Zhao 2020, Eq 2.2).

        Two OLS models fit on controls (period t and base period s).
        Predictions made for ALL treated (both periods).
        OR correction pools ALL treated observations across both periods.

        IF convention
        -------------
        Intermediate terms use R's unnormalized psi_i convention throughout.
        R computes SE as ``sd(psi) / sqrt(n)``; with mean(psi) approx 0 this
        equals ``sqrt(sum(psi^2)) / n``.  At the end we convert to the
        library's pre-scaled phi_i = psi_i / n convention where
        ``se = sqrt(sum(phi^2))``, used by the aggregation/bootstrap layer.

        Returns (att, se, inf_func_concat, idx_concat).
        """
        n_gt = len(y_gt)
        n_gs = len(y_gs)
        n_ct = len(y_ct)
        n_cs = len(y_cs)
        n_all = n_gt + n_gs + n_ct + n_cs

        # --- Fit 2 OLS on control groups (period t and s separately) ---
        beta_t, resid_ct = _linear_regression(
            X_ct,
            y_ct,
            rank_deficient_action=self.rank_deficient_action,
            weights=sw_ct,
        )
        beta_t = np.where(np.isfinite(beta_t), beta_t, 0.0)

        beta_s, resid_cs = _linear_regression(
            X_cs,
            y_cs,
            rank_deficient_action=self.rank_deficient_action,
            weights=sw_cs,
        )
        beta_s = np.where(np.isfinite(beta_s), beta_s, 0.0)

        # --- Predict counterfactual for ALL treated (both periods) ---
        X_gt_int = np.column_stack([np.ones(n_gt), X_gt])
        X_gs_int = np.column_stack([np.ones(n_gs), X_gs])
        X_ct_int = np.column_stack([np.ones(n_ct), X_ct])
        X_cs_int = np.column_stack([np.ones(n_cs), X_cs])

        # mu_hat_{0,t}(X) and mu_hat_{0,s}(X) for each treated obs
        mu_post_gt = X_gt_int @ beta_t  # treated-post predicted at post model
        mu_pre_gt = X_gt_int @ beta_s  # treated-post predicted at pre model
        mu_post_gs = X_gs_int @ beta_t  # treated-pre predicted at post model
        mu_pre_gs = X_gs_int @ beta_s  # treated-pre predicted at pre model

        # --- Group weights (R: w.treat.pre, w.treat.post, w.cont = w.D) ---
        if sw_gt is not None:
            w_treat_post = sw_gt  # treated at t
            w_treat_pre = sw_gs  # treated at s
            w_D_gt = sw_gt  # ALL treated: t portion
            w_D_gs = sw_gs  # ALL treated: s portion
        else:
            w_treat_post = np.ones(n_gt)
            w_treat_pre = np.ones(n_gs)
            w_D_gt = np.ones(n_gt)
            w_D_gs = np.ones(n_gs)

        sum_w_treat_post = np.sum(w_treat_post)
        sum_w_treat_pre = np.sum(w_treat_pre)
        sum_w_D = np.sum(w_D_gt) + np.sum(w_D_gs)  # pool ALL treated

        # R: mean(w.treat.post), mean(w.treat.pre), mean(w.cont)
        mean_w_treat_post = sum_w_treat_post / n_all
        mean_w_treat_pre = sum_w_treat_pre / n_all
        mean_w_D = sum_w_D / n_all

        # --- Treated means (period-specific Hajek means) ---
        eta_treat_post = np.sum(w_treat_post * y_gt) / sum_w_treat_post
        eta_treat_pre = np.sum(w_treat_pre * y_gs) / sum_w_treat_pre

        # --- OR correction: pools ALL treated ---
        # R: out.y.post - out.y.pre for each treated obs
        or_diff_gt = mu_post_gt - mu_pre_gt  # treated at t
        or_diff_gs = mu_post_gs - mu_pre_gs  # treated at s
        eta_cont = (np.sum(w_D_gt * or_diff_gt) + np.sum(w_D_gs * or_diff_gs)) / sum_w_D

        # --- Point estimate ---
        att = float(eta_treat_post - eta_treat_pre - eta_cont)

        # =================================================================
        # Influence function in R's unnormalized psi convention
        # (R: reg_did_rc.R, psi = n * phi)
        # =================================================================

        # --- Treated psi (R: eta.treat.post, eta.treat.pre) ---
        # R: w.treat.post * (y - eta.treat.post) / mean(w.treat.post)
        psi_treat_post = w_treat_post * (y_gt - eta_treat_post) / mean_w_treat_post
        # R: w.treat.pre * (y - eta.treat.pre) / mean(w.treat.pre)
        psi_treat_pre = w_treat_pre * (y_gs - eta_treat_pre) / mean_w_treat_pre

        # --- Control psi: leading term (R: inf.cont.1) ---
        # R: w.cont * (or_diff - eta.cont)   [before /mean(w.cont)]
        psi_cont_1_gt = w_D_gt * (or_diff_gt - eta_cont)
        psi_cont_1_gs = w_D_gs * (or_diff_gs - eta_cont)

        # --- Control psi: estimation effect (R: inf.cont.2) ---
        # R: bread = solve(crossprod(X_ctrl, W * X_ctrl) / n)
        # Here bread is (X'WX)^{-1} (without /n), so asy_lin_rep already
        # absorbs the 1/n that R puts in its bread.  We compensate by using
        # R's colMeans (= sum/n_all) for M1, matching the product exactly.
        W_ct = sw_ct if sw_ct is not None else np.ones(n_ct)
        W_cs = sw_cs if sw_cs is not None else np.ones(n_cs)
        bread_t = _safe_inv(
            X_ct_int.T @ (W_ct[:, None] * X_ct_int),
            tracker=self._safe_inv_tracker,
        )
        bread_s = _safe_inv(
            X_cs_int.T @ (W_cs[:, None] * X_cs_int),
            tracker=self._safe_inv_tracker,
        )

        # R: M1 = colMeans(w.cont * out.x) = sum(w_D * X) / n_all
        M1 = (
            np.sum(w_D_gt[:, None] * X_gt_int, axis=0) + np.sum(w_D_gs[:, None] * X_gs_int, axis=0)
        ) / n_all

        # R: asy.lin.rep.ols  (per-obs OLS score * bread)
        asy_lin_rep_ols_t = (W_ct * resid_ct)[:, None] * X_ct_int @ bread_t
        asy_lin_rep_ols_s = (W_cs * resid_cs)[:, None] * X_cs_int @ bread_s

        # R: inf.cont.2.post = asy.lin.rep.ols_t %*% M1
        psi_cont_2_ct = asy_lin_rep_ols_t @ M1  # (n_ct,)
        # R: inf.cont.2.pre  = asy.lin.rep.ols_s %*% M1
        psi_cont_2_cs = asy_lin_rep_ols_s @ M1  # (n_cs,)

        # --- Assemble per-group psi ---
        # R: inf.treat = inf.treat.post - inf.treat.pre  (across groups)
        # R: inf.cont = (inf.cont.1 + inf.cont.2.post - inf.cont.2.pre) / mean(w.cont)
        # R: att.inf.func = inf.treat - inf.cont
        psi_gt = psi_treat_post - psi_cont_1_gt / mean_w_D
        psi_gs = -psi_treat_pre - psi_cont_1_gs / mean_w_D
        psi_ct = -psi_cont_2_ct / mean_w_D
        psi_cs = psi_cont_2_cs / mean_w_D

        psi_all = np.concatenate([psi_gt, psi_gs, psi_ct, psi_cs])

        # =================================================================
        # Convert to library convention: phi = psi / n_all
        # se = sqrt(sum(phi^2))  ==  sqrt(sum(psi^2)) / n_all
        # =================================================================
        inf_all = psi_all / n_all
        se = float(np.sqrt(np.sum(inf_all**2)))

        idx_all = None  # caller builds idx from masks
        return att, se, inf_all, idx_all

    def _ipw_estimation_rc(
        self,
        y_gt,
        y_gs,
        y_ct,
        y_cs,
        X_gt,
        X_gs,
        X_ct,
        X_cs,
        sw_gt=None,
        sw_gs=None,
        sw_ct=None,
        sw_cs=None,
        context_label: str = "",
        epv_diagnostics_out: Optional[dict] = None,
    ):
        """
        Cross-sectional IPW estimation for ATT(g,t).

        Propensity score P(G=g | X) estimated on pooled treated+control
        observations from both periods. Reweight controls in each period.

        IF convention
        -------------
        Intermediate terms use R's unnormalized psi_i convention throughout
        (R: ``ipw_did_rc``).  R computes SE as ``sd(psi) / sqrt(n)``.
        At the end we convert to the library's pre-scaled phi_i = psi_i / n
        convention where ``se = sqrt(sum(phi^2))``, used by the
        aggregation/bootstrap layer.

        Returns (att, se, inf_func_concat, idx_concat).
        """
        n_gt = len(y_gt)
        n_gs = len(y_gs)
        n_ct = len(y_ct)
        n_cs = len(y_cs)
        n_all = n_gt + n_gs + n_ct + n_cs

        # Pool treated and control for propensity score
        X_all = np.vstack([X_gt, X_gs, X_ct, X_cs])
        D_all = np.concatenate([np.ones(n_gt + n_gs), np.zeros(n_ct + n_cs)])

        sw_all = None
        if sw_gt is not None:
            sw_all = np.concatenate([sw_gt, sw_gs, sw_ct, sw_cs])

        ps_fallback_used = False
        diag = {}
        try:
            beta_logistic, pscore = solve_logit(
                X_all,
                D_all,
                rank_deficient_action=self.rank_deficient_action,
                weights=sw_all,
                epv_threshold=self.epv_threshold,
                context_label=context_label,
                diagnostics_out=diag,
            )
            _check_propensity_diagnostics(pscore, self.pscore_trim)
        except (np.linalg.LinAlgError, ValueError):
            if self.pscore_fallback == "error" or self.rank_deficient_action == "error":
                raise
            ctx = f" for {context_label}" if context_label else ""
            warnings.warn(
                f"Propensity score estimation failed{ctx} (RCS IPW). "
                f"Falling back to unconditional propensity "
                f"(all covariates dropped for this cell). "
                f"Consider estimation_method='reg' to avoid "
                f"propensity scores entirely.",
                UserWarning,
                stacklevel=4,
            )
            if sw_all is not None:
                pos = sw_all > 0
                p_treat = float(np.average(D_all[pos], weights=sw_all[pos]))
            else:
                p_treat = (n_gt + n_gs) / len(D_all)
            pscore = np.full(len(D_all), p_treat)
            ps_fallback_used = True
        if epv_diagnostics_out is not None and diag:
            epv_diagnostics_out.update(diag)

        # Clip propensity scores
        pscore = np.clip(pscore, self.pscore_trim, 1 - self.pscore_trim)

        # Split propensity scores (treated ps not used -- only control IPW weights)
        ps_ct = pscore[n_gt + n_gs : n_gt + n_gs + n_ct]
        ps_cs = pscore[n_gt + n_gs + n_ct :]

        # IPW weights for controls (R: w1.x = ps / (1 - ps))
        w_ct = ps_ct / (1 - ps_ct)
        w_cs = ps_cs / (1 - ps_cs)

        if sw_gt is not None:
            w_ct = sw_ct * w_ct
            w_cs = sw_cs * w_cs

        # R: mean(w.treat.post), mean(w.treat.pre), mean(w.ipw.ct), mean(w.ipw.cs)
        if sw_gt is not None:
            sum_w_treat_post = np.sum(sw_gt)
            sum_w_treat_pre = np.sum(sw_gs)
        else:
            sum_w_treat_post = float(n_gt)
            sum_w_treat_pre = float(n_gs)

        mean_w_treat_post = sum_w_treat_post / n_all
        mean_w_treat_pre = sum_w_treat_pre / n_all

        sum_w_ct = np.sum(w_ct)
        sum_w_cs = np.sum(w_cs)
        mean_w_ct = sum_w_ct / n_all
        mean_w_cs = sum_w_cs / n_all

        # Hajek-normalized weights (R normalizes by sum for point estimate)
        w_ct_norm = w_ct / sum_w_ct if sum_w_ct > 0 else w_ct
        w_cs_norm = w_cs / sum_w_cs if sum_w_cs > 0 else w_cs

        if sw_gt is not None:
            sw_gt_norm = sw_gt / sum_w_treat_post
            sw_gs_norm = sw_gs / sum_w_treat_pre
            mu_gt = float(np.sum(sw_gt_norm * y_gt))
            mu_gs = float(np.sum(sw_gs_norm * y_gs))
        else:
            mu_gt = float(np.mean(y_gt))
            mu_gs = float(np.mean(y_gs))

        mu_ct_ipw = float(np.sum(w_ct_norm * y_ct))
        mu_cs_ipw = float(np.sum(w_cs_norm * y_cs))

        att = (mu_gt - mu_ct_ipw) - (mu_gs - mu_cs_ipw)

        # =================================================================
        # Influence function in R's unnormalized psi convention
        # (R: ipw_did_rc.R, psi = n * phi)
        # =================================================================

        # --- Treated psi (R: eta.treat.post, eta.treat.pre) ---
        # R: w.treat.post * (y - eta.treat.post) / mean(w.treat.post)
        if sw_gt is not None:
            psi_gt = sw_gt * (y_gt - mu_gt) / mean_w_treat_post
            psi_gs = -sw_gs * (y_gs - mu_gs) / mean_w_treat_pre
        else:
            psi_gt = (y_gt - mu_gt) / mean_w_treat_post
            psi_gs = -(y_gs - mu_gs) / mean_w_treat_pre

        # --- Control psi (R: eta.cont.post, eta.cont.pre) ---
        # R: w.ipw * (y - eta.cont) / mean(w.ipw)
        psi_ct = -w_ct * (y_ct - mu_ct_ipw) / mean_w_ct if mean_w_ct > 0 else np.zeros(n_ct)
        psi_cs = w_cs * (y_cs - mu_cs_ipw) / mean_w_cs if mean_w_cs > 0 else np.zeros(n_cs)

        psi_all = np.concatenate([psi_gt, psi_gs, psi_ct, psi_cs])

        # Convert leading psi to phi: phi = psi / n_all
        inf_all = psi_all / n_all

        if not ps_fallback_used:
            # --- PS IF correction — psi convention, convert to phi ---
            X_all_int = np.column_stack([np.ones(n_all), X_all])

            W_ps = pscore * (1 - pscore)
            if sw_all is not None:
                W_ps = W_ps * sw_all
            # R: Hessian.ps = crossprod(X * sqrt(W)) / n
            H_psi = X_all_int.T @ (W_ps[:, None] * X_all_int) / n_all
            H_psi_inv = _safe_inv(H_psi, tracker=self._safe_inv_tracker)

            score_ps = (D_all - pscore)[:, None] * X_all_int
            if sw_all is not None:
                score_ps = score_ps * sw_all[:, None]
            # R: asy.lin.rep.ps = score.ps %*% Hessian.ps  (psi scale, O(1) per obs)
            asy_lin_rep_psi = score_ps @ H_psi_inv

            # PS nuisance correction in psi convention
            # R: M2 = colMeans(w_ipw * (y-mu) * X)
            ipw_resid_ct = w_ct_norm * (y_ct - mu_ct_ipw)
            ipw_resid_cs = w_cs_norm * (y_cs - mu_cs_ipw)

            ct_slice = slice(n_gt + n_gs, n_gt + n_gs + n_ct)
            cs_slice = slice(n_gt + n_gs + n_ct, None)

            M2 = np.zeros(X_all_int.shape[1])
            M2 += np.sum(ipw_resid_ct[:, None] * X_all_int[ct_slice], axis=0)
            M2 -= np.sum(ipw_resid_cs[:, None] * X_all_int[cs_slice], axis=0)

            # psi-scale correction, convert to phi
            # Subtract: R adds PS correction to inf.control, then att = treat - control
            inf_all = inf_all - (asy_lin_rep_psi @ M2) / n_all

        # =================================================================
        # SE from phi: se = sqrt(sum(phi^2))
        # Equivalent to R's sqrt(sum(psi^2)) / n when mean(psi) approx 0.
        # =================================================================
        se = float(np.sqrt(np.sum(inf_all**2)))

        idx_all = None
        return att, se, inf_all, idx_all

    def _doubly_robust_rc(
        self,
        y_gt,
        y_gs,
        y_ct,
        y_cs,
        X_gt,
        X_gs,
        X_ct,
        X_cs,
        sw_gt=None,
        sw_gs=None,
        sw_ct=None,
        sw_cs=None,
        context_label: str = "",
        epv_diagnostics_out: Optional[dict] = None,
    ):
        """
        Cross-sectional doubly robust estimation for ATT(g,t).

        Matches R DRDID::drdid_rc (Sant'Anna & Zhao 2020, Eq 3.1).
        Locally efficient DR estimator with 4 OLS fits (control pre/post,
        treated pre/post) plus propensity score.

        IF convention
        -------------
        Intermediate terms use R's unnormalized psi_i convention throughout
        (R: ``drdid_rc``).  R computes SE as ``sd(psi) / sqrt(n)``.
        At the end we convert to the library's pre-scaled phi_i = psi_i / n
        convention where ``se = sqrt(sum(phi^2))``, used by the
        aggregation/bootstrap layer.

        Returns (att, se, inf_func_concat, idx_concat).
        """
        n_gt = len(y_gt)
        n_gs = len(y_gs)
        n_ct = len(y_ct)
        n_cs = len(y_cs)
        n_all = n_gt + n_gs + n_ct + n_cs

        # =====================================================================
        # 1. Outcome regression: 4 OLS fits
        # =====================================================================
        # Control OLS: E[Y|X, D=0, T=t] and E[Y|X, D=0, T=s]
        beta_ct, resid_ct = _linear_regression(
            X_ct,
            y_ct,
            rank_deficient_action=self.rank_deficient_action,
            weights=sw_ct,
        )
        beta_ct = np.where(np.isfinite(beta_ct), beta_ct, 0.0)

        beta_cs, resid_cs = _linear_regression(
            X_cs,
            y_cs,
            rank_deficient_action=self.rank_deficient_action,
            weights=sw_cs,
        )
        beta_cs = np.where(np.isfinite(beta_cs), beta_cs, 0.0)

        # Treated OLS: E[Y|X, D=1, T=t] and E[Y|X, D=1, T=s]
        beta_gt, resid_gt = _linear_regression(
            X_gt,
            y_gt,
            rank_deficient_action=self.rank_deficient_action,
            weights=sw_gt,
        )
        beta_gt = np.where(np.isfinite(beta_gt), beta_gt, 0.0)

        beta_gs, resid_gs = _linear_regression(
            X_gs,
            y_gs,
            rank_deficient_action=self.rank_deficient_action,
            weights=sw_gs,
        )
        beta_gs = np.where(np.isfinite(beta_gs), beta_gs, 0.0)

        # Intercept-augmented design matrices
        X_gt_int = np.column_stack([np.ones(n_gt), X_gt])
        X_gs_int = np.column_stack([np.ones(n_gs), X_gs])
        X_ct_int = np.column_stack([np.ones(n_ct), X_ct])
        X_cs_int = np.column_stack([np.ones(n_cs), X_cs])

        # Control OR predictions used downstream
        mu0_post_gt = X_gt_int @ beta_ct  # mu_{0,1}(X) for treated-post
        mu0_pre_gt = X_gt_int @ beta_cs  # mu_{0,0}(X) for treated-post
        mu0_post_gs = X_gs_int @ beta_ct  # mu_{0,1}(X) for treated-pre
        mu0_pre_gs = X_gs_int @ beta_cs  # mu_{0,0}(X) for treated-pre
        mu0_post_ct = X_ct_int @ beta_ct  # mu_{0,1}(X) for control-post
        mu0_pre_cs = X_cs_int @ beta_cs  # mu_{0,0}(X) for control-pre
        # The full DRDID R-convention grid also names mu0_pre_ct and
        # mu0_post_cs. They are intentionally not materialized here because no
        # residual, adjustment, or influence-function term consumes them.

        # Treated OR predictions for all groups (for local efficiency adjustment)
        mu1_post_gt = X_gt_int @ beta_gt  # mu_{1,1}(X) for treated-post
        mu1_pre_gt = X_gt_int @ beta_gs  # mu_{1,0}(X) for treated-post
        mu1_post_gs = X_gs_int @ beta_gt  # mu_{1,1}(X) for treated-pre
        mu1_pre_gs = X_gs_int @ beta_gs  # mu_{1,0}(X) for treated-pre

        # mu_{0,Y}(T_i, X_i): control OR evaluated at own period
        mu0Y_gt = mu0_post_gt  # treated-post: use post control model
        mu0Y_gs = mu0_pre_gs  # treated-pre: use pre control model
        mu0Y_ct = mu0_post_ct  # control-post: use post control model
        mu0Y_cs = mu0_pre_cs  # control-pre: use pre control model

        # =====================================================================
        # 2. Propensity score
        # =====================================================================
        X_all = np.vstack([X_gt, X_gs, X_ct, X_cs])
        D_all = np.concatenate([np.ones(n_gt + n_gs), np.zeros(n_ct + n_cs)])
        sw_all = None
        if sw_gt is not None:
            sw_all = np.concatenate([sw_gt, sw_gs, sw_ct, sw_cs])

        ps_fallback_used = False
        diag = {}
        try:
            beta_logistic, pscore = solve_logit(
                X_all,
                D_all,
                rank_deficient_action=self.rank_deficient_action,
                weights=sw_all,
                epv_threshold=self.epv_threshold,
                context_label=context_label,
                diagnostics_out=diag,
            )
            _check_propensity_diagnostics(pscore, self.pscore_trim)
        except (np.linalg.LinAlgError, ValueError):
            if self.pscore_fallback == "error" or self.rank_deficient_action == "error":
                raise
            ctx = f" for {context_label}" if context_label else ""
            warnings.warn(
                f"Propensity score estimation failed{ctx} (RCS DR). "
                f"Falling back to unconditional propensity "
                f"(propensity model ignores covariates; outcome "
                f"regression still uses them). "
                f"Consider estimation_method='reg' to avoid "
                f"propensity scores entirely.",
                UserWarning,
                stacklevel=4,
            )
            if sw_all is not None:
                pos = sw_all > 0
                p_treat = float(np.average(D_all[pos], weights=sw_all[pos]))
            else:
                p_treat = (n_gt + n_gs) / len(D_all)
            pscore = np.full(len(D_all), p_treat)
            ps_fallback_used = True
        if epv_diagnostics_out is not None and diag:
            epv_diagnostics_out.update(diag)

        pscore = np.clip(pscore, self.pscore_trim, 1 - self.pscore_trim)

        # Split control propensity scores per group.
        ps_ct = pscore[n_gt + n_gs : n_gt + n_gs + n_ct]
        ps_cs = pscore[n_gt + n_gs + n_ct :]
        # The symmetric treated slices ps_gt and ps_gs are intentionally not
        # materialized: only control propensity slices feed IPW control weights
        # (the propensity-estimation IF correction consumes the full pscore array directly).

        # =====================================================================
        # 3. Group weights and R-convention means
        # =====================================================================
        if sw_gt is not None:
            w_treat_post = sw_gt
            w_treat_pre = sw_gs
            w_D_gt = sw_gt
            w_D_gs = sw_gs
        else:
            w_treat_post = np.ones(n_gt)
            w_treat_pre = np.ones(n_gs)
            w_D_gt = np.ones(n_gt)
            w_D_gs = np.ones(n_gs)

        sum_w_treat_post = np.sum(w_treat_post)
        sum_w_treat_pre = np.sum(w_treat_pre)
        sum_w_D = np.sum(w_D_gt) + np.sum(w_D_gs)

        # R: mean(w) = sum(w) / n  -- used in psi normalizers
        mean_w_treat_post = sum_w_treat_post / n_all
        mean_w_treat_pre = sum_w_treat_pre / n_all
        mean_w_D = sum_w_D / n_all

        # IPW control weights: sw * ps/(1-ps) for controls
        w_ipw_ct = ps_ct / (1 - ps_ct)
        w_ipw_cs = ps_cs / (1 - ps_cs)
        if sw_ct is not None:
            w_ipw_ct = sw_ct * w_ipw_ct
            w_ipw_cs = sw_cs * w_ipw_cs

        sum_w_ipw_ct = np.sum(w_ipw_ct)
        sum_w_ipw_cs = np.sum(w_ipw_cs)
        mean_w_ipw_ct = sum_w_ipw_ct / n_all
        mean_w_ipw_cs = sum_w_ipw_cs / n_all

        # =====================================================================
        # 4. Point estimate: tau_1 (AIPW using control ORs)
        # =====================================================================
        # Hajek-normalized means of (y - mu0Y) per group
        eta_treat_post = np.sum(w_treat_post * (y_gt - mu0Y_gt)) / sum_w_treat_post
        eta_treat_pre = np.sum(w_treat_pre * (y_gs - mu0Y_gs)) / sum_w_treat_pre

        eta_cont_post = (
            np.sum(w_ipw_ct * (y_ct - mu0Y_ct)) / sum_w_ipw_ct if sum_w_ipw_ct > 0 else 0.0
        )
        eta_cont_pre = (
            np.sum(w_ipw_cs * (y_cs - mu0Y_cs)) / sum_w_ipw_cs if sum_w_ipw_cs > 0 else 0.0
        )

        tau_1 = (eta_treat_post - eta_cont_post) - (eta_treat_pre - eta_cont_pre)

        # =====================================================================
        # 5. Point estimate: local efficiency adjustment (tau_2)
        # =====================================================================
        # Differences mu_{1,t}(X) - mu_{0,t}(X) for treated obs
        or_diff_post_gt = mu1_post_gt - mu0_post_gt  # at treated-post
        or_diff_post_gs = mu1_post_gs - mu0_post_gs  # at treated-pre
        or_diff_pre_gt = mu1_pre_gt - mu0_pre_gt  # at treated-post
        or_diff_pre_gs = mu1_pre_gs - mu0_pre_gs  # at treated-pre

        # att_d_post = mean(w_D * (mu1_post - mu0_post)) / mean(w_D) -- all treated
        att_d_post = (np.sum(w_D_gt * or_diff_post_gt) + np.sum(w_D_gs * or_diff_post_gs)) / sum_w_D
        # att_dt1_post -- treated-post only
        att_dt1_post = np.sum(w_treat_post * or_diff_post_gt) / sum_w_treat_post
        # att_d_pre -- all treated
        att_d_pre = (np.sum(w_D_gt * or_diff_pre_gt) + np.sum(w_D_gs * or_diff_pre_gs)) / sum_w_D
        # att_dt0_pre -- treated-pre only
        att_dt0_pre = np.sum(w_treat_pre * or_diff_pre_gs) / sum_w_treat_pre

        tau_2 = (att_d_post - att_dt1_post) - (att_d_pre - att_dt0_pre)

        att = float(tau_1 + tau_2)

        # =====================================================================
        # 6. Influence function in R's unnormalized psi convention
        #    (R: drdid_rc.R, psi = n * phi)
        # =====================================================================

        # --- tau_1: treated psi (R: eta.treat.post / mean(w.treat.post)) ---
        # R: w.treat.post * (y - mu0Y - eta.treat.post) / mean(w.treat.post)
        psi_treat_post = w_treat_post * (y_gt - mu0Y_gt - eta_treat_post) / mean_w_treat_post
        psi_treat_pre = w_treat_pre * (y_gs - mu0Y_gs - eta_treat_pre) / mean_w_treat_pre

        # --- tau_1: control psi (R: eta.cont.post / mean(w.ipw)) ---
        # R: w.ipw * (y - mu0Y - eta.cont) / mean(w.ipw)
        psi_cont_post_ct = (
            w_ipw_ct * (y_ct - mu0Y_ct - eta_cont_post) / mean_w_ipw_ct
            if mean_w_ipw_ct > 0
            else np.zeros(n_ct)
        )
        psi_cont_pre_cs = (
            w_ipw_cs * (y_cs - mu0Y_cs - eta_cont_pre) / mean_w_ipw_cs
            if mean_w_ipw_cs > 0
            else np.zeros(n_cs)
        )

        # tau_1 psi per group
        psi_gt_tau1 = psi_treat_post
        psi_gs_tau1 = -psi_treat_pre
        psi_ct_tau1 = -psi_cont_post_ct
        psi_cs_tau1 = psi_cont_pre_cs

        # =====================================================================
        # 7. tau_2 leading terms (R: att.d.post, att.dt1.post, etc.)
        # =====================================================================
        # R: w.D * (or_diff - att.d.post) / mean(w.D)
        psi_d_post_gt = w_D_gt * (or_diff_post_gt - att_d_post) / mean_w_D
        psi_d_post_gs = w_D_gs * (or_diff_post_gs - att_d_post) / mean_w_D
        # R: w.treat.post * (or_diff - att.dt1.post) / mean(w.treat.post)
        psi_dt1_post = w_treat_post * (or_diff_post_gt - att_dt1_post) / mean_w_treat_post
        # R: w.D * (or_diff_pre - att.d.pre) / mean(w.D)
        psi_d_pre_gt = w_D_gt * (or_diff_pre_gt - att_d_pre) / mean_w_D
        psi_d_pre_gs = w_D_gs * (or_diff_pre_gs - att_d_pre) / mean_w_D
        # R: w.treat.pre * (or_diff_pre - att.dt0.pre) / mean(w.treat.pre)
        psi_dt0_pre = w_treat_pre * (or_diff_pre_gs - att_dt0_pre) / mean_w_treat_pre

        # tau_2 psi per group (controls contribute zero)
        psi_gt_tau2 = (psi_d_post_gt - psi_dt1_post) - psi_d_pre_gt
        psi_gs_tau2 = psi_d_post_gs - (-psi_dt0_pre + psi_d_pre_gs)

        # =====================================================================
        # 8. Combined plug-in psi (before nuisance corrections)
        # =====================================================================
        psi_gt = psi_gt_tau1 + psi_gt_tau2
        psi_gs = psi_gs_tau1 + psi_gs_tau2
        psi_ct = psi_ct_tau1
        psi_cs = psi_cs_tau1

        psi_all = np.concatenate([psi_gt, psi_gs, psi_ct, psi_cs])

        # =================================================================
        # Convert leading psi to library phi convention: phi = psi / n_all
        # =================================================================
        inf_all = psi_all / n_all

        # =====================================================================
        # 9. PS nuisance correction — psi convention, convert to phi
        # =====================================================================
        X_all_int = np.column_stack([np.ones(n_all), X_all])
        if not ps_fallback_used:
            W_ps = pscore * (1 - pscore)
            if sw_all is not None:
                W_ps = W_ps * sw_all
            # R: Hessian.ps = crossprod(X * sqrt(W)) / n
            H_psi = X_all_int.T @ (W_ps[:, None] * X_all_int) / n_all
            H_psi_inv = _safe_inv(H_psi, tracker=self._safe_inv_tracker)

            score_ps = (D_all - pscore)[:, None] * X_all_int
            if sw_all is not None:
                score_ps = score_ps * sw_all[:, None]
            # R: asy.lin.rep.ps = score.ps %*% Hessian.ps  (psi scale, O(1) per obs)
            asy_lin_rep_psi = score_ps @ H_psi_inv

            # R: M2 = colMeans(w_ipw * dr_resid / mean(w_ipw) * X)
            ct_slice = slice(n_gt + n_gs, n_gt + n_gs + n_ct)
            cs_slice = slice(n_gt + n_gs + n_ct, None)

            dr_resid_ct = y_ct - mu0Y_ct - eta_cont_post
            dr_resid_cs = y_cs - mu0Y_cs - eta_cont_pre

            M2 = np.zeros(X_all_int.shape[1])
            if sum_w_ipw_ct > 0:
                M2 -= np.sum(
                    ((w_ipw_ct * dr_resid_ct / sum_w_ipw_ct)[:, None] * X_all_int[ct_slice]),
                    axis=0,
                )
            if sum_w_ipw_cs > 0:
                M2 += np.sum(
                    ((w_ipw_cs * dr_resid_cs / sum_w_ipw_cs)[:, None] * X_all_int[cs_slice]),
                    axis=0,
                )

            # psi-scale correction, convert to phi
            inf_all = inf_all + (asy_lin_rep_psi @ M2) / n_all

        # =====================================================================
        # 10. Control OR nuisance corrections (phi-scale)
        # =====================================================================
        W_ct_vals = sw_ct if sw_ct is not None else np.ones(n_ct)
        W_cs_vals = sw_cs if sw_cs is not None else np.ones(n_cs)
        bread_ct = _safe_inv(
            X_ct_int.T @ (W_ct_vals[:, None] * X_ct_int),
            tracker=self._safe_inv_tracker,
        )
        bread_cs = _safe_inv(
            X_cs_int.T @ (W_cs_vals[:, None] * X_cs_int),
            tracker=self._safe_inv_tracker,
        )

        # R: asy.lin.rep.ols  (per-obs OLS score * bread)
        asy_lin_rep_ct = (W_ct_vals * resid_ct)[:, None] * X_ct_int @ bread_ct
        asy_lin_rep_cs = (W_cs_vals * resid_cs)[:, None] * X_cs_int @ bread_cs

        # M1 for control-post model (beta_ct): gradient from tau_1 + tau_2
        # tau_1: -w_treat_post*X/sum_w_treat_post (eta_treat_post via mu0Y_gt)
        #        +w_ipw_ct*X/sum_w_ipw_ct (eta_cont_post via mu0Y_ct)
        # tau_2: -w_D*X/sum_w_D (att_d_post via mu0_post at all treated)
        #        +w_treat_post*X/sum_w_treat_post (att_dt1_post via mu0_post)
        M1_ct = np.zeros(X_all_int.shape[1])
        M1_ct -= np.sum(w_treat_post[:, None] * X_gt_int, axis=0) / sum_w_treat_post
        if sum_w_ipw_ct > 0:
            M1_ct += np.sum(w_ipw_ct[:, None] * X_ct_int, axis=0) / sum_w_ipw_ct
        M1_ct -= (
            np.sum(w_D_gt[:, None] * X_gt_int, axis=0) + np.sum(w_D_gs[:, None] * X_gs_int, axis=0)
        ) / sum_w_D
        M1_ct += np.sum(w_treat_post[:, None] * X_gt_int, axis=0) / sum_w_treat_post

        # M1 for control-pre model (beta_cs)
        M1_cs = np.zeros(X_all_int.shape[1])
        M1_cs += np.sum(w_treat_pre[:, None] * X_gs_int, axis=0) / sum_w_treat_pre
        if sum_w_ipw_cs > 0:
            M1_cs -= np.sum(w_ipw_cs[:, None] * X_cs_int, axis=0) / sum_w_ipw_cs
        M1_cs += (
            np.sum(w_D_gt[:, None] * X_gt_int, axis=0) + np.sum(w_D_gs[:, None] * X_gs_int, axis=0)
        ) / sum_w_D
        M1_cs -= np.sum(w_treat_pre[:, None] * X_gs_int, axis=0) / sum_w_treat_pre

        inf_all[n_gt + n_gs : n_gt + n_gs + n_ct] += asy_lin_rep_ct @ M1_ct
        inf_all[n_gt + n_gs + n_ct :] += asy_lin_rep_cs @ M1_cs

        # =====================================================================
        # 11. Treated OR nuisance corrections (phi-scale)
        # =====================================================================
        W_gt_vals = sw_gt if sw_gt is not None else np.ones(n_gt)
        W_gs_vals = sw_gs if sw_gs is not None else np.ones(n_gs)
        bread_gt = _safe_inv(
            X_gt_int.T @ (W_gt_vals[:, None] * X_gt_int),
            tracker=self._safe_inv_tracker,
        )
        bread_gs = _safe_inv(
            X_gs_int.T @ (W_gs_vals[:, None] * X_gs_int),
            tracker=self._safe_inv_tracker,
        )

        asy_lin_rep_gt = (W_gt_vals * resid_gt)[:, None] * X_gt_int @ bread_gt
        asy_lin_rep_gs = (W_gs_vals * resid_gs)[:, None] * X_gs_int @ bread_gs

        # M1 for treated-post model (beta_gt): mu_{1,1}(X)
        # From att_d_post: +w_D*X/sum_w_D (all treated)
        # From att_dt1_post: -w_treat_post*X/sum_w_treat_post (treated-post)
        M1_gt = np.zeros(X_all_int.shape[1])
        M1_gt += (
            np.sum(w_D_gt[:, None] * X_gt_int, axis=0) + np.sum(w_D_gs[:, None] * X_gs_int, axis=0)
        ) / sum_w_D
        M1_gt -= np.sum(w_treat_post[:, None] * X_gt_int, axis=0) / sum_w_treat_post

        # M1 for treated-pre model (beta_gs): mu_{1,0}(X)
        # From att_d_pre: -w_D*X/sum_w_D
        # From att_dt0_pre: +w_treat_pre*X/sum_w_treat_pre
        M1_gs = np.zeros(X_all_int.shape[1])
        M1_gs -= (
            np.sum(w_D_gt[:, None] * X_gt_int, axis=0) + np.sum(w_D_gs[:, None] * X_gs_int, axis=0)
        ) / sum_w_D
        M1_gs += np.sum(w_treat_pre[:, None] * X_gs_int, axis=0) / sum_w_treat_pre

        inf_all[:n_gt] += asy_lin_rep_gt @ M1_gt
        inf_all[n_gt : n_gt + n_gs] += asy_lin_rep_gs @ M1_gs

        # =================================================================
        # SE from phi: se = sqrt(sum(phi^2))
        # Equivalent to R's sqrt(sum(psi^2)) / n when mean(psi) approx 0.
        # =================================================================
        se = float(np.sqrt(np.sum(inf_all**2)))

        idx_all = None
        return att, se, inf_all, idx_all

    @staticmethod
    def _validate_vcov_type(vcov_type: str) -> None:
        """Validate ``vcov_type`` membership against CS's narrow contract.

        Called from ``__init__`` and from ``fit()`` (so sklearn-style
        ``set_params(vcov_type=...)`` mutations are re-checked at use
        time rather than silently passing a bad value through to Results).
        """
        _accepted_vcov = {"hc1"}
        _deferred_vcov = {"conley"}
        _if_incompatible_vcov = {"classical", "hc2", "hc2_bm"}
        if vcov_type in _if_incompatible_vcov:
            raise ValueError(
                f"CallawaySantAnna(vcov_type={vcov_type!r}) is rejected: "
                "CS uses influence-function-based variance per Callaway & "
                "Sant'Anna (2021); the analytical-sandwich families "
                "{'classical', 'hc2', 'hc2_bm'} are defined on a single "
                "regression's hat matrix, and CS's per-(g,t) doubly-robust "
                "/ IPW / outcome-regression structure has no equivalent "
                "single design matrix to compute hat-matrix leverage or "
                "Bell-McCaffrey Satterthwaite DOF on. The rejection is "
                "library-architectural, not paper-prescribed. Use "
                "vcov_type='hc1' (the default) with cluster=<col> for "
                "cluster-robust inference. See docs/methodology/REGISTRY.md "
                "'IF-based variance estimators vs analytical-sandwich "
                "estimators' for the structural taxonomy."
            )
        if vcov_type in _deferred_vcov:
            raise ValueError(
                f"CallawaySantAnna(vcov_type={vcov_type!r}) is not yet "
                "supported: spatial-HAC (Conley) on per-unit influence "
                "functions could conceptually apply (spatial aggregation "
                "of per-unit IFs) but requires separate methodology work. "
                "Tracked as a follow-up row in DEFERRED.md. Use vcov_type='hc1' "
                "(the default) with cluster=<col> for cluster-robust "
                "inference today."
            )
        if vcov_type not in _accepted_vcov:
            raise ValueError(
                f"CallawaySantAnna(vcov_type={vcov_type!r}) is invalid. "
                f"Accepted values: {sorted(_accepted_vcov)}. CS is "
                "permanently narrow to 'hc1' per IF-based variance "
                "structure; see REGISTRY.md."
            )

    # get_params/set_params come from BaseEstimator.
    _DERIVED_CONFIG_ATTRS = ("_vcov_type_explicit",)

    def summary(self) -> str:
        """Get summary of estimation results."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())


#: `precomputed` keys the aggregation machinery actually reads. Verified by
#: enumerating every `precomputed[...]` / `.get(...)` access in
#: `staggered_aggregation.py`. Everything else - notably `outcome_matrix`,
#: `covariate_matrix`, `obs_outcome`, `obs_covariates` - is DATA and is
#: deliberately not retained, so a results object never holds the source panel.
#: `_agg_cache` is excluded too: it is derived memoization, rebuilt on demand.
_AGGREGATION_BOOKKEEPING_KEYS = (
    "unit_to_idx",
    "unit_cohorts",
    "all_units",
    "obs_per_unit",
    "survey_weights",
    "resolved_survey_unit",
    "df_survey",
    "agg_cohort_masses",
    "agg_total_weight",
    "is_panel",
    "canonical_size",
    "n_units",
)


def _build_aggregation_kit(
    estimator: "CallawaySantAnna",
    precomputed: Optional[Dict[str, Any]],
    influence_func_info: Optional[Dict[str, Any]],
    group_time_effects: Optional[Dict[Any, Any]],
) -> Optional["AggregationKit"]:
    """Distil the fit-time state post-fit re-aggregation needs.

    Returns ``None`` when the fit produced nothing to re-aggregate, in which
    case ``aggregate()`` reports that rather than failing obscurely.
    """
    if not influence_func_info or not group_time_effects:
        return None

    bookkeeping: Dict[str, Any] = {}
    if precomputed is not None:
        for key in _AGGREGATION_BOOKKEEPING_KEYS:
            if key in precomputed:
                bookkeeping[key] = precomputed[key]

    # Data minimization: the results object is picklable and users share
    # result artifacts, so the kit must not turn it into a carrier for raw
    # unit identifiers (which are routinely names, emails or administrative
    # IDs). ``all_units`` and ``unit_to_idx`` exist here only to size and
    # position the influence vectors - the influence arrays index by POSITION
    # (``treated_idx`` / ``control_idx``), and the fast aggregation path keys
    # off object identity plus length, never off a label lookup. Substituting
    # canonical 0..n-1 codes is therefore numerically inert (pinned
    # bit-identical by TestRetentionKit) while retaining nothing identifying.
    # ``unit_cohorts`` is kept as-is: those are first-treatment PERIODS, which
    # are reported as group labels and are not unit identifiers.
    if "all_units" in bookkeeping and bookkeeping["all_units"] is not None:
        n_kit_units = len(bookkeeping["all_units"])
        bookkeeping["all_units"] = np.arange(n_kit_units)
        if bookkeeping.get("unit_to_idx") is not None:
            bookkeeping["unit_to_idx"] = {i: i for i in range(n_kit_units)}

    return AggregationKit(
        bookkeeping=bookkeeping,
        influence=influence_func_info,
        alpha=estimator.alpha,
        anticipation=estimator.anticipation,
        cband=bool(estimator.cband),
        # Bootstrap replay is not wired in this PR: a bootstrapped fit's
        # percentile inference cannot be reproduced from analytical state, so
        # aggregate() fails closed on one rather than silently substituting
        # analytical numbers. BootstrapReplaySpec (diff_diff/aggregation.py) is
        # the verified mechanism for the follow-up.
        bootstrap=None,
    )
