"""
Staggered Difference-in-Differences estimators.

Implements modern methods for DiD with variation in treatment timing,
including the Callaway-Sant'Anna (2021) estimator.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg as scipy_linalg

from diff_diff.linalg import (
    _check_propensity_diagnostics,
    _detect_rank_deficiency,
    _format_dropped_columns,
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

# Re-export for backward compatibility
__all__ = [
    "CallawaySantAnna",
    "CallawaySantAnnaResults",
    "CSBootstrapResults",
    "GroupTimeEffect",
]

# Type alias for pre-computed structures
PrecomputedData = Dict[str, Any]


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


def _safe_inv(A: np.ndarray) -> np.ndarray:
    """Invert a square matrix with lstsq fallback for near-singular cases."""
    try:
        return np.linalg.solve(A, np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, np.eye(A.shape[0]), rcond=None)[0]


class CallawaySantAnna(
    CallawaySantAnnaBootstrapMixin,
    CallawaySantAnnaAggregationMixin,
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
        Column name for cluster-robust standard errors.
        Defaults to unit-level clustering.
    n_bootstrap : int, default=0
        Number of bootstrap iterations for inference.
        If 0, uses analytical standard errors.
        Recommended: 999 or more for reliable inference.

        .. note:: Memory Usage
            The bootstrap stores all weights in memory as a (n_bootstrap, n_units)
            float64 array. For large datasets, this can be significant:
            - 1K bootstrap × 10K units = ~80 MB
            - 10K bootstrap × 100K units = ~8 GB
            Consider reducing n_bootstrap if memory is constrained.

    bootstrap_weights : str, default="rademacher"
        Type of weights for multiplier bootstrap:
        - "rademacher": +1/-1 with equal probability (standard choice)
        - "mammen": Two-point distribution (asymptotically valid, matches skewness)
        - "webb": Six-point distribution (recommended when n_clusters < 20)
    bootstrap_weight_type : str, optional
        .. deprecated:: 1.0.1
            Use ``bootstrap_weights`` instead. Will be removed in v3.0.
    seed : int, optional
        Random seed for reproducibility.
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient (linearly dependent columns):
        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning
    base_period : str, default="varying"
        Method for selecting the base (reference) period for computing
        ATT(g,t). Options:
        - "varying": For pre-treatment periods (t < g - anticipation), use
          t-1 as base (consecutive comparisons). For post-treatment, use
          g-1-anticipation. Requires t-1 to exist in data.
        - "universal": Always use g-1-anticipation as base period.
        Both produce identical post-treatment effects. Matches R's
        did::att_gt() base_period parameter.
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

    With event study aggregation:

    >>> cs = CallawaySantAnna()
    >>> results = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat',
    ...                  aggregate='event_study')
    >>>
    >>> # Plot event study
    >>> from diff_diff import plot_event_study
    >>> plot_event_study(results)

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
    With base_period="varying" (default), pre-treatment uses t-1 as base for
    consecutive comparisons useful in parallel trends diagnostics.

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
        bootstrap_weight_type: Optional[str] = None,
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        base_period: str = "varying",
        cband: bool = True,
        pscore_trim: float = 0.01,
        panel: bool = True,
    ):
        import warnings

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

        # Handle bootstrap_weight_type deprecation
        if bootstrap_weight_type is not None:
            warnings.warn(
                "bootstrap_weight_type is deprecated and will be removed in v3.0. "
                "Use bootstrap_weights instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if bootstrap_weights is None:
                bootstrap_weights = bootstrap_weight_type

        # Default to rademacher if neither specified
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

        self.control_group = control_group
        self.anticipation = anticipation
        self.estimation_method = estimation_method
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        # Keep bootstrap_weight_type for backward compatibility
        self.bootstrap_weight_type = bootstrap_weights
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.base_period = base_period

        self.cband = cband
        self.pscore_trim = pscore_trim
        self.panel = panel

        self.is_fitted_ = False
        self.results_: Optional[CallawaySantAnnaResults] = None

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

    def _compute_att_gt_fast(
        self,
        precomputed: PrecomputedData,
        g: Any,
        t: Any,
        covariates: Optional[List[str]],
        pscore_cache: Optional[Dict] = None,
        cho_cache: Optional[Dict] = None,
    ) -> Tuple[Optional[float], float, int, int, Optional[Dict[str, Any]], Optional[float]]:
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
        """
        period_to_col = precomputed["period_to_col"]
        outcome_matrix = precomputed["outcome_matrix"]
        cohort_masks = precomputed["cohort_masks"]
        never_treated_mask = precomputed["never_treated_mask"]
        unit_cohorts = precomputed["unit_cohorts"]
        covariate_by_period = precomputed["covariate_by_period"]

        # Base period selection based on mode
        if self.base_period == "universal":
            # Universal: always use g - 1 - anticipation
            base_period_val = g - 1 - self.anticipation
        else:  # varying
            if t < g - self.anticipation:
                # Pre-treatment: use t - 1 (consecutive comparison)
                base_period_val = t - 1
            else:
                # Post-treatment: use g - 1 - anticipation
                base_period_val = g - 1 - self.anticipation

        if base_period_val not in period_to_col:
            # Base period must exist; no fallback to maintain methodological consistency
            return None, 0.0, 0, 0, None, None

        # Check if periods exist in the data
        if base_period_val not in period_to_col or t not in period_to_col:
            return None, 0.0, 0, 0, None, None

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
            return None, 0.0, 0, 0, None, None

        # Extract outcome changes for treated and control
        treated_change = outcome_change[treated_valid]
        control_change = outcome_change[control_valid]

        # Extract survey weights for treated and control
        survey_w = precomputed.get("survey_weights")
        sw_treated = survey_w[treated_valid] if survey_w is not None else None
        sw_control = survey_w[control_valid] if survey_w is not None else None

        # Guard against zero effective mass after subpopulation filtering
        if sw_treated is not None and np.sum(sw_treated) <= 0:
            return np.nan, np.nan, 0, 0, None, None
        if sw_control is not None and np.sum(sw_control) <= 0:
            return np.nan, np.nan, 0, 0, None, None

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

        # Compute cache key for Cholesky reuse (DR outcome regression)
        cho_key = None
        if cho_cache is not None and X_control is not None:
            is_balanced = precomputed.get("is_balanced", False)
            if is_balanced and self.control_group == "never_treated":
                cho_key = base_period_val
            else:
                cho_key = (g, base_period_val, t)

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
            )
        else:  # doubly robust
            sw_all = np.concatenate([sw_treated, sw_control]) if sw_treated is not None else None
            att_gt, se_gt, inf_func = self._doubly_robust(
                treated_change,
                control_change,
                X_treated,
                X_control,
                pscore_cache=pscore_cache,
                pscore_key=pscore_key,
                cho_cache=cho_cache,
                cho_key=cho_key,
                sw_treated=sw_treated,
                sw_control=sw_control,
                sw_all=sw_all,
            )

        # Package influence function info with index arrays (positions into
        # precomputed['all_units']) for O(1) downstream lookups instead of
        # O(n) Python dict lookups.
        n_t = int(n_treated)
        all_units = precomputed["all_units"]
        treated_positions = np.where(treated_valid)[0]
        control_positions = np.where(control_valid)[0]
        inf_func_info = {
            "treated_idx": treated_positions,
            "control_idx": control_positions,
            "treated_units": all_units[treated_positions],
            "control_units": all_units[control_positions],
            "treated_inf": inf_func[:n_t],
            "control_inf": inf_func[n_t:],
        }

        sw_sum = float(np.sum(sw_treated)) if sw_treated is not None else None
        return att_gt, se_gt, int(n_treated), int(n_control), inf_func_info, sw_sum

    def _compute_all_att_gt_vectorized(
        self,
        precomputed: PrecomputedData,
        treatment_groups: List[Any],
        time_periods: List[Any],
        min_period: Any,
    ) -> Tuple[Dict, Dict]:
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

        # Collect all valid (g, t, base_col, post_col) tuples
        tasks = []
        for g in treatment_groups:
            if self.base_period == "universal":
                universal_base = g - 1 - self.anticipation
                valid_periods = [t for t in time_periods if t != universal_base]
            else:
                valid_periods = [
                    t for t in time_periods if t >= g - self.anticipation or t > min_period
                ]

            for t in valid_periods:
                # Base period selection
                if self.base_period == "universal":
                    base_period_val = g - 1 - self.anticipation
                else:
                    if t < g - self.anticipation:
                        base_period_val = t - 1
                    else:
                        base_period_val = g - 1 - self.anticipation

                if base_period_val not in period_to_col or t not in period_to_col:
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
                att = float(np.mean(treated_change) - np.mean(control_change))

                var_t = float(np.var(treated_change, ddof=1)) if n_t > 1 else 0.0
                var_c = float(np.var(control_change, ddof=1)) if n_c > 1 else 0.0
                se = float(np.sqrt(var_t / n_t + var_c / n_c)) if (n_t > 0 and n_c > 0) else 0.0

                # Influence function
                inf_treated = (treated_change - np.mean(treated_change)) / n_t
                inf_control = -(control_change - np.mean(control_change)) / n_c
                sw_sum = None

            gte_entry = {
                "effect": att,
                "se": se,
                # t_stat, p_value, conf_int filled by batch inference below
                "t_stat": np.nan,
                "p_value": np.nan,
                "conf_int": (np.nan, np.nan),
                "n_treated": n_t,
                "n_control": n_c,
            }
            if sw_sum is not None:
                gte_entry["survey_weight_sum"] = sw_sum
            group_time_effects[(g, t)] = gte_entry

            all_units = precomputed["all_units"]
            treated_positions = np.where(treated_valid)[0]
            control_positions = np.where(control_valid)[0]
            influence_func_info[(g, t)] = {
                "treated_idx": treated_positions,
                "control_idx": control_positions,
                "treated_units": all_units[treated_positions],
                "control_units": all_units[control_positions],
                "treated_inf": inf_treated,
                "control_inf": inf_control,
            }

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

        return group_time_effects, influence_func_info

    def _compute_all_att_gt_covariate_reg(
        self,
        precomputed: PrecomputedData,
        treatment_groups: List[Any],
        time_periods: List[Any],
        min_period: Any,
    ) -> Tuple[Dict, Dict]:
        """
        Optimized computation of all ATT(g,t) for the covariate regression case.

        Groups (g,t) pairs by their control regression key to reuse Cholesky
        factorizations of X^T X across pairs that share the same control design
        matrix.

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

        # Collect all valid (g, t) tasks with their base periods
        tasks_by_group = {}  # control_key -> list of (g, t, base_period_val, base_col, post_col)
        for g in treatment_groups:
            if self.base_period == "universal":
                universal_base = g - 1 - self.anticipation
                valid_periods = [t for t in time_periods if t != universal_base]
            else:
                valid_periods = [
                    t for t in time_periods if t >= g - self.anticipation or t > min_period
                ]

            for t in valid_periods:
                if self.base_period == "universal":
                    base_period_val = g - 1 - self.anticipation
                else:
                    if t < g - self.anticipation:
                        base_period_val = t - 1
                    else:
                        base_period_val = g - 1 - self.anticipation

                if base_period_val not in period_to_col or t not in period_to_col:
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
                continue

            X_ctrl = None
            cho = None
            kept_cols = None
            if not ctrl_has_nan:
                X_ctrl = np.column_stack([np.ones(n_c_base), X_ctrl_raw])

                # One-time rank check for this control group
                rank, dropped_cols, _ = _detect_rank_deficiency(X_ctrl)

                if len(dropped_cols) > 0:
                    # Rank-deficient: force lstsq for both "warn" and "silent".
                    # Cholesky on near-singular XtX could yield unstable coefficients.
                    if self.rank_deficient_action == "warn":
                        col_info = _format_dropped_columns(dropped_cols)
                        warnings.warn(
                            f"Rank-deficient covariate design (control_key={control_key}): "
                            f"dropped columns {col_info}. Rank {rank} < {X_ctrl.shape[1]}. "
                            "Using minimum-norm least-squares solution.",
                            UserWarning,
                            stacklevel=2,
                        )
                    cho = None  # Force lstsq path for ALL rank-deficient cases
                    kept_cols = np.array(
                        [i for i in range(X_ctrl.shape[1]) if i not in dropped_cols]
                    )
                else:
                    kept_cols = None  # Full rank — use all columns
                    with np.errstate(all="ignore"):
                        XtX = X_ctrl.T @ X_ctrl
                    try:
                        cho = scipy_linalg.cho_factor(XtX)
                    except np.linalg.LinAlgError:
                        cho = None

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
                    att = float(np.mean(treated_change) - np.mean(control_change))
                    var_t = float(np.var(treated_change, ddof=1)) if n_t > 1 else 0.0
                    var_c = float(np.var(control_change, ddof=1)) if n_c > 1 else 0.0
                    se = float(np.sqrt(var_t / n_t + var_c / n_c))
                    inf_treated = (treated_change - np.mean(treated_change)) / n_t
                    inf_control = -(control_change - np.mean(control_change)) / n_c
                else:
                    # Build per-pair X_ctrl if control_valid differs from base
                    if is_balanced and self.control_group == "never_treated" and X_ctrl is not None:
                        pair_X_ctrl = X_ctrl
                        pair_n_c = n_c_base
                    else:
                        pair_X_ctrl = np.column_stack([np.ones(n_c), X_control_pair])
                        pair_n_c = n_c

                    # Solve for beta
                    beta = None
                    with np.errstate(all="ignore"):
                        if (
                            cho is not None
                            and is_balanced
                            and self.control_group == "never_treated"
                        ):
                            # Use cached Cholesky
                            Xty = pair_X_ctrl.T @ control_change
                            beta = scipy_linalg.cho_solve(cho, Xty)
                        else:
                            # Compute per-pair Cholesky or lstsq fallback
                            if kept_cols is not None:
                                # Rank-deficient: skip Cholesky, use reduced lstsq
                                pass
                            else:
                                pair_XtX = pair_X_ctrl.T @ pair_X_ctrl
                                try:
                                    pair_cho = scipy_linalg.cho_factor(pair_XtX)
                                    Xty = pair_X_ctrl.T @ control_change
                                    beta = scipy_linalg.cho_solve(pair_cho, Xty)
                                except np.linalg.LinAlgError:
                                    pass

                        if beta is None or np.any(~np.isfinite(beta)):
                            if kept_cols is not None:
                                # Reduced solve for rank-deficient design
                                result = scipy_linalg.lstsq(
                                    pair_X_ctrl[:, kept_cols],
                                    control_change,
                                    cond=1e-07,
                                )
                                beta = np.zeros(pair_X_ctrl.shape[1])
                                beta[kept_cols] = result[0]
                            else:
                                # Full-rank lstsq fallback (Cholesky numerical failure)
                                result = scipy_linalg.lstsq(
                                    pair_X_ctrl,
                                    control_change,
                                    cond=1e-07,
                                )
                                beta = result[0]

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
                        se = np.nan
                        inf_treated = np.zeros(n_t)
                        inf_control = np.zeros(n_c)
                    else:
                        var_t = float(np.var(treated_residuals, ddof=1)) if n_t > 1 else 0.0
                        var_c = float(np.var(residuals, ddof=1)) if pair_n_c > 1 else 0.0
                        se = float(np.sqrt(var_t / n_t + var_c / pair_n_c))
                        inf_treated = (treated_residuals - np.mean(treated_residuals)) / n_t
                        inf_control = -residuals / pair_n_c

                group_time_effects[(g, t)] = {
                    "effect": att,
                    "se": se,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_treated": n_t,
                    "n_control": n_c,
                }

                all_units = precomputed["all_units"]
                treated_positions = np.where(treated_valid)[0]
                control_positions = np.where(control_valid)[0]
                influence_func_info[(g, t)] = {
                    "treated_idx": treated_positions,
                    "control_idx": control_positions,
                    "treated_units": all_units[treated_positions],
                    "control_units": all_units[control_positions],
                    "treated_inf": inf_treated,
                    "control_inf": inf_control,
                }

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

        return group_time_effects, influence_func_info

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
        balance_e: Optional[int] = None,
        survey_design: object = None,
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
            How to aggregate group-time effects:
            - None: Only compute ATT(g,t) (default)
            - "simple": Simple weighted average (overall ATT)
            - "event_study": Aggregate by relative time (event study)
            - "group": Aggregate by treatment cohort
            - "all": Compute all aggregations
        balance_e : int, optional
            For event study, balance the panel at relative time e.
            Ensures all groups contribute to each relative period.
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
        # Validate pscore_trim (may have been changed via set_params)
        if not (0 < self.pscore_trim < 0.5):
            raise ValueError(f"pscore_trim must be in (0, 0.5), got {self.pscore_trim}")

        # Reset stale state from prior fit (prevents leaking event-study VCV)
        self._event_study_vcov = None

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
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        # Validate within-unit constancy for panel survey designs
        if resolved_survey is not None:
            if self.panel:
                _validate_unit_constant_survey(data, unit, survey_design)
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
        df.loc[df[first_treat] == np.inf, first_treat] = 0

        # Identify groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0])

        if self.panel:
            # Panel: count unique units
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
        if self.panel:
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

        if not self.panel:
            # --- Repeated cross-section path ---
            # No vectorized/Cholesky fast paths (panel-only optimizations).
            # Loop using _compute_att_gt_rc() for each (g,t).
            group_time_effects = {}
            influence_func_info = {}

            for g in treatment_groups:
                if self.base_period == "universal":
                    universal_base = g - 1 - self.anticipation
                    valid_periods = [t for t in time_periods if t != universal_base]
                else:
                    valid_periods = [
                        t for t in time_periods if t >= g - self.anticipation or t > min_period
                    ]

                for t in valid_periods:
                    rc_result = self._compute_att_gt_rc(
                        precomputed,
                        g,
                        t,
                        covariates,
                    )
                    att_gt, se_gt, n_treat, n_ctrl, inf_info, sw_sum = rc_result[:6]
                    agg_w = rc_result[6] if len(rc_result) > 6 else n_treat

                    if att_gt is not None:
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
                        }
                        if sw_sum is not None:
                            gte_entry["survey_weight_sum"] = sw_sum
                        group_time_effects[(g, t)] = gte_entry

                        if inf_info is not None:
                            influence_func_info[(g, t)] = inf_info

        elif covariates is None and self.estimation_method == "reg":
            # Fast vectorized path for the common no-covariates regression case
            group_time_effects, influence_func_info = self._compute_all_att_gt_vectorized(
                precomputed, treatment_groups, time_periods, min_period
            )
        elif (
            covariates is not None
            and self.estimation_method == "reg"
            and self.rank_deficient_action != "error"
            and not has_survey  # Cholesky cache uses X'X; survey needs X'WX
        ):
            # Optimized covariate regression path with Cholesky caching
            group_time_effects, influence_func_info = self._compute_all_att_gt_covariate_reg(
                precomputed, treatment_groups, time_periods, min_period
            )
        else:
            # General path: IPW, DR, rank_deficient_action="error", or edge cases
            group_time_effects = {}
            influence_func_info = {}

            # Propensity score cache for IPW/DR with covariates
            pscore_cache = {} if (covariates and self.estimation_method in ("ipw", "dr")) else None
            # Cholesky cache for DR outcome regression component
            # Skip cache when survey weights present (X'WX differs from X'X)
            cho_cache = (
                {}
                if (
                    covariates
                    and self.estimation_method == "dr"
                    and self.rank_deficient_action != "error"
                    and not has_survey
                )
                else None
            )

            for g in treatment_groups:
                if self.base_period == "universal":
                    universal_base = g - 1 - self.anticipation
                    valid_periods = [t for t in time_periods if t != universal_base]
                else:
                    valid_periods = [
                        t for t in time_periods if t >= g - self.anticipation or t > min_period
                    ]

                for t in valid_periods:
                    att_gt, se_gt, n_treat, n_ctrl, inf_info, sw_sum = self._compute_att_gt_fast(
                        precomputed,
                        g,
                        t,
                        covariates,
                        pscore_cache=pscore_cache,
                        cho_cache=cho_cache,
                    )

                    if att_gt is not None:
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
                        }
                        if sw_sum is not None:
                            gte_entry["survey_weight_sum"] = sw_sum
                        group_time_effects[(g, t)] = gte_entry

                        if inf_info is not None:
                            influence_func_info[(g, t)] = inf_info

        if not group_time_effects:
            raise ValueError(
                "Could not estimate any group-time effects. "
                "Check that data has sufficient observations."
            )

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

        if aggregate in ["event_study", "all"]:
            event_study_effects = self._aggregate_event_study(
                group_time_effects,
                influence_func_info,
                treatment_groups,
                time_periods,
                balance_e,
                df,
                unit,
                precomputed,
            )

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

        # Store results
        # Retrieve event-study VCV from aggregation mixin (Phase 7d).
        # Clear it when bootstrap overwrites event-study SEs to prevent
        # HonestDiD from mixing analytical VCV with bootstrap SEs.
        event_study_vcov = getattr(self, "_event_study_vcov", None)
        event_study_vcov_index = getattr(self, "_event_study_vcov_index", None)
        if bootstrap_results is not None and event_study_vcov is not None:
            event_study_vcov = None
            event_study_vcov_index = None

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
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            bootstrap_results=bootstrap_results,
            cband_crit_value=cband_crit_value,
            pscore_trim=self.pscore_trim,
            survey_metadata=survey_metadata,
            event_study_vcov=event_study_vcov,
            event_study_vcov_index=event_study_vcov_index,
            panel=self.panel,
        )

        self.is_fitted_ = True
        return self.results_

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
            beta, residuals = _linear_regression(
                X_control,
                control_change,
                rank_deficient_action=self.rank_deficient_action,
                weights=sw_control,
            )

            # Zero NaN coefficients for prediction (dropped rank-deficient columns
            # contribute 0 to the column space projection, matching DR path convention)
            beta = np.where(np.isfinite(beta), beta, 0.0)

            # Predict counterfactual for treated units
            X_treated_with_intercept = np.column_stack([np.ones(n_t), X_treated])
            predicted_control = np.dot(X_treated_with_intercept, beta)

            # ATT: survey-weighted mean of treated residuals
            treated_residuals = treated_change - predicted_control

            if sw_treated is not None:
                sw_t_sum = float(np.sum(sw_treated))
                sw_c_sum = float(np.sum(sw_control))
                sw_t_norm = sw_treated / sw_t_sum
                sw_c_norm = sw_control / sw_c_sum
                att = float(np.sum(sw_t_norm * treated_residuals))

                # Survey-weighted OR influence function.
                # Mirrors unweighted: inf_treated = (resid-ATT)/n_t,
                # inf_control = -resid/n_c. Survey: w_i/sum(w_group).
                # WLS residuals are orthogonal to W*X by construction.
                X_c_int = np.column_stack([np.ones(n_c), X_control])
                resid_c = control_change - np.dot(X_c_int, beta)

                inf_treated = sw_t_norm * (treated_residuals - att)
                inf_control = -sw_c_norm * resid_c
                inf_func = np.concatenate([inf_treated, inf_control])

                # SE: survey-weighted variance matching unweighted var_t/n_t + var_c/n_c
                var_t = float(np.sum(sw_t_norm * (treated_residuals - att) ** 2))
                var_c = float(np.sum(sw_c_norm * resid_c**2))
                se = float(np.sqrt(var_t + var_c)) if (n_t > 0 and n_c > 0) else 0.0
            else:
                att = float(np.mean(treated_residuals))

                # Standard error using sandwich estimator
                var_t = np.var(treated_residuals, ddof=1) if n_t > 1 else 0.0
                var_c = np.var(residuals, ddof=1) if n_c > 1 else 0.0
                se = float(np.sqrt(var_t / n_t + var_c / n_c)) if (n_t > 0 and n_c > 0) else 0.0

                # Influence function
                inf_treated = (treated_residuals - np.mean(treated_residuals)) / n_t
                inf_control = -residuals / n_c
                inf_func = np.concatenate([inf_treated, inf_control])
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
                att = float(np.mean(treated_change) - np.mean(control_change))

                var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
                var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0
                se = float(np.sqrt(var_t / n_t + var_c / n_c)) if (n_t > 0 and n_c > 0) else 0.0

                # Influence function (for aggregation)
                inf_treated = treated_change - np.mean(treated_change)
                inf_control = control_change - np.mean(control_change)
                inf_func = np.concatenate([inf_treated / n_t, -inf_control / n_c])

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
        n_total = n_treated + n_control

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Covariate-adjusted IPW estimation
            # Check propensity score cache
            cached_pscore = None
            if pscore_cache is not None and pscore_key is not None:
                cached_pscore = pscore_cache.get(pscore_key)

            if cached_pscore is not None:
                # Use cached propensity scores (beta coefficients)
                beta_logistic = cached_pscore
                X_all = np.vstack([X_treated, X_control])
                X_all_with_intercept = np.column_stack([np.ones(n_t + n_c), X_all])
                z = np.dot(X_all_with_intercept, beta_logistic)
                z = np.clip(z, -500, 500)
                pscore = 1 / (1 + np.exp(-z))
            else:
                # Stack covariates and create treatment indicator
                X_all = np.vstack([X_treated, X_control])
                D = np.concatenate([np.ones(n_t), np.zeros(n_c)])

                # Estimate propensity scores using IRLS logistic regression
                try:
                    beta_logistic, pscore = solve_logit(
                        X_all,
                        D,
                        rank_deficient_action=self.rank_deficient_action,
                        weights=sw_all,
                    )
                    _check_propensity_diagnostics(pscore, self.pscore_trim)
                    # Cache the fitted coefficients
                    if pscore_cache is not None and pscore_key is not None:
                        pscore_cache[pscore_key] = beta_logistic
                except (np.linalg.LinAlgError, ValueError):
                    if self.rank_deficient_action == "error":
                        raise
                    # Fallback to unconditional if logistic regression fails
                    warnings.warn(
                        "Propensity score estimation failed. "
                        "Falling back to unconditional estimation.",
                        UserWarning,
                        stacklevel=4,
                    )
                    pscore = np.full(len(D), n_t / (n_t + n_c))

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
                H_psi_inv = _safe_inv(H_psi)

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
                se = float(np.sqrt(var_psi)) if var_psi > 0 else 0.0
            else:
                # IPW weights for control units: p(X) / (1 - p(X))
                # This reweights controls to have same covariate distribution as treated
                weights_control = pscore_control / (1 - pscore_control)
                weights_control = weights_control / np.sum(weights_control)  # normalize

                # ATT = mean(treated) - weighted_mean(control)
                att = float(np.mean(treated_change) - np.sum(weights_control * control_change))

                # Compute standard error
                var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0

                weighted_var_c = np.sum(
                    weights_control
                    * (control_change - np.sum(weights_control * control_change)) ** 2
                )

                se = float(np.sqrt(var_t / n_t + weighted_var_c)) if (n_t > 0 and n_c > 0) else 0.0

                # Influence function
                inf_treated = (treated_change - np.mean(treated_change)) / n_t
                inf_control = -weights_control * (
                    control_change - np.sum(weights_control * control_change)
                )
                inf_func = np.concatenate([inf_treated, inf_control])
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
                p_treat = n_treated / n_total  # unconditional propensity score

                att = float(np.mean(treated_change) - np.mean(control_change))

                var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
                var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

                # Adjusted variance for IPW
                se = float(
                    np.sqrt(var_t / n_t + var_c * (1 - p_treat) / (n_c * p_treat))
                    if (n_t > 0 and n_c > 0 and p_treat > 0)
                    else 0.0
                )

                # Influence function (for aggregation)
                inf_treated = (treated_change - np.mean(treated_change)) / n_t
                inf_control = (control_change - np.mean(control_change)) / n_c
                inf_func = np.concatenate([inf_treated, -inf_control])

        return att, se, inf_func

    def _doubly_robust(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
        X_treated: Optional[np.ndarray] = None,
        X_control: Optional[np.ndarray] = None,
        pscore_cache: Optional[Dict] = None,
        pscore_key: Optional[Any] = None,
        cho_cache: Optional[Dict] = None,
        cho_key: Optional[Any] = None,
        sw_treated: Optional[np.ndarray] = None,
        sw_control: Optional[np.ndarray] = None,
        sw_all: Optional[np.ndarray] = None,
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
            # Step 1: Outcome regression - fit E[Delta Y | X] on control
            # Try Cholesky cache for outcome regression (disabled when survey weights present)
            beta = None
            X_control_with_intercept = np.column_stack([np.ones(n_c), X_control])
            if cho_cache is not None and cho_key is not None:
                cached_cho = cho_cache.get(cho_key)

                if cached_cho is False:
                    # Rank-deficient sentinel: skip Cholesky, fall through
                    pass
                elif cached_cho is not None:
                    Xty = X_control_with_intercept.T @ control_change
                    beta = scipy_linalg.cho_solve(cached_cho, Xty)
                    if np.any(~np.isfinite(beta)):
                        beta = None
                else:
                    # First time for this cho_key: check rank before Cholesky
                    rank_info = _detect_rank_deficiency(X_control_with_intercept)
                    if len(rank_info[1]) > 0:
                        cho_cache[cho_key] = False  # Sentinel
                    else:
                        XtX = X_control_with_intercept.T @ X_control_with_intercept
                        try:
                            cho_factor = scipy_linalg.cho_factor(XtX)
                            cho_cache[cho_key] = cho_factor
                            Xty = X_control_with_intercept.T @ control_change
                            beta = scipy_linalg.cho_solve(cho_factor, Xty)
                            if np.any(~np.isfinite(beta)):
                                beta = None
                        except np.linalg.LinAlgError:
                            pass

            if beta is None:
                beta, _ = _linear_regression(
                    X_control,
                    control_change,
                    rank_deficient_action=self.rank_deficient_action,
                    weights=sw_control,
                )
                # Zero NaN coefficients for prediction only — dropped columns
                # contribute 0 to the column space projection. Note: solve_ols
                # deliberately uses NaN (R's lm() convention) for inference, but
                # here we only need beta for prediction (m_treated, m_control).
                beta = np.where(np.isfinite(beta), beta, 0.0)

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
                beta_logistic = cached_pscore
                X_all = np.vstack([X_treated, X_control])
                X_all_with_intercept = np.column_stack([np.ones(n_t + n_c), X_all])
                z = np.dot(X_all_with_intercept, beta_logistic)
                z = np.clip(z, -500, 500)
                pscore = 1 / (1 + np.exp(-z))
            else:
                X_all = np.vstack([X_treated, X_control])
                D = np.concatenate([np.ones(n_t), np.zeros(n_c)])

                try:
                    beta_logistic, pscore = solve_logit(
                        X_all,
                        D,
                        rank_deficient_action=self.rank_deficient_action,
                        weights=sw_all,
                    )
                    _check_propensity_diagnostics(pscore, self.pscore_trim)
                    if pscore_cache is not None and pscore_key is not None:
                        pscore_cache[pscore_key] = beta_logistic
                except (np.linalg.LinAlgError, ValueError):
                    if self.rank_deficient_action == "error":
                        raise
                    # Fallback to unconditional if logistic regression fails
                    warnings.warn(
                        "Propensity score estimation failed. "
                        "Falling back to unconditional estimation.",
                        UserWarning,
                        stacklevel=4,
                    )
                    pscore = np.full(len(D), n_t / (n_t + n_c))

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
                    H_psi_inv = _safe_inv(H_psi)

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
                    bread = _safe_inv(XtWX)

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
                se = float(np.sqrt(var_psi)) if var_psi > 0 else 0.0
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
                    # --- PS IF correction — psi convention, convert to phi ---
                    n_all_panel = n_t + n_c
                    X_all_int = np.column_stack([np.ones(n_all_panel), X_all])
                    pscore_treated_clipped = np.clip(
                        pscore[:n_t], self.pscore_trim, 1 - self.pscore_trim
                    )
                    pscore_all = np.concatenate([pscore_treated_clipped, pscore_control])

                    W_ps = pscore_all * (1 - pscore_all)
                    H_psi = X_all_int.T @ (W_ps[:, None] * X_all_int) / n_all_panel
                    H_psi_inv = _safe_inv(H_psi)

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
                    bread = _safe_inv(XtX)

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
                se = float(np.sqrt(var_psi)) if var_psi > 0 else 0.0
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
                att = float(np.mean(treated_change) - np.mean(control_change))

                var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
                var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

                se = float(np.sqrt(var_t / n_t + var_c / n_c)) if (n_t > 0 and n_c > 0) else 0.0

                # Influence function for DR estimator
                inf_treated = (treated_change - np.mean(treated_change)) / n_t
                inf_control = (control_change - np.mean(control_change)) / n_c
                inf_func = np.concatenate([inf_treated, -inf_control])

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

        # Fixed cohort masses: total observations per cohort across all periods.
        # Used as aggregation weights so that n_treated is consistent with WIF.
        rcs_cohort_masses = {}
        for g in treatment_groups:
            rcs_cohort_masses[g] = int(np.sum(unit_cohorts == g))

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
        }

    def _compute_att_gt_rc(
        self,
        precomputed: PrecomputedData,
        g: Any,
        t: Any,
        covariates: Optional[List[str]],
    ) -> Tuple[Optional[float], float, int, int, Optional[Dict[str, Any]], Optional[float]]:
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
        """
        cohort_masks = precomputed["cohort_masks"]
        never_treated_mask = precomputed["never_treated_mask"]
        unit_cohorts = precomputed["unit_cohorts"]
        obs_time = precomputed["obs_time"]
        obs_outcome = precomputed["obs_outcome"]
        period_to_col = precomputed["period_to_col"]

        # Base period selection (same logic as panel)
        if self.base_period == "universal":
            base_period_val = g - 1 - self.anticipation
        else:  # varying
            if t < g - self.anticipation:
                base_period_val = t - 1
            else:
                base_period_val = g - 1 - self.anticipation

        if base_period_val not in period_to_col or t not in period_to_col:
            return None, 0.0, 0, 0, None, None

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
            return None, 0.0, 0, 0, None, None

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
                return np.nan, np.nan, 0, 0, None, None
            if np.sum(sw_ct) <= 0 or np.sum(sw_cs) <= 0:
                return np.nan, np.nan, 0, 0, None, None

        # Get covariates if specified
        obs_covariates = precomputed.get("obs_covariates")
        has_covariates = covariates is not None and obs_covariates is not None

        if has_covariates:
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
            )
        elif has_covariates and self.estimation_method == "dr":
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
            )
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
        # For RCS, treated_idx/control_idx combine obs from BOTH periods
        treated_idx = np.concatenate([np.where(treated_t)[0], np.where(treated_s)[0]])
        control_idx = np.concatenate([np.where(control_t)[0], np.where(control_s)[0]])

        n_treated_combined = len(treated_idx)
        inf_func_info = {
            "treated_idx": treated_idx,
            "control_idx": control_idx,
            "treated_units": treated_idx,  # For RCS, obs indices = "units"
            "control_units": control_idx,
            "treated_inf": inf_func_all[:n_treated_combined],
            "control_inf": inf_func_all[n_treated_combined:],
        }

        sw_sum = float(np.sum(sw_gt)) if sw_gt is not None else None
        # n_treated = per-cell treated count at period t (for display).
        # cohort_mass = total treated across all periods (for aggregation weights).
        cohort_mass = precomputed.get("rcs_cohort_masses", {}).get(g, n_gt)
        return att, se, n_gt, n_ct, inf_func_info, sw_sum, cohort_mass

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
        bread_t = _safe_inv(X_ct_int.T @ (W_ct[:, None] * X_ct_int))
        bread_s = _safe_inv(X_cs_int.T @ (W_cs[:, None] * X_cs_int))

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

        try:
            beta_logistic, pscore = solve_logit(
                X_all,
                D_all,
                rank_deficient_action=self.rank_deficient_action,
                weights=sw_all,
            )
            _check_propensity_diagnostics(pscore, self.pscore_trim)
        except (np.linalg.LinAlgError, ValueError):
            if self.rank_deficient_action == "error":
                raise
            warnings.warn(
                "Propensity score estimation failed (RCS IPW). "
                "Falling back to unconditional estimation.",
                UserWarning,
                stacklevel=4,
            )
            p_treat = (n_gt + n_gs) / len(D_all)
            pscore = np.full(len(D_all), p_treat)

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

        # --- PS IF correction — psi convention, convert to phi ---
        X_all_int = np.column_stack([np.ones(n_all), X_all])

        W_ps = pscore * (1 - pscore)
        if sw_all is not None:
            W_ps = W_ps * sw_all
        # R: Hessian.ps = crossprod(X * sqrt(W)) / n
        H_psi = X_all_int.T @ (W_ps[:, None] * X_all_int) / n_all
        H_psi_inv = _safe_inv(H_psi)

        score_ps = (D_all - pscore)[:, None] * X_all_int
        if sw_all is not None:
            score_ps = score_ps * sw_all[:, None]
        # R: asy.lin.rep.ps = score.ps %*% Hessian.ps  (psi scale, O(1) per obs)
        asy_lin_rep_psi = score_ps @ H_psi_inv

        # Convert leading psi to phi: phi = psi / n_all
        inf_all = psi_all / n_all

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

        # Control OR predictions for all groups
        mu0_post_gt = X_gt_int @ beta_ct  # mu_{0,1}(X) for treated-post
        mu0_pre_gt = X_gt_int @ beta_cs  # mu_{0,0}(X) for treated-post
        mu0_post_gs = X_gs_int @ beta_ct  # mu_{0,1}(X) for treated-pre
        mu0_pre_gs = X_gs_int @ beta_cs  # mu_{0,0}(X) for treated-pre
        mu0_post_ct = X_ct_int @ beta_ct  # mu_{0,1}(X) for control-post
        mu0_pre_ct = X_ct_int @ beta_cs  # mu_{0,0}(X) for control-post
        mu0_post_cs = X_cs_int @ beta_ct  # mu_{0,1}(X) for control-pre
        mu0_pre_cs = X_cs_int @ beta_cs  # mu_{0,0}(X) for control-pre

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

        try:
            beta_logistic, pscore = solve_logit(
                X_all,
                D_all,
                rank_deficient_action=self.rank_deficient_action,
                weights=sw_all,
            )
            _check_propensity_diagnostics(pscore, self.pscore_trim)
        except (np.linalg.LinAlgError, ValueError):
            if self.rank_deficient_action == "error":
                raise
            warnings.warn(
                "Propensity score estimation failed (RCS DR). "
                "Falling back to unconditional propensity.",
                UserWarning,
                stacklevel=4,
            )
            p_treat = (n_gt + n_gs) / len(D_all)
            pscore = np.full(len(D_all), p_treat)

        pscore = np.clip(pscore, self.pscore_trim, 1 - self.pscore_trim)

        # Split propensity scores per group
        ps_gt = pscore[:n_gt]
        ps_gs = pscore[n_gt : n_gt + n_gs]
        ps_ct = pscore[n_gt + n_gs : n_gt + n_gs + n_ct]
        ps_cs = pscore[n_gt + n_gs + n_ct :]

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

        W_ps = pscore * (1 - pscore)
        if sw_all is not None:
            W_ps = W_ps * sw_all
        # R: Hessian.ps = crossprod(X * sqrt(W)) / n
        H_psi = X_all_int.T @ (W_ps[:, None] * X_all_int) / n_all
        H_psi_inv = _safe_inv(H_psi)

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
        bread_ct = _safe_inv(X_ct_int.T @ (W_ct_vals[:, None] * X_ct_int))
        bread_cs = _safe_inv(X_cs_int.T @ (W_cs_vals[:, None] * X_cs_int))

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
        bread_gt = _safe_inv(X_gt_int.T @ (W_gt_vals[:, None] * X_gt_int))
        bread_gs = _safe_inv(X_gs_int.T @ (W_gs_vals[:, None] * X_gs_int))

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

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "control_group": self.control_group,
            "anticipation": self.anticipation,
            "estimation_method": self.estimation_method,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            # Deprecated but kept for backward compatibility
            "bootstrap_weight_type": self.bootstrap_weight_type,
            "seed": self.seed,
            "rank_deficient_action": self.rank_deficient_action,
            "base_period": self.base_period,
            "cband": self.cband,
            "pscore_trim": self.pscore_trim,
            "panel": self.panel,
        }

    def set_params(self, **params) -> "CallawaySantAnna":
        """Set estimator parameters (sklearn-compatible)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def summary(self) -> str:
        """Get summary of estimation results."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())
