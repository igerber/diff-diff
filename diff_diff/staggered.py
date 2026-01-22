"""
Staggered Difference-in-Differences estimators.

Implements modern methods for DiD with variation in treatment timing,
including the Callaway-Sant'Anna (2021) estimator.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from diff_diff.linalg import solve_ols
from diff_diff.utils import (
    compute_confidence_interval,
    compute_p_value,
)

# Import from split modules
from diff_diff.staggered_results import (
    GroupTimeEffect,
    CallawaySantAnnaResults,
)
from diff_diff.staggered_bootstrap import (
    CSBootstrapResults,
    CallawaySantAnnaBootstrapMixin,
)
from diff_diff.staggered_aggregation import (
    CallawaySantAnnaAggregationMixin,
)

# Re-export for backward compatibility
__all__ = [
    "CallawaySantAnna",
    "CallawaySantAnnaResults",
    "CSBootstrapResults",
    "GroupTimeEffect",
]

# Type alias for pre-computed structures
PrecomputedData = Dict[str, Any]


def _logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit logistic regression using scipy optimize.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features). Intercept added automatically.
    y : np.ndarray
        Binary outcome (0/1).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    beta : np.ndarray
        Fitted coefficients (including intercept).
    probs : np.ndarray
        Predicted probabilities.
    """
    n, p = X.shape
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])

    def neg_log_likelihood(beta: np.ndarray) -> float:
        z = X_with_intercept @ beta
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        log_lik = np.sum(y * z - np.log(1 + np.exp(z)))
        return -log_lik

    def gradient(beta: np.ndarray) -> np.ndarray:
        z = X_with_intercept @ beta
        z = np.clip(z, -500, 500)
        probs = 1 / (1 + np.exp(-z))
        return -X_with_intercept.T @ (y - probs)

    # Initialize with zeros
    beta_init = np.zeros(p + 1)

    result = optimize.minimize(
        neg_log_likelihood,
        beta_init,
        method='BFGS',
        jac=gradient,
        options={'maxiter': max_iter, 'gtol': tol}
    )

    beta = result.x
    z = X_with_intercept @ beta
    z = np.clip(z, -500, 500)
    probs = 1 / (1 + np.exp(-z))

    return beta, probs


def _linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    rank_deficient_action: str = "warn",
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
        X_with_intercept, y, return_vcov=False,
        rank_deficient_action=rank_deficient_action,
    )

    return beta, residuals


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
    ):
        import warnings

        if control_group not in ["never_treated", "not_yet_treated"]:
            raise ValueError(
                f"control_group must be 'never_treated' or 'not_yet_treated', "
                f"got '{control_group}'"
            )
        if estimation_method not in ["dr", "ipw", "reg"]:
            raise ValueError(
                f"estimation_method must be 'dr', 'ipw', or 'reg', "
                f"got '{estimation_method}'"
            )

        # Handle bootstrap_weight_type deprecation
        if bootstrap_weight_type is not None:
            warnings.warn(
                "bootstrap_weight_type is deprecated and will be removed in v3.0. "
                "Use bootstrap_weights instead.",
                DeprecationWarning,
                stacklevel=2
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
                f"base_period must be 'varying' or 'universal', "
                f"got '{base_period}'"
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

        self.is_fitted_ = False
        self.results_: Optional[CallawaySantAnnaResults] = None

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
        n_units = len(all_units)

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
            cohort_masks[g] = (unit_cohorts == g)

        # Never-treated mask
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

        return {
            'all_units': all_units,
            'unit_to_idx': unit_to_idx,
            'unit_cohorts': unit_cohorts,
            'outcome_matrix': outcome_matrix,
            'period_to_col': period_to_col,
            'cohort_masks': cohort_masks,
            'never_treated_mask': never_treated_mask,
            'covariate_by_period': covariate_by_period,
            'time_periods': time_periods,
        }

    def _compute_att_gt_fast(
        self,
        precomputed: PrecomputedData,
        g: Any,
        t: Any,
        covariates: Optional[List[str]],
    ) -> Tuple[Optional[float], float, int, int, Optional[Dict[str, Any]]]:
        """
        Compute ATT(g,t) using pre-computed data structures (fast version).

        Uses vectorized numpy operations on pre-pivoted outcome matrix
        instead of repeated pandas filtering.
        """
        time_periods = precomputed['time_periods']
        period_to_col = precomputed['period_to_col']
        outcome_matrix = precomputed['outcome_matrix']
        cohort_masks = precomputed['cohort_masks']
        never_treated_mask = precomputed['never_treated_mask']
        unit_cohorts = precomputed['unit_cohorts']
        all_units = precomputed['all_units']
        covariate_by_period = precomputed['covariate_by_period']

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
            return None, 0.0, 0, 0, None

        # Check if periods exist in the data
        if base_period_val not in period_to_col or t not in period_to_col:
            return None, 0.0, 0, 0, None

        base_col = period_to_col[base_period_val]
        post_col = period_to_col[t]

        # Get treated units mask (cohort g)
        treated_mask = cohort_masks[g]

        # Get control units mask
        if self.control_group == "never_treated":
            control_mask = never_treated_mask
        else:  # not_yet_treated
            # Not yet treated at time t: never-treated OR (first_treat > t AND not cohort g)
            # Must exclude cohort g since they are the treated group for this ATT(g,t)
            control_mask = never_treated_mask | ((unit_cohorts > t) & (unit_cohorts != g))

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
            return None, 0.0, 0, 0, None

        # Extract outcome changes for treated and control
        treated_change = outcome_change[treated_valid]
        control_change = outcome_change[control_valid]

        # Get unit IDs for influence function
        treated_units = all_units[treated_valid]
        control_units = all_units[control_valid]

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

        # Estimation method
        if self.estimation_method == "reg":
            att_gt, se_gt, inf_func = self._outcome_regression(
                treated_change, control_change, X_treated, X_control
            )
        elif self.estimation_method == "ipw":
            att_gt, se_gt, inf_func = self._ipw_estimation(
                treated_change, control_change,
                int(n_treated), int(n_control),
                X_treated, X_control
            )
        else:  # doubly robust
            att_gt, se_gt, inf_func = self._doubly_robust(
                treated_change, control_change, X_treated, X_control
            )

        # Package influence function info with unit IDs for bootstrap
        n_t = int(n_treated)
        inf_func_info = {
            'treated_units': list(treated_units),
            'control_units': list(control_units),
            'treated_inf': inf_func[:n_t],
            'control_inf': inf_func[n_t:],
        }

        return att_gt, se_gt, int(n_treated), int(n_control), inf_func_info

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
    ) -> CallawaySantAnnaResults:
        """
        Fit the Callaway-Sant'Anna estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with unit and time identifiers.
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

        Returns
        -------
        CallawaySantAnnaResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
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
        df['first_treat'] = df[first_treat]

        # Identify groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0])

        # Never-treated indicator (first_treat = 0 or inf)
        df['_never_treated'] = (df[first_treat] == 0) | (df[first_treat] == np.inf)

        # Get unique units
        unit_info = df.groupby(unit).agg({
            first_treat: 'first',
            '_never_treated': 'first'
        }).reset_index()

        n_treated_units = (unit_info[first_treat] > 0).sum()
        n_control_units = (unit_info['_never_treated']).sum()

        if n_control_units == 0:
            raise ValueError("No never-treated units found. Check 'first_treat' column.")

        # Pre-compute data structures for efficient ATT(g,t) computation
        precomputed = self._precompute_structures(
            df, outcome, unit, time, first_treat,
            covariates, time_periods, treatment_groups
        )

        # Compute ATT(g,t) for each group-time combination
        group_time_effects = {}
        influence_func_info = {}  # Store influence functions for bootstrap

        # Get minimum period for determining valid pre-treatment periods
        min_period = min(time_periods)

        for g in treatment_groups:
            # Compute valid periods including pre-treatment
            if self.base_period == "universal":
                # Universal: all periods except the base period (which is normalized to 0)
                universal_base = g - 1 - self.anticipation
                valid_periods = [t for t in time_periods if t != universal_base]
            else:
                # Varying: post-treatment + pre-treatment where t-1 exists
                valid_periods = [
                    t for t in time_periods
                    if t >= g - self.anticipation or t > min_period
                ]

            for t in valid_periods:
                att_gt, se_gt, n_treat, n_ctrl, inf_info = self._compute_att_gt_fast(
                    precomputed, g, t, covariates
                )

                if att_gt is not None:
                    t_stat = att_gt / se_gt if np.isfinite(se_gt) and se_gt > 0 else np.nan
                    p_val = compute_p_value(t_stat)
                    ci = compute_confidence_interval(att_gt, se_gt, self.alpha)

                    group_time_effects[(g, t)] = {
                        'effect': att_gt,
                        'se': se_gt,
                        't_stat': t_stat,
                        'p_value': p_val,
                        'conf_int': ci,
                        'n_treated': n_treat,
                        'n_control': n_ctrl,
                    }

                    if inf_info is not None:
                        influence_func_info[(g, t)] = inf_info

        if not group_time_effects:
            raise ValueError(
                "Could not estimate any group-time effects. "
                "Check that data has sufficient observations."
            )

        # Compute overall ATT (simple aggregation)
        overall_att, overall_se = self._aggregate_simple(
            group_time_effects, influence_func_info, df, unit, precomputed
        )
        # Use NaN for t-stat and p-value when SE is undefined (NaN or non-positive)
        if np.isfinite(overall_se) and overall_se > 0:
            overall_t = overall_att / overall_se
            overall_p = compute_p_value(overall_t)
        else:
            overall_t = np.nan
            overall_p = np.nan
        overall_ci = compute_confidence_interval(overall_att, overall_se, self.alpha)

        # Compute additional aggregations if requested
        event_study_effects = None
        group_effects = None

        if aggregate in ["event_study", "all"]:
            event_study_effects = self._aggregate_event_study(
                group_time_effects, influence_func_info,
                treatment_groups, time_periods, balance_e
            )

        if aggregate in ["group", "all"]:
            group_effects = self._aggregate_by_group(
                group_time_effects, influence_func_info, treatment_groups
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
            )

            # Update estimates with bootstrap inference
            overall_se = bootstrap_results.overall_att_se
            # Use NaN for t-stat when SE is undefined; p-value comes from bootstrap
            if np.isfinite(overall_se) and overall_se > 0:
                overall_t = overall_att / overall_se
            else:
                overall_t = np.nan
            overall_p = bootstrap_results.overall_att_p_value
            overall_ci = bootstrap_results.overall_att_ci

            # Update group-time effects with bootstrap SEs
            for gt in group_time_effects:
                if gt in bootstrap_results.group_time_ses:
                    group_time_effects[gt]['se'] = bootstrap_results.group_time_ses[gt]
                    group_time_effects[gt]['conf_int'] = bootstrap_results.group_time_cis[gt]
                    group_time_effects[gt]['p_value'] = bootstrap_results.group_time_p_values[gt]
                    effect = float(group_time_effects[gt]['effect'])
                    se = float(group_time_effects[gt]['se'])
                    group_time_effects[gt]['t_stat'] = effect / se if np.isfinite(se) and se > 0 else np.nan

            # Update event study effects with bootstrap SEs
            if (event_study_effects is not None
                and bootstrap_results.event_study_ses is not None
                and bootstrap_results.event_study_cis is not None
                and bootstrap_results.event_study_p_values is not None):
                for e in event_study_effects:
                    if e in bootstrap_results.event_study_ses:
                        event_study_effects[e]['se'] = bootstrap_results.event_study_ses[e]
                        event_study_effects[e]['conf_int'] = bootstrap_results.event_study_cis[e]
                        p_val = bootstrap_results.event_study_p_values[e]
                        event_study_effects[e]['p_value'] = p_val
                        effect = float(event_study_effects[e]['effect'])
                        se = float(event_study_effects[e]['se'])
                        event_study_effects[e]['t_stat'] = effect / se if np.isfinite(se) and se > 0 else np.nan

            # Update group effects with bootstrap SEs
            if (group_effects is not None
                and bootstrap_results.group_effect_ses is not None
                and bootstrap_results.group_effect_cis is not None
                and bootstrap_results.group_effect_p_values is not None):
                for g in group_effects:
                    if g in bootstrap_results.group_effect_ses:
                        group_effects[g]['se'] = bootstrap_results.group_effect_ses[g]
                        group_effects[g]['conf_int'] = bootstrap_results.group_effect_cis[g]
                        group_effects[g]['p_value'] = bootstrap_results.group_effect_p_values[g]
                        effect = float(group_effects[g]['effect'])
                        se = float(group_effects[g]['se'])
                        group_effects[g]['t_stat'] = effect / se if np.isfinite(se) and se > 0 else np.nan

        # Store results
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
        )

        self.is_fitted_ = True
        return self.results_

    def _outcome_regression(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
        X_treated: Optional[np.ndarray] = None,
        X_control: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using outcome regression.

        With covariates:
        1. Regress outcome changes on covariates for control group
        2. Predict counterfactual for treated using their covariates
        3. ATT = mean(treated_change) - mean(predicted_counterfactual)

        Without covariates:
        Simple difference in means.
        """
        n_t = len(treated_change)
        n_c = len(control_change)

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Covariate-adjusted outcome regression
            # Fit regression on control units: E[Delta Y | X, D=0]
            beta, residuals = _linear_regression(
                X_control, control_change,
                rank_deficient_action=self.rank_deficient_action,
            )

            # Predict counterfactual for treated units
            X_treated_with_intercept = np.column_stack([np.ones(n_t), X_treated])
            predicted_control = X_treated_with_intercept @ beta

            # ATT = mean(observed treated change - predicted counterfactual)
            att = np.mean(treated_change - predicted_control)

            # Standard error using sandwich estimator
            # Variance from treated: Var(Y_1 - m(X))
            treated_residuals = treated_change - predicted_control
            var_t = np.var(treated_residuals, ddof=1) if n_t > 1 else 0.0

            # Variance from control regression (residual variance)
            var_c = np.var(residuals, ddof=1) if n_c > 1 else 0.0

            # Approximate SE (ignoring estimation error in beta for simplicity)
            se = np.sqrt(var_t / n_t + var_c / n_c) if (n_t > 0 and n_c > 0) else 0.0

            # Influence function
            inf_treated = (treated_residuals - np.mean(treated_residuals)) / n_t
            inf_control = -residuals / n_c
            inf_func = np.concatenate([inf_treated, inf_control])
        else:
            # Simple difference in means (no covariates)
            att = np.mean(treated_change) - np.mean(control_change)

            var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
            var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

            se = np.sqrt(var_t / n_t + var_c / n_c) if (n_t > 0 and n_c > 0) else 0.0

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
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using inverse probability weighting.

        With covariates:
        1. Estimate propensity score P(D=1|X) using logistic regression
        2. Reweight control units to match treated covariate distribution
        3. ATT = mean(treated) - weighted_mean(control)

        Without covariates:
        Simple difference in means with unconditional propensity weighting.
        """
        n_t = len(treated_change)
        n_c = len(control_change)
        n_total = n_treated + n_control

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Covariate-adjusted IPW estimation
            # Stack covariates and create treatment indicator
            X_all = np.vstack([X_treated, X_control])
            D = np.concatenate([np.ones(n_t), np.zeros(n_c)])

            # Estimate propensity scores using logistic regression
            try:
                _, pscore = _logistic_regression(X_all, D)
            except (np.linalg.LinAlgError, ValueError):
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
            pscore_control = np.clip(pscore_control, 0.01, 0.99)
            pscore_treated = np.clip(pscore_treated, 0.01, 0.99)

            # IPW weights for control units: p(X) / (1 - p(X))
            # This reweights controls to have same covariate distribution as treated
            weights_control = pscore_control / (1 - pscore_control)
            weights_control = weights_control / np.sum(weights_control)  # normalize

            # ATT = mean(treated) - weighted_mean(control)
            att = np.mean(treated_change) - np.sum(weights_control * control_change)

            # Compute standard error
            # Variance of treated mean
            var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0

            # Variance of weighted control mean
            weighted_var_c = np.sum(weights_control * (control_change - np.sum(weights_control * control_change)) ** 2)

            se = np.sqrt(var_t / n_t + weighted_var_c) if (n_t > 0 and n_c > 0) else 0.0

            # Influence function
            inf_treated = (treated_change - np.mean(treated_change)) / n_t
            inf_control = -weights_control * (control_change - np.sum(weights_control * control_change))
            inf_func = np.concatenate([inf_treated, inf_control])
        else:
            # Unconditional IPW (reduces to difference in means)
            p_treat = n_treated / n_total  # unconditional propensity score

            att = np.mean(treated_change) - np.mean(control_change)

            var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
            var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

            # Adjusted variance for IPW
            se = np.sqrt(var_t / n_t + var_c * (1 - p_treat) / (n_c * p_treat)) if (n_t > 0 and n_c > 0 and p_treat > 0) else 0.0

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
        """
        n_t = len(treated_change)
        n_c = len(control_change)

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Doubly robust estimation with covariates
            # Step 1: Outcome regression - fit E[Delta Y | X] on control
            beta, _ = _linear_regression(
                X_control, control_change,
                rank_deficient_action=self.rank_deficient_action,
            )

            # Predict counterfactual for both treated and control
            X_treated_with_intercept = np.column_stack([np.ones(n_t), X_treated])
            X_control_with_intercept = np.column_stack([np.ones(n_c), X_control])
            m_treated = X_treated_with_intercept @ beta
            m_control = X_control_with_intercept @ beta

            # Step 2: Propensity score estimation
            X_all = np.vstack([X_treated, X_control])
            D = np.concatenate([np.ones(n_t), np.zeros(n_c)])

            try:
                _, pscore = _logistic_regression(X_all, D)
            except (np.linalg.LinAlgError, ValueError):
                # Fallback to unconditional if logistic regression fails
                pscore = np.full(len(D), n_t / (n_t + n_c))

            pscore_control = pscore[n_t:]

            # Clip propensity scores
            pscore_control = np.clip(pscore_control, 0.01, 0.99)

            # IPW weights for control: p(X) / (1 - p(X))
            weights_control = pscore_control / (1 - pscore_control)

            # Step 3: Doubly robust ATT
            # ATT = mean(treated - m(X_treated))
            #     + weighted_mean_control((m(X) - Y) * weight)
            att_treated_part = np.mean(treated_change - m_treated)

            # Augmentation term from control
            augmentation = np.sum(weights_control * (m_control - control_change)) / n_t

            att = att_treated_part + augmentation

            # Step 4: Standard error using influence function
            # Influence function for DR estimator
            psi_treated = (treated_change - m_treated - att) / n_t
            psi_control = (weights_control * (m_control - control_change)) / n_t

            # Variance is sum of squared influence functions
            var_psi = np.sum(psi_treated ** 2) + np.sum(psi_control ** 2)
            se = np.sqrt(var_psi) if var_psi > 0 else 0.0

            # Full influence function
            inf_func = np.concatenate([psi_treated, psi_control])
        else:
            # Without covariates, DR simplifies to difference in means
            att = np.mean(treated_change) - np.mean(control_change)

            var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
            var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

            se = np.sqrt(var_t / n_t + var_c / n_c) if (n_t > 0 and n_c > 0) else 0.0

            # Influence function for DR estimator
            inf_treated = (treated_change - np.mean(treated_change)) / n_t
            inf_control = (control_change - np.mean(control_change)) / n_c
            inf_func = np.concatenate([inf_treated, -inf_control])

        return att, se, inf_func

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
