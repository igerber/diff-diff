"""
Staggered Difference-in-Differences estimators.

Implements modern methods for DiD with variation in treatment timing,
including the Callaway-Sant'Anna (2021) estimator.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _get_significance_stars
from diff_diff.utils import (
    compute_confidence_interval,
    compute_p_value,
)


@dataclass
class GroupTimeEffect:
    """
    Treatment effect for a specific group-time combination.

    Attributes
    ----------
    group : any
        The treatment cohort (first treatment period).
    time : any
        The time period.
    effect : float
        The ATT(g,t) estimate.
    se : float
        Standard error.
    n_treated : int
        Number of treated observations.
    n_control : int
        Number of control observations.
    """
    group: Any
    time: Any
    effect: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_treated: int
    n_control: int

    @property
    def is_significant(self) -> bool:
        """Check if effect is significant at 0.05 level."""
        return bool(self.p_value < 0.05)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)


@dataclass
class CallawaySantAnnaResults:
    """
    Results from Callaway-Sant'Anna (2021) staggered DiD estimation.

    This class stores group-time average treatment effects ATT(g,t) and
    provides methods for aggregation into summary measures.

    Attributes
    ----------
    group_time_effects : dict
        Dictionary mapping (group, time) tuples to effect dictionaries.
    overall_att : float
        Overall average treatment effect (weighted average of ATT(g,t)).
    overall_se : float
        Standard error of overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    groups : list
        List of treatment cohorts (first treatment periods).
    time_periods : list
        List of all time periods.
    n_obs : int
        Total number of observations.
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of never-treated units.
    event_study_effects : dict, optional
        Effects aggregated by relative time (event study).
    group_effects : dict, optional
        Effects aggregated by treatment cohort.
    """
    group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]]
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_units: int
    n_control_units: int
    alpha: float = 0.05
    control_group: str = "never_treated"
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None)
    group_effects: Optional[Dict[Any, Dict[str, Any]]] = field(default=None)
    influence_functions: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"CallawaySantAnnaResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_groups={len(self.groups)}, "
            f"n_periods={len(self.time_periods)})"
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate formatted summary of estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level. Defaults to alpha used in estimation.

        Returns
        -------
        str
            Formatted summary.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 85,
            "Callaway-Sant'Anna Staggered Difference-in-Differences Results".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            f"{'Control group:':<30} {self.control_group:>10}",
            "",
        ]

        # Overall ATT
        lines.extend([
            "-" * 85,
            "Overall Average Treatment Effect on the Treated".center(85),
            "-" * 85,
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
            "-" * 85,
            f"{'ATT':<15} {self.overall_att:>12.4f} {self.overall_se:>12.4f} "
            f"{self.overall_t_stat:>10.3f} {self.overall_p_value:>10.4f} "
            f"{_get_significance_stars(self.overall_p_value):>6}",
            "-" * 85,
            "",
            f"{conf_level}% Confidence Interval: [{self.overall_conf_int[0]:.4f}, {self.overall_conf_int[1]:.4f}]",
            "",
        ])

        # Event study effects if available
        if self.event_study_effects:
            lines.extend([
                "-" * 85,
                "Event Study (Dynamic) Effects".center(85),
                "-" * 85,
                f"{'Rel. Period':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
            ])

            for rel_t in sorted(self.event_study_effects.keys()):
                eff = self.event_study_effects[rel_t]
                sig = _get_significance_stars(eff['p_value'])
                lines.append(
                    f"{rel_t:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                    f"{eff['t_stat']:>10.3f} {eff['p_value']:>10.4f} {sig:>6}"
                )

            lines.extend(["-" * 85, ""])

        # Group effects if available
        if self.group_effects:
            lines.extend([
                "-" * 85,
                "Effects by Treatment Cohort".center(85),
                "-" * 85,
                f"{'Cohort':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
            ])

            for group in sorted(self.group_effects.keys()):
                eff = self.group_effects[group]
                sig = _get_significance_stars(eff['p_value'])
                lines.append(
                    f"{group:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                    f"{eff['t_stat']:>10.3f} {eff['p_value']:>10.4f} {sig:>6}"
                )

            lines.extend(["-" * 85, ""])

        lines.extend([
            "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
            "=" * 85,
        ])

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print summary to stdout."""
        print(self.summary(alpha))

    def to_dataframe(self, level: str = "group_time") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="group_time"
            Level of aggregation: "group_time", "event_study", or "group".

        Returns
        -------
        pd.DataFrame
            Results as DataFrame.
        """
        if level == "group_time":
            rows = []
            for (g, t), data in self.group_time_effects.items():
                rows.append({
                    'group': g,
                    'time': t,
                    'effect': data['effect'],
                    'se': data['se'],
                    't_stat': data['t_stat'],
                    'p_value': data['p_value'],
                    'conf_int_lower': data['conf_int'][0],
                    'conf_int_upper': data['conf_int'][1],
                })
            return pd.DataFrame(rows)

        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError("Event study effects not computed. Use aggregate='event_study'.")
            rows = []
            for rel_t, data in sorted(self.event_study_effects.items()):
                rows.append({
                    'relative_period': rel_t,
                    'effect': data['effect'],
                    'se': data['se'],
                    't_stat': data['t_stat'],
                    'p_value': data['p_value'],
                    'conf_int_lower': data['conf_int'][0],
                    'conf_int_upper': data['conf_int'][1],
                })
            return pd.DataFrame(rows)

        elif level == "group":
            if self.group_effects is None:
                raise ValueError("Group effects not computed. Use aggregate='group'.")
            rows = []
            for group, data in sorted(self.group_effects.items()):
                rows.append({
                    'group': group,
                    'effect': data['effect'],
                    'se': data['se'],
                    't_stat': data['t_stat'],
                    'p_value': data['p_value'],
                    'conf_int_lower': data['conf_int'][0],
                    'conf_int_upper': data['conf_int'][1],
                })
            return pd.DataFrame(rows)

        else:
            raise ValueError(f"Unknown level: {level}. Use 'group_time', 'event_study', or 'group'.")

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)


class CallawaySantAnna:
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
    seed : int, optional
        Random seed for reproducibility.

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

    Notes
    -----
    The key innovation of Callaway & Sant'Anna (2021) is the disaggregated
    approach: instead of estimating a single treatment effect, they estimate
    ATT(g,t) for each cohort-time pair. This avoids the "forbidden comparison"
    problem where already-treated units act as controls.

    The ATT(g,t) is identified under parallel trends conditional on covariates:

        E[Y(0)_t - Y(0)_g-1 | G=g] = E[Y(0)_t - Y(0)_g-1 | C=1]

    where G=g indicates treatment cohort g and C=1 indicates control units.

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
        seed: Optional[int] = None,
    ):
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

        self.control_group = control_group
        self.anticipation = anticipation
        self.estimation_method = estimation_method
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.seed = seed

        self.is_fitted_ = False
        self.results_ = None

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

        # Check for unimplemented features
        if self.n_bootstrap > 0:
            raise NotImplementedError(
                "Bootstrap inference is not yet implemented. "
                "Use n_bootstrap=0 for analytical standard errors."
            )

        if covariates:
            warnings.warn(
                "Covariates are accepted but not yet used in estimation. "
                "The current implementation uses unconditional parallel trends. "
                "Covariate-adjusted estimation will be added in a future version.",
                UserWarning,
                stacklevel=2,
            )

        # Create working copy
        df = data.copy()

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

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

        # Compute ATT(g,t) for each group-time combination
        group_time_effects = {}
        influence_funcs = []

        for g in treatment_groups:
            # Periods for which we compute effects (t >= g - anticipation)
            valid_periods = [t for t in time_periods if t >= g - self.anticipation]

            for t in valid_periods:
                att_gt, se_gt, n_treat, n_ctrl, inf_func = self._compute_att_gt(
                    df, outcome, unit, time, first_treat, g, t,
                    covariates, time_periods
                )

                if att_gt is not None:
                    t_stat = att_gt / se_gt if se_gt > 0 else 0.0
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

                    if inf_func is not None:
                        influence_funcs.append(inf_func)

        if not group_time_effects:
            raise ValueError(
                "Could not estimate any group-time effects. "
                "Check that data has sufficient observations."
            )

        # Compute overall ATT (simple aggregation)
        overall_att, overall_se = self._aggregate_simple(group_time_effects, df, unit)
        overall_t = overall_att / overall_se if overall_se > 0 else 0.0
        overall_p = compute_p_value(overall_t)
        overall_ci = compute_confidence_interval(overall_att, overall_se, self.alpha)

        # Compute additional aggregations if requested
        event_study_effects = None
        group_effects = None

        if aggregate in ["event_study", "all"]:
            event_study_effects = self._aggregate_event_study(
                group_time_effects, treatment_groups, time_periods, balance_e
            )

        if aggregate in ["group", "all"]:
            group_effects = self._aggregate_by_group(
                group_time_effects, treatment_groups
            )

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
            event_study_effects=event_study_effects,
            group_effects=group_effects,
        )

        self.is_fitted_ = True
        return self.results_

    def _compute_att_gt(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        g: Any,
        t: Any,
        covariates: Optional[List[str]],
        all_periods: List[Any],
    ) -> Tuple[Optional[float], float, int, int, Optional[np.ndarray]]:
        """
        Compute ATT(g,t) for a specific group-time combination.

        Uses 2x2 DiD comparing:
        - Treated: Units in cohort g
        - Control: Never-treated units (or not-yet-treated if specified)
        - Pre-period: g - 1 (or earlier if anticipation > 0)
        - Post-period: t
        """
        # Base period for comparison
        base_period = g - 1 - self.anticipation
        if base_period not in all_periods:
            # Find closest earlier period
            earlier = [p for p in all_periods if p < g - self.anticipation]
            if not earlier:
                return None, 0.0, 0, 0, None
            base_period = max(earlier)

        # Treated group: units first treated in period g
        treated_units = df[df[first_treat] == g][unit].unique()

        # Control group
        if self.control_group == "never_treated":
            control_mask = df['_never_treated']
        else:  # not_yet_treated
            # Not yet treated at time t
            control_mask = (df['_never_treated']) | (df[first_treat] > t)

        control_units = df[control_mask][unit].unique()

        if len(treated_units) == 0 or len(control_units) == 0:
            return None, 0.0, 0, 0, None

        # Get data for the two periods
        df_base = df[df[time] == base_period].set_index(unit)
        df_post = df[df[time] == t].set_index(unit)

        # Compute outcome changes for treated
        treated_base = df_base.loc[df_base.index.isin(treated_units), outcome]
        treated_post = df_post.loc[df_post.index.isin(treated_units), outcome]
        treated_common = treated_base.index.intersection(treated_post.index)

        if len(treated_common) == 0:
            return None, 0.0, 0, 0, None

        treated_change = (
            treated_post.loc[treated_common].values -
            treated_base.loc[treated_common].values
        )

        # Compute outcome changes for control
        control_base = df_base.loc[df_base.index.isin(control_units), outcome]
        control_post = df_post.loc[df_post.index.isin(control_units), outcome]
        control_common = control_base.index.intersection(control_post.index)

        if len(control_common) == 0:
            return None, 0.0, 0, 0, None

        control_change = (
            control_post.loc[control_common].values -
            control_base.loc[control_common].values
        )

        # Estimation method
        if self.estimation_method == "reg":
            att_gt, se_gt, inf_func = self._outcome_regression(
                treated_change, control_change
            )
        elif self.estimation_method == "ipw":
            att_gt, se_gt, inf_func = self._ipw_estimation(
                treated_change, control_change,
                len(treated_common), len(control_common)
            )
        else:  # doubly robust
            att_gt, se_gt, inf_func = self._doubly_robust(
                treated_change, control_change
            )

        return att_gt, se_gt, len(treated_common), len(control_common), inf_func

    def _outcome_regression(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using outcome regression (difference in means).
        """
        # Simple difference in means
        att = np.mean(treated_change) - np.mean(control_change)

        # Standard error
        n_t = len(treated_change)
        n_c = len(control_change)

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
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using inverse probability weighting.
        """
        # Without covariates, this reduces to difference in means
        # but with different weighting
        n_total = n_treated + n_control
        p_treat = n_treated / n_total  # propensity score (unconditional)

        # ATT = mean(Y_treated) - weighted mean(Y_control)
        att = np.mean(treated_change) - np.mean(control_change)

        # SE with IPW adjustment
        n_t = len(treated_change)
        n_c = len(control_change)

        var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
        var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

        # Adjusted variance for IPW
        se = np.sqrt(var_t / n_t + var_c * (1 - p_treat) / (n_c * p_treat)) if (n_t > 0 and n_c > 0) else 0.0

        inf_func = np.array([])  # Placeholder

        return att, se, inf_func

    def _doubly_robust(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using doubly robust estimation.

        Without covariates, this is equivalent to the outcome regression
        estimator but provides consistent estimates even if one of the
        two models (outcome or propensity) is misspecified.
        """
        # Without covariates, DR simplifies to difference in means
        # The "doubly robust" property requires covariates
        att = np.mean(treated_change) - np.mean(control_change)

        # Standard error
        n_t = len(treated_change)
        n_c = len(control_change)

        var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
        var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

        se = np.sqrt(var_t / n_t + var_c / n_c) if (n_t > 0 and n_c > 0) else 0.0

        # Influence function for DR estimator
        inf_treated = (treated_change - np.mean(treated_change)) / n_t
        inf_control = (control_change - np.mean(control_change)) / n_c
        inf_func = np.concatenate([inf_treated, -inf_control])

        return att, se, inf_func

    def _aggregate_simple(
        self,
        group_time_effects: Dict,
        df: pd.DataFrame,
        unit: str,
    ) -> Tuple[float, float]:
        """
        Compute simple weighted average of ATT(g,t).

        Weights by group size (number of treated units).
        """
        effects = []
        weights = []
        variances = []

        for (g, t), data in group_time_effects.items():
            effects.append(data['effect'])
            weights.append(data['n_treated'])
            variances.append(data['se'] ** 2)

        effects = np.array(effects)
        weights = np.array(weights, dtype=float)
        variances = np.array(variances)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Weighted average
        overall_att = np.sum(weights * effects)

        # Standard error (assuming independence across g,t)
        overall_var = np.sum((weights ** 2) * variances)
        overall_se = np.sqrt(overall_var)

        return overall_att, overall_se

    def _aggregate_event_study(
        self,
        group_time_effects: Dict,
        groups: List[Any],
        time_periods: List[Any],
        balance_e: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Aggregate effects by relative time (event study).

        Computes average effect at each event time e = t - g.
        """
        # Organize effects by relative time
        effects_by_e: Dict[int, List[Tuple[float, float, int]]] = {}

        for (g, t), data in group_time_effects.items():
            e = t - g  # Relative time
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append((
                data['effect'],
                data['se'],
                data['n_treated']
            ))

        # Balance the panel if requested
        if balance_e is not None:
            # Keep only groups that have effects at relative time balance_e
            groups_at_e = set()
            for (g, t), data in group_time_effects.items():
                if t - g == balance_e:
                    groups_at_e.add(g)

            # Filter effects to only include balanced groups
            balanced_effects: Dict[int, List[Tuple[float, float, int]]] = {}
            for (g, t), data in group_time_effects.items():
                if g in groups_at_e:
                    e = t - g
                    if e not in balanced_effects:
                        balanced_effects[e] = []
                    balanced_effects[e].append((
                        data['effect'],
                        data['se'],
                        data['n_treated']
                    ))
            effects_by_e = balanced_effects

        # Compute aggregated effects
        event_study_effects = {}

        for e, effect_list in sorted(effects_by_e.items()):
            effs = np.array([x[0] for x in effect_list])
            ses = np.array([x[1] for x in effect_list])
            ns = np.array([x[2] for x in effect_list], dtype=float)

            # Weight by group size
            weights = ns / np.sum(ns)

            agg_effect = np.sum(weights * effs)
            agg_var = np.sum((weights ** 2) * (ses ** 2))
            agg_se = np.sqrt(agg_var)

            t_stat = agg_effect / agg_se if agg_se > 0 else 0.0
            p_val = compute_p_value(t_stat)
            ci = compute_confidence_interval(agg_effect, agg_se, self.alpha)

            event_study_effects[e] = {
                'effect': agg_effect,
                'se': agg_se,
                't_stat': t_stat,
                'p_value': p_val,
                'conf_int': ci,
                'n_groups': len(effect_list),
            }

        return event_study_effects

    def _aggregate_by_group(
        self,
        group_time_effects: Dict,
        groups: List[Any],
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Aggregate effects by treatment cohort.

        Computes average effect for each cohort across all post-treatment periods.
        """
        group_effects = {}

        for g in groups:
            # Get all effects for this group (post-treatment only: t >= g)
            g_effects = [
                (data['effect'], data['se'], data['n_treated'])
                for (gg, t), data in group_time_effects.items()
                if gg == g and t >= g
            ]

            if not g_effects:
                continue

            effs = np.array([x[0] for x in g_effects])
            ses = np.array([x[1] for x in g_effects])

            # Equal weight across time periods for a group
            weights = np.ones(len(effs)) / len(effs)

            agg_effect = np.sum(weights * effs)
            agg_var = np.sum((weights ** 2) * (ses ** 2))
            agg_se = np.sqrt(agg_var)

            t_stat = agg_effect / agg_se if agg_se > 0 else 0.0
            p_val = compute_p_value(t_stat)
            ci = compute_confidence_interval(agg_effect, agg_se, self.alpha)

            group_effects[g] = {
                'effect': agg_effect,
                'se': agg_se,
                't_stat': t_stat,
                'p_value': p_val,
                'conf_int': ci,
                'n_periods': len(g_effects),
            }

        return group_effects

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "control_group": self.control_group,
            "anticipation": self.anticipation,
            "estimation_method": self.estimation_method,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
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
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())
