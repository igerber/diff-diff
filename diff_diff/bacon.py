"""
Goodman-Bacon Decomposition for Two-Way Fixed Effects.

Implements the decomposition from Goodman-Bacon (2021) that shows how
TWFE estimates with staggered treatment timing can be written as a
weighted average of all possible 2x2 DiD comparisons.

Reference:
    Goodman-Bacon, A. (2021). Difference-in-differences with variation
    in treatment timing. Journal of Econometrics, 225(2), 254-277.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Comparison2x2:
    """
    A single 2x2 DiD comparison in the Bacon decomposition.

    Attributes
    ----------
    treated_group : Any
        The timing group used as "treated" in this comparison.
    control_group : Any
        The timing group used as "control" in this comparison.
    comparison_type : str
        Type of comparison: "treated_vs_never", "earlier_vs_later",
        or "later_vs_earlier".
    estimate : float
        The 2x2 DiD estimate for this comparison.
    weight : float
        The weight assigned to this comparison in the TWFE average.
    n_treated : int
        Number of treated observations in this comparison.
    n_control : int
        Number of control observations in this comparison.
    time_window : Tuple[Any, Any]
        The (start, end) time period for this comparison.
    """

    treated_group: Any
    control_group: Any
    comparison_type: str
    estimate: float
    weight: float
    n_treated: int
    n_control: int
    time_window: Tuple[Any, Any]

    def __repr__(self) -> str:
        return (
            f"Comparison2x2({self.treated_group} vs {self.control_group}, "
            f"type={self.comparison_type}, β={self.estimate:.4f}, "
            f"weight={self.weight:.4f})"
        )


@dataclass
class BaconDecompositionResults:
    """
    Results from Goodman-Bacon decomposition of TWFE.

    This decomposition shows that the TWFE estimate equals a weighted
    average of all possible 2x2 DiD comparisons between timing groups.

    Attributes
    ----------
    twfe_estimate : float
        The overall TWFE coefficient (should equal weighted sum of 2x2 estimates).
    comparisons : List[Comparison2x2]
        List of all 2x2 comparisons with their estimates and weights.
    total_weight_treated_vs_never : float
        Total weight on treated vs never-treated comparisons.
    total_weight_earlier_vs_later : float
        Total weight on earlier vs later treated comparisons.
    total_weight_later_vs_earlier : float
        Total weight on later vs earlier treated comparisons (forbidden).
    weighted_avg_treated_vs_never : float
        Weighted average effect from treated vs never-treated comparisons.
    weighted_avg_earlier_vs_later : float
        Weighted average effect from earlier vs later comparisons.
    weighted_avg_later_vs_earlier : float
        Weighted average effect from later vs earlier comparisons.
    n_timing_groups : int
        Number of distinct treatment timing groups.
    n_never_treated : int
        Number of never-treated units.
    timing_groups : List[Any]
        List of treatment timing cohorts.
    """

    twfe_estimate: float
    comparisons: List[Comparison2x2]
    total_weight_treated_vs_never: float
    total_weight_earlier_vs_later: float
    total_weight_later_vs_earlier: float
    weighted_avg_treated_vs_never: Optional[float]
    weighted_avg_earlier_vs_later: Optional[float]
    weighted_avg_later_vs_earlier: Optional[float]
    n_timing_groups: int
    n_never_treated: int
    timing_groups: List[Any]
    n_obs: int = 0
    decomposition_error: float = field(default=0.0)

    def __repr__(self) -> str:
        return (
            f"BaconDecompositionResults(TWFE={self.twfe_estimate:.4f}, "
            f"n_comparisons={len(self.comparisons)}, "
            f"n_groups={self.n_timing_groups})"
        )

    def summary(self) -> str:
        """
        Generate a formatted summary of the decomposition.

        Returns
        -------
        str
            Formatted summary table.
        """
        lines = [
            "=" * 85,
            "Goodman-Bacon Decomposition of Two-Way Fixed Effects".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<35} {self.n_obs:>10}",
            f"{'Treatment timing groups:':<35} {self.n_timing_groups:>10}",
            f"{'Never-treated units:':<35} {self.n_never_treated:>10}",
            f"{'Total 2x2 comparisons:':<35} {len(self.comparisons):>10}",
            "",
            "-" * 85,
            "TWFE Decomposition".center(85),
            "-" * 85,
            "",
            f"{'TWFE Estimate:':<35} {self.twfe_estimate:>12.4f}",
            f"{'Weighted Sum of 2x2 Estimates:':<35} {self._weighted_sum():>12.4f}",
            f"{'Decomposition Error:':<35} {self.decomposition_error:>12.6f}",
            "",
        ]

        # Weight breakdown by comparison type
        lines.extend([
            "-" * 85,
            "Weight Breakdown by Comparison Type".center(85),
            "-" * 85,
            f"{'Comparison Type':<30} {'Weight':>12} {'Avg Effect':>12} {'Contribution':>12}",
            "-" * 85,
        ])

        # Treated vs Never-treated
        if self.total_weight_treated_vs_never > 0:
            contrib = self.total_weight_treated_vs_never * (
                self.weighted_avg_treated_vs_never or 0
            )
            lines.append(
                f"{'Treated vs Never-treated':<30} "
                f"{self.total_weight_treated_vs_never:>12.4f} "
                f"{self.weighted_avg_treated_vs_never or 0:>12.4f} "
                f"{contrib:>12.4f}"
            )

        # Earlier vs Later
        if self.total_weight_earlier_vs_later > 0:
            contrib = self.total_weight_earlier_vs_later * (
                self.weighted_avg_earlier_vs_later or 0
            )
            lines.append(
                f"{'Earlier vs Later treated':<30} "
                f"{self.total_weight_earlier_vs_later:>12.4f} "
                f"{self.weighted_avg_earlier_vs_later or 0:>12.4f} "
                f"{contrib:>12.4f}"
            )

        # Later vs Earlier (forbidden)
        if self.total_weight_later_vs_earlier > 0:
            contrib = self.total_weight_later_vs_earlier * (
                self.weighted_avg_later_vs_earlier or 0
            )
            lines.append(
                f"{'Later vs Earlier (forbidden)':<30} "
                f"{self.total_weight_later_vs_earlier:>12.4f} "
                f"{self.weighted_avg_later_vs_earlier or 0:>12.4f} "
                f"{contrib:>12.4f}"
            )

        lines.extend([
            "-" * 85,
            f"{'Total':<30} {self._total_weight():>12.4f} "
            f"{'':>12} {self._weighted_sum():>12.4f}",
            "-" * 85,
            "",
        ])

        # Warning about forbidden comparisons
        if self.total_weight_later_vs_earlier > 0.01:
            pct = self.total_weight_later_vs_earlier * 100
            lines.extend([
                "WARNING: {:.1f}% of weight is on 'forbidden' comparisons where".format(
                    pct
                ),
                "already-treated units serve as controls. This can bias TWFE",
                "when treatment effects are heterogeneous over time.",
                "",
                "Consider using Callaway-Sant'Anna or other robust estimators.",
                "",
            ])

        lines.append("=" * 85)

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def _weighted_sum(self) -> float:
        """Calculate weighted sum of 2x2 estimates."""
        return sum(c.weight * c.estimate for c in self.comparisons)

    def _total_weight(self) -> float:
        """Calculate total weight (should be 1.0)."""
        return sum(c.weight for c in self.comparisons)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert comparisons to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per 2x2 comparison.
        """
        rows = []
        for c in self.comparisons:
            rows.append({
                "treated_group": c.treated_group,
                "control_group": c.control_group,
                "comparison_type": c.comparison_type,
                "estimate": c.estimate,
                "weight": c.weight,
                "n_treated": c.n_treated,
                "n_control": c.n_control,
                "time_start": c.time_window[0],
                "time_end": c.time_window[1],
            })
        return pd.DataFrame(rows)

    def weight_by_type(self) -> Dict[str, float]:
        """
        Get total weight by comparison type.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping comparison type to total weight.
        """
        return {
            "treated_vs_never": self.total_weight_treated_vs_never,
            "earlier_vs_later": self.total_weight_earlier_vs_later,
            "later_vs_earlier": self.total_weight_later_vs_earlier,
        }

    def effect_by_type(self) -> Dict[str, Optional[float]]:
        """
        Get weighted average effect by comparison type.

        Returns
        -------
        Dict[str, Optional[float]]
            Dictionary mapping comparison type to weighted average effect.
        """
        return {
            "treated_vs_never": self.weighted_avg_treated_vs_never,
            "earlier_vs_later": self.weighted_avg_earlier_vs_later,
            "later_vs_earlier": self.weighted_avg_later_vs_earlier,
        }


class BaconDecomposition:
    """
    Goodman-Bacon (2021) decomposition of Two-Way Fixed Effects estimator.

    This class decomposes a TWFE estimate into a weighted average of all
    possible 2x2 DiD comparisons, revealing the implicit comparisons that
    drive the TWFE estimate and their relative importance.

    The decomposition identifies three types of comparisons:

    1. **Treated vs Never-treated**: Uses never-treated units as controls.
       These are "clean" comparisons without bias concerns.

    2. **Earlier vs Later treated**: Units treated earlier are compared to
       units treated later, using the later group as controls before they
       are treated. These are valid comparisons.

    3. **Later vs Earlier treated**: Units treated later are compared to
       units treated earlier, using the earlier group as controls AFTER
       they are already treated. These are "forbidden comparisons" that
       can introduce bias when treatment effects vary over time.

    Parameters
    ----------
    None

    Attributes
    ----------
    results_ : BaconDecompositionResults
        Decomposition results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> from diff_diff import BaconDecomposition
    >>>
    >>> # Panel data with staggered treatment
    >>> data = pd.DataFrame({
    ...     'unit': [...],
    ...     'time': [...],
    ...     'outcome': [...],
    ...     'first_treat': [...]  # 0 for never-treated
    ... })
    >>>
    >>> bacon = BaconDecomposition()
    >>> results = bacon.fit(data, outcome='outcome', unit='unit',
    ...                     time='time', first_treat='first_treat')
    >>> results.print_summary()

    Visualizing the decomposition:

    >>> from diff_diff import plot_bacon
    >>> plot_bacon(results)

    Notes
    -----
    The key insight from Goodman-Bacon (2021) is that TWFE with staggered
    treatment timing implicitly makes comparisons using already-treated
    units as controls. When treatment effects are dynamic (changing over
    time since treatment), these "forbidden comparisons" can bias the
    TWFE estimate, potentially even reversing its sign.

    The decomposition helps diagnose this issue by showing:
    - How much weight is on each type of comparison
    - Whether forbidden comparisons contribute significantly to the estimate
    - How the 2x2 estimates vary across comparison types

    If forbidden comparisons have substantial weight and different estimates
    than clean comparisons, consider using robust estimators like
    Callaway-Sant'Anna that avoid these problematic comparisons.

    References
    ----------
    Goodman-Bacon, A. (2021). Difference-in-differences with variation in
    treatment timing. Journal of Econometrics, 225(2), 254-277.

    See Also
    --------
    CallawaySantAnna : Robust estimator for staggered DiD
    TwoWayFixedEffects : The TWFE estimator being decomposed
    """

    def __init__(self):
        self.results_: Optional[BaconDecompositionResults] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
    ) -> BaconDecompositionResults:
        """
        Perform the Goodman-Bacon decomposition.

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

        Returns
        -------
        BaconDecompositionResults
            Object containing decomposition results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # Validate inputs
        required_cols = [outcome, unit, time, first_treat]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Create working copy
        df = data.copy()

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Get unique time periods and timing groups
        time_periods = sorted(df[time].unique())

        # Identify never-treated and timing groups
        # Never-treated: first_treat = 0 or inf
        never_treated_mask = (df[first_treat] == 0) | (df[first_treat] == np.inf)
        timing_groups = sorted([g for g in df[first_treat].unique()
                               if g > 0 and g != np.inf])

        # Get unit-level treatment timing
        unit_info = df.groupby(unit).agg({first_treat: 'first'}).reset_index()
        n_never_treated = (
            (unit_info[first_treat] == 0) | (unit_info[first_treat] == np.inf)
        ).sum()

        # Create treatment indicator (D_it = 1 if treated at time t)
        df['_treated'] = (~never_treated_mask) & (df[time] >= df[first_treat])

        # First, compute TWFE estimate for reference
        twfe_estimate = self._compute_twfe(df, outcome, unit, time)

        # Perform decomposition
        comparisons = []

        # 1. Treated vs Never-treated comparisons
        if n_never_treated > 0:
            for g in timing_groups:
                comp = self._compute_treated_vs_never(
                    df, outcome, unit, time, first_treat, g, time_periods
                )
                if comp is not None:
                    comparisons.append(comp)

        # 2. Timing group comparisons (earlier vs later and later vs earlier)
        for i, g_early in enumerate(timing_groups):
            for g_late in timing_groups[i + 1:]:
                # Earlier vs Later: g_early treated, g_late as control
                comp_early = self._compute_timing_comparison(
                    df, outcome, unit, time, first_treat,
                    g_early, g_late, time_periods, "earlier_vs_later"
                )
                if comp_early is not None:
                    comparisons.append(comp_early)

                # Later vs Earlier: g_late treated, g_early as control (forbidden)
                comp_late = self._compute_timing_comparison(
                    df, outcome, unit, time, first_treat,
                    g_late, g_early, time_periods, "later_vs_earlier"
                )
                if comp_late is not None:
                    comparisons.append(comp_late)

        # Normalize weights to sum to 1
        total_weight = sum(c.weight for c in comparisons)
        if total_weight > 0:
            for c in comparisons:
                c.weight = c.weight / total_weight

        # Calculate weight totals and weighted averages by type
        weight_by_type = {"treated_vs_never": 0.0, "earlier_vs_later": 0.0,
                         "later_vs_earlier": 0.0}
        weighted_sum_by_type = {"treated_vs_never": 0.0, "earlier_vs_later": 0.0,
                               "later_vs_earlier": 0.0}

        for c in comparisons:
            weight_by_type[c.comparison_type] += c.weight
            weighted_sum_by_type[c.comparison_type] += c.weight * c.estimate

        # Calculate weighted averages
        avg_by_type = {}
        for ctype in weight_by_type:
            if weight_by_type[ctype] > 0:
                avg_by_type[ctype] = (
                    weighted_sum_by_type[ctype] / weight_by_type[ctype]
                )
            else:
                avg_by_type[ctype] = None

        # Calculate decomposition error
        weighted_sum = sum(c.weight * c.estimate for c in comparisons)
        decomp_error = abs(twfe_estimate - weighted_sum)

        self.results_ = BaconDecompositionResults(
            twfe_estimate=twfe_estimate,
            comparisons=comparisons,
            total_weight_treated_vs_never=weight_by_type["treated_vs_never"],
            total_weight_earlier_vs_later=weight_by_type["earlier_vs_later"],
            total_weight_later_vs_earlier=weight_by_type["later_vs_earlier"],
            weighted_avg_treated_vs_never=avg_by_type["treated_vs_never"],
            weighted_avg_earlier_vs_later=avg_by_type["earlier_vs_later"],
            weighted_avg_later_vs_earlier=avg_by_type["later_vs_earlier"],
            n_timing_groups=len(timing_groups),
            n_never_treated=n_never_treated,
            timing_groups=timing_groups,
            n_obs=len(df),
            decomposition_error=decomp_error,
        )

        self.is_fitted_ = True
        return self.results_

    def _compute_twfe(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
    ) -> float:
        """Compute TWFE estimate using within-transformation."""
        # Demean by unit and time
        y = df[outcome].values
        d = df['_treated'].astype(float).values

        # Create unit and time dummies for demeaning
        units = df[unit].values
        times = df[time].values

        # Unit means
        unit_map = {u: i for i, u in enumerate(df[unit].unique())}
        unit_idx = np.array([unit_map[u] for u in units])
        n_units = len(unit_map)

        # Time means
        time_map = {t: i for i, t in enumerate(df[time].unique())}
        time_idx = np.array([time_map[t] for t in times])
        n_times = len(time_map)

        # Compute means
        y_unit_mean = np.zeros(n_units)
        d_unit_mean = np.zeros(n_units)
        unit_counts = np.zeros(n_units)

        for i in range(len(y)):
            u = unit_idx[i]
            y_unit_mean[u] += y[i]
            d_unit_mean[u] += d[i]
            unit_counts[u] += 1

        y_unit_mean /= np.maximum(unit_counts, 1)
        d_unit_mean /= np.maximum(unit_counts, 1)

        y_time_mean = np.zeros(n_times)
        d_time_mean = np.zeros(n_times)
        time_counts = np.zeros(n_times)

        for i in range(len(y)):
            t = time_idx[i]
            y_time_mean[t] += y[i]
            d_time_mean[t] += d[i]
            time_counts[t] += 1

        y_time_mean /= np.maximum(time_counts, 1)
        d_time_mean /= np.maximum(time_counts, 1)

        # Overall mean
        y_mean = np.mean(y)
        d_mean = np.mean(d)

        # Within transformation: y_it - y_i - y_t + y
        y_within = np.zeros(len(y))
        d_within = np.zeros(len(d))

        for i in range(len(y)):
            u = unit_idx[i]
            t = time_idx[i]
            y_within[i] = y[i] - y_unit_mean[u] - y_time_mean[t] + y_mean
            d_within[i] = d[i] - d_unit_mean[u] - d_time_mean[t] + d_mean

        # OLS on demeaned data
        d_var = np.sum(d_within ** 2)
        if d_var > 0:
            beta = np.sum(d_within * y_within) / d_var
        else:
            beta = 0.0

        return beta

    def _compute_treated_vs_never(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        treated_group: Any,
        time_periods: List[Any],
    ) -> Optional[Comparison2x2]:
        """
        Compute 2x2 DiD comparing treated group to never-treated.

        This is a "clean" comparison using the full sample of a treated
        cohort versus never-treated units.
        """
        # Get treated and never-treated units
        never_mask = (df[first_treat] == 0) | (df[first_treat] == np.inf)
        treated_mask = df[first_treat] == treated_group

        df_treated = df[treated_mask]
        df_never = df[never_mask]

        if len(df_treated) == 0 or len(df_never) == 0:
            return None

        # Time window: all periods
        t_min = min(time_periods)
        t_max = max(time_periods)

        # Pre and post periods for this group
        pre_periods = [t for t in time_periods if t < treated_group]
        post_periods = [t for t in time_periods if t >= treated_group]

        if not pre_periods or not post_periods:
            return None

        # Compute 2x2 DiD estimate
        # Mean change for treated
        treated_pre = df_treated[df_treated[time].isin(pre_periods)][outcome].mean()
        treated_post = df_treated[df_treated[time].isin(post_periods)][outcome].mean()

        # Mean change for never-treated
        never_pre = df_never[df_never[time].isin(pre_periods)][outcome].mean()
        never_post = df_never[df_never[time].isin(post_periods)][outcome].mean()

        estimate = (treated_post - treated_pre) - (never_post - never_pre)

        # Calculate weight components
        n_treated = df_treated[unit].nunique()
        n_never = df_never[unit].nunique()
        n_total = n_treated + n_never

        # Group share
        n_k = n_treated / n_total

        # Variance of treatment: proportion of post-treatment periods
        D_k = len(post_periods) / len(time_periods)

        # Weight is proportional to n_k * (1 - n_k) * Var(D_k)
        # Var(D) for treated group = D_k * (1 - D_k)
        weight = n_k * (1 - n_k) * D_k * (1 - D_k)

        return Comparison2x2(
            treated_group=treated_group,
            control_group="never_treated",
            comparison_type="treated_vs_never",
            estimate=estimate,
            weight=weight,
            n_treated=n_treated,
            n_control=n_never,
            time_window=(t_min, t_max),
        )

    def _compute_timing_comparison(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        treated_group: Any,
        control_group: Any,
        time_periods: List[Any],
        comparison_type: str,
    ) -> Optional[Comparison2x2]:
        """
        Compute 2x2 DiD comparing two timing groups.

        For earlier_vs_later: uses later group as controls before they're treated.
        For later_vs_earlier: uses earlier group as controls after treatment (forbidden).
        """
        treated_mask = df[first_treat] == treated_group
        control_mask = df[first_treat] == control_group

        df_treated = df[treated_mask]
        df_control = df[control_mask]

        if len(df_treated) == 0 or len(df_control) == 0:
            return None

        n_treated = df_treated[unit].nunique()
        n_control = df_control[unit].nunique()
        n_total = n_treated + n_control

        if comparison_type == "earlier_vs_later":
            # Earlier treated vs Later treated
            # Time window: from start to when later group gets treated
            # Pre: before earlier group treated
            # Post: after earlier treated but before later treated
            g_early = treated_group
            g_late = control_group

            # Pre-period: before g_early
            pre_periods = [t for t in time_periods if t < g_early]
            # Post-period: g_early <= t < g_late (middle period)
            post_periods = [t for t in time_periods if g_early <= t < g_late]

            if not pre_periods or not post_periods:
                return None

            time_window = (min(time_periods), g_late - 1)

        else:  # later_vs_earlier (forbidden)
            # Later treated vs Earlier treated (used as control after treatment)
            g_late = treated_group
            g_early = control_group

            # Pre-period: after g_early treated but before g_late treated
            pre_periods = [t for t in time_periods if g_early <= t < g_late]
            # Post-period: after g_late treated
            post_periods = [t for t in time_periods if t >= g_late]

            if not pre_periods or not post_periods:
                return None

            time_window = (g_early, max(time_periods))

        # Compute 2x2 DiD estimate
        treated_pre = df_treated[df_treated[time].isin(pre_periods)][outcome].mean()
        treated_post = df_treated[df_treated[time].isin(post_periods)][outcome].mean()

        control_pre = df_control[df_control[time].isin(pre_periods)][outcome].mean()
        control_post = df_control[df_control[time].isin(post_periods)][outcome].mean()

        if np.isnan(treated_pre) or np.isnan(treated_post):
            return None
        if np.isnan(control_pre) or np.isnan(control_post):
            return None

        estimate = (treated_post - treated_pre) - (control_post - control_pre)

        # Calculate weight
        n_k = n_treated / n_total

        # Variance of treatment within the comparison window
        total_periods_in_window = len(pre_periods) + len(post_periods)
        D_k = len(post_periods) / total_periods_in_window if total_periods_in_window > 0 else 0

        # Weight proportional to group sizes and treatment variance
        # Scale by the fraction of total time this comparison covers
        time_share = total_periods_in_window / len(time_periods)
        weight = n_k * (1 - n_k) * D_k * (1 - D_k) * time_share

        return Comparison2x2(
            treated_group=treated_group,
            control_group=control_group,
            comparison_type=comparison_type,
            estimate=estimate,
            weight=weight,
            n_treated=n_treated,
            n_control=n_control,
            time_window=time_window,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {}

    def set_params(self, **params) -> "BaconDecomposition":
        """Set estimator parameters (sklearn-compatible)."""
        return self

    def summary(self) -> str:
        """Get summary of decomposition results."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())


def bacon_decompose(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: str,
) -> BaconDecompositionResults:
    """
    Convenience function for Goodman-Bacon decomposition.

    Decomposes a TWFE estimate into weighted 2x2 DiD comparisons,
    showing which comparisons drive the estimate and whether
    problematic "forbidden comparisons" are involved.

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

    Returns
    -------
    BaconDecompositionResults
        Object containing decomposition results with:
        - twfe_estimate: The overall TWFE coefficient
        - comparisons: List of all 2x2 comparisons with estimates and weights
        - Weight totals by comparison type
        - Methods for visualization and export

    Examples
    --------
    >>> from diff_diff import bacon_decompose
    >>>
    >>> results = bacon_decompose(
    ...     data=panel_df,
    ...     outcome='earnings',
    ...     unit='state',
    ...     time='year',
    ...     first_treat='treatment_year'
    ... )
    >>>
    >>> # View summary
    >>> results.print_summary()
    >>>
    >>> # Check weight on forbidden comparisons
    >>> print(f"Forbidden weight: {results.total_weight_later_vs_earlier:.1%}")
    >>>
    >>> # Convert to DataFrame for analysis
    >>> df = results.to_dataframe()

    See Also
    --------
    BaconDecomposition : Class-based interface with more options
    plot_bacon : Visualize the decomposition
    CallawaySantAnna : Robust estimator that avoids forbidden comparisons
    """
    decomp = BaconDecomposition()
    return decomp.fit(data, outcome, unit, time, first_treat)
