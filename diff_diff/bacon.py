"""
Goodman-Bacon Decomposition for Two-Way Fixed Effects.

Implements the decomposition from Goodman-Bacon (2021) that shows how
TWFE estimates with staggered treatment timing can be written as a
weighted average of all possible 2x2 DiD comparisons.

Reference:
    Goodman-Bacon, A. (2021). Difference-in-differences with variation
    in treatment timing. Journal of Econometrics, 225(2), 254-277.
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _format_survey_block
from diff_diff.utils import pre_demean_norms, snap_absorbed_regressors
from diff_diff.utils import within_transform as _within_transform_util

if TYPE_CHECKING:
    from diff_diff.survey import SurveyDesign


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
        For ``comparison_type="treated_vs_never"``, this is the literal
        string ``"never_treated"``, which refers to the **post-remap U
        bucket** (the paper's ``U`` per Goodman-Bacon 2021 footnote 11).
        On inputs with no remapped always-treated units this is exactly
        the true never-treated set; with remapping it is the broader U
        bucket. Check ``BaconDecompositionResults.n_never_treated`` and
        ``n_always_treated_remapped`` for the precise composition.
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
    time_window : Tuple[float, float]
        The (start, end) time period for this comparison.
    """

    treated_group: Any
    control_group: Any
    comparison_type: str
    estimate: float
    weight: float
    n_treated: int
    n_control: int
    time_window: Tuple[float, float]

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
    n_always_treated_remapped : int
        Number of units whose ``first_treat`` was at or before the first
        observable period (``first_treat <= min(time)``, excluding the
        never-treated sentinels ``0`` and ``np.inf``) and which were
        automatically remapped to the ``U`` (untreated) bucket per
        Goodman-Bacon (2021) footnote 11. Detection uses ordered-time
        logic so negative or zero-crossing period labels work correctly.
        Zero on inputs where the user only used the ``first_treat ∈ {0,
        np.inf}`` sentinels. The user's original ``first_treat`` column
        is preserved unchanged on the input ``data`` frame; remapping
        happens in an internal column.
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
    # Count of units auto-remapped from 0 < first_treat <= min(time) into the
    # U bucket per Goodman-Bacon (2021) footnote 11. Always 0 on legacy inputs.
    n_always_treated_remapped: int = 0
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)

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
        ]
        if self.n_always_treated_remapped > 0:
            lines.append(
                f"{'Always-treated remapped to U:':<35} " f"{self.n_always_treated_remapped:>10}"
            )
        lines.append("")

        # Add survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 85))

        lines.extend(
            [
                "-" * 85,
                "TWFE Decomposition".center(85),
                "-" * 85,
                "",
                f"{'TWFE Estimate:':<35} {self.twfe_estimate:>12.4f}",
                f"{'Weighted Sum of 2x2 Estimates:':<35} {self._weighted_sum():>12.4f}",
                f"{'Decomposition Error:':<35} {self.decomposition_error:>12.6f}",
                "",
            ]
        )

        # Weight breakdown by comparison type
        lines.extend(
            [
                "-" * 85,
                "Weight Breakdown by Comparison Type".center(85),
                "-" * 85,
                f"{'Comparison Type':<30} {'Weight':>12} {'Avg Effect':>12} {'Contribution':>12}",
                "-" * 85,
            ]
        )

        # Treated vs Never-treated
        if self.total_weight_treated_vs_never > 0:
            contrib = self.total_weight_treated_vs_never * (self.weighted_avg_treated_vs_never or 0)
            lines.append(
                f"{'Treated vs Never-treated':<30} "
                f"{self.total_weight_treated_vs_never:>12.4f} "
                f"{self.weighted_avg_treated_vs_never or 0:>12.4f} "
                f"{contrib:>12.4f}"
            )

        # Earlier vs Later
        if self.total_weight_earlier_vs_later > 0:
            contrib = self.total_weight_earlier_vs_later * (self.weighted_avg_earlier_vs_later or 0)
            lines.append(
                f"{'Earlier vs Later treated':<30} "
                f"{self.total_weight_earlier_vs_later:>12.4f} "
                f"{self.weighted_avg_earlier_vs_later or 0:>12.4f} "
                f"{contrib:>12.4f}"
            )

        # Later vs Earlier (forbidden)
        if self.total_weight_later_vs_earlier > 0:
            contrib = self.total_weight_later_vs_earlier * (self.weighted_avg_later_vs_earlier or 0)
            lines.append(
                f"{'Later vs Earlier (forbidden)':<30} "
                f"{self.total_weight_later_vs_earlier:>12.4f} "
                f"{self.weighted_avg_later_vs_earlier or 0:>12.4f} "
                f"{contrib:>12.4f}"
            )

        lines.extend(
            [
                "-" * 85,
                f"{'Total':<30} {self._total_weight():>12.4f} "
                f"{'':>12} {self._weighted_sum():>12.4f}",
                "-" * 85,
                "",
            ]
        )

        # Warning about forbidden comparisons
        if self.total_weight_later_vs_earlier > 0.01:
            pct = self.total_weight_later_vs_earlier * 100
            lines.extend(
                [
                    "WARNING: {:.1f}% of weight is on 'forbidden' comparisons where".format(pct),
                    "already-treated units serve as controls. This can bias TWFE",
                    "when treatment effects are heterogeneous over time.",
                    "",
                    "Consider using Callaway-Sant'Anna or other robust estimators.",
                    "",
                ]
            )

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
            rows.append(
                {
                    "treated_group": c.treated_group,
                    "control_group": c.control_group,
                    "comparison_type": c.comparison_type,
                    "estimate": c.estimate,
                    "weight": c.weight,
                    "n_treated": c.n_treated,
                    "n_control": c.n_control,
                    "time_start": c.time_window[0],
                    "time_end": c.time_window[1],
                }
            )
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
    weights : str, default="exact"
        Weight calculation method:

        - "exact" (default): Variance-based weights from Goodman-Bacon (2021)
          Theorem 1, Eqs. 7-9 and 10e-g. Produces the paper-faithful
          decomposition where the weighted sum matches the TWFE estimate
          to machine precision. Use for publication-quality work and the
          standard methodology contract.
        - "approximate": Fast simplified formula using group shares and
          treatment variance, with post-hoc sum-to-1 normalization. Opt
          in for speed-sensitive diagnostic loops where the relative weight
          structure is sufficient. Approximate-mode results may differ
          numerically from R ``bacondecomp::bacon()``.

    Attributes
    ----------
    weights : str
        The weight calculation method.
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

    def __init__(self, weights: str = "exact"):
        """
        Initialize BaconDecomposition.

        Parameters
        ----------
        weights : str, default="exact"
            Weight calculation method:

            - "exact" (default): Variance-based weights from Goodman-Bacon
              (2021) Theorem 1 (paper-faithful Eqs. 7-9 and 10e-g).
            - "approximate": Fast simplified formula. Opt in for speed.
        """
        if weights not in ("approximate", "exact"):
            raise ValueError(f"weights must be 'approximate' or 'exact', got '{weights}'")
        self.weights = weights
        self.results_: Optional[BaconDecompositionResults] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        survey_design=None,
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
            Name of column indicating when unit was first treated. The
            values ``0`` and ``np.inf`` are **reserved as never-treated
            sentinels** (not configurable today); a real treatment cohort
            with ``first_treat == 0`` would be folded into ``U`` and
            should instead be re-labeled to a non-sentinel value before
            fitting. Units whose ``first_treat`` is at or before the
            first observable period (``first_treat <= min(time)``,
            excluding the never-treated sentinels ``0`` and ``np.inf``)
            are automatically remapped to the ``U`` (untreated) bucket
            per Goodman-Bacon (2021) footnote 11, with a
            ``UserWarning``. **Library boundary extension:** the paper
            uses the strict inequality ``t_i < 1`` (units treated
            *before* the first observable period); the library uses the
            **inclusive** ``first_treat <= min(time)`` rule, additionally
            folding units treated *at* the first observable period
            (``first_treat == min(time)``) into ``U`` because such units
            have no untreated cell in-panel. See REGISTRY's
            ``**Deviation (first-period boundary extension on
            always-treated remap)**`` block for the full contract. Detection uses ordered-time logic on the
            **time axis** so panels whose ``time`` column contains
            negative or zero-crossing labels (e.g. event-time
            ``time ∈ [-2,..,3]``) are handled correctly; the ``0``
            sentinel restriction applies only to ``first_treat``, not to
            ``time``. The user's original ``first_treat`` column on
            ``data`` is preserved unchanged; remapping happens in an
            internal column. The count of remapped units is exposed on
            the result as
            ``BaconDecompositionResults.n_always_treated_remapped``.
        survey_design : SurveyDesign, optional
            Survey design specification for weighted estimation.
            When provided, all means and group shares use survey weights.
            The decomposition remains diagnostic (no survey vcov needed).

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

        # Resolve survey design if provided
        from diff_diff.survey import _resolve_survey_for_fit

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )
        # Reject replicate-weight designs — Bacon decomposition is a
        # diagnostic that does not compute replicate-based variance
        if resolved_survey is not None and resolved_survey.uses_replicate_variance:
            raise NotImplementedError(
                "BaconDecomposition does not support replicate-weight survey "
                "designs. Use a TSL-based survey design (strata/psu/fpc)."
            )

        # Validate within-unit constancy for exact survey weights only.
        # The exact-weight path collapses to per-unit weights via groupby().first(),
        # which requires constant survey columns within units. The approximate path
        # uses observation-level weighted means and does not need this constraint.
        if resolved_survey is not None and self.weights == "exact":
            from diff_diff.survey import _validate_unit_constant_survey

            _validate_unit_constant_survey(data, unit, survey_design)

        # Create working copy
        df = data.copy()

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Preserve the user-provided column name so we can count TRUE
        # never-treated units below (post-remap, `first_treat` will be
        # rebound to the internal column which folds remapped always-treated
        # into the same `0` sentinel bucket as never-treated).
        user_first_treat_col = first_treat

        # Always-treated remap (Goodman-Bacon 2021, footnote 11):
        # The paper convention puts units treated before the first observable
        # period into the U bucket alongside never-treated units. The library's
        # prior sentinel-only convention (`first_treat ∈ {0, np.inf}`) is
        # narrower than the paper's U.
        #
        # Detection uses ORDERED-TIME logic on the `time` axis
        # (`first_treat <= min(time)`), NOT positive-sign restriction, so
        # panels whose `time` column has negative or zero-crossing labels
        # (e.g. event-time `time ∈ [-2,..,3]`) are handled correctly.
        # Sentinel rows (`first_treat ∈ {0, np.inf}`) are excluded from
        # the remap so the never-treated contract is preserved. NOTE: the
        # `0` sentinel restriction applies to `first_treat` only, not to
        # `time`; a real treatment cohort with `first_treat == 0` is not
        # supported today and would be folded into `U` (re-label such
        # cohorts to a non-sentinel value before fitting).
        # Remapping writes to an internal column; the user's `first_treat`
        # column is preserved unchanged (df = data.copy() above).
        df["__bacon_first_treat_internal__"] = df[first_treat]
        min_period = df[time].min()
        is_U_sentinel = (df["__bacon_first_treat_internal__"] == 0) | (
            df["__bacon_first_treat_internal__"] == np.inf
        )
        always_treated_mask = (~is_U_sentinel) & (
            df["__bacon_first_treat_internal__"] <= min_period
        )
        n_always_treated_remapped = int(df.loc[always_treated_mask, unit].nunique())
        if n_always_treated_remapped > 0:
            warnings.warn(
                f"Detected {n_always_treated_remapped} always-treated units "
                f"(first_treat <= {min_period}, excluding sentinel values "
                f"0 and np.inf). Remapping to U bucket per Goodman-Bacon "
                f"(2021) footnote 11. The original first_treat column is "
                f"preserved; remapping happens in an internal column. To "
                f"silence this warning, recode the affected rows' "
                f"first_treat values to 0 or np.inf in your input data "
                f"before fitting.",
                UserWarning,
                stacklevel=2,
            )
            df.loc[always_treated_mask, "__bacon_first_treat_internal__"] = 0
        # Rebind the local first_treat name to the internal column so all
        # downstream df[first_treat] reads (and helper-function passthroughs)
        # see the remapped data without per-site rewrites.
        first_treat = "__bacon_first_treat_internal__"

        # Check for balanced panel
        periods_per_unit = df.groupby(unit)[time].count()
        if periods_per_unit.nunique() > 1:
            warnings.warn(
                "Unbalanced panel detected. Bacon decomposition assumes "
                "balanced panels. Results may be inaccurate.",
                UserWarning,
                stacklevel=2,
            )

        # Get unique time periods and timing groups
        time_periods = sorted(df[time].unique())

        # Identify never-treated and timing groups
        # Never-treated: first_treat = 0 or np.inf (library sentinels).
        # Timing groups: every other value in the (post-remap) internal
        # column. Do NOT restrict to positive values — negative-coded
        # event-time cohorts (e.g. first_treat=-1 on a panel with
        # min(time)=-2) are valid timing groups.
        never_treated_mask = (df[first_treat] == 0) | (df[first_treat] == np.inf)
        timing_groups = sorted([g for g in df[first_treat].unique() if g != 0 and g != np.inf])

        # `n_never_treated` reports TRUE never-treated units, computed from
        # the original user-provided column BEFORE the remap. Remapped
        # always-treated units are reported separately via
        # `n_always_treated_remapped` so the two counts do not double-count.
        unit_info_user = df.groupby(unit).agg({user_first_treat_col: "first"}).reset_index()
        n_never_treated = int(
            (
                (unit_info_user[user_first_treat_col] == 0)
                | (unit_info_user[user_first_treat_col] == np.inf)
            ).sum()
        )

        # `n_units_in_U_bucket` is the POST-remap count used to decide
        # whether `treated_vs_never` (β̂_{kU}^{2x2}) comparisons should
        # be generated. It includes BOTH true never-treated AND remapped
        # always-treated units, since the paper convention puts them in
        # the same `U` bucket (Goodman-Bacon 2021 footnote 11). Without
        # this distinction from `n_never_treated`, panels whose U is
        # composed entirely of remapped always-treated units would
        # silently drop all β̂_{kU}^{2x2} terms and break the Theorem 1
        # identity at the loop gate.
        unit_info_internal = df.groupby(unit).agg({first_treat: "first"}).reset_index()
        n_units_in_U_bucket = int(
            (
                (unit_info_internal[first_treat] == 0) | (unit_info_internal[first_treat] == np.inf)
            ).sum()
        )

        # Create treatment indicator (D_it = 1 if treated at time t)
        # Use unique internal name to avoid conflicts with user data
        _TREAT_COL = "__bacon_treated_internal__"
        df[_TREAT_COL] = (~never_treated_mask) & (df[time] >= df[first_treat])

        # First, compute TWFE estimate for reference
        twfe_estimate = self._compute_twfe(
            df, outcome, unit, time, _TREAT_COL, weights=survey_weights
        )

        # Perform decomposition
        comparisons = []

        # 1. Treated vs Never-treated comparisons.
        # Gate on the POST-remap U bucket count so panels whose U is
        # composed entirely of remapped always-treated units still emit
        # β̂_{kU}^{2x2} terms (paper Goodman-Bacon 2021 footnote 11).
        if n_units_in_U_bucket > 0:
            for g in timing_groups:
                comp = self._compute_treated_vs_never(
                    df,
                    outcome,
                    unit,
                    time,
                    first_treat,
                    g,
                    time_periods,
                    weights=survey_weights,
                )
                if comp is not None:
                    comparisons.append(comp)

        # 2. Timing group comparisons (earlier vs later and later vs earlier)
        for i, g_early in enumerate(timing_groups):
            for g_late in timing_groups[i + 1 :]:
                # Earlier vs Later: g_early treated, g_late as control
                comp_early = self._compute_timing_comparison(
                    df,
                    outcome,
                    unit,
                    time,
                    first_treat,
                    g_early,
                    g_late,
                    time_periods,
                    "earlier_vs_later",
                    weights=survey_weights,
                )
                if comp_early is not None:
                    comparisons.append(comp_early)

                # Later vs Earlier: g_late treated, g_early as control (forbidden)
                comp_late = self._compute_timing_comparison(
                    df,
                    outcome,
                    unit,
                    time,
                    first_treat,
                    g_late,
                    g_early,
                    time_periods,
                    "later_vs_earlier",
                    weights=survey_weights,
                )
                if comp_late is not None:
                    comparisons.append(comp_late)

        # Recompute exact weights if requested
        if self.weights == "exact":
            self._recompute_exact_weights(
                comparisons,
                df,
                outcome,
                unit,
                time,
                first_treat,
                time_periods,
                weights=survey_weights,
            )

        if not comparisons:
            raise ValueError(
                "No valid 2x2 comparisons remain after filtering. "
                "All cells have zero effective weight or insufficient data. "
                "Check subpopulation/domain definition."
            )

        # Normalize weights to sum to 1
        total_weight = sum(c.weight for c in comparisons)
        if total_weight > 0:
            for c in comparisons:
                c.weight = c.weight / total_weight

        # Calculate weight totals and weighted averages by type
        weight_by_type = {"treated_vs_never": 0.0, "earlier_vs_later": 0.0, "later_vs_earlier": 0.0}
        weighted_sum_by_type = {
            "treated_vs_never": 0.0,
            "earlier_vs_later": 0.0,
            "later_vs_earlier": 0.0,
        }

        for c in comparisons:
            weight_by_type[c.comparison_type] += c.weight
            weighted_sum_by_type[c.comparison_type] += c.weight * c.estimate

        # Calculate weighted averages
        avg_by_type = {}
        for ctype in weight_by_type:
            if weight_by_type[ctype] > 0:
                avg_by_type[ctype] = weighted_sum_by_type[ctype] / weight_by_type[ctype]
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
            n_always_treated_remapped=n_always_treated_remapped,
            survey_metadata=survey_metadata,
        )

        self.is_fitted_ = True
        return self.results_

    def _compute_twfe(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treat_col: str = "__bacon_treated_internal__",
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute TWFE estimate using within-transformation."""
        # Apply two-way within transformation (weighted if survey weights provided)
        _pre_norms = pre_demean_norms(df, [treat_col], weights=weights)
        df_dm = _within_transform_util(
            df,
            [outcome, treat_col],
            unit,
            time,
            suffix="_within",
            weights=weights,
        )
        # Snap an FE-spanned treatment to exact zero: the d_var == 0 guard
        # below then returns its deterministic 0.0 (with the cause warning)
        # instead of an arbitrary junk/junk division.
        snap_absorbed_regressors(
            df_dm,
            [treat_col],
            _pre_norms,
            absorbed_desc=f"unit '{unit}' and time '{time}' fixed effects",
            group_vars=[unit, time],
            suffix="_within",
            display_names={treat_col: "treatment"},
            weights=weights,
        )

        # Extract within-transformed values
        y_within = df_dm[f"{outcome}_within"].values
        d_within = df_dm[f"{treat_col}_within"].values

        # OLS on demeaned data: beta = sum(w * d * y) / sum(w * d^2)
        w = weights if weights is not None else np.ones(len(y_within))
        d_var = np.sum(w * d_within**2)
        if d_var > 0:
            beta = np.sum(w * d_within * y_within) / d_var
        else:
            beta = 0.0

        return beta

    def _recompute_exact_weights(
        self,
        comparisons: List[Comparison2x2],
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        time_periods: List[Any],
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Recompute weights using the exact Theorem 1 formula from
        Goodman-Bacon (2021).

        Implements Eqs. 7-9 (subsample FE-adjusted treatment-dummy
        variances) and Eqs. 10e-g (decomposition weights). Only the
        NUMERATORS of Eqs. 10e-g are written here; the post-hoc
        sum-to-1 normalization in ``fit()`` handles the ``V̂^D``
        denominator, which is mathematically equivalent per Theorem 1's
        identity ``V̂^D = Σ numerators``.

        Notation (per paper §2, pp. 256-258):
            n_k        sample share of timing group k (fraction of units)
            n_kU       relative size of group k in pair (k, U): n_k/(n_k+n_U)
            D̄_k        share of periods group k spends treated: (T-k+1)/T

        Equations:
            V̂_{kU}^D     = n_{kU}(1-n_{kU}) · D̄_k(1-D̄_k)                 (Eq. 7)
            V̂_{kℓ}^{D,k} = n_{kℓ}(1-n_{kℓ}) · (D̄_k-D̄_ℓ)/(1-D̄_ℓ)
                          · (1-D̄_k)/(1-D̄_ℓ)                              (Eq. 8)
            V̂_{kℓ}^{D,ℓ} = n_{kℓ}(1-n_{kℓ}) · D̄_ℓ/D̄_k
                          · (D̄_k-D̄_ℓ)/D̄_k                                (Eq. 9)

            s_{kU}    ∝ (n_k+n_U)^2 · V̂_{kU}^D                            (Eq. 10e)
            s_{kℓ}^k  ∝ ((n_k+n_ℓ)(1-D̄_ℓ))^2 · V̂_{kℓ}^{D,k}                (Eq. 10f)
            s_{kℓ}^ℓ  ∝ ((n_k+n_ℓ)·D̄_k)^2 · V̂_{kℓ}^{D,ℓ}                   (Eq. 10g)

        When survey weights are provided, sample shares use weighted unit
        counts (constant-within-unit weights are required and enforced
        upstream by ``_validate_unit_constant_survey``). The ``D̄_k`` term
        is panel-share-based and unaffected by sampling weights.

        Modifies ``comparisons[i].weight`` in place. The caller then
        normalizes to sum to 1.
        """
        # Panel length T (Eq. 7-9 use share-of-periods D̄_k = (T-k+1)/T)
        T = len(time_periods)
        if T <= 0:
            for comp in comparisons:
                comp.weight = 0.0
            return

        # Per-unit first observation (treatment timing is unit-invariant)
        df_copy = df.copy()
        df_copy["_sw"] = weights if weights is not None else np.ones(len(df))
        unit_first = df_copy.groupby(unit).agg({first_treat: "first", "_sw": "first"})
        ft_per_unit = unit_first[first_treat]
        sw_per_unit = unit_first["_sw"]

        # Total weighted unit mass (denominator for n_k, n_U)
        if weights is None:
            unit_mass_total = float(len(ft_per_unit))

            def _mass(mask: pd.Series) -> float:
                return float(int(mask.sum()))

        else:
            unit_mass_total = float(sw_per_unit.sum())

            def _mass(mask: pd.Series) -> float:
                return float(sw_per_unit[mask].sum())

        if unit_mass_total <= 0:
            for comp in comparisons:
                comp.weight = 0.0
            return

        # U bucket sample share (never-treated + always-treated post-remap;
        # remap to 0/inf already happened in fit() via __bacon_first_treat_internal__).
        is_U = (ft_per_unit == 0) | (ft_per_unit == np.inf)
        n_U = _mass(is_U) / unit_mass_total

        # Per-cohort sample shares n_g and panel-share D̄_g.
        # Timing groups: exclude only the U sentinels (0, np.inf). Negative
        # event-time cohorts (e.g. first_treat=-1 on a panel with min(time)=-2)
        # are valid timing groups.
        timing_groups = sorted(g for g in ft_per_unit.unique() if g != 0 and g != np.inf)
        n_g: Dict[Any, float] = {
            g: _mass(ft_per_unit == g) / unit_mass_total for g in timing_groups
        }
        # D̄_g = share of periods group g spends treated.
        # For absorbing treatment with first-treatment time g over panel
        # periods [1, T], the treated periods are [g, T], i.e. T - g + 1
        # periods. We use the actual period values from time_periods to
        # handle panels indexed from 0 / arbitrary start.
        t_arr = np.asarray(sorted(time_periods))
        D_bar: Dict[Any, float] = {g: float(np.sum(t_arr >= g)) / T for g in timing_groups}

        for comp in comparisons:
            if comp.comparison_type == "treated_vs_never":
                k = comp.treated_group
                n_k = n_g.get(k, 0.0)
                if n_k <= 0 or n_U <= 0:
                    comp.weight = 0.0
                    continue
                n_kU = n_k / (n_k + n_U)
                D_k = D_bar.get(k, 0.0)
                # Eq. 7
                V_kU = n_kU * (1.0 - n_kU) * D_k * (1.0 - D_k)
                # Eq. 10e numerator
                comp.weight = (n_k + n_U) ** 2 * V_kU
            elif comp.comparison_type == "earlier_vs_later":
                # k = early (treated in 2x2), ℓ = late (control during MID)
                k = comp.treated_group
                ell = comp.control_group
                n_k = n_g.get(k, 0.0)
                n_ell = n_g.get(ell, 0.0)
                if n_k <= 0 or n_ell <= 0:
                    comp.weight = 0.0
                    continue
                D_k = D_bar.get(k, 0.0)
                D_ell = D_bar.get(ell, 0.0)
                # Eq. 8 requires D̄_ℓ < 1 (denominators 1 - D̄_ℓ)
                if D_ell >= 1.0:
                    comp.weight = 0.0
                    continue
                n_kl = n_k / (n_k + n_ell)
                # Eq. 8
                V_kl_k = (
                    n_kl
                    * (1.0 - n_kl)
                    * (D_k - D_ell)
                    / (1.0 - D_ell)
                    * (1.0 - D_k)
                    / (1.0 - D_ell)
                )
                # Eq. 10f numerator
                comp.weight = ((n_k + n_ell) * (1.0 - D_ell)) ** 2 * V_kl_k
            else:  # later_vs_earlier
                # ℓ = late (treated in 2x2), k = early (already-treated control)
                ell = comp.treated_group
                k = comp.control_group
                n_k = n_g.get(k, 0.0)
                n_ell = n_g.get(ell, 0.0)
                if n_k <= 0 or n_ell <= 0:
                    comp.weight = 0.0
                    continue
                D_k = D_bar.get(k, 0.0)
                D_ell = D_bar.get(ell, 0.0)
                # Eq. 9 requires D̄_k > 0 (denominators D̄_k)
                if D_k <= 0.0:
                    comp.weight = 0.0
                    continue
                n_kl = n_k / (n_k + n_ell)
                # Eq. 9
                V_kl_l = n_kl * (1.0 - n_kl) * D_ell / D_k * (D_k - D_ell) / D_k
                # Eq. 10g numerator
                comp.weight = ((n_k + n_ell) * D_k) ** 2 * V_kl_l

    def _compute_treated_vs_never(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        treated_group: Any,
        time_periods: List[Any],
        weights: Optional[np.ndarray] = None,
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

        # Compute 2x2 DiD estimate using weighted means if survey weights provided
        w = weights if weights is not None else np.ones(len(df))
        y = df[outcome].values

        treated_pre_mask = treated_mask & df[time].isin(pre_periods)
        treated_post_mask = treated_mask & df[time].isin(post_periods)
        never_pre_mask = never_mask & df[time].isin(pre_periods)
        never_post_mask = never_mask & df[time].isin(post_periods)

        # Guard against empty cells (unbalanced/filtered panels)
        # Also check positive weight mass for survey/subpopulation designs
        if not (
            np.any(treated_pre_mask)
            and np.any(treated_post_mask)
            and np.any(never_pre_mask)
            and np.any(never_post_mask)
        ):
            return None
        if (
            np.sum(w[treated_pre_mask]) <= 0
            or np.sum(w[treated_post_mask]) <= 0
            or np.sum(w[never_pre_mask]) <= 0
            or np.sum(w[never_post_mask]) <= 0
        ):
            return None

        treated_pre = np.average(y[treated_pre_mask], weights=w[treated_pre_mask])
        treated_post = np.average(y[treated_post_mask], weights=w[treated_post_mask])
        never_pre = np.average(y[never_pre_mask], weights=w[never_pre_mask])
        never_post = np.average(y[never_post_mask], weights=w[never_post_mask])

        estimate = (treated_post - treated_pre) - (never_post - never_pre)

        # Calculate weight components using weighted group shares
        n_treated = df_treated[unit].nunique()
        n_never = df_never[unit].nunique()

        w_treated_sum = np.sum(w[treated_mask])
        w_never_sum = np.sum(w[never_mask])
        w_total = w_treated_sum + w_never_sum

        # Weighted group share
        n_k = w_treated_sum / w_total if w_total > 0 else 0.0

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
        weights: Optional[np.ndarray] = None,
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

        # Compute 2x2 DiD estimate using weighted means if survey weights provided
        w = weights if weights is not None else np.ones(len(df))
        y = df[outcome].values

        treated_pre_mask = treated_mask & df[time].isin(pre_periods)
        treated_post_mask = treated_mask & df[time].isin(post_periods)
        control_pre_mask = control_mask & df[time].isin(pre_periods)
        control_post_mask = control_mask & df[time].isin(post_periods)

        # Skip if any cell is empty or has zero effective weight
        if (
            treated_pre_mask.sum() == 0
            or treated_post_mask.sum() == 0
            or control_pre_mask.sum() == 0
            or control_post_mask.sum() == 0
        ):
            return None
        if (
            np.sum(w[treated_pre_mask]) <= 0
            or np.sum(w[treated_post_mask]) <= 0
            or np.sum(w[control_pre_mask]) <= 0
            or np.sum(w[control_post_mask]) <= 0
        ):
            return None

        treated_pre = np.average(y[treated_pre_mask], weights=w[treated_pre_mask])
        treated_post = np.average(y[treated_post_mask], weights=w[treated_post_mask])
        control_pre = np.average(y[control_pre_mask], weights=w[control_pre_mask])
        control_post = np.average(y[control_post_mask], weights=w[control_post_mask])

        if np.isnan(treated_pre) or np.isnan(treated_post):
            return None
        if np.isnan(control_pre) or np.isnan(control_post):
            return None

        estimate = (treated_post - treated_pre) - (control_post - control_pre)

        # Calculate weight using weighted group shares
        w_treated_sum = np.sum(w[treated_mask])
        w_control_sum = np.sum(w[control_mask])
        w_total = w_treated_sum + w_control_sum
        n_k = w_treated_sum / w_total if w_total > 0 else 0.0

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
        return {"weights": self.weights}

    def set_params(self, **params) -> "BaconDecomposition":
        """Set estimator parameters (sklearn-compatible)."""
        if "weights" in params:
            if params["weights"] not in ("approximate", "exact"):
                raise ValueError(
                    f"weights must be 'approximate' or 'exact', " f"got '{params['weights']}'"
                )
            self.weights = params["weights"]
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
    weights: str = "exact",
    survey_design: Optional["SurveyDesign"] = None,
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
        Name of column indicating when unit was first treated. The
        values ``0`` and ``np.inf`` are **reserved as never-treated
        sentinels**; a real treatment cohort with ``first_treat == 0``
        would be folded into ``U`` and should be re-labeled to a
        non-sentinel value before fitting. Units whose ``first_treat``
        is at or before the first observable period
        (``first_treat <= min(time)``, excluding the sentinels) are
        automatically remapped to the ``U`` (untreated) bucket per
        Goodman-Bacon (2021) footnote 11, with a ``UserWarning``. See
        ``BaconDecomposition.fit()`` for the full contract and
        ``BaconDecompositionResults.n_always_treated_remapped`` for the
        count. The user's original ``first_treat`` column is preserved
        unchanged.
    weights : str, default="exact"
        Weight calculation method:

        - "exact" (default): Variance-based weights from Goodman-Bacon
          (2021) Theorem 1, Eqs. 7-9 and 10e-g. Paper-faithful.
        - "approximate": Fast simplified formula. Opt in for
          speed-sensitive diagnostic loops; numerical output may differ
          from R ``bacondecomp::bacon()``.
    survey_design : SurveyDesign, optional
        Survey design specification for weighted estimation. When provided,
        cell means, group shares, and within-transform use survey weights.
        The decomposition remains diagnostic (no survey vcov needed).

        **Default-flip caveat (PR-B, 2026-05-16):** the new
        ``weights="exact"`` default routes through
        ``_validate_unit_constant_survey``, which **rejects survey
        designs whose weights / strata / PSU / FPC columns vary within
        a unit across periods** (the exact path collapses to per-unit
        aggregation via ``groupby().first()``). Users whose survey
        design has time-varying within-unit columns must either (a)
        collapse the columns to be unit-constant or (b) pass explicit
        ``weights="approximate"`` to retain the legacy observation-level
        weighted-means path.

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
    >>> # Default: paper-faithful Goodman-Bacon (2021) Theorem 1 weights
    >>> # (weights="exact"); matches R bacondecomp::bacon() at atol=1e-6 on
    >>> # the aggregate (TWFE coefficient + weights-sum) across all panels,
    >>> # and on the per-component breakdown when there are no
    >>> # always-treated / first-period-treated cohorts (i.e. all
    >>> # non-sentinel first_treat values are strictly greater than
    >>> # min(time)). For panels with always-treated units, the
    >>> # per-component breakdown diverges by convention (Python remaps
    >>> # to U per paper footnote 11; R emits `Later vs Always Treated`);
    >>> # see REGISTRY note on R parity convention divergence. Validated
    >>> # via tests/test_methodology_bacon.py::TestBaconParityR.
    >>> results = bacon_decompose(
    ...     data=panel_df,
    ...     outcome='earnings',
    ...     unit='state',
    ...     time='year',
    ...     first_treat='treatment_year'
    ... )
    >>>
    >>> # Opt-in: simplified-variance fast path for diagnostic loops
    >>> # (numerical output may differ from R; sum-to-1 still holds).
    >>> results_approx = bacon_decompose(
    ...     data=panel_df,
    ...     outcome='earnings',
    ...     unit='state',
    ...     time='year',
    ...     first_treat='treatment_year',
    ...     weights='approximate'
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
    decomp = BaconDecomposition(weights=weights)
    return decomp.fit(data, outcome, unit, time, first_treat, survey_design=survey_design)
