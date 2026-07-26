"""
Honest DiD sensitivity analysis (Rambachan & Roth 2023).

Provides robust inference for difference-in-differences designs when
parallel trends may be violated. Instead of assuming parallel trends
holds exactly, this module allows for bounded violations and computes
partially identified treatment effect bounds.

References
----------
Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends.
The Review of Economic Studies, 90(5), 2555-2591.
https://doi.org/10.1093/restud/rdad018

See Also
--------
https://github.com/asheshrambachan/HonestDiD - R package implementation
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize

from diff_diff.results import (
    MultiPeriodDiDResults,
)
from diff_diff.results_base import Diagnostic
from diff_diff.utils import _get_critical_value

# =============================================================================
# Delta Restriction Classes
# =============================================================================


@dataclass
class DeltaSD:
    """
    Smoothness restriction on trend violations (Delta^{SD}).

    Restricts the second differences of the trend violations:

    .. math::

        |\\delta_{t+1} - 2\\delta_t + \\delta_{t-1}| \\le M

    When M=0, this enforces that violations follow a linear trend
    (linear extrapolation of pre-trends). Larger M allows more
    curvature in the violation path.

    Parameters
    ----------
    M : float
        Maximum allowed second difference. M=0 means linear trends only.

    Examples
    --------
    >>> delta = DeltaSD(M=0.5)
    >>> delta.M
    0.5
    """

    M: float = 0.0

    def __post_init__(self):
        if self.M < 0:
            raise ValueError(f"M must be non-negative, got M={self.M}")

    def __repr__(self) -> str:
        return f"DeltaSD(M={self.M})"


@dataclass
class DeltaRM:
    """
    Relative magnitudes restriction on trend violations (Delta^{RM}).

    Post-treatment consecutive first differences are bounded by Mbar
    times the maximum pre-treatment first difference:

    .. math::

        |\\delta_{t+1} - \\delta_t| \\le \\overline{M} \\cdot \\max_{s<0} |\\delta_{s+1} - \\delta_s|

    When Mbar=0, this enforces zero post-treatment first differences.
    Mbar=1 means post-period first differences can be as large as the
    worst observed pre-period first difference.

    Parameters
    ----------
    Mbar : float
        Scaling factor for maximum pre-period first difference.

    Examples
    --------
    >>> delta = DeltaRM(Mbar=1.0)
    >>> delta.Mbar
    1.0
    """

    Mbar: float = 1.0

    def __post_init__(self):
        if self.Mbar < 0:
            raise ValueError(f"Mbar must be non-negative, got Mbar={self.Mbar}")

    def __repr__(self) -> str:
        return f"DeltaRM(Mbar={self.Mbar})"


@dataclass
class DeltaSDRM:
    """
    Combined smoothness and relative magnitudes restriction.

    Imposes both:

    1. Smoothness: :math:`|\\delta_{t+1} - 2\\delta_t + \\delta_{t-1}| \\le M`
    2. Relative magnitudes: :math:`|\\delta_{t+1} - \\delta_t| \\le \\overline{M} \\cdot \\max_{s<0} |\\delta_{s+1} - \\delta_s|`

    This is more restrictive than either constraint alone.

    Parameters
    ----------
    M : float
        Maximum allowed second difference (smoothness).
    Mbar : float
        Scaling factor for maximum pre-period first difference (relative magnitudes).

    Examples
    --------
    >>> delta = DeltaSDRM(M=0.5, Mbar=1.0)
    """

    M: float = 0.0
    Mbar: float = 1.0

    def __post_init__(self):
        if self.M < 0:
            raise ValueError(f"M must be non-negative, got M={self.M}")
        if self.Mbar < 0:
            raise ValueError(f"Mbar must be non-negative, got Mbar={self.Mbar}")

    def __repr__(self) -> str:
        return f"DeltaSDRM(M={self.M}, Mbar={self.Mbar})"


DeltaType = Union[DeltaSD, DeltaRM, DeltaSDRM]


# =============================================================================
# Results Classes
# =============================================================================


@dataclass
class HonestDiDResults(Diagnostic):
    """
    Results from Honest DiD sensitivity analysis.

    Contains bounds on the treatment effect under the specified
    restrictions on violations of parallel trends.

    Attributes
    ----------
    lb : float
        Lower bound of identified set.
    ub : float
        Upper bound of identified set.
    ci_lb : float
        Lower bound of robust confidence interval.
    ci_ub : float
        Upper bound of robust confidence interval.
    M : float
        The restriction parameter value used.
    method : str
        The type of restriction ("smoothness", "relative_magnitude", or "combined").
    original_estimate : float
        The original point estimate (under parallel trends).
    original_se : float
        The original standard error.
    alpha : float
        Significance level for confidence interval.
    ci_method : str
        Method used for CI construction ("FLCI" or "C-LF").
    original_results : Any
        The original estimation results object.
    """

    lb: float
    ub: float
    ci_lb: float
    ci_ub: float
    M: float
    method: str
    original_estimate: float
    original_se: float
    alpha: float = 0.05
    ci_method: str = "FLCI"
    target_label: str = "Equal-weight avg over post horizons"
    pre_periods_used: Optional[List[Any]] = field(default=None, repr=False)
    post_periods_used: Optional[List[Any]] = field(default=None, repr=False)
    original_results: Optional[Any] = field(default=None, repr=False)
    # Event study bounds (optional)
    event_study_bounds: Optional[Dict[Any, Dict[str, float]]] = field(default=None, repr=False)
    # Survey design metadata (Phase 7d)
    survey_metadata: Optional[Any] = field(default=None, repr=False)
    df_survey: Optional[int] = field(default=None, repr=False)

    def _ci_is_finite(self) -> bool:
        """Check if CI endpoints are finite (not NaN/inf)."""
        return np.isfinite(self.ci_lb) and np.isfinite(self.ci_ub)

    def __repr__(self) -> str:
        if not self._ci_is_finite():
            return (
                f"HonestDiDResults(bounds=[{self.lb}, {self.ub}], "
                f"CI=[{self.ci_lb}, {self.ci_ub}] (undefined), "
                f"M={self.M})"
            )
        sig = "" if self.ci_lb <= 0 <= self.ci_ub else "*"
        return (
            f"HonestDiDResults(bounds=[{self.lb:.4f}, {self.ub:.4f}], "
            f"CI=[{self.ci_lb:.4f}, {self.ci_ub:.4f}]{sig}, "
            f"M={self.M})"
        )

    @property
    def is_significant(self) -> bool:
        """Check if CI excludes zero (effect is robust to violations).

        Returns False for undefined (NaN) CIs.
        """
        if not self._ci_is_finite():
            return False
        return not (self.ci_lb <= 0 <= self.ci_ub)

    @property
    def significance_stars(self) -> str:
        """
        Return significance indicator if robust CI excludes zero.

        Note: Unlike point estimation, partial identification does not yield
        a single p-value. This returns "*" if the robust CI excludes zero
        at the specified alpha level, indicating the effect is robust to
        the assumed violations of parallel trends.
        """
        return "*" if self.is_significant else ""

    @property
    def identified_set_width(self) -> float:
        """Width of the identified set."""
        return self.ub - self.lb

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.ci_ub - self.ci_lb

    def summary(self) -> str:
        """
        Generate formatted summary of sensitivity analysis results.

        Returns
        -------
        str
            Formatted summary.
        """
        conf_level = int((1 - self.alpha) * 100)

        method_names = {
            "smoothness": "Smoothness (Delta^SD)",
            "relative_magnitude": "Relative Magnitudes (Delta^RM)",
            "combined": "Combined (Delta^SDRM)",
        }
        method_display = method_names.get(self.method, self.method)

        lines = [
            "=" * 70,
            "Honest DiD Sensitivity Analysis Results".center(70),
            "(Rambachan & Roth 2023)".center(70),
            "=" * 70,
            "",
            f"{'Method:':<30} {method_display}",
            f"{'Target:':<30} {self.target_label}",
            f"{'Restriction parameter (M):':<30} {self.M:.4f}",
            f"{'CI method:':<30} {self.ci_method}",
            "",
            "-" * 70,
            "Original Estimate (under parallel trends)".center(70),
            "-" * 70,
            f"{'Point estimate:':<30} {self.original_estimate:.4f}",
            f"{'Standard error:':<30} {self.original_se:.4f}",
            "",
            "-" * 70,
            "Robust Results (allowing for violations)".center(70),
            "-" * 70,
            f"{'Identified set:':<30} [{self.lb:.4f}, {self.ub:.4f}]",
            f"{f'{conf_level}% Robust CI:':<30} [{self.ci_lb:.4f}, {self.ci_ub:.4f}]",
            "",
            f"{'Effect robust to violations:':<30} {'Yes' if self.is_significant else 'No'}",
            "",
        ]

        # Interpretation
        if self.pre_periods_used is not None:
            lines.append(f"{'Pre horizons used:':<30} {self.pre_periods_used}")
        if self.post_periods_used is not None:
            lines.append(f"{'Post horizons used:':<30} {self.post_periods_used}")
        if self.pre_periods_used is not None or self.post_periods_used is not None:
            lines.append("")

        lines.extend(
            [
                "-" * 70,
                "Interpretation".center(70),
                "-" * 70,
            ]
        )

        if self.method == "relative_magnitude":
            lines.append(
                f"Post-treatment first differences bounded at {self.M:.1f}x max pre-period first difference."
            )
        elif self.method == "smoothness":
            if self.M == 0:
                lines.append("Violations follow linear extrapolation of pre-trends.")
            else:
                lines.append(
                    f"Violation curvature (second diff) bounded by {self.M:.4f} per period."
                )
        else:
            lines.append(f"Combined smoothness (M={self.M:.2f}) and relative magnitude bounds.")

        if self.is_significant:
            if self.ci_lb > 0:
                lines.append(f"Effect remains POSITIVE even with violations up to M={self.M}.")
            else:
                lines.append(f"Effect remains NEGATIVE even with violations up to M={self.M}.")
        else:
            lines.append(f"Cannot rule out zero effect when allowing violations up to M={self.M}.")

        lines.extend(["", "=" * 70])

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "lb": self.lb,
            "ub": self.ub,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
            "M": self.M,
            "method": self.method,
            "target_label": self.target_label,
            "pre_periods_used": self.pre_periods_used,
            "post_periods_used": self.post_periods_used,
            "original_estimate": self.original_estimate,
            "original_se": self.original_se,
            "alpha": self.alpha,
            "ci_method": self.ci_method,
            "is_significant": self.is_significant,
            "identified_set_width": self.identified_set_width,
            "ci_width": self.ci_width,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class SensitivityResults(Diagnostic):
    """
    Results from sensitivity analysis over a grid of M values.

    Contains bounds and confidence intervals for each M value,
    plus the breakdown value.

    Attributes
    ----------
    M_values : np.ndarray
        Grid of M parameter values.
    bounds : List[Tuple[float, float]]
        List of (lb, ub) identified set bounds for each M.
    robust_cis : List[Tuple[float, float]]
        List of (ci_lb, ci_ub) robust CIs for each M.
    breakdown_M : float
        Smallest M where robust CI includes zero.
    method : str
        Type of restriction used.
    original_estimate : float
        Original point estimate.
    original_se : float
        Original standard error.
    alpha : float
        Significance level.
    """

    M_values: np.ndarray
    bounds: List[Tuple[float, float]]
    robust_cis: List[Tuple[float, float]]
    breakdown_M: Optional[float]
    method: str
    original_estimate: float
    original_se: float
    alpha: float = 0.05

    def __repr__(self) -> str:
        breakdown_str = f"{self.breakdown_M:.4f}" if self.breakdown_M else "None"
        return f"SensitivityResults(n_M={len(self.M_values)}, " f"breakdown_M={breakdown_str})"

    @property
    def has_breakdown(self) -> bool:
        """Check if there is a finite breakdown value."""
        return self.breakdown_M is not None

    def summary(self) -> str:
        """Generate formatted summary."""
        lines = [
            "=" * 70,
            "Honest DiD Sensitivity Analysis".center(70),
            "=" * 70,
            "",
            f"{'Method:':<30} {self.method}",
            f"{'Original estimate:':<30} {self.original_estimate:.4f}",
            f"{'Original SE:':<30} {self.original_se:.4f}",
            f"{'M values tested:':<30} {len(self.M_values)}",
            "",
        ]

        if self.breakdown_M is not None:
            lines.append(f"{'Breakdown value:':<30} {self.breakdown_M:.4f}")
            lines.append("")
            lines.append(f"Result is robust to violations up to M = {self.breakdown_M:.4f}")
        else:
            lines.append(f"{'Breakdown value:':<30} None (always significant)")

        lines.extend(
            [
                "",
                "-" * 70,
                f"{'M':<10} {'Lower Bound':>12} {'Upper Bound':>12} {'CI Lower':>12} {'CI Upper':>12}",
                "-" * 70,
            ]
        )

        for i, M in enumerate(self.M_values):
            lb, ub = self.bounds[i]
            ci_lb, ci_ub = self.robust_cis[i]
            lines.append(f"{M:<10.4f} {lb:>12.4f} {ub:>12.4f} {ci_lb:>12.4f} {ci_ub:>12.4f}")

        lines.extend(["", "=" * 70])

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per M value."""
        rows = []
        for i, M in enumerate(self.M_values):
            lb, ub = self.bounds[i]
            ci_lb, ci_ub = self.robust_cis[i]
            rows.append(
                {
                    "M": M,
                    "lb": lb,
                    "ub": ub,
                    "ci_lb": ci_lb,
                    "ci_ub": ci_ub,
                    "is_significant": (
                        np.isfinite(ci_lb) and np.isfinite(ci_ub) and not (ci_lb <= 0 <= ci_ub)
                    ),
                }
            )
        return pd.DataFrame(rows)

    def plot(
        self,
        ax=None,
        show_bounds: bool = True,
        show_ci: bool = True,
        breakdown_line: bool = True,
        **kwargs,
    ):
        """
        Plot sensitivity analysis results.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show_bounds : bool
            Whether to show identified set bounds.
        show_ci : bool
            Whether to show confidence intervals.
        breakdown_line : bool
            Whether to show vertical line at breakdown value.
        **kwargs
            Additional arguments passed to plotting functions.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        M = self.M_values
        bounds_arr = np.array(self.bounds)
        ci_arr = np.array(self.robust_cis)

        # Plot original estimate
        ax.axhline(
            y=self.original_estimate,
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Original estimate",
            alpha=0.7,
        )

        # Plot zero line
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        if show_bounds:
            ax.fill_between(
                M,
                bounds_arr[:, 0],
                bounds_arr[:, 1],
                alpha=0.3,
                color="blue",
                label="Identified set",
            )

        if show_ci:
            ax.plot(M, ci_arr[:, 0], "b-", linewidth=1.5, label="Robust CI")
            ax.plot(M, ci_arr[:, 1], "b-", linewidth=1.5)

        if breakdown_line and self.breakdown_M is not None:
            ax.axvline(
                x=self.breakdown_M,
                color="red",
                linestyle=":",
                linewidth=2,
                label=f"Breakdown (M={self.breakdown_M:.2f})",
            )

        ax.set_xlabel("M (restriction parameter)")
        ax.set_ylabel("Treatment Effect")
        ax.set_title("Sensitivity Analysis: Treatment Effect Bounds")
        ax.legend(loc="best")

        return ax


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_event_study_params(
    results: Union[MultiPeriodDiDResults, Any],
) -> Tuple[np.ndarray, np.ndarray, int, int, List[Any], List[Any], Optional[int]]:
    """
    Extract event study parameters from results objects.

    Parameters
    ----------
    results : MultiPeriodDiDResults, CallawaySantAnnaResults, or ChaisemartinDHaultfoeuilleResults
        Estimation results with event study structure.

    Returns
    -------
    beta_hat : np.ndarray
        Vector of event study coefficients (pre + post periods).
    sigma : np.ndarray
        Variance-covariance matrix of coefficients.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    pre_periods : list
        Pre-period identifiers.
    post_periods : list
        Post-period identifiers.
    df_survey : int or None
        Survey degrees of freedom for t-distribution inference.
    """
    if isinstance(results, MultiPeriodDiDResults):
        # Extract from MultiPeriodDiD
        pre_periods = results.pre_periods
        post_periods = results.post_periods

        # Filter periods with finite effects/SEs, maintaining pre-then-post order
        finite_periods = {
            p
            for p in results.period_effects.keys()
            if np.isfinite(results.period_effects[p].effect)
            and np.isfinite(results.period_effects[p].se)
        }

        pre_estimated = [p for p in pre_periods if p in finite_periods]
        post_estimated = [p for p in post_periods if p in finite_periods]
        all_estimated = pre_estimated + post_estimated

        if not all_estimated:
            raise ValueError(
                "No period effects with finite estimates found. " "Cannot compute HonestDiD bounds."
            )

        effects = [results.period_effects[p].effect for p in all_estimated]
        ses = [results.period_effects[p].se for p in all_estimated]

        beta_hat = np.array(effects)
        num_pre_periods = sum(1 for p in all_estimated if p in pre_periods)
        num_post_periods = sum(1 for p in all_estimated if p in post_periods)

        if num_pre_periods == 0:
            raise ValueError(
                "No pre-period effects with finite estimates found. "
                "HonestDiD requires at least one identified pre-period "
                "coefficient."
            )

        # Extract proper sub-VCV for interaction terms
        if (
            results.vcov is not None
            and hasattr(results, "interaction_indices")
            and results.interaction_indices is not None
        ):
            indices = [results.interaction_indices[p] for p in all_estimated]
            sigma = results.vcov[np.ix_(indices, indices)]
        else:
            # Fallback: diagonal from SEs
            sigma = np.diag(np.array(ses) ** 2)

        # Extract inference df. Prefer ``survey_metadata.df_survey`` (the
        # actual CS-internal df, which may have been tightened post-resolve
        # for replicate designs) over the dedicated ``df_inference`` field.
        # ``df_inference`` is the FALLBACK carrier for bare-``cluster=`` CS
        # fits where ``survey_metadata`` is intentionally None (preserving
        # the survey/non-survey contract for ``DiagnosticReport`` /
        # ``summary()``). Reading ``df_inference`` first would silently
        # overstate the denominator df on panel survey fits whose df was
        # tightened during aggregation. Replicate designs with undefined
        # df → sentinel 0.
        df_survey = None
        if hasattr(results, "survey_metadata") and results.survey_metadata is not None:
            sm = results.survey_metadata
            df_survey = getattr(sm, "df_survey", None)
            if df_survey is None and getattr(sm, "replicate_method", None) is not None:
                df_survey = 0  # undefined replicate df → NaN inference
        if df_survey is None:
            df_inference = getattr(results, "df_inference", None)
            if df_inference is not None:
                df_survey = int(df_inference)

        return (
            beta_hat,
            sigma,
            num_pre_periods,
            num_post_periods,
            pre_periods,
            post_periods,
            df_survey,
        )

    else:
        # Try CallawaySantAnnaResults
        try:
            from diff_diff.staggered import CallawaySantAnnaResults

            if isinstance(results, CallawaySantAnnaResults):
                if results.event_study_effects is None:
                    raise ValueError(
                        "CallawaySantAnnaResults must have event_study_effects for HonestDiD. "
                        "Fit with aggregate='event_study'. HonestDiD reads the fit-time "
                        "surface and does not yet accept the post-fit container returned "
                        "by results.aggregate('event_study')."
                    )

                # Warn if not using universal base period (R's HonestDiD requires it)
                if getattr(results, "base_period", "universal") != "universal":
                    import warnings

                    warnings.warn(
                        "HonestDiD sensitivity analysis on CallawaySantAnna results "
                        "requires base_period='universal' for valid interpretation. "
                        "With base_period='varying', pre-treatment coefficients use "
                        "consecutive comparisons (not a common reference period), "
                        "which changes the meaning of the parallel trends restriction. "
                        "Re-run with CallawaySantAnna(base_period='universal') for "
                        "methodologically valid HonestDiD bounds.",
                        UserWarning,
                        stacklevel=3,
                    )

                # Extract event study effects by relative time
                # Filter out normalization constraints (n_groups=0) and non-finite SEs
                event_effects = {
                    t: data
                    for t, data in results.event_study_effects.items()
                    if data.get("n_groups", 1) > 0 and np.isfinite(data.get("se", np.nan))
                }
                rel_times = sorted(event_effects.keys())

                # Infer the omitted reference period from the normalization
                # marker injected by _aggregate_event_study for universal base.
                # The reference has the exact signature: effect=0.0, se=NaN, n_groups=0.
                # Other empty bins may also have n_groups=0 but with NaN effect.
                ref_period = None
                for t, data in results.event_study_effects.items():
                    if (
                        data.get("n_groups", 1) == 0
                        and data.get("effect", None) == 0.0
                        and not np.isfinite(data.get("se", 0.0))
                    ):
                        ref_period = t
                        break

                if ref_period is not None:
                    # Universal base: split relative to the reference period
                    pre_times = [t for t in rel_times if t < ref_period]
                    post_times = [t for t in rel_times if t > ref_period]
                else:
                    # Varying base or no reference marker: split at t < 0 / t >= 0
                    pre_times = [t for t in rel_times if t < 0]
                    post_times = [t for t in rel_times if t >= 0]

                if len(pre_times) == 0:
                    raise ValueError(
                        "No pre-period effects with finite estimates found in "
                        "CallawaySantAnna event study. HonestDiD requires at "
                        "least one identified pre-period coefficient."
                    )

                effects = []
                ses = []
                for t in rel_times:
                    effects.append(event_effects[t]["effect"])
                    ses.append(event_effects[t]["se"])

                beta_hat = np.array(effects)

                # Use full event-study VCV if available (Phase 7d),
                # otherwise fall back to diagonal from SEs
                if hasattr(results, "event_study_vcov") and results.event_study_vcov is not None:
                    vcov = results.event_study_vcov
                    # VCV is indexed by the aggregated event times (stored in
                    # event_study_vcov_index), NOT by event_study_effects keys
                    # (which may include an injected reference period).
                    # Subset to match the surviving rel_times.
                    vcov_index = getattr(results, "event_study_vcov_index", None)
                    if vcov_index is not None and len(rel_times) < len(vcov_index):
                        idx = [vcov_index.index(t) for t in rel_times if t in vcov_index]
                        if len(idx) == len(rel_times):
                            sigma = vcov[np.ix_(idx, idx)]
                        else:
                            sigma = np.diag(np.array(ses) ** 2)
                    elif vcov.shape[0] == len(rel_times):
                        sigma = vcov
                    else:
                        sigma = np.diag(np.array(ses) ** 2)
                else:
                    # No full VCV available. Check if this is a bootstrap fit
                    # (VCV was cleared to prevent mixing analytical/bootstrap).
                    if (
                        hasattr(results, "bootstrap_results")
                        and results.bootstrap_results is not None
                    ):
                        import warnings

                        warnings.warn(
                            "HonestDiD on bootstrap-fitted CallawaySantAnna results "
                            "uses a diagonal covariance matrix (cross-event-time "
                            "covariance is not available from bootstrap). For full "
                            "covariance structure, use analytical SEs (n_bootstrap=0).",
                            UserWarning,
                            stacklevel=4,
                        )
                    sigma = np.diag(np.array(ses) ** 2)

                # Validate the full event-time grid is consecutive.
                # For universal base: exactly one gap for the omitted reference.
                # For varying base: no gap expected (pre ends at -1, post starts at 0).
                if pre_times and post_times:
                    if ref_period is not None:
                        # Universal: pre[-1]+1 = ref, ref+1 = post[0] → gap of 2
                        ref_gap = post_times[0] - pre_times[-1]
                        has_gap = ref_gap != 2
                    else:
                        # Varying: pre ends at -1, post starts at 0 → gap of 1
                        ref_gap = post_times[0] - pre_times[-1]
                        has_gap = ref_gap != 1
                elif pre_times:
                    has_gap = False  # only pre, no ref gap to check
                elif post_times:
                    has_gap = False  # only post, no ref gap to check
                else:
                    has_gap = False
                # Also check within-block consecutiveness
                for block in [pre_times, post_times]:
                    if len(block) >= 2:
                        for i in range(len(block) - 1):
                            if block[i + 1] - block[i] != 1:
                                has_gap = True
                                break
                if has_gap:
                    raise ValueError(
                        "HonestDiD requires a consecutive event-time grid "
                        "around the omitted reference period. Retained "
                        f"pre-periods {pre_times} and post-periods "
                        f"{post_times} have gaps. This can happen when "
                        "some event-study horizons have non-finite SEs. "
                        "Ensure all event-study periods have valid estimates, "
                        "or use balance_e to restrict to a balanced subset."
                    )

                # Extract inference df. Prefer ``survey_metadata.df_survey``
                # (the actual CS-internal df, which may have been tightened
                # post-resolve for replicate designs) over the dedicated
                # ``df_inference`` field. ``df_inference`` is the FALLBACK
                # carrier for bare-``cluster=`` CS fits where
                # ``survey_metadata`` is intentionally None (preserving the
                # survey/non-survey contract for ``DiagnosticReport`` /
                # ``summary()``). Mirrors the MPD branch preference order
                # at ``_extract_event_study_params`` so all branches agree.
                # Replicate designs with undefined df → sentinel 0.
                df_survey = None
                if hasattr(results, "survey_metadata") and results.survey_metadata is not None:
                    sm = results.survey_metadata
                    df_survey = getattr(sm, "df_survey", None)
                    if df_survey is None and getattr(sm, "replicate_method", None) is not None:
                        df_survey = 0  # undefined replicate df → NaN inference
                if df_survey is None:
                    df_inference = getattr(results, "df_inference", None)
                    if df_inference is not None:
                        df_survey = int(df_inference)

                return (
                    beta_hat,
                    sigma,
                    len(pre_times),
                    len(post_times),
                    pre_times,
                    post_times,
                    df_survey,
                )
        except ImportError:
            pass

        # Try ChaisemartinDHaultfoeuilleResults (dCDH estimator)
        try:
            from diff_diff.chaisemartin_dhaultfoeuille_results import (
                ChaisemartinDHaultfoeuilleResults,
            )

            if isinstance(results, ChaisemartinDHaultfoeuilleResults):
                import warnings

                warnings.warn(
                    "HonestDiD on dCDH results uses DID^{pl}_l placebo "
                    "estimates as pre-period coefficients, not standard "
                    "event-study pre-treatment coefficients. The Rambachan-"
                    "Roth restrictions bound violations of the parallel "
                    "trends assumption underlying the dCDH placebo "
                    "estimand. This is a library extension; interpretation "
                    "differs from canonical event-study HonestDiD.",
                    UserWarning,
                    stacklevel=3,
                )

                if results.placebo_event_study is None:
                    raise ValueError(
                        "ChaisemartinDHaultfoeuilleResults must have placebo_event_study "
                        "for HonestDiD. Re-run ChaisemartinDHaultfoeuille.fit() with "
                        "L_max >= 1 to compute multi-horizon placebos."
                    )
                if results.event_study_effects is None:
                    raise ValueError(
                        "ChaisemartinDHaultfoeuilleResults must have event_study_effects "
                        "for HonestDiD."
                    )

                # Filter for finite SEs in both surfaces
                placebo_finite = {
                    h: data
                    for h, data in results.placebo_event_study.items()
                    if np.isfinite(data.get("se", np.nan))
                }
                effects_finite = {
                    h: data
                    for h, data in results.event_study_effects.items()
                    if np.isfinite(data.get("se", np.nan))
                }

                pre_times = sorted(placebo_finite.keys())  # -P, ..., -1
                post_times = sorted(effects_finite.keys())  # 1, ..., L_max

                if len(pre_times) == 0:
                    raise ValueError(
                        "No placebo horizons with finite SEs found in dCDH results. "
                        "HonestDiD requires at least one identified pre-period "
                        "coefficient."
                    )
                if len(post_times) == 0:
                    raise ValueError(
                        "No event study horizons with finite SEs found in dCDH results. "
                        "HonestDiD requires at least one post-period coefficient."
                    )

                # Consecutiveness check: more permissive than CS because
                # trends_nonparam support-trimming can create legitimate gaps.
                # Filter to the largest consecutive block spanning the -1/+1
                # boundary; warn about dropped horizons.
                def _largest_consecutive_block(times, boundary_val):
                    """Find largest consecutive block containing boundary_val."""
                    if not times:
                        return []
                    if boundary_val not in times:
                        raise ValueError(
                            f"HonestDiD requires horizon {boundary_val} in "
                            f"the dCDH "
                            f"{'placebo' if boundary_val < 0 else 'event study'}"
                            f" surface, but it was removed by finite-SE "
                            f"filtering. Retained horizons: {times}. Ensure "
                            f"horizon {boundary_val} has a finite SE."
                        )
                    # Expand outward from boundary_val
                    block = [boundary_val]
                    idx = times.index(boundary_val)
                    # Expand left
                    for i in range(idx - 1, -1, -1):
                        if times[i] == block[0] - 1:
                            block.insert(0, times[i])
                        else:
                            break
                    # Expand right
                    for i in range(idx + 1, len(times)):
                        if times[i] == block[-1] + 1:
                            block.append(times[i])
                        else:
                            break
                    return block

                pre_consec = _largest_consecutive_block(pre_times, -1)
                post_consec = _largest_consecutive_block(post_times, 1)

                dropped_pre = set(pre_times) - set(pre_consec)
                dropped_post = set(post_times) - set(post_consec)

                if dropped_pre or dropped_post:
                    import warnings

                    dropped = sorted(dropped_pre | dropped_post)
                    warnings.warn(
                        f"HonestDiD requires a consecutive event-time grid. "
                        f"Dropping non-consecutive horizons {dropped} from dCDH "
                        f"results. This can happen when trends_nonparam "
                        f"support-trimming removes horizons. Retained: "
                        f"pre={pre_consec}, post={post_consec}.",
                        UserWarning,
                        stacklevel=3,
                    )
                    pre_times = pre_consec
                    post_times = post_consec

                if len(pre_times) == 0 or len(post_times) == 0:
                    raise ValueError(
                        "After filtering for consecutive horizons, no pre- or "
                        "post-periods remain. Cannot compute HonestDiD bounds."
                    )

                # Build beta_hat and sigma (diagonal - no full VCV for dCDH)
                effects = []
                ses = []
                for h in pre_times:
                    effects.append(placebo_finite[h]["effect"])
                    ses.append(placebo_finite[h]["se"])
                for h in post_times:
                    effects.append(effects_finite[h]["effect"])
                    ses.append(effects_finite[h]["se"])

                beta_hat = np.array(effects)
                sigma = np.diag(np.array(ses) ** 2)

                # Extract inference df. Prefer ``survey_metadata.df_survey``
                # (the actual CS-internal df, which may have been tightened
                # post-resolve for replicate designs) over the dedicated
                # ``df_inference`` field. ``df_inference`` is the FALLBACK
                # carrier for bare-``cluster=`` CS fits where
                # ``survey_metadata`` is intentionally None (preserving the
                # survey/non-survey contract for ``DiagnosticReport`` /
                # ``summary()``). Mirrors the MPD branch preference order
                # at ``_extract_event_study_params`` so all branches agree.
                # Replicate designs with undefined df → sentinel 0.
                df_survey = None
                if hasattr(results, "survey_metadata") and results.survey_metadata is not None:
                    sm = results.survey_metadata
                    df_survey = getattr(sm, "df_survey", None)
                    if df_survey is None and getattr(sm, "replicate_method", None) is not None:
                        df_survey = 0  # undefined replicate df → NaN inference
                if df_survey is None:
                    df_inference = getattr(results, "df_inference", None)
                    if df_inference is not None:
                        df_survey = int(df_inference)

                return (
                    beta_hat,
                    sigma,
                    len(pre_times),
                    len(post_times),
                    pre_times,
                    post_times,
                    df_survey,
                )
        except ImportError:
            pass

        raise TypeError(
            f"Unsupported results type: {type(results)}. "
            "Expected MultiPeriodDiDResults, CallawaySantAnnaResults, "
            "or ChaisemartinDHaultfoeuilleResults."
        )


def _construct_A_sd(num_pre_periods: int, num_post_periods: int) -> np.ndarray:
    """
    Construct constraint matrix for smoothness (second differences).

    Builds the matrix A such that A @ delta gives the second differences,
    accounting for the normalization delta_0 = 0 at the pre-post boundary.

    The delta vector is [delta_{-T}, ..., delta_{-1}, delta_1, ..., delta_{Tbar}]
    (delta_0 = 0 is omitted). Second differences at the boundary use delta_0 = 0:
      t=-1: delta_{-2} - 2*delta_{-1} + 0  (if num_pre >= 2)
      t= 0: delta_{-1} + delta_1           (bridge constraint, always present)
      t= 1: 0 - 2*delta_1 + delta_2        (if num_post >= 2)

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods (T).
    num_post_periods : int
        Number of post-treatment periods (Tbar).

    Returns
    -------
    A : np.ndarray
        Constraint matrix of shape (n_constraints, num_pre + num_post).
        n_constraints = num_pre + num_post - 1 for sufficient periods,
        accounting for the delta_0 = 0 boundary.
    """
    T = num_pre_periods
    Tbar = num_post_periods
    total = T + Tbar

    if total < 2:
        return np.zeros((0, total))

    rows = []

    # Pure pre-period second differences: t = -T+1, ..., -2
    # These involve delta[i-1], delta[i], delta[i+1] all in the pre-period block
    # Row i corresponds to: delta_{-(T-i)} - 2*delta_{-(T-i-1)} + delta_{-(T-i-2)}
    for i in range(T - 2):
        row = np.zeros(total)
        row[i] = 1  # delta_{t-1}
        row[i + 1] = -2  # delta_t
        row[i + 2] = 1  # delta_{t+1}
        rows.append(row)

    # Boundary constraint at t = -1: delta_{-2} - 2*delta_{-1} + delta_0
    # With delta_0 = 0: delta_{-2} - 2*delta_{-1}
    if T >= 2:
        row = np.zeros(total)
        row[T - 2] = 1  # delta_{-2}
        row[T - 1] = -2  # delta_{-1}
        # delta_0 = 0, no entry needed
        rows.append(row)

    # Bridge constraint at t = 0: delta_{-1} - 2*delta_0 + delta_1
    # With delta_0 = 0: delta_{-1} + delta_1
    if T >= 1 and Tbar >= 1:
        row = np.zeros(total)
        row[T - 1] = 1  # delta_{-1}
        row[T] = 1  # delta_1
        rows.append(row)

    # Boundary constraint at t = 1: delta_0 - 2*delta_1 + delta_2
    # With delta_0 = 0: -2*delta_1 + delta_2
    if Tbar >= 2:
        row = np.zeros(total)
        row[T] = -2  # delta_1
        row[T + 1] = 1  # delta_2
        rows.append(row)

    # Pure post-period second differences: event times t = 2, ..., Tbar-1
    # delta_{t+1} - 2*delta_t + delta_{t-1}, all within the post-period block
    for t in range(2, Tbar):
        row = np.zeros(total)
        row[T + t - 2] = 1  # delta_{t-1}
        row[T + t - 1] = -2  # delta_t
        row[T + t] = 1  # delta_{t+1}
        rows.append(row)

    if not rows:
        return np.zeros((0, total))

    return np.array(rows)


def _construct_constraints_sd(
    num_pre_periods: int, num_post_periods: int, M: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct smoothness constraint matrices for Delta^SD(M).

    Returns A, b such that delta in DeltaSD(M) iff |A @ delta| <= b.
    Accounts for delta_0 = 0 normalization at the pre-post boundary.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    M : float
        Smoothness parameter (max second difference).

    Returns
    -------
    A_ineq : np.ndarray
        Inequality constraint matrix.
    b_ineq : np.ndarray
        Inequality constraint vector.
    """
    A_base = _construct_A_sd(num_pre_periods, num_post_periods)

    if A_base.shape[0] == 0:
        total = num_pre_periods + num_post_periods
        return np.zeros((0, total)), np.zeros(0)

    # |A @ delta| <= M becomes:
    # A @ delta <= M  and  -A @ delta <= M
    A_ineq = np.vstack([A_base, -A_base])
    b_ineq = np.full(2 * A_base.shape[0], M)

    return A_ineq, b_ineq


def _construct_constraints_rm_component(
    num_pre_periods: int,
    num_post_periods: int,
    Mbar: float,
    max_pre_first_diff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct constraint matrices for one component of Delta^RM.

    Delta^RM constrains post-treatment FIRST DIFFERENCES (not levels):
        |delta_{t+1} - delta_t| <= Mbar * max_pre_first_diff, for all t >= 0

    With delta_0 = 0 normalization:
        |delta_1| <= bound                         (t=0)
        |delta_{t+1} - delta_t| <= bound           (t=1, ..., Tbar-1)

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    Mbar : float
        Relative magnitude scaling factor.
    max_pre_first_diff : float
        The pre-period first difference for this union component.

    Returns
    -------
    A_ineq : np.ndarray
        Inequality constraint matrix.
    b_ineq : np.ndarray
        Inequality constraint vector.
    """
    T = num_pre_periods
    Tbar = num_post_periods
    total = T + Tbar
    bound = Mbar * max_pre_first_diff

    rows = []

    # t=0: |delta_1 - delta_0| = |delta_1| <= bound (delta_0 = 0)
    if Tbar >= 1:
        row_pos = np.zeros(total)
        row_pos[T] = 1  # delta_1 <= bound
        rows.append(row_pos)
        row_neg = np.zeros(total)
        row_neg[T] = -1  # -delta_1 <= bound
        rows.append(row_neg)

    # t=1, ..., Tbar-1: |delta_{t+1} - delta_t| <= bound
    for t in range(1, Tbar):
        row_pos = np.zeros(total)
        row_pos[T + t] = 1  # delta_{t+1}
        row_pos[T + t - 1] = -1  # -delta_t
        rows.append(row_pos)
        row_neg = np.zeros(total)
        row_neg[T + t] = -1  # -delta_{t+1}
        row_neg[T + t - 1] = 1  # delta_t
        rows.append(row_neg)

    if not rows:
        return np.zeros((0, total)), np.zeros(0)

    A_ineq = np.array(rows)
    b_ineq = np.full(len(rows), bound)
    return A_ineq, b_ineq


def _compute_pre_first_differences(beta_pre: np.ndarray) -> np.ndarray:
    """
    Compute pre-period first differences for Delta^RM.

    With delta_0 = 0 normalization, the pre-period first differences are:
        fd_s = delta_{s+1} - delta_s  for s = -T, ..., -1

    Since delta_pre = beta_pre (by no-anticipation):
        fd_{-T}   = beta_{-T+1} - beta_{-T}
        ...
        fd_{-2}   = beta_{-1} - beta_{-2}
        fd_{-1}   = delta_0 - beta_{-1} = -beta_{-1}  (boundary through delta_0=0)

    Parameters
    ----------
    beta_pre : np.ndarray
        Pre-period coefficient estimates [beta_{-T}, ..., beta_{-1}].

    Returns
    -------
    first_diffs : np.ndarray
        Absolute first differences |fd_{-T}|, ..., |fd_{-1}|.
    """
    if len(beta_pre) == 0:
        return np.array([])

    diffs = []
    # Interior first differences: fd_s = beta_{s+1} - beta_s
    for i in range(len(beta_pre) - 1):
        diffs.append(abs(beta_pre[i + 1] - beta_pre[i]))
    # Boundary: fd_{-1} = delta_0 - delta_{-1} = 0 - beta_{-1} = -beta_{-1}
    diffs.append(abs(beta_pre[-1]))

    return np.array(diffs)


def _solve_rm_bounds_union(
    beta_pre: np.ndarray,
    beta_post: np.ndarray,
    l_vec: np.ndarray,
    num_pre_periods: int,
    Mbar: float,
    lp_method: str = "highs",
) -> Tuple[float, float]:
    """
    Solve identified set bounds for Delta^RM via union of polyhedra.

    Delta^RM is a union of polyhedra (one per location of the max pre-period
    first difference). Per Lemma 2.2 of Rambachan & Roth (2023), the
    identified set is the union of component identified sets.

    With delta_pre = beta_pre pinned, each pre-period first difference is
    a known scalar, so each component LP has simple box constraints on
    post-treatment first differences.

    Parameters
    ----------
    beta_pre : np.ndarray
        Pre-period coefficients.
    beta_post : np.ndarray
        Post-period coefficients.
    l_vec : np.ndarray
        Weighting vector.
    num_pre_periods : int
        Number of pre-periods.
    Mbar : float
        Relative magnitudes scaling factor.
    lp_method : str
        LP solver method.

    Returns
    -------
    lb : float
        Lower bound (min over all components).
    ub : float
        Upper bound (max over all components).
    """
    pre_diffs = _compute_pre_first_differences(beta_pre)
    num_post = len(beta_post)

    if len(pre_diffs) == 0 or np.max(pre_diffs) == 0:
        theta = np.dot(l_vec, beta_post)
        return theta, theta

    # After pinning delta_pre = beta_pre, the RM bound is determined by
    # max(pre_diffs). Smaller components give tighter constraints and thus
    # narrower bounds that are nested inside the max-component bounds.
    # One LP call suffices (Lemma 2.2 union simplifies to max component).
    max_pre_fd = float(np.max(pre_diffs))
    A_ineq, b_ineq = _construct_constraints_rm_component(
        num_pre_periods, num_post, Mbar, max_pre_fd
    )
    return _solve_bounds_lp(beta_pre, beta_post, l_vec, A_ineq, b_ineq, num_pre_periods, lp_method)


def _solve_bounds_lp(
    beta_pre: np.ndarray,
    beta_post: np.ndarray,
    l_vec: np.ndarray,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    num_pre_periods: int,
    lp_method: str = "highs",
) -> Tuple[float, float]:
    """
    Solve for identified set bounds using linear programming.

    Computes the bounds of the identified set S(beta, Delta) per
    Rambachan & Roth (2023) Equations 5-6:

        theta^lb = l'beta_post - max{ l'delta_post : delta in Delta, delta_pre = beta_pre }
        theta^ub = l'beta_post - min{ l'delta_post : delta in Delta, delta_pre = beta_pre }

    The equality constraint delta_pre = beta_pre pins the pre-treatment violations
    to the observed pre-treatment coefficients (since tau_pre = 0 by no-anticipation).

    Parameters
    ----------
    beta_pre : np.ndarray
        Pre-period coefficient estimates (pinned as equality constraints).
    beta_post : np.ndarray
        Post-period coefficient estimates.
    l_vec : np.ndarray
        Weighting vector for aggregation.
    A_ineq : np.ndarray
        Inequality constraint matrix (for all periods).
    b_ineq : np.ndarray
        Inequality constraint vector.
    num_pre_periods : int
        Number of pre-periods (for indexing).
    lp_method : str
        LP solver method for scipy.optimize.linprog. Default 'highs' requires
        scipy >= 1.6.0. Alternatives: 'interior-point', 'revised simplex'.

    Returns
    -------
    lb : float
        Lower bound of identified set.
    ub : float
        Upper bound of identified set.
    """
    num_post = len(beta_post)
    total_periods = A_ineq.shape[1] if A_ineq.shape[0] > 0 else num_pre_periods + num_post

    # Objective: min/max -l' @ delta_post over delta in R^total_periods
    c = np.zeros(total_periods)
    c[num_pre_periods : num_pre_periods + num_post] = -l_vec

    # Equality constraints: delta_pre = beta_pre (Rambachan & Roth Eqs 5-6)
    A_eq = np.zeros((num_pre_periods, total_periods))
    for i in range(num_pre_periods):
        A_eq[i, i] = 1.0
    b_eq = beta_pre

    if A_ineq.shape[0] == 0 and num_pre_periods == 0:
        return -np.inf, np.inf

    lp_kwargs = dict(
        A_ub=A_ineq if A_ineq.shape[0] > 0 else None,
        b_ub=b_ineq if A_ineq.shape[0] > 0 else None,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(None, None),
        method=lp_method,
    )

    # Solve for min(-l'@delta_post) → gives upper bound of theta
    try:
        result_min = optimize.linprog(c, **lp_kwargs)
        if result_min.success:
            min_val = result_min.fun
        elif result_min.status == 2:
            # Infeasible: beta_pre inconsistent with Delta at this M
            return np.nan, np.nan
        else:
            min_val = -np.inf
    except (ValueError, TypeError):
        min_val = -np.inf

    # Solve for max(-l'@delta_post) → gives lower bound of theta
    try:
        result_max = optimize.linprog(-c, **lp_kwargs)
        if result_max.success:
            max_val = -result_max.fun
        elif result_max.status == 2:
            return np.nan, np.nan
        else:
            max_val = np.inf
    except (ValueError, TypeError):
        max_val = np.inf

    theta_base = np.dot(l_vec, beta_post)
    lb = theta_base + min_val  # = l'@beta + min(-l'@delta) = min(l'@(beta-delta))
    ub = theta_base + max_val  # = l'@beta + max(-l'@delta) = max(l'@(beta-delta))

    return lb, ub


def _compute_flci(
    lb: float,
    ub: float,
    se: float,
    alpha: float = 0.05,
    df: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute Fixed Length Confidence Interval (FLCI).

    The FLCI extends the identified set by a critical value times
    the standard error on each side.

    Parameters
    ----------
    lb : float
        Lower bound of identified set.
    ub : float
        Upper bound of identified set.
    se : float
        Standard error of the estimator.
    alpha : float
        Significance level.
    df : int, optional
        Degrees of freedom. If provided, uses t-distribution critical value
        instead of normal (for survey designs with df = n_PSU - n_strata).

    Returns
    -------
    ci_lb : float
        Lower bound of confidence interval.
    ci_ub : float
        Upper bound of confidence interval.

    Raises
    ------
    ValueError
        If se <= 0 or alpha is not in (0, 1).
    """
    if se <= 0:
        raise ValueError(f"Standard error must be positive, got se={se}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got alpha={alpha}")

    z = _get_critical_value(alpha, df)
    ci_lb = lb - z * se
    ci_ub = ub + z * se
    return ci_lb, ci_ub


def _cv_alpha(t: float, alpha: float, df: Optional[int] = None) -> float:
    """
    Compute the (1-alpha) quantile of the folded distribution |X|.

    When df is None: X ~ N(t, 1) (folded normal).
    When df > 0: X ~ nct(df, t) (folded non-central t, for survey inference).
    Per Rambachan & Roth (2023) Equation 18.

    Parameters
    ----------
    t : float
        Non-centrality parameter (bias / se ratio).
    alpha : float
        Significance level.
    df : int, optional
        Degrees of freedom for non-central t. None = normal theory.

    Returns
    -------
    cv : float
        Critical value such that P(|X| <= cv) = 1 - alpha.
    """
    from scipy.stats import norm

    target = 1 - alpha
    t = abs(t)

    if df is not None and df > 0:
        # Folded non-central t: P(|nct(df,t)| <= x) = F(x;df,t) - F(-x;df,t)
        from scipy.stats import nct as nct_dist

        x = nct_dist.ppf(1 - alpha / 2, df, t) + 1.0  # generous start
        for _ in range(30):
            f = nct_dist.cdf(x, df, t) - nct_dist.cdf(-x, df, t) - target
            fprime = nct_dist.pdf(x, df, t) + nct_dist.pdf(-x, df, t)
            if fprime < 1e-15:
                break
            x_new = x - f / fprime
            x_new = max(x_new, 0.0)
            if abs(x_new - x) < 1e-10:
                break
            x = x_new
        return x

    # Folded normal: P(|N(t,1)| <= x) = Phi(x-t) - Phi(-x-t)
    x = norm.ppf(1 - alpha / 2) + t

    for _ in range(20):
        f = norm.cdf(x - t) - norm.cdf(-x - t) - target
        fprime = norm.pdf(x - t) + norm.pdf(-x - t)
        if fprime < 1e-15:
            break
        x_new = x - f / fprime
        x_new = max(x_new, 0.0)
        if abs(x_new - x) < 1e-12:
            break
        x = x_new

    return x


def _build_fd_transform(num_pre: int, num_post: int) -> np.ndarray:
    """
    Build the matrix C mapping first-differences to levels: delta = C @ fd.

    The fd vector has T+Tbar components:
        fd = [fd_{-T}, ..., fd_{-1}, fd_0, ..., fd_{Tbar-1}]
    where fd_s = delta_{s+1} - delta_s (with delta_0 = 0).

    The delta vector is:
        delta = [delta_{-T}, ..., delta_{-1}, delta_1, ..., delta_{Tbar}]

    Pre-period (backward from delta_0=0):
        delta_{-1} = -fd_{T-1}
        delta_{-k} = -(fd_{T-1} + fd_{T-2} + ... + fd_{T-k})

    Post-period (forward from delta_0=0):
        delta_1 = fd_T
        delta_k = fd_T + fd_{T+1} + ... + fd_{T+k-1}
    """
    T = num_pre
    Tbar = num_post
    total = T + Tbar
    C = np.zeros((total, total))

    # Pre-period: delta_{-k} = -(fd_{T-1} + fd_{T-2} + ... + fd_{T-k})
    for k in range(1, T + 1):
        delta_idx = T - k  # delta_{-k} is at index T-k
        for j in range(k):
            fd_idx = T - 1 - j  # fd_{T-1-j}
            C[delta_idx, fd_idx] = -1.0

    # Post-period: delta_k = fd_T + fd_{T+1} + ... + fd_{T+k-1}
    for k in range(1, Tbar + 1):
        delta_idx = T + k - 1  # delta_k is at index T+k-1
        for j in range(k):
            fd_idx = T + j  # fd_{T+j}
            C[delta_idx, fd_idx] = 1.0

    return C


def _build_fd_smoothness_constraints(num_fd: int, M: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build smoothness constraints in first-difference space.

    Delta^SD(M) in fd-space: |fd_{i+1} - fd_i| <= M for all consecutive pairs.
    This is a bounded polyhedron (unlike level-space Delta^SD which is unbounded).
    """
    if num_fd < 2:
        return np.zeros((0, num_fd)), np.zeros(0)

    n_constraints = num_fd - 1
    rows = []
    for i in range(n_constraints):
        row_pos = np.zeros(num_fd)
        row_pos[i + 1] = 1
        row_pos[i] = -1
        rows.append(row_pos)
        row_neg = np.zeros(num_fd)
        row_neg[i + 1] = -1
        row_neg[i] = 1
        rows.append(row_neg)

    A = np.array(rows)
    b = np.full(len(rows), M)
    return A, b


def _w_to_v(w: np.ndarray, l: np.ndarray, num_pre: int) -> np.ndarray:
    """
    Map slope weights w to the full estimator direction v.

    The estimator is: theta_hat = l'beta_post - sum_s w_s (beta_s - beta_{s-1})
    for s = -T+1, ..., 0 (T slopes total, including boundary slope s=0
    where beta_0 = 0).

    This gives v = (v_pre, l) where v_pre is determined by differencing w.

    Parameters
    ----------
    w : np.ndarray
        Weights on slopes (length T). Includes the boundary slope at s=0.
    l : np.ndarray
        Target parameter weights (length Tbar).
    num_pre : int
        Number of pre-periods (T).
    """
    T = num_pre
    Tbar = len(l)
    v = np.zeros(T + Tbar)

    if len(w) > 0:
        # v[0] = w[0] (beta_{-T} from slope s=-T+1)
        v[0] = w[0]
        # v[k] = -w[k-1] + w[k] for k=1,...,T-1
        for k in range(1, T):
            v[k] = -w[k - 1] + w[k]

    v[T:] = l
    return v


def _compute_worst_case_bias(
    w: np.ndarray,
    l: np.ndarray,
    num_pre: int,
    num_post: int,
    M: float,
) -> float:
    """
    Compute worst-case bias of the FLCI affine estimator for Delta^SD.

    Per Rambachan & Roth (2023) Eq. 17, the bias is max |v'delta| over
    Delta^SD(M). This is computed in first-difference space where Delta^SD
    is a bounded polyhedron |fd_{i+1} - fd_i| <= M.

    The bias direction in fd-space is C'v, where C maps fd -> delta and
    v is the estimator direction derived from slope weights w.

    Parameters
    ----------
    w : np.ndarray
        Slope weights (length T), sum(w) = sum_j j*l_j (Eq. 17 neutrality).
    l : np.ndarray
        Target parameter weights.
    num_pre : int
        Number of pre-periods (T).
    num_post : int
        Number of post-periods (Tbar).
    M : float
        Smoothness parameter.

    Returns
    -------
    bias : float
        Maximum worst-case bias (finite for M >= 0).
    """
    if M == 0:
        return 0.0  # Linear trends => zero bias when sum(w)=1

    total = num_pre + num_post
    v = _w_to_v(w, l, num_pre)
    C = _build_fd_transform(num_pre, num_post)
    A_fd, b_fd = _build_fd_smoothness_constraints(total, M)

    # Bias direction in fd-space: max (C'v)' fd subject to smoothness
    bias_dir_fd = C.T @ v

    if A_fd.shape[0] == 0:
        return 0.0

    # Centrosymmetric: max |c'fd| = max c'fd
    try:
        res = optimize.linprog(
            -bias_dir_fd,
            A_ub=A_fd,
            b_ub=b_fd,
            bounds=(None, None),
            method="highs",
        )
        return -res.fun if res.success else np.inf
    except (ValueError, TypeError):
        return np.inf


def _flci_bias_constant(l_vec: np.ndarray, num_post: int) -> float:
    """Constant term of the worst-case-bias objective (R's
    ``.createObjectiveObjectForBias``): ``sum_s |sum_{k=1..s} k*l[Tbar-s+k]| -
    sum_s s*l[s]``. For NONNEGATIVE (averaging) ``l_vec`` the closed form
    ``M*(constant + sum_i |cumsum(w)_i|)`` equals ``_compute_worst_case_bias(w,
    ..., M)`` (the LP oracle) to machine precision; for signed/contrast ``l_vec``
    it intentionally follows R's conservative closed form (the FLCI still matches
    R). See REGISTRY.md Delta^SD FLCI note.
    """
    c = 0.0
    for s in range(1, num_post + 1):
        seg = l_vec[num_post - s : num_post]
        c += abs(float(np.dot(np.arange(1, s + 1), seg)))
    c -= float(np.dot(np.arange(1, num_post + 1), l_vec))
    return c


def _flci_precompute(sigma: np.ndarray, l_vec: np.ndarray, num_pre: int, num_post: int) -> dict:
    """Precompute the FLCI inner problem's quadratic-variance and bias pieces
    (``sigma`` ordered ``[pre; post]``). ``D`` is the first-difference operator
    (identical to the pre-block of ``_w_to_v``), ``L`` the cumsum operator, so the
    estimator variance is ``var(w) = w'Qw + q'w + c0`` and the linear-trend
    neutrality target is ``sum(w) = sum_s s*l[s]`` (R & R 2023 Eq. 17)."""
    s_pre = sigma[:num_pre, :num_pre]
    s_pp = sigma[:num_pre, num_pre:]
    s_post = float(l_vec @ sigma[num_pre:, num_pre:] @ l_vec)
    d_mat = np.eye(num_pre)
    for col in range(num_pre - 1):
        d_mat[col + 1, col] = -1.0
    return {
        "Q": d_mat.T @ s_pre @ d_mat,
        "q": 2.0 * (d_mat.T @ s_pp @ l_vec),
        "c0": s_post,
        "L": np.tril(np.ones((num_pre, num_pre))),
        "target": float(np.dot(np.arange(1, num_post + 1), l_vec)),
        "constant": _flci_bias_constant(l_vec, num_post),
        "num_pre": num_pre,
    }


def _flci_var(w: np.ndarray, P: dict) -> float:
    return float(w @ P["Q"] @ w + P["q"] @ w + P["c0"])


def _flci_w_min(P: dict) -> np.ndarray:
    """Minimum-variance weights subject to ``sum(w)=target`` via the
    equality-constrained KKT system (the abs-value constraints are non-binding for
    the min-variance objective; verified against R's ``.findLowestH``)."""
    k = P["num_pre"]
    kkt = np.zeros((k + 1, k + 1))
    kkt[:k, :k] = 2.0 * P["Q"]
    kkt[:k, k] = 1.0
    kkt[k, :k] = 1.0
    rhs = np.concatenate([-P["q"], [P["target"]]])
    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return sol[:k]


def _flci_h_bounds(P: dict):
    """FLCI bracket: ``hmin`` = minimum estimator SD (R ``.findLowestH``), ``h0`` =
    SD at the minimum-bias weights (all mass on the boundary slope, R
    ``.findHForMinimumBias``). Both match R's closed forms to ~1e-8."""
    w_min = _flci_w_min(P)
    hmin = float(np.sqrt(max(_flci_var(w_min, P), 0.0)))
    w_hi = np.zeros(P["num_pre"])
    w_hi[-1] = P["target"]
    h0 = float(np.sqrt(max(_flci_var(w_hi, P), 0.0)))
    return hmin, h0, w_min


def _flci_min_bias_given_h(P: dict, h: float, x0_w: Optional[np.ndarray] = None):
    """Inner solve (R's ``.findWorstCaseBiasGivenH``): minimize the worst-case bias
    ``constant + sum_i|cumsum(w)_i|`` subject to ``sum(w)=target`` and the
    second-order-cone ``var(w) <= h^2``, as a smooth convex QCQP over ``x=[U; w]``
    (``U >= |Lw|``) via SLSQP. Returns ``(w, bias_unit, feasible)``.

    Gating is on FEASIBILITY (``var<=h^2`` and ``sum=target``), NOT ``res.success``
    -- SLSQP frequently reports ``success=False`` on a correct convex optimum.
    """
    from scipy.optimize import minimize as _sp_minimize

    k = P["num_pre"]
    L = P["L"]
    if k == 1:
        w = np.array([P["target"]])
        return w, P["constant"] + float(np.abs(L @ w).sum()), True
    if x0_w is None:
        x0_w = _flci_w_min(P)
    x0 = np.concatenate([np.abs(L @ x0_w), x0_w])
    c = np.concatenate([np.ones(k), np.zeros(k)])
    cons = [
        {
            "type": "ineq",
            "fun": lambda x: x[:k] - L @ x[k:],
            "jac": lambda x: np.hstack([np.eye(k), -L]),
        },
        {
            "type": "ineq",
            "fun": lambda x: x[:k] + L @ x[k:],
            "jac": lambda x: np.hstack([np.eye(k), L]),
        },
        {
            "type": "eq",
            "fun": lambda x: float(np.sum(x[k:]) - P["target"]),
            "jac": lambda x: np.concatenate([np.zeros(k), np.ones(k)]),
        },
        {
            "type": "ineq",
            "fun": lambda x: h * h - _flci_var(x[k:], P),
            "jac": lambda x: np.concatenate([np.zeros(k), -(2.0 * P["Q"] @ x[k:] + P["q"])]),
        },
    ]
    res = _sp_minimize(
        lambda x: float(c @ x),
        x0,
        jac=lambda x: c,
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    w = res.x[k:]
    feasible = (_flci_var(w, P) <= h * h + 1e-7) and (abs(float(np.sum(w)) - P["target"]) < 1e-6)
    return w, P["constant"] + float(np.abs(L @ w).sum()), bool(feasible)


def _flci_optimal_h(
    P: dict,
    hmin: float,
    h0: float,
    M: float,
    alpha: float,
    df: Optional[int],
    levels=(40, 120, 120),
) -> float:
    """Argmin over the estimator SD ``h`` of the half-length
    ``W(h)=cv(M*bias(h)/h)*h`` (R's ``.findOptimalCIDerivativeBisection``), by a
    multi-level GRID ZOOM. Each level restarts from ``w_min`` and warm-starts the
    inner solve within the level (``h`` ascending), which keeps the objective
    smooth despite ~1e-6 SLSQP noise; zooming refines resolution.

    A grid zoom (not R's derivative-bisection or an envelope-theorem derivative) is
    used deliberately: the width surface is near-flat at the optimum AND the
    min-bias inner solutions are degenerate (most ``cumsum(w)`` components sit at
    the abs-value kink), so ``mu`` is not recoverable from ``sign(Lw)`` and
    finite-difference derivatives are swamped by solver noise. Matches R's
    (deterministic) center to <3e-4 across the validated stress grid.
    """
    if h0 - hmin < 1e-12:
        return h0
    lo, hi = hmin, h0
    gi, grid = 0, np.array([hmin])
    for npts in levels:
        grid = np.linspace(lo, hi, npts)
        seed = _flci_w_min(P)
        vals = np.full(npts, np.inf)
        for i, h in enumerate(grid):
            if h <= 0.0:  # rank-deficient covariance -> hmin=0; skip (M*bias/h undefined)
                continue
            w, bias_unit, ok = _flci_min_bias_given_h(P, h, x0_w=seed)
            if ok and np.isfinite(bias_unit):
                vals[i] = _cv_alpha(M * bias_unit / h, alpha, df=df) * h
                seed = w
        gi = int(np.argmin(vals))
        if not np.isfinite(vals[gi]):
            return float("nan")
        step = (hi - lo) / (npts - 1)
        lo = max(grid[gi] - 2 * step, hmin)
        hi = min(grid[gi] + 2 * step, h0)
    return float(grid[gi])


def _flci_solve(
    beta_pre: np.ndarray,
    beta_post: np.ndarray,
    sigma: np.ndarray,
    l_vec: np.ndarray,
    num_pre: int,
    num_post: int,
    M: float,
    alpha: float = 0.05,
    df: Optional[int] = None,
) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    Compute the optimal Fixed Length Confidence Interval for Delta^SD.

    Per Rambachan & Roth (2023) Section 4.1, the optimal FLCI is:
        CI = (a + v'beta_hat) ± chi
    where (a, v) minimize the half-length chi subject to coverage.

    The estimator is parameterized in terms of slope weights w on
    pre-treatment first differences (Section 4.1.1):
        theta_hat = l'beta_post - sum_s w_s (beta_s - beta_{s-1})
    with constraint sum(w) = sum_j j*l_j (linear trend neutrality, Eq. 17).

    The bias is computed in first-difference space where Delta^SD is
    a bounded polyhedron, making the LP well-posed.

    When df is provided, uses the folded non-central t distribution
    for survey inference (replaces the folded normal).

    Parameters
    ----------
    beta_pre : np.ndarray
        Pre-period coefficients.
    beta_post : np.ndarray
        Post-period coefficients.
    sigma : np.ndarray
        Full variance-covariance matrix (pre + post periods).
    l_vec : np.ndarray
        Target parameter weights.
    num_pre : int
        Number of pre-periods (T).
    num_post : int
        Number of post-periods (Tbar).
    M : float
        Smoothness parameter.
    alpha : float
        Significance level.
    df : int, optional
        Survey degrees of freedom for folded t inference.

    Returns
    -------
    ci_lb : float
        Lower bound of FLCI.
    ci_ub : float
        Upper bound of FLCI.
    optimal_vec : np.ndarray or None
        The optimal affine-estimator direction ``v`` (``theta_hat = v'beta``), for
        parity testing; ``None`` when the CI is NaN.
    """
    # Negative smoothness is invalid (cv_alpha's abs() would silently treat it as
    # +|M|); guard the direct helper path too (fit()/sensitivity_analysis() also
    # validate, but tests call this directly).
    if M < 0:
        raise ValueError(f"M must be non-negative, got M={M}")
    # Survey df<=0 sentinel -> NaN inference.
    if df is not None and df <= 0:
        return np.nan, np.nan, None

    P = _flci_precompute(sigma, l_vec, num_pre, num_post)
    beta_full = np.concatenate([beta_pre, beta_post])

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if M == 0.0:
            # Zero worst-case bias (linear trends): the FLCI estimator is simply the
            # minimum-variance affine estimator, so this branch is unchanged from the
            # prior optimizer's M=0 behaviour (min variance -> half-length z*sd).
            w_opt = _flci_w_min(P)
            hstar = float(np.sqrt(max(_flci_var(w_opt, P), 0.0)))
            if hstar <= 0.0:
                # Zero estimator SD (degenerate covariance) -> undefined inference.
                return np.nan, np.nan, None
            chi = _cv_alpha(0.0, alpha, df=df) * hstar
        else:
            hmin, h0, _ = _flci_h_bounds(P)
            hstar = _flci_optimal_h(P, hmin, h0, M, alpha, df)
            # hstar<=0 (zero SD, e.g. degenerate covariance) would divide by zero in
            # M*bias/hstar; NaN the inference per the zero-SE convention.
            if not np.isfinite(hstar) or hstar <= 0.0:
                return np.nan, np.nan, None
            w_opt, bias_unit, feasible = _flci_min_bias_given_h(P, hstar)
            if not feasible:
                return np.nan, np.nan, None
            chi = _cv_alpha(M * bias_unit / hstar, alpha, df=df) * hstar

    v_opt = _w_to_v(w_opt, l_vec, num_pre)
    theta_hat = float(v_opt @ beta_full)
    if not np.isfinite(chi):
        return np.nan, np.nan, None
    return theta_hat - chi, theta_hat + chi, v_opt


def _compute_optimal_flci(
    beta_pre: np.ndarray,
    beta_post: np.ndarray,
    sigma: np.ndarray,
    l_vec: np.ndarray,
    num_pre: int,
    num_post: int,
    M: float,
    alpha: float = 0.05,
    df: Optional[int] = None,
) -> Tuple[float, float]:
    """Optimal Delta^SD FLCI ``(ci_lb, ci_ub)`` (Rambachan & Roth 2023 §4.1).
    Thin wrapper over :func:`_flci_solve` (which also returns the optimal affine
    estimator direction ``v`` for parity testing)."""
    lb, ub, _ = _flci_solve(beta_pre, beta_post, sigma, l_vec, num_pre, num_post, M, alpha, df)
    return lb, ub


def _setup_moment_inequalities(
    beta_hat: np.ndarray,
    sigma_hat: np.ndarray,
    A: np.ndarray,
    d: np.ndarray,
    l: np.ndarray,
    theta_bar: float,
    num_pre: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform H0: theta = theta_bar into moment inequality form.

    Per Rambachan & Roth (2023) Equations 12-13.

    Returns
    -------
    Y_tilde : np.ndarray
        Transformed statistic.
    X_tilde : np.ndarray
        Transformed nuisance matrix.
    Sigma_tilde : np.ndarray
        Transformed covariance.
    """
    num_post = len(beta_hat) - num_pre

    # Y_n = A @ beta_hat - d
    Y_n = A @ beta_hat - d

    # Build A_tilde: transform to eliminate tau_post nuisance
    # A_tilde_{(.,1)} corresponds to the target direction
    # A_tilde_{(.,rest)} corresponds to nuisance parameters
    L_post = np.zeros((len(beta_hat), num_post))
    L_post[num_pre:, :] = np.eye(num_post)

    A_tilde = A @ L_post  # shape: (n_constraints, num_post)

    # Change of basis: first column = l direction, rest = complement
    # Use QR on l to get orthogonal complement
    l_full = l.reshape(-1, 1)
    Q, _ = np.linalg.qr(np.hstack([l_full, np.eye(num_post)[:, : num_post - 1]]))

    A_tilde_rotated = A_tilde @ Q  # Rotate into (l, complement) basis

    # Y_tilde(theta_bar) = Y_n - A_tilde_{col1} * theta_bar
    Y_tilde = Y_n - A_tilde_rotated[:, 0] * theta_bar

    # X_tilde = remaining columns (nuisance)
    X_tilde = A_tilde_rotated[:, 1:]

    # Sigma_tilde
    Sigma_tilde = A @ sigma_hat @ A.T

    return Y_tilde, X_tilde, Sigma_tilde


def _enumerate_vertices(
    X_tilde: np.ndarray,
    sigma_tilde_diag: np.ndarray,
    n_moments: int,
) -> List[np.ndarray]:
    """
    Enumerate basic feasible solutions of the dual LP.

    The dual feasible set is:
        {gamma >= 0 : gamma' @ X_tilde = 0, gamma' @ sigma_tilde_diag = 1}

    For small problems (typical n_moments <= 15), we enumerate all
    possible bases using combinatorial search.

    Parameters
    ----------
    X_tilde : np.ndarray
        Nuisance constraint matrix, shape (n_moments, n_nuisance).
    sigma_tilde_diag : np.ndarray
        sqrt(diag(Sigma_tilde)), shape (n_moments,).
    n_moments : int
        Number of moment inequalities.

    Returns
    -------
    vertices : list of np.ndarray
        Feasible vertices (gamma vectors).
    """
    import itertools

    n_nuisance = X_tilde.shape[1] if X_tilde.ndim > 1 else 0
    n_eq = n_nuisance + 1  # nuisance zero conditions + normalization

    if n_eq > n_moments:
        return []

    vertices: List[np.ndarray] = []
    n_total = 0
    n_linalg_error = 0
    n_infeasible = 0

    # Each vertex has exactly n_eq non-zero (basic) variables
    for basis_idx in itertools.combinations(range(n_moments), n_eq):
        basis_idx = list(basis_idx)
        n_total += 1

        # Build the system for basic variables
        # gamma[basis_idx]' @ X_tilde[basis_idx, :] = 0
        # gamma[basis_idx]' @ sigma_tilde_diag[basis_idx] = 1
        if n_nuisance > 0:
            A_sys = np.vstack(
                [
                    X_tilde[basis_idx, :].T,
                    sigma_tilde_diag[basis_idx].reshape(1, -1),
                ]
            )
        else:
            A_sys = sigma_tilde_diag[basis_idx].reshape(1, -1)

        b_sys = np.zeros(n_eq)
        b_sys[-1] = 1.0  # normalization

        try:
            gamma_basic = np.linalg.solve(A_sys, b_sys)
        except np.linalg.LinAlgError:
            n_linalg_error += 1
            continue

        # Check feasibility: gamma >= 0
        if np.all(gamma_basic >= -1e-10):
            gamma = np.zeros(n_moments)
            gamma[basis_idx] = np.maximum(gamma_basic, 0)
            vertices.append(gamma)
        else:
            n_infeasible += 1

    # Diagnostic warnings — surface vertex-search pathologies that would
    # otherwise hide behind `_compute_arp_test` returning False
    # (conservative non-rejection).
    if n_total > 0 and len(vertices) == 0:
        warnings.warn(
            f"ARP vertex enumeration exhausted without feasible vertices: "
            f"tried {n_total} bases, {n_linalg_error} rejected for "
            f"LinAlgError, {n_infeasible} infeasible (negative basic "
            f"variables). The caller (_compute_arp_test) will return False "
            f"(conservative non-rejection). This may indicate near-singular "
            f"nuisance constraints (X_tilde) or a degenerate "
            f"moment-inequality system.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif n_total > 0 and n_linalg_error / n_total >= 0.5:
        warnings.warn(
            f"ARP vertex enumeration heavily constrained: "
            f"{n_linalg_error} of {n_total} bases ({100 * n_linalg_error / n_total:.0f}%) "
            f"rejected for LinAlgError, {n_infeasible} infeasible. "
            f"{len(vertices)} feasible vertex(es) recovered. Results may be "
            f"numerically fragile; consider regularizing the moment-inequality "
            f"system or reviewing the nuisance constraints (X_tilde).",
            RuntimeWarning,
            stacklevel=3,
        )

    return vertices


def _compute_arp_test(
    Y_tilde: np.ndarray,
    X_tilde: np.ndarray,
    Sigma_tilde: np.ndarray,
    alpha: float,
    kappa: Optional[float] = None,
) -> bool:
    """
    Run the ARP conditional-LF hybrid test.

    Tests H0 using the ARP framework from Rambachan & Roth (2023)
    Sections 3.2.1-3.2.2.

    Parameters
    ----------
    Y_tilde : np.ndarray
        Transformed statistic.
    X_tilde : np.ndarray
        Nuisance matrix.
    Sigma_tilde : np.ndarray
        Transformed covariance.
    alpha : float
        Significance level.
    kappa : float, optional
        First-stage LF test size. Default: alpha / 10.

    Returns
    -------
    reject : bool
        True if H0 is rejected.
    """
    from scipy.stats import truncnorm

    if kappa is None:
        kappa = alpha / 10.0

    n_moments = len(Y_tilde)
    sigma_tilde_diag = np.sqrt(np.maximum(np.diag(Sigma_tilde), 0))

    # Avoid division by zero
    if np.any(sigma_tilde_diag <= 0):
        return False

    # Enumerate vertices of the dual feasible set
    vertices = _enumerate_vertices(X_tilde, sigma_tilde_diag, n_moments)

    if not vertices:
        # Cannot enumerate vertices; fall back to conservative non-rejection
        return False

    # Compute eta_hat = max_{gamma in vertices} gamma' @ Y_tilde
    eta_values = [gamma @ Y_tilde for gamma in vertices]
    eta_hat = max(eta_values)
    opt_idx = np.argmax(eta_values)
    gamma_star = vertices[opt_idx]

    # Stage 1: LF test (size kappa)
    # c_LF = 1-kappa quantile of max_{gamma in V} gamma' @ xi, xi ~ N(0, Sigma_tilde)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    n_sim = 5000
    L = np.linalg.cholesky(Sigma_tilde + 1e-12 * np.eye(n_moments))
    max_draws = np.zeros(n_sim)
    for i in range(n_sim):
        xi = L @ rng.standard_normal(n_moments)
        max_draws[i] = max(gamma @ xi for gamma in vertices)
    c_LF = np.quantile(max_draws, 1 - kappa)

    if eta_hat > c_LF:
        return True  # Reject via LF test

    # Stage 2: Conditional test (size (alpha - kappa) / (1 - kappa))
    alpha_cond = (alpha - kappa) / (1 - kappa)

    # Compute conditional variance and truncation bounds
    gamma_var = gamma_star @ Sigma_tilde @ gamma_star
    if gamma_var <= 0:
        return False

    sigma_gamma = np.sqrt(gamma_var)

    # Truncation bounds: v_lo is the next-best vertex value
    other_eta = [ev for j, ev in enumerate(eta_values) if j != opt_idx]
    v_lo = max(other_eta) if other_eta else -np.inf

    # v_up for hybrid: min(v_up_cond, c_LF)
    v_up = c_LF  # Upper truncation from first stage non-rejection

    if v_lo >= v_up:
        # Degenerate truncation interval
        return False

    # Truncated normal critical value
    # Under H0, the worst case is mu = 0 (least favorable)
    a = (v_lo - 0) / sigma_gamma
    b = (v_up - 0) / sigma_gamma

    try:
        c_cond = truncnorm.ppf(1 - alpha_cond, a, b, loc=0, scale=sigma_gamma)
    except (ValueError, RuntimeError):
        return False

    return eta_hat > max(0, c_cond)


def _arp_confidence_set(
    beta_hat: np.ndarray,
    sigma_hat: np.ndarray,
    A: np.ndarray,
    d: np.ndarray,
    l: np.ndarray,
    num_pre: int,
    alpha: float = 0.05,
    kappa: Optional[float] = None,
    n_grid: int = 200,
) -> Tuple[float, float]:
    """
    Compute ARP hybrid confidence set by test inversion.

    Per Rambachan & Roth (2023), the confidence set is:
        C = {theta_bar : ARP hybrid test does not reject H0: theta = theta_bar}

    Parameters
    ----------
    beta_hat : np.ndarray
        Full event-study coefficient vector [pre, post].
    sigma_hat : np.ndarray
        Full covariance matrix.
    A : np.ndarray
        Polyhedral constraint matrix (for Delta).
    d : np.ndarray
        Polyhedral constraint vector.
    l : np.ndarray
        Target parameter weights.
    num_pre : int
        Number of pre-periods.
    alpha : float
        Significance level.
    kappa : float, optional
        Hybrid test first-stage size.
    n_grid : int
        Number of grid points for test inversion.

    Returns
    -------
    ci_lb : float
        Lower bound of confidence set.
    ci_ub : float
        Upper bound of confidence set.
    """
    beta_post = beta_hat[num_pre:]

    # Point estimate and SE for grid centering
    theta_hat = l @ beta_post
    se = np.sqrt(l @ sigma_hat[num_pre:, num_pre:] @ l)

    # Grid centered on point estimate
    grid_half = max(5 * se, 1.0)
    theta_grid = np.linspace(theta_hat - grid_half, theta_hat + grid_half, n_grid)

    # Test inversion: find theta_bar values not rejected
    accepted = []
    for theta_bar in theta_grid:
        Y_tilde, X_tilde, Sigma_tilde = _setup_moment_inequalities(
            beta_hat, sigma_hat, A, d, l, theta_bar, num_pre
        )
        reject = _compute_arp_test(Y_tilde, X_tilde, Sigma_tilde, alpha, kappa)
        if not reject:
            accepted.append(theta_bar)

    if not accepted:
        # Everything rejected — empty confidence set (unusual)
        return theta_hat, theta_hat

    ci_lb = min(accepted)
    ci_ub = max(accepted)

    # Refine boundaries with bisection
    for _ in range(15):
        # Refine lower bound
        mid = (ci_lb - grid_half / n_grid + ci_lb) / 2 if ci_lb > theta_grid[0] else ci_lb
        if mid < ci_lb:
            Y_tilde, X_tilde, Sigma_tilde = _setup_moment_inequalities(
                beta_hat, sigma_hat, A, d, l, mid, num_pre
            )
            if not _compute_arp_test(Y_tilde, X_tilde, Sigma_tilde, alpha, kappa):
                ci_lb = mid

        # Refine upper bound
        mid = (ci_ub + grid_half / n_grid + ci_ub) / 2 if ci_ub < theta_grid[-1] else ci_ub
        if mid > ci_ub:
            Y_tilde, X_tilde, Sigma_tilde = _setup_moment_inequalities(
                beta_hat, sigma_hat, A, d, l, mid, num_pre
            )
            if not _compute_arp_test(Y_tilde, X_tilde, Sigma_tilde, alpha, kappa):
                ci_ub = mid

    return ci_lb, ci_ub


# =============================================================================
# Main Class
# =============================================================================


class HonestDiD:
    """
    Honest DiD sensitivity analysis (Rambachan & Roth 2023).

    Computes robust inference for difference-in-differences allowing
    for bounded violations of parallel trends.

    Parameters
    ----------
    method : {"smoothness", "relative_magnitude", "combined"}
        Type of restriction on trend violations:
        - "smoothness": Bounds on second differences of trend violations (Delta^SD)
        - "relative_magnitude": Post first differences <= M * max pre first difference (Delta^RM)
        - "combined": Both restrictions (Delta^SDRM)
    M : float, optional
        Restriction parameter. Interpretation depends on method:
        - smoothness: Max second difference
        - relative_magnitude: Scaling factor for max pre-period first difference
        Default is 1.0 for relative_magnitude, 0.0 for smoothness.
    alpha : float
        Significance level for confidence intervals.
    l_vec : array-like or None
        Weighting vector for scalar parameter (length = num_post_periods).
        If None, uses uniform weights (average effect).

    Examples
    --------
    >>> from diff_diff import MultiPeriodDiD
    >>> from diff_diff.honest_did import HonestDiD
    >>>
    >>> # Fit event study
    >>> mp_did = MultiPeriodDiD()
    >>> results = mp_did.fit(data, outcome='y', treatment='treated',
    ...                      time='period', post_periods=[4,5,6,7])
    >>>
    >>> # Sensitivity analysis with relative magnitudes
    >>> honest = HonestDiD(method='relative_magnitude', M=1.0)
    >>> bounds = honest.fit(results)
    >>> print(bounds.summary())
    >>>
    >>> # Sensitivity curve over M values
    >>> sensitivity = honest.sensitivity_analysis(results, M_grid=[0, 0.5, 1, 1.5, 2])
    >>> sensitivity.plot()
    """

    def __init__(
        self,
        method: Literal["smoothness", "relative_magnitude", "combined"] = "relative_magnitude",
        M: Optional[float] = None,
        alpha: float = 0.05,
        l_vec: Optional[np.ndarray] = None,
    ):
        self.method = method
        self.alpha = alpha
        self.l_vec = l_vec

        # Set default M based on method
        if M is None:
            self.M = 1.0 if method == "relative_magnitude" else 0.0
        else:
            self.M = M

        self._validate_params()

    def _validate_params(self):
        """Validate initialization parameters."""
        if self.method not in ["smoothness", "relative_magnitude", "combined"]:
            raise ValueError(
                f"method must be 'smoothness', 'relative_magnitude', or 'combined', "
                f"got method='{self.method}'"
            )
        if self.M < 0:
            raise ValueError(f"M must be non-negative, got M={self.M}")
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got alpha={self.alpha}")

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "method": self.method,
            "M": self.M,
            "alpha": self.alpha,
            "l_vec": self.l_vec,
        }

    def set_params(self, **params) -> "HonestDiD":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        self._validate_params()
        return self

    def fit(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        M: Optional[float] = None,
    ) -> HonestDiDResults:
        """
        Compute bounds and robust confidence intervals.

        Parameters
        ----------
        results : MultiPeriodDiDResults, CallawaySantAnnaResults, or ChaisemartinDHaultfoeuilleResults
            Results from event study estimation.
        M : float, optional
            Override the M parameter for this fit.

        Returns
        -------
        HonestDiDResults
            Results containing bounds and robust confidence intervals.
        """
        M = M if M is not None else self.M
        # The fit()-time M override bypasses constructor validation; re-check here
        # (a negative M would be silently treated as +|M| via cv_alpha's abs()).
        if M < 0:
            raise ValueError(f"M must be non-negative, got M={M}")

        # Extract event study parameters
        beta_hat, sigma, num_pre, num_post, pre_periods, post_periods, df_survey = (
            _extract_event_study_params(results)
        )

        # beta_hat contains [pre-period effects, post-period effects] in order.
        # Extract pre and post components for the identified set LP.
        # The LP pins delta_pre = beta_pre (Rambachan & Roth Eqs 5-6).
        if len(beta_hat) == num_pre + num_post:
            beta_pre = beta_hat[:num_pre]
            beta_post = beta_hat[num_pre:]
        elif len(beta_hat) == num_post:
            beta_pre = np.zeros(num_pre)
            beta_post = beta_hat
        else:
            beta_pre = np.zeros(num_pre)
            beta_post = beta_hat
            num_post = len(beta_hat)

        # Handle sigma extraction for post periods
        if sigma.shape[0] == num_post and sigma.shape[0] == len(beta_post):
            sigma_post = sigma
        elif sigma.shape[0] == num_pre + num_post:
            sigma_post = sigma[num_pre:, num_pre:]
        else:
            sigma_post = sigma[: len(beta_post), : len(beta_post)]

        # Update num_post to match actual data
        num_post = len(beta_post)

        if num_post == 0:
            raise ValueError(
                "No post-period effects with finite estimates found. "
                "HonestDiD requires at least one identified post-period "
                "coefficient to compute bounds."
            )

        # Set up weighting vector and target label
        if self.l_vec is None:
            l_vec = np.ones(num_post) / num_post  # Uniform weights
            target_label = "Equal-weight avg over post horizons"
        else:
            l_vec = np.asarray(self.l_vec)
            if len(l_vec) != num_post:
                raise ValueError(f"l_vec must have length {num_post}, got {len(l_vec)}")
            # Detect common patterns for a human-readable label
            basis = np.zeros(num_post)
            basis[0] = 1.0
            if np.allclose(l_vec, basis):
                target_label = "First post-treatment effect (on-impact)"
            elif np.allclose(l_vec, np.ones(num_post) / num_post):
                target_label = "Equal-weight avg over post horizons"
            else:
                target_label = f"Custom l_vec ({l_vec.tolist()})"

        # Compute original estimate and SE
        original_estimate = np.dot(l_vec, beta_post)
        original_se = np.sqrt(l_vec @ sigma_post @ l_vec)

        # Compute bounds based on method
        if self.method == "smoothness":
            lb, ub, ci_lb, ci_ub = self._compute_smoothness_bounds(
                beta_pre,
                beta_post,
                sigma,
                sigma_post,
                l_vec,
                num_pre,
                num_post,
                M,
                df=df_survey,
            )
            ci_method = "FLCI"

        elif self.method == "relative_magnitude":
            lb, ub, ci_lb, ci_ub = self._compute_rm_bounds(
                beta_pre,
                beta_post,
                sigma,
                sigma_post,
                l_vec,
                num_pre,
                num_post,
                M,
                pre_periods,
                results,
                df=df_survey,
            )
            ci_method = "FLCI"

        else:  # combined
            lb, ub, ci_lb, ci_ub = self._compute_combined_bounds(
                beta_pre,
                beta_post,
                sigma,
                sigma_post,
                l_vec,
                num_pre,
                num_post,
                M,
                pre_periods,
                results,
                df=df_survey,
            )
            ci_method = "FLCI"

        # Extract survey_metadata for storage on results
        survey_metadata = getattr(results, "survey_metadata", None)

        return HonestDiDResults(
            lb=lb,
            ub=ub,
            ci_lb=ci_lb,
            ci_ub=ci_ub,
            M=M,
            method=self.method,
            original_estimate=original_estimate,
            original_se=original_se,
            alpha=self.alpha,
            ci_method=ci_method,
            target_label=target_label,
            pre_periods_used=list(pre_periods),
            post_periods_used=list(post_periods),
            original_results=results,
            survey_metadata=survey_metadata,
            df_survey=df_survey,
        )

    def _compute_smoothness_bounds(
        self,
        beta_pre: np.ndarray,
        beta_post: np.ndarray,
        sigma_full: np.ndarray,
        sigma_post: np.ndarray,
        l_vec: np.ndarray,
        num_pre: int,
        num_post: int,
        M: float,
        df: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        """Compute bounds under smoothness restriction (Delta^SD).

        Uses the optimal FLCI from Rambachan & Roth (2023) Section 4.1,
        which jointly optimizes the affine estimator direction to minimize
        CI width. Falls back to naive FLCI if the full covariance matrix
        is not available.
        """
        # Construct constraints
        A_ineq, b_ineq = _construct_constraints_sd(num_pre, num_post, M)

        # Solve for the identified set bounds with delta_pre = beta_pre pinned.
        # When the observed pre-trend's own curvature exceeds M this LP is
        # infeasible and the ESTIMATED identified set is empty (lb/ub = NaN) - the
        # point estimate rejects Delta^SD(M). That does NOT invalidate the FLCI: the
        # optimal FLCI is an affine estimator whose worst-case bias is taken over
        # delta in Delta^SD(M) treating beta as random, so it is well-defined given
        # (sigma, M) regardless of whether the realized beta_pre lies in Delta. R's
        # HonestDiD::createSensitivityResults returns the FLCI in exactly this case,
        # so we compute and return it (leaving lb/ub = NaN to flag the empty
        # estimated id-set) rather than NaN-propagating the whole result.
        lb, ub = _solve_bounds_lp(beta_pre, beta_post, l_vec, A_ineq, b_ineq, num_pre)

        # Compute optimal FLCI (Rambachan & Roth Section 4.1) - independent of the
        # identified-set LP above.
        if sigma_full.shape[0] == num_pre + num_post:
            ci_lb, ci_ub = _compute_optimal_flci(
                beta_pre,
                beta_post,
                sigma_full,
                l_vec,
                num_pre,
                num_post,
                M,
                self.alpha,
                df=df,
            )
        else:
            # The naive fallback FLCI extends the identified set by z*se, so it
            # genuinely needs finite id-set bounds; when the LP was infeasible the
            # naive CI is undefined (this branch is only reachable without the full
            # covariance matrix).
            if np.isnan(lb) or np.isnan(ub):
                return lb, ub, np.nan, np.nan
            se = np.sqrt(l_vec @ sigma_post @ l_vec)
            ci_lb, ci_ub = _compute_flci(lb, ub, se, self.alpha, df=df)

        return lb, ub, ci_lb, ci_ub

    def _compute_rm_bounds(
        self,
        beta_pre: np.ndarray,
        beta_post: np.ndarray,
        sigma_full: np.ndarray,
        sigma_post: np.ndarray,
        l_vec: np.ndarray,
        num_pre: int,
        num_post: int,
        Mbar: float,
        pre_periods: List,
        results: Any,
        df: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        """Compute bounds under relative magnitudes restriction (Delta^RM).

        Uses union-of-polyhedra decomposition per Lemma 2.2 of
        Rambachan & Roth (2023). Delta^RM constrains post-treatment
        first differences relative to the max pre-treatment first difference.

        CI construction uses naive FLCI (conservative). The paper recommends
        ARP hybrid confidence sets (Sections 3.2.1-3.2.2); infrastructure
        is implemented but disabled pending calibration of the moment
        inequality transformation.
        """
        # Solve identified set via union of polyhedra
        lb, ub = _solve_rm_bounds_union(beta_pre, beta_post, l_vec, num_pre, Mbar)

        # CI construction for Delta^RM.
        # The paper recommends ARP conditional/hybrid confidence sets
        # (Sections 3.2.1-3.2.2). The ARP infrastructure is implemented
        # (_arp_confidence_set) but the moment inequality transformation
        # requires further calibration to produce valid CIs consistently.
        # Currently uses conservative naive FLCI (extends identified set
        # by z*se); ARP will be enabled once calibrated.
        # TODO: enable ARP hybrid for RM once transformation is validated
        se = np.sqrt(l_vec @ sigma_post @ l_vec)
        if np.isfinite(lb) and np.isfinite(ub):
            ci_lb, ci_ub = _compute_flci(lb, ub, se, self.alpha, df=df)
        else:
            ci_lb, ci_ub = -np.inf, np.inf

        return lb, ub, ci_lb, ci_ub

    def _compute_combined_bounds(
        self,
        beta_pre: np.ndarray,
        beta_post: np.ndarray,
        sigma_full: np.ndarray,
        sigma_post: np.ndarray,
        l_vec: np.ndarray,
        num_pre: int,
        num_post: int,
        M: float,
        pre_periods: List,
        results: Any,
        df: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        """Compute bounds under combined smoothness + RM restriction."""
        import warnings

        warnings.warn(
            "HonestDiD method='combined' (Delta^SDRM) uses naive FLCI on the "
            "intersection of Delta^SD and Delta^RM bounds. The paper proves "
            "FLCI is NOT consistent for Delta^SDRM (Proposition 4.2). "
            "Consider using method='smoothness' or method='relative_magnitude' "
            "separately for paper-supported inference.",
            UserWarning,
            stacklevel=3,
        )
        # Get smoothness bounds
        lb_sd, ub_sd, _, _ = self._compute_smoothness_bounds(
            beta_pre,
            beta_post,
            sigma_full,
            sigma_post,
            l_vec,
            num_pre,
            num_post,
            M,
            df=df,
        )

        # Get RM bounds (use M as Mbar for combined)
        lb_rm, ub_rm, _, _ = self._compute_rm_bounds(
            beta_pre,
            beta_post,
            sigma_full,
            sigma_post,
            l_vec,
            num_pre,
            num_post,
            M,
            pre_periods,
            results,
            df=df,
        )

        # Combined bounds are intersection
        lb = max(lb_sd, lb_rm)
        ub = min(ub_sd, ub_rm)

        # If bounds cross, use the original estimate
        if lb > ub:
            theta = np.dot(l_vec, beta_post)
            lb = ub = theta

        # Compute FLCI on combined bounds
        se = np.sqrt(l_vec @ sigma_post @ l_vec)
        ci_lb, ci_ub = _compute_flci(lb, ub, se, self.alpha, df=df)

        return lb, ub, ci_lb, ci_ub

    def _estimate_max_pre_violation(self, results: Any, pre_periods: List) -> float:
        """
        Estimate the maximum pre-period violation.

        Uses pre-period coefficients if available, otherwise returns
        a default based on the overall SE.
        """
        if isinstance(results, MultiPeriodDiDResults):
            # Pre-period effects are now in period_effects directly
            # Filter out non-finite effects (e.g. from rank-deficient designs)
            pre_effects = [
                abs(results.period_effects[p].effect)
                for p in pre_periods
                if p in results.period_effects and np.isfinite(results.period_effects[p].effect)
            ]
            if pre_effects:
                return max(pre_effects)

            # Fallback: use avg_se as a scale
            return results.avg_se

        # For CallawaySantAnna, use pre-period event study effects
        try:
            from diff_diff.staggered import CallawaySantAnnaResults

            if isinstance(results, CallawaySantAnnaResults):
                if results.event_study_effects:
                    # Use the reference-aware pre_periods from _extract_event_study_params
                    pre_set = set(pre_periods) if pre_periods else set()
                    pre_effects = [
                        abs(results.event_study_effects[t]["effect"])
                        for t in results.event_study_effects
                        if t in pre_set and results.event_study_effects[t].get("n_groups", 1) > 0
                    ]
                    if pre_effects:
                        return max(pre_effects)
                # No valid pre-effects — should have been caught by
                # _extract_event_study_params pre-period validation
                return 0.0
        except ImportError:
            pass

        # Default fallback
        return 0.1

    def sensitivity_analysis(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        M_grid: Optional[List[float]] = None,
    ) -> SensitivityResults:
        """
        Perform sensitivity analysis over a grid of M values.

        Parameters
        ----------
        results : MultiPeriodDiDResults, CallawaySantAnnaResults, or ChaisemartinDHaultfoeuilleResults
            Results from event study estimation.
        M_grid : list of float, optional
            Grid of M values to evaluate. If None, uses default grid
            based on method.

        Returns
        -------
        SensitivityResults
            Results containing bounds and CIs for each M value.
        """
        if M_grid is None:
            if self.method == "relative_magnitude":
                M_grid = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            else:
                M_grid = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

        M_values = np.array(M_grid)
        if np.any(M_values < 0):
            raise ValueError(f"M must be non-negative; M_grid has a negative value: {M_grid}")
        bounds_list = []
        ci_list = []

        for M in M_values:
            result = self.fit(results, M=M)
            bounds_list.append((result.lb, result.ub))
            ci_list.append((result.ci_lb, result.ci_ub))

        # Find breakdown value
        breakdown_M = self._find_breakdown(results, M_values, ci_list)

        # Get original estimate info
        first_result = self.fit(results, M=0)

        return SensitivityResults(
            M_values=M_values,
            bounds=bounds_list,
            robust_cis=ci_list,
            breakdown_M=breakdown_M,
            method=self.method,
            original_estimate=first_result.original_estimate,
            original_se=first_result.original_se,
            alpha=self.alpha,
        )

    def _find_breakdown(
        self, results: Any, M_values: np.ndarray, ci_list: List[Tuple[float, float]]
    ) -> Optional[float]:
        """
        Find the breakdown value where CI first includes zero.

        Uses binary search for precision.
        """

        # Check if any CI includes zero (NaN CIs are treated as undefined, not significant)
        def _ci_includes_zero(ci_lb, ci_ub):
            if not (np.isfinite(ci_lb) and np.isfinite(ci_ub)):
                return True  # Undefined CIs are not "significant"
            return ci_lb <= 0 <= ci_ub

        includes_zero = [_ci_includes_zero(ci_lb, ci_ub) for ci_lb, ci_ub in ci_list]

        if not any(includes_zero):
            # Always significant - no breakdown
            return None

        if all(includes_zero):
            # Never significant - breakdown at 0
            return 0.0

        # Find first transition point
        for i, (inc, M) in enumerate(zip(includes_zero, M_values)):
            if inc and (i == 0 or not includes_zero[i - 1]):
                # Binary search between M_values[i-1] and M_values[i]
                if i == 0:
                    return 0.0

                lo, hi = M_values[i - 1], M_values[i]

                for _ in range(20):  # 20 iterations for precision
                    mid = (lo + hi) / 2
                    result = self.fit(results, M=mid)
                    if _ci_includes_zero(result.ci_lb, result.ci_ub):
                        hi = mid
                    else:
                        lo = mid

                return (lo + hi) / 2

        return None

    def breakdown_value(
        self, results: Union[MultiPeriodDiDResults, Any], tol: float = 0.01
    ) -> Optional[float]:
        """
        Find the breakdown value directly using binary search.

        The breakdown value is the smallest M where the robust
        confidence interval includes zero.

        Parameters
        ----------
        results : MultiPeriodDiDResults, CallawaySantAnnaResults, or ChaisemartinDHaultfoeuilleResults
            Results from event study estimation.
        tol : float
            Tolerance for binary search.

        Returns
        -------
        float or None
            Breakdown value, or None if effect is always significant.
        """

        def _ci_covers_zero(r):
            if not (np.isfinite(r.ci_lb) and np.isfinite(r.ci_ub)):
                return True  # Undefined CIs are not "significant"
            return r.ci_lb <= 0 <= r.ci_ub

        # Check at M=0
        result_0 = self.fit(results, M=0)
        if _ci_covers_zero(result_0):
            return 0.0

        # Check if significant even for large M
        result_large = self.fit(results, M=10)
        if not _ci_covers_zero(result_large):
            return None  # Always significant

        # Binary search
        lo, hi = 0.0, 10.0

        while hi - lo > tol:
            mid = (lo + hi) / 2
            result = self.fit(results, M=mid)
            if _ci_covers_zero(result):
                hi = mid
            else:
                lo = mid

        return (lo + hi) / 2


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_honest_did(
    results: Union[MultiPeriodDiDResults, Any],
    method: str = "relative_magnitude",
    M: float = 1.0,
    alpha: float = 0.05,
    l_vec: Optional[np.ndarray] = None,
) -> HonestDiDResults:
    """
    Convenience function for computing Honest DiD bounds.

    Parameters
    ----------
    results : MultiPeriodDiDResults, CallawaySantAnnaResults, or ChaisemartinDHaultfoeuilleResults
        Results from event study estimation.
    method : str
        Type of restriction ("smoothness", "relative_magnitude", "combined").
    M : float
        Restriction parameter.
    alpha : float
        Significance level.
    l_vec : np.ndarray, optional
        Weight vector defining the scalar target ``theta = l_vec' tau``
        over post-treatment horizons. Length must equal the number of
        post-treatment periods. ``None`` (default) uses equal weights
        (uniform average). To target the on-impact effect only (R's
        default), pass ``np.array([1, 0, ..., 0])``.

    Returns
    -------
    HonestDiDResults
        Bounds and robust confidence intervals.

    Examples
    --------
    >>> bounds = compute_honest_did(event_study_results, method='relative_magnitude', M=1.0)
    >>> print(f"Robust CI: [{bounds.ci_lb:.3f}, {bounds.ci_ub:.3f}]")
    """
    honest = HonestDiD(method=method, M=M, alpha=alpha, l_vec=l_vec)
    return honest.fit(results)


def sensitivity_plot(
    results: Union[MultiPeriodDiDResults, Any],
    method: str = "relative_magnitude",
    M_grid: Optional[List[float]] = None,
    alpha: float = 0.05,
    ax=None,
    **kwargs,
):
    """
    Create a sensitivity analysis plot.

    Parameters
    ----------
    results : MultiPeriodDiDResults, CallawaySantAnnaResults, or ChaisemartinDHaultfoeuilleResults
        Results from event study estimation.
    method : str
        Type of restriction.
    M_grid : list of float, optional
        Grid of M values.
    alpha : float
        Significance level.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    **kwargs
        Additional arguments passed to plot method.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    honest = HonestDiD(method=method, alpha=alpha)
    sensitivity = honest.sensitivity_analysis(results, M_grid=M_grid)
    return sensitivity.plot(ax=ax, **kwargs)
