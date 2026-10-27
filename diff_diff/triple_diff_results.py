"""
Results container for the Triple Difference (DDD) estimator - both designs.

Since the 4.0 cut (item 6 of the cut sequence, ledger rows M-014 / M-140 /
M-141) ``TripleDifference.fit`` returns ONE results shape for both of its
designs: the 2x2x2 cross-sectional design (``group=``, ``partition=``,
``post=``) and the staggered-adoption panel design (``first_treat=``). The
container is the staggered one's structure with the 2x2x2 case as the
degenerate single-ATT view: the canonical quintet ``att`` / ``se`` /
``t_stat`` / ``p_value`` / ``conf_int`` is always populated, the group-time
table is empty in the 2x2x2 design, and ``design`` says which engine fit it.

Post-fit aggregation (spec section 6) lives here too: ``aggregate('simple')``
relays the stored headline in both designs; ``aggregate('event_study')`` and
``aggregate('group')`` recompute from the retained aggregation kit in the
staggered design (bootstrapped fits REPLAY the fit-time multiplier bootstrap);
``aggregate('simple', weights='event_time')`` is the paper's Equation 4.14
overall - the unweighted mean of the post-treatment event-study effects -
which through 3.x was the fit-time ``overall_att_es`` field.

The 3.x ``StaggeredTripleDiffResults`` class (this module's former content)
was removed at 4.0 with row M-014; this module was renamed from
``staggered_triple_diff_results.py`` so the shared staggered engine can import
the container without a cycle through ``triple_diff.py``.
"""

import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.aggregation import (
    AggregationMixin,
    AggregationResult,
    resolve_inference_df,
)
from diff_diff.bootstrap_chunking import effective_weight_backend
from diff_diff.bootstrap_utils import (
    apply_bootstrap_event_study_overrides,
    apply_bootstrap_group_overrides,
)
from diff_diff.results import _format_survey_block, _get_significance_stars
from diff_diff.results_base import (
    _SUMMARY_ALPHA_MESSAGE,
    BaseResults,
    _coverage_pct,
    _require_fit_alpha,
    build_event_study_surface,
)
from diff_diff.staggered_bootstrap import apply_cband_conf_ints
from diff_diff.staggered_results import _KitAggregator, _KitBootstrapAggregator
from diff_diff.utils import safe_inference

if TYPE_CHECKING:
    from diff_diff.staggered_bootstrap import CSBootstrapResults

_DESIGNS = ("2x2x2", "staggered")

#: The four 2x2x2 cell counts. Required in that design (the summary renderer
#: formats them), meaningless in the staggered design (``None`` there).
_CELL_COUNT_FIELDS = (
    "n_treated_eligible",
    "n_treated_ineligible",
    "n_control_eligible",
    "n_control_ineligible",
)

_TWO_BY_TWO_ONLY = (
    "the 2x2x2 TripleDifference design estimates a single ATT from the "
    "(group=, partition=, post=) triple and has no ATT(g, t) cells, event-study "
    "horizons or adoption cohorts to aggregate over; fit the staggered design "
    "(fit(..., unit=, time=, first_treat=, partition=)) for those surfaces."
)


@dataclass
class TripleDifferenceResults(BaseResults, AggregationMixin):
    """
    Results from Triple Difference (DDD) estimation, in either design.

    One container serves both designs of :class:`~diff_diff.TripleDifference`
    (Ortiz-Villavicencio & Sant'Anna 2025). ``design`` records which engine
    produced the fit; the canonical headline quintet is populated in both.

    Attributes
    ----------
    att : float
        Average treatment effect on the treated. 2x2x2: the effect on units in
        the treated group (G=1) and eligible partition (P=1) after treatment
        (T=1). Staggered: the cohort-size-weighted average of the
        post-treatment ``ATT(g, t)`` cells (the library-wide Callaway-Sant'Anna
        "simple" convention; the paper's Eq. 4.14 event-study average is
        ``aggregate('simple', weights='event_time')``).
    se : float
        Standard error of ``att``.
    t_stat : float
        T-statistic for ``att``.
    p_value : float
        P-value for the null hypothesis that ``att`` is zero.
    conf_int : tuple[float, float]
        Confidence interval for ``att`` at ``alpha``.
    n_obs : int
        Total number of observations used in estimation.
    n_treated_eligible, n_treated_ineligible, n_control_eligible, n_control_ineligible : int or None
        2x2x2 cell counts (treated/control group x eligible/ineligible
        partition). ``None`` in the staggered design.
    estimation_method : str
        ``"dr"`` (doubly robust), ``"reg"`` (regression adjustment) or
        ``"ipw"`` (inverse probability weighting).
    alpha : float
        Significance level used for confidence intervals.
    group_means, pscore_stats, r_squared, covariate_balance
        2x2x2 diagnostics (``None`` in the staggered design).
    inference_method : str
        ``"analytical"`` or ``"bootstrap"`` (staggered design with
        ``n_bootstrap > 0``).
    vcov_type : str or None
        2x2x2 sandwich family (``"hc1"`` ...). ``None`` in the staggered
        design, whose SEs come from the GMM-combined influence function or
        the multiplier bootstrap.
    cluster_name, n_clusters : optional
        2x2x2 ``cluster=`` metadata.
    n_bootstrap : int or None
        Bootstrap replications when ``inference_method == "bootstrap"``.
    survey_metadata : SurveyMetadata or None
        Survey design metadata when fitted with ``survey_design=``.
    epv_diagnostics : dict or None
        Events-per-variable diagnostics: keyed by subgroup comparison in the
        2x2x2 design, by ``(g, t)`` cell in the staggered design.
    design : str
        ``"2x2x2"`` or ``"staggered"``.
    df_inference : float or None
        Degrees of freedom used for t-distribution inference on ``att`` when
        a t-reference governed it and no survey design carries it: the 2x2x2
        engine's ``n_obs - 8 - len(covariates)`` on non-survey fits. ``None``
        when inference used the normal reference (the staggered design's
        non-survey fits) or when the governing df lives on
        ``survey_metadata.df_survey``.
    group_time_effects : dict
        Staggered design: ``(group, time)`` -> effect record. Empty in the
        2x2x2 design.
    groups : list
        Staggered design: enabling cohorts that produced at least one cell.
    time_periods : list
        Staggered design: calendar periods.
    n_treated_units, n_control_units, n_never_enabled, n_eligible, n_ineligible : int or None
        Staggered unit counts (``None`` in the 2x2x2 design).
    control_group, base_period, anticipation : optional
        Staggered engine configuration (``None`` in the 2x2x2 design).
    pscore_trim : float
        Propensity-score trimming bound (both engines).
    event_study_effects, group_effects : dict or None
        ``None`` on every 4.0 fit - the event-study and cohort surfaces are
        produced post-fit by :meth:`aggregate`. Retained as carrier fields for
        the aggregation machinery and for 3.x pickles.
    bootstrap_results : CSBootstrapResults or None
        Staggered multiplier-bootstrap payload when ``n_bootstrap > 0``.
    cband_crit_value : float or None
        ``None`` on 4.0 fits (simultaneous bands are produced by
        ``aggregate('event_study')``).
    comparison_group_counts, gmm_weights : dict or None
        Staggered per-cell comparison-cohort counts and GMM weights.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    n_treated_eligible: Optional[int] = None
    n_treated_ineligible: Optional[int] = None
    n_control_eligible: Optional[int] = None
    n_control_ineligible: Optional[int] = None
    estimation_method: str = "dr"
    alpha: float = 0.05
    # Group means for diagnostics (2x2x2)
    group_means: Optional[Dict[str, float]] = field(default=None)
    # Propensity score diagnostics (2x2x2 IPW/DR)
    pscore_stats: Optional[Dict[str, float]] = field(default=None)
    # Regression diagnostics (2x2x2)
    r_squared: Optional[float] = field(default=None)
    # Covariate balance statistics (2x2x2)
    covariate_balance: Optional[pd.DataFrame] = field(default=None, repr=False)
    # Inference details
    inference_method: str = field(default="analytical")
    vcov_type: Optional[str] = field(default="hc1")
    cluster_name: Optional[str] = field(default=None)
    n_bootstrap: Optional[int] = field(default=None)
    n_clusters: Optional[int] = field(default=None)
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)
    # EPV diagnostics: per subgroup comparison (2x2x2) or per (g, t) cell (staggered)
    epv_diagnostics: Optional[Dict[Any, Dict[str, Any]]] = field(default=None, repr=False)
    epv_threshold: float = 10
    pscore_fallback: str = "error"
    # ---- 4.0 unified-container fields (item 6; rows M-014 / M-140 / M-141). ----
    # Everything below is appended with a default so the 2x2x2 positional
    # layout above is byte-stable.
    design: str = "2x2x2"
    df_inference: Optional[float] = None
    group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]] = field(default_factory=dict)
    groups: List[Any] = field(default_factory=list)
    time_periods: List[Any] = field(default_factory=list)
    n_treated_units: Optional[int] = None
    n_control_units: Optional[int] = None
    n_never_enabled: Optional[int] = None
    n_eligible: Optional[int] = None
    n_ineligible: Optional[int] = None
    control_group: Optional[str] = None
    base_period: Optional[str] = None
    anticipation: Optional[int] = None
    pscore_trim: float = 0.01
    # Carrier fields for the post-fit aggregation machinery (the CS pattern):
    # None on every 4.0 fit, populated only on the throwaway copy that
    # ``build_event_study_surface`` reads, and on 3.x pickles.
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None)
    group_effects: Optional[Dict[Any, Dict[str, Any]]] = field(default=None)
    bootstrap_results: Optional["CSBootstrapResults"] = field(default=None, repr=False)
    cband_crit_value: Optional[float] = None
    comparison_group_counts: Optional[Dict[Tuple, int]] = field(default=None, repr=False)
    gmm_weights: Optional[Dict[Tuple, Dict]] = field(default=None, repr=False)
    event_study_vcov: Optional["np.ndarray"] = field(default=None, repr=False)
    event_study_vcov_index: Optional[list] = field(default=None, repr=False)
    event_study_df: Optional[float] = field(default=None, repr=False)
    reference_event_times: Optional[Tuple[Any, ...]] = None
    # Post-fit re-aggregation payload (spec section 6), attached by the
    # staggered engine because nothing it needs survives fit(). Excluded from
    # repr and equality: internal bookkeeping whose arrays would make `==` raise.
    _aggregation_kit: Optional[Any] = field(default=None, repr=False, compare=False)

    #: ``simple`` relays the stored headline in both designs; ``event_study``
    #: and ``group`` recompute from the kit in the staggered design. No
    #: ``total`` - the DDD estimators never offered one.
    _AGGREGATE_SUPPORTED: ClassVar[Tuple[str, ...]] = ("simple", "event_study", "group")
    #: ``balance_e`` is meaningful on ``event_study`` and on
    #: ``simple`` ONLY together with ``weights="event_time"`` (the Eq. 4.14
    #: overall averages the balanced event-study horizons); plain ``simple``
    #: rejects it in ``_aggregate_compute``.
    _AGGREGATE_BALANCE_E_TYPES: ClassVar[Tuple[str, ...]] = ("event_study", "simple")

    def __post_init__(self) -> None:
        if self.design not in _DESIGNS:
            raise ValueError(f"design must be one of {_DESIGNS}, got {self.design!r}")
        if self.design == "staggered":
            if not self.group_time_effects or not self.groups or not self.time_periods:
                raise ValueError(
                    "design='staggered' requires non-empty group_time_effects, groups and "
                    "time_periods (the staggered engine never constructs without them)"
                )
        else:
            missing = [name for name in _CELL_COUNT_FIELDS if getattr(self, name) is None]
            if missing:
                raise ValueError(
                    f"design='2x2x2' requires the four cell counts; missing: {missing}"
                )

    # ------------------------------------------------------------------ #
    # Pickle migration
    # ------------------------------------------------------------------ #

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Load 3.x ``TripleDifferenceResults`` pickles (2x2x2 only).

        The 3.x 2x2x2 container already stored the canonical quintet, so no
        key is renamed; the 4.0 unified-container fields are backfilled with
        their defaults when absent (spec section 5 pickle-migration rule, the
        ``SyntheticDiDResults.__setstate__`` precedent). 3.x
        ``StaggeredTripleDiffResults`` pickles cannot load: the class was
        removed at 4.0 (row M-014).
        """
        self.__dict__.update(state)
        defaults: Dict[str, Any] = {
            "design": "2x2x2",
            "df_inference": None,
            "n_treated_units": None,
            "n_control_units": None,
            "n_never_enabled": None,
            "n_eligible": None,
            "n_ineligible": None,
            "control_group": None,
            "base_period": None,
            "anticipation": None,
            "pscore_trim": 0.01,
            "event_study_effects": None,
            "group_effects": None,
            "bootstrap_results": None,
            "cband_crit_value": None,
            "comparison_group_counts": None,
            "gmm_weights": None,
            "event_study_vcov": None,
            "event_study_vcov_index": None,
            "event_study_df": None,
            "reference_event_times": None,
            "_aggregation_kit": None,
        }
        for key, value in defaults.items():
            if key not in self.__dict__:
                self.__dict__[key] = value
        for key, factory in (
            ("group_time_effects", dict),
            ("groups", list),
            ("time_periods", list),
        ):
            if key not in self.__dict__:
                self.__dict__[key] = factory()

    # ------------------------------------------------------------------ #
    # Scalar views
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        """Concise string representation."""
        if self.design == "staggered":
            return (
                f"TripleDifferenceResults(design='staggered', "
                f"ATT={self.att:.4f}{self.significance_stars}, SE={self.se:.4f}, "
                f"n_groups={len(self.groups)}, n_periods={len(self.time_periods)})"
            )
        return (
            f"TripleDifferenceResults(ATT={self.att:.4f}{self.significance_stars}, "
            f"SE={self.se:.4f}, p={self.p_value:.4f}, method={self.estimation_method})"
        )

    @property
    def is_significant(self) -> bool:
        """Check if the ATT is statistically significant at the alpha level."""
        return bool(self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)

    @property
    def coef_var(self) -> float:
        """Coefficient of variation: SE / abs(ATT). NaN when ATT is 0 or SE non-finite."""
        if not (np.isfinite(self.se) and self.se >= 0):
            return np.nan
        if not np.isfinite(self.att) or self.att == 0:
            return np.nan
        return self.se / abs(self.att)

    # ------------------------------------------------------------------ #
    # summary()
    # ------------------------------------------------------------------ #

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of the estimation results.

        Parameters
        ----------
        alpha : float, optional
            Accepted for signature uniformity. The stored intervals were
            computed at fit time; a value different from the stored
            ``alpha`` raises ValueError rather than silently recomputing
            or relabeling (bootstrap percentile intervals cannot be
            reconstructed from the reported SE). Re-fit at the desired
            alpha instead.

        Returns
        -------
        str
            Formatted summary table.
        """
        if self.design == "staggered":
            return self._summary_staggered(alpha)
        return self._summary_2x2x2(alpha)

    def _summary_2x2x2(self, alpha: Optional[float]) -> str:
        alpha = _require_fit_alpha(alpha, self.alpha, message=_SUMMARY_ALPHA_MESSAGE)
        conf_level = _coverage_pct(alpha)

        lines = [
            "=" * 75,
            "Triple Difference (DDD) Estimation Results".center(75),
            "=" * 75,
            "",
            f"{'Estimation method:':<30} {self.estimation_method:>15}",
            f"{'Total observations:':<30} {self.n_obs:>15}",
            "",
            "Sample Composition by Cell:",
            f"  {'Treated group, Eligible:':<28} {self.n_treated_eligible:>15}",
            f"  {'Treated group, Ineligible:':<28} {self.n_treated_ineligible:>15}",
            f"  {'Control group, Eligible:':<28} {self.n_control_eligible:>15}",
            f"  {'Control group, Ineligible:':<28} {self.n_control_ineligible:>15}",
        ]

        if self.r_squared is not None:
            lines.append(f"{'R-squared:':<30} {self.r_squared:>15.4f}")

        # Add survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 75))

        if self.inference_method != "analytical":
            lines.append(f"{'Inference method:':<30} {self.inference_method:>15}")
            if self.n_bootstrap is not None:
                lines.append(f"{'Bootstrap replications:':<30} {self.n_bootstrap:>15}")
        # Variance-estimator line. Suppressed under survey designs (the survey
        # block above already names the design + n_psu + df; the analytical
        # SE is TSL on the combined IF, not the raw hc1 sandwich). For bare
        # cluster= fits the actual algebra is CR1 Liang-Zeger on the combined
        # IF, so route through the shared _format_vcov_label to render a
        # cluster-aware label rather than raw "hc1".
        if self.survey_metadata is None and self.vcov_type is not None:
            from diff_diff.results import _format_vcov_label

            vcov_label = _format_vcov_label(
                self.vcov_type,
                cluster_name=self.cluster_name,
                n_clusters=self.n_clusters,
                n_obs=self.n_obs,
            )
            if vcov_label:
                lines.append(f"{'Variance estimator:':<30} {vcov_label:>15}")
        if self.n_clusters is not None:
            lines.append(f"{'Number of clusters:':<30} {self.n_clusters:>15}")

        lines.extend(
            [
                "",
                "-" * 75,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'':>5}",
                "-" * 75,
                f"{'ATT':<15} {self.att:>12.4f} {self.se:>12.4f} {self.t_stat:>10.3f} {self.p_value:>10.4f} {self.significance_stars:>5}",
                "-" * 75,
                "",
                f"{conf_level}% Confidence Interval: [{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]",
            ]
        )

        # EPV diagnostics block (if any subgroup has low EPV)
        if self.epv_diagnostics:
            low_epv = {k: v for k, v in self.epv_diagnostics.items() if v.get("is_low")}
            if low_epv:
                n_affected = len(low_epv)
                n_total = len(self.epv_diagnostics)
                min_entry = min(low_epv.values(), key=lambda v: v["epv"])
                lines.extend(
                    [
                        "",
                        "-" * 75,
                        "EPV Diagnostics".center(75),
                        "-" * 75,
                        f"WARNING: Low Events Per Variable (EPV) in "
                        f"{n_affected} of {n_total} subgroup comparison(s).",
                        f"Minimum EPV: {min_entry['epv']:.1f}. "
                        f"Threshold: {self.epv_threshold:.0f}.",
                        "Consider: estimation_method='reg' or fewer covariates.",
                        "Call results.epv_summary() for details.",
                        "-" * 75,
                    ]
                )

        # Show group means if available
        if self.group_means:
            lines.extend(
                [
                    "",
                    "-" * 75,
                    "Cell Means (Y):",
                    "-" * 75,
                ]
            )
            for cell, mean in self.group_means.items():
                lines.append(f"  {cell:<35} {mean:>12.4f}")

        # Show propensity score diagnostics if available
        if self.pscore_stats:
            lines.extend(
                [
                    "",
                    "-" * 75,
                    "Propensity Score Diagnostics:",
                    "-" * 75,
                ]
            )
            for stat, value in self.pscore_stats.items():
                lines.append(f"  {stat:<35} {value:>12.4f}")

        lines.extend(
            [
                "",
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                "=" * 75,
            ]
        )

        return "\n".join(lines)

    def _summary_staggered(self, alpha: Optional[float]) -> str:
        alpha = _require_fit_alpha(alpha, self.alpha)
        conf_level = _coverage_pct(alpha)

        lines = [
            "=" * 85,
            "Staggered Triple Difference (DDD) Results".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated units (S<inf, Q=1):':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Never-enabled units:':<30} {self.n_never_enabled:>10}",
            f"{'Eligible units (Q=1):':<30} {self.n_eligible:>10}",
            f"{'Ineligible units (Q=0):':<30} {self.n_ineligible:>10}",
            f"{'Enabling cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            f"{'Estimation method:':<30} {self.estimation_method:>10}",
            f"{'Control group:':<30} {str(self.control_group):>10}",
            f"{'Base period:':<30} {str(self.base_period):>10}",
            "",
        ]

        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 85))

        if self.inference_method != "analytical":
            lines.append(f"{'Inference method:':<30} {self.inference_method:>10}")
            if self.n_bootstrap is not None:
                lines.append(f"{'Bootstrap replications:':<30} {self.n_bootstrap:>10}")
            lines.append("")

        # Overall ATT
        lines.extend(
            [
                "-" * 85,
                "Overall Average Treatment Effect on the Treated".center(85),
                "-" * 85,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
                f"{'ATT':<15} {self.att:>12.4f} {self.se:>12.4f} "
                f"{self.t_stat:>10.3f} {self.p_value:>10.4f} "
                f"{_get_significance_stars(self.p_value):>6}",
                "-" * 85,
                "",
                f"{conf_level}% Confidence Interval: "
                f"[{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]",
            ]
        )

        cv = self.coef_var
        if np.isfinite(cv):
            lines.append(f"{'CV (SE/abs(ATT)):':<25} {cv:>10.4f}")

        lines.append("")

        # EPV diagnostics block (if any cohort has low EPV)
        if self.epv_diagnostics:
            low_epv = {k: v for k, v in self.epv_diagnostics.items() if v.get("is_low")}
            if low_epv:
                n_affected = len(low_epv)
                n_total = len(self.epv_diagnostics)
                min_entry = min(low_epv.values(), key=lambda v: v["epv"])
                min_g = min(low_epv.keys(), key=lambda k: low_epv[k]["epv"])
                lines.extend(
                    [
                        "-" * 85,
                        "Propensity Score Diagnostics".center(85),
                        "-" * 85,
                        f"WARNING: Low Events Per Variable (EPV) in "
                        f"{n_affected} of {n_total} cohort-time cell(s).",
                        f"Minimum EPV: {min_entry['epv']:.1f} "
                        f"(cohort g={min_g[0]}). "
                        f"Threshold: {self.epv_threshold:.0f}.",
                        "Consider: estimation_method='reg' or fewer covariates.",
                        "Call results.epv_summary() for per-cohort details.",
                        "-" * 85,
                        "",
                    ]
                )

        lines.extend(
            [
                "Event-study and cohort surfaces: results.aggregate('event_study') / "
                "results.aggregate('group').",
                "",
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                "=" * 85,
            ]
        )

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the summary to stdout."""
        print(self.summary(alpha))

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary (canonical key names only).

        Returns
        -------
        Dict[str, Any]
            The headline quintet (``conf_int`` split into
            ``conf_int_lower`` / ``conf_int_upper``), the shared scalars, and
            the design-specific block.
        """
        result: Dict[str, Any] = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "estimation_method": self.estimation_method,
            "alpha": self.alpha,
            "design": self.design,
            "pscore_trim": self.pscore_trim,
            "inference_method": self.inference_method,
            "df_inference": self.df_inference,
        }
        if self.design == "staggered":
            result.update(
                {
                    "n_treated_units": self.n_treated_units,
                    "n_control_units": self.n_control_units,
                    "n_never_enabled": self.n_never_enabled,
                    "n_eligible": self.n_eligible,
                    "n_ineligible": self.n_ineligible,
                    "n_groups": len(self.groups),
                    "n_periods": len(self.time_periods),
                    "groups": self.groups,
                    "time_periods": self.time_periods,
                    "control_group": self.control_group,
                    "base_period": self.base_period,
                    "anticipation": self.anticipation,
                }
            )
            if self.n_bootstrap is not None:
                result["n_bootstrap"] = self.n_bootstrap
            if self.comparison_group_counts is not None:
                result["comparison_group_counts"] = self.comparison_group_counts
        else:
            result.update(
                {
                    "n_treated_eligible": self.n_treated_eligible,
                    "n_treated_ineligible": self.n_treated_ineligible,
                    "n_control_eligible": self.n_control_eligible,
                    "n_control_ineligible": self.n_control_ineligible,
                    "vcov_type": self.vcov_type,
                }
            )
            if self.r_squared is not None:
                result["r_squared"] = self.r_squared
            if self.n_bootstrap is not None:
                result["n_bootstrap"] = self.n_bootstrap
            if self.n_clusters is not None:
                result["n_clusters"] = self.n_clusters
            if self.cluster_name is not None:
                result["cluster_name"] = self.cluster_name
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            result["weight_type"] = sm.weight_type
            result["effective_n"] = sm.effective_n
            result["design_effect"] = sm.design_effect
            result["sum_weights"] = sm.sum_weights
            result["n_strata"] = sm.n_strata
            result["n_psu"] = sm.n_psu
            result["df_survey"] = sm.df_survey
        return result

    def to_dataframe(self, level: Optional[str] = None) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Parameters
        ----------
        level : str, optional
            ``"overall"`` - one row of the scalar ``to_dict()`` entries (both
            designs); ``"group_time"`` - the ``ATT(g, t)`` cells (staggered
            design only). ``None`` selects ``"overall"`` in the 2x2x2 design
            and ``"group_time"`` in the staggered design. ``"event_study"`` and
            ``"group"`` are not stored on 4.0 fits - use
            ``results.aggregate('event_study').to_dataframe()`` /
            ``results.aggregate('group').to_dataframe()``.

        Returns
        -------
        pd.DataFrame
        """
        if level is None:
            level = "group_time" if self.design == "staggered" else "overall"
        if level == "overall":
            scalars = {
                k: v
                for k, v in self.to_dict().items()
                if not isinstance(v, (dict, list, tuple, np.ndarray))
            }
            return pd.DataFrame([scalars])
        if level == "group_time":
            if self.design != "staggered":
                raise ValueError(f"to_dataframe(level='group_time'): {_TWO_BY_TWO_ONLY}")
            rows = []
            for (g, t), data in self.group_time_effects.items():
                row = {
                    "group": g,
                    "time": t,
                    "effect": data["effect"],
                    "se": data["se"],
                    "t_stat": data["t_stat"],
                    "p_value": data["p_value"],
                    "conf_int_lower": data["conf_int"][0],
                    "conf_int_upper": data["conf_int"][1],
                }
                if self.epv_diagnostics and (g, t) in self.epv_diagnostics:
                    row["epv"] = self.epv_diagnostics[(g, t)].get("epv")
                rows.append(row)
            return pd.DataFrame(rows)
        if level in ("event_study", "group"):
            raise ValueError(
                f"{'Event study' if level == 'event_study' else 'Group'} effects are not "
                f"stored on 4.0 fits; use results.aggregate({level!r}).to_dataframe() "
                "(no refit required; staggered design only)."
            )
        raise ValueError(
            f"Unknown level: {level}. Use 'overall' or 'group_time' (or None for the "
            "design's default)."
        )

    def epv_summary(self, show_all: bool = False) -> pd.DataFrame:
        """
        Return the EPV diagnostics as a DataFrame.

        Parameters
        ----------
        show_all : bool, default False
            If False, only show cells with low EPV. If True, show all.

        Returns
        -------
        pd.DataFrame
            2x2x2: columns ``subgroup, epv, n_events, n_params, is_low``.
            Staggered: columns ``group, time, epv, n_events, n_params, is_low``.
        """
        staggered = self.design == "staggered"
        cols = (
            ["group", "time", "epv", "n_events", "n_params", "is_low"]
            if staggered
            else ["subgroup", "epv", "n_events", "n_params", "is_low"]
        )
        if not self.epv_diagnostics:
            return pd.DataFrame(columns=cols)
        rows = []
        for key, diag in sorted(self.epv_diagnostics.items()):
            if show_all or diag.get("is_low", False):
                row: Dict[str, Any] = {}
                if staggered:
                    row["group"], row["time"] = key
                else:
                    row["subgroup"] = key
                row.update(
                    {
                        "epv": diag.get("epv"),
                        "n_events": diag.get("n_events"),
                        "n_params": diag.get("k"),
                        "is_low": diag.get("is_low", False),
                    }
                )
                rows.append(row)
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    # ------------------------------------------------------------------ #
    # Post-fit aggregation (spec section 6; rows M-014 / M-140 / M-141)
    # ------------------------------------------------------------------ #

    def _aggregate_validate_weights(self, weights: Optional[str]) -> None:
        if weights is not None and weights != "event_time":
            raise ValueError(
                f"TripleDifferenceResults.aggregate() accepts weights=None (the "
                f"cohort-size-weighted overall) or weights='event_time' (the paper's "
                f"Eq. 4.14 overall, the unweighted mean of the post-treatment "
                f"event-study effects); got {weights!r}."
            )

    def _aggregate_compute(
        self, level: str, *, weights: Optional[str], balance_e: Optional[int]
    ) -> Any:
        # Every guard runs before any relay returns (nothing may be dropped).
        if weights == "event_time" and level != "simple":
            raise ValueError(
                "weights='event_time' selects the Eq. 4.14 overall and applies to "
                "aggregate('simple') only."
            )
        if self.design != "staggered":
            if level in ("event_study", "group") or weights is not None or balance_e is not None:
                raise ValueError(f"aggregate({level!r}): {_TWO_BY_TWO_ONLY}")
        if level == "simple" and balance_e is not None and weights != "event_time":
            raise ValueError(
                "balance_e applies to aggregate('simple', weights='event_time') and "
                "aggregate('event_study'); the default aggregate('simple') relays the "
                "stored overall ATT, which balance_e does not enter."
            )

        if level == "simple" and weights is None:
            return self._aggregate_simple_result()

        kit = getattr(self, "_aggregation_kit", None)
        if kit is None:
            raise ValueError(
                "This result carries no aggregation kit, so it cannot be "
                "re-aggregated. Kits are attached at fit time by the "
                "estimator; a result unpickled from an older release will "
                "not have one."
            )

        # Shallow copy: shares every array but gives the aggregators a scratch
        # dict to memoize into, so the kit itself is never written to (the CS
        # pattern - and exactly what the engine did at fit through its
        # eligibility-zeroed copy, which is what the kit retains).
        precomputed = dict(kit.bookkeeping)

        # Bootstrap replay (CS pattern): re-run the fit-time multiplier
        # bootstrap from the kit-retained RNG state so the recompute levels
        # carry percentile inference matching a fit-time aggregation.
        boot_replay: Optional["CSBootstrapResults"] = None
        if self.bootstrap_results is not None:
            spec = getattr(kit, "bootstrap", None)
            if spec is None or getattr(spec, "bitgen_state", None) is None:
                raise NotImplementedError(
                    f"aggregate({level!r}) on this bootstrapped fit needs the "
                    "fit-time bootstrap replay state, which this result "
                    "predates - refit to enable the percentile-bootstrap "
                    "replay. aggregate('simple') relays the stored bootstrap "
                    "inference and remains available."
                )
            current_backend = effective_weight_backend()
            if spec.backend not in ("portable", current_backend):
                raise NotImplementedError(
                    f"aggregate({level!r}) cannot replay this bootstrapped "
                    f"fit here: its multiplier bootstrap was generated under "
                    f"the {spec.backend!r} weight backend, but the current "
                    f"backend is {current_backend!r}, and the two produce "
                    "different draws from the same RNG state. Re-fit under "
                    "the current backend, or restore the original one (the "
                    "DIFF_DIFF_BACKEND environment variable / the Rust "
                    "extension install)."
                )
            host = _KitBootstrapAggregator(
                kit.alpha,
                kit.anticipation,
                spec.n_bootstrap,
                spec.weight_type,
                bootstrap_label=kit.bookkeeping.get("bootstrap_label", "TripleDifference"),
            )
            replay_level = "event_study" if level == "simple" else level
            boot_replay = host._run_multiplier_bootstrap(
                group_time_effects=self.group_time_effects,
                influence_func_info=kit.influence,
                aggregate=replay_level,
                balance_e=balance_e if replay_level == "event_study" else None,
                treatment_groups=self.groups,
                time_periods=self.time_periods,
                df=None,
                unit=None,
                precomputed=precomputed,
                cband=kit.cband,
                _replay_bitgen_state=spec.bitgen_state,
            )

        agg = _KitAggregator(kit.alpha, kit.anticipation)

        if level == "group":
            effects = agg._aggregate_by_group(
                self.group_time_effects,
                kit.influence,
                self.groups,
                precomputed=precomputed,
                df=None,
                unit=None,
            )
            if boot_replay is not None:
                apply_bootstrap_group_overrides(effects, boot_replay, kit.alpha)
            return self._group_effects_to_aggregation(effects, kit)

        es = agg._aggregate_event_study(
            self.group_time_effects,
            kit.influence,
            self.groups,
            self.time_periods,
            balance_e,
            None,
            None,
            precomputed,
        )

        if level == "simple":  # weights == "event_time": the Eq. 4.14 overall
            return self._event_time_overall_result(es, boot_replay, kit)

        # event_study -> the unified EventStudyResults container (row M-092)
        if boot_replay is not None:
            apply_bootstrap_event_study_overrides(es.effects, boot_replay, kit.alpha)
            apply_cband_conf_ints(es.effects, boot_replay.cband_crit_value)
        carrier = replace(
            self,
            event_study_effects=es.effects,
            event_study_vcov=boot_replay.event_study_vcov if boot_replay is not None else es.vcov,
            event_study_vcov_index=(
                boot_replay.event_study_vcov_index if boot_replay is not None else es.vcov_index
            ),
            event_study_df=None if boot_replay is not None else es.df_used,
            cband_crit_value=(
                boot_replay.cband_crit_value if boot_replay is not None else self.cband_crit_value
            ),
            reference_event_times=es.reference_event_times,
        )
        return build_event_study_surface(carrier)

    def _aggregate_simple_result(self) -> AggregationResult:
        """The stored headline ATT as a one-row table (a view, both designs).

        Bit-identical to the fit by construction, hence permitted on
        bootstrapped fits: the stored quintet relays verbatim and only the df
        column is NaN'd there (no df governs percentile inference).
        """
        ci = self.conf_int or (np.nan, np.nan)
        if self.design == "staggered":
            n_units = float((self.n_treated_units or 0) + (self.n_control_units or 0))
            n, n_kind = n_units, "units"
        else:
            n, n_kind = float(self.n_obs), "obs"
        return AggregationResult(
            level="simple",
            label=np.array(["overall"], dtype=object),
            target=np.array(["att"], dtype=object),
            att=np.array([self.att], dtype=float),
            se=np.array([self.se], dtype=float),
            t_stat=np.array([self.t_stat], dtype=float),
            p_value=np.array([self.p_value], dtype=float),
            conf_int_lower=np.array([ci[0]], dtype=float),
            conf_int_upper=np.array([ci[1]], dtype=float),
            n=np.array([n], dtype=float),
            df=np.array(
                [np.nan if self.bootstrap_results is not None else _df_or_nan(self)],
                dtype=float,
            ),
            alpha=self.alpha,
            n_kind=n_kind,
            weight=np.array([1.0], dtype=float),
            estimator="TripleDifference",
        )

    def _event_time_overall_result(
        self, es: Any, boot_replay: Optional["CSBootstrapResults"], kit: Any
    ) -> AggregationResult:
        """The paper's Eq. 4.14 overall: the unweighted mean of post-treatment ES(e).

        Reproduces the 3.x fit-time ``overall_att_es`` rule exactly: the point
        estimate averages every finite post-treatment horizon; the analytical
        SE is the influence function of that mean, with the df the engine
        used (the per-statistic effective df when the survey design yields
        one, else the fit's sentinelled survey df); on bootstrapped fits the
        replayed percentile SE / CI / p override it. Both fit-time warnings
        are preserved.
        """
        df_fit = kit.bookkeeping.get("ddd_df_survey_original")
        df_es = df_fit
        overall = es.overall
        if overall is None:
            warnings.warn(
                "Event-study aggregation was requested but no post-treatment "
                "horizons are available for the Eq. 4.14 overall "
                "(aggregate('simple', weights='event_time')); returning NaN.",
                UserWarning,
                stacklevel=4,
            )
            att_es: float = np.nan
            se_es: float = np.nan
            t_es, p_es, ci_es = safe_inference(np.nan, np.nan, alpha=kit.alpha, df=df_fit)
            analytical_se_nonfinite = False
        else:
            att_es = float(overall["att"])
            se_es = float(overall["se"])
            analytical_se_nonfinite = bool(np.isfinite(att_es) and not np.isfinite(se_es))
            eff_df = overall.get("effective_df")
            df_es = eff_df if eff_df is not None else df_fit
            t_es, p_es, ci_es = safe_inference(att_es, se_es, alpha=kit.alpha, df=df_es)
        if boot_replay is not None and boot_replay.overall_att_es_se is not None:
            se_es = boot_replay.overall_att_es_se
            # The engine passes its (possibly survey-mutated) running df here;
            # the two coincide on every bootstrapped fit because bootstrap x
            # replicate-weight designs are rejected and the mutation only
            # happens on replicate designs.
            t_es, _, _ = safe_inference(att_es, se_es, alpha=kit.alpha, df=df_fit)
            ci_es = boot_replay.overall_att_es_ci
            p_es = boot_replay.overall_att_es_p_value
        if analytical_se_nonfinite and not np.isfinite(se_es):
            warnings.warn(
                "Eq. 4.14 overall (aggregate('simple', weights='event_time')) point "
                "estimate is defined but its standard error is undefined (NaN): either a "
                "contributing post-treatment event-study horizon has a non-finite "
                "influence function, or the variance is unidentified (e.g. a "
                "single-PSU/cluster survey design). The SE and its inference fields "
                "are NaN.",
                UserWarning,
                stacklevel=4,
            )
        ci_es = ci_es if ci_es is not None else (np.nan, np.nan)
        n_units = float((self.n_treated_units or 0) + (self.n_control_units or 0))
        return AggregationResult(
            level="simple",
            label=np.array(["overall"], dtype=object),
            target=np.array(["att"], dtype=object),
            att=np.array([att_es], dtype=float),
            se=np.array([se_es], dtype=float),
            t_stat=np.array([t_es], dtype=float),
            p_value=np.array([p_es], dtype=float),
            conf_int_lower=np.array([ci_es[0]], dtype=float),
            conf_int_upper=np.array([ci_es[1]], dtype=float),
            n=np.array([n_units], dtype=float),
            df=np.array(
                [
                    (
                        np.nan
                        if boot_replay is not None
                        else (float(df_es) if overall is not None and df_es is not None else np.nan)
                    )
                ],
                dtype=float,
            ),
            alpha=kit.alpha,
            n_kind="units",
            weight=np.array([1.0], dtype=float),
            estimator="TripleDifference",
        )

    def _group_effects_to_aggregation(
        self, effects: Dict[Any, Dict[str, Any]], kit: Any
    ) -> AggregationResult:
        """Per-cohort aggregation as a table (the CallawaySantAnnaResults shape).

        ``weight`` is left unset: ``_aggregate_by_group`` weights ``(g, t)``
        cells equally WITHIN each cohort and never forms a cross-cohort mass.
        ``n`` is the cohort's finite-contributing cell count
        (``n_kind="cells"``).
        """
        labels = list(effects.keys())
        rows = [effects[g] for g in labels]
        cis = [r.get("conf_int") or (np.nan, np.nan) for r in rows]
        return AggregationResult(
            level="group",
            label=np.array(labels, dtype=object),
            target=np.array(["att"] * len(labels), dtype=object),
            att=np.array([r["effect"] for r in rows], dtype=float),
            se=np.array([r["se"] for r in rows], dtype=float),
            t_stat=np.array([r["t_stat"] for r in rows], dtype=float),
            p_value=np.array([r["p_value"] for r in rows], dtype=float),
            conf_int_lower=np.array([c[0] for c in cis], dtype=float),
            conf_int_upper=np.array([c[1] for c in cis], dtype=float),
            n=np.array([float(r.get("n_periods", np.nan)) for r in rows], dtype=float),
            df=rows[0].get("df_used") if rows else None,
            alpha=kit.alpha,
            n_kind="cells",
            weight=None,
            estimator="TripleDifference",
        )


def _df_or_nan(results: Any) -> float:
    df = resolve_inference_df(results)
    return np.nan if df is None else float(df)
