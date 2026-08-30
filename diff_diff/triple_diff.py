"""
Triple Difference (DDD) estimators.

Implements the methodology from Ortiz-Villavicencio & Sant'Anna (2025)
"Better Understanding Triple Differences Estimators" for causal inference
when treatment requires satisfying two criteria:
1. Belonging to a treated group (e.g., a state with a policy)
2. Being in an eligible partition (e.g., women, low-income, etc.)

This module provides regression adjustment, inverse probability weighting,
and doubly robust estimators that correctly handle covariate adjustment,
unlike naive implementations. Standard errors use the efficient influence
function: SE = std(IF) / sqrt(n), which is inherently heteroskedasticity-
robust. Cluster-robust SEs are available via the ``cluster`` parameter.
The ``vcov_type`` input contract is permanently narrow to ``{"hc1"}``
because the analytical-sandwich families (classical, hc2, hc2_bm) have
no equivalent single design matrix on the 3-pairwise-DiD decomposition;
see REGISTRY.md "IF-based variance estimators vs analytical-sandwich
estimators" for the structural taxonomy.

The DDD is computed via three pairwise DiD comparisons matching R's
``triplediff::ddd()`` package (panel=FALSE mode).

Reference:
    Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
    Better Understanding Triple Differences Estimators.
    arXiv:2505.09942.
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from diff_diff._base import BaseEstimator
from diff_diff._deprecation import (
    NOT_SUPPLIED,
    require_arg,
    resolve_renamed_kwarg,
    warn_deprecated_kwarg,
)
from diff_diff._staggered_triple_diff_engine import _StaggeredTripleDiffEngineMixin
from diff_diff.linalg import _rank_guarded_inv, solve_logit, solve_ols
from diff_diff.results import _format_survey_block, _get_significance_stars
from diff_diff.results_base import _SUMMARY_ALPHA_MESSAGE, BaseResults, _require_fit_alpha
from diff_diff.staggered_aggregation import CallawaySantAnnaAggregationMixin
from diff_diff.staggered_bootstrap import CallawaySantAnnaBootstrapMixin
from diff_diff.staggered_triple_diff_results import StaggeredTripleDiffResults
from diff_diff.utils import (
    safe_inference,
    staggered_ddd_ctor_offenders,
    validate_anticipation,
    validate_n_bootstrap,
    validate_pscore_trim,
)

if TYPE_CHECKING:
    from diff_diff.survey import SurveyDesign


_MIN_CELL_SIZE = 10

# =============================================================================
# Results Classes
# =============================================================================


@dataclass
class TripleDifferenceResults(BaseResults):
    """
    Results from Triple Difference (DDD) estimation.

    Provides access to the estimated average treatment effect on the treated
    (ATT), standard errors, confidence intervals, and diagnostic information.

    Attributes
    ----------
    att : float
        Average Treatment effect on the Treated (ATT).
        This is the effect on units in the treated group (G=1) and eligible
        partition (P=1) after treatment (T=1).
    se : float
        Standard error of the ATT estimate.
    t_stat : float
        T-statistic for the ATT estimate.
    p_value : float
        P-value for the null hypothesis that ATT = 0.
    conf_int : tuple[float, float]
        Confidence interval for the ATT.
    n_obs : int
        Total number of observations used in estimation.
    n_treated_eligible : int
        Number of observations in treated group and eligible partition.
    n_treated_ineligible : int
        Number of observations in treated group and ineligible partition.
    n_control_eligible : int
        Number of observations in control group and eligible partition.
    n_control_ineligible : int
        Number of observations in control group and ineligible partition.
    estimation_method : str
        Estimation method used: "dr" (doubly robust), "reg" (regression
        adjustment), or "ipw" (inverse probability weighting).
    alpha : float
        Significance level used for confidence intervals.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    n_treated_eligible: int
    n_treated_ineligible: int
    n_control_eligible: int
    n_control_ineligible: int
    estimation_method: str
    alpha: float = 0.05
    # Group means for diagnostics
    group_means: Optional[Dict[str, float]] = field(default=None)
    # Propensity score diagnostics (for IPW/DR)
    pscore_stats: Optional[Dict[str, float]] = field(default=None)
    # Regression diagnostics
    r_squared: Optional[float] = field(default=None)
    # Covariate balance statistics
    covariate_balance: Optional[pd.DataFrame] = field(default=None, repr=False)
    # Inference details
    inference_method: str = field(default="analytical")
    vcov_type: str = field(default="hc1")
    cluster_name: Optional[str] = field(default=None)
    n_bootstrap: Optional[int] = field(default=None)
    n_clusters: Optional[int] = field(default=None)
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)
    # EPV diagnostics per subgroup comparison
    epv_diagnostics: Optional[Dict[int, Dict[str, Any]]] = field(default=None, repr=False)
    epv_threshold: float = 10
    pscore_fallback: str = "error"

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"TripleDifferenceResults(ATT={self.att:.4f}{self.significance_stars}, "
            f"SE={self.se:.4f}, p={self.p_value:.4f}, method={self.estimation_method})"
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of the estimation results.

        Parameters
        ----------
        alpha : float, optional
            Accepted for signature uniformity. The stored intervals were
            computed at fit time; a value different from the stored
            ``alpha`` raises ValueError rather than silently recomputing
            or relabeling. Re-fit at the desired alpha instead.

        Returns
        -------
        str
            Formatted summary table.
        """
        alpha = _require_fit_alpha(alpha, self.alpha, message=_SUMMARY_ALPHA_MESSAGE)
        conf_level = int((1 - alpha) * 100)

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
        if self.survey_metadata is None:
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

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the summary to stdout."""
        print(self.summary(alpha))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all estimation results.
        """
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "n_treated_eligible": self.n_treated_eligible,
            "n_treated_ineligible": self.n_treated_ineligible,
            "n_control_eligible": self.n_control_eligible,
            "n_control_ineligible": self.n_control_ineligible,
            "estimation_method": self.estimation_method,
            "inference_method": self.inference_method,
            "vcov_type": self.vcov_type,
        }
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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with estimation results.
        """
        return pd.DataFrame([self.to_dict()])

    @property
    def is_significant(self) -> bool:
        """Check if the ATT is statistically significant at the alpha level."""
        return bool(self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)

    def epv_summary(self, show_all: bool = False) -> pd.DataFrame:
        """
        Return per-subgroup EPV diagnostics as a DataFrame.

        Parameters
        ----------
        show_all : bool, default False
            If False, only show subgroups with low EPV. If True, show all.

        Returns
        -------
        pd.DataFrame
            Columns: subgroup, epv, n_events, n_params, is_low.
        """
        if not self.epv_diagnostics:
            return pd.DataFrame(columns=["subgroup", "epv", "n_events", "n_params", "is_low"])
        rows = []
        for sg, diag in sorted(self.epv_diagnostics.items()):
            if show_all or diag.get("is_low", False):
                rows.append(
                    {
                        "subgroup": sg,
                        "epv": diag.get("epv"),
                        "n_events": diag.get("n_events"),
                        "n_params": diag.get("k"),
                        "is_low": diag.get("is_low", False),
                    }
                )
        cols = ["subgroup", "epv", "n_events", "n_params", "is_low"]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# =============================================================================
# Helper Functions
# =============================================================================


# =============================================================================
# Main Estimator Class
# =============================================================================


class TripleDifference(
    _StaggeredTripleDiffEngineMixin,
    CallawaySantAnnaBootstrapMixin,
    CallawaySantAnnaAggregationMixin,
    BaseEstimator,
):
    """
    Triple Difference (DDD) estimator.

    Estimates the Average Treatment effect on the Treated (ATT) when treatment
    requires satisfying two criteria: belonging to a treated group AND being
    in an eligible partition of the population. The DDD design was popularized
    by Gruber (1994) [2]_.

    This implementation follows Ortiz-Villavicencio & Sant'Anna (2025) [1]_,
    which shows that naive DDD implementations (difference of two DiDs,
    three-way fixed effects) are invalid when covariates are needed for
    identification.

    Parameters
    ----------
    estimation_method : str, default="dr"
        Estimation method to use:

        - "dr": Doubly robust (recommended). Consistent if either the outcome
          model or propensity score model is correctly specified.
        - "reg": Regression adjustment (outcome regression).
        - "ipw": Inverse probability weighting.
    robust : bool, optional
        DEPRECATED (row M-046; warns with ``FutureWarning``, removed in
        4.0 - use ``vcov_type=``). Influence-function SEs are inherently
        robust to heteroskedasticity, so the flag never had an effect
        here; it was retained only for API compatibility.
    cluster : str, optional
        Column name for cluster-robust standard errors. When provided,
        SEs are computed using the Liang-Zeger cluster-robust variance
        estimator on the influence function.
    vcov_type : str, default="hc1"
        Variance estimator. Permanently narrow to ``{"hc1"}`` per the
        IF-based variance decomposition: TripleDifference uses an
        efficient influence function and has no single design matrix on
        which the analytical-sandwich families (``classical``, ``hc2``,
        ``hc2_bm``) could compute hat-matrix leverage or Bell-McCaffrey
        Satterthwaite DOF. ``conley`` is deferred. With ``hc1``, default
        SE is ``std(IF)/sqrt(n)``; with ``hc1`` + ``cluster=<col>``,
        Liang-Zeger CR1 on the combined IF.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    pscore_trim : float, default=0.01
        Trimming threshold for propensity scores. Scores below this value
        or above (1 - pscore_trim) are clipped to avoid extreme weights. Must be in
        ``(0, 0.5)`` and large enough that ``1 - pscore_trim < 1`` in
        float64 (a sub-ulp trim would disable the upper clip).
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient (linearly dependent columns):

        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning
    epv_threshold : float, default=10
        Events Per Variable threshold for propensity score logit.
        When the ratio of minority-class observations to predictor
        variables (excluding intercept) falls below this value, a
        warning is emitted (or ``ValueError`` raised if
        ``rank_deficient_action="error"``). Based on Peduzzi et al.
        (1996). Only applies to IPW and DR estimation methods.
    pscore_fallback : str, default="error"
        Action when propensity score estimation fails:

        - "error": Raise the exception (default)
        - "unconditional": Fall back to unconditional propensity with
          a warning. For IPW, drops all covariates. For DR, the
          propensity model becomes unconditional but outcome regression
          still uses covariates.

        When ``rank_deficient_action="error"``, errors are always
        re-raised regardless of this setting.

    control_group : str, default="not_yet_treated"
        Comparison cohort for staggered mode: ``"not_yet_treated"`` (units
        whose enabling period is still in the future) or ``"never_treated"``
        (the never-enabled cohort only). The compact R spellings
        ``"notyettreated"``/``"nevertreated"`` are accepted only by the
        deprecated ``StaggeredTripleDifference`` and die with it at 4.0.
    anticipation : int, default=0
        Number of periods before the enabling period in which units may
        already respond. Must be a non-negative integer; ``bool`` is
        rejected. Shifts each cohort's
        base period earlier and excludes cohorts entering treatment within the
        window from the comparison group.
    base_period : str, default="varying"
        ``"varying"`` uses consecutive comparisons - each pre-treatment period
        ``t`` is compared to ``t - 1``, while post-treatment periods use the
        fixed cohort reference ``g - 1 - anticipation``. ``"universal"`` uses
        ``g - 1 - anticipation`` for every period, pre and post alike.
        The two therefore differ only on the PRE-treatment effects, which is
        what makes ``"varying"`` the pre-trend-readable default.
    n_bootstrap : int, default=0
        Multiplier-bootstrap replications for staggered mode. ``0`` means the
        analytical influence-function SEs (the library-wide ``0 = off``
        convention, row M-081). Clustered inference in staggered mode is only
        available through this path - ``cluster=`` is rejected there.
    bootstrap_weights : str, default="rademacher"
        Multiplier distribution: ``"rademacher"``, ``"mammen"`` or ``"webb"``.
    seed : int, optional
        Seed for the multiplier bootstrap.
    cband : bool, default=True
        Whether to compute simultaneous (sup-t) confidence bands alongside the
        pointwise ones.

    Attributes
    ----------
    results_ : TripleDifferenceResults or StaggeredTripleDiffResults
        Estimation results after calling fit(). The 2x2x2 mode returns
        ``TripleDifferenceResults``; the staggered mode returns
        ``StaggeredTripleDiffResults`` (the two containers unify at 4.0, row
        M-014).
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage with a DataFrame:

    >>> import pandas as pd
    >>> from diff_diff import TripleDifference
    >>>
    >>> # Data where treatment affects women (partition=1) in states
    >>> # that enacted a policy (group=1)
    >>> data = pd.DataFrame({
    ...     'outcome': [...],
    ...     'group': [1, 1, 0, 0, ...],      # 1=policy state, 0=control state
    ...     'partition': [1, 0, 1, 0, ...],  # 1=women, 0=men
    ...     'post': [0, 0, 1, 1, ...],       # 1=post-treatment period
    ... })
    >>>
    >>> # Fit using doubly robust estimation
    >>> ddd = TripleDifference(estimation_method="dr")
    >>> results = ddd.fit(
    ...     data,
    ...     outcome='outcome',
    ...     group='group',
    ...     partition='partition',
    ...     post='post'
    ... )
    >>> print(results.att)  # ATT estimate

    With covariates (properly handled unlike naive DDD):

    >>> results = ddd.fit(
    ...     data,
    ...     outcome='outcome',
    ...     group='group',
    ...     partition='partition',
    ...     post='post',
    ...     covariates=['age', 'income']
    ... )

    Notes
    -----
    The DDD estimator is appropriate when:

    1. Treatment affects only units satisfying BOTH criteria:
       - Belonging to a treated group (G=1), e.g., states with a policy
       - Being in an eligible partition (P=1), e.g., women, low-income

    2. The DDD parallel trends assumption holds: the differential trend
       between eligible and ineligible partitions would have been the same
       across treated and control groups, absent treatment.

    This is weaker than requiring separate parallel trends for two DiDs,
    as biases can cancel out in the differencing.

    **Which parameters belong to which mode.** ``control_group``,
    ``anticipation``, ``base_period``, ``n_bootstrap``, ``bootstrap_weights``,
    ``seed`` and ``cband`` apply to the STAGGERED mode only
    (``fit(..., first_treat=...)``).

    The first four raise a ``ValueError`` if given a non-default value and then
    used to fit the 2x2x2 design, rather than being silently ignored: that
    engine has one pre/post contrast, so there is no comparison-cohort choice,
    no anticipation window, no base-period rule and no multiplier bootstrap.

    The other three - ``bootstrap_weights``, ``seed`` and ``cband`` - are
    ACCEPTED and inert in 2x2x2 mode. They act only through ``n_bootstrap > 0``,
    which that mode already rejects, so they are unreachable by construction
    rather than silently ignored. The power helpers apply the same boundary, so
    an estimator that is legal to ``fit()`` is legal to simulate.

    References
    ----------
    .. [1] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
           Better Understanding Triple Differences Estimators.
           arXiv:2505.09942.

    .. [2] Gruber, J. (1994). The incidence of mandated maternity benefits.
           American Economic Review, 84(3), 622-641.
    """

    # Names this estimator in the shared bootstrap mixin's user-facing
    # warnings. The mixin is shared with the other hosts, so a hard-coded
    # literal there would misname whichever surface was actually fit.
    _BOOTSTRAP_LABEL: str = "TripleDifference"

    _PARAM_ATTR_ALIASES = {"robust": "_robust_arg"}
    _DERIVED_CONFIG_ATTRS = ("robust",)

    def __init__(
        self,
        estimation_method: str = "dr",
        robust: Optional[bool] = None,
        cluster: Optional[str] = None,
        vcov_type: str = "hc1",
        alpha: float = 0.05,
        pscore_trim: float = 0.01,
        rank_deficient_action: str = "warn",
        epv_threshold: float = 10,
        pscore_fallback: str = "error",
        control_group: str = "not_yet_treated",
        anticipation: int = 0,
        base_period: str = "varying",
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        cband: bool = True,
    ):
        if estimation_method not in ("dr", "reg", "ipw"):
            raise ValueError(
                f"estimation_method must be 'dr', 'reg', or 'ipw', " f"got '{estimation_method}'"
            )
        # Staggered-mode value contracts (M-013). The merged class uses the
        # UNDERSCORED library vocabulary from birth; R's compact spellings
        # ("notyettreated"/"nevertreated") die with StaggeredTripleDifference
        # at 4.0 and are deliberately NOT accepted here.
        if control_group not in ("never_treated", "not_yet_treated"):
            raise ValueError(
                f"control_group must be 'never_treated' or 'not_yet_treated', "
                f"got '{control_group}'"
            )
        # `anticipation` feeds BOTH the base-period rule and the comparison-cohort
        # threshold, so an out-of-domain value silently changes the estimand
        # rather than failing: anticipation=-1 makes the universal base period
        # `g` (an ALREADY-TREATED period) and relaxes the not-yet-treated
        # threshold to max(t, base) - 1, admitting cohorts treated at the
        # evaluation period as "clean" controls. Neither condition is
        # observable in the output. `bool` is rejected too - `True` would
        # otherwise coerce to a silent one-period window. (The whole family
        # now validates uniformly via the shared helper; see the family-wide
        # adoption note, ledger row M-144, in REGISTRY.md.)
        validate_anticipation(anticipation)
        if base_period not in ("varying", "universal"):
            raise ValueError(f"base_period must be 'varying' or 'universal', got '{base_period}'")
        if bootstrap_weights not in ("rademacher", "mammen", "webb"):
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{bootstrap_weights}'"
            )
        validate_n_bootstrap(n_bootstrap)
        # pscore_trim range check per row M-142 (trim=0 disables the overlap
        # guard, trim >= 0.5 inverts the clip bounds); type guard + coercion
        # via the shared utils.validate_pscore_trim helper.
        pscore_trim = validate_pscore_trim(pscore_trim)
        if rank_deficient_action not in ["warn", "error", "silent"]:
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )
        if epv_threshold <= 0:
            raise ValueError(f"epv_threshold must be > 0, got {epv_threshold}")
        if pscore_fallback not in {"error", "unconditional"}:
            raise ValueError(
                f"pscore_fallback must be 'error' or 'unconditional', " f"got '{pscore_fallback}'"
            )
        # vcov_type input contract: TripleDifference is permanently narrow
        # to {"hc1"} because the analytical-sandwich families (classical,
        # hc2, hc2_bm) require a single regression's hat matrix that
        # TripleDifference's 3-pairwise-DiD influence-function decomposition
        # doesn't have. See REGISTRY.md "IF-based variance estimators vs
        # analytical-sandwich estimators" for the structural taxonomy.
        # Factored out so fit() can re-run it: set_params now validates
        # eagerly via the BaseEstimator probe re-init, but DIRECT attribute
        # mutation (est.vcov_type = ...) still bypasses validation until fit.
        self._validate_vcov_type(vcov_type)

        self.estimation_method = estimation_method
        # `robust` is deprecated (row M-046; removed in 4.0): None sentinel,
        # raw arg at `_robust_arg`, resolved legacy bool on the public attr.
        if robust is not None:
            warn_deprecated_kwarg(type(self).__name__, "robust", "use vcov_type= instead")
        self._robust_arg = robust
        self.robust = robust if robust is not None else True
        self.cluster = cluster
        self.vcov_type = vcov_type
        self.alpha = alpha
        self.pscore_trim = pscore_trim
        self.rank_deficient_action = rank_deficient_action
        self.epv_threshold = epv_threshold
        self.pscore_fallback = pscore_fallback
        # Staggered-mode config (M-013). Inert in 2x2x2 mode, and fit() rejects
        # any non-default value there rather than letting it pass unused.
        self.control_group = control_group
        # Already validated above (early-validation ordering); normalize only.
        self.anticipation = int(anticipation)
        self.base_period = base_period
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.cband = cband

        self.is_fitted_ = False
        self.results_: Optional[Union[TripleDifferenceResults, StaggeredTripleDiffResults]] = None

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        group: Any = NOT_SUPPLIED,
        partition: Any = NOT_SUPPLIED,
        post: Any = NOT_SUPPLIED,
        covariates: Optional[List[str]] = None,
        survey_design=None,
        time: Any = NOT_SUPPLIED,
        *,
        unit: Any = NOT_SUPPLIED,
        first_treat: Any = NOT_SUPPLIED,
        aggregate: Optional[str] = None,
        balance_e: Optional[int] = None,
    ) -> Union[TripleDifferenceResults, StaggeredTripleDiffResults]:
        """
        Fit the Triple Difference model, in either of its two designs.

        The 2x2x2 design takes ``(group, partition, post)``; the staggered
        design takes ``(unit, time, first_treat, partition)``. ``first_treat=``
        selects the staggered engine - mixing the two parameter sets is an
        error, never a guess. This mirrors the reference implementation, whose
        ``triplediff::ddd()`` serves both designs from one signature.

        The staggered-only parameters are keyword-only, so a staggered call
        written positionally cannot silently bind to the 2x2x2 slots.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing all variables.
        outcome : str
            Name of the outcome variable column.
        group : str
            2x2x2 mode. Name of the group indicator column (0/1).
            1 = treated group (e.g., states that enacted policy).
            0 = control group.
        partition : str
            BOTH modes. Name of the partition/eligibility indicator column
            (0/1). 1 = eligible partition (e.g., women, targeted demographic);
            0 = ineligible partition. In staggered mode it must be
            time-invariant within units.
        post : str
            Name of the post-period indicator column (0/1).
            1 = post-treatment period.
            0 = pre-treatment period.
        covariates : list of str, optional
            List of covariate column names to adjust for.
            These are properly incorporated using the selected estimation
            method (unlike naive DDD implementations).
        survey_design : SurveyDesign, optional
            Survey design specification for complex survey data. When
            provided, uses survey weights for estimation and Taylor Series
            Linearization (TSL) for variance estimation. Supported with
            all estimation methods ("reg", "ipw", "dr").
        time : str
            In STAGGERED mode, the calendar period column (keyword). In 2x2x2
            mode it is the deprecated alias for ``post`` (row M-031) and warns
            with ``FutureWarning``. From 4.0 it means the calendar column only
            (row M-085).
        unit : str, keyword-only
            Staggered mode. Unit identifier column.
        first_treat : str, keyword-only
            Staggered mode, and the MODE TRIGGER. Column giving each unit's
            enabling period; 0 or ``np.inf`` marks never-enabled units.
        aggregate : str, optional, keyword-only
            Staggered mode. ``"event_study"``, ``"group"``, ``"simple"`` or
            ``"all"``.
        balance_e : int, optional, keyword-only
            Staggered mode. Event time to balance the event study on.

        Returns
        -------
        TripleDifferenceResults or StaggeredTripleDiffResults
            2x2x2 mode returns ``TripleDifferenceResults``; staggered mode
            returns ``StaggeredTripleDiffResults``. The two containers unify at
            4.0 (row M-014).

        Raises
        ------
        ValueError
            If required columns are missing, data validation fails, or the two
            designs' parameters are mixed.
        NotImplementedError
            If survey_design is used with wild_bootstrap inference.
        """
        # ---- Step 0: mode-INDEPENDENT value validation -------------------
        # A bad `aggregate` value must not be masked by a mode error (3(a)'s
        # `spec` precedent). Consequence, deliberate: fit(aggregate="bogus")
        # WITHOUT first_treat reports the invalid value, not the mode error.
        # `_validate_vcov_type` is deliberately NOT hoisted here - moving it
        # ahead of the M-031 shim would make a direct-attribute-mutated
        # vcov_type raise BEFORE the rename FutureWarning that fires today.
        if aggregate is not None and aggregate not in ("event_study", "group", "simple", "all"):
            raise ValueError(
                f"aggregate must be 'event_study', 'group', 'simple', or 'all', "
                f"got '{aggregate}'"
            )

        # ---- Step 1: staggered mode ---------------------------------------
        # Raw sentinels forwarded: the 2x2x2 rename shim must not run, because
        # `time=` is the calendar column on this path, not a deprecated alias.
        if first_treat is not NOT_SUPPLIED:
            return self._fit_staggered(
                data,
                outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                partition=partition,
                group=group,
                post=post,
                covariates=covariates,
                aggregate=aggregate,
                balance_e=balance_e,
                survey_design=survey_design,
            )

        # ---- Step 2: 2x2x2 mode, staggered-only FIT params ----------------
        _staggered_only = [
            name
            for name, supplied in (
                ("unit", unit is not NOT_SUPPLIED),
                ("aggregate", aggregate is not None),
                ("balance_e", balance_e is not None),
            )
            if supplied
        ]
        if _staggered_only:
            raise ValueError(
                f"{', '.join(_staggered_only)} require(s) first_treat= (the staggered DDD "
                f"mode); the 2x2x2 TripleDifference design estimates a single ATT from the "
                f"(group=, partition=, post=) triple and has no adoption cohorts to "
                f"aggregate over. For staggered adoption pass fit(..., unit=<unit id>, "
                f"time=<calendar column>, first_treat=<cohort column>, "
                f"partition=<partition column>)."
            )

        # ---- Step 3: 2x2x2 mode, staggered-only CONSTRUCTOR params --------
        # The mode is only known at fit time, so this coherence check cannot
        # live in __init__. bootstrap_weights/seed/cband are deliberately NOT
        # listed: they only take effect through n_bootstrap > 0, which is
        # rejected here, so they are unreachable-by-construction rather than
        # silently ignored (documented in REGISTRY and the M-013 notes).
        # Roster AND defaults come from the shared helper (derived from this
        # class's own __init__ signature), not a local copy: the power entry
        # points apply the same boundary, and two hand-maintained lists would
        # eventually disagree about what "staggered-configured" means.
        _staggered_ctor = staggered_ddd_ctor_offenders(self)
        if _staggered_ctor:
            raise ValueError(
                f"TripleDifference({', '.join(n + '=' for n in _staggered_ctor)}) applies to "
                f"the staggered DDD mode only (fit(..., first_treat=)): the 2x2x2 engine has "
                f"one pre/post contrast, so there is no comparison-cohort choice, no "
                f"anticipation window, no base-period rule and no multiplier bootstrap - its "
                f"SEs come from the efficient influence function (cluster= for CR1). Drop the "
                f"parameter(s), or pass first_treat= to fit the staggered design."
            )

        # ---- Step 4: 2x2x2 mode, the M-031 rename shim --------------------
        # group/partition are sentinel-defaulted now (they must be omissible in
        # staggered mode), so their missing-argument TypeErrors are restored
        # here. They run BEFORE the rename shim: after it, fit(df, "y",
        # time="t") would newly emit the M-031 FutureWarning before raising the
        # missing-group TypeError (today it raises immediately, with no
        # warning, and under -W error the surfaced exception type would flip).
        require_arg("TripleDifference.fit", "group", group)
        require_arg("TripleDifference.fit", "partition", partition)
        post = resolve_renamed_kwarg(
            "TripleDifference.fit",
            "time",
            time,
            "post",
            post,
            default=NOT_SUPPLIED,
            extra="From 4.0, time= means the calendar column only.",
        )
        require_arg("TripleDifference.fit", "post", post)
        # Body-local name; the public parameter is post (M-031).
        time = post
        # Re-validate vcov_type at fit-time: __init__ and set_params (via
        # the BaseEstimator probe re-init) both validate eagerly, so this
        # second layer only catches DIRECT attribute mutation
        # (est.vcov_type = ...) before it propagates to Results metadata.
        self._validate_vcov_type(self.vcov_type)

        # Reset replicate state from any previous fit
        self._replicate_n_valid = None

        # Resolve survey design if provided
        from diff_diff.survey import (
            _inject_cluster_as_psu,
            _resolve_effective_cluster,
            _resolve_survey_for_fit,
            compute_survey_metadata,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        if resolved_survey is not None and resolved_survey.weight_type != "pweight":
            raise ValueError(
                f"TripleDifference survey support requires weight_type='pweight', "
                f"got '{resolved_survey.weight_type}'. The survey variance math "
                f"assumes probability weights (pweight)."
            )

        # Validate inputs
        self._validate_data(data, outcome, group, partition, time, covariates)

        # Extract data
        y = data[outcome].values.astype(float)
        G = data[group].values.astype(float)
        P = data[partition].values.astype(float)
        T = data[time].values.astype(float)

        # Store cluster IDs for SE computation
        self._cluster_ids = data[self.cluster].values if self.cluster is not None else None
        if self._cluster_ids is not None and np.any(pd.isna(data[self.cluster])):
            raise ValueError(f"Cluster column '{self.cluster}' contains missing values")

        # Reject replicate-weight + cluster=: replicate IF variance is
        # computed by replicate reweighting (BRR / Fay / JK1 / JKn / SDR)
        # and ignores PSU/cluster entirely (survey.py:104-109 enforces
        # replicate_weights are mutually exclusive with strata/psu/fpc).
        # Honoring bare cluster= here would silently have no effect on
        # variance while populating cluster_name/n_clusters on Results
        # dishonestly. Fail-closed per feedback_no_silent_failures.
        # Mirrors CallawaySantAnna guard at staggered.py:1705-1719.
        if (
            self.cluster is not None
            and survey_design is not None
            and getattr(survey_design, "replicate_weights", None) is not None
        ):
            raise NotImplementedError(
                f"TripleDifference(cluster={self.cluster!r}) is not "
                "supported with replicate-weight survey designs. "
                "Replicate-weight variance is computed by replicate "
                "reweighting (BRR / Fay / JK1 / JKn / SDR) and ignores "
                "PSU/cluster entirely — setting cluster= would silently "
                "have no effect on the variance estimate. Either omit "
                "cluster= (the replicate weights encode the design "
                "structure implicitly) or use a non-replicate survey "
                "design (with explicit strata/psu/fpc)."
            )

        # Resolve effective cluster and inject cluster-as-PSU for survey variance
        if resolved_survey is not None:
            effective_cluster_ids = _resolve_effective_cluster(
                resolved_survey, self._cluster_ids, self.cluster
            )
            if effective_cluster_ids is not None:
                resolved_survey = _inject_cluster_as_psu(resolved_survey, effective_cluster_ids)
                if resolved_survey.psu is not None and survey_metadata is not None:
                    raw_w = (
                        data[survey_design.weights].values.astype(np.float64)
                        if survey_design.weights is not None
                        else np.ones(len(data), dtype=np.float64)
                    )
                    survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

        # Get covariates if specified
        X = None
        if covariates:
            X = data[covariates].values.astype(float)
            if np.any(np.isnan(X)):
                raise ValueError("Covariates contain missing values")

        # Count observations in each cell
        n_obs = len(y)
        n_treated_eligible = int(np.sum((G == 1) & (P == 1)))
        n_treated_ineligible = int(np.sum((G == 1) & (P == 0)))
        n_control_eligible = int(np.sum((G == 0) & (P == 1)))
        n_control_ineligible = int(np.sum((G == 0) & (P == 0)))

        # Compute cell means for diagnostics
        group_means = self._compute_cell_means(y, G, P, T, weights=survey_weights)

        # Estimate ATT based on method
        if self.estimation_method == "reg":
            att, se, r_squared, pscore_stats, epv_diag = self._regression_adjustment(
                y,
                G,
                P,
                T,
                X,
                survey_weights=survey_weights,
                resolved_survey=resolved_survey,
            )
        elif self.estimation_method == "ipw":
            att, se, r_squared, pscore_stats, epv_diag = self._ipw_estimation(
                y,
                G,
                P,
                T,
                X,
                survey_weights=survey_weights,
                resolved_survey=resolved_survey,
            )
        else:  # doubly robust
            att, se, r_squared, pscore_stats, epv_diag = self._doubly_robust(
                y,
                G,
                P,
                T,
                X,
                survey_weights=survey_weights,
                resolved_survey=resolved_survey,
            )

        # Compute inference
        # When survey design is active, use survey df (n_PSU - n_strata)
        if survey_metadata is not None and survey_metadata.df_survey is not None:
            df = survey_metadata.df_survey
            # Override with effective replicate df only when replicates were dropped
            if (
                hasattr(self, "_replicate_n_valid")
                and self._replicate_n_valid is not None
                and resolved_survey is not None
                and self._replicate_n_valid < resolved_survey.n_replicates
            ):
                df = self._replicate_n_valid - 1
                survey_metadata.df_survey = self._replicate_n_valid - 1
            # df <= 0 means insufficient rank for t-based inference
            if df is not None and df <= 0:
                df = 0  # Forces NaN from t-distribution
        elif (
            resolved_survey is not None
            and hasattr(resolved_survey, "uses_replicate_variance")
            and resolved_survey.uses_replicate_variance
        ):
            # Replicate design with undefined df (rank <= 1) — NaN inference
            df = 0  # Forces NaN from t-distribution
        else:
            df = n_obs - 8  # Approximate df (8 cell means)
            if covariates:
                df -= len(covariates)
            df = max(df, 1)

        t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=df)

        # Resolve cluster_name / n_clusters for Results metadata.
        # Under survey designs the survey block (PSU/strata/df) is the
        # canonical surface for cluster reporting — suppress the bare
        # cluster_name / n_clusters fields so they don't misreport the
        # raw `cluster=` argument when `survey_design.psu` overrides it.
        # Mirrors the variance-estimator line suppression in summary().
        if resolved_survey is not None:
            cluster_name_for_results: Optional[str] = None
            n_clusters: Optional[int] = None
        elif self.cluster is not None:
            cluster_name_for_results = self.cluster
            n_clusters = data[self.cluster].nunique()
        else:
            cluster_name_for_results = None
            n_clusters = None

        # Create results object
        self.results_ = TripleDifferenceResults(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=n_obs,
            n_treated_eligible=n_treated_eligible,
            n_treated_ineligible=n_treated_ineligible,
            n_control_eligible=n_control_eligible,
            n_control_ineligible=n_control_ineligible,
            estimation_method=self.estimation_method,
            alpha=self.alpha,
            group_means=group_means,
            pscore_stats=pscore_stats,
            r_squared=r_squared,
            inference_method="analytical",
            vcov_type=self.vcov_type,
            cluster_name=cluster_name_for_results,
            n_clusters=n_clusters,
            survey_metadata=survey_metadata,
            epv_diagnostics=epv_diag if epv_diag else None,
            epv_threshold=self.epv_threshold,
            pscore_fallback=self.pscore_fallback,
        )

        self.is_fitted_ = True
        return self.results_

    def _fit_staggered(
        self,
        data: pd.DataFrame,
        outcome: str,
        *,
        unit: Any,
        time: Any,
        first_treat: Any,
        partition: Any,
        group: Any,
        post: Any,
        covariates: Optional[List[str]],
        aggregate: Optional[str],
        balance_e: Optional[int],
        survey_design,
    ) -> StaggeredTripleDiffResults:
        """Staggered branch of the merged fit (row M-013).

        Reached only when ``first_treat=`` was supplied. Rejects the other
        design's parameters rather than ignoring them, then hands off to the
        engine shared with the deprecated ``StaggeredTripleDifference``.
        """
        # Step 1: the 2x2x2 params are rejected, not ignored. A positional 3rd
        # or 5th argument lands in group/post and is caught here.
        _2x2x2_only = [
            name for name, value in (("group", group), ("post", post)) if value is not NOT_SUPPLIED
        ]
        if _2x2x2_only:
            raise ValueError(
                f"{', '.join(_2x2x2_only)} belong(s) to the 2x2x2 mode and cannot be combined "
                f"with first_treat=: staggered DDD reads the treated cohort from first_treat= "
                f"(0 or np.inf = never enabled) and the pre/post contrast from time= (the "
                f"calendar column), so there is no 0/1 group dummy and no 0/1 post dummy. "
                f"Positional arguments 3-5 are the 2x2x2 (group, partition, post) triple - in "
                f"staggered mode pass unit=, time=, first_treat= and partition= as keywords."
            )

        # Step 2: the reverse constructor guard, the mirror of the 2x2x2 one.
        # The two arms have DIFFERENT reachability. `robust=` is constructible
        # (it only warns), so that arm is reached normally. `vcov_type` is
        # validated eagerly in __init__ and by set_params' probe re-init, so a
        # bad value never survives construction - its only live route is direct
        # attribute mutation, the same bypass __init__'s comment documents. That
        # makes this arm the staggered mode's direct-mutation guard, not dead
        # code, and it is why no separate _validate_vcov_type call is needed.
        _2x2x2_ctor = [
            name
            for name, non_default in (
                ("robust", self._robust_arg is not None),
                ("vcov_type", self.vcov_type != "hc1"),
            )
            if non_default
        ]
        if _2x2x2_ctor:
            raise ValueError(
                f"TripleDifference({', '.join(n + '=' for n in _2x2x2_ctor)}) applies to the "
                f"2x2x2 mode only: the staggered engine's SEs come from the GMM-combined "
                f"influence function (analytical) or the multiplier bootstrap "
                f"(n_bootstrap > 0), so there is no sandwich family to select."
            )

        # Step 3: required staggered columns (first_treat is present already).
        require_arg("TripleDifference.fit", "unit", unit)
        require_arg("TripleDifference.fit", "time", time)
        require_arg("TripleDifference.fit", "partition", partition)

        # Step 4: cluster= is a CONSTRUCTOR param and the staggered engine has
        # no clustered analytical path. The deprecated class accepts-then-warns
        # -then-ignores it; this surface is new, so it raises from birth rather
        # than shipping an accepted-but-unhonored parameter (3(a)'s precedent).
        if self.cluster is not None:
            raise ValueError(
                "cluster= is not supported in staggered DDD mode: cluster-robust ANALYTICAL "
                "SEs are not implemented for the staggered engine. Use n_bootstrap > 0 for "
                "unit-level clustered inference via the multiplier bootstrap, or fit the "
                "2x2x2 design, where cluster= gives Liang-Zeger CR1 standard errors."
            )

        results = self._fit_staggered_core(
            data,
            outcome,
            unit,
            time,
            first_treat,
            partition,
            covariates=covariates,
            aggregate=aggregate,
            balance_e=balance_e,
            survey_design=survey_design,
            estimator_name="TripleDifference",
            partition_label="partition",
            _frame_offset=2,
        )
        self.results_ = results
        return results

    def _validate_data(
        self,
        data: pd.DataFrame,
        outcome: str,
        group: str,
        partition: str,
        time: str,
        covariates: Optional[List[str]] = None,
    ) -> None:
        """Validate input data."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # Check required columns exist
        required_cols = [outcome, group, partition, time]
        if covariates:
            required_cols.extend(covariates)
        if self.cluster is not None:
            required_cols.append(self.cluster)

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        # Check for missing values in required columns
        for col in [outcome, group, partition, time]:
            if data[col].isna().any():
                raise ValueError(f"Column '{col}' contains missing values")

        # Validate binary variables
        for col, name in [(group, "group"), (partition, "partition"), (time, "time")]:
            unique_vals = set(data[col].unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(
                    f"'{name}' column must be binary (0/1), " f"got values: {sorted(unique_vals)}"
                )
            if len(unique_vals) < 2:
                raise ValueError(f"'{name}' column must have both 0 and 1 values")

        # Check we have observations in all cells
        G = data[group].values
        P = data[partition].values
        T = data[time].values

        cells = [
            ((G == 1) & (P == 1) & (T == 0), "treated, eligible, pre"),
            ((G == 1) & (P == 1) & (T == 1), "treated, eligible, post"),
            ((G == 1) & (P == 0) & (T == 0), "treated, ineligible, pre"),
            ((G == 1) & (P == 0) & (T == 1), "treated, ineligible, post"),
            ((G == 0) & (P == 1) & (T == 0), "control, eligible, pre"),
            ((G == 0) & (P == 1) & (T == 1), "control, eligible, post"),
            ((G == 0) & (P == 0) & (T == 0), "control, ineligible, pre"),
            ((G == 0) & (P == 0) & (T == 1), "control, ineligible, post"),
        ]

        for mask, cell_name in cells:
            n_cell = int(np.sum(mask))
            if n_cell == 0:
                raise ValueError(
                    f"No observations in cell: {cell_name}. "
                    "DDD requires observations in all 8 cells."
                )
            elif n_cell < _MIN_CELL_SIZE:
                warnings.warn(
                    f"Low observation count ({n_cell}) in cell: {cell_name}. "
                    f"Estimates may be unreliable with fewer than "
                    f"{_MIN_CELL_SIZE} observations per cell.",
                    UserWarning,
                    stacklevel=2,
                )

    def _compute_cell_means(
        self,
        y: np.ndarray,
        G: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute mean outcomes for each of the 8 DDD cells."""
        means = {}
        for g_val, g_name in [(1, "Treated"), (0, "Control")]:
            for p_val, p_name in [(1, "Eligible"), (0, "Ineligible")]:
                for t_val, t_name in [(0, "Pre"), (1, "Post")]:
                    mask = (G == g_val) & (P == p_val) & (T == t_val)
                    cell_name = f"{g_name}, {p_name}, {t_name}"
                    if weights is not None:
                        w_cell = weights[mask]
                        if np.sum(w_cell) <= 0:
                            raise ValueError(
                                f"Cell '{cell_name}' has zero effective survey "
                                f"weight. Cannot compute weighted cell mean. "
                                f"Check subpopulation/domain definition."
                            )
                        means[cell_name] = float(np.average(y[mask], weights=w_cell))
                    else:
                        means[cell_name] = float(np.mean(y[mask]))
        return means

    # =========================================================================
    # Three-DiD Decomposition (matches R's triplediff::ddd())
    # =========================================================================
    #
    # The DDD is decomposed into three pairwise DiD comparisons:
    #   DiD_3: subgroup 3 (G=1,P=0) vs subgroup 4 (G=1,P=1)
    #   DiD_2: subgroup 2 (G=0,P=1) vs subgroup 4 (G=1,P=1)
    #   DiD_1: subgroup 1 (G=0,P=0) vs subgroup 4 (G=1,P=1)
    #
    # DDD = DiD_3 + DiD_2 - DiD_1
    #
    # Each DiD uses the selected estimation method (DR, IPW, or RA).
    # SE is computed from the combined influence function:
    #   inf = w3*inf_3 + w2*inf_2 - w1*inf_1
    #   SE = std(inf, ddof=1) / sqrt(n)
    #
    # Reference: Ortiz-Villavicencio & Sant'Anna (2025), implemented in
    # R's triplediff::ddd() with panel=FALSE (repeated cross-section).
    # =========================================================================

    def _regression_adjustment(
        self,
        y: np.ndarray,
        G: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        X: Optional[np.ndarray],
        survey_weights: Optional[np.ndarray] = None,
        resolved_survey=None,
    ) -> Tuple[
        float, float, Optional[float], Optional[Dict[str, float]], Dict[int, Dict[str, Any]]
    ]:
        """
        Estimate ATT using regression adjustment via three-DiD decomposition.

        For each pairwise comparison (subgroup j vs subgroup 4), fits
        separate outcome models per subgroup-time cell and computes
        imputed counterfactual means. Matches R's triplediff::ddd()
        with est_method="reg".
        """
        return self._estimate_ddd_decomposition(
            y,
            G,
            P,
            T,
            X,
            survey_weights=survey_weights,
            resolved_survey=resolved_survey,
        )

    def _ipw_estimation(
        self,
        y: np.ndarray,
        G: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        X: Optional[np.ndarray],
        survey_weights: Optional[np.ndarray] = None,
        resolved_survey=None,
    ) -> Tuple[
        float, float, Optional[float], Optional[Dict[str, float]], Dict[int, Dict[str, Any]]
    ]:
        """
        Estimate ATT using inverse probability weighting via three-DiD
        decomposition.

        For each pairwise comparison, estimates propensity scores for
        subgroup membership P(subgroup=4|X) within {j, 4} subset.
        Matches R's triplediff::ddd() with est_method="ipw".
        """
        return self._estimate_ddd_decomposition(
            y,
            G,
            P,
            T,
            X,
            survey_weights=survey_weights,
            resolved_survey=resolved_survey,
        )

    def _doubly_robust(
        self,
        y: np.ndarray,
        G: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        X: Optional[np.ndarray],
        survey_weights: Optional[np.ndarray] = None,
        resolved_survey=None,
    ) -> Tuple[
        float, float, Optional[float], Optional[Dict[str, float]], Dict[int, Dict[str, Any]]
    ]:
        """
        Estimate ATT using doubly robust estimation via three-DiD
        decomposition.

        Combines outcome regression and IPW for robustness: consistent
        if either the outcome model or propensity score model is
        correctly specified. Matches R's triplediff::ddd() with
        est_method="dr".
        """
        return self._estimate_ddd_decomposition(
            y,
            G,
            P,
            T,
            X,
            survey_weights=survey_weights,
            resolved_survey=resolved_survey,
        )

    def _estimate_ddd_decomposition(
        self,
        y: np.ndarray,
        G: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        X: Optional[np.ndarray],
        survey_weights: Optional[np.ndarray] = None,
        resolved_survey=None,
    ) -> Tuple[
        float, float, Optional[float], Optional[Dict[str, float]], Dict[int, Dict[str, Any]]
    ]:
        """
        Core DDD estimation via three-DiD decomposition.

        Implements the methodology from Ortiz-Villavicencio & Sant'Anna
        (2025), matching R's triplediff::ddd() for repeated cross-section
        data (panel=FALSE).

        The DDD is decomposed into three pairwise DiD comparisons,
        each using the selected estimation method (DR, IPW, or RA):
          DDD = DiD_3 + DiD_2 - DiD_1

        Standard errors use the efficient influence function:
          SE = std(w3*IF_3 + w2*IF_2 - w1*IF_1) / sqrt(n)

        When resolved_survey is provided, survey-weighted SE is computed
        using TSL on the combined influence function.
        """
        n = len(y)
        est_method = self.estimation_method

        # Assign subgroups following R convention:
        #   4: G=1, P=1 (treated, eligible - reference/"treated")
        #   3: G=1, P=0 (treated, ineligible)
        #   2: G=0, P=1 (control, eligible)
        #   1: G=0, P=0 (control, ineligible)
        subgroup = np.zeros(n, dtype=int)
        subgroup[(G == 1) & (P == 1)] = 4
        subgroup[(G == 1) & (P == 0)] = 3
        subgroup[(G == 0) & (P == 1)] = 2
        subgroup[(G == 0) & (P == 0)] = 1

        post = T.astype(float)

        # Covariate matrix (always includes intercept)
        if X is not None and X.shape[1] > 0:
            covX = np.column_stack([np.ones(n), X])
            has_covariates = True
        else:
            covX = np.ones((n, 1))
            has_covariates = False

        # Three DiD comparisons: j vs 4 for j in {3, 2, 1}
        did_results = {}
        pscore_stats = None
        all_pscores = {}  # Collect pscores for diagnostics
        overlap_issues = []  # Collect overlap diagnostics across comparisons
        any_nonfinite_if = False
        # Tracker for rank-guarded inverse truncations in the OR bread / PS
        # Hessian across the three DiD comparisons. Reset per fit; surfaced as
        # ONE aggregate warning below rather than fanning out per comparison.
        self._lstsq_fallback_tracker: list = []
        epv_all = {}  # Collect EPV diagnostics per subgroup comparison

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for j in [3, 2, 1]:
                mask = (subgroup == j) | (subgroup == 4)
                y_sub = y[mask]
                post_sub = post[mask]
                sg_sub = subgroup[mask]
                covX_sub = covX[mask]
                n_sub = len(y_sub)

                PA4 = (sg_sub == 4).astype(float)
                PAa = (sg_sub == j).astype(float)

                # Subset survey weights for this comparison (needed for logit)
                w_sub = survey_weights[mask] if survey_weights is not None else None

                # --- Propensity scores ---
                if est_method == "reg":
                    # RA: no propensity scores needed
                    pscore_sub = np.ones(n_sub)
                    hessian = None
                elif has_covariates:
                    # Logistic regression: P(subgroup=4 | X) within {j, 4}
                    ps_estimated = True
                    diag = {}
                    try:
                        _, pscore_sub = solve_logit(
                            covX_sub[:, 1:],
                            PA4,
                            rank_deficient_action=self.rank_deficient_action,
                            weights=w_sub,
                            epv_threshold=self.epv_threshold,
                            context_label=f"subgroup {j} vs 4",
                            diagnostics_out=diag,
                        )
                    except Exception:
                        if self.pscore_fallback == "error" or self.rank_deficient_action == "error":
                            raise
                        if w_sub is not None:
                            pos = w_sub > 0
                            if np.any(pos):
                                p_uc = float(np.average(PA4[pos], weights=w_sub[pos]))
                            else:
                                p_uc = float(np.mean(PA4))
                        else:
                            p_uc = float(np.mean(PA4))
                        pscore_sub = np.full(n_sub, p_uc)
                        ps_estimated = False
                        warnings.warn(
                            f"Propensity score estimation failed for subgroup "
                            f"{j} vs 4; using unconditional probability. "
                            f"For DR, outcome regression still uses covariates. "
                            f"Consider estimation_method='reg' to avoid propensity "
                            f"scores entirely.",
                            UserWarning,
                            stacklevel=3,
                        )
                    if diag:
                        epv_all[j] = diag

                    pscore_sub = np.clip(pscore_sub, self.pscore_trim, 1 - self.pscore_trim)
                    all_pscores[j] = pscore_sub

                    # Check overlap: count obs at trim bounds
                    # (1e-10 tolerance for floating-point after np.clip)
                    n_trimmed = int(
                        np.sum(
                            (pscore_sub <= self.pscore_trim + 1e-10)
                            | (pscore_sub >= 1 - self.pscore_trim - 1e-10)
                        )
                    )
                    frac_trimmed = n_trimmed / len(pscore_sub)
                    if frac_trimmed > 0.05:
                        overlap_issues.append((j, frac_trimmed))

                    # Hessian only when PS was actually estimated
                    if ps_estimated:
                        W_ps = pscore_sub * (1 - pscore_sub)
                        if w_sub is not None:
                            W_ps = W_ps * w_sub
                        XWX = covX_sub.T @ (W_ps[:, None] * covX_sub)
                        # Rank-guarded inverse: a near-singular X'WX does not
                        # raise, so np.linalg.inv would return a garbage Hessian
                        # that inflates the IPW/DR influence function.
                        XWX_inv, _, _ = _rank_guarded_inv(
                            XWX,
                            tracker=getattr(self, "_lstsq_fallback_tracker", None),
                        )
                        hessian = XWX_inv * n_sub
                    else:
                        hessian = None
                else:
                    # No covariates: unconditional probability
                    pscore_sub = np.full(n_sub, np.mean(PA4))
                    pscore_sub = np.clip(pscore_sub, self.pscore_trim, 1 - self.pscore_trim)
                    # Check overlap (same logic as covariate branch)
                    n_trimmed = int(
                        np.sum(
                            (pscore_sub <= self.pscore_trim + 1e-10)
                            | (pscore_sub >= 1 - self.pscore_trim - 1e-10)
                        )
                    )
                    frac_trimmed = n_trimmed / len(pscore_sub)
                    if frac_trimmed > 0.05:
                        overlap_issues.append((j, frac_trimmed))
                    hessian = None

                # --- Outcome regression ---
                if est_method == "ipw":
                    # IPW: no outcome regression
                    or_ctrl_pre = np.zeros(n_sub)
                    or_ctrl_post = np.zeros(n_sub)
                    or_trt_pre = np.zeros(n_sub)
                    or_trt_post = np.zeros(n_sub)
                else:
                    # Fit separate OLS per subgroup-time cell, predict for all
                    or_ctrl_pre = self._fit_predict_mu(
                        y_sub,
                        covX_sub,
                        sg_sub == j,
                        post_sub == 0,
                        n_sub,
                        weights=w_sub,
                    )
                    or_ctrl_post = self._fit_predict_mu(
                        y_sub,
                        covX_sub,
                        sg_sub == j,
                        post_sub == 1,
                        n_sub,
                        weights=w_sub,
                    )
                    or_trt_pre = self._fit_predict_mu(
                        y_sub,
                        covX_sub,
                        sg_sub == 4,
                        post_sub == 0,
                        n_sub,
                        weights=w_sub,
                    )
                    or_trt_post = self._fit_predict_mu(
                        y_sub,
                        covX_sub,
                        sg_sub == 4,
                        post_sub == 1,
                        n_sub,
                        weights=w_sub,
                    )

                # --- Compute DiD ATT and influence function ---
                att_j, inf_j = self._compute_did_rc(
                    y_sub,
                    post_sub,
                    PA4,
                    PAa,
                    pscore_sub,
                    covX_sub,
                    or_ctrl_pre,
                    or_ctrl_post,
                    or_trt_pre,
                    or_trt_post,
                    hessian,
                    est_method,
                    n_sub,
                    weights=w_sub,
                )

                # Track non-finite IF values (flag for NaN SE later)
                if not np.all(np.isfinite(inf_j)):
                    any_nonfinite_if = True
                    inf_j = np.where(np.isfinite(inf_j), inf_j, 0.0)

                # Pad influence function to full length
                inf_full = np.zeros(n)
                inf_full[mask] = inf_j

                did_results[j] = {"att": att_j, "inf": inf_full}

        # Emit overlap warning if >5% of observations trimmed in any comparison
        if overlap_issues:
            details = ", ".join(f"subgroup {j} vs 4: {frac:.0%}" for j, frac in overlap_issues)
            warnings.warn(
                f"Poor propensity score overlap ({details} of observations "
                f"trimmed at bounds). IPW/DR estimates may be unreliable.",
                UserWarning,
                stacklevel=3,
            )

        # --- Combine three DiDs ---
        att = (
            float(did_results[3]["att"])
            + float(did_results[2]["att"])
            - float(did_results[1]["att"])
        )

        # Influence function weights (matching R's att_dr_rc)
        if survey_weights is not None:
            n3 = np.sum(survey_weights[(subgroup == 3) | (subgroup == 4)])
            n2 = np.sum(survey_weights[(subgroup == 2) | (subgroup == 4)])
            n1 = np.sum(survey_weights[(subgroup == 1) | (subgroup == 4)])
            n_total = np.sum(survey_weights)
        else:
            n3 = np.sum((subgroup == 3) | (subgroup == 4))
            n2 = np.sum((subgroup == 2) | (subgroup == 4))
            n1 = np.sum((subgroup == 1) | (subgroup == 4))
            n_total = n
        w3 = n_total / n3
        w2 = n_total / n2
        w1 = n_total / n1

        inf_func = (
            w3 * did_results[3]["inf"] + w2 * did_results[2]["inf"] - w1 * did_results[1]["inf"]
        )

        if resolved_survey is not None:
            # Survey-weighted SE via TSL on the combined influence function.
            # For IPW/DR: pairwise IFs include survey weights via weighted Riesz
            # representers, so divide out to avoid double-weighting by TSL.
            # For reg: pairwise IFs are already on the unweighted scale (WLS
            # fits use weights but the IF is residual-based, not Riesz-weighted),
            # so pass directly to TSL without de-weighting.
            inf_for_tsl = inf_func.copy()
            if est_method in ("ipw", "dr") and survey_weights is not None:
                sw = survey_weights
                nz = sw > 0
                inf_for_tsl[nz] = inf_for_tsl[nz] / sw[nz]

            if resolved_survey.uses_replicate_variance:
                from diff_diff.survey import compute_replicate_if_variance

                # Score-scale to match TSL bread (1/sum(w)^2):
                # reg: psi = w * inf_func / sum(w)
                # ipw/dr: psi = inf_func / sum(w)  (survey weights cancel)
                w_sum = float(resolved_survey.weights.sum())
                if est_method in ("ipw", "dr") and survey_weights is not None:
                    psi_rep = inf_func / w_sum
                else:
                    psi_rep = resolved_survey.weights * inf_func / w_sum
                variance, n_valid_rep = compute_replicate_if_variance(psi_rep, resolved_survey)
                se = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan
                # Store effective replicate count for df update in fit()
                self._replicate_n_valid = n_valid_rep
            else:
                from diff_diff.survey import compute_survey_vcov

                vcov_survey = compute_survey_vcov(np.ones((n, 1)), inf_for_tsl, resolved_survey)
                se = float(np.sqrt(vcov_survey[0, 0]))
        elif self._cluster_ids is not None:
            # Cluster-robust SE: sum IF within clusters, then Liang-Zeger variance
            unique_clusters = np.unique(self._cluster_ids)
            n_clusters_val = len(unique_clusters)
            if n_clusters_val < 2:
                raise ValueError(
                    f"Need at least 2 clusters for cluster-robust SEs, " f"got {n_clusters_val}"
                )
            cluster_sums = np.array(
                [np.sum(inf_func[self._cluster_ids == c]) for c in unique_clusters]
            )
            # V = (G/(G-1)) * (1/n^2) * sum(psi_c^2)
            se = float(
                np.sqrt((n_clusters_val / (n_clusters_val - 1)) * np.sum(cluster_sums**2) / n**2)
            )
        else:
            se = float(np.std(inf_func, ddof=1) / np.sqrt(n))

        # Non-finite IF values make SE undefined
        if any_nonfinite_if:
            warnings.warn(
                "Non-finite values in influence function (likely due to "
                "extreme propensity scores or near-singular design). "
                "SE set to NaN.",
                UserWarning,
                stacklevel=3,
            )
            se = np.nan

        # Consolidated rank-guard warning: a near-singular OR bread / PS Hessian
        # (constant/collinear covariate) had a redundant direction truncated by
        # the rank-guarded inverse. Surfaced once per fit rather than per cell;
        # suppressed under rank_deficient_action="silent".
        if self._lstsq_fallback_tracker and self.rank_deficient_action != "silent":
            n_cells = len(self._lstsq_fallback_tracker)
            finite_conds = [c for c in self._lstsq_fallback_tracker if np.isfinite(c)]
            max_cond = max(finite_conds) if finite_conds else float("inf")
            warnings.warn(
                f"Rank-deficient covariate design encountered {n_cells} time(s) "
                f"in the influence-function SE (outcome-regression bread or "
                f"propensity-score Hessian); dropped redundant direction(s) via "
                f"a rank-guarded inverse. Max condition number of affected "
                f"matrix: {max_cond:.2e}. Standard errors use the identified "
                f"covariate subset; consider dropping collinear covariates.",
                UserWarning,
                stacklevel=3,
            )

        # Propensity score stats (for IPW/DR with covariates)
        if has_covariates and est_method != "reg" and all_pscores:
            all_ps = np.concatenate(list(all_pscores.values()))
            pscore_stats = {
                "P(subgroup=4|X) mean": float(np.mean(all_ps)),
                "P(subgroup=4|X) std": float(np.std(all_ps)),
                "P(subgroup=4|X) min": float(np.min(all_ps)),
                "P(subgroup=4|X) max": float(np.max(all_ps)),
            }

        # R-squared for regression-based methods
        r_squared = None
        if est_method in ("reg", "dr") and has_covariates:
            # Compute R-squared from fitted values on full data
            mu_fitted = np.zeros(n)
            for sg_val in [1, 2, 3, 4]:
                for t_val in [0, 1]:
                    cell_mask = (subgroup == sg_val) & (post == t_val)
                    if np.sum(cell_mask) > 0:
                        X_fit = covX[cell_mask]
                        y_fit = y[cell_mask]
                        try:
                            beta_rs, _, _ = solve_ols(
                                X_fit,
                                y_fit,
                                rank_deficient_action=self.rank_deficient_action,
                            )
                            beta_rs = np.where(np.isnan(beta_rs), 0.0, beta_rs)
                            mu_fitted[cell_mask] = X_fit @ beta_rs
                        except (np.linalg.LinAlgError, ValueError):
                            mu_fitted[cell_mask] = np.mean(y_fit)
            ss_res = np.sum((y - mu_fitted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return att, se, r_squared, pscore_stats, epv_all

    def _fit_predict_mu(
        self,
        y: np.ndarray,
        covX: np.ndarray,
        subgroup_mask: np.ndarray,
        time_mask: np.ndarray,
        n_total: int,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit OLS (or WLS with survey weights) on a subgroup-time cell, predict for all observations."""
        fit_mask = subgroup_mask & time_mask
        n_fit = int(np.sum(fit_mask))

        if n_fit == 0:
            return np.zeros(n_total)

        X_fit = covX[fit_mask]
        y_fit = y[fit_mask]

        try:
            if weights is not None:
                # WLS: transform by sqrt(weights) for weighted least squares
                w_fit = weights[fit_mask]
                # Check positive-weight effective sample — subpopulation()
                # can zero weights leaving rows but an underidentified WLS.
                n_eff = int(np.count_nonzero(w_fit > 0))
                n_cols = X_fit.shape[1]
                if n_eff <= n_cols:
                    if np.sum(w_fit) > 0:
                        return np.full(n_total, np.average(y_fit, weights=w_fit))
                    return np.full(n_total, np.mean(y_fit))
                sqrt_w = np.sqrt(w_fit)
                beta, _, _ = solve_ols(
                    X_fit * sqrt_w[:, np.newaxis],
                    y_fit * sqrt_w,
                    rank_deficient_action=self.rank_deficient_action,
                )
            else:
                beta, _, _ = solve_ols(
                    X_fit,
                    y_fit,
                    rank_deficient_action=self.rank_deficient_action,
                )
            # Replace NaN coefficients (dropped columns) with 0 for prediction
            beta = np.where(np.isnan(beta), 0.0, beta)
        except ValueError:
            if self.rank_deficient_action == "error":
                raise
            if weights is not None:
                return np.full(n_total, np.average(y_fit, weights=weights[fit_mask]))
            return np.full(n_total, np.mean(y_fit))
        except np.linalg.LinAlgError:
            if weights is not None:
                return np.full(n_total, np.average(y_fit, weights=weights[fit_mask]))
            return np.full(n_total, np.mean(y_fit))

        # Prediction uses original-scale X (unchanged)
        return covX @ beta

    def _compute_did_rc(
        self,
        y: np.ndarray,
        post: np.ndarray,
        PA4: np.ndarray,
        PAa: np.ndarray,
        pscore: np.ndarray,
        covX: np.ndarray,
        or_ctrl_pre: np.ndarray,
        or_ctrl_post: np.ndarray,
        or_trt_pre: np.ndarray,
        or_trt_post: np.ndarray,
        hessian: Optional[np.ndarray],
        est_method: str,
        n: int,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute a single pairwise DiD (subgroup j vs 4) for RC data.

        Returns ATT and per-observation influence function.
        Matches R's triplediff::compute_did_rc().
        """
        if est_method == "ipw":
            return self._compute_did_rc_ipw(
                y,
                post,
                PA4,
                PAa,
                pscore,
                covX,
                hessian,
                n,
                weights=weights,
            )
        elif est_method == "reg":
            return self._compute_did_rc_reg(
                y,
                post,
                PA4,
                PAa,
                covX,
                or_ctrl_pre,
                or_ctrl_post,
                or_trt_pre,
                or_trt_post,
                n,
                weights=weights,
            )
        else:
            return self._compute_did_rc_dr(
                y,
                post,
                PA4,
                PAa,
                pscore,
                covX,
                or_ctrl_pre,
                or_ctrl_post,
                or_trt_pre,
                or_trt_post,
                hessian,
                n,
                weights=weights,
            )

    def _compute_did_rc_ipw(
        self,
        y: np.ndarray,
        post: np.ndarray,
        PA4: np.ndarray,
        PAa: np.ndarray,
        pscore: np.ndarray,
        covX: np.ndarray,
        hessian: Optional[np.ndarray],
        n: int,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """IPW DiD for a single pairwise comparison (RC)."""
        # Riesz representers (IPW weights * indicators)
        riesz_treat_pre = PA4 * (1 - post)
        riesz_treat_post = PA4 * post
        riesz_control_pre = pscore * PAa * (1 - post) / (1 - pscore)
        riesz_control_post = pscore * PAa * post / (1 - pscore)

        # Incorporate survey weights into Riesz representers
        if weights is not None:
            riesz_treat_pre = riesz_treat_pre * weights
            riesz_treat_post = riesz_treat_post * weights
            riesz_control_pre = riesz_control_pre * weights
            riesz_control_post = riesz_control_post * weights

        # Hajek-normalized cell-time means
        def _hajek(riesz, y_vals):
            denom = np.mean(riesz)
            if denom <= 0:
                return np.zeros_like(riesz), 0.0
            eta = riesz * y_vals / denom
            return eta, float(np.mean(eta))

        eta_treat_pre, att_treat_pre = _hajek(riesz_treat_pre, y)
        eta_treat_post, att_treat_post = _hajek(riesz_treat_post, y)
        eta_control_pre, att_control_pre = _hajek(riesz_control_pre, y)
        eta_control_post, att_control_post = _hajek(riesz_control_post, y)

        att = (att_treat_post - att_treat_pre) - (att_control_post - att_control_pre)

        # Influence function
        inf_treat_pre = eta_treat_pre - riesz_treat_pre * att_treat_pre / np.mean(riesz_treat_pre)
        inf_treat_post = eta_treat_post - riesz_treat_post * att_treat_post / np.mean(
            riesz_treat_post
        )
        inf_treat = inf_treat_post - inf_treat_pre

        inf_control_pre = eta_control_pre - riesz_control_pre * att_control_pre / np.mean(
            riesz_control_pre
        )
        inf_control_post = eta_control_post - riesz_control_post * att_control_post / np.mean(
            riesz_control_post
        )
        inf_control = inf_control_post - inf_control_pre

        # Propensity score correction for influence function
        if hessian is not None:
            score_ps = (PA4 - pscore)[:, None] * covX
            if weights is not None:
                score_ps = score_ps * weights[:, None]
            asy_lin_rep_ps = score_ps @ hessian

            # Riesz representers already incorporate survey weights,
            # so use np.mean (not np.average with weights) to avoid double-weighting.
            M2_pre = np.mean(
                (riesz_control_pre * (y - att_control_pre))[:, None] * covX,
                axis=0,
            ) / np.mean(riesz_control_pre)
            M2_post = np.mean(
                (riesz_control_post * (y - att_control_post))[:, None] * covX,
                axis=0,
            ) / np.mean(riesz_control_post)
            inf_control_ps = asy_lin_rep_ps @ (M2_post - M2_pre)
            inf_control = inf_control + inf_control_ps

        inf_func = inf_treat - inf_control
        return att, inf_func

    def _compute_did_rc_reg(
        self,
        y: np.ndarray,
        post: np.ndarray,
        PA4: np.ndarray,
        PAa: np.ndarray,
        covX: np.ndarray,
        or_ctrl_pre: np.ndarray,
        or_ctrl_post: np.ndarray,
        or_trt_pre: np.ndarray,
        or_trt_post: np.ndarray,
        n: int,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Regression adjustment DiD for a single pairwise comparison (RC)."""

        # Helper: weighted or unweighted mean
        def _wmean(x):
            if weights is not None:
                return np.average(x, weights=weights)
            return np.mean(x)

        # Riesz representers
        riesz_treat_pre = PA4 * (1 - post)
        riesz_treat_post = PA4 * post
        riesz_control = PA4  # weights for OR prediction

        # ATT components
        reg_att_treat_pre = riesz_treat_pre * y
        reg_att_treat_post = riesz_treat_post * y
        reg_att_control = riesz_control * (or_ctrl_post - or_ctrl_pre)

        eta_treat_pre = _wmean(reg_att_treat_pre) / _wmean(riesz_treat_pre)
        eta_treat_post = _wmean(reg_att_treat_post) / _wmean(riesz_treat_post)
        eta_control = _wmean(reg_att_control) / _wmean(riesz_control)

        att = (eta_treat_post - eta_treat_pre) - eta_control

        # Influence function
        # OLS asymptotic linear representation for pre/post
        # When survey weights present, include them in both score and bread
        # for consistency with the weighted OLS fit
        weights_ols_pre = PAa * (1 - post)
        if weights is not None:
            wols_x_pre = (weights_ols_pre * weights)[:, None] * covX
            wols_eX_pre = (weights_ols_pre * weights * (y - or_ctrl_pre))[:, None] * covX
            XpX_pre = wols_x_pre.T @ covX / n
        else:
            wols_x_pre = weights_ols_pre[:, None] * covX
            wols_eX_pre = (weights_ols_pre * (y - or_ctrl_pre))[:, None] * covX
            XpX_pre = wols_x_pre.T @ covX / n
        XpX_inv_pre, _, _ = _rank_guarded_inv(
            XpX_pre, tracker=getattr(self, "_lstsq_fallback_tracker", None)
        )
        asy_lin_rep_ols_pre = wols_eX_pre @ XpX_inv_pre

        weights_ols_post = PAa * post
        if weights is not None:
            wols_x_post = (weights_ols_post * weights)[:, None] * covX
            wols_eX_post = (weights_ols_post * weights * (y - or_ctrl_post))[:, None] * covX
            XpX_post = wols_x_post.T @ covX / n
        else:
            wols_x_post = weights_ols_post[:, None] * covX
            wols_eX_post = (weights_ols_post * (y - or_ctrl_post))[:, None] * covX
            XpX_post = wols_x_post.T @ covX / n
        XpX_inv_post, _, _ = _rank_guarded_inv(
            XpX_post, tracker=getattr(self, "_lstsq_fallback_tracker", None)
        )
        asy_lin_rep_ols_post = wols_eX_post @ XpX_inv_post

        inf_treat_pre = (reg_att_treat_pre - riesz_treat_pre * eta_treat_pre) / _wmean(
            riesz_treat_pre
        )
        inf_treat_post = (reg_att_treat_post - riesz_treat_post * eta_treat_post) / _wmean(
            riesz_treat_post
        )
        inf_treat = inf_treat_post - inf_treat_pre

        inf_control_1 = reg_att_control - riesz_control * eta_control
        if weights is not None:
            M1 = np.average(riesz_control[:, None] * covX, axis=0, weights=weights)
        else:
            M1 = np.mean(riesz_control[:, None] * covX, axis=0)
        inf_control_2_post = asy_lin_rep_ols_post @ M1
        inf_control_2_pre = asy_lin_rep_ols_pre @ M1
        inf_control = (inf_control_1 + inf_control_2_post - inf_control_2_pre) / _wmean(
            riesz_control
        )

        inf_func = inf_treat - inf_control
        return att, inf_func

    def _compute_did_rc_dr(
        self,
        y: np.ndarray,
        post: np.ndarray,
        PA4: np.ndarray,
        PAa: np.ndarray,
        pscore: np.ndarray,
        covX: np.ndarray,
        or_ctrl_pre: np.ndarray,
        or_ctrl_post: np.ndarray,
        or_trt_pre: np.ndarray,
        or_trt_post: np.ndarray,
        hessian: Optional[np.ndarray],
        n: int,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Doubly robust DiD for a single pairwise comparison (RC)."""
        or_ctrl = post * or_ctrl_post + (1 - post) * or_ctrl_pre

        # Riesz representers
        riesz_treat_pre = PA4 * (1 - post)
        riesz_treat_post = PA4 * post
        riesz_control_pre = pscore * PAa * (1 - post) / (1 - pscore)
        riesz_control_post = pscore * PAa * post / (1 - pscore)
        riesz_d = PA4
        riesz_dt1 = PA4 * post
        riesz_dt0 = PA4 * (1 - post)

        # Incorporate survey weights into Riesz representers
        if weights is not None:
            riesz_treat_pre = riesz_treat_pre * weights
            riesz_treat_post = riesz_treat_post * weights
            riesz_control_pre = riesz_control_pre * weights
            riesz_control_post = riesz_control_post * weights
            riesz_d = riesz_d * weights
            riesz_dt1 = riesz_dt1 * weights
            riesz_dt0 = riesz_dt0 * weights

        # DR cell-time components
        def _safe_ratio(num, denom):
            return num / denom if denom > 0 else 0.0

        eta_treat_pre = riesz_treat_pre * (y - or_ctrl) * _safe_ratio(1, np.mean(riesz_treat_pre))
        eta_treat_post = (
            riesz_treat_post * (y - or_ctrl) * _safe_ratio(1, np.mean(riesz_treat_post))
        )
        eta_control_pre = (
            riesz_control_pre * (y - or_ctrl) * _safe_ratio(1, np.mean(riesz_control_pre))
        )
        eta_control_post = (
            riesz_control_post * (y - or_ctrl) * _safe_ratio(1, np.mean(riesz_control_post))
        )

        # Efficiency correction (OR bias correction)
        eta_d_post = riesz_d * (or_trt_post - or_ctrl_post) * _safe_ratio(1, np.mean(riesz_d))
        eta_dt1_post = riesz_dt1 * (or_trt_post - or_ctrl_post) * _safe_ratio(1, np.mean(riesz_dt1))
        eta_d_pre = riesz_d * (or_trt_pre - or_ctrl_pre) * _safe_ratio(1, np.mean(riesz_d))
        eta_dt0_pre = riesz_dt0 * (or_trt_pre - or_ctrl_pre) * _safe_ratio(1, np.mean(riesz_dt0))

        att_treat_pre = float(np.mean(eta_treat_pre))
        att_treat_post = float(np.mean(eta_treat_post))
        att_control_pre = float(np.mean(eta_control_pre))
        att_control_post = float(np.mean(eta_control_post))
        att_d_post = float(np.mean(eta_d_post))
        att_dt1_post = float(np.mean(eta_dt1_post))
        att_d_pre = float(np.mean(eta_d_pre))
        att_dt0_pre = float(np.mean(eta_dt0_pre))

        att = (
            (att_treat_post - att_treat_pre)
            - (att_control_post - att_control_pre)
            + (att_d_post - att_dt1_post)
            - (att_d_pre - att_dt0_pre)
        )

        # --- Influence function ---
        # OLS asymptotic linear representations (control subgroup)
        weights_ols_pre = PAa * (1 - post)
        if weights is not None:
            wols_x_pre = (weights_ols_pre * weights)[:, None] * covX
            wols_eX_pre = (weights_ols_pre * weights * (y - or_ctrl_pre))[:, None] * covX
            XpX_pre = wols_x_pre.T @ covX / n
        else:
            wols_x_pre = weights_ols_pre[:, None] * covX
            wols_eX_pre = (weights_ols_pre * (y - or_ctrl_pre))[:, None] * covX
            XpX_pre = wols_x_pre.T @ covX / n
        XpX_inv_pre, _, _ = _rank_guarded_inv(
            XpX_pre, tracker=getattr(self, "_lstsq_fallback_tracker", None)
        )
        asy_lin_rep_ols_pre = wols_eX_pre @ XpX_inv_pre

        weights_ols_post = PAa * post
        if weights is not None:
            wols_x_post = (weights_ols_post * weights)[:, None] * covX
            wols_eX_post = (weights_ols_post * weights * (y - or_ctrl_post))[:, None] * covX
            XpX_post = wols_x_post.T @ covX / n
        else:
            wols_x_post = weights_ols_post[:, None] * covX
            wols_eX_post = (weights_ols_post * (y - or_ctrl_post))[:, None] * covX
            XpX_post = wols_x_post.T @ covX / n
        XpX_inv_post, _, _ = _rank_guarded_inv(
            XpX_post, tracker=getattr(self, "_lstsq_fallback_tracker", None)
        )
        asy_lin_rep_ols_post = wols_eX_post @ XpX_inv_post

        # OLS representations (treated subgroup)
        weights_ols_pre_treat = PA4 * (1 - post)
        if weights is not None:
            wols_x_pre_treat = (weights_ols_pre_treat * weights)[:, None] * covX
            wols_eX_pre_treat = (weights_ols_pre_treat * weights * (y - or_trt_pre))[:, None] * covX
            XpX_pre_treat = wols_x_pre_treat.T @ covX / n
        else:
            wols_x_pre_treat = weights_ols_pre_treat[:, None] * covX
            wols_eX_pre_treat = (weights_ols_pre_treat * (y - or_trt_pre))[:, None] * covX
            XpX_pre_treat = wols_x_pre_treat.T @ covX / n
        XpX_inv_pre_treat, _, _ = _rank_guarded_inv(
            XpX_pre_treat, tracker=getattr(self, "_lstsq_fallback_tracker", None)
        )
        asy_lin_rep_ols_pre_treat = wols_eX_pre_treat @ XpX_inv_pre_treat

        weights_ols_post_treat = PA4 * post
        if weights is not None:
            wols_x_post_treat = (weights_ols_post_treat * weights)[:, None] * covX
            wols_eX_post_treat = (weights_ols_post_treat * weights * (y - or_trt_post))[
                :, None
            ] * covX
            XpX_post_treat = wols_x_post_treat.T @ covX / n
        else:
            wols_x_post_treat = weights_ols_post_treat[:, None] * covX
            wols_eX_post_treat = (weights_ols_post_treat * (y - or_trt_post))[:, None] * covX
            XpX_post_treat = wols_x_post_treat.T @ covX / n
        XpX_inv_post_treat, _, _ = _rank_guarded_inv(
            XpX_post_treat, tracker=getattr(self, "_lstsq_fallback_tracker", None)
        )
        asy_lin_rep_ols_post_treat = wols_eX_post_treat @ XpX_inv_post_treat

        # Propensity score linear representation
        score_ps = (PA4 - pscore)[:, None] * covX
        if weights is not None:
            score_ps = score_ps * weights[:, None]
        if hessian is not None:
            asy_lin_rep_ps = score_ps @ hessian
        else:
            asy_lin_rep_ps = np.zeros_like(score_ps)

        # Treat influence function components
        m_riesz_treat_pre = np.mean(riesz_treat_pre)
        m_riesz_treat_post = np.mean(riesz_treat_post)

        inf_treat_pre = (
            (eta_treat_pre - riesz_treat_pre * att_treat_pre / m_riesz_treat_pre)
            if m_riesz_treat_pre > 0
            else np.zeros(n)
        )
        inf_treat_post = (
            (eta_treat_post - riesz_treat_post * att_treat_post / m_riesz_treat_post)
            if m_riesz_treat_post > 0
            else np.zeros(n)
        )

        # OR correction for treated
        # Riesz representers already incorporate survey weights,
        # so use np.mean (not weighted average) to avoid double-weighting.
        M1_post = (
            (-np.mean((riesz_treat_post * post)[:, None] * covX, axis=0) / m_riesz_treat_post)
            if m_riesz_treat_post > 0
            else np.zeros(covX.shape[1])
        )
        M1_pre = (
            (-np.mean((riesz_treat_pre * (1 - post))[:, None] * covX, axis=0) / m_riesz_treat_pre)
            if m_riesz_treat_pre > 0
            else np.zeros(covX.shape[1])
        )
        inf_treat_or_post = asy_lin_rep_ols_post @ M1_post
        inf_treat_or_pre = asy_lin_rep_ols_pre @ M1_pre

        # Control influence function components
        m_riesz_control_pre = np.mean(riesz_control_pre)
        m_riesz_control_post = np.mean(riesz_control_post)

        inf_control_pre = (
            (eta_control_pre - riesz_control_pre * att_control_pre / m_riesz_control_pre)
            if m_riesz_control_pre > 0
            else np.zeros(n)
        )
        inf_control_post = (
            (eta_control_post - riesz_control_post * att_control_post / m_riesz_control_post)
            if m_riesz_control_post > 0
            else np.zeros(n)
        )

        # PS correction for control
        M2_pre = (
            (
                np.mean(
                    (riesz_control_pre * (y - or_ctrl - att_control_pre))[:, None] * covX, axis=0
                )
                / m_riesz_control_pre
            )
            if m_riesz_control_pre > 0
            else np.zeros(covX.shape[1])
        )
        M2_post = (
            (
                np.mean(
                    (riesz_control_post * (y - or_ctrl - att_control_post))[:, None] * covX, axis=0
                )
                / m_riesz_control_post
            )
            if m_riesz_control_post > 0
            else np.zeros(covX.shape[1])
        )
        inf_control_ps = asy_lin_rep_ps @ (M2_post - M2_pre)

        # OR correction for control
        M3_post = (
            (-np.mean((riesz_control_post * post)[:, None] * covX, axis=0) / m_riesz_control_post)
            if m_riesz_control_post > 0
            else np.zeros(covX.shape[1])
        )
        M3_pre = (
            (
                -np.mean((riesz_control_pre * (1 - post))[:, None] * covX, axis=0)
                / m_riesz_control_pre
            )
            if m_riesz_control_pre > 0
            else np.zeros(covX.shape[1])
        )
        inf_control_or_post = asy_lin_rep_ols_post @ M3_post
        inf_control_or_pre = asy_lin_rep_ols_pre @ M3_pre

        # Efficiency correction
        m_riesz_d = np.mean(riesz_d)
        m_riesz_dt1 = np.mean(riesz_dt1)
        m_riesz_dt0 = np.mean(riesz_dt0)

        inf_eff1 = (eta_d_post - riesz_d * att_d_post / m_riesz_d) if m_riesz_d > 0 else np.zeros(n)
        inf_eff2 = (
            (eta_dt1_post - riesz_dt1 * att_dt1_post / m_riesz_dt1)
            if m_riesz_dt1 > 0
            else np.zeros(n)
        )
        inf_eff3 = (eta_d_pre - riesz_d * att_d_pre / m_riesz_d) if m_riesz_d > 0 else np.zeros(n)
        inf_eff4 = (
            (eta_dt0_pre - riesz_dt0 * att_dt0_pre / m_riesz_dt0)
            if m_riesz_dt0 > 0
            else np.zeros(n)
        )
        inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4)

        # OR combination
        mom_post = (
            np.mean(
                (riesz_d[:, None] / m_riesz_d - riesz_dt1[:, None] / m_riesz_dt1) * covX, axis=0
            )
            if (m_riesz_d > 0 and m_riesz_dt1 > 0)
            else np.zeros(covX.shape[1])
        )
        mom_pre = (
            np.mean(
                (riesz_d[:, None] / m_riesz_d - riesz_dt0[:, None] / m_riesz_dt0) * covX, axis=0
            )
            if (m_riesz_d > 0 and m_riesz_dt0 > 0)
            else np.zeros(covX.shape[1])
        )
        inf_or_post = (asy_lin_rep_ols_post_treat - asy_lin_rep_ols_post) @ mom_post
        inf_or_pre = (asy_lin_rep_ols_pre_treat - asy_lin_rep_ols_pre) @ mom_pre

        inf_treat_or = inf_treat_or_post + inf_treat_or_pre
        inf_control_or = inf_control_or_post + inf_control_or_pre
        inf_or = inf_or_post - inf_or_pre

        inf_treat = inf_treat_post - inf_treat_pre + inf_treat_or
        inf_control = inf_control_post - inf_control_pre + inf_control_ps + inf_control_or

        inf_func = inf_treat - inf_control + inf_eff + inf_or
        return att, inf_func

    @staticmethod
    def _validate_vcov_type(vcov_type: str) -> None:
        """Validate ``vcov_type`` membership against TripleDifference's
        narrow contract.

        Called from ``__init__`` and from ``fit()`` (so sklearn-style
        ``set_params(vcov_type=...)`` mutations are re-checked at use
        time rather than silently passing a bad value through to Results).
        """
        _accepted_vcov = {"hc1"}
        _deferred_vcov = {"conley"}
        _if_incompatible_vcov = {"classical", "hc2", "hc2_bm"}
        if vcov_type in _if_incompatible_vcov:
            raise ValueError(
                f"TripleDifference(vcov_type={vcov_type!r}) is rejected: "
                "TripleDifference uses influence-function-based variance "
                "per Ortiz-Villavicencio & Sant'Anna (2025) "
                "arXiv:2505.09942; the analytical-sandwich families "
                "{'classical', 'hc2', 'hc2_bm'} are defined on a single "
                "regression's hat matrix, and TripleDifference's "
                "3-pairwise-DiD decomposition (DiD_3 + DiD_2 - DiD_1) "
                "has no equivalent single design matrix to compute "
                "hat-matrix leverage or Bell-McCaffrey Satterthwaite DOF "
                "on. The rejection is library-architectural, not "
                "paper-prescribed. Use vcov_type='hc1' (the default) with "
                "cluster=<col> for cluster-robust inference (Liang-Zeger "
                "CR1 on the combined influence function). See "
                "docs/methodology/REGISTRY.md 'IF-based variance "
                "estimators vs analytical-sandwich estimators' for the "
                "structural taxonomy."
            )
        if vcov_type in _deferred_vcov:
            raise ValueError(
                f"TripleDifference(vcov_type={vcov_type!r}) is not yet "
                "supported: spatial-HAC (Conley) on the 3-pairwise-DiD "
                "influence-function decomposition could conceptually "
                "apply (spatial aggregation of per-unit IFs) but requires "
                "separate methodology work; no reference implementation "
                "exists today. Tracked as a follow-up row in DEFERRED.md. Use "
                "vcov_type='hc1' (the default) with cluster=<col> for "
                "cluster-robust inference today."
            )
        if vcov_type not in _accepted_vcov:
            raise ValueError(
                f"TripleDifference(vcov_type={vcov_type!r}) is invalid. "
                f"Accepted values: {sorted(_accepted_vcov)}. "
                "TripleDifference is permanently narrow to 'hc1' per "
                "IF-based variance structure; see REGISTRY.md."
            )

    # get_params/set_params come from BaseEstimator.

    def summary(self) -> str:
        """
        Get summary of estimation results.

        Returns
        -------
        str
            Formatted summary.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())


# =============================================================================
# Convenience function
# =============================================================================


def triple_difference(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    partition: str,
    time: str,
    covariates: Optional[List[str]] = None,
    estimation_method: str = "dr",
    robust: Optional[bool] = None,
    cluster: Optional[str] = None,
    vcov_type: str = "hc1",
    alpha: float = 0.05,
    rank_deficient_action: str = "warn",
    epv_threshold: float = 10,
    pscore_fallback: str = "error",
    survey_design: Optional["SurveyDesign"] = None,
) -> TripleDifferenceResults:
    """
    Estimate Triple Difference (DDD) treatment effect.

    .. deprecated:: 3.9
        ``triple_difference()`` is deprecated and will be removed in 4.0
        (row M-075). Construct the estimator instead:
        ``TripleDifference(...).fit(data, ...)``.

    Convenience function that creates a TripleDifference estimator and
    fits it to the data in one step.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all variables.
    outcome : str
        Name of the outcome variable column.
    group : str
        Name of the group indicator column (0/1).
        1 = treated group (e.g., states that enacted policy).
    partition : str
        Name of the partition/eligibility indicator column (0/1).
        1 = eligible partition (e.g., women, targeted demographic).
    time : str
        Name of the time period indicator column (0/1).
        1 = post-treatment period.
    covariates : list of str, optional
        List of covariate column names to adjust for.
    estimation_method : str, default="dr"
        Estimation method: "dr" (doubly robust), "reg" (regression),
        or "ipw" (inverse probability weighting).
    robust : bool, optional
        DEPRECATED (row M-046; warns with ``FutureWarning``, removed in
        4.0 - use ``vcov_type=``). Influence-function SEs are inherently
        robust to heteroskedasticity, so the flag never had an effect
        here; it was retained only for API compatibility.
    cluster : str, optional
        Column name for cluster-robust standard errors.
    vcov_type : str, default="hc1"
        Variance estimator. Permanently narrow to ``{"hc1"}`` per the
        IF-based variance decomposition; ``classical``/``hc2``/``hc2_bm``
        are rejected at ``__init__`` and ``conley`` is deferred. See the
        ``TripleDifference`` class docstring for the structural taxonomy.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient:
        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning
    epv_threshold : float, default=10
        Events Per Variable threshold for propensity score logit.
    pscore_fallback : str, default="error"
        Action when propensity score estimation fails:
        - "error": Raise (default)
        - "unconditional": Fall back to unconditional propensity

    Returns
    -------
    TripleDifferenceResults
        Object containing estimation results.

    Examples
    --------
    >>> from diff_diff import triple_difference
    >>> results = triple_difference(
    ...     data,
    ...     outcome='earnings',
    ...     group='policy_state',
    ...     partition='female',
    ...     time='post_policy',
    ...     covariates=['age', 'education']
    ... )
    >>> print(f"ATT: {results.att:.3f} (SE: {results.se:.3f})")
    """
    warnings.warn(
        "triple_difference() is deprecated and will be removed in 4.0; "
        "construct the estimator instead: TripleDifference(...).fit(data, ...).",
        FutureWarning,
        stacklevel=2,
    )
    estimator = TripleDifference(
        estimation_method=estimation_method,
        robust=robust,
        cluster=cluster,
        vcov_type=vcov_type,
        alpha=alpha,
        rank_deficient_action=rank_deficient_action,
        epv_threshold=epv_threshold,
        pscore_fallback=pscore_fallback,
    )
    return estimator.fit(
        data=data,
        outcome=outcome,
        group=group,
        partition=partition,
        post=time,
        covariates=covariates,
        survey_design=survey_design,
    )
