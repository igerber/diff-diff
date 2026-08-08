"""
Staggered Triple Difference (DDD) estimator.

Implements Ortiz-Villavicencio & Sant'Anna (2025) for staggered adoption
settings with an eligibility dimension, combining group-time DDD effects
via GMM-optimal weighting.

Core pairwise DiD computation matches R's triplediff::compute_did() exactly
(Riesz/Hajek normalization, separate M1/M3 OR corrections, hessian = (X'WX)^{-1}*n).

The estimation engine itself lives in `_staggered_triple_diff_engine.py`, shared
verbatim with `TripleDifference`'s staggered mode (ledger row M-013). This module
keeps the deprecated class's 3.x surface: its own constructor (including R's
compact `control_group` spellings) and its own `fit` signature (including the
`eligibility` parameter name), both frozen until the 4.0 removal.
"""

import warnings
from typing import TYPE_CHECKING, ClassVar, List, Optional

from diff_diff._base import BaseEstimator
from diff_diff._staggered_triple_diff_engine import _StaggeredTripleDiffEngineMixin
from diff_diff.staggered_aggregation import (
    CallawaySantAnnaAggregationMixin,
)
from diff_diff.staggered_bootstrap import (
    CallawaySantAnnaBootstrapMixin,
)
from diff_diff.staggered_triple_diff_results import StaggeredTripleDiffResults
from diff_diff.utils import validate_n_bootstrap

if TYPE_CHECKING:
    import pandas as pd

    from diff_diff.survey import SurveyDesign

__all__ = [
    "StaggeredTripleDifference",
    "StaggeredTripleDiffResults",
]

# StaggeredTripleDifference 3.9 deprecation message (row M-013; the SDDD alias
# is the same class object, so constructing via the alias emits this too -
# row M-064). Pinned verbatim by tests/test_v4_merge_ddd.py and the targeted
# pytest filter in pyproject.toml.
_SDDD_DEPRECATION_MSG = (
    "StaggeredTripleDifference is deprecated and will be removed in 4.0; use "
    "TripleDifference().fit(..., unit=, time=, first_treat=, partition=) "
    "instead - the same engine, so the numbers are unchanged. Two vocabulary "
    "changes on the merged surface: the eligibility= parameter is named "
    "partition=, and control_group takes the underscored values "
    "'not_yet_treated'/'never_treated'. The SDDD alias is deprecated with it."
)


class StaggeredTripleDifference(
    _StaggeredTripleDiffEngineMixin,
    CallawaySantAnnaBootstrapMixin,
    CallawaySantAnnaAggregationMixin,
    BaseEstimator,
):
    """
    Staggered Triple Difference (DDD) estimator.

    .. deprecated:: 3.9
        Deprecated in 3.9 and removed in 4.0 (ledger row M-013). Use
        ``TripleDifference().fit(..., unit=, time=, first_treat=, partition=)``,
        which runs this exact engine. The ``eligibility=`` parameter is named
        ``partition=`` there, and ``control_group`` takes the underscored
        values ``"not_yet_treated"``/``"never_treated"``. The ``SDDD`` alias is
        deprecated with the class (row M-064).

    Computes group-time average treatment effects ATT(g,t) for settings
    with staggered adoption and a binary eligibility dimension, using the
    three-DiD decomposition of Ortiz-Villavicencio & Sant'Anna (2025).

    Multiple comparison groups are combined via GMM-optimal (inverse-variance)
    weighting. Event study, group, and overall aggregations are supported.

    Parameters
    ----------
    estimation_method : str, default="dr"
        Estimation method: "dr" (doubly robust), "ipw" (inverse probability
        weighting), or "reg" (regression adjustment).
    alpha : float, default=0.05
        Significance level.
    anticipation : int, default=0
        Number of anticipation periods.
    base_period : str, default="varying"
        Base period selection: "varying" (consecutive comparisons) or
        "universal" (always vs g-1-anticipation).
    n_bootstrap : int, default=0
        Number of multiplier bootstrap repetitions. 0 disables bootstrap.
    bootstrap_weights : str, default="rademacher"
        Bootstrap weight distribution: "rademacher", "mammen", or "webb".
    seed : int or None, default=None
        Random seed for reproducibility.
    cband : bool, default=True
        Whether to compute simultaneous confidence bands.
    pscore_trim : float, default=0.01
        Propensity score trimming bound.
    cluster : str or None, default=None
        Column name for cluster-robust standard errors.
    rank_deficient_action : str, default="warn"
        Action for rank-deficient design matrices: "warn", "error", "silent".
    epv_threshold : float, default=10
        Minimum events per variable for propensity score logistic regression.
        A warning is emitted when EPV falls below this threshold.
    pscore_fallback : str, default="error"
        Action when propensity score estimation fails: "error" (raise) or
        "unconditional" (fall back to unconditional propensity).

    References
    ----------
    Ortiz-Villavicencio, M. & Sant'Anna, P.H.C. (2025). "Better Understanding
    Triple Differences Estimators." arXiv:2505.09942v3.
    """

    # Names this estimator in the shared bootstrap mixin's user-facing
    # warnings. The mixin is shared with the other hosts, so a hard-coded
    # literal there would misname whichever surface was actually fit.
    _BOOTSTRAP_LABEL: ClassVar[str] = "StaggeredTripleDifference"

    def __init__(
        self,
        estimation_method: str = "dr",
        control_group: str = "notyettreated",
        alpha: float = 0.05,
        anticipation: int = 0,
        base_period: str = "varying",
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        cband: bool = True,
        pscore_trim: float = 0.01,
        cluster: Optional[str] = None,
        rank_deficient_action: str = "warn",
        epv_threshold: float = 10,
        pscore_fallback: str = "error",
    ):
        # Row M-013. Emitted per construction (not once): a fitted instance
        # never re-warns, but each new one does, and set_params re-emits via
        # BaseEstimator's transactional probe re-init - the documented side
        # effect MultiPeriodDiD's shim also has.
        warnings.warn(_SDDD_DEPRECATION_MSG, FutureWarning, stacklevel=2)
        if estimation_method not in ["dr", "ipw", "reg"]:
            raise ValueError(
                f"estimation_method must be 'dr', 'ipw', or 'reg', " f"got '{estimation_method}'"
            )
        if control_group not in ["nevertreated", "notyettreated"]:
            raise ValueError(
                f"control_group must be 'nevertreated' or 'notyettreated', "
                f"got '{control_group}'"
            )
        if not (0 < pscore_trim < 0.5):
            raise ValueError(f"pscore_trim must be in (0, 0.5), got {pscore_trim}")
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
        if epv_threshold <= 0:
            raise ValueError(f"epv_threshold must be > 0, got {epv_threshold}")
        if pscore_fallback not in ["error", "unconditional"]:
            raise ValueError(
                f"pscore_fallback must be 'error' or 'unconditional', " f"got '{pscore_fallback}'"
            )

        self.estimation_method = estimation_method
        self.control_group = control_group
        self.alpha = alpha
        self.anticipation = anticipation
        self.base_period = base_period
        validate_n_bootstrap(n_bootstrap)
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.cband = cband
        self.pscore_trim = pscore_trim
        self.cluster = cluster
        self.rank_deficient_action = rank_deficient_action
        self.epv_threshold = epv_threshold
        self.pscore_fallback = pscore_fallback

        self.is_fitted_ = False
        self.results_: Optional[StaggeredTripleDiffResults] = None

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(
        self,
        data: "pd.DataFrame",
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        eligibility: str,
        covariates: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
        balance_e: Optional[int] = None,
        survey_design: Optional["SurveyDesign"] = None,
    ) -> StaggeredTripleDiffResults:
        """
        Fit the staggered triple difference estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data.
        outcome : str
            Outcome variable column name.
        unit : str
            Unit identifier column name.
        time : str
            Time period column name.
        first_treat : str
            Column with the enabling period for each unit's group.
            Use 0 or np.inf for never-enabled units.
        eligibility : str
            Binary eligibility indicator column (0/1, time-invariant).
        covariates : list of str, optional
            Covariate column names.
        aggregate : str, optional
            Aggregation method: "event_study", "group", "simple", or "all".
        balance_e : int, optional
            Event time to balance on for event study.
        survey_design : SurveyDesign, optional
            Survey design specification for complex survey data. When
            provided, uses survey weights for estimation (weighted Riesz
            representers, weighted logit, weighted OLS) and design-based
            variance for aggregated SEs (overall, event study, group) via
            Taylor Series Linearization or replicate weights. Requires
            ``weight_type='pweight'``.

        Returns
        -------
        StaggeredTripleDiffResults
        """
        # The engine is shared with TripleDifference's staggered mode; this
        # class supplies its own vocabulary and its own frame depth
        # (user -> fit -> core), so warnings attribute exactly as in 3.x.
        return self._fit_staggered_core(
            data,
            outcome,
            unit,
            time,
            first_treat,
            eligibility,
            covariates=covariates,
            aggregate=aggregate,
            balance_e=balance_e,
            survey_design=survey_design,
            estimator_name="StaggeredTripleDifference",
            partition_label="eligibility",
            _frame_offset=1,
        )
