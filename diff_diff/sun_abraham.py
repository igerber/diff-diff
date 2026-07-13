"""
Sun-Abraham Interaction-Weighted Estimator for staggered DiD.

Implements the estimator from Sun & Abraham (2021), "Estimating dynamic
treatment effects in event studies with heterogeneous treatment effects",
Journal of Econometrics.

This provides an alternative to Callaway-Sant'Anna using a saturated
regression with cohort × relative-time interactions.
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd

from diff_diff.bootstrap_utils import compute_effect_bootstrap_stats

if TYPE_CHECKING:
    from diff_diff.survey import ResolvedSurveyDesign, SurveyDesign
from diff_diff.linalg import LinearRegression
from diff_diff.results import _format_survey_block, _get_significance_stars
from diff_diff.utils import (
    pre_demean_norms,
    safe_inference,
    snap_absorbed_regressors,
)
from diff_diff.utils import (
    within_transform as _within_transform_util,
)


@dataclass
class SunAbrahamResults:
    """
    Results from Sun-Abraham (2021) interaction-weighted estimation.

    Attributes
    ----------
    event_study_effects : dict
        Dictionary mapping relative time to effect dictionaries with keys:
        'effect', 'se', 't_stat', 'p_value', 'conf_int', 'n_groups'.
    overall_att : float
        Overall average treatment effect (weighted average of post-treatment effects).
    overall_se : float
        Standard error of overall ATT.
    overall_t_stat : float
        T-statistic for overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    cohort_weights : dict
        Dictionary mapping relative time to cohort weight dictionaries.
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
    alpha : float
        Significance level used for confidence intervals.
    control_group : str
        Type of control group used.
    vcov_type : str
        Variance-covariance family from the fit-time configuration
        (``classical``, ``hc1``, ``hc2``, ``hc2_bm``, or ``conley``). On the
        ``"conley"`` (spatial-HAC) path, ``conley_lag_cutoff`` and
        ``cluster_name`` are populated. Note: when a
        ``survey_design=`` is supplied, the survey-design Taylor Series
        Linearization (or replicate-weight refit) variance overrides
        this analytical family — the field still records the
        configured value but ``survey_metadata`` indicates the survey
        path was active. Likewise, on bootstrap fits (``n_bootstrap >
        0``) the SE comes from the pairs bootstrap (or Rao-Wu rescaled
        bootstrap under stratified / PSU survey designs), not the
        analytical family.
    """

    event_study_effects: Dict[int, Dict[str, Any]]
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]
    cohort_weights: Dict[int, Dict[Any, float]]
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_units: int
    n_control_units: int
    alpha: float = 0.05
    control_group: str = "never_treated"
    vcov_type: str = "hc1"
    # Anticipation periods (``k``) used at fit time. Persisted so
    # downstream diagnostics (``BusinessReport`` / ``DiagnosticReport``
    # / ``compute_pretrends_power``) can classify pre-period vs
    # anticipation-window coefficients without re-plumbing the kwarg
    # through every caller.
    anticipation: int = 0
    bootstrap_results: Optional["SABootstrapResults"] = field(default=None, repr=False)
    cohort_effects: Optional[Dict[Tuple[Any, int], Dict[str, Any]]] = field(
        default=None, repr=False
    )
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)
    # Full event-study VCV matrix (PR-B 2026-05-17 for PreTrendsPower
    # canonical Σ_22 fidelity). Built via W @ vcov_cohort @ W.T where W
    # is the |event_times| × n_interactions cohort-aggregation matrix.
    # Set to None for bootstrap fits (analytical VCV is invalidated by
    # bootstrap SE overrides) and for replicate-weight survey fits
    # (analytical vcov_cohort is overridden by replicate refit variance).
    # Consumed by ``compute_pretrends_power`` to route SA through the full
    # pre-period sub-Σ_22 block. Index keys mirror the relative-time labels
    # in ``event_study_vcov_index``.
    event_study_vcov: Optional["np.ndarray"] = field(default=None, repr=False)
    event_study_vcov_index: Optional[list] = field(default=None, repr=False)
    # Conley spatial-HAC metadata (populated only when vcov_type == "conley").
    # ``conley_lag_cutoff`` carries the within-unit Bartlett max lag; ``cluster_name``
    # records an explicit cluster= column (enables the spatial+cluster product-kernel
    # summary label). Both None on non-conley fits.
    conley_lag_cutoff: Optional[int] = None
    cluster_name: Optional[str] = None

    # --- Inference-field aliases (balance/external-adapter compatibility) ---
    @property
    def att(self) -> float:
        return self.overall_att

    @property
    def se(self) -> float:
        return self.overall_se

    @property
    def conf_int(self) -> Tuple[float, float]:
        return self.overall_conf_int

    @property
    def p_value(self) -> float:
        return self.overall_p_value

    @property
    def t_stat(self) -> float:
        return self.overall_t_stat

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        n_rel_periods = len(self.event_study_effects)
        return (
            f"SunAbrahamResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_groups={len(self.groups)}, "
            f"n_rel_periods={n_rel_periods})"
        )

    @property
    def coef_var(self) -> float:
        """Coefficient of variation: SE / abs(overall ATT). NaN when ATT is 0 or SE non-finite."""
        if not (np.isfinite(self.overall_se) and self.overall_se >= 0):
            return np.nan
        if not np.isfinite(self.overall_att) or self.overall_att == 0:
            return np.nan
        return self.overall_se / abs(self.overall_att)

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
            "Sun-Abraham Interaction-Weighted Estimator Results".center(85),
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

        # Add survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 85))

        # Conley spatial-HAC variance label (rendered only on the conley path;
        # a full vcov-family label for all families is a separate follow-up).
        if self.vcov_type == "conley":
            from diff_diff.results import _format_vcov_label

            _vlabel = _format_vcov_label(
                self.vcov_type,
                cluster_name=self.cluster_name,
                n_clusters=None,
                n_obs=self.n_obs,
                conley_lag_cutoff=self.conley_lag_cutoff,
            )
            if _vlabel:
                lines.extend([f"Std. errors: {_vlabel}", ""])

        # Overall ATT
        lines.extend(
            [
                "-" * 85,
                "Overall Average Treatment Effect on the Treated".center(85),
                "-" * 85,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
                f"{'ATT':<15} {self.overall_att:>12.4f} {self.overall_se:>12.4f} "
                f"{self.overall_t_stat:>10.3f} {self.overall_p_value:>10.4f} "
                f"{_get_significance_stars(self.overall_p_value):>6}",
                "-" * 85,
                "",
                f"{conf_level}% Confidence Interval: "
                f"[{self.overall_conf_int[0]:.4f}, {self.overall_conf_int[1]:.4f}]",
            ]
        )

        cv = self.coef_var
        if np.isfinite(cv):
            lines.append(f"{'CV (SE/abs(ATT)):':<25} {cv:>10.4f}")

        lines.append("")

        # Event study effects
        lines.extend(
            [
                "-" * 85,
                "Event Study (Dynamic) Effects".center(85),
                "-" * 85,
                f"{'Rel. Period':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
            ]
        )

        for rel_t in sorted(self.event_study_effects.keys()):
            eff = self.event_study_effects[rel_t]
            sig = _get_significance_stars(eff["p_value"])
            lines.append(
                f"{rel_t:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                f"{eff['t_stat']:>10.3f} {eff['p_value']:>10.4f} {sig:>6}"
            )

        lines.extend(["-" * 85, ""])

        lines.extend(
            [
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                "=" * 85,
            ]
        )

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print summary to stdout."""
        print(self.summary(alpha))

    def to_dataframe(self, level: str = "event_study") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="event_study"
            Level of aggregation: "event_study" or "cohort".

        Returns
        -------
        pd.DataFrame
            Results as DataFrame.
        """
        if level == "event_study":
            rows = []
            for rel_t, data in sorted(self.event_study_effects.items()):
                rows.append(
                    {
                        "relative_period": rel_t,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                    }
                )
            return pd.DataFrame(rows)

        elif level == "cohort":
            if self.cohort_effects is None:
                raise ValueError(
                    "Cohort-level effects not available. "
                    "They are computed internally but not stored by default."
                )
            rows = []
            for (cohort, rel_t), data in sorted(self.cohort_effects.items()):
                rows.append(
                    {
                        "cohort": cohort,
                        "relative_period": rel_t,
                        "effect": data["effect"],
                        "se": data["se"],
                        "weight": data.get("weight", np.nan),
                    }
                )
            return pd.DataFrame(rows)

        else:
            raise ValueError(f"Unknown level: {level}. Use 'event_study' or 'cohort'.")

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)


@dataclass
class SABootstrapResults:
    """
    Results from Sun-Abraham bootstrap inference.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap used (always "pairs" for pairs bootstrap).
    alpha : float
        Significance level used for confidence intervals.
    overall_att_se : float
        Bootstrap standard error for overall ATT.
    overall_att_ci : Tuple[float, float]
        Bootstrap confidence interval for overall ATT.
    overall_att_p_value : float
        Bootstrap p-value for overall ATT.
    event_study_ses : Dict[int, float]
        Bootstrap SEs for event study effects.
    event_study_cis : Dict[int, Tuple[float, float]]
        Bootstrap CIs for event study effects.
    event_study_p_values : Dict[int, float]
        Bootstrap p-values for event study effects.
    bootstrap_distribution : Optional[np.ndarray]
        Full bootstrap distribution of overall ATT.
    """

    n_bootstrap: int
    weight_type: str
    alpha: float
    overall_att_se: float
    overall_att_ci: Tuple[float, float]
    overall_att_p_value: float
    event_study_ses: Dict[int, float]
    event_study_cis: Dict[int, Tuple[float, float]]
    event_study_p_values: Dict[int, float]
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)


class SunAbraham:
    """
    Sun-Abraham (2021) interaction-weighted estimator for staggered DiD.

    This estimator provides event-study coefficients using a saturated
    TWFE regression with cohort × relative-time interactions, following
    the methodology in Sun & Abraham (2021).

    The estimation procedure follows three steps:
    1. Run a saturated TWFE regression with cohort × relative-time dummies
    2. Compute cohort shares (weights) at each relative time
    3. Aggregate cohort-specific effects using interaction weights

    This avoids the negative weighting problem of standard TWFE and provides
    consistent event-study estimates under treatment effect heterogeneity.

    Parameters
    ----------
    control_group : str, default="never_treated"
        Which units to use as controls:
        - "never_treated": Use only never-treated units (recommended)
        - "not_yet_treated": Use never-treated and not-yet-treated units
    anticipation : int, default=0
        Number of periods before treatment where effects may occur.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional
        Column name for cluster-robust standard errors.
        If None, clusters at the unit level by default — UNLESS
        ``vcov_type`` is explicitly set to ``"hc2"`` or ``"classical"``,
        in which case the unit auto-cluster is dropped (both are
        one-way families and the linalg validator rejects them with
        ``cluster_ids``). Use ``vcov_type="hc1"`` (default) or
        ``vcov_type="hc2_bm"`` for cluster-robust inference; the latter
        routes to CR2 Bell-McCaffrey at the cluster level.
    n_bootstrap : int, default=0
        Number of bootstrap iterations for inference.
        If 0, uses analytical cluster-robust standard errors.
    seed : int, optional
        Random seed for reproducibility.
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient (linearly dependent columns):
        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning
    vcov_type : {"classical", "hc1", "hc2", "hc2_bm", "conley"}, default "hc1"
        Variance-covariance family for analytical inference. Defaults to
        ``"hc1"`` (preserves prior behavior bit-equally; SA historically
        hard-coded HC1). ``"conley"`` (Conley 1999 spatial-HAC) threads the
        ``conley_*`` params through the within-transform saturated regression
        (``conley_lag_cutoff=0`` = within-period spatial only; ``conley_lag_cutoff>0``
        adds the within-unit Bartlett serial term — note ``conley_time`` / ``conley_unit``
        are always supplied, so this is the panel-aware path, not pooled cross-sectional);
        the unit auto-cluster is dropped (an explicit
        ``cluster=`` enables the spatial+cluster product kernel) and
        ``survey_design=`` / ``weights`` / ``n_bootstrap>0`` are rejected.

        - ``"classical"``: homoskedastic OLS standard errors. One-way
          only (linalg validator rejects ``classical + cluster_ids``);
          the unit auto-cluster is dropped when ``classical`` is
          explicitly opted into.
        - ``"hc1"``: Eicker-Huber-White HC1 finite-sample correction
          (default; cluster-robust when ``cluster=`` is set or the unit
          auto-cluster fires).
        - ``"hc2"``: Eicker-Huber-White HC2 leverage correction. One-way
          only; the linalg validator rejects combining ``hc2`` with
          clusters. The unit auto-cluster is dropped when ``hc2`` is
          explicitly opted into.
        - ``"hc2_bm"``: HC2 + Bell-McCaffrey CR2 Satterthwaite DOF for
          cluster-robust inference. Routes to CR2-BM at the cluster
          level; preserves the auto-cluster default.

        When ``vcov_type ∈ {"classical","hc2","hc2_bm"}``, the
        saturated regression switches from the within-transform path
        to a full-dummy ``[intercept + interactions + covariates +
        unit_dummies + time_dummies]`` build. For ``hc2`` and
        ``hc2_bm``, the Frisch-Waugh-Lovell theorem preserves
        coefficients but NOT the hat matrix, so HC2 leverage and BM
        Satterthwaite DOF must be computed on the full FE projection.
        ``classical`` also routes through full-dummy so the ``(n-k)``
        finite-sample correction in ``s² × (X'X)^{-1}`` matches R's
        ``lm()`` interpretation. Empirically matches
        ``lm(...) + sandwich::vcovHC(type="HC2")`` and
        ``clubSandwich::vcovCR(..., type="CR2")`` at atol=1e-10.

        ``"hc1"`` keeps the within-transform path (cluster-robust HC1
        does not depend on the hat matrix); empirically close to
        ``fixest::sunab(cluster=~unit)``. See REGISTRY.md for the
        documented HC1 finite-sample-correction deviation.

        Survey designs (``survey_design=``) are rejected for
        ``vcov_type ∈ {"classical","hc2","hc2_bm"}`` because the
        survey-design Taylor Series Linearization (or replicate-weight
        refit) variance overrides the analytical sandwich family, and
        the auto-cluster guard for one-way families would silently
        downgrade unit-level PSUs to per-observation PSUs. Use
        ``vcov_type="hc1"`` (default) for survey designs.

        ``conley`` (Conley-1999 spatial-HAC) is threaded through the
        within-transform saturated regression (pass ``conley_coords`` /
        ``conley_cutoff_km`` / ``conley_lag_cutoff``); ``survey_design=`` /
        ``weights`` / ``n_bootstrap>0`` are rejected. See the ``vcov_type``
        parameter docs above.

    Attributes
    ----------
    results_ : SunAbrahamResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> from diff_diff import SunAbraham
    >>>
    >>> # Panel data with staggered treatment
    >>> data = pd.DataFrame({
    ...     'unit': [...],
    ...     'time': [...],
    ...     'outcome': [...],
    ...     'first_treat': [...]  # 0 for never-treated
    ... })
    >>>
    >>> sa = SunAbraham()
    >>> results = sa.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat')
    >>> results.print_summary()

    With covariates:

    >>> sa = SunAbraham()
    >>> results = sa.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat',
    ...                  covariates=['age', 'income'])

    Notes
    -----
    The Sun-Abraham estimator uses a saturated regression approach:

    Y_it = α_i + λ_t + Σ_g Σ_e [δ_{g,e} × 1(G_i=g) × D_{it}^e] + X'γ + ε_it

    where:
    - α_i = unit fixed effects
    - λ_t = time fixed effects
    - G_i = unit i's treatment cohort (first treatment period)
    - D_{it}^e = indicator for being e periods from treatment
    - δ_{g,e} = cohort-specific effect (CATT) at relative time e

    The event-study coefficients are then computed as:

    β_e = Σ_g w_{g,e} × δ_{g,e}

    where w_{g,e} is the share of cohort g in the treated population at
    relative time e (interaction weights).

    Compared to Callaway-Sant'Anna:
    - SA uses saturated regression; CS uses 2x2 DiD comparisons
    - SA can be more efficient when model is correctly specified
    - Both are consistent under heterogeneous treatment effects
    - Running both provides a useful robustness check

    References
    ----------
    Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in
    event studies with heterogeneous treatment effects. Journal of
    Econometrics, 225(2), 175-199.
    """

    def __init__(
        self,
        control_group: str = "never_treated",
        anticipation: int = 0,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        vcov_type: str = "hc1",
        conley_coords: Optional[Tuple[str, str]] = None,
        conley_cutoff_km: Optional[float] = None,
        conley_metric: str = "haversine",
        conley_kernel: str = "bartlett",
        conley_lag_cutoff: Optional[int] = None,
    ):
        if control_group not in ["never_treated", "not_yet_treated"]:
            raise ValueError(
                f"control_group must be 'never_treated' or 'not_yet_treated', "
                f"got '{control_group}'"
            )

        if rank_deficient_action not in ["warn", "error", "silent"]:
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )

        if vcov_type not in ("classical", "hc1", "hc2", "hc2_bm", "conley"):
            raise ValueError(
                f"vcov_type must be one of "
                f"{{'classical','hc1','hc2','hc2_bm','conley'}}; got '{vcov_type}'"
            )

        self.control_group = control_group
        self.anticipation = anticipation
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.vcov_type = vcov_type
        self.conley_coords = conley_coords
        self.conley_cutoff_km = conley_cutoff_km
        self.conley_metric = conley_metric
        self.conley_kernel = conley_kernel
        self.conley_lag_cutoff = conley_lag_cutoff
        # Track whether the user explicitly opted out of the "hc1" default.
        # The auto-cluster-at-unit default in `fit` is suppressed only when
        # the user explicitly opts into a one-way family — currently
        # ``vcov_type in {"hc2","classical"}``. Both are rejected by the
        # linalg validator when combined with ``cluster_ids``. Leaving the
        # auto-cluster on the default "hc1" path preserves backward compat;
        # ``hc2_bm`` also keeps the auto-cluster (routes to CR2-BM at unit).
        self._vcov_type_explicit = vcov_type != "hc1"

        self.is_fitted_ = False
        self.results_: Optional[SunAbrahamResults] = None
        self._reference_period = -1  # Will be set during fit

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        survey_design: Optional["SurveyDesign"] = None,
    ) -> SunAbrahamResults:
        """
        Fit the Sun-Abraham estimator using saturated regression.

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
            List of covariate column names to include in regression.
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference.
            Supports weighted estimation and Taylor series linearization
            variance with strata, PSU, and FPC.

        Returns
        -------
        SunAbrahamResults
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

        # Validate explicit cluster column upfront. Without this guard, a
        # missing `cluster=` column would cascade through cluster_var=None
        # and silently downgrade clustered inference to one-way (HC1 →
        # heteroskedasticity-only; HC2-BM → singleton CR2-BM). Explicit
        # user input must error, not silently weaken the SE convention.
        if self.cluster is not None:
            if self.cluster not in data.columns:
                raise ValueError(
                    f"cluster column {self.cluster!r} not found in data; "
                    f"available columns: {list(data.columns)}"
                )
            # NA cluster labels are silently dropped by the meat-side
            # `groupby(cluster_ids)` but counted by `np.unique(cluster_ids)`
            # in `n_clusters`, producing malformed cluster-robust SEs. Reject
            # explicitly so the user fixes the cluster column rather than
            # consuming silently-wrong inference.
            if data[self.cluster].isna().any():
                n_na = int(data[self.cluster].isna().sum())
                raise ValueError(
                    f"cluster column {self.cluster!r} contains {n_na} "
                    "NA/NaN values. Cluster labels must be non-missing for "
                    "all observations to produce well-formed cluster-robust "
                    "standard errors. Drop or impute the NA rows before fit."
                )

        # Conley spatial-HAC front-door validation + bootstrap incompatibility.
        # The shared validator gates coords/cutoff/unit/lag/cluster columns and
        # rejects conley + survey_design (deferred). SA has no `inference=` param,
        # so pass the literal "analytical"; the n_bootstrap override is gated
        # separately below (the validator only knows about wild_bootstrap).
        if self.vcov_type == "conley":
            from diff_diff.conley import _validate_conley_estimator_inputs

            _validate_conley_estimator_inputs(
                estimator_name="SunAbraham",
                data=data,
                unit=unit,
                conley_coords=self.conley_coords,
                conley_cutoff_km=self.conley_cutoff_km,
                conley_lag_cutoff=self.conley_lag_cutoff,
                survey_design=survey_design,
                inference="analytical",
                cluster=self.cluster,
            )
            if self.n_bootstrap > 0:
                raise ValueError(
                    "SunAbraham(vcov_type='conley') is incompatible with "
                    "n_bootstrap > 0: the pairs bootstrap overrides the "
                    "analytical Conley sandwich. Use n_bootstrap=0 for the "
                    "analytical Conley SE, or vcov_type='hc1' with the bootstrap."
                )

        # Resolve survey design if provided
        from diff_diff.survey import (
            _resolve_effective_cluster,
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        # Validate survey columns are constant within units (required for
        # unit-level collapse in Rao-Wu bootstrap)
        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)

        _uses_replicate_sa = resolved_survey is not None and resolved_survey.uses_replicate_variance
        if _uses_replicate_sa and self.n_bootstrap > 0:
            raise ValueError(
                "Cannot use n_bootstrap > 0 with replicate-weight survey designs. "
                "Replicate weights provide their own variance estimation."
            )

        # Survey-design + non-HC1 analytical family reject: survey-design
        # Taylor Series Linearization (or replicate-weight refit) variance
        # overrides the analytical sandwich family, so the requested
        # vcov_type ∈ {classical, hc2, hc2_bm} would either silently downgrade
        # unit-as-PSU injection to per-observation PSUs (auto-cluster guard
        # drops cluster_var=None before the survey path injects unit as PSU)
        # or hit the linalg validator's hc2/classical + cluster_ids reject.
        # Explicit reject preserves the "survey TSL overrides analytical"
        # contract documented in REGISTRY. Use vcov_type='hc1' (default) for
        # survey designs.
        if resolved_survey is not None and self.vcov_type in ("classical", "hc2", "hc2_bm"):
            raise NotImplementedError(
                f"SunAbraham(vcov_type={self.vcov_type!r}) with survey_design "
                "is not yet supported: the survey-design TSL (or replicate-"
                "weight refit) variance overrides the analytical sandwich, "
                "so the requested HC2/HC2-BM/classical family would be "
                "silently discarded. Additionally, the auto-cluster guard "
                "for explicit one-way families (classical/hc2) would drop "
                "the unit auto-cluster before survey-PSU injection, "
                "downgrading the panel structure from unit-level to "
                "per-observation PSUs. Use vcov_type='hc1' (default) for "
                "survey designs; the survey TSL machinery computes the "
                "design-aware SE on the within-transform path."
            )

        # Note: the broader survey reject above (line ~625) already covers
        # the replicate-weight + hc2/hc2_bm combo (replicate is a subset of
        # survey). The replicate-only reject that previously lived here is
        # redundant and was removed; see commit history for the rationale.

        # Bootstrap + survey supported via Rao-Wu rescaled bootstrap.
        # Determine Rao-Wu eligibility from the *original* survey_design
        # (before cluster-as-PSU injection which adds PSU to weights-only designs).
        _use_rao_wu = False
        if survey_design is not None and resolved_survey is not None:
            _has_explicit_strata = getattr(survey_design, "strata", None) is not None
            _has_explicit_psu = getattr(survey_design, "psu", None) is not None
            _has_explicit_fpc = getattr(survey_design, "fpc", None) is not None
            if _has_explicit_strata or _has_explicit_psu or _has_explicit_fpc:
                _use_rao_wu = True

        # Create working copy
        df = data.copy()

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Never-treated indicator (must precede treatment_groups to exclude np.inf)
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)
        # Normalize np.inf → 0 so all downstream `> 0` checks exclude never-treated
        df.loc[df[first_treat] == np.inf, first_treat] = 0

        # Identify groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0])

        # Get unique units
        unit_info = (
            df.groupby(unit).agg({first_treat: "first", "_never_treated": "first"}).reset_index()
        )

        n_treated_units = int((unit_info[first_treat] > 0).sum())
        n_control_units = int((unit_info["_never_treated"]).sum())

        if n_control_units == 0:
            raise ValueError("No never-treated units found. Check 'first_treat' column.")

        if len(treatment_groups) == 0:
            raise ValueError("No treated units found. Check 'first_treat' column.")

        # Compute relative time for each observation (vectorized)
        df["_rel_time"] = np.where(df[first_treat] > 0, df[time] - df[first_treat], np.nan)

        # Identify the range of relative time periods to estimate
        rel_times_by_cohort = {}
        for g in treatment_groups:
            g_times = df[df[first_treat] == g][time].unique()
            rel_times_by_cohort[g] = sorted([t - g for t in g_times])

        # Find all relative time values
        all_rel_times: set = set()
        for g, rel_times in rel_times_by_cohort.items():
            all_rel_times.update(rel_times)

        all_rel_times_sorted = sorted(all_rel_times)

        # Use full range of relative times (no artificial truncation, matches R's fixest::sunab())
        min_rel = min(all_rel_times_sorted)
        max_rel = max(all_rel_times_sorted)

        # Reference period: last pre-treatment period (typically -1)
        self._reference_period = -1 - self.anticipation

        # Get relative periods to estimate (excluding reference)
        rel_periods_to_estimate = [
            e
            for e in all_rel_times_sorted
            if min_rel <= e <= max_rel and e != self._reference_period
        ]

        # Determine cluster variable
        # One-way HC2 and classical are single-way only — the linalg
        # validator rejects `vcov_type ∈ {"hc2","classical"} + cluster_ids`.
        # Drop the unit auto-cluster when the user opts into either
        # explicitly. `hc1` and `hc2_bm` preserve the auto-cluster
        # (route to CR1 / CR2-Bell-McCaffrey at unit respectively).
        # SA has no `inference=` parameter — its bootstrap path uses the
        # pairs bootstrap (or Rao-Wu rescaled bootstrap on stratified /
        # PSU survey designs) via `n_bootstrap > 0`, which overrides the
        # analytical SE downstream and does NOT consume the cluster
        # structure of the main fit. So the SA guard simplifies to
        # "explicit-vcov-only", without TWFE's `inference == "analytical"`
        # subguard.
        if self.cluster is not None:
            cluster_var: Optional[str] = self.cluster
        elif self.vcov_type == "conley":
            # Conley: never auto-cluster at unit. A unit-cluster product kernel
            # would zero every between-unit spatial pair, collapsing the spatial
            # pooling. Only an explicit cluster= enables the combined
            # spatial+cluster product kernel (handled by the branch above).
            cluster_var = None
        elif self.vcov_type in ("hc2", "classical") and self._vcov_type_explicit:
            cluster_var = None
        else:
            cluster_var = unit

        # Filter data based on control_group setting
        if self.control_group == "never_treated":
            # Only keep never-treated as controls
            df_reg = df[df["_never_treated"] | (df[first_treat] > 0)].copy()
        else:
            # Keep all units (not_yet_treated will be handled by the regression)
            df_reg = df.copy()

        # Resolve effective cluster and inject cluster-as-PSU.
        # When `cluster_var is None` (one-way HC2 explicit path), the survey
        # path skips PSU injection and the saturated regression receives
        # `cluster_ids=None` downstream.
        if cluster_var is not None and cluster_var in df_reg.columns:
            cluster_ids_raw = df_reg[cluster_var].values
        else:
            cluster_ids_raw = None
        effective_cluster_ids = _resolve_effective_cluster(
            resolved_survey, cluster_ids_raw, cluster_var if self.cluster is not None else None
        )
        if resolved_survey is not None and effective_cluster_ids is not None:
            from diff_diff.survey import _inject_cluster_as_psu, compute_survey_metadata

            resolved_survey = _inject_cluster_as_psu(resolved_survey, effective_cluster_ids)
            if resolved_survey.psu is not None and survey_metadata is not None:
                # resolved_survey non-None implies survey_design was passed.
                assert survey_design is not None
                raw_w = (
                    data[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(data), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

        # Fit saturated regression
        (
            cohort_effects,
            cohort_ses,
            vcov_cohort,
            coef_index_map,
            bm_artifacts,
        ) = self._fit_saturated_regression(
            df_reg,
            outcome,
            unit,
            time,
            first_treat,
            treatment_groups,
            rel_periods_to_estimate,
            covariates,
            cluster_var,
            survey_weights=survey_weights,
            survey_weight_type=survey_weight_type,
            # For replicate designs: pass None to prevent LinearRegression from
            # computing bogus replicate vcov on already-demeaned data.  We
            # override vcov_cohort below with the correct estimator-level refit.
            resolved_survey=None if _uses_replicate_sa else resolved_survey,
            vcov_type=self.vcov_type,
        )

        # Replicate variance override: fully refit the IW estimator per
        # replicate, including recomputing cohort-share aggregation weights
        # from w_r, so replicate SEs reflect the complete estimator.
        _n_valid_rep_sa = None
        if _uses_replicate_sa:
            from diff_diff.survey import compute_replicate_refit_variance

            # The refit returns [overall_att, es_e0, es_e1, ...] after
            # full re-aggregation with replicate-weighted cohort shares.
            _sa_rel_periods = list(rel_periods_to_estimate)

            def _refit_sa(w_r):
                # Drop zero-weight obs for within-transform safety
                nz = w_r > 0
                df_reg_nz = df_reg[nz] if not np.all(nz) else df_reg
                w_nz = w_r[nz] if not np.all(nz) else w_r
                ce_r, _, vcov_r, cim_r, _ = self._fit_saturated_regression(
                    df_reg_nz,
                    outcome,
                    unit,
                    time,
                    first_treat,
                    treatment_groups,
                    _sa_rel_periods,
                    covariates,
                    cluster_var,
                    survey_weights=w_nz,
                    survey_weight_type=survey_weight_type,
                    resolved_survey=None,
                    vcov_type=self.vcov_type,
                )
                # Create temp weight column for IW aggregation with w_r
                # Use full w_r (including zeros) for correct mass computation
                _wt_col = "_rep_wt"
                df[_wt_col] = w_r
                es_r, _ = self._compute_iw_effects(
                    df,
                    unit,
                    first_treat,
                    treatment_groups,
                    _sa_rel_periods,
                    ce_r,
                    {},
                    vcov_r,
                    cim_r,
                    survey_weight_col=_wt_col,
                )
                att_r, _ = self._compute_overall_att(
                    df,
                    first_treat,
                    es_r,
                    ce_r,
                    _,
                    vcov_r,
                    cim_r,
                    survey_weight_col=_wt_col,
                )
                results = [att_r]
                for e in _sa_rel_periods:
                    results.append(es_r[e]["effect"] if e in es_r else np.nan)
                return np.array(results)

        # Resolve survey weight column name for cohort aggregation
        survey_weight_col = (
            survey_design.weights
            if survey_design is not None
            and hasattr(survey_design, "weights")
            and survey_design.weights
            else None
        )

        # Survey degrees of freedom for t-distribution inference
        _sa_survey_df = (
            max(survey_metadata.df_survey, 1)
            if survey_metadata is not None and survey_metadata.df_survey is not None
            else None
        )
        # Replicate df: rank-deficient → NaN inference (dropped-replicate
        # override happens after replicate refit below)
        if _uses_replicate_sa and _sa_survey_df is None:
            _sa_survey_df = 0  # rank-deficient replicate → NaN inference

        # Compute interaction-weighted event study effects
        event_study_effects, cohort_weights = self._compute_iw_effects(
            df,
            unit,
            first_treat,
            treatment_groups,
            rel_periods_to_estimate,
            cohort_effects,
            cohort_ses,
            vcov_cohort,
            coef_index_map,
            survey_weight_col=survey_weight_col,
            survey_df=_sa_survey_df,
        )

        # Build full event-study VCV via W-matrix aggregation (PR-B 2026-05-17).
        # event_study_effects[e] = Σ_g w_{g,e} * cohort_effects[(g, e)] with
        # w_{g,e} = cohort_weights[e][g]. The full event-study VCV is
        #   event_study_vcov = W @ vcov_cohort @ W.T
        # where W is the |event_times| × n_interactions sparse aggregation matrix
        # whose row i has nonzero entries only at columns j = coef_index_map[(g, e_i)]
        # for cohorts g appearing in cohort_weights[e_i]. The diagonal entry
        # [i, i] of this product reproduces the existing per-event-time SE
        # computation in _compute_iw_effects (weight_vec @ vcov_subset @ weight_vec);
        # the off-diagonals give Cov(β̂_{e_i}, β̂_{e_k}) which is what
        # ``compute_pretrends_power`` needs to consume full Σ_22 instead of
        # falling back to diag(ses^2).
        es_vcov_index: Optional[List[int]] = None
        es_vcov: Optional[np.ndarray] = None
        if cohort_weights:
            es_vcov_index = sorted(cohort_weights.keys())
            n_event_times = len(es_vcov_index)
            n_interactions = vcov_cohort.shape[0]
            W_mat = np.zeros((n_event_times, n_interactions))
            for i, e in enumerate(es_vcov_index):
                for g, w in cohort_weights[e].items():
                    # Defensive: only populate when the (g, e) coefficient
                    # actually exists (cohorts with zero observations at e
                    # are filtered upstream by _compute_iw_effects but we
                    # guard explicitly here for clarity).
                    if (g, e) in coef_index_map:
                        j = coef_index_map[(g, e)]
                        W_mat[i, j] = w
            es_vcov = W_mat @ vcov_cohort @ W_mat.T

        # Compute overall ATT (average of post-treatment effects).
        # Capture overall_weights_by_coef for the hc2_bm contrast-DOF path.
        overall_att, overall_se, _overall_weights_by_coef = self._compute_overall_att(
            df,
            first_treat,
            event_study_effects,
            cohort_effects,
            cohort_weights,
            vcov_cohort,
            coef_index_map,
            survey_weight_col=survey_weight_col,
            return_overall_weights=True,
        )

        # Bell-McCaffrey contrast-DOF for analytical hc2_bm aggregated
        # inference. Cohort-level coefficients already use BM DOF via
        # `LinearRegression.get_inference()` inside `_fit_saturated_regression`,
        # but `event_study_effects` (IW-aggregated) and `overall_att` are
        # linear contrasts of the cohort × event-time coefficients. Per
        # the registry contract for `vcov_type="hc2_bm"`, the user-facing
        # aggregated inference must use CR2 Bell-McCaffrey Satterthwaite
        # DOF for each contrast — not the normal distribution that
        # `safe_inference(..., df=None)` would otherwise default to.
        # Mirrors the MultiPeriodDiD post-period-average contrast pattern
        # added in PR #465 (`_compute_cr2_bm_contrast_dof`).
        _es_contrast_dofs: Dict[int, float] = {}
        _overall_att_contrast_dof: Optional[float] = None
        if bm_artifacts is not None and not _uses_replicate_sa:
            from diff_diff.linalg import _compute_cr2_bm_contrast_dof

            X_full, cluster_ids_full, bread_matrix = bm_artifacts
            n_full_coef = X_full.shape[1]
            # `coef_index_map` is 0-indexed within the cohort-effects
            # block; under full-dummy the interactions occupy columns
            # `coef_offset .. coef_offset + n_interactions - 1` in
            # X_full (where coef_offset == 1 for the intercept). Shift
            # by the same offset when building the contrast vector in
            # full-coef space — otherwise the contrast lands on the
            # wrong columns (off-by-one with the intercept).
            _coef_offset_bm = 1  # full-dummy → interactions at cols 1..n
            # Per-event-time contrasts (IW aggregation across cohorts at
            # each event-time): c_e[full_idx(g, e)] = w_{g,e} for each g.
            es_contrast_keys: List[int] = []
            es_contrast_columns: List[np.ndarray] = []
            for e in sorted(event_study_effects.keys()):
                w_dict = cohort_weights.get(e, {})
                if not w_dict:
                    continue
                col = np.zeros(n_full_coef)
                for g, w_ge in w_dict.items():
                    key = (g, e)
                    if key in coef_index_map:
                        col[coef_index_map[key] + _coef_offset_bm] = w_ge
                if np.any(col != 0):
                    es_contrast_keys.append(e)
                    es_contrast_columns.append(col)
            # Overall ATT contrast: c_overall[full_idx(g,e)] = period_w × cohort_w
            overall_col: Optional[np.ndarray] = None
            if _overall_weights_by_coef:
                overall_col = np.zeros(n_full_coef)
                for (g, e), w in _overall_weights_by_coef.items():
                    if (g, e) in coef_index_map:
                        overall_col[coef_index_map[(g, e)] + _coef_offset_bm] = w
            if es_contrast_columns or overall_col is not None:
                contrast_cols: List[np.ndarray] = list(es_contrast_columns)
                if overall_col is not None:
                    contrast_cols.append(overall_col)
                contrasts_matrix = np.column_stack(contrast_cols)
                try:
                    dof_vec = _compute_cr2_bm_contrast_dof(
                        X_full, cluster_ids_full, bread_matrix, contrasts_matrix
                    )
                    for idx, e in enumerate(es_contrast_keys):
                        _es_contrast_dofs[e] = float(dof_vec[idx])
                    if overall_col is not None:
                        _overall_att_contrast_dof = float(dof_vec[-1])
                except (ValueError, np.linalg.LinAlgError) as exc:
                    # Rank-deficient or other linalg issue: fall back to
                    # the shared analytical df (downgraded to normal
                    # inference). Emit a UserWarning so the deviation is
                    # visible.
                    warnings.warn(
                        f"SunAbraham(vcov_type='hc2_bm') aggregated inference "
                        f"could not compute Bell-McCaffrey contrast DOF "
                        f"({type(exc).__name__}: {exc}). Falling back to "
                        "shared df; aggregated p-values/CIs may use normal "
                        "distribution instead of t(BM DOF).",
                        UserWarning,
                        stacklevel=2,
                    )

        # Apply contrast DOFs to the user-facing aggregated inference.
        # Override the per-event-time inference fields with BM-DOF-aware
        # values when available; otherwise leave the `safe_inference`
        # output from `_compute_iw_effects` in place (which used
        # `df=_sa_survey_df`).
        if _es_contrast_dofs:
            for e, df_e in _es_contrast_dofs.items():
                eff_e = event_study_effects[e]["effect"]
                se_e = event_study_effects[e]["se"]
                t_e, p_e, ci_e = safe_inference(eff_e, se_e, alpha=self.alpha, df=df_e)
                event_study_effects[e]["t_stat"] = t_e
                event_study_effects[e]["p_value"] = p_e
                event_study_effects[e]["conf_int"] = ci_e

        overall_t, overall_p, overall_ci = safe_inference(
            overall_att,
            overall_se,
            alpha=self.alpha,
            df=(
                _overall_att_contrast_dof
                if _overall_att_contrast_dof is not None
                else _sa_survey_df
            ),
        )

        # Replicate variance override: refit fully re-aggregated estimates
        if _uses_replicate_sa:
            # Build full-sample estimate vector from actual outputs
            _full_est_sa = [overall_att]
            for e in _sa_rel_periods:
                _full_est_sa.append(
                    event_study_effects[e]["effect"] if e in event_study_effects else np.nan
                )

            _vcov_sa, _n_valid_rep_sa = compute_replicate_refit_variance(
                _refit_sa, np.array(_full_est_sa), resolved_survey
            )

            # Override df if replicates dropped
            # Replicate-refit path is only reached with a resolved design.
            assert resolved_survey is not None
            if _n_valid_rep_sa < resolved_survey.n_replicates:
                _sa_survey_df = _n_valid_rep_sa - 1 if _n_valid_rep_sa > 1 else 0
            if survey_metadata is not None:
                survey_metadata.df_survey = (
                    _sa_survey_df if _sa_survey_df and _sa_survey_df > 0 else None
                )

            # Override overall ATT SE
            overall_se = float(np.sqrt(max(_vcov_sa[0, 0], 0.0)))
            overall_t, overall_p, overall_ci = safe_inference(
                overall_att, overall_se, alpha=self.alpha, df=_sa_survey_df
            )

            # Override event-study SEs
            for i, e in enumerate(_sa_rel_periods):
                if e in event_study_effects and np.isfinite(event_study_effects[e]["effect"]):
                    se_e = float(np.sqrt(max(_vcov_sa[1 + i, 1 + i], 0.0)))
                    eff_e = event_study_effects[e]["effect"]
                    t_e, p_e, ci_e = safe_inference(eff_e, se_e, alpha=self.alpha, df=_sa_survey_df)
                    event_study_effects[e]["se"] = se_e
                    event_study_effects[e]["t_stat"] = t_e
                    event_study_effects[e]["p_value"] = p_e
                    event_study_effects[e]["conf_int"] = ci_e

            # Cohort-level replicate SEs: second refit for raw (g,e) coefficients
            _keys_ordered = sorted(coef_index_map.keys(), key=lambda k: coef_index_map[k])
            _full_cohort_vec = np.array([cohort_effects.get(k, np.nan) for k in _keys_ordered])

            def _refit_sa_cohort(w_r):
                nz = w_r > 0
                df_reg_nz = df_reg[nz] if not np.all(nz) else df_reg
                w_nz = w_r[nz] if not np.all(nz) else w_r
                ce_r, _, _, _, _ = self._fit_saturated_regression(
                    df_reg_nz,
                    outcome,
                    unit,
                    time,
                    first_treat,
                    treatment_groups,
                    _sa_rel_periods,
                    covariates,
                    cluster_var,
                    survey_weights=w_nz,
                    survey_weight_type=survey_weight_type,
                    resolved_survey=None,
                    vcov_type=self.vcov_type,
                )
                return np.array([ce_r.get(k, np.nan) for k in _keys_ordered])

            _vcov_cohort_rep, _ = compute_replicate_refit_variance(
                _refit_sa_cohort, _full_cohort_vec, resolved_survey
            )
            for key in _keys_ordered:
                idx = coef_index_map[key]
                cohort_ses[key] = float(np.sqrt(max(_vcov_cohort_rep[idx, idx], 0.0)))

        # Run bootstrap if requested
        bootstrap_results = None
        if self.n_bootstrap > 0:
            bootstrap_results = self._run_bootstrap(
                df=df_reg,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                treatment_groups=treatment_groups,
                rel_periods_to_estimate=rel_periods_to_estimate,
                covariates=covariates,
                cluster_var=cluster_var,
                original_event_study=event_study_effects,
                original_overall_att=overall_att,
                resolved_survey=resolved_survey,
                survey_weights=survey_weights,
                survey_weight_type=survey_weight_type,
                survey_weight_col=survey_weight_col,
                use_rao_wu=_use_rao_wu,
            )

            # Update results with bootstrap inference
            overall_se = bootstrap_results.overall_att_se
            overall_t = safe_inference(overall_att, overall_se, alpha=self.alpha)[0]
            overall_p = bootstrap_results.overall_att_p_value
            overall_ci = bootstrap_results.overall_att_ci

            # Update event study effects
            for e in event_study_effects:
                if e in bootstrap_results.event_study_ses:
                    event_study_effects[e]["se"] = bootstrap_results.event_study_ses[e]
                    event_study_effects[e]["conf_int"] = bootstrap_results.event_study_cis[e]
                    event_study_effects[e]["p_value"] = bootstrap_results.event_study_p_values[e]
                    eff_val = event_study_effects[e]["effect"]
                    se_val = event_study_effects[e]["se"]
                    event_study_effects[e]["t_stat"] = safe_inference(
                        eff_val, se_val, alpha=self.alpha
                    )[0]

        # Convert cohort effects to storage format
        cohort_effects_storage: Dict[Tuple[Any, int], Dict[str, Any]] = {}
        for (g, e), effect in cohort_effects.items():
            weight = cohort_weights.get(e, {}).get(g, 0.0)
            se = cohort_ses.get((g, e), 0.0)
            cohort_effects_storage[(g, e)] = {
                "effect": effect,
                "se": se,
                "weight": weight,
            }

        # Clear analytical event_study_vcov when bootstrap or replicate-weight
        # survey overrides the analytical SEs. Mirrors the CS pattern at
        # staggered.py:2032-2036 — prevents mixing analytical VCV with
        # bootstrap/replicate SEs downstream in PreTrendsPower (which would
        # silently produce mis-scaled MDV/power output).
        if bootstrap_results is not None or _uses_replicate_sa:
            es_vcov = None
            es_vcov_index = None

        # Store results
        self.results_ = SunAbrahamResults(
            event_study_effects=event_study_effects,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            cohort_weights=cohort_weights,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=len(df),
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            alpha=self.alpha,
            control_group=self.control_group,
            anticipation=self.anticipation,
            vcov_type=self.vcov_type,
            bootstrap_results=bootstrap_results,
            cohort_effects=cohort_effects_storage,
            survey_metadata=survey_metadata,
            event_study_vcov=es_vcov,
            event_study_vcov_index=es_vcov_index,
            conley_lag_cutoff=(self.conley_lag_cutoff if self.vcov_type == "conley" else None),
            cluster_name=(self.cluster if self.vcov_type == "conley" else None),
        )

        self.is_fitted_ = True
        return self.results_

    def _fit_saturated_regression(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        treatment_groups: List[Any],
        rel_periods: List[int],
        covariates: Optional[List[str]],
        cluster_var: Optional[str],
        survey_weights: Optional[np.ndarray] = None,
        survey_weight_type: str = "pweight",
        resolved_survey: Optional["ResolvedSurveyDesign"] = None,
        vcov_type: str = "hc1",
    ) -> Tuple[
        Dict[Tuple[Any, int], float],
        Dict[Tuple[Any, int], float],
        np.ndarray,
        Dict[Tuple[Any, int], int],
        Optional[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]],
    ]:
        """
        Fit saturated TWFE regression with cohort × relative-time interactions.

        Y_it = α_i + λ_t + Σ_g Σ_e [δ_{g,e} × D_{g,e,it}] + X'γ + ε

        Uses within-transformation for unit + time fixed effects when
        ``vcov_type in {"hc1", "conley"}`` (neither the cluster-robust HC1
        sandwich nor the Conley spatial-HAC sandwich depends on the hat
        matrix; matches ``fixest::sunab()`` convention). Routes
        to a full-dummy saturated design when
        ``vcov_type ∈ {"classical","hc2","hc2_bm"}``. For ``hc2`` /
        ``hc2_bm``, FWL preserves coefficients/residuals but NOT the
        hat matrix —
        HC2 leverage and Bell-McCaffrey DOF must be computed on the full
        FE projection. Mirrors the TwoWayFixedEffects Gate 1 pattern
        from PR #469.

        Returns
        -------
        cohort_effects : dict
            Mapping (cohort, rel_period) -> effect estimate δ_{g,e}
        cohort_ses : dict
            Mapping (cohort, rel_period) -> standard error
        vcov : np.ndarray
            Variance-covariance matrix for cohort effects (size
            n_interactions × n_interactions; extracted from the full
            vcov regardless of which path was taken).
        coef_index_map : dict
            Mapping (cohort, rel_period) -> index in the cohort_effects
            block (0-based, NOT the index in the full coefficient vector
            of the underlying regression).
        """
        df = df.copy()

        # Create cohort × relative-time interaction dummies
        # Exclude reference period
        # Build all columns at once to avoid fragmentation.
        # `coef_index_map` is 0-based within the interactions block; the
        # index in the full coefficient vector depends on the branch:
        #  - Within-transform branch: matches coef_index_map directly
        #    (X has no intercept; interactions occupy positions 0..n-1)
        #  - Full-dummy branch: shift by 1 (intercept at position 0;
        #    interactions occupy positions 1..n)
        interaction_data = {}
        coef_index_map: Dict[Tuple[Any, int], int] = {}
        idx = 0

        for g in treatment_groups:
            for e in rel_periods:
                col_name = f"_D_{g}_{e}"
                # Indicator: unit is in cohort g AND at relative time e
                indicator = ((df[first_treat] == g) & (df["_rel_time"] == e)).astype(float)

                # Only include if there are observations
                if indicator.sum() > 0:
                    interaction_data[col_name] = indicator.values
                    coef_index_map[(g, e)] = idx
                    idx += 1

        # Add all interaction columns at once
        interaction_cols = list(interaction_data.keys())
        if interaction_data:
            interaction_df = pd.DataFrame(interaction_data, index=df.index)
            df = pd.concat([df, interaction_df], axis=1)

        if len(interaction_cols) == 0:
            raise ValueError(
                "No valid cohort × relative-time interactions found. " "Check your data structure."
            )

        n_interactions = len(interaction_cols)
        n_units_fe = df[unit].nunique()
        n_times_fe = df[time].nunique()
        # Route through the full-dummy saturated design when the variance
        # family depends on the hat matrix (hc2 / hc2_bm) — FWL preserves
        # coefficients but not the hat matrix, so HC2 leverage and BM DOF
        # must be computed on the full FE projection. Also route classical
        # through full-dummy so the (n-k) finite-sample correction in
        # ``s² × (X'X)^{-1}`` matches R's ``lm(y ~ ... + factor(unit) +
        # factor(time))`` interpretation at atol=1e-12.
        #
        # hc1 stays on the within-transform path: cluster-robust HC1
        # uses the cluster-mean residual outer product (no hat matrix), and
        # matches ``fixest::sunab(cluster=~unit)`` (which also uses
        # within-transform) at atol=1e-8 — fixest is the natural R parity
        # anchor for SA's HC1 default.
        use_full_dummy = vcov_type in ("hc2", "hc2_bm", "classical")

        if use_full_dummy:
            # Full-dummy auto-route: build [intercept, interactions,
            # covariates, unit_dummies, time_dummies] explicitly. FWL
            # preserves cohort coefficients but NOT the hat matrix, so HC2
            # leverage and Bell-McCaffrey Satterthwaite DOF must be
            # computed on the full FE projection (matches lm() +
            # sandwich::vcovHC / clubSandwich::vcovCR). Memory guard
            # mirrors PR #469's TWFE Gate 1 threshold.
            n_obs = len(df)
            n_cov = len(covariates or [])
            dense_cells = n_obs * (1 + n_interactions + n_cov + (n_units_fe - 1) + (n_times_fe - 1))
            if dense_cells > 50_000_000:
                import warnings

                warnings.warn(
                    f"SunAbraham(vcov_type={vcov_type!r}) builds a dense "
                    f"full-dummy saturated design (~{dense_cells:,} float64 "
                    "cells, >50M). FWL preserves coefficients but not the hat "
                    "matrix, so HC2/HC2-BM requires the full-dummy projection "
                    "(within-transform would produce a methodologically "
                    "different statistic). For very high-cardinality panels, "
                    "consider vcov_type='hc1' (within-transform; no full-"
                    "dummy needed) or reducing the panel size.",
                    UserWarning,
                    stacklevel=2,
                )

            interaction_arrs = [df[c].values.astype(np.float64) for c in interaction_cols]
            cov_arrs = [df[c].values.astype(np.float64) for c in (covariates or [])]
            unit_dummies = pd.get_dummies(
                df[unit], prefix=f"_fe_{unit}", drop_first=True
            ).values.astype(np.float64)
            time_dummies = pd.get_dummies(
                df[time], prefix=f"_fe_{time}", drop_first=True
            ).values.astype(np.float64)
            intercept = np.ones(len(df))
            X = np.column_stack(
                [intercept] + interaction_arrs + cov_arrs + [unit_dummies, time_dummies]
            )
            y = df[outcome].values.astype(np.float64)
            if cluster_var is not None and cluster_var in df.columns:
                cluster_ids = df[cluster_var].values
            else:
                cluster_ids = None
            # Full-dummy already counts unit + time dummies in n_params, so
            # no extra adjustment (matches TWFE PR #469 Gate 1).
            df_adj = 0
            # Interactions occupy columns 1..n_interactions (intercept at 0)
            coef_offset = 1
        else:
            # Within-transform path (existing) — used for hc1 and conley
            # (both robust sandwiches that don't need the full FE hat matrix).
            # classical now routes through the full-dummy branch above so its
            # (n-k) finite-sample correction matches R's lm() interpretation.
            variables_to_demean = [outcome] + interaction_cols
            if covariates:
                variables_to_demean.extend(covariates)
            _sa_regressors = variables_to_demean[1:]  # everything except outcome
            _pre_norms = pre_demean_norms(df, _sa_regressors, weights=survey_weights)

            df_demeaned = _within_transform_util(
                df, variables_to_demean, unit, time, suffix="_dm", weights=survey_weights
            )
            # Snap FE-spanned regressors (e.g. a unit-constant covariate) to
            # exact zero so rank handling drops them deterministically.
            snap_absorbed_regressors(
                df_demeaned,
                _sa_regressors,
                _pre_norms,
                absorbed_desc=f"unit '{unit}' and time '{time}' fixed effects",
                group_vars=[unit, time],
                rank_deficient_action=self.rank_deficient_action,
                suffix="_dm",
                weights=survey_weights,
            )

            X_cols = [f"{col}_dm" for col in interaction_cols]
            if covariates:
                X_cols.extend([f"{cov}_dm" for cov in covariates])

            X = df_demeaned[X_cols].values
            y = df_demeaned[f"{outcome}_dm"].values
            if cluster_var is not None and cluster_var in df_demeaned.columns:
                cluster_ids = df_demeaned[cluster_var].values
            else:
                cluster_ids = None
            df_adj = n_units_fe + n_times_fe - 1
            # Interactions occupy columns 0..n_interactions-1 (no intercept)
            coef_offset = 0

        # Conley spatial-HAC arrays, row-aligned to the design X. SA routes
        # conley through the within-transform path (use_full_dummy excludes it),
        # and within_transform preserves row order/count, so coordinates read
        # from `df` (== df_reg, the post-filter frame) align to X's rows.
        if vcov_type == "conley":
            assert self.conley_coords is not None  # guaranteed by _validate_conley_estimator_inputs
            _cl_coords = np.column_stack(
                [
                    df[self.conley_coords[0]].values.astype(np.float64),
                    df[self.conley_coords[1]].values.astype(np.float64),
                ]
            )
            _cl_time = np.asarray(df[time].values)
            _cl_unit = df[unit].values
        else:
            _cl_coords = _cl_time = _cl_unit = None

        reg = LinearRegression(
            include_intercept=False,  # Full design already built (with or without intercept)
            robust=True,  # legacy alias; vcov_type below overrides
            cluster_ids=cluster_ids,
            rank_deficient_action=self.rank_deficient_action,
            weights=survey_weights,
            weight_type=survey_weight_type,
            survey_design=resolved_survey,
            vcov_type=vcov_type,
            conley_coords=_cl_coords,
            conley_cutoff_km=self.conley_cutoff_km,
            conley_metric=self.conley_metric,
            conley_kernel=self.conley_kernel,
            conley_time=_cl_time,
            conley_unit=_cl_unit,
            conley_lag_cutoff=self.conley_lag_cutoff,
        ).fit(X, y, df_adjustment=df_adj)

        vcov = reg.vcov_

        # Extract cohort effects and standard errors using get_inference.
        # coef_index_map is 0-based within the interactions block; under
        # full-dummy we shift by +1 to skip the intercept.
        cohort_effects: Dict[Tuple[Any, int], float] = {}
        cohort_ses: Dict[Tuple[Any, int], float] = {}

        for (g, e), coef_idx in coef_index_map.items():
            full_idx = coef_idx + coef_offset
            inference = reg.get_inference(full_idx)
            cohort_effects[(g, e)] = inference.coefficient
            cohort_ses[(g, e)] = inference.se

        # Extract the vcov sub-block for cohort effects only (covariates
        # and FE dummies excluded). Under full-dummy the interactions
        # start at column 1; under within-transform they start at 0.
        assert vcov is not None
        vcov_cohort = vcov[
            coef_offset : coef_offset + n_interactions,
            coef_offset : coef_offset + n_interactions,
        ]

        # Stash BM contrast-DOF artifacts when hc2_bm — needed by the
        # aggregated inference layer to compute per-event-time and
        # overall-ATT Satterthwaite DOF on user-facing outputs. Under
        # other vcov_type values aggregated inference falls back to the
        # shared analytical df (None → normal distribution).
        if vcov_type == "hc2_bm":
            bread_matrix = X.T @ X
            bm_artifacts: Optional[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]] = (
                X,
                cluster_ids,
                bread_matrix,
            )
        else:
            bm_artifacts = None

        return cohort_effects, cohort_ses, vcov_cohort, coef_index_map, bm_artifacts

    def _compute_iw_effects(
        self,
        df: pd.DataFrame,
        unit: str,
        first_treat: str,
        treatment_groups: List[Any],
        rel_periods: List[int],
        cohort_effects: Dict[Tuple[Any, int], float],
        cohort_ses: Dict[Tuple[Any, int], float],
        vcov_cohort: np.ndarray,
        coef_index_map: Dict[Tuple[Any, int], int],
        survey_weight_col: Optional[str] = None,
        survey_df: Optional[int] = None,
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[Any, float]]]:
        """
        Compute interaction-weighted event study effects.

        β_e = Σ_g w_{g,e} × δ_{g,e}

        where w_{g,e} = n_{g,e} / Σ_g n_{g,e} is the share of observations from cohort g
        at event-time e among all treated observations at that event-time.

        When survey weights are provided, n_{g,e} is the survey-weighted mass
        (sum of weights) rather than raw observation counts, so the estimand
        reflects the survey-weighted cohort composition.

        Returns
        -------
        event_study_effects : dict
            Dictionary mapping relative period to aggregated effect info.
        cohort_weights : dict
            Dictionary mapping relative period to cohort weight dictionary.
        """
        event_study_effects: Dict[int, Dict[str, Any]] = {}
        cohort_weights: Dict[int, Dict[Any, float]] = {}

        # Pre-compute per-event-time observation mass: n_{g,e}
        # With survey weights, use weighted sum; otherwise raw counts.
        treated_mask = df[first_treat] > 0
        if survey_weight_col is not None and survey_weight_col in df.columns:
            event_time_counts = (
                df[treated_mask].groupby([first_treat, "_rel_time"])[survey_weight_col].sum()
            )
        else:
            event_time_counts = df[treated_mask].groupby([first_treat, "_rel_time"]).size()

        for e in rel_periods:
            # Get cohorts that have observations at this relative time
            cohorts_at_e = [g for g in treatment_groups if (g, e) in cohort_effects]

            if not cohorts_at_e:
                continue

            # Compute IW weights: n_{g,e} / Σ_g n_{g,e}
            weights = {}
            total_size = 0
            for g in cohorts_at_e:
                n_g_e = event_time_counts.get((g, e), 0)
                weights[g] = n_g_e
                total_size += n_g_e

            if total_size == 0:
                continue

            # Normalize weights
            for g in weights:
                weights[g] = weights[g] / total_size

            cohort_weights[e] = weights

            # Compute weighted average effect
            agg_effect = 0.0
            for g in cohorts_at_e:
                w = weights[g]
                agg_effect += w * cohort_effects[(g, e)]

            # Compute SE using delta method with vcov
            # Var(β_e) = w' Σ w where w is weight vector and Σ is vcov submatrix
            indices = [coef_index_map[(g, e)] for g in cohorts_at_e]
            weight_vec = np.array([weights[g] for g in cohorts_at_e])
            vcov_subset = vcov_cohort[np.ix_(indices, indices)]
            agg_var = float(weight_vec @ vcov_subset @ weight_vec)
            agg_se = np.sqrt(max(agg_var, 0))

            t_stat, p_val, ci = safe_inference(agg_effect, agg_se, alpha=self.alpha, df=survey_df)

            event_study_effects[e] = {
                "effect": agg_effect,
                "se": agg_se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_groups": len(cohorts_at_e),
            }

        return event_study_effects, cohort_weights

    @overload
    def _compute_overall_att(
        self,
        df: pd.DataFrame,
        first_treat: str,
        event_study_effects: Dict[int, Dict[str, Any]],
        cohort_effects: Dict[Tuple[Any, int], float],
        cohort_weights: Dict[int, Dict[Any, float]],
        vcov_cohort: np.ndarray,
        coef_index_map: Dict[Tuple[Any, int], int],
        survey_weight_col: Optional[str] = None,
        return_overall_weights: Literal[False] = False,
    ) -> Tuple[float, float]: ...

    @overload
    def _compute_overall_att(
        self,
        df: pd.DataFrame,
        first_treat: str,
        event_study_effects: Dict[int, Dict[str, Any]],
        cohort_effects: Dict[Tuple[Any, int], float],
        cohort_weights: Dict[int, Dict[Any, float]],
        vcov_cohort: np.ndarray,
        coef_index_map: Dict[Tuple[Any, int], int],
        survey_weight_col: Optional[str] = None,
        *,
        return_overall_weights: Literal[True],
    ) -> Tuple[float, float, Optional[Dict[Tuple[Any, int], float]]]: ...

    def _compute_overall_att(
        self,
        df: pd.DataFrame,
        first_treat: str,
        event_study_effects: Dict[int, Dict[str, Any]],
        cohort_effects: Dict[Tuple[Any, int], float],
        cohort_weights: Dict[int, Dict[Any, float]],
        vcov_cohort: np.ndarray,
        coef_index_map: Dict[Tuple[Any, int], int],
        survey_weight_col: Optional[str] = None,
        return_overall_weights: bool = False,
    ) -> Union[
        Tuple[float, float],
        Tuple[float, float, Optional[Dict[Tuple[Any, int], float]]],
    ]:
        """
        Compute overall ATT as weighted average of post-treatment effects.

        When survey weights are provided, the per-period weights use
        survey-weighted mass rather than raw observation counts.

        Returns (att, se) tuple. When ``return_overall_weights=True``,
        the returned tuple is extended to (att, se, overall_weights_by_coef)
        where the dict maps (g, e) → weight in the overall ATT
        contrast (i.e. ``c[full_idx(g,e)] = period_weight × cohort_weight``).
        Used by the analytical hc2_bm path to build Bell-McCaffrey
        contrast DOFs for the user-facing aggregated inference. The dict
        is ``None`` when the simplified-variance fallback path was taken.
        """
        post_effects = [(e, eff) for e, eff in event_study_effects.items() if e >= 0]

        if not post_effects:
            if return_overall_weights:
                return np.nan, np.nan, None
            return np.nan, np.nan

        # Weight by (survey-weighted) mass of treated observations at each relative time
        post_weights = []
        post_estimates = []

        for e, eff in post_effects:
            mask = (df["_rel_time"] == e) & (df[first_treat] > 0)
            if survey_weight_col is not None and survey_weight_col in df.columns:
                # No floor for survey weights — valid masses can be < 1
                n_at_e = df.loc[mask, survey_weight_col].sum()
                post_weights.append(n_at_e if n_at_e > 0 else 0.0)
            else:
                n_at_e = len(df[mask])
                post_weights.append(max(n_at_e, 1))
            post_estimates.append(eff["effect"])

        post_weights_arr = np.array(post_weights, dtype=float)
        post_weights_arr = post_weights_arr / post_weights_arr.sum()

        overall_att = float(np.sum(post_weights_arr * np.array(post_estimates)))

        # Compute SE using delta method
        # Need to trace back through the full weighting scheme
        # ATT = Σ_e w_e × β_e = Σ_e w_e × Σ_g w_{g,e} × δ_{g,e}
        # Collect all (g, e) pairs and their overall weights
        overall_weights_by_coef: Dict[Tuple[Any, int], float] = {}

        for i, (e, _) in enumerate(post_effects):
            period_weight = post_weights_arr[i]
            if e in cohort_weights:
                for g, cw in cohort_weights[e].items():
                    key = (g, e)
                    if key in coef_index_map:
                        if key not in overall_weights_by_coef:
                            overall_weights_by_coef[key] = 0.0
                        overall_weights_by_coef[key] += period_weight * cw

        if not overall_weights_by_coef:
            # Fallback to simplified variance that ignores covariances between periods
            warnings.warn(
                "Could not construct full weight vector for overall ATT SE. "
                "Using simplified variance that ignores covariances between periods.",
                UserWarning,
                stacklevel=2,
            )
            overall_var = float(
                np.sum(
                    (post_weights_arr**2) * np.array([eff["se"] ** 2 for _, eff in post_effects])
                )
            )
            if return_overall_weights:
                return overall_att, np.sqrt(overall_var), None
            return overall_att, np.sqrt(overall_var)

        # Build full weight vector and compute variance
        indices = [coef_index_map[key] for key in overall_weights_by_coef.keys()]
        weight_vec = np.array(list(overall_weights_by_coef.values()))
        vcov_subset = vcov_cohort[np.ix_(indices, indices)]
        overall_var = float(weight_vec @ vcov_subset @ weight_vec)
        overall_se = np.sqrt(max(overall_var, 0))

        if return_overall_weights:
            return overall_att, overall_se, overall_weights_by_coef
        return overall_att, overall_se

    def _run_bootstrap(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        treatment_groups: List[Any],
        rel_periods_to_estimate: List[int],
        covariates: Optional[List[str]],
        cluster_var: Optional[str],
        original_event_study: Dict[int, Dict[str, Any]],
        original_overall_att: float,
        resolved_survey: Optional["ResolvedSurveyDesign"] = None,
        survey_weights: Optional[np.ndarray] = None,
        survey_weight_type: str = "pweight",
        survey_weight_col: Optional[str] = None,
        use_rao_wu: bool = False,
    ) -> SABootstrapResults:
        """
        Run bootstrap for inference.

        When use_rao_wu is True (survey design with explicit strata/PSU/FPC),
        uses Rao-Wu rescaled bootstrap (weight perturbation). Otherwise, uses
        pairs bootstrap (resampling units with replacement).
        """
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        if use_rao_wu:
            return self._run_rao_wu_bootstrap(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                treatment_groups=treatment_groups,
                rel_periods_to_estimate=rel_periods_to_estimate,
                covariates=covariates,
                cluster_var=cluster_var,
                original_event_study=original_event_study,
                original_overall_att=original_overall_att,
                resolved_survey=resolved_survey,
                survey_weight_type=survey_weight_type,
                survey_weight_col=survey_weight_col,
                rng=rng,
            )

        # --- Pairs bootstrap (non-survey or weights-only survey) ---

        # Get unique units
        all_units = df[unit].unique()
        n_units = len(all_units)

        # Pre-compute unit -> row indices mapping (avoids repeated boolean scans)
        unit_row_indices = {u: df.index[df[unit] == u].values for u in all_units}
        unit_row_counts = {u: len(idx) for u, idx in unit_row_indices.items()}

        # Store bootstrap samples
        rel_periods = sorted(original_event_study.keys())
        bootstrap_effects = {e: np.zeros(self.n_bootstrap) for e in rel_periods}
        bootstrap_overall = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # Resample units with replacement (pairs bootstrap)
            boot_units = rng.choice(all_units, size=n_units, replace=True)

            # Create bootstrap sample using pre-computed index mapping
            boot_indices = np.concatenate([unit_row_indices[u] for u in boot_units])
            df_b = df.iloc[boot_indices].copy()

            # Reassign unique unit IDs (vectorized)
            rows_per_unit = np.array([unit_row_counts[u] for u in boot_units])
            df_b[unit] = np.repeat(np.arange(n_units), rows_per_unit)

            # Recompute relative time (vectorized)
            df_b["_rel_time"] = np.where(
                df_b[first_treat] > 0, df_b[time] - df_b[first_treat], np.nan
            )
            # np.inf was normalized to 0 in fit(), so the np.inf check is defensive only
            df_b["_never_treated"] = (df_b[first_treat] == 0) | (df_b[first_treat] == np.inf)

            try:
                # Extract survey weights from resampled data if present
                boot_survey_weights = None
                if survey_weight_col is not None and survey_weight_col in df_b.columns:
                    boot_survey_weights = df_b[survey_weight_col].values

                # Re-estimate saturated regression
                (
                    cohort_effects_b,
                    cohort_ses_b,
                    vcov_b,
                    coef_map_b,
                    _,
                ) = self._fit_saturated_regression(
                    df_b,
                    outcome,
                    unit,
                    time,
                    first_treat,
                    treatment_groups,
                    rel_periods_to_estimate,
                    covariates,
                    cluster_var,
                    survey_weights=boot_survey_weights,
                    survey_weight_type=survey_weight_type,
                    resolved_survey=None,  # Use explicit weights, not stale design
                    vcov_type=self.vcov_type,
                )

                # Compute IW effects for this bootstrap sample
                event_study_b, cohort_weights_b = self._compute_iw_effects(
                    df_b,
                    unit,
                    first_treat,
                    treatment_groups,
                    rel_periods_to_estimate,
                    cohort_effects_b,
                    cohort_ses_b,
                    vcov_b,
                    coef_map_b,
                    survey_weight_col=survey_weight_col,
                )

                # Store bootstrap estimates
                for e in rel_periods:
                    if e in event_study_b:
                        bootstrap_effects[e][b] = event_study_b[e]["effect"]
                    else:
                        bootstrap_effects[e][b] = original_event_study[e]["effect"]

                # Compute overall ATT for this bootstrap sample
                overall_b, _ = self._compute_overall_att(
                    df_b,
                    first_treat,
                    event_study_b,
                    cohort_effects_b,
                    cohort_weights_b,
                    vcov_b,
                    coef_map_b,
                    survey_weight_col=survey_weight_col,
                )
                bootstrap_overall[b] = overall_b

            except (ValueError, np.linalg.LinAlgError) as exc:
                # If bootstrap iteration fails, use original
                warnings.warn(
                    f"Bootstrap iteration {b} failed: {exc}. Using original estimate.",
                    UserWarning,
                    stacklevel=2,
                )
                for e in rel_periods:
                    bootstrap_effects[e][b] = original_event_study[e]["effect"]
                bootstrap_overall[b] = original_overall_att

        # Compute bootstrap statistics
        event_study_ses = {}
        event_study_cis = {}
        event_study_p_values = {}

        for e in rel_periods:
            boot_dist = bootstrap_effects[e]
            original_effect = original_event_study[e]["effect"]
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect,
                boot_dist,
                alpha=self.alpha,
                context=f"event study e={e}",
            )
            event_study_ses[e] = se
            event_study_cis[e] = ci
            event_study_p_values[e] = p_value

        # Overall ATT statistics
        overall_se, overall_ci, overall_p = compute_effect_bootstrap_stats(
            original_overall_att,
            bootstrap_overall,
            alpha=self.alpha,
            context="overall ATT",
        )

        return SABootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type="pairs",
            alpha=self.alpha,
            overall_att_se=overall_se,
            overall_att_ci=overall_ci,
            overall_att_p_value=overall_p,
            event_study_ses=event_study_ses,
            event_study_cis=event_study_cis,
            event_study_p_values=event_study_p_values,
            bootstrap_distribution=bootstrap_overall,
        )

    def _run_rao_wu_bootstrap(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        treatment_groups: List[Any],
        rel_periods_to_estimate: List[int],
        covariates: Optional[List[str]],
        cluster_var: Optional[str],
        original_event_study: Dict[int, Dict[str, Any]],
        original_overall_att: float,
        resolved_survey: "ResolvedSurveyDesign",
        survey_weight_type: str,
        survey_weight_col: Optional[str],
        rng: np.random.Generator,
    ) -> SABootstrapResults:
        """
        Run Rao-Wu rescaled bootstrap for survey-aware inference.

        Instead of physically resampling units, each iteration generates
        rescaled observation weights via Rao-Wu (1988) weight perturbation.
        The rescaled weights feed into the existing WLS regression path.
        """
        from diff_diff.bootstrap_utils import generate_rao_wu_weights
        from diff_diff.survey import ResolvedSurveyDesign

        # Column name for rescaled weights in the bootstrap DataFrame
        _rw_col = "__rw_boot_weight"

        # Collapse survey design to unit level so Rao-Wu respects panel
        # structure: each unit gets one set of weights regardless of how
        # many time periods it has.  Without this, when there is no
        # explicit PSU, generate_rao_wu_weights treats each observation as
        # its own PSU and different obs of the same unit can get different
        # weights, breaking panel semantics.
        all_units = df[unit].unique()

        weights_unit = (
            pd.Series(resolved_survey.weights, index=df.index)
            .groupby(df[unit])
            .first()
            .reindex(all_units)
            .values.astype(np.float64)
        )

        strata_unit = None
        if resolved_survey.strata is not None:
            strata_unit = (
                pd.Series(resolved_survey.strata, index=df.index)
                .groupby(df[unit])
                .first()
                .reindex(all_units)
                .values
            )

        psu_unit = None
        if resolved_survey.psu is not None:
            psu_unit = (
                pd.Series(resolved_survey.psu, index=df.index)
                .groupby(df[unit])
                .first()
                .reindex(all_units)
                .values
            )

        fpc_unit = None
        if resolved_survey.fpc is not None:
            fpc_unit = (
                pd.Series(resolved_survey.fpc, index=df.index)
                .groupby(df[unit])
                .first()
                .reindex(all_units)
                .values
            )

        unit_resolved = ResolvedSurveyDesign(
            weights=weights_unit,
            weight_type=resolved_survey.weight_type,
            strata=strata_unit,
            psu=psu_unit,
            fpc=fpc_unit,
            n_strata=resolved_survey.n_strata,
            n_psu=resolved_survey.n_psu,
            lonely_psu=resolved_survey.lonely_psu,
        )

        # Build unit -> row indices mapping for expanding unit-level weights
        unit_to_rows = {u: df.index[df[unit] == u].values for u in all_units}
        unit_order = {u: i for i, u in enumerate(all_units)}

        # Store bootstrap samples
        rel_periods = sorted(original_event_study.keys())
        bootstrap_effects = {e: np.full(self.n_bootstrap, np.nan) for e in rel_periods}
        bootstrap_overall = np.full(self.n_bootstrap, np.nan)

        for b in range(self.n_bootstrap):
            try:
                # Generate Rao-Wu rescaled weights at unit level
                unit_boot_weights = generate_rao_wu_weights(unit_resolved, rng)

                # Expand unit-level weights to observation level
                boot_weights = np.empty(len(df), dtype=np.float64)
                for u, idx in unit_to_rows.items():
                    boot_weights[idx] = unit_boot_weights[unit_order[u]]

                # Drop observations with zero weight (PSUs not drawn in this
                # iteration) to avoid NaN/Inf in within-transformation.
                positive_mask = boot_weights > 0
                if positive_mask.sum() < 2:
                    # Too few observations with positive weight
                    raise ValueError("Rao-Wu iteration produced < 2 positive weights")

                df_b = df[positive_mask].reset_index(drop=True)
                boot_weights_b = boot_weights[positive_mask]
                df_b[_rw_col] = boot_weights_b

                # Verify we still have both treated and control observations
                has_treated = (df_b[first_treat] > 0).any()
                has_control = ((df_b[first_treat] == 0) | (df_b[first_treat] == np.inf)).any()
                if not has_treated or not has_control:
                    raise ValueError("Rao-Wu iteration dropped all treated or control units")

                # Re-estimate saturated regression with rescaled weights.
                # Pass resolved_survey=None since inference comes from the
                # bootstrap distribution, not from within-iteration vcov.
                (
                    cohort_effects_b,
                    cohort_ses_b,
                    vcov_b,
                    coef_map_b,
                    _,
                ) = self._fit_saturated_regression(
                    df_b,
                    outcome,
                    unit,
                    time,
                    first_treat,
                    treatment_groups,
                    rel_periods_to_estimate,
                    covariates,
                    cluster_var,
                    survey_weights=boot_weights_b,
                    survey_weight_type=survey_weight_type,
                    resolved_survey=None,
                    vcov_type=self.vcov_type,
                )

                # Compute IW effects using rescaled weights for cohort shares
                event_study_b, cohort_weights_b = self._compute_iw_effects(
                    df_b,
                    unit,
                    first_treat,
                    treatment_groups,
                    rel_periods_to_estimate,
                    cohort_effects_b,
                    cohort_ses_b,
                    vcov_b,
                    coef_map_b,
                    survey_weight_col=_rw_col,
                )

                # Store bootstrap estimates
                for e in rel_periods:
                    if e in event_study_b:
                        bootstrap_effects[e][b] = event_study_b[e]["effect"]
                    else:
                        bootstrap_effects[e][b] = original_event_study[e]["effect"]

                # Compute overall ATT using rescaled weights
                overall_b, _ = self._compute_overall_att(
                    df_b,
                    first_treat,
                    event_study_b,
                    cohort_effects_b,
                    cohort_weights_b,
                    vcov_b,
                    coef_map_b,
                    survey_weight_col=_rw_col,
                )
                bootstrap_overall[b] = overall_b

            except (ValueError, np.linalg.LinAlgError) as exc:
                # Failed draws stored as NaN (not original estimate) to avoid
                # shrinking bootstrap dispersion.  compute_effect_bootstrap_stats
                # handles NaN draws via nanstd.
                warnings.warn(
                    f"Bootstrap iteration {b} failed: {exc}. Storing NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                for e in rel_periods:
                    bootstrap_effects[e][b] = np.nan
                bootstrap_overall[b] = np.nan

        # Compute bootstrap statistics
        event_study_ses = {}
        event_study_cis = {}
        event_study_p_values = {}

        for e in rel_periods:
            boot_dist = bootstrap_effects[e]
            original_effect = original_event_study[e]["effect"]
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect,
                boot_dist,
                alpha=self.alpha,
                context=f"event study e={e}",
            )
            event_study_ses[e] = se
            event_study_cis[e] = ci
            event_study_p_values[e] = p_value

        # Overall ATT statistics
        overall_se, overall_ci, overall_p = compute_effect_bootstrap_stats(
            original_overall_att,
            bootstrap_overall,
            alpha=self.alpha,
            context="overall ATT",
        )

        return SABootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type="rao_wu",
            alpha=self.alpha,
            overall_att_se=overall_se,
            overall_att_ci=overall_ci,
            overall_att_p_value=overall_p,
            event_study_ses=event_study_ses,
            event_study_cis=event_study_cis,
            event_study_p_values=event_study_p_values,
            bootstrap_distribution=bootstrap_overall,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "control_group": self.control_group,
            "anticipation": self.anticipation,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "rank_deficient_action": self.rank_deficient_action,
            "vcov_type": self.vcov_type,
            "conley_coords": self.conley_coords,
            "conley_cutoff_km": self.conley_cutoff_km,
            "conley_metric": self.conley_metric,
            "conley_kernel": self.conley_kernel,
            "conley_lag_cutoff": self.conley_lag_cutoff,
        }

    def set_params(self, **params) -> "SunAbraham":
        """Set estimator parameters (sklearn-compatible)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        # Refresh the explicit-vcov-type flag if vcov_type changed, so the
        # auto-cluster guard at fit time uses the updated value.
        if "vcov_type" in params:
            self._vcov_type_explicit = self.vcov_type != "hc1"
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
