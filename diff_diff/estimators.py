"""
Difference-in-Differences estimators with sklearn-like API.

This module contains the core DiD estimators:
- DifferenceInDifferences: Basic 2x2 DiD estimator
- MultiPeriodDiD: Event-study style DiD with period-specific treatment effects

Additional estimators are in separate modules:
- TwoWayFixedEffects: See diff_diff.twfe
- SyntheticDiD: See diff_diff.synthetic_did
- SyntheticControl: See diff_diff.synthetic_control

For backward compatibility, all estimators are re-exported from this module.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff._base import BaseEstimator
from diff_diff._deprecation import (
    NOT_SUPPLIED,
    resolve_renamed_kwarg,
    warn_deprecated_kwarg,
)
from diff_diff.linalg import (
    LinearRegression,
    _absorbed_fe_vcov_scale,
    _expand_vcov_with_nan,
    compute_r_squared,
    solve_ols,
)
from diff_diff.results import DiDResults, MultiPeriodDiDResults, PeriodEffect
from diff_diff.utils import (
    WildBootstrapResults,
    absorbed_fe_cr1_k_increment,
    absorbed_fe_rank,
    build_fe_dummy_blocks,
    cluster_nested_fe_dims,
    demean_by_groups,
    fe_dummy_names,
    pre_demean_norms,
    safe_inference,
    snap_absorbed_regressors,
    validate_binary,
    validate_covariate_names,
    validate_design_term_names,
    validate_df_convention,
    validate_n_bootstrap,
    wild_bootstrap_se,
)

# Accepted values for the `inference` selector (M-096). Exposed by exactly
# DifferenceInDifferences, MultiPeriodDiD and TwoWayFixedEffects (the two
# subclasses inherit this __init__); the fail-closed check lives in
# DifferenceInDifferences.__init__ and set_params inherits it via the
# BaseEstimator probe re-init.
_INFERENCE_METHODS = ("analytical", "wild_bootstrap")

# Sentinel for _fit_event_study_core's cluster_override: "use self.cluster".
# A plain None default cannot express that, because None is itself a legal
# override value ("no clustering" - the resolved TWFE event-study carve-outs).
_USE_SELF_CLUSTER: Any = object()

# MultiPeriodDiD 3.9 deprecation message (row M-010; the EventStudy alias is
# the same class object, so constructing via the alias emits this too -
# row M-060). Pinned verbatim by tests/test_v4_merge_mpd.py and the targeted
# pytest filter in pyproject.toml.
_MPD_DEPRECATION_MSG = (
    "MultiPeriodDiD is deprecated and will be removed in 4.0; use "
    "TwoWayFixedEffects().fit(..., event_study=True) instead - "
    "spec='pooled' reproduces the MultiPeriodDiD design; the default "
    "spec='within' adds unit fixed effects. The EventStudy alias is "
    "deprecated with it."
)


class DifferenceInDifferences(BaseEstimator):
    """
    Difference-in-Differences estimator with sklearn-like interface.

    Estimates the Average Treatment effect on the Treated (ATT) using
    the canonical 2x2 DiD design or panel data with two-way fixed effects.

    Parameters
    ----------
    formula : str, optional
        R-style formula for the model (e.g., "outcome ~ treated * post").
        If provided, overrides column name parameters.
    robust : bool, optional
        DEPRECATED legacy alias for ``vcov_type`` (row M-045; warns with
        ``FutureWarning``, removed in 4.0 - use ``vcov_type=``).
        ``robust=True`` maps to ``vcov_type="hc1"``; ``robust=False`` maps
        to ``vcov_type="classical"``. Explicit ``vcov_type`` overrides
        ``robust`` unless the pair is contradictory (e.g.
        ``robust=False, vcov_type="hc2"`` raises).
    cluster : str, optional
        Column name for cluster-robust standard errors. Combined with
        ``vcov_type``: with ``"hc1"`` dispatches to CR1 (Liang-Zeger); with
        ``"hc2_bm"`` dispatches to CR2 Bell-McCaffrey (Pustejovsky-Tipton 2018
        symmetric-sqrt + Satterthwaite DOF).
    vcov_type : {"classical", "hc1", "hc2", "hc2_bm", "hc3", "conley"}, optional
        Variance-covariance family. Defaults to the ``robust`` alias.

        - ``"classical"``: non-robust OLS SEs, ``sigma_hat^2 * (X'X)^{-1}``.
        - ``"hc1"``: heteroskedasticity-robust HC1 with ``n/(n-k)`` adjustment
          (library default). With ``cluster=``, uses CR1 (Liang-Zeger).
        - ``"hc2"``: leverage-corrected meat (one-way only). Errors with
          ``cluster=``; use ``"hc2_bm"`` for clustered Bell-McCaffrey.
        - ``"hc2_bm"``: one-way HC2 + Imbens-Kolesar (2016) Satterthwaite DOF;
          with ``cluster=``, Pustejovsky-Tipton (2018) CR2 cluster-robust.
          ``MultiPeriodDiD(cluster=..., vcov_type="hc2_bm")`` is supported and
          uses a cluster-aware Bell-McCaffrey contrast DOF for the
          post-period-average ATT (see ``_compute_cr2_bm_contrast_dof`` in
          ``linalg.py`` and the REGISTRY.md note). Weighted CR2-BM
          (``survey_design=`` paths) is a separate gate.
        - ``"hc3"``: jackknife-style leverage correction, meat
          ``e_i^2 / (1 - h_ii)^2`` (one-way only; errors with ``cluster=``).
          A leverage-one observation has no defined HC3 variance and the
          vcov fails closed (warning + NaN inference) rather than flooring
          ``1 - h_ii``. With ``absorb=``, routes through the full-dummy
          design like hc2.
        - ``"conley"``: Conley 1999 spatial-HAC sandwich. Pass
          ``conley_coords=(lat_col, lon_col)``, ``conley_cutoff_km=<float>``,
          and ``conley_lag_cutoff=<int>`` on the constructor; pass
          ``unit=<col>`` as a fit-time kwarg to :meth:`fit` (NOT on
          ``__init__``; unused unless Conley is set; not part of
          ``get_params()`` / ``set_params()``). The block-decomposed panel
          sandwich (matches R ``conleyreg`` with ``lag_cutoff > 0``) sums
          within-period spatial pairs plus within-unit Bartlett serial
          pairs (lag=0 excluded). Explicit ``cluster=<col>`` enables the
          combined spatial + cluster product kernel; ``survey_design=``
          and ``inference='wild_bootstrap'`` both raise
          ``NotImplementedError``.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    inference : str, default="analytical"
        Inference method: "analytical" for standard asymptotic inference,
        or "wild_bootstrap" for wild cluster bootstrap (recommended when
        number of clusters is small, <50). Exactly these two (string)
        values are accepted; anything else raises ``ValueError`` at
        construction. ``"wild_bootstrap"`` requires ``cluster=`` — a fit
        without it raises ``ValueError`` (since 3.9; previously it fell
        back to analytical inference silently).
    n_bootstrap : int, default=999
        Number of bootstrap replications when inference="wild_bootstrap".
        Must be a non-negative integer; ``>= 2`` is required when
        ``inference="wild_bootstrap"`` (0 or 1 replications cannot produce
        bootstrap inference — the fit raises ``ValueError``).
    bootstrap_weights : str, default="rademacher"
        Type of bootstrap weights: "rademacher" (standard), "webb"
        (recommended for <10 clusters), or "mammen" (skewness correction).
    p_val_type : str, default="two-tailed"
        Shape of the wild cluster bootstrap test (mirrors
        ``fwildclusterboot::boottest``): "two-tailed" (test on ``|t|``,
        two-tailed inverted CI — which may be asymmetric) or "equal-tailed"
        (each tail at ``alpha/2``, equal-tailed CI). Only used when
        ``inference="wild_bootstrap"``.
    seed : int, optional
        Random seed for reproducibility when using bootstrap inference.
        If None (default), results will vary between runs.
    rank_deficient_action : str, default "warn"
        Action when design matrix is rank-deficient (linearly dependent columns):
        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning
    conley_coords, conley_cutoff_km, conley_metric, conley_kernel, conley_lag_cutoff
        Conley (1999) spatial-HAC variance configuration. Pass
        ``conley_coords=(lat_col, lon_col)``, ``conley_cutoff_km=<float>``,
        and ``conley_lag_cutoff=<int>`` on the constructor; the ``unit``
        identifier is passed as a fit-time arg to ``fit(...)`` (NOT on
        ``__init__``) — it is unused unless ``vcov_type="conley"`` and is
        therefore not part of ``get_params()`` / ``set_params()`` (which
        return constructor-arg dicts). The block-decomposed panel sandwich
        (matching R ``conleyreg`` with ``lag_cutoff > 0``) sums within-period
        spatial pairs plus within-unit Bartlett serial pairs (lag=0 excluded
        to avoid double-counting). Explicit ``cluster=<col>`` + Conley
        enables the combined spatial + cluster product kernel; the cluster
        must be constant within each unit across periods (validator-enforced).
        DiD has no auto-cluster, so cluster is fully opt-in on the Conley
        path — absent ``cluster=``, pure Conley spatial HAC applies.
        ``survey_design=`` + Conley and ``inference='wild_bootstrap'`` +
        Conley both raise ``NotImplementedError``.
    df_convention : {"residual", "cluster", "normal"}, default "residual"
        Degrees-of-freedom convention for analytical t-statistics, p-values,
        and CIs. ``"residual"`` (default) uses the fitted residual df
        (``n − K_full``); ``"cluster"`` uses the Stata/fixest cluster df
        ``G − 1`` on clustered fits — it has no effect on unclustered fits
        or on ``vcov_type="conley"`` (the combined Conley+cluster product
        kernel has no documented ``G − 1`` df reference and keeps the
        residual df); ``"normal"`` deliberately uses normal-theory z
        inference at the fallback level on every fit, clustered or not.
        Applies only at the fallback level of the df resolution under every
        value: survey df and per-coefficient Bell-McCaffrey DOF
        (``vcov_type="hc2_bm"``) are more refined small-sample corrections
        and always take precedence. Point estimates, SEs, and t-statistics
        are unaffected — only the reference distribution changes. The
        default flips to ``"cluster"`` at v4 (see the REGISTRY clustered-CR1
        inference-df deviation note).

    Attributes
    ----------
    results_ : DiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage with a DataFrame:

    >>> import pandas as pd
    >>> from diff_diff import DifferenceInDifferences
    >>>
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     'outcome': [10, 11, 15, 18, 9, 10, 12, 13],
    ...     'treated': [1, 1, 1, 1, 0, 0, 0, 0],
    ...     'post': [0, 0, 1, 1, 0, 0, 1, 1]
    ... })
    >>>
    >>> # Fit the model
    >>> did = DifferenceInDifferences()
    >>> results = did.fit(data, outcome='outcome', treatment='treated', post='post')
    >>>
    >>> # View results
    >>> print(results.att)  # ATT estimate
    >>> results.print_summary()  # Full summary table

    Using formula interface:

    >>> did = DifferenceInDifferences()
    >>> results = did.fit(data, formula='outcome ~ treated * post')

    Notes
    -----
    The ATT is computed using the standard DiD formula:

        ATT = (E[Y|D=1,T=1] - E[Y|D=1,T=0]) - (E[Y|D=0,T=1] - E[Y|D=0,T=0])

    Or equivalently via OLS regression:

        Y = α + β₁*D + β₂*T + β₃*(D×T) + ε

    Where β₃ is the ATT.
    """

    def __init__(
        self,
        robust: Optional[bool] = None,
        cluster: Optional[str] = None,
        vcov_type: Optional[str] = None,
        alpha: float = 0.05,
        inference: str = "analytical",
        n_bootstrap: int = 999,
        bootstrap_weights: str = "rademacher",
        p_val_type: str = "two-tailed",
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        conley_coords: Optional[Tuple[str, str]] = None,
        conley_cutoff_km: Optional[float] = None,
        conley_metric: str = "haversine",
        conley_kernel: str = "bartlett",
        conley_lag_cutoff: Optional[int] = None,
        df_convention: str = "residual",
    ):
        # Resolve vcov_type from the legacy `robust` alias via the shared
        # helper so __init__ and set_params use identical validation logic.
        from diff_diff.linalg import resolve_vcov_type

        validate_df_convention(df_convention)
        validate_n_bootstrap(n_bootstrap)
        # Fail-closed inference selector (M-096): an unrecognized or
        # non-string value must never silently route to analytical. The
        # isinstance guard matters — bare tuple membership admits a
        # one-element ndarray via elementwise __eq__.
        if not isinstance(inference, str) or inference not in _INFERENCE_METHODS:
            raise ValueError(f"inference must be one of {_INFERENCE_METHODS}, got {inference!r}")

        # `robust` is deprecated (rows M-045..M-047; removed in 4.0). None is
        # the not-supplied sentinel: default constructions and get_params
        # round-trips stay silent, only an explicit robust= warns. The raw
        # arg lives at `_robust_arg` (what get_params returns); the PUBLIC
        # `self.robust` keeps the RESOLVED legacy bool so pre-3.9 attribute
        # readers keep seeing True/False until the 4.0 removal.
        if robust is not None:
            warn_deprecated_kwarg(type(self).__name__, "robust", "use vcov_type= instead")
        self._robust_arg = robust
        self.robust = robust if robust is not None else True
        self.cluster = cluster
        self.vcov_type = resolve_vcov_type(robust, vcov_type)
        # Preserve the raw constructor arg (possibly None) alongside the
        # resolved `vcov_type`. `get_params()` returns the raw arg so
        # sklearn clones preserve the implicit-vs-explicit distinction
        # (and therefore the backward-compat remap). Set only in __init__
        # and updated in ``set_params`` so the flag transitions match the
        # user-visible parameter state.
        self._vcov_type_arg = vcov_type
        self._vcov_type_explicit = vcov_type is not None
        self.alpha = alpha
        self.inference = inference
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        # Test shape for wild cluster bootstrap (mirrors fwildclusterboot's
        # p_val_type): "two-tailed" (default) or "equal-tailed".
        self.p_val_type = p_val_type
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        # Conley spatial-HAC parameters; column names (NOT array values) for
        # the coords. Validation happens at fit() when `data` is in scope.
        self.conley_coords = conley_coords
        self.conley_cutoff_km = conley_cutoff_km
        self.conley_metric = conley_metric
        self.conley_kernel = conley_kernel
        # Phase 2 panel block-decomposed kwarg. The conley_time + conley_unit
        # arrays are auto-derived from data[time].values + data[unit].values
        # at fit-time (panel estimators already take time/unit as column names).
        self.conley_lag_cutoff = conley_lag_cutoff
        # Inference df convention for clustered analytical fits: "residual"
        # (default; t/p/CI at the fitted residual df) or "cluster" (the
        # Stata/fixest G-1 convention). Survey df and per-coefficient
        # Bell-McCaffrey DOF always take precedence over either. The default
        # flips to "cluster" at v4 (REGISTRY clustered-CR1 inference-df
        # deviation note).
        self.df_convention = df_convention

        self.is_fitted_ = False
        self.results_ = None
        self._coefficients = None
        self._vcov = None
        self._bootstrap_results = None  # Store WildBootstrapResults if used

    def fit(
        self,
        data: pd.DataFrame,
        outcome: Optional[str] = None,
        treatment: Optional[str] = None,
        post: Any = NOT_SUPPLIED,
        formula: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        fixed_effects: Optional[List[str]] = None,
        absorb: Optional[List[str]] = None,
        survey_design=None,
        unit: Optional[str] = None,
        time: Any = NOT_SUPPLIED,
    ) -> DiDResults:
        """
        Fit the Difference-in-Differences model.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the outcome, treatment, and time variables.
        outcome : str
            Name of the outcome variable column.
        treatment : str
            Name of the treatment group indicator column (0/1).
        post : str
            Name of the post-treatment period indicator column (0/1).
        formula : str, optional
            R-style formula (e.g., "outcome ~ treated * post").
            If provided, overrides outcome, treatment, and post parameters.
        covariates : list, optional
            List of covariate column names to include as linear controls.
            Names must not collide with reserved structural terms (``const``,
            the treatment/time column names, the ``{treatment}:{time}``
            interaction, fixed-effect dummy names, or internal working columns)
            and must be unique; a collision or duplicate raises ``ValueError``
            (it would otherwise silently overwrite a structural coefficient).
        fixed_effects : list, optional
            List of categorical column names to include as fixed effects.
            Creates dummy variables for each category (drops first level).
            Use for low-dimensional fixed effects (e.g., industry, region).
        absorb : list, optional
            List of categorical column names for high-dimensional fixed effects.
            Uses within-transformation (demeaning) instead of dummy variables.
            More efficient for large numbers of categories (e.g., firm, individual).
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference. When provided,
            uses Taylor Series Linearization for variance estimation and
            applies sampling weights to the regression.
        unit : str, optional
            Name of the unit identifier column. Required ONLY when
            ``vcov_type="conley"`` — the panel block-decomposed Conley
            sandwich (matching R ``conleyreg`` with ``lag_cutoff > 0``)
            needs the unit identifier to compute the per-unit serial sum.
            Mirrors :meth:`MultiPeriodDiD.fit(unit=...)` and
            :meth:`TwoWayFixedEffects.fit(unit=...)`. Fit-time only — NOT
            a constructor kwarg, so it is not part of ``get_params()`` /
            ``set_params()`` (which return constructor-arg dicts).
            Ignored when ``vcov_type`` is not ``"conley"``.

        Returns
        -------
        DiDResults
            Object containing estimation results.

        Raises
        ------
        ValueError
            If required parameters are missing or data validation fails, or if
            a covariate name collides with a reserved structural term name or
            duplicates another covariate.

        Examples
        --------
        Using fixed effects (dummy variables):

        >>> did.fit(data, outcome='sales', treatment='treated', post='post',
        ...         fixed_effects=['state', 'industry'])

        Using absorbed fixed effects (within-transformation):

        >>> did.fit(data, outcome='sales', treatment='treated', post='post',
        ...         absorb=['firm_id'])

        The keyword-only ``time`` parameter is a deprecated alias for
        ``post`` (row M-030); it warns with ``FutureWarning`` and will be
        removed in 4.0.
        """
        post = resolve_renamed_kwarg(
            f"{type(self).__name__}.fit", "time", time, "post", post, default=None
        )
        # Body-local name; the public parameter is post (M-030).
        time = post
        # Per-fit bootstrap state: cleared up front so the result builder
        # labels inference from THIS fit only. Without the reset, a wild fit
        # followed by set_params(inference="analytical") + refit reported
        # stale inference_method="wild_bootstrap" + bootstrap metadata.
        self._bootstrap_results = None
        # Parse formula if provided
        if formula is not None:
            outcome, treatment, time, covariates = self._parse_formula(formula, data)
        elif outcome is None or treatment is None or time is None:
            raise ValueError(
                "Must provide either 'formula' or all of 'outcome', 'treatment', and 'post'"
            )

        # Validate inputs
        self._validate_data(data, outcome, treatment, time, covariates)

        # Validate binary variables BEFORE any transformations
        validate_binary(data[treatment].values, "treatment")
        validate_binary(data[time].values, "time")

        # Validate fixed effects and absorb columns
        if fixed_effects:
            for fe in fixed_effects:
                if fe not in data.columns:
                    raise ValueError(f"Fixed effect column '{fe}' not found in data")
        if absorb:
            for ab in absorb:
                if ab not in data.columns:
                    raise ValueError(f"Absorb column '{ab}' not found in data")

        # Resolve survey design if provided
        from diff_diff.survey import _resolve_effective_cluster, _resolve_survey_for_fit

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, self.inference)
        )
        _uses_replicate = resolved_survey is not None and resolved_survey.uses_replicate_variance
        if _uses_replicate and self.inference == "wild_bootstrap":
            raise ValueError(
                "Cannot use inference='wild_bootstrap' with replicate-weight "
                "survey designs. Replicate weights provide their own variance "
                "estimation."
            )
        _replicate_vcov_remap = _uses_replicate and self._warn_replicate_vcov_ignored()

        # Handle absorbed fixed effects (within-transformation)
        working_data = data.copy()
        absorbed_vars = []
        n_absorbed_effects = 0

        # Save raw treatment counts before absorb demeaning
        n_treated_raw = int(np.sum(data[treatment].values.astype(float)))
        n_control_raw = len(data) - n_treated_raw

        # Reject the `absorb + fixed_effects` mutual-exclusion combination
        # BEFORE any auto-route. R4 review caught a contract-drift where the
        # auto-route silently merged the two arguments on the HC2/HC2-BM
        # path — the public API has always treated this combination as
        # invalid (different FE-handling paths; mixing them violates the
        # FWL theorem on the demeaned half), so keep the explicit rejection
        # in front of the auto-route to preserve user-facing behavior.
        if absorb and fixed_effects:
            raise ValueError(
                "Cannot use both absorb and fixed_effects. "
                "The absorb within-transformation does not residualize "
                "fixed_effects dummies, violating the FWL theorem. "
                "Use absorb alone (for high-dimensional FE) "
                "or fixed_effects alone (for low-dimensional FE)."
            )

        # Auto-route absorb → fixed_effects when vcov_type needs the FULL FE
        # hat matrix. HC2 leverage and CR2 Bell-McCaffrey DOF both depend on
        # the full-design hat; FWL preserves coefficients and residuals but
        # not the hat matrix, so the demeaned design's leverage is wrong for
        # these vcov families. Building the full-dummy design and routing
        # through the existing fixed_effects= branch produces the algebraically
        # correct vcov. Empirically matches `lm() + sandwich::vcovHC` and
        # `lm() + clubSandwich::vcovCR` (singleton-cluster trick for one-way
        # HC2-BM; PT2018 §3.3 unweighted CR2 algebra) at ~1e-14.
        # Conley vcov is unaffected: the absorb+Conley path (Wave A) computes
        # the panel sandwich on demeaned scores, which is FWL-correct because
        # Conley's meat uses only residuals (no leverage term).
        # HC1/CR1 paths remain on the demeaned design (no leverage term).
        # Note: the user-facing `result.coefficients` under this auto-route
        # will include the FE-dummy entries (matching the fixed_effects= path),
        # not the slope-only view that a plain `absorb=` returns.
        #
        # Placement: this auto-route runs BEFORE the legacy multi-absorb +
        # survey-weights guard because that guard's rationale ("single-pass
        # demeaning is not the correct weighted FWL projection for N > 1
        # dimensions") doesn't apply when we're about to swap absorb for
        # fixed_effects: the fixed_effects= path builds the full-dummy design
        # and solves WLS directly, with no within-transform step. R2 review
        # surfaced the scope mismatch (REGISTRY/CHANGELOG said "SUPPORTED" but
        # the survey guard fired first on weighted multi-absorb fits).
        # Route on the EFFECTIVE vcov family: under a replicate design the
        # remap to hc1 must also disable this full-dummy swap, or an
        # explicit hc2 request would still change the result surface
        # (full-dummy coefficients vs absorbed reduced fit) despite the
        # "has no effect" warning.
        if absorb and not _replicate_vcov_remap and self.vcov_type in ("hc2", "hc2_bm", "hc3"):
            fixed_effects = list(fixed_effects or []) + list(absorb)
            absorb = None
            absorbed_vars = []
            n_absorbed_effects = 0

        # Weighted multiple absorbed FE is supported: the absorb path below uses
        # iterative alternating projections (demean_by_groups), the exact weighted
        # FWL projection for N > 1 dimensions on both balanced and unbalanced panels.

        # Validate vcov_type="conley" wire-up. DiD.fit() accepts `unit`
        # as a fit-time arg (NOT on __init__) because cluster/unit
        # semantics on DiD are opt-in rather than auto-derived (unlike
        # MultiPeriodDiD / TwoWayFixedEffects which have a unit declaration
        # at fit-time anyway). The panel block-decomposed Conley sandwich
        # (matching R conleyreg with lag_cutoff > 0) needs unit/time/coords
        # to assemble the within-period spatial and within-unit serial
        # sums; we mirror MultiPeriodDiD's reject pattern for missing args
        # and the survey/wild-bootstrap incompatibilities.
        if self.vcov_type == "conley":
            # Shared front-door validation across DiD / MPD / TWFE entry
            # points (Wave A holistic fix: replaces the inline drift that
            # accumulated across CI R1/R2/R6 — same-class validation gaps
            # mirrored across estimator surfaces).
            from diff_diff.conley import _validate_conley_estimator_inputs

            _validate_conley_estimator_inputs(
                estimator_name="DifferenceInDifferences",
                data=data,
                unit=unit,
                conley_coords=self.conley_coords,
                conley_cutoff_km=self.conley_cutoff_km,
                conley_lag_cutoff=self.conley_lag_cutoff,
                survey_design=survey_design,
                inference=self.inference,
                cluster=self.cluster,
            )

        # Fail-closed wild-bootstrap coherence (M-096). Placed AFTER the
        # survey and Conley front doors so their NotImplementedError
        # rejections keep precedence (raising "pass cluster=" on a
        # wild+Conley fit would be contradictory guidance — Conley rejects
        # the combination regardless of cluster).
        if self.inference == "wild_bootstrap":
            if self.cluster is None:
                raise ValueError(
                    "inference='wild_bootstrap' requires cluster=. The wild cluster "
                    "bootstrap resamples at the cluster level; pass cluster= or use "
                    "inference='analytical'."
                )
            if self.n_bootstrap < 2:
                raise ValueError(
                    f"inference='wild_bootstrap' requires n_bootstrap >= 2 "
                    f"(got {self.n_bootstrap}). At least 2 replications are needed "
                    f"for bootstrap inference; use inference='analytical' for "
                    f"analytical SEs."
                )

        if absorb:
            # FWL theorem: demean ALL regressors alongside outcome.
            # Regressors collinear with absorbed FE (e.g., treatment after
            # absorbing unit FE) will zero out and be handled by rank-deficiency.
            working_data["_treat_time"] = working_data[treatment].values.astype(
                float
            ) * working_data[time].values.astype(float)
            vars_to_demean = [outcome, treatment, time, "_treat_time"] + (covariates or [])
            _absorb_regressors = vars_to_demean[1:]  # everything except outcome
            _pre_norms = pre_demean_norms(working_data, _absorb_regressors, weights=survey_weights)
            # Absorbed df MUST be measured before the in-place demean below
            # overwrites the group columns with demeaned floats. Equals
            # demean_by_groups' historical `sum_d (n_d - 1)` on a connected panel;
            # smaller when the incidence graph splits (disconnected/hierarchical).
            _absorbed_df = absorbed_fe_rank(
                working_data,
                list(absorb),
                has_intercept_col=True,
                weights=survey_weights,
            )
            # Stash the raw FE columns: the clustered-CR1 K_reference
            # increment needs them AFTER the effective cluster resolves,
            # but the in-place demean below overwrites them with floats.
            _fe_cols_raw = working_data[list(absorb)].copy()
            # Method of alternating projections: for N > 1 absorbed dimensions a
            # single sequential sweep is only exact on balanced (orthogonal-FE)
            # panels; demean_by_groups iterates to the exact (W)LS-FWL residual.
            working_data, _ = demean_by_groups(  # count superseded by absorbed_fe_rank above
                working_data,
                vars_to_demean,
                list(absorb),
                inplace=True,
                weights=survey_weights,
            )
            # FE-spanned regressors demean to numerical junk, not exact zero;
            # snap them so rank handling drops them deterministically instead
            # of the junk direction perturbing the identified coefficients.
            snap_absorbed_regressors(
                working_data,
                _absorb_regressors,
                _pre_norms,
                absorbed_desc=f"absorb={list(absorb)}",
                group_vars=list(absorb),
                rank_deficient_action=self.rank_deficient_action,
                display_names={"_treat_time": f"{treatment}:{time}"},
                weights=survey_weights,
            )
            n_absorbed_effects += _absorbed_df
            absorbed_vars = list(absorb)

        # Extract variables (may be demeaned if absorb was used)
        y = working_data[outcome].values.astype(float)
        d = working_data[treatment].values.astype(float)
        t = working_data[time].values.astype(float)

        # Create interaction term
        if absorb:
            dt = working_data["_treat_time"].values.astype(float)
        else:
            dt = d * t

        # Reject covariate names that collide with reserved structural terms.
        # Covariate names are appended verbatim to var_names below and zipped
        # into coef_dict, so a covariate named like a structural term would
        # silently overwrite that coefficient (dict last-write-wins). The
        # reserved set covers the intercept, treatment/time indicators, the
        # interaction, the internal _treat_time working column, and any
        # fixed-effect dummy names (derived via fe_dummy_names WITHOUT
        # materializing the dummy matrix; names match the get_dummies build
        # below exactly). validate_design_term_names re-checks the FINAL list.
        _reserved = {"const", treatment, time, f"{treatment}:{time}", "_treat_time"}
        if fixed_effects:
            for fe in fixed_effects:
                _reserved.update(fe_dummy_names(working_data[fe], fe))
        validate_covariate_names(covariates, _reserved, estimator="DifferenceInDifferences")

        # Build design matrix
        X = np.column_stack([np.ones(len(y)), d, t, dt])
        var_names = ["const", treatment, time, f"{treatment}:{time}"]

        # Add covariates if provided
        if covariates:
            for cov in covariates:
                X = np.column_stack([X, working_data[cov].values.astype(float)])
                var_names.append(cov)

        # Add fixed effects as dummy variables
        if fixed_effects:
            # Shared drop-first dummy build (names match fe_dummy_names, the
            # reserved-name guard above). Use working_data to be consistent
            # with absorbed FE if both are used.
            _fe_blocks, _fe_names = build_fe_dummy_blocks(working_data, list(fixed_effects))
            X = np.column_stack([X] + _fe_blocks)
            var_names.extend(_fe_names)

        # Reject any duplicate in the FINAL term list (e.g. a fixed-effect dummy
        # colliding with a structural term) BEFORE the regression — so the fit is
        # not wasted and no misleading multicollinearity warning is emitted ahead
        # of the intended ValueError.
        validate_design_term_names(var_names, estimator="DifferenceInDifferences")

        # Extract ATT index (coefficient on interaction term)
        att_idx = 3  # Index of interaction term
        att_var_name = f"{treatment}:{time}"
        assert var_names[att_idx] == att_var_name, (
            f"ATT index mismatch: expected '{att_var_name}' at index {att_idx}, "
            f"but found '{var_names[att_idx]}'"
        )

        # Always use LinearRegression for initial fit (unified code path)
        # For wild bootstrap, we don't need cluster SEs from the initial fit
        cluster_ids = data[self.cluster].values if self.cluster is not None else None

        # When survey PSU is present, it overrides cluster for variance estimation
        effective_cluster_ids = _resolve_effective_cluster(
            resolved_survey, cluster_ids, self.cluster
        )

        # Inject cluster as effective PSU for survey variance estimation
        if resolved_survey is not None and effective_cluster_ids is not None:
            from diff_diff.survey import _inject_cluster_as_psu, compute_survey_metadata

            resolved_survey = _inject_cluster_as_psu(resolved_survey, effective_cluster_ids)
            if resolved_survey.psu is not None and survey_metadata is not None:
                raw_w = (
                    data[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(data), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

        # When absorb + replicate: pass survey_design=None to prevent
        # LinearRegression from computing replicate vcov on already-demeaned
        # data (demeaning depends on weights, so replicate refits must re-demean).
        _lr_survey = resolved_survey
        if _uses_replicate and absorbed_vars:
            _lr_survey = None

        # Remap implicit "classical" + cluster to CR1 for legacy-alias
        # backward compatibility (see `_resolve_effective_vcov_type`).
        _fit_vcov_type = (
            "hc1"
            if _replicate_vcov_remap
            else self._resolve_effective_vcov_type(effective_cluster_ids)
        )

        # Build Conley coord/time/unit arrays when applicable. CRITICAL:
        # read from the ORIGINAL `data` frame, NOT `working_data` — `absorb`
        # demeans `time` (and any column listed in `absorb`) in working_data,
        # so reading `working_data[time]` would silently partition the
        # within-period spatial sandwich on residualized floats instead of
        # the true pre/post periods (Codex Wave A R1 P0). Coords are likewise
        # read from raw `data` for symmetry with TwoWayFixedEffects
        # (`twfe.py::TwoWayFixedEffects.fit`) which has the same FWL-
        # composability contract: the meat is computed on demeaned scores
        # but the kernel grid uses the original space (coords) and time/unit
        # indexing. `_compute_conley_vcov` normalizes time labels to dense
        # codes 0..T-1 internally, so non-numeric `time` labels (datetime64,
        # pd.Period, strings) still work on the MultiPeriodDiD path; DiD's
        # binary `time` column is integer 0/1 by convention and is unaffected
        # by the normalization.
        if _fit_vcov_type == "conley":
            # Validated by the conley front-door (_validate_conley_estimator_inputs).
            assert self.conley_coords is not None
            _conley_coords_arr: Optional[np.ndarray] = np.column_stack(
                [
                    data[self.conley_coords[0]].values.astype(np.float64),
                    data[self.conley_coords[1]].values.astype(np.float64),
                ]
            )
            _conley_time_arr: Optional[np.ndarray] = np.asarray(data[time].values)
            _conley_unit_arr: Optional[np.ndarray] = data[unit].values
        else:
            _conley_coords_arr = None
            _conley_time_arr = None
            _conley_unit_arr = None

        # Clustered-CR1 K_reference adjustment (variance-conventions.md D2/D1):
        # absorbed FE not nested in the cluster ADD their conditional rank;
        # cluster-nested explicit FE dummies SUBTRACT theirs. Computed only
        # for the effective-hc1 clustered analytical lane — under
        # wild_bootstrap the analytical fit is deliberately unclustered
        # (adjustment travels through the WCB wiring instead), and under a
        # survey design the survey variance replaces the CR1 sandwich
        # wholesale (moot by design).
        _cr1_k_adj = 0
        if (
            _fit_vcov_type == "hc1"
            and self.inference != "wild_bootstrap"
            and effective_cluster_ids is not None
            and resolved_survey is None
        ):
            if absorbed_vars:
                _cr1_k_adj = absorbed_fe_cr1_k_increment(
                    _fe_cols_raw,
                    list(absorb),
                    effective_cluster_ids,
                    has_intercept_col=True,
                    weights=survey_weights,
                )
            elif fixed_effects:
                _nested_fe = cluster_nested_fe_dims(
                    working_data,
                    list(fixed_effects),
                    effective_cluster_ids,
                    weights=survey_weights,
                )
                if _nested_fe:
                    _cr1_k_adj = -absorbed_fe_rank(
                        working_data,
                        _nested_fe,
                        has_intercept_col=True,
                        weights=survey_weights,
                    )

        # Don't forward `robust=self.robust` when the vcov_type has been
        # remapped; `robust=False + vcov_type="hc1"` would otherwise trip
        # the conflict check inside `LinearRegression.__init__`. The
        # remapped vcov_type is the single source of truth for this call.
        reg = LinearRegression(
            include_intercept=False,  # Intercept already in X
            cluster_ids=effective_cluster_ids if self.inference != "wild_bootstrap" else None,
            alpha=self.alpha,
            rank_deficient_action=self.rank_deficient_action,
            weights=survey_weights,
            weight_type=survey_weight_type,
            survey_design=_lr_survey,
            vcov_type=_fit_vcov_type,
            conley_coords=_conley_coords_arr,
            conley_cutoff_km=self.conley_cutoff_km,
            conley_metric=self.conley_metric,
            conley_kernel=self.conley_kernel,
            conley_time=_conley_time_arr,
            conley_unit=_conley_unit_arr,
            conley_lag_cutoff=self.conley_lag_cutoff,
            df_convention=self.df_convention,
        ).fit(X, y, df_adjustment=n_absorbed_effects, cluster_k_adjustment=_cr1_k_adj)

        coefficients = reg.coefficients_
        residuals = reg.residuals_
        fitted = reg.fitted_values_
        assert coefficients is not None
        att = coefficients[att_idx]

        # Get inference - replicate absorb override, bootstrap, or analytical
        if _uses_replicate and absorbed_vars:
            # Estimator-level replicate variance: re-demean + re-solve per replicate
            from diff_diff.survey import compute_replicate_refit_variance
            from diff_diff.utils import safe_inference

            _absorb_list = list(absorbed_vars)  # capture for closure

            # Handle rank-deficient nuisance: refit only identified columns
            _id_mask = ~np.isnan(coefficients)
            _id_cols = np.where(_id_mask)[0]
            _att_idx_reduced = int(np.searchsorted(_id_cols, att_idx))

            def _refit_did_absorb(w_r):
                nz = w_r > 0
                wd = data[nz].copy()
                w_nz = w_r[nz]
                wd["_treat_time"] = wd[treatment].values.astype(float) * wd[time].values.astype(
                    float
                )
                vars_dm = [outcome, treatment, time, "_treat_time"] + (covariates or [])
                _rep_norms = pre_demean_norms(wd, vars_dm[1:], weights=w_nz)
                wd, _ = demean_by_groups(wd, vars_dm, _absorb_list, inplace=True, weights=w_nz)
                # A regressor can become FE-spanned WITHIN a replicate's
                # effective sample (half-sample zeroing): snap silently so the
                # replicate solve drops it (NaN replicate -> invalid) instead
                # of consuming a junk direction.
                snap_absorbed_regressors(
                    wd,
                    vars_dm[1:],
                    _rep_norms,
                    absorbed_desc=f"absorb={_absorb_list}",
                    group_vars=_absorb_list,
                    rank_deficient_action="silent",
                    weights=w_nz,
                )
                y_r = wd[outcome].values.astype(float)
                d_r = wd[treatment].values.astype(float)
                t_r = wd[time].values.astype(float)
                dt_r = wd["_treat_time"].values.astype(float)
                X_r = np.column_stack([np.ones(len(y_r)), d_r, t_r, dt_r])
                if covariates:
                    for cov in covariates:
                        X_r = np.column_stack([X_r, wd[cov].values.astype(float)])
                coef_r, _, _ = solve_ols(
                    X_r[:, _id_cols],
                    y_r,
                    weights=w_nz,
                    weight_type=survey_weight_type,
                    rank_deficient_action="silent",
                    return_vcov=False,
                )
                return coef_r

            vcov_reduced, _n_valid_rep = compute_replicate_refit_variance(
                _refit_did_absorb, coefficients[_id_mask], resolved_survey
            )
            vcov = _expand_vcov_with_nan(vcov_reduced, len(coefficients), _id_cols)
            se = float(np.sqrt(max(vcov[att_idx, att_idx], 0.0)))
            _df_rep = (
                survey_metadata.df_survey
                if survey_metadata and survey_metadata.df_survey
                else 0  # rank-deficient replicate → NaN inference
            )
            # Replicate-refit path is only reached with a resolved design.
            assert resolved_survey is not None
            if _n_valid_rep < resolved_survey.n_replicates:
                _df_rep = _n_valid_rep - 1 if _n_valid_rep > 1 else 0
            if survey_metadata is not None:
                survey_metadata.df_survey = _df_rep if _df_rep > 0 else None
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=_df_rep)
            _inference_df_used = float(_df_rep) if _df_rep is not None and _df_rep > 0 else None
        elif self.inference == "wild_bootstrap" and self.cluster is not None:
            # Override with wild cluster bootstrap inference (bootstrap
            # test-inversion based; no reference t-distribution, so no
            # effective inference df).
            _inference_df_used = None
            # K_reference adjustment for the bootstrap's own CR1 factors,
            # computed against the RAW cluster ids the bootstrap partitions
            # on (NOT effective_cluster_ids — the analytical-lane gate above
            # deliberately passed 0 under wild_bootstrap).
            _wcb_k_adj = 0
            if absorbed_vars:
                _wcb_k_adj = absorbed_fe_cr1_k_increment(
                    _fe_cols_raw,
                    list(absorb),
                    cluster_ids,
                    has_intercept_col=True,
                    weights=survey_weights,
                )
            elif fixed_effects:
                _nested_wcb = cluster_nested_fe_dims(
                    working_data,
                    list(fixed_effects),
                    cluster_ids,
                    weights=survey_weights,
                )
                if _nested_wcb:
                    _wcb_k_adj = -absorbed_fe_rank(
                        working_data,
                        _nested_wcb,
                        has_intercept_col=True,
                        weights=survey_weights,
                    )
            se, p_value, conf_int, t_stat, vcov, _ = self._run_wild_bootstrap_inference(
                X, y, residuals, cluster_ids, att_idx, cluster_k_adjustment=_wcb_k_adj
            )
        else:
            # Use analytical inference from LinearRegression
            # (handles replicate vcov for no-absorb path automatically)
            vcov = reg.vcov_
            inference = reg.get_inference(att_idx)
            se = inference.se
            t_stat = inference.t_stat
            p_value = inference.p_value
            conf_int = inference.conf_int
            _inference_df_used = (
                float(inference.df) if inference.df is not None and inference.df > 0 else None
            )

        r_squared = compute_r_squared(y, residuals)

        # Count observations (use raw counts to avoid demeaned values from absorb)
        n_treated = n_treated_raw
        n_control = n_control_raw

        # Create coefficient dictionary
        coef_dict = {name: coef for name, coef in zip(var_names, coefficients)}

        # Determine inference method and bootstrap info
        inference_method = "analytical"
        n_bootstrap_used = None
        n_clusters_used = None
        p_val_type_used = None
        if self._bootstrap_results is not None:
            inference_method = "wild_bootstrap"
            n_bootstrap_used = self._bootstrap_results.n_bootstrap
            n_clusters_used = self._bootstrap_results.n_clusters
            p_val_type_used = self._bootstrap_results.p_val_type

        # Store results
        self.results_ = DiDResults(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=len(y),
            n_treated=n_treated,
            n_control=n_control,
            alpha=self.alpha,
            coefficients=coef_dict,
            vcov=vcov,
            residuals=residuals,
            fitted_values=fitted,
            r_squared=r_squared,
            inference_method=inference_method,
            n_bootstrap=n_bootstrap_used,
            n_clusters=n_clusters_used,
            p_val_type=p_val_type_used,
            survey_metadata=survey_metadata,
            # Report the family that actually produced the SE, which may be
            # the remapped "hc1" (CR1) under the legacy alias path, not the
            # stored `self.vcov_type`.
            vcov_type=_fit_vcov_type,
            cluster_name=self.cluster,
            conley_lag_cutoff=(self.conley_lag_cutoff if _fit_vcov_type == "conley" else None),
            df_convention=self.df_convention,
            inference_df=_inference_df_used,
        )

        self._coefficients = coefficients
        self._vcov = vcov
        self.is_fitted_ = True

        return self.results_

    def _fit_ols(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Fit OLS regression.

        This method is kept for backwards compatibility. Internally uses the
        unified solve_ols from diff_diff.linalg for optimized computation.

        Parameters
        ----------
        X : np.ndarray
            Design matrix.
        y : np.ndarray
            Outcome vector.

        Returns
        -------
        tuple
            (coefficients, residuals, fitted_values, r_squared)
        """
        # Use unified OLS backend
        coefficients, residuals, fitted, _ = solve_ols(X, y, return_fitted=True, return_vcov=False)
        r_squared = compute_r_squared(y, residuals)

        return coefficients, residuals, fitted, r_squared

    def _run_wild_bootstrap_inference(
        self,
        X: np.ndarray,
        y: np.ndarray,
        residuals: np.ndarray,
        cluster_ids: np.ndarray,
        coefficient_index: int,
        cluster_k_adjustment: int = 0,
    ) -> Tuple[float, float, Tuple[float, float], float, np.ndarray, WildBootstrapResults]:
        """
        Run wild cluster bootstrap inference.

        Parameters
        ----------
        X : np.ndarray
            Design matrix.
        y : np.ndarray
            Outcome vector.
        residuals : np.ndarray
            OLS residuals.
        cluster_ids : np.ndarray
            Cluster identifiers for each observation.
        coefficient_index : int
            Index of the coefficient to compute inference for.
        cluster_k_adjustment : int, default 0
            Signed K_reference adjustment for the bootstrap's CR1 factors
            (nestedness computed by the caller against THESE raw cluster
            ids). Applied to both the analytical SE inside
            ``wild_bootstrap_se`` and the stored-vcov recompute below, so
            ``se == sqrt(vcov[j, j])`` stays exact.

        Returns
        -------
        tuple
            (se, p_value, conf_int, t_stat, vcov, bootstrap_results)
        """
        bootstrap_results = wild_bootstrap_se(
            X,
            y,
            residuals,
            cluster_ids,
            coefficient_index=coefficient_index,
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weights,
            alpha=self.alpha,
            seed=self.seed,
            return_distribution=False,
            p_val_type=self.p_val_type,
            cluster_k_adjustment=cluster_k_adjustment,
        )
        self._bootstrap_results = bootstrap_results

        se = bootstrap_results.se
        p_value = bootstrap_results.p_value
        conf_int = (bootstrap_results.ci_lower, bootstrap_results.ci_upper)
        t_stat = bootstrap_results.t_stat_original

        # Also compute the cluster-robust vcov for storage. Use the rank-aware
        # solve_ols path (silently dropping collinear nuisance columns and
        # NaN-expanding the vcov for them), matching how wild_bootstrap_se itself
        # handles rank-deficient full-dummy designs — `compute_robust_vcov()`
        # inverts the full X'X directly and would raise (or return garbage) on a
        # rank-deficient design even though the ATT and bootstrap are identified.
        # On a saturated design (degenerate bootstrap, NaN se) store a NaN vcov
        # to keep the all-or-nothing NaN contract. (On a full-rank design this
        # vcov is bit-identical to the prior compute_robust_vcov result.)
        if np.isnan(se):
            vcov = np.full((X.shape[1], X.shape[1]), np.nan)
        else:
            _, _, vcov = solve_ols(
                X,
                y,
                cluster_ids=cluster_ids,
                cluster_k_adjustment=cluster_k_adjustment,
                return_vcov=True,
                rank_deficient_action="silent",
            )

        return se, p_value, conf_int, t_stat, vcov, bootstrap_results

    def _parse_formula(
        self, formula: str, data: pd.DataFrame
    ) -> Tuple[str, str, str, Optional[List[str]]]:
        """
        Parse R-style formula.

        Supports basic formulas like:
        - "outcome ~ treatment * time"
        - "outcome ~ treatment + time + treatment:time"
        - "outcome ~ treatment * time + covariate1 + covariate2"

        Parameters
        ----------
        formula : str
            R-style formula string.
        data : pd.DataFrame
            DataFrame to validate column names against.

        Returns
        -------
        tuple
            (outcome, treatment, time, covariates)
        """
        # Split into LHS and RHS
        if "~" not in formula:
            raise ValueError("Formula must contain '~' to separate outcome from predictors")

        lhs, rhs = formula.split("~")
        outcome = lhs.strip()

        # Parse RHS
        rhs = rhs.strip()

        # Check for interaction term
        if "*" in rhs:
            # Handle "treatment * time" syntax
            parts = rhs.split("*")
            if len(parts) != 2:
                raise ValueError("Currently only supports single interaction (treatment * time)")

            treatment = parts[0].strip()
            time = parts[1].strip()

            # Check for additional covariates after interaction
            if "+" in time:
                time_parts = time.split("+")
                time = time_parts[0].strip()
                covariates = [p.strip() for p in time_parts[1:]]
            else:
                covariates = None

        elif ":" in rhs:
            # Handle explicit interaction syntax
            terms = [t.strip() for t in rhs.split("+")]
            interaction_term = None
            main_effects = []
            covariates = []

            for term in terms:
                if ":" in term:
                    interaction_term = term
                else:
                    main_effects.append(term)

            if interaction_term is None:
                raise ValueError("Formula must contain an interaction term (treatment:time)")

            treatment, time = [t.strip() for t in interaction_term.split(":")]

            # Remaining terms after treatment and time are covariates
            for term in main_effects:
                if term != treatment and term != time:
                    covariates.append(term)

            covariates = covariates if covariates else None
        else:
            raise ValueError(
                "Formula must contain interaction term. "
                "Use 'outcome ~ treatment * time' or 'outcome ~ treatment + time + treatment:time'"
            )

        # Validate columns exist
        for col in [outcome, treatment, time]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        if covariates:
            for cov in covariates:
                if cov not in data.columns:
                    raise ValueError(f"Covariate '{cov}' not found in data")

        return outcome, treatment, time, covariates

    def _validate_data(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        covariates: Optional[List[str]] = None,
    ) -> None:
        """Validate input data."""
        # Check DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # Check required columns exist
        required_cols = [outcome, treatment, time]
        if covariates:
            required_cols.extend(covariates)

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        # Check for missing values
        for col in required_cols:
            if data[col].isna().any():
                raise ValueError(f"Column '{col}' contains missing values")

        # Check for sufficient variation
        if data[treatment].nunique() < 2:
            raise ValueError("Treatment variable must have both 0 and 1 values")
        if data[time].nunique() < 2:
            raise ValueError("Time variable must have both 0 and 1 values")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict outcomes using the fitted model.

        Out-of-sample prediction is intentionally unsupported pending a broader
        post-estimation design for estimator result objects. For fitted
        training-data predictions, use ``results_.fitted_values`` after
        :meth:`fit`.

        Parameters
        ----------
        data : pd.DataFrame
            Candidate prediction data. Currently unused because out-of-sample
            prediction is unsupported.

        Returns
        -------
        np.ndarray
            Predicted values.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        NotImplementedError
            Always raised after fitting until the broader post-estimation
            prediction contract is designed.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling predict()")

        raise NotImplementedError(
            "out-of-sample predict() is unsupported pending a broader "
            "post-estimation design. Use results_.fitted_values for fitted "
            "training-data predictions."
        )

    # get_params/set_params come from BaseEstimator. `vcov_type`'s RAW arg
    # lives at `_vcov_type_arg` (the resolved value at `vcov_type`), and
    # get_params must return the raw one: a clone of
    # `DifferenceInDifferences(robust=False, cluster="unit")` must behave
    # the same as the original on a clustered fit, which requires the
    # clone's `__init__` to see `vcov_type=None` (flagging
    # `_vcov_type_explicit=False`) rather than the alias-resolved
    # "classical" (which would mark it explicit and skip the CR1 remap).
    _PARAM_ATTR_ALIASES = {"vcov_type": "_vcov_type_arg", "robust": "_robust_arg"}
    _DERIVED_CONFIG_ATTRS = (
        "vcov_type",
        "_vcov_type_arg",
        "_vcov_type_explicit",
        "robust",
    )

    @classmethod
    def _normalize_set_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        # `robust=` alone re-derives `vcov_type` from the alias: the merged
        # probe config must carry vcov_type=None so `resolve_vcov_type`
        # re-runs on the new `robust` value instead of seeing a conflict
        # with the previously stored raw vcov_type.
        if "robust" in params and "vcov_type" not in params:
            params["vcov_type"] = None
        return params

    def _warn_replicate_vcov_ignored(self, stacklevel: int = 3) -> bool:
        """Warn that an explicit ``vcov_type`` has no effect under a
        replicate-weight survey design, and tell the caller to remap the
        fit-time vcov to ``"hc1"``.

        With ``uses_replicate_variance`` the analytical sandwich is replaced
        wholesale by the replicate-refit variance (the per-replicate refits
        return point estimates only), so the requested vcov family cannot
        influence any reported number. Silently honoring the kwarg would
        report hc1-identical output under an ``hc2``/``hc2_bm``/``classical``
        label; remapping the (discarded) base-fit vcov to ``"hc1"`` also
        avoids wasted CR2-BM work and one-way-only validator rejections.
        ``conley`` is excluded — it carries its own survey-design support
        contract and validators, which must keep firing unchanged. Returns
        True when a remap should be applied (explicit non-hc1, non-conley
        vcov_type).
        """
        if not self._vcov_type_explicit or self.vcov_type == "hc1":
            # Explicit hc1 is exactly the remap target (and the value the
            # old TwoWayFixedEffects NotImplementedError guidance told
            # users to pass) — nothing is being overridden, stay quiet.
            return False
        if self.vcov_type == "conley":
            # Conley keeps its own survey-design support contract (TSL via
            # the stratified-Conley sandwich; dedicated per-design
            # validators in diff_diff.conley) — do not warn-and-remap past
            # those gates; let the conley validation speak for itself.
            return False
        warnings.warn(
            f"vcov_type={self.vcov_type!r} has no effect with replicate-weight "
            "survey designs: the replicate-refit variance replaces the "
            "analytical vcov entirely (per-replicate refits return point "
            "estimates only, identical across vcov families). Proceeding "
            "with replicate variance; the base fit uses 'hc1'.",
            UserWarning,
            stacklevel=stacklevel,
        )
        return True

    def _resolve_effective_vcov_type(self, effective_cluster_ids, stacklevel: int = 3) -> str:
        """Pick the ``vcov_type`` to use for a given fit given cluster context.

        Returns ``self.vcov_type`` unchanged in nearly every case. The one
        exception is the legacy-alias path: if the user supplied
        ``robust=False`` (or nothing) without an explicit ``vcov_type=``,
        ``resolve_vcov_type`` stored ``"classical"`` at ``__init__``.
        But ``"classical"`` is one-way only and the linalg validator
        rejects it with ``cluster_ids`` set, so calls like
        ``DifferenceInDifferences(robust=False, cluster="unit")`` that
        previously produced CR1 inference would now fail. To preserve that
        contract, when the stored vcov_type is implicit ``"classical"``
        and a cluster structure is present at fit time, remap to ``"hc1"``
        (which dispatches to CR1 cluster-robust). Emit a UserWarning so
        the remap is not silent.

        Callers should always route ``vcov_type`` through this method
        before passing it into ``solve_ols``/``compute_robust_vcov`` so
        subclasses (and survey-PSU-injected cluster ids) get the same
        backward-compatible treatment.
        """
        if (
            self.vcov_type == "classical"
            and not self._vcov_type_explicit
            and effective_cluster_ids is not None
        ):
            warnings.warn(
                "robust=False with cluster=... (or an auto-injected "
                "cluster from survey/TWFE) now maps to vcov_type='hc1' "
                "to preserve the legacy CR1 cluster-robust behavior. "
                "Pass vcov_type='classical' explicitly to request "
                "non-robust SEs, or vcov_type='hc1' to silence this "
                "warning.",
                UserWarning,
                stacklevel=stacklevel,
            )
            return "hc1"
        return self.vcov_type

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

    def _fit_event_study_core(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        post_periods: Optional[List[Any]],
        covariates: Optional[List[str]],
        fixed_effects: Optional[List[str]],
        absorb: Optional[List[str]],
        reference_period: Any,
        unit: Optional[str],
        survey_design: Any,
        effective_inference: str,
        *,
        include_treatment_main: bool = True,
        warn_legacy_reference_default: bool = True,
        cluster_override: Any = _USE_SELF_CLUSTER,
        estimator_name: str = "MultiPeriodDiD",
        _frame_offset: int = 0,
    ) -> Tuple[MultiPeriodDiDResults, np.ndarray, np.ndarray]:
        """Shared event-study estimation core (row M-010).

        The relocated body of ``MultiPeriodDiD.fit`` (verbatim through the
        3.9 merge; docs/v4-design.md section 4.1): validation, the
        pooled event-study design build, OLS via ``solve_ols``, survey /
        Conley / hc2_bm variance lanes, per-period inference, and the
        ``MultiPeriodDiDResults`` construction. Called by
        ``MultiPeriodDiD.fit`` (all knob defaults) and by
        ``TwoWayFixedEffects`` in event-study mode.

        Parameters (knobs beyond the MPD fit surface)
        ---------------------------------------------
        include_treatment_main : bool
            False omits the treatment main-effect column - the
            ``spec="within"`` design, where D is absorbed by the unit FE
            (omitting it avoids a spurious snap/collinearity warning).
            Interactions are unaffected (built from the raw indicator).
        warn_legacy_reference_default : bool
            False suppresses the M-007 legacy reference-period default
            FutureWarning (the merged event-study mode has no legacy
            default to warn about; MPD keeps warning until 4.0).
        cluster_override : Any
            The RESOLVED cluster column for this fit. The default
            sentinel means "use ``self.cluster``" (MPD behavior). The
            TWFE event-study branch pre-resolves its auto-cluster with
            the Conley / survey-PSU / one-way carve-outs and passes the
            final value; an explicit column here behaves exactly like
            MPD's own explicit ``cluster=`` on every lane (survey PSU
            injection included).
        estimator_name : str
            Producer name for warnings and validation messages, so TWFE
            event-study fits never steer users toward the deprecated
            MultiPeriodDiD class.
        _frame_offset : int
            Extra call frames between the user's fit call and this body
            (1 via ``MultiPeriodDiD.fit``, 2 via the TWFE event-study
            branch). Added to every warning stacklevel that attributed
            to USER code before the extraction, preserving attribution
            bit-identically; library-attributed warnings (e.g. the
            ``solve_ols`` rank-deficiency chain) are deliberately not
            offset - they attributed to this module before the move and
            still do.
        """
        cluster_resolved: Optional[str] = (
            self.cluster if cluster_override is _USE_SELF_CLUSTER else cluster_override
        )
        # Validate basic inputs
        if outcome is None or treatment is None or time is None:
            raise ValueError("Must provide 'outcome', 'treatment', and 'time'")

        # Validate columns exist
        self._validate_data(data, outcome, treatment, time, covariates)

        # Validate treatment is binary
        validate_binary(data[treatment].values, "treatment")

        # Validate unit column and check for staggered adoption
        if unit is not None:
            if unit not in data.columns:
                raise ValueError(f"Unit column '{unit}' not found in data")

            # Check for staggered treatment timing and absorbing treatment
            unit_time_sorted = data.sort_values([unit, time])
            adoption_times = {}
            has_reversal = False
            for u, group in unit_time_sorted.groupby(unit):
                d_vals = group[treatment].values
                # Check for treatment reversal (non-absorbing treatment)
                if not has_reversal and len(d_vals) > 1 and np.any(np.diff(d_vals) < 0):
                    warnings.warn(
                        f"Treatment reversal detected (unit '{u}' transitions from "
                        f"treated to untreated). {estimator_name} assumes treatment is "
                        f"an absorbing state (once treated, always treated). "
                        f"Treatment reversals violate this assumption and may "
                        f"produce unreliable estimates.",
                        UserWarning,
                        stacklevel=2 + _frame_offset,
                    )
                    has_reversal = True
                # Only use units with observed 0→1 transition for adoption timing
                # (skip units that are always treated — can't determine adoption time)
                if 0 in d_vals and 1 in d_vals:
                    adoption_times[u] = group.loc[group[treatment] == 1, time].iloc[0]

            if len(adoption_times) > 0:
                unique_adoption = len(set(adoption_times.values()))
                if unique_adoption > 1:
                    warnings.warn(
                        "Treatment timing varies across units (staggered adoption "
                        f"detected). {estimator_name} assumes simultaneous adoption "
                        "and may produce biased estimates with staggered treatment. "
                        "Consider using CallawaySantAnna or SunAbraham instead.",
                        UserWarning,
                        stacklevel=2 + _frame_offset,
                    )

                # Check for time-varying treatment (D_it instead of D_i)
                # If any unit has a 0→1 transition, the treatment column is D_it.
                # MultiPeriodDiD expects a time-invariant ever-treated indicator.
                warnings.warn(
                    "Treatment indicator varies within units (time-varying "
                    f"treatment detected). {estimator_name}'s event-study "
                    "specification expects a time-invariant ever-treated "
                    "indicator (D_i = 1 for all periods of eventually-treated "
                    "units). With time-varying treatment, pre-period "
                    "interaction coefficients will be unidentified. Consider: "
                    f"df['ever_treated'] = df.groupby('{unit}')['{treatment}']"
                    ".transform('max')",
                    UserWarning,
                    stacklevel=2 + _frame_offset,
                )

        # Get all unique time periods
        all_periods = sorted(data[time].unique())

        if len(all_periods) < 2:
            raise ValueError("Time variable must have at least 2 unique periods")

        # Determine pre and post periods
        if post_periods is None:
            # Default: last half of periods are post-treatment
            mid_point = len(all_periods) // 2
            post_periods = all_periods[mid_point:]
            pre_periods = all_periods[:mid_point]
        else:
            post_periods = list(post_periods)
            pre_periods = [p for p in all_periods if p not in post_periods]

        if len(post_periods) == 0:
            raise ValueError("Must have at least one post-treatment period")

        if len(pre_periods) == 0:
            raise ValueError("Must have at least one pre-treatment period")

        if len(pre_periods) < 2:
            warnings.warn(
                "Only one pre-treatment period available. At least 2 pre-periods "
                "are needed to assess parallel trends. The treatment effect estimate "
                "is still valid, but pre-period coefficients for parallel trends "
                "testing are not available.",
                UserWarning,
                stacklevel=2 + _frame_offset,
            )

        # Validate post_periods are in the data
        for p in post_periods:
            if p not in all_periods:
                raise ValueError(f"Post-period '{p}' not found in time column")

        # Determine reference period (omitted dummy)
        if reference_period is None:
            # Default: last pre-period (e=-1 convention, matches fixest).
            # The M-007 transition warning is MPD-only: the merged TWFE
            # event-study mode was born on the e=-1 convention and has no
            # legacy default to warn about (docs/v4-design.md section 4.1).
            if len(pre_periods) > 1 and warn_legacy_reference_default:
                warnings.warn(
                    f"The default reference_period has changed from the first "
                    f"pre-period ({pre_periods[0]}) to the last pre-period "
                    f"({pre_periods[-1]}) to match the standard e=-1 convention "
                    f"(as used by fixest, did, etc.). "
                    f"To silence this warning, pass "
                    f"reference_period={pre_periods[-1]} explicitly.",
                    FutureWarning,
                    stacklevel=2 + _frame_offset,
                )
            reference_period = pre_periods[-1]
        elif reference_period not in all_periods:
            raise ValueError(f"Reference period '{reference_period}' not found in time column")

        # Disallow post-period reference (downstream logic assumes reference is pre-period)
        if reference_period in post_periods:
            raise ValueError(
                f"reference_period={reference_period} is a post-treatment period. "
                f"The reference period must be a pre-treatment period "
                f"(e.g., the last pre-period {pre_periods[-1]}). "
                f"Post-period references are not supported because the reference "
                f"period is excluded from estimation, which would bias avg_att "
                f"and break downstream inference."
            )

        # Validate fixed effects and absorb columns
        if fixed_effects:
            for fe in fixed_effects:
                if fe not in data.columns:
                    raise ValueError(f"Fixed effect column '{fe}' not found in data")
        if absorb:
            for ab in absorb:
                if ab not in data.columns:
                    raise ValueError(f"Absorb column '{ab}' not found in data")

        # Resolve survey design if provided
        from diff_diff.survey import _resolve_effective_cluster, _resolve_survey_for_fit

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, effective_inference)
        )
        _uses_replicate_mp = resolved_survey is not None and resolved_survey.uses_replicate_variance
        if _uses_replicate_mp and effective_inference == "wild_bootstrap":
            raise ValueError(
                "Cannot use inference='wild_bootstrap' with replicate-weight "
                "survey designs. Replicate weights provide their own variance "
                "estimation."
            )
        _replicate_vcov_remap_mp = _uses_replicate_mp and self._warn_replicate_vcov_ignored(
            stacklevel=3 + _frame_offset
        )

        # Handle absorbed fixed effects (within-transformation)
        working_data = data.copy()
        n_absorbed_effects = 0

        # Save raw treatment counts before absorb demeaning
        n_treated_raw = int(np.sum(data[treatment].values.astype(float)))
        n_control_raw = len(data) - n_treated_raw

        # Mutual-exclusion check runs ABOVE the auto-route below so that the
        # `absorb=..., fixed_effects=...` combination still rejects rather
        # than being silently merged.
        if absorb and fixed_effects:
            raise ValueError(
                "Cannot use both absorb and fixed_effects. "
                "The absorb within-transformation does not residualize "
                "fixed_effects dummies, violating the FWL theorem. "
                "Use absorb alone (for high-dimensional FE) "
                "or fixed_effects alone (for low-dimensional FE)."
            )

        # Auto-route absorb → fixed_effects when vcov_type needs the FULL FE
        # hat matrix. Mirrors the identical pattern in
        # DifferenceInDifferences.fit (PR #458). HC2 leverage and CR2
        # Bell-McCaffrey DOF both depend on the full-design hat; FWL
        # preserves coefficients and residuals but not the hat matrix, so
        # the demeaned design's leverage is wrong for these vcov families.
        # Building the full-dummy design and routing through the existing
        # fixed_effects= branch produces the algebraically correct vcov.
        # Empirically matches `lm() + sandwich::vcovHC` and
        # `lm() + clubSandwich::vcovCR` (singleton-cluster trick for one-way
        # HC2-BM; PT2018 §3.3 unweighted CR2 algebra) at ~1e-15.
        # Conley vcov is unaffected: the absorb+Conley path computes the
        # panel sandwich on demeaned scores, which is FWL-correct because
        # Conley's meat uses only residuals (no leverage term).
        # HC1/CR1 paths remain on the demeaned design (no leverage term).
        #
        # Survey-replicate scope: this also short-circuits the absorb-refit
        # replicate-variance branch below (search "compute_replicate_refit_variance").
        # Correct: with a fixed full-dummy design, replicate variance doesn't
        # need per-replicate refit — the standard compute_replicate_vcov
        # path applies directly because the design matrix does not depend
        # on the replicate weights.
        #
        # Placement: this auto-route runs BEFORE the multi-absorb +
        # survey-weights guard because that guard's rationale ("single-pass
        # demeaning is not the correct weighted FWL projection for N > 1
        # dimensions") doesn't apply when we're about to swap absorb for
        # fixed_effects: the fixed_effects= path builds the full-dummy
        # design and solves WLS directly, with no within-transform step.
        # Route on the EFFECTIVE vcov family (see DifferenceInDifferences).
        if absorb and not _replicate_vcov_remap_mp and self.vcov_type in ("hc2", "hc2_bm", "hc3"):
            fixed_effects = list(fixed_effects or []) + list(absorb)
            absorb = None
            n_absorbed_effects = 0

        # Weighted multiple absorbed FE is supported: the absorb path below uses
        # iterative alternating projections (demean_by_groups), the exact weighted
        # FWL projection for N > 1 dimensions on both balanced and unbalanced panels.

        # MultiPeriodDiD is intrinsically a multi-period panel estimator;
        # Phase 2 panel block-decomposed Conley (matches R conleyreg) needs
        # `unit`, `conley_lag_cutoff`, and `conley_coords` at fit-time. The
        # validation is shared with DiD / TWFE to avoid the validation-class
        # drift that surfaced across Wave A CI R1/R2/R6.
        if self.vcov_type == "conley":
            from diff_diff.conley import _validate_conley_estimator_inputs

            _validate_conley_estimator_inputs(
                estimator_name=estimator_name,
                data=data,
                unit=unit,
                conley_coords=self.conley_coords,
                conley_cutoff_km=self.conley_cutoff_km,
                conley_lag_cutoff=self.conley_lag_cutoff,
                survey_design=survey_design,
                inference=self.inference,
                cluster=cluster_resolved,
            )
        # Pre-compute non_ref_periods (needed for absorb demeaning)
        non_ref_periods = [p for p in all_periods if p != reference_period]

        if absorb:
            # FWL theorem: demean ALL regressors alongside outcome.
            # Regressors collinear with absorbed FE (e.g., treatment after
            # absorbing unit FE) will zero out and be handled by rank-deficiency.
            d_raw = working_data[treatment].values.astype(float)
            t_raw = working_data[time].values
            # include_treatment_main=False (the TWFE spec="within" design)
            # omits the D main-effect working column entirely: D is absorbed
            # by the unit FE, so demeaning it would only snap it to zero and
            # emit a spurious collinearity warning. Interactions are built
            # from the RAW indicator either way.
            if include_treatment_main:
                working_data["_did_treatment"] = d_raw
            for period in non_ref_periods:
                working_data[f"_did_period_{period}"] = (t_raw == period).astype(float)
                working_data[f"_did_interact_{period}"] = d_raw * (t_raw == period).astype(float)
            vars_to_demean = (
                [outcome]
                + (["_did_treatment"] if include_treatment_main else [])
                + [f"_did_period_{p}" for p in non_ref_periods]
                + [f"_did_interact_{p}" for p in non_ref_periods]
                + (covariates or [])
            )
            _absorb_regressors = vars_to_demean[1:]  # everything except outcome
            _pre_norms = pre_demean_norms(working_data, _absorb_regressors, weights=survey_weights)
            # Absorbed df MUST be measured before the in-place demean below
            # overwrites the group columns (see the DiD path for the rationale).
            _absorbed_df = absorbed_fe_rank(
                working_data,
                list(absorb),
                has_intercept_col=True,
                weights=survey_weights,
            )
            # Stash the raw FE columns for the clustered-CR1 K_reference
            # increment (computed after the effective cluster resolves; the
            # in-place demean below overwrites these with floats).
            _fe_cols_raw_mp = working_data[list(absorb)].copy()
            # Method of alternating projections (exact for unbalanced panels; a
            # single sequential sweep is exact only on balanced orthogonal-FE panels).
            working_data, _ = demean_by_groups(  # count superseded by absorbed_fe_rank above
                working_data,
                vars_to_demean,
                list(absorb),
                inplace=True,
                weights=survey_weights,
            )
            # Snap FE-spanned regressors (e.g. period dummies when a time
            # dimension is absorbed) to exact zero: rank handling then drops
            # them deterministically instead of their junk directions
            # perturbing the identified interaction coefficients.
            snap_absorbed_regressors(
                working_data,
                _absorb_regressors,
                _pre_norms,
                absorbed_desc=f"absorb={list(absorb)}",
                group_vars=list(absorb),
                rank_deficient_action=self.rank_deficient_action,
                stacklevel=3 + _frame_offset,
                display_names={
                    "_did_treatment": treatment,
                    **{f"_did_period_{p}": f"{time}=={p}" for p in non_ref_periods},
                    **{f"_did_interact_{p}": f"{treatment}:{time}=={p}" for p in non_ref_periods},
                },
                weights=survey_weights,
            )
            n_absorbed_effects += _absorbed_df

        # Extract outcome and treatment (may be demeaned if absorb was used)
        y = working_data[outcome].values.astype(float)
        if absorb and include_treatment_main:
            d = working_data["_did_treatment"].values.astype(float)
        else:
            # Raw indicator: the non-absorb design uses it for the main
            # effect and interactions; with include_treatment_main=False it
            # feeds interactions only (never enters X directly).
            d = working_data[treatment].values.astype(float)
        t = working_data[time].values

        # Reject covariate names that collide with reserved structural terms.
        # Covariates are appended verbatim to var_names below and zipped into
        # coef_dict, so a covariate named like a structural term (intercept,
        # treatment, a period dummy, a treatment-period interaction, an internal
        # _did_* working column, or a fixed-effect dummy) would silently
        # overwrite that coefficient (dict last-write-wins). FE dummy names are
        # derived via fe_dummy_names (no dummy-matrix materialization), matching
        # the construction below (and applying the same fe==time skip).
        # validate_design_term_names re-checks the FINAL list before coef_dict.
        _reserved = {"const", treatment, "_did_treatment"}
        _reserved.update(f"period_{p}" for p in non_ref_periods)
        _reserved.update(f"{treatment}:period_{p}" for p in non_ref_periods)
        _reserved.update(f"_did_period_{p}" for p in non_ref_periods)
        _reserved.update(f"_did_interact_{p}" for p in non_ref_periods)
        if fixed_effects:
            for fe in fixed_effects:
                if fe == time:
                    continue
                _reserved.update(fe_dummy_names(working_data[fe], fe))
        validate_covariate_names(covariates, _reserved, estimator=estimator_name)

        # Build design matrix
        # Start with intercept and (unless omitted) the treatment main effect
        if include_treatment_main:
            X = np.column_stack([np.ones(len(y)), d])
            var_names = ["const", treatment]
        else:
            X = np.ones((len(y), 1))
            var_names = ["const"]

        # Add period dummies (excluding reference period)
        period_dummy_indices = {}  # Map period -> column index in X

        for period in non_ref_periods:
            if absorb:
                period_dummy = working_data[f"_did_period_{period}"].values.astype(float)
            else:
                period_dummy = (t == period).astype(float)
            X = np.column_stack([X, period_dummy])
            var_names.append(f"period_{period}")
            period_dummy_indices[period] = X.shape[1] - 1

        # Add treatment × period interactions for ALL non-reference periods
        # Pre-period interactions test parallel trends; post-period interactions
        # estimate dynamic treatment effects
        interaction_indices = {}  # Map period -> column index in X

        for period in non_ref_periods:
            if absorb:
                interaction = working_data[f"_did_interact_{period}"].values.astype(float)
            else:
                interaction = d * (t == period).astype(float)
            X = np.column_stack([X, interaction])
            var_names.append(f"{treatment}:period_{period}")
            interaction_indices[period] = X.shape[1] - 1

        # Add covariates if provided
        if covariates:
            for cov in covariates:
                X = np.column_stack([X, working_data[cov].values.astype(float)])
                var_names.append(cov)

        # Add fixed effects as dummy variables.
        #
        # MPD's design already absorbs the time dimension via non-reference
        # period dummies (the `period_<X>` columns above) and the treatment-
        # period interactions. If the caller passes the same column as a
        # fixed effect (either explicitly or via the absorb -> fixed_effects
        # auto-route for HC2/HC2-BM), the resulting `<time>_<X>` dummies
        # would be perfectly redundant with the existing period dummies,
        # NaN'd by `solve_ols`'s rank-deficiency handling, AND collide on
        # name with the event-study columns in `coef_dict` (silently
        # collapsing the dict and breaking the coefficients-vs-vcov
        # alignment that downstream consumers rely on). Skip those FEs.
        if fixed_effects:
            _mp_fes = [fe for fe in fixed_effects if fe != time]
            if _mp_fes:
                _fe_blocks, _fe_names = build_fe_dummy_blocks(working_data, _mp_fes)
                X = np.column_stack([X] + _fe_blocks)
                var_names.extend(_fe_names)

        # Reject any duplicate in the FINAL term list (e.g. a fixed-effect dummy
        # colliding with a structural period_{p} key) BEFORE the regression — so
        # the fit is not wasted and no misleading multicollinearity warning is
        # emitted ahead of the intended ValueError.
        validate_design_term_names(var_names, estimator=estimator_name)

        # Fit OLS using unified backend
        # Pass cluster_ids to solve_ols for proper vcov computation
        # This handles rank-deficient matrices by returning NaN for dropped columns
        cluster_ids = data[cluster_resolved].values if cluster_resolved is not None else None

        # When survey PSU is present, it overrides cluster for variance estimation
        effective_cluster_ids = _resolve_effective_cluster(
            resolved_survey, cluster_ids, cluster_resolved
        )

        # Inject cluster as effective PSU for survey variance estimation
        if resolved_survey is not None and effective_cluster_ids is not None:
            from diff_diff.survey import _inject_cluster_as_psu, compute_survey_metadata

            resolved_survey = _inject_cluster_as_psu(resolved_survey, effective_cluster_ids)
            if resolved_survey.psu is not None and survey_metadata is not None:
                raw_w = (
                    data[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(data), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

        # Determine if survey vcov should be used
        _use_survey_vcov = resolved_survey is not None and resolved_survey.needs_survey_vcov

        # Remap implicit "classical" + cluster to CR1 (legacy backward compat).
        _fit_vcov_type = (
            "hc1"
            if _replicate_vcov_remap_mp
            else self._resolve_effective_vcov_type(
                effective_cluster_ids, stacklevel=3 + _frame_offset
            )
        )

        # Cluster + CR2 Bell-McCaffrey (non-survey, unweighted) shares the SAME
        # expensive CR2 precomputes (per-cluster A_g eigendecompositions, S_W,
        # the residual-maker M) between the vcov and the per-coef/avg-ATT
        # contrast DOF. Rather than let `solve_ols` build them for the vcov and
        # then rebuild them for the contrast DOF below, skip solve_ols's vcov on
        # this path and compute vcov + DOF together via one
        # `_compute_cr2_bm_vcov_and_dof` call (see the hc2_bm DOF block). The
        # `survey_weights is None` clause keeps the bypass byte-identical to
        # solve_ols: solve_ols would pass `weights=survey_weights`, and the
        # one-call path passes `weights=None`, so they only agree when
        # survey_weights is None (the documented contract on this path — survey
        # designs route through the TSL/replicate paths). If a future weighted
        # entry point set survey_weights here, the flag is False and the code
        # falls back to the original two-call behavior.
        _is_mpd_cr2_path = (
            _fit_vcov_type == "hc2_bm"
            and effective_cluster_ids is not None
            and not _use_survey_vcov
            and survey_weights is None
        )

        # Resolve Conley arrays from column names (init-time) plus the
        # estimator's `time` / `unit` columns. CRITICAL: read from the
        # ORIGINAL `data` frame, NOT `working_data` — if absorb is used
        # with overlapping covariates (e.g. lat/lon or time listed in
        # both `absorb` and `conley_coords`/`time`), `working_data` has
        # those columns demeaned and the Conley helper would silently
        # partition the spatial sandwich on residualized inputs.
        # Mirrors the DiD/TWFE contract at `estimators.py::DifferenceInDifferences.fit`
        # and `twfe.py::TwoWayFixedEffects.fit` (FWL composability: the meat
        # is computed on demeaned scores but the kernel grid uses the raw
        # coords + time/unit). When vcov_type != "conley", these are silently
        # ignored downstream (Phase 1 / 2 convention).
        if _fit_vcov_type == "conley":
            # Validated by the conley front-door (_validate_conley_estimator_inputs).
            assert self.conley_coords is not None
            _conley_coords_arr: Optional[np.ndarray] = np.column_stack(
                [
                    data[self.conley_coords[0]].values.astype(np.float64),
                    data[self.conley_coords[1]].values.astype(np.float64),
                ]
            )
            # Preserve the original time-label dtype (int, datetime64, pd.Period,
            # string). `_compute_conley_vcov` normalizes to dense 0..T-1 codes
            # internally; float coercion here would break datetime64 / Period /
            # string encodings before the normalizer runs.
            _conley_time_arr: Optional[np.ndarray] = np.asarray(data[time].values)
            _conley_unit_arr: Optional[np.ndarray] = data[unit].values
        else:
            _conley_coords_arr = None
            _conley_time_arr = None
            _conley_unit_arr = None

        # Clustered-CR1 K_reference adjustment (variance-conventions.md D1/D2).
        # Absorb lane: the absorbed increment (non-nested rank + no intercept
        # term — MPD's X carries an intercept). Non-absorb lanes: the built-in
        # period dummies ARE MPD's time-FE block (a supplied time FE is
        # skipped as redundant above), so they and any placed `_mp_fes`
        # dummies SUBTRACT their joint rank when nested in the cluster —
        # preserving MPD's absorb/fixed_effects equivalence. Survey lanes
        # pass 0 (survey vcov replaces CR1 wholesale).
        _cr1_k_adj_mp = 0
        if _fit_vcov_type == "hc1" and effective_cluster_ids is not None and not _use_survey_vcov:
            if absorb:
                _cr1_k_adj_mp = absorbed_fe_cr1_k_increment(
                    _fe_cols_raw_mp,
                    list(absorb),
                    effective_cluster_ids,
                    has_intercept_col=True,
                    weights=survey_weights,
                )
            else:
                _mp_fe_blocks = list(_mp_fes) if fixed_effects else []
                _nested_mp = cluster_nested_fe_dims(
                    working_data,
                    _mp_fe_blocks + [time],
                    effective_cluster_ids,
                    weights=survey_weights,
                )
                if _nested_mp:
                    _cr1_k_adj_mp = -absorbed_fe_rank(
                        working_data,
                        _nested_mp,
                        has_intercept_col=True,
                        weights=survey_weights,
                    )

        # Note: Wild bootstrap for multi-period effects is complex (multiple coefficients)
        # For now, we use analytical inference even if inference="wild_bootstrap"
        coefficients, residuals, fitted, vcov = solve_ols(  # type: ignore[call-overload, misc]  # mypy gives up on the Optional-arg union explosion ("Not all union combinations were tried")
            X,
            y,
            return_fitted=True,
            return_vcov=(not _use_survey_vcov) and not _is_mpd_cr2_path,
            cluster_ids=effective_cluster_ids,
            column_names=var_names,
            rank_deficient_action=self.rank_deficient_action,
            weights=survey_weights,
            weight_type=survey_weight_type,
            cluster_k_adjustment=_cr1_k_adj_mp,
            vcov_type=_fit_vcov_type,
            conley_coords=_conley_coords_arr,
            conley_cutoff_km=self.conley_cutoff_km,
            conley_metric=self.conley_metric,
            conley_kernel=self.conley_kernel,
            conley_time=_conley_time_arr,
            conley_unit=_conley_unit_arr,
            conley_lag_cutoff=self.conley_lag_cutoff,
        )

        # Compute survey vcov if applicable
        _n_valid_rep_mp = None
        if _use_survey_vcov and _uses_replicate_mp and absorb:
            # Absorb + replicate: estimator-level refit (demeaning depends on weights)
            from diff_diff.survey import compute_replicate_refit_variance

            _absorb_list_mp = list(absorb)
            # Handle rank-deficient nuisance: refit only identified columns
            _id_mask_mp = ~np.isnan(coefficients)
            _id_cols_mp = np.where(_id_mask_mp)[0]

            def _refit_mp_absorb(w_r):
                nz = w_r > 0
                wd = data[nz].copy()
                w_nz = w_r[nz]
                d_raw_ = wd[treatment].values.astype(float)
                t_raw_ = wd[time].values
                if include_treatment_main:
                    wd["_did_treatment"] = d_raw_
                for period_ in non_ref_periods:
                    wd[f"_did_period_{period_}"] = (t_raw_ == period_).astype(float)
                    wd[f"_did_interact_{period_}"] = d_raw_ * (t_raw_ == period_).astype(float)
                vars_dm_ = (
                    [outcome]
                    + (["_did_treatment"] if include_treatment_main else [])
                    + [f"_did_period_{p}" for p in non_ref_periods]
                    + [f"_did_interact_{p}" for p in non_ref_periods]
                    + (covariates or [])
                )
                _rep_norms_mp = pre_demean_norms(wd, vars_dm_[1:], weights=w_nz)
                wd, _ = demean_by_groups(wd, vars_dm_, _absorb_list_mp, inplace=True, weights=w_nz)
                # Replicate-local FE spanning: snap silently (see DiD closure).
                snap_absorbed_regressors(
                    wd,
                    vars_dm_[1:],
                    _rep_norms_mp,
                    absorbed_desc=f"absorb={_absorb_list_mp}",
                    group_vars=_absorb_list_mp,
                    rank_deficient_action="silent",
                    weights=w_nz,
                )
                y_r = wd[outcome].values.astype(float)
                if include_treatment_main:
                    d_r = wd["_did_treatment"].values.astype(float)
                    X_r = np.column_stack([np.ones(len(y_r)), d_r])
                else:
                    X_r = np.ones((len(y_r), 1))
                for period_ in non_ref_periods:
                    X_r = np.column_stack([X_r, wd[f"_did_period_{period_}"].values.astype(float)])
                for period_ in non_ref_periods:
                    X_r = np.column_stack(
                        [X_r, wd[f"_did_interact_{period_}"].values.astype(float)]
                    )
                if covariates:
                    for cov_ in covariates:
                        X_r = np.column_stack([X_r, wd[cov_].values.astype(float)])
                coef_r, _, _ = solve_ols(
                    X_r[:, _id_cols_mp],
                    y_r,
                    weights=w_nz,
                    weight_type=survey_weight_type,
                    rank_deficient_action="silent",
                    return_vcov=False,
                )
                return coef_r

            vcov_reduced_mp, _n_valid_rep_mp = compute_replicate_refit_variance(
                _refit_mp_absorb, coefficients[_id_mask_mp], resolved_survey
            )
            vcov = _expand_vcov_with_nan(vcov_reduced_mp, len(coefficients), _id_cols_mp)
        elif _use_survey_vcov and _uses_replicate_mp:
            # No absorb + replicate: X is fixed, use compute_replicate_vcov directly
            from diff_diff.survey import compute_replicate_vcov

            nan_mask = np.isnan(coefficients)
            if np.any(nan_mask):
                kept_cols = np.where(~nan_mask)[0]
                if len(kept_cols) > 0:
                    vcov_reduced, _n_valid_rep_mp = compute_replicate_vcov(
                        X[:, kept_cols],
                        y,
                        coefficients[kept_cols],
                        resolved_survey,
                        weight_type=survey_weight_type,
                    )
                    vcov = _expand_vcov_with_nan(vcov_reduced, X.shape[1], kept_cols)
                else:
                    vcov = np.full((X.shape[1], X.shape[1]), np.nan)
                    _n_valid_rep_mp = 0
            else:
                vcov, _n_valid_rep_mp = compute_replicate_vcov(
                    X,
                    y,
                    coefficients,
                    resolved_survey,
                    weight_type=survey_weight_type,
                )
        elif _use_survey_vcov:
            from diff_diff.survey import compute_survey_vcov

            nan_mask = np.isnan(coefficients)
            if np.any(nan_mask):
                kept_cols = np.where(~nan_mask)[0]
                if len(kept_cols) > 0:
                    vcov_reduced = compute_survey_vcov(X[:, kept_cols], residuals, resolved_survey)
                    vcov = _expand_vcov_with_nan(vcov_reduced, X.shape[1], kept_cols)
                else:
                    vcov = np.full((X.shape[1], X.shape[1]), np.nan)
            else:
                vcov = compute_survey_vcov(X, residuals, resolved_survey)
        r_squared = compute_r_squared(y, residuals)

        # Degrees of freedom: survey df overrides standard df
        k_effective = int(np.sum(~np.isnan(coefficients)))
        # For fweights, df uses sum(w) - k (effective sample size)
        n_eff_df = len(y)
        if survey_weights is not None and survey_weight_type == "fweight":
            n_eff_df = int(round(np.sum(survey_weights)))
        df = n_eff_df - k_effective - n_absorbed_effects
        _df_cluster_knob_invalid = False
        # Opt-in Stata/fixest cluster-df convention (df_convention="cluster"):
        # the shared analytical df becomes G - 1 on a clustered fit. Placed
        # BEFORE the survey/replicate overrides below (which overwrite df, so
        # survey df keeps precedence) and upstream of the per-period BM-DOF
        # branch (which wins per coefficient on the hc2_bm path). Mirrors
        # LinearRegression's resolution: only positive-weight clusters count
        # on a weighted fit.
        if (
            self.df_convention == "cluster"
            and effective_cluster_ids is not None
            and _fit_vcov_type != "conley"
        ):
            # conley is excluded: the combined Conley+cluster product kernel is
            # a diff-diff convention with no documented G-1 df reference (see
            # the REGISTRY Conley section); its inference keeps the residual df.
            from diff_diff.linalg import effective_cluster_count

            _g_eff_mp = effective_cluster_count(effective_cluster_ids, survey_weights)
            if _g_eff_mp <= 1:
                # Cluster df G - 1 undefined: fail closed with NaN inference
                # (df=0 forces NaN through safe_inference), mirroring
                # LinearRegression.get_inference's guard. Unreachable via the
                # CR1 vcov path (its validator now counts positive-weight
                # clusters for all weight types and raises) — defense-in-depth.
                warnings.warn(
                    "df_convention='cluster' requires at least 2 effective "
                    f"clusters; got {_g_eff_mp}. Inference fields will be NaN.",
                    UserWarning,
                    stacklevel=2 + _frame_offset,
                )
                _df_cluster_knob_invalid = True
                df = 0
            else:
                df = _g_eff_mp - 1
        elif self.df_convention == "normal":
            # Deliberate normal-theory z inference at the fallback level, on
            # every fit (clustered, unclustered, and conley alike). The
            # survey/replicate overrides below still overwrite df, and the
            # per-period BM-DOF branch still wins per coefficient on hc2_bm.
            # Keep textually parallel with LinearRegression.get_inference's
            # "normal" branch (linalg.py).
            df = None

        # Absorbed-FE variance scale (fixest full-K convention): the within-
        # transform solve_ols above scales the non-clustered classical/hc1 vcov
        # by k_visible, but the correct finite-sample count is
        # K_full = k_effective + n_absorbed_effects (matching `df` just above and
        # fixest feols(vcov="iid"/"hetero")). Rescale so the SE's k agrees with
        # the t-df's. Gated exactly as LinearRegression.fit: clustered CR1
        # carries the K_reference accounting through `cluster_k_adjustment`
        # inside the kernel instead (never this rescale — the gates are
        # mutually exclusive on cluster_ids), hc2/hc2_bm use
        # leverage/Satterthwaite DOF, survey has its own df. When the full-K
        # residual dof is non-positive the helper returns NaN and we void the
        # vcov -> NaN inference (fail-closed, per the non-finite-df contract).
        if (
            n_absorbed_effects > 0
            and effective_cluster_ids is None
            and not _use_survey_vcov
            and _fit_vcov_type in ("classical", "hc1")
        ):
            _fe_scale_mp = _absorbed_fe_vcov_scale(n_eff_df, k_effective, n_absorbed_effects)
            if np.isnan(_fe_scale_mp):
                vcov = np.full_like(vcov, np.nan)
            elif _fe_scale_mp != 1.0:
                vcov = vcov * _fe_scale_mp

        if resolved_survey is not None and resolved_survey.df_survey is not None:
            df = resolved_survey.df_survey
        # Replicate df: rank-deficient → NaN inference; dropped replicates → n_valid-1
        if _uses_replicate_mp:
            # The flag definition above guarantees this (mypy can't track it).
            assert resolved_survey is not None
            if resolved_survey.df_survey is None:
                df = 0  # rank-deficient replicate → NaN inference
            if _n_valid_rep_mp is not None and _n_valid_rep_mp < resolved_survey.n_replicates:
                df = _n_valid_rep_mp - 1 if _n_valid_rep_mp > 1 else 0
                if survey_metadata is not None:
                    survey_metadata.df_survey = df if df > 0 else None

        # Guard: fall back to normal distribution if df is non-positive
        # Skip for replicate designs — df=0 is intentional for NaN inference
        if df is not None and df <= 0 and not _uses_replicate_mp and not _df_cluster_knob_invalid:
            warnings.warn(
                f"Degrees of freedom is non-positive (df={df}). "
                "Using normal distribution instead of t-distribution for inference.",
                UserWarning,
                stacklevel=2 + _frame_offset,
            )
            df = None

        # Note: the prior homoskedastic-vcov fallback conditioned on
        # `not self.robust` has been subsumed by the vcov_type dispatch in
        # solve_ols above, which routes vcov_type="classical" through
        # compute_robust_vcov's classical branch (identical math). The
        # explicit branch is no longer needed; vcov above already matches the
        # requested variance family.

        # For hc2_bm with a non-survey fit, compute per-coefficient and
        # per-contrast Bell-McCaffrey Satterthwaite DOF so period-specific
        # effects and the post-period average use correct small-sample DOF
        # rather than the shared n-k fallback.
        _bm_dof_per_coef: Optional[np.ndarray] = None
        _bm_dof_avg: Optional[float] = None
        # On the `_is_mpd_cr2_path` bypass, solve_ols did not compute vcov
        # (return_vcov=False). If every coefficient was dropped, synthesize the
        # all-NaN vcov solve_ols would have returned (linalg.py:1230/1019); the
        # BM DOF block below is skipped (no identified coefficients).
        if _is_mpd_cr2_path and vcov is None and np.all(np.isnan(coefficients)):
            vcov = np.full((X.shape[1], X.shape[1]), np.nan)
        if (
            self.vcov_type == "hc2_bm"
            and not _use_survey_vcov
            and (vcov is not None or _is_mpd_cr2_path)
            and not np.all(np.isnan(coefficients))
        ):
            from diff_diff.linalg import (
                _compute_bm_dof_from_contrasts,
                _compute_cr2_bm_contrast_dof,
                _compute_cr2_bm_vcov_and_dof,
                _compute_hat_diagonals,
            )

            _identified = ~np.isnan(coefficients)
            _kept = np.where(_identified)[0]
            if len(_kept) > 0:
                X_kept = X[:, _kept]
                bread_kept = X_kept.T @ (
                    X_kept * survey_weights[:, np.newaxis] if survey_weights is not None else X_kept
                )
                # Build the contrast matrix: one column per identified coefficient
                # plus one column for the post-period average contrast (1/n_post
                # on each post-period interaction column, 0 elsewhere).
                n_kept = len(_kept)
                # Post-period contrast in full-width k dims, then subset to kept
                post_contrast_full = np.zeros(X.shape[1])
                _n_post = len(post_periods)
                if _n_post > 0:
                    for _p in post_periods:
                        post_contrast_full[interaction_indices[_p]] = 1.0 / _n_post
                post_contrast_kept = post_contrast_full[_kept]
                contrasts = np.column_stack([np.eye(n_kept), post_contrast_kept[:, np.newaxis]])
                # Branch on cluster: one-way HC2-BM vs cluster-aware CR2-BM.
                # Cluster IDs are per-observation length n and are unchanged
                # by the column-drop applied to X (`_kept` indexes columns
                # only); pass `effective_cluster_ids` unmodified.
                if effective_cluster_ids is None:
                    h_diag_kept = _compute_hat_diagonals(X_kept, bread_kept, weights=survey_weights)
                    _dof_all = _compute_bm_dof_from_contrasts(
                        X_kept,
                        bread_kept,
                        h_diag_kept,
                        contrasts,
                        weights=survey_weights,
                    )
                elif _is_mpd_cr2_path:
                    # Cluster-aware CR2 BM: vcov AND the per-coefficient +
                    # post-period-average compound contrast (Gate 6 lift) DOF
                    # from a SINGLE precompute build — the perf dedup. solve_ols
                    # bypassed vcov on this path, so compute it here from the
                    # same (X_kept, residuals, cluster_ids, bread_kept) it would
                    # have used internally (→ byte-identical vcov), then expand
                    # with NaN for dropped columns. weights=None per the
                    # _is_mpd_cr2_path guard (survey_weights is None here; survey
                    # designs route through the TSL path).
                    _vcov_reduced, _dof_all = _compute_cr2_bm_vcov_and_dof(
                        X_kept,
                        effective_cluster_ids,
                        bread_kept,
                        contrasts,
                        residuals=residuals,
                        weights=None,
                    )
                    vcov = _expand_vcov_with_nan(_vcov_reduced, X.shape[1], _kept)
                else:
                    # Defensive fallback, currently UNREACHABLE: reaching the
                    # cluster sub-branch requires `not _use_survey_vcov`, and any
                    # non-None `survey_weights` comes from a SurveyDesign whose
                    # `needs_survey_vcov` is True — so `_use_survey_vcov` would be
                    # True and the whole hc2_bm block is skipped. Hence
                    # `_is_mpd_cr2_path` is always satisfied here in practice (it
                    # already requires `survey_weights is None`), and this branch
                    # only fails safe for a future weighted-cluster entry point.
                    # It mirrors the pre-refactor call EXACTLY (solve_ols's vcov is
                    # kept; DOF is computed weights-free, exactly as the prior code
                    # did) so it adds no new, untested weighted-CR2 DOF behavior.
                    _dof_all = _compute_cr2_bm_contrast_dof(
                        X_kept,
                        effective_cluster_ids,
                        bread_kept,
                        contrasts,
                    )
                # Expand per-coefficient DOF back to full width (NaN for dropped).
                _bm_dof_per_coef = np.full(X.shape[1], np.nan)
                _bm_dof_per_coef[_kept] = _dof_all[:n_kept]
                # Post-period average: last contrast column.
                # Only meaningful if all post-period coefs are identified.
                if np.all(_identified[[interaction_indices[p] for p in post_periods]]):
                    _bm_dof_avg = float(_dof_all[-1])

        # Extract period-specific treatment effects for ALL non-reference periods
        period_effects = {}
        post_effect_values = []
        post_effect_indices = []
        # Per-period df PROVENANCE for the unified event-study surface
        # (row M-092): the df actually handed to each period's
        # safe_inference below. Distinct from `inference_df`, which stores
        # the POST-AVERAGE contrast df - under hc2_bm the per-period BM
        # DOFs differ from it, so broadcasting inference_df would
        # misrepresent the per-period rows.
        es_df_used: Dict[Any, float] = {}

        assert vcov is not None
        for period in non_ref_periods:
            idx = interaction_indices[period]
            effect = coefficients[idx]
            se = np.sqrt(vcov[idx, idx])
            # Use the per-coefficient BM DOF when available (hc2_bm path);
            # otherwise fall back to the shared analytical df. On the hc2_bm path
            # the coefficient's OWN Bell-McCaffrey DOF governs its inference: a
            # non-finite (guard-suppressed, unreliable) BM DOF makes t/p/CI
            # undefined, so pass it through to `safe_inference` (which returns
            # all-NaN) rather than silently falling back to the residual df.
            if _bm_dof_per_coef is not None:
                period_df = float(_bm_dof_per_coef[idx])
            else:
                period_df = df
            t_stat, p_value, conf_int = safe_inference(effect, se, alpha=self.alpha, df=period_df)
            es_df_used[period] = (
                float(period_df)
                if period_df is not None and np.isfinite(period_df) and period_df > 0
                else float("nan")
            )

            period_effects[period] = PeriodEffect(
                period=period,
                effect=effect,
                se=se,
                t_stat=t_stat,
                p_value=p_value,
                conf_int=conf_int,
            )

            if period in post_periods:
                post_effect_values.append(effect)
                post_effect_indices.append(idx)

        # Compute average treatment effect (post-periods only)
        # R-style NA propagation: if ANY post-period effect is NaN, average is undefined
        effect_arr = np.array(post_effect_values)

        _avg_df = None
        if np.any(np.isnan(effect_arr)):
            # Some period effects are NaN (unidentified) - cannot compute valid average
            # This follows R's default behavior where mean(c(1, 2, NA)) returns NA
            avg_att = np.nan
            avg_se = np.nan
            avg_t_stat = np.nan
            avg_p_value = np.nan
            avg_conf_int = (np.nan, np.nan)
        else:
            # All effects identified - compute average normally
            avg_att = float(np.mean(effect_arr))

            # Standard error of average: need to account for covariance
            n_post = len(post_periods)
            sub_vcov = vcov[np.ix_(post_effect_indices, post_effect_indices)]
            avg_var = np.sum(sub_vcov) / (n_post**2)

            if np.isnan(avg_var) or avg_var < 0:
                # Vcov has NaN (dropped columns) - propagate NaN
                avg_se = np.nan
                avg_t_stat = np.nan
                avg_p_value = np.nan
                avg_conf_int = (np.nan, np.nan)
            else:
                avg_se = float(np.sqrt(avg_var))
                # Prefer the contrast-specific BM DOF for the post-period average
                # when hc2_bm is in use; otherwise fall back to the shared df.
                _avg_df = _bm_dof_avg if _bm_dof_avg is not None else df
                avg_t_stat, avg_p_value, avg_conf_int = safe_inference(
                    avg_att, avg_se, alpha=self.alpha, df=_avg_df
                )

        # Count observations (use raw counts to avoid demeaned values from absorb)
        n_treated = n_treated_raw
        n_control = n_control_raw

        # Create coefficient dictionary (var_names uniqueness already enforced
        # before the fit above).
        coef_dict = {name: coef for name, coef in zip(var_names, coefficients)}

        # Store results
        _core_results = MultiPeriodDiDResults(
            period_effects=period_effects,
            avg_att=avg_att,
            avg_se=avg_se,
            avg_t_stat=avg_t_stat,
            avg_p_value=avg_p_value,
            avg_conf_int=avg_conf_int,
            n_obs=len(y),
            n_treated=n_treated,
            n_control=n_control,
            pre_periods=pre_periods,
            post_periods=post_periods,
            alpha=self.alpha,
            coefficients=coef_dict,
            vcov=vcov,
            residuals=residuals,
            fitted_values=fitted,
            r_squared=r_squared,
            reference_period=reference_period,
            interaction_indices=interaction_indices,
            survey_metadata=survey_metadata,
            # Report the family that actually produced the SE; may be the
            # remapped hc1 under the legacy alias path, not self.vcov_type.
            vcov_type=_fit_vcov_type,
            cluster_name=cluster_resolved,
            n_clusters=(
                len(np.unique(effective_cluster_ids)) if effective_cluster_ids is not None else None
            ),
            conley_lag_cutoff=(self.conley_lag_cutoff if _fit_vcov_type == "conley" else None),
            df_convention=self.df_convention,
            inference_df=(
                float(_avg_df)
                if _avg_df is not None and np.isfinite(_avg_df) and _avg_df > 0
                else None
            ),
            event_study_df=es_df_used,
        )

        return _core_results, coefficients, vcov


class MultiPeriodDiD(DifferenceInDifferences):
    """
    Multi-Period Difference-in-Differences estimator.

    .. deprecated:: 3.9
        MultiPeriodDiD is deprecated and will be removed in 4.0 (ledger
        row M-010): use ``TwoWayFixedEffects().fit(..., event_study=True)``
        instead. ``spec="pooled"`` reproduces this estimator's design
        exactly (treatment-group dummy + period dummies, no unit fixed
        effects - the only spec valid for repeated cross-sections); the
        default ``spec="within"`` estimates the unit-FE event study. The
        ``EventStudy`` alias is deprecated with this class.

    Extends the standard DiD to handle multiple pre-treatment and
    post-treatment time periods, providing period-specific treatment
    effects as well as an aggregate average treatment effect.

    Parameters
    ----------
    robust : bool, optional
        DEPRECATED legacy alias for ``vcov_type`` (row M-045; warns with
        ``FutureWarning``, removed in 4.0 - use ``vcov_type=``).
        ``robust=True`` maps to ``vcov_type="hc1"``; ``robust=False`` maps
        to ``vcov_type="classical"``. Explicit ``vcov_type`` overrides
        ``robust`` unless the pair is contradictory (e.g.
        ``robust=False, vcov_type="hc2"`` raises).
    cluster : str, optional
        Column name for cluster-robust standard errors. With ``vcov_type="hc1"``
        dispatches to CR1 (Liang-Zeger). With ``vcov_type="hc2_bm"`` dispatches
        to CR2 cluster-robust SEs with Bell-McCaffrey Satterthwaite DOF on both
        per-period coefficients and the post-period-average ATT contrast (the
        latter via the new ``_compute_cr2_bm_contrast_dof`` helper in
        ``linalg.py``; matches clubSandwich's
        ``Wald_test(test="HTZ")$df_denom`` at atol=1e-10). Weighted CR2-BM
        (``survey_design=``) is a separate, still-gated path.
    vcov_type : {"classical", "hc1", "hc2", "hc2_bm", "hc3", "conley"}, optional
        Variance-covariance family. Defaults to the ``robust`` alias.

        - ``"classical"``: non-robust OLS SEs, ``sigma_hat^2 * (X'X)^{-1}``.
        - ``"hc1"``: heteroskedasticity-robust HC1 with ``n/(n-k)`` adjustment
          (library default). With ``cluster=``, uses CR1 (Liang-Zeger).
        - ``"hc2"``: leverage-corrected meat (one-way only). Errors with
          ``cluster=``; use ``"hc2_bm"`` without cluster for Bell-McCaffrey.
        - ``"hc2_bm"``: one-way HC2 + Imbens-Kolesar (2016) Satterthwaite DOF
          per coefficient plus a contrast-aware DOF for the post-period-average
          ATT. With ``cluster=``, dispatches to Pustejovsky-Tipton (2018)
          CR2 cluster-robust with a Bell-McCaffrey Satterthwaite contrast DOF
          on the post-period average (see ``cluster`` above for parity
          details). Weighted CR2-BM (``survey_design=``) is still gated.
        - ``"hc3"``: jackknife-style leverage correction, meat
          ``e_i^2 / (1 - h_ii)^2`` (one-way only; errors with ``cluster=``).
          A leverage-one observation has no defined HC3 variance and the
          vcov fails closed (warning + NaN inference) rather than flooring
          ``1 - h_ii``. With ``absorb=``, routes through the full-dummy
          design like hc2.
        - ``"conley"``: Conley 1999 spatial-HAC sandwich via the panel
          block-decomposed form (matches R ``conleyreg`` with
          ``lag_cutoff > 0``). Pass ``conley_coords=(lat_col, lon_col)``,
          ``conley_cutoff_km=<float>``, and ``conley_lag_cutoff=<int>`` on
          the constructor; ``unit=`` must be supplied at fit-time. The
          sandwich sums within-period spatial pairs plus within-unit
          Bartlett serial pairs (lag=0 excluded to avoid double-counting);
          this is NOT a multiplicative product kernel. ``conley_time`` is
          auto-derived from the ``time`` column at fit-time and normalized
          to dense panel-period codes ``0..T-1`` so ``conley_lag_cutoff``
          always counts panel periods (works for int / datetime64 /
          ``pd.Period`` / string encodings). Explicit ``cluster=<col>``
          enables the combined spatial + cluster product kernel
          (Wave A #119; cluster must be constant within each unit across
          periods). Restrictions: ``survey_design=`` and
          ``inference="wild_bootstrap"`` raise on this path
          (Phase 5 / follow-up).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    conley_coords, conley_cutoff_km, conley_metric, conley_kernel, conley_lag_cutoff
        Constructor kwargs that take effect when ``vcov_type="conley"``.
        ``conley_coords`` is a ``(lat_col, lon_col)`` tuple of column names
        on ``data``. ``conley_lag_cutoff`` is the within-unit Bartlett lag
        (non-negative int; 0 means within-period spatial only, no serial
        component).

    Attributes
    ----------
    results_ : MultiPeriodDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage with multiple time periods:

    >>> import pandas as pd
    >>> from diff_diff import MultiPeriodDiD
    >>>
    >>> # Create sample panel data with 6 time periods
    >>> # Periods 0-2 are pre-treatment, periods 3-5 are post-treatment
    >>> data = create_panel_data()  # Your data
    >>>
    >>> # Fit the model
    >>> did = MultiPeriodDiD()
    >>> results = did.fit(
    ...     data,
    ...     outcome='sales',
    ...     treatment='treated',
    ...     time='period',
    ...     post_periods=[3, 4, 5]  # Specify which periods are post-treatment
    ... )
    >>>
    >>> # View period-specific effects
    >>> for period, effect in results.period_effects.items():
    ...     print(f"Period {period}: {effect.effect:.3f} (SE: {effect.se:.3f})")
    >>>
    >>> # View average treatment effect
    >>> print(f"Average ATT: {results.avg_att:.3f}")

    Notes
    -----
    The model estimates:

        Y_it = α + β*D_i + Σ_t γ_t*Period_t + Σ_{t≠ref} δ_t*(D_i × 1{t}) + ε_it

    Where:
    - D_i is the treatment indicator
    - Period_t are time period dummies (all non-reference periods)
    - D_i × 1{t} are treatment-by-period interactions (all non-reference)
    - δ_t are the period-specific treatment effects
    - The reference period (default: last pre-period) has δ_ref = 0 by construction

    Pre-treatment δ_t test the parallel trends assumption (should be ≈ 0).
    Post-treatment δ_t estimate dynamic treatment effects.
    The average ATT is computed from post-treatment δ_t only.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Deprecation shim (row M-010): warn, then defer to DiD's __init__.

        The forwarding ``*args/**kwargs`` keeps the constructor surface
        identical to :class:`DifferenceInDifferences` with zero drift risk;
        the ``__signature__`` mirror below restores introspection for
        ``get_params``/``set_params`` (``BaseEstimator._param_names``
        rejects VAR_* parameter kinds) and ``inspect.signature`` callers.
        Known, accepted side effects: ``set_params`` re-emits the warning
        (its transactional probe re-instantiates), and static type
        checkers lose constructor-argument checking for this class during
        the 3.9 window (waiver recorded in DEFERRED.md's decision record;
        runtime validation is unchanged - ``super().__init__`` validates
        eagerly).
        """
        warnings.warn(_MPD_DEPRECATION_MSG, FutureWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        post_periods: Optional[List[Any]] = None,
        covariates: Optional[List[str]] = None,
        fixed_effects: Optional[List[str]] = None,
        absorb: Optional[List[str]] = None,
        reference_period: Any = None,
        unit: Optional[str] = None,
        survey_design=None,
    ) -> MultiPeriodDiDResults:
        """
        Fit the Multi-Period Difference-in-Differences model.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the outcome, treatment, and time variables.
        outcome : str
            Name of the outcome variable column.
        treatment : str
            Name of the treatment group indicator column (0/1). Should be a
            time-invariant ever-treated indicator (D_i = 1 for all periods of
            treated units). If treatment is time-varying (D_it), pre-period
            interaction coefficients will be unidentified.
        time : str
            Name of the time period column (can have multiple values).
        post_periods : list
            List of time period values that are post-treatment.
            All other periods are treated as pre-treatment.
        covariates : list, optional
            List of covariate column names to include as linear controls.
            Names must not collide with reserved structural terms (``const``,
            the treatment column name, ``period_{p}`` dummies, the
            ``{treatment}:period_{p}`` interactions, fixed-effect dummy names, or
            internal working columns) and must be unique; a collision or
            duplicate raises ``ValueError`` (it would otherwise silently
            overwrite a structural coefficient).
        fixed_effects : list, optional
            List of categorical column names to include as fixed effects.
        absorb : list, optional
            List of categorical column names for high-dimensional fixed effects.
        reference_period : any, optional
            The reference (omitted) time period for the period dummies.
            Defaults to the last pre-treatment period (e=-1 convention).
        unit : str, optional
            Name of the unit identifier column. When provided, checks whether
            treatment timing varies across units and warns if staggered adoption
            is detected (suggests CallawaySantAnna instead). Required when
            ``vcov_type="conley"`` (the panel block-decomposed sandwich computes
            a per-unit serial sum). For other ``vcov_type`` values, use the
            ``cluster`` parameter for cluster-robust SEs.
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference. When provided,
            uses Taylor Series Linearization for variance estimation and
            applies sampling weights to the regression.

        Returns
        -------
        MultiPeriodDiDResults
            Object containing period-specific and average treatment effects.

        Raises
        ------
        ValueError
            If required parameters are missing or data validation fails, or if
            a covariate name collides with a reserved structural term name or
            duplicates another covariate.
        """
        # Fall back to analytical inference if wild bootstrap requested
        # (must happen before _resolve_survey_for_fit which rejects bootstrap+survey).
        # SKIP the warning on the Conley path — the Conley validator below
        # raises NotImplementedError for wild_bootstrap + Conley, so emitting
        # the analytical-fallback warning first would produce contradictory
        # guidance on the same call (warn "falling back" + raise "not
        # supported"). The Conley raise takes precedence. Codex CI R11 P3.
        # NOTE: ``p_val_type`` is inherited from DifferenceInDifferences but is
        # inert here — MultiPeriodDiD has no wild-bootstrap path (it falls back
        # to analytical inference below), so the parameter has no effect.
        effective_inference = self.inference
        if self.inference == "wild_bootstrap" and self.vcov_type != "conley":
            warnings.warn(
                "Wild bootstrap inference is not yet supported for MultiPeriodDiD. "
                "Using analytical inference instead.",
                UserWarning,
            )
            effective_inference = "analytical"

        results, coefficients, vcov = self._fit_event_study_core(
            data,
            outcome,
            treatment,
            time,
            post_periods,
            covariates,
            fixed_effects,
            absorb,
            reference_period,
            unit,
            survey_design,
            effective_inference,
            _frame_offset=1,
        )
        self.results_ = results
        self._coefficients = coefficients
        self._vcov = vcov
        self.is_fitted_ = True
        return self.results_

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


# Mirror DiD's constructor signature onto the deprecation shim so
# introspection keeps working: BaseEstimator._param_names raises on
# *args/**kwargs parameter kinds, and inspect.signature honors an explicit
# __signature__ - so get_params/set_params (and the roster contract in
# tests/test_base_estimator.py) see exactly DiD's parameter surface, with
# zero drift risk when DifferenceInDifferences gains a constructor param.
import inspect as _inspect  # noqa: E402

MultiPeriodDiD.__init__.__signature__ = _inspect.signature(  # type: ignore[attr-defined]
    DifferenceInDifferences.__init__
)


# Re-export estimators from submodules for backward compatibility
# These can also be imported directly from their respective modules:
# - from diff_diff.twfe import TwoWayFixedEffects
# - from diff_diff.synthetic_did import SyntheticDiD
# - from diff_diff.synthetic_control import SyntheticControl
from diff_diff.synthetic_control import SyntheticControl  # noqa: E402
from diff_diff.synthetic_did import SyntheticDiD  # noqa: E402
from diff_diff.twfe import TwoWayFixedEffects  # noqa: E402

__all__ = [
    "DifferenceInDifferences",
    "MultiPeriodDiD",
    "TwoWayFixedEffects",
    "SyntheticDiD",
    "SyntheticControl",
]
