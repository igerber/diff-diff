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
    demean_by_groups,
    fe_dummy_names,
    pre_demean_norms,
    safe_inference,
    snap_absorbed_regressors,
    validate_binary,
    validate_covariate_names,
    validate_design_term_names,
    wild_bootstrap_se,
)


class DifferenceInDifferences:
    """
    Difference-in-Differences estimator with sklearn-like interface.

    Estimates the Average Treatment effect on the Treated (ATT) using
    the canonical 2x2 DiD design or panel data with two-way fixed effects.

    Parameters
    ----------
    formula : str, optional
        R-style formula for the model (e.g., "outcome ~ treated * post").
        If provided, overrides column name parameters.
    robust : bool, default=True
        Legacy alias for ``vcov_type``. ``robust=True`` maps to
        ``vcov_type="hc1"``; ``robust=False`` maps to ``vcov_type="classical"``.
        Explicit ``vcov_type`` overrides ``robust`` unless the pair is
        contradictory (e.g. ``robust=False, vcov_type="hc2"`` raises).
    cluster : str, optional
        Column name for cluster-robust standard errors. Combined with
        ``vcov_type``: with ``"hc1"`` dispatches to CR1 (Liang-Zeger); with
        ``"hc2_bm"`` dispatches to CR2 Bell-McCaffrey (Pustejovsky-Tipton 2018
        symmetric-sqrt + Satterthwaite DOF).
    vcov_type : {"classical", "hc1", "hc2", "hc2_bm", "conley"}, optional
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
        number of clusters is small, <50).
    n_bootstrap : int, default=999
        Number of bootstrap replications when inference="wild_bootstrap".
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
    >>> results = did.fit(data, outcome='outcome', treatment='treated', time='post')
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
        robust: bool = True,
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
    ):
        # Resolve vcov_type from the legacy `robust` alias via the shared
        # helper so __init__ and set_params use identical validation logic.
        from diff_diff.linalg import resolve_vcov_type

        self.robust = robust
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
        time: Optional[str] = None,
        formula: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        fixed_effects: Optional[List[str]] = None,
        absorb: Optional[List[str]] = None,
        survey_design=None,
        unit: Optional[str] = None,
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
        time : str
            Name of the post-treatment period indicator column (0/1).
        formula : str, optional
            R-style formula (e.g., "outcome ~ treated * post").
            If provided, overrides outcome, treatment, and time parameters.
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

        >>> did.fit(data, outcome='sales', treatment='treated', time='post',
        ...         fixed_effects=['state', 'industry'])

        Using absorbed fixed effects (within-transformation):

        >>> did.fit(data, outcome='sales', treatment='treated', time='post',
        ...         absorb=['firm_id'])
        """
        # Parse formula if provided
        if formula is not None:
            outcome, treatment, time, covariates = self._parse_formula(formula, data)
        elif outcome is None or treatment is None or time is None:
            raise ValueError(
                "Must provide either 'formula' or all of 'outcome', 'treatment', and 'time'"
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
        if absorb and self.vcov_type in ("hc2", "hc2_bm"):
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
            # Method of alternating projections: for N > 1 absorbed dimensions a
            # single sequential sweep is only exact on balanced (orthogonal-FE)
            # panels; demean_by_groups iterates to the exact (W)LS-FWL residual.
            working_data, n_fe = demean_by_groups(
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
            n_absorbed_effects += n_fe
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
            for fe in fixed_effects:
                # Create dummies, drop first category to avoid multicollinearity
                # Use working_data to be consistent with absorbed FE if both are used
                dummies = pd.get_dummies(working_data[fe], prefix=fe, drop_first=True)
                for col in dummies.columns:
                    X = np.column_stack([X, dummies[col].values.astype(float)])
                    var_names.append(col)

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
        _fit_vcov_type = self._resolve_effective_vcov_type(effective_cluster_ids)

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
        ).fit(X, y, df_adjustment=n_absorbed_effects)

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
            if _n_valid_rep < resolved_survey.n_replicates:
                _df_rep = _n_valid_rep - 1 if _n_valid_rep > 1 else 0
            if survey_metadata is not None:
                survey_metadata.df_survey = _df_rep if _df_rep > 0 else None
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=_df_rep)
        elif self.inference == "wild_bootstrap" and self.cluster is not None:
            # Override with wild cluster bootstrap inference
            se, p_value, conf_int, t_stat, vcov, _ = self._run_wild_bootstrap_inference(
                X, y, residuals, cluster_ids, att_idx
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

    def get_params(self) -> Dict[str, Any]:
        """
        Get estimator parameters (sklearn-compatible).

        Returns the *raw* user input for ``vcov_type`` (``None`` when
        the value was alias-derived from ``robust``). This preserves
        the backward-compat remap semantics across clones: a clone of
        ``DifferenceInDifferences(robust=False, cluster="unit")`` must
        behave the same as the original on a clustered fit, which
        requires the clone's ``__init__`` to see ``vcov_type=None`` (so
        it flags ``_vcov_type_explicit=False``) rather than the
        alias-resolved ``"classical"`` (which would mark it explicit
        and skip the CR1 remap).

        Returns
        -------
        Dict[str, Any]
            Estimator parameters suitable for passing to ``__init__``.
        """
        return {
            "robust": self.robust,
            "cluster": self.cluster,
            "vcov_type": self._vcov_type_arg,  # raw, possibly None
            "alpha": self.alpha,
            "inference": self.inference,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            "p_val_type": self.p_val_type,
            "seed": self.seed,
            "rank_deficient_action": self.rank_deficient_action,
            "conley_coords": self.conley_coords,
            "conley_cutoff_km": self.conley_cutoff_km,
            "conley_metric": self.conley_metric,
            "conley_kernel": self.conley_kernel,
            "conley_lag_cutoff": self.conley_lag_cutoff,
        }

    def set_params(self, **params) -> "DifferenceInDifferences":
        """
        Set estimator parameters (sklearn-compatible).

        After assignment, the ``robust``/``vcov_type`` pair is re-validated via
        the same :func:`diff_diff.linalg.resolve_vcov_type` helper used by
        ``__init__``. Invalid combinations (e.g. ``robust=False`` with
        ``vcov_type="hc2"``) raise ``ValueError`` instead of leaving the
        object in an inconsistent state.

        Parameters
        ----------
        **params
            Estimator parameters.

        Returns
        -------
        self
        """
        from diff_diff.linalg import resolve_vcov_type

        # Validate BEFORE mutating `self`. A failing call must leave the
        # estimator unchanged so callers that catch `ValueError` can keep
        # reasoning about the object; half-mutated state from an earlier
        # partial assignment defeats that guarantee. Compute the resolved
        # `vcov_type` on local variables, then apply all mutations atomically.
        pending_robust = params.get("robust", self.robust)
        pending_vcov_type = params.get("vcov_type", self.vcov_type)

        # First pass: validate that every incoming key is a known attribute
        # so we don't partially apply a batch that ends in "Unknown parameter".
        for key in params:
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key}")

        # Second pass: resolve the robust/vcov_type pair. When the user passes
        # only `robust=` alongside a previously-set non-aliasing `vcov_type`,
        # re-derive `vcov_type` from the new `robust` value for internal
        # consistency (matching the prior behavior, but now on locals).
        if "vcov_type" in params:
            resolved_vcov = resolve_vcov_type(pending_robust, pending_vcov_type)
        elif "robust" in params:
            resolved_vcov = resolve_vcov_type(pending_robust, None)
        else:
            resolved_vcov = self.vcov_type  # no-op if neither changed

        # All validation passed — apply mutations atomically.
        for key, value in params.items():
            setattr(self, key, value)
        self.vcov_type = resolved_vcov
        # Update the raw-vs-resolved tracking. `vcov_type=` in the call
        # updates `_vcov_type_arg` to whatever the user passed (including
        # None); `robust=` alone clears the raw arg since the resolution
        # re-derives from the alias. The `_vcov_type_explicit` flag is
        # True iff the raw arg is non-None.
        if "vcov_type" in params:
            self._vcov_type_arg = params["vcov_type"]
        elif "robust" in params:
            self._vcov_type_arg = None
        self._vcov_type_explicit = self._vcov_type_arg is not None
        return self

    def _resolve_effective_vcov_type(self, effective_cluster_ids) -> str:
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
                stacklevel=3,
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


class MultiPeriodDiD(DifferenceInDifferences):
    """
    Multi-Period Difference-in-Differences estimator.

    Extends the standard DiD to handle multiple pre-treatment and
    post-treatment time periods, providing period-specific treatment
    effects as well as an aggregate average treatment effect.

    Parameters
    ----------
    robust : bool, default=True
        Legacy alias for ``vcov_type``. ``robust=True`` maps to
        ``vcov_type="hc1"``; ``robust=False`` maps to ``vcov_type="classical"``.
        Explicit ``vcov_type`` overrides ``robust`` unless the pair is
        contradictory (e.g. ``robust=False, vcov_type="hc2"`` raises).
    cluster : str, optional
        Column name for cluster-robust standard errors. With ``vcov_type="hc1"``
        dispatches to CR1 (Liang-Zeger). With ``vcov_type="hc2_bm"`` dispatches
        to CR2 cluster-robust SEs with Bell-McCaffrey Satterthwaite DOF on both
        per-period coefficients and the post-period-average ATT contrast (the
        latter via the new ``_compute_cr2_bm_contrast_dof`` helper in
        ``linalg.py``; matches clubSandwich's
        ``Wald_test(test="HTZ")$df_denom`` at atol=1e-10). Weighted CR2-BM
        (``survey_design=``) is a separate, still-gated path.
    vcov_type : {"classical", "hc1", "hc2", "hc2_bm", "conley"}, optional
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
                        f"treated to untreated). MultiPeriodDiD assumes treatment is "
                        f"an absorbing state (once treated, always treated). "
                        f"Treatment reversals violate this assumption and may "
                        f"produce unreliable estimates.",
                        UserWarning,
                        stacklevel=2,
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
                        "detected). MultiPeriodDiD assumes simultaneous adoption "
                        "and may produce biased estimates with staggered treatment. "
                        "Consider using CallawaySantAnna or SunAbraham instead.",
                        UserWarning,
                        stacklevel=2,
                    )

                # Check for time-varying treatment (D_it instead of D_i)
                # If any unit has a 0→1 transition, the treatment column is D_it.
                # MultiPeriodDiD expects a time-invariant ever-treated indicator.
                warnings.warn(
                    "Treatment indicator varies within units (time-varying "
                    "treatment detected). MultiPeriodDiD's event-study "
                    "specification expects a time-invariant ever-treated "
                    "indicator (D_i = 1 for all periods of eventually-treated "
                    "units). With time-varying treatment, pre-period "
                    "interaction coefficients will be unidentified. Consider: "
                    f"df['ever_treated'] = df.groupby('{unit}')['{treatment}']"
                    ".transform('max')",
                    UserWarning,
                    stacklevel=2,
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
                stacklevel=2,
            )

        # Validate post_periods are in the data
        for p in post_periods:
            if p not in all_periods:
                raise ValueError(f"Post-period '{p}' not found in time column")

        # Determine reference period (omitted dummy)
        if reference_period is None:
            # Default: last pre-period (e=-1 convention, matches fixest)
            if len(pre_periods) > 1:
                warnings.warn(
                    f"The default reference_period has changed from the first "
                    f"pre-period ({pre_periods[0]}) to the last pre-period "
                    f"({pre_periods[-1]}) to match the standard e=-1 convention "
                    f"(as used by fixest, did, etc.). "
                    f"To silence this warning, pass "
                    f"reference_period={pre_periods[-1]} explicitly.",
                    FutureWarning,
                    stacklevel=2,
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
        if absorb and self.vcov_type in ("hc2", "hc2_bm"):
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
                estimator_name="MultiPeriodDiD",
                data=data,
                unit=unit,
                conley_coords=self.conley_coords,
                conley_cutoff_km=self.conley_cutoff_km,
                conley_lag_cutoff=self.conley_lag_cutoff,
                survey_design=survey_design,
                inference=self.inference,
                cluster=self.cluster,
            )
        # Pre-compute non_ref_periods (needed for absorb demeaning)
        non_ref_periods = [p for p in all_periods if p != reference_period]

        if absorb:
            # FWL theorem: demean ALL regressors alongside outcome.
            # Regressors collinear with absorbed FE (e.g., treatment after
            # absorbing unit FE) will zero out and be handled by rank-deficiency.
            d_raw = working_data[treatment].values.astype(float)
            t_raw = working_data[time].values
            working_data["_did_treatment"] = d_raw
            for period in non_ref_periods:
                working_data[f"_did_period_{period}"] = (t_raw == period).astype(float)
                working_data[f"_did_interact_{period}"] = d_raw * (t_raw == period).astype(float)
            vars_to_demean = (
                [outcome, "_did_treatment"]
                + [f"_did_period_{p}" for p in non_ref_periods]
                + [f"_did_interact_{p}" for p in non_ref_periods]
                + (covariates or [])
            )
            _absorb_regressors = vars_to_demean[1:]  # everything except outcome
            _pre_norms = pre_demean_norms(working_data, _absorb_regressors, weights=survey_weights)
            # Method of alternating projections (exact for unbalanced panels; a
            # single sequential sweep is exact only on balanced orthogonal-FE panels).
            working_data, n_fe = demean_by_groups(
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
                display_names={
                    "_did_treatment": treatment,
                    **{f"_did_period_{p}": f"{time}=={p}" for p in non_ref_periods},
                    **{f"_did_interact_{p}": f"{treatment}:{time}=={p}" for p in non_ref_periods},
                },
                weights=survey_weights,
            )
            n_absorbed_effects += n_fe

        # Extract outcome and treatment (may be demeaned if absorb was used)
        y = working_data[outcome].values.astype(float)
        if absorb:
            d = working_data["_did_treatment"].values.astype(float)
        else:
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
        validate_covariate_names(covariates, _reserved, estimator="MultiPeriodDiD")

        # Build design matrix
        # Start with intercept and treatment main effect
        X = np.column_stack([np.ones(len(y)), d])
        var_names = ["const", treatment]

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
            for fe in fixed_effects:
                if fe == time:
                    continue
                dummies = pd.get_dummies(working_data[fe], prefix=fe, drop_first=True)
                for col in dummies.columns:
                    X = np.column_stack([X, dummies[col].values.astype(float)])
                    var_names.append(col)

        # Reject any duplicate in the FINAL term list (e.g. a fixed-effect dummy
        # colliding with a structural period_{p} key) BEFORE the regression — so
        # the fit is not wasted and no misleading multicollinearity warning is
        # emitted ahead of the intended ValueError.
        validate_design_term_names(var_names, estimator="MultiPeriodDiD")

        # Fit OLS using unified backend
        # Pass cluster_ids to solve_ols for proper vcov computation
        # This handles rank-deficient matrices by returning NaN for dropped columns
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

        # Determine if survey vcov should be used
        _use_survey_vcov = resolved_survey is not None and resolved_survey.needs_survey_vcov

        # Remap implicit "classical" + cluster to CR1 (legacy backward compat).
        _fit_vcov_type = self._resolve_effective_vcov_type(effective_cluster_ids)

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

        # Note: Wild bootstrap for multi-period effects is complex (multiple coefficients)
        # For now, we use analytical inference even if inference="wild_bootstrap"
        coefficients, residuals, fitted, vcov = solve_ols(
            X,
            y,
            return_fitted=True,
            return_vcov=(not _use_survey_vcov) and not _is_mpd_cr2_path,
            cluster_ids=effective_cluster_ids,
            column_names=var_names,
            rank_deficient_action=self.rank_deficient_action,
            weights=survey_weights,
            weight_type=survey_weight_type,
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
                wd["_did_treatment"] = d_raw_
                for period_ in non_ref_periods:
                    wd[f"_did_period_{period_}"] = (t_raw_ == period_).astype(float)
                    wd[f"_did_interact_{period_}"] = d_raw_ * (t_raw_ == period_).astype(float)
                vars_dm_ = (
                    [outcome, "_did_treatment"]
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
                d_r = wd["_did_treatment"].values.astype(float)
                X_r = np.column_stack([np.ones(len(y_r)), d_r])
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

        # Absorbed-FE variance scale (fixest full-K convention): the within-
        # transform solve_ols above scales the non-clustered classical/hc1 vcov
        # by k_visible, but the correct finite-sample count is
        # K_full = k_effective + n_absorbed_effects (matching `df` just above and
        # fixest feols(vcov="iid"/"hetero")). Rescale so the SE's k agrees with
        # the t-df's. Gated exactly as LinearRegression.fit: clustered SEs keep
        # k_visible (fixest ssc nested-FE convention), hc2/hc2_bm use
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
            if resolved_survey.df_survey is None:
                df = 0  # rank-deficient replicate → NaN inference
            if _n_valid_rep_mp is not None and _n_valid_rep_mp < resolved_survey.n_replicates:
                df = _n_valid_rep_mp - 1 if _n_valid_rep_mp > 1 else 0
                if survey_metadata is not None:
                    survey_metadata.df_survey = df if df > 0 else None

        # Guard: fall back to normal distribution if df is non-positive
        # Skip for replicate designs — df=0 is intentional for NaN inference
        if df is not None and df <= 0 and not _uses_replicate_mp:
            warnings.warn(
                f"Degrees of freedom is non-positive (df={df}). "
                "Using normal distribution instead of t-distribution for inference.",
                UserWarning,
                stacklevel=2,
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
        self.results_ = MultiPeriodDiDResults(
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
            cluster_name=self.cluster,
            n_clusters=(
                len(np.unique(effective_cluster_ids)) if effective_cluster_ids is not None else None
            ),
            conley_lag_cutoff=(self.conley_lag_cutoff if _fit_vcov_type == "conley" else None),
        )

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
