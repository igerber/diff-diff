"""
Borusyak-Jaravel-Spiess (2024) Imputation DiD Estimator.

Implements the efficient imputation estimator for staggered
Difference-in-Differences from Borusyak, Jaravel & Spiess (2024),
"Revisiting Event-Study Designs: Robust and Efficient Estimation",
Review of Economic Studies.

The estimator:
1. Runs OLS on untreated observations to estimate unit + time fixed effects
2. Imputes counterfactual Y(0) for treated observations
3. Aggregates imputed treatment effects with researcher-chosen weights

Inference uses the conservative clustered variance estimator (Theorem 3).

The ``vcov_type`` input contract is permanently narrow to ``{"hc1"}`` per
the influence-function-based variance decomposition: the per-unit IF
aggregation (Theorem 3 equation 7) has no equivalent single design matrix
on which analytical-sandwich families (``classical``, ``hc2``, ``hc2_bm``)
or spatial-HAC composition (``conley``) can be defined. ``cluster=``
invokes per-cluster IF summation; ``survey_design=`` invokes TSL on the
combined IF. See ``docs/methodology/REGISTRY.md`` for the cross-estimator
IF-vs-sandwich taxonomy.
"""

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from diff_diff._base import BaseEstimator
from diff_diff._deprecation import NOT_SUPPLIED
from diff_diff.aggregation import AggregationKit
from diff_diff.imputation_aggregation import (  # noqa: F401 (compat re-exports)
    _compute_target_weights,
    _ImputationAggregationMixin,
    _lsmr_minnorm_normal_solve,
    _LSMRUnconvergedError,
    _UntreatedProjection,
)
from diff_diff.imputation_bootstrap import ImputationDiDBootstrapMixin
from diff_diff.imputation_results import (  # noqa: F401 (re-export)
    ImputationBootstrapResults,
    ImputationDiDResults,
)
from diff_diff.utils import (
    safe_inference,
    validate_df_convention,
    validate_n_bootstrap,
)

if TYPE_CHECKING:
    from diff_diff.survey import SurveyDesign


# =============================================================================
# Main Estimator
# =============================================================================


class ImputationDiD(ImputationDiDBootstrapMixin, _ImputationAggregationMixin, BaseEstimator):
    """
    Borusyak-Jaravel-Spiess (2024) imputation DiD estimator.

    This is the efficient estimator for staggered Difference-in-Differences
    under parallel trends. It produces shorter confidence intervals than
    Callaway-Sant'Anna (~50% shorter) and Sun-Abraham (2-3.5x shorter)
    under homogeneous treatment effects.

    The estimation procedure:
    1. Run OLS on untreated observations to estimate unit + time fixed effects
    2. Impute counterfactual Y(0) for treated observations
    3. Aggregate imputed treatment effects with researcher-chosen weights

    Inference uses the conservative clustered variance estimator from Theorem 3
    of the paper.

    Parameters
    ----------
    anticipation : int, default=0
        Number of periods before treatment where effects may occur.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional
        Column name for cluster-robust standard errors.
        If None, clusters at the unit level by default.
    vcov_type : str, default="hc1"
        Variance estimator family. Permanently narrow to ``{"hc1"}`` per
        the IF-based variance contract (Theorem 3): analytical-sandwich
        families ``{classical, hc2, hc2_bm}`` and ``conley`` are rejected
        at ``__init__`` with methodology-rooted messages. ``cluster=``
        invokes per-cluster IF summation; ``survey_design=`` invokes TSL
        on the combined IF. See REGISTRY.md for the cross-estimator
        IF-vs-sandwich taxonomy.
    n_bootstrap : int, default=0
        Number of bootstrap iterations. If 0, uses analytical inference
        (conservative variance from Theorem 3).
    bootstrap_weights : str, default="rademacher"
        Type of bootstrap weights: "rademacher", "mammen", or "webb".
    seed : int, optional
        Random seed for reproducibility.
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient:
        - "warn": Issue warning and drop linearly dependent columns
        - "error": Raise ValueError
        - "silent": Drop columns silently
    horizon_max : int, optional
        Maximum event-study horizon. If set, event study effects are only
        computed for abs(h) <= horizon_max.
    aux_partition : str, default="cohort_horizon"
        Controls the auxiliary model partition for Theorem 3 variance:
        - "cohort_horizon": Groups by cohort x relative time (tightest SEs)
        - "cohort": Groups by cohort only (more conservative)
        - "horizon": Groups by relative time only (more conservative)
    pretrends : bool, default=False
        If True, event study includes pre-treatment horizons for visual
        pre-trends assessment. Pre-period effects should be ~0 under
        parallel trends. Only affects event_study aggregation; overall
        ATT and group aggregation are unchanged.
    leave_one_out : bool, default=False
        If True, apply the Borusyak-Jaravel-Spiess (2024) Supplementary
        Appendix A.9 leave-one-out finite-sample refinement to the
        conservative variance. The non-LOO auxiliary aggregate ``tau_tilde_g``
        is built from the fitted ``tau_hat_it`` and thus partially overfits to
        the noise ``epsilon_it``, biasing the variance downward. LOO recomputes
        each unit's group aggregate excluding that unit -- implemented
        efficiently by rescaling each treated auxiliary residual by
        ``1 / (1 - v_ig**2 / sum_j v_jg**2)`` (App. A.9), which is exactly
        equivalent to the direct leave-one-out at the per-unit cluster sum.
        Yields a larger, less-downward-biased SE (Prop. A8: unbiased for an
        upper bound). Default False preserves R ``didimputation`` parity; the
        refinement is an option in the authors' Stata ``did_imputation``. LOO
        is undefined for a group with a single positive-weight unit (App. A.9
        footnote 51): such groups fall back to the non-LOO residual with a
        UserWarning. The Prop. A8 direction (LOO >= non-LOO) is guaranteed at
        the default unit clustering; coarser ``cluster=`` / analytical
        ``survey_design=`` / ``n_bootstrap`` compositions apply the same rescale
        but are a library extension beyond the paper's derivation.
        Replicate-weight survey designs raise ``NotImplementedError`` (their
        variance bypasses the influence-function path where the rescale lives).
    df_convention : {"residual", "cluster", "normal"}, default "residual"
        Degrees-of-freedom convention for the PRETRENDS lead regression's
        per-lead t/p/CI — the one ImputationDiD surface running the shared
        clustered CR1 sandwich (``pretrends=True``, surfaced fit-time via
        the deprecated ``aggregate="event_study"``/``"all"`` or post-fit via
        ``results.aggregate('event_study')``). ``"residual"`` (default) uses
        the lead regression's residual df (``n − k_kept − absorbed
        [time, unit] rank``) — the 3.9 fix: previously silent normal-theory
        z on plain clustered fits; ``"cluster"`` uses ``G − 1``;
        ``"normal"`` deliberately uses z. The full-design survey df keeps
        precedence on survey fits. Everything else — the BJS Theorem-3
        overall/event-study inference and the joint pretrend Wald F (which
        keeps its cluster-robust ``F(q, G − 1)`` reference) — is
        knob-independent; an explicitly non-default value on a
        configuration that never surfaces the per-lead inference warns at
        fit time. The default flips to ``"cluster"`` at v4.

    Attributes
    ----------
    results_ : ImputationDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> from diff_diff import ImputationDiD, generate_staggered_data
    >>> data = generate_staggered_data(n_units=200, seed=42)
    >>> est = ImputationDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='time', first_treat='first_treat')
    >>> results.print_summary()

    With a post-fit event study (M-021):

    >>> est = ImputationDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='time', first_treat='first_treat')
    >>> es = results.aggregate('event_study')
    >>> from diff_diff import plot_event_study
    >>> plot_event_study(es)

    Notes
    -----
    The imputation estimator uses ALL untreated observations (never-treated +
    not-yet-treated periods of eventually-treated units) to estimate the
    counterfactual model. There is no ``control_group`` parameter because this
    is fundamental to the method's efficiency.

    References
    ----------
    Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting Event-Study
    Designs: Robust and Efficient Estimation. Review of Economic Studies,
    91(6), 3253-3285.
    """

    def __init__(
        self,
        anticipation: int = 0,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        vcov_type: str = "hc1",
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        horizon_max: Optional[int] = None,
        aux_partition: str = "cohort_horizon",
        pretrends: bool = False,
        leave_one_out: bool = False,
        df_convention: str = "residual",
    ):
        if rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )
        if bootstrap_weights not in ("rademacher", "mammen", "webb"):
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{bootstrap_weights}'"
            )
        if aux_partition not in ("cohort_horizon", "cohort", "horizon"):
            raise ValueError(
                f"aux_partition must be 'cohort_horizon', 'cohort', or 'horizon', "
                f"got '{aux_partition}'"
            )
        self._validate_vcov_type(vcov_type)
        self._validate_leave_one_out(leave_one_out)
        validate_df_convention(df_convention)

        self.anticipation = anticipation
        self.alpha = alpha
        self.cluster = cluster
        self.vcov_type = vcov_type
        validate_n_bootstrap(n_bootstrap)
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.horizon_max = horizon_max
        self.aux_partition = aux_partition
        self.pretrends = pretrends
        self.leave_one_out = leave_one_out
        self.df_convention = df_convention

        self.is_fitted_ = False
        self.results_: Optional[ImputationDiDResults] = None

        # Internal state preserved for pretrend_test()
        self._fit_data: Optional[Dict[str, Any]] = None

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        aggregate: Any = NOT_SUPPLIED,
        balance_e: Any = NOT_SUPPLIED,
        survey_design: Optional["SurveyDesign"] = None,
    ) -> ImputationDiDResults:
        """
        Fit the imputation DiD estimator.

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
        covariates : list of str, optional
            List of covariate column names.
        aggregate : str, optional
            DEPRECATED (3.9, removed in 4.0; row M-021): aggregate as a
            post-fit step instead — ``results.aggregate('event_study')`` /
            ``.aggregate('group')`` / ``.aggregate('simple')``. Supplying
            ANY value (``None`` included) warns; the deprecated path still
            works and returns exactly the numbers it always did
            (fit-time mode: None/"simple" overall only, "event_study",
            "group", or "all").
        balance_e : int, optional
            DEPRECATED (3.9, removed in 4.0; row M-118): moves onto
            ``results.aggregate('event_study', balance_e=...)``. Restricts
            the event study to cohorts observed at every relative time in
            ``[-balance_e, max_h]`` (the balanced-window rule).
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference. Supports
            pweight only (aweight/fweight raise ValueError). Supports strata,
            PSU, and FPC for design-based variance via compute_survey_if_variance().
            Strata enters survey df for t-distribution inference.
            Both analytical (n_bootstrap=0) and bootstrap inference are supported.

        Returns
        -------
        ImputationDiDResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # M-021/M-118 deprecation shim (CS-style joint warning): a plain
        # fit() never warns; supplying EITHER param with ANY value (None
        # included) warns once, then the legacy routing below runs
        # unchanged - the deprecated path returns exactly the numbers it
        # always did (no new value validation; unknown strings still act
        # like None). The post-fit successor validates its own vocabulary.
        # The sentinel is normalized HERE, before every downstream read of
        # ``aggregate`` (the pretrends+replicate gate and the df_convention
        # reachability warning below both read it).
        _deprecated_passed = [
            n
            for n, v in (("aggregate", aggregate), ("balance_e", balance_e))
            if v is not NOT_SUPPLIED
        ]
        if _deprecated_passed:
            _args = " / ".join(f"{n}=" for n in _deprecated_passed)
            warnings.warn(
                f"ImputationDiD.fit({_args}) is deprecated and will be "
                "removed in 4.0. Fit once, then aggregate as a post-fit "
                "step: results = ImputationDiD().fit(...); "
                "results.aggregate('event_study') / .aggregate('group') / "
                ".aggregate('simple'). balance_e moves onto aggregate() "
                "alongside it: results.aggregate('event_study', "
                "balance_e=2).",
                FutureWarning,
                stacklevel=2,
            )
        if aggregate is NOT_SUPPLIED:
            aggregate = None
        if balance_e is NOT_SUPPLIED:
            balance_e = None

        # Re-validate vcov_type at fit-time: set_params validates eagerly
        # (BaseEstimator probe re-init), so this only catches DIRECT
        # attribute mutation (est.vcov_type = ...).
        self._validate_vcov_type(self.vcov_type)
        self._validate_leave_one_out(self.leave_one_out)

        # Validate inputs
        required_cols = [outcome, unit, time, first_treat]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # pretrends + analytical survey is supported (Phase 8e-iii).
        # Replicate-weight surveys need per-replicate lead regression refits
        # which are not yet implemented — reject that combination.
        if (
            self.pretrends
            and survey_design is not None
            and survey_design.replicate_method is not None
            and aggregate in ("event_study", "all")
        ):
            raise NotImplementedError(
                "pretrends=True is not yet compatible with replicate-weight "
                "survey designs. Analytical survey designs (strata/PSU/FPC) "
                "are supported. Use pretrends=False with replicate weights."
            )

        # Inert-config warning (no-silent-failures): the df_convention knob
        # moves only the pretrends lead regression's per-lead t/p/CI. Since
        # the M-021 post-fit migration that inference is REACHABLE from any
        # analytical pretrends=True fit via results.aggregate('event_study'),
        # so the predicate is reachability-based, not aggregate-keyed:
        # reachable iff pretrends AND not replicate-weight (the gate above
        # rejects fit-time ES and the post-fit path fails closed too) AND
        # (the deprecated fit-time ES/all was supplied OR n_bootstrap <= 0 —
        # a bootstrapped fit builds no ES surface and post-fit aggregate()
        # fails closed on it; validate_n_bootstrap rejects negatives at
        # __init__, so 0 is the only reachable off value and `<= 0` is
        # equivalent to `== 0` — kept as-is, no behavior change). Reachability-BASED, not exact: a
        # fit whose bootstrap later FAILS (bootstrap_results=None) can still
        # aggregate post-fit, so that corner warns spuriously — the warning
        # fires before the bootstrap runs and cannot know. (The post-fit
        # ``pretrend_test()`` reaches the lead helper too, but consumes only
        # gamma/V_gamma — its joint Wald F denominator is knob-independent.)
        _is_replicate_design = (
            survey_design is not None and survey_design.replicate_method is not None
        )
        _lead_inference_reachable = (
            self.pretrends
            and not _is_replicate_design
            and (aggregate in ("event_study", "all") or self.n_bootstrap <= 0)
        )
        if self.df_convention != "residual" and not _lead_inference_reachable:
            warnings.warn(
                f"df_convention={self.df_convention!r} affects only the "
                "pretrends event-study per-lead inference (pretrends=True, "
                "surfaced fit-time via the deprecated aggregate="
                "'event_study'/'all' or post-fit via "
                "results.aggregate('event_study')); it has no effect on "
                "this configuration.",
                UserWarning,
                stacklevel=2,
            )

        # Create working copy
        df = data.copy()

        # Resolve survey design if provided
        from diff_diff.survey import (
            _inject_cluster_as_psu,
            _resolve_effective_cluster,
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, survey_weights, _, survey_metadata = _resolve_survey_for_fit(
            survey_design, data, "analytical"
        )

        _uses_replicate_imp = (
            resolved_survey is not None and resolved_survey.uses_replicate_variance
        )
        if _uses_replicate_imp and self.n_bootstrap > 0:
            raise ValueError(
                "Cannot use n_bootstrap > 0 with replicate-weight survey designs. "
                "Replicate weights provide their own variance estimation."
            )
        # Reject replicate-weight + cluster=: replicate IF variance is
        # computed by replicate reweighting (BRR / Fay / JK1 / JKn / SDR)
        # and ignores PSU/cluster entirely (survey.py enforces that
        # replicate_weights are mutually exclusive with strata/psu/fpc).
        # Honoring bare cluster= here would silently have no effect on
        # variance while populating cluster_name/n_clusters on Results
        # dishonestly. Fail-closed mirroring CallawaySantAnna.
        if (
            self.cluster is not None
            and survey_design is not None
            and getattr(survey_design, "replicate_weights", None) is not None
        ):
            raise NotImplementedError(
                f"ImputationDiD(cluster={self.cluster!r}) is not supported "
                "with replicate-weight survey designs. Replicate-weight "
                "variance is computed by replicate reweighting (BRR / Fay / "
                "JK1 / JKn / SDR) and ignores PSU/cluster entirely — setting "
                "cluster= would silently have no effect on the variance "
                "estimate. Either omit cluster= (the replicate weights encode "
                "the design structure implicitly) or use a non-replicate "
                "survey design (with explicit strata/psu/fpc)."
            )
        # Reject replicate-weight + leave_one_out=: the BJS 2024 App. A.9
        # refinement rescales the conservative influence-function auxiliary
        # residuals, but replicate-weight variance is computed by per-replicate
        # point-estimate refits (not the IF path), so leave_one_out would
        # silently have no effect. Fail-closed (no-silent-failures).
        if _uses_replicate_imp and self.leave_one_out:
            raise NotImplementedError(
                "ImputationDiD(leave_one_out=True) is not supported with "
                "replicate-weight survey designs. The leave-one-out refinement "
                "(Borusyak, Jaravel & Spiess 2024, Supp. App. A.9) rescales the "
                "conservative influence-function residuals, but replicate-weight "
                "variance is computed by per-replicate refits and does not use "
                "that path — leave_one_out would silently have no effect. Use a "
                "non-replicate (Taylor-linearization) survey design, or "
                "leave_one_out=False."
            )
        # Validate within-unit constancy for panel survey designs
        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)
            if resolved_survey.weight_type != "pweight":
                raise ValueError(
                    f"ImputationDiD survey support requires weight_type='pweight', "
                    f"got '{resolved_survey.weight_type}'. The survey variance math "
                    f"assumes probability weights (pweight)."
                )
            # FPC is supported — threaded through compute_survey_if_variance()
            # in _compute_conservative_variance().

        # Bootstrap + survey supported via PSU-level multiplier bootstrap.

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Validate absorbing treatment: first_treat must be constant within each unit
        ft_nunique = df.groupby(unit)[first_treat].nunique()
        non_constant = ft_nunique[ft_nunique > 1]
        if len(non_constant) > 0:
            example_unit = non_constant.index[0]
            example_vals = sorted(df.loc[df[unit] == example_unit, first_treat].unique())
            warnings.warn(
                f"{len(non_constant)} unit(s) have non-constant '{first_treat}' "
                f"values (e.g., unit '{example_unit}' has values {example_vals}). "
                f"ImputationDiD assumes treatment is an absorbing state "
                f"(once treated, always treated) with a single treatment onset "
                f"time per unit. Non-constant first_treat violates this assumption "
                f"and may produce unreliable estimates.",
                UserWarning,
                stacklevel=2,
            )

            # Coerce to per-unit value so downstream code
            # (_never_treated, _treated, _rel_time) uses a single
            # consistent first_treat per unit.
            df[first_treat] = df.groupby(unit)[first_treat].transform("first")

        # Identify treatment status
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)

        # Check for always-treated units (treated in all observed periods)
        min_time = df[time].min()
        always_treated_mask = (~df["_never_treated"]) & (df[first_treat] <= min_time)
        n_always_treated = df.loc[always_treated_mask, unit].nunique()
        if n_always_treated > 0:
            warnings.warn(
                f"{n_always_treated} unit(s) are treated in all observed periods "
                f"(first_treat <= {min_time}). These units have no untreated "
                "observations and cannot contribute to the counterfactual model. "
                "Their treatment effects will be imputed but may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Create treatment indicator D_it
        # D_it = 1 if t >= first_treat and first_treat > 0
        # With anticipation: D_it = 1 if t >= first_treat - anticipation
        effective_treat = df[first_treat] - self.anticipation
        df["_treated"] = (~df["_never_treated"]) & (df[time] >= effective_treat)

        # Identify Omega_0 (untreated) and Omega_1 (treated)
        omega_0_mask = ~df["_treated"]
        omega_1_mask = df["_treated"]

        # Per-fit cache of the target-invariant untreated-projection design +
        # factorization, shared across every estimand target (overall ATT, each
        # event-study horizon, each group) AND the bootstrap precompute. A
        # fit-time local (not self.* state) so fit() stays idempotent; see
        # _compute_cluster_psi_sums for the key derivation.
        proj_cache: Dict[Any, _UntreatedProjection] = {}

        n_omega_0 = int(omega_0_mask.sum())
        n_omega_1 = int(omega_1_mask.sum())

        if n_omega_0 == 0:
            raise ValueError(
                "No untreated observations found. Cannot estimate counterfactual model."
            )
        if n_omega_1 == 0:
            raise ValueError("No treated observations found. Nothing to estimate.")

        # Identify groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0 and g != np.inf])

        if len(treatment_groups) == 0:
            raise ValueError("No treated units found. Check 'first_treat' column.")

        # Unit info
        unit_info = (
            df.groupby(unit).agg({first_treat: "first", "_never_treated": "first"}).reset_index()
        )
        n_treated_units = int((~unit_info["_never_treated"]).sum())
        # Control units = units with at least one untreated observation
        units_in_omega_0 = df.loc[omega_0_mask, unit].unique()
        n_control_units = len(units_in_omega_0)

        # Cluster variable
        cluster_var = self.cluster if self.cluster is not None else unit
        if self.cluster is not None and self.cluster not in df.columns:
            raise ValueError(
                f"Cluster column '{self.cluster}' not found in data. "
                f"Available columns: {list(df.columns)}"
            )

        # Resolve effective cluster and inject cluster-as-PSU for survey variance
        if resolved_survey is not None:
            cluster_ids_raw = df[cluster_var].values if cluster_var in df.columns else None
            effective_cluster_ids = _resolve_effective_cluster(
                resolved_survey,
                cluster_ids_raw,
                cluster_var if self.cluster is not None else None,
            )
            resolved_survey = _inject_cluster_as_psu(resolved_survey, effective_cluster_ids)
            # When survey PSU is present, use it as the effective cluster for
            # Theorem 3 variance (PSU overrides unit-level clustering)
            if resolved_survey.psu is not None:
                # Create a temporary column with PSU IDs for cluster_var
                df["_survey_cluster"] = resolved_survey.psu
                cluster_var = "_survey_cluster"
            # Recompute metadata after PSU injection
            if resolved_survey.psu is not None and survey_metadata is not None:
                from diff_diff.survey import compute_survey_metadata

                # resolved_survey non-None implies survey_design was passed.
                assert survey_design is not None
                raw_w = (
                    data[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(data), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

        # Compute relative time
        df["_rel_time"] = np.where(
            ~df["_never_treated"],
            df[time] - df[first_treat],
            np.nan,
        )

        # ---- Step 1: OLS on untreated observations ----
        unit_fe, time_fe, grand_mean, delta_hat, kept_cov_mask = self._fit_untreated_model(
            df, outcome, unit, time, covariates, omega_0_mask, weights=survey_weights
        )

        # ---- Rank condition checks ----
        # Check: every treated unit should have >= 1 untreated period (for unit FE)
        treated_unit_ids = df.loc[omega_1_mask, unit].unique()
        units_with_fe = set(unit_fe.keys())
        units_missing_fe = set(treated_unit_ids) - units_with_fe

        # Check: every post-treatment period should have >= 1 untreated unit (for time FE)
        post_period_ids = df.loc[omega_1_mask, time].unique()
        periods_with_fe = set(time_fe.keys())
        periods_missing_fe = set(post_period_ids) - periods_with_fe

        if units_missing_fe or periods_missing_fe:
            parts = []
            if units_missing_fe:
                sorted_missing = sorted(units_missing_fe)
                parts.append(
                    f"{len(units_missing_fe)} treated unit(s) have no untreated "
                    f"periods (units: {sorted_missing[:5]}"
                    f"{'...' if len(units_missing_fe) > 5 else ''})"
                )
            if periods_missing_fe:
                sorted_missing = sorted(periods_missing_fe)
                parts.append(
                    f"{len(periods_missing_fe)} post-treatment period(s) have no "
                    f"untreated units (periods: {sorted_missing[:5]}"
                    f"{'...' if len(periods_missing_fe) > 5 else ''})"
                )
            msg = (
                "Rank condition violated: "
                + "; ".join(parts)
                + ". Affected treatment effects will be NaN."
            )
            if self.rank_deficient_action == "error":
                raise ValueError(msg)
            elif self.rank_deficient_action == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)
            # "silent": continue without warning

        # ---- Step 2: Impute treatment effects ----
        tau_hat, y_hat_0 = self._impute_treatment_effects(
            df,
            outcome,
            unit,
            time,
            covariates,
            omega_1_mask,
            unit_fe,
            time_fe,
            grand_mean,
            delta_hat,
        )

        # Store tau_hat in dataframe
        df["_tau_hat"] = np.nan
        df.loc[omega_1_mask, "_tau_hat"] = tau_hat

        # ---- Step 3: Aggregate ----
        # Always compute overall ATT (simple aggregation)
        finite_mask = np.isfinite(tau_hat)
        valid_tau = tau_hat[finite_mask]

        if len(valid_tau) == 0:
            overall_att = np.nan
        elif survey_weights is not None:
            # Survey-weighted ATT: use treated obs' survey weights
            treated_survey_w = survey_weights[omega_1_mask.values]
            w_finite = treated_survey_w[finite_mask]
            overall_att = float(np.average(valid_tau, weights=w_finite))
        else:
            overall_att = float(np.mean(valid_tau))

        # ---- Variance ----
        _n_valid_rep_imp = None
        _vcov_rep_imp = None
        overall_se = np.nan  # placeholder; overridden by replicate or conservative path

        if not _uses_replicate_imp:
            # Conservative variance (Theorem 3)
            overall_weights = np.zeros(n_omega_1)
            n_valid = int(finite_mask.sum())
            if n_valid > 0:
                if survey_weights is not None:
                    treated_sw = survey_weights[omega_1_mask.values]
                    sw_finite = treated_sw[finite_mask]
                    overall_weights[finite_mask] = sw_finite / sw_finite.sum()
                else:
                    overall_weights[finite_mask] = 1.0 / n_valid

            if n_valid == 0:
                overall_se = np.nan
            else:
                overall_se = self._compute_conservative_variance(
                    df=df,
                    outcome=outcome,
                    unit=unit,
                    time=time,
                    first_treat=first_treat,
                    covariates=covariates,
                    omega_0_mask=omega_0_mask,
                    omega_1_mask=omega_1_mask,
                    unit_fe=unit_fe,
                    time_fe=time_fe,
                    grand_mean=grand_mean,
                    delta_hat=delta_hat,
                    weights=overall_weights,
                    cluster_var=cluster_var,
                    kept_cov_mask=kept_cov_mask,
                    survey_weights=survey_weights,
                    resolved_survey=(resolved_survey if not _uses_replicate_imp else None),
                    proj_cache=proj_cache,
                )

        # Survey degrees of freedom for t-distribution inference
        _survey_df = resolved_survey.df_survey if resolved_survey is not None else None
        # Replicate df: rank-deficient → NaN inference; dropped replicates → n_valid-1
        if _uses_replicate_imp and _survey_df is None:
            _survey_df = 0  # rank-deficient replicate → NaN inference

        # Kit df-provenance SEED (M-021): the exact value the analytical
        # aggregators below receive, captured BEFORE the replicate override
        # can rebind _survey_df — post-fit recompute must re-seed from it.
        _survey_df_seed = _survey_df

        # Compute overall inference (may be overridden by replicate below)
        overall_t, overall_p, overall_ci = safe_inference(
            overall_att, overall_se, alpha=self.alpha, df=_survey_df
        )

        # Event study and group aggregation (full-sample, for point estimates)
        event_study_effects = None
        group_effects = None

        if aggregate in ("event_study", "all"):
            event_study_effects = self._aggregate_event_study(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                grand_mean=grand_mean,
                delta_hat=delta_hat,
                cluster_var=cluster_var,
                treatment_groups=treatment_groups,
                balance_e=balance_e,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
                survey_df=_survey_df,
                resolved_survey=(resolved_survey if not _uses_replicate_imp else None),
                proj_cache=proj_cache,
            )

        if aggregate in ("group", "all"):
            group_effects = self._aggregate_group(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                grand_mean=grand_mean,
                delta_hat=delta_hat,
                cluster_var=cluster_var,
                treatment_groups=treatment_groups,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
                survey_df=_survey_df,
                resolved_survey=(resolved_survey if not _uses_replicate_imp else None),
                proj_cache=proj_cache,
            )

        # Replicate variance: derive keys from actual outputs (after filtering)
        if _uses_replicate_imp:
            (
                _vcov_rep_imp,
                _n_valid_rep_imp,
                _survey_df,
            ) = self._replicate_override_aggregates(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                resolved_survey=resolved_survey,
                overall_att=overall_att,
                event_study_effects=event_study_effects,
                group_effects=group_effects,
                balance_e=balance_e,
                survey_df_seed=_survey_df,
            )
            overall_se = float(np.sqrt(max(_vcov_rep_imp[0, 0], 0.0)))
            if survey_metadata is not None:
                survey_metadata.df_survey = _survey_df if _survey_df and _survey_df > 0 else None
            overall_t, overall_p, overall_ci = safe_inference(
                overall_att, overall_se, alpha=self.alpha, df=_survey_df
            )

        # Build treatment effects dataframe
        treated_df = df.loc[omega_1_mask, [unit, time, "_tau_hat", "_rel_time"]].copy()
        treated_df = treated_df.rename(columns={"_tau_hat": "tau_hat", "_rel_time": "rel_time"})
        # Weights consistent with actual ATT: zero for NaN tau_hat
        tau_finite = treated_df["tau_hat"].notna()
        n_valid_te = int(tau_finite.sum())
        if n_valid_te > 0:
            if survey_weights is not None:
                # Survey-weighted: use normalized survey weights for treated obs
                treated_sw = survey_weights[omega_1_mask.values]
                sw_finite = np.where(tau_finite, treated_sw, 0.0)
                sw_sum = sw_finite.sum()
                treated_df["weight"] = sw_finite / sw_sum if sw_sum > 0 else 0.0
            else:
                treated_df["weight"] = np.where(tau_finite, 1.0 / n_valid_te, 0.0)
        else:
            treated_df["weight"] = 0.0

        # Store fit data for pretrend_test
        self._fit_data = {
            "df": df,
            "outcome": outcome,
            "unit": unit,
            "time": time,
            "first_treat": first_treat,
            "covariates": covariates,
            "omega_0_mask": omega_0_mask,
            "omega_1_mask": omega_1_mask,
            "cluster_var": cluster_var,
            "unit_fe": unit_fe,
            "time_fe": time_fe,
            "grand_mean": grand_mean,
            "delta_hat": delta_hat,
            "kept_cov_mask": kept_cov_mask,
            "survey_design": survey_design,
            "resolved_survey": resolved_survey,
            "survey_weights": survey_weights,
        }

        # Pre-compute cluster psi sums for bootstrap
        psi_data = None
        if self.n_bootstrap > 0 and n_valid > 0:
            try:
                # Extract survey weights for untreated obs (same as analytical path)
                _sw_0 = survey_weights[omega_0_mask.values] if survey_weights is not None else None
                # Extract survey weights for treated obs (event-study/group bootstrap paths)
                _sw_1 = survey_weights[omega_1_mask.values] if survey_weights is not None else None
                psi_data = self._precompute_bootstrap_psi(
                    df=df,
                    outcome=outcome,
                    unit=unit,
                    time=time,
                    first_treat=first_treat,
                    covariates=covariates,
                    omega_0_mask=omega_0_mask,
                    omega_1_mask=omega_1_mask,
                    unit_fe=unit_fe,
                    time_fe=time_fe,
                    grand_mean=grand_mean,
                    delta_hat=delta_hat,
                    cluster_var=cluster_var,
                    kept_cov_mask=kept_cov_mask,
                    overall_weights=overall_weights,
                    event_study_effects=event_study_effects,
                    group_effects=group_effects,
                    treatment_groups=treatment_groups,
                    tau_hat=tau_hat,
                    balance_e=balance_e,
                    survey_weights_0=_sw_0,
                    survey_weights_1=_sw_1,
                    proj_cache=proj_cache,
                )
            except Exception as e:
                warnings.warn(
                    f"Bootstrap pre-computation failed: {e}. " "Skipping bootstrap inference.",
                    UserWarning,
                    stacklevel=2,
                )
                psi_data = None

        # Bootstrap
        bootstrap_results = None
        if self.n_bootstrap > 0 and psi_data is not None:
            bootstrap_results = self._run_bootstrap(
                original_att=overall_att,
                original_event_study=event_study_effects,
                original_group=group_effects,
                psi_data=psi_data,
                resolved_survey=resolved_survey,
            )

            # Update inference with bootstrap results
            overall_se = bootstrap_results.overall_att_se
            overall_t = (
                overall_att / overall_se if np.isfinite(overall_se) and overall_se > 0 else np.nan
            )
            overall_p = bootstrap_results.overall_att_p_value
            overall_ci = bootstrap_results.overall_att_ci

            # Update event study
            if event_study_effects and bootstrap_results.event_study_ses:
                for h in event_study_effects:
                    if (
                        h in bootstrap_results.event_study_ses
                        and event_study_effects[h].get("n_obs", 1) > 0
                    ):
                        event_study_effects[h]["se"] = bootstrap_results.event_study_ses[h]
                        assert bootstrap_results.event_study_cis is not None
                        event_study_effects[h]["conf_int"] = bootstrap_results.event_study_cis[h]
                        assert bootstrap_results.event_study_p_values is not None
                        event_study_effects[h]["p_value"] = bootstrap_results.event_study_p_values[
                            h
                        ]
                        eff_val = event_study_effects[h]["effect"]
                        se_val = event_study_effects[h]["se"]
                        event_study_effects[h]["t_stat"] = safe_inference(
                            eff_val, se_val, alpha=self.alpha
                        )[0]

            # Update group effects
            if group_effects and bootstrap_results.group_ses:
                for g in group_effects:
                    if g in bootstrap_results.group_ses:
                        group_effects[g]["se"] = bootstrap_results.group_ses[g]
                        assert bootstrap_results.group_cis is not None
                        group_effects[g]["conf_int"] = bootstrap_results.group_cis[g]
                        assert bootstrap_results.group_p_values is not None
                        group_effects[g]["p_value"] = bootstrap_results.group_p_values[g]
                        eff_val = group_effects[g]["effect"]
                        se_val = group_effects[g]["se"]
                        group_effects[g]["t_stat"] = safe_inference(
                            eff_val, se_val, alpha=self.alpha
                        )[0]
                        # Percentile inference replaced the analytical row —
                        # never publish an analytical df beside it (M-021;
                        # the EfficientDiD M-023 precedent).
                        group_effects[g]["df_used"] = None

        # Resolve cluster_name / n_clusters for Results metadata.
        # Suppress under ANY survey design (the survey block in summary()
        # already renders the design's PSU/strata/replicate metadata, and
        # replicate-weight variance ignores PSU/cluster entirely — keeping
        # cluster_name/n_clusters populated on a replicate fit would
        # misreport the inference source).
        # Otherwise:
        #   bare cluster= -> populate with the user-named cluster column
        #   cluster=None  -> the Theorem 3 variance still clusters at the
        #                    `unit` column by default (cluster_var = unit
        #                    at L418), so the summary label must report
        #                    unit-cluster CR1, not generic HC1.
        if resolved_survey is not None:
            _cluster_name_for_results: Optional[str] = None
            _n_clusters_for_results: Optional[int] = None
        elif self.cluster is not None:
            _cluster_name_for_results = self.cluster
            _n_clusters_for_results = int(data[self.cluster].nunique())
        else:
            _cluster_name_for_results = unit
            _n_clusters_for_results = int(data[unit].nunique())

        # Construct results
        self.results_ = ImputationDiDResults(
            treatment_effects=treated_df,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=len(df),
            n_treated_obs=n_omega_1,
            n_untreated_obs=n_omega_0,
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            alpha=self.alpha,
            anticipation=self.anticipation,
            bootstrap_results=bootstrap_results,
            _estimator_ref=self,
            survey_metadata=survey_metadata,
            vcov_type=self.vcov_type,
            cluster_name=_cluster_name_for_results,
            n_clusters=_n_clusters_for_results,
            leave_one_out=self.leave_one_out,
            df_convention=self.df_convention,
        )

        # Attach the post-fit aggregation kit (M-021/M-118). Unconditional —
        # including bootstrap fits, whose gate lives in _aggregate_compute
        # (a FAILED bootstrap leaves bootstrap_results=None and the fit
        # aggregates normally).
        self.results_._aggregation_kit = _build_imputation_aggregation_kit(
            fit_data=self._fit_data,
            treatment_groups=treatment_groups,
            overall_att=overall_att,
            n_treated_obs=n_omega_1,
            uses_replicate=_uses_replicate_imp,
            survey_df_seed=_survey_df_seed,
            survey_df_final=_survey_df,
            survey_metadata=survey_metadata,
            horizon_max=self.horizon_max,
            pretrends=self.pretrends,
            aux_partition=self.aux_partition,
            leave_one_out=self.leave_one_out,
            rank_deficient_action=self.rank_deficient_action,
            df_convention=self.df_convention,
            alpha=self.alpha,
            anticipation=self.anticipation,
        )

        self.is_fitted_ = True
        return self.results_

    # =========================================================================
    # Step 1: OLS on untreated observations
    # =========================================================================

    # =========================================================================
    # Step 2: Impute counterfactuals
    # =========================================================================

    # =========================================================================
    # Conservative Variance (Theorem 3)
    # =========================================================================

    # =========================================================================
    # Aggregation
    # =========================================================================

    # =========================================================================
    # Pre-trend test (Equation 9) & pre-period lead coefficients
    # =========================================================================

    def _pretrend_test(self, n_leads: Optional[int] = None) -> Dict[str, Any]:
        """
        Run pre-trend test (Equation 9).

        Adds pre-treatment lead indicators to the Step 1 OLS on Omega_0
        and tests their joint significance via Wald F-test (cluster-robust
        or design-based survey VCV when survey_design is present).
        """
        if self._fit_data is None:
            raise RuntimeError("Must call fit() before pretrend_test().")

        fd = self._fit_data
        resolved_survey = fd.get("resolved_survey")
        if resolved_survey is not None and resolved_survey.uses_replicate_variance:
            raise NotImplementedError(
                "pretrend_test() is not yet supported for replicate-weight "
                "survey designs. Per-replicate Equation 9 lead regression "
                "refits are not implemented. Use analytical survey designs "
                "(strata/PSU/FPC) or call pretrend_test() without survey."
            )

        df = fd["df"]
        outcome = fd["outcome"]
        unit = fd["unit"]
        time = fd["time"]
        first_treat = fd["first_treat"]
        covariates = fd["covariates"]
        omega_0_mask = fd["omega_0_mask"]
        cluster_var = fd["cluster_var"]
        resolved_survey = fd.get("resolved_survey")
        survey_weights = fd.get("survey_weights")

        df_0 = df.loc[omega_0_mask].copy()

        # Compute relative time for untreated obs
        rel_time_0 = np.where(
            ~df_0["_never_treated"],
            df_0[time] - df_0[first_treat],
            np.nan,
        )

        # Get available pre-treatment relative times (negative values)
        pre_rel_times = sorted(
            set(int(h) for h in rel_time_0 if np.isfinite(h) and h < -self.anticipation)
        )

        if len(pre_rel_times) == 0:
            return {
                "f_stat": np.nan,
                "p_value": np.nan,
                "df": 0,
                "n_leads": 0,
                "lead_coefficients": {},
            }

        # Exclude the reference period (last pre-treatment period)
        ref = -1 - self.anticipation
        pre_rel_times = [h for h in pre_rel_times if h != ref]

        if n_leads is not None:
            pre_rel_times = sorted(pre_rel_times, reverse=True)[:n_leads]
            pre_rel_times = sorted(pre_rel_times)

        if len(pre_rel_times) == 0:
            return {
                "f_stat": np.nan,
                "p_value": np.nan,
                "df": 0,
                "n_leads": 0,
                "lead_coefficients": {},
            }

        # Survey pretrends: pass full design (subpopulation approach)
        _sw_0_pt = None
        _rs_full_pt = None
        _n_full_pt = None
        _o0_idx_pt = None
        if survey_weights is not None and resolved_survey is not None:
            _sw_0_pt = survey_weights[omega_0_mask.values]
            _rs_full_pt = resolved_survey
            _n_full_pt = len(fd["df"])
            _o0_idx_pt = np.where(omega_0_mask.values)[0]

        # Use shared lead coefficient computation
        effects, gamma, V_gamma = self._compute_lead_coefficients(
            df_0,
            outcome,
            unit,
            time,
            first_treat,
            covariates,
            cluster_var,
            pre_rel_times,
            alpha=self.alpha,
            survey_weights_0=_sw_0_pt,
            resolved_survey_full=_rs_full_pt,
            n_obs_full=_n_full_pt,
            omega_0_indices=_o0_idx_pt,
            survey_df=(resolved_survey.df_survey if resolved_survey is not None else None),
        )

        n_leads_actual = len(pre_rel_times)

        # Wald F-test: F = (gamma' V^{-1} gamma) / n_leads
        try:
            V_inv_gamma = np.linalg.solve(V_gamma, gamma)
            wald_stat = float(gamma @ V_inv_gamma)
            f_stat = wald_stat / n_leads_actual
        except np.linalg.LinAlgError:
            f_stat = np.nan

        # P-value from F distribution (survey df when available)
        if np.isfinite(f_stat) and f_stat >= 0:
            if resolved_survey is not None and resolved_survey.df_survey is not None:
                df_denom = resolved_survey.df_survey
            else:
                cluster_ids = df_0[cluster_var].values
                n_clusters = len(np.unique(cluster_ids))
                df_denom = max(n_clusters - 1, 1)
            if df_denom <= 0:
                p_value = np.nan
            else:
                p_value = float(stats.f.sf(f_stat, n_leads_actual, df_denom))
        else:
            p_value = np.nan

        lead_coefficients = {h: effects[h]["effect"] for h in pre_rel_times}

        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "df": n_leads_actual,
            "n_leads": n_leads_actual,
            "lead_coefficients": lead_coefficients,
        }

    # =========================================================================
    # sklearn-compatible interface
    # =========================================================================

    # get_params/set_params come from BaseEstimator.

    @staticmethod
    def _validate_leave_one_out(leave_one_out: Any) -> None:
        """Validate ``leave_one_out`` is a strict bool.

        Called from ``__init__`` AND ``fit()`` so sklearn-style
        ``set_params(leave_one_out=...)`` mutations are re-checked at use
        time -- the naive ``set_params`` setter would otherwise accept a
        truthy string (e.g. "yes") and silently run the LOO refinement.
        """
        if not isinstance(leave_one_out, bool):
            raise TypeError(f"leave_one_out must be a bool, got {type(leave_one_out).__name__}")

    @staticmethod
    def _validate_vcov_type(vcov_type: str) -> None:
        """Validate ``vcov_type`` membership against ImputationDiD's
        permanently-narrow influence-function variance contract.

        Called from ``__init__`` AND ``fit()``; ``set_params`` validates
        eagerly via the BaseEstimator probe re-init, so the fit-time
        re-check only catches direct attribute mutation.
        Mirrors the TripleDifference / CallawaySantAnna pattern (no
        single design matrix on which hat-matrix leverage or Bell-
        McCaffrey Satterthwaite DOF can be defined).
        """
        _accepted_vcov = {"hc1"}
        _if_incompatible_vcov = {"classical", "hc2", "hc2_bm"}
        _deferred_vcov = {"conley"}

        if vcov_type in _if_incompatible_vcov:
            raise ValueError(
                f"ImputationDiD(vcov_type={vcov_type!r}) is rejected: "
                "ImputationDiD uses influence-function-based variance per "
                "Borusyak, Jaravel, and Spiess (2024) Theorem 3. The "
                "per-unit influence function aggregation has no equivalent "
                "single design matrix on which hat matrix leverage or "
                "Bell-McCaffrey Satterthwaite DOF can be defined, so "
                "analytical-sandwich families {classical, hc2, hc2_bm} are "
                "not paper-prescribed. Use vcov_type='hc1' (the default) "
                "with cluster=<col> for per-cluster influence-function "
                "summation (Theorem 3 equation 7 conservative variance)."
            )
        if vcov_type in _deferred_vcov:
            raise ValueError(
                f"ImputationDiD(vcov_type={vcov_type!r}) is not yet "
                "supported: spatial-HAC composition with Theorem 3 "
                "per-unit IF aggregation has no reference implementation "
                "today. See DEFERRED.md for the deferred follow-up row. Use "
                "vcov_type='hc1' (the default) with cluster=<col> for "
                "cluster-robust inference."
            )
        if vcov_type not in _accepted_vcov:
            raise ValueError(
                f"ImputationDiD(vcov_type={vcov_type!r}) is invalid. "
                f"Accepted: {sorted(_accepted_vcov)}."
            )

    def summary(self) -> str:
        """Get summary of estimation results."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())


# =============================================================================
# Post-fit aggregation kit (M-021/M-118)
# =============================================================================


def _build_imputation_aggregation_kit(
    *,
    fit_data: Dict[str, Any],
    treatment_groups: List[Any],
    overall_att: float,
    n_treated_obs: int,
    uses_replicate: bool,
    survey_df_seed: Optional[int],
    survey_df_final: Optional[int],
    survey_metadata: Optional[Any],
    horizon_max: Optional[int],
    pretrends: bool,
    aux_partition: str,
    leave_one_out: bool,
    rank_deficient_action: str,
    df_convention: str,
    alpha: float,
    anticipation: int,
) -> AggregationKit:
    """Build the PANEL-BACKED post-fit aggregation kit (rows M-021/M-118).

    ImputationDiD's event-study/group aggregation is a target-specific
    Theorem-3 recompute from the working panel + untreated FE model — no
    compact influence payload can honor a different ``balance_e`` post-fit
    — so ``bookkeeping`` holds REFERENCES to the SAME per-fit objects
    ``self._fit_data`` already retains for ``pretrend_test()`` (passed in
    as ``fit_data``; zero marginal memory, and pickles are unchanged via
    memoization — ``_estimator_ref`` already ships these objects). fit()
    rebinds a fresh ``_fit_data`` dict + a fresh working frame per call,
    so kits from different fits never alias.

    Value SNAPSHOTS (not refs) isolate recompute from public-field
    mutation: ``treatment_groups`` is copied (fit hands the same list
    object to the results' public cohort list), scalar config is copied
    by value, and
    ``survey_metadata`` is a ``dataclasses.replace`` copy (the ES carrier
    builds from the kit copy, never the mutable public field). The three
    df-provenance channels: ``survey_df_seed`` (what the analytical
    aggregators received — recompute re-seeds from it),
    ``survey_df_final`` (what the stored overall inference received —
    the 'simple' relay's df), and the metadata copy's own ``df_survey``
    (the fit-final container channel).

    ``influence`` is EMPTY BY DESIGN: the recompute is panel-backed, not
    IF-payload-backed, and the fit-local projection cache holds
    unpicklable factorizations — each ``aggregate()`` call rebuilds a
    call-local ``proj_cache``.
    """
    bookkeeping: Dict[str, Any] = {
        # Panel-backed refs (the _fit_data objects)
        "df": fit_data["df"],
        "outcome": fit_data["outcome"],
        "unit": fit_data["unit"],
        "time": fit_data["time"],
        "first_treat": fit_data["first_treat"],
        "covariates": (list(fit_data["covariates"]) if fit_data["covariates"] else None),
        "omega_0_mask": fit_data["omega_0_mask"],
        "omega_1_mask": fit_data["omega_1_mask"],
        "cluster_var": fit_data["cluster_var"],
        "unit_fe": fit_data["unit_fe"],
        "time_fe": fit_data["time_fe"],
        "grand_mean": fit_data["grand_mean"],
        "delta_hat": fit_data["delta_hat"],
        "kept_cov_mask": fit_data["kept_cov_mask"],
        "resolved_survey": fit_data["resolved_survey"],
        "survey_weights": fit_data["survey_weights"],
        # Value snapshots (isolation from public-field / estimator mutation)
        "treatment_groups": list(treatment_groups),
        "overall_att": float(overall_att),
        "n_treated_obs": int(n_treated_obs),
        "uses_replicate": bool(uses_replicate),
        "horizon_max": horizon_max,
        "pretrends": pretrends,
        "aux_partition": aux_partition,
        "leave_one_out": leave_one_out,
        "rank_deficient_action": rank_deficient_action,
        "df_convention": df_convention,
        "survey_df_seed": survey_df_seed,
        "survey_df_final": survey_df_final,
        "survey_metadata": (
            dataclasses.replace(survey_metadata) if survey_metadata is not None else None
        ),
    }
    return AggregationKit(
        bookkeeping=bookkeeping,
        influence={},
        alpha=alpha,
        anticipation=anticipation,
        cband=False,  # no simultaneous-band concept on this estimator
        bootstrap=None,  # replay not wired; recompute levels fail closed ('simple' relays, M-027)
    )


# =============================================================================
# Convenience function
# =============================================================================


def imputation_did(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: str,
    covariates: Optional[List[str]] = None,
    aggregate: Any = NOT_SUPPLIED,
    balance_e: Any = NOT_SUPPLIED,
    survey_design: Optional["SurveyDesign"] = None,
    vcov_type: str = "hc1",
    **kwargs,
) -> ImputationDiDResults:
    """
    Convenience function for imputation DiD estimation.

    .. deprecated:: 3.9
        ``imputation_did()`` is deprecated and will be removed in 4.0
        (row M-070). Construct the estimator instead:
        ``ImputationDiD(...).fit(data, ...)``.

    This is a shortcut for creating an ImputationDiD estimator and calling fit().

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
        Column indicating first treatment period (0 for never-treated).
    covariates : list of str, optional
        Covariate column names.
    aggregate : str, optional
        DEPRECATED (3.9, removed in 4.0; row M-021): forwarded to ``fit()``,
        which warns — aggregate post-fit via
        ``results.aggregate('event_study')`` instead. A plain wrapper call
        (kwarg not supplied) never fires the aggregate warning; since 3.9
        every wrapper call fires the M-070 wrapper-deprecation warning.
    balance_e : int, optional
        DEPRECATED (3.9, removed in 4.0; row M-118): forwarded to ``fit()``,
        which warns — moves onto ``results.aggregate('event_study',
        balance_e=...)``.
    survey_design : SurveyDesign, optional
        Survey design specification for design-based inference. Supports
        pweight only (aweight/fweight raise ValueError). Supports strata,
        PSU, and FPC for design-based variance. Strata enters survey df
        for t-distribution inference.
        Both analytical (n_bootstrap=0) and bootstrap inference are supported.
    vcov_type : str, default="hc1"
        Variance estimator family. ImputationDiD permanently accepts
        ``{"hc1"}`` only — analytical-sandwich families
        ``{classical, hc2, hc2_bm}`` are rejected at ``__init__`` because the
        Theorem 3 per-unit IF aggregation has no single design matrix on
        which hat-matrix leverage or Bell-McCaffrey Satterthwaite DOF can
        be defined. ``cluster=`` invokes per-cluster IF summation;
        ``survey_design=`` invokes TSL on the combined IF.
    **kwargs
        Additional keyword arguments passed to ImputationDiD constructor.

    Returns
    -------
    ImputationDiDResults
        Estimation results.

    Examples
    --------
    >>> from diff_diff import imputation_did, generate_staggered_data
    >>> data = generate_staggered_data(seed=42)
    >>> results = imputation_did(data, 'outcome', 'unit', 'time', 'first_treat')
    >>> results.print_summary()
    >>> results.aggregate('event_study').summary()  # post-fit aggregation
    """
    warnings.warn(
        "imputation_did() is deprecated and will be removed in 4.0; "
        "construct the estimator instead: ImputationDiD(...).fit(data, ...).",
        FutureWarning,
        stacklevel=2,
    )
    est = ImputationDiD(vcov_type=vcov_type, **kwargs)
    return est.fit(
        data,
        outcome=outcome,
        unit=unit,
        time=time,
        first_treat=first_treat,
        covariates=covariates,
        aggregate=aggregate,
        balance_e=balance_e,
        survey_design=survey_design,
    )
