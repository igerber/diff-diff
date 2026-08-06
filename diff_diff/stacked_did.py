"""
Wing, Freedman & Hollingsworth (2024) Stacked Difference-in-Differences Estimator.

Implements the stacked DiD estimator from Wing, Freedman & Hollingsworth (2024),
NBER Working Paper 32054. The key contribution: naive stacked DiD regressions are
biased because they implicitly weight treatment and control group trends differently
across sub-experiments. The authors derive corrective Q-weights that make a weighted
stacked regression identify the "trimmed aggregate ATT" — a well-defined convex
combination of group-time ATTs with stable composition across event time.

The implementation follows the R reference code at
https://github.com/hollina/stacked-did-weights.

References
----------
Wing, C., Freedman, S. M., & Hollingsworth, A. (2024). Stacked
    Difference-in-Differences. NBER Working Paper 32054.
"""

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff._base import BaseEstimator
from diff_diff._deprecation import (
    NOT_SUPPLIED,
    deprecated_field_property,
    resolve_renamed_kwarg,
)
from diff_diff.balancing import BalanceError, entropy_balance
from diff_diff.linalg import effective_cluster_count, solve_ols
from diff_diff.stacked_did_results import StackedDiDResults  # noqa: F401 (re-export)
from diff_diff.utils import resolve_tail_df, safe_inference, validate_df_convention

__all__ = [
    "StackedDiD",
    "StackedDiDResults",
    "stacked_did",
]


class StackedDiD(BaseEstimator):
    """
    Stacked Difference-in-Differences estimator.

    Implements Wing, Freedman & Hollingsworth (2024). Builds a stacked
    dataset of sub-experiments (one per adoption cohort), applies
    corrective Q-weights to address implicit weighting bias in naive
    stacked regressions, and runs a weighted event-study regression.

    Parameters
    ----------
    kappa_pre : int, default=1
        Number of pre-treatment event-time periods in the event window.
        The event window spans [-kappa_pre, ..., kappa_post].
    kappa_post : int, default=1
        Number of post-treatment event-time periods.
    weighting : str, default="aggregate"
        Target estimand weighting scheme per Table 1 of the paper:
        - "aggregate": Equal weight per adoption event (trimmed aggregate ATT)
        - "population": Weight by population size of treated cohort
        - "sample_share": Weight by sample share of each sub-experiment
    control_group : str, default="not_yet_treated"
        How to define clean controls per Appendix A of the paper:
        - "not_yet_treated": Units with A_s > a + kappa_post
        - "strict": Units with A_s > a + kappa_post + kappa_pre
        - "never_treated": Only units with A_s = infinity
        (``clean_control=`` remains accepted as a deprecated alias, row
        M-043; FutureWarning, removed in 4.0.)
    cluster : str, default="unit"
        Clustering level for standard errors:
        - "unit": Cluster on original unit identifier
        - "unit_subexp": Cluster on (unit, sub_experiment) pairs
    alpha : float, default=0.05
        Significance level for confidence intervals.
    anticipation : int, default=0
        Number of anticipation periods. When anticipation > 0:
        - Reference period shifts from e=-1 to e=-1-anticipation
        - Post-treatment includes anticipation periods (e >= -anticipation)
        - Event window expands by anticipation pre-periods
        Consistent with ImputationDiD, TwoStageDiD, SunAbraham.
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient:
        - "warn": Issue warning and drop linearly dependent columns
        - "error": Raise ValueError
        - "silent": Drop columns silently
    vcov_type : {"classical","hc1","hc2","hc2_bm"}, default="hc1"
        Analytical variance family for the stacked WLS regression. StackedDiD
        is intrinsically clustered (``cluster`` is required, no ``cluster=None``
        opt-out), so one-way families that don't compose with cluster_ids are
        rejected at ``__init__``:

        - ``"hc1"`` (default): CR1 Liang-Zeger cluster-robust on the Q-weighted
          design via ``solve_ols(weights=composed_weights, vcov_type="hc1")``.
          Bit-equal to the prior bake-Q-into-X output up to float64 multiplication
          ordering at machine precision (HC1 WLS sandwich is algebraically
          invariant between the two forms). Matches
          ``clubSandwich::vcovCR(lm(weights=Q,...), cluster=~unit, type="CR1S")``
          at atol=1e-10 (target is ``CR1S`` — Stata-style ``G/(G-1) * (n-1)/(n-p)``
          finite-sample correction — NOT plain ``CR1`` which omits the
          ``(n-1)/(n-p)`` factor and would diverge by ~1.4%).
        - ``"hc2_bm"``: CR2 Bell-McCaffrey via
          ``solve_ols(weights=composed_weights, vcov_type="hc2_bm")``, routed
          through the clubSandwich WLS-CR2 port (matches
          ``clubSandwich::vcovCR(lm(weights=Q,...), cluster=~unit, type="CR2")
          + coef_test()$df_Satt`` at atol=1e-10). See ``REGISTRY.md`` Phase 1a
          ``hc2_bm + weights`` row for the algebra (W not √W in hat matrix,
          W² in bias term, unweighted residuals in score).
        - ``"classical"`` and ``"hc2"`` are REJECTED at ``__init__`` with a
          cluster-incompatibility ``ValueError``: StackedDiD requires a cluster
          structure, so one-way families don't compose with the linalg validator.
          Use ``"hc1"`` or ``"hc2_bm"``.
        - ``"conley"`` is REJECTED at ``__init__`` for a **methodology** reason
          (NOT plumbing): the stacked design replicates units across
          sub-experiments, so Conley would see same-unit copies at distance 0;
          no ``conleyreg`` anchor; paper-gated. Tracked in DEFERRED.md.

        Survey-design precedence: when ``survey_design=`` is supplied to
        ``fit()`` with ``vcov_type != "hc1"``, a ``NotImplementedError`` is
        raised — the survey Taylor-series linearization (or replicate-weight
        refit) variance overrides the analytical sandwich. Use the default
        ``vcov_type="hc1"`` for survey designs.
    balance : {"none", "entropy"}, default="none"
        Within-sub-experiment covariate balancing (Covariate-Balanced Weighted
        Stacked DID; Ustyuzhanin 2026). With ``"entropy"`` and a ``fit(...,
        covariates=[...])`` list, each clean-control group is reweighted by
        entropy balancing (Hainmueller 2012) so its covariate means match the
        treated cohort's (measured at the last pre-treatment period), and the
        resulting design weights ``b_sa`` are composed with the Wing corrective
        weights via the effective control mass into the final stacked weights
        ``W_sa``. This is **control-only reweighting**, so it preserves the
        trimmed-aggregate-ATT estimand (it changes only how untreated trends are
        estimated, not the treated-cohort weights); at ``b_sa=1`` it reduces to
        the paper's unit-count weighted stacked DID, equal to
        ``weighting="aggregate"`` on balanced event windows. v1 requires
        ``weighting="aggregate"`` and **balanced event windows** (ragged windows
        raise a ``ValueError``), and does not support ``survey_design=``;
        matching-based balancing and the repeated-treatment extension are out of
        scope. Default ``"none"`` reproduces plain weighted stacked DID.
    df_convention : {"residual", "cluster", "normal"}, default "residual"
        Degrees-of-freedom convention for the analytical t/p/CI on the
        pooled stacked regression's event-study and overall-ATT inference.
        ``"residual"`` (default) uses the pooled residual df
        (``n_eff − k_kept``, positive-weight rows) — the 3.9 fix: the
        non-BM non-survey lane previously used silent normal-theory z;
        ``"cluster"`` uses ``G − 1`` where G counts positive-weight
        clusters (``effective_cluster_count`` — may differ from the raw
        ``results.n_clusters`` when a cluster's composed weight is zero);
        ``"normal"`` deliberately uses normal-theory z at the fallback
        level. hc2_bm Bell-McCaffrey contrast DOF and survey/replicate df
        always take precedence. Note the stacked design's residual df is
        typically large (control rows replicate across sub-experiments), so
        the default's numeric movement vs the old z is small — the change
        is a convention alignment. The default flips to ``"cluster"`` at v4.

    Attributes
    ----------
    results_ : StackedDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> from diff_diff import StackedDiD, generate_staggered_data
    >>> data = generate_staggered_data(n_units=200, seed=42)
    >>> est = StackedDiD(kappa_pre=2, kappa_post=2)
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='period', first_treat='first_treat')
    >>> results.print_summary()

    The event-study surface is always computed (3.9, row M-024) - view
    it post-fit:

    >>> es = results.aggregate('event_study')
    >>> from diff_diff import plot_event_study
    >>> plot_event_study(results)

    Notes
    -----
    The stacked estimator addresses TWFE bias by:
    1. Creating one sub-experiment per adoption cohort with clean controls
    2. Applying Q-weights to reweight the stacked regression
    3. Running a single event-study WLS regression on the weighted stack

    References
    ----------
    Wing, C., Freedman, S. M., & Hollingsworth, A. (2024). Stacked
        Difference-in-Differences. NBER Working Paper 32054.
    """

    _PARAM_ATTR_ALIASES = {
        "control_group": "_control_group_arg",
        "clean_control": "_clean_control_arg",
    }
    _DERIVED_CONFIG_ATTRS = ("control_group",)

    # Deprecated read-only alias for the resolved ``control_group`` config
    # (row M-043; removed in 4.0). External attribute readers get
    # warn+value instead of a silent AttributeError.
    clean_control = deprecated_field_property("StackedDiD", "clean_control", "control_group")

    @classmethod
    def _normalize_set_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        # During the M-043 shim window `control_group`/`clean_control` are a
        # renamed pair resolved at __init__. get_params() returns the RAW
        # sentinel-era args, so a user migrating in place -
        # StackedDiD(clean_control="strict").set_params(control_group=...) -
        # would merge two supplied values and trip the both-supplied gate.
        # Whichever of the pair the user passes wins; the other resets to
        # its not-supplied sentinel in the merge.
        if "control_group" in params and "clean_control" not in params:
            params["clean_control"] = NOT_SUPPLIED
        elif "clean_control" in params and "control_group" not in params:
            params["control_group"] = NOT_SUPPLIED
        return params

    def __init__(
        self,
        kappa_pre: int = 1,
        kappa_post: int = 1,
        weighting: str = "aggregate",
        control_group: Any = NOT_SUPPLIED,
        cluster: str = "unit",
        alpha: float = 0.05,
        anticipation: int = 0,
        rank_deficient_action: str = "warn",
        vcov_type: str = "hc1",
        balance: str = "none",
        df_convention: str = "residual",
        clean_control: Any = NOT_SUPPLIED,
    ):
        if weighting not in ("aggregate", "population", "sample_share"):
            raise ValueError(
                f"weighting must be 'aggregate', 'population', or 'sample_share', "
                f"got '{weighting}'"
            )
        # M-043: clean_control= is the deprecated alias for control_group=.
        # Raw args are stored for get_params (aliases above); the resolved
        # value lives on self.control_group.
        _control_group_arg = control_group
        _clean_control_arg = clean_control
        control_group = resolve_renamed_kwarg(
            type(self).__name__,
            "clean_control",
            clean_control,
            "control_group",
            control_group,
            default="not_yet_treated",
        )
        if control_group not in ("not_yet_treated", "strict", "never_treated"):
            raise ValueError(
                f"control_group must be 'not_yet_treated', 'strict', or "
                f"'never_treated', got '{control_group}'"
            )
        if cluster not in ("unit", "unit_subexp"):
            raise ValueError(f"cluster must be 'unit' or 'unit_subexp', got '{cluster}'")
        if rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )
        # vcov_type validation (Phase 1b 2/8: thread through StackedDiD).
        # Factored into _validate_vcov_type so set_params() can re-validate.
        self._validate_vcov_type(vcov_type)
        self._validate_balance(balance)
        validate_df_convention(df_convention)

        self.kappa_pre = kappa_pre
        self.kappa_post = kappa_post
        self.weighting = weighting
        self._control_group_arg = _control_group_arg
        self._clean_control_arg = _clean_control_arg
        self.control_group = control_group
        self.cluster = cluster
        self.alpha = alpha
        self.anticipation = anticipation
        self.rank_deficient_action = rank_deficient_action
        self.vcov_type = vcov_type
        self.balance = balance
        self.df_convention = df_convention

        self.is_fitted_ = False
        self.results_: Optional[StackedDiDResults] = None

    @staticmethod
    def _validate_vcov_type(vcov_type: str) -> None:
        """Validate vcov_type. Called from __init__ AND set_params so that
        sklearn-style mutation (`est.set_params(vcov_type="bad")`) hits the
        estimator-level guard rather than failing later in the linalg layer
        with a different message."""
        if vcov_type == "conley":
            raise ValueError(
                "vcov_type='conley' is not supported on StackedDiD and is "
                "deferred for a methodology reason (NOT plumbing, unlike the "
                "SunAbraham / WooldridgeDiD-OLS conley threading): the stacked "
                "design replicates each control unit across every sub-experiment "
                "it qualifies for, so one geographic unit occupies many stacked "
                "rows. Conley's pairwise distance matrix would see those same-unit "
                "copies at distance 0 (K(0)=1, perfectly correlated), conflating "
                "the stacking-replication device with real spatial correlation, "
                "and there is no `conleyreg` analogue for stacked DiD to anchor "
                "parity. A correct treatment needs a per-stack spatial identifier "
                "and is paper-gated (see DEFERRED.md). Use vcov_type='hc1' (default, "
                "CR1) or 'hc2_bm' (CR2 Bell-McCaffrey)."
            )
        if vcov_type not in ("classical", "hc1", "hc2", "hc2_bm"):
            raise ValueError(
                f"vcov_type must be one of {{'classical', 'hc1', 'hc2', 'hc2_bm'}}, "
                f"got '{vcov_type}'"
            )
        if vcov_type in ("classical", "hc2"):
            raise ValueError(
                "StackedDiD clusters intrinsically at 'unit' or 'unit_subexp' "
                "(no cluster=None opt-out). One-way vcov_type='classical'/'hc2' "
                "is rejected by the linalg validator when combined with "
                "cluster_ids. Use vcov_type='hc1' (CR1 Liang-Zeger) or "
                "'hc2_bm' (CR2 Bell-McCaffrey)."
            )

    @staticmethod
    def _validate_balance(balance: str) -> None:
        """Validate the covariate-balancing method (CBWSDID, Ustyuzhanin 2026).

        Called from __init__ AND set_params (mirrors _validate_vcov_type) so
        sklearn-style mutation hits the estimator-level guard. v1 supports only
        entropy balancing; matching-based balancing and IPW are out of scope
        (see docs/methodology/REGISTRY.md StackedDiD)."""
        if balance not in ("none", "entropy"):
            raise ValueError(
                f"balance must be 'none' or 'entropy', got '{balance}'. "
                "Matching-based balancing and IPW are out of scope for v1."
            )

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        aggregate: Any = NOT_SUPPLIED,
        population: Optional[str] = None,
        survey_design=None,
        covariates: Optional[List[str]] = None,
    ) -> StackedDiDResults:
        """
        Fit the stacked DiD estimator.

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
            Use 0 or np.inf for never-treated units.
        aggregate : str, optional
            DEPRECATED (3.9, removed in 4.0; ledger row M-024): the
            event-study surface is now always computed at fit, so this
            parameter is behaviorally inert - passing it (any value,
            ``None`` included) emits a ``FutureWarning``. Aggregate as a
            post-fit step instead: ``results.aggregate('event_study')`` /
            ``results.aggregate('simple')``. Value validation is
            unchanged: ``"group"``/``"all"`` raise ``ValueError``
            (the pooled stacked regression cannot produce
            cohort-specific effects - use CallawaySantAnna or
            ImputationDiD), as do unknown values.
        population : str, optional
            Column name for population weights. Required only when
            weighting="population".
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference. When
            provided, uses Taylor Series Linearization for variance
            estimation and applies sampling weights to the regression.
        covariates : list of str, optional
            Covariate column names to balance the clean controls toward the
            treated cohort (requires ``balance="entropy"``; see the constructor
            ``balance`` parameter). Values are read at the last pre-treatment
            period ``t = a-1-anticipation`` per sub-experiment, so balancing uses
            only pre-treatment information (Assumption 4). Raises ``ValueError``
            if ``balance="none"`` (or vice versa), if a name is absent from
            ``data``, or if a cohort cannot be balanced (infeasible).

        Returns
        -------
        StackedDiDResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # ---- fit(aggregate=) shim (row M-024) ----
        # Deprecated in 3.9, removed at 4.0: the event-study surface is now
        # ALWAYS computed at fit (see the extraction block below), so the
        # param is behaviorally inert and aggregation moves to the post-fit
        # results.aggregate(type=...). The sentinel default means the
        # warning fires ONLY when the caller supplies the argument (None
        # included), never on a plain fit(). Value validation below is
        # unchanged, so the deprecated path returns exactly what a plain
        # fit does - or raises exactly as it always did.
        if aggregate is not NOT_SUPPLIED:
            warnings.warn(
                "StackedDiD.fit(aggregate=) is deprecated and will be "
                "removed in 4.0. The event-study surface is now always "
                "computed; aggregate as a post-fit step: "
                "results.aggregate('event_study') / "
                "results.aggregate('simple').",
                FutureWarning,
                stacklevel=2,
            )
        else:
            aggregate = None

        # ---- Validate inputs ----
        if aggregate in ("group", "all"):
            raise ValueError(
                f"aggregate='{aggregate}' is not supported by StackedDiD. "
                "The pooled stacked regression cannot produce cohort-specific "
                "effects. Use CallawaySantAnna or ImputationDiD for "
                "cohort-level estimates."
            )
        if aggregate not in (None, "simple", "event_study"):
            raise ValueError(
                f"aggregate must be None, 'simple', or 'event_study', " f"got '{aggregate}'"
            )

        required_cols = [outcome, unit, time, first_treat]
        if population is not None:
            required_cols.append(population)
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if self.weighting == "population" and population is None:
            raise ValueError("population column must be specified when weighting='population'")

        # ---- Covariate balancing (CBWSDID, Ustyuzhanin 2026) validation + guards ----
        if isinstance(covariates, str):
            raise TypeError(
                "covariates must be a list of column names, not a string (got "
                f"{covariates!r}). Use covariates=[{covariates!r}]."
            )
        balancing = self.balance != "none"
        if balancing and not covariates:
            raise ValueError(
                "balance='entropy' requires a non-empty covariates= list (the "
                "columns to balance the clean controls toward the treated cohort). "
                "Use balance='none' for unrefined weighted stacked DID."
            )
        if covariates and not balancing:
            raise ValueError(
                "covariates= was provided but balance='none'. Set balance='entropy' "
                "to enable covariate balancing, or drop covariates=."
            )
        if balancing:
            assert covariates is not None  # guaranteed by the cross-validation above
            # Deduplicate (repeated columns are redundant moments) while preserving order.
            covariates = list(dict.fromkeys(covariates))
            if self.weighting != "aggregate":
                raise NotImplementedError(
                    f"balance='entropy' is only supported with weighting='aggregate' "
                    f"(got weighting='{self.weighting}'); the CBWSDID corrective weight "
                    "uses the Wing aggregate (treated-share) form. v1 scope."
                )
            if survey_design is not None:
                raise NotImplementedError(
                    "balance='entropy' with survey_design= is not supported in v1 "
                    "(design-weight + survey-weight composition is out of scope). "
                    "Drop survey_design= or set balance='none'."
                )
            missing_cov = [c for c in covariates if c not in data.columns]
            if missing_cov:
                raise ValueError(f"covariates not found in data columns: {missing_cov}")

        # ---- Resolve survey design ----
        from diff_diff.survey import (
            SurveyDesign,
            _resolve_survey_for_fit,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )
        _uses_replicate_sd = resolved_survey is not None and resolved_survey.uses_replicate_variance

        # Reject fweight and aweight — Q-weight composition is ratio-valued
        # and breaks both frequency-weight (integer) and analytic-weight
        # (inverse-variance) semantics after multiplicative composition
        if (
            survey_design is not None
            and hasattr(survey_design, "weight_type")
            and survey_design.weight_type in ("fweight", "aweight")
        ):
            raise ValueError(
                f"StackedDiD does not support weight_type='{survey_design.weight_type}' "
                "because Q-weight composition changes the weight semantics. "
                "Use weight_type='pweight' (default) instead."
            )

        # Survey-design precedence: when survey_design is supplied, the survey
        # Taylor-series linearization (or replicate-weight refit) variance
        # overrides the analytical sandwich. The non-hc1 analytical families
        # are blocked. Reject ordering matters here: the fweight/aweight check
        # above fires FIRST so users hit the Q-weight semantics error before
        # the vcov error (two-step educational path, matches SA precedent).
        # Future refactors must not swap the order without re-validating tests
        # `test_aweight_plus_hc2_bm_rejected_by_stacked_did_level_guard`.
        if resolved_survey is not None and self.vcov_type != "hc1":
            raise NotImplementedError(
                f"StackedDiD(vcov_type='{self.vcov_type}') is not supported with "
                "survey_design=. The survey TSL (or replicate-weight refit) "
                "variance overrides the analytical sandwich family, so the "
                "small-sample CR2 Bell-McCaffrey correction cannot compose "
                "with the survey variance machinery. Use vcov_type='hc1' "
                "(default) for survey designs."
            )

        # Collect survey design column names for propagation through sub-experiments
        survey_cols: List[str] = []
        if survey_design is not None and isinstance(survey_design, SurveyDesign):
            for attr in ("weights", "strata", "psu", "fpc"):
                col_name = getattr(survey_design, attr, None)
                if col_name is not None:
                    survey_cols.append(col_name)
            # Propagate replicate weight columns through stacked dataset
            if survey_design.replicate_weights is not None:
                survey_cols.extend(survey_design.replicate_weights)

        df = data.copy()
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # ---- Data setup ----
        # Handle never-treated encoding: both 0 and inf -> inf
        df[first_treat] = df[first_treat].replace(0, np.inf)

        # Build unit_info: one row per unit
        unit_info = (
            df.groupby(unit)
            .agg({first_treat: "first"})
            .reset_index()
            .rename(columns={first_treat: "_first_treat"})
        )

        T_min = int(df[time].min())
        T_max = int(df[time].max())
        time_periods = sorted(df[time].unique())

        # Extract unique adoption events (finite first_treat values)
        omega_A = sorted([a for a in unit_info["_first_treat"].unique() if np.isfinite(a)])

        if len(omega_A) == 0:
            raise ValueError(
                "No treated units found. Check 'first_treat' column "
                "(use 0 or np.inf for never-treated units)."
            )

        # ---- Trim adoption events (IC1 + IC2) ----
        omega_kappa, trimmed = self._trim_adoption_events(omega_A, T_min, T_max, unit_info)

        # ---- Build stacked dataset ----
        sub_experiments = []
        skipped_events = []
        for a in omega_kappa:
            sub_exp = self._build_sub_experiment(
                df,
                unit_info,
                a,
                unit,
                time,
                first_treat,
                outcome,
                extra_cols=survey_cols,
            )
            if sub_exp is not None and len(sub_exp) > 0:
                sub_experiments.append(sub_exp)
            else:
                skipped_events.append(a)

        if skipped_events:
            warnings.warn(
                f"Sub-experiments for events {skipped_events} were empty " f"after filtering.",
                UserWarning,
                stacklevel=2,
            )

        if len(sub_experiments) == 0:
            raise ValueError(
                "All sub-experiments are empty after filtering. "
                "Check your data or reduce kappa values."
            )

        stacked_df = pd.concat(sub_experiments, ignore_index=True)

        # ---- Compute Q-weights ----
        stacked_df = self._compute_q_weights(stacked_df, unit, population)

        # ---- Covariate balancing: design weights b_sa -> effective-mass W_sa ----
        # When balancing, this OVERWRITES `_Q_weight` with the CBWSDID final weights
        # W_sa (paper §3.1) so the existing WLS path downstream consumes them
        # transparently; the raw design weights are preserved in `_b_sa`.
        balance_diagnostics: Optional[Dict[Any, Dict[str, Any]]] = None
        if balancing:
            assert covariates is not None  # narrowed by the cross-validation above
            stacked_df, balance_diagnostics = self._compute_balancing_weights(
                stacked_df, df, unit, time, first_treat, covariates
            )

        # ---- Count units ----
        treated_units = stacked_df.loc[stacked_df["_D_sa"] == 1, unit].unique()
        control_units = stacked_df.loc[stacked_df["_D_sa"] == 0, unit].unique()
        n_treated_units = len(treated_units)
        n_control_units = len(control_units)

        # ---- Build design matrix and run WLS ----
        # Always run event study regression (Equation 3 in paper)
        # Reference period: e = -1 - anticipation (shifts when anticipation > 0)
        ref_period = -1 - self.anticipation
        event_times = sorted(
            [
                h
                for h in range(-self.kappa_pre - self.anticipation, self.kappa_post + 1)
                if h != ref_period
            ]
        )

        n = len(stacked_df)
        n_event_dummies = len(event_times)

        # Track column indices for VCV extraction
        # [0] intercept, [1] D_sa, [2..K+1] event-time dummies,
        # [K+2..2K+1] D_sa * event-time interactions
        interaction_indices: Dict[int, int] = {}

        et_vals = stacked_df["_event_time"].values
        d_vals = stacked_df["_D_sa"].values

        # Build design matrix
        X = np.zeros((n, 2 + 2 * n_event_dummies))
        X[:, 0] = 1.0  # intercept
        X[:, 1] = stacked_df["_D_sa"].values  # treatment indicator

        for j, h in enumerate(event_times):
            col_lambda = 2 + j  # event-time dummy
            col_delta = 2 + n_event_dummies + j  # interaction
            mask = et_vals == h
            X[mask, col_lambda] = 1.0
            X[mask, col_delta] = d_vals[mask]
            interaction_indices[h] = col_delta

        # WLS via sqrt(w) transformation
        Q_weights = stacked_df["_Q_weight"].values
        n_stacked = len(stacked_df)

        # Compose Q-weights with survey weights if survey design is present
        if resolved_survey is not None and survey_weights is not None:
            # Survey weights were resolved on the original data; the stacked
            # dataset carries the survey weight column through _build_sub_experiment.
            # Re-extract from the stacked data so lengths match.
            survey_weights_stacked = (
                stacked_df[survey_design.weights].values.astype(np.float64)
                if survey_design.weights is not None
                else np.ones(n_stacked, dtype=np.float64)
            )
            composed_weights = Q_weights * survey_weights_stacked
            # Normalize composed weights to sum = n_stacked
            composed_weights = composed_weights * (n_stacked / np.sum(composed_weights))
        else:
            composed_weights = Q_weights

        # ---- Reference-support guard (row M-024 follow-up, CI R1+R2) ----
        # The omitted reference event time e = -1 - anticipation is the
        # regression's baseline category: every reported delta_h (and the
        # synthesized reference row, and the container's
        # reference_event_times / base_period="universal" provenance) is
        # defined relative to its cells. Equation 3 is a Q-WEIGHTED WLS,
        # so the check runs on the COMPOSED weights, not raw row
        # presence: a cell whose rows all carry zero effective weight
        # (aggregate Q-weights zero a sub-experiment's controls at any
        # event time where that sub-experiment has no treated rows;
        # survey pweights can zero a cell outright) is as empty as a
        # missing one. An effectively-empty cell makes the design
        # rank-deficient and QR pivoting silently re-normalizes against
        # an arbitrary surviving column - the surface would then certify
        # a delta_0 = 0 normalization that never happened (and
        # HonestDiD / PreTrendsPower would trust it). Both cells are
        # load-bearing: without effective TREATED ref weight, D_sa is
        # collinear with the delta block; without effective CONTROL ref
        # weight, the baseline time profile is. Fail closed - a
        # fabricated reference must never reach consumers.
        _ref_mask = et_vals == ref_period
        _w_arr = np.asarray(composed_weights, dtype=float)
        _treated_ref_w = float(np.sum(_w_arr[_ref_mask & (d_vals == 1)]))
        _control_ref_w = float(np.sum(_w_arr[_ref_mask & (d_vals == 0)]))
        if not (_treated_ref_w > 0) or not (_control_ref_w > 0):
            _missing_cell = "treated" if not (_treated_ref_w > 0) else "control"
            raise ValueError(
                f"The omitted reference event time e={ref_period} has no "
                f"{_missing_cell} observations with positive weight in the "
                "stacked WLS, so the event-study normalization "
                "(delta_0 = 0 at the reference) would be fabricated: the "
                "regression's effective baseline cell is empty and rank "
                "handling would silently re-normalize against an "
                "arbitrary horizon. Causes: a gapped panel where every "
                "retained cohort's calendar period a - 1 - anticipation "
                "is absent (the IC1 window check only inspects the "
                "panel's min/max periods, not interior gaps); ragged "
                "panels where the only reference-period control rows sit "
                "in sub-experiments without treated reference rows "
                "(aggregate Q-weights zero those controls); or survey "
                "weights that zero out a reference cell. Fill the gap, "
                "adjust anticipation/kappa_pre so the reference falls on "
                "an observed period, or drop the affected cohorts."
            )

        Y = stacked_df[outcome].values

        # Cluster IDs
        if self.cluster == "unit":
            cluster_ids = stacked_df[unit].values
        else:  # unit_subexp
            cluster_ids = (
                stacked_df[unit].astype(str) + "_" + stacked_df["_sub_exp"].astype(str)
            ).values

        # WLS with weights=composed_weights. solve_ols internally bakes
        # sqrt(w) for the coefficient solve and back-transforms to compute
        # vcov on original-scale data via clubSandwich's WLS-CR2 algebra for
        # hc2_bm (PR #475). The hc1 path remains bit-equal to the prior
        # bake-Q-into-X form (WLS-CR1 score is invariant). Note: this path
        # routes through the Python backend regardless of vcov_type per
        # `linalg.py:747-751` (Rust skips weighted vcov); the prior bake-Q
        # path also went through Python in practice on stacked designs.
        coef, _residuals_unused, vcov = solve_ols(
            X,
            Y,
            cluster_ids=cluster_ids,
            weights=composed_weights,
            weight_type="pweight",
            vcov_type=self.vcov_type,
            return_vcov=True,
            rank_deficient_action=self.rank_deficient_action,
        )
        assert vcov is not None

        # Knob-resolved analytical fallback df (3.9 tail-df consolidation):
        # the pooled stacked design is fully visible (no absorption), so the
        # residual df is n_eff − k_kept with n_eff = positive-weight rows
        # (pweight semantics, mirroring LinearRegression.fit's n_eff). The
        # ``df_convention="cluster"`` G counts POSITIVE-WEIGHT clusters
        # (effective_cluster_count) — deliberately not the raw unique count
        # reported as ``results.n_clusters``; the two diverge only when a
        # cluster's total composed weight is zero (REGISTRY StackedDiD
        # note). Survey/replicate df keeps precedence at the consuming
        # branches below. Previously the non-BM non-survey lane silently
        # used normal theory (the tail-df defect family).
        _n_eff_sd = int(np.count_nonzero(np.asarray(composed_weights, dtype=float) > 0))
        _k_kept_sd = int(np.count_nonzero(~np.isnan(coef)))
        _analytic_fallback_df = resolve_tail_df(
            self.df_convention,
            residual_df=float(_n_eff_sd - _k_kept_sd),
            n_clusters=(
                effective_cluster_count(cluster_ids, composed_weights)
                if self.vcov_type == "hc1"
                else None
            ),
        )

        # Bell-McCaffrey Satterthwaite contrast DOF for hc2_bm. Per the
        # registry contract for `vcov_type="hc2_bm"`, the user-facing
        # aggregated inference (event_study_effects[h]['p_value']/['conf_int']
        # and overall_p_value/overall_conf_int) must use CR2 Bell-McCaffrey
        # Satterthwaite DOF for each contrast — not the normal distribution
        # that safe_inference(df=None) would otherwise default to. Mirrors
        # the SunAbraham aggregated-inference pattern from PR #472
        # (sun_abraham.py:997-1097) and the MPD avg_att pattern from PR #465.
        # Computed BEFORE constructing event_study_effects / overall_*
        # inference so the DOFs can be threaded into the safe_inference calls.
        _bm_contrast_dof_per_event: Dict[int, float] = {}
        _bm_contrast_dof_overall: Optional[float] = None
        if self.vcov_type == "hc2_bm" and not _uses_replicate_sd and not np.all(np.isnan(coef)):
            from diff_diff.linalg import _compute_cr2_bm_contrast_dof

            # Mirror the MultiPeriodDiD rank-deficient pattern (PR #465,
            # estimators.py:1860-1913): solve_ols emits NaN for dropped
            # coefficients under R-style rank handling. Subset X, bread,
            # and contrast vectors to the identified-column block BEFORE
            # calling _compute_cr2_bm_contrast_dof; otherwise the singular
            # full-design bread would raise LinAlgError and downgrade
            # identified contrasts to normal-theory inference (R2 codex
            # P1: catch-and-fallback was too aggressive for identified
            # target contrasts).
            _identified = ~np.isnan(coef)
            _kept = np.where(_identified)[0]
            X_kept = X[:, _kept]
            bread_kept = X_kept.T @ (X_kept * composed_weights[:, np.newaxis])
            k_design = X.shape[1]
            # Per-event-time contrast: unit vector at the delta_h column.
            # Only build contrasts whose target column is identified; if a
            # delta_h column itself was dropped, that event-time will get
            # NaN inference (left to safe_inference's df=None path).
            # 3.9 (row M-024): the per-event contrasts are ALWAYS built -
            # the event-study surface is always materialized so the
            # post-fit results.aggregate("event_study") view is total.
            # (The pre-M-024 gate skipped them when aggregate !=
            # "event_study"; that gate's premise died with the post-fit
            # surface.) NB: the dof helper's degeneracy guard is
            # batch-relative for m > 1 (linalg.py), so the overall-ATT
            # contrast now shares a batch with the per-event contrasts;
            # the widened noise-floor carve-out is documented in the
            # REGISTRY M-024 Note and pinned in test_aggregate_contract.
            es_keys: List[int] = []
            es_cols_full: List[np.ndarray] = []
            for h in event_times:
                if h in interaction_indices and _identified[interaction_indices[h]]:
                    c = np.zeros(k_design)
                    c[interaction_indices[h]] = 1.0
                    es_keys.append(h)
                    es_cols_full.append(c)
            # Overall ATT contrast: average of post-period delta_h columns
            # (the same 1/K * ones contrast used for overall_se below). Only
            # construct if ALL post-period delta_h are identified — otherwise
            # the contrast is undefined.
            _post_event_times_preview = [
                h for h in event_times if h >= -self.anticipation and h in interaction_indices
            ]
            _post_all_identified = all(
                _identified[interaction_indices[h]] for h in _post_event_times_preview
            )
            overall_col_full: Optional[np.ndarray] = None
            if len(_post_event_times_preview) > 0 and _post_all_identified:
                K_prev = len(_post_event_times_preview)
                overall_col_full = np.zeros(k_design)
                for h in _post_event_times_preview:
                    overall_col_full[interaction_indices[h]] = 1.0 / K_prev
            if es_cols_full or overall_col_full is not None:
                # Subset all contrasts to the kept columns. Since each contrast
                # is non-zero only at identified columns (by construction
                # above), no information is lost in the subset.
                cols_full = list(es_cols_full)
                if overall_col_full is not None:
                    cols_full.append(overall_col_full)
                contrasts_full = np.column_stack(cols_full)
                contrasts_kept = contrasts_full[_kept, :]
                try:
                    dof_vec = _compute_cr2_bm_contrast_dof(
                        X_kept,
                        cluster_ids,
                        bread_kept,
                        contrasts_kept,
                        weights=composed_weights,
                    )
                    for idx, h in enumerate(es_keys):
                        _bm_contrast_dof_per_event[h] = float(dof_vec[idx])
                    if overall_col_full is not None:
                        _bm_contrast_dof_overall = float(dof_vec[-1])
                except (ValueError, np.linalg.LinAlgError) as exc:
                    # Genuine singularity on the IDENTIFIED design (very rare
                    # — the rank-deficient handling above already subsets to
                    # identified columns). Emit a UserWarning; the downstream
                    # inference path NaN-closes (per the fail-closed contract
                    # added in this PR) so the user receives undefined
                    # inference rather than silent normal-theory fallback.
                    warnings.warn(
                        f"StackedDiD(vcov_type='hc2_bm') aggregated inference "
                        f"could not compute Bell-McCaffrey contrast DOF on the "
                        f"identified-column design ({type(exc).__name__}: "
                        f"{exc}). Aggregated p-values, t-statistics, and "
                        "confidence intervals will be returned as NaN to "
                        "preserve the hc2_bm contract (small-sample inference "
                        "must use BM Satterthwaite DOF, not normal-theory).",
                        UserWarning,
                        stacklevel=2,
                    )

        # ---- Survey VCV override ----
        _n_valid_rep_sd = None
        resolved_stacked = None
        if resolved_survey is not None and _uses_replicate_sd:
            # Replicate variance: re-run WLS per replicate with composed weights
            from diff_diff.survey import compute_replicate_refit_variance, compute_survey_metadata

            resolved_stacked = survey_design.resolve(stacked_df)

            # Refit closure: compose Q-weights with replicate survey weights.
            # Threads vcov_type=self.vcov_type for grep-consistency though the
            # closure uses return_vcov=False (only the coef is consumed by
            # compute_replicate_refit_variance). The vcov_type passed here is
            # always "hc1" at runtime because the survey + non-hc1 reject in
            # fit() fires before this branch can be reached for any other
            # vcov_type.
            def _refit_stacked(w_r):
                composed_r = Q_weights * w_r
                w_sum = np.sum(composed_r)
                if w_sum > 0:
                    composed_r = composed_r * (n_stacked / w_sum)
                coef_r, _, _ = solve_ols(
                    X,
                    Y,
                    cluster_ids=cluster_ids,
                    weights=composed_r,
                    weight_type="pweight",
                    vcov_type=self.vcov_type,
                    rank_deficient_action="silent",
                    return_vcov=False,
                )
                return coef_r

            # Full-sample cohort effect vector
            vcov, _n_valid_rep_sd = compute_replicate_refit_variance(
                _refit_stacked, coef, resolved_stacked
            )

            # Compute survey metadata
            raw_w_stacked = (
                stacked_df[survey_design.weights].values.astype(np.float64)
                if survey_design.weights is not None
                else np.ones(n_stacked, dtype=np.float64)
            )
            survey_metadata = compute_survey_metadata(resolved_stacked, raw_w_stacked)
        elif resolved_survey is not None:
            from diff_diff.survey import (
                _inject_cluster_as_psu,
                _resolve_effective_cluster,
                compute_survey_metadata,
                compute_survey_vcov,
            )

            # Re-resolve survey design on the stacked data so that strata/PSU
            # arrays have the correct length for TSL variance estimation.
            # (Unlike ContinuousDiD/EfficientDiD, StackedDiD does NOT collapse
            # to unit level via ``subset_to_units_by_row_idx``: control units are
            # duplicated across sub-experiments, so the design must be resolved
            # at the stacked granularity, not one row per original unit.)
            resolved_stacked = survey_design.resolve(stacked_df)

            # Create a copy with composed weights (normalized to sum=n_stacked)
            resolved_composed = copy.copy(resolved_stacked)
            resolved_composed.weights = composed_weights

            # Original-scale residuals for TSL variance
            resid_orig = Y - X @ coef

            # Inject cluster as PSU when survey design has no explicit PSU
            resolved_composed = _inject_cluster_as_psu(resolved_composed, cluster_ids)

            # Resolve effective cluster (PSU overrides user-specified cluster)
            _resolve_effective_cluster(resolved_composed, cluster_ids, self.cluster)

            # Compute TSL variance
            vcov = compute_survey_vcov(X, resid_orig, resolved_composed)

            # Recompute survey metadata on the stacked resolved design
            raw_w_stacked = (
                stacked_df[survey_design.weights].values.astype(np.float64)
                if survey_design.weights is not None
                else np.ones(n_stacked, dtype=np.float64)
            )
            survey_metadata = compute_survey_metadata(resolved_composed, raw_w_stacked)

        # ---- Extract event study effects ----
        # 3.9 (row M-024): the surface is ALWAYS materialized - the
        # regression always includes the event-time interactions, and the
        # post-fit results.aggregate("event_study") view must be total.
        # fit(aggregate=) no longer affects what is computed or stored.
        event_study_effects: Dict[int, Dict[str, Any]] = {}
        es_vcov: Optional[np.ndarray] = None
        es_vcov_index: Optional[List[int]] = None
        es_df_used: Dict[int, float] = {}
        # Reference period (e = -1 - anticipation)
        event_study_effects[ref_period] = {
            "effect": 0.0,
            "se": 0.0,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_int": (np.nan, np.nan),
            "n_obs": 0,
        }
        for h in event_times:
            idx = interaction_indices[h]
            effect = float(coef[idx])
            se = float(np.sqrt(max(vcov[idx, idx], 0.0)))
            _survey_df = (
                max(survey_metadata.df_survey, 1)
                if survey_metadata is not None and survey_metadata.df_survey is not None
                else (0 if _uses_replicate_sd else None)
            )
            # Override df when replicate replicates were dropped
            if _n_valid_rep_sd is not None and resolved_stacked is not None:
                if _n_valid_rep_sd < resolved_stacked.n_replicates:
                    _survey_df = _n_valid_rep_sd - 1 if _n_valid_rep_sd > 1 else 0
                    if survey_metadata is not None:
                        survey_metadata.df_survey = _survey_df if _survey_df > 0 else None
            # Use BM contrast DOF for hc2_bm. Fail-closed: when the
            # hc2_bm contract is in effect but BM DOF is unavailable (helper
            # failed OR noise-floor NaN guard fired), emit all-NaN inference
            # rather than fall back to normal-theory CIs/p-values. Mirrors
            # the fix in LinearRegression.get_inference() from PR #475 R7
            # (linalg.py:3689-3706). `safe_inference` itself has guarded
            # non-finite / <= 0 df since PR #620 (utils.py) with the same
            # all-NaN result; this explicit branch predates that guard and
            # is kept for explicitness (it also skips the call entirely).
            _is_hc2bm_path = self.vcov_type == "hc2_bm" and not _uses_replicate_sd
            _bm_df = _bm_contrast_dof_per_event.get(h)
            if _is_hc2bm_path and (_bm_df is None or not np.isfinite(_bm_df)):
                # BM DOF unavailable on hc2_bm path: NaN-out inference.
                t_stat = float("nan")
                p_value = float("nan")
                conf_int = (float("nan"), float("nan"))
                # No safe_inference call happened -> no df provenance.
                es_df_used[h] = float("nan")
            else:
                # BM DOF > survey/replicate df > the knob-resolved
                # analytical fallback (residual t by default; the
                # replicate 0-sentinel is not None, so it keeps
                # precedence over the fallback).
                _df_eff = (
                    _bm_df
                    if _bm_df is not None
                    else (_survey_df if _survey_df is not None else _analytic_fallback_df)
                )
                t_stat, p_value, conf_int = safe_inference(effect, se, alpha=self.alpha, df=_df_eff)
                # Record the df actually handed to safe_inference iff it
                # governed a t-reference (finite, > 0); None (normal
                # theory) and the df<=0 sentinels record NaN.
                es_df_used[h] = (
                    float(_df_eff)
                    if _df_eff is not None and np.isfinite(_df_eff) and _df_eff > 0
                    else float("nan")
                )
            n_obs_h = int(np.sum((et_vals == h) & (d_vals == 1)))
            event_study_effects[h] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_obs": n_obs_h,
            }

        # Persist the event-time sub-block of the pooled-regression VCV
        # (the reported ES SEs are exactly its diagonal in every
        # inference mode - analytical sandwich, replicate refit, and
        # survey TSL all reassign `vcov` before this block). The
        # reference period is synthesized, never a regression column, so
        # the index is the ESTIMATED event times only.
        if event_times:
            _delta_cols = [interaction_indices[h] for h in event_times]
            es_vcov = vcov[np.ix_(_delta_cols, _delta_cols)]
            es_vcov_index = [int(h) for h in event_times]

        # ---- Compute overall ATT ----
        # Average of post-treatment delta_h coefficients with delta-method SE
        # Post-treatment includes anticipation periods (h >= -anticipation)
        post_event_times = [
            h for h in event_times if h >= -self.anticipation and h in interaction_indices
        ]
        post_indices = [interaction_indices[h] for h in post_event_times]
        K = len(post_indices)

        if K > 0:
            overall_att = sum(float(coef[i]) for i in post_indices) / K
            # Delta method: gradient = 1/K for each post-period coefficient
            sub_vcv = vcov[np.ix_(post_indices, post_indices)]
            ones = np.ones(K)
            overall_se = float(np.sqrt(max(ones @ sub_vcv @ ones, 0.0))) / K
        else:
            overall_att = np.nan
            overall_se = np.nan

        _survey_df_overall = (
            max(survey_metadata.df_survey, 1)
            if survey_metadata is not None and survey_metadata.df_survey is not None
            else (0 if _uses_replicate_sd else None)
        )
        if _n_valid_rep_sd is not None and resolved_stacked is not None:
            if _n_valid_rep_sd < resolved_stacked.n_replicates:
                _survey_df_overall = _n_valid_rep_sd - 1 if _n_valid_rep_sd > 1 else 0
                if survey_metadata is not None:
                    survey_metadata.df_survey = (
                        _survey_df_overall if _survey_df_overall > 0 else None
                    )
        # Use BM contrast DOF for overall ATT (hc2_bm). Fail-closed: when
        # the hc2_bm contract is in effect but BM DOF is unavailable, emit
        # all-NaN inference (per the LinearRegression.get_inference pattern
        # from PR #475 R7). Without this, normal-theory fallback would
        # silently produce wrong p-values/CIs on the overall_* surface.
        _is_hc2bm_path_overall = self.vcov_type == "hc2_bm" and not _uses_replicate_sd
        _overall_df_used: Optional[float] = None
        if _is_hc2bm_path_overall and (
            _bm_contrast_dof_overall is None or not np.isfinite(_bm_contrast_dof_overall)
        ):
            overall_t = float("nan")
            overall_p = float("nan")
            overall_ci = (float("nan"), float("nan"))
        else:
            _df_overall_eff = (
                _bm_contrast_dof_overall
                if _bm_contrast_dof_overall is not None
                else (
                    _survey_df_overall if _survey_df_overall is not None else _analytic_fallback_df
                )
            )
            overall_t, overall_p, overall_ci = safe_inference(
                overall_att, overall_se, alpha=self.alpha, df=_df_overall_eff
            )
            # Scalar provenance: the df the overall safe_inference actually
            # received, recorded iff it governed a t-reference (finite, > 0).
            if _df_overall_eff is not None and np.isfinite(_df_overall_eff) and _df_overall_eff > 0:
                _overall_df_used = float(_df_overall_eff)

        # ---- Construct results ----
        self.results_ = StackedDiDResults(
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            event_study_effects=event_study_effects,
            group_effects=None,
            stacked_data=stacked_df,
            groups=list(omega_kappa),
            trimmed_groups=list(trimmed),
            time_periods=time_periods,
            n_obs=len(data),
            n_stacked_obs=n,
            n_sub_experiments=len(sub_experiments),
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            kappa_pre=self.kappa_pre,
            kappa_post=self.kappa_post,
            weighting=self.weighting,
            control_group=self.control_group,
            alpha=self.alpha,
            anticipation=self.anticipation,
            vcov_type=self.vcov_type,
            cluster_name=self.cluster,
            n_clusters=int(np.unique(cluster_ids).size),
            survey_metadata=survey_metadata,
            balance=self.balance,
            covariates=list(covariates) if balancing else None,
            balance_diagnostics=balance_diagnostics,
            event_study_vcov=es_vcov,
            event_study_vcov_index=es_vcov_index,
            event_study_df=es_df_used,
            df_convention=self.df_convention,
            inference_df=_overall_df_used,
        )

        self.is_fitted_ = True
        return self.results_

    # =========================================================================
    # Trimming (IC1 + IC2)
    # =========================================================================

    def _trim_adoption_events(
        self,
        adoption_events: List[Any],
        T_min: int,
        T_max: int,
        unit_info: pd.DataFrame,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Trim adoption events based on IC1 (window) and IC2 (controls).

        IC1: a - kappa_pre >= T_min AND a + kappa_post <= T_max
        (matches R reference: focalAdoptionTime - kappa_pre >= minTime
        AND focalAdoptionTime + kappa_post <= maxTime)
        With anticipation: a - kappa_pre - anticipation >= T_min

        IC2: Clean controls exist for this adoption event.

        Parameters
        ----------
        adoption_events : list
            Unique finite adoption event times.
        T_min, T_max : int
            Min and max time periods in the data.
        unit_info : pd.DataFrame
            One row per unit with _first_treat column.

        Returns
        -------
        omega_kappa : list
            Included adoption events.
        trimmed : list
            Excluded adoption events.
        """
        omega_kappa = []
        trimmed = []

        for a in adoption_events:
            a_int = int(a)

            # IC1: Event window fits in data
            # a - kappa_pre >= T_min  AND  a + kappa_post <= T_max
            # (matches R reference: focalAdoptionTime - kappa_pre >= minTime)
            # With anticipation: shift window start earlier
            lower_ok = (a_int - self.kappa_pre - self.anticipation) >= T_min
            upper_ok = (a_int + self.kappa_post) <= T_max
            ic1 = lower_ok and upper_ok

            # IC2: Clean controls exist
            ic2 = self._check_clean_controls_exist(a_int, unit_info)

            if ic1 and ic2:
                omega_kappa.append(a)
            else:
                trimmed.append(a)

        if trimmed:
            warnings.warn(
                f"Trimmed {len(trimmed)} adoption event(s) that don't satisfy "
                f"inclusion criteria: {trimmed}. "
                f"IC1 requires event window [{-self.kappa_pre}, {self.kappa_post}] "
                f"to fit within data range [{T_min}, {T_max}]. "
                f"IC2 requires clean controls to exist.",
                UserWarning,
                stacklevel=3,
            )

        if len(omega_kappa) == 0:
            raise ValueError(
                f"All {len(adoption_events)} adoption events were trimmed. "
                f"No valid sub-experiments can be constructed. "
                f"Consider reducing kappa_pre (currently {self.kappa_pre}) "
                f"or kappa_post (currently {self.kappa_post}), or check that "
                f"clean control units exist."
            )

        return omega_kappa, trimmed

    def _check_clean_controls_exist(self, a: int, unit_info: pd.DataFrame) -> bool:
        """Check IC2: whether clean control units exist for adoption event a."""
        ft = unit_info["_first_treat"].values
        if self.control_group == "not_yet_treated":
            return bool(np.any(ft > a + self.kappa_post))
        elif self.control_group == "strict":
            return bool(np.any(ft > a + self.kappa_post + self.kappa_pre))
        else:  # never_treated
            return bool(np.any(np.isinf(ft)))

    # =========================================================================
    # Sub-experiment construction
    # =========================================================================

    def _build_sub_experiment(
        self,
        df: pd.DataFrame,
        unit_info: pd.DataFrame,
        a: Any,
        unit: str,
        time: str,
        first_treat: str,
        outcome: str,
        extra_cols: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Build a single sub-experiment for adoption event a.

        Parameters
        ----------
        df : pd.DataFrame
            Full panel data.
        unit_info : pd.DataFrame
            One row per unit with _first_treat.
        a : int/float
            Adoption event time.
        unit, time, first_treat, outcome : str
            Column names.
        extra_cols : list of str, optional
            Additional columns to propagate from the source data into the
            sub-experiment (e.g., survey design columns: weights, strata,
            psu, fpc).

        Returns
        -------
        pd.DataFrame or None
            Sub-experiment data with _sub_exp, _event_time, _D_sa columns.
        """
        a_int = int(a)
        ft = unit_info["_first_treat"].values
        unit_ids = unit_info[unit].values

        # Treated units: A_s = a
        treated_mask = ft == a
        treated_units = set(unit_ids[treated_mask])

        # Clean control units
        if self.control_group == "not_yet_treated":
            control_mask = ft > a_int + self.kappa_post
        elif self.control_group == "strict":
            control_mask = ft > a_int + self.kappa_post + self.kappa_pre
        else:  # never_treated
            control_mask = np.isinf(ft)
        control_units = set(unit_ids[control_mask])

        if len(treated_units) == 0 or len(control_units) == 0:
            return None

        # Time window: [a - kappa_pre - anticipation, a + kappa_post]
        # Reference period a-1 (event time e=-1) is included when kappa_pre >= 1
        # Matches R reference: (focalAdoptionTime - kappa_pre):(focalAdoptionTime + kappa_post)
        t_start = a_int - self.kappa_pre - self.anticipation
        t_end = a_int + self.kappa_post

        all_units = treated_units | control_units

        # Filter data
        mask = df[unit].isin(all_units) & (df[time] >= t_start) & (df[time] <= t_end)
        sub_df = df.loc[mask].copy()

        if len(sub_df) == 0:
            return None

        # Add sub-experiment columns
        sub_df["_sub_exp"] = a
        sub_df["_event_time"] = sub_df[time] - a_int
        sub_df["_D_sa"] = sub_df[unit].isin(treated_units).astype(int)

        return sub_df

    # =========================================================================
    # Q-weight computation
    # =========================================================================

    def _compute_q_weights(
        self,
        stacked_df: pd.DataFrame,
        unit_col: str,
        population_col: Optional[str],
    ) -> pd.DataFrame:
        """
        Compute Q-weights per Table 1 of Wing et al. (2024).

        Treated observations always get Q = 1.
        Control observations get Q based on the weighting scheme.

        For aggregate weighting, Q-weights are computed using observation
        counts per (event_time, sub_exp), matching the R reference
        ``compute_weights()``. For balanced panels this is equivalent to
        unit counts per sub-experiment. For unbalanced panels the weights
        adjust for varying observation density per event time.

        Population and sample_share weighting use unit counts per
        sub-experiment, following the paper's notation (N_a^D, N_a^C).

        Parameters
        ----------
        stacked_df : pd.DataFrame
            Stacked dataset with _sub_exp, _event_time, and _D_sa columns.
        unit_col : str
            Unit column name.
        population_col : str, optional
            Population column name (for weighting="population").

        Returns
        -------
        pd.DataFrame
            stacked_df with _Q_weight column added.
        """
        if self.weighting == "aggregate":
            return self._compute_q_weights_aggregate(stacked_df)

        # --- Population and sample_share: unit-count-based formulas ---

        # Count distinct units per sub-experiment
        sub_exp_stats = (
            stacked_df.groupby(["_sub_exp", "_D_sa"])[unit_col].nunique().unstack(fill_value=0)
        )

        # N_a^D and N_a^C per sub-experiment
        N_D = sub_exp_stats.get(1, pd.Series(dtype=float)).to_dict()
        N_C = sub_exp_stats.get(0, pd.Series(dtype=float)).to_dict()

        # Totals
        N_Omega_C = sum(N_C.values())

        if self.weighting == "population":
            # Pop_a^D: sum of population values for treated units per sub-exp
            treated_pop = (
                stacked_df[stacked_df["_D_sa"] == 1]
                .drop_duplicates(subset=[unit_col, "_sub_exp"])
                .groupby("_sub_exp")[population_col]
                .sum()
                .to_dict()
            )
            Pop_D_total = sum(treated_pop.values())

            q_control: Dict[Any, float] = {}
            for a in N_D:
                n_c = N_C.get(a, 0)
                if n_c == 0 or N_Omega_C == 0:
                    q_control[a] = 1.0
                    continue
                control_share = n_c / N_Omega_C
                pop_d = treated_pop.get(a, 0)
                pop_share = pop_d / Pop_D_total if Pop_D_total > 0 else 0.0
                q_control[a] = pop_share / control_share if control_share > 0 else 1.0

        else:  # sample_share
            N_Omega_D = sum(N_D.values())
            N_total = {a: N_D.get(a, 0) + N_C.get(a, 0) for a in N_D}
            N_grand = N_Omega_D + N_Omega_C

            q_control = {}
            for a in N_D:
                n_c = N_C.get(a, 0)
                if n_c == 0 or N_Omega_C == 0:
                    q_control[a] = 1.0
                    continue
                control_share = n_c / N_Omega_C
                n_total_a = N_total.get(a, 0)
                sample_share = n_total_a / N_grand if N_grand > 0 else 0.0
                q_control[a] = sample_share / control_share if control_share > 0 else 1.0

        # Assign weights: treated=1, control=q_control[sub_exp]
        sub_exp_vals = stacked_df["_sub_exp"].values
        d_vals = stacked_df["_D_sa"].values
        weights = np.ones(len(stacked_df))
        for i in range(len(stacked_df)):
            if d_vals[i] == 0:
                weights[i] = q_control.get(sub_exp_vals[i], 1.0)

        stacked_df["_Q_weight"] = weights
        return stacked_df

    def _compute_q_weights_aggregate(self, stacked_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute aggregate Q-weights using observation counts per (event_time, sub_exp).

        Matches the R reference ``compute_weights()`` which computes shares at the
        (event_time, sub_exp) level, not the sub_exp level. For balanced panels the
        two approaches are equivalent. For unbalanced panels this adjusts for varying
        observation density per event time.

        R reference pattern::

            stack_treat_n  = count(D==1) BY event_time
            stack_control_n = count(D==0) BY event_time
            sub_treat_n    = count(D==1) BY (sub_exp, event_time)
            sub_control_n  = count(D==0) BY (sub_exp, event_time)
            sub_treat_share = sub_treat_n / stack_treat_n
            sub_control_share = sub_control_n / stack_control_n
            Q = sub_treat_share / sub_control_share  (for controls)
            Q = 1  (for treated)
        """
        # Step 1: Stack-level totals by (event_time, D_sa)
        stack_counts = stacked_df.groupby(["_event_time", "_D_sa"]).size().unstack(fill_value=0)
        stack_treat_n = stack_counts.get(1, pd.Series(0, index=stack_counts.index))
        stack_control_n = stack_counts.get(0, pd.Series(0, index=stack_counts.index))

        # Step 2: Sub-experiment-level counts by (event_time, sub_exp, D_sa)
        sub_counts = (
            stacked_df.groupby(["_event_time", "_sub_exp", "_D_sa"]).size().unstack(fill_value=0)
        )
        sub_treat_n = sub_counts.get(1, pd.Series(0, index=sub_counts.index))
        sub_control_n = sub_counts.get(0, pd.Series(0, index=sub_counts.index))

        # Step 3: Compute shares and Q per (event_time, sub_exp)
        # Q = (sub_treat_n / stack_treat_n) / (sub_control_n / stack_control_n)
        q_lookup: Dict[Tuple[Any, Any], float] = {}
        for et, sub_exp in sub_counts.index:
            s_treat = sub_treat_n.get((et, sub_exp), 0)
            s_control = sub_control_n.get((et, sub_exp), 0)
            st_treat = stack_treat_n.get(et, 0)
            st_control = stack_control_n.get(et, 0)

            if s_control == 0 or st_treat == 0 or st_control == 0:
                q_lookup[(et, sub_exp)] = 1.0
            else:
                treat_share = s_treat / st_treat
                control_share = s_control / st_control
                q_lookup[(et, sub_exp)] = treat_share / control_share if control_share > 0 else 1.0

        # Step 4: Assign weights via vectorized merge
        et_vals = stacked_df["_event_time"].values
        sub_exp_vals = stacked_df["_sub_exp"].values
        d_vals = stacked_df["_D_sa"].values
        weights = np.ones(len(stacked_df))

        for i in range(len(stacked_df)):
            if d_vals[i] == 0:
                weights[i] = q_lookup.get((et_vals[i], sub_exp_vals[i]), 1.0)

        stacked_df["_Q_weight"] = weights
        return stacked_df

    # =========================================================================
    # Covariate balancing (CBWSDID, Ustyuzhanin 2026)
    # =========================================================================

    def _compute_balancing_weights(
        self,
        stacked_df: pd.DataFrame,
        df: pd.DataFrame,
        unit: str,
        time: str,
        first_treat: str,
        covariates: List[str],
    ) -> Tuple[pd.DataFrame, Dict[Any, Dict[str, Any]]]:
        """Compute CBWSDID covariate-balancing weights (Ustyuzhanin 2026, §3.1).

        For each sub-experiment ``a``, balance the clean controls' covariate means —
        measured at the last pre-treatment period ``t = a-1-anticipation`` from the
        SOURCE data, so the design weights use only pre-treatment information
        (Assumption 4) — toward the treated-cohort means via entropy balancing,
        yielding nonnegative design weights ``b_sa`` (treated keep ``b=1``). Then
        compose the final stacked weights with the EFFECTIVE control mass
        ``Ñ^C_a = Σ_{s∈C_a} b_sa``::

            W_sa = b_sa · (N^D_a / N^D_Ω) / (Ñ^C_a / Ñ^C_Ω)   for s ∈ C_a
            W_sa = 1                                          for s ∈ D_a

        A naive ``b_sa · Q_aggregate`` multiply is **not** equivalent: it aggregates the
        cohort control means with weights ∝ (N^D_a/N^D_Ω)·(Ñ^C_a/N^C_a) instead of the
        required ∝ (N^D_a/N^D_Ω), biasing the estimate unless ``b_sa`` is uniform.

        Overwrites ``_Q_weight`` with W_sa (so the downstream WLS consumes it
        transparently) and records the raw design weights in ``_b_sa``. Fail-closed:
        raises a cohort-named ``ValueError`` if a cohort's balance is infeasible —
        dropping the cohort would silently shift the estimand to an overlap-trimmed ATT.
        """
        balance_tol = 1e-8
        sub_exps = list(pd.unique(stacked_df["_sub_exp"]))

        b_lookup: Dict[Tuple[Any, Any], float] = {}
        N_D: Dict[Any, float] = {}
        Nt_C: Dict[Any, float] = {}
        diagnostics: Dict[Any, Dict[str, Any]] = {}

        # Balanced event windows are required: the paper's unit-count corrective and
        # diff-diff's observation-count "aggregate" Q-weights coincide only when every
        # unit is observed at every event time. On ragged windows they diverge (the
        # count-convention is unresolved — out of scope for v1), so fail closed rather
        # than silently producing unit-count estimates that differ from balance="none".
        # The check validates exact (unit x event_time) coverage, not just row counts:
        # it catches (a) eligible units with zero rows in the window (silently dropped
        # by _build_sub_experiment, so invisible to a count), (b) wrong row counts, and
        # (c) duplicate (unit, event_time) rows that a count-only check would let pass
        # alongside a compensating missing row.
        expected_events = list(range(-self.kappa_pre - self.anticipation, self.kappa_post + 1))
        n_expected = len(expected_events)
        ft_by_unit = df.drop_duplicates(subset=[unit]).set_index(unit)[first_treat]

        def _expected_units(a_val: Any) -> set:
            treated = set(ft_by_unit[ft_by_unit == a_val].index)
            if self.control_group == "not_yet_treated":
                controls = set(ft_by_unit[ft_by_unit > a_val + self.kappa_post].index)
            elif self.control_group == "strict":
                controls = set(
                    ft_by_unit[ft_by_unit > a_val + self.kappa_post + self.kappa_pre].index
                )
            else:  # never_treated
                controls = set(ft_by_unit[np.isinf(ft_by_unit)].index)
            return treated | controls

        for a in sub_exps:
            sub = stacked_df[stacked_df["_sub_exp"] == a]
            counts = sub.groupby(unit).size()
            present_units = set(counts.index)
            missing_eligible = _expected_units(a) - present_units  # zero-row eligible units
            wrong_count = set(counts[counts != n_expected].index)
            dup_mask = sub.duplicated(subset=[unit, "_event_time"], keep=False)
            dup_units = set(sub.loc[dup_mask, unit].unique())
            bad = missing_eligible | wrong_count | dup_units
            if bad:
                raise ValueError(
                    f"balance='entropy' requires balanced event windows, but cohort a={a} "
                    f"has {len(bad)} treated/clean-control unit(s) without exact coverage of "
                    f"the {n_expected} event times {expected_events} (zero-row, missing/extra, "
                    f"or duplicated (unit, event_time) rows; e.g. {list(bad)[:3]}). Covariate "
                    "balancing on unbalanced/ragged panels is out of scope for v1 because the "
                    "paper's unit-count corrective and diff-diff's observation-count "
                    "'aggregate' Q-weights diverge off balanced panels. Use balance='none', or "
                    "restrict to a balanced window."
                )
            treated_units = list(pd.unique(sub.loc[sub["_D_sa"] == 1, unit]))
            control_units = list(pd.unique(sub.loc[sub["_D_sa"] == 0, unit]))

            ref_time = a - 1 - self.anticipation
            pre = df.loc[df[time] == ref_time].drop_duplicates(subset=[unit]).set_index(unit)
            missing_units = [u for u in treated_units + control_units if u not in pre.index]
            if missing_units:
                raise ValueError(
                    f"Covariate balancing for cohort a={a}: {len(missing_units)} unit(s) "
                    f"have no observation at the pre-treatment reference period "
                    f"t={ref_time} (e.g. {missing_units[:3]}). Balancing requires the "
                    "covariate values at t=a-1-anticipation for every treated and "
                    "clean-control unit."
                )
            Xt = pre.loc[treated_units, covariates].to_numpy(dtype=np.float64)
            Xc = pre.loc[control_units, covariates].to_numpy(dtype=np.float64)
            if not (np.all(np.isfinite(Xt)) and np.all(np.isfinite(Xc))):
                raise ValueError(
                    f"Covariate balancing for cohort a={a}: covariates contain NaN/inf at "
                    f"t={ref_time}. Balancing requires finite covariate values."
                )

            target = Xt.mean(axis=0)
            pre_imbalance = float(np.max(np.abs(Xc.mean(axis=0) - target)))
            try:
                b_control, info = entropy_balance(Xc, target, tol=balance_tol)
            except BalanceError as exc:
                worst = covariates[int(np.argmax(np.abs(exc.residuals)))]
                raise ValueError(
                    f"Covariate balancing failed for cohort a={a}: could not match the "
                    f"treated covariate means (worst covariate '{worst}', residual "
                    f"{exc.max_residual:.3e}). The treated cohort's covariate profile lies "
                    "outside the clean-control support (infeasible). Remove this cohort, "
                    "reduce the covariate set, or use balance='none'."
                ) from exc

            for u in treated_units:
                b_lookup[(a, u)] = 1.0
            for u, b in zip(control_units, b_control):
                b_lookup[(a, u)] = float(b)
            N_D[a] = float(len(treated_units))
            Nt_C[a] = float(np.sum(b_control))

            ess = float(info["ess"])
            if ess < max(2.0, 0.05 * len(control_units)):
                warnings.warn(
                    f"Covariate balancing for cohort a={a} produced highly concentrated "
                    f"control weights (effective sample size {ess:.1f} of "
                    f"{len(control_units)} controls); estimates for this cohort may be "
                    "unstable.",
                    UserWarning,
                    stacklevel=3,
                )
            diagnostics[a] = {
                "n_treated": int(len(treated_units)),
                "n_control": int(len(control_units)),
                "effective_control_mass": Nt_C[a],
                "ess": ess,
                "max_imbalance_pre": pre_imbalance,
                "max_imbalance_post": float(info["max_residual"]),
                "balance_solver": info["solver"],
            }

        N_D_Omega = sum(N_D.values())
        Nt_C_Omega = sum(Nt_C.values())
        corr = {a: (N_D[a] / N_D_Omega) / (Nt_C[a] / Nt_C_Omega) for a in sub_exps}

        sub_vals = stacked_df["_sub_exp"].to_numpy()
        unit_vals = stacked_df[unit].to_numpy()
        d_vals = stacked_df["_D_sa"].to_numpy()
        b_vals = np.array([b_lookup[(sub_vals[i], unit_vals[i])] for i in range(len(stacked_df))])
        corr_vals = np.array([corr[a] for a in sub_vals])
        W = np.where(d_vals == 0, b_vals * corr_vals, 1.0)

        stacked_df = stacked_df.copy()
        stacked_df["_b_sa"] = b_vals
        stacked_df["_Q_weight"] = W
        return stacked_df, diagnostics

    # =========================================================================
    # sklearn-compatible interface
    # =========================================================================

    # get_params/set_params come from BaseEstimator.

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
# Convenience function
# =============================================================================


def stacked_did(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: str,
    kappa_pre: int = 1,
    kappa_post: int = 1,
    aggregate: Any = NOT_SUPPLIED,
    population: Optional[str] = None,
    survey_design=None,
    covariates: Optional[List[str]] = None,
    **kwargs: Any,
) -> StackedDiDResults:
    """
    Convenience function for stacked DiD estimation.

    .. deprecated:: 3.9
        ``stacked_did()`` is deprecated and will be removed in 4.0
        (row M-072). Construct the estimator instead:
        ``StackedDiD(...).fit(data, ...)``.

    This is a shortcut for creating a StackedDiD estimator and calling fit().

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
        Column indicating first treatment period (0 or inf for never-treated).
    kappa_pre : int, default=1
        Pre-treatment event-time periods.
    kappa_post : int, default=1
        Post-treatment event-time periods.
    aggregate : str, optional
        DEPRECATED (3.9, removed in 4.0; row M-024) - the event-study
        surface is always computed; passing this warns and changes
        nothing. Use ``results.aggregate('event_study')`` /
        ``results.aggregate('simple')`` post-fit.
    population : str, optional
        Population column for weighting="population".
    survey_design : SurveyDesign, optional
        Survey design specification for design-based inference.
    covariates : list of str, optional
        Covariate columns to balance the clean controls toward the treated
        cohort (pass ``balance="entropy"`` via ``**kwargs`` to enable). See
        ``StackedDiD.fit``.
    **kwargs
        Additional keyword arguments passed to StackedDiD constructor
        (e.g. ``balance="entropy"``, ``weighting``, ``cluster``, ``vcov_type``).

    Returns
    -------
    StackedDiDResults
        Estimation results.

    Examples
    --------
    >>> from diff_diff import stacked_did, generate_staggered_data
    >>> data = generate_staggered_data(seed=42)
    >>> results = stacked_did(data, 'outcome', 'unit', 'period',
    ...                       'first_treat', kappa_pre=2, kappa_post=2)
    >>> results.print_summary()
    >>> es = results.aggregate('event_study')  # post-fit view (M-024)
    """
    warnings.warn(
        "stacked_did() is deprecated and will be removed in 4.0; "
        "construct the estimator instead: StackedDiD(...).fit(data, ...).",
        FutureWarning,
        stacklevel=2,
    )
    est = StackedDiD(kappa_pre=kappa_pre, kappa_post=kappa_post, **kwargs)
    return est.fit(
        data,
        outcome=outcome,
        unit=unit,
        time=time,
        first_treat=first_treat,
        aggregate=aggregate,
        population=population,
        survey_design=survey_design,
        covariates=covariates,
    )
