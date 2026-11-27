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
import dataclasses
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff._base import BaseEstimator
from diff_diff.balancing import BalanceError, entropy_balance
from diff_diff.linalg import effective_cluster_count, solve_ols
from diff_diff.stacked_did_results import StackedDiDResults  # noqa: F401 (re-export)
from diff_diff.utils import (
    resolve_tail_df,
    safe_inference,
    validate_anticipation,
    validate_df_convention,
)

__all__ = [
    "StackedDiD",
    "StackedDiDResults",
]


def _cohort_share_weights(
    sub_exp: np.ndarray,
    treated: np.ndarray,
    unit: np.ndarray,
    weighting: str,
    row_weights: Optional[np.ndarray] = None,
    population: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[Any, str]]:
    """Corrective Q-weights that give both arms the same cohort shares.

    Wing, Freedman & Hollingsworth (2024, Section 5.3) show the stacked
    regression recovers a convex combination of the sub-experiment ATTs only
    if the treated arm and the control arm aggregate sub-experiments with the
    SAME shares; otherwise the untreated trends do not cancel. With ``w_s`` a
    unit's survey weight (1 without a survey design) and the sums taken over
    the units of sub-experiment ``a``::

        W_a^D = sum of w_s over treated units,  W_a^C = sum over controls
        pi_a  = W_a^D / W^D                              (aggregate)
        pi_a  = sum of w_s * pop_s over treated / total  (population)
        pi_a  = (W_a^D + W_a^C) / (W^D + W^C)            (sample_share)

        Q_sa = pi_a / (W_a^D / W^D)   for treated rows
        Q_sa = pi_a / (W_a^C / W^C)   for control rows

    Without survey weights these are the paper's Table 1 weights (treated
    ``Q = 1`` under ``aggregate``). A sub-experiment with no treated mass,
    no control mass, or a zero target share gets ``Q = 0`` on every row and
    is left out of the totals; its rows stay in the stack.

    Parameters
    ----------
    sub_exp, treated, unit : np.ndarray
        Per stacked row: sub-experiment label, treated flag (1/0), unit id.
    weighting : {"aggregate", "population", "sample_share"}
    row_weights : np.ndarray, optional
        Per-row survey weight, constant within a unit. ``None`` = all ones.
    population : np.ndarray, optional
        Per-row population (``weighting="population"`` only); read on
        treated rows, constant within a unit.

    Returns
    -------
    q : np.ndarray
        Q-weight per stacked row.
    zero_weight : dict
        Sub-experiments that receive zero weight, mapped to the reason.
    """
    n = len(sub_exp)
    is_treated = np.asarray(treated) == 1
    frame = pd.DataFrame(
        {
            "a": sub_exp,
            "u": unit,
            "d": is_treated,
            "w": np.ones(n) if row_weights is None else np.asarray(row_weights, dtype=float),
        }
    )
    if population is not None:
        frame["p"] = np.asarray(population, dtype=float)
    # One row per (sub-experiment, unit): every stacked unit covers the whole
    # window and its weight is constant, so the first row carries its mass.
    units = frame.loc[~frame.duplicated(subset=["a", "u"])]
    events = list(pd.unique(frame["a"]))
    w_d = units.loc[units["d"]].groupby("a")["w"].sum().reindex(events, fill_value=0.0)
    w_c = units.loc[~units["d"]].groupby("a")["w"].sum().reindex(events, fill_value=0.0)

    zero_weight: Dict[Any, str] = {}
    for a in events:
        if not w_d[a] > 0:
            zero_weight[a] = "no treated unit with positive weight"
        elif not w_c[a] > 0:
            zero_weight[a] = "no clean control with positive weight"
    effective = [a for a in events if a not in zero_weight]

    q_treated = pd.Series(0.0, index=events)
    q_control = pd.Series(0.0, index=events)
    if effective:
        w_d_total = float(w_d[effective].sum())
        w_c_total = float(w_c[effective].sum())
        if weighting == "aggregate":
            target = w_d / w_d_total
        elif weighting == "population":
            treated_units = units.loc[units["d"]]
            pop_mass = (
                (treated_units["w"] * treated_units["p"])
                .groupby(treated_units["a"])
                .sum()
                .reindex(events, fill_value=0.0)
            )
            pop_total = float(pop_mass[effective].sum())
            target = pop_mass / pop_total if pop_total > 0 else pop_mass * 0.0
            for a in effective:
                if not pop_mass[a] > 0:
                    zero_weight[a] = "zero treated population"
        else:  # sample_share
            target = (w_d + w_c) / (w_d_total + w_c_total)
        for a in effective:
            if a in zero_weight:
                continue
            q_treated[a] = target[a] / (w_d[a] / w_d_total)
            q_control[a] = target[a] / (w_c[a] / w_c_total)

    q = np.where(
        is_treated, frame["a"].map(q_treated).to_numpy(), frame["a"].map(q_control).to_numpy()
    ).astype(float)
    return q, zero_weight


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
        Target estimand weighting scheme per Table 1 of the paper. Each sets
        the share ``pi_a`` with which sub-experiment ``a`` enters the aggregate
        ``sum_a pi_a ATT(a, a+e)``, and both the treated and the control rows
        are weighted so that each arm carries that share:

        - "aggregate": share of treated units (trimmed aggregate ATT; treated
          rows have Q = 1)
        - "population": share of the treated units' population
        - "sample_share": share of all units (treated plus clean controls)

        Under ``survey_design=`` the shares are formed from survey-weighted
        unit masses.
    control_group : str, default="not_yet_treated"
        How to define clean controls per Appendix A of the paper:
        - "not_yet_treated": Units with A_s > a + kappa_post + anticipation
        - "strict": Units with A_s > a + kappa_post + kappa_pre + anticipation
        - "never_treated": Only units with A_s = infinity

        ``anticipation`` enters because a later adopter responds from
        ``A_s - anticipation``; with the default ``anticipation=0`` these are
        the paper's rules.
    cluster : str, default="unit"
        Clustering level for standard errors:
        - "unit": Cluster on original unit identifier
        - "unit_subexp": Cluster on (unit, sub_experiment) pairs

        Under ``survey_design=`` with Taylor-linearized variance the survey
        PSU governs: the design's ``psu`` when it names one, otherwise each
        original unit. ``"unit_subexp"`` pairs are never used as survey PSUs.
        Replicate-weight designs take their variance from the replicate
        weights and use no PSU.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    anticipation : int, default=0
        Number of anticipation periods. Must be a non-negative integer;
        ``bool`` is rejected. When anticipation > 0:
        - Reference period shifts from e=-1 to e=-1-anticipation
        - Post-treatment includes anticipation periods (e >= -anticipation)
        - Event window expands by anticipation pre-periods
        - Clean-control cutoffs for later adopters move out by anticipation
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
        ``weighting="aggregate"``. v1 requires ``weighting="aggregate"`` and does
        not support ``survey_design=``; on a ragged panel it runs on the units
        observed over their whole event window, like the default path (see
        Notes);
        matching-based balancing and the repeated-treatment extension are out of
        scope. Default ``"none"`` reproduces plain weighted stacked DID.
    df_convention : {"residual", "cluster", "normal"}, default "cluster"
        Degrees-of-freedom convention for the analytical t/p/CI on the
        pooled stacked regression's event-study and overall-ATT inference.
        ``"cluster"`` (default since 4.0) uses ``G − 1`` where G counts
        positive-weight clusters (``effective_cluster_count`` — may differ
        from the raw ``results.n_clusters`` when a cluster's composed weight
        is zero); ``"residual"`` (the 3.x default) uses the pooled residual
        df (``n_eff − k_kept``, positive-weight rows) — the 3.9 fix: the
        non-BM non-survey lane previously used silent normal-theory z;
        ``"normal"`` deliberately uses normal-theory z at the fallback
        level. hc2_bm Bell-McCaffrey contrast DOF and survey/replicate df
        always take precedence. Note the stacked design's residual df is
        typically large (control rows replicate across sub-experiments), so
        the 3.9 z -> t(residual) step was small and the 4.0 flip to
        ``G − 1`` is the materially moving step (row M-130); pass
        ``df_convention="residual"`` for 3.x inference.

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

    **Complete event windows.** A unit enters a sub-experiment only if it is
    observed at every event time of the window
    ``[-kappa_pre - anticipation, kappa_post]``. The regression contrasts
    event-time cell means, which are within-unit changes only when the same
    units sit in every cell (the paper's stack has each unit observed for
    ``kappa_pre + kappa_post + 1`` periods, Section 5.1). On a ragged panel
    the units that are not are excluded from that sub-experiment with a
    ``UserWarning`` reporting the counts, and the estimand is the trimmed
    aggregate ATT over the units observed across their whole window. An
    adoption event left without a complete treated unit or clean control
    (including one whose window contains a period no unit observes) is
    dropped with a warning and listed in ``results.trimmed_groups``; if none
    is left, ``fit`` raises ``ValueError``. Under ``survey_design=`` both the
    unit-level exclusion and a dropped event are refused
    (``NotImplementedError``). ``time`` must
    be integer-valued with at most one row per unit and period.

    References
    ----------
    Wing, C., Freedman, S. M., & Hollingsworth, A. (2024). Stacked
        Difference-in-Differences. NBER Working Paper 32054.
    """

    def __init__(
        self,
        kappa_pre: int = 1,
        kappa_post: int = 1,
        weighting: str = "aggregate",
        control_group: str = "not_yet_treated",
        cluster: str = "unit",
        alpha: float = 0.05,
        anticipation: int = 0,
        rank_deficient_action: str = "warn",
        vcov_type: str = "hc1",
        balance: str = "none",
        df_convention: str = "cluster",
    ):
        if weighting not in ("aggregate", "population", "sample_share"):
            raise ValueError(
                f"weighting must be 'aggregate', 'population', or 'sample_share', "
                f"got '{weighting}'"
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
        self.control_group = control_group
        self.cluster = cluster
        self.alpha = alpha
        self.anticipation = validate_anticipation(anticipation)
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
        population : str, optional
            Column name for population weights. Required only when
            weighting="population". Read for the treated units in the stack,
            where it must be finite, non-negative and constant within a unit.
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
            Object containing all estimation results. The event-study surface
            is always computed at fit; aggregate as a post-fit step via
            ``results.aggregate('event_study')`` / ``results.aggregate('simple')``
            (``'group'`` is not available: the pooled stacked regression
            cannot produce cohort-specific effects - use CallawaySantAnna or
            ImputationDiD for cohort-level estimates).

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails: an
            empty panel, a missing or non-integer ``time``, a duplicated ``(unit, time)``
            row, a non-integer ``first_treat``, or no adoption event with
            both a treated unit and a clean control observed at every event
            time of the window. Also under ``weighting="population"`` when a
            treated unit's population is missing, negative or not constant
            within the unit, and under ``survey_design=`` when a survey column
            varies within a unit.
        NotImplementedError
            Under ``survey_design=`` when a treated or clean-control unit is
            not observed at every event time of its sub-experiment window, or
            an adoption event would be dropped for lack of complete units
            (removing them would drop their PSUs from the design). Also under
            ``survey_design=`` with ``anticipation > 0`` when the
            anticipation-adjusted clean-control cutoff leaves a later cohort
            out of every sub-experiment, for the same reason.

        Warns
        -----
        UserWarning
            When units are excluded from a sub-experiment because they are
            not observed at every event time of its window, and when an
            adoption event is dropped because no treated unit or no clean
            control is. Also when a sub-experiment carries zero weight (no
            treated or control unit with positive survey weight, or a zero
            treated population under ``weighting="population"``).
        """
        # Fit-time re-check: __init__ and set_params validate eagerly, so
        # this only catches DIRECT attribute mutation (est.anticipation = ...)
        # — an out-of-domain value silently changes the ESTIMAND. The
        # assignment also re-normalizes a mutated numpy scalar to int.
        self.anticipation = validate_anticipation(self.anticipation)

        # ---- Validate inputs ----
        required_cols = [outcome, unit, time, first_treat]
        if population is not None:
            required_cols.append(population)
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if len(data) == 0:
            # Before the survey design is resolved: an empty frame has no
            # rows to read a design from (an FPC column would index an empty
            # array).
            raise ValueError(
                "StackedDiD.fit() received an empty panel (0 rows): there is "
                "nothing to estimate."
            )

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
            _validate_unit_constant_survey,
        )

        # A Taylor-linearized design that names no PSU samples UNITS: the unit
        # column is declared as the PSU here, before anything is trimmed, so
        # the design is validated against the FULL supplied sample (an FPC
        # below a stratum's unit count raises, as it does for an explicit
        # ``psu``) and the same specification is reused on the stack
        # (REGISTRY StackedDiD, survey-PSU Note).
        _implicit_unit_psu = (
            isinstance(survey_design, SurveyDesign)
            and survey_design.psu is None
            and survey_design.replicate_weights is None
        )
        _design_for_fit = (
            dataclasses.replace(survey_design, psu=unit) if _implicit_unit_psu else survey_design
        )
        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(_design_for_fit, data, "analytical")
        )
        # Survey columns must be constant within a unit: the regression
        # contrasts event-time cell means, and a weight that changes over a
        # unit's rows changes each cell's weighted composition exactly as a
        # missing row does (REGISTRY StackedDiD complete-window Note).
        if survey_design is not None and isinstance(survey_design, SurveyDesign):
            _validate_unit_constant_survey(data, unit, survey_design)
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
        self._validate_panel_grid(df, unit, time, first_treat)

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

        # Under a survey design the anticipation-adjusted clean-control cutoff
        # must not take sampled units out of the stack altogether: the design
        # is re-resolved on the stacked rows, so their PSUs and strata would
        # leave the variance and df_survey (REGISTRY StackedDiD, anticipation
        # Note). Refused before any estimate is formed.
        if survey_design is not None:
            lost = self._units_lost_to_anticipation_cutoff(omega_A, T_min, T_max, unit_info)
            if lost:
                raise NotImplementedError(
                    "StackedDiD with survey_design= and "
                    f"anticipation={self.anticipation}: the clean-control cutoff "
                    "moves out by the anticipation periods, and that leaves "
                    f"{sum(lost.values())} unit(s) out of every sub-experiment ("
                    + ", ".join(f"cohort {int(c)}: {n}" for c, n in lost.items())
                    + "). That can take PSUs or strata out of the survey "
                    "variance and its degrees of freedom, which is not domain "
                    "estimation, so the fit is refused whenever a sampled unit "
                    "leaves the stack. Fit without survey_design=, or remove those "
                    "cohorts from the data yourself and adjust the design "
                    "accordingly."
                )

        # ---- Build stacked dataset ----
        # Complete-window rule (REGISTRY StackedDiD Note): a unit enters a
        # sub-experiment only if it has a row at EVERY integer event time of
        # the window. Equation 3 contrasts (treated x event-time) and
        # (control x event-time) cell means, which are within-unit changes
        # only when the same units sit in every cell (Wing et al. 2024,
        # Section 5.1: "each unit will be observed for kappa_pre +
        # kappa_post + 1 time periods").
        win_lo = -self.kappa_pre - self.anticipation
        win_hi = self.kappa_post
        observed_periods = {int(t) for t in time_periods}
        sub_experiments = []
        realized_events: List[Any] = []
        unrealized: Dict[Any, str] = {}
        coverage: Dict[Any, Dict[str, int]] = {}
        for a in omega_kappa:
            a_int = int(a)
            # (i) A window period that NO unit observes: no unit can be
            # complete, so the event cannot be realized.
            absent = [
                t for t in range(a_int + win_lo, a_int + win_hi + 1) if t not in observed_periods
            ]
            if absent:
                unrealized[a] = (
                    f"calendar period(s) {absent} of its window are not observed for any unit"
                )
                continue
            sub_exp, counts = self._build_sub_experiment(
                df,
                unit_info,
                a,
                unit,
                time,
                first_treat,
                outcome,
                extra_cols=survey_cols,
            )
            coverage[a] = counts
            if sub_exp is not None:
                sub_experiments.append(sub_exp)
                realized_events.append(a)
            else:
                reasons = []
                if counts["treated_kept"] == 0:
                    reasons.append(
                        "no treated unit is observed at every event time "
                        f"(0 of {counts['treated_eligible']})"
                    )
                if counts["control_kept"] == 0:
                    reasons.append(
                        "no clean control is observed at every event time "
                        f"(0 of {counts['control_eligible']})"
                    )
                unrealized[a] = " and ".join(reasons)

        n_dropped_treated = sum(
            c["treated_eligible"] - c["treated_kept"] for c in coverage.values()
        )
        n_dropped_control = sum(
            c["control_eligible"] - c["control_kept"] for c in coverage.values()
        )

        # (ii)/(iii) Under a survey design, removing units for unit-level
        # missingness is REFUSED before any estimate is formed: their PSUs
        # and strata would leave the Taylor-linearized meat and df_survey,
        # which is not domain estimation. An event lost to a period the
        # whole panel lacks is refused for the same reason further down,
        # after the "nothing left" diagnosis, which takes precedence when
        # no event at all can be formed.
        if survey_design is not None and (n_dropped_treated + n_dropped_control) > 0:
            raise NotImplementedError(
                "StackedDiD with survey_design= requires every treated and "
                "clean-control unit to be observed at every event time of its "
                f"sub-experiment window [{win_lo}, {win_hi}], but "
                f"{n_dropped_treated} treated and {n_dropped_control} control "
                "unit-by-sub-experiment entries are not ("
                + self._format_coverage(coverage)
                + "). Without a survey design those units are trimmed; under a "
                "survey design removing them would drop their PSUs and strata "
                "from the variance and its degrees of freedom, which is not "
                "domain estimation. Supply a panel in which every unit is "
                "observed at every period, or fit without survey_design=."
            )

        if len(sub_experiments) == 0:
            raise ValueError(
                "No sub-experiment has both a treated unit and a clean control "
                f"observed at every event time of the window [{win_lo}, {win_hi}] "
                f"(kappa_pre={self.kappa_pre}, kappa_post={self.kappa_post}, "
                f"anticipation={self.anticipation}): "
                + "; ".join(f"event {int(a)}: {why}" for a, why in unrealized.items())
                + "."
                + self._describe_time_grid_gaps(T_min, T_max, observed_periods)
                + " Reduce kappa_pre / kappa_post, or supply the missing observations."
            )

        if survey_design is not None and unrealized:
            raise NotImplementedError(
                "StackedDiD with survey_design= cannot drop adoption event(s) "
                "that have no sub-experiment of complete units ("
                + "; ".join(f"event {int(a)}: {why}" for a, why in unrealized.items())
                + "): the dropped cohorts' PSUs and strata would leave the "
                "survey variance and its degrees of freedom, which is not "
                "domain estimation. Remove those cohorts from the data "
                "yourself (and adjust the design accordingly), narrow the "
                "event window, or fit without survey_design=."
            )

        if unrealized:
            warnings.warn(
                f"{len(unrealized)} adoption event(s) were dropped because no "
                "sub-experiment could be formed from units observed at every "
                f"event time of the window [{win_lo}, {win_hi}]: "
                + "; ".join(f"event {int(a)}: {why}" for a, why in unrealized.items())
                + ". They are reported in trimmed_groups and do not enter the "
                "trimmed aggregate ATT.",
                UserWarning,
                stacklevel=2,
            )
            trimmed = sorted(list(trimmed) + list(unrealized))

        realized_cov = {a: coverage[a] for a in realized_events}
        n_trim_t = sum(c["treated_eligible"] - c["treated_kept"] for c in realized_cov.values())
        n_trim_c = sum(c["control_eligible"] - c["control_kept"] for c in realized_cov.values())
        if n_trim_t + n_trim_c > 0:
            warnings.warn(
                f"{n_trim_t} treated and {n_trim_c} control unit-by-sub-experiment "
                "entries were excluded because the unit is not observed at every "
                f"event time of the window [{win_lo}, {win_hi}] ("
                + self._format_coverage(realized_cov)
                + "). The stacked regression contrasts event-time cell means, "
                "which requires the same units at every event time of a "
                "sub-experiment (Wing et al. 2024, Section 5.1), so the estimand "
                "is the trimmed aggregate ATT over the units observed across "
                "their whole window.",
                UserWarning,
                stacklevel=2,
            )

        stacked_df = pd.concat(sub_experiments, ignore_index=True)

        # ---- Compute Q-weights ----
        # Both arms must carry the same cohort shares (helper docstring), so
        # the shares are formed from survey-weighted unit masses when a
        # survey design supplies weights, and recomputed in every replicate.
        _sub_arr = stacked_df["_sub_exp"].to_numpy()
        _d_arr = stacked_df["_D_sa"].to_numpy()
        _unit_arr = stacked_df[unit].to_numpy()
        _survey_w_rows: Optional[np.ndarray] = None
        if survey_design is not None and getattr(survey_design, "weights", None) is not None:
            _survey_w_rows = stacked_df[survey_design.weights].to_numpy(dtype=np.float64)
        _pop_rows: Optional[np.ndarray] = None
        if self.weighting == "population":
            assert population is not None  # validated at the top of fit()
            self._validate_treated_population(stacked_df, unit, population)
            # Only treated rows are read: control populations are unused and
            # may hold anything (missing values, pd.NA in an object column).
            _treated_rows = _d_arr == 1
            _pop_rows = np.zeros(len(stacked_df), dtype=np.float64)
            _pop_rows[_treated_rows] = pd.to_numeric(
                stacked_df.loc[_treated_rows, population]
            ).to_numpy(dtype=np.float64)
        _q_rows, _zero_weight_events = _cohort_share_weights(
            _sub_arr,
            _d_arr,
            _unit_arr,
            self.weighting,
            row_weights=_survey_w_rows,
            population=_pop_rows,
        )
        stacked_df["_Q_weight"] = _q_rows
        if _zero_weight_events:
            warnings.warn(
                f"{len(_zero_weight_events)} sub-experiment(s) carry zero weight "
                "and contribute nothing to the estimates: "
                + "; ".join(f"event {int(a)}: {why}" for a, why in _zero_weight_events.items())
                + ". Their rows stay in the stack (and in groups, "
                "n_sub_experiments and the unit counts).",
                UserWarning,
                stacklevel=2,
            )

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
            # Normalize composed weights to sum = n_stacked. A zero total (no
            # sub-experiment carries weight) is left as is, so the
            # reference-support guard below reports it.
            _composed_sum = float(np.sum(composed_weights))
            if _composed_sum > 0:
                composed_weights = composed_weights * (n_stacked / _composed_sum)
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
        # (survey pweights can zero a cell outright; population
        # weighting zeroes every weight when the treated population is
        # zero) is as empty as a missing one, and with kappa_pre = 0
        # the reference lies outside the window altogether. Gapped and
        # ragged panels no longer reach this guard: the complete-window
        # rule in _build_sub_experiment removes them first. An effectively-empty cell makes the design
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
                "arbitrary horizon. Causes: kappa_pre = 0 (the reference "
                "period lies outside the event window; use kappa_pre >= "
                "1); weighting='population' with zero treated population "
                "(every weight is then zero); or survey weights "
                "that zero out a reference cell."
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
        # (sun_abraham.py:997-1097) and the event-study core's contrast pattern from PR #465.
        # Computed BEFORE constructing event_study_effects / overall_*
        # inference so the DOFs can be threaded into the safe_inference calls.
        _bm_contrast_dof_per_event: Dict[int, float] = {}
        _bm_contrast_dof_overall: Optional[float] = None
        if self.vcov_type == "hc2_bm" and not _uses_replicate_sd and not np.all(np.isnan(coef)):
            from diff_diff.linalg import _compute_cr2_bm_contrast_dof

            # Mirror the event-study core's rank-deficient pattern (PR #465,
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
            # (The pre-3.9 gate skipped them unless an event study was
            # requested at fit; that gate's premise died with the post-fit
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
                # The corrective weights depend on the survey weights (they
                # equalize the arms' survey-weighted cohort shares), so each
                # replicate recomputes them from its own weights.
                q_r, _ = _cohort_share_weights(
                    _sub_arr,
                    _d_arr,
                    _unit_arr,
                    self.weighting,
                    row_weights=w_r,
                    population=_pop_rows,
                )
                composed_r = q_r * w_r
                w_sum = np.sum(composed_r)
                if not w_sum > 0:
                    # No sub-experiment has both arms in this replicate.
                    return np.full(X.shape[1], np.nan)
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
            #
            # A design without an explicit PSU samples UNITS: each original
            # unit is its own PSU, whatever ``cluster`` says. The stacked
            # (unit, sub-experiment) pairs are not sampling units - a unit that
            # serves in several sub-experiments is still one draw, and the pair
            # count moves with the stack. The unit column is therefore declared
            # as the PSU BEFORE resolving, so that the implicit PSUs pass
            # through the same singleton-stratum and FPC validation as an
            # explicit ``survey_design.psu`` (REGISTRY StackedDiD, survey-PSU
            # Note).
            if _implicit_unit_psu:
                if self.cluster == "unit_subexp":
                    warnings.warn(
                        "StackedDiD with survey_design= and no survey_design.psu "
                        "uses each unit as its own PSU for the survey variance; "
                        "cluster='unit_subexp' does not apply there, because a "
                        "unit that appears in several sub-experiments is one "
                        "sampled unit. Name the sampling unit in "
                        "survey_design.psu to choose a different PSU.",
                        UserWarning,
                        stacklevel=2,
                    )
                resolved_stacked = _design_for_fit.resolve(stacked_df)
            else:
                resolved_stacked = survey_design.resolve(stacked_df)

            # Create a copy with composed weights (normalized to sum=n_stacked)
            resolved_composed = copy.copy(resolved_stacked)
            resolved_composed.weights = composed_weights

            # Original-scale residuals for TSL variance
            resid_orig = Y - X @ coef

            if not _implicit_unit_psu:
                # Explicit PSU overrides the user-specified cluster.
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
            if survey_metadata.df_survey is not None and survey_metadata.df_survey <= 0:
                warnings.warn(
                    "StackedDiD: the survey design has zero degrees of freedom "
                    "(n_PSU - n_strata = 0), so its variance is not identified "
                    "for inference. Estimates and standard errors are reported; "
                    "t-statistics, p-values and confidence intervals are NaN.",
                    UserWarning,
                    stacklevel=2,
                )

        # ---- Extract event study effects ----
        # 3.9 (row M-024): the surface is ALWAYS materialized - the
        # regression always includes the event-time interactions, and the
        # post-fit results.aggregate("event_study") view must be total:
        # the event-study surface is always computed at fit.
        event_study_effects: Dict[int, Dict[str, Any]] = {}
        es_vcov: Optional[np.ndarray] = None
        es_vcov_index: Optional[List[int]] = None
        es_df_used: Dict[int, float] = {}
        # Reference period (e = -1 - anticipation): the normalization anchor,
        # marked EXPLICITLY (4.0, ledger row M-093 item 8a) - no count key;
        # the 3.x ``n_obs == 0`` sentinel is retired.
        event_study_effects[ref_period] = {
            "effect": 0.0,
            "se": 0.0,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_int": (np.nan, np.nan),
            "is_reference": True,
        }
        for h in event_times:
            idx = interaction_indices[h]
            effect = float(coef[idx])
            se = float(np.sqrt(max(vcov[idx, idx], 0.0)))
            # The design's own df (n_PSU - n_strata), unclamped: at 0 the
            # variance is not identified and safe_inference returns NaN.
            _survey_df = (
                survey_metadata.df_survey
                if survey_metadata is not None and survey_metadata.df_survey is not None
                else (0 if _uses_replicate_sd else None)
            )
            # Override df when replicate replicates were dropped
            if _n_valid_rep_sd is not None and resolved_stacked is not None:
                if _n_valid_rep_sd < resolved_stacked.n_replicates:
                    # Dropped replicates CAP the df at min(df, n_valid - 1); they never
                    # replace it (a rank-deficient replicate matrix has df < n_valid - 1).
                    from diff_diff.survey import cap_replicate_df

                    _survey_df = cap_replicate_df(_survey_df, _n_valid_rep_sd)
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
                # analytical fallback (t(G-1) under the default; the
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
            survey_metadata.df_survey
            if survey_metadata is not None and survey_metadata.df_survey is not None
            else (0 if _uses_replicate_sd else None)
        )
        if _n_valid_rep_sd is not None and resolved_stacked is not None:
            if _n_valid_rep_sd < resolved_stacked.n_replicates:
                # Dropped replicates CAP the df at min(df, n_valid - 1); they never
                # replace it (a rank-deficient replicate matrix has df < n_valid - 1).
                from diff_diff.survey import cap_replicate_df

                _survey_df_overall = cap_replicate_df(_survey_df_overall, _n_valid_rep_sd)
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
            att=overall_att,
            se=overall_se,
            t_stat=overall_t,
            p_value=overall_p,
            conf_int=overall_ci,
            event_study_effects=event_study_effects,
            group_effects=None,
            stacked_data=stacked_df,
            groups=list(realized_events),
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
            control_cutoff_includes_anticipation=True,
        )

        self.is_fitted_ = True
        return self.results_

    # =========================================================================
    # Input validation and coverage messages
    # =========================================================================

    @staticmethod
    def _validate_panel_grid(df: pd.DataFrame, unit: str, time: str, first_treat: str) -> None:
        """Reject inputs the event-time grid cannot represent.

        Event times are integer offsets ``time - a``, each unit contributes
        one row per event time, and the adoption date ``a`` is an integer
        period. A missing or fractional ``time``, a duplicated ``(unit,
        time)`` row or a fractional ``first_treat`` would otherwise be
        silently mis-assigned (dropped, absorbed into the reference
        category, double-counted, or shifted by ``int()`` truncation).
        """
        t_vals = df[time].to_numpy(dtype=float)
        n_bad = int(np.sum(~np.isfinite(t_vals)))
        if n_bad:
            raise ValueError(
                f"Column '{time}' has {n_bad} missing or non-finite value(s). "
                "StackedDiD places every row at an integer event time; drop or "
                "fill those rows before fitting."
            )
        frac = t_vals != np.floor(t_vals)
        if np.any(frac):
            raise ValueError(
                f"Column '{time}' must be integer-valued, but {int(np.sum(frac))} "
                f"row(s) are not (e.g. {t_vals[frac][:3].tolist()}). Event times "
                "are integer offsets from the adoption period; re-index time to "
                "consecutive integers."
            )
        dup = df.duplicated(subset=[unit, time], keep=False)
        if dup.any():
            example = df.loc[dup, [unit, time]].iloc[0].tolist()
            raise ValueError(
                f"Found {int(dup.sum())} rows sharing a ({unit}, {time}) pair "
                f"(e.g. {example}). StackedDiD requires at most one observation "
                "per unit and period; aggregate or drop the duplicates."
            )
        ft_vals = df[first_treat].to_numpy(dtype=float)
        ft_frac = np.isfinite(ft_vals) & (ft_vals != np.floor(ft_vals))
        if np.any(ft_frac):
            raise ValueError(
                f"Column '{first_treat}' must hold integer adoption periods (or "
                "0 / inf for never-treated units), but found non-integer "
                f"value(s) (e.g. {np.unique(ft_vals[ft_frac])[:3].tolist()})."
            )

    @staticmethod
    def _format_coverage(coverage: Dict[Any, Dict[str, int]], limit: int = 12) -> str:
        """Per-event "kept of eligible" breakdown for the coverage messages."""
        affected = [
            (a, c)
            for a, c in coverage.items()
            if (c["treated_kept"] < c["treated_eligible"])
            or (c["control_kept"] < c["control_eligible"])
        ]
        if not affected:
            return "all units kept"
        text = "kept " + "; ".join(
            f"event {int(a)}: {c['treated_kept']} of {c['treated_eligible']} treated, "
            f"{c['control_kept']} of {c['control_eligible']} control"
            for a, c in affected[:limit]
        )
        if len(affected) > limit:
            text += f"; and {len(affected) - limit} more event(s)"
        return text

    @staticmethod
    def _describe_time_grid_gaps(T_min: int, T_max: int, observed_periods: set) -> str:
        """Sentence naming integer periods the whole panel lacks (bounded cost)."""
        span = T_max - T_min + 1
        n_obs_periods = len(observed_periods)
        if span <= n_obs_periods:
            return ""
        advice = (
            " StackedDiD's event times are integer offsets of the time column, "
            "so re-index time to consecutive integers (for example the rank of "
            "each period) or narrow the window."
        )
        if span > 10_000:
            return (
                f" The time column spans {span} integer values but holds only "
                f"{n_obs_periods} distinct periods (a datetime or coded time "
                "column?)." + advice
            )
        missing = [t for t in range(T_min, T_max + 1) if t not in observed_periods]
        shown = missing[:12]
        more = f" (first 12 of {len(missing)})" if len(missing) > 12 else ""
        return (
            f" The panel's time grid has gaps: period(s) {shown}{more} are not "
            "observed for any unit." + advice
        )

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
            ic1 = self._window_fits(a_int, T_min, T_max)

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

    def _clean_control_mask(self, a: int, ft: np.ndarray) -> np.ndarray:
        """Units that are clean controls for adoption event ``a``.

        A later adopter responds from ``A_s - anticipation`` onward, so it is
        clean for event ``a`` only if that date falls after the event window
        closes at ``a + kappa_post``: ``A_s - anticipation > a + kappa_post``.
        ``"strict"`` also keeps the later adopter's own window, which starts at
        ``A_s - kappa_pre - anticipation``, disjoint from this one. With
        ``anticipation=0`` these are the rules of Wing et al. (2024),
        Appendix A. Never-treated units (``A_s = inf``) satisfy both.
        """
        return self._control_mask_at(a, ft, self.anticipation)

    def _control_mask_at(self, a: int, ft: np.ndarray, shift: int) -> np.ndarray:
        """The clean-control rule with the cutoff moved out by ``shift`` periods
        (``shift = anticipation`` is the rule in force; ``shift = 0`` is the
        paper's rule, which :meth:`_units_lost_to_anticipation_cutoff` compares
        against)."""
        if self.control_group == "not_yet_treated":
            return ft > a + self.kappa_post + shift
        if self.control_group == "strict":
            return ft > a + self.kappa_post + self.kappa_pre + shift
        return np.isinf(ft)  # never_treated

    def _units_lost_to_anticipation_cutoff(
        self, adoption_events: List[Any], T_min: int, T_max: int, unit_info: pd.DataFrame
    ) -> Dict[Any, int]:
        """Units the anticipation term keeps out of EVERY sub-experiment.

        Compares stack membership under the cutoff in force with membership
        under the same cutoff without its ``anticipation`` term: the treated
        units of every event that passes IC1 and has a clean control, plus
        those controls. Returns ``{cohort: number of units}`` for the units in
        the second set only (empty when ``anticipation == 0`` or under
        never-treated controls). Completeness of the event windows is not
        considered; a survey fit refuses incomplete windows separately.
        """
        if self.anticipation == 0 or self.control_group == "never_treated":
            return {}
        ft = unit_info["_first_treat"].values

        def members(shift: int) -> np.ndarray:
            inside = np.zeros(len(ft), dtype=bool)
            for a in adoption_events:
                a_int = int(a)
                if not self._window_fits(a_int, T_min, T_max):
                    continue
                controls = self._control_mask_at(a_int, ft, shift)
                if not controls.any():
                    continue
                inside |= controls | (ft == a)
            return inside

        lost = members(0) & ~members(self.anticipation)
        cohorts, counts = np.unique(ft[lost], return_counts=True)
        return {c: int(n) for c, n in zip(cohorts.tolist(), counts.tolist())}

    def _window_fits(self, a: int, T_min: int, T_max: int) -> bool:
        """IC1: the event window ``[a - kappa_pre - anticipation, a + kappa_post]``
        lies within the observed periods. (The paper text states a stricter
        lower bound, ``T_min + 1``; the co-author's R code uses ``T_min``.)"""
        return (a - self.kappa_pre - self.anticipation) >= T_min and (a + self.kappa_post) <= T_max

    def _check_clean_controls_exist(self, a: int, unit_info: pd.DataFrame) -> bool:
        """Check IC2: whether clean control units exist for adoption event a."""
        return bool(np.any(self._clean_control_mask(a, unit_info["_first_treat"].values)))

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
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, int]]:
        """
        Build a single sub-experiment for adoption event a.

        Only units observed at EVERY integer event time of the window
        ``[-kappa_pre - anticipation, kappa_post]`` are kept (the
        complete-window rule: the paper's stack has each unit observed for
        ``kappa_pre + kappa_post + 1`` periods, Section 5.1, and the
        regression's cell-mean contrasts are within-unit changes only then).

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
        sub_df : pd.DataFrame or None
            Sub-experiment data with _sub_exp, _event_time, _D_sa columns, or
            None when no treated unit or no clean control is complete.
        counts : dict
            ``treated_eligible`` / ``control_eligible`` (units eligible by
            ``first_treat``, with or without rows in the window) and
            ``treated_kept`` / ``control_kept`` (those observed at every
            event time).
        """
        a_int = int(a)
        ft = unit_info["_first_treat"].values
        unit_ids = unit_info[unit].values

        # Treated units: A_s = a
        treated_mask = ft == a
        treated_units = set(unit_ids[treated_mask])

        # Clean control units (same rule as the IC2 check)
        control_mask = self._clean_control_mask(a_int, ft)
        control_units = set(unit_ids[control_mask])

        counts = {
            "treated_eligible": len(treated_units),
            "control_eligible": len(control_units),
            "treated_kept": 0,
            "control_kept": 0,
        }
        if len(treated_units) == 0 or len(control_units) == 0:
            return None, counts

        # Time window: [a - kappa_pre - anticipation, a + kappa_post]
        # Reference period a-1 (event time e=-1) is included when kappa_pre >= 1
        # Matches R reference: (focalAdoptionTime - kappa_pre):(focalAdoptionTime + kappa_post)
        t_start = a_int - self.kappa_pre - self.anticipation
        t_end = a_int + self.kappa_post

        all_units = treated_units | control_units

        # Filter data
        mask = df[unit].isin(all_units) & (df[time] >= t_start) & (df[time] <= t_end)
        sub_df = df.loc[mask].copy()

        # Complete-window rule: keep a unit only if it has a row at every
        # integer event time of the window. Rows are unique per (unit, time)
        # and integer-valued (``_validate_panel_grid``), so a row count equal
        # to the window length is exact coverage. ``len(range(...))`` keeps a
        # non-integer kappa failing with range()'s TypeError.
        n_expected = len(range(t_start, t_end + 1))
        rows_per_unit = sub_df.groupby(unit).size()
        complete = set(rows_per_unit.index[rows_per_unit == n_expected])
        counts["treated_kept"] = len(treated_units & complete)
        counts["control_kept"] = len(control_units & complete)
        if counts["treated_kept"] == 0 or counts["control_kept"] == 0:
            return None, counts
        if len(complete) < len(rows_per_unit):
            sub_df = sub_df.loc[sub_df[unit].isin(complete)].copy()

        # Add sub-experiment columns
        sub_df["_sub_exp"] = a
        sub_df["_event_time"] = sub_df[time] - a_int
        sub_df["_D_sa"] = sub_df[unit].isin(treated_units).astype(int)

        return sub_df, counts

    # =========================================================================
    # Q-weight inputs
    # =========================================================================

    @staticmethod
    def _validate_treated_population(
        stacked_df: pd.DataFrame, unit_col: str, population_col: str
    ) -> None:
        """Validate the population of the treated units in the stack.

        ``weighting="population"`` reads it into both arms' weights (the
        paper's Table 1), so it must be finite, non-negative and constant
        within a unit. Control units, and treated units that never enter the
        stack, are not read and not validated.
        """
        rows = stacked_df.loc[stacked_df["_D_sa"] == 1, [unit_col, population_col]]
        values = pd.to_numeric(rows[population_col], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if np.any(bad):
            raise ValueError(
                f"Column '{population_col}' has {int(np.sum(bad))} missing or "
                "non-finite value(s) on treated units; weighting='population' "
                "needs a finite population for every treated unit."
            )
        if np.any(values < 0):
            raise ValueError(
                f"Column '{population_col}' has negative value(s) on treated "
                "units; weighting='population' needs non-negative populations."
            )
        varying = rows.groupby(unit_col)[population_col].nunique()
        if (varying > 1).any():
            raise ValueError(
                f"Column '{population_col}' varies within {int((varying > 1).sum())} "
                "treated unit(s); weighting='population' needs one population "
                "value per unit."
            )

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

        # Every sub-experiment in the stack is balanced by construction: fit()
        # rejects duplicate (unit, time) rows and _build_sub_experiment keeps
        # only units observed at every event time of the window. The paper's
        # unit-count corrective and the observation-count "aggregate" Q-weights
        # therefore coincide here, on ragged input panels too.
        for a in sub_exps:
            sub = stacked_df[stacked_df["_sub_exp"] == a]
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
