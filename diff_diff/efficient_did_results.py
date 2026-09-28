"""
Result container for the Efficient DiD estimator.

Follows the CallawaySantAnnaResults pattern: dataclass with summary(),
to_dataframe(), and significance properties.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.aggregation import AggregationMixin, AggregationResult, build_total_relay_row
from diff_diff.bootstrap_chunking import effective_weight_backend
from diff_diff.bootstrap_utils import (
    apply_bootstrap_event_study_overrides,
    apply_bootstrap_group_overrides,
)
from diff_diff.efficient_did_aggregation import _EfficientAggregationMixin
from diff_diff.efficient_did_bootstrap import EfficientDiDBootstrapMixin
from diff_diff.results import _format_survey_block, _get_significance_stars
from diff_diff.results_base import (
    BaseResults,
    _coverage_pct,
    _require_fit_alpha,
    build_event_study_surface,
)

if TYPE_CHECKING:
    from diff_diff.efficient_did_bootstrap import EDiDBootstrapResults


@dataclass
class HausmanPretestResult:
    """Result of Hausman pretest for PT-All vs PT-Post (Theorem A.1).

    Under H0 (PT-All holds), both estimators are consistent but PT-All
    is efficient.  Rejection suggests PT-All is too strong; use PT-Post.
    """

    statistic: float
    """Hausman H statistic."""
    p_value: float
    """Chi-squared p-value."""
    df: int
    """Degrees of freedom (effective rank of V)."""
    reject: bool
    """True if p_value < alpha."""
    alpha: float
    """Significance level used."""
    att_all: float
    """Overall ATT under PT-All."""
    att_post: float
    """Overall ATT under PT-Post."""
    recommendation: str
    """``"pt_all"`` if fail to reject, ``"pt_post"`` if reject, ``"inconclusive"`` if test unavailable."""
    gt_details: Optional[pd.DataFrame] = None
    """Per-event-study-horizon details: relative_period, es_all, es_post, delta."""

    def __repr__(self) -> str:
        return (
            f"HausmanPretestResult(H={self.statistic:.3f}, p={self.p_value:.4f}, "
            f"df={self.df}, recommend={self.recommendation})"
        )


class _EDiDKitAggregator(_EfficientAggregationMixin):
    """Throwaway host for post-fit EIF re-aggregation (M-023).

    Exposes exactly the five attributes the extracted
    ``_EfficientAggregationMixin`` methods read (its typed host contract):
    ``alpha``, ``anticipation``, ``_survey_df``, ``_unit_resolved_survey``,
    ``_unit_level_weights``.  A FRESH instance is built per ``aggregate()``
    call, which is what contains the one mutation the mixin performs -
    ``_compute_survey_eif_se`` writes ``self._survey_df`` when a degenerate
    replicate design drops replicates - on the throwaway rather than the
    retained kit, preserving the aggregate() immutability contract.  This
    is also what keeps ``aggregate()`` off an ``_estimator_ref`` (the CS
    ``_KitAggregator`` precedent).
    """

    def __init__(
        self,
        alpha: float,
        anticipation: int,
        survey_df: Optional[float],
        resolved_survey_unit: Optional[Any],
        unit_level_weights: Optional["np.ndarray"],
    ) -> None:
        self.alpha = alpha
        self.anticipation = anticipation
        self._survey_df = survey_df
        self._unit_resolved_survey = resolved_survey_unit
        self._unit_level_weights = unit_level_weights


class _EDiDKitBootstrapAggregator(EfficientDiDBootstrapMixin):
    """Value-bound host that replays the fit-time multiplier bootstrap.

    ``_run_multiplier_bootstrap`` reads exactly five attributes off its
    host (the mixin's typed contract): ``n_bootstrap``,
    ``bootstrap_weights``, ``alpha``, ``seed``, ``anticipation`` — all
    carried here BY VALUE from the kit/spec so post-fit ``set_params`` or
    attribute mutation on the estimator can never desynchronize a replay.
    ``seed`` is None because the replay injects the fit-captured
    bit-generator state directly. Warning attribution note: the engine's
    fit-tuned stacklevels resolve into library frames under the deeper
    ``aggregate()`` chain — accepted as cosmetic (the CS-recorded
    decision).
    """

    def __init__(
        self,
        alpha: float,
        anticipation: int,
        n_bootstrap: int,
        bootstrap_weights: str,
    ) -> None:
        self.alpha = alpha
        self.anticipation = anticipation
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = None  # unused — the replay injects the captured state


@dataclass
class EfficientDiDResults(BaseResults, AggregationMixin):
    """
    Results from Efficient DiD (Chen, Sant'Anna & Xie 2025) estimation.

    Stores group-time ATT(g,t) estimates with efficient weights, plus
    optional aggregations (overall ATT, event study, group effects).

    Attributes
    ----------
    group_time_effects : dict
        ``{(g, t): {'effect', 'se', 't_stat', 'p_value', 'conf_int',
        'n_treated', 'n_control'}}``
    overall_att : float
        Overall ATT (cohort-size weighted average of post-treatment
        group-time effects, matching CallawaySantAnna convention).
    overall_se : float
        Standard error of overall ATT.
    overall_t_stat : float
        t-statistic for overall ATT.
    overall_p_value : float
        p-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    groups : list
        Treatment cohort identifiers.
    time_periods : list
        All time periods.
    n_obs : int
        Total observations (units x periods).
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of never-treated units.
    alpha : float
        Significance level.
    pt_assumption : str
        ``"all"`` or ``"post"``.
    anticipation : int
        Number of anticipation periods used.
    n_bootstrap : int
        Number of bootstrap iterations (0 = analytical only).
    bootstrap_weights : str
        Bootstrap weight distribution (``"rademacher"``, ``"mammen"``, ``"webb"``).
    seed : int or None
        Random seed used for bootstrap.
    event_study_effects : dict, optional
        ``{relative_time: effect_dict}``
    group_effects : dict, optional
        ``{group: effect_dict}``
    efficient_weights : dict, optional
        ``{(g, t): ndarray}`` — diagnostic: weight vector per target.
    omega_condition_numbers : dict, optional
        ``{(g, t): float}`` — diagnostic: Omega* condition numbers.
    cluster_name : str or None
        Cluster column used at fit time (None for unclustered fits;
        suppressed under any survey design). Populated when ``cluster=``
        is passed to :meth:`~EfficientDiD.fit`.
    n_clusters : int or None
        Number of clusters at fit time (None for unclustered or survey
        fits). Renders as ``G=<n>`` in the variance-estimator summary line.
    vcov_type : str
        Variance-estimator family. Permanently ``"hc1"`` per the
        Chen-Sant'Anna-Xie (2025) IF-based variance; see REGISTRY.md.
    influence_functions : dict, optional
        ``{(g, t): ndarray(n_units,)}`` — per-unit EIF values for each
        group-time cell.  Only populated when ``store_eif=True`` in
        :meth:`~EfficientDiD.fit` (used internally by ``hausman_pretest``).
        Since 3.9 (row M-023) the private aggregation kit ALWAYS retains
        the same per-(g,t) EIF dict to power post-fit
        :meth:`aggregate` — ``store_eif`` governs only this public field.
    bootstrap_results : EDiDBootstrapResults, optional
        Bootstrap inference results.
    estimation_path : str
        ``"nocov"`` or ``"dr"`` — which estimation path was used.
    sieve_k_max : int or None
        Maximum polynomial degree for the covariate-path sieves (propensity
        ratio, inverse propensity, and outcome regression); ``1`` forces a
        linear outcome-regression working model.
    sieve_criterion : str
        Information criterion used (``"aic"`` or ``"bic"``) for all
        covariate-path sieve order selection.
    ratio_clip : float
        Clipping bound for sieve propensity ratios.
    kernel_bandwidth : float or None
        Bandwidth used for kernel-smoothed conditional Omega*.
    omega_ridge : float
        Relative ridge used for the Omega* inversion behind the efficient
        weights (0 = legacy exact-inverse/pseudoinverse path).
    event_study_df : float or None
        Scalar df governing the event-study rows' t-inference (the fit's
        survey df). ``None`` on non-survey fits, on bootstrapped fits
        (percentile inference used no df), and when no fit-time
        event-study surface was built.
    """

    group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]]
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_units: int
    n_control_units: int
    alpha: float = 0.05
    pt_assumption: str = "all"
    anticipation: int = 0
    n_bootstrap: int = 0
    bootstrap_weights: str = "rademacher"
    seed: Optional[int] = None
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None)
    group_effects: Optional[Dict[Any, Dict[str, Any]]] = field(default=None)
    efficient_weights: Optional[Dict[Tuple[Any, Any], "np.ndarray"]] = field(
        default=None, repr=False
    )
    omega_condition_numbers: Optional[Dict[Tuple[Any, Any], float]] = field(
        default=None, repr=False
    )
    control_group: str = "never_treated"
    # Cluster column used at fit time (None for unclustered fits, suppressed
    # under survey designs). Persisted so downstream diagnostics — notably
    # ``DiagnosticReport._pt_hausman`` — can replay the Hausman PT-All vs
    # PT-Post pretest under the same clustering as the original estimate
    # rather than silently producing unclustered p-values for a clustered fit.
    cluster_name: Optional[str] = None
    # Number of clusters at fit time (None for unclustered fits, suppressed
    # under survey designs). Used by the shared ``_format_vcov_label`` helper
    # to render the ``G=<n>`` suffix on the variance-estimator summary line.
    n_clusters: Optional[int] = None
    # Variance-estimator family. Permanently narrow to ``{"hc1"}`` per the
    # Chen-Sant'Anna-Xie (2025) EIF-based variance — see REGISTRY.md.
    vcov_type: str = "hc1"
    influence_functions: Optional[Dict[Tuple[Any, Any], "np.ndarray"]] = field(
        default=None, repr=False
    )
    bootstrap_results: Optional["EDiDBootstrapResults"] = field(default=None, repr=False)
    estimation_path: str = "nocov"
    sieve_k_max: Optional[int] = None
    sieve_criterion: str = "bic"
    ratio_clip: float = 20.0
    kernel_bandwidth: Optional[float] = None
    omega_ridge: float = 0.0
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)
    # Post-fit aggregation kit (M-023). The generated __init__'s positional
    # indexes are public API (CS precedent), so new fields are appended
    # AFTER this one, never before.
    _aggregation_kit: Optional[Any] = field(default=None, repr=False, compare=False)
    # Scalar df governing the event-study rows' t-inference (the fit's
    # survey df; None on non-survey fits, on bootstrapped fits — percentile
    # inference used no df — and when no event-study surface was built).
    # Appended last per the positional-__init__ convention above.
    event_study_df: Optional[float] = None

    # Post-fit aggregate() hooks (M-023). Plain class attributes (no
    # annotation) so they never enter dataclasses.fields; the mixin's
    # ClassVar-annotated defaults document the contract. balance_e keeps
    # the mixin default ("event_study",) - CS precedent, do not redeclare.
    _AGGREGATE_SUPPORTED = ("simple", "event_study", "group", "total")

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

    # --- Post-fit aggregation (M-023) -----------------------------------

    @property
    def reference_period(self) -> Optional[int]:
        """Reference event time of the materialized PT-Post anchor, or None.

        MEMBERSHIP-GATED (the SunAbraham rule): returns
        ``-1 - anticipation`` only when this fit ran under
        ``pt_assumption="post"`` AND that event time is actually present in
        ``event_study_effects`` — under PT-Post the per-cohort baseline
        cell ``(g, g-1-anticipation)`` is estimated as a mechanical zero
        whenever it is not the panel's first period, and the builder marks
        that materialized row ``is_reference``.  When the anchor cell was
        never estimated (every surviving cohort baselined at the first
        period) the property is None, so no reference row is ever
        synthesized.  Under ``pt_assumption="all"`` there is no reference
        row (universal first-period baseline; every row is a genuine
        estimate) and the property is None.
        """
        if self.pt_assumption != "post":
            return None
        if not self.event_study_effects:
            return None
        ref = -1 - int(self.anticipation)
        return ref if ref in self.event_study_effects else None

    def _aggregate_compute(
        self, level: str, *, weights: Optional[str], balance_e: Optional[int]
    ) -> Any:
        kit = self._aggregation_kit
        if kit is None:
            raise ValueError(
                "This EfficientDiDResults carries no aggregation kit - it is "
                "attached by EfficientDiD.fit(), so a result unpickled from "
                "an older release will not have one. Re-fit with "
                "diff-diff >= 3.9 to aggregate post-fit."
            )
        # Per-level bootstrap policy (v4-design section 6, converged with row
        # M-027): 'simple' is a bit-exact RELAY of the stored overall row -
        # faithful under any inference regime, bootstrap included - so it
        # dispatches BEFORE the bootstrap branch. On bootstrapped fits the
        # RECOMPUTE levels below REPLAY the fit-time multiplier bootstrap
        # from the kit's BootstrapReplaySpec state (percentile inference,
        # allclose to a fit-time aggregation); the M-023 rationale - never
        # publish analytical provenance beside percentile inference - is
        # honored by the percentile overrides + the NaN df channels.
        if level == "simple":
            return self._aggregate_simple_result(kit)
        if level == "total":
            return self._aggregate_total_result(kit)
        boot_replay = None
        if self.bootstrap_results is not None:
            spec = getattr(kit, "bootstrap", None)
            if spec is None or spec.bitgen_state is None:
                raise NotImplementedError(
                    f"aggregate({level!r}): this bootstrapped result predates "
                    "the fit-time bootstrap replay state (its kit carries no "
                    "BootstrapReplaySpec) - refit with diff-diff >= 3.10 to "
                    "enable the post-fit replay. aggregate('simple') and, "
                    "where supported, aggregate('total') relay the stored "
                    "bootstrap inference and remain available."
                )
            current_backend = effective_weight_backend()
            if spec.backend not in ("portable", current_backend):
                raise NotImplementedError(
                    f"aggregate({level!r}): this fit's bootstrap weights were "
                    f"generated under the {spec.backend!r} weight backend, but "
                    f"the current install uses {current_backend!r} - the two "
                    "backends produce DIFFERENT draws from the same RNG "
                    "state, so replaying here would publish a different "
                    "bootstrap realization beside the stored fit-time "
                    "inference. Re-fit under the current backend, or restore "
                    "the original one (DIFF_DIFF_BACKEND / the Rust "
                    "extension). aggregate('simple') and, where supported, "
                    "aggregate('total') relay the stored inference."
                )
        bk = dict(kit.bookkeeping)
        if self.bootstrap_results is not None:
            spec = kit.bootstrap
            host = _EDiDKitBootstrapAggregator(
                alpha=kit.alpha,
                anticipation=kit.anticipation,
                n_bootstrap=spec.n_bootstrap,
                bootstrap_weights=spec.weight_type,
            )
            # Per-call cost: O(n_bootstrap x n_units x n_gt) - the fused GEMM
            # carries the n_gt per-cell EIF columns; ES/group targets
            # re-aggregate cheaply from the dense (n_bootstrap, n_gt) matrix.
            # No memoization (the recompute convention; the kit deliberately
            # retains no draws). The engine re-derives the generation branch
            # from the kit bookkeeping and re-emits the fit-time bootstrap
            # warnings for the replayed configuration.
            boot_replay = host._run_multiplier_bootstrap(
                group_time_effects=bk["group_time_effects"],
                eif_by_gt=kit.influence,
                n_units=bk["n_units"],
                aggregate=level,
                balance_e=(balance_e if level == "event_study" else None),
                treatment_groups=bk["treatment_groups"],
                cohort_fractions=bk["cohort_fractions"],
                cluster_indices=bk["cluster_indices"],
                n_clusters=bk["n_clusters"],
                resolved_survey=bk["resolved_survey_unit"],
                unit_level_weights=bk["unit_level_weights"],
                _replay_bitgen_state=spec.bitgen_state,
            )
        agg = _EDiDKitAggregator(
            alpha=kit.alpha,
            anticipation=kit.anticipation,
            survey_df=bk["df_survey"],
            resolved_survey_unit=bk["resolved_survey_unit"],
            unit_level_weights=bk["unit_level_weights"],
        )
        if level == "group":
            effects = agg._aggregate_by_group(
                bk["group_time_effects"],
                kit.influence,
                bk["n_units"],
                bk["cohort_fractions"],
                bk["treatment_groups"],
                unit_cohorts=bk["unit_cohorts"],
                cluster_indices=bk["cluster_indices"],
                n_clusters=bk["n_clusters"],
            )
            if boot_replay is not None:
                # Same applier as fit-time (clears each row's df_used, so
                # _group_effects_to_aggregation publishes an all-NaN df).
                apply_bootstrap_group_overrides(effects, boot_replay, kit.alpha)
            return self._group_effects_to_aggregation(effects, kit)
        # level == "event_study" (the mixin validated the vocabulary)
        es = agg._aggregate_event_study(
            bk["group_time_effects"],
            kit.influence,
            bk["n_units"],
            bk["cohort_fractions"],
            bk["treatment_groups"],
            bk["time_periods"],
            balance_e,
            unit_cohorts=bk["unit_cohorts"],
            cluster_indices=bk["cluster_indices"],
            n_clusters=bk["n_clusters"],
        )
        if boot_replay is not None:
            # Same applier as fit-time. The carrier clears only
            # event_study_df below (this class has no vcov/cband ES
            # fields), and the non-None bootstrap_results it retains keeps
            # the container's inference provenance honest.
            apply_bootstrap_event_study_overrides(es, boot_replay, kit.alpha)
        # Carrier + shared builder: EDiD is a _from_relative_dict producer,
        # so the recomputed dict rides the same route as the fit-time
        # surface. The carrier's survey_metadata is a COPY whose df_survey
        # is the kit's post-overall snapshot - _recompute_unit_survey_
        # metadata's is-not-None guard can leave the raw metadata at the
        # resolved design's finite value when the governing df degenerated
        # to None (n_valid <= 1), and the copy keeps the container honest
        # there (in non-degenerate fits the two agree and the copy is
        # inert). Copying also keeps the parent's metadata unmutated.
        meta = self.survey_metadata
        if meta is not None:
            meta = dataclasses.replace(meta, df_survey=bk["df_survey"])
        # The carrier's PROVENANCE fields also come from the kit snapshots
        # (CI review R2): the reference_period property and the container's
        # alpha/anticipation must reflect the FIT's regime, not a possibly
        # mutated public field - a PT-All fit whose public pt_assumption
        # was flipped to "post" would otherwise mark the genuine e=-1
        # estimate as a reference row and zero it.
        # Per-row df provenance: the kit's post-overall df snapshot on
        # analytical recomputes; cleared on bootstrap replays (percentile
        # inference used no df) and for the replicate-undefined 0 sentinel.
        _es_df = bk["df_survey"]
        es_df_carrier = (
            float(_es_df) if (boot_replay is None and _es_df is not None and _es_df > 0) else None
        )
        carrier = dataclasses.replace(
            self,
            event_study_effects=es,
            survey_metadata=meta,
            pt_assumption=bk["pt_assumption"],
            anticipation=kit.anticipation,
            alpha=kit.alpha,
            event_study_df=es_df_carrier,
        )
        return build_event_study_surface(carrier)

    def _aggregate_simple_result(self, kit: Any) -> AggregationResult:
        """One-row relay of the stored overall inference (bit-exact).

        ``n = n_treated_units + n_control_units`` with ``n_kind="units"``:
        EDiD's treated and control unit sets are DISJOINT by construction
        (``last_cohort`` trimming reassigns the last cohort to control
        BEFORE the counts), so a true disjoint total exists - the CS
        convention applies, unlike StackedDiD's overlapping-sets carve-out.

        ``df`` is the kit's post-overall ``df_survey`` snapshot - the very
        value fit passed to ``safe_inference`` for the overall row - NOT
        ``resolve_inference_df(self)``: ``survey_metadata.df_survey`` can
        diverge from the governing df in the degenerate replicate case
        (``n_valid <= 1`` sets the working df to None while the metadata
        keeps the resolved design's finite value), and the snapshot is
        provenance-exact in every state. None → all-NaN df column;
        the replicate-undefined 0-sentinel row NaNs out via post_init.
        Bootstrapped fits relay the stored quintet verbatim (percentile
        se/p/CI beside the finite ``safe_inference`` t) with a NaN df
        column - no df governs percentile inference (the per-level policy
        converged with row M-027).
        """
        df_val = np.nan if self.bootstrap_results is not None else kit.bookkeeping["df_survey"]
        return AggregationResult(
            level="simple",
            label=np.array(["overall"], dtype=object),
            target=np.array(["att"], dtype=object),
            att=np.array([self.overall_att], dtype=float),
            se=np.array([self.overall_se], dtype=float),
            t_stat=np.array([self.overall_t_stat], dtype=float),
            p_value=np.array([self.overall_p_value], dtype=float),
            conf_int_lower=np.array([self.overall_conf_int[0]], dtype=float),
            conf_int_upper=np.array([self.overall_conf_int[1]], dtype=float),
            n=np.array([kit.bookkeeping["n_units_total"]], dtype=float),
            df=df_val,
            alpha=kit.alpha,
            n_kind="units",
            weight=np.array([1.0], dtype=float),
            estimator=type(self).__name__.replace("Results", ""),
        )

    def _aggregate_total_result(self, kit: Any) -> AggregationResult:
        """The estimator-owned total incremental outcome as a one-row table.

        Exact relay ``C x overall`` CONDITIONAL on the realized aggregation
        mass, with ``C = sum(n_treated)`` over the KEPT (g, t) cells of the
        kit's deep-copied ``group_time_effects`` snapshot - keepers are the
        post-anticipation, finite-effect cells, mirroring the simple
        aggregation's ``cohort_fractions`` support. The integer per-cell
        ``n_treated`` sum is used, NEVER the float
        ``n_units x sum(cohort_fractions)`` product (routinely non-integral
        by roundoff). No keepers -> NaN mass -> all-NaN row. Fails closed on
        fits declaring a ``survey_design=`` (see the REGISTRY Note).
        """
        bk = kit.bookkeeping
        # Survey gate, from the kit's own markers: EDiD never synthesizes an
        # internal design, so a non-None marker means the fit DECLARED a
        # survey_design (weight-type-agnostic: unweighted psu-only and
        # analytic fweight designs populate these too). There is NO
        # "survey_weights" key in this kit.
        if bk.get("unit_level_weights") is not None or bk.get("resolved_survey_unit") is not None:
            raise NotImplementedError(
                "aggregate('total') is not available on fits declaring a "
                "survey_design: the realized-mass relay omits the survey "
                "mass-uncertainty variance term, and design-aware "
                "population-scale totals are not implemented (retained "
                "weight scale differs by design family) - tracked in "
                "DEFERRED.md. For an unweighted clustered fit, cluster= "
                "(without survey_design) supports totals."
            )
        support = 0.0
        n_keepers = 0
        for (g, t), cell in bk["group_time_effects"].items():
            if t < g - kit.anticipation:
                continue
            if not np.isfinite(float(cell["effect"])):
                continue
            support += float(cell["n_treated"])
            n_keepers += 1
        mass = support if n_keepers > 0 else float(np.nan)
        return build_total_relay_row(
            mass=mass,
            att=self.overall_att,
            se=self.overall_se,
            t_stat=self.overall_t_stat,
            p_value=self.overall_p_value,
            conf_int=self.overall_conf_int,
            df=(np.nan if self.bootstrap_results is not None else bk["df_survey"]),
            alpha=kit.alpha,
            estimator=type(self).__name__.replace("Results", ""),
        )

    def _group_effects_to_aggregation(
        self, effects: Dict[Any, Dict[str, Any]], kit: Any
    ) -> AggregationResult:
        """Per-cohort AggregationResult from the recomputed group dict.

        ``df`` relays the PER-ROW ``df_used`` array the aggregation
        recorded at each row's ``safe_inference`` call - exact by
        construction (in every constructible fit all rows share one value;
        capture-at-use is robust regardless). ``weight=None``: rows use
        equal within-cohort weights and carry no cross-cohort mass, so a
        weight column would fabricate one (the CS rationale).
        ``n_kind="cells"``: ``n_periods`` counts contributing (g,t) cells.
        """
        labels = list(effects.keys())
        df_arr = np.array(
            [
                (np.nan if effects[g].get("df_used") is None else float(effects[g]["df_used"]))
                for g in labels
            ],
            dtype=float,
        )
        return AggregationResult(
            level="group",
            label=np.array(labels, dtype=object),
            target=np.array(["att"] * len(labels), dtype=object),
            att=np.array([effects[g]["effect"] for g in labels], dtype=float),
            se=np.array([effects[g]["se"] for g in labels], dtype=float),
            t_stat=np.array([effects[g]["t_stat"] for g in labels], dtype=float),
            p_value=np.array([effects[g]["p_value"] for g in labels], dtype=float),
            conf_int_lower=np.array([effects[g]["conf_int"][0] for g in labels], dtype=float),
            conf_int_upper=np.array([effects[g]["conf_int"][1] for g in labels], dtype=float),
            n=np.array([effects[g]["n_periods"] for g in labels], dtype=float),
            df=df_arr,
            alpha=kit.alpha,
            n_kind="cells",
            weight=None,
            estimator=type(self).__name__.replace("Results", ""),
        )

    def __repr__(self) -> str:
        sig = _get_significance_stars(self.overall_p_value)
        path = "DR" if self.estimation_path == "dr" else "nocov"
        return (
            f"EfficientDiDResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"pt={self.pt_assumption}, path={path}, "
            f"n_groups={len(self.groups)}, "
            f"n_periods={len(self.time_periods)})"
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
        """Generate formatted summary of estimation results.

        ``alpha`` is accepted for signature uniformity; a value different
        from the fit-time ``alpha`` raises ValueError (stored inference is
        never recomputed or relabeled - re-fit at the desired alpha).
        """
        alpha = _require_fit_alpha(alpha, self.alpha)
        conf_level = _coverage_pct(alpha)

        lines = [
            "=" * 85,
            "Efficient DiD (Chen-Sant'Anna-Xie 2025) Results".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            f"{'PT assumption:':<30} {self.pt_assumption:>10}",
            f"{'Estimation path:':<30} {'doubly robust' if self.estimation_path == 'dr' else 'no covariates':>10}",
        ]
        if self.control_group != "never_treated":
            lines.append(f"{'Control group:':<30} {self.control_group:>10}")
        if self.anticipation > 0:
            lines.append(f"{'Anticipation periods:':<30} {self.anticipation:>10}")
        # Suppress the legacy ``Bootstrap:`` header when ``bootstrap_results``
        # is present — the new variance/inference-method block below renders
        # the canonical ``Inference method: bootstrap`` + ``Bootstrap
        # replications:`` lines, so the old header would duplicate metadata.
        if self.n_bootstrap > 0 and self.bootstrap_results is None:
            lines.append(f"{'Bootstrap:':<30} {self.n_bootstrap:>10} ({self.bootstrap_weights})")
        lines.append("")

        # Add survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 85))

        # Variance-estimator / inference-method line. Bootstrap takes precedence
        # over analytical because ``bootstrap_results`` overwrites SE/CI/p-value
        # downstream; the analytical HC1/CR1 label would mislabel the reported
        # numbers. Survey-fit summary block already covers TSL/replicate metadata
        # so the variance line is suppressed under ``survey_metadata is not None``.
        if self.bootstrap_results is not None:
            lines.append(f"{'Inference method:':<30} {'bootstrap':>15}")
            lines.append(
                f"{'Bootstrap replications:':<30} {self.bootstrap_results.n_bootstrap:>15}"
            )
        elif self.survey_metadata is None:
            from diff_diff.results import _format_vcov_label

            vcov_label = _format_vcov_label(
                self.vcov_type,
                cluster_name=self.cluster_name,
                n_clusters=self.n_clusters,
                n_obs=self.n_obs,
            )
            if vcov_label:
                lines.append(f"{'Variance estimator:':<30} {vcov_label:>15}")
        if self.n_clusters is not None and self.bootstrap_results is None:
            lines.append(f"{'Number of clusters:':<30} {self.n_clusters:>15}")

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
        if self.event_study_effects:
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

        # Group effects
        if self.group_effects:
            lines.extend(
                [
                    "-" * 85,
                    "Effects by Treatment Cohort".center(85),
                    "-" * 85,
                    f"{'Cohort':<15} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                    "-" * 85,
                ]
            )
            for group in sorted(self.group_effects.keys()):
                eff = self.group_effects[group]
                sig = _get_significance_stars(eff["p_value"])
                lines.append(
                    f"{group:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
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

    def to_dataframe(self, level: str = "group_time") -> pd.DataFrame:
        """Convert results to DataFrame.

        Parameters
        ----------
        level : str
            ``"group_time"``, ``"event_study"``, or ``"group"``.
        """
        if level == "group_time":
            rows = []
            for (g, t), data in self.group_time_effects.items():
                rows.append(
                    {
                        "group": g,
                        "time": t,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                    }
                )
            return pd.DataFrame(rows)

        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError(
                    "Event study effects not computed at fit time. Use "
                    "results.aggregate('event_study') for the post-fit "
                    "event-study container (bootstrapped fits replay the "
                    "fit-time multiplier bootstrap - no refit needed); a "
                    "result unpickled from a pre-3.9 release carries no "
                    "aggregation kit and must be refit."
                )
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

        elif level == "group":
            if self.group_effects is None:
                raise ValueError(
                    "Group effects not computed at fit time. Use "
                    "results.aggregate('group') for the post-fit group "
                    "container (bootstrapped fits replay the fit-time "
                    "multiplier bootstrap - no refit needed); a result "
                    "unpickled from a pre-3.9 release carries no "
                    "aggregation kit and must be refit."
                )
            rows = []
            for group, data in sorted(self.group_effects.items()):
                rows.append(
                    {
                        "group": group,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                    }
                )
            return pd.DataFrame(rows)

        else:
            raise ValueError(
                f"Unknown level: {level}. " "Use 'group_time', 'event_study', or 'group'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert headline results to a flat dictionary.

        Mirrors :meth:`TripleDifferenceResults.to_dict` and
        :meth:`ImputationDiDResults.to_dict` — surfaces variance metadata
        (``vcov_type``, ``cluster_name``, ``n_clusters``, ``n_bootstrap``,
        ``inference_method``) for external adapters that don't render
        the full summary.
        """
        result: Dict[str, Any] = {
            "att": self.overall_att,
            "se": self.overall_se,
            "t_stat": self.overall_t_stat,
            "p_value": self.overall_p_value,
            "conf_int_lower": self.overall_conf_int[0],
            "conf_int_upper": self.overall_conf_int[1],
            "n_obs": self.n_obs,
            "n_treated_units": self.n_treated_units,
            "n_control_units": self.n_control_units,
            "alpha": self.alpha,
            "pt_assumption": self.pt_assumption,
            "estimation_path": self.estimation_path,
            "vcov_type": self.vcov_type,
        }
        if self.cluster_name is not None:
            result["cluster_name"] = self.cluster_name
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        if self.bootstrap_results is not None:
            # Mirror summary() gate exactly — bootstrap_results can be None
            # even when n_bootstrap > 0 (e.g. empty eif_by_gt skips the path).
            result["n_bootstrap"] = self.bootstrap_results.n_bootstrap
            result["inference_method"] = "bootstrap"
        elif self.survey_metadata is not None:
            result["inference_method"] = "survey"
        elif self.cluster_name is not None:
            result["inference_method"] = "cluster_robust"
        else:
            result["inference_method"] = "heteroskedasticity_robust"
        return result

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
