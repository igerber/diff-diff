"""
Result container classes for Callaway-Sant'Anna estimator.

This module provides dataclass containers for storing and presenting
group-time average treatment effects and their aggregations.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.aggregation import (
    AggregationMixin,
    AggregationResult,
    build_total_relay_row,
    resolve_inference_df,
)
from diff_diff.bootstrap_chunking import effective_weight_backend
from diff_diff.bootstrap_utils import (
    apply_bootstrap_event_study_overrides,
    apply_bootstrap_group_overrides,
)
from diff_diff.results import _format_survey_block, _get_significance_stars
from diff_diff.results_base import BaseResults, build_event_study_surface
from diff_diff.staggered_aggregation import (
    CallawaySantAnnaAggregationMixin,
    fixed_cohort_agg_weights,
)
from diff_diff.staggered_bootstrap import (
    CallawaySantAnnaBootstrapMixin,
    CSBootstrapResults,
    apply_cband_conf_ints,
)


class _KitAggregator(CallawaySantAnnaAggregationMixin):
    """Runs the CallawaySantAnna aggregation mixin off a retained kit.

    The aggregation methods are ESTIMATOR-bound but read exactly two attributes
    from their host - ``alpha`` and ``anticipation`` - so a post-fit caller
    needs neither the estimator nor a reference to it, only those two values.
    (Verified by enumerating every ``self.<attr>`` access in
    ``staggered_aggregation.py``: the rest are internal method calls.)

    This is what keeps ``aggregate()`` off an ``_estimator_ref``, which would
    otherwise drag the whole fitted estimator - and its source frame - onto
    every results object.
    """

    def __init__(self, alpha: float, anticipation: int) -> None:
        self.alpha = alpha
        self.anticipation = anticipation


class _KitBootstrapAggregator(CallawaySantAnnaBootstrapMixin, CallawaySantAnnaAggregationMixin):
    """Runs the fit-time multiplier bootstrap off a retained kit (replay).

    Value-bound host for the post-fit bootstrap replay: it carries BY VALUE
    everything ``_run_multiplier_bootstrap`` reads from its estimator host
    (base order matches ``CallawaySantAnna`` — bootstrap mixin first), so the
    replay is immune to post-fit ``set_params``/attribute mutation of the
    estimator. The RNG is injected via ``_replay_bitgen_state`` rather than
    seeded from ``self.seed``. Warning attribution note: the engine's
    fit-tuned stacklevels resolve into library frames on this deeper
    ``aggregate()`` chain (no ``_warn_frame_offset`` is set) — accepted as
    cosmetic; the warnings themselves are accurate statements about the
    replayed bootstrap.
    """

    _BOOTSTRAP_LABEL = "CallawaySantAnna"

    def __init__(
        self, alpha: float, anticipation: int, n_bootstrap: int, bootstrap_weights: str
    ) -> None:
        self.alpha = alpha
        self.anticipation = anticipation
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = None  # unused — the replay injects the captured state


@dataclass
class GroupTimeEffect:
    """
    Treatment effect for a specific group-time combination.

    Attributes
    ----------
    group : any
        The treatment cohort (first treatment period).
    time : any
        The time period.
    effect : float
        The ATT(g,t) estimate.
    se : float
        Standard error.
    n_treated : int
        Number of treated observations.
    n_control : int
        Number of control observations.
    skip_reason : str or None
        ``None`` for an estimable cell; otherwise a machine-readable reason the
        cell is non-estimable (``"missing_period"``, ``"zero_treated_control"``,
        ``"zero_weight_mass"``, ``"non_finite_regression"``, or — DMLDiD
        cells only — ``"cross_fit_degenerate"``, ``"non_finite_score"``) and
        ``effect``/``se`` are NaN. Non-estimable cells are excluded from all
        aggregation.
    """

    group: Any
    time: Any
    effect: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_treated: int
    n_control: int
    skip_reason: Optional[str] = None

    @property
    def is_significant(self) -> bool:
        """Check if effect is significant at 0.05 level."""
        return bool(self.p_value < 0.05)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)


@dataclass
class CallawaySantAnnaResults(BaseResults, AggregationMixin):
    """
    Results from Callaway-Sant'Anna (2021) staggered DiD estimation.

    This class stores group-time average treatment effects ATT(g,t) and
    provides methods for aggregation into summary measures.

    Attributes
    ----------
    group_time_effects : dict
        Dictionary mapping (group, time) tuples to effect dictionaries.
    overall_att : float
        Overall average treatment effect (weighted average of ATT(g,t)).
    overall_se : float
        Standard error of overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    groups : list
        List of treatment cohorts (first treatment periods).
    time_periods : list
        List of all time periods.
    n_obs : int
        Total number of observations.
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of never-treated units (excludes not-yet-treated dynamic controls).
    event_study_effects : dict, optional
        Effects aggregated by relative time (event study).
    group_effects : dict, optional
        Effects aggregated by treatment cohort.
    pscore_trim : float
        Propensity score trimming bound used during estimation.
    vcov_type : str
        Variance type used during estimation. CallawaySantAnna is
        permanently narrow to ``"hc1"`` — see REGISTRY.md
        "IF-based variance estimators vs analytical-sandwich estimators"
        for why analytical-sandwich families don't compose with the
        per-(g,t) doubly-robust / IPW / outcome-regression structure.
    cluster_name : str, optional
        Canonical cluster column. Set to ``survey_design.psu`` when an
        explicit survey PSU was provided (regardless of bare ``cluster=``),
        otherwise to ``self.cluster`` when bare cluster synthesizes or
        injects a PSU. ``None`` when no clustering is active.
    n_clusters : int, optional
        Number of unique clusters (PSUs) used for variance estimation.
        ``None`` when no clustering is active.
    df_inference : int, optional
        Cluster-level degrees of freedom for downstream inference (e.g.,
        ``HonestDiD`` t-critical-value selection) on the bare-``cluster=``
        synthesize path ONLY (the case where ``survey_metadata`` is
        intentionally ``None`` to preserve the survey/non-survey contract
        for ``DiagnosticReport`` / ``summary()``). When the user provides
        an explicit ``survey_design=`` (inject or conflict branches),
        ``df_inference`` stays ``None`` and the canonical df carrier is
        ``survey_metadata.df_survey`` — which holds the actual CS-internal
        df, including any post-resolve tightening (e.g., the
        ``overall_effective_df`` recompute for replicate aggregations).
        ``HonestDiD`` reads ``survey_metadata.df_survey`` first and falls
        back to ``df_inference`` only when ``survey_metadata`` is absent.
        Narrow contract prevents HonestDiD from silently overriding a
        tightened survey df with the original ``resolved_survey.df_survey``.
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
    control_group: str = "never_treated"
    base_period: str = "varying"
    # Anticipation periods (``k``) used at fit time. Persisted on the
    # result so downstream diagnostics (``BusinessReport`` /
    # ``DiagnosticReport`` / ``compute_pretrends_power``) can classify
    # pre-period vs anticipation-window coefficients without re-
    # plumbing the kwarg through every call site. See REGISTRY.md
    # §CallawaySantAnna lines 355-395 for the shifted-boundary
    # contract.
    anticipation: int = 0
    panel: bool = True
    # allow_unbalanced_panel routing (RC-on-panel). `allow_unbalanced_panel`
    # records the fit-time flag; `used_rc_on_unbalanced_panel` is True ONLY when
    # the panel was actually unbalanced and the RC (repeated-cross-section)
    # levels estimator was used (the flag is inert on a balanced panel). When
    # True the ATT(g,t) estimand is R's `allow_unbalanced_panel=TRUE` RC-on-panel
    # estimand, not within-cell panel differencing — and the influence function
    # is clustered by unit (see `cluster_name`).
    allow_unbalanced_panel: bool = False
    used_rc_on_unbalanced_panel: bool = False
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None)
    group_effects: Optional[Dict[Any, Dict[str, Any]]] = field(default=None)
    influence_functions: Optional["np.ndarray"] = field(default=None, repr=False)
    # Full event-study VCV matrix (Phase 7d): indexed by event_study_vcov_index
    event_study_vcov: Optional["np.ndarray"] = field(default=None, repr=False)
    event_study_vcov_index: Optional[list] = field(default=None, repr=False)
    # event_study_df (spec section 5, row M-092): the ONE df actually applied
    # to every stored event-study p-value/CI (safe_inference_batch in
    # _aggregate_event_study). Equals G-1 on the bare-cluster-synthesize
    # path; on explicit-survey fits it is the MINIMUM across per-horizon
    # effective dfs (conservative by design when replicates were dropped) -
    # the value shown is what was USED, not a per-horizon claim. None when
    # the ES rows used normal theory or bootstrap overrode them. Distinct
    # from `df_inference` below, whose narrow bare-cluster-only contract
    # (HonestDiD consumer, PR #487) is unchanged.
    event_study_df: Optional[float] = field(default=None, repr=False)
    bootstrap_results: Optional["CSBootstrapResults"] = field(default=None, repr=False)
    cband_crit_value: Optional[float] = None
    pscore_trim: float = 0.01
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None, repr=False)
    # EPV diagnostics per (group, time) cell
    epv_diagnostics: Optional[Dict[Tuple[Any, Any], Dict[str, Any]]] = field(
        default=None, repr=False
    )
    epv_threshold: float = 10
    pscore_fallback: str = "error"
    # Variance / clustering metadata (PR #XXX — narrow vcov_type contract
    # + cluster= wiring fix). vcov_type is permanently narrow to "hc1" for
    # CS per IF-based variance structure (REGISTRY.md). cluster_name +
    # n_clusters surface the effective clustering level for downstream
    # introspection and label rendering.
    vcov_type: str = "hc1"
    cluster_name: Optional[str] = None
    n_clusters: Optional[int] = None
    # df_inference: cluster-level degrees of freedom for downstream
    # inference, populated on the bare-cluster-synthesize path ONLY.
    # When the user provides an explicit survey_design= (inject or
    # conflict branches), df_inference stays None and the canonical df
    # carrier is survey_metadata.df_survey (which holds the actual
    # CS-internal df, including any post-resolve tightening via the
    # overall_effective_df recompute at staggered.py:~1995-1999).
    # HonestDiD reads survey_metadata.df_survey first and falls back to
    # df_inference only when survey_metadata is absent. Narrow contract
    # prevents HonestDiD from silently overriding a tightened survey df
    # with the original resolved_survey.df_survey.
    df_inference: Optional[int] = None

    # Post-fit re-aggregation payload (spec section 6, rows M-020/M-117),
    # attached by fit() because nothing it needs survives the call. Declared
    # here rather than set dynamically so it is a typed part of the contract.
    # Excluded from repr and equality: it is internal bookkeeping, not a
    # reportable result, and its arrays would make `==` raise.
    # Distinct per-cohort normalization-base EVENT TIMES under
    # base_period="universal" (each cohort's positional base minus its
    # cohort, deduplicated, sorted): the common-reference provenance
    # HonestDiD / PreTrendsPower need on gapped grids, where a cohort's
    # base can overlap another cohort's estimated horizon and no
    # reference-only event-study row marks it. None on varying-base fits
    # (no constant per-cohort reference exists). Declared last among the
    # public fields (positional-compat convention).
    reference_event_times: Optional[Tuple[Any, ...]] = None
    _aggregation_kit: Optional[Any] = field(default=None, repr=False, compare=False)

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
        return (
            f"CallawaySantAnnaResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
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

    # ------------------------------------------------------------------ #
    # Post-fit aggregation (spec section 6; ledger rows M-020 / M-117)
    # ------------------------------------------------------------------ #

    #: CS implements simple / event-study / group / total. ``"calendar"`` is
    #: part of the library-wide vocabulary but CS has no calendar aggregator
    #: (the DEFERRED "Calendar-time aggregation" row), so asking for it raises.
    _AGGREGATE_SUPPORTED = ("simple", "event_study", "group", "total")

    def _aggregate_compute(
        self, level: str, *, weights: Optional[str], balance_e: Optional[int]
    ) -> Any:
        kit = getattr(self, "_aggregation_kit", None)
        if kit is None:
            raise ValueError(
                "This result carries no aggregation kit, so it cannot be "
                "re-aggregated. Kits are attached at fit time by the "
                "estimator; a result unpickled from an older release will "
                "not have one."
            )
        # Per-level bootstrap policy (v4-design section 6, converged with row
        # M-027): 'simple' and 'total' are bit-exact RELAYS of the stored
        # overall inference - faithful under any inference regime, bootstrap
        # included (every SE branch is homogeneous of degree 1 in the x C
        # scaling) - so they dispatch BEFORE the bootstrap machinery. The
        # RECOMPUTE levels below REPLAY the fit-time multiplier bootstrap on
        # bootstrapped fits (kit-retained RNG state; fail-closed only for
        # legacy pickles without the state and cross-backend artifacts).
        if level == "simple":
            return self._aggregate_simple_result(kit)
        if level == "total":
            return self._aggregate_total_result(kit)

        # Shallow copy: shares every array (no data is duplicated) but gives the
        # aggregators a scratch dict to memoize `_agg_cache` into, so the kit
        # itself is never written to. Mirrors what StaggeredTripleDifference
        # already does when it aggregates through a modified copy. Shared by
        # the bootstrap replay and the analytical aggregation below, exactly
        # as fit shares one `precomputed` between them.
        precomputed = dict(kit.bookkeeping)

        # Bootstrap replay: re-run the fit-time multiplier bootstrap from the
        # kit-retained RNG state so the recompute levels carry percentile
        # inference matching a fit-time aggregation (to BLAS reassociation,
        # ~1 ULP on se/CI; p-values are count statistics). NOTE the cost: each
        # replaying call regenerates the full weight stream and re-runs the
        # fused GEMM over the per-cell + per-event-time influence columns -
        # O(n_bootstrap x n_units x (n_gt + n_event_times)) FLOPs per call
        # (no memoization:
        # aggregate() returns a new object and never mutates self, and caching
        # draws would retain matrices the kit design deliberately avoids).
        boot_replay: Optional[CSBootstrapResults] = None
        if self.bootstrap_results is not None:
            spec = getattr(kit, "bootstrap", None)
            if spec is None or getattr(spec, "bitgen_state", None) is None:
                raise NotImplementedError(
                    f"aggregate({level!r}) on this bootstrapped fit needs the "
                    "fit-time bootstrap replay state, which this result "
                    "predates - refit to enable the percentile-bootstrap "
                    "replay. aggregate('simple') and, where supported, "
                    "aggregate('total') relay the stored bootstrap inference "
                    "and remain available."
                )
            current_backend = effective_weight_backend()
            if spec.backend not in ("portable", current_backend):
                # A different weight backend regenerates a DIFFERENT
                # multiplier-weight matrix from the same bit-generator state
                # (Rust row-seeds absolutely from one base seed; NumPy
                # consumes the PCG64 stream), so replaying here would
                # silently desynchronize from this artifact's stored
                # 'simple'/'total' relay inference. "portable"-stamped
                # artifacts (backend-independent generation branches) replay
                # anywhere.
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
                kit.alpha, kit.anticipation, spec.n_bootstrap, spec.weight_type
            )
            boot_replay = host._run_multiplier_bootstrap(
                group_time_effects=self.group_time_effects,
                influence_func_info=kit.influence,
                aggregate=level,
                balance_e=balance_e if level == "event_study" else None,
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
                # Same shared override helper fit uses: percentile se/CI/p +
                # recomputed t, df_used cleared (percentile inference never
                # used the analytical df).
                apply_bootstrap_group_overrides(effects, boot_replay, kit.alpha)
            return self._group_effects_to_aggregation(effects, kit)

        # event_study -> the unified EventStudyResults container (row M-092)
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
        if boot_replay is not None:
            # Mirror fit's clearing rules exactly: percentile se/CI/p + t
            # override, sup-t cband rows, vcov/vcov_index/df cleared (the
            # percentile inference never used them - false provenance
            # otherwise).
            apply_bootstrap_event_study_overrides(es.effects, boot_replay, kit.alpha)
            apply_cband_conf_ints(es.effects, boot_replay.cband_crit_value)
        # `build_event_study_surface` reads the surface off a RESULTS object.
        # Under the immutability contract we cannot populate `self`, so a
        # throwaway carrier holds the freshly computed values.
        carrier = replace(
            self,
            event_study_effects=es.effects,
            event_study_vcov=None if boot_replay is not None else es.vcov,
            event_study_vcov_index=None if boot_replay is not None else es.vcov_index,
            event_study_df=None if boot_replay is not None else es.df_used,
            cband_crit_value=(
                boot_replay.cband_crit_value if boot_replay is not None else self.cband_crit_value
            ),
            # Surface-faithful common-reference provenance: the aggregation
            # recomputes it over the RETAINED cohorts (balance_e can drop
            # the cohort responsible for a second base; the fit-level
            # fit-wide tuple would over-restrict the balanced container).
            # Applies to the bootstrap branch identically.
            reference_event_times=es.reference_event_times,
        )
        return build_event_study_surface(carrier)

    def _aggregate_simple_result(self, kit: Any) -> AggregationResult:
        """The overall ATT as a one-row table.

        ``_aggregate_simple`` runs unconditionally in ``fit()``, so the numbers
        are already stored - this is a view, not a recomputation, and is
        therefore bit-identical to the fit by construction. That is also why
        it is PERMITTED on bootstrapped fits (the per-level policy converged
        with row M-027): the stored quintet - percentile se/p/CI beside the
        finite ``safe_inference`` t - relays verbatim, and only the df COLUMN
        is NaN'd there (no df governs percentile inference, so reporting the
        analytical df beside percentile p would misstate provenance).
        """
        # n_treated_units / n_control_units are UNITS on a panel fit but
        # OBSERVATIONS on a declared repeated cross-section, where fit() counts
        # rows because there is no unit tracking (staggered.py, the panel/RCS
        # branch). n_kind must say which - conflating the two is exactly what
        # the shared vocabulary forbids.
        is_panel = kit.bookkeeping.get("is_panel", True)
        n_kind = "units" if is_panel else "obs"
        ci = self.overall_conf_int or (np.nan, np.nan)
        return AggregationResult(
            level="simple",
            label=np.array(["overall"], dtype=object),
            target=np.array(["att"], dtype=object),
            att=np.array([self.overall_att], dtype=float),
            se=np.array([self.overall_se], dtype=float),
            t_stat=np.array([self.overall_t_stat], dtype=float),
            p_value=np.array([self.overall_p_value], dtype=float),
            conf_int_lower=np.array([ci[0]], dtype=float),
            conf_int_upper=np.array([ci[1]], dtype=float),
            n=np.array([float(self.n_treated_units + self.n_control_units)], dtype=float),
            # NOT ``df_inference``: that field is documented to stay None on
            # explicit ``survey_design=`` fits, where the df that actually
            # governed ``overall_p_value`` lives on ``survey_metadata``.
            # Reading it directly reported df=NaN for survey fits whose CI
            # was built on a finite t-reference. Bootstrapped fits report a
            # NaN df column: no df governs percentile inference.
            df=(np.nan if self.bootstrap_results is not None else resolve_inference_df(self)),
            alpha=self.alpha,
            n_kind=n_kind,
            weight=np.array([1.0], dtype=float),
            estimator=type(self).__name__.replace("Results", ""),
        )

    def _aggregate_total_result(self, kit: Any) -> AggregationResult:
        """The estimator-owned total incremental outcome as a one-row table.

        Exact relay ``C x overall`` CONDITIONAL on the realized aggregation
        mass, where ``C`` is the complete-case treated-observation support
        the fit's own simple aggregation used (``_cs_total_mass``). Fails
        closed - with the reason - on the routings where that mass is not a
        complete-case count: repeated-cross-section routing, declared
        ``survey_design=`` fits, and bare-``cluster=`` fits whose cohort-mass
        weighting diverges from the complete-case count (incomplete treated
        cells). See the REGISTRY CS post-fit ``aggregate('total')`` Note.
        """
        bk = kit.bookkeeping
        # Gate 1 - RC routing. Checked FIRST: a fit that is both
        # survey-declared AND RC-routed must get this (more informative)
        # message deterministically.
        if not bk.get("is_panel", True) or bk.get("agg_cohort_masses") is not None:
            raise NotImplementedError(
                "aggregate('total') is not available on "
                "repeated-cross-section-routed fits: the cohort mass counts "
                "units, not treated observations, so a complete-case total "
                "is not recoverable from the retained fit state - tracked "
                "in DEFERRED.md."
            )
        # Panel-routed legacy kits (pre-upgrade pickles) fail closed: the
        # fit-time snapshots below are the only mutation-proof carriers, and
        # falling back to the mutable public fields would let a post-fit
        # edit silently bypass the survey gate or move the mass.
        cells = bk.get("agg_gt_cells")
        is_survey_fit = bk.get("is_survey_fit")
        if cells is None or is_survey_fit is None:
            raise NotImplementedError(
                "aggregate('total') needs fit-time state this result "
                "predates: it was fitted before aggregate('total') existed - "
                "refit to use it."
            )
        # Gate 2 - declared survey design (immutable fit-time provenance;
        # bare-cluster fits synthesize an internal design but keep this
        # False, so they are ADMITTED).
        if is_survey_fit:
            raise NotImplementedError(
                "aggregate('total') is not available on fits declaring a "
                "survey_design: the realized-mass relay omits the survey "
                "mass-uncertainty variance term, and design-aware "
                "population-scale totals are not implemented (retained "
                "weight scale differs by design family) - tracked in "
                "DEFERRED.md. For an unweighted clustered fit, cluster= "
                "(without survey_design) supports totals."
            )
        ci = self.overall_conf_int or (np.nan, np.nan)
        return build_total_relay_row(
            mass=self._cs_total_mass(kit),
            att=self.overall_att,
            se=self.overall_se,
            t_stat=self.overall_t_stat,
            p_value=self.overall_p_value,
            conf_int=ci,
            df=(np.nan if self.bootstrap_results is not None else resolve_inference_df(self)),
            alpha=self.alpha,
            estimator=type(self).__name__.replace("Results", ""),
        )

    def _cs_total_mass(self, kit: Any) -> float:
        """The realized simple-aggregation mass ``C``, from kit snapshots.

        Replays ``_aggregate_simple``'s cell selection verbatim
        (staggered_aggregation.py: anticipation filter first, finite-effect
        mask second, weight = cohort mass where ``fixed_cohort_agg_weights``
        provides one else per-cell complete-case ``n_treated``, pairwise
        ``np.sum``) over the immutable ``agg_gt_cells`` fit-time snapshot -
        never the mutable public ``group_time_effects``. The ``agg_weight``
        fallback of the source is deliberately absent: that field exists
        only on RC-routed cells, which the RC gate excludes, and the 4-tuple
        snapshot does not carry it.

        On admitted fits the cohort-mass branch (bare-``cluster=``: the
        synthesized all-ones design makes each cohort mass the integer
        cohort size) must COINCIDE with the complete-case count - when kept
        cells have incomplete treated support the two disagree (the same
        overcount that keeps RC routings gated) and this raises rather than
        publishing an ambiguous mass. Empty post set or all-NaN effects
        return NaN WITHOUT re-emitting the fit-time UserWarnings (the
        RELAY-level no-re-warn convention - the NaN row is the signal; the
        RECOMPUTE levels' bootstrap replay re-emits by design).
        """
        bk = kit.bookkeeping
        # TRAP: never bk["agg_total_weight"] - that is the RC all-units WIF
        # mass, not the kept-cell simple mass.
        masses = fixed_cohort_agg_weights(bk)
        effects_list: List[float] = []
        weights_list: List[float] = []
        complete_case: List[float] = []
        for g, t, effect, n_treated in bk["agg_gt_cells"]:
            if t < g - kit.anticipation:
                continue
            effects_list.append(effect)
            if masses is not None and g in masses:
                weights_list.append(masses[g])
            else:
                weights_list.append(n_treated)
            complete_case.append(n_treated)
        if not effects_list:
            return float(np.nan)
        effects = np.array(effects_list, dtype=float)
        weights = np.array(weights_list, dtype=float)
        cc = np.array(complete_case, dtype=float)
        finite_mask = np.isfinite(effects)
        if not np.all(finite_mask):
            weights = weights[finite_mask]
            cc = cc[finite_mask]
        if len(weights) == 0:
            return float(np.nan)
        total_weight = float(np.sum(weights))
        # Coincidence guard - only meaningful when a finite mass exists and
        # the cohort-mass branch was taken.
        if masses is not None and np.isfinite(total_weight) and total_weight != float(np.sum(cc)):
            raise NotImplementedError(
                "aggregate('total') is not available on this fit: the "
                "cohort-mass weighting (from cluster=) counts full cohorts "
                "per period, but some kept cells have incomplete treated "
                "support, so the realized aggregation mass and the "
                "complete-case treated-observation count disagree - tracked "
                "in DEFERRED.md."
            )
        return total_weight

    def _group_effects_to_aggregation(
        self, effects: Dict[Any, Dict[str, Any]], kit: Any
    ) -> AggregationResult:
        """Per-cohort aggregation as a table.

        ``weight`` is left unset: ``_aggregate_by_group`` weights ``(g, t)``
        cells equally WITHIN each cohort and never forms a cross-cohort mass,
        so there is no per-row weight to report and inventing one would be a
        fabricated number. ``n`` is the cohort's finite-contributing cell
        count, hence ``n_kind="cells"`` rather than units.
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
            # The df the GROUP aggregation's own inference used, recorded by
            # ``_aggregate_by_group``. Previously this read ``event_study_df``
            # - a different aggregation's df, and None after a plain fit.
            df=rows[0].get("df_used") if rows else None,
            alpha=kit.alpha,
            n_kind="cells",
            weight=None,
            estimator=type(self).__name__.replace("Results", ""),
        )

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
            "Callaway-Sant'Anna Staggered Difference-in-Differences Results".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated ' + ('obs:' if not self.panel else 'units:'):<30} {self.n_treated_units:>10}",
            f"{'Never-treated ' + ('obs:' if not self.panel else 'units:'):<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            f"{'Control group:':<30} {self.control_group:>10}",
            f"{'Base period:':<30} {self.base_period:>10}",
            "",
        ]

        # Survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 85))

        # Overall ATT
        lines.extend(
            [
                "-" * 85,
                "Overall Average Treatment Effect on the Treated".center(85),
                "-" * 85,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
                f"{'ATT':<15} {self.overall_att:>12.4f} {self.overall_se:>12.4f} "
                f"{self.overall_t_stat:>10.3f} {self.overall_p_value:>10.4f} "
                f"{_get_significance_stars(self.overall_p_value):>6}",
                "-" * 85,
                "",
                f"{conf_level}% Confidence Interval: [{self.overall_conf_int[0]:.4f}, {self.overall_conf_int[1]:.4f}]",
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
                        f"(cohort g={min_g[0]}). Threshold: {self.epv_threshold:.0f}.",
                        "Consider: estimation_method='reg' or fewer covariates.",
                        "Call results.epv_summary() for per-cohort details.",
                        "-" * 85,
                        "",
                    ]
                )

        # Event study effects if available
        if self.event_study_effects:
            ci_label = "Simult. CI" if self.cband_crit_value is not None else "Pointwise CI"
            lines.extend(
                [
                    "-" * 85,
                    "Event Study (Dynamic) Effects".center(85),
                    "-" * 85,
                    f"{'Rel. Period':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
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

            lines.extend(["-" * 85])
            if self.cband_crit_value is not None:
                lines.append(
                    f"{ci_label}: critical value = {self.cband_crit_value:.4f} "
                    f"(sup-t bootstrap, {conf_level}% family-wise)"
                )
            lines.append("")

        # Group effects if available
        if self.group_effects:
            lines.extend(
                [
                    "-" * 85,
                    "Effects by Treatment Cohort".center(85),
                    "-" * 85,
                    f"{'Cohort':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
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

    def epv_summary(self, show_all: bool = False) -> pd.DataFrame:
        """
        Return per-cohort EPV diagnostics as a DataFrame.

        Parameters
        ----------
        show_all : bool, default False
            If False, only show cells with low EPV. If True, show all cells.

        Returns
        -------
        pd.DataFrame
            Columns: group, time, epv, n_events, n_params, is_low.
        """
        if not self.epv_diagnostics:
            return pd.DataFrame(columns=["group", "time", "epv", "n_events", "n_params", "is_low"])
        rows = []
        for (g, t), diag in sorted(self.epv_diagnostics.items()):
            if show_all or diag.get("is_low", False):
                rows.append(
                    {
                        "group": g,
                        "time": t,
                        "epv": diag.get("epv"),
                        "n_events": diag.get("n_events"),
                        "n_params": diag.get("k"),
                        "is_low": diag.get("is_low", False),
                    }
                )
        cols = ["group", "time", "epv", "n_events", "n_params", "is_low"]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print summary to stdout."""
        print(self.summary(alpha))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert headline results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Canonical inference row plus scalar metadata. Detailed
            group-time / event-study tables are available via
            ``to_dataframe(level=...)``.
        """
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.overall_conf_int[0],
            "conf_int_upper": self.overall_conf_int[1],
            "n_obs": self.n_obs,
            "n_treated_units": self.n_treated_units,
            "n_control_units": self.n_control_units,
            "control_group": self.control_group,
            "base_period": self.base_period,
            "anticipation": self.anticipation,
            "panel": self.panel,
            "alpha": self.alpha,
            "vcov_type": self.vcov_type,
        }
        if self.cluster_name is not None:
            result["cluster_name"] = self.cluster_name
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        return result

    def to_dataframe(self, level: str = "group_time") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="group_time"
            Level of aggregation: "group_time", "event_study", or "group".

        Returns
        -------
        pd.DataFrame
            Results as DataFrame.
        """
        if level == "group_time":
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
                    # None for estimable cells; a reason code for non-estimable
                    # (NaN) cells materialized in group_time_effects.
                    "skip_reason": data.get("skip_reason"),
                }
                if self.epv_diagnostics and (g, t) in self.epv_diagnostics:
                    row["epv"] = self.epv_diagnostics[(g, t)].get("epv")
                rows.append(row)
            return pd.DataFrame(rows)

        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError(
                    "Event study effects not computed. "
                    "Call results.aggregate('event_study') to compute them post-fit "
                    "(no refit required)."
                )
            rows = []
            for rel_t, data in sorted(self.event_study_effects.items()):
                cband_ci = data.get("cband_conf_int", (np.nan, np.nan))
                rows.append(
                    {
                        "relative_period": rel_t,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                        "cband_lower": cband_ci[0],
                        "cband_upper": cband_ci[1],
                    }
                )
            return pd.DataFrame(rows)

        elif level == "group":
            if self.group_effects is None:
                raise ValueError(
                    "Group effects not computed. "
                    "Call results.aggregate('group') to compute them post-fit "
                    "(no refit required)."
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
                f"Unknown level: {level}. Use 'group_time', 'event_study', or 'group'."
            )

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
