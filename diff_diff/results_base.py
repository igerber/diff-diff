"""Shared result-contract foundations (4.0 program, Phase 2 PR (a)).

This module is a deliberate LEAF: it imports numpy/pandas only, never other
``diff_diff`` modules, so every results/diagnostic/consumer module can import
it without cycles.

Three public symbols (see ``docs/v4-design.md``):

- :class:`BaseResults` - the shared base for estimator result containers
  (spec section 5, results contract).
- :class:`Diagnostic` - the marker base for diagnostic result containers
  (spec section 3.5, ledger row M-091).
- :class:`EventStudyResults` - the unified event-study representation
  (spec section 5, ledger row M-092).

``build_event_study_surface`` is package-internal: Phase 2 PR (b) wires it
into ``results.aggregate(type="event_study")``, and the Phase 3 merged
TwoWayFixedEffects event-study mode returns the container natively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

__all__ = ["BaseResults", "Diagnostic", "EventStudyResults"]


class Diagnostic:
    """Marker base for diagnostic RESULT containers (spec section 3.5).

    diff-diff distinguishes three object kinds: estimators, estimator
    results, and diagnostics. Exactly one bit is load-bearing: an
    estimator's result carries a causal-effect inference row (the canonical
    quintet ``att``/``se``/``t_stat``/``p_value``/``conf_int``); a
    diagnostic's result does NOT - it assesses a design, an identifying
    assumption, or robustness.

    Marked classes expose ``summary()`` and ``to_dataframe()`` and are
    exempt from the quintet BY TYPE, so consumers route with
    ``isinstance(result, Diagnostic)`` instead of result-class-name
    special-casing. The contract is enforced by the roster test in
    ``tests/test_diagnostic_marker.py`` (ledger row M-091), not by the type
    system - this class intentionally has no methods.
    """

    __slots__ = ()


class BaseResults:
    """Shared base for ESTIMATOR result containers (spec section 5).

    Behaviorally inert in 3.9 - a TRANSITIONAL marker base that makes the
    results contract checkable and anchors the 4.0 storage flip. It does not
    yet enforce a uniform runtime protocol: most subclasses are scalar
    estimator results (one canonical quintet), but :class:`EventStudyResults`
    is intentionally vector-valued, and a few inherited classes still expose
    ``summary()`` without an ``alpha`` argument. Full protocol uniformity
    (uniform ``summary(alpha=None)``, native canonical storage) is enforced in
    later Phase 2 / 4.0 PRs. The target contract every ESTIMATOR-results
    subclass converges to:

    - **Canonical quintet.** ``att``, ``se``, ``t_stat``, ``p_value``,
      ``conf_int`` bound to ONE coherent inference row (native fields or
      property aliases over legacy ``overall_*``/``avg_*`` storage; the
      storage flips to native canonical fields at 4.0, ledger rows
      M-050..M-058, with legacy names living on as FutureWarning
      properties until 5.0).
    - **Serialization.** ``summary(alpha=None)``, ``to_dict()`` (canonical
      key names only - deprecated names never leak into serialized
      output), and ``to_dataframe(level=...)`` where multiple views exist.
    - **Pickle migration.** Classes whose stored field names change ship
      ``__setstate__`` migration following the
      ``SyntheticDiDResults.__setstate__`` precedent
      (``diff_diff/results.py``) so 3.x pickles load under 4.0.
    - **Planned estimand self-description hook.** The per-class headline
      semantics (name / definition / aggregation / headline attribute)
      currently live in ``_reporting_helpers.describe_target_parameter``,
      keyed by result-class name. A later PR lifts that block onto this
      base so results self-describe their target parameter (the MMM
      exporter's verified allowlist is the first consumer); nothing here
      constrains that lift.

    Diagnostic result containers (spec section 3.5) do NOT inherit this
    base - they are marked with :class:`Diagnostic` and are exempt from the
    quintet by type.
    """

    __slots__ = ()


def _json_safe_label(value: Any) -> Any:
    """Convert an event-time label to a JSON-serializable form.

    Calendar labels may be ``pandas.Timestamp`` / ``Period`` / ``datetime``;
    ``.tolist()`` preserves those objects, which ``json.dumps`` cannot
    serialize. Datetimes become ISO strings, pandas ``Period`` its ``str``,
    numpy scalars their Python value; everything else passes through.
    """
    if value is None:
        return None
    if type(value).__name__ == "Period":  # pandas.Period (no isoformat)
        return str(value)
    if hasattr(value, "isoformat"):  # datetime / date / pandas.Timestamp
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


#: Pinned column schema of ``EventStudyResults.to_dataframe()`` - identical
#: for every producer (spec section 5; enforced by
#: ``tests/test_event_study_surface.py``, ledger row M-092).
EVENT_STUDY_SCHEMA: Tuple[str, ...] = (
    "event_time",
    "att",
    "se",
    "t_stat",
    "p_value",
    "conf_int_lower",
    "conf_int_upper",
    "cband_lower",
    "cband_upper",
    "n",
    "df",
    "is_reference",
    # The per-row estimand discriminator (appended with row M-027, the
    # AggregationResult.target precedent): "att" for every ATT producer,
    # the estimand label ("WAS"/"WAS_d_lower") where the att column is
    # NOT an ATT - so a detached frame never mislabels its numbers.
    "estimand",
)

#: Closed vocabulary for the ``n_kind`` field, SHARED by every container that
#: reports a count alongside an estimate (``EventStudyResults`` and
#: ``AggregationResult``), so a consumer can route on ``n_kind`` uniformly
#: across the two. Each value names what one unit of ``n`` counts; never
#: conflate them.
N_KIND_VOCABULARY: Tuple[str, ...] = (
    "groups",
    "switcher_cells",
    "cells",
    "units",
    "obs",
    "clusters",
)


@dataclass
class EventStudyResults(BaseResults):
    """Unified event-study representation (spec section 5, row M-092).

    ONE representation for per-event-time effects across all estimators.
    Columnar numpy arrays index-aligned to ``event_time`` (the
    ``HeterogeneousAdoptionDiDEventStudyResults`` precedent). Values are
    copied bit-exactly from each estimator's native surface - never
    recomputed - except the mandated reference-row normalization
    (``att=0.0``, inference NaN).

    Parameters
    ----------
    event_time : np.ndarray
        Sorted estimator-native event-time labels, NEVER renumbered.
        Relative producers use their own origin (see
        ``event_time_convention``); the pre-4.0 MultiPeriodDiD surface is
        calendar-keyed (``time_scale="calendar"``) and may carry object
        dtype (str/datetime period labels).
    att, se, t_stat, p_value : np.ndarray
        Canonical per-event-time inference columns. On the reference row
        (if any): ``att == 0.0`` and ``se``/``t_stat``/``p_value`` are NaN.
    conf_int_lower, conf_int_upper : np.ndarray
        Confidence-interval bounds at ``alpha`` (NaN on the reference row).
    is_reference : np.ndarray
        Boolean; the EXPLICIT reference-period marking. This column - not
        any count sentinel - is the sole consumer-facing signal. Usually one
        True entry, but MULTIPLE are legal (CallawaySantAnna universal base
        on a gapped grid carries one per cohort's positional base), and ZERO
        when the estimator omits no baseline (e.g. HAD, Wooldridge). Use
        ``reference_periods`` for the general case; ``reference_period`` is
        the single-reference convenience scalar.
    n : np.ndarray
        Per-event-time count as float, NaN where the producer records none
        (and on the reference row - no estimation happened there).
    n_kind : str or None
        Semantic of ``n`` for this producer, drawn from the shared
        :data:`N_KIND_VOCABULARY`: ``"groups"`` (a group-level count -
        cohorts for CallawaySantAnna/SunAbraham, eligible switcher groups
        per horizon for de Chaisemartin-D'Haultfoeuille with ``L_max >= 1``),
        ``"switcher_cells"`` (dCDH legacy ``L_max is None`` path: switching
        ``(g, t)`` cells, where one group may contribute several),
        ``"cells"`` (``(g, t)`` cells generally - what CallawaySantAnna's
        ``"group"`` aggregation counts), ``"units"`` (distinct units, as in
        the overall/simple aggregation), ``"obs"`` (observations),
        ``"clusters"``, or None when no count is recorded. Never conflate
        these units.
    reference_period : Any or None
        Convenience scalar echo of the marked row's ``event_time`` label when
        there is EXACTLY ONE reference row; None when there are zero or
        several. Use ``is_reference`` (or the ``reference_periods`` property)
        for the general case - some estimators (CallawaySantAnna universal
        base on a gapped grid) carry multiple reference-only horizons.
    time_scale : str
        ``"relative"`` or ``"calendar"``.
    event_time_convention : str or None
        Origin documentation for relative scales: ``"e0_first_treated"``
        (e = t - g; first treated period at e=0) or ``"l1_first_switch"``
        (de Chaisemartin-D'Haultfoeuille: instantaneous effect at l=1,
        placebos at negative keys). Horizons are documented, not
        renumbered.
    vcov : np.ndarray or None
        Full event-study variance-covariance matrix where the RESULT
        CONTAINER exposes one (e.g. CallawaySantAnna, SunAbraham,
        MultiPeriodDiD, StackedDiD, and TwoStageDiD's analytical modes),
        ordered by ``vcov_index``. None when the producer records no matrix
        or when the stored SEs are no longer its diagonal (bootstrap and
        replicate-weight overrides clear it rather than ship an
        inconsistent matrix).
    vcov_index : np.ndarray or None
        ``event_time`` labels labelling ``vcov``'s rows/columns (explicit
        ordering for HonestDiD / PreTrendsPower consumption).
    cband_lower, cband_upper : np.ndarray or None
        Simultaneous confidence-band bounds where computed; None when the
        producer has none (``to_dataframe`` then emits NaN columns - the
        schema never changes).
    cband_crit_value : float or None
        Critical value of the simultaneous band, where computed.
    alpha : float
        Significance level of the stored intervals.
    source : str or None
        Producing results-class name (provenance).
    df : float, np.ndarray, or None
        Per-row inference degrees of freedom: ``df[i]`` is the df ACTUALLY
        passed to ``safe_inference`` for row i's stored p-value/CI, threaded
        from the producer. Accepts None (no df exposed -> all-NaN column), a
        scalar (broadcast to every row - e.g. CallawaySantAnna, whose
        explicit-survey event study applies ONE conservative df, the minimum
        per-horizon effective df, to all rows; or de
        Chaisemartin-D'Haultfoeuille, whose effect and placebo rows share one
        design df), or a length-n array (per-row producers: StackedDiD and
        SunAbraham ``hc2_bm`` per-event Bell-McCaffrey df, LPDiD per-horizon
        cluster df, MultiPeriodDiD ``hc2_bm`` per-period df).
        NaN on any row means normal-theory inference, an undefined df,
        bootstrap-overridden inference, or a producer that records none;
        reference rows and rows with NaN p-values are always NaN.
    base_period : str or None
        Producer provenance: the fit's base-period regime where the
        producer has one (CallawaySantAnna vocabulary: ``"varying"`` or
        ``"universal"``). None when the producer has no such notion.
        HonestDiD reads this for its universal-base interpretation
        warning.
    anticipation : int or None
        Producer provenance: the fit's anticipation window in periods,
        where the producer has one. None when the producer has no such
        notion. PreTrendsPower reads this to exclude anticipation-window
        rows (``event_time >= -anticipation``) from the pre-trend set.
    df_survey : float or None
        Producer provenance: the fit's resolved SCALAR inference df,
        with the established semantics of the fit-time consumers -
        ``survey_metadata.df_survey`` where present (``0.0`` = replicate
        design with an undefined df, which fails closed to NaN critical
        values downstream), else ``df_inference`` (the bare-``cluster=``
        carrier), else None (no scalar df notion). Exists beside the
        per-row ``df`` column because that column CANNOT encode the
        replicate-undefined sentinel: ``__post_init__`` forces per-row
        df to NaN wherever the p-value is non-finite.
    reference_event_times : tuple or None
        Producer provenance: the DISTINCT per-cohort normalization-base
        event times (CallawaySantAnna ``base_period="universal"``: each
        cohort's positional base period minus its cohort, deduplicated,
        sorted). More than one entry means the coefficients were
        normalized against DIFFERENT bases (gapped time grid) - and on
        such grids a cohort's base can OVERLAP another cohort's
        estimated horizon, where NO reference-only row exists to mark
        it, so this field (not ``is_reference``) is the authoritative
        common-reference signal. HonestDiD and PreTrendsPower fail
        closed when it carries more than one entry. None when the
        producer records no such notion (varying base, non-CS
        producers, hand-built surfaces).
    """

    event_time: np.ndarray
    att: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    conf_int_lower: np.ndarray
    conf_int_upper: np.ndarray
    is_reference: np.ndarray
    n: np.ndarray
    n_kind: Optional[str] = None
    reference_period: Optional[Any] = None
    time_scale: str = "relative"
    event_time_convention: Optional[str] = None
    vcov: Optional[np.ndarray] = field(default=None, repr=False)
    vcov_index: Optional[np.ndarray] = field(default=None, repr=False)
    cband_lower: Optional[np.ndarray] = None
    cband_upper: Optional[np.ndarray] = None
    cband_crit_value: Optional[float] = None
    alpha: float = 0.05
    source: Optional[str] = None
    df: Optional[Union[float, np.ndarray]] = None
    # Provenance fields declared LAST so every pre-existing field keeps its
    # positional index in the generated __init__ (the constructor signature
    # is public API).
    base_period: Optional[str] = None
    anticipation: Optional[int] = None
    df_survey: Optional[float] = None
    # Distinct per-cohort normalization-base EVENT TIMES (CallawaySantAnna
    # base_period="universal": each cohort's positional base minus its
    # cohort, deduplicated and sorted). NOT the same thing as the
    # ``reference_periods`` property (the is_reference-marked rows): on a
    # gapped grid a cohort's base can OVERLAP another cohort's estimated
    # horizon, where no reference-only row exists to mark it - this field
    # is the consumer-facing signal that coefficients were normalized
    # against more than one base. None when the producer records no such
    # notion (varying base, non-CS producers, hand-built surfaces).
    reference_event_times: Optional[Tuple[Any, ...]] = None
    #: The estimand label for the ``att`` column when the producer's
    #: per-horizon estimand is NOT an ATT - HeterogeneousAdoptionDiD's
    #: per-horizon "WAS"/"WAS_d_lower" (row M-027; ``_from_had`` relays
    #: ``target_parameter``). None means the column is an ATT (every other
    #: producer), keeping their rendering byte-stable. ``summary()`` uses
    #: it as the column heading and ``to_dict()`` serializes it, so the
    #: numbers never silently change meaning. Optional provenance appended
    #: last (the M-092 pre-cut amendment convention).
    estimand: Optional[str] = None
    #: Calendar-partition provenance (M-092 pre-cut amendment #5, with row
    #: M-010): the producer's AUTHORITATIVE post-treatment period labels.
    #: MultiPeriodDiD accepts an ARBITRARY ``post_periods`` subset (a
    #: non-suffix split is legal), and the partition is not recoverable
    #: from ``event_time``/``is_reference`` positionally - consumers that
    #: need pre/post classification on a calendar surface (HonestDiD,
    #: PreTrendsPower, the plotter) read THIS field; pre-periods derive as
    #: the complement: non-reference rows whose ``event_time`` is not in
    #: ``post_periods``. Threaded by ``_from_mpd`` (and the TWFE
    #: event-study producer); None from every other builder.
    post_periods: Optional[Tuple[Any, ...]] = None
    #: TWFE event-study design provenance (M-092 amendment #5, with row
    #: M-010): ``"within"`` (unit + time FE) or ``"pooled"`` (the
    #: MultiPeriodDiD design - treatment-group dummy + period dummies, no
    #: unit FE). The two specs differ materially in SEs, so the design
    #: must be recoverable from the container. None from every builder
    #: (only the TwoWayFixedEffects event-study producer sets it).
    estimation_spec: Optional[str] = None

    _ARRAY_FIELDS = (
        "att",
        "se",
        "t_stat",
        "p_value",
        "conf_int_lower",
        "conf_int_upper",
        "n",
    )

    def __post_init__(self) -> None:
        # Coerce with a COPY (np.array, not np.asarray): reference-row
        # normalization below writes in place, so the container must never
        # mutate a caller-owned buffer (and read-only inputs must not fail).
        # event_time keeps its native dtype (object labels legal in calendar
        # mode - no numeric assumptions anywhere below).
        self.event_time = np.array(self.event_time)
        for name in self._ARRAY_FIELDS:
            setattr(self, name, np.array(getattr(self, name), dtype=float))
        self.is_reference = np.array(self.is_reference, dtype=bool)

        if self.event_time.ndim != 1:
            raise ValueError(
                f"EventStudyResults event_time must be one-dimensional; "
                f"got ndim={self.event_time.ndim}."
            )
        # A zero-row surface is legal: an event study that was REQUESTED but
        # has no estimable horizons (e.g. EfficientDiD's balance_e removing
        # every cohort) is a valid degenerate result, distinct from one that
        # was never requested (which raises upstream in the builder).
        n_rows = self.event_time.shape[0]
        for name in self._ARRAY_FIELDS + ("is_reference",):
            arr = getattr(self, name)
            if arr.shape != (n_rows,):
                raise ValueError(
                    f"EventStudyResults field '{name}' has shape {arr.shape}; "
                    f"expected ({n_rows},) to align with event_time."
                )

        # Multiple reference rows are legal: CallawaySantAnna with
        # base_period="universal" on a gapped period grid materializes each
        # cohort's positional base as its own reference-only horizon (all
        # n_groups==0), so a valid surface can carry several. `is_reference`
        # is the general marker; `reference_period` is a convenience scalar
        # DERIVED from it - authoritative only when there is exactly one
        # reference, None otherwise - so a caller-supplied value can never
        # disagree with `is_reference` / `reference_periods`.
        n_ref = int(self.is_reference.sum())
        if n_ref == 1:
            ref_label = self.event_time[self.is_reference][0]
            # Plain Python scalar so to_dict() stays JSON-serializable.
            self.reference_period = ref_label.item() if hasattr(ref_label, "item") else ref_label
        else:
            self.reference_period = None

        # Simultaneous-band bounds are all-or-nothing (a lower without an upper
        # is meaningless).
        if (self.cband_lower is None) != (self.cband_upper is None):
            raise ValueError(
                "EventStudyResults requires cband_lower and cband_upper together, " "or neither."
            )
        for name in ("cband_lower", "cband_upper"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.array(arr, dtype=float)  # copy: normalized in place below
                if arr.shape != (n_rows,):
                    raise ValueError(
                        f"EventStudyResults field '{name}' has shape {arr.shape}; "
                        f"expected ({n_rows},)."
                    )
                setattr(self, name, arr)

        # Reference rows are normalization anchors: att is exactly 0.0 and all
        # inference / count / band fields are undefined (NaN). Enforce this on
        # EVERY container, not just builder output, so a direct public
        # construction can never expose finite inference on a reference row.
        if self.is_reference.any():
            self.att[self.is_reference] = 0.0
            for name in ("se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper", "n"):
                getattr(self, name)[self.is_reference] = np.nan
            for name in ("cband_lower", "cband_upper"):
                arr = getattr(self, name)
                if arr is not None:
                    arr[self.is_reference] = np.nan

        # df is per-row PROVENANCE: the df actually passed to safe_inference
        # for that row's stored p-value/CI. Normalize None (no df exposed) to
        # an all-NaN column and broadcast scalars (single-df producers), then
        # NaN out rows that carry no safe_inference output - reference rows
        # and rows whose stored p-value is NaN (non-estimable horizons) never
        # used a df. Runs AFTER reference-row normalization so p_value is
        # final; explicit block rather than _ARRAY_FIELDS because df, like
        # the band columns, is optional.
        if self.df is None:
            df_arr = np.full(n_rows, np.nan)
        elif np.ndim(self.df) == 0:
            df_arr = np.full(n_rows, float(cast(float, self.df)))
        else:
            df_arr = np.array(self.df, dtype=float)
            if df_arr.shape != (n_rows,):
                raise ValueError(
                    f"EventStudyResults field 'df' has shape {df_arr.shape}; "
                    f"expected ({n_rows},) to align with event_time."
                )
        df_arr[~np.isfinite(self.p_value)] = np.nan
        # n_kind is a routing key consumers share with AggregationResult, so an
        # off-vocabulary value is a contract break, not a free-form label. Both
        # containers validate it - enforcing on only one would let an unknown
        # value reach a consumer through the unchecked side.
        if self.n_kind is not None and self.n_kind not in N_KIND_VOCABULARY:
            raise ValueError(
                f"EventStudyResults n_kind {self.n_kind!r} is not in the shared "
                f"vocabulary {N_KIND_VOCABULARY}."
            )

        # post_periods is the AUTHORITATIVE calendar partition consumers
        # classify by, so malformed-but-present content is a contract break
        # (the same fail-closed posture as n_kind / the vcov pairing): a
        # hand-built surface must never silently mispartition a consumer.
        if self.post_periods is not None:
            pp = tuple(self.post_periods)
            if len(pp) == 0:
                raise ValueError(
                    "EventStudyResults post_periods must be a nonempty tuple "
                    "when provided (None means 'no partition recorded')."
                )
            if len(set(pp)) != len(pp):
                raise ValueError(f"EventStudyResults post_periods contains duplicates: {pp}.")
            event_set = set(self.event_time.tolist())
            missing = [p for p in pp if p not in event_set]
            if missing:
                raise ValueError(
                    f"EventStudyResults post_periods entries {missing} are not " f"in event_time."
                )
            ref_set = set(self.event_time[self.is_reference].tolist())
            overlap = [p for p in pp if p in ref_set]
            if overlap:
                raise ValueError(
                    f"EventStudyResults post_periods entries {overlap} are "
                    f"reference rows; the reference period is pre-treatment "
                    f"by construction."
                )
            self.post_periods = pp

        # estimation_spec is a two-value design label (TWFE event-study
        # producer only); off-vocabulary values are rejected like n_kind.
        if self.estimation_spec is not None and self.estimation_spec not in (
            "within",
            "pooled",
        ):
            raise ValueError(
                f"EventStudyResults estimation_spec {self.estimation_spec!r} "
                f"is not in ('within', 'pooled')."
            )
        self.df = df_arr

        if (self.vcov is None) != (self.vcov_index is None):
            raise ValueError(
                "EventStudyResults requires vcov and vcov_index together "
                "(explicit ordering) or neither."
            )
        if self.vcov is not None:
            # COPY, not asarray (matching the _ARRAY_FIELDS convention
            # above): np.asarray is a no-op view when the dtype already
            # matches, which would alias the PRODUCER's stored matrix -
            # e.g. StackedDiDResults.event_study_vcov reaches this
            # constructor unmodified via the post-fit aggregate() view
            # (row M-024), and a caller mutating the container would
            # silently corrupt the fitted result. vcov_index keeps its
            # native dtype (int labels must not become floats in
            # to_dict()) but is copied for the same isolation.
            self.vcov = np.array(self.vcov, dtype=float)
            self.vcov_index = np.array(self.vcov_index)
            k = self.vcov_index.shape[0]
            if self.vcov.shape != (k, k):
                raise ValueError(
                    f"EventStudyResults vcov has shape {self.vcov.shape}; "
                    f"expected ({k}, {k}) to match vcov_index."
                )
            event_labels = set(self.event_time.tolist())
            missing = [lbl for lbl in self.vcov_index.tolist() if lbl not in event_labels]
            if missing:
                raise ValueError(
                    f"EventStudyResults vcov_index entries {missing} are not " "event_time labels."
                )

        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}.")

    @property
    def reference_periods(self) -> List[Any]:
        """All reference-row ``event_time`` labels (JSON-safe scalars).

        The general accessor for the normalization anchors: one entry for the
        common single-reference case, several for CallawaySantAnna universal
        base on a gapped grid, empty when the estimator omits no baseline.
        """
        return [_json_safe_label(v) for v in self.event_time[self.is_reference].tolist()]

    # ------------------------------------------------------------------
    # Serialization contract
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Return the pinned per-event-time table (``EVENT_STUDY_SCHEMA``).

        The column schema is identical for every producer; band columns are
        NaN-filled when the producer computes no simultaneous band.
        """
        n_rows = self.event_time.shape[0]
        nan_col = np.full(n_rows, np.nan)
        return pd.DataFrame(
            {
                "event_time": self.event_time,
                "att": self.att,
                "se": self.se,
                "t_stat": self.t_stat,
                "p_value": self.p_value,
                "conf_int_lower": self.conf_int_lower,
                "conf_int_upper": self.conf_int_upper,
                "cband_lower": self.cband_lower if self.cband_lower is not None else nan_col,
                "cband_upper": self.cband_upper if self.cband_upper is not None else nan_col,
                "n": self.n,
                # __post_init__ guarantees an array; cast narrows the
                # scalar-accepting field type for mypy.
                "df": cast(np.ndarray, self.df),
                "is_reference": self.is_reference,
                # Per-row estimand discriminator: a detached frame must not
                # mislabel WAS-family numbers as ATT (row M-027).
                "estimand": np.full(n_rows, self.estimand or "att", dtype=object),
            },
            columns=list(EVENT_STUDY_SCHEMA),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dict (columns as lists, plus metadata).

        Calendar event-time labels (``pandas.Timestamp`` / ``Period`` /
        ``datetime``) are converted to ISO strings so the result is
        JSON-serializable; relative-time integer labels pass through.
        """
        out: Dict[str, Any] = {
            "event_time": [_json_safe_label(v) for v in self.event_time.tolist()],
            "att": self.att.tolist(),
            "se": self.se.tolist(),
            "t_stat": self.t_stat.tolist(),
            "p_value": self.p_value.tolist(),
            "conf_int_lower": self.conf_int_lower.tolist(),
            "conf_int_upper": self.conf_int_upper.tolist(),
            "cband_lower": self.cband_lower.tolist() if self.cband_lower is not None else None,
            "cband_upper": self.cband_upper.tolist() if self.cband_upper is not None else None,
            "n": self.n.tolist(),
            "is_reference": self.is_reference.tolist(),
            "n_kind": self.n_kind,
            "reference_period": _json_safe_label(self.reference_period),
            "reference_periods": self.reference_periods,
            "time_scale": self.time_scale,
            "event_time_convention": self.event_time_convention,
            # Full event-study vcov + its ordered index (labels JSON-safed),
            # so covariance-aware consumers (HonestDiD / PreTrendsPower) can
            # round-trip through the serialized surface.
            "vcov": self.vcov.tolist() if self.vcov is not None else None,
            "vcov_index": (
                [_json_safe_label(v) for v in self.vcov_index.tolist()]
                if self.vcov_index is not None
                else None
            ),
            "cband_crit_value": self.cband_crit_value,
            "alpha": self.alpha,
            "source": self.source,
            # Calendar-partition + design provenance (M-092 amendment #5):
            # post_periods labels JSON-safed like event_time (they carry the
            # same Timestamp/Period label types on calendar surfaces).
            "post_periods": (
                [_json_safe_label(p) for p in self.post_periods]
                if self.post_periods is not None
                else None
            ),
            "estimation_spec": self.estimation_spec,
            "df": cast(np.ndarray, self.df).tolist(),
            "base_period": self.base_period,
            "anticipation": self.anticipation,
            "estimand": self.estimand,
            "df_survey": self.df_survey,
            "reference_event_times": (
                # _json_safe_label per element: CS period arithmetic yields
                # numpy scalars, which json.dumps cannot serialize.
                [_json_safe_label(v) for v in self.reference_event_times]
                if self.reference_event_times is not None
                else None
            ),
        }
        return out

    def summary(self, alpha: Optional[float] = None) -> str:
        """Return a formatted event-study table.

        Parameters
        ----------
        alpha : float, optional
            Accepted for signature uniformity (spec section 5). The stored
            confidence columns were computed at fit time; passing a value
            different from the stored ``alpha`` raises rather than silently
            recomputing or mislabeling - re-aggregate at the desired level
            instead.
        """
        if alpha is not None and alpha != self.alpha:
            raise ValueError(
                f"This event-study surface stores intervals computed at "
                f"alpha={self.alpha}; re-aggregate to obtain alpha={alpha} "
                "intervals (summary() never recomputes stored inference)."
            )
        ci_pct = int(round((1 - self.alpha) * 100))
        lines = [
            "Event-Study Effects",
            "=" * 78,
        ]
        meta_bits = []
        if self.source:
            meta_bits.append(f"source: {self.source}")
        meta_bits.append(f"time scale: {self.time_scale}")
        if self.event_time_convention:
            meta_bits.append(f"convention: {self.event_time_convention}")
        if self.n_kind:
            meta_bits.append(f"n counts: {self.n_kind}")
        if self.estimand:
            # A non-ATT per-horizon estimand (HAD's WAS family, M-027):
            # name it in the metadata and use it as the column heading -
            # the hard-coded ATT would silently relabel the numbers.
            meta_bits.append(f"estimand: {self.estimand}")
        lines.append("  ".join(meta_bits))
        lines.append("-" * 78)
        est_head = self.estimand or "ATT"
        est_w = max(10, len(est_head))
        lines.append(
            f"{'Event time':>12} {est_head:>{est_w}} {'SE':>10} {'t':>8} "
            f"{'P>|t|':>8} {f'[{ci_pct}% CI]':>21}"
        )
        for i in range(self.event_time.shape[0]):
            label = f"{self.event_time[i]}"
            if self.is_reference[i]:
                lines.append(f"{label:>12} {0.0:>{est_w}.4f} {'(reference)':>{50}}")
                continue
            lines.append(
                f"{label:>12} {self.att[i]:>{est_w}.4f} {self.se[i]:>10.4f} "
                f"{self.t_stat[i]:>8.3f} {self.p_value[i]:>8.3f} "
                f"[{self.conf_int_lower[i]:>9.4f}, {self.conf_int_upper[i]:>9.4f}]"
            )
        lines.append("=" * 78)
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Builders: native representation -> EventStudyResults (package-internal)
# ----------------------------------------------------------------------

#: Per-producer remediation for an absent event-study surface.
_ABSENT_SURFACE_HINTS: Dict[str, str] = {
    # Migrated to the post-fit surface (row M-020): no refit needed.
    "CallawaySantAnnaResults": "call results.aggregate('event_study')",
    "ImputationDiDResults": (
        "call results.aggregate('event_study') (on a bootstrapped fit, "
        "re-fit with n_bootstrap=0 or the deprecated fit-time aggregate=)"
    ),
    "TwoStageDiDResults": (
        "call results.aggregate('event_study') (on a bootstrapped fit, "
        "re-fit with n_bootstrap=0 or the deprecated fit-time aggregate=)"
    ),
    # Absence only possible on pre-3.9 pickles: 3.9+ fits always
    # materialize the surface (row M-024).
    "StackedDiDResults": "re-fit with diff-diff >= 3.9, which always computes the surface",
    "StaggeredTripleDiffResults": "refit with aggregate='event_study' (or 'all')",
    "EfficientDiDResults": "call results.aggregate('event_study') (on a bootstrapped fit, re-fit with n_bootstrap=0 or the deprecated fit-time aggregate=)",
    "ContinuousDiDResults": (
        "call results.aggregate('event_study') (on a bootstrapped fit, "
        "re-fit with n_bootstrap=0 or the deprecated fit-time aggregate=)"
    ),
    "WooldridgeDiDResults": "call results.aggregate('event_study') first",
    "SpilloverDiDResults": "refit with event_study=True",
    "ChaisemartinDHaultfoeuilleResults": "refit with L_max >= 1",
    "LPDiDResults": "refit with only_pooled=False",
}


def _absent(results: Any) -> ValueError:
    name = type(results).__name__
    hint = _ABSENT_SURFACE_HINTS.get(name, "request event-study aggregation at fit")
    return ValueError(f"{name} carries no event-study surface - {hint}.")


def _validate_vcov_subblock(
    sigma: np.ndarray,
    ses: np.ndarray,
    consumer: str,
    *,
    allow_singular: bool = True,
) -> np.ndarray:
    """Integrity checks for a consumer-bound covariance sub-block.

    Containers are publicly constructible, so the consumer boundary
    validates what the producers guarantee by construction: finite
    entries, symmetry, a diagonal equal to the stored ``se**2`` (the
    container contract clears ``vcov`` rather than ship a matrix whose
    diagonal disagrees with the stored SEs), and no material
    indefiniteness. Every tolerance is RELATIVE to the matrix scale - an
    absolute floor would wave through a materially indefinite low-scale
    matrix (e.g. diagonal 1e-10 with -1e-10 eigenvalues).

    ``allow_singular=False`` additionally rejects singular/near-singular
    sub-blocks: Rambachan-Roth's inference assumes eigenvalues bounded
    away from zero, so HonestDiD passes False; PreTrendsPower keeps its
    documented singular-covariance handling with the default True.
    """
    sigma = np.asarray(sigma, dtype=float)
    if not np.all(np.isfinite(sigma)):
        raise ValueError(
            f"{consumer}: the event-study container's covariance "
            "sub-block contains non-finite entries."
        )
    _scale = float(np.max(np.abs(sigma))) if sigma.size else 0.0
    if not np.allclose(sigma, sigma.T, rtol=1e-8, atol=1e-12 * _scale):
        raise ValueError(
            f"{consumer}: the event-study container's covariance " "sub-block is not symmetric."
        )
    ses_arr = np.asarray(ses, dtype=float)
    # Pure relative: retained rows have se > 0, so the diagonal target is
    # strictly positive and an absolute atol would mask low-scale
    # inconsistencies.
    if not np.allclose(np.diag(sigma), ses_arr**2, rtol=1e-6, atol=0.0):
        raise ValueError(
            f"{consumer}: the event-study container's covariance diagonal "
            "is inconsistent with the stored standard errors (the "
            "container contract clears vcov rather than ship a matrix "
            "whose diagonal disagrees with se**2)."
        )
    if sigma.size:
        eigs = np.linalg.eigvalsh((sigma + sigma.T) / 2.0)
        _eig_scale = float(np.max(np.abs(eigs)))
        if _eig_scale > 0.0 and float(eigs.min()) < -1e-8 * _eig_scale:
            raise ValueError(
                f"{consumer}: the event-study container's covariance "
                "sub-block is indefinite (most negative eigenvalue "
                f"{float(eigs.min()):.3e} at scale {_eig_scale:.3e}). "
                "Positive semi-definiteness is required."
            )
        if not allow_singular and _eig_scale > 0.0 and float(eigs.min()) < 1e-10 * _eig_scale:
            raise ValueError(
                f"{consumer}: the event-study container's covariance "
                "sub-block is singular or near-singular (smallest "
                f"eigenvalue {float(eigs.min()):.3e} at scale "
                f"{_eig_scale:.3e}). Rambachan-Roth inference assumes "
                "eigenvalues bounded away from zero; drop collinear "
                "horizons or re-estimate before running HonestDiD."
            )
    return sigma


def _resolve_scalar_df_survey(results: Any) -> Optional[float]:
    """Resolve the producer's SCALAR inference df for container provenance.

    Mirrors the fit-time consumers' preference order (honest_did):
    ``survey_metadata.df_survey`` where present, with a replicate design
    whose df is undefined mapping to the ``0.0`` sentinel (fails closed to
    NaN critical values downstream); else the bare-``cluster=``
    ``df_inference`` carrier; else None. Deliberate local sibling of
    ``aggregation.resolve_inference_df``: this module cannot import it
    (aggregation.py imports results_base - the dependency is one-way);
    folding the copies together is tracked in TODO.md (df-resolution /
    adapter-naming consolidation row).
    """
    sm = getattr(results, "survey_metadata", None)
    if sm is not None:
        df_survey = getattr(sm, "df_survey", None)
        if df_survey is not None:
            return float(df_survey)
        if getattr(sm, "replicate_method", None) is not None:
            return 0.0
    df_inference = getattr(results, "df_inference", None)
    if df_inference is not None:
        return float(df_inference)
    return None


def _provenance_kwargs(results: Any) -> Dict[str, Any]:
    """Producer-provenance fields threaded onto the container.

    ``getattr`` with a None default: producers that declare no
    ``base_period``/``anticipation`` notion yield None - values are never
    invented.
    """
    ref_e = getattr(results, "reference_event_times", None)
    return {
        "base_period": getattr(results, "base_period", None),
        "anticipation": getattr(results, "anticipation", None),
        "df_survey": _resolve_scalar_df_survey(results),
        "reference_event_times": tuple(ref_e) if ref_e is not None else None,
    }


def _empty_surface(results: Any) -> EventStudyResults:
    """Zero-row surface for a requested-but-empty event study."""
    empty_f = np.empty(0, dtype=float)
    return EventStudyResults(
        event_time=np.empty(0),
        att=empty_f.copy(),
        se=empty_f.copy(),
        t_stat=empty_f.copy(),
        p_value=empty_f.copy(),
        conf_int_lower=empty_f.copy(),
        conf_int_upper=empty_f.copy(),
        is_reference=np.empty(0, dtype=bool),
        n=empty_f.copy(),
        time_scale="relative",
        alpha=getattr(results, "alpha", 0.05),
        source=type(results).__name__,
        **_provenance_kwargs(results),
    )


def _materialize_cband(
    pairs: Dict[int, Tuple[float, float]], n_rows: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Turn sparse per-row band tuples into aligned arrays (None if empty)."""
    if not pairs:
        return None, None
    lo = np.full(n_rows, np.nan)
    hi = np.full(n_rows, np.nan)
    for i, (lo_i, hi_i) in pairs.items():
        lo[i] = lo_i
        hi[i] = hi_i
    return lo, hi


def _from_relative_dict(results: Any) -> EventStudyResults:
    """Adapter for ``event_study_effects: Dict[int, Dict]`` producers.

    Covers CallawaySantAnna, SunAbraham, ImputationDiD, TwoStageDiD,
    StackedDiD, SpilloverDiD, ContinuousDiD, EfficientDiD, Wooldridge
    (``att`` inner key normalized here), and StaggeredTripleDiff.

    Reference resolution is PRODUCER-AWARE, because a zero count is not a
    universal reference marker:

    1. If the producer names a ``reference_period`` that is PRESENT in the
       effects dict (SpilloverDiD materializes a zero row), that row is THE
       reference. Other zero-count rows - SpilloverDiD emits genuinely
       non-estimable rectangular horizons with ``n_obs == 0`` and NaN
       inference - are preserved as NON-reference NaN rows.
    2. If the producer names a ``reference_period`` that is ABSENT (an
       OMITTED baseline, e.g. SunAbraham's ``e = -1 - anticipation``), the
       anchor row is synthesized ONLY when the estimator reports it was
       genuinely observed (``reference_observed``) - never invented on a
       gapped grid where that period was never in the data.
    3. Otherwise the in-band zero-count sentinel marks the reference: a
       zero-count row is a reference iff its effect is exactly 0 (the
       normalization anchor). A NaN-effect zero-count row is a genuinely
       non-estimable horizon (TwoStageDiD emits effect=NaN, n_obs=0 for a
       horizon whose observations are all filtered) - preserved, never
       normalized. CallawaySantAnna universal base can mark several such
       rows; ContinuousDiD / Wooldridge carry no count key and yield none.
    """
    effects = getattr(results, "event_study_effects", None)
    if effects is None:
        raise _absent(results)  # not requested
    if len(effects) == 0:
        # Requested but no estimable horizons (e.g. EfficientDiD balance_e
        # eliminated every cohort). A zero-row surface, not an error.
        return _empty_surface(results)

    explicit_ref = getattr(results, "reference_period", None)

    # Synthesize an OMITTED-but-OBSERVED baseline before building arrays.
    effects = dict(effects)
    synthesized_ref = None
    if (
        explicit_ref is not None
        and explicit_ref not in effects
        and getattr(results, "reference_observed", False)
    ):
        synthesized_ref = explicit_ref
        effects[explicit_ref] = {
            "effect": 0.0,
            "se": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_int": (np.nan, np.nan),
        }

    keys = sorted(effects.keys())
    n_rows = len(keys)
    att = np.empty(n_rows)
    se = np.empty(n_rows)
    t_stat = np.empty(n_rows)
    p_value = np.empty(n_rows)
    ci_lo = np.empty(n_rows)
    ci_hi = np.empty(n_rows)
    n = np.full(n_rows, np.nan)
    zero_count = np.zeros(n_rows, dtype=bool)
    cband_pairs: Dict[int, Tuple[float, float]] = {}

    n_kind: Optional[str] = None
    for i, k in enumerate(keys):
        row = effects[k]
        att[i] = row["att"] if "att" in row else row["effect"]
        se[i] = row.get("se", np.nan)
        t_stat[i] = row.get("t_stat", np.nan)
        p_value[i] = row.get("p_value", np.nan)
        ci = row.get("conf_int")
        if ci is not None:
            ci_lo[i], ci_hi[i] = float(ci[0]), float(ci[1])
        else:
            ci_lo[i] = ci_hi[i] = np.nan
        if "n_groups" in row:
            n_kind = "groups"
            n[i] = float(row["n_groups"])
            zero_count[i] = row["n_groups"] == 0
        elif "n_obs" in row:
            n_kind = "obs"
            n[i] = float(row["n_obs"])
            zero_count[i] = row["n_obs"] == 0

        cband = row.get("cband_conf_int")
        if cband is not None:
            cband_pairs[i] = (float(cband[0]), float(cband[1]))

    event_time = np.asarray(keys)
    ref_present = explicit_ref is not None and (
        synthesized_ref is not None or explicit_ref in set(event_time.tolist())
    )
    if ref_present:
        # Named reference that is materialized (Spillover) or synthesized
        # (SunAbraham observed): mark exactly that row.
        is_ref = event_time == explicit_ref
    elif explicit_ref is not None:
        # Named reference that is absent and NOT synthesized (SunAbraham on a
        # gapped grid where the anchor was never observed): no reference row.
        is_ref = np.zeros(n_rows, dtype=bool)
    else:
        # In-band zero-count sentinel. A zero-count row is the REFERENCE only
        # when its effect is exactly 0 (the normalization anchor). A NaN-effect
        # zero-count row is a genuinely non-estimable horizon - preserved. A
        # zero-count row with a finite NONZERO effect is malformed - fail
        # loudly rather than silently rewrite it.
        finite = np.isfinite(att)
        bad = zero_count & finite & (att != 0.0)
        if bad.any():
            raise ValueError(
                f"{type(results).__name__} event-study surface has a zero-count "
                f"row with a finite nonzero effect at {event_time[bad].tolist()}; "
                "a reference/normalization row must have effect 0.0."
            )
        is_ref = zero_count & finite

    cband_lo, cband_hi = _materialize_cband(cband_pairs, n_rows)

    vcov = getattr(results, "event_study_vcov", None)
    vcov_index = getattr(results, "event_study_vcov_index", None)
    if vcov is None or vcov_index is None:
        vcov = vcov_index = None

    # Per-row df provenance. Primary channel: `event_study_df` (scalar or
    # {event_time: df} dict of the values actually passed to safe_inference;
    # producers clear it when bootstrap overrides the stored inference).
    # Fallback: CallawaySantAnna's bare-cluster `df_inference` - GATED on no
    # bootstrap, because bare-cluster fits populate df_inference=G-1 even
    # when n_bootstrap>0 replaced the stored ES p/CIs with percentile values
    # that never used that df (and bootstrap p-values are finite, so the
    # container's NaN-p masking cannot catch it).
    df_src: Any = getattr(results, "event_study_df", None)
    if df_src is None and getattr(results, "bootstrap_results", None) is None:
        df_src = getattr(results, "df_inference", None)
    df_arg: Optional[Union[float, np.ndarray]]
    if isinstance(df_src, dict):
        df_arg = np.array([float(df_src[k]) if k in df_src else np.nan for k in keys])
    else:
        df_arg = df_src

    return EventStudyResults(
        event_time=event_time,
        att=att,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        conf_int_lower=ci_lo,
        conf_int_upper=ci_hi,
        is_reference=is_ref,
        n=n,
        n_kind=n_kind,
        time_scale="relative",
        event_time_convention="e0_first_treated",
        vcov=np.asarray(vcov, dtype=float) if vcov is not None else None,
        vcov_index=np.asarray(vcov_index) if vcov_index is not None else None,
        cband_lower=cband_lo,
        cband_upper=cband_hi,
        cband_crit_value=getattr(results, "cband_crit_value", None),
        alpha=getattr(results, "alpha", 0.05),
        source=type(results).__name__,
        df=df_arg,
        **_provenance_kwargs(results),
    )


def _from_mpd(results: Any) -> EventStudyResults:
    """Adapter for MultiPeriodDiD's calendar-keyed ``period_effects``.

    The reference period is omitted from the native dict and recorded on
    ``results.reference_period``; the marked row is synthesized here
    (``att=0.0``, NaN inference). The event-study vcov is the sub-block of
    ``results.vcov`` selected by ``results.interaction_indices`` - pure
    indexing of stored entries, no recomputation.
    """
    period_effects = getattr(results, "period_effects", None)
    if not period_effects:
        raise _absent(results)

    ref = getattr(results, "reference_period", None)
    periods = sorted(period_effects.keys())
    all_periods = sorted(periods + [ref]) if ref is not None else periods

    n_rows = len(all_periods)
    att = np.full(n_rows, np.nan)
    se = np.full(n_rows, np.nan)
    t_stat = np.full(n_rows, np.nan)
    p_value = np.full(n_rows, np.nan)
    ci_lo = np.full(n_rows, np.nan)
    ci_hi = np.full(n_rows, np.nan)
    n = np.full(n_rows, np.nan)
    is_ref = np.zeros(n_rows, dtype=bool)

    for i, p in enumerate(all_periods):
        if ref is not None and p == ref:
            is_ref[i] = True
            continue
        pe = period_effects[p]
        att[i] = pe.effect
        se[i] = pe.se
        t_stat[i] = pe.t_stat
        p_value[i] = pe.p_value
        ci_lo[i], ci_hi[i] = float(pe.conf_int[0]), float(pe.conf_int[1])

    vcov_sub: Optional[np.ndarray] = None
    vcov_index_arr: Optional[np.ndarray] = None
    full_vcov = getattr(results, "vcov", None)
    interaction_indices = getattr(results, "interaction_indices", None)
    if full_vcov is not None and interaction_indices:
        covered = [p for p in periods if p in interaction_indices]
        if covered:
            idx = [interaction_indices[p] for p in covered]
            vcov_sub = np.asarray(full_vcov, dtype=float)[np.ix_(idx, idx)]
            vcov_index_arr = np.asarray(covered)

    # Per-period df: STRICTLY the `event_study_df` channel ({period: df}
    # actually passed to safe_inference - per-period BM DOF under hc2_bm).
    # `inference_df` is deliberately NOT a fallback here: it stores the
    # POST-AVERAGE contrast df, which is the wrong provenance for the
    # per-period rows under hc2_bm.
    df_map = getattr(results, "event_study_df", None)
    df_arg: Optional[np.ndarray] = None
    if isinstance(df_map, dict):
        df_arg = np.array([float(df_map[p]) if p in df_map else np.nan for p in all_periods])

    # Calendar-partition provenance (M-092 amendment #5): the producer's
    # AUTHORITATIVE post_periods list - an arbitrary subset is legal on
    # MultiPeriodDiD, so consumers must never re-derive the partition
    # positionally from the reference row.
    _mpd_post = getattr(results, "post_periods", None)
    post_periods_arg: Optional[Tuple[Any, ...]] = tuple(_mpd_post) if _mpd_post else None

    return EventStudyResults(
        event_time=np.asarray(all_periods),
        att=att,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        conf_int_lower=ci_lo,
        conf_int_upper=ci_hi,
        is_reference=is_ref,
        n=n,
        n_kind=None,
        reference_period=ref,
        time_scale="calendar",
        event_time_convention=None,
        vcov=vcov_sub,
        vcov_index=vcov_index_arr,
        alpha=getattr(results, "alpha", 0.05),
        source=type(results).__name__,
        df=df_arg,
        post_periods=post_periods_arg,
        **_provenance_kwargs(results),
    )


def _from_lpdid(results: Any) -> EventStudyResults:
    """Adapter for LPDiD's per-horizon ``event_study`` DataFrame.

    Native columns ``horizon``/``coefficient``/``conf_low``/``conf_high``
    map onto the canonical names; the materialized ``horizon == -1`` base
    row (coefficient 0.0, NaN inference) translates to the marked
    reference row. The per-horizon ``n_clusters`` column stays on the
    native frame only (``n_kind="obs"``).
    """
    frame = getattr(results, "event_study", None)
    if frame is None or len(frame) == 0:
        raise _absent(results)

    frame = frame.sort_values("horizon")
    horizons = frame["horizon"].to_numpy()
    # copy=True: pandas may hand back a read-only view, and the reference
    # normalization writes into these arrays.
    att = np.array(frame["coefficient"], dtype=float)
    se = np.array(frame["se"], dtype=float)
    t_stat = np.array(frame["t_stat"], dtype=float)
    p_value = np.array(frame["p_value"], dtype=float)
    ci_lo = np.array(frame["conf_low"], dtype=float)
    ci_hi = np.array(frame["conf_high"], dtype=float)
    n = np.array(frame["n_obs"], dtype=float)
    is_ref = horizons == -1

    # Per-horizon df from the producer's `event_study_df` channel ({horizon:
    # df actually passed to safe_inference} - cluster df or per-horizon
    # survey df; NOT re-derivable from the frame, whose n_clusters column
    # cannot recover the survey n_PSU - n_strata rule or the vcov-None
    # guard). The synthetic horizon == -1 base row has no entry -> NaN.
    df_map = getattr(results, "event_study_df", None)
    df_arg: Optional[np.ndarray] = None
    if isinstance(df_map, dict):
        df_arg = np.array([float(df_map.get(int(h), np.nan)) for h in horizons])

    return EventStudyResults(
        event_time=horizons,
        att=att,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        conf_int_lower=ci_lo,
        conf_int_upper=ci_hi,
        is_reference=is_ref,
        n=n,
        n_kind="obs",
        time_scale="relative",
        event_time_convention="e0_first_treated",
        alpha=getattr(results, "alpha", 0.05),
        source=type(results).__name__,
        df=df_arg,
        **_provenance_kwargs(results),
    )


def _from_dcdh(results: Any) -> EventStudyResults:
    """Adapter for de Chaisemartin-D'Haultfoeuille's split dicts.

    Merges ``placebo_event_study`` (negative keys), a synthesized
    reference at 0, and ``event_study_effects`` (post horizons l >= 1,
    l=1 = instantaneous effect) - the same merge the event-study plotter
    performs. Horizons keep the paper's convention
    (``event_time_convention="l1_first_switch"``); they are documented,
    never renumbered.
    """
    effects = getattr(results, "event_study_effects", None)
    if not effects:
        raise _absent(results)
    placebos = getattr(results, "placebo_event_study", None) or {}

    keys: List[Any] = sorted(placebos.keys()) + [0] + sorted(effects.keys())
    n_rows = len(keys)
    att = np.full(n_rows, np.nan)
    se = np.full(n_rows, np.nan)
    t_stat = np.full(n_rows, np.nan)
    p_value = np.full(n_rows, np.nan)
    ci_lo = np.full(n_rows, np.nan)
    ci_hi = np.full(n_rows, np.nan)
    n = np.full(n_rows, np.nan)
    is_ref = np.zeros(n_rows, dtype=bool)
    cband_pairs: Dict[int, Tuple[float, float]] = {}

    for i, k in enumerate(keys):
        if k == 0:
            is_ref[i] = True
            continue
        row = placebos[k] if k < 0 else effects[k]
        att[i] = row["effect"]
        se[i] = row.get("se", np.nan)
        t_stat[i] = row.get("t_stat", np.nan)
        p_value[i] = row.get("p_value", np.nan)
        ci = row.get("conf_int")
        if ci is not None:
            ci_lo[i], ci_hi[i] = float(ci[0]), float(ci[1])
        if "n_obs" in row:
            n[i] = float(row["n_obs"])
        cband = row.get("cband_conf_int")
        if cband is not None:
            cband_pairs[i] = (float(cband[0]), float(cband[1]))

    cband_lo, cband_hi = _materialize_cband(cband_pairs, n_rows)

    # Simultaneous-band critical value lives on sup_t_bands (populated only on
    # a bootstrap fit); carry it so the band's provenance is not lost.
    sup_t = getattr(results, "sup_t_bands", None)
    cband_crit = sup_t.get("crit_value") if isinstance(sup_t, dict) else None

    # dCDH stores its count under the legacy "n_obs" key, but the UNIT is
    # L_max-dependent (never observations): with L_max >= 1 it is N_l, the
    # number of eligible switcher GROUPS per horizon; with L_max is None it is
    # N_S, the number of switching (g, t) CELLS (one group can contribute
    # several). Label each accurately so downstream sample-size logic cannot
    # conflate them.
    l_max = getattr(results, "L_max", None)
    dcdh_n_kind = "groups" if (l_max is not None and l_max >= 1) else "switcher_cells"

    return EventStudyResults(
        event_time=np.asarray(keys),
        att=att,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        conf_int_lower=ci_lo,
        conf_int_upper=ci_hi,
        is_reference=is_ref,
        n=n,
        n_kind=dcdh_n_kind,
        time_scale="relative",
        event_time_convention="l1_first_switch",
        cband_lower=cband_lo,
        cband_upper=cband_hi,
        cband_crit_value=cband_crit,
        alpha=getattr(results, "alpha", 0.05),
        source=type(results).__name__,
        # ONE scalar df for both merged surfaces: the effect rows and the
        # placebo rows are computed from the same design df (and, under
        # replicate weights, refreshed together to the final effective df),
        # so a scalar is faithful - the container broadcasts it and NaN-masks
        # the synthesized reference row at 0 plus any NaN-p rows.
        df=getattr(results, "event_study_df", None),
        **_provenance_kwargs(results),
    )


def _from_had(results: Any) -> EventStudyResults:
    """Adapter for HAD's columnar event-study container (near-passthrough).

    The anchor horizon e = -1 is excluded from the native surface by
    design (trivially zero, WAS not identified there), so this producer
    legitimately has no reference row.
    """
    event_times = getattr(results, "event_times", None)
    if event_times is None or len(event_times) == 0:
        raise _absent(results)

    n_rows = len(event_times)
    n_obs = getattr(results, "n_obs_per_horizon", None)
    cband_lo = getattr(results, "cband_low", None)
    cband_hi = getattr(results, "cband_high", None)

    return EventStudyResults(
        event_time=np.array(event_times),
        att=np.array(results.att, dtype=float),
        se=np.array(results.se, dtype=float),
        t_stat=np.array(results.t_stat, dtype=float),
        p_value=np.array(results.p_value, dtype=float),
        conf_int_lower=np.array(results.conf_int_low, dtype=float),
        conf_int_upper=np.array(results.conf_int_high, dtype=float),
        is_reference=np.zeros(n_rows, dtype=bool),
        # "units", not "obs": ``n_obs_per_horizon`` counts the UNITS
        # contributing at each event time (it equals ``n_units`` at every
        # horizon under the no-NaN validator) - the field docstring says
        # so, and N_KIND_VOCABULARY forbids conflating the two kinds.
        # Corrected with row M-027 (the value previously read "obs").
        n=(np.asarray(n_obs, dtype=float) if n_obs is not None else np.full(n_rows, np.nan)),
        n_kind="units" if n_obs is not None else None,
        time_scale="relative",
        event_time_convention="e0_first_treated",
        cband_lower=np.asarray(cband_lo, dtype=float) if cband_lo is not None else None,
        cband_upper=np.asarray(cband_hi, dtype=float) if cband_hi is not None else None,
        cband_crit_value=getattr(results, "cband_crit_value", None),
        alpha=getattr(results, "alpha", 0.05),
        source=type(results).__name__,
        # The per-horizon estimand is a WAS, not an ATT (row M-027):
        # relay the estimand label so summary()/to_dict never mislabel.
        estimand=getattr(results, "target_parameter", None),
        **_provenance_kwargs(results),
    )


def build_event_study_surface(results: Any) -> EventStudyResults:
    """Build the unified event-study surface from a fitted results object.

    Package-internal (spec section 5 / row M-092): public exposure rides
    ``results.aggregate(type="event_study")`` (Phase 2 PR (b)); the Phase 3
    merged TwoWayFixedEffects event-study mode returns the container
    natively. Values are copied bit-exactly from the native surface;
    an absent surface raises ``ValueError`` naming the call that produces
    it (never a silent empty container).
    """
    # Order matters: dCDH carries both event_study_effects and the split
    # placebo dict; HAD's columnar container and MPD's period_effects have
    # unique attributes; LPDiD's surface is a DataFrame field.
    if hasattr(results, "placebo_event_study") and hasattr(results, "event_study_effects"):
        return _from_dcdh(results)
    if hasattr(results, "event_times") and hasattr(results, "n_obs_per_horizon"):
        return _from_had(results)
    if hasattr(results, "period_effects") and hasattr(results, "interaction_indices"):
        return _from_mpd(results)
    if hasattr(results, "pooled") and hasattr(results, "event_study"):
        return _from_lpdid(results)
    if hasattr(results, "event_study_effects"):
        # SunAbraham (omitted-but-observed baseline) and SpilloverDiD
        # (materialized reference) both flow through here via reference_period.
        return _from_relative_dict(results)
    raise TypeError(
        f"{type(results).__name__} does not expose an event-study surface. "
        "Supported producers: CallawaySantAnna, SunAbraham, ImputationDiD, "
        "TwoStageDiD, StackedDiD, SpilloverDiD, ContinuousDiD, EfficientDiD, "
        "WooldridgeDiD, StaggeredTripleDifference, MultiPeriodDiD, LPDiD, "
        "ChaisemartinDHaultfoeuille, and HeterogeneousAdoptionDiD "
        "event-study results."
    )
