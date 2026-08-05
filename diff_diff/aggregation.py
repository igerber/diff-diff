"""Post-fit aggregation surface (4.0 program Phase 2, spec section 6).

``results.aggregate(type=...)`` replaces the fit-time ``fit(aggregate=)``
argument (ledger rows M-020..M-027): estimate once, aggregate as a post-fit
step - the ecosystem's strongest norm (``did::aggte``, ``etwfe::emfx``, Stata
``estat aggregation``).

Three pieces live here:

- :class:`AggregationResult` - the tabular container every non-event-study
  aggregation returns (row M-122). ``type="event_study"`` returns the
  :class:`~diff_diff.results_base.EventStudyResults` container instead, which
  is where that surface's public exposure finally lands (row M-092).
- :class:`AggregationKit` - the compact per-estimator payload retained at fit
  time so re-aggregation needs neither a refit nor the source frame.
- :class:`AggregationMixin` - shared validation and dispatch, applied PER
  RESULTS CLASS.

The mixin is deliberately NOT mixed into
:class:`~diff_diff.results_base.BaseResults`: ``test_v4_matrix``'s
``resolve_locator`` uses ``inspect.getattr_static``, which walks the MRO, so a
base-class ``aggregate`` would make every still-``planned`` ledger row's
``new`` locator resolve and fail ``test_row_matches_reality``. A results class
gains the mixin in the same diff that flips its row.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results_base import N_KIND_VOCABULARY, BaseResults, _json_safe_label

__all__ = ["AggregationResult", "AGGREGATION_SCHEMA"]


#: Closed aggregation vocabulary (spec section 6). Per-estimator support is a
#: SUBSET of this set - an estimator asked for a type it does not implement
#: raises ``ValueError`` naming what it does support, never silently falling
#: back. Estimator-specific extras (e.g. ContinuousDiD's ``"dose"``) are
#: declared by that estimator's results class, not added here.
AGGREGATION_VOCABULARY: Tuple[str, ...] = (
    "simple",
    "event_study",
    "group",
    "calendar",
)

#: Pinned column schema of ``AggregationResult.to_dataframe()`` - identical for
#: every producer, mirroring ``EVENT_STUDY_SCHEMA``'s contract (spec section 5).
AGGREGATION_SCHEMA: Tuple[str, ...] = (
    "level",
    "label",
    "target",
    "att",
    "se",
    "t_stat",
    "p_value",
    "conf_int_lower",
    "conf_int_upper",
    "n",
    "weight",
    "df",
)


def resolve_inference_df(results: Any) -> Optional[float]:
    """Return the degrees of freedom that governed ``results``' overall statistic.

    Reads the carriers in the order the library documents on
    ``CallawaySantAnnaResults.df_inference``: ``survey_metadata.df_survey``
    first - it holds the actual CS-internal df including any post-resolve
    tightening for replicate designs - and ``df_inference`` only as the
    fallback for bare-``cluster=`` fits, where ``survey_metadata`` is
    intentionally ``None`` to preserve the survey/non-survey contract.
    Reading ``df_inference`` first would overstate the denominator df on
    panel survey fits whose df was tightened during aggregation.

    A replicate design whose df is undefined resolves to the sentinel ``0``,
    which yields NaN inference rather than a t-reference.

    ``AggregationResult.df`` is per-row PROVENANCE - the df actually passed to
    ``safe_inference`` for that row's stored p-value/CI - so it must come from
    here rather than from whichever df field happens to be populated.
    """
    if getattr(results, "survey_metadata", None) is not None:
        sm = results.survey_metadata
        df_survey = getattr(sm, "df_survey", None)
        if df_survey is None and getattr(sm, "replicate_method", None) is not None:
            return 0.0  # undefined replicate df -> NaN inference
        if df_survey is not None:
            return float(df_survey)
    df_inference = getattr(results, "df_inference", None)
    if df_inference is not None:
        return float(df_inference)
    return None


def _sortable(labels: np.ndarray) -> bool:
    """Can ``labels`` be ordered without raising?

    Cohort labels are usually numeric, but the column is object-dtype in
    general and may be mixed-type, where a naive ``argsort`` raises
    ``TypeError``. Producer order is preserved in that case rather than
    guessing an order.
    """
    try:
        np.argsort(labels, kind="stable")
    except TypeError:
        return False
    return True


@dataclass
class AggregationResult(BaseResults):
    """One post-fit aggregation, as a table (spec section 6, row M-122).

    Columnar arrays index-aligned to ``label``. Values are computed by the
    producing estimator's aggregation machinery and stored here verbatim -
    this container never re-derives inference.

    Parameters
    ----------
    level : str
        The aggregation type that produced this table - one of
        :data:`AGGREGATION_VOCABULARY` or a documented per-estimator extra.
    label : np.ndarray
        Per-row aggregation key: the cohort for ``"group"``, the calendar
        period for ``"calendar"``, the dose for ``"dose"``. A single
        ``"overall"`` entry for ``"simple"``.
    target : np.ndarray
        Per-row estimand discriminator, so one container can carry two
        aligned estimands over the same labels (ContinuousDiD's ATT(d) and
        ACRT(d) become 2N rows). ``"att"`` where an estimator has one.
    att, se, t_stat, p_value : np.ndarray
        The canonical quintet, per row, carrying WHATEVER inference the fit
        stored - never recomputed. On a bootstrapped fit that usually means
        the producer's percentile-bootstrap statistics carried through
        unchanged; view-relay producers can mix regimes per row where the
        fit itself did (dCDH's ``L_max >= 2`` cost-benefit delta keeps
        analytical ``safe_inference`` even under ``n_bootstrap > 0`` - see
        the REGISTRY Phase 2 cost-benefit delta SE note).
    conf_int_lower, conf_int_upper : np.ndarray
        Interval bounds at the fit's ``alpha``.
    n : np.ndarray
        Per-row count as float, NaN where the producer records none. Its
        SEMANTIC is ``n_kind`` - never assume units.
    n_kind : str or None
        Semantic of ``n``, from
        :data:`~diff_diff.results_base.N_KIND_VOCABULARY` - the SAME closed
        vocabulary :class:`~diff_diff.results_base.EventStudyResults` draws
        on, so a consumer can route on ``n_kind`` across both containers.
        ``None`` when the producer records no count.
    weight : np.ndarray or None
        Normalized aggregation mass per row, summing to 1 within one
        ``(level, target)`` group. ``None`` where no per-row mass exists -
        CallawaySantAnna's ``"group"`` aggregation weights ``(g, t)`` cells
        equally WITHIN each cohort and has no cross-cohort mass, so inventing
        one would be a fabricated number.
    df : np.ndarray
        Per-row inference degrees of freedom, NaN where none governed the
        stored p-value. NaN on percentile-bootstrap rows (no df governs
        them); a bootstrapped fit's rows can still carry a finite df where
        the fit kept analytical inference for that row (the dCDH delta
        case above).
    alpha : float
        Significance level the interval was computed at.
    estimator : str or None
        Producing estimator class name, for provenance in ``summary()``.
    """

    level: str
    label: np.ndarray
    target: np.ndarray
    att: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    conf_int_lower: np.ndarray
    conf_int_upper: np.ndarray
    n: np.ndarray
    df: np.ndarray
    alpha: float = 0.05
    n_kind: Optional[str] = None
    weight: Optional[np.ndarray] = None
    estimator: Optional[str] = None

    _COLUMN_FIELDS: Tuple[str, ...] = field(
        default=(
            "label",
            "target",
            "att",
            "se",
            "t_stat",
            "p_value",
            "conf_int_lower",
            "conf_int_upper",
            "n",
            "df",
        ),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.label = np.asarray(self.label, dtype=object)
        # Shape check BEFORE reading shape[0]: a 0-d label has an empty shape
        # tuple, so indexing it first raises IndexError instead of the
        # documented ValueError.
        if self.label.ndim != 1:
            raise ValueError(
                f"AggregationResult label must be one-dimensional; got shape {self.label.shape}."
            )
        n_rows = self.label.shape[0]

        for name in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper", "n"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (n_rows,):
                raise ValueError(
                    f"AggregationResult field {name!r} has shape {arr.shape}; "
                    f"expected ({n_rows},) to align with label."
                )
            setattr(self, name, arr)

        target = np.asarray(self.target, dtype=object)
        if target.shape != (n_rows,):
            raise ValueError(
                f"AggregationResult target has shape {target.shape}; "
                f"expected ({n_rows},) - it is a PER-ROW discriminator, not a scalar."
            )
        self.target = target

        # df: scalar broadcasts across rows (the EventStudyResults convention);
        # NaN wherever no df governed the stored p-value.
        if self.df is None:
            df_arr = np.full(n_rows, np.nan)
        elif np.ndim(self.df) == 0:
            df_arr = np.full(n_rows, float(self.df))
        else:
            # np.array (not asarray) so the NaN-out below cannot write through
            # to a caller-owned array or fail on a read-only buffer - matching
            # EventStudyResults' normalization of the same field.
            df_arr = np.array(self.df, dtype=float)
            if df_arr.shape != (n_rows,):
                raise ValueError(
                    f"AggregationResult df has shape {df_arr.shape}; expected ({n_rows},) or scalar."
                )
        df_arr[~np.isfinite(self.p_value)] = np.nan
        self.df = df_arr

        if self.weight is not None:
            w = np.asarray(self.weight, dtype=float)
            if w.shape != (n_rows,):
                raise ValueError(
                    f"AggregationResult weight has shape {w.shape}; expected ({n_rows},) or None."
                )
            self.weight = w

        # n_kind is a routing key consumers share with EventStudyResults, so
        # an off-vocabulary value is a contract break, not a free-form label.
        if self.n_kind is not None and self.n_kind not in N_KIND_VOCABULARY:
            raise ValueError(
                f"AggregationResult n_kind {self.n_kind!r} is not in the shared "
                f"vocabulary {N_KIND_VOCABULARY}."
            )

    # ------------------------------------------------------------------ #
    # Serialization (spec section 5: every main results class has all three)
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Return the pinned :data:`AGGREGATION_SCHEMA` columns, in order.

        Rows are ordered by ``label`` when the labels are homogeneously
        sortable, and in producer order otherwise (mixed-type cohort labels
        cannot be ordered without raising). Heterogeneous-``target``
        containers (ContinuousDiD's att/acrt, row M-025) order by
        FIRST-APPEARANCE target blocks instead - producer order, NOT
        lexicographic, which would put ``"acrt"`` before ``"att"`` - with
        labels ascending within each block under the same sortability
        guard.
        """
        data: Dict[str, Any] = {
            "level": self.level,
            "label": self.label,
            "target": self.target,
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int_lower,
            "conf_int_upper": self.conf_int_upper,
            "n": self.n,
            "weight": self.weight if self.weight is not None else np.nan,
            "df": self.df,
        }
        frame = pd.DataFrame(data, columns=list(AGGREGATION_SCHEMA))
        distinct_targets = list(dict.fromkeys(self.target))
        if len(distinct_targets) > 1 and len(frame) > 1:
            rank = {t: i for i, t in enumerate(distinct_targets)}
            target_rank = np.array([rank[t] for t in self.target])
            if _sortable(self.label):
                # lexsort: LAST key is primary - target blocks first,
                # labels ascending within each block.
                order = np.lexsort((self.label, target_rank))
            else:
                order = np.argsort(target_rank, kind="stable")
            frame = frame.iloc[order].reset_index(drop=True)
        elif len(frame) > 1 and _sortable(self.label):
            order = np.argsort(self.label, kind="stable")
            frame = frame.iloc[order].reset_index(drop=True)
        return frame

    def to_dict(self) -> Dict[str, Any]:
        """Canonical-name mapping (deprecated names never leak into output)."""
        out: Dict[str, Any] = {
            "level": self.level,
            "alpha": self.alpha,
            "n_kind": self.n_kind,
            "estimator": self.estimator,
            "label": [_json_safe_label(v) for v in self.label],
            "target": [str(v) for v in self.target],
        }
        for name in (
            "att",
            "se",
            "t_stat",
            "p_value",
            "conf_int_lower",
            "conf_int_upper",
            "n",
            "df",
        ):
            out[name] = [float(v) for v in getattr(self, name)]
        out["weight"] = None if self.weight is None else [float(v) for v in self.weight]
        return out

    def summary(self, alpha: Optional[float] = None) -> str:
        """Human-readable table.

        Parameters
        ----------
        alpha : float, optional
            Accepted for signature uniformity (spec section 5). The stored
            interval was computed at aggregation time; passing a value
            different from the stored ``alpha`` raises rather than silently
            recomputing or mislabeling it - re-aggregate instead. Mirrors
            :meth:`~diff_diff.results_base.EventStudyResults.summary`.
        """
        if alpha is not None and alpha != self.alpha:
            raise ValueError(
                f"This aggregation stores intervals computed at alpha={self.alpha}; "
                f"re-aggregate to obtain alpha={alpha} intervals "
                "(summary() never recomputes stored inference)."
            )
        who = self.estimator or "estimator"
        lines = [
            f"{who} - aggregate(type={self.level!r})",
            "=" * 64,
        ]
        if len(self.label) == 0:
            lines.append("(no rows - the aggregation selected no cells)")
            return "\n".join(lines)

        n_label = "n" if self.n_kind is None else f"n[{self.n_kind}]"
        frame = self.to_dataframe()
        if len(dict.fromkeys(self.target)) > 1:
            # Heterogeneous targets (ContinuousDiD's att/acrt, row M-025):
            # a target column disambiguates the duplicate labels and the
            # estimate heading goes neutral - the hard-coded 'ATT' would
            # mislabel every acrt row. Uniform-target containers render
            # exactly as before (byte-stable).
            lines.append(
                f"{'label':>14} {'target':>8} {'estimate':>11} {'SE':>10} "
                f"{'t':>8} {'p':>8} {n_label:>10}"
            )
            lines.append("-" * 73)
            for _, row in frame.iterrows():
                n_disp = "" if not np.isfinite(row["n"]) else f"{row['n']:.0f}"
                lines.append(
                    f"{str(row['label']):>14} {str(row['target']):>8} "
                    f"{row['att']:>11.4f} {row['se']:>10.4f} "
                    f"{row['t_stat']:>8.3f} {row['p_value']:>8.4f} {n_disp:>10}"
                )
            lines.append("-" * 73)
        else:
            lines.append(f"{'label':>14} {'ATT':>11} {'SE':>10} {'t':>8} {'p':>8} {n_label:>10}")
            lines.append("-" * 64)
            for _, row in frame.iterrows():
                n_disp = "" if not np.isfinite(row["n"]) else f"{row['n']:.0f}"
                lines.append(
                    f"{str(row['label']):>14} {row['att']:>11.4f} {row['se']:>10.4f} "
                    f"{row['t_stat']:>8.3f} {row['p_value']:>8.4f} {n_disp:>10}"
                )
            lines.append("-" * 64)
        lines.append(f"Confidence intervals at alpha={self.alpha}.")
        if self.weight is None:
            lines.append("Per-row aggregation weights are not defined for this level.")
        return "\n".join(lines)


@dataclass
class AggregationKit:
    """Compact fit-time payload enabling post-fit re-aggregation.

    Built and attached DURING ``fit()`` - neither the estimator's
    ``precomputed`` bookkeeping nor its influence-function payload survives
    the call, so this cannot be reconstructed afterwards.

    Deliberately excludes the data matrices (``outcome_matrix``,
    ``covariate_matrix``, ``obs_outcome``, ``obs_covariates``): re-aggregation
    reads only unit-level bookkeeping, so the source panel is never retained.

    PANEL-BACKED EXCEPTION (ImputationDiD [M-021] and TwoStageDiD [M-022]):
    their recompute is target-specific - a different ``balance_e`` re-masks
    which observations enter each estimand and re-solves the variance from
    the panel + FE model - so no compact influence payload can replace the
    frame, and their kits' ``bookkeeping`` DOES retain panel objects.
    ImputationDiD references the SAME per-fit objects ``_fit_data`` already
    retains for ``pretrend_test()`` (zero marginal memory; pickles unchanged
    via memoization). TwoStageDiD retains a column-subset copy of its
    working frame - the first new panel retention, O(n_obs), documented in
    its ledger row and REGISTRY Note. For those two, ``influence`` is empty
    by design and the exclusion above applies to everything OUTSIDE the
    enumerated bookkeeping payload.

    PRUNED-PAYLOAD VARIANT (ContinuousDiD [M-025]): its event-study
    recompute needs per-(g, t) IF INGREDIENTS rather than a per-unit EIF
    dict or the panel - ``bookkeeping`` retains a pruned
    ``_bootstrap_info`` subset (treated/control indices, ``delta_y_treated``,
    ``ee_control``, masses, the covariate-path ``if_att_glob``), O(n_treated
    + n_control) per cell, plus unit-level arrays and - on survey fits -
    the PANEL-LEVEL ``ResolvedSurveyDesign`` (the recompute performs the
    unit collapse itself; on replicate designs this carries the
    (n_obs x R) replicate matrix). The K-dimensional spline machinery
    (bread, ``ee_treated``, ``Psi_eval``, ``dPsi_*``) is NOT retained;
    bootstrap fits retain scalars only (their event-study route fails
    closed). ``influence`` is empty by design here too. The ``simple`` /
    ``dose`` levels are pure views over stored public results fields and
    read no kit at all.

    Attributes
    ----------
    bookkeeping : dict
        The aggregation-relevant subset of the estimator's ``precomputed``
        mapping. O(n_units) on panel fits; several entries are
        observation-length on repeated cross-sections, where
        ``all_units = np.arange(n_obs)`` by construction.
    influence : dict
        Per-``(g, t)`` influence-function payload. The DOMINANT retained
        object, roughly O(n_units x n_gt).
    alpha, anticipation : float, int
        The only two estimator attributes the aggregation machinery reads.
        Carried explicitly so the aggregators need no estimator reference.
    cband : bool
        Whether the fit requested simultaneous bands. Retained because
        ``cband_crit_value`` is ``None`` both when bands were disabled and
        when no aggregation ran, so it cannot distinguish the two.
    bootstrap : AggregationKit.BootstrapReplaySpec or None
        Value-bound bootstrap replay description; ``None`` on analytical fits.
    """

    bookkeeping: Dict[str, Any]
    influence: Dict[Any, Any]
    alpha: float
    anticipation: int
    cband: bool
    bootstrap: Optional["BootstrapReplaySpec"] = None


@dataclass
class BootstrapReplaySpec:
    """Value-bound description of a bootstrap weight stream.

    Retaining the estimator's ``ReplayableWeightStream`` directly does not
    work: it stores a function-local closure (unpicklable) whose body reads
    ``self.n_bootstrap`` / ``self.bootstrap_weights`` LAZILY, so a post-fit
    ``set_params(n_bootstrap=...)`` silently changes - and can truncate - the
    replayed stream.

    This records the generator state plus the parameters BY VALUE and rebuilds
    the stream through a module-level factory, which replays bit-identically,
    pickles, and is immune to later mutation of the estimator.
    """

    bitgen_state: Dict[str, Any]
    n_bootstrap: int
    n_units: int
    weight_type: str
    block_size: Optional[int] = None
    expand_index: Optional[np.ndarray] = None

    def rebuild(self) -> Any:
        """Reconstruct the replayable weight stream."""
        from diff_diff.bootstrap_chunking import ReplayableWeightStream

        rng = np.random.default_rng()
        rng.bit_generator.state = self.bitgen_state
        return ReplayableWeightStream(_make_weight_iter_from_spec(self), rng)


def _make_weight_iter_from_spec(spec: BootstrapReplaySpec) -> Any:
    """Module-level factory - never a local closure, so the spec stays picklable."""
    from diff_diff.bootstrap_chunking import iter_weight_blocks

    def _factory(rng: np.random.Generator) -> Any:
        return iter_weight_blocks(
            spec.n_bootstrap,
            spec.n_units,
            spec.weight_type,
            rng,
            expand_index=spec.expand_index,
            block_size=spec.block_size,
        )

    return _factory


class AggregationMixin:
    """``aggregate(type=...)`` for one results class (spec section 6).

    Applied PER RESULTS CLASS - never to
    :class:`~diff_diff.results_base.BaseResults`, whose MRO position would
    make every still-``planned`` ledger row's ``new`` locator resolve.

    A results class opts in by setting :attr:`_AGGREGATE_SUPPORTED` and
    implementing ``_aggregate_compute``.
    """

    #: Aggregation types this results class implements. A subset of
    #: :data:`AGGREGATION_VOCABULARY` plus any documented per-estimator extra.
    #: ClassVar so that dataclass results classes overriding these hooks with
    #: an annotation do not grow a spurious ``__init__`` field.
    _AGGREGATE_SUPPORTED: ClassVar[Tuple[str, ...]] = ()

    #: Types for which ``balance_e`` is meaningful. CallawaySantAnna threads it
    #: only through event-study aggregation, so accepting it elsewhere would
    #: silently ignore a user's argument.
    _AGGREGATE_BALANCE_E_TYPES: ClassVar[Tuple[str, ...]] = ("event_study",)

    def aggregate(
        self,
        type: str,  # noqa: A002 - matches the ecosystem's aggte(type=) vocabulary
        weights: Optional[str] = None,
        *,
        balance_e: Optional[int] = None,
    ) -> Any:
        """Re-aggregate this fit without refitting.

        Returns a NEW object; ``self`` is never modified.
        ``type="event_study"`` returns
        :class:`~diff_diff.results_base.EventStudyResults`, every other type an
        :class:`AggregationResult`.

        Raises ``ValueError`` - never falls back silently - on an unsupported
        type, on a ``weights`` value this estimator does not accept, and on
        ``balance_e`` passed with a type that does not use it.
        """
        # NB: the parameter `type` shadows the builtin for this method's body -
        # the ecosystem's aggte(type=) vocabulary is worth the shadow, but every
        # class-name lookup below must go through __class__, not type(self).
        cls_name = self.__class__.__name__
        supported = tuple(self._AGGREGATE_SUPPORTED)
        if not supported:
            raise NotImplementedError(
                f"{cls_name} does not implement aggregate(); it is added "
                "per results class as each estimator's ledger row flips."
            )
        if type not in supported:
            known = ", ".join(repr(t) for t in supported)
            extra = ""
            if type in AGGREGATION_VOCABULARY:
                extra = (
                    f" {type!r} is part of the library-wide aggregation vocabulary "
                    "but this estimator does not implement it."
                )
            raise ValueError(f"Unsupported aggregation type {type!r}. Supported: {known}.{extra}")
        if balance_e is not None and type not in self._AGGREGATE_BALANCE_E_TYPES:
            if self._AGGREGATE_BALANCE_E_TYPES:
                usable = ", ".join(repr(t) for t in self._AGGREGATE_BALANCE_E_TYPES)
                applies = f"It applies to: {usable}."
            else:
                applies = "It applies to no aggregation type on this estimator."
            raise ValueError(
                f"balance_e is not used by aggregate(type={type!r}) and would be "
                f"silently ignored. {applies}"
            )
        self._aggregate_validate_weights(weights)
        return self._aggregate_compute(type, weights=weights, balance_e=balance_e)

    # -- hooks for the results class ----------------------------------- #

    def _aggregate_validate_weights(self, weights: Optional[str]) -> None:
        """Reject a weighting scheme this estimator does not offer.

        Default: no selector, so anything but ``None`` fails closed. Estimators
        with a real scheme (Wooldridge's ``"cell"`` / ``"cohort_share"``)
        override.
        """
        if weights is not None:
            raise ValueError(
                f"{type(self).__name__}.aggregate() does not accept a weights "
                f"selector (got {weights!r}); its aggregation weights are "
                "determined by the estimator."
            )

    def _aggregate_compute(
        self, level: str, *, weights: Optional[str], balance_e: Optional[int]
    ) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} declares _AGGREGATE_SUPPORTED but does not "
            "implement _aggregate_compute()."
        )
