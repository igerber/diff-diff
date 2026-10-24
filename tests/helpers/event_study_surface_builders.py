"""Hand-built calendar ``EventStudyResults`` for consumer tests (4.0 PR 4c-i).

Replaces the hand-built ``MultiPeriodDiDResults`` / ``PeriodEffect`` mocks of
the MultiPeriodDiD-fixture tests with the surface the TWFE event-study
producer emits: one row per calendar label, the reference row marked
``is_reference`` (NaN inference), ``time_scale="calendar"``,
``source="TwoWayFixedEffects"`` and the authoritative ``post_periods``
partition. The calendar consumer route requires exactly one reference row,
which is the last pre-period, so callers supply ``reference`` explicitly.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from diff_diff.results_base import EventStudyResults
from diff_diff.utils import safe_inference

Row = Union[
    Tuple[float, float], Tuple[float, float, float, float, Tuple[float, float]], Dict[str, Any]
]


def _row_fields(row: Row, alpha: float = 0.05) -> Tuple[float, float, float, float, float, float]:
    """(effect, se, t_stat, p_value, ci_lo, ci_hi) from a terse row spec.

    A 2-tuple ``(effect, se)`` fills t/p/CI through ``safe_inference`` (the
    library's own helper, so se <= 0 / non-finite rows fail closed to all-NaN
    inference and ``alpha`` sets the interval level); a 5-tuple
    ``(effect, se, t_stat, p_value, (lo, hi))`` and a dict with those keys are
    deliberate literals taken verbatim.
    """
    if isinstance(row, dict):
        lo, hi = row.get("conf_int", (np.nan, np.nan))
        return (
            float(row["effect"]),
            float(row["se"]),
            float(row.get("t_stat", np.nan)),
            float(row.get("p_value", np.nan)),
            float(lo),
            float(hi),
        )
    if len(row) == 2:
        effect, se = float(row[0]), float(row[1])
        t, p, (lo, hi) = safe_inference(effect, se, alpha=alpha)
        return effect, se, float(t), float(p), float(lo), float(hi)
    effect, se, t, p, ci = row  # type: ignore[misc]
    return float(effect), float(se), float(t), float(p), float(ci[0]), float(ci[1])


def calendar_surface(
    period_effects: Mapping[Any, Row],
    *,
    reference: Any,
    post_periods: Sequence[Any],
    vcov: Optional[np.ndarray] = None,
    vcov_index: Optional[Sequence[Any]] = None,
    source: str = "TwoWayFixedEffects",
    alpha: float = 0.05,
    df_survey: Optional[float] = None,
) -> EventStudyResults:
    """Build a calendar surface from ``{label: row}`` (non-reference labels only).

    ``vcov`` is the event-study sub-block over ``vcov_index`` (default: the
    sorted non-reference labels, i.e. the same order as the rows).
    """
    labels = sorted(period_effects)
    if reference in labels:
        raise ValueError("reference must not be among the estimated labels")
    all_labels = sorted(labels + [reference])
    n = len(all_labels)
    att = np.full(n, np.nan)
    se = np.full(n, np.nan)
    t = np.full(n, np.nan)
    p = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    is_ref = np.zeros(n, dtype=bool)
    for i, lab in enumerate(all_labels):
        if lab == reference:
            is_ref[i] = True
            continue
        att[i], se[i], t[i], p[i], lo[i], hi[i] = _row_fields(period_effects[lab], alpha=alpha)
    if vcov is not None and vcov_index is None:
        vcov_index = labels
    return EventStudyResults(
        event_time=np.asarray(all_labels),
        att=att,
        se=se,
        t_stat=t,
        p_value=p,
        conf_int_lower=lo,
        conf_int_upper=hi,
        is_reference=is_ref,
        n=np.full(n, np.nan),
        reference_period=reference,
        time_scale="calendar",
        vcov=None if vcov is None else np.asarray(vcov, dtype=float),
        vcov_index=None if vcov_index is None else np.asarray(list(vcov_index)),
        alpha=alpha,
        source=source,
        post_periods=tuple(post_periods),
        df_survey=df_survey,
    )
