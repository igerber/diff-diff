"""Post-period average of an event-study surface with MultiPeriodDiD's arithmetic.

The 4.0 cut (PR 4c-i) migrates MultiPeriodDiD-fixture tests to
``TwoWayFixedEffects().fit(..., event_study=True)``, whose
:class:`~diff_diff.results_base.EventStudyResults` carries no ``avg_att`` /
``avg_se``. Ported sites whose subject is the FIT LANE (not the MPD container)
replace those reads with this helper, which reproduces the core's
post-average arithmetic and NaN contract (``diff_diff/estimators.py``,
``_fit_event_study_core``): ``avg_att = mean(post effects)`` and
``avg_se = sqrt(sum(post vcov block) / n_post**2)``; any NaN post effect
voids both; a NaN or negative summed variance voids the SE (guarded before
the sqrt, no RuntimeWarning). Exact equality with MultiPeriodDiD's values on
the pooled twin is pinned by ``tests/test_twfe_event_study_decoupling.py``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from diff_diff.results_base import EventStudyResults


def post_period_average(es: EventStudyResults) -> Tuple[float, float]:
    """Return ``(att_bar, se_bar)`` over ``es.post_periods``.

    ``att_bar`` is the equal-weight mean of the post-period ``att`` rows;
    ``se_bar`` is ``sqrt(sum of the post-period vcov block) / n_post``
    (computed as ``sqrt(sum / n_post**2)``, MultiPeriodDiD's expression).
    """
    if es.post_periods is None:
        raise ValueError("post_period_average requires a surface with post_periods provenance")
    post_rows = np.isin(es.event_time, list(es.post_periods))
    post_att = np.asarray(es.att, dtype=float)[post_rows]
    if post_att.size == 0:
        raise ValueError("post_period_average: no post-period rows on the surface")
    if np.any(np.isnan(post_att)):
        return float("nan"), float("nan")
    att_bar = float(np.mean(post_att))
    if es.vcov is None or es.vcov_index is None:
        return att_bar, float("nan")
    post_cols = np.isin(es.vcov_index, list(es.post_periods))
    n_post = int(post_cols.sum())
    if n_post == 0:
        return att_bar, float("nan")
    avg_var = float(np.sum(np.asarray(es.vcov, dtype=float)[np.ix_(post_cols, post_cols)])) / (
        n_post**2
    )
    if np.isnan(avg_var) or avg_var < 0:
        return att_bar, float("nan")
    return att_bar, float(np.sqrt(avg_var))
