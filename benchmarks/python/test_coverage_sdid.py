"""Harness-level sanity tests for ``coverage_sdid.py``.

Lives under ``benchmarks/`` rather than ``tests/`` because the targets
here (the private DGP helpers inside ``coverage_sdid.py``) are part of
the MC harness, not the shipped ``diff_diff`` package. CI's isolated-
install job deliberately copies only ``tests/``; running the pytest
collection here would just fail with ``ModuleNotFoundError`` on the
``benchmarks`` import.

Invoke manually when about to regenerate the coverage MC artifact::

    pytest benchmarks/python/test_coverage_sdid.py -v

Or from the repo root::

    python -m pytest benchmarks/python/test_coverage_sdid.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys

# Make the top-level ``benchmarks.python.coverage_sdid`` import resolvable
# when pytest is invoked from the repo root. The harness itself does a
# similar sys.path insert at module load; this mirror is only needed for
# test collection from this file.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.python import coverage_sdid  # noqa: E402


def test_stratified_survey_dgp_post_periods_cover_full_post_tail():
    """The ``stratified_survey`` coverage DGP must not drop any post period.

    Regression against PR #355 R13 P1: the harness previously hard-
    coded ``range(7, 12)`` as ``post_periods`` even though
    ``generate_survey_did_data`` is 1-indexed and emits periods
    1..n_periods, so period 12 was silently included in the pre window.
    That contaminated the ``stratified_survey × bootstrap`` calibration
    row and every downstream REGISTRY claim that transcribes from the
    artifact. The fix derives ``post_periods`` from
    ``df["period"].max()``; this test fails fast if a future refactor
    reintroduces the off-by-one.
    """
    df, post_periods = coverage_sdid._stratified_survey_dgp(seed=0)

    # Contract: post_periods covers the full tail from cohort onset
    # through df["period"].max(), with no gaps (unique + sorted +
    # contiguous + maxed at df["period"].max()).
    assert len(post_periods) == len(set(post_periods)), (
        f"post_periods has duplicates: {post_periods}"
    )
    assert post_periods == sorted(post_periods), (
        f"post_periods not sorted: {post_periods}"
    )
    gaps = [
        (a, b) for a, b in zip(post_periods, post_periods[1:]) if b - a != 1
    ]
    assert not gaps, (
        f"post_periods has gaps: {gaps} in {post_periods}"
    )
    assert post_periods[-1] == int(df["period"].max()), (
        f"post_periods max {post_periods[-1]} != df[period].max() "
        f"{int(df['period'].max())} — DGP drops the last post period "
        "(off-by-one on 1-indexed generate_survey_did_data)."
    )
    # Strong form: cohort onset is 7, n_periods=12 → [7,8,9,10,11,12].
    assert post_periods == [7, 8, 9, 10, 11, 12], (
        f"post_periods={post_periods} must equal [7,8,9,10,11,12] for "
        "the documented 6-pre/6-post survey DGP; any other slice "
        "changes the calibration interpretation in REGISTRY."
    )
