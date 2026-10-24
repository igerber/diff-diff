"""Capture the TWFE event-study decoupling oracles for
tests/test_twfe_event_study_decoupling.py (4.0 cut, PR 4c-i).

Run on the UNMODIFIED tree (the commit named in ``provenance.captured_at_head``
of ``tests/fixtures/v4_twfe_event_study_oracles.json``) BEFORE the TWFE
event-study mode is decoupled from the MultiPeriodDiD containers, and write to
stdout like the aggregate precedent::

    DIFF_DIFF_BACKEND=python python tests/_capture_v4_merge_mpd_oracles.py \
        > tests/fixtures/v4_twfe_event_study_oracles.json
    DIFF_DIFF_BACKEND=rust   python tests/_capture_v4_merge_mpd_oracles.py \
        > /path/to/scratch/rust.json   # cross-backend check only

Why. PR 4c-i rewrites how ``TwoWayFixedEffects.fit(..., event_study=True)``
builds its ``EventStudyResults`` (today: an internal ``MultiPeriodDiDResults``
converted by ``results_base._from_mpd``; after: a private payload from
``_fit_event_study_core``). Every ``dataclasses.fields(EventStudyResults)``
value of the TWFE surface and of the MultiPeriodDiD twin
(``_from_mpd(MultiPeriodDiD(...).fit(...))``) is recorded per lane, so the
refactor's output can be compared field by field against the pre-refactor
tree. PR 4c-ii deletes MultiPeriodDiD; re-capturing on a post-4c-ii tree is
NOT a substitute for the MPD-twin half (nothing independent remains).

Panels and lanes come from ``tests/helpers/twfe_es_oracle_lanes.py`` (shared
with the test module); the fixture carries each lane's ``echo()`` so an edited
lane fails as lane drift, not as a numeric regression.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import warnings
from typing import Any, Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "helpers"))

from twfe_es_oracle_lanes import LANES, PANELS  # noqa: E402

from diff_diff import MultiPeriodDiD, TwoWayFixedEffects  # noqa: E402
from diff_diff.results_base import EventStudyResults, _from_mpd  # noqa: E402


def _encode_value(value: Any) -> Any:
    """JSON-safe encoding of one EventStudyResults field value.

    ndarray -> {"dtype": <np dtype str>, "values": [...]} with NaN as null;
    tuple -> list; numpy scalars -> Python scalars; NaN floats -> null.
    """
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "values": _nan_to_none(value.tolist())}
    if isinstance(value, (tuple, list)):
        return [_encode_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _nan_to_none(obj: Any) -> Any:
    """Recursively map float NaN to None in (nested) lists from ``ndarray.tolist()``."""
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def encode_surface(surface: EventStudyResults) -> Dict[str, Any]:
    return {f.name: _encode_value(getattr(surface, f.name)) for f in dataclasses.fields(surface)}


def capture_lane(name: str) -> Dict[str, Any]:
    lane = LANES[name]
    df = PANELS[lane.panel]()
    sd = lane.survey()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MPD's FutureWarning + rank-deficiency notices
        twfe = TwoWayFixedEffects(**lane.twfe_ctor).fit(
            df, event_study=True, survey_design=sd, **lane.twfe_fit
        )
        mpd = _from_mpd(MultiPeriodDiD(**lane.mpd_ctor).fit(df, survey_design=sd, **lane.mpd_fit))
    assert isinstance(twfe, EventStudyResults)
    return {"echo": lane.echo(), "twfe": encode_surface(twfe), "mpd_twin": encode_surface(mpd)}


def dump_fixture(obj: Any) -> str:
    """Sorted keys, one indent level per nesting level, every leaf container
    (a dict or list of scalars) on one line - the aggregate-oracle format."""

    def _fmt(o: Any, ind: int = 0) -> str:
        pad = " " * ind
        if isinstance(o, dict):
            if all(not isinstance(v, (dict, list)) for v in o.values()):
                return json.dumps(o, sort_keys=True)
            items = [f"{pad} {json.dumps(k)}: {_fmt(v, ind + 1)}" for k, v in sorted(o.items())]
            return "{\n" + ",\n".join(items) + f"\n{pad}}}"
        if isinstance(o, list):
            if all(not isinstance(v, (dict, list)) for v in o):
                return json.dumps(o)
            items = [f"{pad} {_fmt(v, ind + 1)}" for v in o]
            return "[\n" + ",\n".join(items) + f"\n{pad}]"
        return json.dumps(o)

    return _fmt(obj) + "\n"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    import platform

    import pandas as pd
    import scipy

    import diff_diff
    from diff_diff.bootstrap_chunking import effective_weight_backend

    payload = {
        "provenance": {
            "captured_at_head": _git_head(),
            "diff_diff_version": diff_diff.__version__,
            "backend": effective_weight_backend(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
        },
        "lanes": {name: capture_lane(name) for name in LANES},
    }
    sys.stdout.write(dump_fixture(payload))
