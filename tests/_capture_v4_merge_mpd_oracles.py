"""Capture the TWFE event-study decoupling oracles for
tests/test_twfe_event_study_decoupling.py (4.0 cut, PR 4c-i / 4c-ii).

The committed records in ``tests/fixtures/v4_twfe_event_study_oracles.json``
are the 68e383bd capture (``provenance.captured_at_head``: origin/main at the
PR 4c-i branch cut, BEFORE the TWFE event-study mode was decoupled from the
MultiPeriodDiD containers) plus, for the two clustered hc2_bm lanes
(``pooled_cluster_hc2_bm``, ``within_cluster_hc2_bm``), the d45d5316 capture
recorded under ``provenance.lane_additions`` (PR 4c-ii, before the core's
post-average block was removed). Both were taken while ``MultiPeriodDiD``, the
independent twin of every lane, still existed; PR 4c-ii deleted it, so a
RE-RUN of this script on the current tree is a RE-BASELINE of the current
code, NOT an oracle - nothing independent remains to disagree with. Re-run
only to deliberately re-baseline after an intended numeric change, and say so
in the commit.

Full capture writes to stdout like the aggregate precedent::

    DIFF_DIFF_BACKEND=python python tests/_capture_v4_merge_mpd_oracles.py \
        > tests/fixtures/v4_twfe_event_study_oracles.json
    DIFF_DIFF_BACKEND=rust   python tests/_capture_v4_merge_mpd_oracles.py \
        > /path/to/scratch/rust.json   # cross-backend check only

Adding lanes splices only the named lanes into the committed file, leaving
every other lane and the top-level provenance byte-identical and recording the
new lanes' own provenance under ``provenance.lane_additions``::

    DIFF_DIFF_BACKEND=python python tests/_capture_v4_merge_mpd_oracles.py \
        --lanes pooled_cluster_hc2_bm within_cluster_hc2_bm \
        --merge-into tests/fixtures/v4_twfe_event_study_oracles.json

What is recorded: every ``dataclasses.fields(EventStudyResults)`` value of the
``TwoWayFixedEffects.fit(..., event_study=True)`` surface, per lane, so the
current tree's output can be compared field by field against the pre-refactor
numbers.

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

from diff_diff import TwoWayFixedEffects  # noqa: E402
from diff_diff.results_base import EventStudyResults  # noqa: E402


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
        warnings.simplefilter("ignore")  # rank-deficiency notices
        twfe = TwoWayFixedEffects(**lane.twfe_ctor).fit(
            df, event_study=True, survey_design=sd, **lane.twfe_fit
        )
    assert isinstance(twfe, EventStudyResults)
    return {"echo": lane.echo(), "twfe": encode_surface(twfe)}


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


def _provenance() -> Dict[str, Any]:
    import platform

    import pandas as pd
    import scipy

    import diff_diff
    from diff_diff.bootstrap_chunking import effective_weight_backend

    return {
        "captured_at_head": _git_head(),
        "diff_diff_version": diff_diff.__version__,
        "backend": effective_weight_backend(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
    }


def _main(argv: list) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--lanes",
        nargs="+",
        metavar="NAME",
        help="capture only these lanes (default: every lane in LANES)",
    )
    ap.add_argument(
        "--merge-into",
        metavar="PATH",
        help=(
            "splice the captured lanes into an existing fixture file in place "
            "(other lanes and the top-level provenance stay byte-identical; each "
            "spliced lane records its own provenance under "
            "provenance.lane_additions.<lane>) instead of writing a full fixture "
            "to stdout"
        ),
    )
    args = ap.parse_args(argv)
    names = args.lanes or list(LANES)
    unknown = [n for n in names if n not in LANES]
    if unknown:
        ap.error(f"unknown lane(s): {unknown}")
    prov = _provenance()
    captured = {name: capture_lane(name) for name in names}
    if args.merge_into is None:
        payload = {"provenance": prov, "lanes": captured}
        sys.stdout.write(dump_fixture(payload))
        return
    with open(args.merge_into) as fh:
        existing = json.load(fh)
    existing["lanes"].update(captured)
    additions = existing["provenance"].setdefault("lane_additions", {})
    for name in names:
        additions[name] = dict(prov)
    with open(args.merge_into, "w") as fh:
        fh.write(dump_fixture(existing))
    sys.stderr.write(f"spliced {len(names)} lane(s) into {args.merge_into}\n")


if __name__ == "__main__":
    _main(sys.argv[1:])
