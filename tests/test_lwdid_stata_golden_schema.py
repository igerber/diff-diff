"""Schema/cardinality validation for the LWDiD Stata parity golden.

This module is deliberately UNGATED (no ``diff_diff.lwdid`` import): the
methodology suite (``tests/test_methodology_lwdid.py``) skips entirely until
the LWDiD estimator lands, so without this file main's CI would never open
``benchmarks/data/lwdid_stata_golden.json`` and a malformed or regenerated
golden could merge unvalidated.

It also HOSTS the expected key-set constants shared with the methodology
suite - the import must run in this direction (the gated module imports from
here), because importing the gated module raises ``Skipped`` at collection
on main.

The golden is produced by ``benchmarks/stata/generate_lwdid_golden.do``
against the authors' SSC ``lwdid`` package; see that file and
``benchmarks/stata/README.md`` for provenance and regeneration.
"""

import json
from pathlib import Path

import pytest

STATA_GOLDEN_PATH = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "lwdid_stata_golden.json"
)

#: The six Walmart (config, outcome) table keys: "<rolling>_<method>__<outcome>"
#: (method in the authors' Stata vocabulary: ra / ipwra).
WALMART_CONFIG_KEYS = (
    "detrend_ra__log_retail_emp",
    "detrend_ipwra__log_retail_emp",
    "demean_ipwra__log_retail_emp",
    "detrend_ra__log_wholesale_emp",
    "detrend_ipwra__log_wholesale_emp",
    "demean_ipwra__log_wholesale_emp",
)

#: Measured per-config event-time label sets of the Stata WATT tables
#: (surface (b) of the cardinality pins): r = -22..13 INCLUDING the anchor
#: rows, identical across all six configs (measured from the generator run;
#: the walmart panel is 1977-1999 with cohorts 1986-1999, so the reachable
#: event-time range is data-determined and config-independent).
WALMART_WATT_LABELS = {key: tuple(range(-22, 14)) for key in WALMART_CONFIG_KEYS}

#: Fixed meta keys the consuming tests read.
REQUIRED_META_KEYS = (
    "ssc_versions",
    "bootstrap_scheme",
    "control_pool",
    "datasets",
    "rireps",
    "riseed",
    "bootstrap_reps",
    "bootstrap_seed",
)


def _load():
    if not STATA_GOLDEN_PATH.exists():
        # The Rust CI matrix copies only tests/ to a temp dir; skip (never
        # error) when benchmarks/data is absent, like every Stata arm test.
        pytest.skip(f"{STATA_GOLDEN_PATH.name} not committed (partial checkout)")
    return json.loads(STATA_GOLDEN_PATH.read_text())


def _is_finite_number(x):
    import math

    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def test_top_level_blocks():
    golden = _load()
    assert sorted(golden.keys()) == ["castle", "meta", "prop99", "walmart"]


def test_meta_keys_and_version_line():
    meta = _load()["meta"]
    for key in REQUIRED_META_KEYS:
        assert key in meta, key
    # ssc drift detection depends on a real version line, not a fallback
    assert "version" in meta["ssc_versions"]["lwdid"].lower()
    assert isinstance(meta["bootstrap_reps"], int) and meta["bootstrap_reps"] >= 999
    assert isinstance(meta["rireps"], int) and meta["rireps"] >= 10000


def test_prop99_block():
    block = _load()["prop99"]
    assert sorted(block.keys()) == ["demean", "detrend"]
    for rolling, cell in block.items():
        assert sorted(cell.keys()) == ["att", "p_ri", "se"], rolling
        assert all(_is_finite_number(v) for v in cell.values()), rolling


def test_castle_block():
    block = _load()["castle"]
    assert sorted(block.keys()) == ["demean", "detrend"]
    for rolling, cell in block.items():
        assert sorted(cell.keys()) == ["att", "se"], rolling
        assert all(_is_finite_number(v) for v in cell.values()), rolling


def test_walmart_block_cardinality():
    block = _load()["walmart"]
    assert sorted(block.keys()) == sorted(WALMART_CONFIG_KEYS)
    for key, table in block.items():
        assert sorted(table.keys()) == ["overall", "watt"], key
        labels = sorted(int(r) for r in table["watt"])
        assert tuple(labels) == tuple(sorted(WALMART_WATT_LABELS[key])), key
        for r, cell in table["watt"].items():
            assert isinstance(cell, list) and len(cell) == 2, f"{key} r={r}"
            # fail closed: a regeneration with missing cells must not pass
            assert all(_is_finite_number(v) for v in cell), f"{key} r={r}"
        assert sorted(table["overall"].keys()) == ["Post_avg", "Pre_avg"], key
        for agg, cell in table["overall"].items():
            assert isinstance(cell, list) and len(cell) == 2, f"{key} {agg}"
            assert all(_is_finite_number(v) for v in cell), f"{key} {agg}"
