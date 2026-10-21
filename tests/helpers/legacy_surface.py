"""Test helpers bridging the post-fit aggregation containers and the 3.x dict shapes.

Since the 4.0 cut (PRs 3b/3c) the fit-time ``fit(aggregate=...)`` route is gone, so a
test that used to read ``results.event_study_effects[h]["effect"]`` after a
fit-time aggregation reads the post-fit container instead::

    es = results.aggregate("event_study")          # EventStudyResults
    grp = results.aggregate("group")               # AggregationResult

Three helpers keep dict-iteration tests short and let the few tests whose
SUBJECT is the native (legacy-pickle) route build a faithful stand-in:

``es_dict(results, balance_e=None)``
    ``aggregate("event_study")`` as the legacy ``event_study_effects`` mapping
    ``{event_time: {"effect", "se", "t_stat", "p_value", "conf_int", <count>}}``.
    The count key follows the container's ``n_kind`` (``"groups"`` -> ``n_groups``,
    ``"obs"`` -> ``n_obs``, ...); a NaN count becomes 0 (the legacy zero-count
    reference sentinel); when the container has NO ``n_kind`` (ContinuousDiD) no
    count key is emitted - ``_from_relative_dict`` rejects a zero-count row with a
    finite nonzero effect. ``cband_conf_int`` is added where the band is finite.
``group_dict(results)``
    ``aggregate("group")`` as the legacy ``group_effects`` mapping (label keys;
    ``df_used`` from the ``df`` column).
``legacy_standin(results, balance_e=None)``
    A COPY of ``results`` whose legacy fields hold the container's surface
    (``event_study_effects`` + ``event_study_vcov``/``_index`` + the surface's
    ``reference_event_times`` + ``cband_crit_value`` + ``event_study_df``), i.e.
    what a 3.x ``fit(aggregate="event_study", balance_e=...)`` stored. Use it ONLY
    where a test exercises the native route that live 4.0 fits no longer
    populate (plotter / HonestDiD / PreTrendsPower raw-field branches,
    ``build_event_study_surface(native)`` fixtures).

``load_oracles()`` / ``boot_oracles()`` load ``tests/fixtures/v4_aggregate_oracles.json``
(see ``tests/fixtures/README.md``); ``boot_oracles`` returns the active weight
backend's section or skips the test when that backend was never captured.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np
import pytest

_COUNT_KEY_BY_N_KIND = {
    "groups": "n_groups",
    "obs": "n_obs",
    "units": "n_units",
    "cells": "n_cells",
    "switcher_cells": "n_switcher_cells",
    "clusters": "n_clusters",
}

_FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")


def _label(x: Any) -> Any:
    """Container labels back to the legacy dict-key type (int where integral)."""
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return int(f) if f.is_integer() else f
    return x


def _count_key(n_kind: Optional[str]) -> Optional[str]:
    if n_kind is None:
        return None
    return _COUNT_KEY_BY_N_KIND.get(n_kind, f"n_{n_kind}")


def es_dict(results: Any, *, balance_e: Optional[int] = None) -> Dict[Any, Dict[str, Any]]:
    es = results.aggregate("event_study", balance_e=balance_e)
    return es_dict_from_container(es)


def es_dict_from_container(es: Any) -> Dict[Any, Dict[str, Any]]:
    """The legacy ``event_study_effects`` shape of an ``EventStudyResults``."""
    key = _count_key(es.n_kind)
    out: Dict[Any, Dict[str, Any]] = {}
    for i, e in enumerate(es.event_time):
        row: Dict[str, Any] = {
            "effect": float(es.att[i]),
            "se": float(es.se[i]),
            "t_stat": float(es.t_stat[i]),
            "p_value": float(es.p_value[i]),
            "conf_int": (float(es.conf_int_lower[i]), float(es.conf_int_upper[i])),
        }
        if key is not None:
            n = float(es.n[i])
            row[key] = 0 if math.isnan(n) else (int(n) if float(n).is_integer() else n)
        if es.cband_lower is not None and es.cband_upper is not None:
            lo, hi = float(es.cband_lower[i]), float(es.cband_upper[i])
            if math.isfinite(lo) and math.isfinite(hi):
                row["cband_conf_int"] = (lo, hi)
        out[_label(e)] = row
    return out


def group_dict(results: Any) -> Dict[Any, Dict[str, Any]]:
    """``aggregate("group")`` as the legacy ``group_effects`` mapping."""
    grp = results.aggregate("group")
    key = _count_key(grp.n_kind)
    out: Dict[Any, Dict[str, Any]] = {}
    for i, g in enumerate(grp.label):
        row: Dict[str, Any] = {
            "effect": float(grp.att[i]),
            "se": float(grp.se[i]),
            "t_stat": float(grp.t_stat[i]),
            "p_value": float(grp.p_value[i]),
            "conf_int": (float(grp.conf_int_lower[i]), float(grp.conf_int_upper[i])),
        }
        if key is not None:
            n = float(grp.n[i])
            row[key] = 0 if math.isnan(n) else (int(n) if float(n).is_integer() else n)
        df = float(grp.df[i])
        row["df_used"] = None if math.isnan(df) else df
        out[_label(g)] = row
    return out


def legacy_standin(results: Any, *, balance_e: Optional[int] = None) -> Any:
    """A copy of ``results`` carrying the container's surface in the 3.x fit-time fields."""
    es = results.aggregate("event_study", balance_e=balance_e)
    names = {f.name for f in dataclasses.fields(results)}
    updates: Dict[str, Any] = {}
    if "event_study_effects" in names:
        updates["event_study_effects"] = es_dict_from_container(es)
    if "event_study_vcov" in names:
        updates["event_study_vcov"] = None if es.vcov is None else np.asarray(es.vcov).copy()
    if "event_study_vcov_index" in names:
        # The 3.x field is a plain list of event-time labels (consumers call
        # ``.index(t)`` on it), not an array.
        updates["event_study_vcov_index"] = (
            None if es.vcov_index is None else [_label(x) for x in es.vcov_index]
        )
    if "reference_event_times" in names:
        updates["reference_event_times"] = es.reference_event_times
    if "cband_crit_value" in names:
        updates["cband_crit_value"] = es.cband_crit_value
    if "event_study_df" in names:
        df = es.df
        if df is None:
            updates["event_study_df"] = None
        else:
            arr = np.asarray(df, dtype=float).ravel()
            finite = arr[np.isfinite(arr)]
            updates["event_study_df"] = (
                float(finite[0]) if finite.size and np.all(finite == finite[0]) else None
            )
    return dataclasses.replace(results, **updates)


@lru_cache(maxsize=1)
def load_oracles() -> Dict[str, Any]:
    with open(os.path.join(_FIXTURE, "v4_aggregate_oracles.json")) as fh:
        return json.load(fh)


def boot_oracles() -> Dict[str, Any]:
    """The bootstrapped oracles for the ACTIVE weight backend, or skip."""
    from diff_diff.bootstrap_chunking import effective_weight_backend

    backend = effective_weight_backend()
    boot = load_oracles()["boot"]
    if backend not in boot:
        pytest.skip(f"no bootstrapped aggregation oracle captured for weight backend {backend!r}")
    return boot[backend]


def oracle_value(x: Any) -> float:
    """JSON ``null`` back to NaN."""
    return float("nan") if x is None else float(x)
