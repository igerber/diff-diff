"""Inference-field aliases on result classes (balance / external-adapter compatibility).

Each in-scope result class exposes flat aliases (``att`` / ``se`` / ``conf_int`` /
``p_value`` / ``t_stat``) that map to the canonical native fields (``overall_*``,
``overall_att_*``, or ``avg_*``). This file pins the alias-canonical contract.

Motivating bug: ``balance.interop.diff_diff.as_balance_diagnostic`` reads
``getattr(res, "se", None)`` and ``getattr(res, "conf_int", None)`` without
fallbacks to ``overall_se`` / ``overall_conf_int``. Pre-alias, every Pattern
B / C / D result class returned ``None`` on those keys, so balance's tutorial
shipped with ``se=NaN`` / ``conf_int=NaN`` in the methods-appendix table.
"""

from __future__ import annotations

import math
from dataclasses import asdict, fields

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    CallawaySantAnna,
    generate_staggered_data,
)
from diff_diff.chaisemartin_dhaultfoeuille_results import (
    ChaisemartinDHaultfoeuilleResults,
)
from diff_diff.continuous_did_results import ContinuousDiDResults
from diff_diff.dml_did_results import DMLDiDResults
from diff_diff.efficient_did_results import EfficientDiDResults
from diff_diff.imputation_results import ImputationDiDResults
from diff_diff.results import MultiPeriodDiDResults
from diff_diff.stacked_did_results import StackedDiDResults
from diff_diff.staggered_results import CallawaySantAnnaResults
from diff_diff.staggered_triple_diff_results import StaggeredTripleDiffResults
from diff_diff.sun_abraham import SunAbrahamResults
from diff_diff.two_stage_results import TwoStageDiDResults
from diff_diff.wooldridge_results import WooldridgeDiDResults

# ============================================================================
# Helpers
# ============================================================================


def _alias_equal(a, b) -> bool:
    """``==`` that treats NaN==NaN as True so aliases inherit NaN consistency."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_alias_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


def _required_init_kwargs(cls, overrides):
    """Return a kwargs dict for constructing a dataclass with sentinel defaults
    for every required field, then merging in ``overrides``.

    Lets us build a minimal result instance for alias-mechanic tests without
    having to enumerate every estimator-specific field. Sentinel values for
    untouched fields are deliberately uninteresting (empty containers, zeros)
    -- they are not exercised by these tests.
    """
    import dataclasses as _dc

    kwargs = {}
    for f in fields(cls):
        if f.name in overrides:
            continue
        # A field is REQUIRED iff both default and default_factory are MISSING.
        # When default_factory is set (e.g. list/dict factory), the dataclass
        # will apply it; we must NOT pre-fill the field with a sentinel or we
        # block the factory.
        if f.default is not _dc.MISSING or f.default_factory is not _dc.MISSING:
            continue
        # Required field — supply a type-compatible sentinel.
        # Order container annotations BEFORE the scalar `"float"` / `"int"`
        # branches so that ``Tuple[float, float]`` is not mis-classified as
        # scalar (``"float" in "Tuple[float, float]"`` is True).
        ann = str(f.type)
        if "Tuple" in ann or "tuple" in ann:
            kwargs[f.name] = (0.0, 0.0)
        elif "List" in ann or "list" in ann:
            kwargs[f.name] = []
        elif "Dict" in ann or "dict" in ann:
            kwargs[f.name] = {}
        elif "DataFrame" in ann:
            kwargs[f.name] = pd.DataFrame()
        elif "ndarray" in ann or "np.ndarray" in ann:
            kwargs[f.name] = np.array([])
        elif "float" in ann:
            kwargs[f.name] = 0.0
        elif "int" in ann:
            kwargs[f.name] = 0
        else:
            kwargs[f.name] = None
    kwargs.update(overrides)
    return kwargs


def test_required_init_kwargs_handles_default_factory_and_tuple_dispatch():
    """Pin the two `_required_init_kwargs()` fixes from PR #437 R2 directly.

    The helper had two latent bugs masked by the specific fields exercised
    by the alias tests: factory-only required fields were pre-filled (so
    the factory never ran), and the type-dispatch checked ``"float"``
    before ``"Tuple"`` (so ``Tuple[float, float]`` annotations matched the
    scalar branch). Both fixes are exercised here against a small local
    dataclass so the contract stays pinned independent of production
    result-dataclass shape.
    """
    from dataclasses import dataclass, field
    from typing import Tuple

    @dataclass
    class _Probe:
        # Required, must be filled with a tuple sentinel — NOT 0.0 — even
        # though "float" appears in the annotation as a substring.
        ci: Tuple[float, float]
        # default_factory-backed: must be OMITTED from kwargs so the
        # factory runs at construction time.
        items: list = field(default_factory=list)
        # default-valued: must also be omitted from kwargs.
        x: float = 1.5

    kwargs = _required_init_kwargs(_Probe, overrides={})
    assert kwargs == {"ci": (0.0, 0.0)}, (
        f"_required_init_kwargs() must (a) supply a tuple sentinel for "
        f"Tuple[float, float] required fields (not a scalar 0.0), and "
        f"(b) omit default_factory / default fields so the dataclass "
        f"applies them at construction time. Got: {kwargs!r}"
    )
    # Round-trip: instance construction must succeed and the factory
    # must have produced an empty list (not been displaced by a sentinel).
    inst = _Probe(**kwargs)
    assert inst.ci == (0.0, 0.0)
    assert inst.items == []
    assert inst.x == 1.5


def _assert_pattern_b_aliases(res, *, att, se, t_stat, p_value, conf_int):
    """Pattern B: 5 flat aliases mapping to the overall_* canonical fields."""
    assert _alias_equal(res.att, att), f"att alias != overall_att ({res.att} vs {att})"
    assert _alias_equal(res.se, se)
    assert _alias_equal(res.conf_int, conf_int)
    assert _alias_equal(res.p_value, p_value)
    assert _alias_equal(res.t_stat, t_stat)


# Sentinel inference values exercised across direct-construction tests.
_ATT = 1.5
_SE = 0.3
_T = 5.0
_P = 0.001
_CI = (1.0, 2.0)


# ============================================================================
# Pattern B (10 classes) — direct-construction alias mechanics
# ============================================================================


@pytest.mark.parametrize(
    "cls",
    [
        CallawaySantAnnaResults,
        DMLDiDResults,
        StackedDiDResults,
        EfficientDiDResults,
        ChaisemartinDHaultfoeuilleResults,
        StaggeredTripleDiffResults,
        WooldridgeDiDResults,
        SunAbrahamResults,
        ImputationDiDResults,
        TwoStageDiDResults,
    ],
    ids=lambda c: c.__name__,
)
def test_pattern_b_aliases_match_overall(cls):
    """Each Pattern B class's flat aliases equal the canonical overall_* fields."""
    overrides = {
        "overall_att": _ATT,
        "overall_se": _SE,
        "overall_t_stat": _T,
        "overall_p_value": _P,
        "overall_conf_int": _CI,
    }
    res = cls(**_required_init_kwargs(cls, overrides))
    _assert_pattern_b_aliases(res, att=_ATT, se=_SE, t_stat=_T, p_value=_P, conf_int=_CI)


@pytest.mark.parametrize(
    "cls",
    [
        CallawaySantAnnaResults,
        DMLDiDResults,
        StackedDiDResults,
        EfficientDiDResults,
        ChaisemartinDHaultfoeuilleResults,
        StaggeredTripleDiffResults,
        WooldridgeDiDResults,
        SunAbrahamResults,
        ImputationDiDResults,
        TwoStageDiDResults,
    ],
    ids=lambda c: c.__name__,
)
def test_pattern_b_aliases_propagate_nan(cls):
    """When canonical overall_* fields are NaN (degenerate fit), aliases are NaN.

    Pins the safe_inference() joint-NaN contract (per CLAUDE.md: ALL inference
    fields are computed together and stay NaN-consistent). Aliases are pure
    read-throughs, so the contract holds without re-computation.
    """
    overrides = {
        "overall_att": np.nan,
        "overall_se": np.nan,
        "overall_t_stat": np.nan,
        "overall_p_value": np.nan,
        "overall_conf_int": (np.nan, np.nan),
    }
    res = cls(**_required_init_kwargs(cls, overrides))
    assert math.isnan(res.att)
    assert math.isnan(res.se)
    assert math.isnan(res.t_stat)
    assert math.isnan(res.p_value)
    assert math.isnan(res.conf_int[0])
    assert math.isnan(res.conf_int[1])


# ============================================================================
# Pattern C — ContinuousDiDResults: flat AND overall_* aliases
# ============================================================================


def _continuous_did_overrides(att=_ATT, se=_SE, t=_T, p=_P, ci=_CI):
    return {
        "overall_att": att,
        "overall_att_se": se,
        "overall_att_t_stat": t,
        "overall_att_p_value": p,
        "overall_att_conf_int": ci,
        "overall_acrt": 0.0,
        "overall_acrt_se": 0.0,
        "overall_acrt_t_stat": 0.0,
        "overall_acrt_p_value": 1.0,
        "overall_acrt_conf_int": (0.0, 0.0),
    }


def test_continuous_did_flat_aliases():
    """ContinuousDiD flat aliases map to the ATT-side overall_att_* fields."""
    res = ContinuousDiDResults(
        **_required_init_kwargs(ContinuousDiDResults, _continuous_did_overrides())
    )
    assert res.att == _ATT
    assert res.se == _SE
    assert res.conf_int == _CI
    assert res.p_value == _P
    assert res.t_stat == _T


def test_continuous_did_overall_aliases():
    """ContinuousDiD overall_* aliases also map to the ATT-side fields
    (consistency with Pattern B family naming)."""
    res = ContinuousDiDResults(
        **_required_init_kwargs(ContinuousDiDResults, _continuous_did_overrides())
    )
    assert res.overall_se == _SE
    assert res.overall_conf_int == _CI
    assert res.overall_p_value == _P
    assert res.overall_t_stat == _T


def test_continuous_did_double_alias_resolves_same_value():
    """``res.se`` and ``res.overall_se`` MUST point at the same value."""
    res = ContinuousDiDResults(
        **_required_init_kwargs(ContinuousDiDResults, _continuous_did_overrides())
    )
    assert res.se == res.overall_se
    assert res.conf_int == res.overall_conf_int
    assert res.p_value == res.overall_p_value
    assert res.t_stat == res.overall_t_stat


# ============================================================================
# Pattern D — MultiPeriodDiDResults: avg_* -> flat aliases
# ============================================================================


def test_multi_period_did_aliases():
    """MultiPeriodDiD flat aliases map to the avg_* canonical fields."""
    overrides = {
        "avg_att": _ATT,
        "avg_se": _SE,
        "avg_t_stat": _T,
        "avg_p_value": _P,
        "avg_conf_int": _CI,
    }
    res = MultiPeriodDiDResults(**_required_init_kwargs(MultiPeriodDiDResults, overrides))
    assert res.att == _ATT
    assert res.se == _SE
    assert res.conf_int == _CI
    assert res.p_value == _P
    assert res.t_stat == _T


# ============================================================================
# Read-only semantics
# ============================================================================


@pytest.mark.parametrize(
    ("cls", "ovr"),
    [
        (
            CallawaySantAnnaResults,
            {
                "overall_att": _ATT,
                "overall_se": _SE,
                "overall_t_stat": _T,
                "overall_p_value": _P,
                "overall_conf_int": _CI,
            },
        ),
        (ContinuousDiDResults, _continuous_did_overrides()),
        (
            MultiPeriodDiDResults,
            {
                "avg_att": _ATT,
                "avg_se": _SE,
                "avg_t_stat": _T,
                "avg_p_value": _P,
                "avg_conf_int": _CI,
            },
        ),
    ],
    ids=lambda v: v.__name__ if hasattr(v, "__name__") else "ovr",
)
def test_aliases_are_read_only(cls, ovr):
    """Assigning to an alias must raise AttributeError (no setter installed).

    Regression: a downstream test in tests/test_practitioner.py used
    `r.overall_se = X` on a `ContinuousDiDResults.__new__()` mock — pre-alias
    that silently created a junk attribute; post-alias the property correctly
    rejects the assignment. Locking read-only here means future contributors
    who write similar fixtures fail loudly via this test rather than via a
    surprise `AttributeError: can't set attribute` deep in another suite.
    """
    res = cls(**_required_init_kwargs(cls, ovr))
    for name in ("att", "se", "conf_int", "p_value", "t_stat"):
        with pytest.raises(AttributeError):
            setattr(res, name, object())
    # ContinuousDiDResults also exposes overall_se / overall_conf_int /
    # overall_p_value / overall_t_stat as read-only aliases over the
    # ATT-side canonical fields (no parallel `overall_att` alias is needed
    # because `overall_att_att` would be confusing; the flat `att` covers
    # that one). These must also reject assignment.
    if cls.__name__ == "ContinuousDiDResults":
        for name in ("overall_se", "overall_conf_int", "overall_p_value", "overall_t_stat"):
            with pytest.raises(AttributeError):
                setattr(res, name, object())


@pytest.mark.parametrize(
    "cls",
    [CallawaySantAnnaResults, ContinuousDiDResults, MultiPeriodDiDResults],
    ids=lambda v: v.__name__,
)
def test_aliases_excluded_from_dataclass_fields_and_asdict(cls):
    """Aliases must not appear in ``dataclasses.fields()`` or
    ``dataclasses.asdict()`` output. This is the contract the registry
    documents (PR for v3.3.3 + REGISTRY note), but the existing alias
    suite only locks read-through and read-only semantics — a future
    refactor that converted an `@property` to a real dataclass field
    would silently surface aliases to serializers and field-walkers
    without this regression catching it.

    Three representative classes cover the three alias patterns:
    Pattern B (`CallawaySantAnnaResults`), the double-alias case
    (`ContinuousDiDResults`), and the `avg_*` mapping
    (`MultiPeriodDiDResults`).
    """
    ovr = {
        CallawaySantAnnaResults: {
            "overall_att": _ATT,
            "overall_se": _SE,
            "overall_t_stat": _T,
            "overall_p_value": _P,
            "overall_conf_int": _CI,
        },
        ContinuousDiDResults: _continuous_did_overrides(),
        MultiPeriodDiDResults: {
            "avg_att": _ATT,
            "avg_se": _SE,
            "avg_t_stat": _T,
            "avg_p_value": _P,
            "avg_conf_int": _CI,
        },
    }[cls]
    res = cls(**_required_init_kwargs(cls, ovr))
    field_names = {f.name for f in fields(res)}
    asdict_keys = set(asdict(res).keys())
    alias_names = ["att", "se", "conf_int", "p_value", "t_stat"]
    # ContinuousDiDResults is the documented double-alias case: it
    # also exposes `overall_se` / `overall_conf_int` / `overall_p_value`
    # / `overall_t_stat` aliases pointing at the ATT side (the native
    # field is `overall_att_*`). Lock those out of fields()/asdict() too
    # so a future refactor cannot silently surface them as dataclass
    # fields (which would duplicate the native `overall_att_*` keys in
    # serializer output).
    if cls is ContinuousDiDResults:
        alias_names += ["overall_se", "overall_conf_int", "overall_p_value", "overall_t_stat"]
    for alias in alias_names:
        assert alias not in field_names, (
            f"{cls.__name__}: alias {alias!r} surfaced in "
            f"dataclasses.fields() output - aliases must remain "
            f"@property descriptors, never real fields."
        )
        assert (
            alias not in asdict_keys
        ), f"{cls.__name__}: alias {alias!r} surfaced in dataclasses.asdict() output."


# ============================================================================
# Cross-cutting regression — balance.interop.diff_diff adapter pattern
# ============================================================================


def test_balance_adapter_pattern_returns_populated_se():
    """Mimic balance.interop.diff_diff.as_balance_diagnostic: real CS fit then
    flat ``getattr(res, "se", None)`` / ``getattr(res, "conf_int", None)``.

    Pre-alias: returned ``None`` on every Pattern B/C/D result class. This is
    the test that would have caught the original bug if balance had exercised
    a real fit instead of a stub class with attributes literally named ``se``.
    """
    df = generate_staggered_data(
        n_units=30,
        n_periods=5,
        cohort_periods=[3],
        never_treated_frac=0.5,
        seed=42,
    )
    res = CallawaySantAnna(estimation_method="reg").fit(
        df,
        outcome="outcome",
        time="period",
        unit="unit",
        first_treat="first_treat",
    )
    se = getattr(res, "se", None)
    conf_int = getattr(res, "conf_int", None)
    p_value = getattr(res, "p_value", None)
    assert se is not None and np.isfinite(
        se
    ), f"balance adapter would see se={se!r}; pre-alias bug returned None"
    assert (
        conf_int is not None and len(conf_int) == 2 and all(np.isfinite(x) for x in conf_int)
    ), f"balance adapter would see conf_int={conf_int!r}; pre-alias bug returned None"
    assert p_value is not None and np.isfinite(p_value)
    # Aliases must equal the canonical overall_* fields.
    assert se == res.overall_se
    assert conf_int == res.overall_conf_int
    assert p_value == res.overall_p_value
