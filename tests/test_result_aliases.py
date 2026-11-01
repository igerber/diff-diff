"""Inference-field names on result classes (balance / external-adapter compatibility).

Since the 4.0 storage flip (ledger rows M-050..M-058, PR 5b) the canonical
quintet ``att`` / ``se`` / ``t_stat`` / ``p_value`` / ``conf_int`` IS the
native storage on every estimator container in this file, and the 3.x
``overall_*`` / ``overall_att_*`` names are GONE (their 4.0 -> 5.0
FutureWarning shims were retired at 4.0 by item 9b of the cut). This file
pins that direction: canonical names are constructor keywords and dataclass
fields, legacy names are absent (``hasattr`` False).
The last ``overall_*`` container, ``StaggeredTripleDiffResults``, was retired by
the item-6 DDD container unification (row M-014): ``TripleDifferenceResults``
now serves both DDD designs with native canonical storage.

Motivating bug: ``balance.interop.diff_diff.as_balance_diagnostic`` reads
``getattr(res, "se", None)`` and ``getattr(res, "conf_int", None)`` without
fallbacks to ``overall_se`` / ``overall_conf_int``. Pre-alias, every Pattern
B / C / D result class returned ``None`` on those keys, so balance's tutorial
shipped with ``se=NaN`` / ``conf_int=NaN`` in the methods-appendix table.
"""

from __future__ import annotations

import inspect
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
from diff_diff.stacked_did_results import StackedDiDResults
from diff_diff.staggered_results import CallawaySantAnnaResults
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


# Sentinel inference values exercised across direct-construction tests.
_ATT = 1.5
_SE = 0.3
_T = 5.0
_P = 0.001
_CI = (1.0, 2.0)


_CANONICAL = {"att": _ATT, "se": _SE, "t_stat": _T, "p_value": _P, "conf_int": _CI}
_CANONICAL_NAN = {
    "att": np.nan,
    "se": np.nan,
    "t_stat": np.nan,
    "p_value": np.nan,
    "conf_int": (np.nan, np.nan),
}

# The nine flipped containers (M-050..M-058) plus DMLDiDResults, which inherits
# CallawaySantAnnaResults' storage and re-declares the shims under its own name.
FLIPPED_PATTERN_B = [
    CallawaySantAnnaResults,
    DMLDiDResults,
    StackedDiDResults,
    EfficientDiDResults,
    ChaisemartinDHaultfoeuilleResults,
    WooldridgeDiDResults,
    SunAbrahamResults,
    ImputationDiDResults,
    TwoStageDiDResults,
]


# ============================================================================
# Pattern B (9 flipped classes) — direct-construction mechanics
# ============================================================================


@pytest.mark.parametrize("cls", FLIPPED_PATTERN_B, ids=lambda c: c.__name__)
def test_pattern_b_canonical_keywords_and_no_legacy_names(cls):
    """Each flipped class constructs with canonical keywords; the legacy
    ``overall_*`` names are gone (removed at 4.0, item 9b)."""
    res = cls(**_required_init_kwargs(cls, dict(_CANONICAL)))
    assert res.att == _ATT
    assert res.se == _SE
    assert res.t_stat == _T
    assert res.p_value == _P
    assert res.conf_int == _CI
    for old in (
        "overall_att",
        "overall_se",
        "overall_t_stat",
        "overall_p_value",
        "overall_conf_int",
    ):
        assert not hasattr(res, old), f"{cls.__name__}.{old} resurrected"


@pytest.mark.parametrize("cls", FLIPPED_PATTERN_B, ids=lambda c: c.__name__)
def test_pattern_b_legacy_constructor_keywords_rejected(cls):
    """The 3.x ``overall_*`` names are no longer constructor keywords."""
    kwargs = _required_init_kwargs(cls, dict(_CANONICAL))
    kwargs["overall_att"] = _ATT
    with pytest.raises(TypeError, match="unexpected keyword argument 'overall_att'"):
        cls(**kwargs)


# ============================================================================
# Pattern C — ContinuousDiDResults: nine legacy spellings onto the canonical five
# ============================================================================


def _continuous_did_overrides(att=_ATT, se=_SE, t=_T, p=_P, ci=_CI):
    return {
        "att": att,
        "se": se,
        "t_stat": t,
        "p_value": p,
        "conf_int": ci,
        "overall_acrt": 0.0,
        "overall_acrt_se": 0.0,
        "overall_acrt_t_stat": 0.0,
        "overall_acrt_p_value": 1.0,
        "overall_acrt_conf_int": (0.0, 0.0),
    }


def test_continuous_did_canonical_fields_native():
    """ContinuousDiD stores the ATT-side headline under the canonical quintet."""
    res = ContinuousDiDResults(
        **_required_init_kwargs(ContinuousDiDResults, _continuous_did_overrides())
    )
    assert res.att == _ATT
    assert res.se == _SE
    assert res.conf_int == _CI
    assert res.p_value == _P
    assert res.t_stat == _T
    assert res.overall_acrt == 0.0  # the separate ACRT estimand is untouched


# ============================================================================
# Read-only semantics
# ============================================================================


@pytest.mark.parametrize(
    ("cls", "ovr", "legacy_names"),
    [
        (
            CallawaySantAnnaResults,
            dict(_CANONICAL),
            ("overall_att", "overall_se", "overall_t_stat", "overall_p_value", "overall_conf_int"),
        ),
        (
            ContinuousDiDResults,
            _continuous_did_overrides(),
            (
                "overall_att",
                "overall_att_se",
                "overall_att_t_stat",
                "overall_att_p_value",
                "overall_att_conf_int",
                "overall_se",
                "overall_t_stat",
                "overall_p_value",
                "overall_conf_int",
            ),
        ),
    ],
    ids=["CallawaySantAnnaResults", "ContinuousDiDResults"],
)
def test_legacy_names_absent_canonical_fields_writable(cls, ovr, legacy_names):
    """The legacy ``overall_*`` / ``overall_att_*`` names are absent (neither
    field nor descriptor; ``hasattr`` False), while the canonical quintet are
    plain dataclass fields and accept assignment (the
    ``WooldridgeDiDResults.aggregate("simple")`` write path relies on this).

    Note: with the descriptors gone, assigning a legacy name creates a shadow
    instance attribute on these non-slotted dataclasses instead of raising -
    the former "wrong name fails loudly" guard was retired with the shims at
    item 9b; the naming guard's consumer sweep is the in-repo defense.
    """
    res = cls(**_required_init_kwargs(cls, ovr))
    for name in legacy_names:
        assert inspect.getattr_static(cls, name, None) is None, f"{cls.__name__}.{name}"
        assert not hasattr(res, name), f"{cls.__name__}.{name}"
    for name in ("att", "se", "conf_int", "p_value", "t_stat"):
        setattr(res, name, 7.0)
        assert getattr(res, name) == 7.0


@pytest.mark.parametrize(
    "cls",
    [CallawaySantAnnaResults, ContinuousDiDResults],
    ids=lambda v: v.__name__,
)
def test_legacy_aliases_excluded_from_dataclass_fields_and_asdict(cls):
    """The legacy ``overall_*`` properties must not appear in
    ``dataclasses.fields()`` or ``dataclasses.asdict()`` output, and the
    canonical quintet MUST (it is the storage). A future refactor that
    annotated a shim (turning it into a real field) would silently surface
    the deprecated names to serializers and field-walkers without this
    regression catching it.

    Two representative classes cover the flipped patterns: Pattern B
    (`CallawaySantAnnaResults`) and the nine-spelling case
    (`ContinuousDiDResults`).
    """
    ovr = {
        CallawaySantAnnaResults: dict(_CANONICAL),
        ContinuousDiDResults: _continuous_did_overrides(),
    }[cls]
    res = cls(**_required_init_kwargs(cls, ovr))
    field_names = {f.name for f in fields(res)}
    asdict_keys = set(asdict(res).keys())
    canonical = ["att", "se", "conf_int", "p_value", "t_stat"]
    legacy = ["overall_att", "overall_se", "overall_conf_int", "overall_p_value", "overall_t_stat"]
    if cls is ContinuousDiDResults:
        legacy += [
            "overall_att_se",
            "overall_att_conf_int",
            "overall_att_p_value",
            "overall_att_t_stat",
        ]
    for name in canonical:
        assert name in field_names, f"{cls.__name__}: canonical {name!r} is not a dataclass field"
        assert name in asdict_keys
    for alias in legacy:
        assert alias not in field_names, (
            f"{cls.__name__}: legacy {alias!r} surfaced in "
            f"dataclasses.fields() output - shims must remain "
            f"descriptors, never real fields."
        )
        assert (
            alias not in asdict_keys
        ), f"{cls.__name__}: legacy {alias!r} surfaced in dataclasses.asdict() output."


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
    # The legacy overall_* names are gone (removed at 4.0, item 9b).
    assert not hasattr(res, "overall_se")
    assert not hasattr(res, "overall_conf_int")
    assert not hasattr(res, "overall_p_value")
