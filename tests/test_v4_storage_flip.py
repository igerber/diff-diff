"""Storage flip of the headline quintet at the 4.0 cut (ledger rows M-050..M-058).

``test_ref`` for M-050 (CallawaySantAnnaResults; DMLDiDResults rides it),
M-051 (SunAbrahamResults), M-052 (ImputationDiDResults), M-053
(TwoStageDiDResults), M-054 (StackedDiDResults), M-055 (EfficientDiDResults),
M-056 (WooldridgeDiDResults), M-057 (ChaisemartinDHaultfoeuilleResults) and
M-058 (ContinuousDiDResults). Spec: ``docs/v4-design.md`` section 5.

Contract pinned here, per class:

1. the canonical quintet ``att`` / ``se`` / ``t_stat`` / ``p_value`` /
   ``conf_int`` are dataclass fields; every legacy name is a plain
   descriptor (``deprecated_field_property``), not a field;
2. reading a legacy name emits the pinned ``FutureWarning`` text naming the
   class (the subclass for ``DMLDiDResults``) and relays the canonical value;
   assigning to it raises ``AttributeError``;
3. a 3.x pickle state (legacy keys) loads through ``__setstate__`` into the
   canonical fields and no ``overall_*`` key lingers in ``__dict__``;
4. a 4.0 pickle round-trips with every field present and ``to_dict()`` emits
   the canonical key;
5. constructors take canonical keywords only (``overall_att=`` raises);
6. every public read surface (``to_dict`` / ``summary`` / ``repr`` /
   ``to_dataframe`` at every level / ``coef_var`` / ``is_significant`` /
   ``significance_stars`` / ``aggregate``) runs under
   ``simplefilter("error", FutureWarning)`` - no internal ``self.overall_*``
   read survived the flip;
7. the Wooldridge self-WRITE sites assign the canonical fields;
8. the ``migrate_renamed_state_keys`` helper contract.

Fixtures are module-scoped fits built locally (fixtures do not cross
modules); every ``aggregate()`` call runs on a FRESH fit because
``WooldridgeDiDResults.aggregate()`` mutates ``self``.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    CallawaySantAnna,
    ChaisemartinDHaultfoeuille,
    ContinuousDiD,
    DMLDiD,
    EfficientDiD,
    ImputationDiD,
    StackedDiD,
    SunAbraham,
    TwoStageDiD,
    WooldridgeDiD,
    generate_continuous_did_data,
    generate_reversible_did_data,
    generate_staggered_data,
    load_mpdta,
)
from diff_diff._deprecation import LEGACY_OVERALL_QUINTET, migrate_renamed_state_keys
from diff_diff.chaisemartin_dhaultfoeuille_results import ChaisemartinDHaultfoeuilleResults
from diff_diff.continuous_did_results import (
    _LEGACY_OVERALL_ATT_QUINTET,
    ContinuousDiDResults,
)
from diff_diff.dml_did_results import DMLDiDResults
from diff_diff.efficient_did_results import EfficientDiDResults
from diff_diff.imputation_results import ImputationDiDResults
from diff_diff.stacked_did_results import StackedDiDResults
from diff_diff.staggered_results import CallawaySantAnnaResults
from diff_diff.sun_abraham import SunAbrahamResults
from diff_diff.two_stage_results import TwoStageDiDResults
from diff_diff.wooldridge_results import WooldridgeDiDResults

CANONICAL = ("att", "se", "t_stat", "p_value", "conf_int")

# Legacy property name -> canonical field, per class (what WARNS).
PROPERTY_MAP = {
    CallawaySantAnnaResults: dict(LEGACY_OVERALL_QUINTET),
    DMLDiDResults: dict(LEGACY_OVERALL_QUINTET),
    SunAbrahamResults: dict(LEGACY_OVERALL_QUINTET),
    ImputationDiDResults: dict(LEGACY_OVERALL_QUINTET),
    TwoStageDiDResults: dict(LEGACY_OVERALL_QUINTET),
    StackedDiDResults: dict(LEGACY_OVERALL_QUINTET),
    EfficientDiDResults: dict(LEGACY_OVERALL_QUINTET),
    WooldridgeDiDResults: dict(LEGACY_OVERALL_QUINTET),
    ChaisemartinDHaultfoeuilleResults: dict(LEGACY_OVERALL_QUINTET),
    ContinuousDiDResults: {**_LEGACY_OVERALL_ATT_QUINTET, **LEGACY_OVERALL_QUINTET},
}
# Legacy STORED key -> canonical field, per class (what a 3.x pickle carries).
STATE_MAP = {
    cls: (
        dict(_LEGACY_OVERALL_ATT_QUINTET)
        if cls is ContinuousDiDResults
        else dict(LEGACY_OVERALL_QUINTET)
    )
    for cls in PROPERTY_MAP
}
CLASSES = list(PROPERTY_MAP)
IDS = [c.__name__ for c in CLASSES]

# ``to_dataframe(level=...)`` vocabulary per class (read from each container).
LEVELS = {
    CallawaySantAnnaResults: ("group_time", "event_study", "group"),
    DMLDiDResults: ("group_time", "event_study", "group"),
    SunAbrahamResults: ("event_study", "cohort"),
    ImputationDiDResults: ("observation", "event_study", "group"),
    TwoStageDiDResults: ("observation", "event_study", "group"),
    StackedDiDResults: ("event_study", "group"),
    EfficientDiDResults: ("group_time", "event_study", "group"),
    WooldridgeDiDResults: ("simple", "group", "calendar", "event_study", "gt"),
    ChaisemartinDHaultfoeuilleResults: ("overall", "event_study"),
    ContinuousDiDResults: ("dose_response", "group_time", "event_study"),
}
# Classes whose fits may legitimately refuse a level because the surface
# was not stored on a 4.0 fit (the documented ValueError) - since M-093 8b
# only the "group" level does so (event_study relays aggregate()).
NOT_STORED_RE = re.compile(
    r"not stored|aggregate\(|not available|not supported|no .* surface|None", re.I
)


def _nan_eq(a, b) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_nan_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


# ---------------------------------------------------------------------------
# Fixtures (module-scoped fits; FRESH fits via the builders below)
# ---------------------------------------------------------------------------

_STAGGERED_FIT_KW = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


def _staggered_panel():
    return generate_staggered_data(
        n_units=80, n_periods=8, cohort_periods=[4], treatment_effect=2.0, seed=42
    )


def _sa_panel(n_units=80, n_periods=8, n_cohorts=3, seed=42):
    rng = np.random.RandomState(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    n_never = int(n_units * 0.3)
    n_treated = n_units - n_never
    cohort_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int)
    first_treat = np.zeros(n_units)
    if n_treated > 0:
        first_treat[n_never:] = cohort_periods[rng.choice(len(cohort_periods), size=n_treated)]
    fte = np.repeat(first_treat, n_periods)
    unit_fe = np.repeat(rng.randn(n_units) * 2, n_periods)
    time_fe = np.tile(np.linspace(0, 1, n_periods), n_units)
    post = (times >= fte) & (fte > 0)
    y = unit_fe + time_fe + 2.0 * post + rng.randn(len(units)) * 0.5
    return pd.DataFrame(
        {"unit": units, "time": times, "outcome": y, "first_treat": fte.astype(int)}
    )


def _dml_panel(n_units=120, n_periods=5, seed=0):
    rng = np.random.default_rng(seed)
    cohort = rng.choice([0, 3, 4], size=n_units, p=[0.4, 0.3, 0.3])
    rows = []
    for i in range(n_units):
        x1, x2 = rng.normal(), rng.normal()
        u = rng.normal()
        for t in range(1, n_periods + 1):
            y = u + 0.5 * t + 0.3 * x1 - 0.2 * x2 + rng.normal(scale=0.5)
            if cohort[i] > 0 and t >= cohort[i]:
                y += 1.5
            rows.append((i, t, cohort[i], y, x1, x2))
    return pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1", "x2"])


def _fit(cls):
    """Build a FRESH fit of ``cls`` (never cached - aggregate() may mutate)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cls is CallawaySantAnnaResults:
            return CallawaySantAnna(n_bootstrap=0).fit(_staggered_panel(), **_STAGGERED_FIT_KW)
        if cls is DMLDiDResults:
            return DMLDiD(seed=0).fit(
                _dml_panel(),
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
            )
        if cls is SunAbrahamResults:
            return SunAbraham().fit(
                _sa_panel(), outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )
        if cls is ImputationDiDResults:
            return ImputationDiD().fit(_staggered_panel(), **_STAGGERED_FIT_KW)
        if cls is TwoStageDiDResults:
            return TwoStageDiD().fit(_staggered_panel(), **_STAGGERED_FIT_KW)
        if cls is StackedDiDResults:
            data = generate_staggered_data(
                n_units=120,
                n_periods=12,
                cohort_periods=[4, 6, 8],
                never_treated_frac=0.3,
                treatment_effect=5.0,
                dynamic_effects=True,
                seed=42,
            )
            return StackedDiD(kappa_pre=2, kappa_post=2).fit(data, **_STAGGERED_FIT_KW)
        if cls is EfficientDiDResults:
            return EfficientDiD().fit(_staggered_panel(), **_STAGGERED_FIT_KW)
        if cls is WooldridgeDiDResults:
            return WooldridgeDiD().fit(
                load_mpdta(),
                outcome="lemp",
                unit="countyreal",
                time="year",
                first_treat="first_treat",
            )
        if cls is ChaisemartinDHaultfoeuilleResults:
            data = generate_reversible_did_data(
                n_groups=50, n_periods=8, pattern="joiners_only", seed=42
            )
            return ChaisemartinDHaultfoeuille(twfe_diagnostic=False).fit(
                data,
                outcome="outcome",
                unit="group",
                time="period",
                treatment="treatment",
                L_max=1,
            )
        if cls is ContinuousDiDResults:
            data = generate_continuous_did_data(
                n_units=200, n_periods=5, cohort_periods=[2, 4], seed=42, noise_sd=0.5
            )
            return ContinuousDiD(control_group="not_yet_treated", n_bootstrap=0).fit(
                data, "outcome", "unit", "period", "first_treat", "dose"
            )
    raise AssertionError(cls)


@pytest.fixture(scope="module")
def fits():
    return {cls: _fit(cls) for cls in CLASSES}


# ---------------------------------------------------------------------------
# 1. Field shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_canonical_quintet_are_fields_legacy_names_are_descriptors(cls):
    field_names = set(cls.__dataclass_fields__)
    for name in CANONICAL:
        assert name in field_names, f"{cls.__name__}.{name} is not a dataclass field"
    for old in PROPERTY_MAP[cls]:
        assert old not in field_names, f"{cls.__name__}.{old} is still a dataclass field"
        assert isinstance(
            inspect.getattr_static(cls, old), property
        ), f"{cls.__name__}.{old} does not resolve as a property descriptor"
    # The ACRT family on ContinuousDiD is a separate estimand and stays stored.
    if cls is ContinuousDiDResults:
        assert {"overall_acrt", "overall_acrt_se", "overall_acrt_conf_int"} <= field_names


# ---------------------------------------------------------------------------
# 2. Warning + relay + read-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_legacy_properties_warn_relay_and_reject_assignment(fits, cls):
    res = fits[cls]
    assert type(res) is cls
    for old, new in PROPERTY_MAP[cls].items():
        pattern = (
            rf"^{cls.__name__}\.{old} is deprecated; use {new} instead\. Will be removed in 5\.0\.$"
        )
        with pytest.warns(FutureWarning, match=pattern):
            got = getattr(res, old)
        assert _nan_eq(got, getattr(res, new)), (cls.__name__, old)
        with pytest.raises(AttributeError):
            setattr(res, old, 1.0)
        doc = inspect.getattr_static(cls, old).__doc__ or ""
        assert "removed in 5.0" in doc, (cls.__name__, old, doc)


def test_dml_warning_names_the_subclass(fits):
    res = fits[DMLDiDResults]
    with pytest.warns(FutureWarning, match=r"^DMLDiDResults\.overall_att is deprecated"):
        res.overall_att
    # The storage is the inherited field set - no DML-specific field appeared.
    assert set(DMLDiDResults.__dataclass_fields__) >= set(
        CallawaySantAnnaResults.__dataclass_fields__
    )
    assert "overall_att" not in DMLDiDResults.__dataclass_fields__


# ---------------------------------------------------------------------------
# 3. Legacy pickle state migrates
# ---------------------------------------------------------------------------


def _assert_no_legacy_keys(obj, cls):
    allowed_prefix = "overall_acrt" if cls is ContinuousDiDResults else None
    leaked = [
        k
        for k in obj.__dict__
        if k.startswith("overall_") and not (allowed_prefix and k.startswith(allowed_prefix))
    ]
    assert not leaked, f"{cls.__name__}: legacy keys linger in __dict__: {leaked}"


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_legacy_state_loads_into_canonical_fields(fits, cls):
    res = fits[cls]
    state = dict(res.__dict__)
    for old, new in STATE_MAP[cls].items():
        state[old] = state.pop(new)
    if cls is StackedDiDResults:
        # The pre-existing clean_control -> control_group migration still runs.
        state["clean_control"] = state.pop("control_group")
    restored = object.__new__(cls)
    restored.__setstate__(state)
    for name in CANONICAL:
        assert _nan_eq(getattr(restored, name), getattr(res, name)), (cls.__name__, name)
    _assert_no_legacy_keys(restored, cls)
    if cls is StackedDiDResults:
        assert restored.control_group == res.control_group
    with pytest.warns(FutureWarning):
        assert _nan_eq(restored.overall_att, res.att)


# ---------------------------------------------------------------------------
# 4. 4.0 pickle round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_pickle_round_trip_keeps_every_field_and_canonical_to_dict(fits, cls):
    res = fits[cls]
    loaded = pickle.loads(pickle.dumps(res))
    for f in dataclasses.fields(cls):
        assert f.name in loaded.__dict__, (cls.__name__, f.name)
    _assert_no_legacy_keys(loaded, cls)
    for name in CANONICAL:
        assert _nan_eq(getattr(loaded, name), getattr(res, name)), (cls.__name__, name)
    d = loaded.to_dict()
    assert _nan_eq(d["att"], res.att)
    assert not [k for k in d if k.startswith("overall_") and not k.startswith("overall_acrt")]


# ---------------------------------------------------------------------------
# 5. Constructor keywords
# ---------------------------------------------------------------------------


def _required_init_kwargs(cls, overrides):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in overrides:
            continue
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            continue
        ann = str(f.type)
        if "Tuple" in ann or "tuple" in ann:
            kwargs[f.name] = (0.0, 0.0)
        elif "List" in ann or "list" in ann:
            kwargs[f.name] = []
        elif "Dict" in ann or "dict" in ann:
            kwargs[f.name] = {}
        elif "DataFrame" in ann:
            kwargs[f.name] = pd.DataFrame()
        elif "ndarray" in ann:
            kwargs[f.name] = np.array([])
        elif "float" in ann:
            kwargs[f.name] = 0.0
        elif "int" in ann:
            kwargs[f.name] = 0
        else:
            kwargs[f.name] = None
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_constructor_takes_canonical_keywords_only(cls):
    canonical = {"att": 1.5, "se": 0.3, "t_stat": 5.0, "p_value": 0.001, "conf_int": (1.0, 2.0)}
    res = cls(**_required_init_kwargs(cls, canonical))
    for name, value in canonical.items():
        assert getattr(res, name) == value
    for old in STATE_MAP[cls]:
        kwargs = _required_init_kwargs(cls, canonical)
        kwargs[old] = 1.5
        with pytest.raises(TypeError, match=rf"unexpected keyword argument '{old}'"):
            cls(**kwargs)


def test_positional_layout_unchanged_canonical_in_legacy_slots():
    """The fields were renamed IN SLOT: the generated ``__init__`` positional
    layout (public API on StackedDiDResults / dCDH) puts the quintet exactly
    where ``overall_*`` sat under 3.x."""
    stacked = [f.name for f in dataclasses.fields(StackedDiDResults)]
    assert stacked[:5] == list(CANONICAL)
    dcdh = [f.name for f in dataclasses.fields(ChaisemartinDHaultfoeuilleResults)]
    assert dcdh[:5] == list(CANONICAL)
    cs = [f.name for f in dataclasses.fields(CallawaySantAnnaResults)]
    assert cs[:6] == ["group_time_effects", *CANONICAL]
    cont = [f.name for f in dataclasses.fields(ContinuousDiDResults)]
    assert cont[:7] == ["dose_response_att", "dose_response_acrt", *CANONICAL]
    assert cont[7:12] == [
        "overall_acrt",
        "overall_acrt_se",
        "overall_acrt_t_stat",
        "overall_acrt_p_value",
        "overall_acrt_conf_int",
    ]


# ---------------------------------------------------------------------------
# 6. No internal legacy read survives: every surface under FutureWarning=error
# ---------------------------------------------------------------------------


def _assert_documented_value_error(exc: ValueError, cls, level):
    assert NOT_STORED_RE.search(
        str(exc)
    ), f"{cls.__name__}.to_dataframe({level!r}) raised an undocumented ValueError: {exc}"


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_read_surfaces_emit_no_future_warning(fits, cls):
    res = fits[cls]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        res.to_dict()
        res.summary()
        repr(res)
        for level in LEVELS[cls]:
            try:
                res.to_dataframe(level=level)
            except ValueError as exc:
                _assert_documented_value_error(exc, cls, level)
        if cls is not WooldridgeDiDResults:
            res.coef_var
            res.is_significant
            res.significance_stars


@pytest.mark.parametrize(
    "cls",
    [CallawaySantAnnaResults, ImputationDiDResults, TwoStageDiDResults, EfficientDiDResults],
    ids=lambda c: c.__name__,
)
def test_aggregate_simple_and_total_emit_no_future_warning(cls):
    res = _fit(cls)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        simple = res.aggregate("simple")
        assert np.isfinite(simple.to_dict()["att"]) if hasattr(simple, "to_dict") else True
        try:
            res.aggregate("total")
        except ValueError as exc:
            assert "total" in str(exc), exc


@pytest.mark.parametrize(
    "cls",
    [StackedDiDResults, ChaisemartinDHaultfoeuilleResults, ContinuousDiDResults, SunAbrahamResults],
    ids=lambda c: c.__name__,
)
def test_aggregate_simple_emits_no_future_warning_other_containers(cls):
    res = _fit(cls)
    if not hasattr(res, "aggregate"):
        pytest.skip(f"{cls.__name__} has no aggregate()")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        try:
            res.aggregate("simple")
        except ValueError as exc:
            assert "simple" in str(exc), exc


# ---------------------------------------------------------------------------
# 7. Wooldridge self-writes assign the canonical fields
# ---------------------------------------------------------------------------


def _three_cohort_panel(rng, *, tau_by_gt, sigma, cohort_unit_counts):
    """Balanced 4-period panel, cohorts g=0 (never), 2, 3; the fixture of
    tests/test_methodology_wooldridge.py's cohort-share test."""
    rows = []
    unit_id = 0
    for g in (0, 2, 3):
        for _ in range(cohort_unit_counts[g]):
            c_i = rng.standard_normal()
            for t in (1, 2, 3, 4):
                w = 1 if (g > 0 and t >= g) else 0
                tau = tau_by_gt.get((g, t), 0.0) if w else 0.0
                y = c_i + 0.5 * t + w * tau + sigma * rng.standard_normal()
                rows.append({"unit": unit_id, "time": t, "cohort": g, "y": y})
            unit_id += 1
    return pd.DataFrame(rows)


def test_wooldridge_aggregate_simple_writes_canonical_fields():
    rng = np.random.default_rng(20261026)
    panel = _three_cohort_panel(
        rng,
        tau_by_gt={(2, 2): 1.0, (2, 3): 1.0, (2, 4): 1.0, (3, 3): 2.0, (3, 4): 2.0},
        sigma=0.05,
        cohort_unit_counts={0: 100, 2: 50, 3: 200},
    )
    res = WooldridgeDiD(method="ols").fit(
        panel, outcome="y", unit="unit", time="time", first_treat="cohort"
    )
    n_g = res._n_g_per_cohort
    assert n_g == {2: 50, 3: 200}
    gt = res.group_time_effects
    # Paper Eq. 7.4: ATT_simple = sum_{(g,t): t>=g} N_g tau_gt / sum N_g.
    expected = (
        n_g[2] * (gt[(2, 2)]["att"] + gt[(2, 3)]["att"] + gt[(2, 4)]["att"])
        + n_g[3] * (gt[(3, 3)]["att"] + gt[(3, 4)]["att"])
    ) / (n_g[2] * 3 + n_g[3] * 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        # cohort_share inference fails closed with a documented UserWarning
        # (paper W2025 Section 7.5); only FutureWarning is under test here.
        warnings.simplefilter("ignore", UserWarning)
        res.aggregate("simple", weights="cohort_share")
    # On a balanced panel cell and cohort-share weights coincide, so the value
    # need not CHANGE - the point is that the canonical-field WRITE path ran
    # and landed Eq. 7.4 under the error filter.
    assert res.att == pytest.approx(expected, abs=1e-12)
    assert np.isfinite(res.se)
    assert res.aggregation_weights["simple"] == "cohort_share"
    _assert_no_legacy_keys(res, WooldridgeDiDResults)


def test_wooldridge_bootstrap_override_writes_canonical_fields():
    res = WooldridgeDiD(n_bootstrap=20, seed=0).fit(
        load_mpdta(), outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
    )
    assert res._bootstrap_used is True
    assert np.isfinite(res.se)
    assert np.isfinite(res.t_stat) and np.isfinite(res.p_value)
    assert all(np.isfinite(res.conf_int))
    _assert_no_legacy_keys(res, WooldridgeDiDResults)


# ---------------------------------------------------------------------------
# 8. migrate_renamed_state_keys helper contract
# ---------------------------------------------------------------------------


class TestMigrateRenamedStateKeys:
    def test_canonical_value_wins_and_legacy_key_is_dropped(self):
        out = migrate_renamed_state_keys({"overall_att": 1.0, "att": 2.0}, LEGACY_OVERALL_QUINTET)
        assert out == {"att": 2.0}

    def test_legacy_only_state_seeds_canonical(self):
        out = migrate_renamed_state_keys(
            {"overall_att": 1.0, "overall_se": 0.5, "n_obs": 3}, LEGACY_OVERALL_QUINTET
        )
        assert out == {"att": 1.0, "se": 0.5, "n_obs": 3}

    def test_canonical_only_state_returned_equal_but_distinct(self):
        state = {"att": 1.0, "se": 0.5}
        out = migrate_renamed_state_keys(state, LEGACY_OVERALL_QUINTET)
        assert out == state and out is not state

    def test_input_not_mutated(self):
        state = {"overall_att": 1.0, "att": 2.0}
        snapshot = dict(state)
        migrate_renamed_state_keys(state, LEGACY_OVERALL_QUINTET)
        assert state == snapshot

    def test_quintet_map_is_the_five_names(self):
        assert LEGACY_OVERALL_QUINTET == {
            "overall_att": "att",
            "overall_se": "se",
            "overall_t_stat": "t_stat",
            "overall_p_value": "p_value",
            "overall_conf_int": "conf_int",
        }
        assert _LEGACY_OVERALL_ATT_QUINTET == {
            "overall_att": "att",
            "overall_att_se": "se",
            "overall_att_t_stat": "t_stat",
            "overall_att_p_value": "p_value",
            "overall_att_conf_int": "conf_int",
        }
