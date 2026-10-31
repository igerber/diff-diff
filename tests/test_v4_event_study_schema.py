"""Item 8b of the 4.0 cut - the unified ``to_dataframe(level="event_study")`` schema (ledger row M-093, second half).

Since 4.0 every results class's ``to_dataframe(level="event_study")`` emits the
``EVENT_STUDY_SCHEMA`` columns (``results_base.EventStudyResults.to_dataframe``)
as an identical PREFIX: the ``AggregationMixin`` classes return
``aggregate("event_study").to_dataframe()``; SunAbraham, LPDiD, WooldridgeDiD
and SpilloverDiD return ``build_event_study_surface(self).to_dataframe()``.
Only two producers append documented fit-configuration columns after the
schema - LWDiD (``rolling``, ``estimation_method``, ``vcov_type``; row M-153)
and WooldridgeDiD (``cohort_trends``, ``aggregation_weights``). The 3.x
bespoke frames (``relative_period`` / ``effect`` / ``att_glob`` / ``horizon`` /
``conf_int_lo`` / ``n_obs`` ...) and the "not stored on 4.0 fits" raises on
CallawaySantAnna / EfficientDiD / TripleDifference and on analytical
ImputationDiD / TwoStageDiD / ContinuousDiD are gone.

The first half of M-093 (the reference-row sentinel retirement) is pinned by
``tests/test_v4_sentinel_retirement.py``.
"""

import copy
import dataclasses
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from diff_diff import (
    CallawaySantAnna,
    ChaisemartinDHaultfoeuille,
    ContinuousDiD,
    DMLDiD,
    DurationDiD,
    EfficientDiD,
    ImputationDiD,
    LPDiD,
    LPDiDResults,
    LWDiD,
    StackedDiD,
    SunAbraham,
    TripleDifference,
    TwoStageDiD,
    TwoWayFixedEffects,
    WooldridgeDiD,
    generate_continuous_did_data,
    generate_ddd_data,
    generate_reversible_did_data,
    generate_staggered_data,
    generate_staggered_ddd_data,
    load_mpdta,
)
from diff_diff.results_base import EVENT_STUDY_SCHEMA, build_event_study_surface
from diff_diff.spillover import SpilloverDiD
from tests._dgp_utils import generate_butts_staggered_dgp
from tests.test_duration_did import simulate_panel
from tests.test_event_study_surface import (
    _bjs_panel,
    _efficient_panel,
    _had_event_study,
    _lpdid_panel,
    _sa_panel,
    _two_group_calendar_panel,
)

THIS_TEST_REF = "tests/test_v4_event_study_schema.py"
SCHEMA = list(EVENT_STUDY_SCHEMA)
N_SCHEMA = len(SCHEMA)
_PANEL_KW = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
_BJS_KW = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")

# Producers that append documented fit-configuration columns after the schema.
TRAILING = {
    "wool": ["cohort_trends", "aggregation_weights"],
    "lwdid": ["rolling", "estimation_method", "vcov_type"],
}
# Producers WITHOUT the AggregationMixin: compared against the builder frame.
NON_MIXIN = {"sa", "lpdid", "spill_es", "wool"}


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# --------------------------------------------------------------------------- #
# One fresh fit per producer (module-scoped)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cs_varying():
    d = generate_staggered_data(n_units=120, n_periods=8, cohort_periods=[4, 6], seed=7)
    return _quiet(CallawaySantAnna(n_bootstrap=0).fit, d, **_PANEL_KW)


@pytest.fixture(scope="module")
def cs_universal():
    d = generate_staggered_data(n_units=120, n_periods=8, cohort_periods=[4, 6], seed=7)
    return _quiet(CallawaySantAnna(n_bootstrap=0, base_period="universal").fit, d, **_PANEL_KW)


@pytest.fixture(scope="module")
def cs_boot():
    d = generate_staggered_data(n_units=120, n_periods=8, cohort_periods=[4, 6], seed=7)
    return _quiet(CallawaySantAnna(n_bootstrap=49, seed=11).fit, d, **_PANEL_KW)


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


@pytest.fixture(scope="module")
def dml():
    return _quiet(
        DMLDiD(seed=0).fit,
        _dml_panel(),
        outcome="y",
        unit="unit",
        time="time",
        first_treat="first_treat",
        covariates=["x1", "x2"],
    )


@pytest.fixture(scope="module")
def sa():
    return _quiet(SunAbraham().fit, _sa_panel(), **_BJS_KW)


@pytest.fixture(scope="module")
def imp():
    return _quiet(ImputationDiD().fit, _bjs_panel(), **_BJS_KW)


@pytest.fixture(scope="module")
def imp_boot():
    return _quiet(ImputationDiD(n_bootstrap=49, seed=42).fit, _bjs_panel(), **_BJS_KW)


@pytest.fixture(scope="module")
def ts():
    return _quiet(TwoStageDiD().fit, _bjs_panel(), **_BJS_KW)


@pytest.fixture(scope="module")
def ts_boot():
    return _quiet(TwoStageDiD(n_bootstrap=49, seed=42).fit, _bjs_panel(), **_BJS_KW)


@pytest.fixture(scope="module")
def stacked():
    d = generate_staggered_data(n_units=120, n_periods=12, cohort_periods=[4, 6, 8], seed=42)
    return _quiet(StackedDiD(kappa_pre=2, kappa_post=2).fit, d, **_PANEL_KW)


def _spillover(event_study):
    d = generate_butts_staggered_dgp(seed=42)
    kw: Dict[str, Any] = dict(
        rings=[0.0, 50.0, 200.0],
        d_bar=200.0,
        conley_coords=("lat", "lon"),
        conley_metric="haversine",
        conley_cutoff_km=200.0,
        conley_lag_cutoff=0,
    )
    if event_study:
        kw.update(event_study=True, horizon_max=2)
    return _quiet(
        SpilloverDiD(**kw).fit, d, outcome="y", unit="unit", time="time", first_treat="first_treat"
    )


@pytest.fixture(scope="module")
def spill_es():
    return _spillover(True)


@pytest.fixture(scope="module")
def spill_agg():
    return _spillover(False)


def _cont_panel():
    return generate_continuous_did_data(
        n_units=200, n_periods=5, cohort_periods=[2, 4], seed=42, noise_sd=0.5
    )


@pytest.fixture(scope="module")
def cont():
    return _quiet(
        ContinuousDiD(control_group="not_yet_treated", n_bootstrap=0).fit,
        _cont_panel(),
        "outcome",
        "unit",
        "period",
        "first_treat",
        "dose",
    )


@pytest.fixture(scope="module")
def cont_boot():
    return _quiet(
        ContinuousDiD(control_group="not_yet_treated", n_bootstrap=49, seed=42).fit,
        _cont_panel(),
        "outcome",
        "unit",
        "period",
        "first_treat",
        "dose",
    )


@pytest.fixture(scope="module")
def edid():
    return _quiet(
        EfficientDiD().fit,
        _efficient_panel(),
        outcome="y",
        unit="unit",
        time="time",
        first_treat="first_treat",
    )


def _wooldridge():
    return _quiet(
        WooldridgeDiD(control_group="never_treated").fit,
        load_mpdta(),
        outcome="lemp",
        unit="countyreal",
        time="year",
        first_treat="first_treat",
    )


@pytest.fixture(scope="module")
def wool():
    res = _wooldridge()
    res.aggregate("event_study")
    return res


@pytest.fixture(scope="module")
def wool_fresh():
    return _wooldridge()


@pytest.fixture(scope="module")
def ddd_stag():
    d = generate_staggered_ddd_data(n_units=300, treatment_effect=3.0, seed=42)
    return _quiet(TripleDifference().fit, d, partition="eligibility", **_PANEL_KW)


@pytest.fixture(scope="module")
def ddd_2x2x2():
    d = generate_ddd_data(
        n_per_cell=200,
        treatment_effect=2.0,
        group_effect=1.0,
        partition_effect=0.5,
        time_effect=0.7,
        noise_sd=1.0,
        seed=42,
    )
    return _quiet(
        TripleDifference().fit,
        d,
        outcome="outcome",
        group="group",
        partition="partition",
        post="time",
    )


@pytest.fixture(scope="module")
def dcdh():
    d = generate_reversible_did_data(n_groups=50, n_periods=10, pattern="joiners_only", seed=42)
    return _quiet(
        ChaisemartinDHaultfoeuille(twfe_diagnostic=False, placebo=True).fit,
        d,
        outcome="outcome",
        unit="group",
        time="period",
        treatment="treatment",
        L_max=3,
    )


@pytest.fixture(scope="module")
def lpdid():
    return _quiet(
        LPDiD(pre_window=2, post_window=2).fit,
        _lpdid_panel(),
        outcome="y",
        unit="unit",
        time="time",
        treatment="treat",
    )


@pytest.fixture(scope="module")
def had_es():
    return _had_event_study()[1]


@pytest.fixture(scope="module")
def duration():
    return _quiet(
        DurationDiD(n_bootstrap=60, seed=3).fit,
        simulate_panel(),
        outcome="exited",
        unit="unit",
        time="time",
        treatment="treated",
        last_pre_period=4,
    )


@pytest.fixture(scope="module")
def lwdid():
    rng = np.random.default_rng(29)
    rows = []
    for u in range(30):
        g = 3 if u < 10 else (4 if u < 20 else 0)
        alpha = rng.normal()
        for t in range(1, 7):
            d = int(g > 0 and t >= g)
            rows.append(dict(unit=u, time=t, first=g, treat=d, y=alpha + rng.normal() + d))
    return _quiet(
        LWDiD().fit,
        pd.DataFrame(rows),
        outcome="y",
        unit="unit",
        time="time",
        treatment="treat",
        first_treat="first",
    )


@pytest.fixture(scope="module")
def twfe_es():
    return _quiet(
        TwoWayFixedEffects().fit,
        _two_group_calendar_panel(),
        outcome="outcome",
        unit="unit",
        time="period",
        treatment="treated",
        event_study=True,
        spec="within",
        post_periods=[3, 4, 5],
        reference_period=2,
    )


PRODUCERS = [
    "cs_varying",
    "cs_universal",
    "cs_boot",
    "dml",
    "sa",
    "imp",
    "imp_boot",
    "ts",
    "ts_boot",
    "stacked",
    "spill_es",
    "cont",
    "cont_boot",
    "edid",
    "wool",
    "ddd_stag",
    "dcdh",
    "lpdid",
    "had_es",
    "duration",
    "lwdid",
]


def _frame(res):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return res.to_dataframe(level="event_study")


# --------------------------------------------------------------------------- #
# (1) The schema is an identical prefix on every producer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("producer", PRODUCERS)
def test_prefix_schema_and_trailing_config_columns(request, producer):
    res = request.getfixturevalue(producer)
    cols = list(_frame(res).columns)
    assert cols[:N_SCHEMA] == SCHEMA
    assert cols[N_SCHEMA:] == TRAILING.get(producer, [])


def test_twfe_event_study_mode_is_the_container_itself(twfe_es):
    # The merged TWFE event-study mode RETURNS EventStudyResults: its level-less
    # to_dataframe() is the schema frame every other class now relays.
    assert list(twfe_es.to_dataframe().columns) == SCHEMA


# --------------------------------------------------------------------------- #
# (2) The frame IS the aggregate / builder frame (bit-equal)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("producer", PRODUCERS)
def test_equals_aggregate_or_builder_route(request, producer):
    res = request.getfixturevalue(producer)
    df = _frame(res)
    if producer in NON_MIXIN:
        expected = build_event_study_surface(res).to_dataframe()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = res.aggregate("event_study").to_dataframe()
    assert_frame_equal(df.iloc[:, :N_SCHEMA], expected)
    if producer == "wool":
        # Config columns are constant per fit; the weighting scheme is the
        # stored event-study aggregation's ("cell" by default).
        assert df["cohort_trends"].eq(res.cohort_trends).all()
        assert df["aggregation_weights"].eq("cell").all()
    if producer == "lwdid":
        assert df["rolling"].nunique() == 1
        assert df["vcov_type"].eq(res.vcov_type).all()


@pytest.mark.parametrize("producer", ["ts", "stacked", "sa", "lpdid", "wool"])
def test_default_level_is_event_study_where_it_was(request, producer):
    # These classes default to level="event_study": a plain to_dataframe() is
    # the unified frame (on analytical TwoStageDiD it now RECOMPUTES from the
    # kit instead of raising).
    res = request.getfixturevalue(producer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_frame_equal(res.to_dataframe(), res.to_dataframe(level="event_study"))


# --------------------------------------------------------------------------- #
# (3) Reference / estimand / count spot checks on the unified frame
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("producer", ["stacked", "imp_boot", "ts_boot", "spill_es", "duration"])
def test_reference_row_at_minus_one_with_nan_count(request, producer):
    df = _frame(request.getfixturevalue(producer))
    ref = df["is_reference"].to_numpy(dtype=bool)
    assert df.loc[ref, "event_time"].tolist() == [-1]
    assert df.loc[ref, "att"].tolist() == [0.0]
    assert np.isnan(df.loc[ref, "n"]).all()
    if producer != "duration":
        assert np.isfinite(df.loc[~ref, "n"].to_numpy(dtype=float)).all()


def test_sun_abraham_synthesizes_the_omitted_reference(sa):
    df = _frame(sa)
    ref = df["is_reference"].to_numpy(dtype=bool)
    assert df.loc[ref, "event_time"].tolist() == [-1]
    assert np.isnan(df.loc[ref, "n"]).all()
    # SunAbraham records the contributing-cohort count on every estimated row.
    assert np.isfinite(df.loc[~ref, "n"].to_numpy(dtype=float)).all()


def test_dcdh_frame_is_the_paper_convention_without_native_labels(dcdh):
    df = _frame(dcdh)
    ref = df["is_reference"].to_numpy(dtype=bool)
    assert df.loc[ref, "event_time"].tolist() == [0]
    assert (df["event_time"] < 0).any()  # placebo rows
    assert (df["event_time"] > 0).any()
    assert set(df["estimand"]) == {"att"}  # the DID^pl_l / "ref" / DID_l labels left
    # The retired native row shape is gone.
    assert "horizon" not in df.columns and "effect" not in df.columns
    assert np.isnan(df.loc[df["event_time"] < 0, "n"]).sum() == 0


def test_had_frame_carries_the_was_estimand_and_no_reference(had_es):
    df = _frame(had_es)
    assert set(df["estimand"]) == {had_es.target_parameter}
    assert not df["is_reference"].any()
    assert len(df) == len(had_es.event_times)


def test_wooldridge_frame_has_no_reference_row_and_nan_counts(wool):
    df = _frame(wool)
    assert not df["is_reference"].any()
    assert np.isnan(df["n"]).all()
    assert np.isnan(df["df"]).all()


def test_spillover_default_level_is_the_inherited_headline_frame(spill_es):
    assert_frame_equal(spill_es.to_dataframe(), pd.DataFrame([spill_es.to_dict()]))


# --------------------------------------------------------------------------- #
# (4) Error contract
# --------------------------------------------------------------------------- #


def test_wooldridge_before_aggregate_raises_instead_of_an_empty_frame(wool_fresh):
    pat = r"carries no event-study surface - call results\.aggregate\('event_study'\) first"
    with pytest.raises(ValueError, match=pat):
        wool_fresh.to_dataframe()
    with pytest.raises(ValueError, match=pat):
        wool_fresh.to_dataframe(level="event_study")


def test_spillover_aggregate_mode_raises(spill_agg):
    assert spill_agg.event_study_effects is None
    with pytest.raises(ValueError, match="refit with event_study=True"):
        spill_agg.to_dataframe(level="event_study")


def test_triple_difference_2x2x2_raises_the_single_att_error(ddd_2x2x2):
    with pytest.raises(
        ValueError, match=r"aggregate\('event_study'\): the 2x2x2 TripleDifference design"
    ):
        ddd_2x2x2.to_dataframe(level="event_study")


def _lpdid_results(event_study):
    return LPDiDResults(
        event_study=event_study,
        pooled=None,
        n_obs=10,
        n_treated_units=4,
        n_control_units=6,
        pre_window=2,
        post_window=0,
        control_group="clean",
        reweight=False,
        no_composition=False,
        pmd=None,
    )


_LPDID_COLUMNS = [
    "horizon",
    "coefficient",
    "se",
    "t_stat",
    "p_value",
    "conf_low",
    "conf_high",
    "n_obs",
]


def test_lpdid_absent_and_empty_native_frames_raise():
    with pytest.raises(ValueError, match="event_study dataframe was not computed"):
        _lpdid_results(None).to_dataframe(level="event_study")
    # A zero-row stored frame used to pass through as an empty frame; it now
    # raises the builder's absent-surface hint.
    with pytest.raises(ValueError, match="refit with only_pooled=False"):
        _lpdid_results(pd.DataFrame(columns=_LPDID_COLUMNS)).to_dataframe(level="event_study")


def test_stacked_legacy_pickle_without_surface_raises_the_refit_hint(stacked):
    legacy = dataclasses.replace(stacked, event_study_effects=None)
    with pytest.raises(ValueError, match=r"re-fit with diff-diff >= 3\.9"):
        legacy.to_dataframe(level="event_study")


def test_kit_less_legacy_pickle_raises_through_to_dataframe(ts):
    legacy = copy.copy(ts)
    object.__setattr__(legacy, "_aggregation_kit", None)
    with pytest.raises(ValueError, match="carries no aggregation kit"):
        legacy.to_dataframe()  # the default level is event_study


def test_bad_levels_on_the_newly_leveled_classes(had_es, spill_es, duration):
    with pytest.raises(ValueError, match="Use 'event_study'"):
        had_es.to_dataframe(level="periods")
    with pytest.raises(ValueError, match="Use 'simple' or 'event_study'"):
        spill_es.to_dataframe(level="group")
    with pytest.raises(ValueError, match="'periods', 'att' or 'event_study'"):
        duration.to_dataframe(level="bogus")


# --------------------------------------------------------------------------- #
# (5) Ledger
# --------------------------------------------------------------------------- #


def test_ledger_row_is_done_and_points_here():
    from tests.test_v4_matrix import ROWS as LEDGER_ROWS

    row = {r["id"]: r for r in LEDGER_ROWS}["M-093"]
    assert row["kind"] == "behavior"
    assert row["status"] == "done"
    assert row["test_ref"] == THIS_TEST_REF
    assert "TENTH AMENDMENT" in row["notes"]
