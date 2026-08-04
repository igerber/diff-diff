"""Unified event-study representation (ledger M-092, spec section 5).

This file is the ``test_ref`` for ledger row M-092. It covers:

- ``EventStudyResults`` container validation, reference marking, the pinned
  ``to_dataframe`` schema, and the summary-alpha contract.
- ``build_event_study_surface`` over all FOURTEEN producers: identical
  column schema, bit-exact quintet vs the native representation on
  non-reference rows, explicit ``is_reference`` marking (sentinels never
  leaking into the container schema), per-producer expected reference-row
  count, and absent-surface ValueErrors.

Fits are small and analytical (no bootstrap is needed for any producer's
event-study surface); fixtures are module-scoped.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.results_base import (
    EVENT_STUDY_SCHEMA,
    EventStudyResults,
    build_event_study_surface,
)
from diff_diff.spillover import SpilloverDiD
from diff_diff.sun_abraham import SunAbraham

# ===========================================================================
# Container unit tests (hand-built instances - no fits)
# ===========================================================================


def _tiny_surface(**overrides):
    kwargs = dict(
        event_time=np.array([-1, 0, 1]),
        att=np.array([0.0, 0.5, 0.6]),
        se=np.array([np.nan, 0.1, 0.1]),
        t_stat=np.array([np.nan, 5.0, 6.0]),
        p_value=np.array([np.nan, 0.0, 0.0]),
        conf_int_lower=np.array([np.nan, 0.3, 0.4]),
        conf_int_upper=np.array([np.nan, 0.7, 0.8]),
        is_reference=np.array([True, False, False]),
        n=np.array([np.nan, 10.0, 10.0]),
        n_kind="groups",
    )
    kwargs.update(overrides)
    return EventStudyResults(**kwargs)


def test_to_dataframe_schema_is_pinned():
    df = _tiny_surface().to_dataframe()
    assert tuple(df.columns) == EVENT_STUDY_SCHEMA


def test_reference_row_inferred_and_marked():
    surface = _tiny_surface()
    assert surface.reference_period == -1
    assert surface.is_reference.sum() == 1


def test_empty_surface_allowed():
    # A requested-but-empty event study (no estimable horizons) is a valid
    # zero-row surface with the pinned schema, not an error.
    surface = EventStudyResults(
        event_time=np.array([]),
        att=np.array([]),
        se=np.array([]),
        t_stat=np.array([]),
        p_value=np.array([]),
        conf_int_lower=np.array([]),
        conf_int_upper=np.array([]),
        is_reference=np.array([], dtype=bool),
        n=np.array([]),
    )
    assert surface.reference_period is None
    assert surface.reference_periods == []
    df = surface.to_dataframe()
    assert tuple(df.columns) == EVENT_STUDY_SCHEMA
    assert len(df) == 0
    assert isinstance(surface.summary(), str)


def test_multiple_reference_rows_allowed():
    # CallawaySantAnna universal base on a gapped grid materializes several
    # reference-only horizons; the container must represent them.
    surface = _tiny_surface(is_reference=np.array([True, True, False]))
    assert int(surface.is_reference.sum()) == 2
    # reference_period (single-scalar convenience) is None when not exactly one.
    assert surface.reference_period is None
    assert surface.reference_periods == [-1, 0]


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="align with event_time"):
        _tiny_surface(att=np.array([0.0, 0.5]))


def test_vcov_requires_index():
    with pytest.raises(ValueError, match="vcov and vcov_index together"):
        _tiny_surface(vcov=np.eye(3))


def test_summary_alpha_mismatch_raises():
    surface = _tiny_surface(alpha=0.05)
    with pytest.raises(ValueError, match="re-aggregate"):
        surface.summary(alpha=0.10)
    # None and the stored alpha are both fine.
    assert isinstance(surface.summary(), str)
    assert isinstance(surface.summary(alpha=0.05), str)


def test_object_dtype_calendar_event_time():
    # Calendar mode may carry string/period labels - no numeric assumptions.
    surface = _tiny_surface(
        event_time=np.array(["2018", "2019", "2020"], dtype=object),
        is_reference=np.array([True, False, False]),
        time_scale="calendar",
    )
    df = surface.to_dataframe()
    assert list(df["event_time"]) == ["2018", "2019", "2020"]
    assert surface.reference_period == "2018"


def test_to_dict_json_safe_timestamp_labels():
    # pandas.Timestamp calendar labels must serialize (json.dumps would raise
    # on raw Timestamp objects from .tolist()).
    import json

    surface = _tiny_surface(
        event_time=np.array(
            [pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01")],
            dtype=object,
        ),
        is_reference=np.array([True, False, False]),
        time_scale="calendar",
    )
    d = surface.to_dict()
    json.dumps(d)  # must not raise
    assert d["event_time"][0] == "2018-01-01T00:00:00"
    assert d["reference_period"] == "2018-01-01T00:00:00"


def test_to_dict_json_safe_period_labels():
    import json

    surface = _tiny_surface(
        event_time=np.array(
            [pd.Period("2018", "Y"), pd.Period("2019", "Y"), pd.Period("2020", "Y")],
            dtype=object,
        ),
        is_reference=np.array([True, False, False]),
        time_scale="calendar",
    )
    d = surface.to_dict()
    json.dumps(d)  # must not raise
    assert d["event_time"][0] == "2018"
    assert d["reference_period"] == "2018"


# ===========================================================================
# Provenance fields (base_period / anticipation / df_survey)
# ===========================================================================


def test_provenance_fields_default_none():
    surface = _tiny_surface()
    assert surface.base_period is None
    assert surface.anticipation is None
    assert surface.df_survey is None


def test_provenance_round_trips_through_to_dict():
    d = _tiny_surface(base_period="universal", anticipation=1, df_survey=7.0).to_dict()
    assert d["base_period"] == "universal"
    assert d["anticipation"] == 1
    assert d["df_survey"] == 7.0


def test_builder_threads_cs_provenance():
    # The relative-dict builder reads base_period/anticipation off the
    # producer and resolves the scalar df_survey (None here: no survey
    # design and no bare-cluster df carrier).
    class _FakeCS:
        alpha = 0.05
        base_period = "universal"
        anticipation = 1
        event_study_effects = {
            -2: {"effect": 0.0, "se": np.nan, "n_groups": 0},
            0: {"effect": 1.0, "se": 0.1, "n_groups": 5},
        }

    surface = build_event_study_surface(_FakeCS())
    assert surface.base_period == "universal"
    assert surface.anticipation == 1
    assert surface.df_survey is None


def test_empty_surface_threads_provenance():
    # The requested-but-empty early return must carry provenance too - a
    # balance_e-emptied aggregation would otherwise read as provenance-free.
    class _EmptyCS:
        alpha = 0.05
        base_period = "universal"
        anticipation = 1
        event_study_effects: dict = {}

    surface = build_event_study_surface(_EmptyCS())
    assert surface.event_time.shape[0] == 0
    assert surface.base_period == "universal"
    assert surface.anticipation == 1
    assert surface.df_survey is None


def test_df_survey_replicate_undefined_maps_to_zero_sentinel():
    # survey_metadata present, df_survey undefined, replicate design ->
    # the 0.0 sentinel (fails closed to NaN critical values downstream).
    class _SM:
        df_survey = None
        replicate_method = "brr"

    class _Fake:
        alpha = 0.05
        survey_metadata = _SM()
        event_study_effects = {0: {"effect": 1.0, "se": 0.1, "n_groups": 5}}

    surface = build_event_study_surface(_Fake())
    assert surface.df_survey == 0.0


def test_df_survey_prefers_survey_metadata_over_df_inference():
    class _SM:
        df_survey = 12
        replicate_method = None

    class _Fake:
        alpha = 0.05
        survey_metadata = _SM()
        df_inference = 30
        event_study_effects = {0: {"effect": 1.0, "se": 0.1, "n_groups": 5}}

    surface = build_event_study_surface(_Fake())
    assert surface.df_survey == 12.0


def test_df_survey_bare_cluster_falls_back_to_df_inference():
    class _Fake:
        alpha = 0.05
        survey_metadata = None
        df_inference = 19
        event_study_effects = {0: {"effect": 1.0, "se": 0.1, "n_groups": 5}}

    surface = build_event_study_surface(_Fake())
    assert surface.df_survey == 19.0


# ===========================================================================
# Producer builders (small analytical fits)
# ===========================================================================


@pytest.fixture(scope="module")
def surfaces():
    """Build every producer's surface once; return {label: (native, surface)}."""
    out = {}

    from diff_diff import (
        CallawaySantAnna,
        EfficientDiD,
        LPDiD,
        MultiPeriodDiD,
        StackedDiD,
        StaggeredTripleDifference,
        WooldridgeDiD,
        generate_staggered_ddd_data,
    )
    from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille
    from diff_diff.continuous_did import ContinuousDiD
    from diff_diff.datasets import load_mpdta
    from diff_diff.imputation import ImputationDiD
    from diff_diff.prep import generate_staggered_data as prep_staggered
    from diff_diff.prep_dgp import (
        generate_continuous_did_data,
        generate_reversible_did_data,
    )
    from diff_diff.prep_dgp import generate_staggered_data as dgp_staggered
    from diff_diff.spillover import SpilloverDiD
    from diff_diff.sun_abraham import SunAbraham
    from diff_diff.two_stage import TwoStageDiD

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 1. CallawaySantAnna
        cs_data = prep_staggered(
            n_units=80, n_periods=8, cohort_periods=[4], treatment_effect=2.0, seed=42
        )
        cs = CallawaySantAnna(n_bootstrap=0).fit(
            cs_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        out["CallawaySantAnna"] = (cs, build_event_study_surface(cs))

        # 2. SunAbraham (always-on)
        sa_data = _sa_panel()
        sa = SunAbraham().fit(
            sa_data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        out["SunAbraham"] = (sa, build_event_study_surface(sa))

        # 3. ImputationDiD
        imp_data = _bjs_panel()
        imp = ImputationDiD().fit(
            imp_data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        out["ImputationDiD"] = (imp, build_event_study_surface(imp))

        # 4. TwoStageDiD (same panel shape)
        ts = TwoStageDiD().fit(
            imp_data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        out["TwoStageDiD"] = (ts, build_event_study_surface(ts))

        # 5. StackedDiD
        st_data = dgp_staggered(
            n_units=120,
            n_periods=12,
            cohort_periods=[4, 6, 8],
            never_treated_frac=0.3,
            treatment_effect=5.0,
            dynamic_effects=True,
            seed=42,
        )
        st = StackedDiD(kappa_pre=2, kappa_post=2).fit(
            st_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        out["StackedDiD"] = (st, build_event_study_surface(st))

        # 6. SpilloverDiD
        from tests._dgp_utils import generate_butts_staggered_dgp

        sp_data = generate_butts_staggered_dgp(seed=42)
        sp = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
            vcov_type="hc1",
            event_study=True,
            horizon_max=2,
            anticipation=0,
        ).fit(sp_data, outcome="y", unit="unit", time="time", first_treat="first_treat")
        out["SpilloverDiD"] = (sp, build_event_study_surface(sp))

        # 7. ContinuousDiD
        cd_data = generate_continuous_did_data(
            n_units=200, n_periods=5, cohort_periods=[2, 4], seed=42, noise_sd=0.5
        )
        cd = ContinuousDiD(control_group="not_yet_treated", n_bootstrap=0).fit(
            cd_data, "outcome", "unit", "period", "first_treat", "dose", aggregate="eventstudy"
        )
        out["ContinuousDiD"] = (cd, build_event_study_surface(cd))

        # 8. EfficientDiD
        ed_data = _efficient_panel()
        ed = EfficientDiD().fit(
            ed_data, "y", "unit", "time", "first_treat", aggregate="event_study"
        )
        out["EfficientDiD"] = (ed, build_event_study_surface(ed))

        # 9. WooldridgeDiD (post-fit aggregate)
        mp = load_mpdta()
        wd = WooldridgeDiD(control_group="never_treated").fit(
            mp, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
        wd.aggregate("event")
        out["WooldridgeDiD"] = (wd, build_event_study_surface(wd))

        # 10. StaggeredTripleDifference
        sddd_data = generate_staggered_ddd_data(n_units=300, treatment_effect=3.0, seed=42)
        sddd = StaggeredTripleDifference().fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
        out["StaggeredTripleDifference"] = (sddd, build_event_study_surface(sddd))

        # 11. MultiPeriodDiD
        mpd_data = _mpd_panel()
        mpd = MultiPeriodDiD().fit(
            mpd_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5],
            reference_period=2,
        )
        out["MultiPeriodDiD"] = (mpd, build_event_study_surface(mpd))

        # 12. LPDiD
        lp_data = _lpdid_panel()
        lp = LPDiD(pre_window=2, post_window=2).fit(
            lp_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        out["LPDiD"] = (lp, build_event_study_surface(lp))

        # 13. ChaisemartinDHaultfoeuille (event + placebo)
        dcdh_data = generate_reversible_did_data(
            n_groups=50, n_periods=10, pattern="joiners_only", seed=42
        )
        dcdh = ChaisemartinDHaultfoeuille(twfe_diagnostic=False).fit(
            dcdh_data,
            outcome="outcome",
            unit="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        out["ChaisemartinDHaultfoeuille"] = (dcdh, build_event_study_surface(dcdh))

        # 14. HeterogeneousAdoptionDiD event study
        had_data, had_es = _had_event_study()
        out["HeterogeneousAdoptionDiD"] = (had_es, build_event_study_surface(had_es))

    return out


ALL_PRODUCERS = [
    "CallawaySantAnna",
    "SunAbraham",
    "ImputationDiD",
    "TwoStageDiD",
    "StackedDiD",
    "SpilloverDiD",
    "ContinuousDiD",
    "EfficientDiD",
    "WooldridgeDiD",
    "StaggeredTripleDifference",
    "MultiPeriodDiD",
    "LPDiD",
    "ChaisemartinDHaultfoeuille",
    "HeterogeneousAdoptionDiD",
]


def _expected_reference_labels(label, native):
    """The exact event_time(s) the surface should mark as reference.

    Reference resolution is producer-specific, NOT a count heuristic (a zero
    count is not a universal reference marker - SpilloverDiD uses n_obs=0 for
    both its genuine reference and its non-estimable rectangular horizons):

    - Explicit reference_period wins (SpilloverDiD, MultiPeriodDiD).
    - Structurally-omitted baselines are synthesized (SunAbraham at
      -1-anticipation; dCDH at 0).
    - LPDiD's horizon==-1 base row.
    - Wooldridge / HAD omit no baseline (no reference).
    - The remaining relative-dict producers mark EVERY zero-count sentinel
      row (CallawaySantAnna universal base on a gapped grid can materialize
      several); CS varying and EfficientDiD estimate e=-1, so none.
    """
    if label in ("MultiPeriodDiD", "SpilloverDiD"):
        return [] if native.reference_period is None else [native.reference_period]
    if label == "ChaisemartinDHaultfoeuille":
        return [0]
    if label == "SunAbraham":
        return [-1 - (getattr(native, "anticipation", 0) or 0)]
    if label == "LPDiD":
        return [-1] if (native.event_study["horizon"] == -1).any() else []
    if label in ("WooldridgeDiD", "HeterogeneousAdoptionDiD"):
        return []
    # Count-sentinel producers: a zero-count row is the reference only when its
    # effect is finite (0.0); a NaN-effect zero-count row is non-estimable.
    return sorted(
        e
        for e, row in native.event_study_effects.items()
        if (row.get("n_groups", 1) == 0 or row.get("n_obs", 1) == 0)
        and np.isfinite(row.get("att", row.get("effect", np.nan)))
    )


def test_all_producers_built(surfaces):
    assert set(surfaces) == set(ALL_PRODUCERS)


@pytest.mark.parametrize("label", ALL_PRODUCERS)
def test_identical_schema(label, surfaces):
    _, surface = surfaces[label]
    assert tuple(surface.to_dataframe().columns) == EVENT_STUDY_SCHEMA


@pytest.mark.parametrize("label", ALL_PRODUCERS)
def test_reference_labels_match_native(label, surfaces):
    # is_reference marks EXACTLY the producer-specific reference label(s) -
    # never an arbitrary zero-count row (the count heuristic would mislabel
    # SpilloverDiD's non-estimable horizons and miss SunAbraham's anchor).
    native, surface = surfaces[label]
    expected = _expected_reference_labels(label, native)
    marked = sorted(surface.event_time[surface.is_reference].tolist())
    assert marked == expected, f"{label}: marked {marked}, expected {expected}"


@pytest.mark.parametrize("label", ALL_PRODUCERS)
def test_reference_row_values_normalized(label, surfaces):
    _, surface = surfaces[label]
    ref = surface.is_reference
    if ref.any():
        assert surface.att[ref][0] == 0.0
        assert np.isnan(surface.se[ref][0])
        assert np.isnan(surface.conf_int_lower[ref][0])


@pytest.mark.parametrize("label", ALL_PRODUCERS)
def test_no_sentinel_in_schema(label, surfaces):
    # The retiring sentinels (n_groups==0 / n_obs==0) must not surface as a
    # zero count on the marked reference row: it is NaN there.
    _, surface = surfaces[label]
    ref = surface.is_reference
    if ref.any():
        assert np.isnan(surface.n[ref][0])


@pytest.mark.parametrize("label", ALL_PRODUCERS)
def test_bit_exact_quintet_nonreference(label, surfaces):
    # Full canonical quintet (att/se/t_stat/p_value/both CI bounds) on every
    # native non-reference row, across all fourteen producers.
    native, surface = surfaces[label]
    non_ref = ~surface.is_reference
    native_map = _native_event_map(label, native)
    checked = 0
    for i in np.where(non_ref)[0]:
        e = surface.event_time[i]
        if e not in native_map:
            continue
        exp = native_map[e]
        assert surface.att[i] == pytest.approx(exp["att"], nan_ok=True)
        assert surface.se[i] == pytest.approx(exp["se"], nan_ok=True)
        assert surface.t_stat[i] == pytest.approx(exp["t_stat"], nan_ok=True)
        assert surface.p_value[i] == pytest.approx(exp["p_value"], nan_ok=True)
        assert surface.conf_int_lower[i] == pytest.approx(exp["ci_lo"], nan_ok=True)
        assert surface.conf_int_upper[i] == pytest.approx(exp["ci_hi"], nan_ok=True)
        checked += 1
    assert checked > 0, f"{label}: no non-reference rows compared"


def test_cs_vcov_alignment(surfaces):
    native, surface = surfaces["CallawaySantAnna"]
    assert surface.vcov is not None
    assert surface.vcov_index is not None
    # vcov_index labels are a subset of event_time labels.
    assert set(surface.vcov_index.tolist()).issubset(set(surface.event_time.tolist()))
    # Same ordered index the native surface exposes.
    np.testing.assert_array_equal(np.asarray(native.event_study_vcov_index), surface.vcov_index)


def test_mpd_vcov_subblock(surfaces):
    native, surface = surfaces["MultiPeriodDiD"]
    # MPD supplies a full vcov + interaction_indices; the surface exposes an
    # ordered sub-block. Presence (not exact values) is the contract here.
    if native.vcov is not None and native.interaction_indices:
        assert surface.vcov is not None
        assert surface.vcov.shape[0] == surface.vcov_index.shape[0]


def test_dcdh_convention_and_placebo_merge(surfaces):
    native, surface = surfaces["ChaisemartinDHaultfoeuille"]
    assert surface.event_time_convention == "l1_first_switch"
    times = surface.event_time.tolist()
    assert 0 in times  # synthesized reference
    assert any(t < 0 for t in times)  # placebo horizons merged in
    assert any(t >= 1 for t in times)  # post horizons


def test_dcdh_n_kind_is_groups_not_obs(surfaces):
    # dCDH stores N_l (eligible switcher GROUPS) under its legacy "n_obs" key.
    # The unified surface must label it "groups", never "obs" - a consumer
    # doing sample-size logic would otherwise misread switcher groups as
    # observations.
    native, surface = surfaces["ChaisemartinDHaultfoeuille"]
    assert surface.n_kind == "groups"
    df = surface.to_dataframe()
    # Non-reference n values equal the native N_l (from event_study_effects /
    # placebo_event_study), not an observation count.
    native_n = {}
    for k, row in native.event_study_effects.items():
        native_n[k] = row.get("n_obs")
    for k, row in (native.placebo_event_study or {}).items():
        native_n[k] = row.get("n_obs")
    for _, r in df[~df["is_reference"]].iterrows():
        e = r["event_time"]
        if e in native_n and native_n[e] is not None:
            assert r["n"] == float(native_n[e])


def test_dcdh_l_max_none_count_kind_is_switcher_cells():
    # Legacy single-horizon path (L_max=None): the count under "n_obs" is N_S,
    # the number of switching (g,t) CELLS - one group can contribute several -
    # so n_kind is "switcher_cells", NOT "groups" and NOT "obs".
    class _FakeDCDHLegacy:
        alpha = 0.05
        L_max = None
        placebo_event_study = None
        event_study_effects = {
            1: {
                "effect": 1.0,
                "se": 0.2,
                "t_stat": 5.0,
                "p_value": 0.0,
                "conf_int": (0.6, 1.4),
                "n_obs": 30,  # N_S switching cells, NOT observations or groups
            }
        }
        sup_t_bands = None

    surface = build_event_study_surface(_FakeDCDHLegacy())
    assert surface.n_kind == "switcher_cells"


def test_spillover_oversized_horizon_single_reference():
    # Regression for the count-heuristic bug: with horizon_max=4 SpilloverDiD
    # emits BOTH a genuine reference (reference_period, n_obs=0, coef=0) and a
    # non-estimable rectangular horizon (n_obs=0, coef=NaN). Only the former
    # is the reference; the latter must survive as a non-reference NaN row.
    from tests._dgp_utils import generate_butts_staggered_dgp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
            vcov_type="hc1",
            event_study=True,
            horizon_max=4,
            anticipation=0,
        ).fit(
            generate_butts_staggered_dgp(seed=42),
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
    surface = build_event_study_surface(sp)
    df = surface.to_dataframe()
    # Native has >=2 zero-count rows; exactly one is the reference.
    zero_native = [k for k, r in sp.event_study_effects.items() if r.get("n_obs") == 0]
    assert len(zero_native) >= 2
    marked = df[df["is_reference"]]["event_time"].tolist()
    assert marked == [sp.reference_period]
    # A non-reference zero-count horizon keeps its NaN att and n=0.
    other_zero = next(k for k in zero_native if k != sp.reference_period)
    row = df[df["event_time"] == other_zero].iloc[0]
    assert not bool(row["is_reference"])
    assert np.isnan(row["att"])
    assert row["n"] == 0.0


def test_sun_abraham_synthesizes_anticipation_reference():
    # SunAbraham omits its baseline; positive anticipation shifts the
    # synthesized reference to e = -1 - anticipation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sa = SunAbraham(anticipation=1).fit(
            _sa_panel(n_periods=10),
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
    surface = build_event_study_surface(sa)
    marked = surface.event_time[surface.is_reference].tolist()
    assert marked == [-2]  # -1 - anticipation
    ref_idx = int(np.where(surface.is_reference)[0][0])
    assert surface.att[ref_idx] == 0.0
    assert np.isnan(surface.se[ref_idx])


def test_sun_abraham_unobserved_reference_not_fabricated():
    # SunAbraham records reference_period = -1 - anticipation even when that
    # period was never observed (reference_observed=False). The adapter must
    # NOT invent the anchor row.
    class _FakeSA:
        alpha = 0.05
        anticipation = 0
        reference_period = -1
        reference_observed = False
        event_study_effects = {
            0: {"effect": 1.0, "se": 0.1, "n_groups": 5},
            1: {"effect": 1.2, "se": 0.1, "n_groups": 5},
            2: {"effect": 1.3, "se": 0.1, "n_groups": 5},
        }

    surface = build_event_study_surface(_FakeSA())
    assert int(surface.is_reference.sum()) == 0
    assert -1 not in surface.event_time.tolist()


def test_sun_abraham_gapped_grid_unobserved_reference():
    # Reviewer's case: native keys {-2, 0}. -1 falls BETWEEN them, so a
    # range-based guard would wrongly synthesize it. reference_observed=False
    # (the -1 gap was never in the data) => no reference row.
    class _FakeSAGap:
        alpha = 0.05
        anticipation = 0
        reference_period = -1
        reference_observed = False
        event_study_effects = {
            -2: {"effect": -0.1, "se": 0.1, "n_groups": 5},
            0: {"effect": 1.0, "se": 0.1, "n_groups": 5},
        }

    surface = build_event_study_surface(_FakeSAGap())
    assert int(surface.is_reference.sum()) == 0
    assert -1 not in surface.event_time.tolist()


def test_sun_abraham_gapped_grid_observed_reference_synthesized():
    # Same keys {-2, 0} but the -1 anchor WAS observed (omitted baseline):
    # synthesize the reference row.
    class _FakeSAObs:
        alpha = 0.05
        anticipation = 0
        reference_period = -1
        reference_observed = True
        event_study_effects = {
            -2: {"effect": -0.1, "se": 0.1, "n_groups": 5},
            0: {"effect": 1.0, "se": 0.1, "n_groups": 5},
        }

    surface = build_event_study_surface(_FakeSAObs())
    marked = surface.event_time[surface.is_reference].tolist()
    assert marked == [-1]
    ref_idx = int(np.where(surface.is_reference)[0][0])
    assert surface.att[ref_idx] == 0.0
    assert np.isnan(surface.se[ref_idx])


def test_cs_universal_gapped_multiple_references():
    # Regression: CS base_period="universal" on a gapped grid ({1,3,6},
    # cohorts {3,6}) materializes each cohort's positional base as its own
    # reference-only horizon (all n_groups==0). The surface must represent
    # all of them - the count path marks every sentinel, and the container
    # allows multiple references.
    from diff_diff import CallawaySantAnna

    rng = np.random.RandomState(3)
    rows = []
    for u in range(40):
        g = 3 if u < 15 else (6 if u < 30 else 0)
        ufe = rng.randn() * 2
        for t in (1, 3, 6):
            post = 1 if (g > 0 and t >= g) else 0
            rows.append(
                {
                    "unit": u,
                    "period": t,
                    "outcome": ufe + 0.3 * t + 2.0 * post + rng.randn() * 0.5,
                    "first_treat": g,
                }
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = CallawaySantAnna(n_bootstrap=0, base_period="universal").fit(
            pd.DataFrame(rows),
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
    native_refs = sorted(e for e, r in cs.event_study_effects.items() if r.get("n_groups") == 0)
    assert len(native_refs) >= 2  # the case only matters when >1
    surface = build_event_study_surface(cs)
    marked = sorted(surface.event_time[surface.is_reference].tolist())
    assert marked == native_refs
    assert surface.reference_period is None  # not exactly one
    assert surface.reference_periods == native_refs
    # Every reference row is normalized (att=0, NaN inference).
    df = surface.to_dataframe()
    ref_df = df[df["is_reference"]]
    assert (ref_df["att"] == 0.0).all()
    assert ref_df["se"].isna().all()


def test_requested_but_empty_event_study_builds_zero_row_surface():
    # event_study_effects == {} means REQUESTED but no estimable horizons
    # (EfficientDiD balance_e removing every cohort) - a zero-row surface,
    # distinct from None (never requested -> raises).
    class _RequestedEmpty:
        alpha = 0.05
        event_study_effects: dict = {}

    surface = build_event_study_surface(_RequestedEmpty())
    assert len(surface.to_dataframe()) == 0
    assert surface.reference_periods == []


def test_absent_surface_distinguished_from_empty():
    # None (attribute present but None / missing) still raises the refit hint.
    class _NotRequested:
        alpha = 0.05
        event_study_effects = None

    with pytest.raises(ValueError, match="no event-study surface"):
        build_event_study_surface(_NotRequested())


def test_zero_count_finite_nonzero_effect_raises():
    # A zero-count row with a finite NONZERO effect is malformed (a reference
    # must be exactly 0) - fail loudly rather than silently rewrite to 0.
    class _Malformed:
        alpha = 0.05
        event_study_effects = {
            0: {"effect": 1.0, "se": 0.1, "n_obs": 40},
            1: {"effect": 3.3, "se": np.nan, "n_obs": 0},  # nonzero + zero count
        }

    with pytest.raises(ValueError, match="finite nonzero effect"):
        build_event_study_surface(_Malformed())


def test_zero_count_nan_horizon_is_not_a_reference():
    # TwoStageDiD emits effect=NaN, n_obs=0 for an estimated horizon whose
    # observations are all filtered (two_stage_aggregation.py::_stage2_event_study) - distinct from
    # its effect=0.0, n_obs=0 reference. The count-sentinel adapter path must
    # mark ONLY the finite-effect (0.0) row as reference and preserve the
    # NaN-effect horizon as a non-reference NaN row (never normalize it to 0).
    class _FakeTwoStage:
        alpha = 0.05
        event_study_effects = {
            -1: {  # the reference: effect 0.0, n_obs 0
                "effect": 0.0,
                "se": 0.0,
                "t_stat": np.nan,
                "p_value": np.nan,
                "conf_int": (0.0, 0.0),
                "n_obs": 0,
            },
            0: {  # an identified horizon
                "effect": 1.5,
                "se": 0.2,
                "t_stat": 7.5,
                "p_value": 0.0,
                "conf_int": (1.1, 1.9),
                "n_obs": 40,
            },
            1: {  # non-estimable: effect NaN, n_obs 0 (NOT a reference)
                "effect": np.nan,
                "se": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "conf_int": (np.nan, np.nan),
                "n_obs": 0,
            },
        }

    surface = build_event_study_surface(_FakeTwoStage())
    df = surface.to_dataframe()
    marked = sorted(surface.event_time[surface.is_reference].tolist())
    assert marked == [-1], "only the finite-effect zero-count row is the reference"
    # The NaN-effect zero-count horizon survives untouched.
    row = df[df["event_time"] == 1].iloc[0]
    assert not bool(row["is_reference"])
    assert np.isnan(row["att"])
    assert row["n"] == 0.0


def test_reference_period_scalar_forced_none_when_not_unique():
    # A caller cannot leave reference_period disagreeing with reference_periods:
    # the scalar is authoritative only when there is exactly one reference.
    surface = _tiny_surface(
        is_reference=np.array([True, True, False]),
        reference_period=99,  # bogus caller-supplied scalar
    )
    assert surface.reference_period is None
    assert surface.reference_periods == [-1, 0]


def test_single_reference_scalar_derived_over_supplied():
    # With exactly one reference, a mismatched caller-supplied scalar is
    # overridden by the marked label - never retained.
    surface = _tiny_surface(
        is_reference=np.array([False, True, False]),  # marks event_time 0
        reference_period=99,
    )
    assert surface.reference_period == 0


def test_reference_row_inference_normalized_by_container():
    # The public container enforces the reference-row contract itself: finite
    # inference on a marked row is normalized to att=0 / NaN, not trusted.
    surface = _tiny_surface(
        att=np.array([5.0, 0.5, 0.6]),  # bogus finite att on the reference
        se=np.array([9.9, 0.1, 0.1]),
        n=np.array([99.0, 10.0, 10.0]),
        is_reference=np.array([True, False, False]),
    )
    assert surface.att[0] == 0.0
    assert np.isnan(surface.se[0])
    assert np.isnan(surface.t_stat[0])
    assert np.isnan(surface.n[0])


def test_cband_requires_both_bounds():
    with pytest.raises(ValueError, match="cband_lower and cband_upper together"):
        _tiny_surface(cband_lower=np.array([np.nan, 0.2, 0.3]))


def test_container_does_not_mutate_caller_arrays():
    # __post_init__ normalizes reference rows in place; it must copy first so
    # the caller's own arrays are never overwritten.
    att = np.array([5.0, 0.5, 0.6])  # finite att on the reference row (index 0)
    se = np.array([9.9, 0.1, 0.1])
    _tiny_surface(att=att, se=se, is_reference=np.array([True, False, False]))
    assert att[0] == 5.0  # untouched
    assert se[0] == 9.9


def test_container_accepts_read_only_arrays():
    att = np.array([0.0, 0.5, 0.6])
    att.setflags(write=False)
    se = np.array([np.nan, 0.1, 0.1])
    se.setflags(write=False)
    surface = _tiny_surface(att=att, se=se, is_reference=np.array([True, False, False]))
    assert surface.att[0] == 0.0  # constructed without error


def test_vcov_round_trips_through_to_dict():
    import json

    vcov = np.array([[0.04, 0.01], [0.01, 0.09]])
    surface = _tiny_surface(vcov=vcov, vcov_index=np.array([0, 1]))
    d = surface.to_dict()
    json.dumps(d)  # must not raise
    np.testing.assert_array_equal(np.asarray(d["vcov"]), vcov)
    assert d["vcov_index"] == [0, 1]


def test_dcdh_crit_value_carried_from_sup_t_bands():
    # Constructed dCDH-shaped result: cband_crit_value is copied from
    # sup_t_bands["crit_value"] (bootstrap-only in practice).
    class _FakeDCDH:
        alpha = 0.05
        placebo_event_study = {-1: {"effect": 0.1, "se": 0.05, "n_obs": 40}}
        event_study_effects = {
            1: {
                "effect": 1.0,
                "se": 0.2,
                "t_stat": 5.0,
                "p_value": 0.0,
                "conf_int": (0.6, 1.4),
                "n_obs": 50,
                "cband_conf_int": (0.4, 1.6),
            }
        }
        sup_t_bands = {"crit_value": 2.71, "alpha": 0.05}

    surface = build_event_study_surface(_FakeDCDH())
    assert surface.cband_crit_value == 2.71
    assert surface.event_time_convention == "l1_first_switch"


def test_had_carries_cband_and_crit_value():
    # HAD's simultaneous-band bounds must travel with their critical value.
    class _FakeHAD:
        alpha = 0.05
        event_times = np.array([0, 1, 2])
        att = np.array([1.0, 1.2, 1.3])
        se = np.array([0.2, 0.2, 0.2])
        t_stat = np.array([5.0, 6.0, 6.5])
        p_value = np.array([0.0, 0.0, 0.0])
        conf_int_low = np.array([0.6, 0.8, 0.9])
        conf_int_high = np.array([1.4, 1.6, 1.7])
        n_obs_per_horizon = np.array([40, 40, 40])
        cband_low = np.array([0.4, 0.6, 0.7])
        cband_high = np.array([1.6, 1.8, 1.9])
        cband_crit_value = 2.5

    surface = build_event_study_surface(_FakeHAD())
    assert surface.cband_crit_value == 2.5
    np.testing.assert_array_equal(surface.cband_lower, _FakeHAD.cband_low)
    np.testing.assert_array_equal(surface.cband_upper, _FakeHAD.cband_high)


def test_wooldridge_key_normalized(surfaces):
    native, surface = surfaces["WooldridgeDiD"]
    # Native uses inner key "att"; the surface exposes canonical att column.
    assert "att" in surface.to_dataframe().columns
    assert surface.n_kind is None  # Wooldridge records no per-row count


def test_lpdid_n_kind_obs(surfaces):
    _, surface = surfaces["LPDiD"]
    assert surface.n_kind == "obs"
    # n_clusters dropped from the unified view; the schema is the pinned one.
    assert "n_clusters" not in surface.to_dataframe().columns


def test_absent_surface_raises():
    from diff_diff import CallawaySantAnna
    from diff_diff.prep import generate_staggered_data as prep_staggered

    data = prep_staggered(n_units=60, n_periods=6, cohort_periods=[3], treatment_effect=2.0, seed=1)
    # aggregate defaults to "simple": no event_study_effects populated.
    cs = CallawaySantAnna(n_bootstrap=0).fit(
        data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
    )
    with pytest.raises(ValueError, match="event-study"):
        build_event_study_surface(cs)


# ===========================================================================
# Panel builders (inlined from the existing suite; small + analytical)
# ===========================================================================


def _sa_panel(n_units=80, n_periods=8, n_cohorts=3, seed=42):
    np.random.seed(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    n_never = int(n_units * 0.3)
    n_treated = n_units - n_never
    cohort_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int)
    first_treat = np.zeros(n_units)
    if n_treated > 0:
        first_treat[n_never:] = cohort_periods[
            np.random.choice(len(cohort_periods), size=n_treated)
        ]
    fte = np.repeat(first_treat, n_periods)
    unit_fe = np.repeat(np.random.randn(n_units) * 2, n_periods)
    time_fe = np.tile(np.linspace(0, 1, n_periods), n_units)
    post = (times >= fte) & (fte > 0)
    dyn = 2.0 * (1 + 0.1 * np.maximum(times - fte, 0))
    y = unit_fe + time_fe + dyn * post + np.random.randn(len(units)) * 0.5
    return pd.DataFrame(
        {"unit": units, "time": times, "outcome": y, "first_treat": fte.astype(int)}
    )


def _bjs_panel(n_units=100, n_periods=10, seed=42):
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    n_never = int(n_units * 0.3)
    n_treated = n_units - n_never
    cohort_periods = np.array([3, 5, 7])
    first_treat = np.zeros(n_units, dtype=int)
    if n_treated > 0:
        first_treat[n_never:] = cohort_periods[rng.choice(len(cohort_periods), size=n_treated)]
    fte = np.repeat(first_treat, n_periods)
    unit_fe = np.repeat(rng.standard_normal(n_units) * 2.0, n_periods)
    time_fe = np.tile(np.linspace(0, 1, n_periods), n_units)
    post = (times >= fte) & (fte > 0)
    rel = times - fte
    mult = 1 + 0.1 * np.maximum(rel, 0)
    y = unit_fe + time_fe + (2.0 * mult) * post + rng.standard_normal(len(units)) * 0.5
    return pd.DataFrame({"unit": units, "time": times, "outcome": y, "first_treat": fte})


def _efficient_panel(n_units=100, n_periods=5, n_treated=50, treat_period=3, seed=42):
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    ft = np.full(n_units, np.inf)
    ft[:n_treated] = treat_period
    ft_col = np.repeat(ft, n_periods)
    unit_fe = np.repeat(rng.normal(0, 1, n_units), n_periods)
    time_fe = np.tile(np.arange(1, n_periods + 1) * 0.5, n_units)
    tau = np.where((ft_col < np.inf) & (times >= ft_col), 2.0, 0.0)
    y = unit_fe + time_fe + tau + rng.normal(0, 0.5, len(units))
    return pd.DataFrame({"unit": units, "time": times, "first_treat": ft_col, "y": y})


def _mpd_panel(n_units=100, n_periods=6, seed=42):
    np.random.seed(seed)
    rows = []
    for unit in range(n_units):
        is_treated = unit < n_units // 2
        unit_effect = np.random.normal(0, 1)
        for period in range(n_periods):
            y = 10.0 + unit_effect + period * 0.5
            if is_treated and period >= 3:
                y += 3.0
            y += np.random.normal(0, 0.5)
            rows.append({"unit": unit, "period": period, "treated": int(is_treated), "outcome": y})
    return pd.DataFrame(rows)


def _lpdid_panel(seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for _ in range(12):
        uid += 1
        alpha = rng.normal(0.0, 0.5)
        for t in range(8):
            y0 = alpha + 0.5 * t + rng.normal(0.0, 0.2)
            k = t - 4
            effect = (1.0 + 0.5 * k) if k >= 0 else 0.0
            rows.append(
                {"unit": uid, "time": t, "y": y0 + effect, "treat": int(t >= 4), "first_treat": 4.0}
            )
    for _ in range(12):
        uid += 1
        alpha = rng.normal(0.0, 0.5)
        for t in range(8):
            y0 = alpha + 0.5 * t + rng.normal(0.0, 0.2)
            rows.append({"unit": uid, "time": t, "y": y0, "treat": 0, "first_treat": np.inf})
    return pd.DataFrame(rows)


def _had_event_study():
    from diff_diff.had import HeterogeneousAdoptionDiD

    rng = np.random.default_rng(0)
    G = 300
    mass_n = int(0.3 * G)
    d_at_F = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5, 1.0, G - mass_n)])
    units = np.arange(G)
    alpha_g = 0.5 * rng.standard_normal(G)
    F, n_periods = 3, 5
    rows = []
    for g in units:
        d_g = float(d_at_F[g])
        for t in range(1, n_periods + 1):
            dose = d_g if t >= F else 0.0
            outcome = alpha_g[g] + 0.3 * dose + 0.1 * rng.standard_normal()
            rows.append({"unit": g, "period": t, "dose": dose, "outcome": outcome})
    panel = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es = HeterogeneousAdoptionDiD(design="mass_point").fit(
            panel, "outcome", "dose", "period", "unit", aggregate="event_study"
        )
    return panel, es


def _native_event_map(label, native):
    """Map event_time -> full canonical quintet from the native representation."""
    out = {}
    if label == "LPDiD":
        for _, row in native.event_study.iterrows():
            out[row["horizon"]] = {
                "att": row["coefficient"],
                "se": row["se"],
                "t_stat": row["t_stat"],
                "p_value": row["p_value"],
                "ci_lo": row["conf_low"],
                "ci_hi": row["conf_high"],
            }
    elif label == "MultiPeriodDiD":
        for period, pe in native.period_effects.items():
            out[period] = {
                "att": pe.effect,
                "se": pe.se,
                "t_stat": pe.t_stat,
                "p_value": pe.p_value,
                "ci_lo": pe.conf_int[0],
                "ci_hi": pe.conf_int[1],
            }
    elif label == "HeterogeneousAdoptionDiD":
        for i, e in enumerate(native.event_times):
            out[e] = {
                "att": native.att[i],
                "se": native.se[i],
                "t_stat": native.t_stat[i],
                "p_value": native.p_value[i],
                "ci_lo": native.conf_int_low[i],
                "ci_hi": native.conf_int_high[i],
            }
    else:  # relative-dict + dCDH producers
        source = dict(native.event_study_effects)
        placebos = getattr(native, "placebo_event_study", None)
        if placebos:
            source.update(placebos)
        for e, row in source.items():
            att = row["att"] if "att" in row else row["effect"]
            ci = row.get("conf_int", (np.nan, np.nan))
            out[e] = {
                "att": att,
                "se": row.get("se", np.nan),
                "t_stat": row.get("t_stat", np.nan),
                "p_value": row.get("p_value", np.nan),
                "ci_lo": ci[0],
                "ci_hi": ci[1],
            }
    return out


# ===========================================================================
# Per-row df provenance + Stacked/TwoStage vcov persistence (M-092 follow-up)
# ===========================================================================


def test_df_scalar_broadcasts_per_row():
    surface = _tiny_surface(df=12.0)
    assert isinstance(surface.df, np.ndarray)
    assert np.isnan(surface.df[0])  # reference row: no df provenance
    assert surface.df[1] == 12.0 and surface.df[2] == 12.0


def test_df_none_yields_nan_column():
    surface = _tiny_surface()
    assert isinstance(surface.df, np.ndarray)
    assert np.isnan(surface.df).all()


def test_df_array_length_validated():
    with pytest.raises(ValueError, match="'df' has shape"):
        _tiny_surface(df=np.array([1.0, 2.0]))


def test_df_nan_where_p_value_nan():
    # Row 2: finite se but NaN p (non-estimable inference) - its df must be
    # stripped, because df is provenance of a STORED p-value.
    surface = _tiny_surface(
        p_value=np.array([np.nan, 0.0, np.nan]),
        df=np.array([5.0, 12.0, 12.0]),
    )
    assert np.isnan(surface.df[0])  # reference
    assert surface.df[1] == 12.0
    assert np.isnan(surface.df[2])  # NaN-p row never used a df


def test_df_round_trips_through_to_dict():
    import json

    d = _tiny_surface(df=7.0).to_dict()
    json.dumps(d)  # must not raise
    assert isinstance(d["df"], list) and len(d["df"]) == 3
    assert d["df"][1] == 7.0 and np.isnan(d["df"][0])


def test_df_column_in_pinned_frame():
    surface = _tiny_surface(df=9.0)
    frame = surface.to_dataframe()
    assert "df" in frame.columns  # via EVENT_STUDY_SCHEMA
    np.testing.assert_array_equal(frame["df"].to_numpy(), surface.df)


@pytest.mark.parametrize("label", ALL_PRODUCERS)
def test_df_nan_on_reference_and_nan_p_rows(label, surfaces):
    _, surface = surfaces[label]
    assert isinstance(surface.df, np.ndarray)
    assert surface.df.shape == surface.event_time.shape
    assert np.isnan(surface.df[surface.is_reference]).all()
    assert np.isnan(surface.df[~np.isfinite(surface.p_value)]).all()


def test_stacked_vcov_alignment(surfaces):
    native, surface = surfaces["StackedDiD"]
    assert surface.vcov is not None
    assert surface.vcov_index is not None
    idx = surface.vcov_index.tolist()
    assert set(idx).issubset(set(surface.event_time.tolist()))
    # The reference period is synthesized, never a regression column.
    assert -1 not in idx
    assert idx == native.event_study_vcov_index
    # The marginal ES SEs are literally this matrix's diagonal (pure copy).
    ses = {h: d["se"] for h, d in native.event_study_effects.items()}
    diag = np.sqrt(np.maximum(np.diag(surface.vcov), 0.0))
    np.testing.assert_allclose(diag, [ses[h] for h in idx], rtol=1e-14)


def test_two_stage_vcov_alignment(surfaces):
    native, surface = surfaces["TwoStageDiD"]
    assert surface.vcov is not None
    assert surface.vcov_index is not None
    idx = surface.vcov_index.tolist()
    assert set(idx).issubset(set(surface.event_time.tolist()))
    assert -1 not in idx  # ref_period never a Stage-2 column
    assert idx == native.event_study_vcov_index
    ses = {h: d["se"] for h, d in native.event_study_effects.items()}
    diag = np.sqrt(np.maximum(np.diag(surface.vcov), 0.0))
    for i, h in enumerate(idx):
        if np.isfinite(ses[h]):
            np.testing.assert_allclose(diag[i], ses[h], rtol=1e-14)
        else:
            assert np.isnan(diag[i])  # rank-guard NaN row == NaN marginal SE


def test_two_stage_prop5_horizons_excluded_from_vcov_index():
    # No never-treated units -> late horizons are Proposition-5 unidentified
    # (NaN effect, n_obs > 0). They live in the effects dict but were never
    # Stage-2 columns, so the persisted vcov_index must exclude them.
    from diff_diff.two_stage import TwoStageDiD

    rng = np.random.default_rng(9)
    rows = []
    for u in range(60):
        g = 3 if u % 2 == 0 else 5
        for t in range(1, 9):
            y = 1.0 + 0.1 * t + u * 0.01 + (1.0 if t >= g else 0.0) + rng.normal(0, 0.3)
            rows.append({"unit": u, "time": t, "outcome": y, "first_treat": g})
    data = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
    prop5 = [
        h
        for h, d in res.event_study_effects.items()
        if d["n_obs"] > 0 and not np.isfinite(d["effect"])
    ]
    assert prop5, "fixture must produce Proposition-5 horizons"
    assert res.event_study_vcov_index is not None
    assert not set(prop5) & set(res.event_study_vcov_index)
    surface = build_event_study_surface(res)
    assert surface.vcov is not None
    assert set(surface.vcov_index.tolist()).issubset(set(surface.event_time.tolist()))


def _stacked_panel(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(60):
        g = [4, 6, 0][u % 3]
        for t in range(1, 11):
            y = 1.0 + 0.1 * t + u * 0.01 + (1.5 if g and t >= g else 0.0) + rng.normal(0, 0.3)
            rows.append({"unit": u, "time": t, "outcome": y, "first_treat": g})
    return pd.DataFrame(rows)


def test_stacked_hc2_bm_per_row_df():
    from diff_diff import StackedDiD

    data = _stacked_panel()
    kwargs = dict(
        outcome="outcome",
        unit="unit",
        time="time",
        first_treat="first_treat",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm").fit(data, **kwargs)
        hc1 = StackedDiD(kappa_pre=2, kappa_post=2).fit(data, **kwargs)

    # hc2_bm: every finite-p non-reference row carries its own BM df,
    # bit-equal to the producer's event_study_df provenance dict.
    s_bm = build_event_study_surface(bm)
    finite_p = np.isfinite(s_bm.p_value)
    assert finite_p.any()
    assert np.isfinite(s_bm.df[finite_p]).all()
    for et, df_val in zip(s_bm.event_time.tolist(), s_bm.df.tolist()):
        if et in bm.event_study_df and np.isfinite(bm.event_study_df[et]):
            assert df_val == bm.event_study_df[et]

    # hc1 non-survey (3.9 / M-127): the df_convention default resolves to
    # t(pooled residual df), so every finite-p row now carries that df.
    s_hc1 = build_event_study_surface(hc1)
    finite_hc1 = np.isfinite(s_hc1.p_value)
    assert np.isfinite(s_hc1.df[finite_hc1]).all()
    vals = {v for v in s_hc1.df[finite_hc1].tolist()}
    assert len(vals) == 1 and vals == {hc1.inference_df}


def test_stacked_survey_df_broadcast():
    from diff_diff import StackedDiD
    from diff_diff.survey import SurveyDesign

    data = _stacked_panel()
    data["w"] = 1.0
    data["strat"] = data["unit"] % 2
    data["psu_id"] = data["unit"]
    design = SurveyDesign(weights="w", weight_type="pweight", strata="strat", psu="psu_id")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = StackedDiD(kappa_pre=2, kappa_post=2).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
    assert res.survey_metadata is not None and res.survey_metadata.df_survey is not None
    surface = build_event_study_surface(res)
    finite_p = np.isfinite(surface.p_value)
    assert finite_p.any()
    vals = set(surface.df[finite_p].tolist())
    assert vals == {float(max(res.survey_metadata.df_survey, 1))}


def test_two_stage_survey_df_threaded():
    from diff_diff.survey import SurveyDesign
    from diff_diff.two_stage import TwoStageDiD

    data = _stacked_panel()
    data["w"] = 1.0
    data["strat"] = data["unit"] % 2
    data["psu_id"] = data["unit"]
    design = SurveyDesign(weights="w", weight_type="pweight", strata="strat", psu="psu_id")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            survey_design=design,
        )
    assert res.event_study_df is not None and res.event_study_df > 0
    surface = build_event_study_surface(res)
    finite_p = np.isfinite(surface.p_value)
    assert finite_p.any()
    assert set(surface.df[finite_p].tolist()) == {float(res.event_study_df)}


def test_stacked_replicate_weight_vcov_diag_matches_ses():
    # Replicate-weight designs REASSIGN the coefficient covariance before
    # the ES extraction loop (replicate refit), so StackedDiD persists the
    # replicate VCV sub-block and the reported ES SEs remain exactly its
    # diagonal - the every-inference-mode guarantee.
    from diff_diff import StackedDiD
    from diff_diff.survey import SurveyDesign

    data = _stacked_panel()
    data["w"] = 1.0
    units = np.sort(data["unit"].unique())
    n_rep = 8
    unit_pos = {u: i for i, u in enumerate(units)}
    rows = data["unit"].map(unit_pos).to_numpy()
    per = max(len(units) // n_rep, 1)
    rep_cols = []
    for r in range(n_rep):
        w_r = np.ones(len(units))
        w_r[r * per : min((r + 1) * per, len(units))] = 0.0
        nz = w_r > 0
        w_r[nz] = w_r[nz] * n_rep / (n_rep - 1)
        data[f"rep_{r}"] = w_r[rows]
        rep_cols.append(f"rep_{r}")
    design = SurveyDesign(
        weights="w", weight_type="pweight", replicate_weights=rep_cols, replicate_method="JK1"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = StackedDiD(kappa_pre=2, kappa_post=2).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
    assert res.event_study_vcov is not None
    ses = {h: d["se"] for h, d in res.event_study_effects.items()}
    diag = np.sqrt(np.maximum(np.diag(res.event_study_vcov), 0.0))
    for i, h in enumerate(res.event_study_vcov_index):
        np.testing.assert_allclose(diag[i], ses[h], rtol=1e-14)
    surface = build_event_study_surface(res)
    assert surface.vcov is not None


def _cs_cluster_panel(seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(80):
        g = [4, 6, 0][u % 3]
        for t in range(1, 9):
            y = 1 + 0.1 * t + u * 0.01 + (1.0 if g and t >= g else 0.0) + rng.normal(0, 0.3)
            rows.append({"unit": u, "time": t, "outcome": y, "first_treat": g, "st": u % 8})
    return pd.DataFrame(rows)


def test_cs_bare_cluster_df_fallback():
    from diff_diff import CallawaySantAnna

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CallawaySantAnna(cluster="st").fit(
            _cs_cluster_panel(),
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
    # Bare-cluster synthesize path: the ES rows' safe_inference used G-1,
    # recorded on BOTH channels (primary event_study_df; df_inference keeps
    # its narrow HonestDiD contract).
    assert res.event_study_df == float(res.df_inference) == 7.0
    surface = build_event_study_surface(res)
    finite_p = np.isfinite(surface.p_value)
    assert set(surface.df[finite_p].tolist()) == {7.0}


def test_cs_bare_cluster_bootstrap_df_is_nan():
    from diff_diff import CallawaySantAnna

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CallawaySantAnna(cluster="st", n_bootstrap=20, seed=1).fit(
            _cs_cluster_panel(),
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
    # Bootstrap overrode the stored ES p/CIs with percentile values that
    # never used the analytical df: the surface must show NO df even though
    # df_inference (the HonestDiD contract field) still carries G-1 on the
    # container. Bootstrap p-values are finite, so only the gated
    # df_inference fallback - not the NaN-p mask - prevents the leak.
    assert res.df_inference == 7
    assert res.event_study_df is None
    surface = build_event_study_surface(res)
    assert np.isfinite(surface.p_value).any()
    assert np.isnan(surface.df).all()


def test_cs_survey_min_df_on_surface():
    from diff_diff import CallawaySantAnna
    from diff_diff.survey import SurveyDesign

    data = _cs_cluster_panel()
    data["w"] = 1.0
    design = SurveyDesign(weights="w", weight_type="pweight", strata="st", psu="unit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CallawaySantAnna().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            survey_design=design,
        )
    # Explicit-survey fits: df_inference stays None (PR #487 narrow
    # HonestDiD contract - channel separation), while the surface carries
    # the ONE df every ES row's safe_inference_batch actually applied.
    assert res.df_inference is None
    assert res.event_study_df is not None and res.event_study_df > 0
    surface = build_event_study_surface(res)
    finite_p = np.isfinite(surface.p_value)
    assert finite_p.any()
    assert set(surface.df[finite_p].tolist()) == {float(res.event_study_df)}


def test_mpd_hc2_bm_per_period_df():
    from diff_diff import MultiPeriodDiD

    rng = np.random.default_rng(5)
    rows = []
    for u in range(50):
        tr = u % 2
        # Unbalanced panel: later periods observed for a shrinking subset,
        # so the per-period BM Satterthwaite dfs genuinely differ.
        t_max = 8 - (u % 4)
        for t in range(1, t_max + 1):
            y = 1 + 0.1 * t + 0.3 * tr + (0.8 if tr and t >= 5 else 0.0) + rng.normal(0, 0.3)
            rows.append({"unit": u, "time": t, "outcome": y, "treated": tr})
    data = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MultiPeriodDiD(vcov_type="hc2_bm", cluster="unit").fit(
            data, outcome="outcome", treatment="treated", time="time", unit="unit"
        )
    finite_vals = [v for v in res.event_study_df.values() if np.isfinite(v)]
    assert finite_vals
    # Per-period dfs vary across rows on an unbalanced panel...
    assert len({round(v, 6) for v in finite_vals}) > 1
    # ...and are NOT the post-average contrast df broadcast.
    assert any(v != res.inference_df for v in finite_vals)
    surface = build_event_study_surface(res)
    for et, df_val in zip(surface.event_time.tolist(), surface.df.tolist()):
        if et in res.event_study_df and np.isfinite(res.event_study_df[et]):
            assert df_val == res.event_study_df[et]


def test_lpdid_per_horizon_df(surfaces):
    native, surface = surfaces["LPDiD"]
    frame = native.event_study
    ncl = dict(zip(frame["horizon"].tolist(), frame["n_clusters"].tolist()))
    for et, df_val in zip(surface.event_time.tolist(), surface.df.tolist()):
        if et == -1:
            assert np.isnan(df_val)  # synthetic base row: no provenance
        elif np.isfinite(df_val):
            # Non-survey cluster rule: realized per-horizon G - 1.
            assert df_val == ncl[et] - 1
    # The native frame schema is UNCHANGED (df lives on event_study_df only).
    assert "df" not in frame.columns


# ===========================================================================
# df provenance: remaining producers (SunAbraham, dCDH) + LPDiD pooled window
# ===========================================================================


def _sa_df_panel(seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(60):
        g = [4, 6, 0][u % 3]
        for t in range(1, 9):
            y = 1.0 + 0.1 * t + u * 0.01 + (1.0 if g and t >= g else 0.0) + rng.normal(0, 0.3)
            rows.append({"unit": u, "time": t, "outcome": y, "first_treat": g})
    return pd.DataFrame(rows)


def _sa_kwargs():
    return dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")


def test_sun_abraham_hc2_bm_per_row_df():
    data = _sa_df_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm = SunAbraham(vcov_type="hc2_bm").fit(data, **_sa_kwargs())
        plain = SunAbraham().fit(data, **_sa_kwargs())

    # hc2_bm: each finite-p row carries its own BM contrast df, bit-equal to
    # the producer's provenance dict (pure copies, no recomputation).
    s_bm = build_event_study_surface(bm)
    finite_p = np.isfinite(s_bm.p_value)
    assert finite_p.any()
    assert np.isfinite(s_bm.df[finite_p]).all()
    for et, df_val in zip(s_bm.event_time.tolist(), s_bm.df.tolist()):
        if et in bm.event_study_df and np.isfinite(bm.event_study_df[et]):
            assert df_val == bm.event_study_df[et]

    # Plain analytic fit (3.9 / M-127, the D4 fix): aggregates share the
    # saturated fit's residual df, so provenance is FINITE on every row.
    assert all(np.isfinite(v) and v > 0 for v in plain.event_study_df.values())
    s_plain = build_event_study_surface(plain)
    finite_plain = np.isfinite(s_plain.p_value)
    assert np.isfinite(s_plain.df[finite_plain]).all()


def test_sun_abraham_survey_df_broadcast():
    from diff_diff.survey import SurveyDesign

    data = _sa_df_panel()
    data["w"] = 1.0
    data["strat"] = data["unit"] % 4
    data["psu_id"] = data["unit"]
    design = SurveyDesign(weights="w", weight_type="pweight", strata="strat", psu="psu_id")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SunAbraham().fit(data, survey_design=design, **_sa_kwargs())
    assert res.survey_metadata is not None and res.survey_metadata.df_survey is not None
    surface = build_event_study_surface(res)
    finite_p = np.isfinite(surface.p_value)
    assert finite_p.any()
    assert set(surface.df[finite_p].tolist()) == {float(max(res.survey_metadata.df_survey, 1))}


def test_sun_abraham_replicate_df_kept_while_vcov_cleared():
    # The clear predicates deliberately DIFFER: the analytical vcov is
    # invalidated by the replicate refit, but the recomputed rows genuinely
    # used the (post-drop) replicate df, so that provenance is kept.
    from diff_diff.survey import SurveyDesign

    data = _sa_df_panel()
    data["w"] = 1.0
    units = np.sort(data["unit"].unique())
    n_rep = 8
    unit_pos = {u: i for i, u in enumerate(units)}
    rows_idx = data["unit"].map(unit_pos).to_numpy()
    per = max(len(units) // n_rep, 1)
    rep_cols = []
    for r in range(n_rep):
        w_r = np.ones(len(units))
        w_r[r * per : min((r + 1) * per, len(units))] = 0.0
        nz = w_r > 0
        w_r[nz] = w_r[nz] * n_rep / (n_rep - 1)
        data[f"rep_{r}"] = w_r[rows_idx]
        rep_cols.append(f"rep_{r}")
    design = SurveyDesign(
        weights="w",
        weight_type="pweight",
        replicate_weights=rep_cols,
        replicate_method="JK1",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SunAbraham().fit(data, survey_design=design, **_sa_kwargs())
    assert res.event_study_vcov is None  # existing clear
    assert res.event_study_df is not None  # provenance KEPT
    surface = build_event_study_surface(res)
    finite_p = np.isfinite(surface.p_value)
    # Unconditional: a regression that NaN-ed every replicate row's
    # inference must fail here rather than pass vacuously.
    assert finite_p.any()
    assert np.isfinite(surface.df[finite_p]).all()
    # The surfaced values are the producer's record, which is the (post-drop)
    # design df the replicate refit's safe_inference calls received.
    expected = float(max(res.survey_metadata.df_survey, 1))
    assert set(surface.df[finite_p].tolist()) == {expected}
    for et, df_val in zip(surface.event_time.tolist(), surface.df.tolist()):
        if et in res.event_study_df and np.isfinite(res.event_study_df[et]):
            assert df_val == res.event_study_df[et]


def test_sun_abraham_bootstrap_clears_df():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SunAbraham(n_bootstrap=20, seed=1).fit(_sa_df_panel(), **_sa_kwargs())
    assert res.bootstrap_results is not None
    assert res.event_study_df is None
    surface = build_event_study_surface(res)
    # Bootstrap p-values are FINITE, so only the producer-side clear (not the
    # container's NaN-p masking) can prevent a false df from surfacing.
    assert np.isfinite(surface.p_value).any()
    assert np.isnan(surface.df).all()


def _dcdh_df_panel(seed=8):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(60):
        g = [4, 6, 0][u % 3]
        for t in range(1, 9):
            y = 1.0 + 0.1 * t + u * 0.01 + (1.0 if g and t >= g else 0.0) + rng.normal(0, 0.3)
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "outcome": y,
                    "treat": 1 if (g and t >= g) else 0,
                    "w": 1.0,
                    "strat": u % 4,
                    "psu_id": u,
                }
            )
    return pd.DataFrame(rows)


def _dcdh_kwargs():
    return dict(outcome="outcome", group="unit", time="time", treatment="treat", L_max=2)


def test_dcdh_analytic_df_all_nan():
    from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ChaisemartinDHaultfoeuille().fit(_dcdh_df_panel(), **_dcdh_kwargs())
    assert res.event_study_df is None  # z-inference, no df
    assert np.isnan(build_event_study_surface(res).df).all()


def test_dcdh_survey_df_on_surface_covers_placebos():
    from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille
    from diff_diff.survey import SurveyDesign

    design = SurveyDesign(weights="w", weight_type="pweight", strata="strat", psu="psu_id")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ChaisemartinDHaultfoeuille().fit(
            _dcdh_df_panel(), survey_design=design, **_dcdh_kwargs()
        )
    assert res.event_study_df == float(res.survey_metadata.df_survey)
    surface = build_event_study_surface(res)
    # Placebo rows (negative keys) are merged into the same surface and share
    # the ONE design df - that is why a scalar is faithful here.
    assert (surface.event_time < 0).any()
    finite_p = np.isfinite(surface.p_value)
    assert finite_p.any()
    assert set(surface.df[finite_p].tolist()) == {float(res.event_study_df)}
    assert np.isnan(surface.df[surface.is_reference]).all()


def test_dcdh_survey_bootstrap_clears_df():
    # TSL survey + bootstrap is a LIVE mode (only replicate + bootstrap is
    # rejected): the df expression evaluates finite while the stored p/CIs
    # are bootstrap percentiles, so the clear is load-bearing here.
    from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille
    from diff_diff.survey import SurveyDesign

    design = SurveyDesign(weights="w", weight_type="pweight", strata="strat", psu="psu_id")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ChaisemartinDHaultfoeuille(n_bootstrap=20, seed=1).fit(
            _dcdh_df_panel(), survey_design=design, **_dcdh_kwargs()
        )
    assert res.survey_metadata is not None and res.survey_metadata.df_survey is not None
    assert res.event_study_df is None
    assert np.isnan(build_event_study_surface(res).df).all()


def test_lpdid_pooled_df_threaded():
    from diff_diff import LPDiD

    rng = np.random.default_rng(11)
    rows = []
    for u in range(40):
        g = [5, 7, 0][u % 3]
        for t in range(1, 12):
            y = 1.0 + 0.05 * t + u * 0.02 + (1.0 if g and t >= g else 0.0) + rng.normal(0, 0.3)
            rows.append({"unit": u, "time": t, "outcome": y, "treat": 1 if (g and t >= g) else 0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = LPDiD(pre_window=3, post_window=3).fit(
            pd.DataFrame(rows), outcome="outcome", unit="unit", time="time", treatment="treat"
        )
    assert res.pooled_df is not None and set(res.pooled_df) == {"pre", "post"}
    ncl = dict(zip(res.pooled["window"].tolist(), res.pooled["n_clusters"].tolist()))
    for window, df_val in res.pooled_df.items():
        if np.isfinite(df_val):
            assert df_val == ncl[window] - 1
    assert res.to_dict()["pooled_df"] == res.pooled_df
    # Native pooled frame schema is UNCHANGED.
    assert "df" not in res.pooled.columns
