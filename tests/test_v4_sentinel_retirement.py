"""Item 8a of the 4.0 cut - the event-study reference-row sentinel retirement (ledger row M-093, first half).

Through 3.x every native event-study dict marked its normalization (reference)
row with a zero-count SENTINEL: ``n_groups == 0`` on CallawaySantAnna,
``n_obs == 0`` on StackedDiD / ImputationDiD / TwoStageDiD / SpilloverDiD.
Since 4.0 the reference row carries an explicit ``"is_reference": True`` key
and NO count key, SpilloverDiD's event-study frames carry a boolean
``is_reference`` column (NaN ``n_obs`` on the reference rows), the unified
builder reads the key only (a zero count is a genuine count), the consumers'
raw-dict routes read the key, and 3.x pickles are migrated on load by
``_deprecation.migrate_event_study_reference_sentinel``.

The second half of M-093 (every results class's ``to_dataframe(level=
"event_study")`` emitting the unified schema) is item 8b; the row flips to
``done`` there.
"""

import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from legacy_surface import es_dict_from_container, legacy_standin

from diff_diff import (
    CallawaySantAnna,
    ImputationDiD,
    StackedDiD,
    TwoStageDiD,
    compute_honest_did,
    compute_pretrends_power,
)
from diff_diff._deprecation import migrate_event_study_reference_sentinel
from diff_diff.prep_dgp import generate_staggered_data
from diff_diff.results_base import build_event_study_surface
from diff_diff.spillover import SpilloverDiD
from tests._dgp_utils import generate_butts_staggered_dgp

THIS_TEST_REF = "tests/test_v4_sentinel_retirement.py"

_PANEL_KW = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Fixtures: one fit per producer that materializes a reference row
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def stacked_fit():
    d = generate_staggered_data(n_units=120, n_periods=12, cohort_periods=[4, 6, 8], seed=42)
    return _quiet(StackedDiD(kappa_pre=2, kappa_post=2).fit, d, **_PANEL_KW)


@pytest.fixture(scope="module")
def stacked_fit_anticipation_1():
    d = generate_staggered_data(n_units=120, n_periods=12, cohort_periods=[4, 6, 8], seed=42)
    return _quiet(StackedDiD(kappa_pre=3, kappa_post=2, anticipation=1).fit, d, **_PANEL_KW)


@pytest.fixture(scope="module")
def imputation_boot_fit():
    d = generate_staggered_data(n_units=120, n_periods=8, cohort_periods=[4, 6], seed=42)
    return _quiet(ImputationDiD(n_bootstrap=19, seed=1).fit, d, **_PANEL_KW)


@pytest.fixture(scope="module")
def twostage_boot_fit():
    d = generate_staggered_data(n_units=120, n_periods=8, cohort_periods=[4, 6], seed=42)
    return _quiet(TwoStageDiD(n_bootstrap=19, seed=1).fit, d, **_PANEL_KW)


@pytest.fixture(scope="module")
def spillover_fit():
    d = generate_butts_staggered_dgp(seed=42)
    est = SpilloverDiD(
        rings=[0.0, 50.0, 200.0],
        d_bar=200.0,
        conley_coords=("lat", "lon"),
        conley_metric="haversine",
        conley_cutoff_km=200.0,
        conley_lag_cutoff=0,
        event_study=True,
        horizon_max=2,
    )
    return _quiet(est.fit, d, outcome="y", unit="unit", time="time", first_treat="first_treat")


def _gapped_universal_panel():
    """CS universal base on a gapped period grid: one positional base per cohort."""
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
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cs_universal_gapped():
    return _quiet(
        CallawaySantAnna(n_bootstrap=0, base_period="universal").fit,
        _gapped_universal_panel(),
        **_PANEL_KW,
    )


@pytest.fixture(scope="module")
def cs_universal_regular():
    d = generate_staggered_data(n_units=120, n_periods=8, cohort_periods=[4, 6], seed=7)
    return _quiet(CallawaySantAnna(n_bootstrap=0, base_period="universal").fit, d, **_PANEL_KW)


def _ref_rows(effects):
    return sorted(k for k, r in effects.items() if r.get("is_reference"))


# --------------------------------------------------------------------------- #
# (1) Native dict shape on every producer that materializes a reference row
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fixture_name, expected_ref",
    [
        ("stacked_fit", -1),
        ("stacked_fit_anticipation_1", -2),
        ("imputation_boot_fit", -1),
        ("twostage_boot_fit", -1),
        ("spillover_fit", -1),
    ],
)
def test_native_reference_row_is_keyed_without_count(request, fixture_name, expected_ref):
    res = request.getfixturevalue(fixture_name)
    effects = res.event_study_effects
    assert effects is not None
    assert _ref_rows(effects) == [expected_ref]
    ref = effects[expected_ref]
    assert ref["is_reference"] is True
    assert ref["effect"] == 0.0
    assert "n_obs" not in ref and "n_groups" not in ref
    for k, row in effects.items():
        if k == expected_ref:
            continue
        assert "is_reference" not in row, k
        assert "n_obs" in row and isinstance(row["n_obs"], (int, np.integer)), k
    # The container marks exactly that row, with a NaN count.
    es = build_event_study_surface(res)
    marked = es.event_time[es.is_reference].tolist()
    assert marked == [expected_ref]
    assert np.isnan(es.n[es.is_reference][0])


def test_spillover_frames_carry_is_reference_with_nan_n_obs(spillover_fit):
    res = spillover_fit
    ref = res.reference_period
    for frame in (res.att_dynamic, res.spillover_effects):
        assert "is_reference" in frame.columns
        assert frame["is_reference"].dtype == bool
        k = (
            frame.index.get_level_values("k")
            if isinstance(frame.index, pd.MultiIndex)
            else frame.index
        )
        is_ref = np.asarray(k == ref)
        assert np.array_equal(frame["is_reference"].to_numpy(), is_ref)
        assert np.all(np.isnan(frame.loc[frame["is_reference"], "n_obs"].to_numpy(dtype=float)))
        non_ref = frame.loc[~frame["is_reference"], "n_obs"].to_numpy(dtype=float)
        assert np.all(np.isfinite(non_ref))
        # Rectangular empty cells (if any) keep a GENUINE zero count.
        empty = frame.loc[~frame["is_reference"] & frame["coef"].isna(), "n_obs"]
        assert np.all(empty.to_numpy(dtype=float) == 0.0)
    # One reference row per ring on the spillover frame.
    n_rings = res.spillover_effects.index.get_level_values("ring").nunique()
    assert int(res.spillover_effects["is_reference"].sum()) == n_rings
    # to_dict() records carry the column.
    rec = res.to_dict()["att_dynamic"]
    assert all("is_reference" in r for r in rec)
    assert sum(1 for r in rec if r["is_reference"]) == 1


@pytest.mark.parametrize(
    "fixture_name", ["stacked_fit", "imputation_boot_fit", "twostage_boot_fit"]
)
def test_native_to_dataframe_event_study_reference_n_obs_is_nan(request, fixture_name):
    # The 8a visible change on the native level frames (8b replaces them with
    # the unified schema): the reference row's n_obs renders NaN, not 0.
    res = request.getfixturevalue(fixture_name)
    df = res.to_dataframe(level="event_study")
    ref_mask = df["relative_period"] == -1
    assert ref_mask.sum() == 1
    assert np.isnan(float(df.loc[ref_mask, "n_obs"].iloc[0]))
    assert np.all(np.isfinite(df.loc[~ref_mask, "n_obs"].to_numpy(dtype=float)))


# --------------------------------------------------------------------------- #
# (2) The CallawaySantAnna aggregator carrier
# --------------------------------------------------------------------------- #


def _capture_carrier_dict(monkeypatch, res):
    """Run aggregate('event_study') and capture the PRODUCER dict the carrier
    hands to the builder (never reconstructed from the container)."""
    import diff_diff.staggered_results as sr

    captured = {}
    real = sr.build_event_study_surface

    def _spy(carrier):
        captured["effects"] = carrier.event_study_effects
        return real(carrier)

    monkeypatch.setattr(sr, "build_event_study_surface", _spy)
    es = _quiet(res.aggregate, "event_study")
    return captured["effects"], es


def test_cs_universal_gapped_carrier_marks_every_positional_base(monkeypatch, cs_universal_gapped):
    effects, es = _capture_carrier_dict(monkeypatch, cs_universal_gapped)
    refs = _ref_rows(effects)
    assert len(refs) >= 2  # one per cohort on the gapped grid
    for k in refs:
        assert "n_groups" not in effects[k]
        assert effects[k]["effect"] == 0.0 and np.isnan(effects[k]["se"])
    for k, row in effects.items():
        if k not in refs:
            assert "is_reference" not in row and row["n_groups"] > 0
    assert sorted(es.event_time[es.is_reference].tolist()) == refs
    assert es.reference_periods == refs


def test_cs_universal_regular_carrier_single_reference(monkeypatch, cs_universal_regular):
    effects, es = _capture_carrier_dict(monkeypatch, cs_universal_regular)
    assert _ref_rows(effects) == [-1]
    assert "n_groups" not in effects[-1]
    assert es.reference_period == -1


# --------------------------------------------------------------------------- #
# (3) The builder is count-agnostic
# --------------------------------------------------------------------------- #


class _Dict:
    alpha = 0.05

    def __init__(self, effects):
        self.event_study_effects = effects


def test_builder_ignores_zero_count_without_key():
    es = build_event_study_surface(
        _Dict(
            {
                -1: {"effect": 0.0, "se": np.nan, "n_groups": 0},
                0: {"effect": 1.0, "se": 0.1, "n_groups": 40},
            }
        )
    )
    assert not es.is_reference.any()
    assert es.n[0] == 0.0 and es.n_kind == "groups"


def test_builder_marks_keyed_row_with_nan_count():
    es = build_event_study_surface(
        _Dict(
            {
                -1: {"effect": 0.0, "se": 0.0, "is_reference": True},
                0: {"effect": 1.0, "se": 0.1, "n_obs": 40},
                1: {"effect": np.nan, "se": np.nan, "n_obs": 0},  # genuine empty horizon
            }
        )
    )
    assert es.is_reference.tolist() == [True, False, False]
    assert np.isnan(es.n[0]) and es.n[1] == 40.0 and es.n[2] == 0.0
    assert np.isnan(es.att[2]) and np.isnan(es.se[0])
    assert es.n_kind == "obs"


def test_builder_rejects_keyed_row_with_finite_nonzero_effect():
    with pytest.raises(ValueError, match="is_reference row with a finite nonzero effect"):
        build_event_study_surface(
            _Dict(
                {
                    -1: {"effect": 0.4, "se": 0.0, "is_reference": True},
                    0: {"effect": 1.0, "se": 0.1, "n_obs": 40},
                }
            )
        )


def test_builder_reference_only_dict_has_no_n_kind():
    # TwoStageDiD's balance_e no-cohort early return / ref_only with no
    # Prop-5 rows: the only row records no count, so n_kind is None.
    es = build_event_study_surface(_Dict({-1: {"effect": 0.0, "se": 0.0, "is_reference": True}}))
    assert es.is_reference.tolist() == [True]
    assert es.n_kind is None and np.isnan(es.n[0])
    legacy = es_dict_from_container(es)
    assert list(legacy) == [-1]
    row = legacy[-1]
    assert set(row) == {"effect", "se", "t_stat", "p_value", "conf_int", "is_reference"}
    assert row["is_reference"] is True and row["effect"] == 0.0
    assert np.isnan(row["se"]) and np.isnan(row["t_stat"]) and np.isnan(row["p_value"])
    assert np.isnan(row["conf_int"][0]) and np.isnan(row["conf_int"][1])


def test_spillover_explicit_reference_and_key_agree(spillover_fit):
    es = build_event_study_surface(spillover_fit)
    assert es.event_time[es.is_reference].tolist() == [spillover_fit.reference_period]


# --------------------------------------------------------------------------- #
# (4) Legacy (3.x) pickle migration
# --------------------------------------------------------------------------- #


def _legacy_shape(effects, count_key):
    """Rewrite keyed reference rows into the 3.x zero-count sentinel shape."""
    out = {}
    for k, row in effects.items():
        row = dict(row)
        if row.pop("is_reference", False):
            row[count_key] = 0
        out[k] = row
    return out


def test_migration_helper_rewrites_only_the_anchor_signature():
    legacy = {
        -2: {"effect": np.nan, "se": np.nan, "n_obs": 0},  # genuine: stays
        -1: {"effect": 0.0, "se": 0.0, "n_obs": 0},  # sentinel: rewritten
        0: {"effect": 0.0, "se": 0.2, "n_obs": 30},  # estimated zero: stays
        1: {"att": 0.0, "se": np.nan, "n_groups": 0},  # Wooldridge-style key
    }
    out = migrate_event_study_reference_sentinel(legacy)
    assert out is not legacy
    assert out[-2] == legacy[-2] and "is_reference" not in out[-2]
    assert out[-1] == {"effect": 0.0, "se": 0.0, "is_reference": True}
    assert out[0] == legacy[0]
    assert out[1] == {"att": 0.0, "se": np.nan, "is_reference": True}
    assert migrate_event_study_reference_sentinel(None) is None
    # Idempotent on the 4.0 shape.
    assert migrate_event_study_reference_sentinel(out) == out


def _reload_from_state(res, state):
    obj = res.__class__.__new__(res.__class__)
    obj.__setstate__(state)
    return obj


@pytest.mark.parametrize(
    "fixture_name, count_key",
    [
        ("stacked_fit", "n_obs"),
        ("imputation_boot_fit", "n_obs"),
        ("twostage_boot_fit", "n_obs"),
    ],
)
def test_setstate_migrates_legacy_dict_sentinel(request, fixture_name, count_key):
    res = request.getfixturevalue(fixture_name)
    state = copy.deepcopy(res.__dict__)
    state["event_study_effects"] = _legacy_shape(res.event_study_effects, count_key)
    assert state["event_study_effects"][-1][count_key] == 0
    loaded = _reload_from_state(res, state)
    assert _ref_rows(loaded.event_study_effects) == [-1]
    assert count_key not in loaded.event_study_effects[-1]
    es = _quiet(loaded.aggregate, "event_study")
    assert es.event_time[es.is_reference].tolist() == [-1]


def test_setstate_migrates_legacy_cs_universal_dict(cs_universal_gapped):
    standin = legacy_standin(cs_universal_gapped)  # 3.x fit-time fields, 4.0 keyed rows
    refs = _ref_rows(standin.event_study_effects)
    assert len(refs) >= 2
    state = copy.deepcopy(standin.__dict__)
    state["event_study_effects"] = _legacy_shape(standin.event_study_effects, "n_groups")
    loaded = _reload_from_state(standin, state)
    assert _ref_rows(loaded.event_study_effects) == refs
    assert all("n_groups" not in loaded.event_study_effects[k] for k in refs)
    es = build_event_study_surface(loaded)
    assert sorted(es.event_time[es.is_reference].tolist()) == refs


def test_setstate_migrates_legacy_spillover_frames(spillover_fit):
    res = spillover_fit
    ref = res.reference_period
    state = copy.deepcopy(res.__dict__)
    state["event_study_effects"] = _legacy_shape(res.event_study_effects, "n_obs")
    for name in ("att_dynamic", "spillover_effects"):
        frame = state[name].drop(columns=["is_reference"]).copy()
        frame["n_obs"] = frame["n_obs"].fillna(0).astype(int)
        state[name] = frame
    loaded = _reload_from_state(res, state)
    assert _ref_rows(loaded.event_study_effects) == [ref]
    assert "n_obs" not in loaded.event_study_effects[ref]
    pd.testing.assert_frame_equal(loaded.att_dynamic, res.att_dynamic)
    pd.testing.assert_frame_equal(loaded.spillover_effects, res.spillover_effects)


def test_setstate_leaves_aggregate_mode_spillover_untouched():
    d = generate_butts_staggered_dgp(seed=42)
    est = SpilloverDiD(
        rings=[0.0, 50.0, 200.0],
        d_bar=200.0,
        conley_coords=("lat", "lon"),
        conley_metric="haversine",
        conley_cutoff_km=200.0,
        conley_lag_cutoff=0,
    )
    res = _quiet(est.fit, d, outcome="y", unit="unit", time="time", first_treat="first_treat")
    assert res.reference_period is None and res.event_study_effects is None
    assert "is_reference" not in res.spillover_effects.columns
    loaded = pickle.loads(pickle.dumps(res))
    pd.testing.assert_frame_equal(loaded.spillover_effects, res.spillover_effects)
    assert "is_reference" not in loaded.spillover_effects.columns


@pytest.mark.parametrize(
    "fixture_name",
    ["stacked_fit", "imputation_boot_fit", "twostage_boot_fit", "spillover_fit"],
)
def test_fresh_pickle_round_trip_is_a_noop(request, fixture_name):
    res = request.getfixturevalue(fixture_name)
    loaded = pickle.loads(pickle.dumps(res))
    a, b = res.event_study_effects, loaded.event_study_effects
    assert set(a) == set(b)
    for k in a:
        assert set(a[k]) == set(b[k]), k
        assert a[k].get("is_reference") == b[k].get("is_reference"), k
        if "n_obs" in a[k]:
            assert a[k]["n_obs"] == b[k]["n_obs"]
    if getattr(res, "att_dynamic", None) is not None:
        pd.testing.assert_frame_equal(loaded.att_dynamic, res.att_dynamic)
        pd.testing.assert_frame_equal(loaded.spillover_effects, res.spillover_effects)


# --------------------------------------------------------------------------- #
# (5) Consumers read the key on their raw-dict routes
# --------------------------------------------------------------------------- #


def test_plot_extractor_resolves_reference_by_key(stacked_fit_anticipation_1):
    from diff_diff.visualization._event_study import _extract_plot_data

    out = _extract_plot_data(stacked_fit_anticipation_1, None, None, None, None)
    assert out[5] == -2 and out[6] is True  # reference_period, reference_inferred


def test_plot_extractor_never_treats_zero_count_as_reference():
    from diff_diff.visualization._event_study import _extract_plot_data

    class _Res:
        alpha = 0.05
        anticipation = 1
        event_study_effects = {
            -3: {"effect": np.nan, "se": np.nan, "n_obs": 0},  # genuine empty horizon
            -2: {"effect": 0.0, "se": 0.0, "is_reference": True},
            -1: {"effect": 0.2, "se": 0.1, "n_obs": 30},
            0: {"effect": 1.0, "se": 0.1, "n_obs": 30},
        }

    out = _extract_plot_data(_Res(), None, None, None, None)
    assert out[5] == -2 and out[6] is True

    class _Keyless:
        alpha = 0.05
        event_study_effects = {
            -2: {"effect": 0.1, "se": 0.1, "n_obs": 30},
            -1: {"effect": 0.0, "se": 0.0, "n_obs": 0},  # a 3.x sentinel, unmigrated
            0: {"effect": 1.0, "se": 0.1, "n_obs": 30},
        }

    out = _extract_plot_data(_Keyless(), None, None, None, None)
    assert out[5] == -1 and out[6] is True  # the backward-compatibility fallback


def test_honest_plot_anchor_from_key_and_all_undefined_guard(cs_universal_regular):
    # matplotlib is an optional dependency (the CI test legs install only the
    # core stack), so the plotting test skips without it - the
    # test_event_study_consumers.py::TestPlotting convention.
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from diff_diff.visualization._event_study import plot_honest_event_study

    standin = legacy_standin(cs_universal_regular)
    h = _quiet(compute_honest_did, standin, M=0.5)
    es = cs_universal_regular.aggregate("event_study")
    h.event_study_bounds = {
        int(t): {"ci_lb": float(a) - 0.5, "ci_ub": float(a) + 0.5}
        for t, a, r in zip(es.event_time, es.att, es.is_reference)
        if not r
    }
    fig = plot_honest_event_study(h, show=False)
    assert fig is not None
    with pytest.raises(ValueError, match="No valid data to plot"):
        plot_honest_event_study(h, periods=[-1], show=False)
    plt.close("all")


def test_honest_and_pretrends_raw_route_parity(cs_universal_regular):
    standin = legacy_standin(cs_universal_regular)
    container = cs_universal_regular.aggregate("event_study")
    h_raw = _quiet(compute_honest_did, standin, M=0.5)
    h_con = _quiet(compute_honest_did, container, M=0.5)
    np.testing.assert_allclose([h_raw.lb, h_raw.ub], [h_con.lb, h_con.ub], rtol=1e-10)
    p_raw = _quiet(compute_pretrends_power, standin, M=0.5)
    p_con = _quiet(compute_pretrends_power, container, M=0.5)
    np.testing.assert_allclose(p_raw.power, p_con.power, rtol=1e-10)


def test_diagnostic_report_pre_period_collector_skips_reference_and_empty_rows():
    from diff_diff.diagnostic_report import _collect_pre_period_coefs

    class _Res:
        anticipation = 0
        event_study_effects = {
            -4: {"effect": np.nan, "se": np.nan, "p_value": np.nan, "n_obs": 0},  # empty
            -3: {"effect": 0.1, "se": 0.05, "p_value": 0.05, "n_obs": 20},
            -2: {"effect": -0.1, "se": 0.05, "p_value": 0.05, "n_obs": 20},
            -1: {"effect": 0.0, "se": 0.0, "p_value": np.nan, "is_reference": True},
            0: {"effect": 1.0, "se": 0.1, "p_value": 0.0, "n_obs": 20},
        }

    rows, n_dropped = _collect_pre_period_coefs(_Res())
    assert [k for k, *_ in rows] == [-3, -2]
    assert n_dropped == 0


@pytest.mark.parametrize(
    "fixture_name", ["stacked_fit", "imputation_boot_fit", "twostage_boot_fit"]
)
def test_summary_marks_reference_row(request, fixture_name):
    res = request.getfixturevalue(fixture_name)
    assert "[ref: -1]" in res.summary()


@pytest.mark.parametrize("fixture_name", ["imputation_boot_fit", "twostage_boot_fit"])
def test_bootstrap_override_skips_reference_row(request, fixture_name):
    res = request.getfixturevalue(fixture_name)
    ref = res.event_study_effects[-1]
    assert ref["se"] == 0.0 and np.isnan(ref["p_value"])  # no percentile override
    for k, row in res.event_study_effects.items():
        if k != -1 and np.isfinite(row["effect"]):
            assert np.isfinite(row["se"]) and row["se"] > 0


# --------------------------------------------------------------------------- #
# (6) Ledger
# --------------------------------------------------------------------------- #


def test_ledger_row_points_here_and_stays_planned_until_8b():
    from tests.test_v4_matrix import ROWS as LEDGER_ROWS

    row = {r["id"]: r for r in LEDGER_ROWS}["M-093"]
    assert row["kind"] == "behavior"
    assert row["status"] == "planned"  # flips to done in 8b (the schema rewire)
    assert row["test_ref"] == THIS_TEST_REF
    assert row["old"] == "diff_diff:CallawaySantAnnaResults.event_study_effects"
