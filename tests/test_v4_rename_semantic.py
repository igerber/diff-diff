"""Behavioral pins for the 2(c)-ii semantic rename wave (3.9 shims).

The dedicated shim test file required by the ledger rows this PR flips
(``test_ref`` on M-030/M-031, M-043/M-095, M-044/M-086/M-087,
M-045..M-047/M-115, M-084, and the missed-rename amendments M-136..M-138 in
``docs/v4-deprecations.yaml``). Per the section 2 per-PR gate, every shimmed
surface pins: the old spelling's ``FutureWarning`` with the migration
message, canonical/positional silence, the both-supplied ``ValueError``,
bit-exact routing parity, the renamed-field trio (dataclass shape + warning
property + pickle migration + dual serialized keys), and the raw-keep rows'
probe re-warn counts.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff._deprecation import _NotSupplied


def _assert_no_future_warning(record):
    fw = [w for w in record if issubclass(w.category, FutureWarning)]
    assert fw == [], [str(w.message) for w in fw]


@pytest.fixture(scope="module")
def did_2x2():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "d": np.repeat([0, 1], n // 2),
            "p": np.tile(np.repeat([0, 1], n // 4), 2),
        }
    )


@pytest.fixture(scope="module")
def ddd_2x2x2():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": np.repeat([0, 1], n // 2),
            "q": np.tile(np.repeat([0, 1], n // 4), 2),
            "t": np.tile([0, 1], n // 2),
        }
    )
    df["y"] += 1.5 * df.g * df.q * df.t
    return df


@pytest.fixture(scope="module")
def placebo_panel():
    rng = np.random.default_rng(2)
    rows = []
    for u in range(30):
        d = 1 if u < 15 else 0
        for t2 in (0, 1):
            rows.append((u, d, t2, 0.5 + 0.3 * t2 + 0.8 * d * t2 + rng.normal(0, 0.3)))
    return pd.DataFrame(rows, columns=["u", "d", "p", "y"])


@pytest.fixture(scope="module")
def stacked_panel():
    rng = np.random.default_rng(5)
    rows = []
    for u in range(40):
        g = 3 if u < 12 else (4 if u < 24 else 0)
        for t in range(1, 7):
            y = 0.2 * t + (1.0 if g and t >= g else 0) + rng.normal(0, 0.3)
            rows.append((u, t, g, y))
    return pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y"])


@pytest.fixture(scope="module")
def etwfe_result():
    from diff_diff import WooldridgeDiD

    rng = np.random.default_rng(3)
    rows = []
    for u in range(40):
        g = 3 if u < 20 else 0
        for t in range(1, 6):
            rows.append(
                (u, t, g, 1.0 + 0.5 * t + (1.5 if g and t >= g else 0) + rng.normal(0, 0.3))
            )
    df = pd.DataFrame(rows, columns=["unit", "time", "g", "y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WooldridgeDiD().fit(df, "y", "unit", "time", "g")


# ---------------------------------------------------------------------------
# M-030 / M-031: DiD.fit and TripleDifference.fit time= -> post=
# ---------------------------------------------------------------------------


class TestTimeToPost:
    def test_did_canonical_positional_formula_silent(self, did_2x2):
        from diff_diff import DifferenceInDifferences as DiD

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_kw = DiD().fit(did_2x2, outcome="y", treatment="d", post="p")
            r_pos = DiD().fit(did_2x2, "y", "d", "p")
            r_form = DiD().fit(did_2x2, formula="y ~ d * p")
        _assert_no_future_warning(record)
        assert r_kw.att == r_pos.att == r_form.att

    def test_did_time_warns_and_routes_identically(self, did_2x2):
        from diff_diff import DifferenceInDifferences as DiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_new = DiD().fit(did_2x2, outcome="y", treatment="d", post="p")
        with pytest.warns(
            FutureWarning,
            match=r"DifferenceInDifferences\.fit\(time=\) is deprecated and "
            r"will be removed in 4\.0; use post= instead\.",
        ):
            r_old = DiD().fit(did_2x2, outcome="y", treatment="d", time="p")
        assert r_old.att == r_new.att
        assert r_old.se == r_new.se

    def test_did_both_supplied_raises(self, did_2x2):
        from diff_diff import DifferenceInDifferences as DiD

        with pytest.raises(ValueError, match=r"pass only post="):
            DiD().fit(did_2x2, outcome="y", treatment="d", post="p", time="p")

    def test_ddd_time_warns_with_calendar_note(self, ddd_2x2x2):
        from diff_diff import TripleDifference

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_new = TripleDifference().fit(ddd_2x2x2, "y", "g", "q", post="t")
            r_pos = TripleDifference().fit(ddd_2x2x2, "y", "g", "q", "t")
        with pytest.warns(FutureWarning, match=r"calendar column only"):
            r_old = TripleDifference().fit(ddd_2x2x2, "y", "g", "q", time="t")
        assert r_old.att == r_new.att == r_pos.att

    def test_ddd_missing_post_raises(self, ddd_2x2x2):
        from diff_diff import TripleDifference

        with pytest.raises(TypeError, match=r"missing required argument: 'post'"):
            TripleDifference().fit(ddd_2x2x2, "y", "g", "q")

    def test_wrapper_forwards_time_without_rename_warning(self, ddd_2x2x2):
        # Flipped BY DESIGN in the 2(d) PR-A (M-075): the wrapper now
        # fires its OWN deprecation FutureWarning, but the contract this
        # test exists for is unchanged and only the wrapper path can
        # express it - the wrapper's legacy positional `time` argument
        # maps onto the renamed `post=` fit kwarg WITHOUT tripping
        # M-031's inner rename warning. So: exactly ONE FutureWarning,
        # the wrapper's, never the rename shim's.
        from diff_diff.triple_diff import triple_difference

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r = triple_difference(ddd_2x2x2, "y", "g", "q", "t")
        fw = [w for w in record if issubclass(w.category, FutureWarning)]
        assert len(fw) == 1, [str(w.message) for w in fw]
        assert "triple_difference() is deprecated" in str(fw[0].message)
        assert "time=" not in str(fw[0].message)
        assert np.isfinite(r.att)


# ---------------------------------------------------------------------------
# M-137 / M-138: permutation_test / leave_one_out_test time= -> post=
# ---------------------------------------------------------------------------


class TestDiagnosticsTimeToPost:
    def test_permutation_rename(self, placebo_panel):
        from diff_diff import permutation_test

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_new = permutation_test(
                placebo_panel, "y", "d", post="p", unit="u", n_permutations=50, seed=1
            )
            r_pos = permutation_test(placebo_panel, "y", "d", "p", "u", n_permutations=50, seed=1)
        _assert_no_future_warning(record)
        with pytest.warns(FutureWarning, match=r"permutation_test\(time=\) is deprecated"):
            r_old = permutation_test(
                placebo_panel, "y", "d", time="p", unit="u", n_permutations=50, seed=1
            )
        assert r_new.p_value == r_old.p_value == r_pos.p_value
        with pytest.raises(ValueError, match=r"pass only post="):
            permutation_test(placebo_panel, "y", "d", post="p", unit="u", time="p")
        with pytest.raises(TypeError, match=r"missing required argument: 'unit'"):
            permutation_test(placebo_panel, "y", "d", post="p")

    def test_leave_one_out_rename(self, placebo_panel):
        from diff_diff import leave_one_out_test

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            l_new = leave_one_out_test(placebo_panel, "y", "d", post="p", unit="u")
        _assert_no_future_warning(record)
        with pytest.warns(FutureWarning, match=r"leave_one_out_test\(time=\) is deprecated"):
            l_old = leave_one_out_test(placebo_panel, "y", "d", time="p", unit="u")
        assert l_new.original_effect == l_old.original_effect

    def test_wrapper_paths_silent_and_no_error_dicts(self, placebo_panel):
        """The run_all_placebo_tests calls sit inside ``except Exception``
        blocks that would SWALLOW a FutureWarning-raised-as-error into
        ``{"error": ...}`` dicts - so the zero-warning contract is pinned
        here explicitly (dual-review R2 consensus P1)."""
        from diff_diff.diagnostics import run_all_placebo_tests, run_placebo_test

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            run_placebo_test(
                placebo_panel,
                "y",
                "d",
                "p",
                unit="u",
                test_type="permutation",
                n_permutations=30,
                seed=2,
            )
            allr = run_all_placebo_tests(
                placebo_panel,
                "y",
                "d",
                "p",
                unit="u",
                pre_periods=[0],
                post_periods=[1],
                n_permutations=30,
                seed=2,
            )
        _assert_no_future_warning(record)
        for key in ("permutation", "leave_one_out"):
            v = allr.get(key)
            assert v is not None
            assert not (isinstance(v, dict) and "error" in v), (key, v)


# ---------------------------------------------------------------------------
# M-045..M-047 / M-115: the robust drop
# ---------------------------------------------------------------------------


class TestRobustDrop:
    @pytest.mark.parametrize(
        "cls_name,legacy_default",
        [
            ("DifferenceInDifferences", True),
            ("TwoWayFixedEffects", True),
            ("TripleDifference", True),
            ("HeterogeneousAdoptionDiD", False),
            ("LinearRegression", True),
        ],
    )
    def test_default_silent_and_resolved_attr(self, cls_name, legacy_default):
        import diff_diff

        cls = getattr(diff_diff, cls_name)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            est = cls()
        _assert_no_future_warning(record)
        # The PUBLIC attr keeps the RESOLVED legacy bool through 3.9 so
        # pre-rename attribute readers keep working.
        assert est.robust is legacy_default
        assert est._robust_arg is None

    @pytest.mark.parametrize(
        "cls_name", ["DifferenceInDifferences", "TripleDifference", "LinearRegression"]
    )
    def test_explicit_robust_warns(self, cls_name):
        import diff_diff

        cls = getattr(diff_diff, cls_name)
        with pytest.warns(
            FutureWarning,
            match=rf"{cls_name}\(robust=\) is deprecated and will be removed "
            r"in 4\.0; use vcov_type= instead\.",
        ):
            est = cls(robust=False)
        assert est.robust is False

    def test_did_routing_parity(self, did_2x2):
        from diff_diff import DifferenceInDifferences as DiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_old = DiD(robust=False).fit(did_2x2, "y", "d", "p")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_new = DiD(vcov_type="classical").fit(did_2x2, "y", "d", "p")
        _assert_no_future_warning(record)
        assert r_old.att == r_new.att
        assert r_old.se == r_new.se

    def test_get_params_round_trip_silent_on_default(self):
        from diff_diff import DifferenceInDifferences as DiD

        est = DiD()
        params = est.get_params()
        assert params["robust"] is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clone = DiD(**params)
        assert clone.vcov_type == est.vcov_type

    def test_probe_rewarns_on_deprecated_configured_instance(self):
        """Accepted + documented: a robust=-configured instance re-warns on
        set_params probe re-init (the config is still deprecated)."""
        from diff_diff import DifferenceInDifferences as DiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = DiD(robust=False)
        with pytest.warns(FutureWarning, match=r"\(robust=\) is deprecated"):
            est.set_params(alpha=0.10)
        assert est.vcov_type == "classical"

    def test_set_params_robust_alone_rederives_vcov(self):
        from diff_diff import DifferenceInDifferences as DiD

        est = DiD(vcov_type="hc1")
        with pytest.warns(FutureWarning, match=r"\(robust=\) is deprecated"):
            est.set_params(robust=False)
        assert est.vcov_type == "classical"

    def test_conflict_still_raises(self):
        from diff_diff import DifferenceInDifferences as DiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="conflicts with vcov_type"):
                DiD(robust=False, vcov_type="hc1")


# ---------------------------------------------------------------------------
# M-043 / M-095: StackedDiD clean_control= -> control_group=
# ---------------------------------------------------------------------------


class TestCleanControlRename:
    def test_canonical_silent_and_default(self):
        from diff_diff import StackedDiD

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            s = StackedDiD()
            s2 = StackedDiD(control_group="strict")
        _assert_no_future_warning(record)
        assert s.control_group == "not_yet_treated"
        assert s2.control_group == "strict"

    def test_old_name_warns_and_maps(self):
        from diff_diff import StackedDiD

        with pytest.warns(
            FutureWarning,
            match=r"StackedDiD\(clean_control=\) is deprecated and will be "
            r"removed in 4\.0; use control_group= instead\.",
        ):
            s = StackedDiD(clean_control="never_treated")
        assert s.control_group == "never_treated"

    def test_both_supplied_raises(self):
        from diff_diff import StackedDiD

        with pytest.raises(ValueError, match=r"pass only control_group="):
            StackedDiD(control_group="strict", clean_control="strict")

    def test_estimator_property_alias(self):
        from diff_diff import StackedDiD

        s = StackedDiD(control_group="strict")
        with pytest.warns(FutureWarning, match=r"StackedDiD\.clean_control is deprecated"):
            assert s.clean_control == "strict"
        with pytest.raises(AttributeError):
            s.clean_control = "never_treated"

    def test_get_params_round_trip_silent(self):
        from diff_diff import StackedDiD

        est = StackedDiD(control_group="strict")
        params = est.get_params()
        assert isinstance(params["clean_control"], _NotSupplied)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clone = StackedDiD(**params)
        assert clone.control_group == "strict"

    def test_set_params_migrates_both_directions(self):
        from diff_diff import StackedDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = StackedDiD(clean_control="strict")
            s.set_params(control_group="never_treated")
            assert s.control_group == "never_treated"
        s2 = StackedDiD(control_group="never_treated")
        with pytest.warns(FutureWarning, match=r"clean_control"):
            s2.set_params(clean_control="strict")
        assert s2.control_group == "strict"

    def test_routing_parity(self, stacked_panel):
        from diff_diff import StackedDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_old = StackedDiD(clean_control="not_yet_treated").fit(
                stacked_panel,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_new = StackedDiD(control_group="not_yet_treated").fit(
                stacked_panel,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        _assert_no_future_warning(record)
        assert r_old.overall_att == r_new.overall_att

    def test_results_field_trio_and_dual_keys(self, stacked_panel):
        from diff_diff import StackedDiD
        from diff_diff.stacked_did_results import StackedDiDResults

        assert "control_group" in StackedDiDResults.__dataclass_fields__
        assert "clean_control" not in StackedDiDResults.__dataclass_fields__
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = StackedDiD().fit(
                stacked_panel,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        assert res.control_group == "not_yet_treated"
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            d = res.to_dict()
            s = res.summary()
        assert d["control_group"] == "not_yet_treated" == d["clean_control"]
        assert "Control group:" in s
        with pytest.warns(FutureWarning, match=r"StackedDiDResults\.clean_control is deprecated"):
            assert res.clean_control == "not_yet_treated"

    def test_pickle_migration(self, stacked_panel):
        from diff_diff import StackedDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = StackedDiD().fit(
                stacked_panel,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        clone = pickle.loads(pickle.dumps(res))
        assert clone.control_group == res.control_group
        state = dict(res.__dict__)
        state["clean_control"] = state.pop("control_group")
        old_style = object.__new__(type(res))
        old_style.__setstate__(state)
        assert old_style.control_group == "not_yet_treated"

    def test_power_survey_gate_estimator_scoped(self):
        """Post-M-095 StackedDiD exposes control_group; the survey_config
        gate must keep accepting the default (bit-exact with the
        pre-rename gate, which rejected only strict) - dual-review R2 P1
        regression."""
        from diff_diff import StackedDiD
        from diff_diff.power import SurveyPowerConfig, simulate_power

        cfg = SurveyPowerConfig(weight_cv=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # default (not_yet_treated) passes the gate
            simulate_power(
                StackedDiD(),
                n_units=30,
                n_periods=4,
                survey_config=cfg,
                n_simulations=1,
                seed=1,
                progress=False,
            )
            with pytest.raises(ValueError, match="control_group='strict'"):
                simulate_power(
                    StackedDiD(control_group="strict"),
                    n_units=30,
                    n_periods=4,
                    survey_config=cfg,
                    n_simulations=1,
                    seed=1,
                    progress=False,
                )


# ---------------------------------------------------------------------------
# M-044 / M-086 / M-087 (+ M-136): the Wooldridge triple and LPDiD level
# ---------------------------------------------------------------------------


class TestWooldridgeTriple:
    def test_aggregate_event_study_canonical(self, etwfe_result):
        r = etwfe_result
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r.aggregate("event_study")
        _assert_no_future_warning(record)
        assert r.aggregation_weights["event_study"] == "cell"
        assert r.aggregation_weights["event"] == "cell"  # dual key (3.9)

    def test_aggregate_event_warns_and_maps(self, etwfe_result):
        r = etwfe_result
        with pytest.warns(FutureWarning, match=r"use type='event_study' instead"):
            r.aggregate("event")
        assert r.event_study_effects

    def test_to_dataframe_level_rename(self, etwfe_result):
        r = etwfe_result
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.aggregate("event_study")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            f_default = r.to_dataframe()
            f_level = r.to_dataframe(level="event_study")
            f_gt = r.to_dataframe("gt")
        _assert_no_future_warning(record)
        assert not f_gt.empty
        with pytest.warns(FutureWarning, match=r"use level= instead"):
            f_old = r.to_dataframe(aggregation="event_study")
        pd.testing.assert_frame_equal(f_default, f_level)
        pd.testing.assert_frame_equal(f_default, f_old)
        with pytest.warns(FutureWarning, match=r"use level='event_study' instead"):
            f_oldval = r.to_dataframe(level="event")
        pd.testing.assert_frame_equal(f_default, f_oldval)
        with pytest.raises(ValueError, match=r"pass only level="):
            r.to_dataframe(level="gt", aggregation="gt")

    def test_summary_alpha_keyword_only(self, etwfe_result):
        r = etwfe_result
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            s = r.summary()
            s_same = r.summary(alpha=r.alpha)
        _assert_no_future_warning(record)
        assert "95% CI" in s and "95% CI" in s_same
        # The shared section-5 contract: a DIFFERENT alpha raises rather
        # than silently relabeling stored intervals.
        with pytest.raises(ValueError, match="never recomputes"):
            r.summary(alpha=0.10)
        with pytest.warns(FutureWarning, match=r"summary\(aggregation=\)"):
            r.summary("simple")
        with pytest.raises(TypeError, match="KEYWORD-ONLY"):
            r.summary(0.10)

    def test_aggregation_weights_pickle_mirror(self, etwfe_result):
        r = etwfe_result
        state = dict(r.__dict__)
        state["aggregation_weights"] = {"event": "cell"}
        r2 = object.__new__(type(r))
        r2.__setstate__(state)
        assert r2.aggregation_weights["event_study"] == "cell"


class TestLPDiDLevelValue:
    """M-136."""

    def test_level_value_shim(self):
        from diff_diff import LPDiD

        rng = np.random.default_rng(7)
        rows = []
        for u in range(40):
            g = 4 if u < 20 else 0
            for t in range(1, 9):
                d = 1 if (g and t >= g) else 0
                rows.append((u, t, d, 0.2 * t + 1.0 * d + rng.normal(0, 0.3)))
        df = pd.DataFrame(rows, columns=["unit", "time", "d", "y"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LPDiD(pre_window=2, post_window=2).fit(
                df, outcome="y", unit="unit", time="time", treatment="d"
            )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            f_new = res.to_dataframe()
            f_canon = res.to_dataframe(level="event_study")
        _assert_no_future_warning(record)
        with pytest.warns(FutureWarning, match=r"use level='event_study' instead"):
            f_old = res.to_dataframe(level="event")
        pd.testing.assert_frame_equal(f_new, f_canon)
        pd.testing.assert_frame_equal(f_new, f_old)
        with pytest.raises(ValueError, match="event_study"):
            res.to_dataframe(level="bogus")


# ---------------------------------------------------------------------------
# M-084: ContinuousDiD covariates move to fit()
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cdid_panel():
    rng = np.random.default_rng(6)
    rows = []
    for u in range(60):
        g = 3 if u < 30 else 0
        d = max(0.1, rng.normal(1, 0.4)) if g else 0.0
        x = rng.normal()
        for t in range(1, 6):
            rows.append(
                (
                    u,
                    t,
                    g,
                    d,
                    x,
                    0.3 * t + 0.9 * (d if (g and t >= g) else 0) + 0.2 * x + rng.normal(0, 0.3),
                )
            )
    return pd.DataFrame(rows, columns=["u", "t", "g", "dose", "x", "y"])


class TestCovariatesMove:
    def test_fit_level_canonical_silent(self, cdid_panel):
        from diff_diff import ContinuousDiD

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r = ContinuousDiD().fit(cdid_panel, "y", "u", "t", "g", "dose", covariates=["x"])
        _assert_no_future_warning(record)
        assert r.covariates == ["x"]

    def test_ctor_warns_and_routes_identically(self, cdid_panel):
        from diff_diff import ContinuousDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_new = ContinuousDiD().fit(cdid_panel, "y", "u", "t", "g", "dose", covariates=["x"])
        with pytest.warns(
            FutureWarning,
            match=r"ContinuousDiD\(covariates=\) is deprecated and will be "
            r"removed in 4\.0; pass covariates to fit\(\) instead\.",
        ):
            est = ContinuousDiD(covariates=["x"])
        r_old = est.fit(cdid_panel, "y", "u", "t", "g", "dose")
        assert r_old.att == r_new.att
        assert r_old.covariates == r_new.covariates == ["x"]

    def test_both_supplied_raises(self, cdid_panel):
        from diff_diff import ContinuousDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = ContinuousDiD(covariates=["x"])
        with pytest.raises(ValueError, match=r"fit\(\) only"):
            est.fit(cdid_panel, "y", "u", "t", "g", "dose", covariates=["x"])

    def test_lowest_dose_guard_fires_for_fit_level(self, cdid_panel):
        from diff_diff import ContinuousDiD

        with pytest.raises(NotImplementedError, match="lowest_dose"):
            ContinuousDiD(control_group="lowest_dose").fit(
                cdid_panel, "y", "u", "t", "g", "dose", covariates=["x"]
            )

    def test_probe_rewarns_on_ctor_configured_instance(self):
        from diff_diff import ContinuousDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = ContinuousDiD(covariates=["x"])
        with pytest.warns(FutureWarning, match=r"\(covariates=\) is deprecated"):
            est.set_params(alpha=0.10)
