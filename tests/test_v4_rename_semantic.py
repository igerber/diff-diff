"""Removal pins for the 2(c)-ii semantic rename wave (3.9 shims, removed 4.0).

The dedicated ``test_ref`` file for ledger rows M-030/M-031, M-043/M-095,
M-044/M-086/M-087, M-045..M-047/M-115, M-084, and the missed-rename
amendments M-136..M-138 in ``docs/v4-deprecations.yaml``. At the 4.0 cut
(PR 2) every row but M-031 flipped to ``removed``; per surface this file now
pins a canonical/positional positive smoke plus the removal evidence:
``TypeError`` matching ``"unexpected keyword argument '<old>'"`` for renamed
params, ``AttributeError`` for removed estimator/results attributes,
``ValueError`` for removed value spellings and ``set_params`` keys, and the
canonical-only serialized keys. ``__setstate__`` pickle migrations are KEPT
(spec section 5: 3.x pickles load under 4.0) and stay pinned.

M-031 (``TripleDifference.fit(time=)``) stays shimmed: ``time`` keeps its
calendar meaning in staggered mode. Its 4.0 enforcement, M-085, landed in
PR 4a, so the 2x2x2 ``time=`` pin below is a raise pin.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest


def _assert_no_future_warning(record):
    fw = [w for w in record if issubclass(w.category, FutureWarning)]
    assert fw == [], [str(w.message) for w in fw]


def _unexpected(old: str) -> str:
    return rf"unexpected keyword argument '{old}'"


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
# M-030 (removed): DiD.fit time= -> post=; M-031 (shimmed, name kept for the
# staggered calendar column) + M-085 (done): TripleDifference.fit time= raises in 2x2x2
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

    def test_did_time_raises_typeerror(self, did_2x2):
        """M-030 removal pin: ``time=`` is no longer a ``DiD.fit`` parameter."""
        from diff_diff import DifferenceInDifferences as DiD

        with pytest.raises(TypeError, match=_unexpected("time")):
            DiD().fit(did_2x2, outcome="y", treatment="d", time="p")
        with pytest.raises(TypeError, match=_unexpected("time")):
            DiD().fit(did_2x2, outcome="y", treatment="d", post="p", time="p")

    def test_ddd_time_raises_naming_post(self, ddd_2x2x2):
        """M-085: in 2x2x2 mode ``time=`` is the staggered calendar column only."""
        from diff_diff import TripleDifference

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r_new = TripleDifference().fit(ddd_2x2x2, "y", "g", "q", post="t")
            r_pos = TripleDifference().fit(ddd_2x2x2, "y", "g", "q", "t")
        assert r_new.att == r_pos.att
        with pytest.raises(ValueError, match=r"staggered calendar column only"):
            TripleDifference().fit(ddd_2x2x2, "y", "g", "q", time="t")

    def test_ddd_missing_post_raises(self, ddd_2x2x2):
        from diff_diff import TripleDifference

        with pytest.raises(TypeError, match=r"missing required argument: 'post'"):
            TripleDifference().fit(ddd_2x2x2, "y", "g", "q")


# ---------------------------------------------------------------------------
# M-137 / M-138 (removed): permutation_test / leave_one_out_test time= -> post=
# ---------------------------------------------------------------------------


class TestDiagnosticsTimeToPost:
    """Both functions take ``**estimator_kwargs``, so a stray ``time=`` has
    two loud failure shapes: without ``post=`` Python's own missing-argument
    error fires first; with ``post=`` the kwarg reaches
    ``DifferenceInDifferences(time=...)`` and is rejected there."""

    def test_permutation_canonical_and_positional_silent(self, placebo_panel):
        from diff_diff import permutation_test

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_kw = permutation_test(
                placebo_panel, "y", "d", post="p", unit="u", n_permutations=50, seed=1
            )
            r_pos = permutation_test(placebo_panel, "y", "d", "p", "u", n_permutations=50, seed=1)
        _assert_no_future_warning(record)
        assert r_kw.p_value == r_pos.p_value

    def test_permutation_time_raises_typeerror(self, placebo_panel):
        from diff_diff import permutation_test

        with pytest.raises(TypeError, match=r"missing 1 required positional argument: 'post'"):
            permutation_test(placebo_panel, "y", "d", time="p", unit="u", n_permutations=5)
        with pytest.raises(TypeError, match=_unexpected("time")):
            permutation_test(
                placebo_panel, "y", "d", post="p", unit="u", time="p", n_permutations=5
            )
        with pytest.raises(TypeError, match=r"missing 1 required positional argument: 'unit'"):
            permutation_test(placebo_panel, "y", "d", post="p")

    def test_leave_one_out_canonical_silent_and_time_raises(self, placebo_panel):
        from diff_diff import leave_one_out_test

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            l_kw = leave_one_out_test(placebo_panel, "y", "d", post="p", unit="u")
            l_pos = leave_one_out_test(placebo_panel, "y", "d", "p", "u")
        _assert_no_future_warning(record)
        assert l_kw.original_effect == l_pos.original_effect
        with pytest.raises(TypeError, match=r"missing 1 required positional argument: 'post'"):
            leave_one_out_test(placebo_panel, "y", "d", time="p", unit="u")
        with pytest.raises(TypeError, match=_unexpected("time")):
            leave_one_out_test(placebo_panel, "y", "d", post="p", unit="u", time="p")

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
# M-045..M-047 / M-115 (removed): the robust drop
# ---------------------------------------------------------------------------

ROBUST_CLASSES = [
    "DifferenceInDifferences",
    "TwoWayFixedEffects",
    "TripleDifference",
    "HeterogeneousAdoptionDiD",
    "LinearRegression",
]
# LinearRegression does not mix BaseEstimator in (tests/test_base_estimator.py
# NON_PARTICIPANTS), so it has no get_params() to inspect.
ROBUST_BASE_ESTIMATORS = ROBUST_CLASSES[:-1]


class TestRobustDrop:
    @pytest.mark.parametrize("cls_name", ROBUST_CLASSES)
    def test_default_construction_silent(self, cls_name):
        import diff_diff

        cls = getattr(diff_diff, cls_name)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            cls()
        _assert_no_future_warning(record)

    @pytest.mark.parametrize("cls_name", ROBUST_CLASSES)
    def test_robust_kwarg_raises_typeerror(self, cls_name):
        import diff_diff

        cls = getattr(diff_diff, cls_name)
        with pytest.raises(TypeError, match=_unexpected("robust")):
            cls(robust=False)
        with pytest.raises(TypeError, match=_unexpected("robust")):
            cls(robust=True)

    @pytest.mark.parametrize("cls_name", ROBUST_BASE_ESTIMATORS)
    def test_robust_absent_from_get_params(self, cls_name):
        import diff_diff

        cls = getattr(diff_diff, cls_name)
        est = cls()
        params = est.get_params()
        assert "robust" not in params
        assert "vcov_type" in params
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(robust=False)

    def test_vcov_type_canonical_path(self, did_2x2):
        from diff_diff import DifferenceInDifferences as DiD

        # On the balanced saturated 2x2 fixture HC1 and classical SEs are
        # ALGEBRAICALLY identical for ANY residuals (equal cell sizes make
        # n/(n-k) * sum SSE_c/n_c^2 equal SSE/(n-k) * sum 1/n_c), so a plain
        # ``!=`` only held by a 1-ULP BLAS roundoff and was bit-equal on CI's
        # Rust backend. Shrink the treated-post cell (unequal n_c) and scale
        # its noise so the two SEs differ by construction.
        df = did_2x2.iloc[:170].copy()  # cell (d=1, p=1) keeps 20 of 50 rows
        df["y"] = df["y"] * (1.0 + 3.0 * df["d"] * df["p"])

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_classical = DiD(vcov_type="classical").fit(df, "y", "d", "p")
            r_hc1 = DiD(vcov_type="hc1").fit(df, "y", "d", "p")
        _assert_no_future_warning(record)
        np.testing.assert_allclose(r_classical.att, r_hc1.att, rtol=1e-12)
        assert np.isfinite(r_classical.se) and np.isfinite(r_hc1.se)
        assert abs(r_classical.se - r_hc1.se) > 1e-3 * r_classical.se, (
            f"classical SE {r_classical.se!r} and HC1 SE {r_hc1.se!r} should differ "
            "under cell-dependent variance; vcov_type is not being dispatched"
        )

    def test_get_params_round_trip_silent_on_default(self):
        from diff_diff import DifferenceInDifferences as DiD

        est = DiD()
        params = est.get_params()
        assert "robust" not in params
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clone = DiD(**params)
        assert clone.vcov_type == est.vcov_type


# ---------------------------------------------------------------------------
# M-043 / M-095 (removed): StackedDiD clean_control= -> control_group=
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

    def test_ctor_old_name_raises_typeerror(self):
        from diff_diff import StackedDiD

        with pytest.raises(TypeError, match=_unexpected("clean_control")):
            StackedDiD(clean_control="never_treated")
        with pytest.raises(TypeError, match=_unexpected("clean_control")):
            StackedDiD(control_group="strict", clean_control="strict")

    def test_estimator_attribute_removed(self):
        from diff_diff import StackedDiD

        s = StackedDiD(control_group="strict")
        with pytest.raises(AttributeError):
            s.clean_control

    def test_get_params_round_trip_silent(self):
        from diff_diff import StackedDiD

        est = StackedDiD(control_group="strict")
        params = est.get_params()
        assert "clean_control" not in params
        assert params["control_group"] == "strict"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clone = StackedDiD(**params)
        assert clone.control_group == "strict"

    def test_sklearn_clone_preserves_explicit_control_group(self):
        from diff_diff import StackedDiD

        base = pytest.importorskip("sklearn.base")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", FutureWarning)
            est = StackedDiD(control_group="strict")
            cloned = base.clone(est)
        assert cloned is not est
        assert cloned.get_params() == est.get_params()
        assert cloned.control_group == est.control_group == "strict"
        _assert_no_future_warning(record)

    def test_set_params_canonical_works_and_old_name_rejected(self):
        from diff_diff import StackedDiD

        s = StackedDiD(control_group="strict")
        s.set_params(control_group="never_treated")
        assert s.control_group == "never_treated"
        with pytest.raises(ValueError, match="Unknown parameter"):
            s.set_params(clean_control="strict")
        # transactional: the failed call left the instance untouched
        assert s.control_group == "never_treated"

    def test_results_field_canonical_only(self, stacked_panel):
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
        assert d["control_group"] == "not_yet_treated"
        assert "clean_control" not in d
        assert "Control group:" in s
        with pytest.raises(AttributeError):
            res.clean_control

    def test_pickle_migration(self, stacked_panel):
        """``__setstate__`` keeps mapping the pre-M-095 ``clean_control`` key (kept at 4.0)."""
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
        with pytest.raises(AttributeError):
            old_style.clean_control

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
# M-044 / M-086 / M-087 / M-150 (+ M-136): the Wooldridge triple and LPDiD level
# ---------------------------------------------------------------------------


class TestWooldridgeTriple:
    def test_aggregate_event_study_canonical(self, etwfe_result):
        r = etwfe_result
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r.aggregate("event_study")
        _assert_no_future_warning(record)
        assert r.aggregation_weights["event_study"] == "cell"
        assert "event" not in r.aggregation_weights  # the 3.9 mirror key is gone

    def test_aggregate_event_rejected(self, etwfe_result):
        """M-086: ``type="event"`` is an unknown aggregation type at 4.0."""
        with pytest.raises(ValueError, match="type must be one of"):
            etwfe_result.aggregate("event")

    def test_to_dataframe_level_canonical_and_removed_spellings(self, etwfe_result):
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
        pd.testing.assert_frame_equal(f_default, f_level)
        # M-044: the aggregation= parameter is gone
        with pytest.raises(TypeError, match=_unexpected("aggregation")):
            r.to_dataframe(aggregation="event_study")
        with pytest.raises(TypeError, match=_unexpected("aggregation")):
            r.to_dataframe(level="gt", aggregation="gt")
        # M-150: the "event" level spelling is rejected, as is any unknown level
        with pytest.raises(ValueError):
            r.to_dataframe(level="event")
        with pytest.raises(ValueError):
            r.to_dataframe(level="bogus")

    def test_summary_alpha_positional_like_siblings(self, etwfe_result):
        r = etwfe_result
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            s = r.summary()
            s_same = r.summary(alpha=r.alpha)
            s_pos = r.summary(r.alpha)
        _assert_no_future_warning(record)
        assert "95% CI" in s
        assert s_same == s
        assert s_pos == s
        # The shared section-5 contract: a DIFFERENT alpha raises rather
        # than silently relabeling stored intervals.
        with pytest.raises(ValueError, match="never recomputes"):
            r.summary(alpha=0.10)
        # M-087: the aggregation= parameter is gone ...
        with pytest.raises(TypeError, match=_unexpected("aggregation")):
            r.summary(aggregation="simple")
        # ... and an old positional aggregation string binds to ``alpha``,
        # where the family's fit-alpha guard rejects it (a ValueError, not a
        # TypeError - summary(alpha) is positional like every sibling now).
        with pytest.raises(ValueError, match="never recomputes"):
            r.summary("simple")
        with pytest.raises(ValueError, match="never recomputes"):
            r.summary("event")

    def test_aggregation_weights_pickle_mirror(self, etwfe_result):
        """``__setstate__`` keeps migrating the pre-M-086 ``"event"`` key (kept at 4.0)."""
        r = etwfe_result
        # <=3.8 pickle: only the legacy key
        state = dict(r.__dict__)
        state["aggregation_weights"] = {"event": "cell"}
        r2 = object.__new__(type(r))
        r2.__setstate__(state)
        assert r2.aggregation_weights["event_study"] == "cell"
        assert "event" not in r2.aggregation_weights
        # 3.9-3.13 pickle: BOTH keys (the to_dict/aggregate mirror wrote the
        # legacy key alongside the canonical one) - the canonical value wins
        # and the legacy key is dropped.
        state = dict(r.__dict__)
        state["aggregation_weights"] = {"event": "cell", "event_study": "cohort_share"}
        r3 = object.__new__(type(r))
        r3.__setstate__(state)
        assert r3.aggregation_weights["event_study"] == "cohort_share"
        assert "event" not in r3.aggregation_weights


class TestLPDiDLevelValue:
    """M-136 (removed)."""

    def test_level_event_rejected(self):
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
        pd.testing.assert_frame_equal(f_new, f_canon)
        with pytest.raises(ValueError, match="event_study"):
            res.to_dataframe(level="event")
        with pytest.raises(ValueError, match="event_study"):
            res.to_dataframe(level="bogus")


# ---------------------------------------------------------------------------
# M-084 (removed): ContinuousDiD covariates live on fit() only
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
        assert np.isfinite(r.att)

    def test_ctor_covariates_raises_typeerror(self, cdid_panel):
        from diff_diff import ContinuousDiD

        with pytest.raises(TypeError, match=_unexpected("covariates")):
            ContinuousDiD(covariates=["x"])
        assert "covariates" not in ContinuousDiD().get_params()
        with pytest.raises(ValueError, match="Unknown parameter"):
            ContinuousDiD().set_params(covariates=["x"])

    def test_lowest_dose_guard_fires_for_fit_level(self, cdid_panel):
        from diff_diff import ContinuousDiD

        with pytest.raises(NotImplementedError, match="lowest_dose"):
            ContinuousDiD(control_group="lowest_dose").fit(
                cdid_panel, "y", "u", "t", "g", "dose", covariates=["x"]
            )
