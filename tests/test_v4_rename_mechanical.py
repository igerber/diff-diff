"""Behavioral pins for the 2(c)-ii mechanical rename wave (3.9 shims).

The dedicated shim test file required by the ledger rows this PR flips
(``test_ref`` on M-032, M-033..M-042, M-088/M-089, M-094, M-097..M-114 in
``docs/v4-deprecations.yaml``). Per the section 2 per-PR gate, every shimmed
surface pins here:

- ``pytest.warns(FutureWarning, match=...)`` on the OLD spelling with the
  migration message;
- zero warnings on the canonical spelling (and on positional calls, which
  bind to the new name in the old position);
- the both-supplied ``ValueError``;
- the sentinel-restored missing-argument ``TypeError`` naming the NEW param;
- bit-exact routing parity: the deprecated path returns the same numbers as
  the canonical path;
- for renamed results FIELDS: the new name is the dataclass field, the old
  name is a read-only warning property returning the same object, pickles
  from the old field name migrate via ``__setstate__``, and ``to_dict()``
  emits the dual keys.
"""

import pickle
import warnings
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import pytest

from diff_diff._deprecation import (
    NOT_SUPPLIED,
    _NotSupplied,
    deprecated_field_property,
    deprecated_kwarg_message,
    require_arg,
    resolve_renamed_kwarg,
)


def _assert_no_future_warning(record):
    fw = [w for w in record if issubclass(w.category, FutureWarning)]
    assert fw == [], [str(w.message) for w in fw]


# ---------------------------------------------------------------------------
# The shared helper module
# ---------------------------------------------------------------------------


class TestDeprecationHelpers:
    def test_sentinel_repr(self):
        assert repr(NOT_SUPPLIED) == "<not supplied>"
        assert isinstance(NOT_SUPPLIED, _NotSupplied)

    def test_message_template(self):
        msg = deprecated_kwarg_message("Cls.fit", "old", "use new= instead")
        assert msg == (
            "Cls.fit(old=) is deprecated and will be removed in 4.0; " "use new= instead."
        )

    def test_resolve_neither_returns_default(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = resolve_renamed_kwarg(
                "Cls.fit", "old", NOT_SUPPLIED, "new", NOT_SUPPLIED, default="d"
            )
        assert out == "d"

    def test_resolve_new_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = resolve_renamed_kwarg("Cls.fit", "old", NOT_SUPPLIED, "new", "value")
        assert out == "value"

    def test_resolve_old_warns_and_maps(self):
        with pytest.warns(FutureWarning, match=r"Cls\.fit\(old=\) is deprecated"):
            out = resolve_renamed_kwarg("Cls.fit", "old", "value", "new", NOT_SUPPLIED)
        assert out == "value"

    def test_resolve_old_extra_appended(self):
        with pytest.warns(FutureWarning, match="calendar column"):
            resolve_renamed_kwarg(
                "Cls.fit",
                "old",
                "value",
                "new",
                NOT_SUPPLIED,
                extra="From 4.0, old= means the calendar column only.",
            )

    def test_resolve_both_raises(self):
        with pytest.raises(ValueError, match=r"pass only new="):
            resolve_renamed_kwarg("Cls.fit", "old", "a", "new", "b")

    def test_require_arg(self):
        require_arg("Cls.fit", "new", "value")  # no raise
        with pytest.raises(TypeError, match=r"missing required argument: 'new'"):
            require_arg("Cls.fit", "new", NOT_SUPPLIED)

    def test_deprecated_field_property_shape(self):
        @dataclass
        class Toy:
            new_name: int = 3
            old_name = deprecated_field_property("Toy", "old_name", "new_name")

        toy = Toy()
        assert "new_name" in {f.name for f in fields(toy)}
        assert "old_name" not in {f.name for f in fields(toy)}
        with pytest.warns(FutureWarning, match=r"Toy\.old_name is deprecated"):
            assert toy.old_name == 3
        with pytest.raises(AttributeError):
            toy.old_name = 5


# ---------------------------------------------------------------------------
# M-032: WooldridgeDiD.fit cohort= -> first_treat=
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def etwfe_panel():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(40):
        g = 3 if u < 20 else 0
        for t in range(1, 6):
            y = 1.0 + 0.5 * t + (1.5 if g and t >= g else 0.0) + rng.normal(0, 0.3)
            rows.append((u, t, g, y))
    return pd.DataFrame(rows, columns=["unit", "time", "g", "y"])


class TestWooldridgeCohortRename:
    """M-032."""

    def test_canonical_and_positional_silent(self, etwfe_panel):
        from diff_diff import WooldridgeDiD

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_kw = WooldridgeDiD().fit(etwfe_panel, "y", "unit", "time", first_treat="g")
            r_pos = WooldridgeDiD().fit(etwfe_panel, "y", "unit", "time", "g")
        _assert_no_future_warning(record)
        assert r_kw.att == r_pos.att

    def test_old_name_warns_and_routes_identically(self, etwfe_panel):
        from diff_diff import WooldridgeDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            r_new = WooldridgeDiD().fit(etwfe_panel, "y", "unit", "time", first_treat="g")
        with pytest.warns(
            FutureWarning,
            match=r"WooldridgeDiD\.fit\(cohort=\) is deprecated and will be "
            r"removed in 4\.0; use first_treat= instead\.",
        ):
            r_old = WooldridgeDiD().fit(etwfe_panel, "y", "unit", "time", cohort="g")
        assert r_old.att == r_new.att
        assert r_old.se == r_new.se

    def test_both_supplied_raises(self, etwfe_panel):
        from diff_diff import WooldridgeDiD

        with pytest.raises(ValueError, match=r"pass only first_treat="):
            WooldridgeDiD().fit(etwfe_panel, "y", "unit", "time", first_treat="g", cohort="g")

    def test_missing_raises_typeerror(self, etwfe_panel):
        from diff_diff import WooldridgeDiD

        with pytest.raises(TypeError, match=r"missing required argument: 'first_treat'"):
            WooldridgeDiD().fit(etwfe_panel, "y", "unit", "time")


# ---------------------------------------------------------------------------
# M-033/M-034: ChaisemartinDHaultfoeuille.fit group= -> unit=, controls= ->
# covariates=; M-097: twowayfeweights group= -> unit=; M-114:
# ChaisemartinDHaultfoeuilleResults.groups -> units
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dcdh_panel():
    rng = np.random.default_rng(1)
    rows = []
    for g in range(12):
        switch = 3 if g < 6 else 99
        for t in range(1, 6):
            d = 1.0 if t >= switch else 0.0
            y = 0.3 * t + 1.2 * d + rng.normal(0, 0.2)
            rows.append((g, t, d, y))
    return pd.DataFrame(rows, columns=["g", "t", "d", "y"])


class TestDCDHRenames:
    """M-033 / M-034 / M-097 / M-114."""

    def test_canonical_and_positional_silent(self, dcdh_panel):
        from diff_diff import ChaisemartinDHaultfoeuille

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_kw = ChaisemartinDHaultfoeuille().fit(
                dcdh_panel, "y", unit="g", time="t", treatment="d"
            )
            r_pos = ChaisemartinDHaultfoeuille().fit(dcdh_panel, "y", "g", "t", "d")
        _assert_no_future_warning(record)
        assert r_kw.overall_att == r_pos.overall_att

    def test_group_warns_and_routes_identically(self, dcdh_panel):
        from diff_diff import ChaisemartinDHaultfoeuille

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_new = ChaisemartinDHaultfoeuille().fit(
                dcdh_panel, "y", unit="g", time="t", treatment="d"
            )
        with pytest.warns(
            FutureWarning,
            match=r"ChaisemartinDHaultfoeuille\.fit\(group=\) is deprecated and "
            r"will be removed in 4\.0; use unit= instead\.",
        ):
            r_old = ChaisemartinDHaultfoeuille().fit(
                dcdh_panel, "y", group="g", time="t", treatment="d"
            )
        assert r_old.overall_att == r_new.overall_att
        assert r_old.overall_se == r_new.overall_se

    def test_controls_warns_and_routes_identically(self, dcdh_panel):
        from diff_diff import ChaisemartinDHaultfoeuille

        rng = np.random.default_rng(9)
        df = dcdh_panel.assign(x=rng.normal(size=len(dcdh_panel)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_new = ChaisemartinDHaultfoeuille().fit(
                df, "y", "g", "t", "d", L_max=1, covariates=["x"]
            )
        with pytest.warns(FutureWarning, match=r"fit\(controls=\) is deprecated"):
            r_old = ChaisemartinDHaultfoeuille().fit(
                df, "y", "g", "t", "d", L_max=1, controls=["x"]
            )
        assert r_old.overall_att == r_new.overall_att

    def test_both_supplied_raises(self, dcdh_panel):
        from diff_diff import ChaisemartinDHaultfoeuille

        with pytest.raises(ValueError, match=r"pass only unit="):
            ChaisemartinDHaultfoeuille().fit(
                dcdh_panel, "y", unit="g", group="g", time="t", treatment="d"
            )

    def test_missing_followers_raise(self, dcdh_panel):
        from diff_diff import ChaisemartinDHaultfoeuille

        with pytest.raises(TypeError, match=r"missing required argument: 'unit'"):
            ChaisemartinDHaultfoeuille().fit(dcdh_panel, "y")
        with pytest.raises(TypeError, match=r"missing required argument: 'time'"):
            ChaisemartinDHaultfoeuille().fit(dcdh_panel, "y", unit="g")
        with pytest.raises(TypeError, match=r"missing required argument: 'treatment'"):
            ChaisemartinDHaultfoeuille().fit(dcdh_panel, "y", unit="g", time="t")

    def test_twowayfeweights_rename(self, dcdh_panel):
        from diff_diff import twowayfeweights

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            tw_kw = twowayfeweights(dcdh_panel, "y", unit="g", time="t", treatment="d")
            tw_pos = twowayfeweights(dcdh_panel, "y", "g", "t", "d")
        _assert_no_future_warning(record)
        with pytest.warns(FutureWarning, match=r"twowayfeweights\(group=\) is deprecated"):
            tw_old = twowayfeweights(dcdh_panel, "y", group="g", time="t", treatment="d")
        assert tw_kw.beta_fe == tw_old.beta_fe == tw_pos.beta_fe
        with pytest.raises(TypeError, match=r"missing required argument: 'unit'"):
            twowayfeweights(dcdh_panel, "y")

    def test_units_field_and_groups_alias(self, dcdh_panel):
        from diff_diff import ChaisemartinDHaultfoeuille
        from diff_diff.chaisemartin_dhaultfoeuille_results import (
            ChaisemartinDHaultfoeuilleResults,
        )

        assert "units" in ChaisemartinDHaultfoeuilleResults.__dataclass_fields__
        assert "groups" not in ChaisemartinDHaultfoeuilleResults.__dataclass_fields__
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ChaisemartinDHaultfoeuille().fit(dcdh_panel, "y", "g", "t", "d")
        with pytest.warns(
            FutureWarning,
            match=r"ChaisemartinDHaultfoeuilleResults\.groups is deprecated",
        ):
            alias = res.groups
        assert alias is res.units
        with pytest.raises(AttributeError):
            res.groups = [1]
        assert "n_units=" in repr(res)
        assert "Units (post-filter):" in res.summary()

    def test_to_dict_never_serialized_the_unit_list(self, dcdh_panel):
        """dCDH ``to_dict()`` is a hand-built headline dict: it serialized
        neither ``groups`` (pre-M-114) nor ``units`` (post), so no dual-key
        window applies here - unlike M-094's ``treatment_col``, which WAS
        an emitted key. Serialization must also never touch the warning
        property."""
        from diff_diff import ChaisemartinDHaultfoeuille

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ChaisemartinDHaultfoeuille().fit(dcdh_panel, "y", "g", "t", "d")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            d = res.to_dict()
        assert "groups" not in d
        assert "units" not in d

    def test_pickle_migration(self, dcdh_panel):
        from diff_diff import ChaisemartinDHaultfoeuille

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ChaisemartinDHaultfoeuille().fit(dcdh_panel, "y", "g", "t", "d")
        clone = pickle.loads(pickle.dumps(res))
        assert clone.units == res.units
        # Simulate a 3.8-era pickle payload keyed on the old field name.
        state = dict(res.__dict__)
        state["groups"] = state.pop("units")
        old_style = object.__new__(type(res))
        old_style.__setstate__(state)
        assert old_style.units == res.units


# ---------------------------------------------------------------------------
# M-035..M-039: HeterogeneousAdoptionDiD.fit *_col -> bare names
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def had_panel():
    rng = np.random.default_rng(2)
    rows = []
    for u in range(120):
        d = max(0.0, rng.normal(1.0, 0.6))
        for t in (0, 1):
            y = 0.5 + 0.4 * t + 1.1 * d * t + rng.normal(0, 0.3)
            rows.append((u, t, d if t == 1 else 0.0, y))
    return pd.DataFrame(rows, columns=["u", "t", "dose", "y"])


class TestHADColRenames:
    """M-035..M-039."""

    def test_canonical_and_positional_silent(self, had_panel):
        from diff_diff import HeterogeneousAdoptionDiD

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_pos = HeterogeneousAdoptionDiD().fit(had_panel, "y", "dose", "t", "u")
            r_kw = HeterogeneousAdoptionDiD().fit(
                had_panel, outcome="y", dose="dose", time="t", unit="u"
            )
        _assert_no_future_warning(record)
        assert r_pos.att == r_kw.att

    def test_old_names_warn_once_each_and_route_identically(self, had_panel):
        from diff_diff import HeterogeneousAdoptionDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_new = HeterogeneousAdoptionDiD().fit(had_panel, "y", "dose", "t", "u")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_old = HeterogeneousAdoptionDiD().fit(
                had_panel,
                outcome_col="y",
                dose_col="dose",
                time_col="t",
                unit_col="u",
            )
        fw = [w for w in record if issubclass(w.category, FutureWarning)]
        assert len(fw) == 4
        messages = "\n".join(str(w.message) for w in fw)
        for old in ("outcome_col", "dose_col", "time_col", "unit_col"):
            assert f"HeterogeneousAdoptionDiD.fit({old}=) is deprecated" in messages
        assert r_old.att == r_new.att
        assert r_old.se == r_new.se

    def test_both_supplied_raises(self, had_panel):
        from diff_diff import HeterogeneousAdoptionDiD

        with pytest.raises(ValueError, match=r"pass only unit="):
            HeterogeneousAdoptionDiD().fit(had_panel, "y", "dose", "t", "u", unit_col="u")

    def test_missing_raises_typeerror(self, had_panel):
        from diff_diff import HeterogeneousAdoptionDiD

        with pytest.raises(TypeError, match=r"missing required argument: 'time'"):
            HeterogeneousAdoptionDiD().fit(had_panel, "y", "dose")


# ---------------------------------------------------------------------------
# M-040..M-042: RegressionDiscontinuity.fit renames; M-094: results field
# treatment_col -> takeup; M-088/M-089: RDPlot.fit renames
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rd_data():
    rng = np.random.default_rng(3)
    n = 500
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 0.8 * (x >= 0) + 1.2 * x + rng.normal(0, 0.3, n)
    tk = ((x >= 0) & (rng.uniform(size=n) > 0.2)).astype(float)
    return pd.DataFrame({"x": x, "y": y, "takeup": tk})


class TestRDDRenames:
    """M-040..M-042 + M-094."""

    def test_canonical_and_positional_silent(self, rd_data):
        from diff_diff import RegressionDiscontinuity

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_pos = RegressionDiscontinuity().fit(rd_data, "y", "x")
            r_fz = RegressionDiscontinuity().fit(rd_data, "y", "x", takeup="takeup")
        _assert_no_future_warning(record)
        assert r_pos.att != r_fz.att  # sharp vs fuzzy are different fits

    def test_old_names_warn_and_route_identically(self, rd_data):
        from diff_diff import RegressionDiscontinuity

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_new = RegressionDiscontinuity().fit(rd_data, "y", "x", takeup="takeup")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_old = RegressionDiscontinuity().fit(
                rd_data, outcome_col="y", running_col="x", treatment_col="takeup"
            )
        fw = [w for w in record if issubclass(w.category, FutureWarning)]
        assert len(fw) == 3
        messages = "\n".join(str(w.message) for w in fw)
        assert "use takeup= instead" in messages
        assert r_old.att == r_new.att
        assert r_old.se == r_new.se

    def test_both_supplied_raises(self, rd_data):
        from diff_diff import RegressionDiscontinuity

        with pytest.raises(ValueError, match=r"pass only takeup="):
            RegressionDiscontinuity().fit(
                rd_data, "y", "x", takeup="takeup", treatment_col="takeup"
            )

    def test_missing_raises_typeerror(self, rd_data):
        from diff_diff import RegressionDiscontinuity

        with pytest.raises(TypeError, match=r"missing required argument: 'running'"):
            RegressionDiscontinuity().fit(rd_data, "y")

    def test_takeup_field_alias_and_dual_keys(self, rd_data):
        from diff_diff import RegressionDiscontinuity
        from diff_diff.rdd import RegressionDiscontinuityResults

        assert "takeup" in RegressionDiscontinuityResults.__dataclass_fields__
        assert "treatment_col" not in RegressionDiscontinuityResults.__dataclass_fields__
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = RegressionDiscontinuity().fit(rd_data, "y", "x", takeup="takeup")
        assert res.takeup == "takeup"
        with pytest.warns(
            FutureWarning,
            match=r"RegressionDiscontinuityResults\.treatment_col is deprecated",
        ):
            alias = res.treatment_col
        assert alias == "takeup"
        with pytest.raises(AttributeError):
            res.treatment_col = "x"
        d = res.to_dict()
        assert d["takeup"] == "takeup"
        assert d["treatment_col"] == "takeup"

    def test_pickle_migration(self, rd_data):
        from diff_diff import RegressionDiscontinuity

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = RegressionDiscontinuity().fit(rd_data, "y", "x", takeup="takeup")
        clone = pickle.loads(pickle.dumps(res))
        assert clone.takeup == res.takeup
        state = dict(res.__dict__)
        state["treatment_col"] = state.pop("takeup")
        old_style = object.__new__(type(res))
        old_style.__setstate__(state)
        assert old_style.takeup == "takeup"


class TestRDPlotRenames:
    """M-088/M-089."""

    def test_rename_pair(self, rd_data):
        from diff_diff import RDPlot

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            p_new = RDPlot().fit(rd_data, "y", "x")
        _assert_no_future_warning(record)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            p_old = RDPlot().fit(rd_data, outcome_col="y", running_col="x")
        fw = [w for w in record if issubclass(w.category, FutureWarning)]
        assert len(fw) == 2
        assert "RDPlot.fit(outcome_col=) is deprecated" in str(fw[0].message)
        np.testing.assert_array_equal(p_new.coef, p_old.coef)
        pd.testing.assert_frame_equal(p_new.vars_bins, p_old.vars_bins)
        with pytest.raises(ValueError, match=r"pass only running="):
            RDPlot().fit(rd_data, "y", "x", running_col="x")
        with pytest.raises(TypeError, match=r"missing required argument: 'outcome'"):
            RDPlot().fit(rd_data)


# ---------------------------------------------------------------------------
# M-098..M-112: the three HAD pretest entry points; M-113: trim_weights
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def had_multi_panel():
    rng = np.random.default_rng(4)
    rows = []
    for u in range(80):
        d = max(0.0, rng.normal(1.0, 0.5))
        for t in (0, 1, 2, 3):
            dose = d if t >= 2 else 0.0
            yv = 0.4 * t + 0.9 * dose + rng.normal(0, 0.3)
            rows.append((u, t, dose, yv))
    return pd.DataFrame(rows, columns=["u", "t", "dose", "y"])


class TestPretestFunctionRenames:
    """M-098..M-112 (three functions, five params each)."""

    def test_joint_pretrends_rename(self, had_multi_panel):
        from diff_diff import joint_pretrends_test

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            j_new = joint_pretrends_test(
                had_multi_panel,
                "y",
                "dose",
                "t",
                "u",
                pre_periods=[0],
                base_period=1,
                n_bootstrap=99,
                seed=7,
            )
        _assert_no_future_warning(record)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            j_old = joint_pretrends_test(
                had_multi_panel,
                outcome_col="y",
                dose_col="dose",
                time_col="t",
                unit_col="u",
                pre_periods=[0],
                base_period=1,
                n_bootstrap=99,
                seed=7,
            )
        fw = [w for w in record if issubclass(w.category, FutureWarning)]
        assert len(fw) == 4
        assert "joint_pretrends_test(outcome_col=) is deprecated" in str(fw[0].message)
        assert j_new.cvm_stat_joint == j_old.cvm_stat_joint
        assert j_new.p_value == j_old.p_value
        with pytest.raises(TypeError, match=r"missing required argument: 'pre_periods'"):
            joint_pretrends_test(had_multi_panel, "y", "dose", "t", "u", base_period=1)

    def test_joint_homogeneity_rename(self, had_multi_panel):
        from diff_diff import joint_homogeneity_test

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            j_new = joint_homogeneity_test(
                had_multi_panel,
                "y",
                "dose",
                "t",
                "u",
                post_periods=[2, 3],
                base_period=1,
                n_bootstrap=99,
                seed=7,
            )
        _assert_no_future_warning(record)
        with pytest.warns(
            FutureWarning,
            match=r"joint_homogeneity_test\(dose_col=\) is deprecated",
        ):
            j_old = joint_homogeneity_test(
                had_multi_panel,
                "y",
                dose_col="dose",
                time="t",
                unit="u",
                post_periods=[2, 3],
                base_period=1,
                n_bootstrap=99,
                seed=7,
            )
        assert j_new.cvm_stat_joint == j_old.cvm_stat_joint
        with pytest.raises(TypeError, match=r"missing required argument: 'post_periods'"):
            joint_homogeneity_test(had_multi_panel, "y", "dose", "t", "u", base_period=1)

    def test_workflow_rename(self, had_multi_panel):
        from diff_diff import did_had_pretest_workflow

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            r_new = did_had_pretest_workflow(
                had_multi_panel,
                "y",
                "dose",
                "t",
                "u",
                n_bootstrap=99,
                seed=7,
                aggregate="event_study",
            )
        _assert_no_future_warning(record)
        with pytest.warns(
            FutureWarning,
            match=r"did_had_pretest_workflow\(unit_col=\) is deprecated",
        ):
            r_old = did_had_pretest_workflow(
                had_multi_panel,
                "y",
                "dose",
                "t",
                unit_col="u",
                n_bootstrap=99,
                seed=7,
                aggregate="event_study",
            )
        assert r_new.homogeneity_joint.cvm_stat_joint == r_old.homogeneity_joint.cvm_stat_joint
        with pytest.raises(ValueError, match=r"pass only time="):
            did_had_pretest_workflow(had_multi_panel, "y", "dose", "t", "u", time_col="t")


class TestValidationMessagesRecommendCanonicalNames:
    """User-facing validation messages must steer callers to the NEW
    spellings (CI review R1 on PR #742): the staggered fail-closed error
    and the cohort-mismatch family recommend first_treat=, never the
    deprecated first_treat_col=."""

    def test_staggered_fail_closed_recommends_first_treat(self, had_multi_panel):
        from diff_diff import HeterogeneousAdoptionDiD

        rng = np.random.default_rng(11)
        rows = []
        for u in range(60):
            start = 1 if u % 2 == 0 else 2
            d = max(0.1, rng.normal(1.0, 0.4))
            for t in (0, 1, 2, 3):
                dose = d if t >= start else 0.0
                rows.append((u, t, dose, 0.3 * t + 0.8 * dose + rng.normal(0, 0.2)))
        staggered = pd.DataFrame(rows, columns=["u", "t", "dose", "y"])
        with pytest.raises(ValueError) as exc:
            HeterogeneousAdoptionDiD().fit(
                staggered, "y", "dose", "t", "u", aggregate="event_study"
            )
        message = str(exc.value)
        assert "Pass first_treat=" in message
        assert "first_treat_col" not in message


class TestPretestValidationMessagesUseCanonicalNames:
    """CI review R3 on PR #742: invalid-period errors from the renamed
    pretest APIs must describe the canonical ``time`` parameter, never the
    deprecated ``time_col`` spelling."""

    def test_invalid_period_messages_exclude_time_col(self, had_multi_panel):
        from diff_diff import joint_homogeneity_test, joint_pretrends_test

        for func, kw in (
            (joint_pretrends_test, {"pre_periods": [99], "base_period": 1}),
            (joint_homogeneity_test, {"post_periods": [99], "base_period": 1}),
        ):
            with pytest.raises(ValueError) as exc:
                func(had_multi_panel, "y", "dose", "t", "u", n_bootstrap=9, **kw)
            message = str(exc.value)
            assert "time_col" not in message, message


class TestFirstTreatOptionalAliasPerSurface:
    """M-039 / M-102 / M-107 / M-112: the OPTIONAL first_treat_col alias,
    pinned per surface (warn + routing parity + both-supplied rejection)."""

    @staticmethod
    def _with_ft(panel):
        ft = panel.groupby("u")["dose"].transform(lambda s: 2 if (s > 0).any() else 0)
        return panel.assign(ft=ft)

    def test_had_fit_first_treat_col(self, had_multi_panel):
        from diff_diff import HeterogeneousAdoptionDiD

        df = self._with_ft(had_multi_panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_new = HeterogeneousAdoptionDiD().fit(
                df, "y", "dose", "t", "u", first_treat="ft", aggregate="event_study"
            )
        with pytest.warns(
            FutureWarning,
            match=r"HeterogeneousAdoptionDiD\.fit\(first_treat_col=\) is deprecated",
        ):
            r_old = HeterogeneousAdoptionDiD().fit(
                df,
                "y",
                "dose",
                "t",
                "u",
                first_treat_col="ft",
                aggregate="event_study",
            )
        np.testing.assert_array_equal(r_old.att, r_new.att)
        with pytest.raises(ValueError, match=r"pass only first_treat="):
            HeterogeneousAdoptionDiD().fit(
                df,
                "y",
                "dose",
                "t",
                "u",
                first_treat="ft",
                first_treat_col="ft",
                aggregate="event_study",
            )

    @pytest.mark.parametrize(
        "func_name,extra",
        [
            ("joint_pretrends_test", {"pre_periods": [0], "base_period": 1}),
            ("joint_homogeneity_test", {"post_periods": [2, 3], "base_period": 1}),
            ("did_had_pretest_workflow", {"aggregate": "event_study"}),
        ],
    )
    def test_pretest_functions_first_treat_col(self, had_multi_panel, func_name, extra):
        import diff_diff

        func = getattr(diff_diff, func_name)
        df = self._with_ft(had_multi_panel)
        kw = dict(n_bootstrap=99, seed=7, **extra)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_new = func(df, "y", "dose", "t", "u", first_treat="ft", **kw)
        with pytest.warns(
            FutureWarning,
            match=rf"{func_name}\(first_treat_col=\) is deprecated",
        ):
            r_old = func(df, "y", "dose", "t", "u", first_treat_col="ft", **kw)
        new_p = r_new.p_value if hasattr(r_new, "p_value") else r_new.homogeneity_joint.p_value
        old_p = r_old.p_value if hasattr(r_old, "p_value") else r_old.homogeneity_joint.p_value
        assert new_p == old_p
        with pytest.raises(ValueError, match=r"pass only first_treat="):
            func(df, "y", "dose", "t", "u", first_treat="ft", first_treat_col="ft", **kw)


class TestTrimWeightsRename:
    """M-113."""

    def test_rename_pair(self):
        from diff_diff import trim_weights

        df = pd.DataFrame({"w": [1.0, 2.0, 50.0], "y": [1, 2, 3]})
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out_pos = trim_weights(df, "w", upper=10.0)
            out_kw = trim_weights(df, weights="w", upper=10.0)
        _assert_no_future_warning(record)
        with pytest.warns(FutureWarning, match=r"trim_weights\(weight_col=\) is deprecated"):
            out_old = trim_weights(df, weight_col="w", upper=10.0)
        assert (out_pos["w"] == out_old["w"]).all()
        assert (out_kw["w"] == out_old["w"]).all()
        assert out_old["w"].max() == 10.0
        with pytest.raises(ValueError, match=r"pass only weights="):
            trim_weights(df, "w", weight_col="w")
        with pytest.raises(TypeError, match=r"missing required argument: 'weights'"):
            trim_weights(df, upper=10.0)
