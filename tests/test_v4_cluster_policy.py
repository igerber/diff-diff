"""Item 7b of the 4.0 cut - the uniform cluster policy (ledger rows M-080 and M-155).

Pins, in one place:

1. The shared ``cluster`` contract (``diff_diff.utils.validate_cluster``): ``str``
   (a column name), ``None`` (the estimator's default clustering) or ``False``
   (explicitly unclustered) on every ``BaseEstimator`` subclass exposing the
   parameter; ``True`` / non-``str`` labels raise ONE shared ``ValueError`` at
   ``__init__``, through ``set_params`` and again at ``fit()`` (direct-mutation
   defense). ``StackedDiD`` keeps its own ``'unit'``/``'unit_subexp'`` vocabulary.
2. ``SpilloverDiD`` auto-clusters at ``unit`` when ``cluster=`` is omitted (the ONE
   default SE that moves at 4.0) with ``TwoWayFixedEffects``' carve-outs mirrored:
   explicit ``cluster=`` wins, the auto-cluster is dropped on the Conley path and is
   never injected as a survey PSU; ``cluster=False`` reproduces the 3.x
   observation-level HC1 lane (its goldens are pinned in ``tests/test_spillover.py``).
3. ``cluster=False`` on the estimators whose clustering is a CHOICE (category c1:
   TWFE static + event-study, SunAbraham, WooldridgeDiD OLS + GLM, LPDiD): the plain
   heteroskedasticity-robust sandwich with residual-df t inference (the
   ``df_convention`` knob is inert without a cluster count), no cluster metadata; the
   lanes whose bootstrap draws AT the analytical cluster (DiD / TWFE wild,
   WooldridgeDiD multiplier) and the cluster-only Bell-McCaffrey DOF lanes of
   SunAbraham / WooldridgeDiD reject it.
4. ``cluster=False`` on the unit-level-by-construction estimators (category b / c2)
   and on the 2x2 surfaces is a bit-identical no-op.
5. Cluster metadata (4.0): DiD / TWFE analytical fits report ``n_clusters`` from the
   regression's own effective cluster count on non-survey fits; ``SunAbrahamResults``
   gains ``n_clusters`` and reports ``cluster_name`` on clustered analytical fits;
   TWFE's survey-lane auto label is None (the regression received no cluster ids).
6. The ledger rows read ``done`` with this file as ``test_ref``.
"""

from __future__ import annotations

import inspect
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import diff_diff
from diff_diff import (
    CallawaySantAnna,
    ChaisemartinDHaultfoeuille,
    DifferenceInDifferences,
    EfficientDiD,
    LPDiD,
    SpilloverDiD,
    StackedDiD,
    SunAbraham,
    SurveyDesign,
    TripleDifference,
    TwoWayFixedEffects,
    WooldridgeDiD,
    generate_ddd_data,
    generate_reversible_did_data,
    generate_staggered_ddd_data,
)
from diff_diff._base import BaseEstimator
from diff_diff.linalg import LinearRegression
from diff_diff.lwdid_sensitivity import robustness_pre_periods, sensitivity_no_anticipation
from diff_diff.utils import ClusterArg, validate_cluster
from tests.test_dml_did import FIT_KW as DML_FIT_KW
from tests.test_dml_did import make_staggered_dml_data
from tests.test_lwdid import _make_common_timing_panel
from tests.test_t23_spillover_tva_drift import D_BAR_KM, _build_t23_panel
from tests.test_v4_matrix import ROWS as LEDGER_ROWS

THIS_TEST_REF = "tests/test_v4_cluster_policy.py"

CONTRACT_MSG = r"cluster must be a column name \(str\), None"


def _t_tail(t, df):
    return 2 * stats.t.sf(abs(t), df)


def _quintet(r):
    return (r.att, r.se, r.t_stat, r.p_value, tuple(r.conf_int))


def _meta(r):
    return (getattr(r, "cluster_name", "absent"), getattr(r, "n_clusters", "absent"))


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def panel():
    """60 units x 8 periods, cohorts {never, 4, 6}; unit-constant survey weight."""
    rng = np.random.default_rng(7)
    rows = []
    for u in range(60):
        ft = [0, 4, 6][u % 3]
        lat, lon = 40.0 + 0.05 * (u // 6), -100.0 + 0.05 * (u % 6)
        w = rng.uniform(0.5, 1.5)
        for t in range(1, 9):
            eff = 0.6 if (ft and t >= ft) else 0.0
            rows.append(
                dict(
                    unit=u,
                    time=t,
                    first_treat=ft,
                    grp=int(ft > 0),
                    post=int(t >= 4),
                    outcome=0.3 * u / 60 + 0.1 * t + eff + rng.standard_normal() * 0.6,
                    lat=lat,
                    lon=lon,
                    x=rng.standard_normal(),
                    w=w,
                    row=len(rows),
                )
            )
    df = pd.DataFrame(rows)
    df["d"] = df["grp"] * df["post"]
    df["treated"] = ((df["first_treat"] > 0) & (df["time"] >= df["first_treat"])).astype(int)
    df["yb"] = (df["outcome"] > df["outcome"].median()).astype(float)
    return df


@pytest.fixture(scope="module")
def spill_panel():
    return _build_t23_panel()


SA_KW = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
TW_KW = dict(outcome="outcome", treatment="d", post="post", unit="unit")
ES_KW = dict(
    outcome="outcome",
    treatment="grp",
    unit="unit",
    time="time",
    event_study=True,
    post_periods=[4, 5, 6, 7, 8],
)
LP_KW = dict(outcome="outcome", unit="unit", time="time", treatment="treated")
LP_CTOR = dict(pre_window=2, post_window=2)
SP_KW = dict(outcome="y", unit="unit", time="time", treatment="D")


def _spill(**kw):
    return SpilloverDiD(rings=[0.0, D_BAR_KM], conley_coords=("lat", "lon"), **kw)


def _fit_quiet(est, df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit(df, **kw)


# --------------------------------------------------------------------------- #
# (1) The shared contract
# --------------------------------------------------------------------------- #


def _cluster_roster():
    """Every exported BaseEstimator subclass (deduplicated by class object;
    aliases such as CS / SA / TWFE point at the same classes) whose __init__
    exposes ``cluster``. Functions such as bias_corrected_local_linear (an
    ndarray ``cluster``) are not estimators and are excluded."""
    seen, out = set(), []
    for name in diff_diff.__all__:
        obj = getattr(diff_diff, name)
        if not inspect.isclass(obj) or id(obj) in seen or not issubclass(obj, BaseEstimator):
            continue
        seen.add(id(obj))
        if "cluster" in inspect.signature(obj.__init__).parameters:
            out.append(obj)
    return sorted(out, key=lambda c: c.__name__)


ROSTER = _cluster_roster()
CTOR_KW = {"SpilloverDiD": {"rings": [0.0, D_BAR_KM]}, "LPDiD": LP_CTOR}


def test_roster_is_the_sixteen_documented_classes():
    assert {c.__name__ for c in ROSTER} == {
        "CallawaySantAnna",
        "ChaisemartinDHaultfoeuille",
        "DMLDiD",
        "DifferenceInDifferences",
        "EfficientDiD",
        "HeterogeneousAdoptionDiD",
        "ImputationDiD",
        "LPDiD",
        "LWDiD",
        "SpilloverDiD",
        "StackedDiD",
        "SunAbraham",
        "TripleDifference",
        "TwoStageDiD",
        "TwoWayFixedEffects",
        "WooldridgeDiD",
    }


class TestValidateCluster:
    @pytest.mark.parametrize("ok", [None, False, "unit", "state"])
    def test_accepts_and_returns_unchanged(self, ok):
        assert validate_cluster(ok) is ok

    @pytest.mark.parametrize("bad", [True, 1, 0, 1.5, [], ("a",), b"x", np.bool_(False)])
    def test_rejects_everything_else(self, bad):
        with pytest.raises(ValueError, match=CONTRACT_MSG):
            validate_cluster(bad)

    def test_alias_is_the_documented_union(self):
        import typing

        assert set(typing.get_args(ClusterArg)) == {str, typing.Literal[False], type(None)}


@pytest.mark.parametrize("cls", ROSTER, ids=lambda c: c.__name__)
def test_false_accepted_and_round_trips_except_stacked(cls):
    kw = CTOR_KW.get(cls.__name__, {})
    if cls is StackedDiD:
        with pytest.raises(ValueError, match="cluster must be 'unit' or 'unit_subexp'"):
            StackedDiD(cluster=False)
        return
    est = cls(cluster=False, **kw)
    assert est.cluster is False
    assert est.get_params()["cluster"] is False
    est2 = cls(**kw)
    est2.set_params(cluster=False)
    assert est2.cluster is False
    est2.set_params(cluster=None)
    assert est2.cluster is None


@pytest.mark.parametrize("cls", ROSTER, ids=lambda c: c.__name__)
def test_true_and_non_str_labels_rejected_everywhere(cls):
    kw = CTOR_KW.get(cls.__name__, {})
    if cls is StackedDiD:
        for bad in (True, 7):
            with pytest.raises(ValueError, match="cluster must be 'unit' or 'unit_subexp'"):
                StackedDiD(cluster=bad)
        return
    for bad in (True, 7):
        with pytest.raises(ValueError, match=CONTRACT_MSG):
            cls(cluster=bad, **kw)
        est = cls(**kw)
        with pytest.raises(ValueError, match=CONTRACT_MSG):
            est.set_params(cluster=bad)
        assert est.cluster is None  # transactional rollback


def test_dcdh_column_name_still_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="user-specified clustering"):
        ChaisemartinDHaultfoeuille(cluster="unit")


def _fit_recipes(panel, spill_panel):
    """Per-class (ctor kwargs, data, fit kwargs) for a minimal fit."""
    had_g = 200
    rng = np.random.default_rng(3)
    d = rng.uniform(0.0, 1.0, had_g)
    d[0] = 0.0
    dy = 0.3 * d + 0.1 * rng.standard_normal(had_g)
    had = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(had_g), 2),
            "period": np.tile([1, 2], had_g),
            "dose": np.column_stack([np.zeros(had_g), d]).ravel(),
            "outcome": np.column_stack([np.zeros(had_g), dy]).ravel(),
        }
    )
    return {
        "DifferenceInDifferences": (
            {},
            panel,
            dict(outcome="outcome", treatment="grp", post="post"),
        ),
        "TwoWayFixedEffects": ({}, panel, TW_KW),
        "SunAbraham": ({}, panel, SA_KW),
        "WooldridgeDiD": ({}, panel, SA_KW),
        "ImputationDiD": ({}, panel, SA_KW),
        "TwoStageDiD": ({}, panel, SA_KW),
        "CallawaySantAnna": ({}, panel, SA_KW),
        "EfficientDiD": ({}, panel, SA_KW),
        "LPDiD": (LP_CTOR, panel, LP_KW),
        "SpilloverDiD": (
            dict(rings=[0.0, D_BAR_KM], conley_coords=("lat", "lon")),
            spill_panel,
            SP_KW,
        ),
        "DMLDiD": (
            dict(seed=0, n_folds=2),
            make_staggered_dml_data(),
            dict(covariates=["x1", "x2"], **DML_FIT_KW),
        ),
        "ChaisemartinDHaultfoeuille": (
            {},
            generate_reversible_did_data(n_groups=40, n_periods=5, seed=1),
            dict(outcome="outcome", unit="group", time="period", treatment="treatment"),
        ),
        "TripleDifference": (
            {},
            generate_ddd_data(n_per_cell=100, treatment_effect=2.0, seed=42),
            dict(outcome="outcome", group="group", partition="partition", post="time"),
        ),
        "LWDiD": (
            {},
            _make_common_timing_panel(),
            dict(outcome="y", unit="unit", time="time", treatment="treat"),
        ),
        "HeterogeneousAdoptionDiD": (
            dict(design="continuous_at_zero"),
            had,
            dict(outcome="outcome", dose="dose", time="period", unit="unit"),
        ),
    }


@pytest.fixture(scope="module")
def recipes(panel, spill_panel):
    return _fit_recipes(panel, spill_panel)


@pytest.mark.parametrize(
    "name", sorted(c.__name__ for c in ROSTER if c is not StackedDiD), ids=lambda n: n
)
def test_direct_mutation_to_true_raises_at_fit_on_every_class(name, recipes):
    """The fit-time re-check is wired on EVERY fit path, not only the c1 lanes:
    without it ``True`` would fall through to a column lookup (KeyError)."""
    ctor, data, fit_kw = recipes[name]
    est = getattr(diff_diff, name)(**ctor)
    est.cluster = True  # bypasses __init__ and set_params on purpose
    with pytest.raises(ValueError, match=CONTRACT_MSG):
        _fit_quiet(est, data, **fit_kw)


# --------------------------------------------------------------------------- #
# (2) SpilloverDiD auto-cluster (category a)
# --------------------------------------------------------------------------- #


class TestSpilloverAutoCluster:
    @pytest.fixture(scope="class")
    def default(self, spill_panel):
        return _fit_quiet(_spill(), spill_panel, **SP_KW)

    @pytest.fixture(scope="class")
    def explicit_unit(self, spill_panel):
        return _fit_quiet(_spill(cluster="unit"), spill_panel, **SP_KW)

    @pytest.fixture(scope="class")
    def off(self, spill_panel):
        return _fit_quiet(_spill(cluster=False), spill_panel, **SP_KW)

    def test_default_is_cr1_at_unit(self, default, explicit_unit, spill_panel):
        G = spill_panel["unit"].nunique()
        assert default.cluster_name == "unit"
        assert default.n_clusters == G == explicit_unit.n_clusters
        assert default.att == explicit_unit.att
        np.testing.assert_allclose(default.se, explicit_unit.se, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(default.p_value, explicit_unit.p_value, rtol=1e-12)
        assert "CR1 cluster-robust at unit, G=" in default.summary()

    def test_false_is_the_observation_level_hc1_lane(self, default, off):
        assert off.cluster_name is None and off.n_clusters is None
        assert off.att == default.att
        assert off.se != default.se
        assert "HC1 heteroskedasticity-robust" in off.summary()

    def test_false_differs_from_none(self, default, off):
        assert default.se != off.se  # None is NOT the off-switch

    def test_conley_drops_the_auto_cluster(self, spill_panel):
        kw = dict(vcov_type="conley", conley_cutoff_km=D_BAR_KM, conley_lag_cutoff=0)
        auto = _fit_quiet(_spill(**kw), spill_panel, **SP_KW)
        off = _fit_quiet(_spill(cluster=False, **kw), spill_panel, **SP_KW)
        assert auto.cluster_name is None and auto.n_clusters is None
        assert auto.se == off.se and auto.p_value == off.p_value
        assert off.cluster_name is None

    def test_conley_explicit_cluster_keeps_the_product_kernel(self, spill_panel):
        kw = dict(vcov_type="conley", conley_cutoff_km=D_BAR_KM, conley_lag_cutoff=0)
        auto = _fit_quiet(_spill(**kw), spill_panel, **SP_KW)
        prod = _fit_quiet(_spill(cluster="unit", **kw), spill_panel, **SP_KW)
        assert prod.cluster_name == "unit" and prod.n_clusters == spill_panel["unit"].nunique()
        assert prod.se != auto.se

    def test_survey_never_injects_the_auto_cluster(self, spill_panel):
        df = spill_panel.copy()
        codes = {u: i for i, u in enumerate(df["unit"].unique())}
        df["w"] = df["unit"].map(lambda u: 0.5 + (codes[u] % 7) / 7)
        auto = _fit_quiet(_spill(), df, survey_design=SurveyDesign(weights="w"), **SP_KW)
        off = _fit_quiet(
            _spill(cluster=False), df, survey_design=SurveyDesign(weights="w"), **SP_KW
        )
        explicit = _fit_quiet(
            _spill(cluster="unit"), df, survey_design=SurveyDesign(weights="w"), **SP_KW
        )
        # weights-only design: implicit per-observation PSUs under auto AND False
        assert auto.n_psu == len(df) == off.n_psu
        assert auto.survey_metadata.df_survey == len(df) - 1
        assert auto.cluster_name is None and auto.n_clusters is None
        assert auto.se == off.se
        # explicit cluster= still becomes the PSU
        assert explicit.n_psu == df["unit"].nunique() == explicit.n_clusters
        assert explicit.cluster_name == "unit"
        assert explicit.se != auto.se

    def test_point_estimates_never_move(self, default, off, explicit_unit):
        for a, b in ((default, off), (default, explicit_unit)):
            assert a.att == b.att
            pd.testing.assert_frame_equal(
                a.spillover_effects[["coef"]], b.spillover_effects[["coef"]]
            )


# --------------------------------------------------------------------------- #
# (3) The c1 off-switch
# --------------------------------------------------------------------------- #


class TestTWFEOffSwitch:
    def test_static_plain_hc1_full_k_parity(self, panel):
        """cluster=False on the within-transform TWFE = HC1 on the explicit
        unit + post dummy design (the absorbed-FE full-K rescale), residual-df t."""
        default = TwoWayFixedEffects().fit(panel, **TW_KW)
        off = TwoWayFixedEffects(cluster=False).fit(panel, **TW_KW)
        assert off.cluster_name is None and off.n_clusters is None
        assert off.att == default.att and off.se != default.se
        X = pd.concat(
            [
                panel[["d"]].astype(float),
                pd.get_dummies(panel["unit"], prefix="u", dtype=float),
                pd.get_dummies(panel["post"], prefix="p", drop_first=True, dtype=float),
            ],
            axis=1,
        )
        reg = LinearRegression(include_intercept=False).fit(X.values, panel["outcome"].values)
        inf = reg.get_inference(0)
        np.testing.assert_allclose(off.se, inf.se, rtol=1e-10)
        np.testing.assert_allclose(off.p_value, inf.p_value, rtol=1e-10)
        assert off.inference_df == pytest.approx(inf.df)
        n, k_full = len(panel), X.shape[1]
        assert off.inference_df == n - k_full

    @pytest.mark.parametrize("knob", ["cluster", "residual"])
    def test_static_knob_inert_without_a_cluster(self, panel, knob):
        off = TwoWayFixedEffects(cluster=False, df_convention=knob).fit(panel, **TW_KW)
        base = TwoWayFixedEffects(cluster=False).fit(panel, **TW_KW)
        assert _quintet(off) == _quintet(base)
        z = TwoWayFixedEffects(cluster=False, df_convention="normal").fit(panel, **TW_KW)
        assert z.inference_df is None and z.p_value != base.p_value

    def test_static_hc2_bm_one_way_lane(self, panel):
        default = TwoWayFixedEffects(vcov_type="hc2_bm").fit(panel, **TW_KW)
        off = TwoWayFixedEffects(cluster=False, vcov_type="hc2_bm").fit(panel, **TW_KW)
        assert off.cluster_name is None and off.n_clusters is None
        assert default.cluster_name == "unit"
        assert np.isfinite(off.se) and np.isfinite(off.p_value)
        assert off.inference_df is not None and off.inference_df != default.inference_df
        X = pd.concat(
            [
                panel[["d"]].astype(float),
                pd.get_dummies(panel["unit"], prefix="u", dtype=float),
                pd.get_dummies(panel["post"], prefix="p", drop_first=True, dtype=float),
            ],
            axis=1,
        )
        reg = LinearRegression(include_intercept=False, vcov_type="hc2_bm").fit(
            X.values, panel["outcome"].values
        )
        np.testing.assert_allclose(off.se, reg.get_inference(0).se, rtol=1e-10)

    def test_event_study_off_switch(self, panel):
        default = TwoWayFixedEffects().fit(panel, **ES_KW)
        off = TwoWayFixedEffects(cluster=False).fit(panel, **ES_KW)
        keep = ~off.is_reference
        np.testing.assert_allclose(off.att[keep], default.att[keep], rtol=1e-12)
        assert np.all(off.se[keep] != default.se[keep])
        # residual df on every row (knob inert), G-1 on the default
        assert np.all(off.df[keep] == off.df[keep][0]) and off.df[keep][0] > panel["unit"].nunique()
        assert np.all(default.df[keep] == panel["unit"].nunique() - 1)
        res = TwoWayFixedEffects(cluster=False, df_convention="residual").fit(panel, **ES_KW)
        np.testing.assert_array_equal(res.df[keep], off.df[keep])
        np.testing.assert_allclose(res.p_value[keep], off.p_value[keep], rtol=1e-12)

    def test_event_study_hc2_bm_one_way_lane_is_finite(self, panel):
        default = TwoWayFixedEffects(vcov_type="hc2_bm").fit(panel, **ES_KW)
        off = TwoWayFixedEffects(cluster=False, vcov_type="hc2_bm").fit(panel, **ES_KW)
        keep = ~off.is_reference
        assert np.all(np.isfinite(off.se[keep])) and np.all(np.isfinite(off.df[keep]))
        assert np.all(off.df[keep] != default.df[keep])

    def test_wild_bootstrap_rejects_false(self, panel):
        with pytest.raises(ValueError, match="requires a cluster"):
            TwoWayFixedEffects(cluster=False, inference="wild_bootstrap", n_bootstrap=9).fit(
                panel, **TW_KW
            )
        with pytest.raises(ValueError, match="requires cluster=<col>"):
            DifferenceInDifferences(cluster=False, inference="wild_bootstrap", n_bootstrap=9).fit(
                panel, outcome="outcome", treatment="grp", post="post"
            )

    def test_front_doors_keep_precedence_over_the_wild_reject(self, panel):
        est = TwoWayFixedEffects(cluster=False, inference="wild_bootstrap", n_bootstrap=9)
        with pytest.raises(NotImplementedError):
            est.fit(panel, survey_design=SurveyDesign(weights="w"), **TW_KW)

    def test_conley_plus_false_equals_conley_default(self, panel):
        kw = dict(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=20.0,
            conley_lag_cutoff=0,
        )
        a = TwoWayFixedEffects(**kw).fit(panel, **TW_KW)
        b = TwoWayFixedEffects(cluster=False, **kw).fit(panel, **TW_KW)
        assert _quintet(a) == _quintet(b)
        assert a.cluster_name is None and b.cluster_name is None
        assert a.n_clusters is None and b.n_clusters is None

    def test_survey_default_has_no_psu_injection(self, panel):
        r = TwoWayFixedEffects().fit(panel, survey_design=SurveyDesign(weights="w"), **TW_KW)
        assert r.survey_metadata.n_psu == len(panel)
        assert r.cluster_name is None and r.n_clusters is None  # 4.0: honest label
        off = TwoWayFixedEffects(cluster=False).fit(
            panel, survey_design=SurveyDesign(weights="w"), **TW_KW
        )
        assert _quintet(off) == _quintet(r)


class TestSunAbrahamOffSwitch:
    def test_plain_hc1_residual_df(self, panel):
        default = SunAbraham().fit(panel, **SA_KW)
        off = SunAbraham(cluster=False).fit(panel, **SA_KW)
        assert off.att == default.att and off.se != default.se
        assert off.cluster_name is None and off.n_clusters is None
        assert off.inference_df > panel["unit"].nunique()  # residual df, not G-1
        np.testing.assert_allclose(off.p_value, _t_tail(off.t_stat, off.inference_df), rtol=1e-12)
        res = SunAbraham(cluster=False, df_convention="residual").fit(panel, **SA_KW)
        assert _quintet(res) == _quintet(off)
        assert "HC1 heteroskedasticity-robust" in off.summary()

    def test_hc2_bm_rejected(self, panel):
        with pytest.raises(ValueError, match="Bell-McCaffrey DOF lane is cluster-only"):
            SunAbraham(cluster=False, vcov_type="hc2_bm").fit(panel, **SA_KW)

    def test_pairs_bootstrap_accepts_false_as_a_no_op(self, panel):
        a = SunAbraham(n_bootstrap=19, seed=1).fit(panel, **SA_KW)
        b = SunAbraham(cluster=False, n_bootstrap=19, seed=1).fit(panel, **SA_KW)
        assert a.se == b.se and a.p_value == b.p_value
        assert b.cluster_name is None and b.n_clusters is None
        assert a.cluster_name is None and a.n_clusters is None  # bootstrap: no analytical cluster
        assert "CR1 cluster-robust" not in a.summary()

    def test_survey_false_means_no_unit_psu_injection(self, panel):
        default = SunAbraham().fit(panel, survey_design=SurveyDesign(weights="w"), **SA_KW)
        off = SunAbraham(cluster=False).fit(panel, survey_design=SurveyDesign(weights="w"), **SA_KW)
        assert default.survey_metadata.n_psu == panel["unit"].nunique()
        assert off.survey_metadata.n_psu == len(panel)
        assert off.survey_metadata.df_survey == len(panel) - 1
        assert off.se != default.se
        assert default.cluster_name is None and default.n_clusters is None
        assert off.cluster_name is None and off.n_clusters is None
        assert "Std. errors:" not in default.summary()

    def test_conley_plus_false_equals_conley_default(self, panel):
        kw = dict(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=20.0,
            conley_lag_cutoff=0,
        )
        a = SunAbraham(**kw).fit(panel, **SA_KW)
        b = SunAbraham(cluster=False, **kw).fit(panel, **SA_KW)
        assert _quintet(a) == _quintet(b) and a.cluster_name is None and a.n_clusters is None


class TestWooldridgeOffSwitch:
    def test_ols_within_plain_hc1_cell_parity(self, panel):
        """The unclustered within lane carries the absorbed-FE full-K rescale, so
        each cohort-time cell SE equals HC1 on the explicit full-dummy design."""
        default = WooldridgeDiD().fit(panel, **SA_KW)
        off = WooldridgeDiD(cluster=False).fit(panel, **SA_KW)
        assert off.att == pytest.approx(default.att, rel=1e-12) and off.se != default.se
        assert off.cluster_name is None and off.n_clusters is None
        cells = sorted(off.group_time_effects)
        cell_cols = {
            (g, t): ((panel["first_treat"] == g) & (panel["time"] == t)).astype(float).values
            for (g, t) in cells
        }
        X = np.column_stack(
            [cell_cols[c] for c in cells]
            + [pd.get_dummies(panel["unit"], dtype=float).values]
            + [pd.get_dummies(panel["time"], drop_first=True, dtype=float).values]
        )
        reg = LinearRegression(include_intercept=False).fit(X, panel["outcome"].values)
        for i, c in enumerate(cells):
            np.testing.assert_allclose(
                off.group_time_effects[c]["se"], reg.get_inference(i).se, rtol=1e-9
            )

    def test_ols_knob_inert_without_a_cluster(self, panel):
        off = WooldridgeDiD(cluster=False).fit(panel, **SA_KW)
        res = WooldridgeDiD(cluster=False, df_convention="residual").fit(panel, **SA_KW)
        assert _quintet(off) == _quintet(res)
        z = WooldridgeDiD(cluster=False, df_convention="normal").fit(panel, **SA_KW)
        assert z.p_value != off.p_value
        np.testing.assert_allclose(z.p_value, 2 * stats.norm.sf(abs(z.t_stat)), rtol=1e-12)

    def test_ols_cohort_trends_full_dummy_lane(self, panel):
        default = WooldridgeDiD(cohort_trends=True).fit(panel, **SA_KW)
        off = WooldridgeDiD(cluster=False, cohort_trends=True).fit(panel, **SA_KW)
        assert off.se != default.se and off.cluster_name is None and off.n_clusters is None

    @pytest.mark.parametrize("method", ["logit", "poisson"])
    def test_glm_plain_qmle_sandwich_normal_theory(self, panel, method):
        kw = dict(outcome="yb", unit="unit", time="time", first_treat="first_treat")
        default = WooldridgeDiD(method=method).fit(panel, **kw)
        off = WooldridgeDiD(method=method, cluster=False).fit(panel, **kw)
        assert off.att == pytest.approx(default.att, rel=1e-10) and off.se != default.se
        assert off.cluster_name is None and off.n_clusters is None
        assert default.cluster_name == "unit" and default.n_clusters == panel["unit"].nunique()
        # GLM inference stays normal-theory (knob-independent): only the SE moved
        np.testing.assert_allclose(off.p_value, 2 * stats.norm.sf(abs(off.t_stat)), rtol=1e-12)

    def test_bootstrap_rejects_false(self, panel):
        with pytest.raises(ValueError, match="multiplier bootstrap draws one weight per cluster"):
            WooldridgeDiD(cluster=False, n_bootstrap=9).fit(panel, **SA_KW)

    def test_hc2_bm_rejects_false(self, panel):
        with pytest.raises(ValueError, match="Bell-McCaffrey DOF lane is cluster-only"):
            WooldridgeDiD(cluster=False, vcov_type="hc2_bm").fit(panel, **SA_KW)

    def test_conley_plus_false_equals_conley_default(self, panel):
        kw = dict(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=20.0,
            conley_lag_cutoff=0,
        )
        a = WooldridgeDiD(**kw).fit(panel, **SA_KW)
        b = WooldridgeDiD(cluster=False, **kw).fit(panel, **SA_KW)
        assert _quintet(a) == _quintet(b) and a.cluster_name is None

    def test_survey_default_has_no_psu_injection(self, panel):
        r = WooldridgeDiD().fit(panel, survey_design=SurveyDesign(weights="w"), **SA_KW)
        assert r.survey_metadata.n_psu == len(panel)
        assert r.cluster_name is None and r.n_clusters is None

    def test_saturated_within_full_k_df_voids_inference(self):
        """Full-K residual df <= 0 on the unclustered within lane -> all-NaN
        inference (the helper's NaN contract), never a finite under-counted SE."""
        # 2 units x 2 periods, one treated cell: n = 4, k = 1 cell dummy, absorbed unit +
        # time rank = 3 -> full-K residual df = 0.
        rows = []
        for u, ft in enumerate([0, 2]):
            for t in (1, 2):
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        first_treat=ft,
                        outcome=float(0.7 * u + t + (ft and t >= ft)),
                    )
                )
        df = pd.DataFrame(rows)
        r = _fit_quiet(WooldridgeDiD(cluster=False), df, **SA_KW)
        assert np.isfinite(r.att)
        assert np.isnan(r.se) and np.isnan(r.t_stat) and np.isnan(r.p_value)
        assert np.isnan(r.conf_int[0]) and np.isnan(r.conf_int[1])


class TestLPDiDOffSwitch:
    def test_ols_lane(self, panel):
        default = LPDiD(**LP_CTOR).fit(panel, **LP_KW)
        off = LPDiD(cluster=False, **LP_CTOR).fit(panel, **LP_KW)
        assert off.att == default.att and off.se != default.se
        assert off.cluster_name is None and off.n_clusters is None
        assert off.vcov_type == "hc1"
        assert off.event_study["n_clusters"].isna().all()
        assert default.event_study["n_clusters"].notna().any()
        # residual df per horizon (knob inert), G-1 on the default
        assert all(v > panel["unit"].nunique() for v in off.event_study_df.values())
        res = LPDiD(cluster=False, df_convention="residual", **LP_CTOR).fit(panel, **LP_KW)
        assert _quintet(res) == _quintet(off)
        assert "HC1 heteroskedasticity-robust" in off.summary()
        assert off.to_dict()["inference_method"] == "heteroskedasticity_robust"

    def test_ra_lane_unclustered_if_variance(self, panel):
        default = LPDiD(reweight=True, **LP_CTOR).fit(panel, covariates=["x"], **LP_KW)
        off = LPDiD(cluster=False, reweight=True, **LP_CTOR).fit(panel, covariates=["x"], **LP_KW)
        assert off.vcov_type == "if_robust" and default.vcov_type == "if_cluster"
        assert off.cluster_name is None and off.n_clusters is None
        assert off.se != default.se
        # Singleton clusters make sum_g s_g^2 == sum_i phi_i^2 and neither lane
        # carries a finite-sample factor: the SEs agree EXACTLY, the p/CI differ
        # (t(G-1 = n-1) vs the residual df).
        single = LPDiD(cluster="row", reweight=True, **LP_CTOR).fit(
            panel, covariates=["x"], **LP_KW
        )
        np.testing.assert_allclose(off.se, single.se, rtol=1e-14)
        assert off.p_value != single.p_value
        assert "Influence-function heteroskedasticity-robust (unclustered)" in off.summary()

    def test_survey_false_implicit_observation_psus(self, panel):
        default = LPDiD(**LP_CTOR).fit(panel, survey_design=SurveyDesign(weights="w"), **LP_KW)
        off = LPDiD(cluster=False, **LP_CTOR).fit(
            panel, survey_design=SurveyDesign(weights="w"), **LP_KW
        )
        assert default.n_psu == panel["unit"].nunique() == default.n_clusters
        assert off.n_psu == len(panel)
        assert off.cluster_name is None and off.n_clusters is None
        assert off.se != default.se


# --------------------------------------------------------------------------- #
# (4) The b / c2 / n-a no-op
# --------------------------------------------------------------------------- #


NOOP = [
    "CallawaySantAnna",
    "EfficientDiD",
    "DMLDiD",
    "ChaisemartinDHaultfoeuille",
    "LWDiD",
    "HeterogeneousAdoptionDiD",
    "ImputationDiD",
    "TwoStageDiD",
    "DifferenceInDifferences",
    "TripleDifference",
]


@pytest.mark.parametrize("name", NOOP)
def test_false_is_a_bit_identical_no_op(name, recipes):
    ctor, data, fit_kw = recipes[name]
    cls = getattr(diff_diff, name)
    a = _fit_quiet(cls(**ctor), data, **fit_kw)
    b = _fit_quiet(cls(cluster=False, **ctor), data, **fit_kw)
    assert _quintet(a) == _quintet(b)
    assert _meta(a) == _meta(b)


def test_false_no_op_on_the_staggered_ddd_mode():
    stag = generate_staggered_ddd_data(n_units=96, n_periods=6, cohort_periods=[3, 4], seed=42)
    kw = dict(
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
        partition="eligibility",
    )
    a = _fit_quiet(TripleDifference(n_bootstrap=19, seed=1), stag, **kw)
    b = _fit_quiet(TripleDifference(cluster=False, n_bootstrap=19, seed=1), stag, **kw)
    assert _quintet(a) == _quintet(b) and _meta(a) == _meta(b)
    with pytest.raises(ValueError, match="not supported in staggered DDD mode"):
        _fit_quiet(TripleDifference(cluster="unit", n_bootstrap=19, seed=1), stag, **kw)


def test_false_no_op_on_the_cs_rc_auto_psu_path(panel):
    unbalanced = panel.drop(panel.index[::17])
    kw = dict(allow_unbalanced_panel=True)
    a = _fit_quiet(CallawaySantAnna(**kw), unbalanced, **SA_KW)
    b = _fit_quiet(CallawaySantAnna(cluster=False, **kw), unbalanced, **SA_KW)
    assert a.cluster_name == "unit" == b.cluster_name  # the RC unit PSU survives False
    assert _quintet(a) == _quintet(b) and a.n_clusters == b.n_clusters


def test_hausman_pretest_accepts_false(panel):
    a = EfficientDiD.hausman_pretest(panel, **SA_KW)
    b = EfficientDiD.hausman_pretest(panel, cluster=False, **SA_KW)
    assert a.statistic == b.statistic and a.p_value == b.p_value


def test_lwdid_sensitivity_helpers_accept_false():
    df = _make_common_timing_panel()
    kw = dict(outcome="y", unit="unit", time="time", treatment="treat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = robustness_pre_periods(df, **kw)
        b = robustness_pre_periods(df, cluster=False, **kw)
        c = sensitivity_no_anticipation(df, **kw)
        d = sensitivity_no_anticipation(df, cluster=False, **kw)
    pd.testing.assert_frame_equal(a.to_dataframe(), b.to_dataframe())
    pd.testing.assert_frame_equal(c.to_dataframe(), d.to_dataframe())
    with pytest.raises(ValueError, match=CONTRACT_MSG):
        robustness_pre_periods(df, cluster=True, **kw)


def test_stacked_did_still_rejects_false():
    with pytest.raises(ValueError, match="cluster must be 'unit' or 'unit_subexp'"):
        StackedDiD(cluster=False)


# --------------------------------------------------------------------------- #
# (5) Cluster metadata
# --------------------------------------------------------------------------- #


class TestClusterMetadata:
    def test_did_analytical_n_clusters(self, panel):
        G = panel["unit"].nunique()
        r = DifferenceInDifferences(cluster="unit").fit(
            panel, outcome="outcome", treatment="grp", post="post"
        )
        assert (r.cluster_name, r.n_clusters) == ("unit", G)
        assert r.inference_df == G - 1
        plain = DifferenceInDifferences().fit(
            panel, outcome="outcome", treatment="grp", post="post"
        )
        assert plain.n_clusters is None and plain.cluster_name is None

    def test_did_explicit_cluster_under_psu_survey_keeps_none_count(self, panel):
        df = panel.copy()
        df["psu"] = df["unit"] // 2
        r = DifferenceInDifferences(cluster="unit").fit(
            df,
            outcome="outcome",
            treatment="grp",
            post="post",
            survey_design=SurveyDesign(weights="w", psu="psu"),
        )
        assert r.cluster_name == "unit" and r.n_clusters is None

    def test_twfe_metadata_lanes(self, panel):
        G = panel["unit"].nunique()
        auto = TwoWayFixedEffects().fit(panel, **TW_KW)
        assert (auto.cluster_name, auto.n_clusters) == ("unit", G)
        assert auto.summary().count(f"G={G}") == 1
        wild = TwoWayFixedEffects(inference="wild_bootstrap", n_bootstrap=9, seed=1).fit(
            panel, **TW_KW
        )
        assert wild.n_clusters == G and wild.inference_method == "wild_bootstrap"
        kw = dict(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=20.0,
            conley_lag_cutoff=0,
        )
        c_auto = TwoWayFixedEffects(**kw).fit(panel, **TW_KW)
        assert (c_auto.cluster_name, c_auto.n_clusters) == (None, None)
        panel2 = panel.assign(region=panel["unit"] % 5)
        c_exp = TwoWayFixedEffects(cluster="region", **kw).fit(panel2, **TW_KW)
        assert (c_exp.cluster_name, c_exp.n_clusters) == ("region", 5)
        one_way = TwoWayFixedEffects(vcov_type="hc2").fit(panel, **TW_KW)
        assert (one_way.cluster_name, one_way.n_clusters) == (None, None)

    def test_sun_abraham_metadata_lanes(self, panel):
        G = panel["unit"].nunique()
        auto = SunAbraham().fit(panel, **SA_KW)
        assert (auto.cluster_name, auto.n_clusters) == ("unit", G)
        assert f"CR1 cluster-robust at unit, G={G}" in auto.summary()
        d = auto.to_dict()
        assert d["cluster_name"] == "unit" and d["n_clusters"] == G
        panel2 = panel.assign(state=panel["unit"] % 5)
        exp = SunAbraham(cluster="state").fit(panel2, **SA_KW)
        assert (exp.cluster_name, exp.n_clusters) == ("state", 5)
        off = SunAbraham(cluster=False).fit(panel, **SA_KW)
        assert (off.cluster_name, off.n_clusters) == (None, None)
        assert "n_clusters" not in off.to_dict()
        one_way = SunAbraham(vcov_type="hc2").fit(panel, **SA_KW)
        assert (one_way.cluster_name, one_way.n_clusters) == (None, None)
        kw = dict(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=20.0,
            conley_lag_cutoff=0,
        )
        c_exp = SunAbraham(cluster="state", **kw).fit(panel2, **SA_KW)
        assert (c_exp.cluster_name, c_exp.n_clusters) == ("state", 5)
        back = pickle.loads(pickle.dumps(auto))
        assert back.n_clusters == G and back.cluster_name == "unit"

    def test_sun_abraham_results_field_is_appended_last(self):
        import dataclasses

        from diff_diff.sun_abraham import SunAbrahamResults

        names = [f.name for f in dataclasses.fields(SunAbrahamResults)]
        assert names[-1] == "n_clusters" and names[-2] == "inference_df"


# --------------------------------------------------------------------------- #
# (6) Ledger
# --------------------------------------------------------------------------- #


def test_ledger_rows_are_done_and_point_here():
    rows = {r["id"]: r for r in LEDGER_ROWS}
    for rid in ("M-080", "M-155"):
        row = rows[rid]
        assert row["kind"] == "behavior", rid
        assert row["status"] == "done", rid
        assert row["test_ref"] == THIS_TEST_REF, rid
    assert rows["M-080"]["old"] == "diff_diff:SpilloverDiD[cluster]"
    assert rows["M-155"]["old"] == "diff_diff.utils:validate_cluster"
    assert rows["M-154"]["status"] == "done"  # 7c shipped the SpilloverDiD df_convention knob
    assert rows["M-154"]["old"] == "diff_diff:SpilloverDiD[df_convention]"
