"""Item 7a of the 4.0 cut: the ``df_convention`` default flip.

Ledger rows M-004 (DifferenceInDifferences), M-005 (TwoWayFixedEffects),
M-006 (LinearRegression), M-128 (SunAbraham), M-129 (WooldridgeDiD), M-130
(StackedDiD), M-131 (ImputationDiD): the analytical tail-df convention for
CLUSTERED fits flipped from ``"residual"`` (the fit's ``n - K_full``) to
``"cluster"`` (the Stata/fixest ``G - 1``). LPDiD adopted ``"cluster"`` at 3.9
and has no row.

What moves: p-values and confidence intervals on clustered analytical fits
(TWFE / SunAbraham / WooldridgeDiD-OLS / StackedDiD / ImputationDiD-pretrends
cluster at unit BY DEFAULT, so their default fits move; plain
DifferenceInDifferences moves only with an explicit ``cluster=``). What does
not: point estimates, SEs and t-statistics; survey/replicate df and hc2_bm
Bell-McCaffrey DOF (they keep precedence); bootstrap percentile p/CIs;
unclustered and Conley fits (the knob is inert without a cluster count).
``df_convention="residual"`` restores the 3.x inference bit-for-bit.

Every G-1 pin below takes G from the surface's own clustered sample:
``data[unit].nunique()`` for DiD / TWFE / SunAbraham (their results carry no
analytical ``n_clusters``), ``res.n_clusters`` for WooldridgeDiD and
StackedDiD, and the lead regression's untreated sample for the ImputationDiD
leads (every unit of the shared panel has untreated rows, so that count is
the panel's unit count too).
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from diff_diff import (
    DifferenceInDifferences,
    ImputationDiD,
    LPDiD,
    StackedDiD,
    SunAbraham,
    SurveyDesign,
    SyntheticDiD,
    TwoWayFixedEffects,
    WooldridgeDiD,
    generate_staggered_data,
)
from diff_diff.linalg import LinearRegression

# The ledger is restricted YAML; the matrix gate parses it without pyyaml (not a
# test dependency) and exposes the parsed rows.
from tests.test_v4_matrix import ROWS as LEDGER_ROWS

THIS_TEST_REF = "tests/test_v4_inference_defaults.py"

FLIP_ROWS = {
    "M-004": DifferenceInDifferences,
    "M-005": TwoWayFixedEffects,
    "M-006": LinearRegression,
    "M-128": SunAbraham,
    "M-129": WooldridgeDiD,
    "M-130": StackedDiD,
    "M-131": ImputationDiD,
}
# LinearRegression is a plain class (no BaseEstimator mixin): no get/set_params.
BASE_ESTIMATOR_KNOB_CLASSES = [
    DifferenceInDifferences,
    TwoWayFixedEffects,
    SunAbraham,
    WooldridgeDiD,
    StackedDiD,
    ImputationDiD,
    LPDiD,
]


def t_tail(t_stat, df):
    return 2 * stats.t.sf(abs(t_stat), df)


def z_tail(t_stat):
    return 2 * stats.norm.sf(abs(t_stat))


@pytest.fixture(scope="module")
def panel():
    """60 units x 8 periods, cohorts {never, 4, 6}; every unit has untreated
    rows (treatment starts at period 4 at the earliest)."""
    rng = np.random.default_rng(7)
    rows = []
    for u in range(60):
        ft = [0, 4, 6][u % 3]
        lat, lon = 40.0 + 0.05 * (u // 6), -100.0 + 0.05 * (u % 6)
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
                )
            )
    df = pd.DataFrame(rows)
    df["d"] = df["grp"] * df["post"]
    return df


SA_KW = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")


# --------------------------------------------------------------------------- #
# (1) Signature / params pins
# --------------------------------------------------------------------------- #


class TestSignatureDefaults:
    @pytest.mark.parametrize("cls", list(FLIP_ROWS.values()) + [LPDiD], ids=lambda c: c.__name__)
    def test_signature_default_is_cluster(self, cls):
        default = inspect.signature(cls.__init__).parameters["df_convention"].default
        assert default == "cluster"

    @pytest.mark.parametrize("cls", BASE_ESTIMATOR_KNOB_CLASSES, ids=lambda c: c.__name__)
    def test_get_params_reads_cluster_and_set_params_round_trips(self, cls):
        est = cls()
        assert est.get_params()["df_convention"] == "cluster"
        for value in ("residual", "normal", "cluster"):
            est.set_params(df_convention=value)
            assert est.get_params()["df_convention"] == value
            assert cls(**est.get_params()).df_convention == value

    def test_synthetic_did_inherits_the_flip_inertly(self):
        """SyntheticDiD subclasses DifferenceInDifferences and never reads the
        knob (placebo / bootstrap / jackknife variance); the inherited
        attribute flips with the base class and is NOT a SyntheticDiD
        parameter."""
        sdid = SyntheticDiD()
        assert sdid.df_convention == "cluster"
        assert "df_convention" not in sdid.get_params()


# --------------------------------------------------------------------------- #
# (2) Default fits sit on t(G-1); "residual" restores the 3.x tail
# --------------------------------------------------------------------------- #


class TestDefaultFitsUseClusterDf:
    def test_did_explicit_cluster(self, panel):
        G = panel["unit"].nunique()
        kw = dict(outcome="outcome", treatment="grp", post="post", unit="unit")
        res = DifferenceInDifferences(cluster="unit").fit(panel, **kw)
        assert res.inference_df == G - 1
        np.testing.assert_allclose(res.p_value, t_tail(res.t_stat, G - 1), rtol=1e-12)
        rr = DifferenceInDifferences(cluster="unit", df_convention="residual").fit(panel, **kw)
        assert rr.inference_df > G - 1  # n - K_full
        assert (rr.att, rr.se, rr.t_stat) == (res.att, res.se, res.t_stat)
        assert rr.p_value != res.p_value
        np.testing.assert_allclose(rr.p_value, t_tail(rr.t_stat, rr.inference_df), rtol=1e-12)

    def test_twfe_static_auto_cluster(self, panel):
        G = panel["unit"].nunique()
        kw = dict(outcome="outcome", treatment="d", post="post", unit="unit")
        res = TwoWayFixedEffects().fit(panel, **kw)
        assert res.inference_df == G - 1
        np.testing.assert_allclose(res.p_value, t_tail(res.t_stat, G - 1), rtol=1e-12)
        rr = TwoWayFixedEffects(df_convention="residual").fit(panel, **kw)
        assert rr.se == res.se and rr.t_stat == res.t_stat and rr.p_value != res.p_value

    @pytest.mark.parametrize("spec", ["within", "pooled"])
    def test_twfe_event_study_auto_cluster(self, panel, spec):
        """Both specs auto-cluster at unit when ``unit=`` is passed (a pooled
        fit WITHOUT ``unit=`` is unclustered and the knob is inert there -
        that is the committed ``pooled_default`` oracle lane)."""
        G = panel["unit"].nunique()
        kw = dict(
            outcome="outcome",
            treatment="grp",
            unit="unit",
            time="time",
            event_study=True,
            spec=spec,
            post_periods=[4, 5, 6, 7, 8],
            reference_period=3,
        )
        res = TwoWayFixedEffects().fit(panel, **kw)
        est = ~res.is_reference
        assert est.sum() == 7  # periods 1, 2 (pre) + 4..8 (post)
        np.testing.assert_array_equal(res.df[est], float(G - 1))
        np.testing.assert_allclose(
            res.p_value[est], t_tail(res.att[est] / res.se[est], G - 1), rtol=1e-12
        )
        rr = TwoWayFixedEffects(df_convention="residual").fit(panel, **kw)
        np.testing.assert_array_equal(rr.se, res.se)
        assert (rr.df[est] > G - 1).all()
        assert (rr.p_value[est] != res.p_value[est]).all()

    def test_sun_abraham(self, panel):
        G = panel["unit"].nunique()
        res = SunAbraham().fit(panel, **SA_KW)
        assert res.inference_df == G - 1
        np.testing.assert_allclose(res.p_value, t_tail(res.t_stat, G - 1), rtol=1e-12)
        assert set(res.event_study_df.values()) == {float(G - 1)}
        # every event-study row (the cohort x event-time cells keep only
        # coefficient/SE on the public surface)
        for eff in res.event_study_effects.values():
            np.testing.assert_allclose(eff["p_value"], t_tail(eff["t_stat"], G - 1), rtol=1e-12)
        rr = SunAbraham(df_convention="residual").fit(panel, **SA_KW)
        assert rr.inference_df > G - 1 and rr.se == res.se and rr.p_value != res.p_value

    def test_wooldridge_ols(self, panel):
        res = WooldridgeDiD(method="ols").fit(panel, **SA_KW)
        G = res.n_clusters
        assert G == panel["unit"].nunique()
        np.testing.assert_allclose(res.p_value, t_tail(res.t_stat, G - 1), rtol=1e-12)
        for eff in res.group_time_effects.values():
            np.testing.assert_allclose(eff["p_value"], t_tail(eff["t_stat"], G - 1), rtol=1e-12)
        rr = WooldridgeDiD(method="ols", df_convention="residual").fit(panel, **SA_KW)
        assert rr.se == res.se and rr.p_value != res.p_value
        assert rr.p_value != t_tail(rr.t_stat, G - 1)

    def test_wooldridge_bootstrap_fit_keeps_analytical_lanes_on_g_minus_1(self, panel):
        """Only the overall p/CI are percentile-overridden on a bootstrap fit;
        post-fit aggregate('group') re-derives ANALYTICAL inference from the
        stored fallback, which now carries G-1."""
        res = WooldridgeDiD(method="ols", n_bootstrap=19, seed=1).fit(panel, **SA_KW)
        G = res.n_clusters
        for ge in res.aggregate("group").group_effects.values():
            np.testing.assert_allclose(ge["p_value"], t_tail(ge["t_stat"], G - 1), rtol=1e-12)

    def test_wooldridge_glm_inert_config_warning_tracks_the_new_default(self, panel):
        data = panel.copy()
        data["bin"] = (data["outcome"] > data["outcome"].median()).astype(int)
        kw = dict(outcome="bin", unit="unit", time="time", first_treat="first_treat")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WooldridgeDiD(method="logit").fit(data, **kw)
        assert not any("no effect on the logit/poisson" in str(w.message) for w in caught)
        with pytest.warns(UserWarning, match="no effect on the logit/poisson"):
            WooldridgeDiD(method="logit", df_convention="residual").fit(data, **kw)

    def test_stacked_did(self):
        data = generate_staggered_data(n_units=60, n_periods=10, seed=9)
        kw = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
        res = StackedDiD(kappa_pre=2, kappa_post=2).fit(data, **kw)
        G = res.n_clusters
        assert res.inference_df == G - 1
        np.testing.assert_allclose(res.p_value, t_tail(res.t_stat, G - 1), rtol=1e-12)
        assert set(res.event_study_df.values()) == {float(G - 1)}
        surface = res.aggregate("event_study")
        rows = ~surface.is_reference & np.isfinite(surface.p_value)
        assert rows.any()
        np.testing.assert_array_equal(surface.df[rows], float(G - 1))
        rr = StackedDiD(kappa_pre=2, kappa_post=2, df_convention="residual").fit(data, **kw)
        assert rr.inference_df > G - 1 and rr.se == res.se

    def test_imputation_pretrends_leads(self, panel):
        """The lead regression clusters on the untreated sample; every unit of
        the panel has untreated rows, so G_lead is the unit count. The
        surface's ``df`` column is NaN on plain fits by design (it carries the
        survey df only), so the pin goes through the p-value."""
        G_lead = panel.loc[
            panel["time"] < panel["first_treat"].replace(0, np.inf), "unit"
        ].nunique()
        assert G_lead == panel["unit"].nunique()
        res = ImputationDiD(pretrends=True).fit(panel, **SA_KW)
        es = res.aggregate("event_study")
        leads = (es.event_time < -1) & np.isfinite(es.p_value)
        assert leads.sum() >= 2
        np.testing.assert_allclose(
            es.p_value[leads], t_tail(es.att[leads] / es.se[leads], G_lead - 1), rtol=1e-12
        )
        assert np.isnan(np.asarray(es.df, dtype=float)[leads]).all()
        rr = ImputationDiD(pretrends=True, df_convention="residual").fit(panel, **SA_KW)
        es_r = rr.aggregate("event_study")
        np.testing.assert_array_equal(es_r.se[leads], es.se[leads])
        assert (es_r.p_value[leads] != es.p_value[leads]).all()
        # post rows are BJS (knob-independent)
        post = es.event_time >= 0
        np.testing.assert_array_equal(es_r.p_value[post], es.p_value[post])

    def test_imputation_bootstrap_fit_keeps_analytical_leads_on_g_minus_1(self, panel):
        G_lead = panel["unit"].nunique()
        res = ImputationDiD(pretrends=True, n_bootstrap=19, seed=1).fit(panel, **SA_KW)
        es = res.aggregate("event_study")
        leads = (es.event_time < -1) & np.isfinite(es.p_value)
        np.testing.assert_allclose(
            es.p_value[leads], t_tail(es.att[leads] / es.se[leads], G_lead - 1), rtol=1e-12
        )

    def test_imputation_inert_config_warning_tracks_the_new_default(self, panel):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ImputationDiD().fit(panel, **SA_KW)
        assert not any("affects only the pretrends" in str(w.message) for w in caught)
        with pytest.warns(UserWarning, match="affects only the pretrends"):
            ImputationDiD(df_convention="residual").fit(panel, **SA_KW)

    def test_linear_regression_direct(self):
        rng = np.random.default_rng(3)
        n, G = 120, 12
        X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
        y = X @ np.array([1.0, 0.4, 2.0]) + rng.normal(size=n)
        cl = np.repeat(np.arange(G), n // G)
        reg = LinearRegression(cluster_ids=cl, include_intercept=False).fit(X, y)
        inf = reg.get_inference(1)
        assert inf.df == G - 1
        np.testing.assert_allclose(inf.p_value, t_tail(inf.t_stat, G - 1), rtol=1e-12)
        reg_r = LinearRegression(
            cluster_ids=cl, include_intercept=False, df_convention="residual"
        ).fit(X, y)
        inf_r = reg_r.get_inference(1)
        assert inf_r.df == n - 3 and inf_r.se == inf.se and inf_r.p_value != inf.p_value


# --------------------------------------------------------------------------- #
# (3) Where the knob is inert, nothing moved
# --------------------------------------------------------------------------- #


class TestInertSurfaces:
    def test_unclustered_did_is_bit_identical(self, panel):
        kw = dict(outcome="outcome", treatment="grp", post="post")
        r0 = DifferenceInDifferences().fit(panel, **kw)
        rr = DifferenceInDifferences(df_convention="residual").fit(panel, **kw)
        assert (r0.p_value, r0.conf_int, r0.inference_df) == (
            rr.p_value,
            rr.conf_int,
            rr.inference_df,
        )

    def test_conley_twfe_is_bit_identical(self, panel):
        kw = dict(outcome="outcome", treatment="d", post="post", unit="unit")
        common = dict(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=50.0,
            conley_lag_cutoff=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r0 = TwoWayFixedEffects(**common).fit(panel, **kw)
            rr = TwoWayFixedEffects(**common, df_convention="residual").fit(panel, **kw)
        assert (r0.p_value, r0.conf_int) == (rr.p_value, rr.conf_int)

    def test_survey_df_keeps_precedence(self, panel):
        data = panel.copy()
        data["w"] = 1.0 + (data["unit"] % 5) * 0.2
        data["stratum"] = data["unit"] % 4
        design = SurveyDesign(weights="w", strata="stratum", psu="unit")
        kw = dict(outcome="outcome", treatment="grp", post="post", unit="unit")
        r0 = DifferenceInDifferences().fit(data, survey_design=design, **kw)
        rr = DifferenceInDifferences(df_convention="residual").fit(data, survey_design=design, **kw)
        assert (r0.p_value, r0.conf_int) == (rr.p_value, rr.conf_int)
        assert r0.inference_df is not None and r0.inference_df != panel["unit"].nunique() - 1

    def test_normal_is_deliberate_z(self, panel):
        kw = dict(outcome="outcome", treatment="grp", post="post", unit="unit")
        r0 = DifferenceInDifferences(cluster="unit").fit(panel, **kw)
        rn = DifferenceInDifferences(cluster="unit", df_convention="normal").fit(panel, **kw)
        assert rn.se == r0.se and rn.p_value != r0.p_value
        np.testing.assert_allclose(rn.p_value, z_tail(rn.t_stat), rtol=1e-14)

    def test_one_effective_cluster_still_raises_at_the_vcov_layer(self):
        """Unchanged from 3.x: a weighted CR1 fit with ONE positive-weight
        cluster raises before any df is resolved (under every convention)."""
        rng = np.random.default_rng(5)
        n = 60
        X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
        y = X @ np.array([1.0, 0.5, 2.0]) + rng.normal(size=n)
        cl = np.repeat(np.arange(3), 20)
        w = np.where(cl == 0, 2.0, 0.0)
        with pytest.raises(ValueError, match="at least 2 clusters"):
            LinearRegression(cluster_ids=cl, include_intercept=False, weights=w).fit(X, y)


# --------------------------------------------------------------------------- #
# (4) Ledger
# --------------------------------------------------------------------------- #


def test_ledger_rows_are_done_and_point_here():
    rows = {r["id"]: r for r in LEDGER_ROWS}
    for rid, cls in FLIP_ROWS.items():
        row = rows[rid]
        assert row["kind"] == "default-flip"
        assert row["status"] == "done", rid
        assert row["test_ref"] == THIS_TEST_REF, rid
        assert row["new_default"] == "'cluster'"
        assert row["old"].endswith(f":{cls.__name__}[df_convention]"), (rid, row["old"])
