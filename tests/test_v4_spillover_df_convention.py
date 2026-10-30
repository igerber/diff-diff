"""Item 7c of the 4.0 cut: the SpilloverDiD ``df_convention`` knob (ledger row M-154).

7b made SpilloverDiD auto-cluster at ``unit`` but left its inference on the OLS
residual df ``t(n_obs_eff - k_effective)``; 7c adds the shared three-value knob
with the 4.0 default ``"cluster"`` so the clustered lanes (the unit auto-cluster
and an explicit ``cluster=``) run the Stata/fixest ``t(G - 1)`` like every other
auto-clustered default. What moves: p-values and CIs on clustered fits. What does
not: point estimates, SEs and t-statistics; ``cluster=False`` and EVERY Conley fit
(G is forced to None, the SunAbraham precedent - an explicit product-kernel
``cluster=`` included); the survey ``df_survey`` (precedence under every value);
the saturated-sample all-NaN contract. ``df_convention="residual"`` restores the
3.x / 7b inference bit-for-bit.

The df ladder is the pure helper ``diff_diff.spillover._resolve_inference_df``
(survey df -> saturation -> ``utils.resolve_tail_df``); its saturation branch is
pinned here directly because a saturated SpilloverDiD FIT is unreachable through
the balanced-panel + far-away validation (no test constructs one).

The ledger import lives inside its test: ``tests.test_v4_matrix`` skips at import
when ``docs/v4-deprecations.yaml`` is absent (the wheel-install CI legs copy only
``tests/``), and the behavioral pins must still run there.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
from scipy import stats

from diff_diff import SpilloverDiD, SurveyDesign
from diff_diff.spillover import _resolve_inference_df
from diff_diff.utils import safe_inference
from tests._dgp_utils import generate_butts_staggered_dgp
from tests.test_t23_spillover_tva_drift import (
    D_BAR_KM,
    _build_t23_panel,
    _silence_spillover_matmul_warnings,
)

THIS_TEST_REF = "tests/test_v4_spillover_df_convention.py"
SP_KW = dict(outcome="y", unit="unit", time="time", treatment="D")
N_RINGS = 1  # rings=[0, D_BAR_KM] -> one ring coefficient; k_effective = 1 + N_RINGS


def _t_tail(t, df):
    return 2.0 * stats.t.sf(np.abs(t), df)


def _z_tail(t):
    return 2.0 * stats.norm.sf(np.abs(t))


def _fit(panel, *, survey_design=None, **kwargs):
    est = SpilloverDiD(rings=[0.0, D_BAR_KM], conley_coords=("lat", "lon"), **kwargs)
    with warnings.catch_warnings():
        _silence_spillover_matmul_warnings()
        warnings.simplefilter("ignore", UserWarning)
        if survey_design is None:
            return est.fit(panel, **SP_KW)
        return est.fit(panel, survey_design=survey_design, **SP_KW)


CONLEY_KW = dict(vcov_type="conley", conley_cutoff_km=D_BAR_KM, conley_lag_cutoff=0)


@pytest.fixture(scope="module")
def panel():
    df = _build_t23_panel()
    idx = df["unit"].str[1:].astype(int)
    df["blk"] = idx % 10  # a unit-constant coarser cluster: 10 labels
    df["w"] = 1.0 + (idx % 5) * 0.2  # unit-constant survey weights
    df["stratum"] = idx % 4
    return df


@pytest.fixture(scope="module")
def default(panel):
    return _fit(panel)


@pytest.fixture(scope="module")
def residual(panel):
    return _fit(panel, df_convention="residual")


# --------------------------------------------------------------------------- #
# (1) Signature / validation
# --------------------------------------------------------------------------- #


class TestKnobSurface:
    def test_signature_default_is_cluster(self):
        assert inspect.signature(SpilloverDiD).parameters["df_convention"].default == "cluster"

    def test_get_set_params_round_trip(self):
        est = SpilloverDiD(rings=[0.0, 50.0])
        assert est.get_params()["df_convention"] == "cluster"
        est.set_params(df_convention="residual")
        assert est.df_convention == "residual" == est.get_params()["df_convention"]

    def test_constructor_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="df_convention must be one of"):
            SpilloverDiD(rings=[0.0, 50.0], df_convention="bogus")

    def test_set_params_is_transactional(self):
        est = SpilloverDiD(rings=[0.0, 50.0])
        with pytest.raises(ValueError, match="df_convention must be one of"):
            est.set_params(df_convention="bogus")
        assert est.df_convention == "cluster"

    def test_direct_mutation_rejected_at_fit(self, panel):
        est = SpilloverDiD(rings=[0.0, D_BAR_KM], conley_coords=("lat", "lon"))
        est.df_convention = "bogus"
        with pytest.raises(ValueError, match="df_convention must be one of"):
            est.fit(panel, **SP_KW)


# --------------------------------------------------------------------------- #
# (2)-(4) The three values on the default (CR1 at unit) lane
# --------------------------------------------------------------------------- #


class TestDefaultLane:
    def test_default_is_t_g_minus_1(self, default, panel):
        G = panel["unit"].nunique()
        assert default.cluster_name == "unit" and default.n_clusters == G
        assert default.df_convention == "cluster"
        assert default.inference_df == G - 1 == default.n_clusters - 1
        np.testing.assert_allclose(default.p_value, _t_tail(default.t_stat, G - 1), rtol=1e-12)
        lo, hi = default.conf_int
        np.testing.assert_allclose(
            (hi - lo) / 2, stats.t.ppf(1 - default.alpha / 2, G - 1) * default.se, rtol=1e-12
        )

    def test_ring_rows_share_the_df(self, default, panel):
        G = panel["unit"].nunique()
        eff = default.spillover_effects
        assert eff is not None and len(eff) == N_RINGS
        for _, row in eff.iterrows():
            np.testing.assert_allclose(row["p_value"], _t_tail(row["t_stat"], G - 1), rtol=1e-12)
            np.testing.assert_allclose(
                (row["ci_high"] - row["ci_low"]) / 2,
                stats.t.ppf(1 - default.alpha / 2, G - 1) * row["se"],
                rtol=1e-12,
            )

    def test_explicit_unit_cluster_is_bit_equal(self, default, panel):
        explicit = _fit(panel, cluster="unit")
        assert (explicit.p_value, explicit.conf_int, explicit.inference_df) == (
            default.p_value,
            default.conf_int,
            default.inference_df,
        )

    def test_to_dict_carries_both_fields(self, default, panel):
        d = default.to_dict()
        assert d["df_convention"] == "cluster"
        assert d["inference_df"] == panel["unit"].nunique() - 1

    def test_explicit_coarser_cluster_uses_its_own_g(self, panel):
        """G comes from the effective cluster labels, not the unit count."""
        blk = _fit(panel, cluster="blk")
        assert blk.cluster_name == "blk" and blk.n_clusters == 10
        assert blk.inference_df == 9.0
        np.testing.assert_allclose(blk.p_value, _t_tail(blk.t_stat, 9), rtol=1e-12)
        blk_res = _fit(panel, cluster="blk", df_convention="residual")
        assert blk_res.se == blk.se
        assert blk_res.inference_df == blk.n_obs - (1 + N_RINGS)

    def test_residual_restores_the_7b_inference(self, default, residual):
        k = 1 + N_RINGS
        assert residual.df_convention == "residual"
        assert residual.inference_df == residual.n_obs - k
        np.testing.assert_allclose(
            residual.p_value, _t_tail(residual.t_stat, residual.n_obs - k), rtol=1e-12
        )
        assert residual.att == default.att and residual.se == default.se
        assert residual.t_stat == default.t_stat
        assert residual.p_value != default.p_value
        assert residual.conf_int != default.conf_int

    def test_normal_is_deliberate_z(self, default, panel):
        z = _fit(panel, df_convention="normal")
        assert z.inference_df is None and z.df_convention == "normal"
        assert z.se == default.se and z.t_stat == default.t_stat
        np.testing.assert_allclose(z.p_value, _z_tail(z.t_stat), rtol=1e-12)
        lo, hi = z.conf_int
        np.testing.assert_allclose(
            (hi - lo) / 2, stats.norm.ppf(1 - z.alpha / 2) * z.se, rtol=1e-12
        )
        assert "inference_df" not in z.to_dict()


# --------------------------------------------------------------------------- #
# (5) Inert lanes: cluster=False and every Conley fit
# --------------------------------------------------------------------------- #


class TestInertLanes:
    def test_cluster_false_is_knob_inert(self, panel):
        off = _fit(panel, cluster=False)
        off_res = _fit(panel, cluster=False, df_convention="residual")
        assert off.n_clusters is None and off.cluster_name is None
        assert off.inference_df == off.n_obs - (1 + N_RINGS)
        assert (off.p_value, off.conf_int, off.inference_df) == (
            off_res.p_value,
            off_res.conf_int,
            off_res.inference_df,
        )

    def test_conley_auto_is_knob_inert(self, panel):
        c = _fit(panel, **CONLEY_KW)
        c_res = _fit(panel, df_convention="residual", **CONLEY_KW)
        assert c.n_clusters is None
        assert c.inference_df == c.n_obs - (1 + N_RINGS)
        assert (c.p_value, c.conf_int, c.inference_df) == (
            c_res.p_value,
            c_res.conf_int,
            c_res.inference_df,
        )

    def test_conley_with_explicit_cluster_keeps_residual_df(self, panel):
        """The product-kernel cluster is reported (n_clusters) but never a df source."""
        cu = _fit(panel, cluster="unit", **CONLEY_KW)
        cu_res = _fit(panel, cluster="unit", df_convention="residual", **CONLEY_KW)
        assert cu.n_clusters == panel["unit"].nunique() and cu.cluster_name == "unit"
        assert cu.inference_df == cu.n_obs - (1 + N_RINGS)
        assert (cu.p_value, cu.conf_int) == (cu_res.p_value, cu_res.conf_int)

    def test_conley_normal_is_z(self, panel):
        cz = _fit(panel, df_convention="normal", **CONLEY_KW)
        assert cz.inference_df is None
        np.testing.assert_allclose(cz.p_value, _z_tail(cz.t_stat), rtol=1e-12)


# --------------------------------------------------------------------------- #
# (6) Survey df keeps precedence under every value
# --------------------------------------------------------------------------- #


class TestSurveyPrecedence:
    @pytest.mark.parametrize(
        "design, expected_df",
        [
            (SurveyDesign(weights="w"), lambda df: len(df) - 1),  # implicit per-obs PSUs
            (
                SurveyDesign(weights="w", strata="stratum", psu="unit"),
                lambda df: df["unit"].nunique() - df["stratum"].nunique(),
            ),
        ],
        ids=["weights_only", "psu_strata"],
    )
    def test_survey_df_wins(self, panel, design, expected_df):
        fits = {
            v: _fit(panel, df_convention=v, survey_design=design)
            for v in ("cluster", "residual", "normal")
        }
        exp = expected_df(panel)
        for v, r in fits.items():
            assert r.survey_metadata is not None and r.survey_metadata.df_survey == exp, v
            assert r.inference_df == exp, v
        base = fits["cluster"]
        for v in ("residual", "normal"):
            assert (fits[v].p_value, fits[v].conf_int) == (base.p_value, base.conf_int), v
        # The PSU design's df (G - n_strata) is distinguishable from G - 1.
        assert exp != panel["unit"].nunique() - 1 or design.psu is None


# --------------------------------------------------------------------------- #
# (7) Event-study mode
# --------------------------------------------------------------------------- #


def _fit_es(df, **kwargs):
    est = SpilloverDiD(
        rings=[0.0, 50.0, 200.0],
        d_bar=200.0,
        conley_coords=("lat", "lon"),
        event_study=True,
        horizon_max=2,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        return est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")


def _k_effective(res):
    """Finite stage-2 coefficients - the synthetic 'ATT' lincom entry excluded."""
    return sum(1 for name, v in res.coefficients.items() if name != "ATT" and np.isfinite(v))


@pytest.fixture(scope="module")
def es_panel():
    return generate_butts_staggered_dgp(seed=0)


class TestEventStudy:
    def test_cluster_default_is_g_minus_1(self, es_panel):
        res = _fit_es(es_panel)
        G = es_panel["unit"].nunique()
        assert res.n_clusters == G and res.inference_df == G - 1
        dyn = res.att_dynamic
        finite = dyn["p_value"].notna() & (dyn["se"] > 0)
        assert finite.sum() >= 3
        np.testing.assert_allclose(
            dyn.loc[finite, "p_value"], _t_tail(dyn.loc[finite, "t_stat"], G - 1), rtol=1e-12
        )
        np.testing.assert_allclose(res.p_value, _t_tail(res.t_stat, G - 1), rtol=1e-12)

    def test_residual_uses_n_minus_k(self, es_panel):
        res = _fit_es(es_panel, df_convention="residual")
        k = _k_effective(res)
        assert res.inference_df == res.n_obs - k
        dyn = res.att_dynamic
        finite = dyn["p_value"].notna() & (dyn["se"] > 0)
        np.testing.assert_allclose(
            dyn.loc[finite, "p_value"],
            _t_tail(dyn.loc[finite, "t_stat"], res.n_obs - k),
            rtol=1e-12,
        )

    def test_normal_is_z_through_the_extractor(self, es_panel):
        res = _fit_es(es_panel, df_convention="normal")
        assert res.inference_df is None
        dyn = res.att_dynamic
        finite = dyn["p_value"].notna() & (dyn["se"] > 0)
        np.testing.assert_allclose(
            dyn.loc[finite, "p_value"], _z_tail(dyn.loc[finite, "t_stat"]), rtol=1e-12
        )
        np.testing.assert_allclose(res.p_value, _z_tail(res.t_stat), rtol=1e-12)


# --------------------------------------------------------------------------- #
# (8) The df ladder itself (saturation is unreachable through a fit)
# --------------------------------------------------------------------------- #


KNOBS = ("cluster", "residual", "normal")


class TestResolveInferenceDf:
    @pytest.mark.parametrize("knob", KNOBS)
    @pytest.mark.parametrize(
        "n_obs_eff, k_effective, substring",
        [
            (4, 4, "inference df = 0 (n_obs=4, effective_rank=4"),
            (5, 8, "inference df = -3 (n_obs=5, effective_rank=8"),
        ],
    )
    def test_saturated_sample_is_nan_sentinel(self, knob, n_obs_eff, k_effective, substring):
        with pytest.warns(UserWarning, match=substring.replace("(", r"\(")):
            df = _resolve_inference_df(
                df_convention=knob,
                n_obs_eff=n_obs_eff,
                k_effective=k_effective,
                n_clusters=20,
                vcov_mode="cluster",
                survey_df=None,
                has_survey=False,
            )
        assert df == 0.0

    @pytest.mark.parametrize("knob", KNOBS)
    def test_survey_saturated_is_nan_sentinel(self, knob):
        with pytest.warns(UserWarning, match="df_survey = 0"):
            df = _resolve_inference_df(
                df_convention=knob,
                n_obs_eff=302,
                k_effective=2,
                n_clusters=20,
                vcov_mode="cluster",
                survey_df=0,
                has_survey=True,
            )
        assert df == 0.0

    @pytest.mark.parametrize("knob", KNOBS)
    def test_survey_df_beats_every_knob(self, knob):
        df = _resolve_inference_df(
            df_convention=knob,
            n_obs_eff=302,
            k_effective=2,
            n_clusters=20,
            vcov_mode="cluster",
            survey_df=12,
            has_survey=True,
        )
        assert df == 12.0

    @pytest.mark.parametrize(
        "vcov_mode, n_clusters, expected",
        [
            ("cluster", 20, {"cluster": 19.0, "residual": 300.0, "normal": None}),
            ("conley", 20, {"cluster": 300.0, "residual": 300.0, "normal": None}),
            ("hc1", None, {"cluster": 300.0, "residual": 300.0, "normal": None}),
        ],
        ids=["cluster_mode", "conley_mode", "hc1_mode"],
    )
    def test_knob_branches(self, vcov_mode, n_clusters, expected):
        for knob, want in expected.items():
            got = _resolve_inference_df(
                df_convention=knob,
                n_obs_eff=302,
                k_effective=2,
                n_clusters=n_clusters,
                vcov_mode=vcov_mode,
                survey_df=None,
                has_survey=False,
            )
            assert got == want, (vcov_mode, knob)

    def test_zero_sentinel_is_all_nan_in_safe_inference(self):
        t, p, ci = safe_inference(1.0, 0.5, alpha=0.05, df=0.0)
        assert np.isnan(t) and np.isnan(p) and np.isnan(ci[0]) and np.isnan(ci[1])


# --------------------------------------------------------------------------- #
# (9) Ledger
# --------------------------------------------------------------------------- #


def test_ledger_row_is_done_and_points_here():
    from tests.test_v4_matrix import ROWS as LEDGER_ROWS

    row = {r["id"]: r for r in LEDGER_ROWS}["M-154"]
    assert row["kind"] == "behavior"
    assert row["status"] == "done"
    assert row["test_ref"] == THIS_TEST_REF
    assert row["old"] == "diff_diff:SpilloverDiD[df_convention]"
    assert row["introduced_in"] == "4.0"
