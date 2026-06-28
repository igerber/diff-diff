"""Tests for the LPDiD (Local Projections DiD) estimator.

Validation strategy (B1, pure-Python, no R required):

- ``TestLPDiDAPI`` / ``TestLPDiDEdgeCases``: parameter validation, idempotence,
  NaN-consistent inference, absorbing-path enforcement.
- ``TestLPDiDFormula``: small hand-built panels with analytically-known
  coefficients (true effect derivable by hand).
- ``TestLPDiDMethodology``: DGP-recovery on panels with a known dynamic effect
  path, plus the clean-control / weighting properties.
- ``TestLPDiDCrossEstimator``: the equivalences proved in Dube, Girardi, Jordà &
  Taylor (2025), validated against estimators already in the library
  (point estimates only; SEs are anchored separately by the B2 R-parity layer).

External R-parity (authors' ``danielegirardi/lpdid`` + ``alexCardazzi/lpdid``)
lives in ``tests/test_methodology_lpdid.py`` (PR-B2).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dgp_utils import make_lpdid_panel  # noqa: E402

from diff_diff import (  # noqa: E402
    CallawaySantAnna,
    ImputationDiD,
    LPDiD,
    LPDiDResults,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_linear_panel(units):
    """Build a panel from per-unit specs (treat_start=99 => never-treated)."""
    rows = []
    for spec in units:
        for time in spec["times"]:
            rows.append(
                {
                    "unit": spec["unit"],
                    "time": time,
                    "y": spec["y"][time],
                    "treat": int(time >= spec["treat_start"]),
                    **{
                        k: v
                        for k, v in spec.items()
                        if k not in {"unit", "times", "y", "treat_start"}
                    },
                }
            )
    return pd.DataFrame(rows)


def _event_coef(res, horizon=0):
    es = res.event_study
    return es.loc[es["horizon"] == horizon, "coefficient"].iloc[0]


def _event_row(res, horizon=0):
    es = res.event_study
    return es.loc[es["horizon"] == horizon].iloc[0]


# ===========================================================================
# API / validation
# ===========================================================================
class TestLPDiDAPI:
    def test_get_params_round_trip(self):
        est = LPDiD(pre_window=4, post_window=6, reweight=True, no_composition=True)
        params = est.get_params()
        assert params["pre_window"] == 4
        assert params["post_window"] == 6
        assert params["reweight"] is True
        assert params["no_composition"] is True

    def test_set_params_updates_attributes(self):
        est = LPDiD()
        returned = est.set_params(pre_window=5, control_group="never_treated")
        assert returned is est
        assert est.pre_window == 5
        assert est.control_group == "never_treated"

    def test_rejects_invalid_control_group(self):
        with pytest.raises(ValueError, match="control_group"):
            LPDiD(control_group="bad")

    def test_rejects_invalid_rank_deficient_action(self):
        with pytest.raises(ValueError, match="rank_deficient_action"):
            LPDiD(rank_deficient_action="bad")

    def test_rejects_invalid_pmd_value(self):
        with pytest.raises(ValueError, match="pmd"):
            LPDiD(pmd="bad")

    def test_rejects_negative_window(self):
        with pytest.raises(ValueError, match="pre_window"):
            LPDiD(pre_window=-1)
        with pytest.raises(ValueError, match="post_window"):
            LPDiD(post_window=-2)
        with pytest.raises(ValueError, match="pre_window"):
            LPDiD(pre_window=True)  # bool is not a valid int window

    def test_set_params_rejects_unknown_key(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            LPDiD().set_params(nonexistent_param=1)

    def test_set_params_is_transactional(self):
        # An invalid update must roll back ALL fields (snapshot + restore).
        est = LPDiD(pre_window=2)
        with pytest.raises(ValueError, match="control_group"):
            est.set_params(pre_window=7, control_group="bad")
        assert est.pre_window == 2  # rolled back, not left at 7

    def test_requires_core_columns(self):
        df = pd.DataFrame({"y": [1.0], "id": [1], "t": [0]})
        with pytest.raises(ValueError, match="Missing columns"):
            LPDiD().fit(df, outcome="y", unit="id", time="t", treatment="treat")

    def test_rejects_only_event_and_only_pooled_together(self):
        df = pd.DataFrame({"y": [1.0], "id": [1], "t": [0], "treat": [0]})
        with pytest.raises(ValueError, match="only_event"):
            LPDiD().fit(
                df,
                outcome="y",
                unit="id",
                time="t",
                treatment="treat",
                only_event=True,
                only_pooled=True,
            )

    def test_rejects_non_numeric_treatment_values(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "y": [1, 2, 1, 1],
                "treat": [0, "treated", 0, 0],
            }
        )
        with pytest.raises(ValueError, match="binary numeric"):
            LPDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_rejects_non_binary_treatment_values(self):
        df = pd.DataFrame(
            {"unit": [1, 1, 2, 2], "time": [0, 1, 0, 1], "y": [1, 2, 1, 1], "treat": [0, 2, 0, 0]}
        )
        with pytest.raises(ValueError, match="binary numeric"):
            LPDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_event_dataframe_contains_reference_and_requested_horizons(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "time": [0, 1, 2] * 3,
                "y": [1, 2, 4, 1, 1, 1, 2, 2, 2],
                "treat": [0, 1, 1, 0, 0, 0, 0, 0, 0],
            }
        )
        res = LPDiD(pre_window=2, post_window=1).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        assert list(res.event_study["horizon"]) == [-2, -1, 0, 1]

    def test_pooled_dataframe_has_pre_and_post_rows(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 1, 1, 2, 2, 2, 2],
                "time": [0, 1, 2, 3] * 2,
                "y": [1, 2, 4, 6, 1, 1, 1, 1],
                "treat": [0, 0, 1, 1, 0, 0, 0, 0],
            }
        )
        res = LPDiD(pre_window=2, post_window=1).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert set(res.pooled["window"]) == {"pre", "post"}

    def test_rejects_pooled_horizon_outside_supported_window(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 0, 1, 2],
                "y": [1, 3, 5, 1, 1, 1],
                "treat": [0, 1, 1, 0, 0, 0],
            }
        )
        with pytest.raises(ValueError, match="outside the supported pre window"):
            LPDiD(pre_window=2, post_window=1).fit(
                df, outcome="y", unit="unit", time="time", treatment="treat", pre_pooled=(-3, -1)
            )

    def test_results_to_dataframe_and_repr(self):
        df = pd.DataFrame({"horizon": [0], "coefficient": [1.0], "se": [0.1]})
        results = LPDiDResults(
            event_study=df,
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
        assert results.to_dataframe(level="event").equals(df)
        with pytest.raises(ValueError, match="not computed"):
            results.to_dataframe(level="pooled")
        with pytest.raises(ValueError, match="level must be"):
            results.to_dataframe(level="bad")
        assert "LPDiDResults" in repr(results)

    def test_results_summary_and_to_dict_run(self):
        df = make_lpdid_panel(cohorts=(4,), n_per_cohort=15, n_never=15, n_periods=8, seed=11)
        res = LPDiD(pre_window=2, post_window=2).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert isinstance(res.summary(), str)
        assert "Local Projections DiD" in res.summary()
        d = res.to_dict()
        assert d["inference_method"] == "cluster_robust"
        assert d["cluster_name"] == "unit"
        assert np.isfinite(d["att"])

    def test_clears_fitted_state_after_failed_refit(self):
        good = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 0, 1, 2],
                "y": [1, 3, 5, 1, 1, 1],
                "treat": [0, 1, 1, 0, 0, 0],
            }
        )
        bad = good.copy()
        bad["treat"] = [0, "bad", 1, 0, 0, 0]
        est = LPDiD(pre_window=2, post_window=1)
        est.fit(good, outcome="y", unit="unit", time="time", treatment="treat", only_event=True)
        with pytest.raises(ValueError, match="binary numeric"):
            est.fit(bad, outcome="y", unit="unit", time="time", treatment="treat")
        assert est.is_fitted_ is False
        assert est.results_ is None

    def test_fit_is_idempotent_on_config(self):
        # Re-fitting with the same config must give identical results (fit
        # writes no config-mutating state).
        df = make_lpdid_panel(cohorts=(4,), n_per_cohort=12, n_never=12, n_periods=8, seed=7)
        est = LPDiD(pre_window=2, post_window=2)
        r1 = est.fit(df, outcome="y", unit="unit", time="time", treatment="treat")
        params_after = est.get_params()
        r2 = est.fit(df, outcome="y", unit="unit", time="time", treatment="treat")
        assert est.get_params() == params_after
        pd.testing.assert_frame_equal(r1.event_study, r2.event_study)


# ===========================================================================
# Analytical closed-form coefficients (true effect derivable by hand)
# ===========================================================================
class TestLPDiDFormula:
    def test_event_estimation_controls_for_calendar_time(self):
        df = pd.DataFrame(
            {
                "unit": ["c1"] * 3
                + ["c2"] * 3
                + ["t1"] * 3
                + ["t2a"] * 3
                + ["t2b"] * 3
                + ["t2c"] * 3,
                "time": [0, 1, 2] * 6,
                "y": [0, 0, 100, 0, 0, 100, 0, 5, 105, 0, 0, 105, 0, 0, 105, 0, 0, 105],
                "treat": [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            }
        )
        res = LPDiD(pre_window=2, post_window=0).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        assert _event_coef(res, 0) == pytest.approx(5.0, abs=1e-8)

    def test_pmd_max_uses_mean_of_all_available_pre_periods(self):
        df = _make_linear_panel(
            [
                {
                    "unit": "t1",
                    "times": [0, 1, 2, 3],
                    "treat_start": 3,
                    "y": {0: 0, 1: 2, 2: 4, 3: 6},
                },
                {
                    "unit": "t2",
                    "times": [0, 1, 2, 3],
                    "treat_start": 3,
                    "y": {0: 0, 1: 2, 2: 4, 3: 6},
                },
                {
                    "unit": "c1",
                    "times": [0, 1, 2, 3],
                    "treat_start": 99,
                    "y": {0: 0, 1: 1, 2: 2, 3: 3},
                },
                {
                    "unit": "c2",
                    "times": [0, 1, 2, 3],
                    "treat_start": 99,
                    "y": {0: 0, 1: 1, 2: 2, 3: 3},
                },
            ]
        )
        standard = LPDiD(pre_window=2, post_window=0).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        pmd_max = LPDiD(pre_window=2, post_window=0, pmd="max").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        assert _event_coef(standard) == pytest.approx(1.0, abs=1e-8)
        assert _event_coef(pmd_max) == pytest.approx(2.0, abs=1e-8)

    def test_pmd_integer_uses_last_k_pre_periods(self):
        df = _make_linear_panel(
            [
                {
                    "unit": "t1",
                    "times": [0, 1, 2, 3],
                    "treat_start": 3,
                    "y": {0: 0, 1: 2, 2: 4, 3: 6},
                },
                {
                    "unit": "t2",
                    "times": [0, 1, 2, 3],
                    "treat_start": 3,
                    "y": {0: 0, 1: 2, 2: 4, 3: 6},
                },
                {
                    "unit": "c1",
                    "times": [0, 1, 2, 3],
                    "treat_start": 99,
                    "y": {0: 0, 1: 1, 2: 2, 3: 3},
                },
                {
                    "unit": "c2",
                    "times": [0, 1, 2, 3],
                    "treat_start": 99,
                    "y": {0: 0, 1: 1, 2: 2, 3: 3},
                },
            ]
        )
        pmd_two = LPDiD(pre_window=2, post_window=0, pmd=2).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        assert _event_coef(pmd_two) == pytest.approx(1.5, abs=1e-8)

    def test_covariates_and_absorb_remove_confounding_bias(self):
        df = _make_linear_panel(
            [
                {
                    "unit": "tA1",
                    "times": [0, 1, 2],
                    "treat_start": 1,
                    "y": {0: 0, 1: 6, 2: 12},
                    "x1": 2,
                    "region": "A",
                },
                {
                    "unit": "tA2",
                    "times": [0, 1, 2],
                    "treat_start": 1,
                    "y": {0: 0, 1: 5, 2: 10},
                    "x1": 1,
                    "region": "A",
                },
                {
                    "unit": "tB1",
                    "times": [0, 1, 2],
                    "treat_start": 1,
                    "y": {0: 0, 1: -1, 2: -2},
                    "x1": 1,
                    "region": "B",
                },
                {
                    "unit": "cA1",
                    "times": [0, 1, 2],
                    "treat_start": 99,
                    "y": {0: 0, 1: 4, 2: 8},
                    "x1": 0,
                    "region": "A",
                },
                {
                    "unit": "cB1",
                    "times": [0, 1, 2],
                    "treat_start": 99,
                    "y": {0: 0, 1: -1, 2: -2},
                    "x1": 1,
                    "region": "B",
                },
                {
                    "unit": "cB2",
                    "times": [0, 1, 2],
                    "treat_start": 99,
                    "y": {0: 0, 1: -2, 2: -4},
                    "x1": 0,
                    "region": "B",
                },
            ]
        )
        uncontrolled = LPDiD(pre_window=2, post_window=0).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        controlled = LPDiD(pre_window=2, post_window=0).fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            covariates=["x1"],
            absorb=["region"],
            only_event=True,
        )
        assert _event_coef(uncontrolled) > 2.5
        assert abs(_event_coef(controlled)) < 1e-8

    def test_ylags_remove_lagged_outcome_bias(self):
        df = _make_linear_panel(
            [
                {
                    "unit": "t1",
                    "times": [0, 1, 2, 3],
                    "treat_start": 2,
                    "y": {0: 8, 1: 4, 2: 2, 3: 1},
                },
                {
                    "unit": "t2",
                    "times": [0, 1, 2, 3],
                    "treat_start": 2,
                    "y": {0: 6, 1: 3, 2: 1.5, 3: 0.75},
                },
                {
                    "unit": "c1",
                    "times": [0, 1, 2, 3],
                    "treat_start": 99,
                    "y": {0: 4, 1: 2, 2: 1, 3: 0.5},
                },
                {
                    "unit": "c2",
                    "times": [0, 1, 2, 3],
                    "treat_start": 99,
                    "y": {0: 2, 1: 1, 2: 0.5, 3: 0.25},
                },
            ]
        )
        uncontrolled = LPDiD(pre_window=2, post_window=0).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        with_ylag = LPDiD(pre_window=2, post_window=0).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", ylags=1, only_event=True
        )
        assert _event_coef(uncontrolled) < -0.5
        assert abs(_event_coef(with_ylag)) < 1e-8

    def test_dylags_remove_lagged_difference_bias(self):
        df = _make_linear_panel(
            [
                {
                    "unit": "t1",
                    "times": [0, 1, 2, 3],
                    "treat_start": 3,
                    "y": {0: 0, 1: 0, 2: 4, 3: 8},
                },
                {
                    "unit": "t2",
                    "times": [0, 1, 2, 3],
                    "treat_start": 3,
                    "y": {0: 0, 1: 0, 2: 3, 3: 6},
                },
                {
                    "unit": "c1",
                    "times": [0, 1, 2, 3],
                    "treat_start": 99,
                    "y": {0: 0, 1: 0, 2: 2, 3: 4},
                },
                {
                    "unit": "c2",
                    "times": [0, 1, 2, 3],
                    "treat_start": 99,
                    "y": {0: 0, 1: 0, 2: 1, 3: 2},
                },
            ]
        )
        uncontrolled = LPDiD(pre_window=2, post_window=0).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        with_dylag = LPDiD(pre_window=2, post_window=0).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", dylags=1, only_event=True
        )
        assert _event_coef(uncontrolled) > 1.5
        assert abs(_event_coef(with_dylag)) < 1e-8


# ===========================================================================
# DGP recovery + methodological properties
# ===========================================================================
class TestLPDiDMethodology:
    def test_single_cohort_exact_dynamic_recovery(self):
        # Noiseless single-cohort DGP: LP-DiD recovers tau_h = 1 + 0.5h exactly,
        # and pre-treatment placebos are zero.
        df = make_lpdid_panel(
            cohorts=(5,),
            n_per_cohort=20,
            n_never=20,
            n_periods=12,
            tau=lambda k: 1.0 + 0.5 * k,
            unit_fe_sd=0.0,
            error_sd=0.0,
            heterogeneous=False,
            seed=1,
        )
        res = LPDiD(pre_window=3, post_window=4).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        for h in range(0, 5):
            assert _event_coef(res, h) == pytest.approx(1.0 + 0.5 * h, abs=1e-8)
        for h in (-3, -2):
            assert _event_coef(res, h) == pytest.approx(0.0, abs=1e-8)

    def test_clean_control_rule_defeats_already_treated_contamination(self):
        # An early cohort with a large ongoing effect would bias a naive
        # long-difference that used it as a control; LP-DiD's clean-control
        # restriction excludes it and recovers the late cohort's true effect.
        df = make_lpdid_panel(
            cohorts=(2, 8),
            n_per_cohort=25,
            n_never=25,
            n_periods=12,
            tau=lambda k: 5.0,
            unit_fe_sd=0.0,
            error_sd=0.0,
            heterogeneous=False,
            seed=2,
        )
        res = LPDiD(pre_window=2, post_window=1).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        # Homogeneous effect 5.0 across cohorts -> h=0 effect is exactly 5.0.
        assert _event_coef(res, 0) == pytest.approx(5.0, abs=1e-8)

    def test_pooled_post_equals_mean_of_event_study(self):
        df = make_lpdid_panel(
            cohorts=(5,),
            n_per_cohort=20,
            n_never=20,
            n_periods=12,
            tau=lambda k: 1.0 + 0.5 * k,
            unit_fe_sd=0.0,
            error_sd=0.0,
            heterogeneous=False,
            seed=3,
        )
        res = LPDiD(pre_window=2, post_window=3).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        es = res.event_study
        post_mean = es.loc[es["horizon"].between(0, 3), "coefficient"].mean()
        post_pooled = res.pooled.loc[res.pooled["window"] == "post", "coefficient"].iloc[0]
        assert post_pooled == pytest.approx(post_mean, abs=1e-8)

    def test_variance_and_equal_weighting_differ_under_heterogeneity(self):
        # With heterogeneous cohort effects and unequal cohort sizes, the
        # variance-weighted (default) and equally-weighted (reweight) ATTs differ.
        df = make_lpdid_panel(
            cohorts=(4, 8), n_per_cohort=20, n_never=20, n_periods=12, heterogeneous=True, seed=4
        )
        vw = LPDiD(pre_window=2, post_window=2).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        ew = LPDiD(pre_window=2, post_window=2, reweight=True).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        assert _event_coef(vw, 2) != pytest.approx(_event_coef(ew, 2), abs=1e-6)

    def test_cluster_se_finite_and_nan_consistent(self):
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=20, n_never=20, n_periods=10, seed=5)
        res = LPDiD(pre_window=2, post_window=2).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        row0 = _event_row(res, 0)
        assert np.isfinite(row0["se"]) and row0["se"] > 0
        assert np.isfinite(row0["t_stat"]) and np.isfinite(row0["conf_low"])

    def test_single_cluster_yields_nan_inference(self):
        # All obs in one cluster -> cluster vcov undefined -> NaN-consistent.
        df = make_lpdid_panel(cohorts=(2,), n_per_cohort=4, n_never=4, n_periods=4, seed=6)
        df["only"] = 1
        res = LPDiD(pre_window=1, post_window=1, cluster="only").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        row0 = _event_row(res, 0)
        for col in ("se", "t_stat", "p_value", "conf_low", "conf_high"):
            assert pd.isna(row0[col])


# ===========================================================================
# Cross-estimator equivalences proved in Dube et al. (2025)
# ===========================================================================
class TestLPDiDCrossEstimator:
    def test_ce1_2x2_h0_equals_first_difference_did(self):
        # 2 periods, treated-at-1 vs never: LP-DiD h=0 == the closed-form 2x2
        # DiD = mean(dy | treated) - mean(dy | control)  [paper Appendix A.1].
        df = make_lpdid_panel(
            cohorts=(1,),
            n_per_cohort=60,
            n_never=60,
            n_periods=2,
            tau=lambda k: 3.0,
            heterogeneous=False,
            seed=10,
        )
        lp = LPDiD(pre_window=1, post_window=0).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        wide = df.pivot(index="unit", columns="time", values="y")
        dy = wide[1] - wide[0]
        treated_units = df.loc[df["first_treat"] == 1, "unit"].unique()
        is_t = wide.index.isin(treated_units)
        did_2x2 = dy[is_t].mean() - dy[~is_t].mean()
        assert _event_coef(lp, 0) == pytest.approx(did_2x2, abs=1e-9)

    def test_ce2_reweighted_equals_callaway_santanna(self):
        # Reweighted LP-DiD == Callaway-Sant'Anna event-study effects
        # [paper Section 3.7]. Match CS to LP-DiD's not-yet-treated controls.
        df = make_lpdid_panel(cohorts=(5, 8), n_per_cohort=25, n_never=30, n_periods=12, seed=21)
        lp = LPDiD(pre_window=3, post_window=3, reweight=True).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        cs = CallawaySantAnna(control_group="not_yet_treated").fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        for h in range(0, 4):
            assert _event_coef(lp, h) == pytest.approx(
                cs.event_study_effects[h]["effect"], abs=1e-9
            )

    def test_ce3_pmd_single_cohort_equals_bjs_imputation(self):
        # PMD LP-DiD (k=t-1, "max") with a SINGLE treated cohort == BJS
        # imputation [paper Section 3.4, footnotes 10-11; only single-cohort
        # is an exact equality, multi-cohort is only "very similar"].
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=40, n_never=40, n_periods=12, seed=31)
        assert df.loc[np.isfinite(df["first_treat"]), "first_treat"].nunique() == 1
        lp = LPDiD(pre_window=3, post_window=3, pmd="max").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        bjs = ImputationDiD().fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        for h in range(0, 4):
            assert _event_coef(lp, h) == pytest.approx(
                bjs.event_study_effects[h]["effect"], abs=1e-6
            )

    # CE-4 (variance-weighted LP-DiD == stacked regression) is intentionally
    # omitted: the paper's equivalence is to the Cengiz et al. (2019) stacking
    # scheme, whereas diff-diff's `StackedDiD` implements Wing, Freedman &
    # Hollingsworth (2024) with Q-weights — a different (corrected) scheme that
    # does not numerically coincide. See REGISTRY `## LPDiD`.


# ===========================================================================
# Edge cases
# ===========================================================================
class TestLPDiDEdgeCases:
    def test_rejects_non_absorbing_treatment(self):
        # Treatment that turns off must raise (absorbing-path scope).
        df = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 0, 1, 2],
                "y": [1.0, 2, 3, 1, 1, 1],
                "treat": [0, 1, 0, 0, 0, 0],
            }
        )
        with pytest.raises(ValueError, match="absorbing"):
            LPDiD(pre_window=1, post_window=1).fit(
                df, outcome="y", unit="unit", time="time", treatment="treat"
            )

    def test_no_composition_drops_controls(self):
        df = _make_linear_panel(
            [
                {
                    "unit": "t1",
                    "times": [0, 1, 2, 3],
                    "treat_start": 1,
                    "y": {0: 0, 1: 1, 2: 2, 3: 3},
                },
                {
                    "unit": "t2",
                    "times": [0, 1, 2, 3],
                    "treat_start": 2,
                    "y": {0: 0, 1: 0, 2: 2, 3: 4},
                },
                {
                    "unit": "c1",
                    "times": [0, 1, 2, 3],
                    "treat_start": 99,
                    "y": {0: 0, 1: 0, 2: 0, 3: 0},
                },
            ]
        )
        unrestricted = LPDiD(pre_window=2, post_window=2).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        common = LPDiD(pre_window=2, post_window=2, no_composition=True).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        u0 = unrestricted.event_study.loc[unrestricted.event_study["horizon"] == 0, "n_obs"].iloc[0]
        c0 = common.event_study.loc[common.event_study["horizon"] == 0, "n_obs"].iloc[0]
        assert c0 < u0


class TestLPDiDUnbalanced:
    """Unbalanced-panel correctness (review round 1: reweight denominators, RA
    identification, no_composition all computed from the realized sample)."""

    def _unbalanced(self):
        df = make_lpdid_panel(cohorts=(5, 8), n_per_cohort=25, n_never=30, n_periods=12, seed=21)
        drop = (
            (df["first_treat"] == np.inf) & (df["time"].isin([9, 10, 11])) & (df["unit"] % 3 == 0)
        )
        return df.loc[~drop].reset_index(drop=True)

    def test_reweight_matches_cs_on_unbalanced(self):
        # Equal-weighting denominators must come from the realized (post-drop)
        # sample, else the Callaway-Sant'Anna equivalence breaks on missing rows.
        ub = self._unbalanced()
        lp = LPDiD(pre_window=3, post_window=3, reweight=True).fit(
            ub, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        cs = CallawaySantAnna(control_group="not_yet_treated").fit(
            ub,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        for h in range(0, 4):
            assert _event_coef(lp, h) == pytest.approx(
                cs.event_study_effects[h]["effect"], abs=1e-9
            )

    def test_ra_nan_consistent_on_uncontrolled_event_time(self):
        # An event time with treated units but no clean control is unidentified:
        # the RA path drops those treated (with a warning) and returns jointly
        # NaN inference, never a coef=NaN / se=0 mismatch.
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=10, n_never=10, n_periods=10, seed=3)
        df["x"] = np.arange(len(df)) % 3
        df = df.loc[~((df["first_treat"] == np.inf) & (df["time"] == 5))].reset_index(drop=True)
        with pytest.warns(UserWarning, match="no clean control"):
            res = LPDiD(pre_window=2, post_window=2, reweight=True).fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                covariates=["x"],
                only_event=True,
            )
        row0 = _event_row(res, 0)
        assert pd.isna(row0["coefficient"]) and pd.isna(row0["se"])
        for col in ("t_stat", "p_value", "conf_low", "conf_high"):
            assert pd.isna(row0[col])

    def test_no_composition_holds_post_horizons_fixed(self):
        # no_composition fixes the POST-treatment composition: every post horizon
        # shares the same realized n_obs even on an unbalanced panel.
        ub = self._unbalanced()
        res = LPDiD(pre_window=3, post_window=3, no_composition=True).fit(
            ub, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        post = res.event_study.loc[res.event_study["horizon"].between(0, 3), "n_obs"]
        assert post.nunique() == 1

    def test_pmd_retains_treated_missing_exact_t_minus_1(self):
        # Under PMD the long difference uses the premean baseline, so a treated
        # observation missing the exact t-1 outcome (but with earlier pre-data)
        # must NOT be dropped.
        df = make_lpdid_panel(
            cohorts=(5,),
            n_per_cohort=12,
            n_never=12,
            n_periods=10,
            unit_fe_sd=0.0,
            error_sd=0.0,
            heterogeneous=False,
            tau=lambda k: 2.0,
            seed=1,
        )
        treated = df.loc[df["first_treat"] == 5, "unit"].unique()
        df = df.loc[~(df["unit"].isin(treated[:6]) & (df["time"] == 4))].reset_index(drop=True)
        std = LPDiD(pre_window=2, post_window=2).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        pmd = LPDiD(pre_window=2, post_window=2, pmd="max").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        std_n = std.event_study.loc[std.event_study["horizon"] == 0, "n_obs"].iloc[0]
        pmd_n = pmd.event_study.loc[pmd.event_study["horizon"] == 0, "n_obs"].iloc[0]
        assert pmd_n > std_n  # PMD keeps missing-t-1 treated that first-lag drops
        assert np.isfinite(_event_coef(pmd, 0))

    def test_pre_window_too_small_for_pooled_raises(self):
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=8, n_never=8, n_periods=10, seed=2)
        with pytest.raises(ValueError, match="pooled pre window is empty"):
            LPDiD(pre_window=1, post_window=1).fit(
                df, outcome="y", unit="unit", time="time", treatment="treat"
            )
        # only_event=True does not need a pooled pre window
        r = LPDiD(pre_window=1, post_window=1).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        assert list(r.event_study["horizon"]) == [-1, 0, 1]

    def test_pre_pooled_rejects_reference_horizon(self):
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=8, n_never=8, n_periods=10, seed=3)
        with pytest.raises(ValueError, match="outside the supported pre window"):
            LPDiD(pre_window=3, post_window=1).fit(
                df, outcome="y", unit="unit", time="time", treatment="treat", pre_pooled=(-2, -1)
            )

    def test_no_composition_post_fixed_under_nonmonotone_missingness(self):
        # Missing an INTERMEDIATE target (t+1) while the max target (t+2) is
        # observed must still give identical post-horizon n_obs (the common mask
        # requires every post target, not just the maximum horizon).
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=20, n_never=20, n_periods=12, seed=7)
        drop = (df["first_treat"] == np.inf) & (df["time"] == 6) & (df["unit"] % 4 == 0)
        ub = df.loc[~drop].reset_index(drop=True)
        res = LPDiD(pre_window=2, post_window=2, no_composition=True).fit(
            ub, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        post = res.event_study.loc[res.event_study["horizon"].between(0, 2), "n_obs"]
        assert post.nunique() == 1

    def test_rank_deficient_action_propagates_to_results(self):
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=10, n_never=10, n_periods=8, seed=8)
        res = LPDiD(pre_window=2, post_window=2, rank_deficient_action="silent").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.rank_deficient_action == "silent"
        assert res.to_dict()["rank_deficient_action"] == "silent"

    def test_ra_path_reports_if_cluster_variance_label(self):
        # The regression-adjustment covariate path uses an influence-function
        # cluster variance, not an OLS CR1 sandwich: the results metadata and the
        # summary label must say so (not "hc1" / "CR1").
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=12, n_never=12, n_periods=8, seed=9)
        df["x"] = np.arange(len(df)) % 3
        ra = LPDiD(pre_window=2, post_window=2, reweight=True).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", covariates=["x"]
        )
        assert ra.to_dict()["vcov_type"] == "if_cluster"
        assert "Influence-function" in ra.summary()
        # the default (non-RA) path stays hc1 / CR1
        base = LPDiD(pre_window=2, post_window=2).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert base.to_dict()["vcov_type"] == "hc1"
        assert "CR1 cluster-robust" in base.summary()

    def test_pmd_max_excludes_present_but_nan_pretreatment(self):
        # A present-but-NaN pretreatment outcome must not deflate the premean
        # baseline: the denominator counts non-missing prior outcomes, not rows.
        rows = []
        for u in (1, 2):
            for t, y in enumerate([10.0, np.nan, 30.0, 40.0, 50.0]):
                rows.append({"unit": u, "time": t, "y": y, "treat": int(t >= 3)})
        for u, base in {3: 0, 4: 1, 5: 2, 6: 3}.items():
            for t in range(5):
                rows.append({"unit": u, "time": t, "y": float(base + t), "treat": 0})
        df = pd.DataFrame(rows)
        est = LPDiD(pre_window=2, post_window=1, pmd="max")
        est.fit(df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True)
        panel = est._prepare_panel(
            df, outcome="y", unit="unit", time="time", treatment="treat", cluster="unit"
        )
        baseline = panel.loc[(panel["unit"] == 1) & (panel["time"] == 3), "_pmd_all_baseline"].iloc[
            0
        ]
        assert baseline == pytest.approx(20.0, abs=1e-9)  # mean(10, 30); NaN at t=1 excluded

    def test_ylags_dylags_trigger_direct_inclusion_warning(self):
        # Outcome/first-difference lags are direct-included controls under
        # reweight=False, so they fire the same homogeneity warning as covariates.
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=10, n_never=10, n_periods=10, seed=4)
        with pytest.warns(UserWarning, match="covariate-style controls"):
            LPDiD(pre_window=2, post_window=2).fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                ylags=1,
                only_event=True,
            )
        with pytest.warns(UserWarning, match="covariate-style controls"):
            LPDiD(pre_window=2, post_window=2).fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                dylags=1,
                only_event=True,
            )

    def test_no_composition_post_fixed_with_present_but_nan_outcome(self):
        # Fixed composition must hold even when a target row exists but its
        # outcome is NaN (value-based availability, not row existence).
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=20, n_never=20, n_periods=12, seed=7)
        df.loc[(df["first_treat"] == np.inf) & (df["time"] == 6) & (df["unit"] % 4 == 0), "y"] = (
            np.nan
        )
        res = LPDiD(pre_window=2, post_window=2, no_composition=True).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
        )
        post = res.event_study.loc[res.event_study["horizon"].between(0, 2), "n_obs"]
        assert post.nunique() == 1

    def test_pooled_rejects_bool_window(self):
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=8, n_never=8, n_periods=10, seed=2)
        with pytest.raises(ValueError, match="not bool"):
            LPDiD(pre_window=3, post_window=1).fit(
                df, outcome="y", unit="unit", time="time", treatment="treat", pre_pooled=True
            )

    def test_ylags_dylags_recorded_in_results(self):
        df = make_lpdid_panel(cohorts=(5,), n_per_cohort=10, n_never=10, n_periods=10, seed=4)
        res = LPDiD(pre_window=2, post_window=2, reweight=True).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", ylags=2, dylags=1
        )
        assert res.ylags == 2 and res.dylags == 1
        d = res.to_dict()
        assert d["ylags"] == 2 and d["dylags"] == 1
