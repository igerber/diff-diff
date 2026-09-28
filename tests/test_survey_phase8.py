"""Tests for Phase 8 Survey Maturity: SDR replicate method + FPC support."""

import numpy as np
import pandas as pd
import pytest

from diff_diff.survey import (
    SurveyDesign,
    _replicate_variance_factor,
    compute_replicate_if_variance,
    compute_replicate_vcov,
)

# ---------------------------------------------------------------------------
# Rows M-021/M-022 (+ M-118/M-119): ImputationDiD / TwoStageDiD
# ``fit(aggregate=, balance_e=)`` is deprecated (3.9, removed 4.0) and warns on
# ANY supplied value. The deprecated fit-time route is kept DELIBERATELY here:
# these tests pin FIT-TIME surface behaviour (bit-equality grids, bootstrap
# aggregation, R/Stata parity, replicate overrides, native effect dicts) that
# the post-fit ``results.aggregate(...)`` container route does not reproduce
# shape-for-shape. The shim warning is therefore filtered BY MESSAGE, scoped to
# these two estimators only - every other FutureWarning (including the other
# estimators' aggregate() shims) still surfaces.
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.filterwarnings(
    r"ignore:(ImputationDiD|TwoStageDiD)\.fit\((aggregate=|balance_e=|aggregate= / balance_e=)\):FutureWarning"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_replicate_data(n=200, k=20, seed=123):
    """Data with full-sample + replicate weight columns."""
    np.random.seed(seed)
    x = np.random.randn(n)
    eps = np.random.randn(n) * 0.5
    y = 1.0 + 2.0 * x + eps
    w = 1.0 + np.random.exponential(0.5, n)
    data = pd.DataFrame({"x": x, "y": y, "weight": w})

    # BRR-style replicates: w * (1 + 0.5 * sign) with random signs per cluster
    rng = np.random.RandomState(seed + 1)
    cluster_size = n // k
    rep_cols = []
    for r in range(k):
        signs = rng.choice([-1, 1], size=k)
        perturbation = np.repeat(signs, cluster_size)[:n]
        w_r = w * (1.0 + 0.5 * perturbation)
        w_r = np.maximum(w_r, 0.0)
        col = f"rep_{r}"
        data[col] = w_r
        rep_cols.append(col)
    return data, rep_cols


@pytest.fixture
def replicate_data():
    return _make_replicate_data()


def _make_staggered_panel(n_units=50, n_periods=8, seed=456):
    """Multi-period staggered panel for ImputationDiD/TwoStageDiD tests."""
    np.random.seed(seed)
    rows = []
    for i in range(n_units):
        if i < 15:
            ft = 4  # cohort 1
        elif i < 30:
            ft = 6  # cohort 2
        else:
            ft = 0  # never-treated
        # Assign to strata and PSUs for survey design
        stratum = i % 3
        psu = i  # one PSU per unit
        wt = 1.0 + 0.3 * (i % 5)
        for t in range(1, n_periods + 1):
            y = 10.0 + i * 0.03 + t * 0.2
            if ft > 0 and t >= ft:
                y += 2.0
            y += np.random.normal(0, 0.4)
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "first_treat": ft,
                    "outcome": y,
                    "weight": wt,
                    "stratum": stratum,
                    "psu": psu,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def staggered_data():
    return _make_staggered_panel()


# ===========================================================================
# 8a: SDR Replicate Method Tests
# ===========================================================================


class TestSDR:
    """Tests for Successive Difference Replication (SDR) method."""

    def test_sdr_accepted(self, replicate_data):
        """SurveyDesign with replicate_method='SDR' resolves without error."""
        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved = sd.resolve(data)
        assert resolved.replicate_method == "SDR"
        assert resolved.n_replicates == len(rep_cols)

    def test_sdr_variance_factor(self):
        """SDR variance factor is 4/R."""
        assert _replicate_variance_factor("SDR", 80, 0.0) == pytest.approx(4.0 / 80)
        assert _replicate_variance_factor("SDR", 20, 0.0) == pytest.approx(4.0 / 20)
        assert _replicate_variance_factor("SDR", 1, 0.0) == pytest.approx(4.0)

    def test_sdr_4x_brr_scaling(self, replicate_data):
        """SDR variance == 4x BRR variance on same data (4/R vs 1/R)."""
        from diff_diff.linalg import solve_ols

        data, rep_cols = replicate_data
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        sd_brr = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        resolved_brr = sd_brr.resolve(data)
        coef, _, _ = solve_ols(X, y, weights=resolved_brr.weights, weight_type="pweight")
        vcov_brr, _ = compute_replicate_vcov(X, y, coef, resolved_brr)

        sd_sdr = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved_sdr = sd_sdr.resolve(data)
        vcov_sdr, _ = compute_replicate_vcov(X, y, coef, resolved_sdr)

        # SDR uses 4/R, BRR uses 1/R => SDR = 4 * BRR
        np.testing.assert_allclose(np.diag(vcov_sdr), 4.0 * np.diag(vcov_brr))

    def test_sdr_vcov_finite(self, replicate_data):
        """compute_replicate_vcov produces finite results with SDR."""
        from diff_diff.linalg import solve_ols

        data, rep_cols = replicate_data
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved = sd.resolve(data)
        coef, _, _ = solve_ols(X, y, weights=resolved.weights, weight_type="pweight")
        vcov, n_valid = compute_replicate_vcov(X, y, coef, resolved)
        assert np.all(np.isfinite(np.diag(vcov)))
        assert np.all(np.diag(vcov) > 0)
        assert n_valid == len(rep_cols)

    def test_sdr_if_variance_finite(self, replicate_data):
        """compute_replicate_if_variance produces finite results with SDR."""
        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved = sd.resolve(data)
        psi = np.random.randn(len(data)) * 0.1
        var, n_valid = compute_replicate_if_variance(psi, resolved)
        assert np.isfinite(var)
        assert var >= 0
        assert n_valid == len(rep_cols)

    def test_sdr_if_4x_brr(self, replicate_data):
        """SDR IF variance == 4x BRR IF variance."""
        data, rep_cols = replicate_data
        np.random.seed(99)
        psi = np.random.randn(len(data)) * 0.1

        sd_brr = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        resolved_brr = sd_brr.resolve(data)
        var_brr, _ = compute_replicate_if_variance(psi, resolved_brr)

        sd_sdr = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved_sdr = sd_sdr.resolve(data)
        var_sdr, _ = compute_replicate_if_variance(psi, resolved_sdr)

        assert var_sdr == pytest.approx(4.0 * var_brr)

    def test_sdr_rejects_fay_rho(self, replicate_data):
        """SDR with fay_rho > 0 raises ValueError (generic validation)."""
        data, rep_cols = replicate_data
        with pytest.raises(ValueError, match="fay_rho must be 0"):
            SurveyDesign(
                weights="weight",
                replicate_weights=rep_cols,
                replicate_method="SDR",
                fay_rho=0.3,
            )

    def test_sdr_ignores_custom_scales(self, replicate_data):
        """SDR warns and ignores custom replicate_scale/replicate_rscales."""
        from diff_diff.linalg import solve_ols

        data, rep_cols = replicate_data
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        sd_plain = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved_plain = sd_plain.resolve(data)
        coef, _, _ = solve_ols(X, y, weights=resolved_plain.weights, weight_type="pweight")
        vcov_plain, _ = compute_replicate_vcov(X, y, coef, resolved_plain)

        sd_scale = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
            replicate_scale=99.0,
        )
        resolved_scale = sd_scale.resolve(data)
        with pytest.warns(UserWarning, match="ignored for SDR"):
            vcov_scale, _ = compute_replicate_vcov(X, y, coef, resolved_scale)

        # Custom scale should be ignored — results identical
        np.testing.assert_allclose(np.diag(vcov_scale), np.diag(vcov_plain))

    def test_sdr_end_to_end_callaway_santanna(self):
        """CallawaySantAnna + SDR end-to-end smoke test."""
        from diff_diff import CallawaySantAnna

        np.random.seed(42)
        n_units, n_periods = 40, 6
        rows = []
        for i in range(n_units):
            ft = 4 if i < 15 else (0 if i >= 30 else 6)
            wt = 1.0 + 0.2 * (i % 5)
            for t in range(1, n_periods + 1):
                y = 5.0 + i * 0.02 + t * 0.1
                if ft > 0 and t >= ft:
                    y += 1.5
                y += np.random.normal(0, 0.3)
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "first_treat": ft,
                        "outcome": y,
                        "weight": wt,
                    }
                )
        data = pd.DataFrame(rows)

        # Add BRR-style replicate weights (re-used as SDR)
        rng = np.random.RandomState(100)
        n_rep = 16
        rep_cols = []
        units = sorted(data["unit"].unique())
        for r in range(n_rep):
            signs = rng.choice([-1, 1], size=len(units))
            sign_map = dict(zip(units, signs))
            perturbation = data["unit"].map(sign_map).values.astype(float)
            w_r = data["weight"].values * (1.0 + 0.5 * perturbation)
            w_r = np.maximum(w_r, 0.0)
            col = f"sdr_{r}"
            data[col] = w_r
            rep_cols.append(col)

        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )

        est = CallawaySantAnna()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0


# ===========================================================================
# 8b: FPC in ImputationDiD and TwoStageDiD
# ===========================================================================


class TestImputationDiDFPC:
    """FPC support in ImputationDiD."""

    def test_imputation_fpc_no_raise(self, staggered_data):
        """ImputationDiD with FPC runs without error."""
        from diff_diff import ImputationDiD

        data = staggered_data
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        # Add FPC column: population size per stratum (larger than sample)
        n_units = data["unit"].nunique()
        data["fpc_col"] = data["stratum"].map({0: n_units * 3, 1: n_units * 3, 2: n_units * 3})

        est = ImputationDiD(horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_imputation_fpc_reduces_se(self, staggered_data):
        """FPC SE should be <= no-FPC SE (by definition)."""
        from diff_diff import ImputationDiD

        data = staggered_data
        # Large FPC (small correction)
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd_no_fpc = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        sd_fpc = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )

        est = ImputationDiD(horizon_max=3)
        result_no_fpc = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd_no_fpc,
        )
        result_fpc = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd_fpc,
        )
        assert result_fpc.overall_se <= result_no_fpc.overall_se + 1e-10

    def test_imputation_fpc_event_study(self, staggered_data):
        """FPC works with aggregate='event_study'."""
        from diff_diff import ImputationDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        est = ImputationDiD(horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="event_study",
        )
        assert result.event_study_effects is not None
        for h, eff in result.event_study_effects.items():
            assert np.isfinite(eff["se"]), f"Non-finite SE at horizon {h}"

    def test_imputation_fpc_group(self, staggered_data):
        """FPC works with aggregate='group'."""
        from diff_diff import ImputationDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        est = ImputationDiD(horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="group",
        )
        assert result.group_effects is not None
        for g, eff in result.group_effects.items():
            assert np.isfinite(eff["se"]), f"Non-finite SE for group {g}"


class TestTwoStageDiDFPC:
    """FPC support in TwoStageDiD."""

    def test_two_stage_fpc_no_raise(self, staggered_data):
        """TwoStageDiD with FPC runs without error."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        est = TwoStageDiD()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_two_stage_fpc_reduces_se(self, staggered_data):
        """FPC SE should be <= no-FPC SE."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd_no_fpc = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        sd_fpc = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )

        est = TwoStageDiD()
        result_no_fpc = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd_no_fpc,
        )
        result_fpc = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd_fpc,
        )
        assert result_fpc.overall_se <= result_no_fpc.overall_se + 1e-10

    def test_two_stage_fpc_event_study(self, staggered_data):
        """FPC works with aggregate='event_study'."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        est = TwoStageDiD()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="event_study",
        )
        assert result.event_study_effects is not None
        for h, eff in result.event_study_effects.items():
            assert np.isfinite(eff["se"]), f"Non-finite SE at horizon {h}"

    def test_two_stage_fpc_group(self, staggered_data):
        """FPC works with aggregate='group'."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        est = TwoStageDiD()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="group",
        )
        assert result.group_effects is not None
        for g, eff in result.group_effects.items():
            assert np.isfinite(eff["se"]), f"Non-finite SE for group {g}"


class TestUnstratifiedFPC:
    """Regression tests for FPC without explicit strata (P1 fix)."""

    def test_imputation_unstratified_fpc(self, staggered_data):
        """ImputationDiD with weights + psu + fpc (no strata) runs."""
        from diff_diff import ImputationDiD

        data = staggered_data
        data["fpc_col"] = 500  # constant FPC for all

        sd = SurveyDesign(weights="weight", psu="psu", fpc="fpc_col")
        est = ImputationDiD(horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_two_stage_unstratified_fpc(self, staggered_data):
        """TwoStageDiD with weights + psu + fpc (no strata) runs."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = 500

        sd = SurveyDesign(weights="weight", psu="psu", fpc="fpc_col")
        est = TwoStageDiD()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_two_stage_unstratified_fpc_event_study(self, staggered_data):
        """TwoStageDiD unstratified FPC with event study."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = 500

        sd = SurveyDesign(weights="weight", psu="psu", fpc="fpc_col")
        est = TwoStageDiD()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="event_study",
        )
        assert result.event_study_effects is not None
        for h, eff in result.event_study_effects.items():
            assert np.isfinite(eff["se"]), f"Non-finite SE at horizon {h}"


class TestSinglePSUUnstratifiedFPC:
    """Regression: unstratified single-PSU FPC → NaN SE (variance unidentified)."""

    def test_two_stage_single_psu_nan_se(self):
        """TwoStageDiD with 1 PSU and FPC should produce NaN SE."""
        from diff_diff import TwoStageDiD

        # Build data where all units are in one PSU
        np.random.seed(99)
        n_units, n_periods = 30, 6
        rows = []
        for i in range(n_units):
            ft = 4 if i < 10 else (6 if i < 20 else 0)
            for t in range(1, n_periods + 1):
                y = 5.0 + i * 0.02 + t * 0.1
                if ft > 0 and t >= ft:
                    y += 1.5
                y += np.random.normal(0, 0.3)
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "first_treat": ft,
                        "outcome": y,
                        "weight": 1.0,
                        "psu": 0,  # single PSU
                        "fpc_col": 1000,
                    }
                )
        data = pd.DataFrame(rows)

        sd = SurveyDesign(weights="weight", psu="psu", fpc="fpc_col")
        est = TwoStageDiD()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        # Single PSU = variance unidentified → NaN SE
        assert np.isnan(result.overall_se)


class TestPSUMeatHelper:
    """Unit tests for _compute_stratified_meat_from_psu_scores."""

    def test_no_strata_matches_simple(self):
        """Without strata/FPC, helper should match S.T @ S."""
        from diff_diff.survey import _compute_stratified_meat_from_psu_scores

        np.random.seed(42)
        G, k = 20, 3
        S = np.random.randn(G, k) * 0.1
        # All in one stratum, no FPC
        psu_strata = np.zeros(G, dtype=int)
        meat, var_computed, legit_zero = _compute_stratified_meat_from_psu_scores(S, psu_strata)

        # Without FPC: adjustment = (1 - 0) * (G / (G-1)) = G/(G-1)
        expected = (G / (G - 1)) * (S - S.mean(axis=0)).T @ (S - S.mean(axis=0))
        np.testing.assert_allclose(meat, expected, rtol=1e-10)
        assert var_computed is True

    def test_fpc_reduces_meat(self):
        """FPC should reduce the meat matrix vs no-FPC."""
        from diff_diff.survey import _compute_stratified_meat_from_psu_scores

        np.random.seed(42)
        G, k = 20, 3
        S = np.random.randn(G, k) * 0.1
        psu_strata = np.zeros(G, dtype=int)

        meat_no_fpc, _, _ = _compute_stratified_meat_from_psu_scores(S, psu_strata)
        fpc = np.full(G, 100.0)  # 20 out of 100 sampled
        meat_fpc, _, _ = _compute_stratified_meat_from_psu_scores(
            S,
            psu_strata,
            fpc_per_psu=fpc,
        )

        # FPC reduces variance
        assert np.all(np.diag(meat_fpc) <= np.diag(meat_no_fpc) + 1e-15)
        # Not zero (f = 20/100 = 0.2, so 1-f = 0.8)
        assert np.all(np.diag(meat_fpc) > 0)

    def test_lonely_psu_adjust(self):
        """Lonely PSU with adjust contributes nonzero variance."""
        from diff_diff.survey import _compute_stratified_meat_from_psu_scores

        np.random.seed(42)
        G, k = 5, 2
        S = np.random.randn(G, k)
        # Stratum 0 has 1 PSU (lonely), stratum 1 has 4 PSUs
        psu_strata = np.array([0, 1, 1, 1, 1])

        meat_remove, vc_remove, _ = _compute_stratified_meat_from_psu_scores(
            S,
            psu_strata,
            lonely_psu="remove",
        )
        meat_adjust, vc_adjust, _ = _compute_stratified_meat_from_psu_scores(
            S,
            psu_strata,
            lonely_psu="adjust",
        )

        # "adjust" should produce larger variance (lonely PSU contributes)
        assert np.all(np.diag(meat_adjust) >= np.diag(meat_remove) - 1e-15)
        assert vc_adjust is True

    def test_all_singleton_remove_unidentified(self):
        """All singleton strata with remove: variance unidentified."""
        from diff_diff.survey import _compute_stratified_meat_from_psu_scores

        np.random.seed(42)
        G, k = 5, 2
        S = np.random.randn(G, k)
        # Every PSU is in its own stratum (all singletons)
        psu_strata = np.arange(G)

        meat, var_computed, legit_zero = _compute_stratified_meat_from_psu_scores(
            S,
            psu_strata,
            lonely_psu="remove",
        )
        # No variance was computed, no legitimate zeros
        assert var_computed is False
        assert legit_zero == 0
        # Meat is zero matrix (unidentified — caller should produce NaN)
        assert np.all(meat == 0.0)


# ===========================================================================
# 8d: Lonely PSU "adjust" in Bootstrap
# ===========================================================================


class TestBootstrapLonelyPSUAdjust:
    """Tests for lonely_psu='adjust' in survey-aware bootstrap."""

    def _make_survey_data_with_singletons(self):
        """Create staggered panel data with singleton strata for bootstrap tests."""
        np.random.seed(42)
        n_units, n_periods = 40, 6
        rows = []
        for i in range(n_units):
            ft = 4 if i < 15 else (0 if i >= 30 else 6)
            # Stratum 0 has 1 PSU (singleton), strata 1-3 have multiple
            if i == 0:
                stratum = 0
            elif i < 15:
                stratum = 1
            elif i < 30:
                stratum = 2
            else:
                stratum = 3
            for t in range(1, n_periods + 1):
                y = 5.0 + i * 0.02 + t * 0.1
                if ft > 0 and t >= ft:
                    y += 1.5
                y += np.random.normal(0, 0.3)
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "first_treat": ft,
                        "outcome": y,
                        "weight": 1.0 + 0.2 * (i % 5),
                        "stratum": stratum,
                        "psu": i,
                    }
                )
        return pd.DataFrame(rows)

    def test_multiplier_adjust_no_raise(self):
        """Multiplier bootstrap with lonely_psu='adjust' runs without error."""
        from diff_diff import CallawaySantAnna

        data = self._make_survey_data_with_singletons()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            lonely_psu="adjust",
        )
        est = CallawaySantAnna(n_bootstrap=50, seed=42)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_adjust_nonzero_variance(self):
        """With 'adjust', singleton strata contribute nonzero bootstrap variance."""
        from diff_diff.bootstrap_utils import generate_survey_multiplier_weights_batch
        from diff_diff.survey import ResolvedSurveyDesign

        np.random.seed(42)
        n = 20
        # Strata 0 and 1 are singletons (1 PSU each), strata 2 and 3 have multiple
        strata = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        psu = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        n_unique_psu = len(np.unique(psu))
        resolved_remove = ResolvedSurveyDesign(
            weights=np.ones(n),
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=4,
            n_psu=n_unique_psu,
            lonely_psu="remove",
        )
        resolved_adjust = ResolvedSurveyDesign(
            weights=np.ones(n),
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=4,
            n_psu=n_unique_psu,
            lonely_psu="adjust",
        )

        rng = np.random.default_rng(42)
        w_remove, _ = generate_survey_multiplier_weights_batch(
            100,
            resolved_remove,
            "rademacher",
            rng,
        )
        rng = np.random.default_rng(42)
        w_adjust, _ = generate_survey_multiplier_weights_batch(
            100,
            resolved_adjust,
            "rademacher",
            rng,
        )

        # With remove, column 0 (singleton PSU) is all zeros
        assert np.all(w_remove[:, 0] == 0.0)
        # With adjust, column 0 gets nonzero pooled weights
        assert not np.all(w_adjust[:, 0] == 0.0)

    def test_adjust_all_singleton(self):
        """All singleton strata: adjust pools them into one pseudo-stratum."""
        from diff_diff import CallawaySantAnna

        np.random.seed(99)
        n_units, n_periods = 30, 6
        rows = []
        for i in range(n_units):
            ft = 4 if i < 10 else (6 if i < 20 else 0)
            for t in range(1, n_periods + 1):
                y = 5.0 + i * 0.02 + t * 0.1
                if ft > 0 and t >= ft:
                    y += 1.5
                y += np.random.normal(0, 0.3)
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "first_treat": ft,
                        "outcome": y,
                        "weight": 1.0,
                        # Every unit is its own stratum (all singletons)
                        "stratum": i,
                        "psu": i,
                    }
                )
        data = pd.DataFrame(rows)

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            lonely_psu="adjust",
        )
        est = CallawaySantAnna(n_bootstrap=50, seed=42)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0


class TestSingleSingletonAdjust:
    """Single singleton stratum with adjust: warns and contributes zero variance."""

    def test_multiplier_single_singleton_warns(self):
        """Multiplier bootstrap with 1 singleton warns."""
        from diff_diff.bootstrap_utils import generate_survey_multiplier_weights_batch
        from diff_diff.survey import ResolvedSurveyDesign

        np.random.seed(42)
        n = 15
        # Stratum 0 has 1 PSU (singleton), strata 1-2 have multiple
        strata = np.array([0] * 5 + [1] * 5 + [2] * 5)
        psu = np.array([0] * 5 + [1, 2, 3, 4, 5] + [6, 7, 8, 9, 10])

        resolved = ResolvedSurveyDesign(
            weights=np.ones(n),
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=3,
            n_psu=11,
            lonely_psu="adjust",
        )
        rng = np.random.default_rng(42)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            weights, _ = generate_survey_multiplier_weights_batch(
                50,
                resolved,
                "rademacher",
                rng,
            )
            user_warns = [x for x in w if issubclass(x.category, UserWarning)]
            assert any("1 singleton" in str(x.message) for x in user_warns)
        # Singleton column is zero (same as remove)
        assert np.all(weights[:, 0] == 0.0)

    def test_rao_wu_single_singleton_warns(self):
        """Rao-Wu with 1 singleton warns."""
        from diff_diff.bootstrap_utils import generate_rao_wu_weights
        from diff_diff.survey import ResolvedSurveyDesign

        np.random.seed(42)
        n = 15
        strata = np.array([0] * 5 + [1] * 5 + [2] * 5)
        psu = np.array([0] * 5 + [1, 2, 3, 4, 5] + [6, 7, 8, 9, 10])

        resolved = ResolvedSurveyDesign(
            weights=np.ones(n),
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=3,
            n_psu=11,
            lonely_psu="adjust",
        )
        rng = np.random.default_rng(42)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rescaled = generate_rao_wu_weights(resolved, rng)
            user_warns = [x for x in w if issubclass(x.category, UserWarning)]
            assert any("1 singleton" in str(x.message) for x in user_warns)
        # Singleton obs keep base weights (zero variance contribution)
        np.testing.assert_allclose(rescaled[:5], np.ones(5))


# ===========================================================================
class TestRaoWuLonelyPSUAdjust:
    """Tests for lonely_psu='adjust' in Rao-Wu bootstrap (SunAbraham consumer)."""

    def test_rao_wu_adjust_direct(self):
        """Direct test of generate_rao_wu_weights with lonely_psu='adjust'."""
        from diff_diff.bootstrap_utils import generate_rao_wu_weights
        from diff_diff.survey import ResolvedSurveyDesign

        np.random.seed(42)
        n = 20
        # Two singleton strata (0, 1), two multi-PSU strata (2, 3)
        strata = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
        psu = np.array([0] * 5 + [1] * 5 + [2, 3, 4, 5, 6] + [7, 8, 9, 10, 11])

        resolved = ResolvedSurveyDesign(
            weights=np.ones(n),
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=4,
            n_psu=12,
            lonely_psu="adjust",
        )
        # Run multiple draws to check that singleton weights are not constant
        base = resolved.weights.copy()
        any_different = False
        for seed in range(42, 52):
            rng = np.random.default_rng(seed)
            rescaled = generate_rao_wu_weights(resolved, rng)
            # Singleton PSU weights should vary across draws (pooled resampling)
            if not np.allclose(rescaled[:5], base[:5]):
                any_different = True
                break
        assert any_different, "Singleton PSU weights should vary with 'adjust'"


# ===========================================================================
# 8e-i: CV on Estimates
# ===========================================================================


class TestCoefficientOfVariation:
    """Tests for coef_var property on results classes."""

    def test_coef_var_basic(self):
        """Known values: att=2.0, se=0.5 -> CV=0.25."""
        from diff_diff.results import DiDResults

        r = DiDResults(
            att=2.0,
            se=0.5,
            t_stat=4.0,
            p_value=0.0001,
            conf_int=(1.0, 3.0),
            n_obs=100,
            n_treated=50,
            n_control=50,
        )
        assert r.coef_var == pytest.approx(0.25)

    def test_coef_var_zero_estimate(self):
        """ATT=0 -> CV=NaN."""
        from diff_diff.results import DiDResults

        r = DiDResults(
            att=0.0,
            se=0.5,
            t_stat=0.0,
            p_value=1.0,
            conf_int=(-1.0, 1.0),
            n_obs=100,
            n_treated=50,
            n_control=50,
        )
        assert np.isnan(r.coef_var)

    def test_coef_var_zero_se(self):
        """SE=0 with nonzero estimate -> CV=0.0 (e.g., FPC census)."""
        from diff_diff.results import DiDResults

        r = DiDResults(
            att=2.0,
            se=0.0,
            t_stat=np.inf,
            p_value=0.0,
            conf_int=(2.0, 2.0),
            n_obs=100,
            n_treated=50,
            n_control=50,
        )
        assert r.coef_var == 0.0

    def test_coef_var_nan_se(self):
        """NaN SE -> CV=NaN."""
        from diff_diff.results import DiDResults

        r = DiDResults(
            att=2.0,
            se=np.nan,
            t_stat=np.nan,
            p_value=np.nan,
            conf_int=(np.nan, np.nan),
            n_obs=100,
            n_treated=50,
            n_control=50,
        )
        assert np.isnan(r.coef_var)

    def test_coef_var_in_summary(self):
        """CV appears in summary output."""
        from diff_diff.results import DiDResults

        r = DiDResults(
            att=2.0,
            se=0.5,
            t_stat=4.0,
            p_value=0.0001,
            conf_int=(1.0, 3.0),
            n_obs=100,
            n_treated=50,
            n_control=50,
        )
        assert "CV" in r.summary()

    def test_coef_var_overall_att_pattern(self):
        """CV works on overall_att/overall_se pattern (CallawaySantAnna)."""
        from diff_diff import CallawaySantAnna

        np.random.seed(42)
        n_units, n_periods = 30, 6
        rows = []
        for i in range(n_units):
            ft = 4 if i < 10 else (6 if i < 20 else 0)
            for t in range(1, n_periods + 1):
                y = 5.0 + i * 0.02 + t * 0.1
                if ft > 0 and t >= ft:
                    y += 1.5
                y += np.random.normal(0, 0.3)
                rows.append({"unit": i, "time": t, "first_treat": ft, "outcome": y})
        data = pd.DataFrame(rows)

        result = CallawaySantAnna().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        cv = result.coef_var
        assert np.isfinite(cv)
        assert cv == pytest.approx(result.overall_se / abs(result.overall_att))


# ===========================================================================
# 8e-ii: Weight Trimming
# ===========================================================================


class TestTrimWeights:
    """Tests for trim_weights utility."""

    def test_trim_upper(self):
        """Upper cap trims large weights."""
        from diff_diff.prep import trim_weights

        data = pd.DataFrame({"w": [1.0, 2.0, 5.0, 10.0, 20.0]})
        result = trim_weights(data, "w", upper=5.0)
        assert result["w"].max() == 5.0
        assert result["w"].tolist() == [1.0, 2.0, 5.0, 5.0, 5.0]

    def test_trim_quantile(self):
        """Quantile-based trimming."""
        from diff_diff.prep import trim_weights

        np.random.seed(42)
        data = pd.DataFrame({"w": np.random.exponential(1.0, 1000)})
        result = trim_weights(data, "w", quantile=0.95)
        q95 = np.quantile(data["w"].values, 0.95)
        assert result["w"].max() == pytest.approx(q95)

    def test_trim_lower(self):
        """Lower floor raises small weights."""
        from diff_diff.prep import trim_weights

        data = pd.DataFrame({"w": [0.01, 0.1, 1.0, 5.0]})
        result = trim_weights(data, "w", lower=0.5)
        assert result["w"].min() == 0.5

    def test_trim_returns_copy(self):
        """Original DataFrame is not modified."""
        from diff_diff.prep import trim_weights

        data = pd.DataFrame({"w": [1.0, 10.0, 100.0]})
        original_max = data["w"].max()
        _ = trim_weights(data, "w", upper=5.0)
        assert data["w"].max() == original_max

    def test_trim_upper_and_quantile_raises(self):
        """Both upper and quantile raises ValueError."""
        from diff_diff.prep import trim_weights

        data = pd.DataFrame({"w": [1.0, 2.0]})
        with pytest.raises(ValueError, match="upper.*quantile"):
            trim_weights(data, "w", upper=5.0, quantile=0.95)

    def test_trim_quantile_with_nan(self):
        """Quantile trimming with NaN weights doesn't poison the column."""
        from diff_diff.prep import trim_weights

        data = pd.DataFrame({"w": [1.0, 2.0, np.nan, 5.0, 10.0]})
        result = trim_weights(data, "w", quantile=0.9)
        # NaN stays NaN, non-NaN values are trimmed
        assert np.isnan(result["w"].iloc[2])
        assert np.isfinite(result["w"].iloc[0])
        assert np.isfinite(result["w"].iloc[4])

    def test_trim_lower_exceeds_upper_raises(self):
        """lower > upper raises ValueError."""
        from diff_diff.prep import trim_weights

        data = pd.DataFrame({"w": [1.0, 5.0, 10.0]})
        with pytest.raises(ValueError, match="lower.*<=.*upper"):
            trim_weights(data, "w", upper=5.0, lower=6.0)


# ===========================================================================
# 8e-iii: ImputationDiD Pretrends + Survey
# ===========================================================================


class TestImputationPretrendsSurvey:
    """Tests for pretrends + survey support in ImputationDiD."""

    def test_pretrends_survey_no_raise(self, staggered_data):
        """ImputationDiD with pretrends=True + survey runs without error."""
        from diff_diff import ImputationDiD

        data = staggered_data
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = ImputationDiD(pretrends=True, horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="event_study",
        )
        assert result.event_study_effects is not None
        # Should have negative-horizon (pre-period) effects
        pre_horizons = [h for h in result.event_study_effects if h < 0]
        assert len(pre_horizons) > 0

    def test_pretrends_survey_finite_se(self, staggered_data):
        """Pre-period lead SEs are finite with survey design."""
        from diff_diff import ImputationDiD

        data = staggered_data
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = ImputationDiD(pretrends=True, horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="event_study",
        )
        for h, eff in result.event_study_effects.items():
            if h < -1:  # Skip reference period (h=-1)
                assert np.isfinite(eff["se"]), f"Non-finite SE at pre-horizon {h}"
                assert eff["se"] > 0, f"Zero SE at pre-horizon {h}"

    def test_pretrend_test_survey_no_raise(self, staggered_data):
        """pretrend_test() works with survey design."""
        from diff_diff import ImputationDiD

        data = staggered_data
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = ImputationDiD(pretrends=True, horizon_max=3)
        est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="event_study",
        )
        pt = est._pretrend_test()
        assert "f_stat" in pt
        assert "p_value" in pt
        if pt["n_leads"] > 0:
            assert np.isfinite(pt["f_stat"])
            assert np.isfinite(pt["p_value"])

    def test_pretrends_survey_with_covariates(self, staggered_data):
        """Survey pretrends with covariates uses the survey+covariate path."""
        from diff_diff import ImputationDiD

        data = staggered_data
        data["x1"] = np.random.default_rng(42).normal(0, 1, len(data))
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = ImputationDiD(pretrends=True, horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1"],
            survey_design=sd,
            aggregate="event_study",
        )
        pre = [h for h in result.event_study_effects if h < -1]
        for h in pre:
            assert np.isfinite(result.event_study_effects[h]["se"])

    def test_pretrends_survey_with_bootstrap(self, staggered_data):
        """Survey pretrends + bootstrap: bootstrap doesn't overwrite pre-period SEs."""
        from diff_diff import ImputationDiD

        data = staggered_data
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = ImputationDiD(pretrends=True, horizon_max=3, n_bootstrap=20, seed=42)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="event_study",
        )
        pre = [h for h in result.event_study_effects if h < -1]
        for h in pre:
            # Pre-period SEs come from lead regression, not bootstrap
            assert np.isfinite(result.event_study_effects[h]["se"])

    def test_pretrends_survey_always_treated_psu(self):
        """Survey pretrends with a PSU/stratum that has no untreated obs."""
        from diff_diff import ImputationDiD

        np.random.seed(77)
        n_periods = 8
        rows = []
        for i in range(40):
            if i < 10:
                ft = 1  # always-treated: treated from period 1 (zero untreated rows)
            elif i < 25:
                ft = 5  # late cohort
            else:
                ft = 0  # never-treated
            # PSU 0-9 are in stratum 0 (all always-treated → zero untreated obs)
            # PSU 10-39 are in strata 1-2 (mix of treated/never-treated)
            stratum = 0 if i < 10 else (1 if i < 25 else 2)
            for t in range(1, n_periods + 1):
                y = 5.0 + i * 0.02 + t * 0.1
                if ft > 0 and t >= ft:
                    y += 1.5
                y += np.random.normal(0, 0.3)
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "first_treat": ft,
                        "outcome": y,
                        "weight": 1.0 + 0.2 * (i % 5),
                        "stratum": stratum,
                        "psu": i,
                    }
                )
        data = pd.DataFrame(rows)

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = ImputationDiD(pretrends=True, horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="event_study",
        )
        # Should produce finite results even though stratum 0 has
        # mostly treated PSUs (sparse untreated domain)
        assert np.isfinite(result.overall_att)
        pre = [h for h in result.event_study_effects if h < -1]
        for h in pre:
            assert np.isfinite(result.event_study_effects[h]["se"])

    def test_pretrends_replicate_raises(self, staggered_data):
        """pretrends=True + replicate-weight survey raises NotImplementedError."""
        from diff_diff import ImputationDiD

        data = staggered_data
        # Add BRR replicate weights
        rng = np.random.RandomState(42)
        units = sorted(data["unit"].unique())
        rep_cols = []
        for r in range(10):
            signs = rng.choice([-1, 1], size=len(units))
            sign_map = dict(zip(units, signs))
            pert = data["unit"].map(sign_map).values.astype(float)
            w_r = data["weight"].values * (1.0 + 0.5 * pert)
            w_r = np.maximum(w_r, 0.0)
            col = f"brr_{r}"
            data[col] = w_r
            rep_cols.append(col)

        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        est = ImputationDiD(pretrends=True, horizon_max=3)
        with pytest.raises(NotImplementedError, match="replicate"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sd,
                aggregate="event_study",
            )

    def test_pretrend_test_replicate_raises(self, staggered_data):
        """pretrend_test() with replicate-weight survey raises NotImplementedError."""
        from diff_diff import ImputationDiD

        data = staggered_data
        rng = np.random.RandomState(42)
        units = sorted(data["unit"].unique())
        rep_cols = []
        for r in range(10):
            signs = rng.choice([-1, 1], size=len(units))
            sign_map = dict(zip(units, signs))
            pert = data["unit"].map(sign_map).values.astype(float)
            w_r = data["weight"].values * (1.0 + 0.5 * pert)
            w_r = np.maximum(w_r, 0.0)
            col = f"brr_{r}"
            data[col] = w_r
            rep_cols.append(col)

        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        # fit() without pretrends succeeds
        est = ImputationDiD(horizon_max=3)
        est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        # But pretrend_test() should reject replicate designs
        with pytest.raises(NotImplementedError, match="replicate"):
            est._pretrend_test()
