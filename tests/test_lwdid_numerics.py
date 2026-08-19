"""Numerical precision and edge case tests for LWDiD."""

import time
import warnings

import numpy as np
import pandas as pd

from diff_diff import LWDiD, LWDiDResults

# ─── Data Helpers ───────────────────────────────────────────────────────────


def _make_common_timing_panel(
    n_treated=30,
    n_control=50,
    n_pre=5,
    n_post=3,
    true_att=2.0,
    seed=42,
):
    """Generate balanced common-timing panel with known ATT."""
    rng = np.random.default_rng(seed)
    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    rows = []
    for i in range(n_units):
        is_treated = i < n_treated
        unit_fe = rng.normal(0, 1)
        for t in range(1, n_periods + 1):
            time_trend = 0.3 * t
            noise = rng.normal(0, 0.5)
            post = 1 if t > n_pre else 0
            treat = 1 if (is_treated and post) else 0
            y = unit_fe + time_trend + noise + (true_att if treat else 0)
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "y": y,
                    "treat": treat,
                }
            )
    return pd.DataFrame(rows)


def _make_large_panel(n_units=1000, n_periods=20, seed=42):
    """Large panel for performance testing."""
    rng = np.random.default_rng(seed)
    n_treated = n_units // 3
    n_pre = n_periods // 2

    unit_ids = np.repeat(np.arange(n_units), n_periods)
    time_ids = np.tile(np.arange(1, n_periods + 1), n_units)

    is_treated = (unit_ids < n_treated).astype(float)
    is_post = (time_ids > n_pre).astype(float)
    treat = is_treated * is_post

    # Unit FEs + time trend + noise + treatment effect
    unit_fes = rng.normal(0, 2, size=n_units)
    y = unit_fes[unit_ids] + 0.3 * time_ids + rng.normal(0, 0.5, size=len(unit_ids)) + 2.0 * treat

    return pd.DataFrame(
        {
            "unit": unit_ids,
            "time": time_ids,
            "y": y,
            "treat": treat.astype(int),
        }
    )


# ─── Hand-Computed ATT Tests ───────────────────────────────────────────────


class TestLWDiDHandComputed:
    """Tests where ATT can be computed by hand."""

    def test_hand_computed_att_3units(self):
        """3 units, 4 periods, hand-computable ATT.

        Unit 0 (control): y = [1, 2, 3, 4], pre_mean = 1.5
            demeaned post: [3-1.5, 4-1.5] = [1.5, 2.5] → avg = 2.0
        Unit 1 (control): y = [2, 4, 6, 8], pre_mean = 3
            demeaned post: [6-3, 8-3] = [3, 5] → avg = 4.0
        Unit 2 (treated): y = [1, 3, 10, 12], pre_mean = 2
            demeaned post: [10-2, 12-2] = [8, 10] → avg = 9.0

        Cross-section: control_mean = (2.0 + 4.0)/2 = 3.0
                       treated_mean = 9.0
        ATT = 9.0 - 3.0 = 6.0

        But RA is y = alpha + tau*D, so:
        Intercept = mean of controls = 3.0
        tau = mean(treated) - mean(controls) = 9.0 - 3.0 = 6.0
        """
        df = pd.DataFrame(
            {
                "unit": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                "y": [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 10.0, 12.0],
                "treat": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            }
        )
        res = LWDiD(rolling="demean", estimation_method="reg", vcov_type="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        np.testing.assert_allclose(res.att, 6.0, atol=1e-10)

    def test_hand_computed_att_zero_effect(self):
        """When treatment effect is exactly 0, ATT should be ~0.

        Both treated and controls have same DGP: y = unit_fe + t.
        """
        df = pd.DataFrame(
            {
                "unit": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "y": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0],
                "treat": [0, 0, 0, 0, 0, 0, 0, 0, 1],
            }
        )
        res = LWDiD(rolling="demean", estimation_method="reg", vcov_type="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # All units have pre_mean = 1.5, 2.5, 3.5
        # Post demeaned: control = [3-1.5, 4-2.5] = [1.5, 1.5] avg=1.5
        # Treated: 5-3.5 = 1.5
        # ATT = 1.5 - 1.5 = 0
        np.testing.assert_allclose(res.att, 0.0, atol=1e-10)

    def test_detrend_perfect_linear_zero_effect(self):
        """Perfect linear trend, no treatment effect → ATT = 0.

        All units follow y = a_i + b_i * t with no treatment effect.
        After detrending, residuals are 0 everywhere.
        """
        df = pd.DataFrame(
            {
                "unit": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                "y": [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 0.0, 1.0, 2.0, 3.0],
                "treat": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            }
        )
        res = LWDiD(rolling="detrend", estimation_method="reg", vcov_type="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        np.testing.assert_allclose(res.att, 0.0, atol=1e-10)

    def test_detrend_with_known_effect(self):
        """Linear trend + constant treatment effect.

        Controls: y = a_i + t (perfectly linear)
        Treated: y = a_i + t in pre, y = a_i + t + 3 in post
        After detrend, control residuals = 0, treated residuals = 3.
        ATT = 3 - 0 = 3.
        """
        df = pd.DataFrame(
            {
                "unit": [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4,
                "time": [1, 2, 3, 4] * 4,
                "y": [
                    2.0,
                    3.0,
                    4.0,
                    5.0,  # control 0: y = 1 + t
                    3.0,
                    4.0,
                    5.0,
                    6.0,  # control 1: y = 2 + t
                    4.0,
                    5.0,
                    6.0,
                    7.0,  # control 2: y = 3 + t
                    2.0,
                    3.0,
                    7.0,
                    8.0,  # treated: y = 1 + t + 3*post
                ],
                "treat": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
            }
        )
        res = LWDiD(rolling="detrend", estimation_method="reg", vcov_type="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        np.testing.assert_allclose(res.att, 3.0, atol=1e-10)


# ─── Numerical Precision Tests ──────────────────────────────────────────────


class TestLWDiDNumericalPrecision:
    """Test numerical stability with challenging data configurations."""

    def test_collinear_controls_handled(self):
        """Rank-deficient design matrix should not crash."""
        panel = _make_common_timing_panel(seed=11)
        # Add duplicate (unit-constant) control column
        rng = np.random.default_rng(11)
        units = panel["unit"].unique()
        xmap = dict(zip(units, rng.normal(size=len(units))))
        panel["x1"] = panel["unit"].map(xmap)
        panel["x2"] = panel["x1"]  # perfectly collinear

        # Should produce a result (possibly with warning), not crash
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(estimation_method="reg").fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                covariates=["x1", "x2"],
            )
        assert np.isfinite(res.att)

    def test_near_singular_design(self):
        """Near-singular design should still produce finite estimate."""
        rng = np.random.default_rng(22)
        panel = _make_common_timing_panel(seed=22)
        # Add nearly collinear (unit-constant) controls
        units = panel["unit"].unique()
        xmap = dict(zip(units, rng.normal(size=len(units))))
        emap = dict(zip(units, rng.normal(0, 1e-8, size=len(units))))
        panel["x1"] = panel["unit"].map(xmap)
        panel["x2"] = panel["x1"] + panel["unit"].map(emap)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(estimation_method="reg").fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                covariates=["x1", "x2"],
            )
        assert np.isfinite(res.att)

    def test_zero_variance_outcome_handled(self):
        """Constant outcome should be handled gracefully."""
        df = pd.DataFrame(
            {
                "unit": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "y": [5.0] * 9,  # constant outcome
                "treat": [0, 0, 0, 0, 0, 0, 0, 0, 1],
            }
        )
        # Should not crash; ATT should be 0 or NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", vcov_type="classical").fit(
                df, outcome="y", unit="unit", time="time", treatment="treat"
            )
        # With constant outcome, demeaned values are all 0, ATT = 0
        assert res.att == 0.0 or np.isnan(res.att)

    def test_single_treated_unit(self):
        """Only 1 treated unit should still produce a result."""
        df = pd.DataFrame(
            {
                "unit": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "y": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 2.0, 8.0],
                "treat": [0, 0, 0, 0, 0, 0, 0, 0, 1],
            }
        )
        res = LWDiD(rolling="demean", vcov_type="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert isinstance(res, LWDiDResults)
        assert np.isfinite(res.att)
        assert res.n_treated == 1

    def test_large_outcome_values(self):
        """Large outcome values should not cause overflow."""
        panel = _make_common_timing_panel(seed=33)
        panel["y"] = panel["y"] * 1e8

        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)

    def test_small_outcome_values(self):
        """Small outcome values should not underflow."""
        panel = _make_common_timing_panel(seed=44)
        panel["y"] = panel["y"] * 1e-8

        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isfinite(res.att)

    def test_negative_outcomes(self):
        """Negative outcomes should work fine."""
        panel = _make_common_timing_panel(seed=55)
        panel["y"] = panel["y"] - 100  # shift all negative

        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isfinite(res.att)
        # ATT should still be positive (shift doesn't affect demeaned values)
        assert res.att > 0


# ─── Performance Tests ──────────────────────────────────────────────────────


class TestLWDiDPerformance:
    """Test that estimation completes in reasonable time."""

    def test_large_panel_performance(self):
        """1000 units × 20 periods should complete in reasonable time."""
        panel = _make_large_panel(n_units=1000, n_periods=20)
        start = time.time()
        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        elapsed = time.time() - start
        assert elapsed < 30  # Should complete in < 30 seconds
        assert np.isfinite(res.att)

    def test_moderate_staggered_performance(self):
        """200 units × 10 periods staggered should be fast."""
        from tests.test_lwdid import _make_staggered_panel

        panel = _make_staggered_panel(n_units=200, n_periods=10, seed=77)
        start = time.time()
        res = LWDiD(control_group="never_treated").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat", first_treat="cohort"
        )
        elapsed = time.time() - start
        assert elapsed < 30
        assert np.isfinite(res.att)


# ─── VCE Consistency Tests ──────────────────────────────────────────────────


class TestLWDiDVCEConsistency:
    """Test variance-covariance estimation properties."""

    def test_hc1_se_positive(self):
        """HC1 SE must be strictly positive when ATT is identified."""
        panel = _make_common_timing_panel()
        res = LWDiD(vcov_type="hc1").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.se > 0

    def test_cluster_se_invariant_to_row_order(self):
        """Shuffling rows should not change cluster-robust SE."""
        panel = _make_common_timing_panel(seed=66)
        panel["cluster_id"] = panel["unit"] % 10

        # Fit on original order
        res1 = LWDiD(cluster="cluster_id").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )

        # Shuffle rows
        panel_shuffled = panel.sample(frac=1, random_state=99).reset_index(drop=True)
        res2 = LWDiD(cluster="cluster_id").fit(
            panel_shuffled,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )

        np.testing.assert_allclose(res1.att, res2.att, atol=1e-12)
        np.testing.assert_allclose(res1.se, res2.se, atol=1e-12)

    def test_vcov_symmetric(self):
        """VCE matrix must be symmetric."""
        panel = _make_common_timing_panel()
        res = LWDiD(vcov_type="hc1").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        if res.vcov is not None:
            np.testing.assert_allclose(res.vcov, res.vcov.T, atol=1e-14)

    def test_vcov_positive_semidefinite(self):
        """VCE matrix diagonal should be non-negative."""
        panel = _make_common_timing_panel()
        res = LWDiD(vcov_type="hc1").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        if res.vcov is not None:
            diag = np.diag(res.vcov)
            assert np.all(diag >= -1e-15)  # allow small numerical error

    def test_se_consistent_with_vcov(self):
        """SE should equal sqrt(vcov[1,1]) for the treatment coefficient."""
        panel = _make_common_timing_panel()
        res = LWDiD(vcov_type="hc1", estimation_method="reg").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        if res.vcov is not None:
            expected_se = np.sqrt(max(res.vcov[1, 1], 0.0))
            np.testing.assert_allclose(res.se, expected_se, atol=1e-14)


# ─── Determinism Tests ──────────────────────────────────────────────────────


class TestLWDiDDeterminism:
    """Test that results are deterministic (same input → same output)."""

    def test_same_data_same_result(self):
        """Running twice on same data gives identical results."""
        panel = _make_common_timing_panel(seed=42)

        res1 = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        res2 = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")

        assert res1.att == res2.att
        assert res1.se == res2.se
        assert res1.t_stat == res2.t_stat

    def test_copy_invariance(self):
        """Deep copy of data should give same results."""
        panel = _make_common_timing_panel(seed=42)
        panel_copy = panel.copy(deep=True)

        res1 = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        res2 = LWDiD().fit(panel_copy, outcome="y", unit="unit", time="time", treatment="treat")

        assert res1.att == res2.att
        assert res1.se == res2.se


# ─── Multiple Post-Period Aggregation ───────────────────────────────────────


class TestLWDiDPostPeriodAggregation:
    """Test that multiple post-periods are correctly averaged."""

    def test_single_post_period(self):
        """Single post period = no averaging needed."""
        panel = _make_common_timing_panel(n_pre=5, n_post=1, seed=42)
        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isfinite(res.att)

    def test_many_post_periods(self):
        """Many post periods should be averaged correctly."""
        panel = _make_common_timing_panel(n_pre=3, n_post=10, seed=42)
        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isfinite(res.att)
        assert res.att > 0  # True ATT = 2.0

    def test_more_pre_than_post(self):
        """Many pre periods, few post."""
        panel = _make_common_timing_panel(n_pre=10, n_post=2, seed=42)
        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isfinite(res.att)
        assert res.att > 0
