"""Tests for LWDiD estimator (Lee & Wooldridge 2025, 2026)."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import LWDiD, LWDiDResults

# ─── Test Data Generators ───────────────────────────────────────────────────


def _make_common_timing_panel(
    n_treated=30,
    n_control=50,
    n_pre=5,
    n_post=3,
    true_att=2.0,
    seed=42,
):
    """Generate balanced common-timing panel with known ATT.

    Pre-treatment periods: 1..n_pre (treatment=0 for all)
    Post-treatment periods: n_pre+1..n_pre+n_post (treatment=1 for treated)
    """
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


def _make_staggered_panel(
    n_units=120,
    n_periods=10,
    n_cohorts=3,
    true_att=1.5,
    seed=42,
):
    """Generate staggered adoption panel with multiple cohorts.

    Cohort assignment:
    - First ~1/4 units: never-treated (cohort=0)
    - Remaining units split across n_cohorts with treatment times spread.
    """
    rng = np.random.default_rng(seed)
    n_never = n_units // 4
    n_per_cohort = (n_units - n_never) // n_cohorts

    # Cohort adoption times (spread across middle periods)
    cohort_times = [3 + i * 2 for i in range(n_cohorts)]

    rows = []
    uid = 0
    for i in range(n_never):
        unit_fe = rng.normal(0, 1)
        for t in range(1, n_periods + 1):
            y = unit_fe + 0.2 * t + rng.normal(0, 0.5)
            rows.append(
                {
                    "unit": uid,
                    "time": t,
                    "y": y,
                    "treat": 0,
                    "cohort": 0,
                }
            )
        uid += 1

    for c_idx, g in enumerate(cohort_times):
        for i in range(n_per_cohort):
            unit_fe = rng.normal(0, 1)
            for t in range(1, n_periods + 1):
                post = 1 if t >= g else 0
                treat = post  # treated once cohort adopts
                effect = true_att * post
                y = unit_fe + 0.2 * t + rng.normal(0, 0.5) + effect
                rows.append(
                    {
                        "unit": uid,
                        "time": t,
                        "y": y,
                        "treat": treat,
                        "cohort": g,
                    }
                )
            uid += 1

    return pd.DataFrame(rows)


# ─── Parameter Interface Tests ──────────────────────────────────────────────


class TestLWDiDParams:
    """Test parameter setting, getting, and validation."""

    def test_get_params_returns_all(self):
        est = LWDiD(rolling="demean", estimation_method="reg", vcov_type="hc1")
        params = est.get_params()
        assert "rolling" in params
        assert "estimation_method" in params
        assert "vcov_type" in params
        assert "control_group" in params
        assert "alpha" in params
        assert "n_bootstrap" in params
        assert params["rolling"] == "demean"
        assert params["estimation_method"] == "reg"
        assert params["vcov_type"] == "hc1"

    def test_set_params_modifies(self):
        est = LWDiD()
        est.set_params(rolling="detrend")
        assert est.rolling == "detrend"

    def test_set_params_returns_self(self):
        est = LWDiD()
        ret = est.set_params(estimation_method="ipw")
        assert ret is est

    def test_invalid_rolling_raises(self):
        with pytest.raises(ValueError, match="rolling"):
            LWDiD(rolling="invalid")

    def test_invalid_estimation_method_raises(self):
        with pytest.raises(ValueError, match="estimation_method"):
            LWDiD(estimation_method="invalid")

    def test_invalid_vcov_type_raises(self):
        with pytest.raises(ValueError, match="vcov_type"):
            LWDiD(vcov_type="invalid")

    def test_invalid_control_group_raises(self):
        with pytest.raises(ValueError, match="control_group"):
            LWDiD(control_group="invalid")

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            LWDiD(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            LWDiD(alpha=1.0)

    def test_invalid_n_bootstrap_raises(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            LWDiD(n_bootstrap=-1)

    def test_LW_alias_removed(self):
        import diff_diff

        assert not hasattr(diff_diff, "LW")
        assert "LW" not in diff_diff.__all__

    def test_default_params(self):
        est = LWDiD()
        assert est.rolling == "demean"
        assert est.estimation_method == "reg"
        assert est.vcov_type == "hc1"
        assert est.control_group == "not_yet_treated"
        assert est.alpha == 0.05
        assert est.n_bootstrap == 0

    def test_repr(self):
        est = LWDiD(rolling="demean", estimation_method="reg")
        r = repr(est)
        assert "LWDiD" in r
        assert "demean" in r
        assert "reg" in r

    def test_set_params_invalid_key_raises(self):
        est = LWDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(bad_param="x")


# ─── Input Validation Tests ─────────────────────────────────────────────────


class TestLWDiDInputValidation:
    """Test input data validation."""

    def test_missing_column_raises(self):
        df = pd.DataFrame({"unit": [1], "time": [1], "y": [1.0]})
        with pytest.raises(ValueError, match="Columns not found"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_nan_in_outcome_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, np.nan, 2.0, 3.0],
                "treat": [0, 1, 0, 0],
            }
        )
        with pytest.raises(ValueError, match="missing values"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_nan_in_treatment_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 2.0, 3.0],
                "treat": [0, np.nan, 0, 0],
            }
        )
        with pytest.raises(ValueError, match="missing values"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_duplicate_unit_time_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2],
                "time": [1, 1, 2, 1],
                "y": [1.0, 1.5, 2.0, 3.0],
                "treat": [0, 0, 1, 0],
            }
        )
        with pytest.raises(ValueError, match="duplicate"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_non_binary_treatment_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0],
                "treat": [0, 2, 0, 0],  # not binary
            }
        )
        with pytest.raises(ValueError):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_vcov_type_cluster_rejected(self):
        with pytest.raises(ValueError, match="cluster"):
            LWDiD(vcov_type="cluster")

    def test_no_treated_units_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0],
                "treat": [0, 0, 0, 0],
            }
        )
        with pytest.raises(ValueError, match="[Nn]o treated|[Nn]o post"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_no_control_units_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0],
                "treat": [0, 1, 0, 1],
            }
        )
        with pytest.raises(ValueError, match="[Nn]o control"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")


# ─── Treatment Design Validation Tests ──────────────────────────────────────


def _make_design_panel(cohort_map, n_periods=5, seed=7):
    """Small panel (len(cohort_map) units x n_periods) with D_it = 1[t >= g_i].

    cohort_map: {unit_id: g} with g=0 for never-treated. Returns columns
    unit/time/y/treat/cohort so tests can freely corrupt treat or cohort.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for uid, g in cohort_map.items():
        for t in range(1, n_periods + 1):
            treat = int(g > 0 and t >= g)
            rows.append(
                {
                    "unit": uid,
                    "time": t,
                    "y": rng.normal(0, 0.5) + 1.5 * treat,
                    "treat": treat,
                    "cohort": g,
                }
            )
    return pd.DataFrame(rows)


class TestTreatmentDesignValidation:
    """Unified vectorized design checks (_check_treatment_design)."""

    @staticmethod
    def _cohorts(n_treated_3=5, n_treated_4=5, n_never=10):
        cohorts = {}
        uid = 0
        for _ in range(n_treated_3):
            cohorts[uid] = 3
            uid += 1
        for _ in range(n_treated_4):
            cohorts[uid] = 4
            uid += 1
        for _ in range(n_never):
            cohorts[uid] = 0
            uid += 1
        return cohorts

    # ── (a) absorbing treatment ──

    def test_non_absorbing_common_timing_raises(self):
        panel = _make_design_panel({u: (3 if u < 8 else 0) for u in range(20)})
        # unit 0 switches back to 0 at the last period
        panel.loc[(panel["unit"] == 0) & (panel["time"] == 5), "treat"] = 0
        with pytest.raises(ValueError, match="Non-absorbing"):
            LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")

    def test_non_absorbing_staggered_raises(self):
        panel = _make_design_panel(self._cohorts())
        panel.loc[(panel["unit"] == 0) & (panel["time"] == 5), "treat"] = 0
        with pytest.raises(ValueError, match="Non-absorbing"):
            LWDiD().fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
            )

    def test_non_absorbing_unsorted_input_raises(self):
        """Detection must not depend on the input row order."""
        panel = _make_design_panel({u: (3 if u < 8 else 0) for u in range(20)})
        panel.loc[(panel["unit"] == 0) & (panel["time"] == 4), "treat"] = 0
        shuffled = panel.sample(frac=1.0, random_state=0).reset_index(drop=True)
        with pytest.raises(ValueError, match="Non-absorbing"):
            LWDiD().fit(shuffled, outcome="y", unit="unit", time="time", treatment="treat")

    # ── (b) common timing: unique onset ──

    def test_heterogeneous_onset_without_cohort_raises(self):
        cohorts = {u: 3 for u in range(5)}
        cohorts.update({u: 4 for u in range(5, 10)})
        cohorts.update({u: 0 for u in range(10, 20)})
        panel = _make_design_panel(cohorts)
        with pytest.raises(ValueError, match="heterogeneous first-treatment"):
            LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")

    def test_common_timing_valid_passes(self):
        panel = _make_design_panel({u: (3 if u < 8 else 0) for u in range(20)})
        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isfinite(res.att)

    # ── (c) staggered: onset == cohort ──

    def test_onset_cohort_mismatch_raises(self):
        panel = _make_design_panel(self._cohorts())
        # unit 0 (cohort 3) starts treatment one period early
        panel.loc[(panel["unit"] == 0) & (panel["time"] == 2), "treat"] = 1
        with pytest.raises(ValueError, match="inconsistent with cohort"):
            LWDiD().fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
            )

    def test_never_treated_with_treatment_rows_raises(self):
        panel = _make_design_panel(self._cohorts())
        # unit 19 is never-treated by cohort but has a treatment=1 row
        panel.loc[(panel["unit"] == 19) & (panel["time"] == 5), "treat"] = 1
        with pytest.raises(ValueError, match="inconsistent with cohort"):
            LWDiD().fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
            )

    def test_cohort_in_window_never_switching_on_raises(self):
        panel = _make_design_panel(self._cohorts())
        # unit 0 keeps cohort=3 but never actually switches on
        panel.loc[panel["unit"] == 0, "treat"] = 0
        with pytest.raises(ValueError, match="no\\s+treatment=1 rows"):
            LWDiD().fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
            )

    def test_staggered_valid_passes(self):
        panel = _make_design_panel(self._cohorts())
        res = LWDiD(control_group="never_treated").fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert np.isfinite(res.att)

    def test_staggered_nan_cohort_never_treated_passes(self):
        """Never-treated encoded as NaN cohort is a valid design."""
        panel = _make_design_panel(self._cohorts())
        panel["cohort"] = panel["cohort"].replace(0, np.nan)
        res = LWDiD(control_group="never_treated").fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert np.isfinite(res.att)

    def test_cohort_beyond_window_recoded_to_never_treated(self):
        """Beyond-window cohorts are recoded to never-treated by the
        normalizer (with a warning), then pass the design check as
        never-treated units. The design check itself now documents a
        normalized-input precondition, so direct callers normalize first.
        """
        from diff_diff.lwdid import _check_treatment_design, _normalize_cohorts

        cohorts = self._cohorts()
        cohorts[0] = 9  # beyond n_periods=5: all treat rows are 0
        panel = _make_design_panel(cohorts)
        with pytest.warns(UserWarning, match="exceed the last observed period"):
            panel["cohort"], n_inf, n_beyond = _normalize_cohorts(
                panel["cohort"], max_time=panel["time"].max()
            )
        assert n_inf == 0 and n_beyond > 0
        assert (panel.loc[panel["unit"] == 0, "cohort"] == 0).all()
        # Must not raise: the recoded unit is never-treated with no D=1 rows
        _check_treatment_design(panel, "unit", "time", "treat", "cohort")


# ─── Transformation Tests ───────────────────────────────────────────────────


class TestLWDiDTransformations:
    """Test that rolling transformations are correctly applied."""

    def test_demean_subtracts_pre_mean(self):
        """Construct simple 2-unit panel where pre-mean is known."""
        # Unit 0 (control): y = [2, 4, 6] → pre_mean = 3
        # Unit 1 (treated): y = [1, 3, 10] → pre_mean = 2
        df = pd.DataFrame(
            {
                "unit": [0, 0, 0, 1, 1, 1],
                "time": [1, 2, 3, 1, 2, 3],
                "y": [2.0, 4.0, 6.0, 1.0, 3.0, 10.0],
                "treat": [0, 0, 0, 0, 0, 1],
            }
        )
        res = LWDiD(rolling="demean", estimation_method="reg").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # The method demeaned using pre-treatment periods (time 1,2)
        # Unit 0: pre_mean = 3, post (time 3) ydot = 6-3 = 3
        # Unit 1: pre_mean = 2, post (time 3) ydot = 10-2 = 8
        # ATT = 8 - 3 = 5 (treatment effect + any trend difference)
        assert isinstance(res, LWDiDResults)
        assert np.isfinite(res.att)

    def test_detrend_removes_linear_trend(self):
        """Construct unit with perfect linear trend y = 1 + 2*t.

        After detrend, residuals should be ~0 in pre-period.
        """
        # Need at least 2 pre periods for detrend
        # Unit 0 (control): y = 1 + 2*t for all t
        # Unit 1 (treated): y = 1 + 2*t in pre, + 5 in post
        df = pd.DataFrame(
            {
                "unit": [0, 0, 0, 0, 1, 1, 1, 1],
                "time": [1, 2, 3, 4, 1, 2, 3, 4],
                "y": [3.0, 5.0, 7.0, 9.0, 3.0, 5.0, 12.0, 14.0],
                "treat": [0, 0, 0, 0, 0, 0, 1, 1],
            }
        )
        res = LWDiD(rolling="detrend", estimation_method="reg").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert isinstance(res, LWDiDResults)
        # Detrended control should be ~0, detrended treated should show effect
        assert res.att > 0

    def test_transform_preserves_treatment_effect(self):
        """After demean, the treatment effect should still be visible."""
        panel = _make_common_timing_panel(true_att=5.0, seed=123)
        res = LWDiD(rolling="demean", estimation_method="reg").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # True ATT is 5.0, estimate should be positive and in range
        assert res.att > 2.0


# ─── Common Timing Tests ────────────────────────────────────────────────────


class TestLWDiDCommonTiming:
    """Test common-timing estimation paths."""

    @pytest.fixture
    def panel(self):
        return _make_common_timing_panel(true_att=2.0)

    def test_ra_returns_results(self, panel):
        est = LWDiD(rolling="demean", estimation_method="reg")
        res = est.fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert isinstance(res, LWDiDResults)

    def test_ra_demean_positive_att(self, panel):
        res = LWDiD(rolling="demean", estimation_method="reg").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.att > 0  # True ATT is 2.0

    def test_ra_detrend_positive_att(self, panel):
        res = LWDiD(rolling="detrend", estimation_method="reg").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.att > 0

    def test_ra_att_close_to_truth(self, panel):
        """RA demean should recover ATT near 2.0 with enough data."""
        res = LWDiD(rolling="demean", estimation_method="reg").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # Allow generous tolerance due to small sample noise
        assert 0.5 < res.att < 4.0

    def test_ipw_positive_att(self, panel):
        """IPW needs controls for propensity score."""
        panel_with_x = panel.copy()
        rng = np.random.default_rng(0)
        units = panel_with_x["unit"].unique()
        xmap = dict(zip(units, rng.normal(size=len(units))))
        panel_with_x["x1"] = panel_with_x["unit"].map(xmap)  # unit-constant
        res = LWDiD(rolling="demean", estimation_method="ipw").fit(
            panel_with_x,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            covariates=["x1"],
        )
        assert res.att > 0

    def test_dr_positive_att(self, panel):
        """DR (doubly robust) should recover positive ATT."""
        panel_with_x = panel.copy()
        rng = np.random.default_rng(0)
        units = panel_with_x["unit"].unique()
        xmap = dict(zip(units, rng.normal(size=len(units))))
        panel_with_x["x1"] = panel_with_x["unit"].map(xmap)  # unit-constant
        res = LWDiD(rolling="demean", estimation_method="dr").fit(
            panel_with_x,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            covariates=["x1"],
        )
        assert res.att > 0

    def test_hc1_se_positive(self, panel):
        res = LWDiD(vcov_type="hc1").fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert res.se > 0

    def test_classical_se_positive(self, panel):
        res = LWDiD(vcov_type="classical").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.se > 0

    def test_cluster_robust_se(self, panel):
        """Cluster-robust SE should be positive."""
        # Create a cluster variable (group units into clusters)
        panel_cl = panel.copy()
        panel_cl["cluster_id"] = panel_cl["unit"] % 10
        res = LWDiD(cluster="cluster_id").fit(
            panel_cl, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.se > 0

    def test_n_obs_n_treated_n_control(self, panel):
        """Sample sizes should be consistent."""
        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert res.n_treated == 30
        assert res.n_control == 50
        assert res.n_obs == 80

    def test_result_not_staggered(self, panel):
        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert not res.is_staggered
        assert res.cohort_effects is None

    def test_params_stored(self, panel):
        """RA should store coefficient vector."""
        res = LWDiD(rolling="demean", estimation_method="reg").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.params is not None
        assert len(res.params) >= 2  # intercept + treatment

    def test_vcov_stored(self, panel):
        """RA should store vcov matrix."""
        res = LWDiD(rolling="demean", estimation_method="reg").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.vcov is not None
        assert res.vcov.shape[0] == res.vcov.shape[1]

    def test_controls_improve_precision(self):
        """Adding relevant controls should reduce SE (most cases)."""
        rng = np.random.default_rng(99)
        panel = _make_common_timing_panel(n_treated=50, n_control=100, seed=99)
        # Add control correlated with outcome
        unit_map = {}
        for uid in panel["unit"].unique():
            unit_map[uid] = rng.normal(0, 2)
        panel["x_corr"] = panel["unit"].map(unit_map)

        res_no_ctrl = LWDiD(estimation_method="reg").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        res_ctrl = LWDiD(estimation_method="reg").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat", covariates=["x_corr"]
        )
        # Both should produce finite results
        assert np.isfinite(res_no_ctrl.se)
        assert np.isfinite(res_ctrl.se)


# ─── Staggered Design Tests ─────────────────────────────────────────────────


class TestLWDiDStaggered:
    """Test staggered adoption designs."""

    @pytest.fixture
    def stag_panel(self):
        return _make_staggered_panel(true_att=1.5)

    def test_staggered_never_treated(self, stag_panel):
        res = LWDiD(control_group="never_treated").fit(
            stag_panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert isinstance(res, LWDiDResults)
        assert res.cohort_effects is not None

    def test_staggered_not_yet_treated(self, stag_panel):
        res = LWDiD(control_group="not_yet_treated").fit(
            stag_panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert res.att is not None
        assert np.isfinite(res.att)

    def test_cohort_effects_populated(self, stag_panel):
        res = LWDiD().fit(
            stag_panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert res.cohort_effects is not None
        assert len(res.cohort_effects) > 0

    def test_staggered_att_positive(self, stag_panel):
        """Overall ATT should be positive (true_att=1.5)."""
        res = LWDiD(control_group="never_treated").fit(
            stag_panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert res.att > 0

    def test_staggered_is_staggered(self, stag_panel):
        res = LWDiD().fit(
            stag_panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert res.is_staggered

    def test_staggered_se_positive(self, stag_panel):
        res = LWDiD(control_group="never_treated").fit(
            stag_panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert res.se > 0

    def test_staggered_detrend(self, stag_panel):
        """Detrend should also work for staggered."""
        res = LWDiD(rolling="detrend", control_group="never_treated").fit(
            stag_panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert isinstance(res, LWDiDResults)
        assert res.att > 0

    def test_staggered_cluster_equals_unit_column(self, stag_panel):
        """Regression: cluster= the unit column must not raise KeyError.

        The unit column is consumed by set_index inside the staggered
        engine, so looking it up as a regular column used to crash when
        cluster == unit (the most common by-unit clustering spelling).
        """
        res = LWDiD(cluster="unit", control_group="never_treated").fit(
            stag_panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0

        # An explicit copy of the unit column under a different name must
        # give exactly the same estimates.
        copied = stag_panel.copy()
        copied["cluster_id"] = copied["unit"]
        res_copy = LWDiD(cluster="cluster_id", control_group="never_treated").fit(
            copied,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert res.att == res_copy.att
        assert res.se == res_copy.se

    def test_no_treated_cohorts_raises(self):
        """All cohort=0 should raise."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0],
                "treat": [0, 0, 0, 0],
                "cohort": [0, 0, 0, 0],
            }
        )
        with pytest.raises(ValueError, match="[Nn]o treated cohort"):
            LWDiD().fit(
                df, outcome="y", unit="unit", time="time", treatment="treat", first_treat="cohort"
            )

    def test_never_treated_required_when_specified(self):
        """control_group='never_treated' requires at least one cohort=0 unit."""
        # All units are in cohort 3 (treated)
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2, 3, 3],
                "time": [1, 2, 1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "treat": [0, 1, 0, 1, 0, 0],
                "cohort": [2, 2, 2, 2, 3, 3],
            }
        )
        with pytest.raises(ValueError, match="never-treated"):
            LWDiD(control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", treatment="treat", first_treat="cohort"
            )


# ─── Results Container Tests ────────────────────────────────────────────────


class TestLWDiDResults:
    """Test the LWDiDResults dataclass interface."""

    @pytest.fixture
    def result(self):
        panel = _make_common_timing_panel()
        return LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")

    def test_inference_consistency(self, result):
        """t_stat ≈ att / se."""
        if result.se > 0 and np.isfinite(result.se):
            np.testing.assert_allclose(result.t_stat, result.att / result.se, rtol=1e-10)

    def test_conf_int_bounds(self, result):
        """CI should bracket ATT."""
        lo, hi = result.conf_int
        assert lo < result.att < hi

    def test_conf_int_symmetric(self, result):
        """CI should be symmetric around ATT (normal-based)."""
        lo, hi = result.conf_int
        half_width_lo = result.att - lo
        half_width_hi = hi - result.att
        np.testing.assert_allclose(half_width_lo, half_width_hi, rtol=1e-10)

    def test_p_value_range(self, result):
        """p-value should be in [0, 1]."""
        assert 0 <= result.p_value <= 1

    def test_summary_contains_fields(self, result):
        s = result.summary()
        assert "ATT" in s or "att" in s.lower()
        assert "LWDiD" in s

    def test_to_dataframe(self, result):
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        assert "att" in df.columns

    def test_to_dict_serializable(self, result):
        """to_dict() should produce JSON-serializable output."""
        d = result.to_dict()
        json.dumps(d, default=str)

    def test_to_dict_contains_keys(self, result):
        d = result.to_dict()
        assert "att" in d
        assert "se" in d
        assert "rolling" in d
        assert "estimation_method" in d

    def test_repr_informative(self, result):
        r = repr(result)
        assert "LWDiDResults" in r
        assert "ATT" in r

    def test_rolling_metadata(self, result):
        assert result.rolling == "demean"
        assert result.estimation_method == "reg"
        assert result.vcov_type == "hc1"
        assert result.alpha == 0.05

    def test_nan_inference_when_se_zero(self):
        """Direct construction with se=0 should give NaN inference."""
        res = LWDiDResults(
            att=1.0,
            se=0.0,
            t_stat=float("nan"),
            p_value=float("nan"),
            conf_int=(float("nan"), float("nan")),
            n_obs=100,
            n_treated=30,
            n_control=70,
            rolling="demean",
            estimation_method="reg",
            vcov_type="hc1",
            alpha=0.05,
        )
        assert np.isnan(res.t_stat)
        assert np.isnan(res.p_value)
        assert np.isnan(res.conf_int[0])
        assert np.isnan(res.conf_int[1])


# ─── Different VCE Comparisons ──────────────────────────────────────────────


class TestLWDiDVCEComparisons:
    """Compare VCE methods produce different but finite SEs."""

    @pytest.fixture
    def panel(self):
        return _make_common_timing_panel(n_treated=40, n_control=80, seed=77)

    def test_hc1_vs_classical(self, panel):
        res_cl = LWDiD(vcov_type="classical").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        res_hc1 = LWDiD(vcov_type="hc1").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # ATTs should be the same (same point estimate)
        np.testing.assert_allclose(res_cl.att, res_hc1.att, atol=1e-12)
        # SEs differ
        assert res_cl.se > 0
        assert res_hc1.se > 0

    def test_cluster_vs_hc1(self, panel):
        panel_cl = panel.copy()
        panel_cl["cluster_id"] = panel_cl["unit"] % 10
        res_hc1 = LWDiD(vcov_type="hc1").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        res_cl = LWDiD(cluster="cluster_id").fit(
            panel_cl, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # Point estimates should be identical
        np.testing.assert_allclose(res_hc1.att, res_cl.att, atol=1e-12)
        # Both SEs positive
        assert res_cl.se > 0
        assert res_hc1.se > 0


# ─── Estimator Consistency Tests ────────────────────────────────────────────


class TestLWDiDEstimatorConsistency:
    """Test that different estimators produce consistent results."""

    @pytest.fixture
    def panel_with_controls(self):
        panel = _make_common_timing_panel(n_treated=50, n_control=100, seed=55)
        rng = np.random.default_rng(55)
        units = panel["unit"].unique()
        xmap = dict(zip(units, rng.normal(size=len(units))))
        panel["x1"] = panel["unit"].map(xmap)  # unit-constant
        return panel

    def test_ra_ipw_same_sign(self, panel_with_controls):
        """RA and IPW should give same-sign ATT."""
        res_ra = LWDiD(estimation_method="reg").fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            covariates=["x1"],
        )
        res_ipw = LWDiD(estimation_method="ipw").fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            covariates=["x1"],
        )
        assert np.sign(res_ra.att) == np.sign(res_ipw.att)

    def test_reg_dr_same_sign(self, panel_with_controls):
        """Regression adjustment and DR should give same-sign ATT."""
        res_ra = LWDiD(estimation_method="reg").fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            covariates=["x1"],
        )
        res_dr = LWDiD(estimation_method="dr").fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            covariates=["x1"],
        )
        assert np.sign(res_ra.att) == np.sign(res_dr.att)

    def test_ipw_without_controls_warns(self):
        """IPW without controls should warn and behave like RA."""
        panel = _make_common_timing_panel(seed=88)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = LWDiD(estimation_method="ipw").fit(
                panel, outcome="y", unit="unit", time="time", treatment="treat"
            )
            # Should produce a warning about no controls
            ipw_warnings = [x for x in w if "IPW" in str(x.message)]
            assert len(ipw_warnings) > 0
        assert np.isfinite(res.att)


# ─── Cohort-Time Cell Support (issue #734) ──────────────────────────────────


def _make_eligibility_panel(seed=7):
    """Staggered panel with distinct cohort sizes.

    Sizes are distinct so a cell's control count identifies the eligible
    pool uniquely: never-treated 2, cohort 3 has 4 units, cohort 8 has 3,
    cohort 10 has 5.
    """
    sizes = {0: 2, 3: 4, 8: 3, 10: 5}
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g, size in sizes.items():
        for _ in range(size):
            unit_fe = rng.normal()
            for t in range(1, 13):
                treated = g > 0 and t >= g
                rows.append(
                    {
                        "unit": uid,
                        "time": t,
                        "cohort": g,
                        "treat": int(treated),
                        "y": (unit_fe + 0.3 * t + rng.normal(0, 0.5) + (2.0 if treated else 0.0)),
                    }
                )
            uid += 1
    return pd.DataFrame(rows), sizes


def _make_trend_only_panel(shift=None):
    """The issue #734 reproduction: a pure common time trend, zero effect.

    Cohort 3 (5 units) and cohort 5 (5 units) over t = 1..6, plus two
    never-treated units observed only through t = 4. No control is
    available from t = 5 on: cohort 3 loses every control there and
    cohort 5 never has a post-treatment control.
    """
    rows = []
    for unit in range(12):
        cohort = 3 if unit < 5 else (5 if unit < 10 else 0)
        last_period = 4 if cohort == 0 else 6
        for time in range(1, last_period + 1):
            y = float(time)
            if shift is not None:
                y += shift(time)
            rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "cohort": cohort,
                    "treat": int(cohort > 0 and time >= cohort),
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def _fit_trend_only(data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = LWDiD(
            rolling="demean",
            estimation_method="reg",
            vcov_type="classical",
            control_group="not_yet_treated",
        ).fit(
            data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
    return res, [str(x.message) for x in caught]


class TestCohortTimeCellSupport:
    """Per-(g, t) cells with calendar-time-specific control eligibility.

    The estimand is built from cohort-time cells whose control pool is
    A_{g,t} = {G = g} u {G = 0} u {G > max(g, t)} (LW 2026 Sec. 7). Applying
    eligibility as a unit-level filter and then averaging each unit's
    transformed outcomes over unequal calendar windows produces a non-zero
    ATT under a pure common time trend, which is the defect these tests pin.
    """

    def test_later_cohort_eligibility_is_period_specific(self):
        """A later cohort is a valid control at r = 3 but not at r = 5."""
        panel, sizes = _make_eligibility_panel()
        res = LWDiD(
            rolling="demean",
            estimation_method="reg",
            vcov_type="classical",
            control_group="not_yet_treated",
        ).fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        cells = res.cohort_time_effects

        # r = 3 is calendar t = 6: cohorts 8 and 10 are both still untreated.
        at_r3 = cells[(3, 6)]
        assert at_r3["n_treated"] == sizes[3]
        assert at_r3["n_control"] == sizes[0] + sizes[8] + sizes[10]

        # r = 5 is calendar t = 8: cohort 8 is treated by then and drops out.
        at_r5 = cells[(3, 8)]
        assert at_r5["n_treated"] == sizes[3]
        assert at_r5["n_control"] == sizes[0] + sizes[10]

        # By t = 10 only the never-treated remain eligible.
        assert cells[(3, 10)]["n_control"] == sizes[0]

    def test_eligibility_matches_formula_for_every_cell(self):
        """Every cohort-3 cell's control count equals |A_{3,t}| - |G = 3|."""
        panel, sizes = _make_eligibility_panel()
        res = LWDiD(
            rolling="demean",
            estimation_method="reg",
            vcov_type="classical",
            control_group="not_yet_treated",
        ).fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        for t in range(3, 13):
            expected = sizes[0] + sum(size for g, size in sizes.items() if g > 0 and g > max(3, t))
            assert res.cohort_time_effects[(3, t)]["n_control"] == expected, t

    def test_common_time_trend_yields_zero_att(self):
        """A pure time trend with no treatment effect must estimate zero."""
        res, _ = _fit_trend_only(_make_trend_only_panel())
        assert abs(res.att) < 1e-10

    @pytest.mark.parametrize(
        "shift",
        [
            lambda t: 100.0,
            lambda t: 0.5 * t**2,
            lambda t: (-1.0) ** t * 3.0,
        ],
        ids=["level", "quadratic", "sawtooth"],
    )
    def test_common_time_shift_leaves_att_unchanged(self, shift):
        """Adding any time-only h(t) to every unit cannot move the ATT."""
        base, _ = _fit_trend_only(_make_trend_only_panel())
        shifted, _ = _fit_trend_only(_make_trend_only_panel(shift=shift))
        assert abs(shifted.att - base.att) < 1e-10

    def test_unsupported_cells_are_reported(self):
        """Cells with an empty control pool are recorded and warned about."""
        res, messages = _fit_trend_only(_make_trend_only_panel())

        # Cohort 3 keeps no controls from t = 5 onward.
        for t in (5, 6):
            cell = res.cohort_time_effects[(3, t)]
            assert cell["skip_reason"] == "zero_treated_control"
            assert cell["inference_status"] == "not_estimable"
            assert np.isnan(cell["att"])

        assert any("skipped" in m and "unsupported" in m for m in messages)

    def test_cohort_without_any_supported_cell_is_dropped(self):
        """Cohort 5 has no eligible post-treatment control and is dropped."""
        res, _ = _fit_trend_only(_make_trend_only_panel())
        assert 5 not in res.cohort_effects
        assert all(
            res.cohort_time_effects[key]["skip_reason"] == "zero_treated_control"
            for key in res.cohort_time_effects
            if key[0] == 5 and key[1] >= 5
        )

    def test_degenerate_standard_errors_are_not_reported(self):
        """An exactly-fitting design must not report a ~0 SE as inference."""
        res, messages = _fit_trend_only(_make_trend_only_panel())
        assert np.isnan(res.se)
        assert np.isnan(res.p_value)
        assert res.inference_basis == "unavailable_degenerate_cells"
        assert any("degenerate or non-finite standard error" in m for m in messages)

    def test_supported_design_reports_joint_influence_inference(self):
        """A well-identified staggered panel still gets finite inference."""
        panel, _ = _make_eligibility_panel()
        res = LWDiD(
            rolling="demean",
            estimation_method="reg",
            vcov_type="hc1",
            control_group="not_yet_treated",
        ).fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert res.inference_basis == "joint_influence_function"
        assert np.isfinite(res.se) and res.se > 0
        assert res.att == pytest.approx(2.0, abs=0.5)


# ─── Joint Influence-Function Inference (issue #735) ────────────────────────


def _cluster_sums(values, ids):
    frame = pd.DataFrame({"value": values, "cluster": ids})
    return frame.groupby("cluster", sort=False)["value"].sum().to_numpy()


def _make_shared_control_panel(seed=101, n_never=40, per_cohort=20, cohorts=(5, 7, 9)):
    """Staggered panel whose cohorts all draw on the same never-treated pool."""
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g in (0,) + tuple(cohorts):
        size = n_never if g == 0 else per_cohort
        for _ in range(size):
            unit_fe = rng.normal()
            for t in range(1, 13):
                treated = g > 0 and t >= g
                rows.append(
                    {
                        "unit": uid,
                        "time": t,
                        "cohort": g,
                        "treat": int(treated),
                        "y": (unit_fe + 0.2 * t + rng.normal(0, 0.7) + (1.5 if treated else 0.0)),
                    }
                )
            uid += 1
    return pd.DataFrame(rows)


class TestInfluenceFunctionReconciliation:
    """Each estimator returns the influence function behind its own SE.

    Cohort effects that share control units are not independent, so the
    staggered aggregation combines per-cell influence functions rather than
    summing marginal variances. That is only sound if a single cell's
    influence function reproduces that cell's standard error exactly, which
    is the identity pinned here: the contributions are the estimator's own
    asymptotically linear representation reweighted by the variance
    estimator, not a proxy rescaled to hit a target.
    """

    @pytest.fixture(scope="class")
    def sample(self):
        rng = np.random.default_rng(11)
        n = 200
        controls = rng.normal(size=(n, 2))
        index = 0.6 * controls[:, 0] - 0.4 * controls[:, 1]
        treatment = (rng.uniform(size=n) < 1 / (1 + np.exp(-index))).astype(float)
        y = 1.0 + 2.0 * treatment + controls @ np.array([0.5, -0.3]) + rng.normal(0, 1.2, size=n)
        clusters = rng.integers(0, 12, size=n)
        return y, treatment, controls, clusters, n

    @pytest.mark.parametrize("estimation_method", ["reg", "ipw", "dr"])
    @pytest.mark.parametrize("vcov", ["classical", "hc1", "hc2", "hc3", "cluster"])
    def test_influence_reproduces_standard_error(self, sample, estimation_method, vcov):
        y, treatment, controls, clusters, n = sample
        cluster_ids = clusters if vcov == "cluster" else None
        if vcov == "cluster":
            est = LWDiD(estimation_method=estimation_method, cluster="cl")
        else:
            est = LWDiD(estimation_method=estimation_method, vcov_type=vcov)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _att, se, _, _, _, influence = getattr(est, f"_estimate_{estimation_method}")(
                y, treatment, controls, cluster_ids, n
            )
        assert influence is not None
        effective = influence if cluster_ids is None else _cluster_sums(influence, cluster_ids)
        assert float(np.sqrt(np.sum(effective**2))) == pytest.approx(se, rel=1e-10)

    @pytest.mark.parametrize("vcov", ["classical", "hc1", "hc2", "hc3", "cluster"])
    def test_influence_reproduces_standard_error_without_controls(self, sample, vcov):
        """The regression design matrix drops the interaction block without controls."""
        y, treatment, _controls, clusters, n = sample
        cluster_ids = clusters if vcov == "cluster" else None
        if vcov == "cluster":
            est = LWDiD(estimation_method="reg", cluster="cl")
        else:
            est = LWDiD(estimation_method="reg", vcov_type=vcov)
        _att, se, _, _, _, influence = est._estimate_reg(y, treatment, None, cluster_ids, n)
        effective = influence if cluster_ids is None else _cluster_sums(influence, cluster_ids)
        assert float(np.sqrt(np.sum(effective**2))) == pytest.approx(se, rel=1e-10)

    def test_matching_reports_no_influence_function(self, sample):
        """PSM has no influence-function representation and must say so."""
        y, treatment, controls, _clusters, n = sample
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            *_, influence = LWDiD(estimation_method="psm")._estimate_psm(
                y, treatment, controls, None, n
            )
        assert influence is None

    def test_staggered_psm_reports_unavailable_basis(self):
        """Overall PSM inference is NaN rather than an independence guess."""
        panel = _make_shared_control_panel(per_cohort=15, n_never=30)
        rng = np.random.default_rng(5)
        x_by_unit = pd.Series(
            rng.normal(size=panel["unit"].nunique()), index=panel["unit"].unique()
        )
        panel["x1"] = panel["unit"].map(x_by_unit)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = LWDiD(
                rolling="demean",
                estimation_method="psm",
                control_group="never_treated",
            ).fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
                covariates=["x1"],
            )
        assert res.inference_basis == "unavailable_matching"
        assert np.isnan(res.se)
        assert any("matching" in str(w.message) for w in caught)


class TestStaggeredJointInference:
    """Overall staggered inference accounts for shared control units.

    Cohorts estimated against a common never-treated pool are positively
    correlated. Summing marginal cohort variances therefore understates the
    overall standard error; combining influence functions does not.
    """

    @pytest.fixture(scope="class")
    def fitted(self):
        panel = _make_shared_control_panel()
        res = LWDiD(
            rolling="demean",
            estimation_method="reg",
            vcov_type="hc1",
            control_group="never_treated",
        ).fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        return panel, res

    def test_reports_joint_influence_basis(self, fitted):
        _panel, res = fitted
        assert res.inference_basis == "joint_influence_function"
        assert np.isfinite(res.se) and res.se > 0

    def test_wider_than_independence_assumption(self, fitted):
        """The independence formula is the specific thing being corrected."""
        _panel, res = fitted
        cohort_se = np.array([v["se"] for v in res.cohort_effects.values()])
        weights = np.array([v["weight"] for v in res.cohort_effects.values()])
        independence_se = float(np.sqrt(np.sum(weights**2 * cohort_se**2)))
        assert res.se > independence_se

    @pytest.mark.slow
    def test_matches_unit_cluster_bootstrap(self, fitted, ci_params):
        """Concordance with a unit-level bootstrap, which needs no
        independence assumption. The independence formula misses by ~24% on
        this design; the joint influence function lands within 10%."""
        panel, res = fitted
        units = panel["unit"].unique()
        blocks = {u: g for u, g in panel.groupby("unit")}
        rng = np.random.default_rng(2024)
        draws = []
        for _ in range(ci_params.bootstrap(300, min_n=60)):
            picked = rng.choice(units, size=len(units), replace=True)
            frames = []
            for new_id, u in enumerate(picked):
                block = blocks[u].copy()
                block["unit"] = new_id
                frames.append(block)
            sample = pd.concat(frames, ignore_index=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    att = (
                        LWDiD(
                            rolling="demean",
                            estimation_method="reg",
                            vcov_type="hc1",
                            control_group="never_treated",
                        )
                        .fit(
                            sample,
                            outcome="y",
                            unit="unit",
                            time="time",
                            treatment="treat",
                            first_treat="cohort",
                        )
                        .att
                    )
                except ValueError:
                    continue
            if np.isfinite(att):
                draws.append(att)

        bootstrap_se = float(np.std(np.array(draws), ddof=1))
        assert bootstrap_se == pytest.approx(res.se, rel=0.10)


# ─── Post-Fit Aggregation Contract (issues #732, #733) ──────────────────────


class TestAggregationContract:
    """``aggregate()`` reports the fit; it never re-derives inference.

    A staggered fit already chooses an inference basis - the composite
    regression where the paper's theory applies, joint influence functions
    otherwise. Recomputing an overall ATT from marginal cohort effects would
    substitute a cohort-independence assumption for that basis and quietly
    report a different standard error for the same estimand.
    """

    @pytest.fixture(scope="class")
    def staggered(self):
        return _make_shared_control_panel(n_never=30, per_cohort=15)

    @pytest.fixture(scope="class")
    def composite_fit(self, staggered):
        """The composite-regression path (never-treated + RA + classical)."""
        return LWDiD(
            rolling="demean",
            estimation_method="reg",
            vcov_type="classical",
            control_group="never_treated",
        ).fit(
            staggered,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )

    def test_uses_composite_regression(self, composite_fit):
        assert composite_fit.inference_basis == "composite_regression"

    def test_simple_preserves_the_fitted_result(self, composite_fit):
        """Exact agreement, including the finite-sample degrees of freedom."""
        agg = composite_fit.aggregate("simple")
        assert agg.att[0] == composite_fit.att
        assert agg.se[0] == composite_fit.se
        assert agg.t_stat[0] == composite_fit.t_stat
        assert agg.p_value[0] == composite_fit.p_value
        assert agg.conf_int_lower[0] == composite_fit.conf_int[0]
        assert agg.conf_int_upper[0] == composite_fit.conf_int[1]
        assert agg.df[0] == composite_fit.df_inference
        assert agg.alpha == composite_fit.alpha

    @pytest.mark.parametrize("vcov", ["classical", "hc1"])
    @pytest.mark.parametrize("control_group", ["never_treated", "not_yet_treated"])
    def test_simple_preserves_every_inference_basis(self, staggered, vcov, control_group):
        """Holds off the composite path too, not just where it is gated on."""
        res = LWDiD(
            rolling="demean",
            estimation_method="reg",
            vcov_type=vcov,
            control_group=control_group,
        ).fit(
            staggered,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        agg = res.aggregate("simple")
        assert agg.att[0] == res.att
        assert agg.se[0] == res.se
        if res.df_inference is None:
            assert np.isnan(agg.df[0])
        else:
            assert agg.df[0] == res.df_inference

    def test_group_reports_cohort_effects_with_weights(self, composite_fit):
        agg = composite_fit.aggregate("group")
        assert agg.level == "group"
        assert list(agg.label) == list(composite_fit.cohort_effects)
        assert agg.weight is not None
        assert float(np.nansum(agg.weight)) == pytest.approx(1.0)
        for i, cohort in enumerate(agg.label):
            assert agg.att[i] == composite_fit.cohort_effects[cohort]["att"]

    def test_group_dataframe_matches_shared_schema(self, composite_fit):
        from diff_diff.aggregation import AGGREGATION_SCHEMA

        frame = composite_fit.aggregate("group").to_dataframe()
        assert tuple(frame.columns) == AGGREGATION_SCHEMA

    def test_event_study_returns_shared_container(self, staggered):
        from diff_diff.results_base import EVENT_STUDY_SCHEMA, EventStudyResults

        res = LWDiD(rolling="demean", estimation_method="reg", n_bootstrap=199, seed=7).fit(
            staggered,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        es = res.aggregate("event_study")
        assert isinstance(es, EventStudyResults)
        frame = es.to_dataframe()
        assert tuple(frame.columns) == EVENT_STUDY_SCHEMA

        # The anchor period is carried as a reference row, not dropped.
        assert list(frame.loc[frame["is_reference"], "event_time"]) == [-1]
        assert frame.loc[frame["is_reference"], "att"].tolist() == [0.0]
        assert es.cband_lower is not None
        assert es.cband_crit_value > 0

    def test_event_study_serialises_through_to_dict(self, staggered):
        res = LWDiD(rolling="demean", estimation_method="reg", n_bootstrap=199, seed=7).fit(
            staggered,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        payload = res.to_dict()
        assert payload["reference_periods"] == [-1]
        assert payload["cband_method"] == "multiplier_bootstrap_sup_t"
        assert payload["cband_n_bootstrap"] == 199
        assert payload["inference_basis"] == res.inference_basis
        assert set(payload["event_study_effects"]) == {str(r) for r in res.event_study_effects}

    def test_unsupported_type_names_the_supported_set(self, composite_fit):
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            composite_fit.aggregate("overall")
        with pytest.raises(ValueError, match="'simple', 'event_study', 'group'"):
            composite_fit.aggregate("calendar")

    def test_weights_selector_is_rejected(self, composite_fit):
        with pytest.raises(ValueError, match="does not accept a weights selector"):
            composite_fit.aggregate("simple", weights="cell")

    def test_balance_e_is_rejected_off_event_study(self, composite_fit):
        with pytest.raises(ValueError, match="balance_e"):
            composite_fit.aggregate("simple", balance_e=2)

    def test_common_timing_aggregate_simple_and_group(self):
        """Guard relaxation: simple relays the fit on common timing, while
        group still raises (there is no cohort dimension)."""
        panel = _make_common_timing_panel(seed=3)
        res = LWDiD(rolling="demean").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        agg = res.aggregate("simple")
        frame = agg.to_dataframe()
        assert frame["att"].iloc[0] == res.att
        assert frame["se"].iloc[0] == res.se
        with pytest.raises(ValueError, match="only available for staggered"):
            res.aggregate("group")

    def test_fit_time_aggregate_is_gone(self):
        """Aggregation is post-fit only: fit() no longer takes aggregate."""
        panel = _make_shared_control_panel(n_never=20, per_cohort=10)
        with pytest.raises(TypeError, match="aggregate"):
            LWDiD(rolling="demean").fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
                aggregate="group",
            )


# ─── PR #588 review: statistical-core fixes ────────────────────────────────────


class TestClassicalJointInference:
    """Classical joint covariance is built from residual-based influence.

    The former ``sigma * basis`` contributions gave every shared control
    unit a non-zero cross-cell product regardless of its actual outcome
    draw, fabricating correlation between cohort-time cells and inflating
    the classical joint SE roughly two-fold against a unit-level bootstrap.
    """

    @pytest.fixture(scope="class")
    def fitted(self):
        panel = _make_shared_control_panel(seed=303)
        res = LWDiD(
            rolling="demean",
            estimation_method="reg",
            vcov_type="classical",
            control_group="not_yet_treated",
        ).fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        return panel, res

    def test_reports_joint_influence_basis(self, fitted):
        _panel, res = fitted
        assert res.inference_basis == "joint_influence_function"
        assert np.isfinite(res.se) and res.se > 0

    @pytest.mark.slow
    def test_matches_unit_level_bootstrap(self, fitted, ci_params):
        """Shared not-yet-treated controls: the classical joint/overall SE
        must agree with a unit-level bootstrap that assumes no independence.
        The sigma * basis contributions missed by ~2x on this design."""
        panel, res = fitted
        units = panel["unit"].unique()
        blocks = {u: g for u, g in panel.groupby("unit")}
        rng = np.random.default_rng(588)
        draws = []
        for _ in range(ci_params.bootstrap(400, min_n=60)):
            picked = rng.choice(units, size=len(units), replace=True)
            frames = []
            for new_id, u in enumerate(picked):
                block = blocks[u].copy()
                block["unit"] = new_id
                frames.append(block)
            sample = pd.concat(frames, ignore_index=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    att = (
                        LWDiD(
                            rolling="demean",
                            estimation_method="reg",
                            vcov_type="classical",
                            control_group="not_yet_treated",
                        )
                        .fit(
                            sample,
                            outcome="y",
                            unit="unit",
                            time="time",
                            treatment="treat",
                            first_treat="cohort",
                        )
                        .att
                    )
                except ValueError:
                    continue
            if np.isfinite(att):
                draws.append(att)

        bootstrap_se = float(np.std(np.array(draws), ddof=1))
        assert res.se == pytest.approx(bootstrap_se, rel=0.3)

    def test_event_study_simultaneous_band_is_sane(self):
        """The sup-t band exists and is at least as wide as pointwise CIs."""
        panel = _make_shared_control_panel(seed=303)
        res = LWDiD(
            rolling="demean",
            estimation_method="reg",
            vcov_type="classical",
            control_group="not_yet_treated",
            n_bootstrap=199,
            seed=7,
        ).fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        assert res.cband_method == "multiplier_bootstrap_sup_t"
        assert np.isfinite(res.cband_crit_value) and res.cband_crit_value > 0
        tolerance = 1e-12
        for row in res.event_study_effects.values():
            if "cband_conf_int" not in row:
                continue
            lo, hi = row["cband_conf_int"]
            assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
            assert lo <= row["conf_int"][0] + tolerance
            assert hi >= row["conf_int"][1] - tolerance


class TestAllEventuallyTreatedRejection:
    """No never-treated units + not_yet_treated controls is rejected.

    The final-period cohort-time cells of such designs have an empty
    control pool, so estimating them would silently truncate the estimand
    (e.g. cohorts {3, 5} over T = 5 lose (3, 5) and (5, 5), dropping event
    time 2 entirely).
    """

    @staticmethod
    def _all_treated_panel(cohorts=(3, 5), n_periods=5, per_cohort=6, seed=11):
        rng = np.random.default_rng(seed)
        rows = []
        uid = 0
        for g in cohorts:
            for _ in range(per_cohort):
                unit_fe = rng.normal()
                for t in range(1, n_periods + 1):
                    treated = t >= g
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "treat": int(treated),
                            "y": unit_fe + 0.3 * t + rng.normal(0, 0.4) + float(treated),
                        }
                    )
                uid += 1
        return pd.DataFrame(rows)

    def test_all_eventually_treated_raises(self):
        panel = self._all_treated_panel()
        with pytest.raises(ValueError, match="eventually treated"):
            LWDiD(
                rolling="demean",
                estimation_method="reg",
                vcov_type="hc1",
                control_group="not_yet_treated",
            ).fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
            )

    def test_design_with_never_treated_runs_complete(self):
        """A regular staggered design estimates every relative event time."""
        panel = _make_shared_control_panel(seed=101, cohorts=(5, 7, 9))
        res = LWDiD(
            rolling="demean",
            estimation_method="reg",
            vcov_type="hc1",
            control_group="not_yet_treated",
        ).fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        expected = {t - g for g in (5, 7, 9) for t in range(1, 13)} - {-1}
        assert set(res.event_study_effects) == expected
        assert all(np.isfinite(row["effect"]) for row in res.event_study_effects.values())
        assert np.isfinite(res.att) and np.isfinite(res.se)


class TestStaggeredCovariateConstancy:
    """Staggered LWDiD only supports unit-constant covariates.

    Cohort-time cells read covariates at each calendar time, so a column
    that changes after treatment would silently move the ATT; such columns
    are rejected up front (matching the lwdid-py reference behaviour).
    """

    @staticmethod
    def _panel_with_covariate(time_varying):
        panel = _make_shared_control_panel(seed=17, n_never=20, per_cohort=10)
        rng = np.random.default_rng(23)
        x_by_unit = pd.Series(
            rng.normal(size=panel["unit"].nunique()), index=panel["unit"].unique()
        )
        panel["x1"] = panel["unit"].map(x_by_unit)
        if time_varying:
            # Post-treatment shift: constant pre-treatment, jumps at adoption.
            panel["x1"] += 0.5 * panel["treat"]
        return panel

    def test_post_treatment_varying_covariate_raises(self):
        panel = self._panel_with_covariate(time_varying=True)
        with pytest.raises(ValueError, match="not unit-constant"):
            LWDiD(rolling="demean", estimation_method="reg", vcov_type="hc1").fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
                covariates=["x1"],
            )

    def test_unit_constant_covariate_estimates(self):
        panel = self._panel_with_covariate(time_varying=False)
        res = LWDiD(rolling="demean", estimation_method="reg", vcov_type="hc1").fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
            covariates=["x1"],
        )
        assert np.isfinite(res.att)
        assert res.att == pytest.approx(1.5, abs=0.5)


# ─── Datetime/Period Time Scale Tests ───────────────────────────────────────


class TestDatetimeTimeScale:
    """Staggered fits on datetime64/Period panels via integer-position encoding."""

    @staticmethod
    def _datetime_panel():
        """Numeric staggered panel plus a quarterly datetime relabeling."""
        numeric = _make_staggered_panel(seed=42)
        date_map = {
            t: pd.Timestamp("2000-01-01") + pd.DateOffset(months=3 * (t - 1))
            for t in sorted(numeric["time"].unique())
        }
        panel = numeric.copy()
        panel["date"] = panel["time"].map(date_map)
        panel["adopt"] = panel["cohort"].map(lambda g: date_map[g] if g > 0 else pd.NaT)
        return numeric, panel, date_map

    def test_datetime_staggered_matches_numeric(self):
        numeric, panel, date_map = self._datetime_panel()
        model = LWDiD(rolling="demean", estimation_method="reg", vcov_type="hc1")
        res_num = model.fit(
            numeric,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        res_dt = model.fit(
            panel,
            outcome="y",
            unit="unit",
            time="date",
            treatment="treat",
            first_treat="adopt",
        )
        assert res_dt.att == pytest.approx(res_num.att)
        assert res_dt.se == pytest.approx(res_num.se)
        # Cohort keys are restored to the original datetime labels
        expected_cohorts = {date_map[g] for g in res_num.cohort_effects}
        assert set(res_dt.cohort_effects) == expected_cohorts
        for g, info in res_dt.cohort_effects.items():
            assert info["cohort"] == g
        # Cohort-time cells carry datetime labels with integer event times
        for (g, t), info in res_dt.cohort_time_effects.items():
            assert isinstance(g, pd.Timestamp) and isinstance(t, pd.Timestamp)
            assert info["cohort"] == g and info["time"] == t
            assert int(info["relative_time"]) == info["relative_time"]
        # Event-study labels stay integer position differences
        assert list(res_dt.event_study_effects) == list(res_num.event_study_effects)
        for label, row in res_num.event_study_effects.items():
            assert res_dt.event_study_effects[label]["effect"] == pytest.approx(row["effect"])

    def test_period_dtype_staggered_fits(self):
        numeric, panel, _ = self._datetime_panel()
        panel["date"] = panel["date"].dt.to_period("Q")
        panel["adopt"] = pd.PeriodIndex(panel["adopt"], freq="Q")
        model = LWDiD(rolling="demean", estimation_method="reg", vcov_type="hc1")
        res_num = model.fit(
            numeric,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
        )
        res_p = model.fit(
            panel,
            outcome="y",
            unit="unit",
            time="date",
            treatment="treat",
            first_treat="adopt",
        )
        assert res_p.att == pytest.approx(res_num.att)
        assert all(isinstance(g, pd.Period) for g in res_p.cohort_effects)

    def test_mixed_time_scales_raise(self):
        _, panel, _ = self._datetime_panel()
        with pytest.raises(ValueError, match="same time scale"):
            LWDiD(rolling="demean").fit(
                panel,
                outcome="y",
                unit="unit",
                time="date",
                treatment="treat",
                first_treat="cohort",
            )

    def test_datetime_all_eventually_treated_rejected(self):
        """The all-eventually-treated guard must also fire on datetime panels."""
        _, panel, _ = self._datetime_panel()
        eventually = panel.loc[panel["adopt"].notna()]
        with pytest.raises(ValueError, match="eventually treated"):
            LWDiD(rolling="demean", control_group="not_yet_treated").fit(
                eventually,
                outcome="y",
                unit="unit",
                time="date",
                treatment="treat",
                first_treat="adopt",
            )

    def test_datetime_time_varying_covariate_rejected(self):
        """The covariate constancy guard must also fire on datetime panels."""
        _, panel, _ = self._datetime_panel()
        rng = np.random.default_rng(0)
        panel["x1"] = rng.normal(size=len(panel))
        with pytest.raises(ValueError, match="not unit-constant"):
            LWDiD(rolling="demean").fit(
                panel,
                outcome="y",
                unit="unit",
                time="date",
                treatment="treat",
                first_treat="adopt",
                covariates=["x1"],
            )

    def test_datetime_transformation_diagnostics_keys(self):
        _, panel, date_map = self._datetime_panel()
        diagnostics = LWDiD(rolling="demean").get_transformation_diagnostics(
            panel,
            outcome="y",
            unit="unit",
            time="date",
            treatment="treat",
            first_treat="adopt",
        )
        assert diagnostics["design"] == "staggered"
        assert all(isinstance(g, pd.Timestamp) for g in diagnostics["by_cohort"])


class TestCohortNormalization:
    """LWDiD fix-wave WS9: one shared cohort normalizer (`_normalize_cohorts`)
    applied after time-scale encoding and before the design check, making
    every downstream never-treated predicate coherent. Campaign findings
    (execution-verified): first_treat=inf was iterated as a real cohort that
    consumed tau_omega weight mass; beyond-window cohorts distorted the
    composite; unbalanced panels missing the onset row were falsely
    rejected; validator and fit() disagreed on the never-treated encoding.
    """

    def _cohorts(self):
        # 8 treated across two cohorts + 12 never-treated
        return {u: (3 if u < 4 else (4 if u < 8 else 0)) for u in range(20)}

    def test_inf_cohort_recoded_to_never_treated(self):
        cohorts = self._cohorts()
        cohorts[19] = np.inf
        panel = _make_design_panel(cohorts)
        est = LWDiD(rolling="demean", estimation_method="reg", control_group="never_treated")
        with pytest.warns(UserWarning, match="first_treat=inf"):
            res = est.fit(
                panel, outcome="y", unit="unit", time="time",
                treatment="treat", first_treat="cohort",
            )
        assert np.isfinite(res.att)
        # inf never appears as a cohort anywhere in the results
        assert all(np.isfinite(g) and g > 0 for g in res.cohort_effects)

    def test_beyond_window_cohort_recoded_and_counts_as_control(self):
        cohorts = self._cohorts()
        cohorts[19] = 9  # beyond n_periods=5
        panel = _make_design_panel(cohorts)
        est = LWDiD(rolling="demean", estimation_method="reg", control_group="never_treated")
        with pytest.warns(UserWarning, match="exceed the last observed period"):
            res = est.fit(
                panel, outcome="y", unit="unit", time="time",
                treatment="treat", first_treat="cohort",
            )
        assert np.isfinite(res.att)
        assert all(g <= 5 for g in res.cohort_effects)

    def test_negative_cohort_rejected(self):
        cohorts = self._cohorts()
        cohorts[19] = -2
        panel = _make_design_panel(cohorts)
        # treat rows for a negative cohort: 1[t >= -2] would be all-1; keep 0
        panel.loc[panel["unit"] == 19, "treat"] = 0
        est = LWDiD(rolling="demean", estimation_method="reg")
        with pytest.raises(ValueError, match="negative"):
            est.fit(
                panel, outcome="y", unit="unit", time="time",
                treatment="treat", first_treat="cohort",
            )

    def test_between_period_numeric_cohort_rejected_with_clear_message(self):
        cohorts = self._cohorts()
        cohorts[0] = 3.5  # between observed periods 3 and 4
        panel = _make_design_panel(cohorts)
        # D_it = 1[t >= 3.5] -> treated at t=4,5
        panel.loc[panel["unit"] == 0, "treat"] = (
            panel.loc[panel["unit"] == 0, "time"] >= 3.5
        ).astype(int)
        est = LWDiD(rolling="demean", estimation_method="reg")
        with pytest.raises(ValueError, match="not observed time periods"):
            est.fit(
                panel, outcome="y", unit="unit", time="time",
                treatment="treat", first_treat="cohort",
            )

    def test_unobserved_onset_row_accepted(self):
        # Campaign finding: requiring the onset row itself to be observed
        # falsely rejected valid unbalanced panels.
        panel = _make_design_panel(self._cohorts())
        drop_mask = (panel["unit"] == 0) & (panel["time"] == 3)  # unit 0's onset row
        panel = panel.loc[~drop_mask].reset_index(drop=True)
        est = LWDiD(rolling="demean", estimation_method="reg")
        res = est.fit(
            panel, outcome="y", unit="unit", time="time",
            treatment="treat", first_treat="cohort",
        )
        assert np.isfinite(res.att)

    def test_untreated_observed_row_after_onset_rejected(self):
        # Review-caught hole: a first-observed-treated-row inequality alone
        # would ACCEPT a unit whose observed post-onset rows are all D=0
        # (no D=1 rows anywhere, onset row unobserved). The equality
        # predicate D_it == 1[t >= g_i] over observed rows must reject it.
        panel = _make_design_panel(self._cohorts())
        u0 = panel["unit"] == 0  # cohort 3
        panel = panel.loc[~(u0 & (panel["time"] == 3))]  # onset row unobserved
        panel.loc[panel["unit"] == 0, "treat"] = 0  # observed t=4,5 stay D=0
        est = LWDiD(rolling="demean", estimation_method="reg")
        with pytest.raises(ValueError, match="1\\[t >= cohort\\]"):
            est.fit(
                panel, outcome="y", unit="unit", time="time",
                treatment="treat", first_treat="cohort",
            )

    def test_validator_accepts_nan_coded_never_treated(self):
        # Validator/fit split: fit() accepts NaN never-treated; the
        # validator previously required cohort==0 and rejected it.
        from diff_diff.lwdid import validate_staggered_data

        panel = _make_design_panel(self._cohorts())
        panel["cohort"] = panel["cohort"].astype(float).replace(0.0, np.nan)
        out = validate_staggered_data(panel, "unit", "time", "cohort")
        assert out["valid"], out["errors"]
        assert out["n_never_treated"] == 12
        assert out["n_cohorts"] == 2

    def test_validator_flags_nat_mixed_with_finite_cohort(self):
        # nunique() excluded missing values, so a unit mixing NaT/NaN with
        # a finite cohort passed validation then raised inside fit().
        from diff_diff.lwdid import validate_staggered_data

        panel = _make_design_panel(self._cohorts())
        panel["cohort"] = panel["cohort"].astype(float)
        mix = (panel["unit"] == 0) & (panel["time"] == 1)
        panel.loc[mix, "cohort"] = np.nan
        out = validate_staggered_data(panel, "unit", "time", "cohort")
        assert not out["valid"]
        assert any("time-varying cohort" in e for e in out["errors"])

    def test_validator_reports_no_treated_cohorts(self):
        from diff_diff.lwdid import validate_staggered_data

        panel = _make_design_panel({u: 0 for u in range(6)})
        out = validate_staggered_data(panel, "unit", "time", "cohort")
        assert not out["valid"]
        assert any("No treated cohorts found" in e for e in out["errors"])

    def test_validator_handles_datetime_cohorts_without_raw_errors(self):
        # Previously df[cohort] > 0 raised a raw pandas TypeError on
        # datetime cohorts and df[cohort] == 0 silently reported "no
        # never-treated units".
        from diff_diff.lwdid import validate_staggered_data

        base = _make_design_panel(self._cohorts())
        time_map = {t: pd.Timestamp(f"2020-0{t}-01") for t in range(1, 6)}
        panel = base.assign(
            time=base["time"].map(time_map),
            cohort=base["cohort"].map(lambda g: time_map.get(g, pd.NaT)),
        )
        out = validate_staggered_data(panel, "unit", "time", "cohort")
        assert out["valid"], out["errors"]
        assert out["n_never_treated"] == 12
        assert out["n_cohorts"] == 2

    def test_is_never_treated_time_aware(self):
        from diff_diff.lwdid import is_never_treated

        cohorts = self._cohorts()
        cohorts[18] = np.inf
        cohorts[19] = 9  # beyond the window
        panel = _make_design_panel(cohorts)
        panel.loc[panel["unit"].isin([18, 19]), "treat"] = 0
        base = is_never_treated(panel, "unit", "cohort")
        aware = is_never_treated(panel, "unit", "cohort", time="time")
        units = panel.drop_duplicates("unit")["unit"].to_numpy()
        base_map = dict(zip(units, base))
        aware_map = dict(zip(units, aware))
        assert base_map[18] and aware_map[18]  # inf is never-treated either way
        assert not base_map[19]  # beyond-window needs the time support
        assert aware_map[19]

    def test_diagnostics_iterate_normalized_cohorts_only(self):
        cohorts = self._cohorts()
        cohorts[19] = np.inf
        panel = _make_design_panel(cohorts)
        est = LWDiD(rolling="demean", estimation_method="reg")
        with pytest.warns(UserWarning, match="first_treat=inf"):
            diag = est.get_transformation_diagnostics(
                panel, outcome="y", unit="unit", time="time",
                treatment="treat", first_treat="cohort",
            )
        assert set(diag["by_cohort"]) == {3, 4}


class TestSeasonalTransformFailClosed:
    """LWDiD fix-wave WS2: the seasonal transforms fail closed. Campaign
    findings: detrendq silently fit intercept+trend (plain detrend) per
    unit when pre-periods < seasonal parameter count - with quarterly data
    and <=5 pre-periods EVERY unit fell back, so the whole fit was
    numerically identical to rolling='detrend' while reporting 'detrendq';
    both q transforms silently extrapolated quarters unobserved in the
    pre-period at the reference-season level.
    """

    @staticmethod
    def _quarterly_common_panel(n_pre, t_max=12, n_units=30, seed=5):
        rng = np.random.default_rng(seed)
        season = np.array([1.0, -0.5, 0.8, -1.3])
        rows = []
        onset = n_pre + 1
        for u in range(n_units):
            alpha = rng.normal()
            treated = u < n_units // 2
            for t in range(1, t_max + 1):
                d = int(treated and t >= onset)
                y = alpha + season[(t - 1) % 4] + 0.05 * t + rng.normal(scale=0.3) + 1.2 * d
                rows.append(dict(unit=u, time=t, treat=d, y=y))
        return pd.DataFrame(rows)

    def test_detrendq_insufficient_pre_fails_closed_not_silent_detrend(self):
        # 4 pre-periods cover all 4 seasons -> n_params = 1 + 1 + 3 = 5 > 4:
        # every unit is unidentified. Pre-fix this silently produced the
        # detrend numbers; now the fit warns and the ATT is NaN with a
        # consistent inference tuple.
        df = self._quarterly_common_panel(n_pre=4)
        est = LWDiD(rolling="detrendq", estimation_method="reg")
        with pytest.warns(UserWarning, match="detrendq requires at least"):
            res = est.fit(df, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isnan(res.att)
        from tests.conftest import assert_nan_inference

        assert_nan_inference(
            {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
        )

    def test_detrendq_identified_differs_from_detrend(self):
        # With enough pre-periods the seasonal fit is identified and must
        # NOT equal plain detrend on a seasonal DGP.
        df = self._quarterly_common_panel(n_pre=8, t_max=16)
        kw = dict(outcome="y", unit="unit", time="time", treatment="treat")
        rq = LWDiD(rolling="detrendq", estimation_method="reg").fit(df, **kw)
        rp = LWDiD(rolling="detrend", estimation_method="reg").fit(df, **kw)
        assert np.isfinite(rq.att)
        assert abs(rq.att - rp.att) > 1e-8

    @pytest.mark.parametrize("rolling", ["demeanq", "detrendq"])
    def test_unobserved_pre_season_fails_closed(self, rolling):
        # Pre-period covers quarters 1-3 only; post includes quarter 4 ->
        # out-of-support prediction must warn + NaN, never extrapolate.
        df = self._quarterly_common_panel(n_pre=7, t_max=8)
        # onset at t=8 (quarter 4); pre t=1..7 covers quarters 1,2,3,4?
        # t=1..7 -> quarters 1,2,3,4,1,2,3: quarter 4 IS observed. Drop
        # every pre row in quarter 4 instead.
        df = df.loc[~((df["time"] == 4))].reset_index(drop=True)
        est = LWDiD(rolling=rolling, estimation_method="reg")
        with pytest.warns(UserWarning, match="cannot predict quarter"):
            res = est.fit(df, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isnan(res.att)


class TestBootstrapIntegrity:
    """LWDiD fix-wave WS3: common-timing bootstrap resampling integrity.

    Campaign findings (execution-verified): the bootstrap collected index
    LABELS but fetched rows POSITIONALLY (raw IndexError on offset indexes;
    silently doubled SEs on row-shuffled frames); cluster= was accepted but
    dead inside _bootstrap (iid unit bootstrap labeled clustered); the
    reported df_inference was G-1 while the bootstrap p-value used N-k.
    """

    @staticmethod
    def _common_panel(n_units=40, t_max=6, onset=4, seed=11, n_clusters=8):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(n_units):
            alpha = rng.normal()
            cl = u % n_clusters
            cl_shock = np.sin(cl)  # cluster-correlated level
            treated = u < n_units // 2
            for t in range(1, t_max + 1):
                d = int(treated and t >= onset)
                y = alpha + cl_shock + 0.1 * t + rng.normal(scale=0.4) + 1.3 * d
                rows.append(dict(unit=u, time=t, treat=d, y=y, cl=cl))
        return pd.DataFrame(rows)

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_bootstrap_invariant_to_index_labels_and_row_order(self):
        df = self._common_panel()
        est = lambda: LWDiD(rolling="demean", estimation_method="reg", n_bootstrap=60, seed=1)  # noqa: E731
        base = est().fit(df, **self.KW)
        shifted = est().fit(df.set_axis(df.index + 1000), **self.KW)  # offset labels
        rng = np.random.default_rng(3)
        shuffled = df.sample(frac=1.0, random_state=5)  # permuted labels
        res_shuffled = est().fit(shuffled, **self.KW)
        assert np.isfinite(base.se)
        np.testing.assert_allclose(shifted.att, base.att, rtol=0, atol=1e-12)
        np.testing.assert_allclose(shifted.se, base.se, rtol=0, atol=1e-12)
        np.testing.assert_allclose(res_shuffled.att, base.att, rtol=0, atol=1e-12)
        # Same seed + same units resampled -> the SE must not move with
        # row order (pre-fix it more than doubled).
        np.testing.assert_allclose(res_shuffled.se, base.se, rtol=1e-10)
        del rng

    def test_cluster_bootstrap_resamples_clusters(self, ci_params):
        df = self._common_panel()
        n_boot = ci_params.bootstrap(120)
        iid = LWDiD(rolling="demean", estimation_method="reg", n_bootstrap=n_boot, seed=7).fit(
            df, **self.KW
        )
        clustered = LWDiD(
            rolling="demean", estimation_method="reg", n_bootstrap=n_boot, seed=7, cluster="cl"
        ).fit(df, **self.KW)
        # Points identical (resampling never moves the full-sample point)
        np.testing.assert_allclose(clustered.att, iid.att, rtol=0, atol=1e-12)
        # SEs genuinely differ on a cluster-correlated DGP
        assert np.isfinite(clustered.se) and clustered.se > 0
        assert abs(clustered.se - iid.se) / iid.se > 1e-3
        # df matches the analytical clustered rule (G-1), not N-k
        assert clustered.df_inference == 8 - 1
        assert clustered.cluster_name == "cl"
        assert clustered.n_clusters == 8

    def test_cluster_bootstrap_concords_with_analytical_cr1(self, ci_params):
        df = self._common_panel(n_units=80, n_clusters=16)
        n_boot = ci_params.bootstrap(300, min_n=199)
        analytical = LWDiD(rolling="demean", estimation_method="reg", cluster="cl").fit(
            df, **self.KW
        )
        boot = LWDiD(
            rolling="demean", estimation_method="reg", cluster="cl",
            n_bootstrap=n_boot, seed=13,
        ).fit(df, **self.KW)
        threshold = 0.40 if n_boot < 100 else 0.15
        assert abs(boot.se - analytical.se) / analytical.se < threshold, (boot.se, analytical.se)
