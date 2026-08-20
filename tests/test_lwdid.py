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
        """Construct simple 3-unit panel where pre-means are known.

        (Fix-wave update: the former 2-unit fixture is an INVALID exact
        design - 2 collapsed observations for 2 parameters - which the
        Registry small-sample guard now rejects; a second control keeps
        the hand-computed arithmetic with a positive residual df.)
        """
        # Unit 0 (control): y = [2, 4, 6] -> pre_mean = 3, post ydot = 3
        # Unit 2 (control): y = [4, 6, 8] -> pre_mean = 5, post ydot = 3
        # Unit 1 (treated): y = [1, 3, 10] -> pre_mean = 2, post ydot = 8
        df = pd.DataFrame(
            {
                "unit": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "y": [2.0, 4.0, 6.0, 1.0, 3.0, 10.0, 4.0, 6.0, 8.0],
                "treat": [0, 0, 0, 0, 0, 1, 0, 0, 0],
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
        # Need at least 2 pre periods for detrend; a third unit keeps the
        # collapsed design valid (fix-wave Registry small-sample guard).
        # Units 0/2 (controls): y = 1 + 2*t (unit 2 offset by +2)
        # Unit 1 (treated): y = 1 + 2*t in pre, + 5 in post
        df = pd.DataFrame(
            {
                "unit": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                "y": [3.0, 5.0, 7.0, 9.0, 3.0, 5.0, 12.0, 14.0, 5.0, 7.0, 9.0, 11.0],
                "treat": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
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
        res = LWDiD(vcov_type="hc1").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
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

    # Fix-wave WS6: ipw/dr accept vcov_type='hc1' ONLY (the IF sandwich);
    # other families were silently inert and are now rejected at
    # construction, so the reconciliation grid enumerates real configs.
    @pytest.mark.parametrize(
        "estimation_method,vcov",
        [
            ("reg", "classical"),
            ("reg", "hc1"),
            ("reg", "hc2"),
            ("reg", "hc3"),
            ("reg", "cluster"),
            ("ipw", "hc1"),
            ("ipw", "cluster"),
            ("dr", "hc1"),
            ("dr", "cluster"),
        ],
    )
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

    @pytest.mark.parametrize("estimation_method", ["ipw", "dr", "psm"])
    @pytest.mark.parametrize("vcov", ["classical", "hc2", "hc3"])
    def test_inert_vcov_values_rejected(self, estimation_method, vcov):
        with pytest.raises(ValueError, match="silently inert"):
            LWDiD(estimation_method=estimation_method, vcov_type=vcov)

    def test_cluster_composes_only_with_hc1(self):
        with pytest.raises(ValueError, match="composes only with vcov_type='hc1'"):
            LWDiD(estimation_method="reg", vcov_type="hc3", cluster="cl")

    def test_psm_cluster_rejected(self):
        with pytest.raises(ValueError, match="psm.*does not support cluster"):
            LWDiD(estimation_method="psm", cluster="cl")

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
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
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
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
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
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
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
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
            )

    def test_unobserved_onset_row_accepted(self):
        # Campaign finding: requiring the onset row itself to be observed
        # falsely rejected valid unbalanced panels.
        panel = _make_design_panel(self._cohorts())
        drop_mask = (panel["unit"] == 0) & (panel["time"] == 3)  # unit 0's onset row
        panel = panel.loc[~drop_mask].reset_index(drop=True)
        est = LWDiD(rolling="demean", estimation_method="reg")
        res = est.fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="cohort",
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
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
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
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="cohort",
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

        def est():
            return LWDiD(rolling="demean", estimation_method="reg", n_bootstrap=60, seed=1)

        base = est().fit(df, **self.KW)
        shifted = est().fit(df.set_axis(df.index + 1000), **self.KW)  # offset labels
        shuffled = df.sample(frac=1.0, random_state=5)  # permuted labels
        res_shuffled = est().fit(shuffled, **self.KW)
        assert np.isfinite(base.se)
        np.testing.assert_allclose(shifted.att, base.att, rtol=0, atol=1e-12)
        np.testing.assert_allclose(shifted.se, base.se, rtol=0, atol=1e-12)
        np.testing.assert_allclose(res_shuffled.att, base.att, rtol=0, atol=1e-12)
        # Same seed + same units resampled -> the SE must not move with
        # row order (pre-fix it more than doubled).
        np.testing.assert_allclose(res_shuffled.se, base.se, rtol=1e-10)

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
            rolling="demean",
            estimation_method="reg",
            cluster="cl",
            n_bootstrap=n_boot,
            seed=13,
        ).fit(df, **self.KW)
        threshold = 0.40 if n_boot < 100 else 0.15
        assert abs(boot.se - analytical.se) / analytical.se < threshold, (boot.se, analytical.se)


class TestPSMCaliperContract:
    """LWDiD fix-wave WS5 (campaign finding, deterministic repro): with a
    caliper and n_neighbors > 1, argsort kept selecting np.inf-distance
    (out-of-caliper) controls whenever fewer than n_neighbors controls fell
    inside the caliper, silently averaging arbitrarily distant controls
    into the counterfactual (ATT -49 vs the correct 1.0 on this fixture).
    """

    @staticmethod
    def _fixture():
        # 2 treated (pscore ~0.64-0.69), 1 near control (~0.67, ydot=0
        # effect scale), 3 far controls (pscore ~0) whose transformed
        # outcome is +100.
        rows = []
        units = [
            ("t1", 1, 5.0, 1.0),
            ("t2", 1, 4.6, 1.0),
            ("c_near", 0, 4.8, 0.0),
            ("c_far1", 0, -9.0, 100.0),
            ("c_far2", 0, -9.4, 100.0),
            ("c_far3", 0, -9.8, 100.0),
        ]
        for name, d, x, post_shift in units:
            for t in (1, 2):
                y = 1.0 if d and t == 2 else 0.0
                y += post_shift if t == 2 else 0.0
                rows.append(dict(unit=name, time=t, treat=d * int(t == 2), y=y, x=x))
        return pd.DataFrame(rows)

    def test_partial_caliper_shortfall_averages_survivors_only(self):
        df = self._fixture()
        est = LWDiD(
            rolling="demean",
            estimation_method="psm",
            n_neighbors=2,
            caliper=0.05,
        )
        with pytest.warns(UserWarning, match="fewer than n_neighbors"):
            res = est.fit(
                df, outcome="y", unit="unit", time="time", treatment="treat", covariates=["x"]
            )
        # Demeaned outcomes: treated ydot = 2.0, near control ydot = 0,
        # far controls ydot = +100. Caliper-respecting match (c_near only):
        # ATT = 2.0. Contaminated pre-fix value: 2.0 - (0+100)/2 = -48.
        np.testing.assert_allclose(res.att, 2.0, atol=1e-10)


class TestSilentDataHandling:
    """LWDiD fix-wave WS8 (campaign findings): NaN covariates silently
    dropped units on cell paths while poisoning the common-timing OLS;
    staggered n_obs/n_treated counted every input unit regardless of cell
    drops; rank-deficient designs used the NOMINAL parameter count for the
    df and a full-width pinv bread, breaking the IF == solve_ols SE
    identity the docstring claims.
    """

    @staticmethod
    def _staggered_panel(seed=31):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(24):
            g = 3 if u < 5 else (4 if u < 10 else 0)
            alpha = rng.normal()
            x = rng.normal()
            for t in range(1, 7):
                d = int(g > 0 and t >= g)
                y = alpha + 0.4 * x + rng.normal(scale=0.4) + 1.5 * d
                rows.append(dict(unit=u, time=t, first=g, treat=d, y=y, x=x))
        return pd.DataFrame(rows)

    def test_nan_covariate_rejected_explicitly(self):
        df = self._staggered_panel()
        df.loc[3, "x"] = np.nan
        with pytest.raises(ValueError, match="missing value"):
            LWDiD(rolling="demean", estimation_method="reg").fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="first",
                covariates=["x"],
            )

    def test_nan_cluster_rejected_explicitly(self):
        df = self._staggered_panel()
        df["cl"] = df["unit"] % 4
        df["cl"] = df["cl"].astype(float)
        df.loc[df["unit"] == 2, "cl"] = np.nan
        with pytest.raises(ValueError, match="Cluster column .* missing"):
            LWDiD(rolling="demean", estimation_method="reg", cluster="cl").fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="first",
            )

    def test_rank_deficient_covariate_if_reproduces_solve_ols_se(self):
        # A NON-trailing collinear covariate is dropped by solve_ols; the
        # influence function must be rebuilt on the kept columns so its
        # norm still reproduces the reported SE (docstring identity).
        rng = np.random.default_rng(37)
        n = 200
        controls = rng.normal(size=(n, 3))
        controls[:, 0] = 2.0 * controls[:, 2] + 1.0  # column 0 collinear
        treatment = (rng.uniform(size=n) < 0.4).astype(float)
        y = 1.0 + 2.0 * treatment + controls[:, 1] * 0.5 + rng.normal(size=n)
        est = LWDiD(estimation_method="reg", vcov_type="hc1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            att, se, coefs, _, n_params, influence = est._estimate_reg(
                y, treatment, controls, None, n
            )
        assert np.isfinite(att) and np.isfinite(se)
        assert np.isnan(coefs).any()  # a column really was dropped
        assert n_params == int(np.sum(~np.isnan(coefs)))
        assert influence is not None
        assert float(np.sqrt(np.sum(influence**2))) == pytest.approx(se, rel=1e-10)

    def test_treatment_column_dropped_yields_nan_att(self):
        # If the treatment column itself is pivoted out (collinear with a
        # control), the ATT is unidentified: NaN point + no influence.
        rng = np.random.default_rng(41)
        n = 120
        treatment = (rng.uniform(size=n) < 0.5).astype(float)
        controls = np.column_stack([treatment * 3.0, rng.normal(size=n)])
        y = 1.0 + rng.normal(size=n)
        est = LWDiD(estimation_method="reg", vcov_type="hc1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            att, se, _, _, _, influence = est._estimate_reg(y, treatment, controls, None, n)
        # Either the treatment or its collinear twin is dropped; if the
        # treatment survives the ATT is finite - accept both resolutions
        # but NEVER a finite ATT with se=0-style inference.
        if np.isnan(att):
            assert np.isnan(se) and influence is None
        else:
            assert np.isfinite(se) and se > 0

    def test_staggered_metadata_counts_contributing_units(self):
        # Under rolling='detrend', a unit with a single pre-period row has
        # NaN transformed outcomes in EVERY cell (per-unit trend needs >= 2
        # pre points), so it is dropped from every cell's finite filter and
        # contributes nothing - the estimation-sample metadata must not
        # count it (campaign finding: n_obs/n_treated covered every input
        # unit regardless of cell drops).
        df = self._staggered_panel()
        rng = np.random.default_rng(5)
        extra = [
            dict(unit=99, time=t, first=0, treat=0, y=rng.normal(), x=0.0) for t in (1, 4, 5, 6)
        ]  # ONE pre row (t=1) + post rows
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="detrend", estimation_method="reg").fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="first",
            )
        assert res.n_obs == 24  # unit 99 contributed to no estimated cell
        assert res.n_control == 14
        assert res.n_treated == 10


class TestResultsPolish:
    """LWDiD fix-wave WS10 result-surface pins."""

    def test_to_latex_removed(self):
        from diff_diff.lwdid_results import LWDiDResults

        assert not hasattr(LWDiDResults, "to_latex")

    def test_staggered_to_dataframe_carries_config_columns(self):
        rng = np.random.default_rng(19)
        rows = []
        for u in range(20):
            g = 3 if u < 8 else 0
            alpha = rng.normal()
            for t in range(1, 6):
                d = int(g > 0 and t >= g)
                rows.append(dict(unit=u, time=t, first=g, treat=d, y=alpha + rng.normal() + d))
        df = pd.DataFrame(rows)
        res = LWDiD(rolling="demean", estimation_method="reg").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat", first_treat="first"
        )
        frame = res.to_dataframe()
        for col in ("rolling", "estimation_method", "vcov_type"):
            assert col in frame.columns
            assert frame[col].nunique() == 1

    def test_detrend_degenerate_cohort_composite_is_graceful(self):
        # Campaign finding: a cohort with < 2 pre-periods crashed the
        # composite with a raw LinAlgError under detrend + never_treated.
        # The complete-case machinery now drops it with warnings.
        rng = np.random.default_rng(2)
        rows = []
        for u in range(20):
            g = 2 if u < 4 else (5 if u < 9 else 0)  # cohort 2: ONE pre period
            alpha = rng.normal()
            for t in range(1, 8):
                d = int(g > 0 and t >= g)
                rows.append(
                    dict(unit=u, time=t, first=g, treat=d, y=alpha + 0.1 * t + rng.normal() + d)
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(
                rolling="detrend",
                estimation_method="reg",
                vcov_type="classical",
                control_group="never_treated",
            ).fit(df, outcome="y", unit="unit", time="time", treatment="treat", first_treat="first")
        assert np.isfinite(res.att)
        assert res.n_composite_treated_dropped == 4


class TestNonNumericTimeContract:
    """LWDiD fix-wave (campaign finding): string time columns made
    detrend/demeanq/detrendq raise raw numpy conversion errors while
    demean succeeded - now an informative ValueError states the contract.
    """

    @staticmethod
    def _string_time_panel():
        rng = np.random.default_rng(1)
        rows = []
        labels = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
        for u in range(10):
            for i, t in enumerate(labels):
                d = int(u < 5 and i >= 3)
                rows.append(dict(unit=u, time=t, treat=d, y=rng.normal() + d))
        return pd.DataFrame(rows)

    def test_demean_accepts_ordered_categorical_time(self):
        # Round-20 review: plain string labels sort lexicographically
        # ('Q10' < 'Q2'), so the chronology must be DECLARED - demean
        # accepts an ordered categorical and rejects plain object labels.
        df = self._string_time_panel()
        labels = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
        df_cat = df.assign(time=pd.Categorical(df["time"], categories=labels, ordered=True))
        res = LWDiD(rolling="demean", estimation_method="reg").fit(
            df_cat, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert np.isfinite(res.att)

    def test_demean_rejects_plain_string_time(self):
        df = self._string_time_panel()
        with pytest.raises(ValueError, match="ORDERED categorical"):
            LWDiD(rolling="demean", estimation_method="reg").fit(
                df, outcome="y", unit="unit", time="time", treatment="treat"
            )

    @pytest.mark.parametrize("rolling", ["detrend", "demeanq", "detrendq"])
    def test_trend_seasonal_transforms_reject_string_time_informatively(self, rolling):
        df = self._string_time_panel()
        with pytest.raises(ValueError, match="numeric or datetime"):
            LWDiD(rolling=rolling, estimation_method="reg").fit(
                df, outcome="y", unit="unit", time="time", treatment="treat"
            )


class TestReviewRound1Guards:
    """Local-review round 1 (fix wave): execution-verified guards.

    - HC3 with a leverage-one observation (single treated unit) fabricated
      finite inference via a 1e-10 floor on 1 - h_ii
    - N=2 / N=3,K=1 collapsed designs hit ZeroDivisionError or a coerced
      df=1 instead of the Registry's small-sample guards
    - n_bootstrap=1 was accepted; staggered PSM + bootstrap silently no-oped
    - PSM's naive matched-pairs SE ignored control reuse and first-stage
      uncertainty (now fail-closed NaN inference, point retained)
    - the common-timing single-cluster fallback warned then raised
    - Period time crashed detrend with a raw TypeError
    """

    @staticmethod
    def _panel(n_units=12, n_treated=1, t_max=6, onset=4, seed=0, **cols):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(n_units):
            treated = u < n_treated
            for t in range(1, t_max + 1):
                d = int(treated and t >= onset)
                row = dict(unit=u, time=t, treat=d, y=rng.normal() + d)
                for k, fn in cols.items():
                    row[k] = fn(u)
                rows.append(row)
        return pd.DataFrame(rows)

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_hc3_leverage_one_fails_closed(self):
        df = self._panel(n_units=12, n_treated=1)
        with pytest.warns(UserWarning, match="HC3 variance is undefined"):
            res = LWDiD(rolling="demean", estimation_method="reg", vcov_type="hc3").fit(
                df, **self.KW
            )
        assert np.isfinite(res.att)
        from tests.conftest import assert_nan_inference

        assert_nan_inference(
            {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
        )

    def test_invalid_exact_designs_rejected(self):
        df2 = self._panel(n_units=2, n_treated=1)
        with pytest.raises(ValueError, match="Invalid exact-inference design"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                LWDiD(rolling="demean", estimation_method="reg", vcov_type="classical").fit(
                    df2, **self.KW
                )
        df3 = self._panel(n_units=3, n_treated=1, x=lambda u: float(u))
        with pytest.raises(ValueError, match="Invalid exact-inference design"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                LWDiD(rolling="demean", estimation_method="reg", vcov_type="classical").fit(
                    df3, covariates=["x"], **self.KW
                )

    def test_n_bootstrap_one_rejected(self):
        with pytest.raises(ValueError, match="n_bootstrap must be 0"):
            LWDiD(n_bootstrap=1)

    def test_staggered_psm_bootstrap_rejected(self):
        df = self._panel(n_units=16, n_treated=6)
        df["first"] = np.where(df["unit"] < 6, 4, 0)
        est = LWDiD(estimation_method="psm", n_bootstrap=50)
        with pytest.raises(ValueError, match="psm.*does not support n_bootstrap"):
            est.fit(df, first_treat="first", **self.KW)

    def test_psm_inference_fails_closed_point_retained(self):
        df = self._panel(n_units=20, n_treated=8, x=lambda u: float(u % 4))
        with pytest.warns(UserWarning, match="no valid matching variance"):
            res = LWDiD(rolling="demean", estimation_method="psm").fit(
                df, covariates=["x"], **self.KW
            )
        assert np.isfinite(res.att)
        assert np.isnan(res.se) and np.isnan(res.p_value)
        assert res.psm_config is not None
        assert res.psm_config["n_neighbors"] == 1

    def test_matching_params_strictly_validated(self):
        with pytest.raises(ValueError, match="n_neighbors must be an integer"):
            LWDiD(n_neighbors=1.5)
        with pytest.raises(ValueError, match="with_replacement must be a boolean"):
            LWDiD(with_replacement="yes")
        with pytest.raises(ValueError, match="caliper must be a positive"):
            LWDiD(caliper=-0.1)

    def test_common_single_cluster_post_drop_point_retained(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(10):
            for t in range(1, 7):
                d = int(u < 5 and t >= 4)
                rows.append(dict(unit=u, time=t, treat=d, y=rng.normal() + d, cl=0 if u < 9 else 1))
        df = pd.DataFrame(rows)
        df = df.loc[~((df.unit == 9) & (df.time.isin([2, 3])))]
        with pytest.warns(UserWarning, match="fewer than 2 clusters"):
            res = LWDiD(rolling="detrend", estimation_method="reg", cluster="cl").fit(df, **self.KW)
        assert np.isfinite(res.att)
        assert np.isnan(res.se)

    def test_period_time_detrend_rejected_informatively(self):
        rng = np.random.default_rng(1)
        times = pd.period_range("2020Q1", periods=8, freq="Q")
        rows = []
        for u in range(10):
            for i, t in enumerate(times):
                d = int(u < 5 and i >= 5)
                rows.append(dict(unit=u, time=t, treat=d, y=rng.normal() + d))
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="does not support a Period"):
            LWDiD(rolling="detrend", estimation_method="reg").fit(df, **self.KW)

    def test_provenance_fields_round_trip(self):
        df = self._panel(n_units=12, n_treated=5)
        res = LWDiD(
            rolling="demean",
            estimation_method="reg",
            control_group="never_treated",
            n_bootstrap=0,
            seed=7,
        ).fit(df, **self.KW)
        assert res.control_group == "never_treated"
        assert res.n_bootstrap == 0
        assert res.seed == 7
        assert res.psm_config is None
        d = res.to_dict()
        assert d["control_group"] == "never_treated"
        assert d["seed"] == 7


class TestReviewRound2Guards:
    """Local-review round 2: execution-verified guards.

    - cluster='_treat' with numeric labels silently reported the cluster
      labels' coefficient as the ATT (reserved-name collision)
    - the same seed produced different bootstrap SEs across n_jobs (the
      serial path consumed one sequential RNG stream while the parallel
      path spawned per-replicate streams)
    - pscore_trim was absent from ipw/dr result provenance
    - an event cell with degenerate multiplier-bootstrap draws silently
      kept its analytical SE (undocumented mixture of inference families)
    """

    @staticmethod
    def _panel(n_units=12, n_treated=6, t_max=6, onset=4, seed=0, **cols):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(n_units):
            treated = u < n_treated
            for t in range(1, t_max + 1):
                d = int(treated and t >= onset)
                row = dict(unit=u, time=t, treat=d, y=rng.normal() + 2 * d + 0.5 * t)
                for k, fn in cols.items():
                    row[k] = fn(u)
                rows.append(row)
        return pd.DataFrame(rows)

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_reserved_internal_names_rejected(self):
        df = self._panel(cl=lambda u: float(u % 3)).rename(columns={"cl": "_treat"})
        with pytest.raises(ValueError, match="reserved for LWDiD internal use"):
            LWDiD(rolling="demean", cluster="_treat").fit(df, **self.KW)
        df2 = self._panel(x=lambda u: float(u)).rename(columns={"x": "_ydot"})
        with pytest.raises(ValueError, match="reserved for LWDiD internal use"):
            LWDiD(rolling="demean").fit(df2, covariates=["_ydot"], **self.KW)
        df3 = self._panel().rename(columns={"unit": "_boot_unit"})
        with pytest.raises(ValueError, match="reserved for LWDiD internal use"):
            LWDiD(rolling="demean").fit(
                df3, outcome="y", unit="_boot_unit", time="time", treatment="treat"
            )

    def test_duplicate_role_columns_rejected(self):
        df = self._panel()
        with pytest.raises(ValueError, match="distinct column"):
            LWDiD(rolling="demean").fit(df, outcome="y", unit="unit", time="time", treatment="y")
        df2 = self._panel(x=lambda u: float(u))
        with pytest.raises(ValueError, match="already supplied"):
            LWDiD(rolling="demean").fit(df2, covariates=["y"], **self.KW)
        # cluster == unit stays supported (documented intentional case)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", cluster="unit").fit(df, **self.KW)
        assert np.isfinite(res.att)

    def test_seeded_bootstrap_invariant_to_n_jobs(self):
        df = self._panel(n_units=20, n_treated=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = LWDiD(rolling="demean", n_bootstrap=49, seed=7, n_jobs=1).fit(df, **self.KW)
            r2 = LWDiD(rolling="demean", n_bootstrap=49, seed=7, n_jobs=2).fit(df, **self.KW)
        assert r1.se == r2.se
        assert r1.att == r2.att

    def test_pscore_trim_provenance(self):
        df = self._panel(x=lambda u: float(u % 4))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ipw = LWDiD(rolling="demean", estimation_method="ipw", pscore_trim=0.02).fit(
                df, covariates=["x"], **self.KW
            )
            reg = LWDiD(rolling="demean", estimation_method="reg").fit(df, **self.KW)
        assert ipw.pscore_trim == 0.02
        assert ipw.to_dict()["pscore_trim"] == 0.02
        assert reg.pscore_trim is None
        assert "pscore_trim" not in reg.to_dict()

    def test_degenerate_bootstrap_event_cell_fails_closed(self):
        from types import SimpleNamespace

        from diff_diff.lwdid_staggered import compute_event_study_bands

        rng = np.random.default_rng(0)
        estimator = SimpleNamespace(n_bootstrap=199, seed=3, alpha=0.05)
        event_effects = {
            0: {
                "effect": 1.0,
                "se": 0.2,
                "t_stat": 5.0,
                "p_value": 0.0,
                "conf_int": (0.6, 1.4),
                "df": None,
            },
            1: {
                "effect": 0.5,
                "se": 0.1,
                "t_stat": 5.0,
                "p_value": 0.0,
                "conf_int": (0.3, 0.7),
                "df": None,
            },
        }
        event_influence = {
            0: np.zeros(30),  # degenerate: zero influence column
            1: rng.normal(size=30),
        }
        with pytest.warns(UserWarning, match="degenerate draws"):
            compute_event_study_bands(estimator, event_effects, event_influence, None)
        assert np.isnan(event_effects[0]["se"])
        assert np.isnan(event_effects[0]["p_value"])
        assert event_effects[0]["inference_status"] == "degenerate_bootstrap"
        assert event_effects[0]["effect"] == 1.0  # point retained
        assert np.isfinite(event_effects[1]["se"])  # valid cell bootstrapped
        assert "cband_conf_int" in event_effects[1]


class TestReviewRound3Guards:
    """Local-review round 3: execution-verified guards.

    - encoded staggered panels fed dense POSITIONS to the seasonal
      transforms' (t-1)%4+1 fallback, so a globally missing calendar
      quarter silently relabeled every later season (probe: zero-effect
      gapped quarterly panel with treated/control-differential
      seasonality biased demeanq ATT to ~0.12)
    - PSM bypassed its fail-closed NaN-inference contract via the
      covariate-less delegation, the common-timing pairs bootstrap, and
      the logit-failure regression fallback
    - common-timing bootstrap fits reported bootstrap headline inference
      with no provenance while params/vcov stayed analytical
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    @staticmethod
    def _gapped_quarterly(as_period=True):
        rng = np.random.default_rng(5)
        periods = pd.period_range("2018Q1", "2023Q4", freq="Q")
        periods = periods[periods != pd.Period("2020Q3", freq="Q")]
        seas = {1: 2.0, 2: -1.0, 3: 0.5, 4: -1.5}
        onset = pd.Period("2022Q1", freq="Q")
        rows = []
        for u in range(24):
            treated_unit = u < 12
            amp = 3.0 if treated_unit else 1.0
            for p in periods:
                d = int(treated_unit and p >= onset)
                y = 1.0 + amp * seas[p.quarter] + rng.normal(0, 0.1)
                rows.append(dict(unit=u, p=p, y=y, treat=d, g=onset if treated_unit else pd.NaT))
        df = pd.DataFrame(rows)
        if as_period:
            df["time"] = pd.PeriodIndex(df["p"], freq="Q")
            df["gv"] = pd.PeriodIndex(df["g"], freq="Q")
        else:
            # ordinal numeric encoding preserves calendar-quarter identity
            # under (t-1)%4+1 (Q ordinals advance one per quarter), so the
            # numeric path is the correct-season oracle
            df["time"] = pd.PeriodIndex(df["p"], freq="Q").map(lambda v: v.ordinal + 1)
            df["gv"] = [pd.Period(v, freq="Q").ordinal + 1 if pd.notna(v) else 0 for v in df["g"]]
        return df.drop(columns=["p", "g"])

    def test_gapped_calendar_seasonal_parity(self):
        # Period path (encoded to dense positions) must match the
        # ordinal-numeric oracle on the SAME gapped data. demeanq is
        # trend-free, so seasonal-grouping parity is exact. detrendq
        # additionally uses the time values as its trend coordinate
        # (dense positions on the encoded path vs calendar ordinals on
        # the numeric path - a documented encoding choice), so its pin is
        # the no-seasonal-leakage bound, not bitwise parity.
        got, want = {}, {}
        for rolling in ("demeanq", "detrendq"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_period = LWDiD(rolling=rolling).fit(
                    self._gapped_quarterly(True), first_treat="gv", **self.KW
                )
                r_numeric = LWDiD(rolling=rolling).fit(
                    self._gapped_quarterly(False), first_treat="gv", **self.KW
                )
            got[rolling], want[rolling] = r_period.att, r_numeric.att
        np.testing.assert_allclose(got["demeanq"], want["demeanq"], rtol=1e-10)
        for rolling in got:
            # zero-effect DGP with 3x treated seasonal amplitude: the
            # pre-fix position-modulo labeling biased this to ~0.12
            assert abs(got[rolling]) < 0.06, rolling

    @staticmethod
    def _panel(n_units=16, x=True):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(n_units):
            for t in range(1, 7):
                d = 1 if (u < n_units // 2 and t >= 4) else 0
                row = dict(unit=u, time=t, treat=d, y=1 + 0.5 * t + 2 * d + rng.normal(0, 0.5))
                if x:
                    row["x"] = float(u % 4)
                rows.append(row)
        return pd.DataFrame(rows)

    def test_psm_requires_covariates(self):
        with pytest.raises(ValueError, match="requires covariates"):
            LWDiD(rolling="demean", estimation_method="psm").fit(self._panel(x=False), **self.KW)

    def test_psm_bootstrap_rejected_common_timing(self):
        with pytest.raises(ValueError, match="does not support n_bootstrap"):
            LWDiD(rolling="demean", estimation_method="psm", n_bootstrap=50).fit(
                self._panel(), covariates=["x"], **self.KW
            )

    def test_psm_logit_failure_fails_closed(self, monkeypatch):
        import diff_diff.lwdid as lwdid_mod

        est = LWDiD(rolling="demean", estimation_method="psm")
        rng = np.random.default_rng(1)
        y = rng.normal(size=20)
        treatment = np.array([1.0] * 8 + [0.0] * 12)
        controls = rng.normal(size=(20, 1))
        # Non-finite PROBABILITIES = genuine solver failure (round 19:
        # NaN coefs with finite probs is now the reduced-rank
        # continuation path, not the fallback).
        monkeypatch.setattr(
            lwdid_mod,
            "solve_logit",
            lambda X, d: (np.array([np.nan, np.nan]), np.full(len(d), np.nan)),
        )
        with pytest.warns(UserWarning, match="PSM fail-closed"):
            att, se, coefs, vcov, _, influence = est._estimate_psm(y, treatment, controls, None, 20)
        assert np.isfinite(att)
        assert np.isnan(se)
        assert coefs is None and vcov is None and influence is None

    def test_bootstrap_inference_basis_provenance(self):
        df = self._panel(x=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = LWDiD(rolling="demean").fit(df, **self.KW)
            boot = LWDiD(rolling="demean", n_bootstrap=49, seed=3).fit(df, **self.KW)
        assert plain.inference_basis is None
        assert boot.inference_basis == "unit_bootstrap"
        assert "bootstrap" in boot.summary()
        assert boot.to_dict()["inference_basis"] == "unit_bootstrap"
        df["cl"] = df["unit"] % 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cboot = LWDiD(rolling="demean", n_bootstrap=49, seed=3, cluster="cl").fit(df, **self.KW)
        assert cboot.inference_basis == "cluster_bootstrap"

    def test_diagnostics_shares_fit_validation(self):
        df = self._panel(x=False)
        bad = df.copy()
        bad["treat"] = bad["treat"] * 2  # non-binary
        with pytest.raises(ValueError):
            LWDiD(rolling="demean").get_transformation_diagnostics(
                bad, outcome="y", unit="unit", time="time", treatment="treat"
            )
        dup = pd.concat([df, df.iloc[:6]])  # duplicate unit-time rows
        with pytest.raises(ValueError, match="duplicate"):
            LWDiD(rolling="demean").get_transformation_diagnostics(
                dup, outcome="y", unit="unit", time="time", treatment="treat"
            )

    def test_duplicate_covariates_rejected(self):
        with pytest.raises(ValueError, match="duplicate column"):
            LWDiD(rolling="demean").fit(self._panel(), covariates=["x", "x"], **self.KW)


class TestReviewRound4Guards:
    """Local-review round 4: execution-verified guards.

    - RI and the WCR wrapper fit via np.linalg.lstsq, so a control
      duplicating treatment returned a finite MINIMUM-NORM ATT (probe:
      true ATT 2.0 reported as 0.877 with finite p-values in both)
    - staggered NT-only cells could estimate on a single surviving
      control after transformation drops
    - unobserved staggered anchors were synthesized as zero-valued
      reference rows
    - the standalone WCR wrapper accepted n_bootstrap=1; the result-level
      wrapper ignored the fitted alpha
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    @staticmethod
    def _arrays(seed=0, n=40):
        rng = np.random.default_rng(seed)
        treat = np.array([1.0] * (n // 2) + [0.0] * (n // 2))
        y = 2.0 * treat + rng.normal(0, 1, n)
        cl = np.arange(n) % 8
        return y, treat, cl, rng

    def test_collinear_control_identified_in_ri_and_wcb(self):
        # Pre-fix, lstsq split the effect across the duplicate columns
        # (minimum-norm: true ATT 2.0 reported as ~0.88). The shared
        # rank-aware solver pivots, keeps the treatment column, drops the
        # duplicate control, and reports the IDENTIFIED ATT (here exactly
        # the difference in means) with a rank warning. The raise branch
        # remains as a backstop should the treatment column itself be
        # pivoted out.
        from diff_diff.lwdid_randomization import randomization_inference
        from diff_diff.lwdid_wild_bootstrap import wild_cluster_bootstrap

        y, treat, cl, rng = self._arrays()
        dup = treat.reshape(-1, 1).copy()
        truth = y[treat == 1].mean() - y[treat == 0].mean()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ri = randomization_inference(y, treat, controls=dup, n_reps=49, seed=3)
            wb = wild_cluster_bootstrap(y, treat, cl, controls=dup, n_bootstrap=49, seed=3)
        np.testing.assert_allclose(ri.att_observed, truth, rtol=1e-12)
        np.testing.assert_allclose(wb.att, truth, rtol=1e-12)
        assert any("rank" in str(x.message).lower() for x in caught)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # valid controls: studentization stays coherent
            x = rng.normal(size=(len(y), 1))
            wb2 = wild_cluster_bootstrap(y, treat, cl, controls=x, n_bootstrap=49, seed=3)
            np.testing.assert_allclose(wb2.t_stat_original, wb2.att / wb2.se)

    def test_wcb_n_bootstrap_validation(self):
        from diff_diff.lwdid_wild_bootstrap import wild_cluster_bootstrap

        y, treat, cl, _ = self._arrays()
        for bad in (1, 0, -3, 2.5, True):
            with pytest.raises(ValueError, match="integer >= 2"):
                wild_cluster_bootstrap(y, treat, cl, n_bootstrap=bad)

    def test_ri_n_reps_validation(self):
        from diff_diff.lwdid_randomization import randomization_inference

        y, treat, _, _ = self._arrays()
        for bad in (0, -1, 99.5, True):
            with pytest.raises(ValueError, match="n_reps must be"):
                randomization_inference(y, treat, n_reps=bad)

    def test_results_wcb_inherits_fitted_alpha(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(16):
            for t in range(1, 7):
                d = 1 if (u < 8 and t >= 4) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=1 + 2 * d + rng.normal(0, 0.5)))
        df = pd.DataFrame(rows)
        df["cl"] = df["unit"] % 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", alpha=0.10, cluster="cl").fit(df, **self.KW)
            wb = res.wild_cluster_bootstrap(n_bootstrap=49, seed=1)
            assert wb.alpha == 0.10
            wb2 = res.wild_cluster_bootstrap(n_bootstrap=49, seed=1, alpha=0.05)
            assert wb2.alpha == 0.05

    def test_nt_only_cell_needs_two_surviving_controls(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(8):
            g = 4 if u < 6 else 0  # 6 treated, exactly 2 never-treated
            for t in range(1, 7):
                d = int(g > 0 and t >= g)
                y = 1 + 0.5 * t + d + rng.normal(0, 0.3)
                if u == 6 and t == 5:
                    y = np.nan  # one NT control loses its t=5 outcome
                rows.append(dict(unit=u, time=t, treat=d, g=g, y=y))
        df = pd.DataFrame(rows).dropna(subset=[]).copy()
        df.loc[(df.unit == 6) & (df.time == 5), "y"] = np.nan
        df = df.dropna(subset=["y"]) if False else df
        # NaN y raises in validation; drop the row instead (unbalanced panel)
        df = df[~((df.unit == 6) & (df.time == 5))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", control_group="never_treated").fit(
                df, first_treat="g", **self.KW
            )
        cell = res.cohort_time_effects[(4, 5)]
        assert cell["skip_reason"] == "insufficient_never_treated_controls"
        assert np.isnan(cell["att"])
        # other post cells still estimated with both controls
        assert np.isfinite(res.cohort_time_effects[(4, 4)]["att"])

    def test_unobserved_anchor_not_synthesized(self):
        rng = np.random.default_rng(0)
        rows = []
        times = [1, 2, 3, 5, 6]  # time 4 (= g-1 anchor for g=5) missing
        for u in range(10):
            g = 5 if u < 5 else 0
            for t in times:
                d = int(g > 0 and t >= g)
                rows.append(
                    dict(unit=u, time=t, treat=d, g=g, y=1 + 0.3 * t + d + rng.normal(0, 0.3))
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", control_group="never_treated").fit(
                df, first_treat="g", **self.KW
            )
        assert res.reference_periods == ()  # anchor r=-1 unobserved -> not emitted


class TestReviewRound5Guards:
    """Local-review round 5: execution-verified guards.

    - post-fit WCR/RI accepted arbitrary arrays + a non-interacted design,
      caching p-values for a DIFFERENT estimand than .att (probe: fitted
      3.98 vs tested 3.26 on a covariate-unbalanced RA fit) - now replay
      the fit spec (pinned in test_lwdid_wild_bootstrap.py)
    - a requested bootstrap overwrote the single-effective-cluster
      fail-closed NaN inference with a near-zero SE from the raw cluster
      map
    - aggregate(balance_e=) was accepted but silently ignored
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_bootstrap_preserves_single_cluster_fail_closed(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(12):
            cl = 0 if u < 6 else 1
            # cluster 1 units observe only ONE pre period: detrend needs 2,
            # so their transformed outcomes are NaN and the whole cluster
            # drops from the collapsed cross-section
            times = range(3, 7) if cl == 1 else range(1, 7)
            for t in times:
                d = 1 if (u % 6 < 3 and t >= 4) else 0
                rows.append(
                    dict(unit=u, time=t, treat=d, cl=cl, y=1 + 0.5 * t + 2 * d + rng.normal(0, 0.3))
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = LWDiD(rolling="detrend", cluster="cl", n_bootstrap=49, seed=1).fit(df, **self.KW)
        assert np.isfinite(res.att)
        from tests.conftest import assert_nan_inference

        assert_nan_inference(
            {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
        )
        assert any("bootstrap skipped" in str(x.message) for x in caught)
        assert res.inference_basis is None  # no bootstrap ran

    def test_balance_e_rejected(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(16):
            g = 4 if u < 4 else (5 if u < 8 else 0)
            for t in range(1, 8):
                d = int(g > 0 and t >= g)
                rows.append(
                    dict(unit=u, time=t, treat=d, g=g, y=1 + 0.3 * t + d + rng.normal(0, 0.3))
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean").fit(df, first_treat="g", **self.KW)
        with pytest.raises((ValueError, TypeError), match="balance_e"):
            res.aggregate("event_study", balance_e=1)
        # without balance_e the aggregation still works
        assert res.aggregate("event_study") is not None


class TestReviewRound6Guards:
    """Local-review round 6: execution-verified guards.

    - fweight + HC2/HC3 used the WLS-hat (weighted) leverage, so the
      compressed variance was up to ~5x the literal np.repeat expansion
      (fweights are replicated data by definition)
    - the common-timing headline averaged whichever post periods each
      unit observed, letting calendar composition masquerade as ATT
    - the degenerate-SE guard's max(1, |effect|) floor NaN'd valid
      inference under outcome rescaling
    - complete-case drops could empty an arm and dispatch a one-arm design
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_fweight_hc2_hc3_expansion_parity(self):
        from diff_diff.linalg import solve_ols

        X = np.column_stack([np.ones(5), np.arange(5.0)])
        y = np.array([1.0, 2.2, 2.9, 4.1, 5.3])
        w = np.array([3.0, 1.0, 2.0, 1.0, 4.0])
        Xe = np.repeat(X, w.astype(int), axis=0)
        ye = np.repeat(y, w.astype(int))
        for vt in ("hc2", "hc3"):
            _, _, v_c = solve_ols(
                X, y, return_vcov=True, vcov_type=vt, weights=w, weight_type="fweight"
            )
            _, _, v_e = solve_ols(Xe, ye, return_vcov=True, vcov_type=vt)
            np.testing.assert_allclose(np.diag(v_c), np.diag(v_e), rtol=1e-12, err_msg=vt)

    def test_fixed_window_complete_case_headline(self):
        # Zero-effect panel, Y_it = t: half the controls miss the last
        # post period. Pre-fix their shorter post average biased the
        # headline; complete-case drops them (warned) and ATT ~ 0.
        rows = []
        for u in range(12):
            treated = u < 6
            t_max = 6 if (treated or u < 9) else 5  # controls 9-11 miss t=6
            for t in range(1, t_max + 1):
                d = 1 if (treated and t >= 4) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=float(t)))
        df = pd.DataFrame(rows)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = LWDiD(rolling="demean").fit(df, **self.KW)
        assert any("fixed-window" in str(x.message) for x in caught)
        np.testing.assert_allclose(res.att, 0.0, atol=1e-10)
        assert res.n_control == 3  # complete controls only

    def test_se_guard_scale_equivariant(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(16):
            g = 4 if u < 8 else 0
            for t in range(1, 7):
                d = int(g > 0 and t >= g)
                rows.append(
                    dict(unit=u, time=t, treat=d, g=g, y=1 + 0.3 * t + d + rng.normal(0, 0.3))
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = LWDiD(rolling="demean").fit(df, first_treat="g", **self.KW)
            df2 = df.assign(y=df["y"] * 1e-10)
            r2 = LWDiD(rolling="demean").fit(df2, first_treat="g", **self.KW)
        # t-statistic is invariant to outcome rescaling
        np.testing.assert_allclose(r2.t_stat, r1.t_stat, rtol=1e-8)
        np.testing.assert_allclose(r2.att, r1.att * 1e-10, rtol=1e-8)
        assert np.isfinite(r2.se)

    def test_empty_arm_after_drops_raises(self):
        # All treated units observe only one pre period -> detrend NaNs
        # every treated unit; pre-fix a one-arm design was dispatched.
        rng = np.random.default_rng(0)
        rows = []
        for u in range(8):
            treated = u < 4
            times = range(3, 7) if treated else range(1, 7)
            for t in times:
                d = 1 if (treated and t >= 4) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=1 + 0.5 * t + rng.normal(0, 0.3)))
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="at\\s+least one of each"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                LWDiD(rolling="detrend").fit(df, **self.KW)

    def test_constructor_numeric_validation(self):
        with pytest.raises(ValueError, match="pscore_trim"):
            LWDiD(pscore_trim=True)
        with pytest.raises(ValueError, match="pscore_trim"):
            LWDiD(pscore_trim="0.1")
        with pytest.raises(ValueError, match="n_jobs"):
            LWDiD(n_jobs=True)

    def test_df_inference_serializes(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(12):
            for t in range(1, 7):
                d = 1 if (u < 6 and t >= 4) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=1 + 2 * d + rng.normal(0, 0.4)))
        res = LWDiD(rolling="demean").fit(pd.DataFrame(rows), **self.KW)
        d = res.to_dict()
        assert d["df_inference"] == res.df_inference


class TestReviewRound7Guards:
    """Local-review round 7: execution-verified guards.

    - the post-fit replay always rebuilt the covariate interactions, but
      the fit uses plain (1, D, X) when an arm has N <= K+1 (LW eq. 3.3
      gate) - small-arm fits' replayed statistic mismatched .att and the
      round-5 coherence assert made their inference unusable
    - the sensitivity helpers swallowed treatment-design violations
      (absorbing/onset/cohort-consistency) as not_estimable specs
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_small_arm_plain_design_replay(self):
        # K=1 covariate, exactly 2 treated units: n_treated <= K+1, so the
        # fit uses the plain design; the replay must follow it.
        rng = np.random.default_rng(3)
        rows = []
        for u in range(12):
            treated = u < 2
            x = float(u % 4)
            for t in range(1, 7):
                d = 1 if (treated and t >= 4) else 0
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        treat=d,
                        x=x,
                        cl=u % 4,
                        y=1 + 0.4 * x + 2 * d + rng.normal(0, 0.3),
                    )
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", cluster="cl").fit(df, covariates=["x"], **self.KW)
            ri = res.randomization_test(n_reps=99, seed=1)
            wb = res.wild_cluster_bootstrap(n_bootstrap=49, seed=1)
        np.testing.assert_allclose(ri.att_observed, res.att, rtol=1e-10)
        np.testing.assert_allclose(wb.att, res.att, rtol=1e-10)

    def test_sensitivity_rejects_design_violations(self):
        from diff_diff.lwdid_sensitivity import (
            robustness_pre_periods,
            sensitivity_no_anticipation,
        )

        rng = np.random.default_rng(0)
        rows = []
        for u in range(10):
            for t in range(1, 9):
                d = 1 if (u < 5 and t >= 6) else 0
                if u == 0 and t == 7:
                    d = 0  # 1 -> 0 reversal: non-absorbing treatment
                rows.append(dict(unit=u, time=t, treat=d, y=rng.normal() + d))
        df = pd.DataFrame(rows)
        for fn in (robustness_pre_periods, sensitivity_no_anticipation):
            with pytest.raises(ValueError, match="absorbing|revert"):
                fn(df, outcome="y", unit="unit", time="time", treatment="treat")
        # heterogeneous onsets without first_treat: also a raise, not
        # a silent not_estimable
        rows2 = []
        for u in range(10):
            onset = 5 if u < 3 else (6 if u < 5 else 99)
            for t in range(1, 9):
                d = 1 if (u < 5 and t >= onset) else 0
                rows2.append(dict(unit=u, time=t, treat=d, y=rng.normal() + d))
        df2 = pd.DataFrame(rows2)
        with pytest.raises(ValueError, match="common timing|first_treat|onset"):
            robustness_pre_periods(df2, outcome="y", unit="unit", time="time", treatment="treat")


class TestReviewRound8Guards:
    """Local-review round 8: execution-verified guards.

    - the per-period max(D) partition classified a post period with no
      observed treated rows as PRE-treatment (zero-effect trend probe:
      ATT 0.75); the partition now derives from the single onset S
    - the common-timing onset check rejected units missing their t = S
      row as heterogeneous timing (the staggered branch permits it)
    - the tau_omega completeness check accepted any finite post average,
      so a control observing part of a cohort's window survived
    - Inf covariates passed the NaN check; non-numeric covariates crashed
      with raw conversion errors
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    @staticmethod
    def _trend_panel(drop=lambda u, t: False, n_units=12, onset=4, t_max=6, effect=0.0):
        rows = []
        for u in range(n_units):
            treated = u < n_units // 2
            for t in range(1, t_max + 1):
                if drop(u, t) and treated:
                    continue
                d = 1 if (treated and t >= onset) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=float(t) + effect * d))
        return pd.DataFrame(rows)

    def test_controls_only_post_period_not_misclassified(self):
        # 2 of 6 treated units miss post period t=5: pre-fix that period
        # was classified as pre (contaminating the pre window, ATT 0.75 on
        # this zero-effect trend); now it stays post and the incomplete
        # treated units are complete-case dropped.
        df = self._trend_panel(drop=lambda u, t: u < 2 and t == 5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean").fit(df, **self.KW)
        np.testing.assert_allclose(res.att, 0.0, atol=1e-10)
        assert res.n_treated == 4
        # ALL treated missing the period -> no fixed-window comparison
        df_all = self._trend_panel(drop=lambda u, t: t == 5)
        with pytest.raises(ValueError, match="at\\s+least one of each"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                LWDiD(rolling="demean").fit(df_all, **self.KW)

    def test_missing_onset_row_accepted_common_timing(self):
        df = self._trend_panel(drop=lambda u, t: u == 0 and t == 4, effect=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean").fit(df, **self.KW)
        np.testing.assert_allclose(res.att, 2.0, atol=1e-10)
        # genuinely heterogeneous onsets still rejected
        rows = []
        for u in range(8):
            onset = 4 if u < 2 else (5 if u < 4 else 99)
            for t in range(1, 7):
                d = 1 if (u < 4 and t >= onset) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=float(t)))
        with pytest.raises(ValueError, match="heterogeneous|common onset"):
            LWDiD(rolling="demean").fit(pd.DataFrame(rows), **self.KW)

    def test_tau_omega_partial_window_semantics_pinned(self):
        # ADJUDICATED (round 8): completeness = a finite average over the
        # OBSERVED post-g rows, symmetric across arms - a control missing
        # one window period is RETAINED (its component averages observed
        # rows); a control missing the ENTIRE window is dropped. The
        # acceptance suite's frozen reference oracle pins the same rule.
        def build(missing):
            rows = []
            for u in range(12):
                g = 5 if u < 6 else 0
                for t in range(1, 9):
                    if u == 11 and t in missing:
                        continue
                    d = int(g > 0 and t >= g)
                    rows.append(dict(unit=u, time=t, treat=d, g=g, y=float(t)))
            return pd.DataFrame(rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            part = LWDiD(
                rolling="demean", control_group="never_treated", vcov_type="classical"
            ).fit(build({8}), first_treat="g", **self.KW)
            whole = LWDiD(
                rolling="demean", control_group="never_treated", vcov_type="classical"
            ).fit(build({5, 6, 7, 8}), first_treat="g", **self.KW)
        assert part.n_composite_controls_dropped == 0  # partial window retained
        assert whole.n_composite_controls_dropped == 1  # entire window missing
        # Documented composition caveat, pinned exactly: with y = t and
        # demean (cohort-5 pre-mean 2.5), complete units average ydot
        # over t=5..8 (=4.0) while the partial control averages t=5..7
        # (=3.5), so tau_omega = 4.0 - (5*4.0 + 3.5)/6 = 1/12.
        np.testing.assert_allclose(part.att, 1.0 / 12.0, atol=1e-10)
        np.testing.assert_allclose(whole.att, 0.0, atol=1e-10)

    def test_nonfinite_and_nonnumeric_covariates_rejected(self):
        df = self._trend_panel()
        df["x"] = 1.0
        df.loc[df.index[3], "x"] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            LWDiD(rolling="demean").fit(df, covariates=["x"], **self.KW)
        df["x2"] = "a"
        with pytest.raises(ValueError, match="not numeric"):
            df_ok = df.assign(x=1.0)
            LWDiD(rolling="demean").fit(df_ok, covariates=["x2"], **self.KW)

    def test_validate_staggered_data_rejects_mixed_families(self):
        from diff_diff.lwdid import validate_staggered_data

        rows = []
        for u in range(6):
            g = pd.Period("2020Q1", freq="Q") if u < 3 else pd.NaT
            for i, ts in enumerate(pd.date_range("2019-01-01", periods=6, freq="QS")):
                d = int(u < 3 and i >= 4)
                rows.append(dict(unit=u, time=ts, treat=d, g=g, y=float(i)))
        df = pd.DataFrame(rows)
        df["g"] = pd.PeriodIndex(df["g"], freq="Q")
        out = validate_staggered_data(df, unit="unit", time="time", cohort="g")
        assert out["valid"] is False
        assert any("same time scale" in e for e in out["errors"])


class TestReviewRound9Guards:
    """Local-review round 9: execution-verified guards.

    - staggered aggregation stored effects under int(t - g), silently
      merging distinct fractional horizons; the common interface used
      positional labels while staggered numeric used arithmetic, so the
      same gapped design got different event keys per interface
    - the round-8 onset partition was not propagated to diagnostics and
      the sensitivity helpers
    - Inf outcomes passed the NaN check and were silently cell-filtered
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_fractional_horizons_fail_closed(self):
        rows = []
        times = [0.5, 1.0, 1.5, 2.0, 2.5]
        for u in range(10):
            g = 1.5 if u < 5 else 0
            for t in times:
                d = int(g > 0 and t >= g)
                rows.append(dict(unit=u, time=t, treat=d, g=g, y=float(t) + d))
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="not an integer"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                LWDiD(rolling="demean").fit(df, first_treat="g", **self.KW)

    def test_gapped_calendar_common_staggered_label_parity(self):
        # {1, 2, 4, 6} with onset 4: both interfaces must label events by
        # arithmetic t - g on numeric calendars (pre-fix: common reported
        # {0, 1} positional while staggered reported {0, 2}).
        rows = []
        for u in range(12):
            treated = u < 6
            for t in (1, 2, 4, 6):
                d = 1 if (treated and t >= 4) else 0
                rows.append(
                    dict(unit=u, time=t, treat=d, g=4 if treated else 0, y=float(t) + 2 * d)
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            common = LWDiD(rolling="demean").fit(df, **self.KW)
            stag = LWDiD(rolling="demean", control_group="never_treated").fit(
                df, first_treat="g", **self.KW
            )
        common_post = sorted(common.event_study_effects)
        stag_post = sorted(k for k, v in stag.event_study_effects.items() if k >= 0)
        assert common_post == [0, 2]
        assert stag_post == [0, 2]

    def test_diagnostics_and_sensitivity_use_onset_partition(self):
        from diff_diff.lwdid_sensitivity import _get_pre_periods

        # controls-only post period t=5 (all treated rows missing there)
        rows = []
        for u in range(12):
            treated = u < 6
            for t in range(1, 7):
                if treated and t == 5:
                    continue
                d = 1 if (treated and t >= 4) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=float(t)))
        df = pd.DataFrame(rows)
        pre = _get_pre_periods(df, "time", "treat")
        assert list(pre) == [1, 2, 3]  # t=5 stays POST despite no treated rows
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diag = LWDiD(rolling="demean").get_transformation_diagnostics(
                df, outcome="y", unit="unit", time="time", treatment="treat"
            )
        # control units' pre window excludes t=5: with y = t the pre mean
        # over {1,2,3} is 2.0 for every unit
        assert diag is not None

    def test_inf_outcome_rejected(self):
        rows = []
        for u in range(8):
            for t in range(1, 7):
                d = 1 if (u < 4 and t >= 4) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=1.0 + d))
        df = pd.DataFrame(rows)
        df.loc[df.index[5], "y"] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            LWDiD(rolling="demean").fit(df, **self.KW)
        df["g"] = np.where(df["unit"] < 4, 4, 0)
        with pytest.raises(ValueError, match="non-finite"):
            LWDiD(rolling="demean").fit(df, first_treat="g", **self.KW)


class TestReviewRound10Guards:
    """Local-review round 10: execution-verified guards.

    - the generic over-one-leverage HC1 fallback ran BEFORE hc3's
      fail-closed check, so numerically over-one designs got an HC1
      result still labeled hc3 (and a clipped hc3 influence vector)
    - the sensitivity multi-cohort count used RAW cohorts, rejecting
      valid single-cohort designs with beyond-window encodings
    - zero-post-row units evaded the fixed-window drop warning
    - baseline sensitivity fits swallowed config errors as not_estimable
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_hc3_over_one_leverage_fails_closed(self):
        from diff_diff.linalg import compute_robust_vcov

        rng = np.random.default_rng(0)
        # near-duplicate rows -> numerically over-one leverage is hard to
        # force deterministically; drive the guard directly with h >= 1
        X = np.column_stack([np.ones(4), np.array([0.0, 0.0, 0.0, 1.0])])
        y = np.array([1.0, 1.1, 0.9, 5.0])
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        with pytest.warns(UserWarning, match="HC3 variance is undefined"):
            v = compute_robust_vcov(X, resid, vcov_type="hc3")
        assert np.all(np.isnan(v))
        del rng

    def test_sensitivity_accepts_beyond_window_single_cohort(self):
        from diff_diff.lwdid_sensitivity import robustness_pre_periods

        rng = np.random.default_rng(0)
        rows = []
        for u in range(16):
            # one real cohort (5); 4 units carry a beyond-window encoding
            # (99) that normalizes to never-treated; rest never-treated
            g = 5 if u < 6 else (99 if u < 10 else 0)
            for t in range(1, 10):
                d = int(g == 5 and t >= 5)
                rows.append(dict(unit=u, time=t, treat=d, g=g, y=rng.normal() + d))
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = robustness_pre_periods(
                df, outcome="y", unit="unit", time="time", treatment="treat", cohort="g"
            )
        assert np.isfinite(res.baseline_att)

    def test_zero_post_unit_counted_in_drop_warning(self):
        rows = []
        for u in range(12):
            treated = u < 6
            t_range = range(1, 4) if u == 11 else range(1, 7)  # unit 11: pre rows only
            for t in t_range:
                d = 1 if (treated and t >= 4) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=float(t) + 2 * d))
        df = pd.DataFrame(rows)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = LWDiD(rolling="demean").fit(df, **self.KW)
        assert any("fixed-window" in str(x.message) for x in caught)
        assert res.n_control == 5  # unit 11 dropped and accounted for

    def test_sensitivity_baseline_config_errors_raise(self):
        from diff_diff.lwdid_sensitivity import (
            robustness_pre_periods,
            sensitivity_no_anticipation,
        )

        rng = np.random.default_rng(0)
        rows = []
        for u in range(12):
            for t in range(1, 9):
                d = 1 if (u < 6 and t >= 6) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=rng.normal() + d))
        df = pd.DataFrame(rows)
        for fn in (robustness_pre_periods, sensitivity_no_anticipation):
            with pytest.raises(ValueError, match="requires covariates"):
                fn(
                    df,
                    outcome="y",
                    unit="unit",
                    time="time",
                    treatment="treat",
                    estimation_method="psm",
                )


class TestReviewRound11Guards:
    """Local-review round 11: execution-verified guards.

    - the IPW/DR logit score/Hessian used CLIPPED propensities, breaking
      the estimating-equation linearization whenever trimming fired (the
      MLE's score is ~0 in the RAW fitted probabilities only), and clipped
      observations kept a nonzero weight-derivative
    - NaN logit coefficients from a rank-deficient (collinear) propensity
      model were treated as non-convergence and silently substituted
      regression adjustment under ipw/dr provenance
    - the RA interaction gate counted NOMINAL covariate columns, so a
      perfectly collinear control flipped the eq. 3.3 design off and
      changed the ATT
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    @staticmethod
    def _panel(x_fn, n_units=24, extra=None):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(n_units):
            treated = u < n_units // 2
            x = x_fn(u)
            for t in range(1, 7):
                d = 1 if (treated and t >= 4) else 0
                row = dict(unit=u, time=t, treat=d, x=x, y=1 + 0.4 * x + 2 * d + rng.normal(0, 0.3))
                if extra is not None:
                    row["x2"] = extra(x)
                rows.append(row)
        return pd.DataFrame(rows)

    def test_rank_deficient_propensity_stays_ipw(self):
        # x2 = 2x: the logit drops a column (NaN coef) but the fitted
        # probabilities are valid - the fit must REMAIN IPW (pre-fix it
        # silently became regression adjustment under ipw provenance).
        df_full = self._panel(lambda u: float(u % 5))
        df_dup = self._panel(lambda u: float(u % 5), extra=lambda x: 2.0 * x)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r_dup = LWDiD(rolling="demean", estimation_method="ipw").fit(
                df_dup, covariates=["x", "x2"], **self.KW
            )
            r_ipw = LWDiD(rolling="demean", estimation_method="ipw").fit(
                df_full, covariates=["x"], **self.KW
            )
            r_reg = LWDiD(rolling="demean", estimation_method="reg").fit(
                df_dup, covariates=["x", "x2"], **self.KW
            )
        assert any("reduced-rank propensity" in str(x.message) for x in caught)
        # the duplicated-column IPW fit equals the identified IPW fit,
        # NOT the regression-adjustment fit
        np.testing.assert_allclose(r_dup.att, r_ipw.att, rtol=1e-10)
        assert abs(r_dup.att - r_reg.att) > 1e-12 or abs(r_dup.se - r_reg.se) > 1e-12

    def test_redundant_control_does_not_change_ra_estimand(self):
        df_full = self._panel(lambda u: float(u % 3), n_units=8)
        df_dup = self._panel(lambda u: float(u % 3), n_units=8, extra=lambda x: 2.0 * x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = LWDiD(rolling="demean", estimation_method="reg").fit(
                df_full, covariates=["x"], **self.KW
            )
            dup = LWDiD(rolling="demean", estimation_method="reg").fit(
                df_dup, covariates=["x", "x2"], **self.KW
            )
        # identical identified design -> identical ATT (pre-fix: the
        # nominal K flipped the interaction gate and moved the point)
        np.testing.assert_allclose(dup.att, base.att, rtol=1e-10)

    def test_trimmed_propensity_score_uses_raw_fit(self):
        # Strong-heterogeneity DGP that activates trimming: the logit
        # score at the MLE, as constructed by the IF code path, must be
        # ~0 (raw probabilities), not the clipped-probability residual.
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(3)
        n = 300
        x = rng.normal(0, 2.5, n)
        p = 1 / (1 + np.exp(-2.5 * x))
        d = (rng.random(n) < p).astype(float)
        X = x.reshape(-1, 1)
        coefs, probs_raw = solve_logit(X, d)
        trim_lo, trim_hi = 0.01, 0.99
        assert ((probs_raw < trim_lo) | (probs_raw > trim_hi)).any()  # trimming active
        X_ps = np.column_stack([np.ones(n), X])
        score_raw = ((d - probs_raw)[:, None] * X_ps).sum(axis=0)
        probs_clipped = np.clip(probs_raw, trim_lo, trim_hi)
        score_clipped = ((d - probs_clipped)[:, None] * X_ps).sum(axis=0)
        assert np.abs(score_raw).max() < 1e-6  # MLE estimating equation
        assert np.abs(score_clipped).max() > 1e-2  # the pre-fix construction


class TestReviewRound12Guards:
    """Local-review round 12: execution-verified guards.

    - the DR outcome WLS inverted the raw nominal Gram (not rank-aware /
      scale-equilibrated): an exactly redundant 1e12-rescaled duplicate
      changed the DR SE by ~2.5x
    - IPW/DR returned NOMINAL parameter counts, so a redundant control
      shrank residual df and moved p-values/CIs with ATT/SE unchanged
    - the drops-route staggered aggregation weighted cohorts by RAW
      masses, keeping dropped treated units in the cohort weights
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    @staticmethod
    def _panel(extra_scale=None, n_units=24):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(n_units):
            treated = u < n_units // 2
            x = float(u % 5)
            for t in range(1, 7):
                d = 1 if (treated and t >= 4) else 0
                row = dict(unit=u, time=t, treat=d, x=x, y=1 + 0.4 * x + 2 * d + rng.normal(0, 0.3))
                if extra_scale is not None:
                    row["x2"] = extra_scale * x
                rows.append(row)
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("method", ["ipw", "dr"])
    @pytest.mark.parametrize("scale", [2.0, 1e12])
    def test_redundant_control_full_inference_invariance(self, method, scale):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = LWDiD(rolling="demean", estimation_method=method).fit(
                self._panel(), covariates=["x"], **self.KW
            )
            dup = LWDiD(rolling="demean", estimation_method=method).fit(
                self._panel(extra_scale=scale), covariates=["x", "x2"], **self.KW
            )
        np.testing.assert_allclose(dup.att, base.att, rtol=1e-8, err_msg=f"{method} att")
        np.testing.assert_allclose(dup.se, base.se, rtol=1e-8, err_msg=f"{method} se")
        assert dup.df_inference == base.df_inference, f"{method} df"
        np.testing.assert_allclose(dup.p_value, base.p_value, rtol=1e-8)
        np.testing.assert_allclose(dup.conf_int, base.conf_int, rtol=1e-8)

    def test_drops_route_uses_survivor_cohort_masses(self):
        # Independent oracle: cohorts {3: 4 units, 5: 4 units}; ONE
        # cohort-5 treated unit observes only t=1..4 (missing its own post
        # window entirely) -> dropped. Survivor masses 4/7 and 3/7 must
        # weight the cohort effects (raw masses would use 4/8, 4/8).
        rng = np.random.default_rng(1)
        rows = []
        uid = 0
        spec = [(0, 8, None), (3, 4, None), (5, 3, None), (5, 1, (1, 2, 3, 4))]
        for g, n, keep in spec:
            for _ in range(n):
                alpha = rng.normal()
                for t in range(1, 7):
                    if keep is not None and t not in keep:
                        continue
                    d = int(g > 0 and t >= g)
                    y = alpha + 0.2 * t + rng.normal(scale=0.3) + (1.5 + 0.4 * (g == 5)) * d
                    rows.append(dict(unit=uid, time=t, treat=d, g=g, y=y))
                uid += 1
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", control_group="never_treated", vcov_type="classical").fit(
                df, first_treat="g", **self.KW
            )
        assert res.n_composite_treated_dropped == 1
        att3 = res.cohort_effects[3]["att"]
        att5 = res.cohort_effects[5]["att"]
        expected = (4.0 * att3 + 3.0 * att5) / 7.0  # SURVIVOR masses
        raw_weighted = (4.0 * att3 + 4.0 * att5) / 8.0
        np.testing.assert_allclose(res.att, expected, rtol=1e-12)
        assert abs(res.att - raw_weighted) > 1e-6  # distinguishes the rules


class TestReviewRound13Guards:
    """Local-review round 13: execution-verified guards.

    - the RA influence bread was rebuilt with a RAW-Gram pinv after
      solve_ols's scale-equilibrated fit: at large covariate units the
      pinv silently dropped low-scale directions, so cell ATT/SE were
      invariant while every AGGREGATE SE/p/CI (and the multiplier-
      bootstrap inputs) depended on covariate units
    - the exact-inference guard counted nominal columns, rejecting
      redundant-column designs with positive effective residual df
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_aggregate_inference_invariant_to_covariate_units(self):
        rng = np.random.default_rng(2)

        def build(scale):
            rows = []
            for u in range(20):
                g = 4 if u < 5 else (5 if u < 10 else 0)
                x = float(u % 4) * scale
                for t in range(1, 8):
                    d = int(g > 0 and t >= g)
                    rows.append(
                        dict(
                            unit=u,
                            time=t,
                            treat=d,
                            g=g,
                            x=x,
                            y=1 + 0.2 * t + 0.3 * (x / scale) + 1.5 * d + rng.normal(0, 0.3),
                        )
                    )
            return pd.DataFrame(rows)

        df1 = build(1.0)
        rng = np.random.default_rng(2)  # same noise stream
        df2 = build(10.0**7.25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = LWDiD(rolling="demean").fit(df1, first_treat="g", covariates=["x"], **self.KW)
            r2 = LWDiD(rolling="demean").fit(df2, first_treat="g", covariates=["x"], **self.KW)
        np.testing.assert_allclose(r2.att, r1.att, rtol=1e-8)
        np.testing.assert_allclose(r2.se, r1.se, rtol=1e-6)  # aggregate IF SE
        np.testing.assert_allclose(r2.p_value, r1.p_value, rtol=1e-5, atol=1e-300)
        for k in r1.event_study_effects:
            np.testing.assert_allclose(
                r2.event_study_effects[k]["se"],
                r1.event_study_effects[k]["se"],
                rtol=1e-6,
                err_msg=f"event {k}",
            )

    def test_redundant_column_small_sample_fits(self):
        # 4 collapsed units, design [1, D, x, 2x]: effective rank 3,
        # residual df 1 -> must FIT (pre-fix: nominal width 4 raised).
        rows = []
        for u in range(4):
            x = float(u)
            for t in range(1, 5):
                d = 1 if (u < 2 and t >= 3) else 0
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        treat=d,
                        x=x,
                        x2=2.0 * x,
                        y=1 + 0.5 * t + 0.3 * x + 2 * d + 0.01 * u * t,
                    )
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean").fit(df, covariates=["x", "x2"], **self.KW)
        assert np.isfinite(res.att)
        # a genuinely saturated full-rank design still raises
        rows2 = []
        for u in range(3):
            for t in range(1, 5):
                d = 1 if (u < 1 and t >= 3) else 0
                rows2.append(
                    dict(unit=u, time=t, treat=d, x=float(u**2), y=1 + 0.5 * t + 2 * d + 0.01 * u)
                )
        with pytest.raises(ValueError, match="Invalid exact-inference design"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                LWDiD(rolling="demean").fit(pd.DataFrame(rows2), covariates=["x"], **self.KW)


class TestReviewRound14Guards:
    """Local-review round 14: Inf time values raised a raw OverflowError
    in event-time arithmetic instead of a validation error."""

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_nonfinite_time_rejected(self, bad):
        rows = []
        for u in range(8):
            for t in range(1, 7):
                d = 1 if (u < 4 and t >= 4) else 0
                rows.append(dict(unit=u, time=float(t), treat=d, y=1.0 + d))
        df = pd.DataFrame(rows)
        df.loc[df.index[2], "time"] = bad
        with pytest.raises(ValueError, match="Time column .* non-finite"):
            LWDiD(rolling="demean").fit(df, **self.KW)
        df["g"] = np.where(df["unit"] < 4, 4.0, 0.0)
        with pytest.raises(ValueError, match="Time column .* non-finite"):
            LWDiD(rolling="demean").fit(df, first_treat="g", **self.KW)


class TestReviewRound16Guards:
    """Local-review round 16: execution-verified guards.

    - a degenerate staggered event row (NaN inference) still contributed
      a 0.0-diagonal column to the ANALYTICAL event-study covariance
    - k_min=1 was silently clamped to 2, dropping a valid demeaning spec
    - the degenerate early-return results lost psm_config / cluster_name
    """

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_degenerate_event_row_excluded_from_analytical_vcov(self):
        from types import SimpleNamespace

        from diff_diff.lwdid_staggered import compute_event_study_bands

        rng = np.random.default_rng(0)
        estimator = SimpleNamespace(n_bootstrap=0, seed=None, alpha=0.05)
        event_effects = {
            0: {
                "effect": 1.0,
                "se": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "conf_int": (np.nan, np.nan),
                "df": None,
            },
            1: {
                "effect": 0.5,
                "se": 0.1,
                "t_stat": 5.0,
                "p_value": 0.0,
                "conf_int": (0.3, 0.7),
                "df": None,
            },
        }
        event_influence = {0: np.zeros(30), 1: rng.normal(size=30)}
        vcov, index, *_ = compute_event_study_bands(estimator, event_effects, event_influence, None)
        assert list(index) == [1]  # NaN-inference row excluded
        assert vcov.shape == (1, 1) and np.isfinite(vcov[0, 0])

    def test_k_min_one_honored_for_demean(self):
        from diff_diff.lwdid_sensitivity import robustness_pre_periods

        rng = np.random.default_rng(0)
        rows = []
        for u in range(12):
            for t in range(1, 9):
                d = 1 if (u < 6 and t >= 6) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=rng.normal() + d))
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = robustness_pre_periods(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                rolling="demean",
                k_min=1,
            )
        labels = [s.label for s in res.specifications]
        assert "k=1_pre_periods" in labels
        with pytest.raises(ValueError, match="minimum pre-period requirement"):
            robustness_pre_periods(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                rolling="detrend",
                k_min=1,
            )

    def test_degenerate_return_keeps_psm_and_cluster_provenance(self):
        # detrend with a single pre-period: every transformed outcome is
        # NaN -> the degenerate early return must still carry the fit
        # configuration.
        rows = []
        for u in range(8):
            for t in range(2, 7):  # one pre-period (t=2), onset t=3
                d = 1 if (u < 4 and t >= 3) else 0
                rows.append(dict(unit=u, time=t, treat=d, cl=u % 3, x=float(u % 4), y=1.0 + d))
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_psm = LWDiD(rolling="detrend", estimation_method="psm", caliper=0.5).fit(
                df, covariates=["x"], **self.KW
            )
            res_cl = LWDiD(rolling="detrend", cluster="cl").fit(df, **self.KW)
        assert np.isnan(res_psm.att)
        assert res_psm.psm_config is not None
        assert res_psm.psm_config["caliper"] == 0.5
        assert np.isnan(res_cl.att)
        assert res_cl.cluster_name == "cl"


class TestReviewRound17Guards:
    """Local-review round 17: the RA gates used matrix_rank's looser
    default tolerance, disagreeing with solve_ols's pivoted-QR 1e-7
    convention on NEAR-collinear controls (x2 = x + 1e-10): the gate
    could count two identified controls, turn the interacted design off,
    and move the ATT while the solver fit the identified single-control
    model."""

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    @staticmethod
    def _panel(near=False, n_units=8):
        rng = np.random.default_rng(4)
        rows = []
        for u in range(n_units):
            treated = u < n_units // 2
            x = float(u % 4)
            for t in range(1, 7):
                d = 1 if (treated and t >= 4) else 0
                row = dict(unit=u, time=t, treat=d, x=x, y=1 + 0.4 * x + 2 * d + rng.normal(0, 0.3))
                if near:
                    row["x2"] = x + 1e-10
                rows.append(row)
        return pd.DataFrame(rows)

    def test_near_collinear_control_matches_identified_design(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = LWDiD(rolling="demean", estimation_method="reg").fit(
                self._panel(), covariates=["x"], **self.KW
            )
            near = LWDiD(rolling="demean", estimation_method="reg").fit(
                self._panel(near=True), covariates=["x", "x2"], **self.KW
            )
        # same identified design under the SHARED rank convention: the
        # near-duplicate is dropped, the interaction gate stays on, and
        # the ATT matches the single-control fit
        np.testing.assert_allclose(near.att, base.att, rtol=1e-6)
        # replay coherence (the mirror uses the same shared detector)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ri = near.randomization_test(n_reps=49, seed=1)
        np.testing.assert_allclose(ri.att_observed, near.att, rtol=1e-10)


class TestReviewRound18Guards:
    """Local-review round 18: diagnostics bypassed the common-timing
    time-scale checks; RI applied the finite mask before shape checks."""

    def test_diagnostics_reject_period_detrend_and_string_time(self):
        est = LWDiD(rolling="detrend")
        periods = pd.period_range("2020Q1", periods=6, freq="Q")
        rows = []
        for u in range(6):
            for i, p in enumerate(periods):
                d = 1 if (u < 3 and i >= 4) else 0
                rows.append(dict(unit=u, time=p, treat=d, y=float(i) + d))
        df = pd.DataFrame(rows)
        df["time"] = pd.PeriodIndex(df["time"], freq="Q")
        with pytest.raises(ValueError, match="Period"):
            est.get_transformation_diagnostics(
                df, outcome="y", unit="unit", time="time", treatment="treat"
            )
        df2 = df.assign(time=[f"P{i%6}" for i in range(len(df))])
        with pytest.raises(ValueError, match="numeric or"):
            est.get_transformation_diagnostics(
                df2, outcome="y", unit="unit", time="time", treatment="treat"
            )

    def test_ri_shape_checks_precede_finite_mask(self):
        from diff_diff.lwdid_randomization import randomization_inference

        y = np.array([1.0, np.inf, 2.0, 3.0])
        with pytest.raises(ValueError, match="same length"):
            randomization_inference(y, np.array([1.0, 0.0]), n_reps=9)
        with pytest.raises(ValueError, match="controls must have"):
            randomization_inference(
                y,
                np.array([1.0, 0.0, 1.0, 0.0]),
                controls=np.zeros((2, 1)),
                n_reps=9,
            )


class TestReviewRound19Guards:
    """Local-review round 19: PSM treated a rank-deficient (finite-
    probability) propensity fit as non-convergence and substituted a
    regression-adjustment point under psm provenance (ipw/dr already
    continued reduced-rank)."""

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    @staticmethod
    def _panel(extra=False):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(24):
            treated = u < 12
            x = float(u % 5)
            for t in range(1, 7):
                d = 1 if (treated and t >= 4) else 0
                row = dict(unit=u, time=t, treat=d, x=x, y=1 + 0.4 * x + 2 * d + rng.normal(0, 0.3))
                if extra:
                    row["x2"] = 2.0 * x
                rows.append(row)
        return pd.DataFrame(rows)

    def test_psm_continues_reduced_rank(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            base = LWDiD(rolling="demean", estimation_method="psm").fit(
                self._panel(), covariates=["x"], **self.KW
            )
            dup = LWDiD(rolling="demean", estimation_method="psm").fit(
                self._panel(extra=True), covariates=["x", "x2"], **self.KW
            )
        assert any("reduced-rank" in str(x.message) for x in caught)
        # identical propensity fit -> identical matches -> identical ATT
        np.testing.assert_allclose(dup.att, base.att, rtol=1e-10)
        from tests.conftest import assert_nan_inference

        assert_nan_inference(
            {"se": dup.se, "t_stat": dup.t_stat, "p_value": dup.p_value, "conf_int": dup.conf_int}
        )


class TestReviewRound20Guards:
    """Local-review round 20: PSM was rejected by the exact-OLS df guard;
    staggered diagnostics returned an empty success on all-never-treated
    panels; ordered chronology enforced for non-numeric common time."""

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_psm_point_only_exempt_from_residual_df_guard(self):
        # 2 treated + 2 controls with 3 redundant unit-constant
        # covariates: nominal width exhausts an OLS df count PSM never
        # uses - the point-only matching fit must still run.
        rng = np.random.default_rng(0)
        rows = []
        for u in range(4):
            x = float(u)
            for t in range(1, 7):
                d = 1 if (u < 2 and t >= 4) else 0
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        treat=d,
                        x=x,
                        x2=2 * x,
                        x3=3 * x,
                        y=1 + 0.3 * x + 2 * d + rng.normal(0, 0.2),
                    )
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", estimation_method="psm").fit(
                df, covariates=["x", "x2", "x3"], **self.KW
            )
        assert np.isfinite(res.att)
        assert res.df_inference is None
        from tests.conftest import assert_nan_inference

        assert_nan_inference(
            {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
        )

    def test_diagnostics_reject_all_never_treated(self):
        rows = []
        for u in range(6):
            for t in range(1, 7):
                rows.append(dict(unit=u, time=t, treat=0, g=0, y=float(t)))
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="No treated cohorts"):
            LWDiD(rolling="demean").get_transformation_diagnostics(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="g",
            )

    def test_ordered_categorical_nonlexicographic_chronology(self):
        # 'Q10' sorts before 'Q2' lexicographically; the declared order
        # must win (zero-effect trend panel -> att 0 under the correct
        # chronology).
        labels = [f"Q{i}" for i in range(1, 11)]  # Q1..Q10
        rng = np.random.default_rng(0)
        rows = []
        for u in range(10):
            for i, lab in enumerate(labels):
                d = 1 if (u < 5 and i >= 7) else 0
                rows.append(dict(unit=u, time=lab, treat=d, y=float(i)))
        df = pd.DataFrame(rows)
        df["time"] = pd.Categorical(df["time"], categories=labels, ordered=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean").fit(df, **self.KW)
        np.testing.assert_allclose(res.att, 0.0, atol=1e-12)
        del rng


class TestReviewRound21Guards:
    """Local-review round 21: hc2 fabricated finite inference at leverage
    one on the new LWDiD surface; the common-timing headline bypassed the
    degenerate-SE guard; plots rendered +/-1.96*SE instead of the fitted
    intervals."""

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_hc2_leverage_one_fails_closed_on_lwdid(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(12):
            for t in range(1, 7):
                d = 1 if (u < 1 and t >= 4) else 0  # single treated unit
                rows.append(dict(unit=u, time=t, treat=d, y=rng.normal() + d))
        df = pd.DataFrame(rows)
        with pytest.warns(UserWarning, match="HC2 variance is undefined"):
            res = LWDiD(rolling="demean", vcov_type="hc2").fit(df, **self.KW)
        assert np.isfinite(res.att)
        from tests.conftest import assert_nan_inference

        assert_nan_inference(
            {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
        )

    @pytest.mark.parametrize("vcov", ["classical", "hc1"])
    def test_exact_fit_headline_fails_closed(self, vcov):
        # y = t exactly: the collapsed regression fits exactly, so the SE
        # is roundoff of zero - pre-fix t ~ 1e16 was reported.
        rows = []
        for u in range(6):
            for t in range(1, 7):
                d = 1 if (u < 3 and t >= 4) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=float(t)))
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", vcov_type=vcov).fit(df, **self.KW)
        np.testing.assert_allclose(res.att, 0.0, atol=1e-12)
        from tests.conftest import assert_nan_inference

        assert_nan_inference(
            {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
        )

    def test_event_plot_uses_fitted_interval_endpoints(self):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from diff_diff.lwdid_visualization import plot_event_study

        rng = np.random.default_rng(0)
        rows = []
        for u in range(14):
            g = 4 if u < 7 else 0
            for t in range(1, 8):
                d = int(g > 0 and t >= g)
                rows.append(
                    dict(unit=u, time=t, treat=d, g=g, y=1 + 0.2 * t + d + rng.normal(0, 0.4))
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", alpha=0.10).fit(df, first_treat="g", **self.KW)
        fig = plot_event_study(res)
        ax = fig.axes[0]
        # the rendered whiskers must match the FITTED interval endpoints
        # (alpha=0.10 t-intervals), not +/-1.96*SE
        seg_ys = sorted(
            y
            for coll in ax.collections
            for seg in coll.get_segments()
            for y in (seg[0][1], seg[-1][1])
        )
        row = res.event_study_effects[max(res.event_study_effects)]
        lo, hi = row["conf_int"]
        assert any(abs(y - lo) < 1e-9 for y in seg_ys)
        assert any(abs(y - hi) < 1e-9 for y in seg_ys)
        naive = 1.96 * row["se"]
        fitted_half = row["effect"] - lo
        assert abs(naive - fitted_half) > 1e-6  # the two conventions differ here
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestReviewRound22Guards:
    """Local-review round 22: array-valued alpha passed construction and
    failed later with a raw TypeError."""

    @pytest.mark.parametrize(
        "bad", [np.array([0.05]), "0.05", None, complex(0.05), np.nan, np.inf, True]
    )
    def test_alpha_scalar_validation(self, bad):
        with pytest.raises((ValueError, TypeError)):
            LWDiD(alpha=bad)
        from diff_diff.lwdid_wild_bootstrap import wild_cluster_bootstrap

        y = np.random.default_rng(0).normal(size=20)
        d = np.array([1.0] * 10 + [0.0] * 10)
        cl = np.arange(20) % 5
        with pytest.raises((ValueError, TypeError)):
            wild_cluster_bootstrap(y, d, cl, alpha=bad, n_bootstrap=19)


class TestReviewRound23Guards:
    """Local-review round 23: raw cohort masses weighted non-contributing
    treated units into staggered overall aggregates outside the tau_omega
    route; plot_cohort_trends silently ignored cohort=; validator
    reported missing unit/time as warnings only."""

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_overall_weights_use_contributing_treated_units(self):
        # Two cohorts under NOT_YET_TREATED (non-tau_omega route): 3 of 4
        # cohort-5 treated units observe NO post rows -> only 1
        # contributes. Overall masses must be 4 (cohort 3) and 1
        # (cohort 5), not the raw 4 and 4.
        rng = np.random.default_rng(2)
        rows = []
        uid = 0
        spec = [(0, 8, None), (3, 4, None), (5, 1, None), (5, 3, (1, 2, 3, 4))]
        for g, n, keep in spec:
            for _ in range(n):
                alpha = rng.normal()
                for t in range(1, 7):
                    if keep is not None and t not in keep:
                        continue
                    d = int(g > 0 and t >= g)
                    rows.append(
                        dict(
                            unit=uid,
                            time=t,
                            treat=d,
                            g=g,
                            y=alpha + 0.2 * t + 1.5 * d + rng.normal(0, 0.3),
                        )
                    )
                uid += 1
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", control_group="not_yet_treated").fit(
                df, first_treat="g", **self.KW
            )
        assert res.cohort_effects[3]["n_treated"] == 4
        assert res.cohort_effects[5]["n_treated"] == 1  # contributing only
        att3 = res.cohort_effects[3]["att"]
        att5 = res.cohort_effects[5]["att"]
        expected = (4.0 * att3 + 1.0 * att5) / 5.0
        raw = (4.0 * att3 + 4.0 * att5) / 8.0
        np.testing.assert_allclose(res.att, expected, rtol=1e-12)
        assert abs(res.att - raw) > 1e-9

    def test_plot_cohort_trends_honors_cohort(self):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from diff_diff.lwdid_visualization import plot_cohort_trends

        rng = np.random.default_rng(0)
        rows = []
        for u in range(12):
            g = 3 if u < 4 else (5 if u < 8 else 0)
            for t in range(1, 7):
                d = int(g > 0 and t >= g)
                rows.append(dict(unit=u, time=t, treat=d, g=g, y=rng.normal() + d))
        df = pd.DataFrame(rows)
        fig = plot_cohort_trends(
            df, outcome="y", unit="unit", time="time", treatment="treat", cohort="g"
        )
        labels = [line.get_label() for ax in fig.axes for line in ax.get_lines()]
        assert any("Cohort 3" in lab for lab in labels)
        assert any("Cohort 5" in lab for lab in labels)
        assert any(lab == "Control" for lab in labels)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_validator_missing_unit_time_invalid(self):
        from diff_diff.lwdid import validate_staggered_data

        rows = []
        for u in range(6):
            g = 4 if u < 3 else 0
            for t in range(1, 7):
                rows.append(dict(unit=u, time=t, g=g, y=1.0))
        df = pd.DataFrame(rows)
        df.loc[df.index[3], "unit"] = np.nan
        out = validate_staggered_data(df, unit="unit", time="time", cohort="g")
        assert out["valid"] is False
        assert any("missing values" in e for e in out["errors"])


class TestReviewRound24Guards:
    """Local-review round 24 P2s: unusable small n_reps rejected up
    front; HC3 fail-closed keeps the length-k DOF contract; datetime
    cohort relabeling is canonical (collision/row-order independent)."""

    def test_ri_small_n_reps_rejected_up_front(self):
        from diff_diff.lwdid_randomization import randomization_inference

        y = np.random.default_rng(0).normal(size=20)
        d = np.array([1.0] * 10 + [0.0] * 10)
        with pytest.raises(ValueError, match="integer >= 10"):
            randomization_inference(y, d, n_reps=9)
        res = randomization_inference(y, d, n_reps=10, seed=0)
        assert 0 < res.pvalue <= 1

    def test_hc3_fail_closed_dof_vector(self):
        from diff_diff.linalg import compute_robust_vcov

        X = np.column_stack([np.ones(4), np.array([0.0, 0.0, 0.0, 1.0])])
        y = np.array([1.0, 1.1, 0.9, 5.0])
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        with pytest.warns(UserWarning, match="HC3 variance is undefined"):
            vcov, dof = compute_robust_vcov(X, resid, vcov_type="hc3", return_dof=True)
        assert np.all(np.isnan(vcov))
        assert dof.shape == (2,) and np.all(np.isnan(dof))

    def test_datetime_cohort_relabel_canonical(self):
        # Two raw between-period cohort dates map to the SAME observed
        # onset; the reported cohort key must be the canonical observed
        # period regardless of row order.
        rng = np.random.default_rng(0)
        times = pd.date_range("2020-01-01", periods=6, freq="QS")
        onset = times[4]
        raw_a = onset - pd.Timedelta(days=10)
        raw_b = onset - pd.Timedelta(days=20)

        def build(order):
            rows = []
            for idx, u in enumerate(order):
                g = {0: raw_a, 1: raw_b}.get(u, pd.NaT) if u < 2 else pd.NaT
                for i, ts in enumerate(times):
                    d = int(u < 2 and ts >= onset)
                    rows.append(dict(unit=u, time=ts, treat=d, g=g, y=rng.normal() + d))
            return pd.DataFrame(rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = LWDiD(rolling="demean").fit(
                build([0, 1, 2, 3, 4, 5]),
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="g",
            )
            r2 = LWDiD(rolling="demean").fit(
                build([1, 0, 2, 3, 4, 5]),
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                first_treat="g",
            )
        assert list(r1.cohort_effects) == [onset]
        assert list(r2.cohort_effects) == [onset]


class TestReviewRoundCI7Guards:
    """CI review round 7: pin the DOCUMENTED within-cohort cell-mass
    convention on an unbalanced panel where it provably differs from the
    LW 2026 eq. 7.10 unit-average estimand (a treated unit observing
    more post periods carries more cell-mass weight)."""

    KW = dict(outcome="y", unit="unit", time="time", treatment="treat")

    def test_cohort_effect_cell_mass_oracle_unbalanced(self):
        # One cohort (g=4), 2 treated units: unit 0 observes post {4,5,6},
        # unit 1 observes post {4} only. 6 never-treated controls observe
        # everything. Deterministic outcomes.
        rows = []
        for u in range(8):
            g = 4 if u < 2 else 0
            times = range(1, 7)
            for t in times:
                if u == 1 and t > 4:
                    continue  # unit 1 misses post periods 5, 6
                d = int(g > 0 and t >= g)
                # unit-specific level + zero noise; treated add u-dependent effect
                y = 10.0 * u + 0.0 * t + (2.0 + 3.0 * u) * d
                rows.append(dict(unit=u, time=t, treat=d, g=g, y=y))
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(rolling="demean", control_group="never_treated", vcov_type="hc1").fit(
                df, first_treat="g", **self.KW
            )
        # Independent oracle. Demean: pre-mean = level (zero trend), so
        # ydot = effect for treated rows, 0 for controls.
        # Cell ATTs: t=4 has units {0,1} -> mean(2, 5) = 3.5, n_treated=2;
        # t=5, t=6 have unit 0 only -> 2.0, n_treated=1.
        # CELL-MASS cohort effect = (2*3.5 + 1*2 + 1*2) / 4 = 2.75.
        # eq. 7.10 UNIT-AVERAGE estimand: unit post-averages are 2.0
        # (unit 0) and 5.0 (unit 1) -> cohort effect 3.5. The documented
        # convention is cell-mass.
        cell_mass = res.cohort_effects[4]["att"]
        np.testing.assert_allclose(cell_mass, 2.75, atol=1e-10)
        assert abs(cell_mass - 3.5) > 0.5  # distinguishes eq. 7.10
        # .att on this NT/reg route is the tau_omega COMPOSITE (7.18),
        # built from unit post-averages - here exactly the eq. 7.10
        # unit-average value (3.5). The two surfaces answer different,
        # separately documented estimands on unbalanced panels.
        np.testing.assert_allclose(res.att, 3.5, atol=1e-10)
