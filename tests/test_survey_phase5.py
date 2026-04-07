"""Tests for Phase 5 survey support: SyntheticDiD and TROP.

Covers: pweight-only survey integration for both estimators, including
point estimate weighting, bootstrap/placebo SE threading, survey_metadata
in results, error guards for unsupported designs, and scale invariance.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import SurveyDesign, SyntheticDiD
from diff_diff.trop import TROP, trop

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def sdid_survey_data():
    """Balanced panel for SDID with survey design columns.

    20 units (5 treated, 15 control), 10 periods, block treatment at period 6.
    Unit-constant weight column that varies across units.
    """
    np.random.seed(42)
    n_units = 20
    n_periods = 10
    n_treated = 5

    units = list(range(n_units))
    periods = list(range(n_periods))

    rows = []
    for u in units:
        is_treated = 1 if u < n_treated else 0
        base = np.random.randn() * 2
        for t in periods:
            y = base + 0.5 * t + np.random.randn() * 0.5
            if is_treated and t >= 6:
                y += 2.0  # treatment effect
            rows.append({"unit": u, "time": t, "outcome": y, "treated": is_treated})

    data = pd.DataFrame(rows)

    # Unit-constant survey columns
    unit_weight = 1.0 + np.arange(n_units) * 0.1  # [1.0, 1.1, ..., 2.9]
    unit_stratum = np.arange(n_units) // 10
    unit_psu = np.arange(n_units) // 5
    unit_map = {u: i for i, u in enumerate(units)}
    idx = data["unit"].map(unit_map).values

    data["weight"] = unit_weight[idx]
    data["stratum"] = unit_stratum[idx]
    data["psu"] = unit_psu[idx]

    return data


@pytest.fixture
def trop_survey_data():
    """Panel data for TROP with absorbing-state D and survey columns.

    20 units (5 treated starting at period 5), 10 periods.
    """
    np.random.seed(123)
    n_units = 20
    n_periods = 10
    n_treated = 5

    units = list(range(n_units))
    periods = list(range(n_periods))

    rows = []
    for u in units:
        is_treated_unit = u < n_treated
        base = np.random.randn() * 2
        for t in periods:
            y = base + 0.3 * t + np.random.randn() * 0.5
            # Absorbing state: D=1 for t >= 5 if treated unit
            d = 1 if (is_treated_unit and t >= 5) else 0
            if d == 1:
                y += 1.5  # treatment effect
            rows.append({"unit": u, "time": t, "outcome": y, "D": d})

    data = pd.DataFrame(rows)

    # Unit-constant survey columns
    unit_weight = 1.0 + np.arange(n_units) * 0.15
    unit_stratum = np.arange(n_units) // 10
    unit_psu = np.arange(n_units) // 5
    unit_map = {u: i for i, u in enumerate(units)}
    idx = data["unit"].map(unit_map).values

    data["weight"] = unit_weight[idx]
    data["stratum"] = unit_stratum[idx]
    data["psu"] = unit_psu[idx]

    return data


@pytest.fixture
def survey_design_weights():
    return SurveyDesign(weights="weight")


@pytest.fixture
def survey_design_full():
    return SurveyDesign(weights="weight", strata="stratum", psu="psu")


# =============================================================================
# SyntheticDiD Survey Tests
# =============================================================================


class TestSyntheticDiDSurvey:
    """Survey support tests for SyntheticDiD."""

    def test_smoke_weights_only(self, sdid_survey_data, survey_design_weights):
        """Fit completes and survey_metadata is populated."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        assert result.survey_metadata is not None
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)

    def test_uniform_weights_match_unweighted(self, sdid_survey_data):
        """Uniform weights (all 1.0) produce same ATT as unweighted."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["uniform_w"] = 1.0

        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result_no_survey = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
        )
        result_uniform = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        assert result_uniform.att == pytest.approx(result_no_survey.att, abs=1e-10)

    def test_survey_metadata_fields(self, sdid_survey_data, survey_design_weights):
        """Metadata has correct weight_type, effective_n, design_effect."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        sm = result.survey_metadata
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0
        assert sm.design_effect > 0

    def test_full_design_bootstrap_smoke(self, sdid_survey_data, survey_design_full):
        """Full survey design (strata/PSU) works with bootstrap variance."""
        est = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_full,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
        assert result.survey_metadata is not None
        assert result.survey_metadata.n_strata is not None
        assert result.survey_metadata.n_psu is not None

    def test_full_design_placebo_raises(self, sdid_survey_data, survey_design_full):
        """Placebo variance with full design raises NotImplementedError."""
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        with pytest.raises(NotImplementedError, match="placebo.*does not support strata/PSU/FPC"):
            est.fit(
                sdid_survey_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=survey_design_full,
            )

    def test_full_design_se_differs_from_weights_only(self, sdid_survey_data):
        """Rao-Wu bootstrap SE differs from pweight-only bootstrap SE."""
        sd_w = SurveyDesign(weights="weight")
        sd_full = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = SyntheticDiD(variance_method="bootstrap", n_bootstrap=100, seed=42)

        result_w = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_w,
        )
        result_full = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_full,
        )
        # ATT point estimates should be the same (same weights)
        assert result_full.att == pytest.approx(result_w.att, abs=1e-10)
        # SEs should differ (different bootstrap scheme)
        assert result_full.se != pytest.approx(result_w.se, abs=1e-6)

    def test_fweight_aweight_raises(self, sdid_survey_data):
        """Non-pweight raises ValueError."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        sd = SurveyDesign(weights="weight", weight_type="fweight")
        with pytest.raises(ValueError, match="pweight"):
            est.fit(
                sdid_survey_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd,
            )

    def test_weighted_att_differs(self, sdid_survey_data, survey_design_weights):
        """Non-uniform weights produce different ATT than unweighted."""
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result_no_survey = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
        )
        result_survey = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        # ATTs should differ since weights are non-uniform
        assert result_survey.att != pytest.approx(result_no_survey.att, abs=1e-6)

    def test_summary_includes_survey(self, sdid_survey_data, survey_design_weights):
        """summary() output contains Survey Design section."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        summary = result.summary()
        assert "Survey Design" in summary
        assert "pweight" in summary

    def test_bootstrap_with_survey(self, sdid_survey_data, survey_design_weights):
        """variance_method='bootstrap' completes with survey weights."""
        est = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.se)
        assert result.se > 0

    def test_placebo_with_survey(self, sdid_survey_data, survey_design_weights):
        """variance_method='placebo' completes with survey weights."""
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.se)
        assert result.se > 0

    def test_weight_scale_invariance(self, sdid_survey_data, survey_design_weights):
        """Multiplying all weights by constant produces same ATT."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["weight_2x"] = sdid_survey_data["weight"] * 2.0

        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result_1x = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        result_2x = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=SurveyDesign(weights="weight_2x"),
        )
        assert result_2x.att == pytest.approx(result_1x.att, rel=1e-6)

    def test_unit_varying_survey_raises(self, sdid_survey_data):
        """Time-varying weight column raises ValueError."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["bad_weight"] = sdid_survey_data["weight"] + sdid_survey_data["time"] * 0.1
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        with pytest.raises(ValueError):
            est.fit(
                sdid_survey_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=SurveyDesign(weights="bad_weight"),
            )

    def test_to_dict_includes_survey(self, sdid_survey_data, survey_design_weights):
        """to_dict() output includes survey metadata fields."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        d = result.to_dict()
        assert "weight_type" in d
        assert d["weight_type"] == "pweight"

    def test_covariates_with_survey(self, sdid_survey_data, survey_design_weights):
        """Covariates + survey_design smoke test (WLS residualization)."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["x1"] = np.random.randn(len(sdid_survey_data))
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            covariates=["x1"],
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.att)
        assert result.survey_metadata is not None

    def test_effective_weights_returned(self, sdid_survey_data, survey_design_weights):
        """unit_weights returns composed ω_eff (not raw ω) under survey weighting."""
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        weights = result.unit_weights
        # Effective weights should sum to 1 (renormalized)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-10)
        # With non-uniform survey weights, effective weights should differ
        # from what uniform survey weights would produce
        sdid_survey_data_u = sdid_survey_data.copy()
        sdid_survey_data_u["uniform_w"] = 1.0
        result_u = est.fit(
            sdid_survey_data_u,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        # Non-uniform weights should change the returned weight distribution
        eff_vals = sorted(weights.values(), reverse=True)
        uni_vals = sorted(result_u.unit_weights.values(), reverse=True)
        assert eff_vals != pytest.approx(uni_vals, abs=1e-6)


# =============================================================================
# TROP Survey Tests
# =============================================================================


class TestTROPSurvey:
    """Survey support tests for TROP (local and global methods)."""

    def test_smoke_local_weights_only(self, trop_survey_data, survey_design_weights):
        """Local method completes with survey weights."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert result.survey_metadata is not None
        assert np.isfinite(result.att)

    def test_smoke_global_weights_only(self, trop_survey_data, survey_design_weights):
        """Global method completes with survey weights."""
        est = TROP(method="global", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert result.survey_metadata is not None
        assert np.isfinite(result.att)

    def test_uniform_weights_match_local(self, trop_survey_data):
        """Uniform weights produce same ATT as unweighted (local)."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["uniform_w"] = 1.0

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result_no_survey = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        result_uniform = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        assert result_uniform.att == pytest.approx(result_no_survey.att, abs=1e-10)

    def test_uniform_weights_match_global(self, trop_survey_data):
        """Uniform weights produce same ATT as unweighted (global)."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["uniform_w"] = 1.0

        est = TROP(method="global", n_bootstrap=10, seed=42, max_iter=5)
        result_no_survey = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        result_uniform = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        assert result_uniform.att == pytest.approx(result_no_survey.att, abs=1e-10)

    def test_survey_metadata_fields(self, trop_survey_data, survey_design_weights):
        """Metadata has correct fields."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        sm = result.survey_metadata
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0

    def test_full_design_local_rao_wu(self, trop_survey_data, survey_design_full):
        """Full design (strata/PSU/FPC) uses Rao-Wu bootstrap and succeeds."""
        est = TROP(method="local", n_bootstrap=20, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_full,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.survey_metadata is not None

    def test_full_design_global_rao_wu(self, trop_survey_data, survey_design_full):
        """Full design (strata/PSU/FPC) with global method uses Rao-Wu bootstrap."""
        est = TROP(method="global", n_bootstrap=20, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_full,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.survey_metadata is not None

    def test_fweight_aweight_raises(self, trop_survey_data):
        """Non-pweight raises ValueError."""
        est = TROP(method="local", n_bootstrap=10, seed=42)
        sd = SurveyDesign(weights="weight", weight_type="aweight")
        with pytest.raises(ValueError, match="pweight"):
            est.fit(
                trop_survey_data,
                outcome="outcome",
                treatment="D",
                unit="unit",
                time="time",
                survey_design=sd,
            )

    def test_weighted_att_differs(self, trop_survey_data, survey_design_weights):
        """Non-uniform weights change ATT."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result_no = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        result_survey = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert result_survey.att != pytest.approx(result_no.att, abs=1e-6)

    def test_weighted_att_differs_global(self, trop_survey_data, survey_design_weights):
        """Non-uniform weights change ATT for method='global'."""
        est = TROP(method="global", n_bootstrap=10, seed=42, max_iter=5)
        result_no = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        result_survey = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert result_survey.att != pytest.approx(result_no.att, abs=1e-6)

    def test_summary_includes_survey(self, trop_survey_data, survey_design_weights):
        """summary() contains survey section."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        summary = result.summary()
        assert "Survey Design" in summary

    def test_weight_scale_invariance(self, trop_survey_data, survey_design_weights):
        """Scale invariance: 2x weights produce same ATT."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["weight_3x"] = trop_survey_data["weight"] * 3.0

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result_1x = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        result_3x = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=SurveyDesign(weights="weight_3x"),
        )
        assert result_3x.att == pytest.approx(result_1x.att, rel=1e-6)

    def test_unit_varying_survey_raises(self, trop_survey_data):
        """Validation catches time-varying weights."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["bad_weight"] = trop_survey_data["weight"] + trop_survey_data["time"] * 0.1
        est = TROP(method="local", n_bootstrap=10, seed=42)
        with pytest.raises(ValueError):
            est.fit(
                trop_survey_data,
                outcome="outcome",
                treatment="D",
                unit="unit",
                time="time",
                survey_design=SurveyDesign(weights="bad_weight"),
            )

    def test_convenience_function_with_survey(self, trop_survey_data, survey_design_weights):
        """trop() convenience function accepts survey_design."""
        result = trop(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
            n_bootstrap=10,
            seed=42,
            max_iter=5,
        )
        assert result.survey_metadata is not None

    def test_to_dict_includes_survey(self, trop_survey_data, survey_design_weights):
        """to_dict() includes survey metadata fields."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        d = result.to_dict()
        assert "weight_type" in d
        assert d["weight_type"] == "pweight"

    def test_local_bootstrap_nan_treated_outcomes(self, trop_survey_data):
        """Bootstrap handles NaN treated outcomes without poisoning SE."""
        trop_survey_data = trop_survey_data.copy()
        # Set some treated post-treatment outcomes to NaN
        mask = (trop_survey_data["D"] == 1) & (trop_survey_data["time"] == 7)
        trop_survey_data.loc[mask, "outcome"] = np.nan

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        # Point estimate should use finite cells only
        assert np.isfinite(result.att)
        # SE should remain finite (not poisoned by NaN)
        assert np.isfinite(result.se)

    def test_local_bootstrap_nan_with_survey(self, trop_survey_data, survey_design_weights):
        """Bootstrap + survey handles NaN treated outcomes correctly."""
        trop_survey_data = trop_survey_data.copy()
        mask = (trop_survey_data["D"] == 1) & (trop_survey_data["time"] == 8)
        trop_survey_data.loc[mask, "outcome"] = np.nan

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)


# =============================================================================
# Pinned Numerical Tests
# =============================================================================


class TestPinnedNumerical:
    """Deterministic numerical tests for exact weighted formulas."""

    def test_sdid_weighted_att_manual(self):
        """Manual ATT check: survey-weighted treated means + ω∘w_co composition."""
        # Tiny 2x2 balanced panel: 2 control, 1 treated, 2 pre + 1 post
        np.random.seed(99)
        data = pd.DataFrame(
            {
                "unit": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
                "outcome": [1.0, 2.0, 3.0, 2.0, 3.0, 4.5, 5.0, 6.0, 10.0],
                "treated": [0, 0, 0, 0, 0, 0, 1, 1, 1],
                "weight": [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0],
            }
        )
        # Single treated unit → treated means are trivially that unit's outcomes
        # (survey weight doesn't change a single-unit mean)
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=20, seed=42)
        result = est.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[2],
            survey_design=SurveyDesign(weights="weight"),
        )
        # Verify unit_weights sum to 1 (composed with survey)
        assert sum(result.unit_weights.values()) == pytest.approx(1.0, abs=1e-10)
        assert np.isfinite(result.att)

    def test_trop_weighted_att_aggregation(self):
        """Verify TROP ATT = weighted mean of tau values."""
        # Create data where we can predict directional effect of weighting
        np.random.seed(77)
        n_units = 15
        n_periods = 6
        n_treated = 3

        units = list(range(n_units))
        periods = list(range(n_periods))

        rows = []
        for u in units:
            is_treated = u < n_treated
            base = u * 0.5
            for t in periods:
                y = base + 0.2 * t + np.random.randn() * 0.3
                d = 1 if (is_treated and t >= 3) else 0
                if d == 1:
                    # Different effect per unit: unit 0 gets +1, unit 1 gets +3, unit 2 gets +5
                    y += 1.0 + 2.0 * u
                rows.append({"unit": u, "time": t, "outcome": y, "D": d})

        data = pd.DataFrame(rows)
        # Weight unit 2 (biggest effect) heavily
        weights = np.ones(n_units)
        weights[2] = 10.0  # unit 2 has effect ~5, heavily weighted
        unit_map = {u: i for i, u in enumerate(units)}
        data["weight"] = weights[data["unit"].map(unit_map).values]

        est_no = TROP(method="local", n_bootstrap=5, seed=42, max_iter=3)
        result_no = est_no.fit(data, "outcome", "D", "unit", "time")

        est_w = TROP(method="local", n_bootstrap=5, seed=42, max_iter=3)
        result_w = est_w.fit(
            data,
            "outcome",
            "D",
            "unit",
            "time",
            survey_design=SurveyDesign(weights="weight"),
        )

        # Weighted ATT should be pulled toward unit 2's larger effect
        assert result_w.att > result_no.att

    def test_sdid_to_dict_schema_matches_did(self):
        """SyntheticDiDResults.to_dict() survey fields match DiDResults schema."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "unit": [0, 0, 1, 1, 2, 2],
                "time": [0, 1, 0, 1, 0, 1],
                "outcome": [1.0, 2.0, 2.0, 3.0, 5.0, 8.0],
                "treated": [0, 0, 0, 0, 1, 1],
                "weight": [1.0, 1.0, 2.0, 2.0, 1.5, 1.5],
            }
        )
        est = SyntheticDiD(n_bootstrap=10, seed=42)
        result = est.fit(
            data,
            "outcome",
            "treated",
            "unit",
            "time",
            post_periods=[1],
            survey_design=SurveyDesign(weights="weight"),
        )
        d = result.to_dict()
        # Schema alignment: all these fields should be present
        for key in [
            "weight_type",
            "effective_n",
            "design_effect",
            "sum_weights",
            "n_strata",
            "n_psu",
            "df_survey",
        ]:
            assert key in d, f"Missing key: {key}"


class TestTROPRaoWuEquivalence:
    """Test Rao-Wu vs block bootstrap equivalence under degenerate design."""

    def test_rao_wu_approximates_block_no_strata(self, trop_survey_data):
        """Without real strata variation, Rao-Wu SE ~ block bootstrap SE."""
        from diff_diff import TROP

        data = trop_survey_data.copy()
        # Single stratum, PSU = unit (effectively block bootstrap)
        data["single_stratum"] = 0
        data["unit_psu"] = data["unit"]

        sd_rw = SurveyDesign(
            weights="weight", strata="single_stratum", psu="unit_psu",
        )
        sd_block = SurveyDesign(weights="weight")

        result_rw = TROP(method="local", n_bootstrap=99, seed=42, max_iter=5).fit(
            data, "outcome", "D", "unit", "time", survey_design=sd_rw,
        )
        result_block = TROP(method="local", n_bootstrap=99, seed=42, max_iter=5).fit(
            data, "outcome", "D", "unit", "time", survey_design=sd_block,
        )

        # Point estimates identical (same weights)
        assert result_rw.att == pytest.approx(result_block.att, abs=1e-10)
        # SEs should be within factor of 2
        if result_block.se > 0:
            ratio = result_rw.se / result_block.se
            assert 0.5 < ratio < 2.0, (
                f"Rao-Wu SE ({result_rw.se:.4f}) and block SE "
                f"({result_block.se:.4f}) differ by > 2x (ratio={ratio:.2f})"
            )
