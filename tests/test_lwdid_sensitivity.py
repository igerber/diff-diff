"""Tests for lwdid_sensitivity module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.lwdid_sensitivity import (
    _classify_robustness,
    _compute_sensitivity_ratio,
    robustness_pre_periods,
    sensitivity_no_anticipation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_data():
    rng = np.random.default_rng(42)
    records = []
    for i in range(80):
        d = int(i < 25)
        for t in range(1, 9):
            y = 1.0 + 0.1 * t + rng.normal(0, 0.3)
            if d and t > 4:
                y += 2.0
            records.append({"unit": i, "time": t, "y": y, "treat": d * int(t > 4)})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# SensitivityResult fields
# ---------------------------------------------------------------------------


class TestSensitivityResultFields:
    """Test SensitivityResult dataclass has all expected fields."""

    def test_result_fields_present(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        assert hasattr(r, "specifications")
        assert hasattr(r, "baseline_att")
        assert hasattr(r, "baseline_se")
        assert hasattr(r, "sensitivity_ratio")
        assert hasattr(r, "robustness_level")
        assert hasattr(r, "n_specifications")

    def test_result_types(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        assert isinstance(r.specifications, list)
        assert isinstance(r.baseline_att, float)
        assert isinstance(r.baseline_se, float)
        assert isinstance(r.sensitivity_ratio, float)
        assert isinstance(r.robustness_level, str)
        assert isinstance(r.n_specifications, int)


# ---------------------------------------------------------------------------
# Robustness level valid
# ---------------------------------------------------------------------------


class TestRobustnessLevel:
    """Test robustness_level is a valid classification."""

    VALID_LEVELS = {"highly_robust", "moderately_robust", "sensitive", "highly_sensitive"}

    def test_robustness_level_valid(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        assert r.robustness_level in self.VALID_LEVELS

    def test_classify_robustness_helper(self):
        assert _classify_robustness(0.05) == "highly_robust"
        assert _classify_robustness(0.15) == "moderately_robust"
        assert _classify_robustness(0.35) == "sensitive"
        assert _classify_robustness(0.60) == "highly_sensitive"


# ---------------------------------------------------------------------------
# Sensitivity ratio non-negative
# ---------------------------------------------------------------------------


class TestSensitivityRatio:
    """Test sensitivity_ratio is non-negative."""

    def test_ratio_non_negative(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        assert r.sensitivity_ratio >= 0.0

    def test_compute_sensitivity_ratio_helper(self):
        assert _compute_sensitivity_ratio(2.0, [2.0, 2.1, 1.9]) == pytest.approx(0.1)
        # Single finite estimate: robustness cannot be assessed
        assert np.isnan(_compute_sensitivity_ratio(2.0, [2.0]))
        # Near-zero baseline: the ratio is undefined -> not estimable (NaN)
        assert np.isnan(_compute_sensitivity_ratio(1e-15, [1e-15, 0.5]))


# ---------------------------------------------------------------------------
# Specifications list populated
# ---------------------------------------------------------------------------


class TestSpecifications:
    """Test specifications list is populated."""

    def test_specs_populated(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        # Should have at least 1 specification
        assert len(r.specifications) >= 1
        assert r.n_specifications >= 2  # baseline + at least 1 alternative

    def test_spec_has_expected_attributes(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        if r.specifications:
            spec = r.specifications[0]
            assert hasattr(spec, "label")
            assert hasattr(spec, "rolling")
            assert hasattr(spec, "estimation_method")
            assert hasattr(spec, "att")
            assert hasattr(spec, "se")
            assert hasattr(spec, "pvalue")


# ---------------------------------------------------------------------------
# to_dataframe()
# ---------------------------------------------------------------------------


class TestToDataframe:
    """Test to_dataframe() returns a DataFrame."""

    def test_to_dataframe_returns_df(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        df = r.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        assert "att" in df.columns
        assert "label" in df.columns

    def test_summary_returns_string(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        s = r.summary()
        assert isinstance(s, str)
        assert "Sensitivity" in s


# ---------------------------------------------------------------------------
# not_estimable classification and failure reporting
# ---------------------------------------------------------------------------


class TestNotEstimable:
    """Failed fits must be reported as 'not_estimable', never as robust."""

    def test_nan_baseline_ratio_is_nan(self):
        assert np.isnan(_compute_sensitivity_ratio(np.nan, [np.nan, 1.0, 2.0]))

    def test_classify_nan_ratio_not_estimable(self):
        assert _classify_robustness(float("nan")) == "not_estimable"

    @staticmethod
    def _unestimable_but_valid_panel():
        # Data fit() ACCEPTS but cannot estimate: rolling='detrendq' with
        # 4 pre-periods covering all 4 seasons -> every unit is seasonal-
        # unidentified (warn + NaN ATT on every spec).
        rng = np.random.default_rng(3)
        records = []
        for i in range(20):
            d = int(i < 8)
            for t in range(1, 9):
                records.append({"unit": i, "time": t, "y": rng.normal(), "treat": d * int(t >= 5)})
        return pd.DataFrame(records)

    def test_all_specs_fail_reports_not_estimable(self):
        """Every spec unestimable (fit accepts the data but NaNs) -> the
        result must be 'not_estimable' with a NaN ratio, not
        'highly_robust'."""
        df = self._unestimable_but_valid_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = robustness_pre_periods(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                rolling="detrendq",
            )
        assert r.robustness_level == "not_estimable"
        assert np.isnan(r.sensitivity_ratio)

    def test_data_fit_would_reject_raises_not_silently_swallowed(self):
        # Fix-wave WS10 (campaign finding): genuine specification errors
        # were swallowed as per-spec 'failed fits'. Data that LWDiD.fit()
        # itself rejects (all-NaN outcome) must RAISE from the sensitivity
        # helpers too.
        records = []
        for i in range(20):
            d = int(i < 8)
            for t in range(1, 7):
                records.append({"unit": i, "time": t, "y": np.nan, "treat": d * int(t > 3)})
        df = pd.DataFrame(records)
        with pytest.raises(ValueError):
            robustness_pre_periods(df, outcome="y", unit="unit", time="time", treatment="treat")
        with pytest.raises(ValueError):
            sensitivity_no_anticipation(
                df, outcome="y", unit="unit", time="time", treatment="treat"
            )

    def test_all_specs_fail_no_anticipation_not_estimable(self):
        df = self._unestimable_but_valid_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sensitivity_no_anticipation(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                rolling="detrendq",
            )
        assert r.robustness_level == "not_estimable"
        assert np.isnan(r.sensitivity_ratio)

    def test_missing_outcome_column_raises(self, panel_data):
        with pytest.raises(ValueError, match="not found in data"):
            robustness_pre_periods(
                panel_data,
                outcome="no_such_column",
                unit="unit",
                time="time",
                treatment="treat",
            )

    def test_missing_control_column_raises(self, panel_data):
        with pytest.raises(ValueError, match="not found in data"):
            robustness_pre_periods(
                panel_data,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                controls=["no_such_control"],
            )


class TestMultiCohortRejection:
    """Round-2 finding: pre-period windows are earliest-adoption-relative,
    so multi-cohort staggered inputs would mislabel later cohorts'
    transformation samples. They fail closed; a single treated cohort is
    exactly the global rule and stays supported."""

    @staticmethod
    def _staggered(n_cohorts=2):
        rng = np.random.default_rng(0)
        rows = []
        onsets = [5, 7][:n_cohorts]
        for u in range(16):
            if u < 4 * n_cohorts:
                g = onsets[u % n_cohorts]
            else:
                g = 0
            for t in range(1, 10):
                d = int(g > 0 and t >= g)
                rows.append(dict(unit=u, time=t, treat=d, g=g, y=rng.normal() + d))
        return pd.DataFrame(rows)

    def test_multi_cohort_rejected_both_functions(self):
        df = self._staggered(n_cohorts=2)
        with pytest.raises(ValueError, match="single treated\\s+cohort"):
            robustness_pre_periods(
                df, outcome="y", unit="unit", time="time", treatment="treat", cohort="g"
            )
        with pytest.raises(ValueError, match="single treated\\s+cohort"):
            sensitivity_no_anticipation(
                df, outcome="y", unit="unit", time="time", treatment="treat", cohort="g"
            )

    def test_single_cohort_still_supported(self):
        df = self._staggered(n_cohorts=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = robustness_pre_periods(
                df, outcome="y", unit="unit", time="time", treatment="treat", cohort="g"
            )
        assert np.isfinite(res.baseline_att)


class TestParameterValidation:
    """Round-3: strict validation of exclusion/window parameters
    (exclude_periods=0 previously sliced pre_periods[:-0] == EMPTY,
    silently dropping every pre-period)."""

    @staticmethod
    def _panel():
        rng = np.random.default_rng(0)
        rows = []
        for u in range(12):
            for t in range(1, 9):
                d = 1 if (u < 6 and t >= 6) else 0
                rows.append(dict(unit=u, time=t, treat=d, y=rng.normal() + d))
        return pd.DataFrame(rows)

    def test_exclude_periods_zero_rejected(self):
        with pytest.raises(ValueError, match="positive integers"):
            sensitivity_no_anticipation(
                self._panel(),
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                exclude_periods=[0],
            )

    def test_exclude_periods_duplicates_and_types_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            sensitivity_no_anticipation(
                self._panel(),
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                exclude_periods=[1, 1],
            )
        with pytest.raises(ValueError, match="positive integers"):
            sensitivity_no_anticipation(
                self._panel(),
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                exclude_periods=[True],
            )

    def test_k_bounds_validated(self):
        with pytest.raises(ValueError, match="k_min must be"):
            robustness_pre_periods(
                self._panel(),
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                k_min=2.5,
            )
