"""Tests for difference-in-differences estimators."""

import warnings

import numpy as np
import pandas as pd
import pytest
from event_study_average import post_period_average

from diff_diff import (
    DiDResults,
    DifferenceInDifferences,
    SyntheticDiD,
    SyntheticDiDResults,
    TwoWayFixedEffects,
)


@pytest.fixture
def simple_did_data():
    """Create simple 2x2 DiD data with known ATT."""
    np.random.seed(42)

    # Create balanced panel: 100 units, 2 periods
    n_units = 100
    n_treated = 50

    data = []
    for unit in range(n_units):
        is_treated = unit < n_treated

        for period in [0, 1]:
            # Base outcome
            y = 10.0

            # Unit effect
            y += unit * 0.1

            # Time effect (period 1 is higher for everyone)
            if period == 1:
                y += 5.0

            # Treatment effect (only for treated in post period)
            if is_treated and period == 1:
                y += 3.0  # True ATT = 3.0

            # Add noise
            y += np.random.normal(0, 1)

            data.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": period,
                    "outcome": y,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def simple_2x2_data():
    """Minimal 2x2 DiD data."""
    return pd.DataFrame(
        {
            "outcome": [10, 11, 15, 18, 9, 10, 12, 13],
            "treated": [1, 1, 1, 1, 0, 0, 0, 0],
            "post": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )


class TestDifferenceInDifferences:
    """Tests for DifferenceInDifferences estimator."""

    def test_basic_fit(self, simple_2x2_data):
        """Test basic model fitting."""
        did = DifferenceInDifferences()
        results = did.fit(simple_2x2_data, outcome="outcome", treatment="treated", post="post")

        assert isinstance(results, DiDResults)
        assert did.is_fitted_
        assert results.n_obs == 8
        assert results.n_treated == 4
        assert results.n_control == 4

    def test_att_direction(self, simple_did_data):
        """Test that ATT is estimated in correct direction."""
        did = DifferenceInDifferences()
        results = did.fit(simple_did_data, outcome="outcome", treatment="treated", post="post")

        # True ATT is 3.0, estimate should be close
        assert results.att > 0
        assert abs(results.att - 3.0) < 1.0  # Within 1 unit

    def test_formula_interface(self, simple_2x2_data):
        """Test formula-based fitting."""
        did = DifferenceInDifferences()
        results = did.fit(simple_2x2_data, formula="outcome ~ treated * post")

        assert isinstance(results, DiDResults)
        assert did.is_fitted_

    def test_formula_with_explicit_interaction(self, simple_2x2_data):
        """Test formula with explicit interaction syntax."""
        did = DifferenceInDifferences()
        results = did.fit(simple_2x2_data, formula="outcome ~ treated + post + treated:post")

        assert isinstance(results, DiDResults)

    def test_robust_vs_classical_se(self, simple_did_data):
        """Test that robust and classical SEs differ."""
        did_robust = DifferenceInDifferences()
        did_classical = DifferenceInDifferences(vcov_type="classical")

        results_robust = did_robust.fit(
            simple_did_data, outcome="outcome", treatment="treated", post="post"
        )
        results_classical = did_classical.fit(
            simple_did_data, outcome="outcome", treatment="treated", post="post"
        )

        # The vcov matrices should differ (HC1 vs classical)
        # Note: For balanced designs with homoskedastic errors, the ATT SE
        # may coincidentally be equal, but other coefficients will differ
        assert not np.allclose(results_robust.vcov, results_classical.vcov)
        # But ATT should be the same
        assert results_robust.att == results_classical.att

    def test_confidence_interval(self, simple_did_data):
        """Test confidence interval properties."""
        did = DifferenceInDifferences(alpha=0.05)
        results = did.fit(simple_did_data, outcome="outcome", treatment="treated", post="post")

        lower, upper = results.conf_int
        assert lower < results.att < upper
        assert lower < upper

    def test_get_set_params(self):
        """Test sklearn-compatible get_params and set_params."""
        did = DifferenceInDifferences(alpha=0.05)

        params = did.get_params()
        # get_params returns the RAW vcov_type arg (None when not supplied);
        # the resolved family lives on the attr. ``robust=`` is gone at 4.0
        # (rows M-045..M-047).
        assert "robust" not in params
        assert params["vcov_type"] is None
        assert did.vcov_type == "hc1"
        assert params["alpha"] == 0.05

        did.set_params(alpha=0.10)
        assert did.alpha == 0.10

    def test_summary_output(self, simple_2x2_data):
        """Test that summary produces string output."""
        did = DifferenceInDifferences()
        did.fit(simple_2x2_data, outcome="outcome", treatment="treated", post="post")

        summary = did.summary()
        assert isinstance(summary, str)
        assert "ATT" in summary
        assert "Difference-in-Differences" in summary

    def test_invalid_treatment_values(self):
        """Test error on non-binary treatment."""
        data = pd.DataFrame(
            {
                "outcome": [1, 2, 3, 4],
                "treated": [0, 1, 2, 3],  # Invalid: not binary
                "post": [0, 0, 1, 1],
            }
        )

        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="binary"):
            did.fit(data, outcome="outcome", treatment="treated", post="post")

    def test_missing_column_error(self):
        """Test error when column is missing."""
        data = pd.DataFrame(
            {
                "outcome": [1, 2, 3, 4],
                "treated": [0, 0, 1, 1],
            }
        )

        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="Missing columns"):
            did.fit(data, outcome="outcome", treatment="treated", post="post")

    def test_unfitted_model_error(self):
        """Test error when accessing results before fitting."""
        did = DifferenceInDifferences()

        with pytest.raises(RuntimeError, match="fitted"):
            did.summary()

    def test_rank_deficient_action_error_raises(self, simple_2x2_data):
        """Test that rank_deficient_action='error' raises ValueError on collinear data."""
        # Add a covariate that is perfectly collinear with treatment
        data = simple_2x2_data.copy()
        data["collinear_cov"] = data["treated"].copy()

        did = DifferenceInDifferences(rank_deficient_action="error")
        with pytest.raises(ValueError, match="rank-deficient"):
            did.fit(
                data,
                outcome="outcome",
                treatment="treated",
                post="post",
                covariates=["collinear_cov"],
            )

    def test_rank_deficient_action_silent_no_warning(self, simple_2x2_data):
        """Test that rank_deficient_action='silent' produces no warning."""
        import warnings

        # Add a covariate that is perfectly collinear with treatment
        data = simple_2x2_data.copy()
        data["collinear_cov"] = data["treated"].copy()

        did = DifferenceInDifferences(rank_deficient_action="silent")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = did.fit(
                data,
                outcome="outcome",
                treatment="treated",
                post="post",
                covariates=["collinear_cov"],
            )

            # No warnings about rank deficiency should be emitted
            rank_warnings = [
                x
                for x in w
                if "Rank-deficient" in str(x.message) or "rank-deficient" in str(x.message).lower()
            ]
            assert len(rank_warnings) == 0, f"Expected no rank warnings, got {rank_warnings}"

        # Should still have NaN for dropped coefficient
        assert "collinear_cov" in results.coefficients
        # Either collinear_cov or treated will be NaN
        has_nan = np.isnan(results.coefficients.get("collinear_cov", 0)) or np.isnan(
            results.coefficients.get("treated", 0)
        )
        assert has_nan, "Expected NaN for one of the collinear coefficients"

    def test_rank_deficient_action_warn_default(self, simple_2x2_data):
        """Test that rank_deficient_action='warn' (default) emits warning."""
        import warnings

        # Add a covariate that is perfectly collinear with treatment
        data = simple_2x2_data.copy()
        data["collinear_cov"] = data["treated"].copy()

        did = DifferenceInDifferences()  # Default is "warn"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            did.fit(
                data,
                outcome="outcome",
                treatment="treated",
                post="post",
                covariates=["collinear_cov"],
            )

            # Should have a warning about rank deficiency
            rank_warnings = [
                x
                for x in w
                if "Rank-deficient" in str(x.message) or "rank-deficient" in str(x.message).lower()
            ]
            assert len(rank_warnings) > 0, "Expected warning about rank deficiency"


class TestDiDResults:
    """Tests for DiDResults class."""

    def test_repr(self, simple_2x2_data):
        """Test string representation."""
        did = DifferenceInDifferences()
        results = did.fit(simple_2x2_data, outcome="outcome", treatment="treated", post="post")

        repr_str = repr(results)
        assert "DiDResults" in repr_str
        assert "ATT=" in repr_str

    def test_to_dict(self, simple_2x2_data):
        """Test conversion to dictionary."""
        did = DifferenceInDifferences()
        results = did.fit(simple_2x2_data, outcome="outcome", treatment="treated", post="post")

        result_dict = results.to_dict()
        assert "att" in result_dict
        assert "se" in result_dict
        assert "p_value" in result_dict

    def test_to_dataframe(self, simple_2x2_data):
        """Test conversion to DataFrame."""
        did = DifferenceInDifferences()
        results = did.fit(simple_2x2_data, outcome="outcome", treatment="treated", post="post")

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "att" in df.columns

    def test_significance_stars(self, simple_did_data):
        """Test significance star notation."""
        did = DifferenceInDifferences()
        results = did.fit(simple_did_data, outcome="outcome", treatment="treated", post="post")

        # With true effect of 3.0 and n=200, should be significant
        assert results.significance_stars in ["*", "**", "***"]

    def test_is_significant_property(self, simple_did_data):
        """Test is_significant property."""
        did = DifferenceInDifferences(alpha=0.05)
        results = did.fit(simple_did_data, outcome="outcome", treatment="treated", post="post")

        # Boolean check
        assert isinstance(results.is_significant, bool)
        # With true effect, should be significant
        assert results.is_significant


class TestFixedEffects:
    """Tests for fixed effects functionality."""

    @pytest.fixture
    def panel_data_with_fe(self):
        """Create panel data with fixed effects."""
        np.random.seed(42)
        n_units = 50
        n_periods = 4
        n_states = 5

        data = []
        for unit in range(n_units):
            state = unit % n_states
            is_treated = unit < n_units // 2
            # State-level effect
            state_effect = state * 2.0

            for period in range(n_periods):
                post = 1 if period >= 2 else 0

                y = 10.0 + state_effect + period * 0.5
                if is_treated and post:
                    y += 3.0  # True ATT

                y += np.random.normal(0, 0.5)

                data.append(
                    {
                        "unit": unit,
                        "state": f"state_{state}",
                        "period": period,
                        "treated": int(is_treated),
                        "post": post,
                        "outcome": y,
                    }
                )

        return pd.DataFrame(data)

    def test_fixed_effects_dummy(self, panel_data_with_fe):
        """Test fixed effects using dummy variables."""
        did = DifferenceInDifferences()
        results = did.fit(
            panel_data_with_fe,
            outcome="outcome",
            treatment="treated",
            post="post",
            fixed_effects=["state"],
        )

        assert results is not None
        assert did.is_fitted_
        # ATT should still be close to 3.0
        assert abs(results.att - 3.0) < 1.0

    def test_fixed_effects_coefficients_include_dummies(self, panel_data_with_fe):
        """Test that dummy coefficients are included in results."""
        did = DifferenceInDifferences()
        results = did.fit(
            panel_data_with_fe,
            outcome="outcome",
            treatment="treated",
            post="post",
            fixed_effects=["state"],
        )

        # Should have state dummy coefficients
        state_coefs = [k for k in results.coefficients.keys() if k.startswith("state_")]
        assert len(state_coefs) == 4  # 5 states - 1 (dropped first)

    def test_absorb_fixed_effects(self, panel_data_with_fe):
        """Test absorbed (within-transformed) fixed effects."""
        did = DifferenceInDifferences()
        results = did.fit(
            panel_data_with_fe, outcome="outcome", treatment="treated", post="post", absorb=["unit"]
        )

        assert results is not None
        assert did.is_fitted_
        # ATT should still be close to 3.0
        assert abs(results.att - 3.0) < 1.0

    def test_fixed_effects_vs_no_fe(self, panel_data_with_fe):
        """Test that FE produces different (usually better) estimates."""
        did_no_fe = DifferenceInDifferences()
        did_with_fe = DifferenceInDifferences()

        results_no_fe = did_no_fe.fit(
            panel_data_with_fe, outcome="outcome", treatment="treated", post="post"
        )

        results_with_fe = did_with_fe.fit(
            panel_data_with_fe,
            outcome="outcome",
            treatment="treated",
            post="post",
            fixed_effects=["state"],
        )

        # Both should estimate positive ATT
        assert results_no_fe.att > 0
        assert results_with_fe.att > 0

        # FE model should have higher R-squared (explains more variance)
        assert results_with_fe.r_squared >= results_no_fe.r_squared

    def test_invalid_fixed_effects_column(self, panel_data_with_fe):
        """Test error when fixed effects column doesn't exist."""
        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="not found"):
            did.fit(
                panel_data_with_fe,
                outcome="outcome",
                treatment="treated",
                post="post",
                fixed_effects=["nonexistent_column"],
            )

    def test_invalid_absorb_column(self, panel_data_with_fe):
        """Test error when absorb column doesn't exist."""
        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="not found"):
            did.fit(
                panel_data_with_fe,
                outcome="outcome",
                treatment="treated",
                post="post",
                absorb=["nonexistent_column"],
            )

    def test_multiple_fixed_effects(self, panel_data_with_fe):
        """Test multiple fixed effects."""
        # Add another categorical variable
        panel_data_with_fe["industry"] = panel_data_with_fe["unit"] % 3

        did = DifferenceInDifferences()
        results = did.fit(
            panel_data_with_fe,
            outcome="outcome",
            treatment="treated",
            post="post",
            fixed_effects=["state", "industry"],
        )

        assert results is not None
        # Should have both state and industry dummies
        state_coefs = [k for k in results.coefficients.keys() if k.startswith("state_")]
        industry_coefs = [k for k in results.coefficients.keys() if k.startswith("industry_")]
        assert len(state_coefs) > 0
        assert len(industry_coefs) > 0

    def test_covariates_with_fixed_effects(self, panel_data_with_fe):
        """Test combining covariates with fixed effects."""
        # Add a continuous covariate
        panel_data_with_fe["size"] = np.random.normal(100, 10, len(panel_data_with_fe))

        did = DifferenceInDifferences()
        results = did.fit(
            panel_data_with_fe,
            outcome="outcome",
            treatment="treated",
            post="post",
            covariates=["size"],
            fixed_effects=["state"],
        )

        assert results is not None
        assert "size" in results.coefficients


class TestParallelTrendsRobust:
    """Tests for robust parallel trends checking."""

    @pytest.fixture
    def parallel_trends_data(self):
        """Create panel data where parallel trends holds."""
        np.random.seed(42)
        n_units = 100
        n_periods = 6  # 3 pre, 3 post

        data = []
        for unit in range(n_units):
            is_treated = unit < n_units // 2
            unit_effect = np.random.normal(0, 2)

            for period in range(n_periods):
                # Common trend for both groups
                time_effect = period * 1.5

                y = 10.0 + unit_effect + time_effect

                # Treatment effect only in post period (period >= 3)
                if is_treated and period >= 3:
                    y += 5.0

                y += np.random.normal(0, 0.5)

                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": int(is_treated),
                        "outcome": y,
                    }
                )

        return pd.DataFrame(data)

    @pytest.fixture
    def non_parallel_trends_data(self):
        """Create panel data where parallel trends is violated."""
        np.random.seed(42)
        n_units = 100
        n_periods = 6

        data = []
        for unit in range(n_units):
            is_treated = unit < n_units // 2
            unit_effect = np.random.normal(0, 2)

            for period in range(n_periods):
                # Different trends for treated vs control
                if is_treated:
                    time_effect = period * 3.0  # Steeper trend
                else:
                    time_effect = period * 1.0  # Flatter trend

                y = 10.0 + unit_effect + time_effect

                # Treatment effect in post period
                if is_treated and period >= 3:
                    y += 5.0

                y += np.random.normal(0, 0.5)

                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": int(is_treated),
                        "outcome": y,
                    }
                )

        return pd.DataFrame(data)

    def test_wasserstein_parallel_trends_valid(self, parallel_trends_data):
        """Test Wasserstein check when parallel trends holds."""
        from diff_diff.utils import check_parallel_trends_robust

        results = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=42,
        )

        assert "wasserstein_distance" in results
        assert "wasserstein_p_value" in results
        assert "ks_statistic" in results
        # When trends are parallel, p-value should be high
        assert results["wasserstein_p_value"] > 0.05
        assert results["parallel_trends_plausible"] is True

    def test_wasserstein_parallel_trends_violated(self, non_parallel_trends_data):
        """Test Wasserstein check when parallel trends is violated."""
        from diff_diff.utils import check_parallel_trends_robust

        results = check_parallel_trends_robust(
            non_parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=42,
        )

        # When trends are not parallel, should detect it
        # Either low p-value or high normalized Wasserstein
        assert results["wasserstein_distance"] > 0
        # The test should flag this as problematic
        assert results["parallel_trends_plausible"] is False

    def test_wasserstein_returns_changes(self, parallel_trends_data):
        """Test that changes arrays are returned."""
        from diff_diff.utils import check_parallel_trends_robust

        results = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=42,
        )

        assert "treated_changes" in results
        assert "control_changes" in results
        assert len(results["treated_changes"]) > 0
        assert len(results["control_changes"]) > 0

    def test_wasserstein_without_unit(self, parallel_trends_data):
        """Test Wasserstein check without unit specification."""
        from diff_diff.utils import check_parallel_trends_robust

        results = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            pre_periods=[0, 1, 2],
            seed=42,
        )

        assert "wasserstein_distance" in results
        assert not np.isnan(results["wasserstein_distance"])

    def test_equivalence_test_parallel(self, parallel_trends_data):
        """Test equivalence testing when trends are parallel."""
        from diff_diff.utils import equivalence_test_trends

        results = equivalence_test_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
        )

        assert "tost_p_value" in results
        assert "equivalent" in results
        assert "equivalence_margin" in results
        # When trends are parallel, should be equivalent
        assert results["equivalent"] is True

    def test_equivalence_test_non_parallel(self, non_parallel_trends_data):
        """Test equivalence testing when trends are not parallel."""
        from diff_diff.utils import equivalence_test_trends

        results = equivalence_test_trends(
            non_parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
        )

        # When trends are not parallel, should not be equivalent
        assert results["equivalent"] is False

    def test_equivalence_test_custom_margin(self, parallel_trends_data):
        """Test equivalence testing with custom margin."""
        from diff_diff.utils import equivalence_test_trends

        results = equivalence_test_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            equivalence_margin=0.1,  # Very tight margin
        )

        assert results["equivalence_margin"] == 0.1

    def test_ks_test_included(self, parallel_trends_data):
        """Test that KS test results are included."""
        from diff_diff.utils import check_parallel_trends_robust

        results = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=42,
        )

        assert "ks_statistic" in results
        assert "ks_p_value" in results
        assert 0 <= results["ks_statistic"] <= 1
        assert 0 <= results["ks_p_value"] <= 1

    def test_variance_ratio(self, parallel_trends_data):
        """Test that variance ratio is computed."""
        from diff_diff.utils import check_parallel_trends_robust

        results = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=42,
        )

        assert "variance_ratio" in results
        assert results["variance_ratio"] > 0


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_multicollinearity_detection(self):
        """Test that perfect multicollinearity is detected and warning is emitted."""
        import warnings

        # Create data where a covariate is perfectly correlated with treatment
        data = pd.DataFrame(
            {
                "outcome": [10, 11, 15, 18, 9, 10, 12, 13],
                "treated": [1, 1, 1, 1, 0, 0, 0, 0],
                "post": [0, 0, 1, 1, 0, 0, 1, 1],
                "duplicate_treated": [1, 1, 1, 1, 0, 0, 0, 0],  # Same as treated
            }
        )

        did = DifferenceInDifferences()

        # With R-style rank deficiency handling, a warning is emitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = did.fit(
                data,
                outcome="outcome",
                treatment="treated",
                post="post",
                covariates=["duplicate_treated"],
            )
            # Should emit a warning about rank deficiency
            rank_warnings = [x for x in w if "Rank-deficient" in str(x.message)]
            assert len(rank_warnings) > 0, "Expected warning about rank deficiency"

        # ATT should still be finite
        assert np.isfinite(result.att)

    def test_wasserstein_custom_threshold(self):
        """Test that custom Wasserstein threshold is respected."""
        from diff_diff.utils import check_parallel_trends_robust

        np.random.seed(42)
        n_units = 50
        n_periods = 4

        data = []
        for unit in range(n_units):
            is_treated = unit < n_units // 2
            for period in range(n_periods):
                y = 10.0 + period * 1.5 + np.random.normal(0, 0.5)
                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": int(is_treated),
                        "outcome": y,
                    }
                )

        df = pd.DataFrame(data)

        # Test with very low threshold (more strict)
        results_strict = check_parallel_trends_robust(
            df,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1],
            seed=42,
            wasserstein_threshold=0.01,  # Very strict
        )

        # Test with high threshold (more lenient)
        results_lenient = check_parallel_trends_robust(
            df,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1],
            seed=42,
            wasserstein_threshold=1.0,  # Very lenient
        )

        # Both should return valid results
        assert "wasserstein_distance" in results_strict
        assert "wasserstein_distance" in results_lenient

    def test_equivalence_test_insufficient_data(self):
        """Test equivalence test handles insufficient data gracefully."""
        from diff_diff.utils import equivalence_test_trends

        # Create minimal data with only 1 observation per group
        data = pd.DataFrame(
            {
                "outcome": [10, 15],
                "period": [0, 1],
                "treated": [1, 0],
                "unit": [0, 1],
            }
        )

        results = equivalence_test_trends(
            data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0],
        )

        # Should return NaN values with error message
        assert np.isnan(results["tost_p_value"])
        assert results["equivalent"] is None
        assert "error" in results

    def test_parallel_trends_single_period(self):
        """Test that single pre-period returns NaN values."""
        from diff_diff.utils import check_parallel_trends

        data = pd.DataFrame(
            {
                "outcome": [10, 11, 12, 13],
                "time": [0, 0, 0, 0],  # All same period
                "treated": [1, 1, 0, 0],
            }
        )

        results = check_parallel_trends(
            data, outcome="outcome", time="time", treatment_group="treated", pre_periods=[0]
        )

        # Should handle gracefully with NaN
        assert np.isnan(results["treated_trend"]) or results["treated_trend"] is None


class TestTwoWayFixedEffects:
    """Tests for TwoWayFixedEffects estimator."""

    @pytest.fixture
    def twfe_panel_data(self):
        """Create panel data for TWFE testing."""
        np.random.seed(42)
        n_units = 20
        n_periods = 4

        data = []
        for unit in range(n_units):
            is_treated = unit < n_units // 2
            unit_effect = np.random.normal(0, 2)

            for period in range(n_periods):
                time_effect = period * 1.0
                post = 1 if period >= 2 else 0

                y = 10.0 + unit_effect + time_effect
                if is_treated and post:
                    y += 3.0  # True ATT

                y += np.random.normal(0, 0.5)

                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": int(is_treated),
                        "post": post,
                        "outcome": y,
                    }
                )

        return pd.DataFrame(data)

    def test_twfe_basic_fit(self, twfe_panel_data):
        """Test basic TWFE model fitting."""
        from diff_diff.estimators import TwoWayFixedEffects

        twfe = TwoWayFixedEffects()
        results = twfe.fit(
            twfe_panel_data, outcome="outcome", treatment="treated", post="post", unit="unit"
        )

        assert results is not None
        assert twfe.is_fitted_
        # ATT should be positive (true effect is 3.0)
        # Note: TWFE with within-transformation may give different estimates
        # due to the mechanics of two-way demeaning
        assert results.att > 0
        assert results.se > 0

    def test_twfe_with_covariates(self, twfe_panel_data):
        """Test TWFE with covariates."""
        from diff_diff.estimators import TwoWayFixedEffects

        # Add a covariate
        twfe_panel_data["size"] = np.random.normal(100, 10, len(twfe_panel_data))

        twfe = TwoWayFixedEffects()
        results = twfe.fit(
            twfe_panel_data,
            outcome="outcome",
            treatment="treated",
            post="post",
            unit="unit",
            covariates=["size"],
        )

        assert results is not None
        assert twfe.is_fitted_

    def test_twfe_invalid_unit_column(self, twfe_panel_data):
        """Test error when unit column doesn't exist."""
        from diff_diff.estimators import TwoWayFixedEffects

        twfe = TwoWayFixedEffects()
        with pytest.raises(ValueError, match="not found"):
            twfe.fit(
                twfe_panel_data,
                outcome="outcome",
                treatment="treated",
                post="post",
                unit="nonexistent_unit",
            )

    def test_twfe_clusters_at_unit_level(self, twfe_panel_data):
        """Test that TWFE defaults to clustering at unit level."""
        from diff_diff.estimators import TwoWayFixedEffects

        twfe = TwoWayFixedEffects()
        results = twfe.fit(
            twfe_panel_data, outcome="outcome", treatment="treated", post="post", unit="unit"
        )

        # Cluster should NOT be mutated (remains None) - clustering is handled internally
        # This ensures the estimator config is immutable as per sklearn convention
        assert twfe.cluster is None
        # But the results should still reflect cluster-robust SEs were computed correctly
        assert results.se > 0

    def test_twfe_treatment_collinearity_raises_error(self):
        """Test that TWFE raises informative error when treatment is collinear."""
        from diff_diff.estimators import TwoWayFixedEffects

        # Create data where treatment is perfectly collinear with fixed effects
        # (all treated units are treated in all periods)
        data = []
        for unit in range(10):
            is_treated = unit < 5
            for period in range(4):
                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": int(is_treated),  # Same for all periods
                        "post": 1 if period >= 2 else 0,
                        "outcome": 10.0 + unit * 0.5 + period * 0.3 + np.random.normal(0, 0.1),
                    }
                )
        df = pd.DataFrame(data)

        # Make treatment_post constant for treated units (collinear)
        # by making treatment only occur in post periods
        df_collinear = df.copy()
        # This creates perfect collinearity: treatment is perfectly predicted by unit FE
        # since treated units always have treated=1 and control units always have treated=0

        twfe = TwoWayFixedEffects()

        # Should raise or warn about collinearity - depends on what columns get dropped
        # The key is that it should NOT silently produce misleading results
        try:
            results = twfe.fit(
                df_collinear, outcome="outcome", treatment="treated", post="post", unit="unit"
            )
            # If we get here without error, the ATT should still be computed
            # (this means only covariates were dropped, not the treatment)
            assert results is not None
        except ValueError as e:
            # If treatment column is dropped, should get informative error
            assert "collinear" in str(e).lower() or "Treatment effect cannot be identified" in str(
                e
            )

    def test_rank_deficient_action_error_raises(self, twfe_panel_data):
        """Test that rank_deficient_action='error' raises ValueError on collinear data."""
        from diff_diff.estimators import TwoWayFixedEffects

        # Add a covariate that is perfectly collinear with post
        twfe_panel_data = twfe_panel_data.copy()
        twfe_panel_data["collinear_cov"] = twfe_panel_data["post"].copy()

        twfe = TwoWayFixedEffects(rank_deficient_action="error")
        with pytest.raises(ValueError, match="rank-deficient"):
            twfe.fit(
                twfe_panel_data,
                outcome="outcome",
                treatment="treated",
                post="post",
                unit="unit",
                covariates=["collinear_cov"],
            )

    def test_rank_deficient_action_silent_no_warning(self, twfe_panel_data):
        """Test that rank_deficient_action='silent' produces no warning."""
        import warnings

        from diff_diff.estimators import TwoWayFixedEffects

        # Add a covariate that is perfectly collinear with another
        twfe_panel_data = twfe_panel_data.copy()
        twfe_panel_data["size"] = np.random.normal(100, 10, len(twfe_panel_data))
        twfe_panel_data["size_dup"] = twfe_panel_data["size"].copy()  # Perfect collinearity

        twfe = TwoWayFixedEffects(rank_deficient_action="silent")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = twfe.fit(
                twfe_panel_data,
                outcome="outcome",
                treatment="treated",
                post="post",
                unit="unit",
                covariates=["size", "size_dup"],
            )

            # No warnings about rank deficiency or collinearity should be emitted
            rank_warnings = [
                x
                for x in w
                if "Rank-deficient" in str(x.message)
                or "rank-deficient" in str(x.message).lower()
                or "collinear" in str(x.message).lower()
            ]
            assert len(rank_warnings) == 0, f"Expected no rank warnings, got {rank_warnings}"

        # Should still get valid results
        assert results is not None
        assert twfe.is_fitted_


class TestClusterRobustSE:
    """Tests for cluster-robust standard errors."""

    def test_cluster_robust_se(self):
        """Test cluster-robust standard errors in base DiD."""
        np.random.seed(42)

        # Create clustered data
        data = []
        for cluster in range(10):
            for obs in range(10):
                treated = cluster < 5
                post = obs >= 5
                y = 10 + (3.0 if treated and post else 0) + np.random.normal(0, 1)
                data.append(
                    {
                        "cluster": cluster,
                        "outcome": y,
                        "treated": int(treated),
                        "post": int(post),
                    }
                )

        df = pd.DataFrame(data)

        # With clustering
        did_cluster = DifferenceInDifferences(cluster="cluster")
        results_cluster = did_cluster.fit(df, outcome="outcome", treatment="treated", post="post")

        # Without clustering
        did_no_cluster = DifferenceInDifferences()
        results_no_cluster = did_no_cluster.fit(
            df, outcome="outcome", treatment="treated", post="post"
        )

        # ATT should be similar
        assert abs(results_cluster.att - results_no_cluster.att) < 0.01

        # SEs should be different (cluster-robust typically larger)
        assert results_cluster.se != results_no_cluster.se


def _multi_period_panel() -> pd.DataFrame:
    """Panel data with multiple time periods and known ATT (6 periods, 3 pre / 3 post,
    effect 3.0 from period 3). Module-level so the TWFE
    event-study tests share it (4.0 cut, PR 4c-i)."""
    np.random.seed(42)
    n_units = 100
    n_periods = 6
    data = []
    for unit in range(n_units):
        is_treated = unit < n_units // 2
        unit_effect = np.random.normal(0, 1)
        for period in range(n_periods):
            y = 10.0 + unit_effect + period * 0.5
            if is_treated and period >= 3:
                y += 3.0
            y += np.random.normal(0, 0.5)
            data.append({"unit": unit, "period": period, "treated": int(is_treated), "outcome": y})
    return pd.DataFrame(data)


def _event_study_panel() -> pd.DataFrame:
    """Panel data with 6 periods, treatment at period 3 (no unit effects)."""
    np.random.seed(42)
    data = []
    for unit in range(100):
        is_treated = unit < 50
        for period in range(6):
            y = 10.0 + period * 0.5
            if is_treated and period >= 3:
                y += 3.0
            y += np.random.normal(0, 0.5)
            data.append({"unit": unit, "period": period, "treated": int(is_treated), "outcome": y})
    return pd.DataFrame(data)


class TestTwoWayFixedEffectsEventStudy:
    """TWFE event-study twins of the former MultiPeriodDiD core-behaviour pins that had no
    TWFE coverage (4.0 cut, PR 4c-i). The unit-driven validation twins keep
    `unit=` (the warnings run only inside the core's `if unit is not None:`
    block) and assert the warning or its absence, never SE values - the pooled
    spec auto-clusters at unit when `unit=` is supplied."""

    @pytest.fixture
    def multi_period_panel(self):
        return _multi_period_panel()

    @pytest.fixture
    def event_study_panel(self):
        return _event_study_panel()

    def _fit(self, data, **kw):
        base = dict(outcome="outcome", treatment="treated", time="period", event_study=True)
        base.update(kw)
        return TwoWayFixedEffects().fit(data, **base)

    def test_reference_period_in_post_raises(self, event_study_panel):
        with pytest.raises(ValueError, match="post-treatment period"):
            self._fit(event_study_panel, spec="pooled", post_periods=[3, 4, 5], reference_period=4)

    def test_staggered_treatment_warning(self):
        np.random.seed(42)
        data = []
        for unit in range(40):
            if unit < 10:
                treat_start = 3
            elif unit < 20:
                treat_start = 5
            else:
                treat_start = None
            for period in range(8):
                is_treated = treat_start is not None and period >= treat_start
                y = 10.0 + period * 0.5 + (2.0 if is_treated else 0) + np.random.normal(0, 0.3)
                data.append(
                    {"unit": unit, "period": period, "treated": int(is_treated), "outcome": y}
                )
        df = pd.DataFrame(data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._fit(
                df, spec="pooled", post_periods=[3, 4, 5, 6, 7], reference_period=2, unit="unit"
            )
        assert any("staggered" in str(x.message).lower() for x in w)

    def test_single_pre_period_warns(self):
        np.random.seed(42)
        data = []
        for unit_id in range(40):
            is_treated = unit_id < 20
            for period in range(3):
                y = 10.0 + period * 0.5 + (2.0 if is_treated and period >= 1 else 0)
                y += np.random.normal(0, 0.3)
                data.append(
                    {"unit": unit_id, "period": period, "treated": int(is_treated), "outcome": y}
                )
        df = pd.DataFrame(data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = self._fit(df, spec="pooled", post_periods=[1, 2], reference_period=0)
        assert any("Only one pre-treatment period" in str(x.message) for x in w)
        att_bar, _ = post_period_average(results)
        assert np.isfinite(att_bar)

    def test_treatment_reversal_warns(self):
        np.random.seed(42)
        data = []
        for unit_id in range(40):
            for period in range(6):
                is_treated = unit_id < 20
                if unit_id == 0:
                    d = 1 if 2 <= period <= 3 else 0
                elif is_treated:
                    d = 1 if period >= 3 else 0
                else:
                    d = 0
                y = 10.0 + period * 0.5 + (2.0 if d else 0) + np.random.normal(0, 0.3)
                data.append({"unit": unit_id, "period": period, "treated": d, "outcome": y})
        df = pd.DataFrame(data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._fit(df, spec="pooled", post_periods=[3, 4, 5], reference_period=2, unit="unit")
        assert any("Treatment reversal" in str(x.message) for x in w)

    def test_time_varying_treatment_warning(self):
        np.random.seed(42)
        data = []
        for unit_id in range(40):
            is_treated = unit_id < 20
            for period in range(6):
                d = 1 if is_treated and period >= 3 else 0
                y = 10.0 + period * 0.5 + (2.0 if d else 0) + np.random.normal(0, 0.3)
                data.append({"unit": unit_id, "period": period, "treated": d, "outcome": y})
        df = pd.DataFrame(data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = self._fit(
                df, spec="pooled", post_periods=[3, 4, 5], reference_period=2, unit="unit"
            )
        dit_warnings = [x for x in w if "time-varying" in str(x.message).lower()]
        assert dit_warnings, "Expected warning about time-varying treatment"
        assert "ever_treated" in str(dit_warnings[0].message)
        assert results is not None
        assert np.sum(~results.is_reference) > 0

    def test_staggered_no_false_positive_unbalanced(self):
        np.random.seed(42)
        data = []
        for unit_id in range(40):
            is_treated = unit_id < 20
            start_period = 4 if (is_treated and unit_id < 5) else 0
            for period in range(start_period, 8):
                d = 1 if is_treated and period >= 3 else 0
                y = 10.0 + period * 0.5 + (2.0 if d else 0) + np.random.normal(0, 0.3)
                data.append({"unit": unit_id, "period": period, "treated": d, "outcome": y})
        df = pd.DataFrame(data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._fit(
                df, spec="pooled", post_periods=[3, 4, 5, 6, 7], reference_period=2, unit="unit"
            )
        assert not any("staggered" in str(x.message).lower() for x in w)

    def _collinear(self, panel):
        panel = panel.copy()
        panel["collinear_cov"] = panel["treated"].copy()
        return panel

    def test_rank_deficient_design_warns_and_keeps_period_rows_finite(self, multi_period_panel):
        """A collinear covariate warns; the per-period quintet stays finite (the
        dropped column is the covariate, not a period interaction)."""
        df = self._collinear(multi_period_panel)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = self._fit(
                df,
                spec="pooled",
                post_periods=[3, 4, 5],
                reference_period=2,
                covariates=["collinear_cov"],
            )
        assert any(
            "Rank-deficient" in str(x.message) or "collinear" in str(x.message).lower() for x in w
        )
        rows = ~results.is_reference
        for field in ("att", "se", "t_stat", "p_value"):
            assert np.all(np.isfinite(getattr(results, field)[rows])), field

    def test_rank_deficient_action_error_raises(self, multi_period_panel):
        df = self._collinear(multi_period_panel)
        with pytest.raises(ValueError, match="rank-deficient"):
            TwoWayFixedEffects(rank_deficient_action="error").fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="period",
                event_study=True,
                spec="pooled",
                post_periods=[3, 4, 5],
                reference_period=2,
                covariates=["collinear_cov"],
            )

    def test_rank_deficient_action_silent_no_warning(self, multi_period_panel):
        df = self._collinear(multi_period_panel)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = TwoWayFixedEffects(rank_deficient_action="silent").fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="period",
                event_study=True,
                spec="pooled",
                post_periods=[3, 4, 5],
                reference_period=2,
                covariates=["collinear_cov"],
            )
        assert not any(
            "Rank-deficient" in str(x.message) or "rank-deficient" in str(x.message).lower()
            for x in w
        ), [str(x.message) for x in w]
        rows = ~results.is_reference
        assert np.all(np.isfinite(results.att[rows]))


class TestSyntheticDiD:
    """Tests for SyntheticDiD estimator."""

    @pytest.fixture
    def sdid_panel_data(self):
        """Create panel data suitable for Synthetic DiD with known ATT."""
        np.random.seed(42)
        n_units = 30
        n_periods = 8  # 4 pre, 4 post
        n_treated = 5  # Few treated units (good use case for SDID)

        data = []
        for unit in range(n_units):
            is_treated = unit < n_treated
            # Unit-specific intercept (varies across units)
            unit_effect = np.random.normal(0, 3)

            for period in range(n_periods):
                # Common time trend
                time_effect = period * 0.5

                y = 10.0 + unit_effect + time_effect

                # Treatment effect in post-periods (periods 4-7)
                if is_treated and period >= 4:
                    y += 5.0  # True ATT = 5.0

                y += np.random.normal(0, 0.5)

                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": int(is_treated),
                        "outcome": y,
                    }
                )

        return pd.DataFrame(data)

    @pytest.fixture
    def single_treated_unit_data(self):
        """Create data with a single treated unit (classic SC case)."""
        np.random.seed(42)
        n_controls = 20
        n_periods = 10  # 5 pre, 5 post

        data = []

        # Single treated unit with distinct pattern
        for period in range(n_periods):
            y = 50.0 + period * 2.0  # Steeper trend
            if period >= 5:
                y += 10.0  # True ATT = 10
            y += np.random.normal(0, 1)
            data.append(
                {
                    "unit": 0,
                    "period": period,
                    "treated": 1,
                    "outcome": y,
                }
            )

        # Control units with various patterns
        for unit in range(1, n_controls + 1):
            unit_intercept = np.random.uniform(30, 70)
            unit_slope = np.random.uniform(0.5, 3.5)  # Various slopes
            for period in range(n_periods):
                y = unit_intercept + period * unit_slope
                y += np.random.normal(0, 1)
                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": 0,
                        "outcome": y,
                    }
                )

        return pd.DataFrame(data)

    def test_basic_fit(self, sdid_panel_data, ci_params):
        """Test basic SDID model fitting."""
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        assert isinstance(results, SyntheticDiDResults)
        assert sdid.is_fitted_
        assert results.n_obs == 240  # 30 units * 8 periods
        assert results.n_treated == 5
        assert results.n_control == 25

    def test_att_direction(self, sdid_panel_data, ci_params):
        """Test that ATT is estimated in correct direction."""
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        # True ATT is 5.0
        assert results.att > 0
        assert abs(results.att - 5.0) < 2.0

    def test_unit_weights_sum_to_one(self, sdid_panel_data):
        """Test that unit weights sum to 1."""
        sdid = SyntheticDiD(variance_method="placebo", seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        weight_sum = sum(results.unit_weights.values())
        assert abs(weight_sum - 1.0) < 1e-6

    def test_time_weights_sum_to_one(self, sdid_panel_data):
        """Test that time weights sum to 1."""
        sdid = SyntheticDiD(variance_method="placebo", seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        weight_sum = sum(results.time_weights.values())
        assert abs(weight_sum - 1.0) < 1e-6

    def test_unit_weights_nonnegative(self, sdid_panel_data):
        """Test that unit weights are non-negative."""
        sdid = SyntheticDiD(variance_method="placebo", seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        for w in results.unit_weights.values():
            assert w >= 0

    def test_single_treated_unit(self, single_treated_unit_data, ci_params):
        """Test SDID with a single treated unit (classic SC scenario)."""
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            single_treated_unit_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7, 8, 9],
        )

        assert results.n_treated == 1
        # True ATT is 10.0
        assert results.att > 0
        # With good controls, should be reasonably close
        assert abs(results.att - 10.0) < 5.0

    def test_regularization_effect(self, sdid_panel_data):
        """Test that regularization affects weight dispersion."""
        sdid_no_reg = SyntheticDiD(zeta_omega=0.01, variance_method="placebo", seed=42)
        sdid_high_reg = SyntheticDiD(zeta_omega=100.0, variance_method="placebo", seed=42)

        results_no_reg = sdid_no_reg.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        results_high_reg = sdid_high_reg.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        # High regularization should give more uniform weights
        weights_no_reg = np.array(list(results_no_reg.unit_weights.values()))
        weights_high_reg = np.array(list(results_high_reg.unit_weights.values()))

        # Variance of weights should be lower with more regularization
        assert np.var(weights_high_reg) <= np.var(weights_no_reg) + 0.01

    def test_placebo_inference(self, sdid_panel_data):
        """Test placebo-based variance estimation (variance_method='placebo')."""
        sdid = SyntheticDiD(variance_method="placebo", seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        assert results.variance_method == "placebo"
        assert results.variance_effects is not None
        assert len(results.variance_effects) > 0
        assert results.se > 0

    def test_bootstrap_inference(self, sdid_panel_data, ci_params):
        """Test bootstrap-based inference."""
        n_boot = ci_params.bootstrap(100)
        sdid = SyntheticDiD(variance_method="bootstrap", n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        assert results.variance_method == "bootstrap"
        assert results.n_bootstrap == n_boot
        assert results.se > 0
        assert results.conf_int[0] < results.att < results.conf_int[1]

    def test_jackknife_inference(self, sdid_panel_data):
        """Test jackknife-based variance estimation."""
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )
        assert results.variance_method == "jackknife"
        assert results.se > 0
        assert results.conf_int[0] < results.att < results.conf_int[1]
        # n_bootstrap should be None for jackknife
        assert results.n_bootstrap is None

    def test_jackknife_valid_method(self):
        """Test that 'jackknife' is accepted as a variance_method."""
        sdid = SyntheticDiD(variance_method="jackknife")
        assert sdid.variance_method == "jackknife"

    def test_invalid_variance_method(self):
        """Test that invalid variance_method raises ValueError."""
        with pytest.raises(ValueError, match="variance_method must be one of"):
            SyntheticDiD(variance_method="invalid")

    def test_get_unit_weights_df(self, sdid_panel_data):
        """Test getting unit weights as DataFrame."""
        sdid = SyntheticDiD(variance_method="placebo", seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        weights_df = results.get_unit_weights_df()
        assert isinstance(weights_df, pd.DataFrame)
        assert "unit" in weights_df.columns
        assert "weight" in weights_df.columns
        assert len(weights_df) == 25  # Number of control units

    def test_get_time_weights_df(self, sdid_panel_data):
        """Test getting time weights as DataFrame."""
        sdid = SyntheticDiD(variance_method="placebo", seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        weights_df = results.get_time_weights_df()
        assert isinstance(weights_df, pd.DataFrame)
        assert "period" in weights_df.columns
        assert "weight" in weights_df.columns
        assert len(weights_df) == 4  # Number of pre-periods

    def test_pre_treatment_fit(self, sdid_panel_data):
        """Test that pre-treatment fit is computed."""
        sdid = SyntheticDiD(variance_method="placebo", seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        assert results.pre_treatment_fit is not None
        assert results.pre_treatment_fit >= 0
        assert isinstance(results.pre_treatment_level_gap, float)
        assert np.isfinite(results.pre_treatment_level_gap)

    def test_summary_output(self, sdid_panel_data, ci_params):
        """Test that summary produces string output."""
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        summary = results.summary()
        assert isinstance(summary, str)
        assert "Synthetic Difference-in-Differences" in summary
        assert "ATT" in summary
        assert "Unit Weights" in summary

    def test_to_dict(self, sdid_panel_data, ci_params):
        """Test conversion to dictionary."""
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        result_dict = results.to_dict()
        assert "att" in result_dict
        assert "se" in result_dict
        assert "n_pre_periods" in result_dict
        assert "n_post_periods" in result_dict
        assert "pre_treatment_fit" in result_dict
        assert "pre_treatment_level_gap" in result_dict
        assert np.isfinite(result_dict["pre_treatment_level_gap"])
        assert "pre_fit_placebo_pvalue" in result_dict
        assert 0.0 < result_dict["pre_fit_placebo_pvalue"] <= 1.0

    def test_to_dataframe(self, sdid_panel_data, ci_params):
        """Test conversion to DataFrame."""
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "att" in df.columns

    def test_repr(self, sdid_panel_data, ci_params):
        """Test string representation."""
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        repr_str = repr(results)
        assert "SyntheticDiDResults" in repr_str
        assert "ATT=" in repr_str

    def test_is_significant_property(self, sdid_panel_data, ci_params):
        """Test is_significant property."""
        n_boot = ci_params.bootstrap(100)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        assert isinstance(results.is_significant, bool)

    def test_get_set_params(self):
        """Test get_params and set_params with new parameter names."""
        sdid = SyntheticDiD(zeta_omega=1.0, zeta_lambda=0.5, alpha=0.10, variance_method="placebo")

        params = sdid.get_params()
        assert params["zeta_omega"] == 1.0
        assert params["zeta_lambda"] == 0.5
        assert params["alpha"] == 0.10
        assert params["variance_method"] == "placebo"

        sdid.set_params(zeta_omega=2.0)
        assert sdid.zeta_omega == 2.0

        sdid.set_params(variance_method="bootstrap")
        assert sdid.variance_method == "bootstrap"

    def test_removed_params(self):
        """M-001/M-002 removal pins: the pre-3.5 ``lambda_reg`` / ``zeta``
        names are rejected by the constructor and by set_params."""
        with pytest.raises(TypeError, match="unexpected keyword argument 'lambda_reg'"):
            SyntheticDiD(lambda_reg=1.0)
        with pytest.raises(TypeError, match="unexpected keyword argument 'zeta'"):
            SyntheticDiD(zeta=0.5)

        sdid = SyntheticDiD()
        assert "lambda_reg" not in sdid.get_params()
        assert "zeta" not in sdid.get_params()
        with pytest.raises(ValueError, match="Unknown parameter"):
            sdid.set_params(lambda_reg=2.0)
        # auto-computed regularization is the default
        assert sdid.zeta_omega is None
        assert sdid.zeta_lambda is None

    def test_missing_unit_column(self, sdid_panel_data):
        """Test error when unit column is missing."""
        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="Missing columns"):
            sdid.fit(
                sdid_panel_data,
                outcome="outcome",
                treatment="treated",
                unit="nonexistent",
                time="period",
                post_periods=[4, 5, 6, 7],
            )

    def test_missing_time_column(self, sdid_panel_data):
        """Test error when time column is missing."""
        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="Missing columns"):
            sdid.fit(
                sdid_panel_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="nonexistent",
                post_periods=[4, 5, 6, 7],
            )

    def test_no_treated_units_error(self):
        """Test error when no treated units."""
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "period": [0, 1, 0, 1],
                "treated": [0, 0, 0, 0],
                "outcome": [10, 11, 12, 13],
            }
        )

        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="No treated units"):
            sdid.fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[1],
            )

    def test_no_control_units_error(self):
        """Test error when no control units."""
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "period": [0, 1, 0, 1],
                "treated": [1, 1, 1, 1],
                "outcome": [10, 11, 12, 13],
            }
        )

        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="No control units"):
            sdid.fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[1],
            )

    def test_auto_infer_post_periods(self, sdid_panel_data):
        """Test automatic inference of post-periods."""
        sdid = SyntheticDiD(variance_method="placebo", seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            # post_periods not specified
        )

        # With 8 periods, should infer last 4 as post
        assert results.pre_periods == [0, 1, 2, 3]
        assert results.post_periods == [4, 5, 6, 7]

    def test_with_covariates(self, sdid_panel_data, ci_params):
        """Test SDID with covariates."""
        # Add a covariate
        sdid_panel_data["size"] = np.random.normal(100, 10, len(sdid_panel_data))

        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
            covariates=["size"],
        )

        assert results is not None
        assert sdid.is_fitted_

    def test_confidence_interval_contains_estimate(self, sdid_panel_data, ci_params):
        """Test that confidence interval contains the estimate."""
        n_boot = ci_params.bootstrap(100)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        lower, upper = results.conf_int
        assert lower < results.att < upper

    def test_reproducibility_with_seed(self, sdid_panel_data, ci_params):
        """Test that results are reproducible with the same seed."""
        n_boot = ci_params.bootstrap(50)
        results1 = SyntheticDiD(n_bootstrap=n_boot, seed=42).fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        results2 = SyntheticDiD(n_bootstrap=n_boot, seed=42).fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
        )

        assert results1.att == results2.att
        assert results1.se == results2.se

    def test_insufficient_pre_periods_warning(self, ci_params):
        """Test that SDID warns with very few pre-treatment periods."""
        np.random.seed(42)

        # Create data with only 2 pre-treatment periods
        n_control = 8
        n_periods = 4  # 2 pre, 2 post
        post_periods = [2, 3]

        data = []
        # Treated unit
        for t in range(n_periods):
            y = 10.0 + t * 0.5 + np.random.normal(0, 0.3)
            if t in post_periods:
                y += 3.0
            data.append(
                {
                    "unit": 0,
                    "period": t,
                    "outcome": y,
                    "treated": 1,
                }
            )

        # Control units
        for unit in range(1, n_control + 1):
            for t in range(n_periods):
                y = 8.0 + unit * 0.2 + t * 0.4 + np.random.normal(0, 0.3)
                data.append(
                    {
                        "unit": unit,
                        "period": t,
                        "outcome": y,
                        "treated": 0,
                    }
                )

        df = pd.DataFrame(data)

        n_boot = ci_params.bootstrap(30)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)

        # Should work but may warn about few pre-periods
        # (Depending on implementation - some may warn, some may not)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=post_periods,
        )

        # Results should still be valid
        assert np.isfinite(results.att)
        assert results.se > 0

    def test_single_pre_period_edge_case(self, ci_params):
        """Test SDID with single pre-treatment period (extreme edge case)."""
        np.random.seed(42)

        n_control = 5
        n_periods = 3  # 1 pre, 2 post
        post_periods = [1, 2]

        data = []
        # Treated unit
        for t in range(n_periods):
            y = 10.0 + np.random.normal(0, 0.2)
            if t in post_periods:
                y += 2.0
            data.append(
                {
                    "unit": 0,
                    "period": t,
                    "outcome": y,
                    "treated": 1,
                }
            )

        # Control units
        for unit in range(1, n_control + 1):
            for t in range(n_periods):
                y = 9.0 + np.random.normal(0, 0.2)
                data.append(
                    {
                        "unit": unit,
                        "period": t,
                        "outcome": y,
                        "treated": 0,
                    }
                )

        df = pd.DataFrame(data)

        n_boot = ci_params.bootstrap(30)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)

        # With single pre-period, time weights will be trivially [1.0]
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=post_periods,
        )

        # Should still produce results
        assert np.isfinite(results.att)
        # Time weights should have single entry
        assert len(results.time_weights) == 1

    def test_more_pre_periods_than_control_units(self, ci_params):
        """Test SDID when n_pre_periods > n_control_units (underdetermined)."""
        np.random.seed(42)

        n_control = 3  # Few control units
        n_periods = 10  # Many periods
        post_periods = [8, 9]  # 8 pre-treatment periods

        data = []
        # Treated unit
        for t in range(n_periods):
            y = 10.0 + t * 0.2 + np.random.normal(0, 0.3)
            if t in post_periods:
                y += 2.5
            data.append(
                {
                    "unit": 0,
                    "period": t,
                    "outcome": y,
                    "treated": 1,
                }
            )

        # Control units
        for unit in range(1, n_control + 1):
            for t in range(n_periods):
                y = 8.0 + t * 0.15 + np.random.normal(0, 0.3)
                data.append(
                    {
                        "unit": unit,
                        "period": t,
                        "outcome": y,
                        "treated": 0,
                    }
                )

        df = pd.DataFrame(data)

        # Use regularization to help with underdetermined system
        n_boot = ci_params.bootstrap(30)
        sdid = SyntheticDiD(zeta_omega=1.0, n_bootstrap=n_boot, seed=42)

        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=post_periods,
        )

        # Should produce valid results with regularization
        assert np.isfinite(results.att)
        assert results.se > 0

    def test_unknown_kwargs_rejected(self):
        """Unknown kwargs raise TypeError instead of being silently swallowed."""
        with pytest.raises(TypeError):
            SyntheticDiD(bogus_param=True)


class TestSyntheticWeightsUtils:
    """Tests for synthetic weight utility functions."""

    def test_project_simplex(self):
        """Test simplex projection."""
        from diff_diff.utils import _project_simplex

        # Already on simplex
        v = np.array([0.3, 0.3, 0.4])
        projected = _project_simplex(v)
        assert abs(np.sum(projected) - 1.0) < 1e-6
        assert np.all(projected >= 0)

        # Negative values
        v = np.array([-0.5, 0.5, 1.0])
        projected = _project_simplex(v)
        assert abs(np.sum(projected) - 1.0) < 1e-6
        assert np.all(projected >= 0)

        # Values summing to more than 1
        v = np.array([0.5, 0.5, 0.5])
        projected = _project_simplex(v)
        assert abs(np.sum(projected) - 1.0) < 1e-6
        assert np.all(projected >= 0)

    # test_compute_synthetic_weights removed in the silent-failures audit
    # post-cleanup (finding #22). Helper deleted; behavior now covered via
    # tests/test_prep.py::TestRankControlUnits (its sole caller).

    def test_compute_time_weights(self):
        """Test time weight computation with Frank-Wolfe solver."""
        from diff_diff.utils import compute_time_weights

        np.random.seed(42)
        n_pre = 5
        n_post = 3
        n_control = 10

        Y_pre = np.random.randn(n_pre, n_control)
        Y_post = np.random.randn(n_post, n_control)

        weights = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        # Weights should sum to 1
        assert abs(np.sum(weights) - 1.0) < 1e-6
        # Weights should be non-negative
        assert np.all(weights >= -1e-10)
        # Should have correct length
        assert len(weights) == n_pre

    def test_compute_sdid_estimator(self):
        """Test SDID estimator computation."""
        from diff_diff.utils import compute_sdid_estimator

        # Simple case with known answer
        Y_pre_control = np.array([[10.0], [10.0]])
        Y_post_control = np.array([[12.0], [12.0]])
        Y_pre_treated = np.array([10.0, 10.0])
        Y_post_treated = np.array([15.0, 15.0])

        unit_weights = np.array([1.0])
        time_weights = np.array([0.5, 0.5])

        tau = compute_sdid_estimator(
            Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated, unit_weights, time_weights
        )

        # Treated: 15 - 10 = 5
        # Control: 12 - 10 = 2
        # SDID: 5 - 2 = 3
        assert abs(tau - 3.0) < 1e-6


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestUnbalancedPanels:
    """Tests for handling unbalanced panels with missing periods."""

    def test_did_with_missing_periods(self):
        """Test DifferenceInDifferences handles missing periods gracefully."""
        # Create unbalanced panel - some units missing some periods
        np.random.seed(42)
        data = []

        for unit in range(20):
            is_treated = unit < 10
            # Some units missing period 0, some missing period 1
            periods = [0, 1]
            if unit % 5 == 0:
                periods = [1]  # Missing pre-period
            elif unit % 7 == 0:
                periods = [0]  # Missing post-period

            for period in periods:
                y = 10.0 + unit * 0.1
                if period == 1:
                    y += 5.0
                if is_treated and period == 1:
                    y += 3.0
                y += np.random.normal(0, 1)

                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": int(is_treated),
                        "post": period,
                        "outcome": y,
                    }
                )

        df = pd.DataFrame(data)

        did = DifferenceInDifferences()
        results = did.fit(df, outcome="outcome", treatment="treated", post="post")

        # Should still produce valid results
        assert np.isfinite(results.att)
        assert results.se > 0
        assert results.n_obs == len(df)

    def test_twfe_with_unbalanced_panel(self):
        """Test TwoWayFixedEffects handles unbalanced panels."""
        from diff_diff import TwoWayFixedEffects

        np.random.seed(42)
        data = []

        for unit in range(15):
            is_treated = unit < 8
            unit_effect = np.random.normal(0, 2)

            # Create unbalanced panel - varying number of periods per unit
            if unit < 5:
                periods = [0, 1, 2, 3]  # Full panel
            elif unit < 10:
                periods = [0, 1, 3]  # Missing period 2
            else:
                periods = [1, 2, 3]  # Missing period 0

            for period in periods:
                time_effect = period * 0.5
                post = 1 if period >= 2 else 0

                y = 10.0 + unit_effect + time_effect
                if is_treated and post:
                    y += 3.0
                y += np.random.normal(0, 0.5)

                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": int(is_treated),
                        "post": post,
                        "outcome": y,
                    }
                )

        df = pd.DataFrame(data)

        twfe = TwoWayFixedEffects()
        # NOTE: this test previously passed treatment="post" with time="period",
        # making the treatment interaction a pure function of the period FE -
        # an unidentifiable specification that "worked" only because the
        # FE-spanned junk column survived rank detection and produced a finite
        # garbage ATT. The v3.6.x span guard now routes that spec into TWFE's
        # collinearity error, so the test uses the well-specified form.
        results = twfe.fit(df, outcome="outcome", treatment="treated", unit="unit", post="post")

        # Should produce valid results
        assert np.isfinite(results.att)
        assert results.se > 0
        # the degenerate spec now raises the pre-existing identification error
        # (routed there deterministically by the FE-span snap):
        with pytest.raises(ValueError, match="cannot be identified"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                TwoWayFixedEffects().fit(
                    df, outcome="outcome", treatment="post", unit="unit", post="period"
                )

    def test_event_study_with_sparse_data(self):
        """Test the TWFE event-study mode with sparse data across periods."""
        np.random.seed(42)
        data = []

        n_units = 30
        for unit in range(n_units):
            is_treated = unit < n_units // 2

            # Each unit observed in random subset of periods
            available_periods = np.random.choice([0, 1, 2, 3, 4], size=3, replace=False)
            available_periods = sorted(available_periods)

            for period in available_periods:
                y = 10.0 + np.random.normal(0, 1)
                if period >= 2:
                    y += 2.0  # Time effect
                if is_treated and period >= 2:
                    y += 3.0  # Treatment effect

                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": int(is_treated),
                        "outcome": y,
                    }
                )

        df = pd.DataFrame(data)

        # Pooled event study (the former MultiPeriodDiD fit); the midpoint default
        # over the observed periods 0-4 resolved to post [2, 3, 4].
        results = TwoWayFixedEffects().fit(
            df,
            outcome="outcome",
            treatment="treated",
            time="period",
            event_study=True,
            spec="pooled",
            post_periods=[2, 3, 4],
            reference_period=1,
        )

        # Should produce valid results
        att_bar, _ = post_period_average(results)
        assert np.isfinite(att_bar)
        assert np.sum(~results.is_reference) > 0


class TestSingleTreatedUnit:
    """Tests for scenarios with only one treated unit."""

    def test_did_single_treated_unit(self):
        """Test DifferenceInDifferences with single treated unit."""
        np.random.seed(42)
        data = []

        # 1 treated unit, 10 control units
        for unit in range(11):
            is_treated = unit == 0

            for period in [0, 1]:
                y = 10.0 + np.random.normal(0, 0.5)
                if period == 1:
                    y += 2.0
                if is_treated and period == 1:
                    y += 5.0  # Large effect for single unit

                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": int(is_treated),
                        "post": period,
                        "outcome": y,
                    }
                )

        df = pd.DataFrame(data)

        did = DifferenceInDifferences()
        results = did.fit(df, outcome="outcome", treatment="treated", post="post")

        # Should produce valid results
        assert np.isfinite(results.att)
        assert results.se > 0
        assert results.n_treated == 2  # 1 unit × 2 periods

    def test_sdid_single_treated_unit(self):
        """Test SyntheticDiD with single treated unit (primary use case)."""
        np.random.seed(42)

        n_control = 10
        n_periods = 6
        post_periods = [4, 5]

        data = []

        # Generate treated unit
        treated_base = 15.0
        treated_trend = 0.5
        for t in range(n_periods):
            y = treated_base + treated_trend * t + np.random.normal(0, 0.3)
            if t in post_periods:
                y += 3.0  # Treatment effect
            data.append(
                {
                    "unit": 0,
                    "period": t,
                    "outcome": y,
                    "treated": 1,
                }
            )

        # Generate control units
        for unit in range(1, n_control + 1):
            unit_base = 10.0 + np.random.normal(0, 2)
            unit_trend = 0.4 + np.random.normal(0, 0.1)
            for t in range(n_periods):
                y = unit_base + unit_trend * t + np.random.normal(0, 0.3)
                data.append(
                    {
                        "unit": unit,
                        "period": t,
                        "outcome": y,
                        "treated": 0,
                    }
                )

        df = pd.DataFrame(data)

        sdid = SyntheticDiD(n_bootstrap=50, seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=post_periods,
        )

        # SDID is designed for single/few treated units
        assert np.isfinite(results.att)
        assert results.se > 0
        # Effect should be roughly correct
        assert abs(results.att - 3.0) < 2.0


class TestCollinearityDetection:
    """Tests for handling perfect or near collinearity."""

    def test_did_with_redundant_covariate_emits_warning(self):
        """Test DiD emits warning for perfectly collinear covariates.

        Following R's lm() approach, rank-deficient matrices emit a warning
        and set NaN for coefficients of dropped columns. The ATT should still
        be identified if the treatment interaction is not in the dropped set.
        """
        import warnings

        np.random.seed(42)
        data = pd.DataFrame(
            {
                "outcome": np.random.normal(10, 1, 100),
                "treated": np.repeat([0, 1], 50),
                "post": np.tile([0, 1], 50),
                "x1": np.random.normal(0, 1, 100),
            }
        )
        # Add perfectly collinear covariate
        data["x2"] = data["x1"] * 2 + 3

        did = DifferenceInDifferences()

        # With R-style rank deficiency handling, a warning is emitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = did.fit(
                data, outcome="outcome", treatment="treated", post="post", covariates=["x1", "x2"]
            )
            # Should emit a warning about rank deficiency
            rank_warnings = [x for x in w if "Rank-deficient" in str(x.message)]
            assert len(rank_warnings) > 0, "Expected warning about rank deficiency"

        # ATT should still be finite (the interaction term is identified)
        assert np.isfinite(result.att)

    def test_did_with_constant_covariate_emits_warning(self):
        """Test DiD emits warning for constant covariates.

        Constant covariates are collinear with the intercept and are dropped.
        Following R's lm() approach, a warning is emitted and the coefficient
        for the constant covariate is set to NaN.
        """
        import warnings

        np.random.seed(42)
        data = pd.DataFrame(
            {
                "outcome": np.random.normal(10, 1, 100),
                "treated": np.repeat([0, 1], 50),
                "post": np.tile([0, 1], 50),
                "constant_x": np.ones(100),  # Constant covariate
            }
        )

        did = DifferenceInDifferences()

        # Constant covariate is collinear with intercept
        # Should emit warning about rank deficiency
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = did.fit(
                data, outcome="outcome", treatment="treated", post="post", covariates=["constant_x"]
            )
            # Should emit a warning about rank deficiency
            rank_warnings = [x for x in w if "Rank-deficient" in str(x.message)]
            assert len(rank_warnings) > 0, "Expected warning about rank deficiency"

        # ATT should still be finite
        assert np.isfinite(result.att)

    def test_did_with_near_collinear_covariates(self):
        """Test DiD handles near-collinear covariates (not perfectly collinear)."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "outcome": np.random.normal(10, 1, 100),
                "treated": np.repeat([0, 1], 50),
                "post": np.tile([0, 1], 50),
                "x1": np.random.normal(0, 1, 100),
            }
        )
        # Add near-collinear covariate (small noise breaks perfect collinearity)
        data["x2"] = data["x1"] * 2 + 3 + np.random.normal(0, 0.1, 100)

        did = DifferenceInDifferences()

        # Near-collinear should work (not perfectly rank-deficient)
        results = did.fit(
            data, outcome="outcome", treatment="treated", post="post", covariates=["x1", "x2"]
        )

        assert np.isfinite(results.att)

    def test_twfe_with_absorbed_covariate(self):
        """Test TWFE handles covariate absorbed by fixed effects.

        A unit-level covariate (constant within unit) becomes zero after
        within-transformation, causing rank deficiency. TWFE should handle
        this gracefully (warn but still estimate ATT).
        """
        from diff_diff import TwoWayFixedEffects

        np.random.seed(42)
        n_units = 20
        n_periods = 4

        data = []
        for unit in range(n_units):
            # Unit-specific covariate (absorbed by unit FE)
            unit_x = np.random.normal(0, 1)

            for period in range(n_periods):
                y = 10.0 + unit * 0.5 + period * 0.3 + np.random.normal(0, 0.5)
                post = 1 if period >= 2 else 0
                if unit < n_units // 2 and post:
                    y += 2.0

                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "outcome": y,
                        "treated": int(unit < n_units // 2),
                        "post": post,
                        "unit_covariate": unit_x,  # Same for all periods within unit
                    }
                )

        df = pd.DataFrame(data)

        # Use correct TWFE specification: treatment="treated", time="post"
        # Include unit_covariate which is constant within unit and will be
        # absorbed by unit FE (becomes zero after within-transformation)
        twfe = TwoWayFixedEffects(rank_deficient_action="silent")
        results = twfe.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            post="post",
            covariates=["unit_covariate"],
        )

        assert np.isfinite(results.att)
        assert results.se > 0


class TestCovariateNameCollision:
    """PR3: a covariate whose name equals a reserved structural term silently
    overwrote that coefficient in the zip()-built coef_dict (dict last-write-wins).
    The DiD-family estimators now raise ValueError; non-colliding covariates leave
    the structural coefficients intact. TWFE collision tests live in
    tests/test_estimators_vcov_type.py (next to the full-dummy invariant)."""

    @staticmethod
    def _did_data():
        rng = np.random.default_rng(7)
        rows = []
        for unit in range(60):
            treated = int(unit < 30)
            for post in (0, 1):
                y = 1.0 + 2.0 * treated * post + rng.normal()
                rows.append({"unit": unit, "treated": treated, "post": post, "outcome": y})
        df = pd.DataFrame(rows)
        df["x1"] = np.random.default_rng(11).normal(size=len(df))
        return df

    @staticmethod
    def _event_study_collision_panel():
        rng = np.random.default_rng(7)
        rows = []
        for unit in range(60):
            treated = int(unit < 30)
            ueff = rng.normal()
            for period in range(6):
                y = (
                    10.0
                    + ueff
                    + 0.5 * period
                    + (3.0 if (treated and period >= 3) else 0.0)
                    + rng.normal(0, 0.5)
                )
                rows.append({"unit": unit, "treated": treated, "time": period, "outcome": y})
        df = pd.DataFrame(rows)
        df["x1"] = np.random.default_rng(11).normal(size=len(df))
        return df

    def test_did_collision_for_each_structural_name(self):
        # Drive the reserved set from a clean fit's actual coefficient keys so
        # the test cannot drift from the real var_names (const/treated/post/
        # treated:post), plus the internal _treat_time working column.
        data = self._did_data()
        clean = DifferenceInDifferences().fit(data, "outcome", "treated", "post", covariates=["x1"])
        reserved = [k for k in clean.coefficients if k != "x1"] + ["_treat_time"]
        assert "const" in reserved and "treated:post" in reserved  # sanity
        for name in reserved:
            d2 = data.copy()
            if name not in d2.columns:
                d2[name] = np.random.default_rng(5).normal(size=len(d2))
            with pytest.raises(ValueError, match="collide"):
                DifferenceInDifferences().fit(d2, "outcome", "treated", "post", covariates=[name])

    def test_did_fixed_effects_dummy_collision(self):
        # get_dummies(region, prefix="region", drop_first=True) drops "region_A"
        # and keeps "region_B" -> a covariate named "region_B" collides.
        data = self._did_data()
        data["region"] = np.where(data["unit"] % 2 == 0, "A", "B")
        data["region_B"] = np.random.default_rng(2).normal(size=len(data))
        with pytest.raises(ValueError, match="collide"):
            DifferenceInDifferences().fit(
                data,
                "outcome",
                "treated",
                "post",
                covariates=["region_B"],
                fixed_effects=["region"],
            )

    def test_did_noncolliding_preserves_structural_coefs(self):
        data = self._did_data()
        r = DifferenceInDifferences().fit(data, "outcome", "treated", "post", covariates=["x1"])
        ck = r.coefficients
        assert np.isfinite(ck["const"]) and np.isfinite(ck["treated:post"])
        assert "x1" in ck and ck["x1"] != ck["treated:post"]
        # No key was overwritten: dict size matches the vcov rank.
        assert len(ck) == r.vcov.shape[0]

    def test_did_formula_colliding_raises(self):
        data = self._did_data()
        data["const"] = np.random.default_rng(3).normal(size=len(data))
        with pytest.raises(ValueError, match="collide"):
            DifferenceInDifferences().fit(data, formula="outcome ~ treated*post + const")

    def test_did_formula_noncolliding_works(self):
        data = self._did_data()
        r = DifferenceInDifferences().fit(data, formula="outcome ~ treated*post + x1")
        assert "x1" in r.coefficients

    def test_did_duplicate_covariates_raise(self):
        data = self._did_data()
        with pytest.raises(ValueError, match="duplicate"):
            DifferenceInDifferences().fit(
                data, "outcome", "treated", "post", covariates=["x1", "x1"]
            )

    def test_event_study_collision_for_each_structural_name(self):
        # Port of the former MultiPeriodDiD pin onto the TWFE event-study
        # surface (4.0 cut, PR 4c-ii): the only test of the event-study core's
        # reserved-name guard. Drive the period set from a clean fit's actual
        # vcov_index (the non-reference calendar periods) so the test cannot
        # drift from the real structural names (const/treated/period_*/
        # treated:period_*), plus the internal _did_treatment working column.
        data = self._event_study_collision_panel()
        kw = dict(
            outcome="outcome",
            treatment="treated",
            time="time",
            event_study=True,
            spec="pooled",
            post_periods=[3, 4, 5],
            reference_period=2,
        )
        clean = TwoWayFixedEffects().fit(data, covariates=["x1"], **kw)
        idx = [int(p) for p in clean.vcov_index.tolist()]
        assert idx == [0, 1, 3, 4, 5]  # sanity: every non-reference period
        reserved = (
            ["const", "treated", "_did_treatment"]
            + [f"period_{p}" for p in idx]
            + [f"treated:period_{p}" for p in idx]
        )
        for name in reserved:
            d2 = data.copy()
            if name not in d2.columns:
                d2[name] = np.random.default_rng(5).normal(size=len(d2))
            with pytest.raises(ValueError, match="collide"):
                TwoWayFixedEffects().fit(d2, covariates=[name], **kw)

    def test_synthetic_did_unaffected_by_guard(self):
        # SyntheticDiD overrides fit() and never reaches the base-class guard;
        # a covariate must still fit (regression lock that the guard didn't leak).
        rng = np.random.default_rng(42)
        rows = []
        for unit in range(30):
            treated = int(unit < 5)
            ueff = rng.normal(0, 3)
            for period in range(8):
                y = (
                    10.0
                    + ueff
                    + 0.5 * period
                    + (5.0 if (treated and period >= 4) else 0.0)
                    + rng.normal(0, 0.5)
                )
                rows.append({"unit": unit, "period": period, "treated": treated, "outcome": y})
        d = pd.DataFrame(rows)
        d["x1"] = rng.normal(size=len(d))
        r = SyntheticDiD().fit(
            d,
            "outcome",
            "treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
            covariates=["x1"],
        )
        assert np.isfinite(r.att)


class TestAbsorbedRegressorSnap:
    """FE-spanned regressors snap to zero -> deterministic NaN + warning
    (REGISTRY 'Absorbed Fixed Effects'; the pre-snap junk columns perturbed
    identified coefficients at ~1e-5, tolerance- and implementation-dependent).
    """

    @staticmethod
    def _geo_frame(seed=0, n_states=20, n_months=12, n_rows=4_000):
        rng = np.random.default_rng(seed)
        state = rng.integers(0, n_states, n_rows)
        month = rng.integers(0, n_months, n_rows)
        treated = (state < n_states // 2).astype(int)
        post = (month >= n_months // 2).astype(int)
        y = (
            rng.normal(0, 1, n_states)[state]
            + rng.normal(0, 0.5, n_months)[month]
            + 0.3 * treated * post
            + rng.normal(size=n_rows)
        )
        return pd.DataFrame(
            {"y": y, "treated": treated, "post": post, "state": state, "month": month}
        )

    def test_did_absorb_spanned_main_effects_nan_with_cause_warning(self):
        df = self._geo_frame()
        est = DifferenceInDifferences()
        with pytest.warns(UserWarning, match="collinear with the absorbed"):
            res = est.fit(
                df,
                outcome="y",
                treatment="treated",
                post="post",
                absorb=["state", "month"],
            )
        assert np.isnan(res.coefficients["treated"])
        assert np.isnan(res.coefficients["post"])
        assert np.isfinite(res.att) and np.isfinite(res.se)

    def test_did_absorb_att_matches_clean_fwl_ground_truth(self):
        """With the junk columns snapped, ATT must equal the regression on the
        demeaned interaction ALONE (the identified FWL estimand) at 1e-10."""
        from diff_diff.utils import demean_by_groups

        df = self._geo_frame(seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DifferenceInDifferences().fit(
                df,
                outcome="y",
                treatment="treated",
                post="post",
                absorb=["state", "month"],
            )
        d = df.copy()
        d["_tt"] = d["treated"].values.astype(float) * d["post"].values.astype(float)
        out, _ = demean_by_groups(d, ["y", "_tt"], ["state", "month"], suffix="_dm", tol=1e-12)
        tt = out["_tt_dm"].values
        gt = float(np.dot(tt, out["y_dm"].values) / np.dot(tt, tt))
        assert res.att == pytest.approx(gt, abs=1e-10)

    def test_twfe_unit_constant_covariate_nan_att_unaffected(self):
        rng = np.random.default_rng(3)
        n_units, n_periods = 80, 6
        unit = np.repeat(np.arange(n_units), n_periods)
        time = np.tile(np.arange(n_periods), n_units)
        treated = (unit < n_units // 2).astype(int)
        post = (time >= n_periods // 2).astype(int)
        y = 0.4 * treated * post + rng.normal(size=unit.size)
        df = pd.DataFrame({"y": y, "treated": treated, "post": post, "unit": unit, "time": time})
        df["xc"] = np.repeat(rng.normal(size=n_units), n_periods)  # unit-constant

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = TwoWayFixedEffects().fit(
                df, outcome="y", treatment="treated", post="post", unit="unit"
            )
        with pytest.warns(UserWarning, match="collinear with the absorbed"):
            res = TwoWayFixedEffects().fit(
                df,
                outcome="y",
                treatment="treated",
                post="post",
                unit="unit",
                covariates=["xc"],
            )
        # TWFE results expose only the ATT coefficient; the snapped covariate
        # column is dropped by rank handling, so the fit must reduce EXACTLY
        # to the no-covariate design.
        assert res.att == pytest.approx(base.att, abs=1e-10)
        assert res.se == pytest.approx(base.se, rel=1e-8)


class TestReplicateRefitSnap:
    """Replicate refits must fail closed when a regressor becomes FE-spanned
    within a replicate's effective half-sample (local-review follow-up)."""

    def test_replicate_local_spanning_yields_finite_inference(self):
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(30)
        n_states, n_months, n_rows = 8, 10, 4_000
        state = rng.integers(0, n_states, n_rows)
        month = rng.integers(0, n_months, n_rows)
        treated = (state < n_states // 2).astype(int)
        post = (month >= n_months // 2).astype(int)
        # covariate varies ONLY within state 0's rows; on any replicate that
        # zeroes state 0 it becomes state-spanned in the effective sample
        xc = np.where(state == 0, rng.normal(size=n_rows), 1.0)
        y = (
            rng.normal(0, 1, n_states)[state]
            + rng.normal(0, 0.3, n_months)[month]
            + 0.3 * treated * post
            + 0.1 * xc
            + rng.normal(size=n_rows)
        )
        df = pd.DataFrame(
            {
                "y": y,
                "treated": treated,
                "post": post,
                "xc": xc,
                "state": state,
                "month": month,
                "w": np.ones(n_rows),
            }
        )
        # BRR-style replicates; replicate 1 zeroes states {0, 1} so xc is
        # constant (spanned by state FE) on its effective sample
        halves = [
            np.array([1, 0, 1, 0, 1, 0, 1, 0]),
            np.array([0, 1, 0, 1, 1, 0, 0, 1]),
            np.array([1, 1, 0, 0, 0, 1, 1, 0]),
            np.array([0, 0, 1, 1, 0, 1, 0, 1]),
        ]
        for i, h in enumerate(halves, 1):
            df[f"rw{i}"] = 2.0 * h[state]
        design = SurveyDesign(
            weights="w",
            replicate_weights=[f"rw{i}" for i in range(1, 5)],
            replicate_method="BRR",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DifferenceInDifferences().fit(
                df,
                outcome="y",
                treatment="treated",
                post="post",
                absorb=["state", "month"],
                covariates=["xc"],
                survey_design=design,
            )
        # fail-closed contract: inference is finite (invalid replicates are
        # dropped by compute_replicate_refit_variance), never junk-inflated
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0


class TestJointSpanCovariateSnap:
    """Local-review P0 end-to-end: a covariate exactly in the joint span of
    the absorbed dimensions must be dropped (NaN + warning) and leave the
    identified ATT untouched, even when MAP truncation masks it from the
    fast-path norm test (unbalanced, correlated FE incidence)."""

    @staticmethod
    def _frame(seed=7):
        rng = np.random.default_rng(seed)
        n_units, n_periods, span = 150, 30, 9
        unit = np.repeat(np.arange(n_units), n_periods)
        time = np.tile(np.arange(n_periods), n_units)
        entry = rng.integers(0, n_periods - span, n_units)
        keep = (time >= entry[unit]) & (time < entry[unit] + span)
        unit, time = unit[keep], time[keep]
        treated = (unit < n_units // 2).astype(int)
        post = (time >= n_periods // 2).astype(int)
        y = 0.4 * treated * post + rng.normal(size=unit.size)
        a = rng.normal(0, 1, n_units)
        b = rng.normal(0, 1, n_periods)
        return pd.DataFrame(
            {
                "y": y,
                "treated": treated,
                "post": post,
                "unit": unit,
                "time": time,
                "xspan": a[unit] + b[time],
            }
        )

    def test_did_absorb_joint_span_covariate_dropped_att_stable(self):
        df = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = DifferenceInDifferences().fit(
                df,
                outcome="y",
                treatment="treated",
                post="post",
                absorb=["unit", "time"],
            )
        with pytest.warns(UserWarning, match=r"xspan.*collinear with the absorbed"):
            res = DifferenceInDifferences().fit(
                df,
                outcome="y",
                treatment="treated",
                post="post",
                absorb=["unit", "time"],
                covariates=["xspan"],
            )
        assert np.isnan(res.coefficients["xspan"])
        assert res.att == pytest.approx(base.att, abs=1e-10)

    def test_sun_abraham_joint_span_covariate_dropped_att_stable(self):
        from diff_diff import SunAbraham

        df = self._frame(seed=8)
        first = np.where(np.arange(df["unit"].nunique()) % 3 == 0, 0, 15)[
            df["unit"].values % df["unit"].nunique()
        ]
        df = df.assign(first_treat=first)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = SunAbraham().fit(
                df, outcome="y", unit="unit", time="time", first_treat="first_treat"
            )
        with pytest.warns(UserWarning, match="collinear with the absorbed"):
            res = SunAbraham().fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["xspan"],
            )
        assert res.att == pytest.approx(base.att, abs=1e-8)


# ---------------------------------------------------------------------------
# Family-wide summary(alpha=) contract (M-146 completion)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def did_alpha_fitted():
    np.random.seed(11)
    n = 40
    rows = []
    for unit in range(n):
        for post in (0, 1):
            rows.append(
                {
                    "unit": unit,
                    "post": post,
                    "treated": int(unit < n // 2),
                    "outcome": np.random.normal() + 2.0 * (unit < n // 2) * post,
                }
            )
    data = pd.DataFrame(rows)
    return DifferenceInDifferences().fit(data, outcome="outcome", treatment="treated", post="post")


@pytest.fixture(scope="module")
def sdid_alpha_fitted(ci_params):
    np.random.seed(13)
    n, t = 30, 8
    rows = []
    for unit in range(n):
        ue = np.random.normal(0, 3)
        for period in range(t):
            rows.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(unit < 5),
                    "outcome": 10.0
                    + ue
                    + 0.5 * period
                    + 5.0 * (unit < 5) * (period >= 4)
                    + np.random.normal(0, 0.5),
                }
            )
    data = pd.DataFrame(rows)
    n_boot = ci_params.bootstrap(20)
    return SyntheticDiD(n_bootstrap=n_boot, seed=42).fit(
        data,
        outcome="outcome",
        treatment="treated",
        unit="unit",
        time="period",
        post_periods=[4, 5, 6, 7],
    )


class TestSummaryAlphaContractNonStaggered:
    """summary(alpha=...) never recomputes stored inference (M-146 family-wide).

    A non-fit alpha raises instead of silently relabeling the
    confidence-interval header over fit-time stored intervals; alpha=0.0
    (previously swallowed by the falsy `alpha or self.alpha` idiom) raises
    too. The four generic-message sites must emit the accurate shared
    message, never the staggered default's bootstrap-percentile rationale.
    """

    @pytest.mark.parametrize("bad_alpha", [0.10, 0.0])
    @pytest.mark.parametrize("fixture", ["did_alpha_fitted", "sdid_alpha_fitted"])
    def test_summary_rejects_non_fit_alpha(self, request, fixture, bad_alpha):
        fitted = request.getfixturevalue(fixture)
        with pytest.raises(ValueError, match="never recomputes") as exc:
            fitted.summary(alpha=bad_alpha)
        msg = str(exc.value)
        assert "re-fit with the desired alpha" in msg
        assert "bootstrap percentile" not in msg

    @pytest.mark.parametrize("fixture", ["did_alpha_fitted", "sdid_alpha_fitted"])
    def test_summary_accepts_fit_alpha(self, request, fixture):
        fitted = request.getfixturevalue(fixture)
        assert fitted.summary(alpha=fitted.alpha) == fitted.summary()

    def test_print_summary_relays_the_guard(self, did_alpha_fitted):
        with pytest.raises(ValueError, match="never recomputes"):
            did_alpha_fitted.print_summary(alpha=0.10)


class TestRequireFitAlphaMessageOverride:
    """The helper's message= override replaces the whole message; the default
    stays byte-identical (staggered pins depend on it)."""

    def test_default_message_unchanged(self):
        from diff_diff.results_base import _require_fit_alpha

        with pytest.raises(ValueError) as exc:
            _require_fit_alpha(0.10, 0.05)
        assert str(exc.value) == (
            "This result stores intervals computed at alpha=0.05; "
            "summary() never recomputes or relabels stored inference "
            "(requested alpha=0.1). Re-fit with the desired alpha "
            "(bootstrap percentile intervals cannot be reconstructed from "
            "the reported SE)."
        )

    def test_message_override_replaces_and_formats(self):
        from diff_diff.results_base import _require_fit_alpha

        with pytest.raises(ValueError) as exc:
            _require_fit_alpha(
                0.10, 0.05, message="custom never recomputes a={alpha} f={fit_alpha}"
            )
        assert str(exc.value) == "custom never recomputes a=0.1 f=0.05"

    def test_none_and_fit_alpha_pass(self):
        from diff_diff.results_base import _require_fit_alpha

        assert _require_fit_alpha(None, 0.05, message="x never recomputes") == 0.05
        assert _require_fit_alpha(0.05, 0.05) == 0.05
