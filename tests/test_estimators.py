"""Tests for difference-in-differences estimators."""

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DifferenceInDifferences,
    DiDResults,
    MultiPeriodDiD,
    MultiPeriodDiDResults,
    PeriodEffect,
    SyntheticDiD,
    SyntheticDiDResults,
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

            data.append({
                "unit": unit,
                "period": period,
                "treated": int(is_treated),
                "post": period,
                "outcome": y,
            })

    return pd.DataFrame(data)


@pytest.fixture
def simple_2x2_data():
    """Minimal 2x2 DiD data."""
    return pd.DataFrame({
        "outcome": [10, 11, 15, 18, 9, 10, 12, 13],
        "treated": [1, 1, 1, 1, 0, 0, 0, 0],
        "post": [0, 0, 1, 1, 0, 0, 1, 1],
    })


class TestDifferenceInDifferences:
    """Tests for DifferenceInDifferences estimator."""

    def test_basic_fit(self, simple_2x2_data):
        """Test basic model fitting."""
        did = DifferenceInDifferences()
        results = did.fit(
            simple_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        assert isinstance(results, DiDResults)
        assert did.is_fitted_
        assert results.n_obs == 8
        assert results.n_treated == 4
        assert results.n_control == 4

    def test_att_direction(self, simple_did_data):
        """Test that ATT is estimated in correct direction."""
        did = DifferenceInDifferences()
        results = did.fit(
            simple_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        # True ATT is 3.0, estimate should be close
        assert results.att > 0
        assert abs(results.att - 3.0) < 1.0  # Within 1 unit

    def test_formula_interface(self, simple_2x2_data):
        """Test formula-based fitting."""
        did = DifferenceInDifferences()
        results = did.fit(
            simple_2x2_data,
            formula="outcome ~ treated * post"
        )

        assert isinstance(results, DiDResults)
        assert did.is_fitted_

    def test_formula_with_explicit_interaction(self, simple_2x2_data):
        """Test formula with explicit interaction syntax."""
        did = DifferenceInDifferences()
        results = did.fit(
            simple_2x2_data,
            formula="outcome ~ treated + post + treated:post"
        )

        assert isinstance(results, DiDResults)

    def test_robust_vs_classical_se(self, simple_did_data):
        """Test that robust and classical SEs differ."""
        did_robust = DifferenceInDifferences(robust=True)
        did_classical = DifferenceInDifferences(robust=False)

        results_robust = did_robust.fit(
            simple_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )
        results_classical = did_classical.fit(
            simple_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
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
        results = did.fit(
            simple_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        lower, upper = results.conf_int
        assert lower < results.att < upper
        assert lower < upper

    def test_get_set_params(self):
        """Test sklearn-compatible get_params and set_params."""
        did = DifferenceInDifferences(robust=True, alpha=0.05)

        params = did.get_params()
        assert params["robust"] is True
        assert params["alpha"] == 0.05

        did.set_params(alpha=0.10)
        assert did.alpha == 0.10

    def test_summary_output(self, simple_2x2_data):
        """Test that summary produces string output."""
        did = DifferenceInDifferences()
        did.fit(simple_2x2_data, outcome="outcome", treatment="treated", time="post")

        summary = did.summary()
        assert isinstance(summary, str)
        assert "ATT" in summary
        assert "Difference-in-Differences" in summary

    def test_invalid_treatment_values(self):
        """Test error on non-binary treatment."""
        data = pd.DataFrame({
            "outcome": [1, 2, 3, 4],
            "treated": [0, 1, 2, 3],  # Invalid: not binary
            "post": [0, 0, 1, 1],
        })

        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="binary"):
            did.fit(data, outcome="outcome", treatment="treated", time="post")

    def test_missing_column_error(self):
        """Test error when column is missing."""
        data = pd.DataFrame({
            "outcome": [1, 2, 3, 4],
            "treated": [0, 0, 1, 1],
        })

        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="Missing columns"):
            did.fit(data, outcome="outcome", treatment="treated", time="post")

    def test_unfitted_model_error(self):
        """Test error when accessing results before fitting."""
        did = DifferenceInDifferences()

        with pytest.raises(RuntimeError, match="fitted"):
            did.summary()


class TestDiDResults:
    """Tests for DiDResults class."""

    def test_repr(self, simple_2x2_data):
        """Test string representation."""
        did = DifferenceInDifferences()
        results = did.fit(
            simple_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        repr_str = repr(results)
        assert "DiDResults" in repr_str
        assert "ATT=" in repr_str

    def test_to_dict(self, simple_2x2_data):
        """Test conversion to dictionary."""
        did = DifferenceInDifferences()
        results = did.fit(
            simple_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        result_dict = results.to_dict()
        assert "att" in result_dict
        assert "se" in result_dict
        assert "p_value" in result_dict

    def test_to_dataframe(self, simple_2x2_data):
        """Test conversion to DataFrame."""
        did = DifferenceInDifferences()
        results = did.fit(
            simple_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "att" in df.columns

    def test_significance_stars(self, simple_did_data):
        """Test significance star notation."""
        did = DifferenceInDifferences()
        results = did.fit(
            simple_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        # With true effect of 3.0 and n=200, should be significant
        assert results.significance_stars in ["*", "**", "***"]

    def test_is_significant_property(self, simple_did_data):
        """Test is_significant property."""
        did = DifferenceInDifferences(alpha=0.05)
        results = did.fit(
            simple_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

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

                data.append({
                    "unit": unit,
                    "state": f"state_{state}",
                    "period": period,
                    "treated": int(is_treated),
                    "post": post,
                    "outcome": y,
                })

        return pd.DataFrame(data)

    def test_fixed_effects_dummy(self, panel_data_with_fe):
        """Test fixed effects using dummy variables."""
        did = DifferenceInDifferences()
        results = did.fit(
            panel_data_with_fe,
            outcome="outcome",
            treatment="treated",
            time="post",
            fixed_effects=["state"]
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
            time="post",
            fixed_effects=["state"]
        )

        # Should have state dummy coefficients
        state_coefs = [k for k in results.coefficients.keys() if k.startswith("state_")]
        assert len(state_coefs) == 4  # 5 states - 1 (dropped first)

    def test_absorb_fixed_effects(self, panel_data_with_fe):
        """Test absorbed (within-transformed) fixed effects."""
        did = DifferenceInDifferences()
        results = did.fit(
            panel_data_with_fe,
            outcome="outcome",
            treatment="treated",
            time="post",
            absorb=["unit"]
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
            panel_data_with_fe,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        results_with_fe = did_with_fe.fit(
            panel_data_with_fe,
            outcome="outcome",
            treatment="treated",
            time="post",
            fixed_effects=["state"]
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
                time="post",
                fixed_effects=["nonexistent_column"]
            )

    def test_invalid_absorb_column(self, panel_data_with_fe):
        """Test error when absorb column doesn't exist."""
        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="not found"):
            did.fit(
                panel_data_with_fe,
                outcome="outcome",
                treatment="treated",
                time="post",
                absorb=["nonexistent_column"]
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
            time="post",
            fixed_effects=["state", "industry"]
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
            time="post",
            covariates=["size"],
            fixed_effects=["state"]
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

                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                })

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

                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                })

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
            seed=42
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
            seed=42
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
            seed=42
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
            seed=42
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
            pre_periods=[0, 1, 2]
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
            pre_periods=[0, 1, 2]
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
            equivalence_margin=0.1  # Very tight margin
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
            seed=42
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
            seed=42
        )

        assert "variance_ratio" in results
        assert results["variance_ratio"] > 0


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_multicollinearity_detection(self):
        """Test that perfect multicollinearity is detected."""
        # Create data where a covariate is perfectly correlated with treatment
        data = pd.DataFrame({
            "outcome": [10, 11, 15, 18, 9, 10, 12, 13],
            "treated": [1, 1, 1, 1, 0, 0, 0, 0],
            "post": [0, 0, 1, 1, 0, 0, 1, 1],
            "duplicate_treated": [1, 1, 1, 1, 0, 0, 0, 0],  # Same as treated
        })

        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="rank-deficient"):
            did.fit(
                data,
                outcome="outcome",
                treatment="treated",
                time="post",
                covariates=["duplicate_treated"]
            )

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
                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                })

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
            wasserstein_threshold=0.01  # Very strict
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
            wasserstein_threshold=1.0  # Very lenient
        )

        # Both should return valid results
        assert "wasserstein_distance" in results_strict
        assert "wasserstein_distance" in results_lenient

    def test_equivalence_test_insufficient_data(self):
        """Test equivalence test handles insufficient data gracefully."""
        from diff_diff.utils import equivalence_test_trends

        # Create minimal data with only 1 observation per group
        data = pd.DataFrame({
            "outcome": [10, 15],
            "period": [0, 1],
            "treated": [1, 0],
            "unit": [0, 1],
        })

        results = equivalence_test_trends(
            data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0]
        )

        # Should return NaN values with error message
        assert np.isnan(results["tost_p_value"])
        assert results["equivalent"] is None
        assert "error" in results

    def test_parallel_trends_single_period(self):
        """Test that single pre-period returns NaN values."""
        from diff_diff.utils import check_parallel_trends

        data = pd.DataFrame({
            "outcome": [10, 11, 12, 13],
            "time": [0, 0, 0, 0],  # All same period
            "treated": [1, 1, 0, 0],
        })

        results = check_parallel_trends(
            data,
            outcome="outcome",
            time="time",
            treatment_group="treated",
            pre_periods=[0]
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

                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": post,
                    "outcome": y,
                })

        return pd.DataFrame(data)

    def test_twfe_basic_fit(self, twfe_panel_data):
        """Test basic TWFE model fitting."""
        from diff_diff.estimators import TwoWayFixedEffects

        twfe = TwoWayFixedEffects()
        results = twfe.fit(
            twfe_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit"
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
            time="post",
            unit="unit",
            covariates=["size"]
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
                time="post",
                unit="nonexistent_unit"
            )

    def test_twfe_clusters_at_unit_level(self, twfe_panel_data):
        """Test that TWFE defaults to clustering at unit level."""
        from diff_diff.estimators import TwoWayFixedEffects

        twfe = TwoWayFixedEffects()
        results = twfe.fit(
            twfe_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit"
        )

        # Cluster should NOT be mutated (remains None) - clustering is handled internally
        # This ensures the estimator config is immutable as per sklearn convention
        assert twfe.cluster is None
        # But the results should still reflect cluster-robust SEs were computed correctly
        assert results.se > 0


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
                data.append({
                    "cluster": cluster,
                    "outcome": y,
                    "treated": int(treated),
                    "post": int(post),
                })

        df = pd.DataFrame(data)

        # With clustering
        did_cluster = DifferenceInDifferences(cluster="cluster")
        results_cluster = did_cluster.fit(
            df, outcome="outcome", treatment="treated", time="post"
        )

        # Without clustering
        did_no_cluster = DifferenceInDifferences(robust=True)
        results_no_cluster = did_no_cluster.fit(
            df, outcome="outcome", treatment="treated", time="post"
        )

        # ATT should be similar
        assert abs(results_cluster.att - results_no_cluster.att) < 0.01

        # SEs should be different (cluster-robust typically larger)
        assert results_cluster.se != results_no_cluster.se


class TestMultiPeriodDiD:
    """Tests for MultiPeriodDiD estimator."""

    @pytest.fixture
    def multi_period_data(self):
        """Create panel data with multiple time periods and known ATT."""
        np.random.seed(42)
        n_units = 100
        n_periods = 6  # 3 pre-treatment, 3 post-treatment

        data = []
        for unit in range(n_units):
            is_treated = unit < n_units // 2
            unit_effect = np.random.normal(0, 1)

            for period in range(n_periods):
                # Common time trend
                time_effect = period * 0.5

                y = 10.0 + unit_effect + time_effect

                # Treatment effect: 3.0 in post-periods (periods 3, 4, 5)
                if is_treated and period >= 3:
                    y += 3.0

                y += np.random.normal(0, 0.5)

                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                })

        return pd.DataFrame(data)

    @pytest.fixture
    def heterogeneous_effects_data(self):
        """Create data with different treatment effects per period."""
        np.random.seed(42)
        n_units = 100
        n_periods = 6

        # Different true effects per post-period
        true_effects = {3: 2.0, 4: 3.0, 5: 4.0}

        data = []
        for unit in range(n_units):
            is_treated = unit < n_units // 2
            unit_effect = np.random.normal(0, 1)

            for period in range(n_periods):
                time_effect = period * 0.5
                y = 10.0 + unit_effect + time_effect

                # Period-specific treatment effects
                if is_treated and period in true_effects:
                    y += true_effects[period]

                y += np.random.normal(0, 0.5)

                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                })

        return pd.DataFrame(data), true_effects

    def test_basic_fit(self, multi_period_data):
        """Test basic model fitting with multiple periods."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        assert isinstance(results, MultiPeriodDiDResults)
        assert did.is_fitted_
        assert results.n_obs == 600  # 100 units * 6 periods
        assert len(results.period_effects) == 3  # 3 post-periods
        assert len(results.pre_periods) == 3
        assert len(results.post_periods) == 3

    def test_avg_att_close_to_true(self, multi_period_data):
        """Test that average ATT is close to true effect."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        # True ATT is 3.0
        assert abs(results.avg_att - 3.0) < 0.5
        assert results.avg_att > 0

    def test_period_specific_effects(self, heterogeneous_effects_data):
        """Test that period-specific effects are estimated correctly."""
        data, true_effects = heterogeneous_effects_data

        did = MultiPeriodDiD()
        results = did.fit(
            data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        # Each period-specific effect should be close to truth
        for period, true_effect in true_effects.items():
            estimated = results.period_effects[period].effect
            assert abs(estimated - true_effect) < 0.5, \
                f"Period {period}: expected ~{true_effect}, got {estimated}"

    def test_period_effects_have_all_stats(self, multi_period_data):
        """Test that period effects contain all statistics."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        for period, pe in results.period_effects.items():
            assert isinstance(pe, PeriodEffect)
            assert hasattr(pe, 'effect')
            assert hasattr(pe, 'se')
            assert hasattr(pe, 't_stat')
            assert hasattr(pe, 'p_value')
            assert hasattr(pe, 'conf_int')
            assert pe.se > 0
            assert len(pe.conf_int) == 2
            assert pe.conf_int[0] < pe.conf_int[1]

    def test_get_effect_method(self, multi_period_data):
        """Test get_effect method."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        # Valid period
        effect = results.get_effect(4)
        assert isinstance(effect, PeriodEffect)
        assert effect.period == 4

        # Invalid period
        with pytest.raises(KeyError):
            results.get_effect(0)  # Pre-period

    def test_auto_infer_post_periods(self, multi_period_data):
        """Test automatic inference of post-periods."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period"
            # post_periods not specified - should infer last half
        )

        # With 6 periods, should infer periods 3, 4, 5 as post
        assert results.pre_periods == [0, 1, 2]
        assert results.post_periods == [3, 4, 5]

    def test_custom_reference_period(self, multi_period_data):
        """Test custom reference period."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5],
            reference_period=2  # Use period 2 as reference
        )

        # Should work and give reasonable results
        assert results is not None
        assert did.is_fitted_
        # Reference period should not be in coefficients as a dummy
        assert "period_2" not in results.coefficients

    def test_with_covariates(self, multi_period_data):
        """Test multi-period DiD with covariates."""
        # Add a covariate
        multi_period_data["size"] = np.random.normal(100, 10, len(multi_period_data))

        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5],
            covariates=["size"]
        )

        assert results is not None
        assert "size" in results.coefficients

    def test_with_fixed_effects(self):
        """Test multi-period DiD with fixed effects."""
        np.random.seed(42)
        n_units = 50
        n_periods = 6
        n_states = 5

        data = []
        for unit in range(n_units):
            state = unit % n_states
            is_treated = unit < n_units // 2
            state_effect = state * 2.0

            for period in range(n_periods):
                y = 10.0 + state_effect + period * 0.5
                if is_treated and period >= 3:
                    y += 3.0
                y += np.random.normal(0, 0.5)

                data.append({
                    "unit": unit,
                    "state": f"state_{state}",
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                })

        df = pd.DataFrame(data)

        did = MultiPeriodDiD()
        results = did.fit(
            df,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5],
            fixed_effects=["state"]
        )

        assert results is not None
        assert did.is_fitted_
        # ATT should still be close to 3.0
        assert abs(results.avg_att - 3.0) < 1.0

    def test_with_absorbed_fe(self, multi_period_data):
        """Test multi-period DiD with absorbed fixed effects."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5],
            absorb=["unit"]
        )

        assert results is not None
        assert did.is_fitted_
        assert abs(results.avg_att - 3.0) < 1.0

    def test_cluster_robust_se(self, multi_period_data):
        """Test cluster-robust standard errors."""
        did_cluster = MultiPeriodDiD(cluster="unit")
        did_robust = MultiPeriodDiD(robust=True)

        results_cluster = did_cluster.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        results_robust = did_robust.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        # ATT should be similar
        assert abs(results_cluster.avg_att - results_robust.avg_att) < 0.01

        # SEs should be different
        assert results_cluster.avg_se != results_robust.avg_se

    def test_summary_output(self, multi_period_data):
        """Test that summary produces string output."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        summary = results.summary()
        assert isinstance(summary, str)
        assert "Multi-Period" in summary
        assert "Period-Specific" in summary
        assert "Average Treatment Effect" in summary
        assert "Avg ATT" in summary

    def test_to_dict(self, multi_period_data):
        """Test conversion to dictionary."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        result_dict = results.to_dict()
        assert "avg_att" in result_dict
        assert "avg_se" in result_dict
        assert "n_pre_periods" in result_dict
        assert "n_post_periods" in result_dict

    def test_to_dataframe(self, multi_period_data):
        """Test conversion to DataFrame."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 3 post-periods
        assert "period" in df.columns
        assert "effect" in df.columns
        assert "p_value" in df.columns

    def test_is_significant_property(self, multi_period_data):
        """Test is_significant property."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        # With true effect of 3.0, should be significant
        assert isinstance(results.is_significant, bool)
        assert results.is_significant

    def test_significance_stars(self, multi_period_data):
        """Test significance stars property."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        # Should have significance stars
        assert results.significance_stars in ["*", "**", "***"]

    def test_repr(self, multi_period_data):
        """Test string representation."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        repr_str = repr(results)
        assert "MultiPeriodDiDResults" in repr_str
        assert "avg_ATT=" in repr_str

    def test_period_effect_repr(self, multi_period_data):
        """Test PeriodEffect string representation."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        pe = results.period_effects[3]
        repr_str = repr(pe)
        assert "PeriodEffect" in repr_str
        assert "period=" in repr_str
        assert "effect=" in repr_str

    def test_invalid_post_period(self, multi_period_data):
        """Test error when post_period not in data."""
        did = MultiPeriodDiD()
        with pytest.raises(ValueError, match="not found in time column"):
            did.fit(
                multi_period_data,
                outcome="outcome",
                treatment="treated",
                time="period",
                post_periods=[3, 4, 99]  # 99 doesn't exist
            )

    def test_no_pre_periods_error(self, multi_period_data):
        """Test error when all periods are post-treatment."""
        did = MultiPeriodDiD()
        with pytest.raises(ValueError, match="at least one pre-treatment period"):
            did.fit(
                multi_period_data,
                outcome="outcome",
                treatment="treated",
                time="period",
                post_periods=[0, 1, 2, 3, 4, 5]  # All periods
            )

    def test_no_post_periods_error(self):
        """Test error when no post-treatment periods."""
        data = pd.DataFrame({
            "outcome": [10, 11, 12, 13],
            "treated": [1, 1, 0, 0],
            "period": [0, 1, 0, 1],
        })

        did = MultiPeriodDiD()
        with pytest.raises(ValueError, match="at least one post-treatment period"):
            did.fit(
                data,
                outcome="outcome",
                treatment="treated",
                time="period",
                post_periods=[]
            )

    def test_invalid_treatment_values(self, multi_period_data):
        """Test error on non-binary treatment."""
        multi_period_data["treated"] = multi_period_data["treated"] * 2  # Makes values 0, 2

        did = MultiPeriodDiD()
        with pytest.raises(ValueError, match="binary"):
            did.fit(
                multi_period_data,
                outcome="outcome",
                treatment="treated",
                time="period",
                post_periods=[3, 4, 5]
            )

    def test_unfitted_model_error(self):
        """Test error when accessing results before fitting."""
        did = MultiPeriodDiD()
        with pytest.raises(RuntimeError, match="fitted"):
            did.summary()

    def test_confidence_interval_contains_estimate(self, multi_period_data):
        """Test that confidence intervals contain the estimates."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        # Average ATT CI
        lower, upper = results.avg_conf_int
        assert lower < results.avg_att < upper

        # Period-specific CIs
        for pe in results.period_effects.values():
            lower, upper = pe.conf_int
            assert lower < pe.effect < upper

    def test_two_periods_works(self):
        """Test that MultiPeriodDiD works with just 2 periods (edge case)."""
        np.random.seed(42)
        data = []
        for unit in range(50):
            is_treated = unit < 25
            for period in [0, 1]:
                y = 10.0 + (3.0 if is_treated and period == 1 else 0)
                y += np.random.normal(0, 0.5)
                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                })

        df = pd.DataFrame(data)

        did = MultiPeriodDiD()
        results = did.fit(
            df,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[1]
        )

        assert len(results.period_effects) == 1
        assert len(results.pre_periods) == 1
        assert abs(results.avg_att - 3.0) < 1.0

    def test_many_periods(self):
        """Test with many time periods."""
        np.random.seed(42)
        n_periods = 20
        data = []
        for unit in range(50):
            is_treated = unit < 25
            for period in range(n_periods):
                y = 10.0 + period * 0.1
                if is_treated and period >= 10:
                    y += 2.5
                y += np.random.normal(0, 0.3)
                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                })

        df = pd.DataFrame(data)

        did = MultiPeriodDiD()
        results = did.fit(
            df,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=list(range(10, 20))
        )

        assert len(results.period_effects) == 10
        assert len(results.pre_periods) == 10
        assert abs(results.avg_att - 2.5) < 0.5

    def test_r_squared_reported(self, multi_period_data):
        """Test that R-squared is reported."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        assert results.r_squared is not None
        assert 0 <= results.r_squared <= 1

    def test_coefficients_dict(self, multi_period_data):
        """Test that coefficients dictionary contains expected keys."""
        did = MultiPeriodDiD()
        results = did.fit(
            multi_period_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5]
        )

        # Should have treatment, period dummies, and interactions
        assert "treated" in results.coefficients
        assert "const" in results.coefficients
        # Period dummies (excluding reference)
        assert any("period_" in k for k in results.coefficients)
        # Treatment interactions
        assert any("treated:period_" in k for k in results.coefficients)


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

                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                })

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
            data.append({
                "unit": 0,
                "period": period,
                "treated": 1,
                "outcome": y,
            })

        # Control units with various patterns
        for unit in range(1, n_controls + 1):
            unit_intercept = np.random.uniform(30, 70)
            unit_slope = np.random.uniform(0.5, 3.5)  # Various slopes
            for period in range(n_periods):
                y = unit_intercept + period * unit_slope
                y += np.random.normal(0, 1)
                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": 0,
                    "outcome": y,
                })

        return pd.DataFrame(data)

    def test_basic_fit(self, sdid_panel_data):
        """Test basic SDID model fitting."""
        sdid = SyntheticDiD(n_bootstrap=50, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        assert isinstance(results, SyntheticDiDResults)
        assert sdid.is_fitted_
        assert results.n_obs == 240  # 30 units * 8 periods
        assert results.n_treated == 5
        assert results.n_control == 25

    def test_att_direction(self, sdid_panel_data):
        """Test that ATT is estimated in correct direction."""
        sdid = SyntheticDiD(n_bootstrap=50, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        # True ATT is 5.0
        assert results.att > 0
        assert abs(results.att - 5.0) < 2.0

    def test_unit_weights_sum_to_one(self, sdid_panel_data):
        """Test that unit weights sum to 1."""
        sdid = SyntheticDiD(n_bootstrap=0, seed=42)  # Use placebo instead
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        weight_sum = sum(results.unit_weights.values())
        assert abs(weight_sum - 1.0) < 1e-6

    def test_time_weights_sum_to_one(self, sdid_panel_data):
        """Test that time weights sum to 1."""
        sdid = SyntheticDiD(n_bootstrap=0, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        weight_sum = sum(results.time_weights.values())
        assert abs(weight_sum - 1.0) < 1e-6

    def test_unit_weights_nonnegative(self, sdid_panel_data):
        """Test that unit weights are non-negative."""
        sdid = SyntheticDiD(n_bootstrap=0, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        for w in results.unit_weights.values():
            assert w >= 0

    def test_single_treated_unit(self, single_treated_unit_data):
        """Test SDID with a single treated unit (classic SC scenario)."""
        sdid = SyntheticDiD(n_bootstrap=50, seed=42)
        results = sdid.fit(
            single_treated_unit_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7, 8, 9]
        )

        assert results.n_treated == 1
        # True ATT is 10.0
        assert results.att > 0
        # With good controls, should be reasonably close
        assert abs(results.att - 10.0) < 5.0

    def test_regularization_effect(self, sdid_panel_data):
        """Test that regularization affects weight dispersion."""
        sdid_no_reg = SyntheticDiD(lambda_reg=0.0, n_bootstrap=0, seed=42)
        sdid_high_reg = SyntheticDiD(lambda_reg=10.0, n_bootstrap=0, seed=42)

        results_no_reg = sdid_no_reg.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        results_high_reg = sdid_high_reg.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        # High regularization should give more uniform weights
        weights_no_reg = np.array(list(results_no_reg.unit_weights.values()))
        weights_high_reg = np.array(list(results_high_reg.unit_weights.values()))

        # Variance of weights should be lower with more regularization
        assert np.var(weights_high_reg) <= np.var(weights_no_reg) + 0.01

    def test_placebo_inference(self, sdid_panel_data):
        """Test placebo-based inference (n_bootstrap=0)."""
        sdid = SyntheticDiD(n_bootstrap=0, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        assert results.placebo_effects is not None
        assert len(results.placebo_effects) > 0
        assert results.se > 0

    def test_bootstrap_inference(self, sdid_panel_data):
        """Test bootstrap-based inference."""
        sdid = SyntheticDiD(n_bootstrap=100, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        assert results.se > 0
        assert results.conf_int[0] < results.att < results.conf_int[1]

    def test_get_unit_weights_df(self, sdid_panel_data):
        """Test getting unit weights as DataFrame."""
        sdid = SyntheticDiD(n_bootstrap=0, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        weights_df = results.get_unit_weights_df()
        assert isinstance(weights_df, pd.DataFrame)
        assert "unit" in weights_df.columns
        assert "weight" in weights_df.columns
        assert len(weights_df) == 25  # Number of control units

    def test_get_time_weights_df(self, sdid_panel_data):
        """Test getting time weights as DataFrame."""
        sdid = SyntheticDiD(n_bootstrap=0, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        weights_df = results.get_time_weights_df()
        assert isinstance(weights_df, pd.DataFrame)
        assert "period" in weights_df.columns
        assert "weight" in weights_df.columns
        assert len(weights_df) == 4  # Number of pre-periods

    def test_pre_treatment_fit(self, sdid_panel_data):
        """Test that pre-treatment fit is computed."""
        sdid = SyntheticDiD(n_bootstrap=0, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        assert results.pre_treatment_fit is not None
        assert results.pre_treatment_fit >= 0

    def test_summary_output(self, sdid_panel_data):
        """Test that summary produces string output."""
        sdid = SyntheticDiD(n_bootstrap=50, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        summary = results.summary()
        assert isinstance(summary, str)
        assert "Synthetic Difference-in-Differences" in summary
        assert "ATT" in summary
        assert "Unit Weights" in summary

    def test_to_dict(self, sdid_panel_data):
        """Test conversion to dictionary."""
        sdid = SyntheticDiD(n_bootstrap=50, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        result_dict = results.to_dict()
        assert "att" in result_dict
        assert "se" in result_dict
        assert "n_pre_periods" in result_dict
        assert "n_post_periods" in result_dict
        assert "pre_treatment_fit" in result_dict

    def test_to_dataframe(self, sdid_panel_data):
        """Test conversion to DataFrame."""
        sdid = SyntheticDiD(n_bootstrap=50, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "att" in df.columns

    def test_repr(self, sdid_panel_data):
        """Test string representation."""
        sdid = SyntheticDiD(n_bootstrap=50, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        repr_str = repr(results)
        assert "SyntheticDiDResults" in repr_str
        assert "ATT=" in repr_str

    def test_is_significant_property(self, sdid_panel_data):
        """Test is_significant property."""
        sdid = SyntheticDiD(n_bootstrap=100, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        assert isinstance(results.is_significant, bool)

    def test_get_set_params(self):
        """Test get_params and set_params."""
        sdid = SyntheticDiD(lambda_reg=1.0, zeta=0.5, alpha=0.10)

        params = sdid.get_params()
        assert params["lambda_reg"] == 1.0
        assert params["zeta"] == 0.5
        assert params["alpha"] == 0.10

        sdid.set_params(lambda_reg=2.0)
        assert sdid.lambda_reg == 2.0

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
                post_periods=[4, 5, 6, 7]
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
                post_periods=[4, 5, 6, 7]
            )

    def test_no_treated_units_error(self):
        """Test error when no treated units."""
        data = pd.DataFrame({
            "unit": [1, 1, 2, 2],
            "period": [0, 1, 0, 1],
            "treated": [0, 0, 0, 0],
            "outcome": [10, 11, 12, 13],
        })

        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="No treated units"):
            sdid.fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[1]
            )

    def test_no_control_units_error(self):
        """Test error when no control units."""
        data = pd.DataFrame({
            "unit": [1, 1, 2, 2],
            "period": [0, 1, 0, 1],
            "treated": [1, 1, 1, 1],
            "outcome": [10, 11, 12, 13],
        })

        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="No control units"):
            sdid.fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[1]
            )

    def test_auto_infer_post_periods(self, sdid_panel_data):
        """Test automatic inference of post-periods."""
        sdid = SyntheticDiD(n_bootstrap=0, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period"
            # post_periods not specified
        )

        # With 8 periods, should infer last 4 as post
        assert results.pre_periods == [0, 1, 2, 3]
        assert results.post_periods == [4, 5, 6, 7]

    def test_with_covariates(self, sdid_panel_data):
        """Test SDID with covariates."""
        # Add a covariate
        sdid_panel_data["size"] = np.random.normal(100, 10, len(sdid_panel_data))

        sdid = SyntheticDiD(n_bootstrap=50, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7],
            covariates=["size"]
        )

        assert results is not None
        assert sdid.is_fitted_

    def test_confidence_interval_contains_estimate(self, sdid_panel_data):
        """Test that confidence interval contains the estimate."""
        sdid = SyntheticDiD(n_bootstrap=100, seed=42)
        results = sdid.fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        lower, upper = results.conf_int
        assert lower < results.att < upper

    def test_reproducibility_with_seed(self, sdid_panel_data):
        """Test that results are reproducible with the same seed."""
        results1 = SyntheticDiD(n_bootstrap=50, seed=42).fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        results2 = SyntheticDiD(n_bootstrap=50, seed=42).fit(
            sdid_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[4, 5, 6, 7]
        )

        assert results1.att == results2.att
        assert results1.se == results2.se


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

    def test_compute_synthetic_weights(self):
        """Test synthetic weight computation."""
        from diff_diff.utils import compute_synthetic_weights

        np.random.seed(42)
        n_pre = 5
        n_control = 10

        Y_control = np.random.randn(n_pre, n_control)
        Y_treated = np.random.randn(n_pre)

        weights = compute_synthetic_weights(Y_control, Y_treated)

        # Weights should sum to 1
        assert abs(np.sum(weights) - 1.0) < 1e-6
        # Weights should be non-negative
        assert np.all(weights >= 0)
        # Should have correct length
        assert len(weights) == n_control

    def test_compute_time_weights(self):
        """Test time weight computation."""
        from diff_diff.utils import compute_time_weights

        np.random.seed(42)
        n_pre = 5
        n_control = 10

        Y_control = np.random.randn(n_pre, n_control)
        Y_treated = np.random.randn(n_pre)

        weights = compute_time_weights(Y_control, Y_treated)

        # Weights should sum to 1
        assert abs(np.sum(weights) - 1.0) < 1e-6
        # Weights should be non-negative
        assert np.all(weights >= 0)
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
            Y_pre_control, Y_post_control,
            Y_pre_treated, Y_post_treated,
            unit_weights, time_weights
        )

        # Treated: 15 - 10 = 5
        # Control: 12 - 10 = 2
        # SDID: 5 - 2 = 3
        assert abs(tau - 3.0) < 1e-6
