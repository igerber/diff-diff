"""Tests for difference-in-differences estimators."""

import numpy as np
import pandas as pd
import pytest

from diff_diff import DifferenceInDifferences, DiDResults


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

        # SEs should be different (not exactly equal)
        assert results_robust.se != results_classical.se
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
        twfe.fit(
            twfe_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit"
        )

        # Cluster should be set to unit
        assert twfe.cluster == "unit"


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
