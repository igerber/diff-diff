"""
Tests for Triple Difference (DDD) estimator.

Tests cover:
- Basic DDD estimation without covariates
- Covariate-adjusted estimation (RA, IPW, DR)
- Edge cases and error handling
- Results object functionality
- Comparison between estimation methods
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.survey import SurveyDesign
from diff_diff.triple_diff import (
    TripleDifference,
    TripleDifferenceResults,
    triple_difference,
)

# Note: The library exports generate_ddd_data in diff_diff.prep, but tests use
# a local implementation with test-specific parameter names and covariate handling.


# =============================================================================
# Fixtures for test data generation
# =============================================================================


def generate_ddd_data(
    n_per_cell: int = 100,
    true_att: float = 2.0,
    noise_sd: float = 1.0,
    seed: int = 42,
    add_covariates: bool = False,
    covariate_effect: float = 0.5,
) -> pd.DataFrame:
    """
    Generate synthetic DDD data with known treatment effect.

    This is a test-specific implementation that maintains backward compatibility
    with existing tests. For general use, prefer diff_diff.prep.generate_ddd_data.
    """
    rng = np.random.default_rng(seed)

    rows = []
    for g in [0, 1]:  # group
        for p in [0, 1]:  # partition
            for t in [0, 1]:  # time
                for _ in range(n_per_cell):
                    # Base outcome depends on cell
                    y = 10 + 2 * g + 1 * p + 0.5 * t

                    # Add second-order interactions (non-treatment)
                    y += 0.3 * g * p  # group-partition interaction
                    y += 0.2 * g * t  # group-time interaction
                    y += 0.1 * p * t  # partition-time interaction

                    # Treatment effect: only for G=1, P=1, T=1
                    if g == 1 and p == 1 and t == 1:
                        y += true_att

                    # Add covariates if requested
                    if add_covariates:
                        x1 = rng.normal(0, 1)
                        x2 = rng.choice([0, 1])
                        y += covariate_effect * x1 + 0.3 * x2
                    else:
                        x1 = rng.normal(0, 1)
                        x2 = rng.choice([0, 1])

                    # Add noise
                    y += rng.normal(0, noise_sd)

                    rows.append(
                        {
                            "outcome": y,
                            "group": g,
                            "partition": p,
                            "time": t,
                            "x1": x1,
                            "x2": x2,
                            "unit_id": len(rows),
                        }
                    )

    return pd.DataFrame(rows)


@pytest.fixture
def simple_ddd_data():
    """Simple DDD data without covariates affecting outcome."""
    return generate_ddd_data(n_per_cell=100, true_att=2.0, seed=42)


@pytest.fixture
def ddd_data_with_covariates():
    """DDD data where covariates affect outcome."""
    return generate_ddd_data(
        n_per_cell=100,
        true_att=2.0,
        seed=42,
        add_covariates=True,
        covariate_effect=0.5,
    )


@pytest.fixture
def small_ddd_data():
    """Small DDD dataset for edge case testing."""
    return generate_ddd_data(n_per_cell=10, true_att=2.0, seed=42)


# =============================================================================
# Basic Tests
# =============================================================================


class TestTripleDifferenceBasic:
    """Basic tests for TripleDifference estimator."""

    def test_init_default_params(self):
        """Test default parameter initialization."""
        ddd = TripleDifference()
        assert ddd.estimation_method == "dr"
        assert ddd.robust is True
        assert ddd.cluster is None
        assert ddd.alpha == 0.05
        assert ddd.pscore_trim == 0.01
        assert ddd.is_fitted_ is False

    def test_init_custom_params(self):
        """Test custom parameter initialization."""
        ddd = TripleDifference(
            estimation_method="reg",
            robust=False,
            alpha=0.10,
            pscore_trim=0.05,
        )
        assert ddd.estimation_method == "reg"
        assert ddd.robust is False
        assert ddd.alpha == 0.10
        assert ddd.pscore_trim == 0.05

    def test_init_invalid_method(self):
        """Test that invalid estimation method raises error."""
        with pytest.raises(ValueError, match="estimation_method must be"):
            TripleDifference(estimation_method="invalid")

    def test_fit_basic(self, simple_ddd_data):
        """Test basic fitting with default settings."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert ddd.is_fitted_ is True
        assert isinstance(results, TripleDifferenceResults)
        assert results.n_obs == len(simple_ddd_data)

    def test_fit_returns_results(self, simple_ddd_data):
        """Test that fit returns results object."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # Check results attributes
        assert hasattr(results, "att")
        assert hasattr(results, "se")
        assert hasattr(results, "t_stat")
        assert hasattr(results, "p_value")
        assert hasattr(results, "conf_int")

    def test_att_estimate_reasonable(self, simple_ddd_data):
        """Test that ATT estimate is close to true value."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # True ATT is 2.0, should be within reasonable range
        assert abs(results.att - 2.0) < 0.5

    def test_standard_error_positive(self, simple_ddd_data):
        """Test that standard error is positive."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert results.se > 0

    def test_confidence_interval_contains_att(self, simple_ddd_data):
        """Test that confidence interval is properly ordered."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert results.conf_int[0] < results.conf_int[1]
        assert results.conf_int[0] < results.att < results.conf_int[1]


# =============================================================================
# Estimation Method Tests
# =============================================================================


class TestEstimationMethods:
    """Test different estimation methods."""

    def test_regression_adjustment(self, simple_ddd_data):
        """Test regression adjustment estimation."""
        ddd = TripleDifference(estimation_method="reg")
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert results.estimation_method == "reg"
        # r_squared is only computed when covariates are present
        # (the decomposition approach doesn't use a single OLS)
        assert abs(results.att - 2.0) < 0.5

    def test_ipw_estimation(self, simple_ddd_data):
        """Test inverse probability weighting estimation."""
        ddd = TripleDifference(estimation_method="ipw")
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert results.estimation_method == "ipw"
        assert abs(results.att - 2.0) < 0.5

    def test_doubly_robust_estimation(self, simple_ddd_data):
        """Test doubly robust estimation."""
        ddd = TripleDifference(estimation_method="dr")
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert results.estimation_method == "dr"
        assert abs(results.att - 2.0) < 0.5

    def test_methods_give_similar_results_no_covariates(self, simple_ddd_data):
        """Test that methods give similar results without covariates."""
        results_reg = TripleDifference(estimation_method="reg").fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        results_ipw = TripleDifference(estimation_method="ipw").fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        results_dr = TripleDifference(estimation_method="dr").fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # All methods should give similar point estimates
        assert abs(results_reg.att - results_ipw.att) < 0.3
        assert abs(results_reg.att - results_dr.att) < 0.3
        assert abs(results_ipw.att - results_dr.att) < 0.3


# =============================================================================
# Covariate Tests
# =============================================================================


class TestCovariates:
    """Test covariate adjustment functionality."""

    def test_with_single_covariate(self, ddd_data_with_covariates):
        """Test estimation with a single covariate."""
        ddd = TripleDifference(estimation_method="dr")
        results = ddd.fit(
            ddd_data_with_covariates,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            covariates=["x1"],
        )

        assert results is not None
        # Tolerance is wider with covariates due to estimation uncertainty
        assert abs(results.att - 2.0) < 1.0

    def test_with_multiple_covariates(self, ddd_data_with_covariates):
        """Test estimation with multiple covariates."""
        ddd = TripleDifference(estimation_method="dr")
        results = ddd.fit(
            ddd_data_with_covariates,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            covariates=["x1", "x2"],
        )

        assert results is not None
        # Tolerance is wider with covariates due to estimation uncertainty
        assert abs(results.att - 2.0) < 1.0

    def test_covariates_improve_precision(self, ddd_data_with_covariates):
        """Test that covariates can improve precision."""
        # Without covariates
        results_no_cov = TripleDifference(estimation_method="reg").fit(
            ddd_data_with_covariates,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # With covariates
        results_with_cov = TripleDifference(estimation_method="reg").fit(
            ddd_data_with_covariates,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            covariates=["x1", "x2"],
        )

        # Covariates should improve precision (lower SE)
        assert results_with_cov.se <= results_no_cov.se

    def test_ipw_with_covariates_has_pscore_stats(self, ddd_data_with_covariates):
        """Test that IPW with covariates provides propensity score stats."""
        ddd = TripleDifference(estimation_method="ipw")
        results = ddd.fit(
            ddd_data_with_covariates,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            covariates=["x1", "x2"],
        )

        assert results.pscore_stats is not None
        assert "P(subgroup=4|X) mean" in results.pscore_stats


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Test input validation and error handling."""

    def test_missing_outcome_column(self, simple_ddd_data):
        """Test error when outcome column is missing."""
        ddd = TripleDifference()
        with pytest.raises(ValueError, match="Missing columns"):
            ddd.fit(
                simple_ddd_data,
                outcome="nonexistent",
                group="group",
                partition="partition",
                time="time",
            )

    def test_missing_cluster_column(self, simple_ddd_data):
        """Test error when cluster column is missing from data."""
        ddd = TripleDifference(cluster="nonexistent")
        with pytest.raises(ValueError, match="Missing columns"):
            ddd.fit(
                simple_ddd_data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )

    def test_missing_group_column(self, simple_ddd_data):
        """Test error when group column is missing."""
        ddd = TripleDifference()
        with pytest.raises(ValueError, match="Missing columns"):
            ddd.fit(
                simple_ddd_data,
                outcome="outcome",
                group="nonexistent",
                partition="partition",
                time="time",
            )

    def test_non_binary_group(self, simple_ddd_data):
        """Test error when group is not binary."""
        data = simple_ddd_data.copy()
        data["group"] = data["group"] + 1  # Now 1 and 2

        ddd = TripleDifference()
        with pytest.raises(ValueError, match="must be binary"):
            ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )

    def test_non_binary_partition(self, simple_ddd_data):
        """Test error when partition is not binary."""
        data = simple_ddd_data.copy()
        data["partition"] = data["partition"] * 2  # Now 0 and 2

        ddd = TripleDifference()
        with pytest.raises(ValueError, match="must be binary"):
            ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )

    def test_missing_cell(self, simple_ddd_data):
        """Test error when a cell has no observations."""
        # Remove all observations from one cell
        data = simple_ddd_data[
            ~(
                (simple_ddd_data["group"] == 1)
                & (simple_ddd_data["partition"] == 1)
                & (simple_ddd_data["time"] == 0)
            )
        ]

        ddd = TripleDifference()
        with pytest.raises(ValueError, match="No observations in cell"):
            ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )

    def test_missing_values_in_outcome(self, simple_ddd_data):
        """Test error when outcome has missing values."""
        data = simple_ddd_data.copy()
        data.loc[0, "outcome"] = np.nan

        ddd = TripleDifference()
        with pytest.raises(ValueError, match="contains missing values"):
            ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )

    def test_non_dataframe_input(self):
        """Test error when input is not a DataFrame."""
        ddd = TripleDifference()
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            ddd.fit(
                {"outcome": [1, 2, 3]},  # dict, not DataFrame
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )


# =============================================================================
# Results Object Tests
# =============================================================================


class TestTripleDifferenceResults:
    """Test TripleDifferenceResults functionality."""

    def test_summary_output(self, simple_ddd_data):
        """Test that summary generates output."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        summary = results.summary()
        assert isinstance(summary, str)
        assert "Triple Difference" in summary
        assert "ATT" in summary

    def test_to_dict(self, simple_ddd_data):
        """Test conversion to dictionary."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        result_dict = results.to_dict()
        assert isinstance(result_dict, dict)
        assert "att" in result_dict
        assert "se" in result_dict
        assert "p_value" in result_dict

    def test_to_dataframe(self, simple_ddd_data):
        """Test conversion to DataFrame."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        result_df = results.to_dataframe()
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1
        assert "att" in result_df.columns

    def test_is_significant_property(self, simple_ddd_data):
        """Test is_significant property."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # With true ATT of 2.0 and reasonable sample size, should be significant
        assert isinstance(results.is_significant, bool)

    def test_significance_stars_property(self, simple_ddd_data):
        """Test significance_stars property."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        stars = results.significance_stars
        assert isinstance(stars, str)
        assert stars in ["***", "**", "*", ".", ""]

    def test_group_means_available(self, simple_ddd_data):
        """Test that cell means are computed."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert results.group_means is not None
        assert len(results.group_means) == 8  # 2x2x2 cells

    def test_cell_counts(self, simple_ddd_data):
        """Test that cell counts are correct."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        total = (
            results.n_treated_eligible
            + results.n_treated_ineligible
            + results.n_control_eligible
            + results.n_control_ineligible
        )
        # Each cell has n observations for pre and post periods
        assert total == results.n_obs


# =============================================================================
# sklearn Compatibility Tests
# =============================================================================


class TestSklearnCompatibility:
    """Test sklearn-like interface."""

    def test_get_params(self):
        """Test get_params method."""
        ddd = TripleDifference(estimation_method="ipw", alpha=0.10)
        params = ddd.get_params()

        assert params["estimation_method"] == "ipw"
        assert params["alpha"] == 0.10

    def test_set_params(self):
        """Test set_params method."""
        ddd = TripleDifference()
        ddd.set_params(estimation_method="reg", alpha=0.01)

        assert ddd.estimation_method == "reg"
        assert ddd.alpha == 0.01

    def test_set_params_returns_self(self):
        """Test that set_params returns self for chaining."""
        ddd = TripleDifference()
        result = ddd.set_params(alpha=0.10)

        assert result is ddd

    def test_set_invalid_param(self):
        """Test error on invalid parameter."""
        ddd = TripleDifference()
        with pytest.raises(ValueError, match="Unknown parameter"):
            ddd.set_params(invalid_param=42)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Test triple_difference convenience function."""

    def test_basic_usage(self, simple_ddd_data):
        """Test basic usage of convenience function."""
        results = triple_difference(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert isinstance(results, TripleDifferenceResults)
        assert abs(results.att - 2.0) < 0.5

    def test_with_method_specification(self, simple_ddd_data):
        """Test convenience function with method specification."""
        results = triple_difference(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            estimation_method="reg",
        )

        assert results.estimation_method == "reg"

    def test_with_covariates(self, ddd_data_with_covariates):
        """Test convenience function with covariates."""
        results = triple_difference(
            ddd_data_with_covariates,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            covariates=["x1", "x2"],
        )

        assert results is not None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_sample(self, small_ddd_data):
        """Test with small sample size."""
        ddd = TripleDifference()
        results = ddd.fit(
            small_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # Should still produce results
        assert results is not None
        assert np.isfinite(results.att)
        assert np.isfinite(results.se)

    def test_zero_treatment_effect(self):
        """Test when true treatment effect is zero."""
        data = generate_ddd_data(n_per_cell=200, true_att=0.0, seed=123)

        ddd = TripleDifference()
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # ATT should be close to zero
        assert abs(results.att) < 0.5

    def test_large_treatment_effect(self):
        """Test with large treatment effect."""
        data = generate_ddd_data(n_per_cell=100, true_att=10.0, seed=42)

        ddd = TripleDifference()
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert abs(results.att - 10.0) < 1.0

    def test_low_noise(self):
        """Test with very low noise."""
        data = generate_ddd_data(n_per_cell=100, true_att=2.0, noise_sd=0.1, seed=42)

        ddd = TripleDifference()
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # Should recover ATT very precisely
        assert abs(results.att - 2.0) < 0.2
        # Should be significant at 0.05 level
        assert results.p_value < 0.05

    def test_high_noise(self):
        """Test with high noise."""
        data = generate_ddd_data(n_per_cell=50, true_att=2.0, noise_sd=5.0, seed=42)

        ddd = TripleDifference()
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # Should still run, but with wider confidence intervals
        assert results is not None
        ci_width = results.conf_int[1] - results.conf_int[0]
        assert ci_width > 0.5  # Wide CI due to noise


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegression:
    """Regression tests to ensure consistent behavior."""

    def test_reproducibility(self, simple_ddd_data):
        """Test that results are reproducible."""
        ddd1 = TripleDifference()
        results1 = ddd1.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        ddd2 = TripleDifference()
        results2 = ddd2.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert results1.att == results2.att
        assert results1.se == results2.se

    def test_summary_does_not_raise(self, simple_ddd_data):
        """Test that summary() doesn't raise exceptions."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # Should not raise
        results.summary()
        results.print_summary()

    def test_repr_does_not_raise(self, simple_ddd_data):
        """Test that repr doesn't raise exceptions."""
        ddd = TripleDifference()
        results = ddd.fit(
            simple_ddd_data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # Should not raise
        repr_str = repr(results)
        assert "TripleDifferenceResults" in repr_str


# =============================================================================
# Rank Deficiency Tests
# =============================================================================


class TestRankDeficientAction:
    """Tests for rank_deficient_action parameter handling."""

    @pytest.fixture
    def ddd_data_with_covariates(self):
        """Create DDD data with covariates for testing."""
        np.random.seed(42)
        n = 400
        data = pd.DataFrame(
            {
                "group": np.repeat([0, 1], n // 2),
                "partition": np.tile(np.repeat([0, 1], n // 4), 2),
                "time": np.tile([0, 1], n // 2),
                "x1": np.random.randn(n),
            }
        )

        # Generate outcome with effect
        data["outcome"] = (
            1.0
            + 0.5 * data["x1"]
            + 0.5 * data["group"]
            + 0.3 * data["partition"]
            + 0.2 * data["time"]
            + 2.0 * data["group"] * data["partition"] * data["time"]
            + np.random.randn(n) * 0.5
        )

        return data

    def test_rank_deficient_action_error_raises(self, ddd_data_with_covariates):
        """Test that rank_deficient_action='error' raises ValueError on collinear data."""
        # Add a covariate that is perfectly collinear with x1
        ddd_data_with_covariates["x1_dup"] = ddd_data_with_covariates["x1"].copy()

        ddd = TripleDifference(
            estimation_method="reg",  # Use regression method to test OLS path
            rank_deficient_action="error",
        )
        with pytest.raises(ValueError, match="[Rr]ank-deficient"):
            ddd.fit(
                ddd_data_with_covariates,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
                covariates=["x1", "x1_dup"],
            )

    def test_rank_deficient_action_silent_no_warning(self, ddd_data_with_covariates):
        """Test that rank_deficient_action='silent' produces no warning."""
        import warnings

        # Add a covariate that is perfectly collinear with x1
        ddd_data_with_covariates["x1_dup"] = ddd_data_with_covariates["x1"].copy()

        ddd = TripleDifference(
            estimation_method="reg",  # Use regression method to test OLS path
            rank_deficient_action="silent",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = ddd.fit(
                ddd_data_with_covariates,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
                covariates=["x1", "x1_dup"],
            )

            # No warnings about rank deficiency should be emitted
            rank_warnings = [
                x
                for x in w
                if "Rank-deficient" in str(x.message) or "rank-deficient" in str(x.message).lower()
            ]
            assert len(rank_warnings) == 0, f"Expected no rank warnings, got {rank_warnings}"

        # Should still get valid results
        assert results is not None
        assert results.att is not None

    def test_convenience_function_passes_rank_deficient_action(self, ddd_data_with_covariates):
        """Test that triple_difference() convenience function passes rank_deficient_action."""
        from diff_diff import triple_difference

        # Add a covariate that is perfectly collinear with x1
        ddd_data_with_covariates["x1_dup"] = ddd_data_with_covariates["x1"].copy()

        # Should raise with "error" action
        with pytest.raises(ValueError, match="[Rr]ank-deficient"):
            triple_difference(
                ddd_data_with_covariates,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
                estimation_method="reg",
                covariates=["x1", "x1_dup"],
                rank_deficient_action="error",
            )


class TestTripleDifferenceTStatNaN:
    """Tests for NaN t_stat when SE is invalid."""

    def test_tstat_nan_when_se_zero(self):
        """t_stat is NaN (not 0.0) when SE is zero or non-finite."""
        # Generate standard DDD data
        data = generate_ddd_data(n_per_cell=100, true_att=2.0, seed=42)

        td = TripleDifference(estimation_method="reg")
        results = td.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        se = results.se
        t_stat = results.t_stat

        if not np.isfinite(se) or se == 0:
            assert np.isnan(t_stat), f"t_stat should be NaN when SE={se}, got {t_stat}"
            ci = results.conf_int
            assert np.isnan(ci[0]) and np.isnan(
                ci[1]
            ), f"conf_int should be (NaN, NaN) when SE={se}, got {ci}"
        else:
            expected = results.att / se
            assert np.isclose(
                t_stat, expected
            ), f"t_stat should be ATT/SE, expected {expected}, got {t_stat}"

    def test_tstat_consistency_all_methods(self):
        """t_stat follows NaN pattern across all estimation methods."""
        data = generate_ddd_data(
            n_per_cell=50,
            true_att=2.0,
            seed=42,
            add_covariates=True,
            covariate_effect=0.5,
        )

        for method in ["reg", "ipw", "dr"]:
            td = TripleDifference(estimation_method=method)
            results = td.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
                covariates=["x1"],
            )

            se = results.se
            t_stat = results.t_stat

            if not np.isfinite(se) or se == 0:
                assert np.isnan(
                    t_stat
                ), f"[{method}] t_stat should be NaN when SE={se}, got {t_stat}"
            else:
                expected = results.att / se
                assert np.isclose(t_stat, expected), (
                    f"[{method}] t_stat should be ATT/SE, " f"expected {expected}, got {t_stat}"
                )


def _generate_ddd_data_with_state_clusters(
    n_states: int = 25,
    units_per_state: int = 8,
    state_effect_sd: float = 3.0,
    true_att: float = 2.0,
    seed: int = 53,
) -> pd.DataFrame:
    """Generate DDD data with state-level random effects.

    Used by the defensive cluster-changes-SE test below. Per
    ``feedback_homogeneous_dgp_no_twfe_bias``, assertive cluster-vs-no-cluster
    SE tests need a panel with intra-cluster correlation; without state
    random effects, cluster-robust SE collapses to per-unit SE.
    """
    rng = np.random.default_rng(seed)
    state_effects = rng.normal(0.0, state_effect_sd, n_states)
    rows = []
    next_unit = 0
    for s in range(n_states):
        for _ in range(units_per_state):
            for g in [0, 1]:
                for p in [0, 1]:
                    for t in [0, 1]:
                        y = (
                            state_effects[s]
                            + 10.0
                            + 2 * g
                            + 1 * p
                            + 0.5 * t
                            + 0.3 * g * p
                            + 0.2 * g * t
                            + 0.1 * p * t
                            + (true_att if (g == 1 and p == 1 and t == 1) else 0.0)
                            + rng.normal(0.0, 0.5)
                        )
                        rows.append(
                            {
                                "outcome": y,
                                "group": g,
                                "partition": p,
                                "time": t,
                                "state": s,
                                "unit_id": next_unit,
                            }
                        )
                        next_unit += 1
    return pd.DataFrame(rows)


class TestTripleDifferenceClusterDefensive:
    """Defensive: TripleDifference cluster= produces SE differing from
    cluster=None on a panel with intra-cluster correlation.

    Added because the audit found that TripleDifference's bare-cluster
    code path (``triple_diff.py:1245-1259``) is correct but had no
    positive regression test (only an error-handling test for missing
    cluster columns). Without this assertive test, a future refactor
    could silently regress the cluster wiring to a no-op (matching the
    CS class of bug just fixed). Mirrors
    ``tests/test_two_stage.py::test_cluster_changes_ses``.
    """

    def test_cluster_changes_ses(self):
        data = _generate_ddd_data_with_state_clusters(seed=53)

        td_unit = TripleDifference()  # cluster=None default
        res_unit = td_unit.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        td_cluster = TripleDifference(cluster="state")
        res_cluster = td_cluster.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert np.isfinite(res_unit.se) and res_unit.se > 0
        assert np.isfinite(res_cluster.se) and res_cluster.se > 0
        assert abs(res_unit.se - res_cluster.se) > 1e-6, (
            f"TripleDifference cluster='state' SE ({res_cluster.se:.6f}) "
            f"is effectively identical to cluster=None SE "
            f"({res_unit.se:.6f}) — the cluster= parameter may "
            "have regressed to a silent no-op."
        )


def _ddd_survey_panel(seed: int = 71, n: int = 400) -> pd.DataFrame:
    """Cross-sectional DDD data with survey columns for vcov_type bit-equal tests.

    Mirrors ``tests/test_survey_phase3.py::ddd_survey_data`` but uses
    ``default_rng`` for reproducibility independent of global state.
    """
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "outcome": rng.standard_normal(n) + 0.5,
            "group": rng.choice([0, 1], n),
            "partition": rng.choice([0, 1], n),
            "time": rng.choice([0, 1], n),
            "weight": rng.uniform(0.5, 2.0, n),
            "stratum": rng.choice([1, 2, 3], n),
        }
    )
    mask = (data["group"] == 1) & (data["partition"] == 1) & (data["time"] == 1)
    data.loc[mask, "outcome"] += 1.5
    return data


def _ddd_replicate_panel(seed: int = 89, n: int = 200, n_rep: int = 10):
    """DDD panel with JK1 replicate-weight columns for testing the
    replicate-variance inference branch. Mirrors the pattern in
    ``tests/test_survey_phase6.py::test_triple_diff_replicate_all_methods``
    but uses ``default_rng`` for reproducibility independent of global state.

    Returns (DataFrame with outcome/group/partition/time/weight + rep_0..rep_{n_rep-1},
    list of replicate column names).
    """
    rng = np.random.default_rng(seed)
    d1 = np.repeat([0, 1], n // 2)
    d2 = np.tile([0, 1], n // 2)
    post = rng.choice([0, 1], n)
    y = 1.0 + 0.5 * d1 + 0.3 * d2 + 2.0 * d1 * d2 * post + rng.standard_normal(n) * 0.5
    w = 1.0 + rng.exponential(0.3, n)
    data = pd.DataFrame(
        {
            "outcome": y,
            "group": d1,
            "partition": d2,
            "time": post,
            "weight": w,
        }
    )
    cluster_size = n // n_rep
    rep_cols = []
    for r in range(n_rep):
        w_r = w.copy()
        start = r * cluster_size
        end = min((r + 1) * cluster_size, n)
        w_r[start:end] = 0.0
        w_r[w_r > 0] *= n_rep / (n_rep - 1)
        col = f"rep_{r}"
        data[col] = w_r
        rep_cols.append(col)
    return data, rep_cols


class TestTripleDifferenceVcovType:
    """Phase 1b interstitial #2: vcov_type input contract on TripleDifference.

    TripleDifference uses IF-based variance per Ortiz-Villavicencio &
    Sant'Anna (2025); vcov_type is permanently narrow to {"hc1"}.
    Analytical-sandwich families {classical, hc2, hc2_bm} and conley are
    rejected at __init__ with methodology-rooted messages. Mirrors CS
    PR #487 template at tests/test_staggered.py.

    5-surface matrix:
      1. Default preserved bit-equally (3 estimation methods)
      2. Cluster path preserved bit-equally
      3. Survey path preserved bit-equally
      4. Input rejection at __init__ (methodology terminology)
      5. fit()-time revalidation (set_params can't bypass)

    Plus introspection tests for Results carrier, summary render,
    to_dict, get_params, fit-clone idempotence, and convenience function.
    """

    # -- Surface 1: default behavior preserved bit-equally ---------------

    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_default_hc1_bit_equal_baseline(self, method):
        """vcov_type='hc1' (explicit) is bit-equal to the default for every
        estimation method. Guards against drift between __init__ defaults
        and Results construction when vcov_type was threaded through."""
        data = generate_ddd_data(n_per_cell=80, true_att=2.0, seed=11)

        r_default = TripleDifference(estimation_method=method).fit(
            data, outcome="outcome", group="group", partition="partition", time="time"
        )
        r_explicit = TripleDifference(estimation_method=method, vcov_type="hc1").fit(
            data, outcome="outcome", group="group", partition="partition", time="time"
        )
        assert r_default.att == r_explicit.att, f"[{method}] ATT not bit-equal"
        assert r_default.se == r_explicit.se, f"[{method}] SE not bit-equal"

    # -- Surface 2: cluster path preserved bit-equally -------------------

    def test_cluster_hc1_bit_equal_baseline(self):
        """cluster=<col> + vcov_type='hc1' bit-equal to cluster=<col> alone."""
        data = _generate_ddd_data_with_state_clusters(seed=23)

        r_default = TripleDifference(cluster="state").fit(
            data, outcome="outcome", group="group", partition="partition", time="time"
        )
        r_explicit = TripleDifference(cluster="state", vcov_type="hc1").fit(
            data, outcome="outcome", group="group", partition="partition", time="time"
        )
        assert r_default.att == r_explicit.att
        assert r_default.se == r_explicit.se

    # -- Surface 3: survey path preserved bit-equally --------------------

    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_survey_hc1_bit_equal_baseline(self, method):
        """survey_design + vcov_type='hc1' bit-equal to survey_design alone.

        Pre-empt: CS PR #487 R2 caught a survey_metadata overload bug;
        same risk class here when threading vcov_type alongside survey_design.
        """
        data = _ddd_survey_panel(seed=29)
        sd = SurveyDesign(weights="weight", strata="stratum")

        r_default = TripleDifference(estimation_method=method).fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            survey_design=sd,
        )
        r_explicit = TripleDifference(estimation_method=method, vcov_type="hc1").fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            survey_design=sd,
        )
        assert r_default.att == r_explicit.att, f"[{method}] survey ATT not bit-equal"
        assert r_default.se == r_explicit.se, f"[{method}] survey SE not bit-equal"

    # -- Surface 3b: replicate-weight survey path preserved bit-equally --

    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_replicate_survey_hc1_bit_equal_baseline(self, method):
        """Replicate-weight survey design + vcov_type='hc1' bit-equal to
        replicate-weight survey design alone. Exercises the distinct
        replicate-df branch in fit() (separate from the TSL branch in
        Surface 3 above).

        Addresses codex R5 P1 (.claude/reviews/local-review-latest.md):
        the prior survey bit-equal coverage only exercised the analytical
        TSL path; the replicate-variance path was unverified."""
        data, rep_cols = _ddd_replicate_panel(seed=89)
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        r_default = TripleDifference(estimation_method=method).fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            survey_design=sd,
        )
        r_explicit = TripleDifference(estimation_method=method, vcov_type="hc1").fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            survey_design=sd,
        )
        assert r_default.att == r_explicit.att, f"[{method}] replicate ATT not bit-equal"
        assert r_default.se == r_explicit.se, f"[{method}] replicate SE not bit-equal"
        # Results-surface assertion: vcov_type carries through on the
        # replicate path AND summary still suppresses the raw variance
        # line (survey block remains the canonical surface).
        assert r_explicit.vcov_type == "hc1"
        assert r_explicit.survey_metadata is not None
        out = r_explicit.summary()
        assert "Survey Design" in out
        assert "Variance estimator" not in out

    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_cluster_plus_replicate_weights_rejected(self, method):
        """cluster= + survey_design(replicate_weights=...) raises
        NotImplementedError because replicate-weight variance is computed
        by replicate reweighting (BRR / Fay / JK1 / JKn / SDR) and ignores
        PSU/cluster entirely — honoring the cluster argument would silently
        have no effect on the variance estimate.

        Addresses codex R7 P1 (.claude/reviews/local-review-latest.md):
        the silent no-op was caught by direct interpreter inspection of
        the new JK1 replicate fixture. Mirrors CallawaySantAnna's guard
        at diff_diff/staggered.py:1705-1719 (CS PR #487)."""
        data, rep_cols = _ddd_replicate_panel(seed=89)
        # Add a 'state' column to attempt as the cluster argument
        rng = np.random.default_rng(seed=89)
        data["state"] = rng.choice(range(5), size=len(data))
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        with pytest.raises(NotImplementedError, match="replicate-weight"):
            TripleDifference(estimation_method=method, cluster="state").fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
                survey_design=sd,
            )

    # -- Surface 4: input rejection at __init__ --------------------------

    def test_reject_classical_at_init(self):
        with pytest.raises(ValueError, match="influence-function"):
            TripleDifference(vcov_type="classical")

    def test_reject_hc2_at_init(self):
        with pytest.raises(ValueError, match="Ortiz-Villavicencio"):
            TripleDifference(vcov_type="hc2")

    def test_reject_hc2_bm_at_init(self):
        with pytest.raises(ValueError, match="hat matrix"):
            TripleDifference(vcov_type="hc2_bm")

    def test_reject_hc2_bm_at_init_bm_keyword(self):
        """Distinct keyword pin: Bell-McCaffrey terminology in the message."""
        with pytest.raises(ValueError, match="Bell-McCaffrey"):
            TripleDifference(vcov_type="hc2_bm")

    def test_reject_conley_at_init(self):
        with pytest.raises(ValueError, match="spatial-HAC"):
            TripleDifference(vcov_type="conley")

    def test_reject_conley_at_init_todo_pointer(self):
        """Conley rejection cites the TODO follow-up row."""
        with pytest.raises(ValueError, match="TODO"):
            TripleDifference(vcov_type="conley")

    def test_reject_unknown_vcov_type(self):
        """Generic membership rejection for unrecognized values."""
        with pytest.raises(ValueError, match="invalid"):
            TripleDifference(vcov_type="hc4")

    # -- Surface 5: fit()-time revalidation (set_params can't bypass) ----

    def test_set_params_bad_vcov_caught_at_fit_time(self):
        """set_params is strict-mirror sklearn (no atomic validation), but
        fit() re-validates so a bad set_params(vcov_type='hc4') surfaces a
        clear error at fit-time rather than silently propagating a bad
        value to Results metadata. Mirrors CS
        tests/test_staggered.py::test_set_params_bad_vcov_caught_at_fit_time."""
        td = TripleDifference()
        # set_params succeeds (sklearn-style mutate-then-validate-at-use)
        td.set_params(vcov_type="hc4")
        assert td.vcov_type == "hc4"
        # fit() re-validates and raises
        data = generate_ddd_data(n_per_cell=40, true_att=2.0, seed=37)
        with pytest.raises(ValueError, match="hc4"):
            td.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )

    def test_set_params_bad_vcov_classical_caught_at_fit_time(self):
        """Same as above but with an IF-incompatible family (classical).
        Catches the silent-propagation path on the methodology-rooted
        rejection branch."""
        td = TripleDifference()
        td.set_params(vcov_type="classical")
        data = generate_ddd_data(n_per_cell=40, true_att=2.0, seed=39)
        with pytest.raises(ValueError, match="influence-function"):
            td.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )

    # -- Introspection contract -------------------------------------------

    def test_default_vcov_type_is_hc1(self):
        """Attribute default sanity (pre-fit)."""
        assert TripleDifference().vcov_type == "hc1"

    def test_get_params_includes_vcov_type(self):
        td = TripleDifference()
        params = td.get_params()
        assert "vcov_type" in params
        assert params["vcov_type"] == "hc1"

    def test_results_carries_vcov_type(self):
        data = generate_ddd_data(n_per_cell=40, true_att=2.0, seed=43)
        res = TripleDifference().fit(
            data, outcome="outcome", group="group", partition="partition", time="time"
        )
        assert res.vcov_type == "hc1"

    def test_to_dict_includes_vcov_type(self):
        """CS R7 caught the same Results-introspection gap on the dict surface."""
        data = generate_ddd_data(n_per_cell=40, true_att=2.0, seed=47)
        res = TripleDifference().fit(
            data, outcome="outcome", group="group", partition="partition", time="time"
        )
        d = res.to_dict()
        assert "vcov_type" in d
        assert d["vcov_type"] == "hc1"

    def test_summary_includes_vcov_type(self):
        """Default (no cluster, no survey) renders the variance-family label
        via the shared _format_vcov_label, not the raw vcov_type string."""
        data = generate_ddd_data(n_per_cell=40, true_att=2.0, seed=51)
        res = TripleDifference().fit(
            data, outcome="outcome", group="group", partition="partition", time="time"
        )
        out = res.summary()
        assert "Variance estimator" in out
        assert "HC1 heteroskedasticity-robust" in out

    def test_summary_cluster_label_is_cr1_not_raw_hc1(self):
        """Cluster fit renders the cluster-aware CR1 Liang-Zeger label rather
        than 'hc1', since the actual algebra is CR1 on the combined IF.
        Addresses codex local-review P2 — raw 'hc1' line was misleading."""
        data = _generate_ddd_data_with_state_clusters(seed=53)
        res = TripleDifference(cluster="state").fit(
            data, outcome="outcome", group="group", partition="partition", time="time"
        )
        out = res.summary()
        assert "CR1 cluster-robust at state" in out
        # G=<n_clusters> suffix present
        assert f"G={res.n_clusters}" in out

    def test_summary_no_variance_estimator_line_under_survey(self):
        """Survey fit suppresses the variance-estimator line; the Survey Design
        block above already names design + n_psu + df. The analytical SE is
        TSL on the combined IF (or replicate refit), not the raw hc1 sandwich,
        so a 'Variance estimator: hc1' line would be misleading. Addresses
        codex local-review P2 + P3 (summary regression coverage gap)."""
        data = _ddd_survey_panel(seed=29)
        sd = SurveyDesign(weights="weight", strata="stratum")
        res = TripleDifference(estimation_method="reg").fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            survey_design=sd,
        )
        out = res.summary()
        # The Survey Design block remains the canonical surface
        assert "Survey Design" in out
        # No misleading variance-estimator line on survey-backed fits
        assert "Variance estimator" not in out

    def test_results_cluster_name_carries_through(self):
        """cluster_name field on Results: populated when cluster= set, None otherwise.
        Mirrors CS PR #487 pattern; consumed by _format_vcov_label in summary()."""
        data = generate_ddd_data(n_per_cell=40, true_att=2.0, seed=63)
        r_none = TripleDifference().fit(
            data, outcome="outcome", group="group", partition="partition", time="time"
        )
        assert r_none.cluster_name is None

        data2 = _generate_ddd_data_with_state_clusters(seed=67)
        r_cluster = TripleDifference(cluster="state").fit(
            data2, outcome="outcome", group="group", partition="partition", time="time"
        )
        assert r_cluster.cluster_name == "state"
        # And it flows through to_dict
        d = r_cluster.to_dict()
        assert d.get("cluster_name") == "state"

    def test_cluster_name_suppressed_under_survey_design(self):
        """When survey_design overrides the bare cluster= argument, the Results
        cluster_name + n_clusters fields are suppressed so they don't misreport
        the ignored argument. The Survey Design block on summary() is the
        canonical surface for cluster/PSU reporting on survey-backed fits.

        Addresses codex local-review R2 P2 (.claude/reviews/local-review-latest.md):
        Under cluster='state' + survey_design(psu='psu') with conflicting
        partitions, _resolve_effective_cluster picks survey_design.psu and
        warns; the records on Results should reflect that, not the raw
        `self.cluster` argument the user passed."""
        # Build DDD survey panel with BOTH a 'state' column (user's cluster=)
        # and a 'psu' column (survey_design.psu) at DIFFERENT partitions.
        # The survey-design PSU wins; cluster= is overridden with a warning.
        data = _ddd_survey_panel(seed=83).copy()
        rng = np.random.default_rng(seed=83)
        # 'psu' is a coarser partition than 'state' — distinct grouping
        data["state"] = rng.choice(range(20), size=len(data))
        data["psu"] = rng.choice(range(5), size=len(data))

        sd = SurveyDesign(weights="weight", psu="psu")
        with pytest.warns(UserWarning, match="PSU will be used"):
            res = TripleDifference(estimation_method="reg", cluster="state").fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
                survey_design=sd,
            )
        # cluster_name + n_clusters suppressed under survey-backed fit
        assert res.cluster_name is None, (
            f"cluster_name should be suppressed under survey-backed fit, "
            f"got {res.cluster_name!r} (the raw cluster= argument)"
        )
        assert res.n_clusters is None, (
            f"n_clusters should be suppressed under survey-backed fit, "
            f"got {res.n_clusters} (would be raw data['state'].nunique())"
        )
        # And to_dict doesn't leak the misleading raw cluster
        d = res.to_dict()
        assert "cluster_name" not in d or d.get("cluster_name") is None
        assert "n_clusters" not in d or d.get("n_clusters") is None
        # Survey block remains the canonical surface for cluster/PSU reporting
        assert "Survey Design" in res.summary()
        assert res.survey_metadata is not None

    def test_fit_clone_idempotent_on_vcov_type(self):
        """get_params -> reconstruct -> refit -> identical SE.
        Catches drift between __init__ defaults, attribute storage, and
        Results construction (sklearn clone() pattern)."""
        data = generate_ddd_data(n_per_cell=40, true_att=2.0, seed=57)
        td1 = TripleDifference(vcov_type="hc1")
        r1 = td1.fit(data, outcome="outcome", group="group", partition="partition", time="time")
        td2 = TripleDifference(**td1.get_params())
        r2 = td2.fit(data, outcome="outcome", group="group", partition="partition", time="time")
        assert r1.att == r2.att
        assert r1.se == r2.se
        assert r2.vcov_type == "hc1"

    # -- Convenience function threading ----------------------------------

    def test_triple_difference_convenience_func_rejects_invalid_vcov_type(self):
        """Invalid vcov_type rejected at the function entry point too."""
        data = generate_ddd_data(n_per_cell=40, true_att=2.0, seed=59)
        with pytest.raises(ValueError, match="influence-function"):
            triple_difference(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
                vcov_type="classical",
            )

    def test_triple_difference_convenience_func_threads_valid_vcov_type(self):
        """Valid vcov_type='hc1' fits successfully AND lands on Results."""
        data = generate_ddd_data(n_per_cell=40, true_att=2.0, seed=61)
        res = triple_difference(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            vcov_type="hc1",
        )
        assert res.vcov_type == "hc1"
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
