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

from diff_diff.triple_diff import (
    TripleDifference,
    TripleDifferenceResults,
    triple_difference,
)


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

    Parameters
    ----------
    n_per_cell : int
        Number of observations per cell (8 cells total).
    true_att : float
        True average treatment effect on the treated.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed for reproducibility.
    add_covariates : bool
        Whether to add covariates that affect the outcome.
    covariate_effect : float
        Effect size of covariates on outcome.

    Returns
    -------
    pd.DataFrame
        Synthetic DDD data with known ATT.
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

                    rows.append({
                        "outcome": y,
                        "group": g,
                        "partition": p,
                        "time": t,
                        "x1": x1,
                        "x2": x2,
                        "unit_id": len(rows),
                    })

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
        assert results.r_squared is not None
        assert 0 <= results.r_squared <= 1
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

        # Covariates should improve R-squared
        assert results_with_cov.r_squared >= results_no_cov.r_squared

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
        assert "P(G=1) mean" in results.pscore_stats


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
            ~((simple_ddd_data["group"] == 1) &
              (simple_ddd_data["partition"] == 1) &
              (simple_ddd_data["time"] == 0))
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
        summary = results.summary()
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
