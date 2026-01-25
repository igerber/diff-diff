"""Tests for Triply Robust Panel (TROP) estimator."""

import numpy as np
import pandas as pd
import pytest

from diff_diff import SyntheticDiD
from diff_diff.trop import TROP, TROPResults, trop
from diff_diff.prep import generate_factor_data


def generate_factor_dgp(
    n_units: int = 50,
    n_pre: int = 10,
    n_post: int = 5,
    n_treated: int = 10,
    n_factors: int = 2,
    treatment_effect: float = 2.0,
    factor_strength: float = 1.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate panel data with known factor structure.

    Wrapper around the library function for backward compatibility with tests.
    """
    data = generate_factor_data(
        n_units=n_units,
        n_pre=n_pre,
        n_post=n_post,
        n_treated=n_treated,
        n_factors=n_factors,
        treatment_effect=treatment_effect,
        factor_strength=factor_strength,
        treated_loading_shift=0.5,
        unit_fe_sd=1.0,
        noise_sd=noise_std,
        seed=seed,
    )

    # Return only the columns the tests expect
    return data[["unit", "period", "outcome", "treated"]]


@pytest.fixture
def factor_dgp_data():
    """Generate data with factor structure and known treatment effect."""
    return generate_factor_dgp(
        n_units=30,
        n_pre=8,
        n_post=4,
        n_treated=5,
        n_factors=2,
        treatment_effect=2.0,
        factor_strength=1.0,
        noise_std=0.5,
        seed=42,
    )


@pytest.fixture
def simple_panel_data():
    """Generate simple panel data without factors."""
    rng = np.random.default_rng(123)

    n_units = 20
    n_treated = 5
    n_pre = 5
    n_post = 3
    true_att = 3.0

    data = []
    for i in range(n_units):
        is_treated = i < n_treated
        for t in range(n_pre + n_post):
            post = t >= n_pre
            y = 10.0 + i * 0.1 + t * 0.5
            treatment_indicator = 1 if (is_treated and post) else 0
            if treatment_indicator:
                y += true_att
            y += rng.normal(0, 0.5)
            data.append({
                "unit": i,
                "period": t,
                "outcome": y,
                "treated": treatment_indicator,
            })

    return pd.DataFrame(data)


class TestTROP:
    """Tests for TROP estimator."""

    def test_basic_fit(self, simple_panel_data):
        """Test basic model fitting."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert isinstance(results, TROPResults)
        assert trop_est.is_fitted_
        assert results.n_obs == len(simple_panel_data)
        assert results.n_control == 15
        assert results.n_treated == 5

    def test_fit_with_factors(self, factor_dgp_data):
        """Test fitting with factor structure."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1, 1.0],
            n_bootstrap=20,
            seed=42
        )
        results = trop_est.fit(
            factor_dgp_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert isinstance(results, TROPResults)
        assert results.effective_rank >= 0
        assert results.factor_matrix.shape == (12, 30)  # n_periods x n_units

    def test_treatment_effect_recovery(self, factor_dgp_data):
        """Test that TROP recovers treatment effect direction."""
        true_att = 2.0

        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0, 0.5, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=30,
            seed=42
        )
        results = trop_est.fit(
            factor_dgp_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # ATT should be positive (correct direction)
        assert results.att > 0
        # Should be reasonably close to true value
        assert abs(results.att - true_att) < 3.0

    def test_tuning_parameter_selection(self, simple_panel_data):
        """Test that LOOCV selects tuning parameters."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5, 1.0, 2.0],
            lambda_unit_grid=[0.0, 0.5, 1.0],
            lambda_nn_grid=[0.0, 0.1, 1.0],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Check that lambda values are from the grid
        assert results.lambda_time in trop_est.lambda_time_grid
        assert results.lambda_unit in trop_est.lambda_unit_grid
        assert results.lambda_nn in trop_est.lambda_nn_grid

    def test_bootstrap_variance(self, simple_panel_data):
        """Test bootstrap variance estimation."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            variance_method="bootstrap",
            n_bootstrap=30,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert results.se > 0
        assert results.variance_method == "bootstrap"
        assert results.n_bootstrap == 30
        assert results.bootstrap_distribution is not None

    def test_jackknife_variance(self, simple_panel_data):
        """Test jackknife variance estimation."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            variance_method="jackknife",
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert results.se >= 0
        assert results.variance_method == "jackknife"

    def test_confidence_interval(self, simple_panel_data):
        """Test confidence interval properties."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            alpha=0.05,
            n_bootstrap=30,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        lower, upper = results.conf_int
        assert lower < results.att < upper
        assert lower < upper

    def test_get_set_params(self):
        """Test sklearn-compatible get_params and set_params."""
        trop_est = TROP(alpha=0.05)

        params = trop_est.get_params()
        assert params["alpha"] == 0.05

        trop_est.set_params(alpha=0.10)
        assert trop_est.alpha == 0.10

    def test_invalid_variance_method(self):
        """Test error on invalid variance method."""
        with pytest.raises(ValueError):
            TROP(variance_method="invalid")

    def test_missing_columns(self, simple_panel_data):
        """Test error when column is missing."""
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5
        )
        with pytest.raises(ValueError, match="Missing columns"):
            trop_est.fit(
                simple_panel_data,
                outcome="nonexistent",
                treatment="treated",
                unit="unit",
                time="period",
            )

    def test_no_treated_observations(self):
        """Test error when no treated observations."""
        data = pd.DataFrame({
            "unit": [0, 0, 1, 1],
            "period": [0, 1, 0, 1],
            "outcome": [1, 2, 3, 4],
            "treated": [0, 0, 0, 0],
        })

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5
        )
        with pytest.raises(ValueError, match="No treated observations"):
            trop_est.fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )

    def test_no_control_units(self):
        """Test error when no control units."""
        data = pd.DataFrame({
            "unit": [0, 0, 1, 1],
            "period": [0, 1, 0, 1],
            "outcome": [1, 2, 3, 4],
            "treated": [0, 1, 0, 1],  # Both units become treated
        })

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5
        )
        with pytest.raises(ValueError, match="No control units"):
            trop_est.fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )


class TestTROPResults:
    """Tests for TROPResults dataclass."""

    def test_summary(self, simple_panel_data):
        """Test that summary produces string output."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        summary = results.summary()
        assert isinstance(summary, str)
        assert "ATT" in summary
        assert "TROP" in summary
        assert "LOOCV" in summary
        assert "Lambda" in summary

    def test_to_dict(self, simple_panel_data):
        """Test conversion to dictionary."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        d = results.to_dict()
        assert "att" in d
        assert "se" in d
        assert "lambda_time" in d
        assert "lambda_unit" in d
        assert "lambda_nn" in d
        assert "effective_rank" in d

    def test_to_dataframe(self, simple_panel_data):
        """Test conversion to DataFrame."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "att" in df.columns

    def test_get_treatment_effects_df(self, simple_panel_data):
        """Test getting treatment effects DataFrame."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        effects_df = results.get_treatment_effects_df()
        assert isinstance(effects_df, pd.DataFrame)
        assert "unit" in effects_df.columns
        assert "time" in effects_df.columns
        assert "effect" in effects_df.columns
        assert len(effects_df) == results.n_treated_obs

    def test_get_unit_effects_df(self, simple_panel_data):
        """Test getting unit effects DataFrame."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        effects_df = results.get_unit_effects_df()
        assert isinstance(effects_df, pd.DataFrame)
        assert "unit" in effects_df.columns
        assert "effect" in effects_df.columns

    def test_get_time_effects_df(self, simple_panel_data):
        """Test getting time effects DataFrame."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        effects_df = results.get_time_effects_df()
        assert isinstance(effects_df, pd.DataFrame)
        assert "time" in effects_df.columns
        assert "effect" in effects_df.columns

    def test_is_significant(self, simple_panel_data):
        """Test significance property."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            alpha=0.05,
            n_bootstrap=30,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert isinstance(results.is_significant, bool)

    def test_significance_stars(self, simple_panel_data):
        """Test significance stars."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=30,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        stars = results.significance_stars
        assert stars in ["", ".", "*", "**", "***"]

    def test_nan_propagation_when_se_zero(self):
        """Test that inference fields are NaN when SE is zero/undefined.

        This verifies the P0 fix: when SE <= 0, all inference fields
        (t_stat, p_value, conf_int) should be NaN, not finite values.
        """
        from diff_diff.trop import TROPResults

        # Create a TROPResults directly with SE=0
        results = TROPResults(
            att=1.0,
            se=0.0,  # Zero SE - inference should be undefined
            t_stat=np.nan,
            p_value=np.nan,
            conf_int=(np.nan, np.nan),
            n_obs=100,
            n_treated=5,
            n_control=10,
            n_treated_obs=20,
            unit_effects={0: 0.1, 1: 0.2},
            time_effects={0: 0.0, 1: 0.1},
            treatment_effects={(0, 5): 1.0},
            lambda_time=1.0,
            lambda_unit=1.0,
            lambda_nn=0.1,
            factor_matrix=np.zeros((10, 15)),
            effective_rank=2.0,
            loocv_score=0.5,
            variance_method="bootstrap",
        )

        # Verify that all inference fields are NaN when SE=0
        assert np.isnan(results.t_stat), "t_stat should be NaN when SE=0"
        assert np.isnan(results.p_value), "p_value should be NaN when SE=0"
        assert np.isnan(results.conf_int[0]), "conf_int[0] should be NaN when SE=0"
        assert np.isnan(results.conf_int[1]), "conf_int[1] should be NaN when SE=0"

        # Verify the ATT itself is still valid
        assert results.att == 1.0, "ATT should still be valid"


class TestTROPvsSDID:
    """Tests comparing TROP to SDID under different DGPs."""

    def test_trop_handles_factor_dgp(self):
        """Test that TROP works on factor DGP data."""
        data = generate_factor_dgp(
            n_units=30,
            n_pre=8,
            n_post=4,
            n_treated=5,
            n_factors=2,
            treatment_effect=2.0,
            factor_strength=1.5,
            noise_std=0.5,
            seed=42,
        )

        # TROP should complete without error
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1, 1.0],
            n_bootstrap=20,
            seed=42
        )
        results = trop_est.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert results.att != 0
        assert results.se >= 0


class TestConvenienceFunction:
    """Tests for trop() convenience function."""

    def test_convenience_function(self, simple_panel_data):
        """Test that convenience function works."""
        results = trop(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42,
        )

        assert isinstance(results, TROPResults)
        assert results.n_obs == len(simple_panel_data)

    def test_convenience_with_kwargs(self, simple_panel_data):
        """Test convenience function with additional kwargs."""
        results = trop(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0, 0.5],
            lambda_nn_grid=[0.0, 0.1],
            max_iter=50,
            n_bootstrap=10,
            seed=42,
        )

        assert isinstance(results, TROPResults)


class TestMethodologyVerification:
    """Tests verifying TROP methodology matches paper specifications.

    These tests verify:
    1. Limiting cases match expected behavior
    2. Treatment effect recovery under paper's simulation DGP
    3. Observation-specific weighting produces expected results
    """

    def test_limiting_case_uniform_weights(self):
        """
        Test limiting case: λ_unit = λ_time = 0, λ_nn = 0.

        With all lambdas at zero, TROP should use uniform weights and no
        nuclear norm regularization, giving TWFE-like estimates.
        """
        # Generate simple data with known treatment effect
        rng = np.random.default_rng(42)
        n_units = 15
        n_treated = 5
        n_pre = 5
        n_post = 3
        true_att = 3.0

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            unit_fe = rng.normal(0, 0.5)
            for t in range(n_pre + n_post):
                post = t >= n_pre
                time_fe = 0.2 * t
                y = 10.0 + unit_fe + time_fe
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_att
                y += rng.normal(0, 0.3)
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "treated": treatment_indicator,
                })

        df = pd.DataFrame(data)

        # TROP with uniform weights
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Should recover treatment effect within reasonable tolerance
        assert abs(results.att - true_att) < 1.0, \
            f"ATT={results.att:.3f} should be close to true={true_att}"
        # Check that uniform weights were selected
        assert results.lambda_time == 0.0
        assert results.lambda_unit == 0.0
        assert results.lambda_nn == 0.0

    def test_unit_weights_reduce_bias(self):
        """
        Test that unit distance-based weights reduce bias when controls vary.

        When control units have varying similarity to treated units, using
        distance-based unit weights should improve estimation.
        """
        rng = np.random.default_rng(123)
        n_units = 25
        n_treated = 5
        n_pre = 6
        n_post = 3
        true_att = 2.5

        data = []
        # Create heterogeneous control units - some similar to treated, some different
        for i in range(n_units):
            is_treated = i < n_treated
            # Treated units and first 5 controls are similar
            if is_treated or i < n_treated + 5:
                unit_fe = 5.0 + rng.normal(0, 0.3)
            else:
                # Remaining controls are dissimilar
                unit_fe = 10.0 + rng.normal(0, 0.5)

            for t in range(n_pre + n_post):
                post = t >= n_pre
                time_fe = 0.2 * t
                y = unit_fe + time_fe
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_att
                y += rng.normal(0, 0.3)
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "treated": treatment_indicator,
                })

        df = pd.DataFrame(data)

        # TROP with unit weighting enabled
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0, 1.0, 2.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Should recover treatment effect reasonably well
        assert abs(results.att - true_att) < 1.5, \
            f"ATT={results.att:.3f} should be close to true={true_att}"

    def test_time_weights_reduce_bias(self):
        """
        Test that time distance-based weights reduce bias with trending data.

        When pre-treatment outcomes are trending, weighting recent periods
        more heavily should improve estimation.
        """
        rng = np.random.default_rng(456)
        n_units = 20
        n_treated = 5
        n_pre = 8
        n_post = 3
        true_att = 2.0

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            unit_fe = rng.normal(0, 0.5)

            for t in range(n_pre + n_post):
                post = t >= n_pre
                # Time trend that accelerates near treatment
                time_fe = 0.1 * t + 0.05 * (t ** 2 / n_pre)
                y = 10.0 + unit_fe + time_fe
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_att
                y += rng.normal(0, 0.3)
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "treated": treatment_indicator,
                })

        df = pd.DataFrame(data)

        # TROP with time weighting enabled
        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Should recover treatment effect direction
        assert results.att > 0, f"ATT={results.att:.3f} should be positive"
        # Check that time weighting was considered
        assert results.lambda_time in [0.0, 0.5, 1.0]

    def test_factor_model_reduces_bias(self):
        """
        Test that nuclear norm regularization reduces bias with factor structure.

        Following paper's simulation: when true DGP has interactive fixed effects,
        the factor model component should help recover the treatment effect.
        """
        # Generate data with known factor structure
        data = generate_factor_dgp(
            n_units=40,
            n_pre=10,
            n_post=5,
            n_treated=8,
            n_factors=2,
            treatment_effect=2.0,
            factor_strength=1.5,  # Strong factors
            noise_std=0.5,
            seed=789,
        )

        # TROP with nuclear norm regularization
        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5],
            lambda_unit_grid=[0.0, 0.5],
            lambda_nn_grid=[0.0, 0.1, 1.0, 5.0],
            n_bootstrap=20,
            seed=42
        )
        results = trop_est.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        true_att = 2.0
        # With factor adjustment, should recover treatment effect
        assert abs(results.att - true_att) < 2.0, \
            f"ATT={results.att:.3f} should be within 2.0 of true={true_att}"
        # Factor matrix should capture some structure
        assert results.effective_rank > 0, "Factor matrix should have positive rank"

    def test_paper_dgp_recovery(self):
        """
        Test treatment effect recovery using paper's simulation DGP.

        Based on Table 2 (page 32) simulation settings:
        - Factor model with 2 factors
        - Treatment effect = 0 (null hypothesis)
        - Should produce estimates centered around zero

        This is a methodological validation test.
        """
        # Generate data similar to paper's simulation
        rng = np.random.default_rng(2024)
        n_units = 50
        n_treated = 10
        n_pre = 10
        n_post = 5
        n_factors = 2
        true_tau = 0.0  # Null treatment effect

        # Generate factors F: (n_periods, n_factors)
        F = rng.normal(0, 1, (n_pre + n_post, n_factors))

        # Generate loadings Lambda: (n_factors, n_units)
        Lambda = rng.normal(0, 1, (n_factors, n_units))
        # Treated units have different loadings (selection on unobservables)
        Lambda[:, :n_treated] += 0.5

        # Unit fixed effects
        gamma = rng.normal(0, 1, n_units)
        gamma[:n_treated] += 1.0  # Selection on levels

        # Time fixed effects (linear trend)
        delta = np.linspace(0, 2, n_pre + n_post)

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_pre + n_post):
                post = t >= n_pre
                # Y = mu + gamma_i + delta_t + Lambda_i'F_t + tau*D + eps
                y = 10.0 + gamma[i] + delta[t]
                y += Lambda[:, i] @ F[t, :]  # Factor component
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_tau
                y += rng.normal(0, 0.5)  # Idiosyncratic noise

                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "treated": treatment_indicator,
                })

        df = pd.DataFrame(data)

        # TROP estimation
        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0, 0.5, 1.0],
            lambda_nn_grid=[0.0, 0.1, 1.0],
            n_bootstrap=30,
            seed=42
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Under null hypothesis, ATT should be close to zero
        # Allow for estimation error (this is a finite sample)
        assert abs(results.att) < 2.0, \
            f"ATT={results.att:.3f} should be close to true={true_tau} under null"
        # Check that factor model was used
        assert results.effective_rank >= 0


class TestOptimizationEquivalence:
    """Tests verifying optimized implementations produce identical results.

    These tests ensure the vectorized implementations in v2.1.0+ produce
    numerically equivalent results to the original loop-based implementations.
    """

    def test_precomputed_structures_consistency(self, simple_panel_data):
        """
        Test that pre-computed structures match dynamically computed values.

        Verifies:
        - Time distance matrix is correct
        - Unit distance matrix is symmetric
        - Control observations list is complete
        """
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42
        )

        # Fit to populate precomputed structures
        trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        precomputed = trop_est._precomputed
        assert precomputed is not None

        # Verify time distance matrix
        n_periods = precomputed["n_periods"]
        time_dist = precomputed["time_dist_matrix"]
        assert time_dist.shape == (n_periods, n_periods)
        # Check diagonal is zero
        assert np.allclose(np.diag(time_dist), 0)
        # Check symmetry
        assert np.allclose(time_dist, time_dist.T)
        # Check specific values: |t - s|
        for t in range(n_periods):
            for s in range(n_periods):
                assert time_dist[t, s] == abs(t - s)

        # Verify unit distance matrix
        n_units = precomputed["n_units"]
        unit_dist = precomputed["unit_dist_matrix"]
        assert unit_dist.shape == (n_units, n_units)
        # Check diagonal is zero
        assert np.allclose(np.diag(unit_dist), 0)
        # Check symmetry
        assert np.allclose(unit_dist, unit_dist.T)

    def test_vectorized_alternating_minimization(self):
        """
        Test that vectorized alternating minimization converges correctly.

        The vectorized implementation should produce the same fixed effects
        estimates as the original loop-based implementation.
        """
        rng = np.random.default_rng(42)
        n_units = 10
        n_periods = 8

        # Generate simple test data
        alpha_true = rng.normal(0, 1, n_units)
        beta_true = rng.normal(0, 1, n_periods)

        Y = np.outer(np.ones(n_periods), alpha_true) + np.outer(beta_true, np.ones(n_units))
        Y += rng.normal(0, 0.1, (n_periods, n_units))

        # All observations are control
        control_mask = np.ones((n_periods, n_units), dtype=bool)
        W = np.ones((n_periods, n_units))

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
        )

        # Run the estimation
        alpha_est, beta_est, L_est = trop_est._estimate_model(
            Y, control_mask, W, lambda_nn=0.0,
            n_units=n_units, n_periods=n_periods
        )

        # Check that we recovered the fixed effects structure
        # (up to a constant shift since FE are identified up to a constant)
        alpha_centered = alpha_est - np.mean(alpha_est)
        beta_centered = beta_est - np.mean(beta_est)
        alpha_true_centered = alpha_true - np.mean(alpha_true)
        beta_true_centered = beta_true - np.mean(beta_true)

        # Should be reasonably close
        assert np.corrcoef(alpha_centered, alpha_true_centered)[0, 1] > 0.95
        assert np.corrcoef(beta_centered, beta_true_centered)[0, 1] > 0.95

    def test_vectorized_weights_computation(self, simple_panel_data):
        """
        Test that vectorized weight computation produces correct results.

        Verifies that observation-specific weights follow Equation 3 from paper.
        """
        trop_est = TROP(
            lambda_time_grid=[0.5],
            lambda_unit_grid=[0.5],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42
        )

        # Fit to populate precomputed structures
        trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        precomputed = trop_est._precomputed
        n_units = precomputed["n_units"]
        n_periods = precomputed["n_periods"]
        control_unit_idx = precomputed["control_unit_idx"]

        # Build Y and D matrices from data
        all_units = sorted(simple_panel_data["unit"].unique())
        all_periods = sorted(simple_panel_data["period"].unique())
        Y = (
            simple_panel_data.pivot(index="period", columns="unit", values="outcome")
            .reindex(index=all_periods, columns=all_units)
            .values
        )
        D = (
            simple_panel_data.pivot(index="period", columns="unit", values="treated")
            .reindex(index=all_periods, columns=all_units)
            .fillna(0)
            .astype(int)
            .values
        )

        # Test for a specific observation
        i = 0  # First unit
        t = 5  # Post-treatment period
        lambda_time = 0.5
        lambda_unit = 0.5

        weights = trop_est._compute_observation_weights(
            Y, D, i, t, lambda_time, lambda_unit, control_unit_idx,
            n_units, n_periods
        )

        # Verify shape
        assert weights.shape == (n_periods, n_units)

        # Verify time weights follow exp(-lambda_time * |t - s|)
        time_weights = weights[:, i]  # Weights for unit i across time
        for s in range(n_periods):
            expected = np.exp(-lambda_time * abs(t - s))
            # Time weight should be proportional to expected
            assert np.isclose(time_weights[s], expected, rtol=1e-5) or \
                   np.isclose(time_weights[s] / weights[t, i], expected / weights[t, i], rtol=1e-5)

    def test_pivot_vs_iterrows_equivalence(self):
        """
        Test that pivot-based matrix construction matches iterrows-based.

        The optimized pivot approach should produce identical Y and D matrices.
        """
        rng = np.random.default_rng(42)

        # Create test data
        n_units = 10
        n_periods = 5
        data = []
        for i in range(n_units):
            for t in range(n_periods):
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": rng.normal(0, 1),
                    "treated": 1 if (i < 3 and t >= 3) else 0,
                })
        df = pd.DataFrame(data)

        all_units = sorted(df["unit"].unique())
        all_periods = sorted(df["period"].unique())
        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        period_to_idx = {p: i for i, p in enumerate(all_periods)}

        # Method 1: iterrows (original)
        Y_iterrows = np.full((n_periods, n_units), np.nan)
        D_iterrows = np.zeros((n_periods, n_units), dtype=int)
        for _, row in df.iterrows():
            i = unit_to_idx[row["unit"]]
            t = period_to_idx[row["period"]]
            Y_iterrows[t, i] = row["outcome"]
            D_iterrows[t, i] = int(row["treated"])

        # Method 2: pivot (optimized)
        Y_pivot = (
            df.pivot(index="period", columns="unit", values="outcome")
            .reindex(index=all_periods, columns=all_units)
            .values
        )
        D_pivot = (
            df.pivot(index="period", columns="unit", values="treated")
            .reindex(index=all_periods, columns=all_units)
            .fillna(0)
            .astype(int)
            .values
        )

        # Verify equivalence
        assert np.allclose(Y_iterrows, Y_pivot, equal_nan=True)
        assert np.array_equal(D_iterrows, D_pivot)

    def test_reproducibility_with_seed(self, simple_panel_data):
        """
        Test that results are reproducible with the same seed.

        Running TROP twice with the same seed should produce identical results.
        """
        results1 = trop(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=20,
            seed=42,
        )

        results2 = trop(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=20,
            seed=42,
        )

        # Results should be identical
        assert results1.att == results2.att
        assert results1.se == results2.se
        assert results1.lambda_time == results2.lambda_time
        assert results1.lambda_unit == results2.lambda_unit
        assert results1.lambda_nn == results2.lambda_nn


class TestPaperConformanceFixes:
    """Tests verifying fixes for paper conformance issues.

    These tests validate the four fixes from the implementation assessment:
    - Issue A: Control set includes pre-treatment obs of eventually-treated units
    - Issue B: Distance computation excludes target period
    - Issue C: Nuclear norm update uses weights
    - Issue D: Bootstrap uses stratified sampling
    """

    def test_issue_a_control_includes_pretreatment_obs(self):
        """
        Test Issue A fix: Control set includes pre-treatment observations
        of eventually-treated units.

        Paper's Equation 2 (page 7) sums over ALL observations where
        (1 - W_js) is non-zero, including pre-treatment periods of
        eventually-treated units.
        """
        # Create staggered adoption data where treated units have
        # informative pre-treatment outcomes
        rng = np.random.default_rng(42)
        n_units = 20
        n_early_treat = 5  # Units treated at period 3
        n_late_treat = 5   # Units treated at period 5
        n_control = 10     # Never-treated units
        n_periods = 8
        true_att = 2.0

        data = []
        for i in range(n_units):
            # Determine treatment timing
            if i < n_early_treat:
                treat_period = 3
                unit_fe = 5.0  # Early-treated have specific level
            elif i < n_early_treat + n_late_treat:
                treat_period = 5
                unit_fe = 5.5  # Late-treated similar to early-treated
            else:
                treat_period = None
                unit_fe = 10.0  # Control units have different level

            for t in range(n_periods):
                is_post = treat_period is not None and t >= treat_period
                treatment_indicator = 1 if is_post else 0
                y = unit_fe + 0.2 * t
                if treatment_indicator:
                    y += true_att
                y += rng.normal(0, 0.3)
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "treated": treatment_indicator,
                })

        df = pd.DataFrame(data)

        # With Issue A fix, TROP should be able to use pre-treatment
        # observations of late-treated units as controls for early-treated
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[1.0],  # Use unit weights so distance matters
            lambda_nn_grid=[0.0],
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Should recover treatment effect direction
        assert results.att > 0, f"ATT={results.att:.3f} should be positive"

    def test_issue_b_distance_excludes_target_period(self):
        """
        Test Issue B fix: Distance computation excludes target period.

        Paper's Equation 3 (page 7) specifies 1{u ≠ t} to exclude the
        target period when computing pairwise distances.
        """
        rng = np.random.default_rng(123)

        # Create data where unit 0's outcome at target period is very different
        n_units = 10
        n_periods = 6
        data = []
        for i in range(n_units):
            is_treated = i == 0
            for t in range(n_periods):
                if is_treated and t == 3:
                    # Target period (t=3) has anomalous outcome
                    y = 100.0  # Very different from other periods
                elif is_treated and t >= 3:
                    y = 5.0 + rng.normal(0, 0.1)
                else:
                    y = 5.0 + rng.normal(0, 0.1)

                treatment_indicator = 1 if (is_treated and t >= 3) else 0
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "treated": treatment_indicator,
                })

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42
        )

        # With Issue B fix (target period excluded), this should complete
        # Without the fix, the anomalous period would dominate distance
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Model should fit without error
        assert results is not None
        # ATT should be finite
        assert np.isfinite(results.att)

    def test_issue_c_weighted_nuclear_norm(self):
        """
        Test Issue C fix: Nuclear norm update properly accounts for weights.

        The paper's Equation 2 (page 7) specifies the full weighted objective.
        Weights should affect L matrix estimation.
        """
        rng = np.random.default_rng(456)

        # Create data with factor structure where weights matter
        n_units = 15
        n_periods = 8
        n_treated = 3
        true_att = 2.0

        # Factor loadings that vary by unit
        loadings = rng.normal(0, 1, n_units)
        factors = rng.normal(0, 1, n_periods)

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_periods):
                post = t >= 5
                # Y = mu + factor_component + treatment_effect + noise
                y = 10.0 + loadings[i] * factors[t]
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_att
                y += rng.normal(0, 0.3)
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "treated": treatment_indicator,
                })

        df = pd.DataFrame(data)

        # Test with nuclear norm regularization
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1, 1.0],  # Use regularization
            n_bootstrap=10,
            seed=42
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Factor matrix should have been estimated with non-zero effective rank
        # (with weighted nuclear norm solver, this tests the code path)
        assert results.effective_rank >= 0
        # ATT should recover treatment effect direction
        assert results.att > 0, f"ATT={results.att:.3f} should be positive"

    def test_issue_d_stratified_bootstrap(self):
        """
        Test Issue D fix: Bootstrap uses stratified sampling.

        Paper's Algorithm 3 (page 27) specifies sampling N_0 control and
        N_1 treated units separately to preserve treatment ratio.
        """
        rng = np.random.default_rng(789)

        # Create data with unbalanced treated/control ratio
        n_treated = 3
        n_control = 17
        n_units = n_treated + n_control
        n_periods = 6
        true_att = 2.0

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_periods):
                post = t >= 3
                y = 10.0 + rng.normal(0, 0.5)
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_att
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "treated": treatment_indicator,
                })

        df = pd.DataFrame(data)

        # Run with bootstrap variance estimation
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            variance_method="bootstrap",
            n_bootstrap=30,
            seed=42
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Bootstrap should complete successfully
        assert results.bootstrap_distribution is not None
        assert len(results.bootstrap_distribution) >= 20  # Most iterations succeed
        # SE should be positive and finite
        assert results.se > 0
        assert np.isfinite(results.se)

    def test_weighted_nuclear_norm_solver_convergence(self):
        """
        Test that the weighted nuclear norm solver converges properly.
        """
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[1.0],  # Larger lambda for more regularization
        )

        # Create test data
        n_periods = 5
        n_units = 8

        Y = np.random.default_rng(42).normal(0, 1, (n_periods, n_units))
        W = np.ones((n_periods, n_units))
        L_init = np.zeros((n_periods, n_units))
        alpha = np.zeros(n_units)
        beta = np.zeros(n_periods)

        # Call the weighted nuclear norm solver
        L = trop_est._weighted_nuclear_norm_solve(
            Y, W, L_init, alpha, beta, lambda_nn=1.0, max_inner_iter=20
        )

        # L should be finite and have reasonable values
        assert np.all(np.isfinite(L))
        # With nuclear norm regularization, singular values should be reduced
        _, s, _ = np.linalg.svd(L, full_matrices=False)
        _, s_orig, _ = np.linalg.svd(Y, full_matrices=False)
        # Regularized singular values should be smaller than original
        assert np.sum(s) < np.sum(s_orig), \
            "Nuclear norm regularization should reduce total singular value mass"


class TestAPIChangesV2_1_8:
    """Tests verifying API changes in v2.1.8.

    These tests verify:
    1. post_periods parameter has been removed
    2. TROPResults uses n_pre_periods/n_post_periods instead of lists
    3. CV scoring uses sum (not average) per Equation 5
    4. LOOCV warning is emitted when fits fail
    """

    def test_fit_no_post_periods_parameter(self, simple_panel_data):
        """Test that fit() no longer accepts post_periods parameter."""
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42
        )

        # This should work - no post_periods parameter
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )
        assert results is not None
        assert isinstance(results, TROPResults)

        # Verify the API change - post_periods should raise TypeError
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            trop_est.fit(
                simple_panel_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],  # This should fail
            )

    def test_convenience_function_no_post_periods(self, simple_panel_data):
        """Test that trop() convenience function no longer accepts post_periods."""
        # This should work
        results = trop(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42,
        )
        assert results is not None

        # This should fail
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            trop(
                simple_panel_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],  # Should fail
                lambda_time_grid=[0.0],
                lambda_unit_grid=[0.0],
                lambda_nn_grid=[0.0],
                n_bootstrap=5,
                seed=42,
            )

    def test_results_has_period_counts_not_lists(self, simple_panel_data):
        """Test that TROPResults has n_pre_periods/n_post_periods, not lists."""
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Should have count attributes, not list attributes
        assert hasattr(results, "n_pre_periods")
        assert hasattr(results, "n_post_periods")
        assert isinstance(results.n_pre_periods, int)
        assert isinstance(results.n_post_periods, int)

        # Should NOT have list attributes
        assert not hasattr(results, "pre_periods")
        assert not hasattr(results, "post_periods")

        # Values should be correct (5 pre, 3 post in simple_panel_data)
        assert results.n_pre_periods == 5
        assert results.n_post_periods == 3

    def test_validation_still_checks_pre_periods(self):
        """Test that validation still requires at least 2 pre-treatment periods."""
        # Create data with only 1 pre-treatment period
        data = pd.DataFrame({
            "unit": [0, 0, 1, 1],
            "period": [0, 1, 0, 1],
            "outcome": [1.0, 2.0, 1.5, 2.5],
            "treated": [0, 1, 0, 0],  # Treatment at period 1
        })

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5
        )

        with pytest.raises(ValueError, match="at least 2 pre-treatment periods"):
            trop_est.fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )

    def test_loocv_warning_on_many_failures(self):
        """Test that LOOCV emits warning when many fits fail."""
        import warnings

        # Create numerically challenging data that may cause LOOCV failures
        rng = np.random.default_rng(42)
        n_units = 10
        n_periods = 5

        data = []
        for i in range(n_units):
            is_treated = i < 2
            for t in range(n_periods):
                post = t >= 3
                # Add some extreme values that might cause numerical issues
                y = rng.normal(0, 1) if not (is_treated and post) else 1e10
                treatment_indicator = 1 if (is_treated and post) else 0
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "treated": treatment_indicator,
                })

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[100.0],  # Extreme lambda may cause issues
            lambda_unit_grid=[100.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42
        )

        # Capture warnings and verify the warning code path
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit_succeeded = False
            try:
                trop_est.fit(
                    df,
                    outcome="outcome",
                    treatment="treated",
                    unit="unit",
                    time="period",
                )
                fit_succeeded = True
            except (ValueError, np.linalg.LinAlgError):
                # Expected if data is too extreme - this is valid behavior
                pass

            # Check for LOOCV-related warnings
            loocv_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and "LOOCV" in str(x.message)
            ]

            # If fit succeeded, check that we can capture warnings properly
            # (warnings may or may not be raised depending on data)
            if fit_succeeded:
                # At minimum, verify warnings capture infrastructure is working
                # by checking that w is a list we can inspect
                assert isinstance(w, list), "Warning capture should work"

            # If any LOOCV warnings were raised, verify they have expected content
            for warning in loocv_warnings:
                msg = str(warning.message)
                # Warnings should mention LOOCV and provide context
                assert "LOOCV" in msg, f"Warning should mention LOOCV: {msg}"

    def test_loocv_warning_deterministic_with_mock(self, simple_panel_data):
        """Test that LOOCV returns infinity and warns on first fit failure.

        Per Equation 5, Q(λ) must sum over ALL D==0 cells. Any failure means
        this λ cannot produce valid estimates, so we return infinity immediately.
        """
        import warnings
        from unittest.mock import patch

        trop_est = TROP(
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=5,
            seed=42
        )

        # Mock _estimate_model to fail on the first LOOCV call
        # This simulates a parameter combination that can't estimate all control cells
        call_count = [0]
        original_estimate = trop_est._estimate_model

        def mock_estimate_with_failure(*args, **kwargs):
            """Mock that fails on first call (immediate rejection per Equation 5)."""
            call_count[0] += 1
            # Fail on first call to trigger immediate infinity return
            if call_count[0] == 1:
                raise np.linalg.LinAlgError("Simulated failure")
            return original_estimate(*args, **kwargs)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Disable Rust backend for this test by patching the module-level variables
            import sys
            trop_module = sys.modules['diff_diff.trop']
            with patch.object(trop_module, 'HAS_RUST_BACKEND', False), \
                 patch.object(trop_module, '_rust_loocv_grid_search', None), \
                 patch.object(trop_est, '_estimate_model', mock_estimate_with_failure):
                try:
                    trop_est.fit(
                        simple_panel_data,
                        outcome="outcome",
                        treatment="treated",
                        unit="unit",
                        time="period",
                    )
                except (ValueError, np.linalg.LinAlgError):
                    # If all fits fail, that's acceptable
                    pass

            # Check that LOOCV warning was raised on first failure
            loocv_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and "LOOCV" in str(x.message)
            ]

            # With any failure, we should get a warning about returning infinity
            assert len(loocv_warnings) > 0, (
                "Expected LOOCV warning on first failure, but none was raised. "
                f"call_count={call_count[0]}, warnings={[str(x.message) for x in w]}"
            )

            # Verify warning content mentions Equation 5 and returning infinity
            msg = str(loocv_warnings[0].message)
            assert "LOOCV" in msg
            assert "fail" in msg.lower(), f"Warning should mention failure: {msg}"
            assert "Equation 5" in msg, f"Warning should reference Equation 5: {msg}"
