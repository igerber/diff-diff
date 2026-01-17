"""Tests for Triply Robust Panel (TROP) estimator."""

import numpy as np
import pandas as pd
import pytest

from diff_diff import SyntheticDiD
from diff_diff.trop import TROP, TROPResults, trop


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

    DGP: Y_it = mu + gamma_i + delta_t + Lambda_i'F_t + tau*D_it + eps_it
    """
    rng = np.random.default_rng(seed)

    n_control = n_units - n_treated
    n_periods = n_pre + n_post

    # Generate factors F: (n_periods, n_factors)
    F = rng.normal(0, 1, (n_periods, n_factors))

    # Generate loadings Lambda: (n_factors, n_units)
    Lambda = rng.normal(0, 1, (n_factors, n_units))
    Lambda[:, :n_treated] += 0.5

    # Unit fixed effects
    gamma = rng.normal(0, 1, n_units)
    gamma[:n_treated] += 1.0

    # Time fixed effects
    delta = np.linspace(0, 2, n_periods)

    # Generate outcomes
    data = []
    for i in range(n_units):
        is_treated = i < n_treated

        for t in range(n_periods):
            period = t
            post = t >= n_pre

            y = 10.0 + gamma[i] + delta[t]
            y += factor_strength * (Lambda[:, i] @ F[t, :])

            # Treatment effect only for treated units in post period
            treatment_indicator = 1 if (is_treated and post) else 0
            if treatment_indicator:
                y += treatment_effect

            y += rng.normal(0, noise_std)

            data.append({
                "unit": i,
                "period": period,
                "outcome": y,
                "treated": treatment_indicator,
            })

    return pd.DataFrame(data)


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
            post_periods=[5, 6, 7],
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
        post_periods = list(range(8, 12))
        results = trop_est.fit(
            factor_dgp_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=post_periods,
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
        post_periods = list(range(8, 12))
        results = trop_est.fit(
            factor_dgp_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=post_periods,
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
        )

        stars = results.significance_stars
        assert stars in ["", ".", "*", "**", "***"]


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
        post_periods = list(range(8, 12))

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
            post_periods=post_periods,
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
            post_periods=[5, 6, 7],
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
            post_periods=[5, 6, 7],
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0, 0.5],
            lambda_nn_grid=[0.0, 0.1],
            max_iter=50,
            n_bootstrap=10,
            seed=42,
        )

        assert isinstance(results, TROPResults)
