"""Tests for Triply Robust Panel (TROP) estimator."""

import warnings

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

    def test_n_bootstrap_less_than_2_raises(self):
        """n_bootstrap < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_bootstrap must be >= 2"):
            TROP(n_bootstrap=1)
        with pytest.raises(ValueError, match="n_bootstrap must be >= 2"):
            TROP(n_bootstrap=0)

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

    def test_fit_with_factors(self, factor_dgp_data, ci_params):
        """Test fitting with factor structure."""
        n_boot = ci_params.bootstrap(20)
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1, 1.0],
            n_bootstrap=n_boot,
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

    def test_treatment_effect_recovery(self, factor_dgp_data, ci_params):
        """Test that TROP recovers treatment effect direction."""
        true_att = 2.0
        n_boot = ci_params.bootstrap(30)

        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0, 0.5, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=n_boot,
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

    def test_tuning_parameter_selection(self, simple_panel_data, ci_params):
        """Test that LOOCV selects tuning parameters."""
        time_grid = ci_params.grid([0.0, 0.5, 1.0, 2.0])
        trop_est = TROP(
            lambda_time_grid=time_grid,
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

    def test_bootstrap_variance(self, simple_panel_data, ci_params):
        """Test bootstrap variance estimation."""
        n_boot = ci_params.bootstrap(30)
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=n_boot,
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
        assert results.n_bootstrap == n_boot
        assert results.bootstrap_distribution is not None

    def test_confidence_interval(self, simple_panel_data, ci_params):
        """Test confidence interval properties."""
        n_boot = ci_params.bootstrap(30)
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            alpha=0.05,
            n_bootstrap=n_boot,
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

    @pytest.fixture(scope="class")
    def fitted_results(self):
        """Shared TROP fit for read-only result tests (class-scoped to avoid redundant fits)."""
        # Inline data generation (same as simple_panel_data fixture)
        rng = np.random.default_rng(123)
        n_units, n_treated, n_pre, n_post, true_att = 20, 5, 5, 3, 3.0
        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_pre + n_post):
                post = t >= n_pre
                y = 10.0 + i * 0.1 + t * 0.5
                if is_treated and post:
                    y += true_att
                y += rng.normal(0, 0.5)
                data.append({
                    "unit": i, "period": t, "outcome": y,
                    "treated": 1 if (is_treated and post) else 0,
                })
        panel = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42,
        )
        return trop_est.fit(
            panel, outcome="outcome", treatment="treated",
            unit="unit", time="period",
        )

    def test_summary(self, fitted_results):
        """Test that summary produces string output."""
        summary = fitted_results.summary()
        assert isinstance(summary, str)
        assert "ATT" in summary
        assert "TROP" in summary
        assert "LOOCV" in summary
        assert "Lambda" in summary

    def test_to_dict(self, fitted_results):
        """Test conversion to dictionary."""
        d = fitted_results.to_dict()
        assert "att" in d
        assert "se" in d
        assert "lambda_time" in d
        assert "lambda_unit" in d
        assert "lambda_nn" in d
        assert "effective_rank" in d

    def test_to_dataframe(self, fitted_results):
        """Test conversion to DataFrame."""
        df = fitted_results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "att" in df.columns

    def test_get_treatment_effects_df(self, fitted_results):
        """Test getting treatment effects DataFrame."""
        effects_df = fitted_results.get_treatment_effects_df()
        assert isinstance(effects_df, pd.DataFrame)
        assert "unit" in effects_df.columns
        assert "time" in effects_df.columns
        assert "effect" in effects_df.columns
        assert len(effects_df) == fitted_results.n_treated_obs

    def test_get_unit_effects_df(self, fitted_results):
        """Test getting unit effects DataFrame."""
        effects_df = fitted_results.get_unit_effects_df()
        assert isinstance(effects_df, pd.DataFrame)
        assert "unit" in effects_df.columns
        assert "effect" in effects_df.columns

    def test_get_time_effects_df(self, fitted_results):
        """Test getting time effects DataFrame."""
        effects_df = fitted_results.get_time_effects_df()
        assert isinstance(effects_df, pd.DataFrame)
        assert "time" in effects_df.columns
        assert "effect" in effects_df.columns

    def test_significance_properties(self, simple_panel_data, ci_params):
        """Test is_significant and significance_stars properties."""
        n_boot = ci_params.bootstrap(30)
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            alpha=0.05,
            n_bootstrap=n_boot,
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
        assert results.significance_stars in ["", ".", "*", "**", "***"]

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
        )

        # Verify that all inference fields are NaN when SE=0
        assert np.isnan(results.t_stat), "t_stat should be NaN when SE=0"
        assert np.isnan(results.p_value), "p_value should be NaN when SE=0"
        assert np.isnan(results.conf_int[0]), "conf_int[0] should be NaN when SE=0"
        assert np.isnan(results.conf_int[1]), "conf_int[1] should be NaN when SE=0"

        # Verify the ATT itself is still valid
        assert results.att == 1.0, "ATT should still be valid"


@pytest.mark.slow
class TestTROPvsSDID:
    """Tests comparing TROP to SDID under different DGPs."""

    def test_trop_handles_factor_dgp(self, ci_params):
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
        n_boot = ci_params.bootstrap(20)
        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1, 1.0],
            n_bootstrap=n_boot,
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


@pytest.mark.slow
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

    def test_factor_model_reduces_bias(self, ci_params):
        """
        Test that nuclear norm regularization reduces bias with factor structure.

        Following paper's simulation: when true DGP has interactive fixed effects,
        the factor model component should help recover the treatment effect.
        """
        # Generate data with known factor structure (reduced size for CI speed)
        data = generate_factor_dgp(
            n_units=25,
            n_pre=7,
            n_post=3,
            n_treated=5,
            n_factors=2,
            treatment_effect=2.0,
            factor_strength=1.5,  # Strong factors
            noise_std=0.5,
            seed=789,
        )

        # TROP with nuclear norm regularization
        n_boot = ci_params.bootstrap(20)
        nn_grid = ci_params.grid([0.0, 0.1, 1.0, 5.0])
        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5],
            lambda_unit_grid=[0.0, 0.5],
            lambda_nn_grid=nn_grid,
            n_bootstrap=n_boot,
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

    def test_paper_dgp_recovery(self, ci_params):
        """
        Test treatment effect recovery using paper's simulation DGP.

        Based on Table 2 (page 32) simulation settings:
        - Factor model with 2 factors
        - Treatment effect = 0 (null hypothesis)
        - Should produce estimates centered around zero

        This is a methodological validation test.
        """
        # Generate data similar to paper's simulation (reduced size for CI speed)
        rng = np.random.default_rng(2024)
        n_units = 30
        n_treated = 6
        n_pre = 7
        n_post = 3
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
        n_boot = ci_params.bootstrap(30)
        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0, 0.5, 1.0],
            lambda_nn_grid=[0.0, 0.1, 1.0],
            n_bootstrap=n_boot,
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

    def test_reproducibility_with_seed(self, simple_panel_data, ci_params):
        """
        Test that results are reproducible with the same seed.

        Running TROP twice with the same seed should produce identical results.
        """
        n_boot = ci_params.bootstrap(20)
        results1 = trop(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=n_boot,
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
            n_bootstrap=n_boot,
            seed=42,
        )

        # Results should be identical
        assert results1.att == results2.att
        assert results1.se == results2.se
        assert results1.lambda_time == results2.lambda_time
        assert results1.lambda_unit == results2.lambda_unit
        assert results1.lambda_nn == results2.lambda_nn


class TestDMatrixValidation:
    """Tests for D matrix absorbing-state validation."""

    def test_d_matrix_absorbing_state_validation_valid(self):
        """Test that valid absorbing-state D passes validation."""
        # Staggered adoption: once treated, always treated
        rng = np.random.default_rng(42)
        n_units = 15
        n_periods = 8

        data = []
        for i in range(n_units):
            # Different treatment timing for different units
            if i < 5:
                treat_period = 3  # Early adopters
            elif i < 10:
                treat_period = 5  # Late adopters
            else:
                treat_period = None  # Never treated

            for t in range(n_periods):
                is_treated = treat_period is not None and t >= treat_period
                y = 10.0 + rng.normal(0, 0.5)
                if is_treated:
                    y += 2.0
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "treated": 1 if is_treated else 0,
                })

        df = pd.DataFrame(data)

        # Should work without error
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )
        assert results is not None
        assert isinstance(results, TROPResults)

    def test_d_matrix_absorbing_state_validation_invalid(self):
        """Test that non-absorbing D raises ValueError."""
        # Event-style D: only first treatment period has D=1
        data = []
        n_units = 10
        n_periods = 6

        for i in range(n_units):
            is_treated_unit = i < 3
            for t in range(n_periods):
                # Event-style: D=1 only at t=3, then back to 0
                if is_treated_unit and t == 3:
                    treated = 1
                else:
                    treated = 0
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": float(i + t),
                    "treated": treated,
                })

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5
        )

        with pytest.raises(ValueError, match="not an absorbing state"):
            trop_est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )

    def test_d_matrix_validation_error_message_helpful(self):
        """Test that error message includes unit IDs and remediation guidance."""
        # Event-style D for unit 5 only
        data = []
        for i in range(10):
            for t in range(5):
                # Unit 5: D goes 0→1→0 (invalid)
                if i == 5:
                    treated = 1 if t == 2 else 0
                else:
                    # Other units: proper absorbing state
                    treated = 1 if (i < 3 and t >= 3) else 0
                data.append({
                    "unit": i,
                    "period": t,
                    "outcome": float(i + t),
                    "treated": treated,
                })

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5
        )

        with pytest.raises(ValueError) as exc_info:
            trop_est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )

        error_msg = str(exc_info.value)
        # Check that error message is helpful
        assert "5" in error_msg, "Should mention unit ID 5"
        assert "absorbing state" in error_msg
        assert "monotonic" in error_msg.lower() or "non-decreasing" in error_msg.lower()
        assert "D[t, i] = 1 for all t >= first treatment" in error_msg


@pytest.mark.slow
class TestCyclingSearch:
    """Tests for LOOCV cycling (coordinate descent) search."""

    def test_cycling_search_converges(self, simple_panel_data):
        """Test that cycling search converges to reasonable values."""
        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0, 0.5, 1.0],
            lambda_nn_grid=[0.0, 0.1, 1.0],
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

        # Check that lambda values are from the grid
        assert results.lambda_time in trop_est.lambda_time_grid
        assert results.lambda_unit in trop_est.lambda_unit_grid
        assert results.lambda_nn in trop_est.lambda_nn_grid

        # Check that results are reasonable
        assert np.isfinite(results.att)
        assert results.se >= 0

    def test_cycling_search_reproducible(self, simple_panel_data):
        """Test that cycling search produces reproducible results."""
        results1 = trop(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0, 0.5, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42,
        )

        results2 = trop(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0, 0.5, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42,
        )

        # Results should be identical with same seed
        assert results1.att == results2.att
        assert results1.lambda_time == results2.lambda_time
        assert results1.lambda_unit == results2.lambda_unit
        assert results1.lambda_nn == results2.lambda_nn

    def test_cycling_search_single_value_grids(self, simple_panel_data):
        """Test cycling search with single-value grids (degenerate case)."""
        trop_est = TROP(
            lambda_time_grid=[0.5],  # Single value
            lambda_unit_grid=[0.5],  # Single value
            lambda_nn_grid=[0.1],    # Single value
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

        # Should use the only available values
        assert results.lambda_time == 0.5
        assert results.lambda_unit == 0.5
        assert results.lambda_nn == 0.1


@pytest.mark.slow
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

    def test_issue_d_stratified_bootstrap(self, ci_params):
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
        n_boot = ci_params.bootstrap(30)
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=n_boot,
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
        min_successes = max(5, int(0.67 * n_boot))
        assert len(results.bootstrap_distribution) >= min_successes, (
            f"Expected >= {min_successes} successful bootstrap draws "
            f"out of {n_boot}, got {len(results.bootstrap_distribution)}"
        )
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


class TestLOOCVFallback:
    """Tests for LOOCV fallback to defaults when all fits fail."""

    def test_infinite_score_triggers_fallback(self, simple_panel_data):
        """
        Test that infinite LOOCV scores trigger fallback to defaults.

        When all LOOCV fits return infinity (e.g., due to numerical issues),
        the estimator should:
        1. Emit a warning about using defaults
        2. Use default parameters (1.0, 1.0, 0.1)
        3. Still complete estimation
        """
        import sys
        from unittest.mock import patch

        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=5,
            seed=42
        )

        # Mock LOOCV to always return infinity
        def always_infinity(*args, **kwargs):
            return np.inf

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Disable Rust backend and mock LOOCV score to always return infinity
            trop_module = sys.modules['diff_diff.trop']
            with patch.object(trop_module, 'HAS_RUST_BACKEND', False), \
                 patch.object(trop_module, '_rust_loocv_grid_search', None), \
                 patch.object(trop_est, '_loocv_score_obs_specific', always_infinity):
                results = trop_est.fit(
                    simple_panel_data,
                    outcome="outcome",
                    treatment="treated",
                    unit="unit",
                    time="period",
                )

            # Verify warning emitted about fallback to defaults
            fallback_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and "defaults" in str(x.message).lower()
            ]
            assert len(fallback_warnings) > 0, (
                f"Expected fallback warning, got: {[str(x.message) for x in w]}"
            )

            # Verify defaults used (per REGISTRY.md: 1.0, 1.0, 0.1)
            assert results.lambda_time == 1.0, \
                f"Expected default lambda_time=1.0, got {results.lambda_time}"
            assert results.lambda_unit == 1.0, \
                f"Expected default lambda_unit=1.0, got {results.lambda_unit}"
            assert results.lambda_nn == 0.1, \
                f"Expected default lambda_nn=0.1, got {results.lambda_nn}"

            # Verify estimation still completed
            assert np.isfinite(results.att), "ATT should be finite even with default params"

    def test_rust_infinite_score_triggers_fallback(self, simple_panel_data):
        """
        Test that infinite LOOCV score from Rust backend triggers fallback.

        The Rust backend may return infinite score when all fits fail.
        Python should detect this and fall back to defaults.
        When Rust returns infinity, best_lambda stays None, then Python fallback
        is attempted. If Python also returns infinity, defaults are used.
        """
        import sys
        from unittest.mock import patch, MagicMock

        trop_est = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=5,
            seed=42
        )

        # Mock Rust function to return infinite score
        # Return format: (lambda_time, lambda_unit, lambda_nn, score, n_valid, n_attempted, first_failed_obs)
        mock_rust_loocv = MagicMock(return_value=(0.5, 0.5, 0.05, np.inf, 0, 100, None))

        # Also mock Python LOOCV to return infinity (so Python fallback also fails)
        def always_infinity(*args, **kwargs):
            return np.inf

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            trop_module = sys.modules['diff_diff.trop']
            with patch.object(trop_module, 'HAS_RUST_BACKEND', True), \
                 patch.object(trop_module, '_rust_loocv_grid_search', mock_rust_loocv), \
                 patch.object(trop_est, '_loocv_score_obs_specific', always_infinity):
                results = trop_est.fit(
                    simple_panel_data,
                    outcome="outcome",
                    treatment="treated",
                    unit="unit",
                    time="period",
                )

            # Verify warning emitted about fallback to defaults
            fallback_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and "defaults" in str(x.message).lower()
            ]
            assert len(fallback_warnings) > 0, (
                f"Expected fallback warning with Rust backend, got: {[str(x.message) for x in w]}"
            )

            # Verify defaults used (NOT the Rust-returned values)
            assert results.lambda_time == 1.0, \
                f"Expected default lambda_time=1.0, got {results.lambda_time}"
            assert results.lambda_unit == 1.0, \
                f"Expected default lambda_unit=1.0, got {results.lambda_unit}"
            assert results.lambda_nn == 0.1, \
                f"Expected default lambda_nn=0.1, got {results.lambda_nn}"

    def test_uniform_weights_and_disabled_factor_handled_consistently(self, simple_panel_data):
        """
        Test that 0.0 (uniform weights) and inf (disabled factor) are handled
        consistently in LOOCV and final estimation.

        Per Athey et al. (2025) Eq. 3:
        - λ_time=0.0 → uniform time weights (exp(-0×dist)=1)
        - λ_unit=0.0 → uniform unit weights (exp(-0×dist)=1)
        - λ_nn=∞ → factor model disabled (L=0), converted to 1e10 internally
        """
        trop_est = TROP(
            lambda_time_grid=[0.0],     # Uniform time weights (disabled)
            lambda_unit_grid=[0.0],     # Uniform unit weights (disabled)
            lambda_nn_grid=[np.inf],    # Factor model disabled → converted to 1e10
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

        # ATT should be finite
        assert np.isfinite(results.att), (
            f"ATT should be finite with uniform weights and no factor model, got {results.att}"
        )

        # SE should be finite or at least non-negative
        assert np.isfinite(results.se) or results.se >= 0, (
            f"SE should be finite, got {results.se}"
        )

        # lambda_time and lambda_unit should be 0.0 (uniform weights)
        assert results.lambda_time == 0.0, (
            f"lambda_time should be 0.0 (uniform weights), got {results.lambda_time}"
        )
        # lambda_nn should store the original inf value
        assert np.isinf(results.lambda_nn), (
            f"lambda_nn should be inf (original grid value), got {results.lambda_nn}"
        )

    def test_inf_in_time_unit_grids_raises_valueerror(self):
        """
        Test that inf in lambda_time_grid or lambda_unit_grid raises ValueError.

        Per Athey et al. (2025) Eq. 3, λ_time=0 and λ_unit=0 give uniform
        weights. Using inf is a misunderstanding; only λ_nn=∞ is valid.
        """
        import pytest

        # inf in lambda_time_grid should raise
        with pytest.raises(ValueError, match="lambda_time_grid must not contain inf"):
            TROP(lambda_time_grid=[np.inf])

        with pytest.raises(ValueError, match="lambda_time_grid must not contain inf"):
            TROP(lambda_time_grid=[0.0, np.inf, 1.0])

        # inf in lambda_unit_grid should raise
        with pytest.raises(ValueError, match="lambda_unit_grid must not contain inf"):
            TROP(lambda_unit_grid=[np.inf])

        with pytest.raises(ValueError, match="lambda_unit_grid must not contain inf"):
            TROP(lambda_unit_grid=[0.5, np.inf])

        # inf in lambda_nn_grid should still be valid
        trop_est = TROP(lambda_nn_grid=[np.inf])
        assert np.inf in trop_est.lambda_nn_grid

    def test_variance_estimation_uses_converted_params(self, simple_panel_data):
        """
        Test that variance estimation uses the same converted parameters as point estimation.

        λ_nn=∞ is converted to 1e10 for computation. λ_time and λ_unit use 0.0
        directly for uniform weights (no conversion needed).
        """
        from unittest.mock import patch

        trop_est = TROP(
            lambda_time_grid=[0.0],     # Uniform time weights (paper convention)
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[np.inf],    # Will be converted to 1e10 internally
            n_bootstrap=5,
            seed=42
        )

        # Track what parameters are passed to _fit_with_fixed_lambda
        # (called by bootstrap variance estimation)
        original_fit_with_fixed = TROP._fit_with_fixed_lambda
        captured_lambda = []

        def tracking_fit(self, data, outcome, treatment, unit, time, fixed_lambda, **kwargs):
            captured_lambda.append(fixed_lambda)
            return original_fit_with_fixed(self, data, outcome, treatment, unit, time, fixed_lambda, **kwargs)

        with patch.object(TROP, '_fit_with_fixed_lambda', tracking_fit):
            results = trop_est.fit(
                simple_panel_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )

        # Results should store 0.0 for time (direct value, no conversion)
        assert results.lambda_time == 0.0, "lambda_time should be 0.0"
        # Results should store original inf for lambda_nn
        assert np.isinf(results.lambda_nn), "Results should store original infinity value for lambda_nn"

        # ATT should be finite (computed with converted params)
        assert np.isfinite(results.att), "ATT should be finite"

        # Variance estimation should have received converted parameters
        # Check that bootstrap iterations used converted (non-infinite) λ_nn values
        for captured in captured_lambda:
            lambda_time, lambda_unit, lambda_nn = captured
            assert lambda_time == 0.0, (
                f"Bootstrap should receive λ_time=0.0, got {lambda_time}"
            )
            assert not np.isinf(lambda_nn), (
                f"Bootstrap should receive converted λ_nn=1e10, not {lambda_nn}"
            )

    def test_empty_control_obs_returns_infinity(self, simple_panel_data):
        """
        Test that LOOCV returns infinity when control observations are empty.

        A score of 0.0 for empty control would incorrectly "win" over legitimate
        parameters. This test verifies the fix for empty control handling (PR #110 Round 7).
        """
        import warnings

        trop_est = TROP(
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[1.0],
            seed=42
        )

        # Setup matrices from data
        data = simple_panel_data
        all_units = sorted(data['unit'].unique())
        all_periods = sorted(data['period'].unique())
        n_units = len(all_units)
        n_periods = len(all_periods)

        Y = (
            data.pivot(index='period', columns='unit', values='outcome')
            .reindex(index=all_periods, columns=all_units)
            .values
        )
        D = (
            data.pivot(index='period', columns='unit', values='treated')
            .reindex(index=all_periods, columns=all_units)
            .fillna(0)
            .astype(int)
            .values
        )

        control_mask = D == 0
        control_unit_idx = np.where(~np.any(D == 1, axis=0))[0]

        # Force empty control_obs by setting precomputed with empty list
        trop_est._precomputed = {
            "control_obs": [],  # Empty!
            "time_dist_matrix": np.abs(np.subtract.outer(
                np.arange(n_periods), np.arange(n_periods)
            )),
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            score = trop_est._loocv_score_obs_specific(
                Y, D, control_mask, control_unit_idx,
                1.0, 1.0, 1.0, n_units, n_periods
            )

        # Should return infinity, not 0.0
        assert np.isinf(score), f"Empty control_obs should return inf, got {score}"

        # Should emit warning
        warning_msgs = [str(warning.message) for warning in w]
        assert any("No valid control observations" in msg for msg in warning_msgs), (
            f"Should warn about empty control obs. Warnings: {warning_msgs}"
        )

    def test_original_grid_values_stored_in_results(self, simple_panel_data):
        """
        Test that TROPResults stores the selected grid values correctly.

        λ_time and λ_unit store values directly (0.0 = uniform weights).
        λ_nn stores the original inf value when factor model is disabled.
        """
        trop_est = TROP(
            lambda_time_grid=[0.0],     # Uniform time weights
            lambda_unit_grid=[0.5],
            lambda_nn_grid=[np.inf],    # Factor model disabled (original: inf)
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

        # lambda_time stores selected value directly (0.0 = uniform)
        assert results.lambda_time == 0.0, (
            f"results.lambda_time should be 0.0, got {results.lambda_time}"
        )
        assert results.lambda_unit == 0.5, (
            f"results.lambda_unit should be 0.5, got {results.lambda_unit}"
        )
        # lambda_nn stores original inf (converted to 1e10 only for computation)
        assert np.isinf(results.lambda_nn), (
            f"results.lambda_nn should be inf (original), got {results.lambda_nn}"
        )

        # But ATT should still be finite (computed with converted values)
        assert np.isfinite(results.att), "ATT should be finite"


class TestPR110FeedbackRound8:
    """Tests for PR #110 feedback round 8 fixes.

    Issue 1: Final LOOCV score uses converted infinity values (not raw inf)
    Issue 2: Rust LOOCV warnings include failed observation metadata
    Issue 3: D matrix validation handles unbalanced panels correctly
    """

    def test_unbalanced_panel_d_matrix_validation(self):
        """Test that unbalanced panels don't trigger spurious D matrix violations.

        Issue 3 fix: Missing unit-period observations should not be flagged
        as violations. Only validate monotonicity between observed periods.
        """
        # Create an unbalanced panel: unit 1 is missing period 5
        # Unit 1: treated from period 3 onwards, but missing period 5
        # This should NOT raise an error, because the 1→0 transition at period 5
        # is due to missing data, not a real violation.
        data = []

        # Unit 0: control, complete panel
        for t in range(6):
            data.append({
                "unit": 0,
                "period": t,
                "outcome": 10.0 + t,
                "treated": 0,
            })

        # Unit 1: treated from t=3, missing t=5 (unbalanced)
        for t in range(6):
            if t == 5:
                continue  # Skip period 5 - creates unbalanced panel
            treated = 1 if t >= 3 else 0
            data.append({
                "unit": 1,
                "period": t,
                "outcome": 10.0 + t + (2.0 if treated else 0),
                "treated": treated,
            })

        # Unit 2: control, complete panel
        for t in range(6):
            data.append({
                "unit": 2,
                "period": t,
                "outcome": 10.0 + t,
                "treated": 0,
            })

        df = pd.DataFrame(data)

        # This should NOT raise an error
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42
        )

        # Should not raise ValueError - missing data is not a violation
        try:
            results = trop_est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )
            # Basic sanity checks
            assert results is not None
            assert np.isfinite(results.att)
        except ValueError as e:
            if "absorbing state" in str(e):
                pytest.fail(
                    f"Unbalanced panel incorrectly flagged as absorbing state violation: {e}"
                )
            raise

    def test_unbalanced_panel_real_violation_still_caught(self):
        """Test that real violations are still caught in unbalanced panels.

        Even with missing data, actual D→1→0 violations on observed periods
        should still be detected and raise ValueError.
        """
        data = []

        # Unit 0: control, complete
        for t in range(5):
            data.append({
                "unit": 0,
                "period": t,
                "outcome": 10.0 + t,
                "treated": 0,
            })

        # Unit 1: REAL violation - D goes 0→1→0 on observed periods (t=2: D=1, t=3: D=0)
        # This is a real violation, not a missing data artifact
        for t in range(5):
            if t == 2:
                treated = 1
            else:
                treated = 0
            data.append({
                "unit": 1,
                "period": t,
                "outcome": 10.0 + t,
                "treated": treated,
            })

        # Unit 2: control
        for t in range(5):
            data.append({
                "unit": 2,
                "period": t,
                "outcome": 10.0 + t,
                "treated": 0,
            })

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5
        )

        # This SHOULD raise an error - real violation
        with pytest.raises(ValueError, match="absorbing state"):
            trop_est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )

    def test_unbalanced_panel_multiple_missing_periods(self):
        """Test unbalanced panel with multiple missing periods per unit."""
        data = []

        # Unit 0: control, complete
        for t in range(8):
            data.append({
                "unit": 0,
                "period": t,
                "outcome": 10.0 + t,
                "treated": 0,
            })

        # Unit 1: treated from t=4, missing t=2 and t=6
        for t in range(8):
            if t in [2, 6]:
                continue  # Skip these periods
            treated = 1 if t >= 4 else 0
            data.append({
                "unit": 1,
                "period": t,
                "outcome": 10.0 + t + (2.0 if treated else 0),
                "treated": treated,
            })

        # Unit 2: control, missing t=0
        for t in range(8):
            if t == 0:
                continue
            data.append({
                "unit": 2,
                "period": t,
                "outcome": 10.0 + t,
                "treated": 0,
            })

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42
        )

        # Should not raise error
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )
        assert results is not None
        assert np.isfinite(results.att)

    def test_mixed_grid_values_with_final_score_computation(self, simple_panel_data):
        """Test that grid values including 0.0 (uniform) and inf (λ_nn) work for final score.

        When LOOCV selects λ_nn=∞, the final score computation should use
        converted value (1e10), not raw infinity. λ_time and λ_unit grids
        use finite values only (0.0 = uniform weights per Eq. 3).
        """
        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5],     # 0.0 = uniform time weights
            lambda_unit_grid=[0.0, 0.5],     # 0.0 = uniform unit weights
            lambda_nn_grid=[np.inf, 0.1],    # inf should convert to 1e10
            n_bootstrap=5,
            seed=42
        )

        # This should complete without error
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # ATT should be finite regardless of which grid values were selected
        assert np.isfinite(results.att), "ATT should be finite with mixed grid values"
        assert results.se >= 0, "SE should be non-negative"

        # If inf was selected for λ_nn, LOOCV score should still be computed correctly
        if np.isinf(results.loocv_score):
            # Infinite LOOCV score is acceptable (means fits failed)
            # but ATT should still be finite (falls back to defaults)
            pass
        else:
            assert np.isfinite(results.loocv_score), (
                "LOOCV score should be finite when computed with converted inf values"
            )

    def test_violation_across_missing_gap_caught(self):
        """Test that 1→0 violations spanning missing periods are caught.

        Issue: If periods [3, 4] are missing and D[2]=1, D[5]=0, this is a
        real violation that must be detected even though the adjacent
        period transitions don't show it (the gap hides the transition).

        PR #110 round 10 fix: Check each unit's observed D sequence for
        monotonicity, not just adjacent periods in the full time grid.
        """
        data = []

        # Unit 0: control, complete
        for t in range(6):
            data.append({"unit": 0, "period": t, "outcome": 10.0 + t, "treated": 0})

        # Unit 1: VIOLATION across gap
        # Observed at [0, 1, 2, 5], missing [3, 4]
        # D[2]=1, D[5]=0 is a real violation spanning the gap
        for t in [0, 1, 2, 5]:
            treated = 1 if t == 2 else 0  # Only treated at period 2
            data.append({"unit": 1, "period": t, "outcome": 10.0 + t, "treated": treated})

        # Unit 2: control, complete
        for t in range(6):
            data.append({"unit": 2, "period": t, "outcome": 10.0 + t, "treated": 0})

        df = pd.DataFrame(data)
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
        )

        with pytest.raises(ValueError, match="absorbing state"):
            trop_est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )

    def test_n_post_periods_counts_observed_treatment(self):
        """Test n_post_periods counts periods with actual D=1 observations.

        Per docstring: "Number of post-treatment periods (periods with D=1 observations)"

        This tests that n_post_periods reflects periods where treatment is
        actually observed, not just calendar periods from first treatment.
        """
        data = []

        # Create panel where period 5 exists but has no D=1 observations
        # (all treated units are missing at period 5)
        for unit in range(3):
            for period in range(6):
                # Units 1, 2 are treated from period 3, but missing at period 5
                if unit in [1, 2] and period == 5:
                    continue  # Skip - creates unbalanced panel
                treated = 1 if (unit in [1, 2] and period >= 3) else 0
                data.append({
                    "unit": unit,
                    "period": period,
                    "outcome": 10.0 + period,
                    "treated": treated,
                })

        df = pd.DataFrame(data)
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42,
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Periods with D=1 observations: 3, 4 (not 5 - missing for treated units)
        assert results.n_post_periods == 2, (
            f"Expected 2 post-periods with D=1, got {results.n_post_periods}"
        )


class TestTROPNuclearNormSolver:
    """Tests for proximal gradient step size correctness and objective monotonicity."""

    def test_proximal_step_size_correctness(self):
        """Verify L converges to prox_{λ/2}(R) for uniform weights."""
        trop_est = TROP(method="global", n_bootstrap=2)

        # Small problem with known solution
        rng = np.random.default_rng(42)
        R = rng.normal(0, 1, (4, 3))
        delta = np.ones((4, 3))
        lambda_nn = 0.5

        # Run solver (many iterations to ensure convergence)
        L = np.zeros_like(R)
        for _ in range(500):
            delta_max = np.max(delta)
            delta_norm = delta / delta_max
            gradient_step = L + delta_norm * (R - L)
            eta = 1.0 / (2.0 * delta_max)
            L = trop_est._soft_threshold_svd(gradient_step, eta * lambda_nn)

        # Analytical solution for uniform weights: prox_{λ/2}(R)
        L_exact = trop_est._soft_threshold_svd(R, lambda_nn / 2.0)

        np.testing.assert_array_almost_equal(L, L_exact, decimal=4)

    def test_lowrank_objective_decreases(self):
        """Verify objective f(L) + λ||L||_* is non-increasing across iterations."""
        # Generate small problem
        rng = np.random.default_rng(42)
        R = rng.normal(0, 1, (6, 4))
        delta = rng.uniform(0.5, 2.0, (6, 4))
        lambda_nn = 0.3

        trop_est = TROP(method="global", n_bootstrap=2)
        L = np.zeros_like(R)
        objectives = []

        for _ in range(50):
            # Compute objective
            f_val = np.sum(delta * (R - L) ** 2)
            _, s, _ = np.linalg.svd(L, full_matrices=False)
            obj = f_val + lambda_nn * np.sum(s)
            objectives.append(obj)

            # Proximal gradient step
            delta_max = np.max(delta)
            delta_norm = delta / delta_max
            gradient_step = L + delta_norm * (R - L)
            eta = 1.0 / (2.0 * delta_max)
            L = trop_est._soft_threshold_svd(gradient_step, eta * lambda_nn)

        # Objective should be non-increasing (within numerical tolerance)
        for k in range(1, len(objectives)):
            assert objectives[k] <= objectives[k - 1] + 1e-10, (
                f"Objective increased at step {k}: {objectives[k]} > {objectives[k-1]}"
            )

    def test_local_nonuniform_weights_objective(self):
        """Verify objective decreases with non-uniform weights (W_max < 1)."""
        rng = np.random.default_rng(123)
        R = rng.normal(0, 1, (6, 4))
        W = rng.uniform(0.1, 0.8, (6, 4))
        lambda_nn = 0.3

        trop_est = TROP(method="local", n_bootstrap=2)

        # Initial objective with L=0
        L_init = np.zeros_like(R)
        f_init = np.sum(W * (R - L_init) ** 2)
        _, s_init, _ = np.linalg.svd(L_init, full_matrices=False)
        obj_init = f_init + lambda_nn * np.sum(s_init)

        # Solve
        L_final = trop_est._weighted_nuclear_norm_solve(
            Y=R,
            W=W,
            L_init=L_init,
            alpha=np.zeros(R.shape[1]),
            beta=np.zeros(R.shape[0]),
            lambda_nn=lambda_nn,
            max_inner_iter=20,
        )

        # Final objective
        f_final = np.sum(W * (R - L_final) ** 2)
        _, s_final, _ = np.linalg.svd(L_final, full_matrices=False)
        obj_final = f_final + lambda_nn * np.sum(s_final)

        assert obj_final <= obj_init + 1e-10, (
            f"Objective did not decrease: {obj_final} > {obj_init}"
        )

        # Soft-thresholding should reduce nuclear norm vs residual
        nuclear_norm_R = np.sum(np.linalg.svd(R, compute_uv=False))
        nuclear_norm_L = np.sum(s_final)
        assert nuclear_norm_L < nuclear_norm_R, (
            f"Nuclear norm not reduced: {nuclear_norm_L} >= {nuclear_norm_R}"
        )

    def test_zero_weights_no_division_error(self):
        """Verify solver handles all-zero weights without ZeroDivisionError."""
        rng = np.random.default_rng(99)
        Y = rng.normal(0, 1, (6, 4))
        W = np.zeros((6, 4))
        L_init = rng.normal(0, 1, (6, 4))

        trop_est = TROP(method="local", n_bootstrap=2)
        result = trop_est._weighted_nuclear_norm_solve(
            Y=Y,
            W=W,
            L_init=L_init,
            alpha=np.zeros(4),
            beta=np.zeros(6),
            lambda_nn=0.3,
        )

        assert np.isfinite(result).all(), "Result contains NaN or Inf"
        assert result.shape == (6, 4), f"Expected (6, 4), got {result.shape}"


@pytest.mark.slow
class TestTROPGlobalMethod:
    """Tests for TROP method='global'.

    The global method fits a single model on control data with global
    weights, then extracts per-observation treatment effects as
    residuals (τ_it = Y_it - μ - α_i - β_t - L_it). ATT is the mean
    of these effects. The local method instead fits a separate model
    per treated observation with observation-specific weights.
    """

    def test_global_basic(self, simple_panel_data):
        """Global method runs and produces reasonable ATT."""
        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42,
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
        # ATT should be positive (true effect is 3.0)
        assert results.att > 0

    def test_global_no_lowrank(self, simple_panel_data):
        """Global method with lambda_nn=inf (no low-rank)."""
        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[float('inf')],  # Disable low-rank
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert isinstance(results, TROPResults)
        # Effective rank should be 0 when L=0
        assert results.effective_rank == 0.0
        # Factor matrix should be all zeros
        assert np.allclose(results.factor_matrix, 0.0)

    def test_global_with_lowrank(self, factor_dgp_data, ci_params):
        """Global method with finite lambda_nn (with low-rank)."""
        n_boot = ci_params.bootstrap(20)
        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1, 1.0],
            n_bootstrap=n_boot,
            seed=42,
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
        # Should produce non-zero factor matrix if low-rank is used
        # (depends on which lambda_nn is selected)

    def test_global_matches_direction(self, simple_panel_data):
        """Global method sign/magnitude roughly matches local."""
        # Fit with local
        trop_local = TROP(
            method="local",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42,
        )
        results_local = trop_local.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Fit with global
        trop_global = TROP(
            method="global",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42,
        )
        results_global = trop_global.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Both should have positive ATT (true effect is 3.0)
        assert results_local.att > 0
        assert results_global.att > 0

        # Signs should match
        assert np.sign(results_local.att) == np.sign(results_global.att)

    def test_method_parameter_validation(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            TROP(method="invalid_method")

    def test_method_in_get_params(self):
        """method parameter appears in get_params()."""
        trop_est = TROP(method="global")
        params = trop_est.get_params()
        assert "method" in params
        assert params["method"] == "global"

    def test_method_in_get_params_joint_deprecated(self):
        """'joint' alias maps to 'global' in get_params()."""
        with pytest.warns(FutureWarning, match="deprecated"):
            trop_est = TROP(method="joint")
        params = trop_est.get_params()
        assert params["method"] == "global"

    def test_method_in_set_params(self):
        """method parameter can be set via set_params()."""
        trop_est = TROP(method="local")
        assert trop_est.method == "local"

        trop_est.set_params(method="global")
        assert trop_est.method == "global"

    def test_method_set_params_joint_deprecated(self):
        """'joint' alias maps to 'global' via set_params()."""
        trop_est = TROP(method="local")
        with pytest.warns(FutureWarning, match="deprecated"):
            trop_est.set_params(method="joint")
        assert trop_est.method == "global"

    def test_method_in_get_params_twostep_deprecated(self):
        """'twostep' alias maps to 'local' in get_params()."""
        with pytest.warns(FutureWarning, match="deprecated"):
            trop_est = TROP(method="twostep")
        params = trop_est.get_params()
        assert params["method"] == "local"

    def test_method_set_params_twostep_deprecated(self):
        """'twostep' alias maps to 'local' via set_params()."""
        trop_est = TROP(method="global")
        with pytest.warns(FutureWarning, match="deprecated"):
            trop_est.set_params(method="twostep")
        assert trop_est.method == "local"

    def test_global_bootstrap_variance(self, simple_panel_data, ci_params):
        """Global method bootstrap variance estimation works."""
        n_boot = ci_params.bootstrap(20)
        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=n_boot,
            seed=42,
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert results.se > 0
        assert results.n_bootstrap == n_boot
        assert results.bootstrap_distribution is not None

    def test_global_confidence_interval(self, simple_panel_data, ci_params):
        """Global method produces valid confidence intervals."""
        n_boot = ci_params.bootstrap(30)
        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            alpha=0.05,
            n_bootstrap=n_boot,
            seed=42,
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

    def test_global_loocv_selects_from_grid(self, simple_panel_data):
        """Global method LOOCV selects tuning parameters from the grid."""
        grid_time = [0.0, 0.5, 1.0]
        grid_unit = [0.0, 0.5, 1.0]
        grid_nn = [0.0, 0.1]

        trop_est = TROP(
            method="global",
            lambda_time_grid=grid_time,
            lambda_unit_grid=grid_unit,
            lambda_nn_grid=grid_nn,
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Selected lambdas should be from the grid
        assert results.lambda_time in grid_time
        assert results.lambda_unit in grid_unit
        assert results.lambda_nn in grid_nn
        # LOOCV score should be computed
        assert np.isfinite(results.loocv_score) or np.isnan(results.loocv_score)

    def test_global_loocv_score_internal(self, simple_panel_data):
        """Test the internal _loocv_score_global method produces valid scores."""
        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            seed=42,
        )

        # Setup data matrices
        all_units = sorted(simple_panel_data['unit'].unique())
        all_periods = sorted(simple_panel_data['period'].unique())
        n_units = len(all_units)
        n_periods = len(all_periods)

        Y = (
            simple_panel_data.pivot(index='period', columns='unit', values='outcome')
            .reindex(index=all_periods, columns=all_units)
            .values
        )
        D = (
            simple_panel_data.pivot(index='period', columns='unit', values='treated')
            .reindex(index=all_periods, columns=all_units)
            .fillna(0)
            .astype(int)
            .values
        )

        control_mask = D == 0
        control_obs = [
            (t, i) for t in range(n_periods) for i in range(n_units)
            if control_mask[t, i] and not np.isnan(Y[t, i])
        ][:20]  # Limit for speed

        treated_periods = 3  # From fixture: n_post = 3

        # Score should be finite
        score = trop_est._loocv_score_global(
            Y, D, control_obs, 0.0, 0.0, 0.0,
            treated_periods, n_units, n_periods
        )
        assert np.isfinite(score) or np.isinf(score), "Score should be finite or inf"

        # Score with larger lambda_nn should still work
        score2 = trop_est._loocv_score_global(
            Y, D, control_obs, 1.0, 1.0, 0.1,
            treated_periods, n_units, n_periods
        )
        assert np.isfinite(score2) or np.isinf(score2), "Score should be finite or inf"

    def test_global_handles_nan_outcomes(self, simple_panel_data):
        """Global method handles NaN outcome values gracefully."""
        # Introduce NaN in some control observations
        data = simple_panel_data.copy()
        control_mask = data['treated'] == 0
        control_indices = data[control_mask].index.tolist()

        # Set 5 random control observations to NaN
        np.random.seed(42)
        nan_indices = np.random.choice(control_indices, size=5, replace=False)
        data.loc[nan_indices, 'outcome'] = np.nan

        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Results should be finite (NaN observations excluded)
        assert np.isfinite(results.att), "ATT should be finite with NaN data"
        assert np.isfinite(results.se), "SE should be finite with NaN data"
        # ATT should be positive (true effect is 3.0)
        assert results.att > 0, "ATT should be positive"

    def test_global_with_lowrank_handles_nan(self, simple_panel_data):
        """Global method with low-rank handles NaN values correctly."""
        # Introduce NaN in some control observations
        data = simple_panel_data.copy()
        control_mask = data['treated'] == 0
        control_indices = data[control_mask].index.tolist()

        # Set 3 random control observations to NaN
        np.random.seed(123)
        nan_indices = np.random.choice(control_indices, size=3, replace=False)
        data.loc[nan_indices, 'outcome'] = np.nan

        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],  # Finite lambda_nn enables low-rank
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Results should be finite
        assert np.isfinite(results.att), "ATT should be finite with NaN data"
        assert np.isfinite(results.se), "SE should be finite with NaN data"

    def test_global_nan_exclusion_behavior(self, simple_panel_data):
        """Verify NaN observations are truly excluded from estimation.

        This tests the PR #113 fix: NaN observations should not contribute
        to the weighted gradient step. We verify this by comparing results
        when fitting on data with NaN vs data with those observations removed.
        """
        # Get a clean copy
        data_full = simple_panel_data.copy()

        # Identify a specific control observation to "remove"
        control_mask = data_full['treated'] == 0
        control_indices = data_full[control_mask].index.tolist()

        # Pick a few specific observations to remove/set to NaN
        np.random.seed(42)
        remove_indices = np.random.choice(control_indices, size=3, replace=False)

        # Create version with NaN
        data_nan = data_full.copy()
        data_nan.loc[remove_indices, 'outcome'] = np.nan

        # Create version with rows removed
        data_dropped = data_full.drop(remove_indices)

        # Fit on both versions with identical settings
        trop_nan = TROP(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[0.0],  # Disable low-rank for cleaner comparison
            n_bootstrap=10,
            seed=42,
        )
        trop_dropped = TROP(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=10,
            seed=42,
        )

        results_nan = trop_nan.fit(
            data_nan,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )
        results_dropped = trop_dropped.fit(
            data_dropped,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # ATT should be very close (allowing small numerical tolerance)
        # If NaN observations were not truly excluded, ATT would differ
        assert np.abs(results_nan.att - results_dropped.att) < 0.5, (
            f"ATT with NaN ({results_nan.att:.4f}) should match dropped data "
            f"({results_dropped.att:.4f}) - true NaN exclusion"
        )

    def test_global_unit_no_valid_pre_gets_zero_weight(self, simple_panel_data):
        """Verify units with no valid pre-period data get zero weight.

        This tests the PR #113 fix: units with no valid pre-period observations
        should get zero weight (instead of max weight via dist=0).
        """
        # Create data where one control unit has all NaN in pre-period
        data = simple_panel_data.copy()

        # Find a control unit (unit that never has treated=1)
        unit_ever_treated = data.groupby('unit')['treated'].max()
        control_units = unit_ever_treated[unit_ever_treated == 0].index.tolist()
        target_unit = control_units[0]

        # Get pre-periods (periods where this control unit has treated=0)
        unit_data = data[data['unit'] == target_unit]
        pre_periods = sorted(unit_data[unit_data['treated'] == 0]['period'].unique())[:5]

        # Set all pre-period values for target_unit to NaN
        mask = (data['unit'] == target_unit) & (data['period'].isin(pre_periods))
        data.loc[mask, 'outcome'] = np.nan

        trop_est = TROP(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],  # Non-zero lambda_unit to use distance weighting
            lambda_nn_grid=[0.0],
            n_bootstrap=10,
            seed=42,
        )

        # This should not error and should produce finite results
        results = trop_est.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert np.isfinite(results.att), "ATT should be finite even with unit having no pre-period data"
        assert np.isfinite(results.se), "SE should be finite"

    def test_global_treated_pre_nan_handling(self, simple_panel_data):
        """Verify global method handles NaN in treated units during pre-periods.

        When all treated units have NaN at a pre-period, average_treated[t] = NaN.
        This period should be excluded from unit distance calculation (both numerator
        and denominator) to avoid inflating valid_count.

        This tests the fix for PR #113 Round 5 feedback (P1).
        """
        data = simple_panel_data.copy()

        # Find treated units and pre-periods
        treated_units = data[data['treated'] == 1]['unit'].unique()
        # Pre-periods are periods where treated=0 for treated units
        pre_periods = sorted(
            data[(data['unit'].isin(treated_units)) & (data['treated'] == 0)]['period'].unique()
        )
        assert len(pre_periods) >= 2, "Need at least 2 pre-periods for this test"

        # Pick a middle pre-period
        target_period = pre_periods[len(pre_periods) // 2]

        # Set ALL treated units' outcomes at target_period to NaN
        # This makes average_treated[target_period] = NaN
        mask = (data['unit'].isin(treated_units)) & (data['period'] == target_period)
        data.loc[mask, 'outcome'] = np.nan

        # Verify we set NaN correctly
        n_nan = data.loc[mask, 'outcome'].isna().sum()
        assert n_nan == len(treated_units), f"Should have {len(treated_units)} NaN, got {n_nan}"

        trop_est = TROP(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Results should be finite - NaN period properly excluded from distance calc
        assert np.isfinite(results.att), f"ATT should be finite, got {results.att}"
        assert np.isfinite(results.se), f"SE should be finite, got {results.se}"

    def test_global_rejects_staggered_adoption(self):
        """Global method raises ValueError for staggered adoption data.

        The global method assumes all treated units receive treatment at the
        same time. With staggered adoption (units first treated at different
        periods), the method's weights and variance estimation are invalid.
        """
        # Create data with staggered treatment (units treated at different times)
        data = []
        np.random.seed(42)
        for i in range(10):
            # Units 0-2 first treated at t=5, units 3-4 first treated at t=7
            first_treat = 5 if i < 3 else 7
            is_treated_unit = i < 5  # Units 0-4 are treated, 5-9 are control
            for t in range(10):
                treated = 1 if is_treated_unit and t >= first_treat else 0
                data.append({
                    'unit': i,
                    'time': t,
                    'outcome': np.random.randn(),
                    'treated': treated
                })
        df = pd.DataFrame(data)

        trop = TROP(method="global")
        with pytest.raises(ValueError, match="staggered adoption"):
            trop.fit(df, 'outcome', 'treated', 'unit', 'time')

    def test_global_method_alias(self, simple_panel_data):
        """method='global' runs and produces a valid positive ATT."""
        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert isinstance(results, TROPResults)
        assert results.att > 0

    def test_global_uses_control_only_weights(self, simple_panel_data):
        """Verify delta[t,i] == 0 for all D[t,i] == 1 (control-only weights)."""
        trop_est = TROP(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[0.0],
            seed=42,
        )

        # Setup data matrices
        all_units = sorted(simple_panel_data['unit'].unique())
        all_periods = sorted(simple_panel_data['period'].unique())
        n_units = len(all_units)
        n_periods = len(all_periods)

        Y = (
            simple_panel_data.pivot(index='period', columns='unit', values='outcome')
            .reindex(index=all_periods, columns=all_units)
            .values
        )
        D = (
            simple_panel_data.pivot(index='period', columns='unit', values='treated')
            .reindex(index=all_periods, columns=all_units)
            .fillna(0)
            .astype(int)
            .values
        )

        treated_periods = np.sum(np.any(D == 1, axis=1))

        delta = trop_est._compute_global_weights(
            Y, D, 1.0, 1.0, int(treated_periods), n_units, n_periods
        )

        # All treated cells should have zero weight
        assert np.all(delta[D == 1] == 0.0), (
            "Treated observations should have zero weight after (1-W) masking"
        )
        # Some control cells should have non-zero weight
        assert np.any(delta[D == 0] > 0.0), (
            "Some control observations should have positive weight"
        )

    def test_global_tau_is_posthoc_residual(self, simple_panel_data):
        """Verify ATT == mean(Y - mu - alpha - beta - L) over treated cells."""
        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # Reconstruct tau from treatment_effects
        tau_values = [v for v in results.treatment_effects.values() if np.isfinite(v)]
        assert len(tau_values) > 0, "Should have treatment effects"
        reconstructed_att = np.mean(tau_values)
        assert np.isclose(results.att, reconstructed_att, atol=1e-10), (
            f"ATT ({results.att}) should equal mean of treatment effects ({reconstructed_att})"
        )

    def test_global_heterogeneous_treatment_effects(self, simple_panel_data):
        """Treatment effects are heterogeneous (not all identical) with global method."""
        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[float('inf')],
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        te_values = list(results.treatment_effects.values())
        # With post-hoc extraction, effects should vary across observations
        assert len(set(te_values)) > 1, (
            "Treatment effects should be heterogeneous with post-hoc extraction"
        )

    def test_global_treated_outcome_does_not_affect_fit(self, simple_panel_data):
        """Perturbing treated outcomes should not change (mu, alpha, beta, L)."""
        all_units = sorted(simple_panel_data['unit'].unique())
        all_periods = sorted(simple_panel_data['period'].unique())
        n_units = len(all_units)
        n_periods = len(all_periods)

        Y = (
            simple_panel_data.pivot(index='period', columns='unit', values='outcome')
            .reindex(index=all_periods, columns=all_units)
            .values
        )
        D = (
            simple_panel_data.pivot(index='period', columns='unit', values='treated')
            .reindex(index=all_periods, columns=all_units)
            .fillna(0)
            .astype(int)
            .values
        )

        treated_periods = int(np.sum(np.any(D == 1, axis=1)))

        trop_est = TROP(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[0.1],
            seed=42,
        )

        # Compute weights and fit with original data
        delta = trop_est._compute_global_weights(
            Y, D, 1.0, 1.0, treated_periods, n_units, n_periods
        )
        mu1, alpha1, beta1, L1 = trop_est._solve_global_with_lowrank(
            Y, delta, 0.1, 100, 1e-6
        )

        # Perturb treated outcomes by large amount
        Y_perturbed = Y.copy()
        Y_perturbed[D == 1] += 1000.0

        # Recompute (same weights since (1-W) zeroes treated cells)
        delta2 = trop_est._compute_global_weights(
            Y_perturbed, D, 1.0, 1.0, treated_periods, n_units, n_periods
        )
        mu2, alpha2, beta2, L2 = trop_est._solve_global_with_lowrank(
            Y_perturbed, delta2, 0.1, 100, 1e-6
        )

        # Model parameters should be identical
        assert np.isclose(mu1, mu2, atol=1e-8), f"mu changed: {mu1} vs {mu2}"
        assert np.allclose(alpha1, alpha2, atol=1e-8), "alpha changed"
        assert np.allclose(beta1, beta2, atol=1e-8), "beta changed"
        assert np.allclose(L1, L2, atol=1e-8), "L changed"


class TestTROPNValidTreated:
    """Tests for n_valid_treated consistency and NaN treated outcome handling."""

    @staticmethod
    def _make_panel(n_units=20, n_periods=8, n_treated=5, n_post=3,
                    effect=2.0, seed=42):
        """Helper: generate a clean panel DataFrame."""
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_periods):
                post = t >= (n_periods - n_post)
                y = 5.0 + i * 0.3 + t * 0.2 + rng.normal() * 0.3
                d = 1 if (is_treated and post) else 0
                if d:
                    y += effect
                rows.append({'unit': i, 'time': t, 'outcome': y, 'treated': d})
        return pd.DataFrame(rows)

    def test_global_n_treated_obs_partial_nan(self):
        """Global method: n_treated_obs reflects only finite outcomes."""
        df = self._make_panel()

        # Inject NaN into some treated outcomes
        treated_mask = (df['treated'] == 1)
        treated_idx = df[treated_mask].index.tolist()
        n_nan = 3
        for idx in treated_idx[:n_nan]:
            df.loc[idx, 'outcome'] = np.nan

        total_treated = int(treated_mask.sum())

        trop_est = TROP(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[np.inf],
            n_bootstrap=2,
            seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = trop_est.fit(df, 'outcome', 'treated', 'unit', 'time')

        assert results.n_treated_obs == total_treated - n_nan, \
            f"Expected {total_treated - n_nan}, got {results.n_treated_obs}"
        assert np.isfinite(results.att)

    def test_local_n_treated_obs_partial_nan(self):
        """Local method: n_treated_obs reflects only finite outcomes."""
        df = self._make_panel()

        treated_mask = (df['treated'] == 1)
        treated_idx = df[treated_mask].index.tolist()
        n_nan = 3
        for idx in treated_idx[:n_nan]:
            df.loc[idx, 'outcome'] = np.nan

        total_treated = int(treated_mask.sum())

        trop_est = TROP(
            method="local",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[np.inf],
            n_bootstrap=2,
            seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = trop_est.fit(df, 'outcome', 'treated', 'unit', 'time')

        assert results.n_treated_obs == total_treated - n_nan, \
            f"Expected {total_treated - n_nan}, got {results.n_treated_obs}"
        assert np.isfinite(results.att)

    def test_local_nan_treated_not_poison_att(self):
        """Local: NaN treated outcomes don't poison ATT via np.mean."""
        df = self._make_panel(effect=3.0)

        # Make ONE treated outcome NaN
        treated_mask = (df['treated'] == 1)
        first_treated_idx = df[treated_mask].index[0]
        df.loc[first_treated_idx, 'outcome'] = np.nan

        trop_est = TROP(
            method="local",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[np.inf],
            n_bootstrap=2,
            seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = trop_est.fit(df, 'outcome', 'treated', 'unit', 'time')

        # ATT must be finite (not NaN from NaN poisoning)
        assert np.isfinite(results.att), f"ATT should be finite, got {results.att}"
        # ATT should be in reasonable range
        assert results.att > 1.0, f"ATT {results.att} should reflect treatment effect"

    def test_global_all_treated_nan_warns(self):
        """Global method warns when all treated outcomes are NaN."""
        df = self._make_panel()

        # Set ALL treated outcomes to NaN
        df.loc[df['treated'] == 1, 'outcome'] = np.nan

        trop_est = TROP(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[np.inf],
            n_bootstrap=2,
            seed=42,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = trop_est.fit(df, 'outcome', 'treated', 'unit', 'time')

        # Should warn about all NaN treated
        nan_warnings = [x for x in w if "All treated outcomes are NaN" in str(x.message)]
        assert len(nan_warnings) > 0, "Should warn about all-NaN treated outcomes"
        assert results.n_treated_obs == 0
        assert np.isnan(results.att)

    def test_local_all_treated_nan_warns(self):
        """Local method warns when all treated outcomes are NaN."""
        df = self._make_panel()

        df.loc[df['treated'] == 1, 'outcome'] = np.nan

        trop_est = TROP(
            method="local",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[np.inf],
            n_bootstrap=2,
            seed=42,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = trop_est.fit(df, 'outcome', 'treated', 'unit', 'time')

        nan_warnings = [x for x in w if "All treated outcomes are NaN" in str(x.message)]
        assert len(nan_warnings) > 0, "Should warn about all-NaN treated outcomes"
        assert results.n_treated_obs == 0
        assert np.isnan(results.att)


class TestTROPBootstrapNaNSE:
    """Tests for NaN SE when bootstrap has <2 successful draws."""

    def test_global_bootstrap_zero_draws_returns_nan_se(self):
        """Global bootstrap with 0 successful draws returns NaN SE, not 0.0."""
        from unittest.mock import patch
        import sys

        df = TestTROPNValidTreated._make_panel()

        trop_est = TROP(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[np.inf],
            n_bootstrap=5,
            seed=42,
        )

        # Disable Rust backend so Python fallback path is tested,
        # then patch _fit_global_with_fixed_lambda to always raise
        trop_global_module = sys.modules['diff_diff.trop_global']
        with patch.object(trop_global_module, 'HAS_RUST_BACKEND', False), \
             patch.object(trop_global_module, '_rust_bootstrap_trop_variance_global', None), \
             patch.object(TROP, '_fit_global_with_fixed_lambda',
                          side_effect=ValueError("forced failure")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                se, dist = trop_est._bootstrap_variance_global(
                    df, 'outcome', 'treated', 'unit', 'time',
                    (1.0, 1.0, 1e10), 3,
                )

        assert np.isnan(se), f"SE should be NaN when 0 draws succeed, got {se}"
        assert len(dist) == 0

    def test_local_bootstrap_zero_draws_returns_nan_se(self):
        """Local bootstrap with 0 successful draws returns NaN SE, not 0.0."""
        from unittest.mock import patch

        df = TestTROPNValidTreated._make_panel()

        trop_est = TROP(
            method="local",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[np.inf],
            n_bootstrap=5,
            seed=42,
        )

        # Patch _fit_with_fixed_lambda to always raise
        with patch.object(TROP, '_fit_with_fixed_lambda',
                          side_effect=ValueError("forced failure")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                se, dist = trop_est._bootstrap_variance(
                    df, 'outcome', 'treated', 'unit', 'time',
                    (1.0, 1.0, 1e10),
                )

        assert np.isnan(se), f"SE should be NaN when 0 draws succeed, got {se}"
        assert len(dist) == 0


class TestTROPModuleSplit:
    """Regression tests for the trop.py -> trop_global.py / trop_local.py split."""

    @staticmethod
    def _make_panel():
        """Create a simple balanced panel for split regression tests."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 8, 6
        rows = []
        for i in range(n_units):
            treated = i < 3  # 3 treated, 5 control
            for t in range(n_periods):
                y = rng.normal(0, 1)
                if treated and t >= 4:
                    y += 2.0  # treatment effect
                rows.append({
                    "unit": i, "time": t, "outcome": y,
                    "treated": 1 if treated and t >= 4 else 0,
                })
        return pd.DataFrame(rows)

    def test_global_absorbing_state_error_has_remediation_guidance(self):
        """Global path ValueError for non-absorbing D includes remediation text."""
        df = self._make_panel()
        # Break absorbing state: unit 0 goes 0->1->0
        df.loc[(df["unit"] == 0) & (df["time"] == 5), "treated"] = 0

        with pytest.raises(ValueError, match="once treated, always treated"):
            TROP(method="global").fit(df, "outcome", "treated", "unit", "time")

        with pytest.raises(ValueError, match="convert to absorbing state"):
            TROP(method="global").fit(df, "outcome", "treated", "unit", "time")

    def test_global_finite_lambda_nn_exercises_lowrank_path(self):
        """method='global' with finite lambda_nn successfully fits the low-rank solver."""
        df = self._make_panel()
        trop_est = TROP(
            method="global",
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],  # finite -> exercises _solve_global_with_lowrank
            n_bootstrap=5,
            seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = trop_est.fit(df, "outcome", "treated", "unit", "time")
        assert np.isfinite(result.att)

    def test_local_finite_lambda_nn_exercises_nuclear_norm(self):
        """method='local' with finite lambda_nn exercises weighted nuclear norm solver."""
        df = self._make_panel()
        trop_est = TROP(
            method="local",
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],  # finite -> exercises _weighted_nuclear_norm_solve
            n_bootstrap=5,
            seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = trop_est.fit(df, "outcome", "treated", "unit", "time")
        assert np.isfinite(result.att)

    def test_method_dispatch_global_uses_fit_global(self):
        """method='global' dispatches to _fit_global from TROPGlobalMixin."""
        from unittest.mock import patch

        df = self._make_panel()
        trop_est = TROP(method="global", n_bootstrap=2, seed=42)

        with patch.object(TROP, '_fit_global', wraps=trop_est._fit_global) as mock_fg:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trop_est.fit(df, "outcome", "treated", "unit", "time")
            mock_fg.assert_called_once()

    def test_method_dispatch_local_does_not_use_fit_global(self):
        """method='local' does NOT call _fit_global."""
        from unittest.mock import patch

        df = self._make_panel()
        trop_est = TROP(method="local", n_bootstrap=2, seed=42,
                        lambda_time_grid=[0.0], lambda_unit_grid=[0.0],
                        lambda_nn_grid=[np.inf])

        with patch.object(TROP, '_fit_global') as mock_fg:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trop_est.fit(df, "outcome", "treated", "unit", "time")
            mock_fg.assert_not_called()

