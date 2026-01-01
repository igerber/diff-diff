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
