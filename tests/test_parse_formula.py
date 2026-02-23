"""Tests for _parse_formula edge cases.

These tests document bugs in the current formula parser and should be
fixed alongside the parser rewrite.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import DifferenceInDifferences


@pytest.fixture
def formula_data():
    """Simple data with outcome, treatment, time, and two covariates."""
    np.random.seed(42)
    n = 200
    treated = np.array([1] * (n // 2) + [0] * (n // 2))
    post = np.tile([0, 1], n // 2)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    outcome = 1.0 + 2.0 * treated + 3.0 * post + 5.0 * treated * post + 0.5 * x1 + np.random.randn(n) * 0.1
    return pd.DataFrame({
        "outcome": outcome,
        "treated": treated,
        "post": post,
        "x1": x1,
        "x2": x2,
    })


class TestParseFormulaEdgeCases:
    """Edge cases for the _parse_formula method."""

    # ------------------------------------------------------------------
    # Bug: covariates BEFORE * interaction
    # ------------------------------------------------------------------
    def test_covariates_before_star(self, formula_data):
        """'outcome ~ x1 + treated * post' should work (covariates before *)."""
        did = DifferenceInDifferences()
        results = did.fit(formula_data, formula="outcome ~ x1 + treated * post")

        assert np.isfinite(results.att)
        assert "x1" in results.coefficients

    def test_covariates_both_sides_of_star(self, formula_data):
        """'outcome ~ x1 + treated * post + x2' should parse both covariates."""
        did = DifferenceInDifferences()
        results = did.fit(formula_data, formula="outcome ~ x1 + treated * post + x2")

        assert np.isfinite(results.att)
        assert "x1" in results.coefficients
        assert "x2" in results.coefficients

    # ------------------------------------------------------------------
    # Bug: multiple ~ in formula
    # ------------------------------------------------------------------
    def test_multiple_tildes_raises_clear_error(self, formula_data):
        """'outcome ~ treated ~ post' should give a clear ValueError, not crash."""
        did = DifferenceInDifferences()
        with pytest.raises(ValueError):
            did.fit(formula_data, formula="outcome ~ treated ~ post")

    # ------------------------------------------------------------------
    # Bug: whitespace variations
    # ------------------------------------------------------------------
    def test_extra_whitespace(self, formula_data):
        """Formula with extra whitespace should still parse correctly."""
        did = DifferenceInDifferences()
        results = did.fit(formula_data, formula="  outcome  ~  treated  *  post  ")
        assert np.isfinite(results.att)

    def test_no_whitespace(self, formula_data):
        """Formula with no whitespace should still parse correctly."""
        did = DifferenceInDifferences()
        results = did.fit(formula_data, formula="outcome~treated*post")
        assert np.isfinite(results.att)

    # ------------------------------------------------------------------
    # Consistency: formula vs explicit params should match
    # ------------------------------------------------------------------
    def test_covariates_before_star_matches_explicit(self, formula_data):
        """Formula 'y ~ x1 + D * T' should give same ATT as explicit params."""
        did1 = DifferenceInDifferences(seed=42)
        r1 = did1.fit(formula_data, formula="outcome ~ x1 + treated * post")

        did2 = DifferenceInDifferences(seed=42)
        r2 = did2.fit(
            formula_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            covariates=["x1"],
        )

        assert np.isclose(r1.att, r2.att, rtol=1e-10)
        assert np.isclose(r1.se, r2.se, rtol=1e-10)
