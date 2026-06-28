"""
Unit and integration tests for ContinuousDiD estimator.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.continuous_did import ContinuousDiD
from diff_diff.continuous_did_bspline import (
    bspline_derivative_design_matrix,
    bspline_design_matrix,
    build_bspline_basis,
    default_dose_grid,
)
from diff_diff.continuous_did_results import ContinuousDiDResults
from diff_diff.prep_dgp import generate_continuous_did_data

# =============================================================================
# B-Spline Basis Tests
# =============================================================================


class TestBSplineBasis:
    """Test B-spline utility functions."""

    def test_knot_construction_no_interior(self):
        dose = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=0)
        assert deg == 3
        # Boundary knots repeated degree+1 times
        assert knots[0] == 1.0
        assert knots[-1] == 5.0
        assert len(knots) == 3 + 1 + 3 + 1  # (degree+1)*2

    def test_knot_construction_with_interior(self):
        dose = np.linspace(1, 10, 100)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=2)
        # Interior knots at 1/3 and 2/3 quantiles
        n_expected = 2 * (3 + 1) + 2  # boundary + interior
        assert len(knots) == n_expected

    def test_design_matrix_shape(self):
        dose = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=0)
        B = bspline_design_matrix(dose, knots, deg, include_intercept=True)
        n_basis = len(knots) - deg - 1  # Total basis functions
        assert B.shape == (5, n_basis)  # Same columns (intercept replaces first)

    def test_design_matrix_intercept_column(self):
        dose = np.linspace(1, 5, 20)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=0)
        B = bspline_design_matrix(dose, knots, deg, include_intercept=True)
        # First column should be all ones
        np.testing.assert_array_equal(B[:, 0], np.ones(20))

    def test_design_matrix_no_intercept(self):
        dose = np.linspace(1, 5, 20)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=0)
        B_no = bspline_design_matrix(dose, knots, deg, include_intercept=False)
        n_basis = len(knots) - deg - 1
        assert B_no.shape == (20, n_basis)
        # First column should NOT be all ones
        assert not np.allclose(B_no[:, 0], 1.0)

    def test_derivative_numerical_check(self):
        """Verify B-spline derivatives match finite differences."""
        dose = np.linspace(1, 5, 50)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=1)

        # Evaluate at interior points (avoid boundaries)
        x = np.linspace(1.5, 4.5, 30)
        dB = bspline_derivative_design_matrix(x, knots, deg, include_intercept=True)

        # Finite difference check
        h = 1e-6
        x_plus = x + h
        x_minus = x - h
        B_plus = bspline_design_matrix(x_plus, knots, deg, include_intercept=True)
        B_minus = bspline_design_matrix(x_minus, knots, deg, include_intercept=True)
        fd = (B_plus - B_minus) / (2 * h)

        # Intercept derivative should be 0
        np.testing.assert_allclose(dB[:, 0], 0.0, atol=1e-10)
        # Other columns should match finite differences
        np.testing.assert_allclose(dB[:, 1:], fd[:, 1:], atol=1e-4)

    def test_partition_of_unity(self):
        """B-spline basis without intercept should sum to ~1 at interior points."""
        dose = np.linspace(1, 5, 50)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=2)
        x = np.linspace(1.1, 4.9, 30)
        B = bspline_design_matrix(x, knots, deg, include_intercept=False)
        row_sums = B.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_linear_basis(self):
        """Degree 1 with 0 knots: 2 basis functions (intercept + linear)."""
        dose = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        knots, deg = build_bspline_basis(dose, degree=1, num_knots=0)
        B = bspline_design_matrix(dose, knots, deg, include_intercept=True)
        assert B.shape[1] == 2  # intercept + 1 basis fn


# ---------------------------------------------------------------------------
# Finding #12 (axis C, silent-failures audit). Previously
# `bspline_derivative_design_matrix` silently swallowed ValueError in the
# per-basis derivative loop, leaving affected columns of the derivative
# design matrix as zero with no user-visible signal. ContinuousDiD's
# analytical inference then fed a biased dPsi into downstream SE
# computation. The fix aggregates failed-basis indices and emits ONE
# UserWarning naming them.
# ---------------------------------------------------------------------------


class TestBSplineDerivativeDegenerateBasis:
    def test_single_dose_is_silent(self):
        """All-identical knots (single dose value) is a well-defined
        degenerate case — derivatives are mathematically zero and the
        function returns silently. Regression-guard the existing contract."""
        x = np.array([3.0, 3.0, 3.0, 3.0])
        knots = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])  # all identical
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            dB = bspline_derivative_design_matrix(x, knots, degree=3, include_intercept=True)
        deriv_warnings = [
            w for w in caught if "B-spline derivative construction failed" in str(w.message)
        ]
        assert deriv_warnings == [], (
            "All-identical knots should be handled silently (mathematically "
            "well-defined zero-derivative case); warning fired unexpectedly: "
            f"{[str(w.message) for w in deriv_warnings]}"
        )
        np.testing.assert_array_equal(dB, np.zeros_like(dB))

    def test_valueerror_from_bspline_emits_aggregate_warning(self):
        """When BSpline construction raises ValueError for some basis
        functions (malformed knot vector, etc.), the new aggregate
        UserWarning must fire naming the affected indices."""
        from unittest.mock import patch

        import diff_diff.continuous_did_bspline as bspline_mod

        dose = np.linspace(1, 5, 30)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=1)
        x = np.linspace(1.5, 4.5, 20)

        # Force ValueError on basis indices 1 and 3 only; the rest run
        # through normally. This is the partial-failure mode the audit
        # called out.
        real_bspline = bspline_mod.BSpline
        call_counter = {"n": 0}

        def flaky_bspline(knots, c, degree):
            # c is a one-hot vector; the index set to 1 is the basis j
            j = int(np.argmax(c))
            call_counter["n"] += 1
            if j in (1, 3):
                raise ValueError(f"forced test failure for basis j={j}")
            return real_bspline(knots, c, degree)

        import warnings as _w

        with patch.object(bspline_mod, "BSpline", side_effect=flaky_bspline):
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                dB = bspline_derivative_design_matrix(x, knots, degree=deg, include_intercept=True)

        deriv_warnings = [
            w for w in caught if "B-spline derivative construction failed" in str(w.message)
        ]
        assert len(deriv_warnings) == 1, (
            f"Expected exactly one aggregate warning, got {len(deriv_warnings)}: "
            f"{[str(w.message) for w in deriv_warnings]}"
        )
        msg = str(deriv_warnings[0].message)
        # Message must name the failed basis indices so the user can debug.
        assert "[1, 3]" in msg, f"Expected indices [1, 3] in warning; got: {msg}"
        assert "2 of" in msg, f"Expected failure count '2 of ...' in warning; got: {msg}"
        # Affected columns should be zero.
        # With include_intercept=True, column 0 is always zero (intercept
        # derivative) and basis index j is at dB column j (the drop-first
        # then prepend-zeros logic keeps the same per-j mapping for j>=1).
        np.testing.assert_array_equal(dB[:, 1], np.zeros(len(x)))  # failed basis j=1
        np.testing.assert_array_equal(dB[:, 3], np.zeros(len(x)))  # failed basis j=3

        # Unaffected columns must match the un-patched baseline exactly
        # (except columns 1 and 3 which were forced to zero). This guards
        # a regression that would zero or corrupt the entire derivative
        # matrix on any ValueError.
        dB_baseline = bspline_derivative_design_matrix(x, knots, degree=deg, include_intercept=True)
        for col in range(dB.shape[1]):
            if col in (1, 3):
                continue
            np.testing.assert_array_equal(
                dB[:, col],
                dB_baseline[:, col],
                err_msg=f"Unaffected column {col} diverges from baseline",
            )
        # At least one non-intercept, non-failed column must be non-zero,
        # confirming the function still produces meaningful derivatives.
        non_failed_cols = [c for c in range(1, dB.shape[1]) if c not in (1, 3)]
        assert any(np.any(dB[:, c] != 0) for c in non_failed_cols), (
            "Expected at least one unaffected non-intercept column to have "
            "non-zero derivatives; got all-zero dB outside failed cols."
        )

    def test_clean_knots_emit_no_warning(self):
        """Well-formed knot vector → no ValueError path taken → no
        warning. Regression-guard the happy path."""
        dose = np.linspace(1, 5, 50)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=2)
        x = np.linspace(1.5, 4.5, 30)
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            bspline_derivative_design_matrix(x, knots, deg, include_intercept=True)
        deriv_warnings = [
            w for w in caught if "B-spline derivative construction failed" in str(w.message)
        ]
        assert deriv_warnings == []


class TestDoseGrid:
    """Test dose grid computation."""

    def test_default_grid_size(self):
        dose = np.random.default_rng(42).lognormal(0.5, 0.5, size=100)
        grid = default_dose_grid(dose)
        assert len(grid) == 90  # quantiles 0.10 to 0.99

    def test_default_grid_sorted(self):
        dose = np.random.default_rng(42).lognormal(0.5, 0.5, size=100)
        grid = default_dose_grid(dose)
        assert np.all(np.diff(grid) >= 0)

    def test_custom_grid_passthrough(self):
        custom = np.array([1.0, 2.0, 3.0])
        est = ContinuousDiD(dvals=custom)
        np.testing.assert_array_equal(est.dvals, custom)

    def test_empty_dose(self):
        grid = default_dose_grid(np.array([0.0, 0.0]))
        assert len(grid) == 0


# =============================================================================
# ContinuousDiD Estimator Tests
# =============================================================================


class TestContinuousDiDInit:
    """Test constructor, get_params, set_params."""

    def test_default_params(self):
        est = ContinuousDiD()
        params = est.get_params()
        assert params["degree"] == 3
        assert params["num_knots"] == 0
        assert params["control_group"] == "never_treated"
        assert params["alpha"] == 0.05
        assert params["n_bootstrap"] == 0

    def test_set_params(self):
        est = ContinuousDiD()
        est.set_params(degree=1, num_knots=2)
        assert est.degree == 1
        assert est.num_knots == 2

    def test_set_invalid_param(self):
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="Invalid parameter"):
            est.set_params(nonexistent_param=42)


class TestContinuousDiDDataValidation:
    """Test data validation in fit()."""

    def test_missing_column(self):
        data = pd.DataFrame({"unit": [1], "period": [1], "outcome": [1.0]})
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="Column.*not found"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

    def test_non_time_invariant_dose(self):
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "period": [1, 2, 1, 2],
                "outcome": [1.0, 2.0, 1.0, 2.0],
                "first_treat": [2, 2, 0, 0],
                "dose": [1.0, 2.0, 0.0, 0.0],  # Dose changes over time!
            }
        )
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="time-invariant"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

    def test_drop_zero_dose_treated(self):
        """Units with positive first_treat but zero dose should be dropped."""
        # Need enough treated units for OLS: degree=1 → 2 basis fns → need >2 treated
        rows = []
        uid = 0
        # 1 treated unit with zero dose (should be dropped)
        rows += [
            {"unit": uid, "period": 1, "outcome": 1.0, "first_treat": 2, "dose": 0.0},
            {"unit": uid, "period": 2, "outcome": 3.0, "first_treat": 2, "dose": 0.0},
        ]
        uid += 1
        # 4 treated units with positive dose (should remain)
        for d in [1.0, 2.0, 3.0, 4.0]:
            rows += [
                {"unit": uid, "period": 1, "outcome": 0.0, "first_treat": 2, "dose": d},
                {"unit": uid, "period": 2, "outcome": 2 * d, "first_treat": 2, "dose": d},
            ]
            uid += 1
        # 3 control units
        for _ in range(3):
            rows += [
                {"unit": uid, "period": 1, "outcome": 0.0, "first_treat": 0, "dose": 0.0},
                {"unit": uid, "period": 2, "outcome": 0.0, "first_treat": 0, "dose": 0.0},
            ]
            uid += 1

        data = pd.DataFrame(rows)
        est = ContinuousDiD(degree=1, num_knots=0)
        with pytest.warns(UserWarning, match="Dropping.*units"):
            results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        # Unit 0 dropped (zero dose but treated), 4 treated remain
        assert results.n_treated_units == 4

    def test_unbalanced_panel_error(self):
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2],
                "period": [1, 2, 1],
                "outcome": [1.0, 2.0, 1.0],
                "first_treat": [2, 2, 0],
                "dose": [1.0, 1.0, 0.0],
            }
        )
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="[Uu]nbalanced"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

    def test_unbalanced_panel_same_count_different_periods(self):
        """Units with same period count but different periods should be caught."""
        data = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2, 2, 2],
                "period": [1, 2, 3, 1, 2, 4],  # Same count (3) but unit 2 has {1,2,4} vs {1,2,3}
                "outcome": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                "first_treat": [2, 2, 2, 0, 0, 0],
                "dose": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            }
        )
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="[Uu]nbalanced"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

    def test_invalid_aggregate_raises(self):
        """Invalid aggregate value should raise ValueError."""
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "period": [1, 2, 1, 2],
                "outcome": [1.0, 2.0, 1.0, 2.0],
                "first_treat": [2, 2, 0, 0],
                "dose": [1.0, 1.0, 0.0, 0.0],
            }
        )
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="Invalid aggregate"):
            est.fit(
                data, "outcome", "unit", "period", "first_treat", "dose", aggregate="event_study"
            )

    def test_no_never_treated_error(self):
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "period": [1, 2, 1, 2],
                "outcome": [1.0, 3.0, 1.0, 4.0],
                "first_treat": [2, 2, 2, 2],
                "dose": [1.0, 1.0, 2.0, 2.0],
            }
        )
        est = ContinuousDiD(control_group="never_treated")
        with pytest.raises(ValueError, match="[Nn]ever-treated"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")


class TestContinuousDiDFit:
    """Test basic fit returns correct types and shapes."""

    @pytest.fixture
    def basic_data(self):
        return generate_continuous_did_data(n_units=100, n_periods=3, seed=42, noise_sd=0.5)

    def test_fit_returns_results(self, basic_data):
        est = ContinuousDiD()
        results = est.fit(basic_data, "outcome", "unit", "period", "first_treat", "dose")
        assert isinstance(results, ContinuousDiDResults)

    def test_dose_response_shapes(self, basic_data):
        est = ContinuousDiD()
        results = est.fit(basic_data, "outcome", "unit", "period", "first_treat", "dose")
        n_grid = len(results.dose_grid)
        assert results.dose_response_att.effects.shape == (n_grid,)
        assert results.dose_response_acrt.effects.shape == (n_grid,)
        assert results.dose_response_att.se.shape == (n_grid,)
        assert results.dose_response_acrt.se.shape == (n_grid,)

    def test_overall_parameters(self, basic_data):
        est = ContinuousDiD()
        results = est.fit(basic_data, "outcome", "unit", "period", "first_treat", "dose")
        assert np.isfinite(results.overall_att)
        assert np.isfinite(results.overall_acrt)

    def test_group_time_effects_populated(self, basic_data):
        est = ContinuousDiD()
        results = est.fit(basic_data, "outcome", "unit", "period", "first_treat", "dose")
        assert len(results.group_time_effects) > 0

    def test_results_contain_init_params(self, basic_data):
        est = ContinuousDiD(
            base_period="universal",
            anticipation=0,
            n_bootstrap=49,
            bootstrap_weights="mammen",
            seed=123,
            rank_deficient_action="error",
        )
        results = est.fit(basic_data, "outcome", "unit", "period", "first_treat", "dose")
        assert results.base_period == "universal"
        assert results.anticipation == 0
        assert results.n_bootstrap == 49
        assert results.bootstrap_weights == "mammen"
        assert results.seed == 123
        assert results.rank_deficient_action == "error"

    def test_not_yet_treated_control(self):
        data = generate_continuous_did_data(
            n_units=100,
            n_periods=4,
            cohort_periods=[2, 3],
            seed=42,
        )
        est = ContinuousDiD(control_group="not_yet_treated")
        results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        assert isinstance(results, ContinuousDiDResults)


class TestContinuousDiDResults:
    """Test results object methods."""

    @pytest.fixture
    def results(self):
        data = generate_continuous_did_data(n_units=100, n_periods=3, seed=42, noise_sd=0.1)
        est = ContinuousDiD(n_bootstrap=49, seed=42)
        return est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

    def test_summary(self, results):
        s = results.summary()
        assert "ATT_glob" in s
        assert "ACRT_glob" in s
        assert "Continuous" in s

    def test_print_summary(self, results, capsys):
        results.print_summary()
        captured = capsys.readouterr()
        assert "ATT_glob" in captured.out

    def test_to_dataframe_dose_response(self, results):
        df = results.to_dataframe(level="dose_response")
        assert "dose" in df.columns
        assert "att" in df.columns
        assert "acrt" in df.columns
        assert len(df) == len(results.dose_grid)

    def test_to_dataframe_group_time(self, results):
        df = results.to_dataframe(level="group_time")
        assert "group" in df.columns
        assert "time" in df.columns
        assert "att_glob" in df.columns

    def test_to_dataframe_event_study_error(self, results):
        """Should error if event study not computed."""
        with pytest.raises(ValueError, match="[Ee]vent study"):
            results.to_dataframe(level="event_study")

    def test_to_dataframe_invalid_level(self, results):
        with pytest.raises(ValueError, match="Unknown level"):
            results.to_dataframe(level="invalid")

    def test_is_significant(self, results):
        assert isinstance(results.is_significant, bool)

    def test_significance_stars(self, results):
        stars = results.significance_stars
        assert stars in ("", ".", "*", "**", "***")

    def test_repr(self, results):
        r = repr(results)
        assert "ContinuousDiDResults" in r


class TestDoseAggregation:
    """Test dose-response aggregation across (g,t) cells."""

    def test_multi_period_aggregation(self):
        data = generate_continuous_did_data(
            n_units=200,
            n_periods=5,
            cohort_periods=[2, 4],
            seed=42,
            noise_sd=0.1,
        )
        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
            aggregate="dose",
        )
        # With linear DGP (ATT(d) = 1 + 2d) and degree=1, should recover well
        # ACRT should be close to 2.0
        assert abs(results.overall_acrt - 2.0) < 0.3

    def test_single_cohort_aggregation(self):
        data = generate_continuous_did_data(
            n_units=100,
            n_periods=3,
            seed=42,
            noise_sd=0.1,
        )
        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
            aggregate="dose",
        )
        assert len(results.groups) == 1
        assert np.isfinite(results.overall_att)


class TestEventStudyAggregation:
    """Test event-study aggregation path."""

    def test_event_study_computed(self):
        data = generate_continuous_did_data(
            n_units=200,
            n_periods=5,
            cohort_periods=[2, 4],
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=49, seed=42)
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
            aggregate="eventstudy",
        )
        assert results.event_study_effects is not None
        # Should have pre and post relative periods
        rel_periods = sorted(results.event_study_effects.keys())
        assert min(rel_periods) < 0  # Pre-treatment
        assert max(rel_periods) >= 0  # Post-treatment

    def test_event_study_to_dataframe(self):
        data = generate_continuous_did_data(
            n_units=200,
            n_periods=4,
            cohort_periods=[2, 3],
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD()
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
            aggregate="eventstudy",
        )
        df = results.to_dataframe(level="event_study")
        assert "relative_period" in df.columns
        assert "att_glob" in df.columns

    def test_event_study_not_yet_treated(self):
        """Event study with control_group='not_yet_treated' and analytic SE."""
        data = generate_continuous_did_data(
            n_units=200,
            n_periods=5,
            cohort_periods=[2, 4],
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(control_group="not_yet_treated", n_bootstrap=0)
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
            aggregate="eventstudy",
        )
        assert results.event_study_effects is not None
        rel_periods = sorted(results.event_study_effects.keys())
        assert min(rel_periods) < 0  # Pre-treatment
        assert max(rel_periods) >= 0  # Post-treatment
        for e, info in results.event_study_effects.items():
            assert np.isfinite(info["effect"]), f"effect is NaN for e={e}"
            assert np.isfinite(info["se"]), f"SE is NaN for e={e}"

    def test_event_study_universal_base_period(self):
        """Event study with base_period='universal' and analytic SE."""
        data = generate_continuous_did_data(
            n_units=200,
            n_periods=5,
            cohort_periods=[2, 4],
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(base_period="universal", n_bootstrap=0)
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
            aggregate="eventstudy",
        )
        assert results.event_study_effects is not None
        rel_periods = sorted(results.event_study_effects.keys())
        assert min(rel_periods) < 0  # Pre-treatment
        assert max(rel_periods) >= 0  # Post-treatment
        for e, info in results.event_study_effects.items():
            assert np.isfinite(info["effect"]), f"effect is NaN for e={e}"
            assert np.isfinite(info["se"]), f"SE is NaN for e={e}"

    def test_event_study_not_yet_treated_bootstrap(self, ci_params):
        """Event study with not_yet_treated control group and bootstrap SE."""
        n_boot = ci_params.bootstrap(99)
        data = generate_continuous_did_data(
            n_units=200,
            n_periods=5,
            cohort_periods=[2, 4],
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(
            control_group="not_yet_treated",
            n_bootstrap=n_boot,
            seed=42,
        )
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
            aggregate="eventstudy",
        )
        assert results.event_study_effects is not None
        rel_periods = sorted(results.event_study_effects.keys())
        assert min(rel_periods) < 0  # Pre-treatment
        assert max(rel_periods) >= 0  # Post-treatment
        for e, info in results.event_study_effects.items():
            if e >= 0:  # Post-treatment: SE and p-value should be finite
                assert np.isfinite(info["se"]), f"SE is NaN for post e={e}"
                assert np.isfinite(info["p_value"]), f"p_value is NaN for post e={e}"


class TestBootstrap:
    """Test bootstrap inference."""

    def test_bootstrap_ses_positive(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        data = generate_continuous_did_data(
            n_units=100,
            n_periods=3,
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        assert results.overall_att_se > 0
        assert results.overall_acrt_se > 0
        # Dose-response SEs should be positive
        assert np.all(results.dose_response_att.se > 0)

    def test_bootstrap_ci_contains_estimate(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        data = generate_continuous_did_data(
            n_units=100,
            n_periods=3,
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        lo, hi = results.overall_att_conf_int
        assert lo <= results.overall_att <= hi

    def test_bootstrap_acrt_ci_centered(self, ci_params):
        """Bootstrap ACRT CI should bracket the point estimate, not zero."""
        n_boot = ci_params.bootstrap(99)
        data = generate_continuous_did_data(
            n_units=200,
            n_periods=3,
            seed=42,
            noise_sd=0.5,
            att_function="linear",
            att_slope=2.0,
            att_intercept=1.0,
        )
        est = ContinuousDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        lo, hi = results.overall_acrt_conf_int
        assert lo <= results.overall_acrt <= hi, (
            f"ACRT CI [{lo:.4f}, {hi:.4f}] does not bracket "
            f"point estimate {results.overall_acrt:.4f}"
        )
        # CI midpoint should be closer to estimate than to 0
        midpoint = (lo + hi) / 2
        assert abs(midpoint - results.overall_acrt) < abs(midpoint), (
            f"CI midpoint {midpoint:.4f} is closer to 0 than to "
            f"estimate {results.overall_acrt:.4f} — bootstrap distribution "
            f"may still be mis-centered"
        )

    def test_bootstrap_p_values_valid(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        data = generate_continuous_did_data(
            n_units=100,
            n_periods=3,
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        assert 0 <= results.overall_att_p_value <= 1
        assert 0 <= results.overall_acrt_p_value <= 1

    def test_bootstrap_dose_response_p_values(self, ci_params):
        """Bootstrap dose-response should use bootstrap p-values, not normal approx."""
        n_boot = ci_params.bootstrap(99)
        data = generate_continuous_did_data(
            n_units=100,
            n_periods=3,
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        for curve in [results.dose_response_att, results.dose_response_acrt]:
            df = curve.to_dataframe()
            # Bootstrap mode: t-stat is undefined
            assert all(
                np.isnan(df["t_stat"])
            ), f"t_stat should be NaN in bootstrap mode for {curve.target}"
            # Bootstrap p-values should be present and valid
            assert all(
                np.isfinite(df["p_value"])
            ), f"p_value should be finite in bootstrap mode for {curve.target}"
            assert all(
                (df["p_value"] >= 0) & (df["p_value"] <= 1)
            ), f"p_value out of [0,1] range for {curve.target}"


class TestAnalyticalSE:
    """Test analytical standard errors (n_bootstrap=0)."""

    def test_analytical_se_positive(self):
        data = generate_continuous_did_data(
            n_units=100,
            n_periods=3,
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=0)
        results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        assert results.overall_att_se > 0
        assert results.overall_acrt_se > 0

    def test_analytical_ci(self):
        data = generate_continuous_did_data(
            n_units=100,
            n_periods=3,
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=0)
        results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        lo, hi = results.overall_att_conf_int
        assert lo < results.overall_att < hi


class TestEdgeCases:
    """Test edge cases."""

    def test_few_treated_units(self):
        """Estimator should handle very few treated units."""
        data = generate_continuous_did_data(
            n_units=30,
            n_periods=3,
            seed=42,
            never_treated_frac=0.8,  # Only ~6 treated
        )
        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        assert isinstance(results, ContinuousDiDResults)

    def test_inf_first_treat_normalization(self):
        """first_treat=inf should be treated as never-treated, and the caller
        must receive a UserWarning reporting the affected row count so the
        recategorization is not silent (axis-E counter)."""
        data = generate_continuous_did_data(n_units=50, n_periods=3, seed=42)
        data["first_treat"] = data["first_treat"].astype(float)
        inf_mask = data["first_treat"] == 0
        n_inf_rows = int(inf_mask.sum())
        data.loc[inf_mask, "first_treat"] = np.inf
        est = ContinuousDiD()

        with pytest.warns(
            UserWarning,
            match=rf"{n_inf_rows} row\(s\) have inf in 'first_treat'",
        ):
            results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        assert results.n_control_units > 0

    def test_no_inf_first_treat_no_warning(self):
        """No inf rows in first_treat — no recategorization warning."""
        import warnings

        data = generate_continuous_did_data(n_units=50, n_periods=3, seed=42)
        data["first_treat"] = data["first_treat"].astype(float)
        est = ContinuousDiD()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

        inf_warnings = [x for x in w if "inf in 'first_treat'" in str(x.message)]
        assert inf_warnings == []

    def test_nonzero_dose_on_never_treated_warns(self):
        """first_treat=0 (never-treated) rows with nonzero dose must now surface
        a UserWarning with the affected row count before the zeroing coercion.
        Before PR #331's CI-review follow-up this was silent."""
        # 4 units x 3 periods (12 rows). 2 units are never-treated (first_treat=0)
        # but carry dose=1.5 on every row — 6 rows should be reported.
        rows = []
        for unit in range(4):
            if unit < 2:
                ft, dose_val = 0.0, 1.5  # never-treated with nonzero dose
            else:
                ft, dose_val = 2.0, 1.0  # treated
            for t in range(1, 4):
                rows.append(
                    {
                        "unit": unit,
                        "period": t,
                        "outcome": float(unit + t),
                        "first_treat": ft,
                        "dose": dose_val,
                    }
                )
        data = pd.DataFrame(rows)
        est = ContinuousDiD()

        with pytest.warns(
            UserWarning,
            match=r"6 row\(s\) have 'first_treat'=0 \(never-treated\) but nonzero 'dose'",
        ):
            try:
                est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
            except Exception:
                # Downstream validation may reject this minimal panel (too few
                # treated for OLS); we only care about the dose-coercion warning.
                pass

    def test_negative_dose_on_never_treated_coerces_not_rejects(self):
        """Force-zero coercion applies to ANY nonzero dose on `first_treat=0`
        rows, including negative values. The negative-dose rejection at
        line 287-294 of continuous_did.py applies only to treated units
        (`first_treat > 0`); never-treated rows are coerced to dose=0
        with a `UserWarning` regardless of sign. This is observed
        implementation behavior for inconsistent inputs (an
        accidentally-nonzero dose on a never-treated row), NOT a
        documented routing option for manufacturing never-treated
        controls — REGISTRY does not list relabeling as a fallback.
        The test locks in the coercion contract; the autonomous guide
        §5.2 counter-example #5 explicitly tells agents not to use this
        path methodologically."""
        rows = []
        for unit in range(4):
            if unit < 2:
                ft, dose_val = 0.0, -1.5  # never-treated with NEGATIVE dose
            else:
                ft, dose_val = 2.0, 1.0  # treated, positive dose
            for t in range(1, 4):
                rows.append(
                    {
                        "unit": unit,
                        "period": t,
                        "outcome": float(unit + t),
                        "first_treat": ft,
                        "dose": dose_val,
                    }
                )
        data = pd.DataFrame(rows)
        est = ContinuousDiD()
        with pytest.warns(
            UserWarning,
            match=r"6 row\(s\) have 'first_treat'=0 \(never-treated\) but nonzero 'dose'",
        ):
            try:
                est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
            except ValueError as e:
                # Downstream may reject the minimal panel for other reasons,
                # but the rejection MUST NOT be the "negative dose" message
                # (which only applies to treated units).
                assert "negative dose" not in str(e).lower(), (
                    "Negative dose on never-treated rows must coerce, not "
                    "raise the treated-unit negative-dose error."
                )
            except Exception:
                # Other downstream errors (small-panel OLS) are acceptable;
                # the warning emission is what we are guarding here.
                pass

    def test_clean_never_treated_doses_silent(self):
        """Never-treated rows with dose=0 must not trigger the coercion warning."""
        import warnings

        data = generate_continuous_did_data(n_units=50, n_periods=3, seed=42)
        # generate_continuous_did_data already sets dose=0 for never-treated.
        est = ContinuousDiD()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

        coerce_warnings = [
            x for x in w if "never-treated" in str(x.message) and "nonzero 'dose'" in str(x.message)
        ]
        assert coerce_warnings == []

    def test_negative_first_treat_raises_with_row_count(self):
        """Negative `first_treat` (including -inf) must raise ValueError with
        the affected row count. Without this guard the affected units fall
        out of both the treated (g > 0) and never-treated (g == 0) masks and
        are silently excluded from the estimator."""
        rows = []
        for unit in range(4):
            # Unit 0: -inf. Unit 1: -2. Others: valid (0 or positive).
            if unit == 0:
                ft = -np.inf
            elif unit == 1:
                ft = -2.0
            else:
                ft = 0.0
            for t in range(1, 4):
                rows.append(
                    {
                        "unit": unit,
                        "period": t,
                        "outcome": float(unit + t),
                        "first_treat": ft,
                        "dose": 0.0,
                    }
                )
        data = pd.DataFrame(rows)
        est = ContinuousDiD()

        with pytest.raises(
            ValueError,
            match=r"6 row\(s\) have negative 'first_treat' values",
        ):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

    def test_nan_first_treat_raises_with_row_count(self):
        """NaN `first_treat` must raise ValueError with the row count. Without
        this guard, NaN rows survive preprocessing but match neither the
        treated (g > 0) nor never-treated (g == 0) mask, so the affected
        units would be silently excluded."""
        rows = []
        for unit in range(4):
            # Unit 0 has NaN first_treat across all 3 periods (3 NaN rows).
            ft = np.nan if unit == 0 else 0.0
            for t in range(1, 4):
                rows.append(
                    {
                        "unit": unit,
                        "period": t,
                        "outcome": float(unit + t),
                        "first_treat": ft,
                        "dose": 0.0,
                    }
                )
        data = pd.DataFrame(rows)
        est = ContinuousDiD()

        with pytest.raises(
            ValueError,
            match=r"3 row\(s\) have NaN 'first_treat' values",
        ):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

    def test_positive_inf_warning_silent_when_no_inf(self):
        """+inf warning is gated on +inf rows only; panels with only valid
        non-negative values (including just 0 and positive periods) must
        never trigger the recategorization warning."""
        import warnings

        rows = []
        for unit in range(4):
            ft = 0.0 if unit < 2 else 2.0
            for t in range(1, 4):
                rows.append(
                    {
                        "unit": unit,
                        "period": t,
                        "outcome": float(unit + t),
                        "first_treat": ft,
                        "dose": 0.0 if unit < 2 else 1.0,
                    }
                )
        data = pd.DataFrame(rows)
        est = ContinuousDiD()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
            except Exception:
                pass

        inf_warnings = [x for x in w if "inf in 'first_treat'" in str(x.message)]
        assert inf_warnings == []

    def test_inf_first_treat_warning_counts_rows_not_units(self):
        """The warning counts affected rows (not units). On a panel with
        multiple periods per unit, each inf row must count separately so the
        message surface matches the per-row semantics of `.replace(inf, 0)`."""
        # Build a 4-unit, 3-period panel (12 rows). 2 units have inf across
        # all 3 periods → 6 inf rows, 2 units, so row-count != unit-count.
        rows = []
        for unit in range(4):
            ft = np.inf if unit < 2 else 2.0
            dose = 0.0 if unit < 2 else 1.0
            for t in range(1, 4):
                rows.append(
                    {
                        "unit": unit,
                        "period": t,
                        "outcome": float(unit + t),
                        "first_treat": ft,
                        "dose": dose,
                    }
                )
        data = pd.DataFrame(rows)
        est = ContinuousDiD()

        with pytest.warns(
            UserWarning,
            match=r"6 row\(s\) have inf in 'first_treat'",
        ):
            try:
                est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
            except Exception:
                # Downstream validation may reject this minimal panel (too few
                # treated for OLS). We only care that the inf-row warning fires
                # with the correct row count.
                pass

    def test_custom_dvals(self):
        data = generate_continuous_did_data(n_units=100, n_periods=3, seed=42)
        custom_grid = np.array([1.0, 2.0, 3.0])
        est = ContinuousDiD(dvals=custom_grid)
        results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        np.testing.assert_array_equal(results.dose_grid, custom_grid)
        assert len(results.dose_response_att.effects) == 3

    def test_negative_dose_raises(self):
        """Negative doses among treated units should raise ValueError."""
        data = generate_continuous_did_data(n_units=50, n_periods=3, seed=42)
        # Set one treated unit's dose to negative
        treated_units = data.loc[data["first_treat"] > 0, "unit"].unique()
        data.loc[data["unit"] == treated_units[0], "dose"] = -1.0
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="negative dose"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

    def test_not_yet_treated_excludes_own_cohort(self):
        """not_yet_treated control group must not include the treated cohort itself.

        Construct a panel where contamination from including cohort g=2 in its own
        control set would produce a biased pre-treatment effect. With the fix,
        the pre-treatment ATT(g=2,t=1) should be near zero.
        """
        rng = np.random.RandomState(99)
        n_per_group = 20
        periods = [1, 2, 3, 4]

        rows = []
        # Group 1: never-treated (first_treat=0, dose=0)
        for i in range(n_per_group):
            uid = i
            for t in periods:
                rows.append(
                    {
                        "unit": uid,
                        "period": t,
                        "first_treat": 0,
                        "dose": 0.0,
                        "outcome": rng.normal(0, 0.5),
                    }
                )
        # Group 2: treated at period 2 (g=2), moderate dose
        for i in range(n_per_group):
            uid = n_per_group + i
            dose_i = rng.uniform(1, 3)
            for t in periods:
                y = rng.normal(0, 0.5)
                if t >= 2:
                    y += 5.0 * dose_i  # strong treatment effect
                rows.append(
                    {
                        "unit": uid,
                        "period": t,
                        "first_treat": 2,
                        "dose": dose_i,
                        "outcome": y,
                    }
                )
        # Group 3: treated at period 3 (g=3), high dose
        for i in range(n_per_group):
            uid = 2 * n_per_group + i
            dose_i = rng.uniform(1, 3)
            for t in periods:
                y = rng.normal(0, 0.5)
                if t >= 3:
                    y += 5.0 * dose_i
                rows.append(
                    {
                        "unit": uid,
                        "period": t,
                        "first_treat": 3,
                        "dose": dose_i,
                        "outcome": y,
                    }
                )

        data = pd.DataFrame(rows)
        est = ContinuousDiD(
            control_group="not_yet_treated",
            degree=1,
            num_knots=0,
            n_bootstrap=0,
        )
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
        )

        # Pre-treatment cells for g=2 should be near zero (t=1 is pre-treatment)
        # If cohort g=2 were included in its own control set, the pre-treatment
        # difference would be contaminated by the cohort's own outcomes
        pre_treatment_effects = {
            (g, t): v for (g, t), v in results.group_time_effects.items() if t < g
        }
        for (g, t), cell in pre_treatment_effects.items():
            att_glob = cell.get("att_glob", 0)
            assert abs(att_glob) < 2.0, (
                f"Pre-treatment ATT(g={g},t={t}) = {att_glob:.4f} is too large; "
                f"cohort may be contaminating its own control group"
            )


class TestAnalyticalSEParity:
    """Test analytical SE vs bootstrap SE agreement."""

    def test_analytical_se_matches_bootstrap(self, ci_params):
        """Analytical SEs should be within ~50% of bootstrap SEs."""
        n_boot = ci_params.bootstrap(999, min_n=199)
        data = generate_continuous_did_data(
            n_units=200,
            n_periods=3,
            seed=42,
            noise_sd=1.0,
        )
        est_boot = ContinuousDiD(n_bootstrap=n_boot, seed=42)
        results_boot = est_boot.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        est_analytic = ContinuousDiD(n_bootstrap=0)
        results_analytic = est_analytic.fit(
            data, "outcome", "unit", "period", "first_treat", "dose"
        )
        threshold = 0.50 if n_boot < 100 else 0.30
        ratio = results_analytic.overall_att_se / results_boot.overall_att_se
        assert (1 - threshold) < ratio < (1 + threshold) / (1 - threshold), (
            f"Analytical/bootstrap SE ratio = {ratio:.3f}, "
            f"expected within [{1 - threshold:.2f}, {(1 + threshold) / (1 - threshold):.2f}]"
        )


class TestDiscreteDoseWarning:
    """Test discrete dose detection warning."""

    def test_discrete_dose_warning(self):
        """Integer-valued doses should trigger a discrete dose warning."""
        data = generate_continuous_did_data(
            n_units=100,
            n_periods=3,
            seed=42,
        )
        data["dose"] = data["dose"].round().astype(float)
        data.loc[data["first_treat"] == 0, "dose"] = 0.0
        est = ContinuousDiD()
        with pytest.warns(UserWarning, match="[Dd]iscrete"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")


class TestAnticipationEventStudy:
    """Test event study with anticipation > 0."""

    def test_anticipation_event_study(self):
        """Event study with anticipation > 0 should include anticipation periods."""
        data = generate_continuous_did_data(
            n_units=100,
            n_periods=5,
            cohort_periods=[3],
            seed=42,
        )
        est = ContinuousDiD(anticipation=1, n_bootstrap=0)
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
            aggregate="eventstudy",
        )
        assert results.event_study_effects is not None
        # With anticipation=1 and g=3, post-treatment starts at t=2 (g - anticipation).
        # Relative times e = t - g, so t=2 → e=-1 (the anticipation period).
        rel_times = sorted(results.event_study_effects.keys())
        assert (
            -1 in rel_times
        ), f"Anticipation period e=-1 missing from event study; got {rel_times}"
        assert np.isfinite(results.event_study_effects[-1]["effect"])

    def test_anticipation_event_study_excludes_contaminated_periods(self):
        """With anticipation=2, event study should not contain e < -2."""
        rng = np.random.default_rng(42)
        n_per_group = 30
        periods = list(range(1, 9))  # 8 periods

        rows = []
        # Never-treated
        for i in range(n_per_group):
            for t in periods:
                rows.append(
                    {
                        "unit": i,
                        "period": t,
                        "first_treat": 0,
                        "dose": 0.0,
                        "outcome": rng.normal(0, 0.5),
                    }
                )
        # Cohort g=5 — treatment at t=5, anticipation=2 means post at t>=3
        for i in range(n_per_group):
            uid = n_per_group + i
            d = rng.uniform(0.5, 2.0)
            for t in periods:
                y = rng.normal(0, 0.5) + (2.0 * d if t >= 5 else 0)
                rows.append(
                    {
                        "unit": uid,
                        "period": t,
                        "first_treat": 5,
                        "dose": d,
                        "outcome": y,
                    }
                )

        data = pd.DataFrame(rows)
        est = ContinuousDiD(anticipation=2, n_bootstrap=0)
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
            aggregate="eventstudy",
        )
        assert results.event_study_effects is not None
        for e in results.event_study_effects.keys():
            assert e >= -2, f"Found relative period e={e} with anticipation=2; " f"expected e >= -2"

    def test_anticipation_not_yet_treated_excludes_anticipation_window(self):
        """Not-yet-treated controls must exclude cohorts in the anticipation window.

        With anticipation=1 and cohort g=3, computing ATT(g=3, t=4) should use
        threshold t + anticipation = 5, so cohort g=5 (unit_cohorts == 5) fails
        > 5 and is correctly excluded. Without the fix, threshold is t=4 and
        cohort g=5 passes > 4, contaminating controls with treated units.
        """
        rng = np.random.default_rng(42)
        n_per_group = 20
        periods = [1, 2, 3, 4, 5, 6]

        rows = []
        # Never-treated group
        for i in range(n_per_group):
            uid = i
            for t in periods:
                rows.append(
                    {
                        "unit": uid,
                        "period": t,
                        "first_treat": 0,
                        "dose": 0.0,
                        "outcome": rng.normal(0, 0.5),
                    }
                )

        # Early cohort: g=3, treatment effect = +5*dose at t>=3
        for i in range(n_per_group):
            uid = n_per_group + i
            d = rng.uniform(1, 3)
            for t in periods:
                y = rng.normal(0, 0.5) + (5.0 * d if t >= 3 else 0)
                rows.append(
                    {
                        "unit": uid,
                        "period": t,
                        "first_treat": 3,
                        "dose": d,
                        "outcome": y,
                    }
                )

        # Late cohort: g=5, treatment effect = +5*dose at t>=5
        for i in range(n_per_group):
            uid = 2 * n_per_group + i
            d = rng.uniform(1, 3)
            for t in periods:
                y = rng.normal(0, 0.5) + (5.0 * d if t >= 5 else 0)
                rows.append(
                    {
                        "unit": uid,
                        "period": t,
                        "first_treat": 5,
                        "dose": d,
                        "outcome": y,
                    }
                )

        data = pd.DataFrame(rows)

        est = ContinuousDiD(
            anticipation=1,
            control_group="not_yet_treated",
            n_bootstrap=0,
        )
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
        )

        assert np.isfinite(
            results.overall_att
        ), "overall_att should be finite with anticipation + not_yet_treated"
        assert results.dose_response_att is not None, "dose-response curve should exist"


class TestEmptyPostTreatment:
    """Test guard for empty post-treatment cells."""

    def test_no_post_treatment_cells_warns(self):
        """When no post-treatment cells exist, should warn and return NaN."""
        data = generate_continuous_did_data(
            n_units=50,
            n_periods=3,
            cohort_periods=[5],
            seed=42,
        )
        est = ContinuousDiD()
        with pytest.warns(UserWarning, match="[Nn]o post-treatment"):
            results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        assert np.isnan(results.overall_att)
        assert np.isnan(results.overall_acrt)


class TestParameterValidation:
    """Test parameter validation for constrained values."""

    def test_invalid_control_group_raises(self):
        """Invalid control_group should raise ValueError."""
        with pytest.raises(ValueError, match="control_group"):
            ContinuousDiD(control_group="invalid")

    def test_invalid_base_period_raises(self):
        """Invalid base_period should raise ValueError."""
        with pytest.raises(ValueError, match="base_period"):
            ContinuousDiD(base_period="invalid")

    def test_set_params_invalid_control_group_raises(self):
        """set_params with invalid control_group should raise ValueError."""
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="control_group"):
            est.set_params(control_group="NEVER_TREATED")

    def test_set_params_invalid_base_period_raises(self):
        """set_params with invalid base_period should raise ValueError."""
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="base_period"):
            est.set_params(base_period="VARYING")


class TestBootstrapPercentileInference:
    """Test that bootstrap uses percentile CI/p-value, not normal approximation."""

    def test_bootstrap_percentile_ci(self, ci_params):
        """Bootstrap CIs should use percentile method (generally asymmetric)."""
        n_boot = ci_params.bootstrap(499, min_n=199)
        data = generate_continuous_did_data(
            n_units=200,
            n_periods=3,
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        lo, hi = results.overall_att_conf_int
        estimate = results.overall_att
        # CI should contain estimate
        assert lo <= estimate <= hi
        # p-value should be finite and in [0, 1]
        assert 0 <= results.overall_att_p_value <= 1
        # Percentile CIs are generally asymmetric around the estimate.
        # With enough bootstrap reps, the upper and lower distances differ.
        upper_dist = hi - estimate
        lower_dist = estimate - lo
        # Just verify both distances are positive (CI is non-degenerate)
        assert upper_dist > 0
        assert lower_dist > 0


class TestNotYetTreatedNoDZeroError:
    """Test P(D=0)>0 error for not_yet_treated with no never-treated units."""

    def test_no_never_treated_raises(self):
        """not_yet_treated with zero never-treated units should raise ValueError."""
        data = generate_continuous_did_data(
            n_units=100,
            n_periods=4,
            cohort_periods=[2, 3],
            never_treated_frac=0.0,
            seed=42,
        )
        est = ContinuousDiD(control_group="not_yet_treated", degree=1, num_knots=0)
        with pytest.raises(ValueError, match="D=0"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")


class TestEventStudyAnalyticalSE:
    """Test analytical SEs for event study aggregation (n_bootstrap=0)."""

    def test_event_study_analytical_se_finite(self):
        """Event study with n_bootstrap=0 should produce finite SE/t/p for all bins."""
        data = generate_continuous_did_data(
            n_units=200,
            n_periods=5,
            cohort_periods=[2, 4],
            seed=42,
            noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=0)
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
            aggregate="eventstudy",
        )
        assert results.event_study_effects is not None
        for e, info in results.event_study_effects.items():
            assert np.isfinite(info["se"]), f"SE is NaN for e={e}"
            assert info["se"] > 0, f"SE is non-positive for e={e}"
            assert np.isfinite(info["t_stat"]), f"t_stat is NaN for e={e}"
            assert np.isfinite(info["p_value"]), f"p_value is NaN for e={e}"
            assert 0 <= info["p_value"] <= 1, f"p_value out of range for e={e}"
            lo, hi = info["conf_int"]
            assert np.isfinite(lo) and np.isfinite(hi), f"conf_int contains NaN for e={e}"


class TestContinuousDiDBreadRankGuard:
    """The ContinuousDiD ACRT-variance bread (Psi'WPsi) now routes through the
    shared ``_rank_guarded_inv``: a near-singular B-spline design Gram rank-
    reduces to a finite SE on the identified subspace and warns (the prior
    ``pinv`` fallback was minimum-norm AND silent)."""

    def test_rank_deficient_bspline_gram_warns_and_finite_se(self):
        import warnings
        from unittest.mock import patch

        import diff_diff.continuous_did as cd_mod

        data = generate_continuous_did_data(n_units=100, n_periods=3, seed=42, noise_sd=0.5)
        real_rgi = cd_mod._rank_guarded_inv

        def force_drop(A, **kwargs):
            # Finite inverse, but report a dropped direction to exercise the
            # rank-reduce warning path deterministically (B-spline Gram rank-
            # deficiency is hard to force via dose data alone).
            inv, _, rank = real_rgi(A, **kwargs)
            return inv, 1, rank

        with patch.object(cd_mod, "_rank_guarded_inv", side_effect=force_drop):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                results = ContinuousDiD().fit(
                    data, "outcome", "unit", "period", "first_treat", "dose"
                )
        msgs = [str(w.message) for w in caught]
        assert any(
            "ContinuousDiD ACRT variance" in m and "rank-deficient" in m for m in msgs
        ), msgs
        # rank-reduced bread still yields finite ACRT SEs.
        assert np.all(np.isfinite(results.dose_response_acrt.se))
