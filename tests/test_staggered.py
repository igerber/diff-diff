"""
Tests for Callaway-Sant'Anna staggered DiD estimator.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, CallawaySantAnnaResults


def generate_staggered_data(
    n_units: int = 100,
    n_periods: int = 10,
    n_cohorts: int = 3,
    treatment_effect: float = 2.0,
    never_treated_frac: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic staggered adoption data."""
    np.random.seed(seed)

    # Generate unit and time identifiers
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    # Assign treatment cohorts
    # Some units never treated, others treated in different periods
    n_never = int(n_units * never_treated_frac)
    n_treated = n_units - n_never

    # Treatment periods start from period 3 onwards
    cohort_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int)

    first_treat = np.zeros(n_units)
    if n_treated > 0:
        cohort_assignments = np.random.choice(len(cohort_periods), size=n_treated)
        first_treat[n_never:] = cohort_periods[cohort_assignments]

    first_treat_expanded = np.repeat(first_treat, n_periods)

    # Generate outcomes
    # Y = unit_fe + time_fe + treatment_effect * post + noise
    unit_fe = np.random.randn(n_units) * 2
    time_fe = np.linspace(0, 1, n_periods)

    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    # Treatment indicator
    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)

    # Dynamic treatment effects (effect grows over time)
    relative_time = times - first_treat_expanded
    dynamic_effect = treatment_effect * (1 + 0.1 * np.maximum(relative_time, 0))

    outcomes = (
        unit_fe_expanded +
        time_fe_expanded +
        dynamic_effect * post +
        np.random.randn(len(units)) * 0.5
    )

    df = pd.DataFrame({
        'unit': units,
        'time': times,
        'outcome': outcomes,
        'first_treat': first_treat_expanded.astype(int),
    })

    return df


class TestCallawaySantAnna:
    """Tests for CallawaySantAnna estimator."""

    def test_basic_fit(self):
        """Test basic model fitting."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert cs.is_fitted_
        assert isinstance(results, CallawaySantAnnaResults)
        assert results.overall_att is not None
        assert results.overall_se > 0
        assert len(results.group_time_effects) > 0

    def test_positive_treatment_effect(self):
        """Test that estimator recovers positive treatment effect."""
        data = generate_staggered_data(treatment_effect=3.0, seed=123)

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Should detect positive effect
        assert results.overall_att > 0
        # Effect should be roughly correct (within 2 SE)
        assert abs(results.overall_att - 3.0) < 2 * results.overall_se + 1.0

    def test_zero_treatment_effect(self):
        """Test with no treatment effect."""
        data = generate_staggered_data(treatment_effect=0.0, seed=456)

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Effect should be close to zero
        assert abs(results.overall_att) < 3 * results.overall_se

    def test_event_study_aggregation(self):
        """Test event study aggregation."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='event_study'
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0

        # Check that relative periods are present
        rel_periods = list(results.event_study_effects.keys())
        assert any(p >= 0 for p in rel_periods)  # Post-treatment

    def test_group_aggregation(self):
        """Test aggregation by treatment cohort."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='group'
        )

        assert results.group_effects is not None
        assert len(results.group_effects) > 0

    def test_all_aggregation(self):
        """Test computing all aggregations."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='all'
        )

        assert results.event_study_effects is not None
        assert results.group_effects is not None

    def test_control_group_options(self):
        """Test different control group options."""
        data = generate_staggered_data()

        # Never treated only
        cs1 = CallawaySantAnna(control_group="never_treated")
        results1 = cs1.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Not yet treated
        cs2 = CallawaySantAnna(control_group="not_yet_treated")
        results2 = cs2.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert results1.control_group == "never_treated"
        assert results2.control_group == "not_yet_treated"
        # Results should be different
        assert results1.overall_att != results2.overall_att

    def test_estimation_methods(self):
        """Test different estimation methods."""
        data = generate_staggered_data()

        methods = ["reg", "ipw", "dr"]
        results = {}

        for method in methods:
            cs = CallawaySantAnna(estimation_method=method)
            results[method] = cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )

        # All methods should produce results
        for method, res in results.items():
            assert res.overall_att is not None

    def test_summary_output(self):
        """Test summary output formatting."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='event_study'
        )

        summary = results.summary()

        assert "Callaway-Sant'Anna" in summary
        assert "ATT" in summary
        assert "Std. Err." in summary

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='all'
        )

        # Group-time DataFrame
        df_gt = results.to_dataframe(level='group_time')
        assert 'group' in df_gt.columns
        assert 'time' in df_gt.columns
        assert 'effect' in df_gt.columns

        # Event study DataFrame
        df_es = results.to_dataframe(level='event_study')
        assert 'relative_period' in df_es.columns

        # Group DataFrame
        df_g = results.to_dataframe(level='group')
        assert 'group' in df_g.columns

    def test_get_set_params(self):
        """Test sklearn-compatible parameter access."""
        cs = CallawaySantAnna(alpha=0.10, control_group="not_yet_treated")

        params = cs.get_params()
        assert params['alpha'] == 0.10
        assert params['control_group'] == "not_yet_treated"

        cs.set_params(alpha=0.05)
        assert cs.alpha == 0.05

    def test_missing_column_error(self):
        """Test error on missing columns."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()

        with pytest.raises(ValueError, match="Missing columns"):
            cs.fit(
                data,
                outcome='nonexistent',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )

    def test_no_control_units_error(self):
        """Test error when no control units exist."""
        data = generate_staggered_data(never_treated_frac=0.0)

        # All units are treated, no controls
        cs = CallawaySantAnna()

        with pytest.raises(ValueError, match="No never-treated units"):
            cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )

    def test_significance_properties(self):
        """Test significance-related properties."""
        data = generate_staggered_data(treatment_effect=5.0)

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # With strong effect, should be significant
        assert results.is_significant
        assert results.significance_stars in ["*", "**", "***"]


class TestCallawaySantAnnaResults:
    """Tests for CallawaySantAnnaResults class."""

    def test_repr(self):
        """Test string representation."""
        data = generate_staggered_data()
        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        repr_str = repr(results)
        assert "CallawaySantAnnaResults" in repr_str
        assert "ATT=" in repr_str

    def test_invalid_level_error(self):
        """Test error on invalid DataFrame level."""
        data = generate_staggered_data()
        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        with pytest.raises(ValueError, match="Unknown level"):
            results.to_dataframe(level='invalid')

    def test_event_study_not_computed_error(self):
        """Test error when event study not computed."""
        data = generate_staggered_data()
        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        with pytest.raises(ValueError, match="Event study effects not computed"):
            results.to_dataframe(level='event_study')
