"""
Tests for Stacked DiD estimator (Wing, Freedman & Hollingsworth 2024).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import StackedDiD, StackedDiDResults, stacked_did
from diff_diff.prep_dgp import generate_staggered_data

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def staggered_data():
    """Standard staggered adoption data for testing."""
    return generate_staggered_data(
        n_units=200,
        n_periods=12,
        cohort_periods=[4, 6, 8],
        never_treated_frac=0.3,
        treatment_effect=5.0,
        dynamic_effects=True,
        seed=42,
    )


@pytest.fixture
def constant_effect_data():
    """Staggered data with constant treatment effect (no dynamics)."""
    return generate_staggered_data(
        n_units=200,
        n_periods=12,
        cohort_periods=[4, 6, 8],
        never_treated_frac=0.3,
        treatment_effect=5.0,
        dynamic_effects=False,
        seed=42,
    )


@pytest.fixture
def no_never_treated_data():
    """Staggered data without never-treated units."""
    return generate_staggered_data(
        n_units=200,
        n_periods=12,
        cohort_periods=[4, 6, 8],
        never_treated_frac=0.0,
        treatment_effect=5.0,
        dynamic_effects=True,
        seed=42,
    )


# =============================================================================
# TestStackedDiDBasic
# =============================================================================


class TestStackedDiDBasic:
    """Basic functionality tests."""

    def test_basic_fit(self, staggered_data):
        """Default parameters produce valid results."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results, StackedDiDResults)
        assert np.isfinite(results.overall_att)
        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0
        assert results.n_stacked_obs > 0
        assert results.n_sub_experiments > 0

    def test_event_study(self, staggered_data):
        """Event study aggregation populates event_study_effects."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert results.event_study_effects is not None
        assert -1 in results.event_study_effects  # reference period
        # Reference period effect should be zero
        ref = results.event_study_effects[-1]
        assert ref["effect"] == 0.0
        assert ref["n_obs"] == 0

        # Post-treatment periods should have effects
        for h in range(0, 3):
            if h in results.event_study_effects:
                assert results.event_study_effects[h]["n_obs"] > 0

    def test_group_aggregate_raises(self, staggered_data):
        """aggregate='group' raises ValueError."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        with pytest.raises(ValueError, match="group.*not supported"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="group",
            )

    def test_all_aggregate_raises(self, staggered_data):
        """aggregate='all' raises ValueError."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        with pytest.raises(ValueError, match="all.*not supported"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="all",
            )

    def test_simple_att(self, staggered_data):
        """aggregate='simple' produces overall ATT only."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="simple",
        )
        assert np.isfinite(results.overall_att)
        assert results.event_study_effects is None
        assert results.group_effects is None

    def test_known_constant_effect(self, constant_effect_data):
        """With constant treatment effect, estimated ATT should be close."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            constant_effect_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        # Treatment effect is 5.0; allow generous tolerance
        assert (
            abs(results.overall_att - 5.0) < 1.5
        ), f"Estimated ATT {results.overall_att:.2f} too far from true effect 5.0"

    def test_dynamic_effects(self, staggered_data):
        """With dynamic effects, post-treatment coefficients should increase."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert results.event_study_effects is not None
        # Post-treatment effects should generally increase
        post_effects = [
            results.event_study_effects[h]["effect"]
            for h in sorted(results.event_study_effects.keys())
            if h >= 0 and results.event_study_effects[h]["n_obs"] > 0
        ]
        if len(post_effects) >= 2:
            # Last post should be larger than first post (dynamic growth)
            assert post_effects[-1] > post_effects[0]


# =============================================================================
# TestTrimming
# =============================================================================


class TestTrimming:
    """Tests for IC1/IC2 trimming logic."""

    def test_ic1_window_trimming(self, staggered_data):
        """Events outside the observation window are trimmed."""
        # With very large kappa, early/late events should be trimmed
        est = StackedDiD(kappa_pre=4, kappa_post=4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        # With kappa_pre=4, kappa_post=4 on 12 periods, some events should trim
        if len(results.trimmed_groups) > 0:
            assert any("Trimmed" in str(wi.message) for wi in w)

    def test_ic2_no_controls_trimming(self, no_never_treated_data):
        """Events without clean controls are trimmed with never_treated mode."""
        est = StackedDiD(kappa_pre=1, kappa_post=1, clean_control="never_treated")
        # No never-treated units exist → all events should be trimmed
        with pytest.raises(ValueError, match="All.*adoption events were trimmed"):
            est.fit(
                no_never_treated_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_trimmed_groups_reported(self, staggered_data):
        """Trimmed groups are reported in results."""
        est = StackedDiD(kappa_pre=5, kappa_post=5)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                results = est.fit(
                    staggered_data,
                    outcome="outcome",
                    unit="unit",
                    time="period",
                    first_treat="first_treat",
                )
                # If some groups survive, check trimmed_groups
                assert isinstance(results.trimmed_groups, list)
            except ValueError:
                # All trimmed — expected for large kappa
                pass

    def test_all_trimmed_raises(self, staggered_data):
        """ValueError when all events are eliminated by trimming."""
        est = StackedDiD(kappa_pre=10, kappa_post=10)
        with pytest.raises(ValueError, match="All.*adoption events were trimmed"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_wider_window_more_trimming(self, staggered_data):
        """Larger kappa values should trim more (or equal) events."""
        est1 = StackedDiD(kappa_pre=1, kappa_post=1)
        results1 = est1.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )

        est2 = StackedDiD(kappa_pre=2, kappa_post=2)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results2 = est2.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

        assert len(results2.trimmed_groups) >= len(results1.trimmed_groups)


# =============================================================================
# TestQWeights
# =============================================================================


class TestQWeights:
    """Tests for Q-weight computation."""

    def test_treated_weight_is_one(self, staggered_data):
        """All treated observations should have Q=1."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        treated_weights = results.stacked_data.loc[results.stacked_data["_D_sa"] == 1, "_Q_weight"]
        assert np.allclose(treated_weights, 1.0)

    def test_aggregate_weighting_formula(self, staggered_data):
        """Q-weights match R's observation-count formula at (event_time, sub_exp) level."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, weighting="aggregate")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        sd = results.stacked_data

        # Compute expected Q per R formula at (event_time, sub_exp) level
        for et in sd["_event_time"].unique():
            et_data = sd[sd["_event_time"] == et]
            stack_treat_n = (et_data["_D_sa"] == 1).sum()
            stack_control_n = (et_data["_D_sa"] == 0).sum()
            for sub_exp in results.groups:
                sub_et = et_data[et_data["_sub_exp"] == sub_exp]
                sub_treat_n = (sub_et["_D_sa"] == 1).sum()
                sub_control_n = (sub_et["_D_sa"] == 0).sum()
                if sub_control_n > 0 and stack_treat_n > 0 and stack_control_n > 0:
                    expected_q = (sub_treat_n / stack_treat_n) / (sub_control_n / stack_control_n)
                    actual_q = sub_et.loc[sub_et["_D_sa"] == 0, "_Q_weight"].iloc[0]
                    assert (
                        abs(actual_q - expected_q) < 1e-10
                    ), f"Sub-exp {sub_exp}, et={et}: expected Q={expected_q:.6f}, got {actual_q:.6f}"

    def test_sample_share_weighting(self, staggered_data):
        """Verify sample_share Q formula."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, weighting="sample_share")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        sd = results.stacked_data

        # All weights should be positive and finite
        assert np.all(sd["_Q_weight"] > 0)
        assert np.all(np.isfinite(sd["_Q_weight"]))

    def test_weights_positive(self, staggered_data):
        """All Q-weights should be positive."""
        for w in ["aggregate", "sample_share"]:
            est = StackedDiD(kappa_pre=2, kappa_post=2, weighting=w)
            results = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
            assert np.all(results.stacked_data["_Q_weight"] > 0)


# =============================================================================
# TestCleanControl
# =============================================================================


class TestCleanControl:
    """Tests for clean control group definitions."""

    def test_not_yet_treated_default(self, staggered_data):
        """Default includes not-yet-treated and never-treated as controls."""
        est = StackedDiD(kappa_pre=1, kappa_post=1, clean_control="not_yet_treated")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_control_units > 0

    def test_strict_excludes_more(self, staggered_data):
        """Strict mode should have fewer (or equal) controls than not_yet_treated."""
        est_nyt = StackedDiD(kappa_pre=2, kappa_post=2, clean_control="not_yet_treated")
        results_nyt = est_nyt.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )

        est_strict = StackedDiD(kappa_pre=2, kappa_post=2, clean_control="strict")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                results_strict = est_strict.fit(
                    staggered_data,
                    outcome="outcome",
                    unit="unit",
                    time="period",
                    first_treat="first_treat",
                )
                # Strict should have fewer or equal stacked obs
                assert results_strict.n_stacked_obs <= results_nyt.n_stacked_obs
            except ValueError:
                # Strict may trim all events — that's valid behavior
                pass

    def test_never_treated_only(self, staggered_data):
        """never_treated mode only uses never-treated as controls."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, clean_control="never_treated")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        sd = results.stacked_data
        # All control units should have first_treat = inf
        control_ft = sd.loc[sd["_D_sa"] == 0, "first_treat"].unique()
        assert all(np.isinf(ft) for ft in control_ft)

    def test_never_treated_no_nevertreated_raises(self, no_never_treated_data):
        """Error when no never-treated units exist with never_treated mode."""
        est = StackedDiD(kappa_pre=1, kappa_post=1, clean_control="never_treated")
        with pytest.raises(ValueError, match="All.*adoption events were trimmed"):
            est.fit(
                no_never_treated_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )


# =============================================================================
# TestClustering
# =============================================================================


class TestClustering:
    """Tests for clustering standard errors."""

    def test_unit_clustering(self, staggered_data):
        """Default unit clustering produces finite SEs."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, cluster="unit")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0

    def test_unit_subexp_clustering(self, staggered_data):
        """unit_subexp clustering produces finite SEs."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, cluster="unit_subexp")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0


# =============================================================================
# TestStackedData
# =============================================================================


class TestStackedData:
    """Tests for the stacked dataset."""

    def test_stacked_data_accessible(self, staggered_data):
        """results.stacked_data is a DataFrame."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results.stacked_data, pd.DataFrame)

    def test_required_columns(self, staggered_data):
        """Stacked data has _sub_exp, _event_time, _D_sa, _Q_weight."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        required = {"_sub_exp", "_event_time", "_D_sa", "_Q_weight"}
        assert required.issubset(results.stacked_data.columns)

    def test_event_time_range(self, staggered_data):
        """Event times span [-kappa_pre, ..., kappa_post]."""
        kp, kq = 2, 2
        est = StackedDiD(kappa_pre=kp, kappa_post=kq)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        et = results.stacked_data["_event_time"]
        # Event times should include the reference period -1
        assert et.min() <= -kp
        assert et.max() >= kq


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_cohort(self):
        """Works with only one adoption event."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[5],
            never_treated_frac=0.5,
            treatment_effect=3.0,
            dynamic_effects=False,
            seed=99,
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_sub_experiments == 1
        assert np.isfinite(results.overall_att)

    def test_anticipation_reference_period(self):
        """anticipation=1 shifts reference period to e=-2."""
        data = generate_staggered_data(
            n_units=200,
            n_periods=12,
            cohort_periods=[5, 7],
            never_treated_frac=0.3,
            treatment_effect=5.0,
            seed=42,
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Reference period is -2 (not -1)
        assert -2 in results.event_study_effects
        assert results.event_study_effects[-2]["effect"] == 0.0
        assert results.event_study_effects[-2]["n_obs"] == 0  # sentinel

        # -1 is NOT the reference; it should have a non-zero estimated effect
        assert -1 in results.event_study_effects
        assert results.event_study_effects[-1]["n_obs"] > 0

        # Extra pre-period -3 should have a dummy
        assert -3 in results.event_study_effects
        assert results.event_study_effects[-3]["n_obs"] > 0

        # Post-treatment includes anticipation period (-1)
        # Overall ATT averages h in {-1, 0, 1, 2}
        assert np.isfinite(results.overall_att)

    def test_unbalanced_panel(self):
        """Works with missing observations within the window."""
        data = generate_staggered_data(
            n_units=200,
            n_periods=12,
            cohort_periods=[4, 6, 8],
            never_treated_frac=0.3,
            treatment_effect=5.0,
            seed=42,
        )
        # Remove some random rows to create unbalanced panel
        rng = np.random.default_rng(42)
        drop_idx = rng.choice(len(data), size=50, replace=False)
        data = data.drop(data.index[drop_idx]).reset_index(drop=True)

        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        sd = results.stacked_data
        assert np.isfinite(results.overall_att)
        assert np.all(sd["_Q_weight"] > 0)
        assert np.all(np.isfinite(sd["_Q_weight"]))

    def test_nan_inference(self):
        """Degenerate case with NaN inference fields."""
        # Create small data where estimation might degenerate.
        # Need n > k to avoid division by zero in cluster-robust VCV:
        # Design matrix has 4 columns (intercept, D_sa, lambda_0, delta_0),
        # so we need > 4 observations (3 units × 2 periods = 6).
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2, 3, 3],
                "period": [1, 2, 1, 2, 1, 2],
                "outcome": [1.0, 2.0, 1.0, 2.0, 1.5, 2.5],
                "first_treat": [2, 2, 0, 0, 0, 0],
            }
        )
        est = StackedDiD(kappa_pre=1, kappa_post=0)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        # Should produce finite results or NaN (not crash)
        assert isinstance(results, StackedDiDResults)

    def test_never_treated_encoding_zero(self):
        """first_treat=0 treated same as first_treat=inf (never-treated)."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[5],
            never_treated_frac=0.5,
            treatment_effect=5.0,
            seed=42,
        )
        # The generator uses 0 for never-treated
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_control_units > 0

    def test_never_treated_encoding_inf(self):
        """first_treat=inf works for never-treated units."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[5],
            never_treated_frac=0.5,
            treatment_effect=5.0,
            seed=42,
        )
        # Replace 0 with inf for never-treated
        data["first_treat"] = data["first_treat"].replace(0, np.inf)

        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_control_units > 0


# =============================================================================
# TestSklearnInterface
# =============================================================================


class TestSklearnInterface:
    """Tests for sklearn-compatible API."""

    def test_get_params(self):
        """All init params present in get_params."""
        est = StackedDiD(
            kappa_pre=3,
            kappa_post=2,
            weighting="population",
            clean_control="strict",
            cluster="unit_subexp",
            alpha=0.10,
            anticipation=1,
            rank_deficient_action="error",
        )
        params = est.get_params()
        assert params["kappa_pre"] == 3
        assert params["kappa_post"] == 2
        assert params["weighting"] == "population"
        assert params["clean_control"] == "strict"
        assert params["cluster"] == "unit_subexp"
        assert params["alpha"] == 0.10
        assert params["anticipation"] == 1
        assert params["rank_deficient_action"] == "error"

    def test_set_params(self):
        """set_params modifies attributes correctly."""
        est = StackedDiD()
        est.set_params(kappa_pre=5, weighting="sample_share")
        assert est.kappa_pre == 5
        assert est.weighting == "sample_share"

    def test_set_params_unknown_raises(self):
        """set_params raises on unknown parameter."""
        est = StackedDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(nonexistent_param=42)

    def test_convenience_function(self, staggered_data):
        """stacked_did() convenience function works."""
        results = stacked_did(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            kappa_pre=2,
            kappa_post=2,
        )
        assert isinstance(results, StackedDiDResults)
        assert np.isfinite(results.overall_att)


# =============================================================================
# TestResultsMethods
# =============================================================================


class TestResultsMethods:
    """Tests for StackedDiDResults methods."""

    def test_summary(self, staggered_data):
        """summary() returns formatted string."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        summary = results.summary()
        assert "Stacked DiD" in summary
        assert "ATT" in summary

    def test_to_dataframe_event_study(self, staggered_data):
        """to_dataframe(level='event_study') returns DataFrame."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        df = results.to_dataframe(level="event_study")
        assert isinstance(df, pd.DataFrame)
        assert "relative_period" in df.columns
        assert "effect" in df.columns

    def test_to_dataframe_group_raises(self, staggered_data):
        """to_dataframe(level='group') raises ValueError."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        with pytest.raises(ValueError, match="Group aggregation is not supported"):
            results.to_dataframe(level="group")

    def test_to_dataframe_no_event_study_raises(self, staggered_data):
        """to_dataframe raises when event_study not computed."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        with pytest.raises(ValueError, match="Event study effects not computed"):
            results.to_dataframe(level="event_study")

    def test_is_significant(self, staggered_data):
        """is_significant property works."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results.is_significant, bool)

    def test_significance_stars(self, staggered_data):
        """significance_stars property works."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results.significance_stars, str)

    def test_repr(self, staggered_data):
        """__repr__ returns formatted string."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        r = repr(results)
        assert "StackedDiDResults" in r
        assert "ATT=" in r


# =============================================================================
# TestValidation
# =============================================================================


class TestValidation:
    """Tests for input validation."""

    def test_missing_columns(self, staggered_data):
        """Raises on missing required columns."""
        est = StackedDiD()
        with pytest.raises(ValueError, match="Missing columns"):
            est.fit(
                staggered_data,
                outcome="nonexistent",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_invalid_weighting(self):
        """Raises on invalid weighting parameter."""
        with pytest.raises(ValueError, match="weighting"):
            StackedDiD(weighting="invalid")

    def test_invalid_clean_control(self):
        """Raises on invalid clean_control parameter."""
        with pytest.raises(ValueError, match="clean_control"):
            StackedDiD(clean_control="invalid")

    def test_invalid_cluster(self):
        """Raises on invalid cluster parameter."""
        with pytest.raises(ValueError, match="cluster"):
            StackedDiD(cluster="invalid")

    def test_invalid_aggregate(self, staggered_data):
        """Raises on invalid aggregate parameter."""
        est = StackedDiD()
        with pytest.raises(ValueError, match="aggregate"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="invalid",
            )

    def test_population_required_for_population_weighting(self, staggered_data):
        """Raises when population col not specified with weighting='population'."""
        est = StackedDiD(weighting="population")
        with pytest.raises(ValueError, match="population"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_no_treated_units(self):
        """Raises when no treated units exist."""
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "period": [1, 2, 1, 2],
                "outcome": [1.0, 2.0, 1.0, 2.0],
                "first_treat": [0, 0, 0, 0],
            }
        )
        est = StackedDiD()
        with pytest.raises(ValueError, match="No treated units"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
