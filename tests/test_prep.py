"""
Tests for data preparation utility functions.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.prep import (
    aggregate_survey,
    aggregate_to_cohorts,
    balance_panel,
    create_event_time,
    generate_did_data,
    make_post_indicator,
    make_treatment_indicator,
    summarize_did_data,
    validate_did_data,
    wide_to_long,
)
from diff_diff.survey import SurveyDesign


class TestMakeTreatmentIndicator:
    """Tests for make_treatment_indicator function."""

    def test_categorical_single_value(self):
        """Test treatment from single categorical value."""
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "y": [1, 2, 3, 4]})
        result = make_treatment_indicator(df, "group", treated_values="A")
        assert result["treated"].tolist() == [1, 1, 0, 0]

    def test_categorical_multiple_values(self):
        """Test treatment from multiple categorical values."""
        df = pd.DataFrame({"group": ["A", "B", "C", "D"], "y": [1, 2, 3, 4]})
        result = make_treatment_indicator(df, "group", treated_values=["A", "B"])
        assert result["treated"].tolist() == [1, 1, 0, 0]

    def test_threshold_above(self):
        """Test treatment from numeric threshold (above)."""
        df = pd.DataFrame({"size": [10, 50, 100, 200], "y": [1, 2, 3, 4]})
        result = make_treatment_indicator(df, "size", threshold=75)
        assert result["treated"].tolist() == [0, 0, 1, 1]

    def test_threshold_below(self):
        """Test treatment from numeric threshold (below)."""
        df = pd.DataFrame({"size": [10, 50, 100, 200], "y": [1, 2, 3, 4]})
        result = make_treatment_indicator(df, "size", threshold=75, above_threshold=False)
        assert result["treated"].tolist() == [1, 1, 0, 0]

    def test_custom_column_name(self):
        """Test custom output column name."""
        df = pd.DataFrame({"group": ["A", "B"], "y": [1, 2]})
        result = make_treatment_indicator(df, "group", treated_values="A", new_column="is_treated")
        assert "is_treated" in result.columns
        assert result["is_treated"].tolist() == [1, 0]

    def test_original_unchanged(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({"group": ["A", "B"], "y": [1, 2]})
        original_cols = df.columns.tolist()
        make_treatment_indicator(df, "group", treated_values="A")
        assert df.columns.tolist() == original_cols

    def test_error_both_params(self):
        """Test error when both treated_values and threshold specified."""
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
        with pytest.raises(ValueError, match="Specify either"):
            make_treatment_indicator(df, "x", treated_values=1, threshold=1.5)

    def test_error_neither_param(self):
        """Test error when neither treated_values nor threshold specified."""
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
        with pytest.raises(ValueError, match="Must specify either"):
            make_treatment_indicator(df, "x")

    def test_error_column_not_found(self):
        """Test error when column doesn't exist."""
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            make_treatment_indicator(df, "missing", treated_values=1)


class TestMakePostIndicator:
    """Tests for make_post_indicator function."""

    def test_post_periods_single(self):
        """Test post indicator from single period value."""
        df = pd.DataFrame({"year": [2018, 2019, 2020, 2021], "y": [1, 2, 3, 4]})
        result = make_post_indicator(df, "year", post_periods=2020)
        assert result["post"].tolist() == [0, 0, 1, 0]

    def test_post_periods_multiple(self):
        """Test post indicator from multiple period values."""
        df = pd.DataFrame({"year": [2018, 2019, 2020, 2021], "y": [1, 2, 3, 4]})
        result = make_post_indicator(df, "year", post_periods=[2020, 2021])
        assert result["post"].tolist() == [0, 0, 1, 1]

    def test_treatment_start(self):
        """Test post indicator from treatment start."""
        df = pd.DataFrame({"year": [2018, 2019, 2020, 2021], "y": [1, 2, 3, 4]})
        result = make_post_indicator(df, "year", treatment_start=2020)
        assert result["post"].tolist() == [0, 0, 1, 1]

    def test_datetime_column(self):
        """Test with datetime column."""
        df = pd.DataFrame(
            {"date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]), "y": [1, 2, 3]}
        )
        result = make_post_indicator(df, "date", treatment_start="2020-06-01")
        assert result["post"].tolist() == [0, 1, 1]

    def test_custom_column_name(self):
        """Test custom output column name."""
        df = pd.DataFrame({"year": [2018, 2019], "y": [1, 2]})
        result = make_post_indicator(df, "year", post_periods=2019, new_column="after")
        assert "after" in result.columns

    def test_error_both_params(self):
        """Test error when both post_periods and treatment_start specified."""
        df = pd.DataFrame({"year": [2018, 2019], "y": [1, 2]})
        with pytest.raises(ValueError, match="Specify either"):
            make_post_indicator(df, "year", post_periods=[2019], treatment_start=2019)

    def test_error_neither_param(self):
        """Test error when neither parameter specified."""
        df = pd.DataFrame({"year": [2018, 2019], "y": [1, 2]})
        with pytest.raises(ValueError, match="Must specify either"):
            make_post_indicator(df, "year")


class TestWideToLong:
    """Tests for wide_to_long function."""

    def test_basic_conversion(self):
        """Test basic wide to long conversion."""
        wide_df = pd.DataFrame(
            {
                "firm_id": [1, 2],
                "sales_2019": [100, 150],
                "sales_2020": [110, 160],
                "sales_2021": [120, 170],
            }
        )
        result = wide_to_long(
            wide_df,
            value_columns=["sales_2019", "sales_2020", "sales_2021"],
            id_column="firm_id",
            time_name="year",
            value_name="sales",
        )
        assert len(result) == 6
        assert set(result.columns) == {"firm_id", "year", "sales"}

    def test_with_time_values(self):
        """Test with explicit time values."""
        wide_df = pd.DataFrame({"id": [1], "t1": [10], "t2": [20]})
        result = wide_to_long(
            wide_df, value_columns=["t1", "t2"], id_column="id", time_values=[2020, 2021]
        )
        assert result["period"].tolist() == [2020, 2021]

    def test_preserves_other_columns(self):
        """Test that other columns are preserved."""
        wide_df = pd.DataFrame({"id": [1, 2], "group": ["A", "B"], "t1": [10, 20], "t2": [15, 25]})
        result = wide_to_long(wide_df, value_columns=["t1", "t2"], id_column="id")
        assert "group" in result.columns
        assert result[result["id"] == 1]["group"].tolist() == ["A", "A"]

    def test_error_empty_value_columns(self):
        """Test error with empty value columns."""
        df = pd.DataFrame({"id": [1]})
        with pytest.raises(ValueError, match="cannot be empty"):
            wide_to_long(df, value_columns=[], id_column="id")

    def test_error_mismatched_time_values(self):
        """Test error when time_values length doesn't match."""
        df = pd.DataFrame({"id": [1], "t1": [10], "t2": [20]})
        with pytest.raises(ValueError, match="length"):
            wide_to_long(df, value_columns=["t1", "t2"], id_column="id", time_values=[2020])


class TestBalancePanel:
    """Tests for balance_panel function."""

    def test_inner_balance(self):
        """Test inner balance (keep complete units only)."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2, 2, 3, 3, 3],
                "period": [1, 2, 3, 1, 2, 1, 2, 3],
                "y": [10, 11, 12, 20, 21, 30, 31, 32],
            }
        )
        result = balance_panel(df, "unit", "period", method="inner")
        assert set(result["unit"].unique()) == {1, 3}
        assert len(result) == 6

    def test_outer_balance(self):
        """Test outer balance (include all combinations)."""
        df = pd.DataFrame({"unit": [1, 1, 2], "period": [1, 2, 1], "y": [10, 11, 20]})
        result = balance_panel(df, "unit", "period", method="outer")
        assert len(result) == 4  # 2 units x 2 periods

    def test_fill_with_value(self):
        """Test fill method with specific value."""
        df = pd.DataFrame({"unit": [1, 1, 2], "period": [1, 2, 1], "y": [10.0, 11.0, 20.0]})
        result = balance_panel(df, "unit", "period", method="fill", fill_value=0.0)
        assert len(result) == 4
        missing_row = result[(result["unit"] == 2) & (result["period"] == 2)]
        assert missing_row["y"].values[0] == 0.0

    def test_fill_forward_backward(self):
        """Test fill method with forward/backward fill."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2, 2],
                "period": [1, 2, 3, 1, 3],  # Unit 2 missing period 2
                "y": [10.0, 11.0, 12.0, 20.0, 22.0],
            }
        )
        result = balance_panel(df, "unit", "period", method="fill", fill_value=None)
        assert len(result) == 6
        # Check that unit 2, period 2 was filled
        filled_row = result[(result["unit"] == 2) & (result["period"] == 2)]
        assert len(filled_row) == 1
        assert filled_row["y"].values[0] == 20.0  # Forward filled from period 1

    def test_error_invalid_method(self):
        """Test error with invalid method."""
        df = pd.DataFrame({"unit": [1], "period": [1], "y": [10]})
        with pytest.raises(ValueError, match="method must be"):
            balance_panel(df, "unit", "period", method="invalid")


class TestValidateDidData:
    """Tests for validate_did_data function."""

    def test_valid_data(self):
        """Test validation of valid data."""
        df = pd.DataFrame(
            {"y": [1.0, 2.0, 3.0, 4.0], "treated": [0, 0, 1, 1], "post": [0, 1, 0, 1]}
        )
        result = validate_did_data(df, "y", "treated", "post", raise_on_error=False)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_column(self):
        """Test validation catches missing columns."""
        df = pd.DataFrame({"y": [1, 2], "treated": [0, 1]})
        result = validate_did_data(df, "y", "treated", "post", raise_on_error=False)
        assert result["valid"] is False
        assert any("not found" in e for e in result["errors"])

    def test_non_numeric_outcome(self):
        """Test validation catches non-numeric outcome."""
        df = pd.DataFrame(
            {"y": ["a", "b", "c", "d"], "treated": [0, 0, 1, 1], "post": [0, 1, 0, 1]}
        )
        result = validate_did_data(df, "y", "treated", "post", raise_on_error=False)
        assert result["valid"] is False
        assert any("numeric" in e for e in result["errors"])

    def test_non_binary_treatment(self):
        """Test validation catches non-binary treatment."""
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "treated": [0, 1, 2], "post": [0, 1, 0]})
        result = validate_did_data(df, "y", "treated", "post", raise_on_error=False)
        assert result["valid"] is False
        assert any("binary" in e for e in result["errors"])

    def test_missing_values(self):
        """Test validation catches missing values."""
        df = pd.DataFrame(
            {"y": [1.0, np.nan, 3.0, 4.0], "treated": [0, 0, 1, 1], "post": [0, 1, 0, 1]}
        )
        result = validate_did_data(df, "y", "treated", "post", raise_on_error=False)
        assert result["valid"] is False
        assert any("missing" in e for e in result["errors"])

    def test_raises_on_error(self):
        """Test that validation raises when raise_on_error=True."""
        df = pd.DataFrame({"y": [1], "treated": [0]})  # Missing post column
        with pytest.raises(ValueError):
            validate_did_data(df, "y", "treated", "post", raise_on_error=True)

    def test_panel_validation(self):
        """Test panel-specific validation."""
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "treated": [0, 0, 1, 1],
                "post": [0, 1, 0, 1],
                "unit": [1, 1, 2, 2],
            }
        )
        result = validate_did_data(df, "y", "treated", "post", unit="unit", raise_on_error=False)
        assert result["valid"] is True
        assert result["summary"]["n_units"] == 2


class TestSummarizeDidData:
    """Tests for summarize_did_data function."""

    def test_basic_summary(self):
        """Test basic summary statistics."""
        df = pd.DataFrame(
            {
                "y": [10, 11, 12, 13, 20, 21, 22, 23],
                "treated": [0, 0, 1, 1, 0, 0, 1, 1],
                "post": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        summary = summarize_did_data(df, "y", "treated", "post")
        assert len(summary) == 5  # 4 groups + DiD estimate

    def test_did_estimate_included(self):
        """Test that DiD estimate is calculated."""
        df = pd.DataFrame(
            {
                "y": [10, 20, 15, 30],  # Perfect DiD = 30-15 - (20-10) = 5
                "treated": [0, 0, 1, 1],
                "post": [0, 1, 0, 1],
            }
        )
        summary = summarize_did_data(df, "y", "treated", "post")
        assert "DiD Estimate" in summary.index
        assert summary.loc["DiD Estimate", "mean"] == 5.0


class TestGenerateDidData:
    """Tests for generate_did_data function."""

    def test_basic_generation(self):
        """Test basic data generation."""
        data = generate_did_data(n_units=50, n_periods=4, seed=42)
        assert len(data) == 200  # 50 units x 4 periods
        assert set(data.columns) == {"unit", "period", "treated", "post", "outcome", "true_effect"}

    def test_treatment_fraction(self):
        """Test that treatment fraction is respected."""
        data = generate_did_data(n_units=100, treatment_fraction=0.3, seed=42)
        n_treated_units = data.groupby("unit")["treated"].first().sum()
        assert n_treated_units == 30

    def test_treatment_effect_recovery(self):
        """Test that treatment effect can be roughly recovered."""
        from diff_diff import DifferenceInDifferences

        true_effect = 5.0
        data = generate_did_data(
            n_units=200, n_periods=4, treatment_effect=true_effect, noise_sd=0.5, seed=42
        )

        did = DifferenceInDifferences()
        results = did.fit(data, outcome="outcome", treatment="treated", post="post")

        # Effect should be within 1 unit of true effect
        assert abs(results.att - true_effect) < 1.0

    def test_reproducibility(self):
        """Test that seed produces reproducible data."""
        data1 = generate_did_data(seed=123)
        data2 = generate_did_data(seed=123)
        pd.testing.assert_frame_equal(data1, data2)

    def test_true_effect_column(self):
        """Test that true_effect column is correct."""
        data = generate_did_data(n_units=10, n_periods=4, treatment_effect=3.0, seed=42)

        # True effect should only be non-zero for treated units in post period
        treated_post = data[(data["treated"] == 1) & (data["post"] == 1)]
        not_treated_post = data[~((data["treated"] == 1) & (data["post"] == 1))]

        assert (treated_post["true_effect"] == 3.0).all()
        assert (not_treated_post["true_effect"] == 0.0).all()


class TestCreateEventTime:
    """Tests for create_event_time function."""

    def test_basic_event_time(self):
        """Test basic event time calculation."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2, 2, 2],
                "year": [2018, 2019, 2020, 2018, 2019, 2020],
                "treatment_year": [2019, 2019, 2019, 2020, 2020, 2020],
            }
        )
        result = create_event_time(df, "year", "treatment_year")
        assert result["event_time"].tolist() == [-1, 0, 1, -2, -1, 0]

    def test_never_treated(self):
        """Test handling of never-treated units."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "year": [2019, 2020, 2019, 2020],
                "treatment_year": [2020, 2020, np.nan, np.nan],
            }
        )
        result = create_event_time(df, "year", "treatment_year")
        assert result.loc[0, "event_time"] == -1
        assert result.loc[1, "event_time"] == 0
        assert pd.isna(result.loc[2, "event_time"])
        assert pd.isna(result.loc[3, "event_time"])

    def test_custom_column_name(self):
        """Test custom output column name."""
        df = pd.DataFrame({"year": [2019, 2020], "treat_time": [2020, 2020]})
        result = create_event_time(df, "year", "treat_time", new_column="rel_time")
        assert "rel_time" in result.columns


class TestAggregateToCohorts:
    """Tests for aggregate_to_cohorts function."""

    def test_basic_aggregation(self):
        """Test basic cohort aggregation."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2, 3, 3, 4, 4],
                "period": [0, 1, 0, 1, 0, 1, 0, 1],
                "treated": [1, 1, 1, 1, 0, 0, 0, 0],
                "y": [10, 15, 12, 17, 8, 10, 9, 11],
            }
        )
        result = aggregate_to_cohorts(df, "unit", "period", "treated", "y")
        assert len(result) == 4  # 2 treatment groups x 2 periods
        assert "mean_y" in result.columns
        assert "n_units" in result.columns

    def test_with_covariates(self):
        """Test aggregation with covariates."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "period": [0, 1, 0, 1],
                "treated": [1, 1, 0, 0],
                "y": [10, 15, 8, 10],
                "x": [1.0, 1.5, 0.5, 0.8],
            }
        )
        result = aggregate_to_cohorts(df, "unit", "period", "treated", "y", covariates=["x"])
        assert "x" in result.columns


class TestRankControlUnits:
    """Tests for rank_control_units function."""

    def test_basic_ranking(self):
        """Test basic control unit ranking."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=20, n_periods=6, seed=42)
        result = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
        )
        assert "quality_score" in result.columns
        assert "outcome_trend_score" in result.columns
        assert "synthetic_weight" in result.columns
        assert len(result) > 0
        # Check sorted descending
        assert result["quality_score"].is_monotonic_decreasing

    def test_with_covariates(self):
        """Test ranking with covariate matching."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=20, n_periods=6, seed=42)
        # Add covariate
        np.random.seed(42)
        data["x1"] = np.random.randn(len(data))

        result = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
            covariates=["x1"],
        )
        assert not result["covariate_score"].isna().all()

    def test_explicit_treated_units(self):
        """Test with explicitly specified treated units."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=20, n_periods=6, seed=42)

        result = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treated_units=[0, 1, 2],
        )
        # Should not include treated units in ranking
        assert 0 not in result["unit"].values
        assert 1 not in result["unit"].values
        assert 2 not in result["unit"].values

    def test_exclude_units(self):
        """Test unit exclusion."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=20, n_periods=6, seed=42)

        result = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
            exclude_units=[15, 16, 17],
        )
        assert 15 not in result["unit"].values
        assert 16 not in result["unit"].values
        assert 17 not in result["unit"].values

    def test_require_units(self):
        """Test required units are always included."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=30, n_periods=6, seed=42)

        # Get control units (not treated)
        control_units = data[data["treated"] == 0]["unit"].unique()
        require = [control_units[-1], control_units[-2]]  # Pick last two controls

        result = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
            require_units=require,
            n_top=5,
        )
        # Required units should be present
        for u in require:
            assert u in result["unit"].values
        # is_required flag should be set
        assert result[result["unit"].isin(require)]["is_required"].all()

    def test_n_top_limit(self):
        """Test limiting to top N controls."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=30, n_periods=6, seed=42)

        result = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
            n_top=10,
        )
        assert len(result) == 10

    def test_suggest_treatment_candidates(self):
        """Test treatment candidate suggestion mode."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=20, n_periods=6, seed=42)
        # Remove treatment column to simulate unknown treatment
        data = data.drop(columns=["treated"])

        result = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            suggest_treatment_candidates=True,
            n_treatment_candidates=5,
        )
        assert "treatment_candidate_score" in result.columns
        assert len(result) == 5

    def test_original_unchanged(self):
        """Test that original DataFrame is not modified."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=20, n_periods=6, seed=42)
        original_cols = data.columns.tolist()

        rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
        )
        assert data.columns.tolist() == original_cols

    def test_error_missing_column(self):
        """Test error when column doesn't exist."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=10, n_periods=4, seed=42)

        with pytest.raises(ValueError, match="not found"):
            rank_control_units(
                data, unit_column="missing_col", time_column="period", outcome_column="outcome"
            )

    def test_error_both_treatment_specs(self):
        """Test error when both treatment specifications provided."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=10, n_periods=4, seed=42)

        with pytest.raises(ValueError, match="Specify either"):
            rank_control_units(
                data,
                unit_column="unit",
                time_column="period",
                outcome_column="outcome",
                treatment_column="treated",
                treated_units=[0, 1],
            )

    def test_error_require_and_exclude_same_unit(self):
        """Test error when same unit is required and excluded."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=10, n_periods=4, seed=42)

        with pytest.raises(ValueError, match="both required and excluded"):
            rank_control_units(
                data,
                unit_column="unit",
                time_column="period",
                outcome_column="outcome",
                treatment_column="treated",
                require_units=[5],
                exclude_units=[5],
            )

    def test_synthetic_weight_sum(self):
        """Test that synthetic weights sum to approximately 1."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=20, n_periods=6, seed=42)

        result = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
        )

        # Synthetic weights should sum to approximately 1
        assert abs(result["synthetic_weight"].sum() - 1.0) < 0.01

    def test_pre_periods_explicit(self):
        """Test with explicitly specified pre-periods."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=20, n_periods=6, seed=42)

        result = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
            pre_periods=[0, 1],  # Only use first two periods
        )
        assert len(result) > 0

    def test_weight_parameters(self):
        """Test different outcome/covariate weight settings."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=20, n_periods=6, seed=42)
        np.random.seed(42)
        data["x1"] = np.random.randn(len(data))

        # All weight on outcome
        result1 = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
            covariates=["x1"],
            outcome_weight=1.0,
            covariate_weight=0.0,
        )

        # All weight on covariates
        result2 = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
            covariates=["x1"],
            outcome_weight=0.0,
            covariate_weight=1.0,
        )

        # Rankings should differ
        # (just check both work, exact comparison is data-dependent)
        assert len(result1) > 0
        assert len(result2) > 0

    def test_unbalanced_panel(self):
        """Test handling of unbalanced panels with missing data."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=20, n_periods=6, seed=42)

        # Remove some observations to create unbalanced panel
        # Remove all pre-period data for one control unit
        control_units = data[data["treated"] == 0]["unit"].unique()
        unit_to_partially_remove = control_units[0]
        mask = ~((data["unit"] == unit_to_partially_remove) & (data["period"] < 3))
        unbalanced_data = data[mask].copy()

        result = rank_control_units(
            unbalanced_data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
        )

        # Should still work and exclude the unit with no pre-treatment data
        assert len(result) > 0
        # The unit with missing pre-treatment data should not be in results
        assert unit_to_partially_remove not in result["unit"].values

    def test_single_control_unit(self):
        """Test edge case with only one control unit."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=10, n_periods=6, seed=42)

        # Keep only one control unit
        treated_units = data[data["treated"] == 1]["unit"].unique()
        control_units = data[data["treated"] == 0]["unit"].unique()
        single_control = control_units[0]

        filtered_data = data[
            (data["unit"].isin(treated_units)) | (data["unit"] == single_control)
        ].copy()

        result = rank_control_units(
            filtered_data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
        )

        assert len(result) == 1
        assert result["unit"].iloc[0] == single_control
        # Single control should get score of 1.0 (best possible)
        assert result["quality_score"].iloc[0] == 1.0

    def test_extreme_Y_scale_synthetic_weight_column(self):
        """Finding #22 (post-audit cleanup): `synthetic_weight` column must
        remain a valid non-degenerate simplex vector even at extreme Y
        scale (Y ~ 1e9). The previous `compute_synthetic_weights` wrapper
        had two bugs here: Rust PGD collapsed to a single vertex, Python
        PGD stalled at uniform. The inlined Frank-Wolfe solver in
        ``rank_control_units`` handles both cases correctly."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=12, n_periods=8, seed=42)
        # Shift outcomes to extreme scale — the exact condition the deleted
        # wrapper mishandled.
        data = data.copy()
        data["outcome"] = data["outcome"] + 1e9

        result = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
        )

        weights = result["synthetic_weight"].to_numpy()
        # Valid simplex: non-negative, sums to 1.
        assert np.all(weights >= 0), "synthetic_weight must be non-negative"
        assert (
            abs(weights.sum() - 1.0) < 1e-10
        ), f"synthetic_weight should sum to 1.0, got {weights.sum()}"
        # Non-degenerate: at least 2 controls receive non-trivial weight.
        # This guards the Rust-PGD collapse-to-one-vertex bug that
        # previously fired at Y ~ 1e9 under the deleted wrapper.
        assert int(np.sum(weights > 1e-6)) >= 2, (
            f"synthetic_weight collapsed to a single vertex at extreme Y "
            f"scale; n_nonzero={int(np.sum(weights > 1e-6))}. weights={weights}"
        )

    def test_synthetic_weight_column_with_lambda_reg(self):
        """`rank_control_units(lambda_reg > 0)` regression. Non-zero
        regularization must produce a valid simplex vector and should pull
        weights toward a more uniform distribution vs `lambda_reg=0`."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=10, n_periods=8, seed=42)

        res_unreg = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
            lambda_reg=0.0,
        )
        res_reg = rank_control_units(
            data,
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
            lambda_reg=1.0,
        )

        w_unreg = res_unreg["synthetic_weight"].to_numpy()
        w_reg = res_reg["synthetic_weight"].to_numpy()

        for w, label in [(w_unreg, "unregularized"), (w_reg, "regularized")]:
            assert np.all(w >= 0), f"{label} weights must be non-negative"
            assert abs(w.sum() - 1.0) < 1e-10, f"{label} weights should sum to 1.0, got {w.sum()}"

        # Regularization should increase entropy (pull toward uniform) or at
        # least not collapse the simplex — a valid regularized solution must
        # put non-trivial mass on at least as many donors as the unregularized
        # one for a well-conditioned input.
        assert int(np.sum(w_reg > 1e-6)) >= int(np.sum(w_unreg > 1e-6)) - 1, (
            f"lambda_reg=1.0 should not collapse support vs lambda_reg=0; "
            f"n_nonzero_reg={int(np.sum(w_reg > 1e-6))}, "
            f"n_nonzero_unreg={int(np.sum(w_unreg > 1e-6))}"
        )

    def test_synthetic_weight_column_backend_parity(self):
        """Full-pipeline Rust/Python parity for `rank_control_units`. Forces
        the Python FW path via `HAS_RUST_BACKEND=False` and verifies the
        resulting `synthetic_weight` column agrees with the default (Rust
        when available) path to a loose tolerance. Backend divergence on
        this caller path was the trigger for deleting the old wrapper."""
        from unittest.mock import patch

        from diff_diff import utils as utils_mod
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=10, n_periods=8, seed=123)

        kwargs = dict(
            unit_column="unit",
            time_column="period",
            outcome_column="outcome",
            treatment_column="treated",
        )

        res_default = rank_control_units(data, **kwargs)
        with patch.object(utils_mod, "HAS_RUST_BACKEND", False):
            res_python = rank_control_units(data, **kwargs)

        # Align by unit id (ranking order is not a backend-parity property;
        # the simplex values on common donors are).
        merged = res_default[["unit", "synthetic_weight"]].merge(
            res_python[["unit", "synthetic_weight"]],
            on="unit",
            suffixes=("_default", "_python"),
        )
        # FW stopping threshold is scale-dependent, so use a loose tolerance.
        np.testing.assert_allclose(
            merged["synthetic_weight_default"].to_numpy(),
            merged["synthetic_weight_python"].to_numpy(),
            atol=1e-4,
            rtol=1e-4,
        )


class TestGenerateStaggeredData:
    """Tests for generate_staggered_data function."""

    def test_basic_generation(self):
        """Test basic staggered data generation."""
        from diff_diff.prep import generate_staggered_data

        data = generate_staggered_data(n_units=50, n_periods=8, seed=42)
        assert len(data) == 400  # 50 units x 8 periods
        assert set(data.columns) == {
            "unit",
            "period",
            "outcome",
            "first_treat",
            "treated",
            "treat",
            "true_effect",
        }

    def test_never_treated_fraction(self):
        """Test that never_treated_frac is respected."""
        from diff_diff.prep import generate_staggered_data

        data = generate_staggered_data(n_units=100, never_treated_frac=0.3, seed=42)
        n_never = (data.groupby("unit")["first_treat"].first() == 0).sum()
        assert n_never == 30

    def test_cohort_periods(self):
        """Test custom cohort periods."""
        from diff_diff.prep import generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=10, cohort_periods=[4, 6], seed=42)
        cohorts = data.groupby("unit")["first_treat"].first().unique()
        assert set(cohorts) == {0, 4, 6}

    def test_treatment_effect_direction(self):
        """Test that treatment effect is positive."""
        from diff_diff.prep import generate_staggered_data

        data = generate_staggered_data(n_units=100, treatment_effect=3.0, noise_sd=0.1, seed=42)
        # Treated observations should have positive true_effect
        treated_effects = data[data["treated"] == 1]["true_effect"]
        assert (treated_effects > 0).all()

    def test_dynamic_effects(self):
        """Test dynamic treatment effects."""
        from diff_diff.prep import generate_staggered_data

        data = generate_staggered_data(
            n_units=50,
            n_periods=10,
            treatment_effect=2.0,
            dynamic_effects=True,
            effect_growth=0.1,
            seed=42,
        )
        # Effects should grow over time since treatment
        # Check a treated unit
        treated_units = data[data["treat"] == 1]["unit"].unique()
        unit_data = data[data["unit"] == treated_units[0]].sort_values("period")
        first_treat = unit_data["first_treat"].iloc[0]
        effects = unit_data[unit_data["period"] >= first_treat]["true_effect"].values
        # Effects should be increasing (with dynamic effects)
        assert all(effects[i] <= effects[i + 1] for i in range(len(effects) - 1))

    def test_reproducibility(self):
        """Test seed produces reproducible data."""
        from diff_diff.prep import generate_staggered_data

        data1 = generate_staggered_data(seed=123)
        data2 = generate_staggered_data(seed=123)
        pd.testing.assert_frame_equal(data1, data2)

    def test_invalid_cohort_period(self):
        """Test error on invalid cohort period."""
        from diff_diff.prep import generate_staggered_data

        with pytest.raises(ValueError, match="must be between"):
            generate_staggered_data(n_periods=10, cohort_periods=[0, 5])  # 0 invalid

        with pytest.raises(ValueError, match="must be between"):
            generate_staggered_data(n_periods=10, cohort_periods=[5, 10])  # 10 invalid


class TestGenerateFactorData:
    """Tests for generate_factor_data function."""

    def test_basic_generation(self):
        """Test basic factor data generation."""
        from diff_diff.prep import generate_factor_data

        data = generate_factor_data(n_units=30, n_pre=8, n_post=4, n_treated=5, seed=42)
        assert len(data) == 360  # 30 units x 12 periods
        assert set(data.columns) == {"unit", "period", "outcome", "treated", "treat", "true_effect"}

    def test_treated_units_count(self):
        """Test that n_treated is respected."""
        from diff_diff.prep import generate_factor_data

        data = generate_factor_data(n_units=50, n_treated=10, seed=42)
        n_treated = data.groupby("unit")["treat"].first().sum()
        assert n_treated == 10

    def test_treatment_in_post_only(self):
        """Test that treatment indicator is 1 only in post-treatment."""
        from diff_diff.prep import generate_factor_data

        data = generate_factor_data(n_pre=10, n_post=5, n_treated=10, seed=42)
        # Pre-treatment observations should have treated=0
        pre_data = data[data["period"] < 10]
        assert (pre_data["treated"] == 0).all()

    def test_treatment_effect_recovery(self):
        """Test that treatment effect can be roughly recovered."""
        from diff_diff.prep import generate_factor_data

        true_effect = 3.0
        data = generate_factor_data(
            n_units=100,
            n_pre=10,
            n_post=5,
            n_treated=30,
            treatment_effect=true_effect,
            noise_sd=0.1,
            factor_strength=0.1,
            seed=42,
        )
        # Simple DiD on treated vs control, post vs pre
        treated_post = data[(data["treat"] == 1) & (data["period"] >= 10)]["outcome"].mean()
        treated_pre = data[(data["treat"] == 1) & (data["period"] < 10)]["outcome"].mean()
        control_post = data[(data["treat"] == 0) & (data["period"] >= 10)]["outcome"].mean()
        control_pre = data[(data["treat"] == 0) & (data["period"] < 10)]["outcome"].mean()
        did_estimate = (treated_post - treated_pre) - (control_post - control_pre)
        # With low noise and factor strength, should be reasonably close
        assert abs(did_estimate - true_effect) < 2.0

    def test_reproducibility(self):
        """Test seed produces reproducible data."""
        from diff_diff.prep import generate_factor_data

        data1 = generate_factor_data(seed=123)
        data2 = generate_factor_data(seed=123)
        pd.testing.assert_frame_equal(data1, data2)

    def test_invalid_n_treated(self):
        """Test error on invalid n_treated."""
        from diff_diff.prep import generate_factor_data

        with pytest.raises(ValueError, match="cannot exceed"):
            generate_factor_data(n_units=10, n_treated=20)

        with pytest.raises(ValueError, match="at least 1"):
            generate_factor_data(n_units=10, n_treated=0)


class TestGenerateSyntheticControlData:
    """Tests for generate_synthetic_control_data function."""

    def test_basic_generation(self):
        """Shape and exact column set (incl. predictor columns)."""
        from diff_diff.prep import generate_synthetic_control_data

        data = generate_synthetic_control_data(
            n_donors=10, n_pre=12, n_post=4, n_predictors=3, seed=42
        )
        assert len(data) == (10 + 1) * (12 + 4)  # (donors + treated) x periods
        assert set(data.columns) == {
            "unit",
            "period",
            "outcome",
            "treatment",
            "treat",
            "true_effect",
            "x1",
            "x2",
            "x3",
        }

    def test_single_treated_unit(self):
        """Exactly one ever-treated unit (unit 0)."""
        from diff_diff.prep import generate_synthetic_control_data

        data = generate_synthetic_control_data(n_donors=15, seed=42)
        ever_treated = data.groupby("unit")["treat"].first()
        assert ever_treated.sum() == 1
        assert ever_treated.loc[0] == 1  # treated unit is id 0

    def test_treatment_absorbing_treated_post_only(self):
        """treatment==1 only for the treated unit in post periods."""
        from diff_diff.prep import generate_synthetic_control_data

        data = generate_synthetic_control_data(n_donors=8, n_pre=10, n_post=4, seed=1)
        treated_rows = data[data["treatment"] == 1]
        assert (treated_rows["unit"] == 0).all()
        assert (treated_rows["period"] >= 10).all()
        # Donors are never treated; treated unit's pre periods are untreated.
        assert (data[data["unit"] != 0]["treatment"] == 0).all()
        assert (data[(data["unit"] == 0) & (data["period"] < 10)]["treatment"] == 0).all()

    def test_ramp_effect_increases(self):
        """Ramp effect strictly increases across post periods; pre/donor are zero."""
        from diff_diff.prep import generate_synthetic_control_data

        data = generate_synthetic_control_data(
            n_donors=8,
            n_pre=10,
            n_post=5,
            effect_type="ramp",
            treatment_effect=5.0,
            effect_growth=1.0,
            seed=1,
        )
        post_eff = (
            data[(data["unit"] == 0) & (data["period"] >= 10)]
            .sort_values("period")["true_effect"]
            .to_numpy()
        )
        assert np.allclose(post_eff, [5.0, 6.0, 7.0, 8.0, 9.0])
        assert np.all(np.diff(post_eff) > 0)
        # true_effect is zero everywhere else.
        assert (data[data["treatment"] == 0]["true_effect"] == 0).all()

    def test_constant_effect_flat(self):
        """Constant effect is flat across post periods."""
        from diff_diff.prep import generate_synthetic_control_data

        data = generate_synthetic_control_data(
            n_donors=8,
            n_pre=10,
            n_post=5,
            effect_type="constant",
            treatment_effect=4.0,
            seed=1,
        )
        post_eff = data[(data["unit"] == 0) & (data["period"] >= 10)]["true_effect"]
        assert (post_eff == 4.0).all()

    def test_reproducibility(self):
        """Seed produces reproducible data."""
        from diff_diff.prep import generate_synthetic_control_data

        data1 = generate_synthetic_control_data(seed=123)
        data2 = generate_synthetic_control_data(seed=123)
        pd.testing.assert_frame_equal(data1, data2)

    def test_treated_in_hull_recovers_effect(self):
        """The treated unit's latent trajectory is in the donor hull, so on noisy
        observed data a synthetic control achieves a small (approximate) pre-period
        RMSPE and recovers the average effect (loose tolerance). The exact-hull
        property is pinned separately by ``test_noiseless_outcome_path_in_hull``."""
        from diff_diff import SyntheticControl
        from diff_diff.prep import generate_synthetic_control_data

        data = generate_synthetic_control_data(
            n_donors=15,
            n_pre=20,
            n_post=4,
            effect_type="constant",
            treatment_effect=5.0,
            noise_sd=0.4,
            seed=0,
        )
        res = SyntheticControl(
            n_starts=1,
            inner_min_decrease=1e-3,
            optimizer_options={"maxiter": 50},
            seed=0,
        ).fit(
            data,
            outcome="outcome",
            treatment="treatment",
            unit="unit",
            time="period",
            predictors=["x1", "x2", "x3"],
        )
        # Good pre-fit (treated reproducible from donors) and recovered effect.
        assert res.pre_rmspe < 2.0
        assert abs(res.att - 5.0) < 2.0

    def test_noiseless_outcome_path_in_hull(self):
        """At zero noise the treated unit's outcome path IS an exact convex combination
        of the donor paths (it lies in the donor convex hull), so a synthetic control
        matching the pre-period outcome path reproduces it to ~zero pre-RMSPE — orders
        of magnitude below the noisy case. Pins the *noiseless* in-hull claim the
        docstring/tutorial make (the observed fit on noisy data is only approximate)."""
        from diff_diff import SyntheticControl
        from diff_diff.prep import generate_synthetic_control_data

        data = generate_synthetic_control_data(
            n_donors=12,
            n_pre=20,
            n_post=4,
            effect_type="constant",
            treatment_effect=5.0,
            noise_sd=0.0,
            predictor_noise_sd=0.0,
            seed=0,
        )
        # Match on the full pre-period outcome path (no predictors=): the path itself is
        # the exact convex combination, so the simplex fit is (near-)exact.
        res = SyntheticControl(
            n_starts=1,
            inner_min_decrease=1e-4,
            optimizer_options={"maxiter": 50},
            seed=0,
        ).fit(data, outcome="outcome", treatment="treatment", unit="unit", time="period")
        assert res.pre_rmspe < 0.05, f"noiseless pre_rmspe={res.pre_rmspe:.4f} not ~0"
        assert abs(res.att - 5.0) < 0.5

    def test_invalid_arguments(self):
        """Each validation guard raises ValueError with an informative message."""
        from diff_diff.prep import generate_synthetic_control_data

        with pytest.raises(ValueError, match="n_donors"):
            generate_synthetic_control_data(n_donors=1)
        with pytest.raises(ValueError, match="n_pre"):
            generate_synthetic_control_data(n_pre=0)
        with pytest.raises(ValueError, match="n_post"):
            generate_synthetic_control_data(n_post=0)
        with pytest.raises(ValueError, match="n_factors"):
            generate_synthetic_control_data(n_factors=0)
        with pytest.raises(ValueError, match="n_predictors"):
            generate_synthetic_control_data(n_predictors=-1)
        with pytest.raises(ValueError, match="n_convex_donors"):
            generate_synthetic_control_data(n_donors=5, n_convex_donors=6)
        with pytest.raises(ValueError, match="effect_type"):
            generate_synthetic_control_data(effect_type="quadratic")
        with pytest.raises(ValueError, match="factor_persistence"):
            generate_synthetic_control_data(factor_persistence=1.5)


class TestGenerateDddData:
    """Tests for generate_ddd_data function."""

    def test_basic_generation(self):
        """Test basic DDD data generation."""
        from diff_diff.prep import generate_ddd_data

        data = generate_ddd_data(n_per_cell=50, seed=42)
        assert len(data) == 400  # 50 x 8 cells
        expected_cols = {"outcome", "group", "partition", "time", "unit_id", "true_effect"}
        assert expected_cols.issubset(set(data.columns))

    def test_cell_structure(self):
        """Test that all 8 cells have correct counts."""
        from diff_diff.prep import generate_ddd_data

        data = generate_ddd_data(n_per_cell=100, seed=42)
        cell_counts = data.groupby(["group", "partition", "time"]).size()
        assert len(cell_counts) == 8
        assert (cell_counts == 100).all()

    def test_treatment_effect_location(self):
        """Test that true_effect is only non-zero for G=1, P=1, T=1."""
        from diff_diff.prep import generate_ddd_data

        data = generate_ddd_data(n_per_cell=50, treatment_effect=5.0, seed=42)
        # Only G=1, P=1, T=1 should have non-zero true_effect
        treated = data[(data["group"] == 1) & (data["partition"] == 1) & (data["time"] == 1)]
        not_treated = data[~((data["group"] == 1) & (data["partition"] == 1) & (data["time"] == 1))]

        assert (treated["true_effect"] == 5.0).all()
        assert (not_treated["true_effect"] == 0.0).all()

    def test_with_covariates(self):
        """Test data generation with covariates."""
        from diff_diff.prep import generate_ddd_data

        data = generate_ddd_data(n_per_cell=50, add_covariates=True, seed=42)
        assert "age" in data.columns
        assert "education" in data.columns

    def test_without_covariates(self):
        """Test data generation without covariates."""
        from diff_diff.prep import generate_ddd_data

        data = generate_ddd_data(n_per_cell=50, add_covariates=False, seed=42)
        assert "age" not in data.columns
        assert "education" not in data.columns

    def test_treatment_effect_recovery(self):
        """Test that treatment effect can be recovered with DDD."""
        from diff_diff.prep import generate_ddd_data

        true_effect = 3.0
        data = generate_ddd_data(
            n_per_cell=200, treatment_effect=true_effect, noise_sd=0.5, seed=42
        )

        # Manual DDD calculation
        y_111 = data[(data["group"] == 1) & (data["partition"] == 1) & (data["time"] == 1)][
            "outcome"
        ].mean()
        y_110 = data[(data["group"] == 1) & (data["partition"] == 1) & (data["time"] == 0)][
            "outcome"
        ].mean()
        y_101 = data[(data["group"] == 1) & (data["partition"] == 0) & (data["time"] == 1)][
            "outcome"
        ].mean()
        y_100 = data[(data["group"] == 1) & (data["partition"] == 0) & (data["time"] == 0)][
            "outcome"
        ].mean()
        y_011 = data[(data["group"] == 0) & (data["partition"] == 1) & (data["time"] == 1)][
            "outcome"
        ].mean()
        y_010 = data[(data["group"] == 0) & (data["partition"] == 1) & (data["time"] == 0)][
            "outcome"
        ].mean()
        y_001 = data[(data["group"] == 0) & (data["partition"] == 0) & (data["time"] == 1)][
            "outcome"
        ].mean()
        y_000 = data[(data["group"] == 0) & (data["partition"] == 0) & (data["time"] == 0)][
            "outcome"
        ].mean()

        manual_ddd = (y_111 - y_110) - (y_101 - y_100) - (y_011 - y_010) + (y_001 - y_000)
        assert abs(manual_ddd - true_effect) < 0.5

    def test_reproducibility(self):
        """Test seed produces reproducible data."""
        from diff_diff.prep import generate_ddd_data

        data1 = generate_ddd_data(seed=123)
        data2 = generate_ddd_data(seed=123)
        pd.testing.assert_frame_equal(data1, data2)


class TestGenerateDddPanelData:
    """Tests for generate_ddd_panel_data function (panel-structured DDD DGP)."""

    def test_shape(self):
        """len(data) == n_units * n_periods (balanced panel)."""
        from diff_diff.prep import generate_ddd_panel_data

        data = generate_ddd_panel_data(n_units=200, n_periods=8, seed=42)
        assert len(data) == 200 * 8
        expected_cols = {
            "unit",
            "period",
            "outcome",
            "group",
            "partition",
            "post",
            "treated",
            "true_effect",
        }
        assert expected_cols.issubset(set(data.columns))

    def test_balanced_cells(self):
        """Each (group, partition) cell has roughly n_units * group_frac * partition_frac units."""
        from diff_diff.prep import generate_ddd_panel_data

        data = generate_ddd_panel_data(
            n_units=400, n_periods=8, group_frac=0.5, partition_frac=0.5, seed=42
        )
        # Cell counts measured at unit level (not row level)
        unit_cells = data.groupby("unit")[["group", "partition"]].first()
        cell_counts = unit_cells.groupby(["group", "partition"]).size()
        assert len(cell_counts) == 4
        expected_per_cell = 400 * 0.5 * 0.5  # = 100
        for count in cell_counts.values:
            assert abs(count - expected_per_cell) <= 2

    def test_treatment_effect_location(self):
        """true_effect != 0 only when group==1 AND partition==1 AND post==1."""
        from diff_diff.prep import generate_ddd_panel_data

        data = generate_ddd_panel_data(
            n_units=200, n_periods=8, treatment_period=4, treatment_effect=3.0, seed=42
        )
        treated_mask = (data["group"] == 1) & (data["partition"] == 1) & (data["post"] == 1)
        assert (data.loc[treated_mask, "true_effect"] == 3.0).all()
        assert (data.loc[~treated_mask, "true_effect"] == 0.0).all()
        assert (data["treated"] == treated_mask.astype(int)).all()

    def test_panel_structure(self):
        """Each unit observed in exactly n_periods periods."""
        from diff_diff.prep import generate_ddd_panel_data

        data = generate_ddd_panel_data(n_units=100, n_periods=6, seed=42)
        assert data.groupby("unit")["period"].count().eq(6).all()

    def test_post_derived_from_period(self):
        """post is exactly 1[period >= treatment_period]."""
        from diff_diff.prep import generate_ddd_panel_data

        data = generate_ddd_panel_data(n_units=100, n_periods=8, treatment_period=3, seed=42)
        expected_post = (data["period"] >= 3).astype(int)
        assert (data["post"] == expected_post).all()

    def test_group_partition_time_invariant(self):
        """group and partition are unit-level (constant within unit) — DDD-CPT requirement."""
        from diff_diff.prep import generate_ddd_panel_data

        data = generate_ddd_panel_data(n_units=200, n_periods=8, seed=42)
        assert data.groupby("unit")["group"].nunique().eq(1).all()
        assert data.groupby("unit")["partition"].nunique().eq(1).all()

    def test_treatment_period_validation(self):
        """treatment_period must satisfy 1 <= treatment_period < n_periods."""
        from diff_diff.prep import generate_ddd_panel_data

        with pytest.raises(ValueError, match="treatment_period"):
            generate_ddd_panel_data(n_units=100, n_periods=8, treatment_period=0, seed=42)
        with pytest.raises(ValueError, match="treatment_period"):
            generate_ddd_panel_data(n_units=100, n_periods=8, treatment_period=8, seed=42)

    def test_group_frac_validation(self):
        """group_frac and partition_frac must be strictly in (0, 1)."""
        from diff_diff.prep import generate_ddd_panel_data

        with pytest.raises(ValueError, match="group_frac"):
            generate_ddd_panel_data(n_units=100, group_frac=0.0, seed=42)
        with pytest.raises(ValueError, match="group_frac"):
            generate_ddd_panel_data(n_units=100, group_frac=1.0, seed=42)
        with pytest.raises(ValueError, match="partition_frac"):
            generate_ddd_panel_data(n_units=100, partition_frac=0.0, seed=42)

    def test_n_units_validation(self):
        """n_units must be >= 4."""
        from diff_diff.prep import generate_ddd_panel_data

        with pytest.raises(ValueError, match="n_units"):
            generate_ddd_panel_data(n_units=3, seed=42)

    def test_infeasible_cell_counts_raise(self):
        """Configs that round to empty (group, partition) cells fail fast."""
        from diff_diff.prep import generate_ddd_panel_data

        # n_units=4, group_frac=0.25 → n_group_1=1; partition_frac=0.25 inside
        # the 1-unit group=1 stratum rounds to 0, leaving the (1, 1) cell empty.
        with pytest.raises(ValueError, match=r"every \(group, partition\) cell"):
            generate_ddd_panel_data(
                n_units=4,
                group_frac=0.25,
                partition_frac=0.25,
                seed=42,
            )
        # n_units=10, group_frac=0.1 → n_group_1=1 again; same failure mode for
        # the (1, 1) cell.
        with pytest.raises(ValueError, match=r"every \(group, partition\) cell"):
            generate_ddd_panel_data(
                n_units=10,
                group_frac=0.1,
                partition_frac=0.5,
                seed=42,
            )

    def test_smallest_feasible_config_populates_all_cells(self):
        """n_units=4 with fracs=0.5 yields one unit per (group, partition) cell."""
        from diff_diff import TripleDifference
        from diff_diff.prep import generate_ddd_panel_data

        data = generate_ddd_panel_data(
            n_units=4,
            n_periods=4,
            treatment_period=2,
            group_frac=0.5,
            partition_frac=0.5,
            noise_sd=0.0,
            seed=42,
        )
        # All four (group, partition) cells must receive exactly one unit.
        unit_cells = data.groupby("unit")[["group", "partition"]].first()
        cell_counts = unit_cells.groupby(["group", "partition"]).size()
        assert len(cell_counts) == 4
        assert (cell_counts == 1).all()
        # TripleDifference.fit (which requires all 8 G×P×T cells) must succeed
        # on the smallest feasible config — validates the public contract
        # advertised in the docstring's compatibility paragraph.
        TripleDifference().fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            post="post",
        )

    def test_recommended_clustered_panel_path(self):
        """Documented clustered-by-unit path produces n_clusters == n_units inference.

        Locks the docstring's recommended pattern (`cluster="unit"` for panel data)
        against silent regression: TripleDifference is the repeated-cross-section
        `panel=FALSE` estimator, so unclustered SE on panel-generated rows
        understates variance. Clustering by `unit` aggregates per-row IFs within
        unit before variance computation (Liang-Zeger CR1).
        """
        from diff_diff import TripleDifference
        from diff_diff.prep import generate_ddd_panel_data

        n_units = 200
        data = generate_ddd_panel_data(
            n_units=n_units,
            n_periods=8,
            treatment_period=4,
            seed=42,
        )
        result = TripleDifference(cluster="unit").fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            post="post",
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        # Cluster-robust path records the number of clusters used.
        assert result.n_clusters == n_units
        # Unclustered fit on the same data MUST produce a different SE (clustering
        # is materially different when within-unit serial correlation exists).
        unclustered = TripleDifference().fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            post="post",
        )
        # Point estimate is invariant to clustering.
        np.testing.assert_allclose(unclustered.att, result.att, atol=1e-10)
        # SE differs because panel rows are not iid.
        assert unclustered.se != result.se

    def test_reproducibility(self):
        """Same seed produces identical DataFrames."""
        from diff_diff.prep import generate_ddd_panel_data

        data1 = generate_ddd_panel_data(n_units=100, n_periods=6, seed=123)
        data2 = generate_ddd_panel_data(n_units=100, n_periods=6, seed=123)
        pd.testing.assert_frame_equal(data1, data2)

    def test_ddd_effect_recovery_deterministic(self):
        """noise_sd=0 deterministic recovery within 1e-6 (catches DGP transcription bugs)."""
        from diff_diff import TripleDifference
        from diff_diff.prep import generate_ddd_panel_data

        true_effect = 2.0
        data = generate_ddd_panel_data(
            n_units=400,
            n_periods=8,
            treatment_period=4,
            treatment_effect=true_effect,
            noise_sd=0.0,
            seed=42,
        )
        result = TripleDifference().fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            post="post",
        )
        assert abs(result.att - true_effect) < 1e-6

    def test_ddd_effect_recovery(self):
        """Finite-sample recovery within tolerance under default noise."""
        from diff_diff import TripleDifference
        from diff_diff.prep import generate_ddd_panel_data

        true_effect = 2.0
        data = generate_ddd_panel_data(
            n_units=1000,
            n_periods=8,
            treatment_period=4,
            treatment_effect=true_effect,
            seed=42,
        )
        result = TripleDifference().fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            post="post",
        )
        assert abs(result.att - true_effect) < 0.7

    def test_covariates_added(self):
        """add_covariates=True adds x1 and x2 columns."""
        from diff_diff.prep import generate_ddd_panel_data

        data = generate_ddd_panel_data(
            n_units=100, n_periods=6, treatment_period=3, add_covariates=True, seed=42
        )
        assert "x1" in data.columns
        assert "x2" in data.columns

    def test_covariates_omitted(self):
        """add_covariates=False omits x1 and x2 columns."""
        from diff_diff.prep import generate_ddd_panel_data

        data = generate_ddd_panel_data(
            n_units=100, n_periods=6, treatment_period=3, add_covariates=False, seed=42
        )
        assert "x1" not in data.columns
        assert "x2" not in data.columns


class TestGeneratePanelData:
    """Tests for generate_panel_data function."""

    def test_basic_generation(self):
        """Test basic panel data generation."""
        from diff_diff.prep import generate_panel_data

        data = generate_panel_data(n_units=50, n_periods=6, seed=42)
        assert len(data) == 300  # 50 units x 6 periods
        assert set(data.columns) == {"unit", "period", "treated", "post", "outcome", "true_effect"}

    def test_treatment_fraction(self):
        """Test that treatment_fraction is respected."""
        from diff_diff.prep import generate_panel_data

        data = generate_panel_data(n_units=100, treatment_fraction=0.4, seed=42)
        n_treated_units = data.groupby("unit")["treated"].first().sum()
        assert n_treated_units == 40

    def test_treatment_period(self):
        """Test that treatment_period is respected."""
        from diff_diff.prep import generate_panel_data

        data = generate_panel_data(n_periods=10, treatment_period=5, seed=42)
        # Post should be 1 for periods >= 5
        assert (data[data["period"] < 5]["post"] == 0).all()
        assert (data[data["period"] >= 5]["post"] == 1).all()

    def test_parallel_trends(self):
        """Test data generation with parallel trends."""
        from diff_diff.prep import generate_panel_data

        data = generate_panel_data(
            n_units=200, n_periods=8, parallel_trends=True, noise_sd=0.1, seed=42
        )
        # Calculate pre-treatment trends
        pre_data = data[data["post"] == 0]
        treated_trend = pre_data[pre_data["treated"] == 1].groupby("period")["outcome"].mean()
        control_trend = pre_data[pre_data["treated"] == 0].groupby("period")["outcome"].mean()

        # Calculate slopes
        treated_slope = np.polyfit(treated_trend.index, treated_trend.values, 1)[0]
        control_slope = np.polyfit(control_trend.index, control_trend.values, 1)[0]

        # Slopes should be similar (parallel trends)
        assert abs(treated_slope - control_slope) < 0.5

    def test_non_parallel_trends(self):
        """Test data generation with trend violation."""
        from diff_diff.prep import generate_panel_data

        data = generate_panel_data(
            n_units=200,
            n_periods=8,
            parallel_trends=False,
            trend_violation=1.0,
            noise_sd=0.1,
            seed=42,
        )
        # Calculate pre-treatment trends
        pre_data = data[data["post"] == 0]
        treated_trend = pre_data[pre_data["treated"] == 1].groupby("period")["outcome"].mean()
        control_trend = pre_data[pre_data["treated"] == 0].groupby("period")["outcome"].mean()

        # Calculate slopes
        treated_slope = np.polyfit(treated_trend.index, treated_trend.values, 1)[0]
        control_slope = np.polyfit(control_trend.index, control_trend.values, 1)[0]

        # Treated slope should be steeper (trend violation)
        assert treated_slope > control_slope + 0.5

    def test_reproducibility(self):
        """Test seed produces reproducible data."""
        from diff_diff.prep import generate_panel_data

        data1 = generate_panel_data(seed=123)
        data2 = generate_panel_data(seed=123)
        pd.testing.assert_frame_equal(data1, data2)

    def test_invalid_treatment_period(self):
        """Test error on invalid treatment_period."""
        from diff_diff.prep import generate_panel_data

        with pytest.raises(ValueError, match="at least 1"):
            generate_panel_data(n_periods=10, treatment_period=0)

        with pytest.raises(ValueError, match="less than n_periods"):
            generate_panel_data(n_periods=10, treatment_period=10)


class TestGenerateEventStudyData:
    """Tests for generate_event_study_data function."""

    def test_basic_generation(self):
        """Test basic event study data generation."""
        from diff_diff.prep import generate_event_study_data

        data = generate_event_study_data(n_units=100, n_pre=5, n_post=5, seed=42)
        assert len(data) == 1000  # 100 units x 10 periods
        assert set(data.columns) == {
            "unit",
            "period",
            "treated",
            "post",
            "outcome",
            "event_time",
            "true_effect",
        }

    def test_event_time(self):
        """Test that event_time is correctly calculated."""
        from diff_diff.prep import generate_event_study_data

        data = generate_event_study_data(n_pre=5, n_post=5, seed=42)
        # Event time should range from -5 to 4
        assert data["event_time"].min() == -5
        assert data["event_time"].max() == 4

    def test_treatment_at_correct_period(self):
        """Test that treatment starts at period n_pre."""
        from diff_diff.prep import generate_event_study_data

        data = generate_event_study_data(n_pre=4, n_post=3, seed=42)
        # Post should be 1 for periods >= 4
        assert (data[data["period"] < 4]["post"] == 0).all()
        assert (data[data["period"] >= 4]["post"] == 1).all()

    def test_treatment_effect_recovery(self):
        """Test that treatment effect can be recovered."""
        from diff_diff.prep import generate_event_study_data

        true_effect = 4.0
        data = generate_event_study_data(
            n_units=500, n_pre=5, n_post=5, treatment_effect=true_effect, noise_sd=0.5, seed=42
        )

        # Simple DiD
        treated_post = data[(data["treated"] == 1) & (data["post"] == 1)]["outcome"].mean()
        treated_pre = data[(data["treated"] == 1) & (data["post"] == 0)]["outcome"].mean()
        control_post = data[(data["treated"] == 0) & (data["post"] == 1)]["outcome"].mean()
        control_pre = data[(data["treated"] == 0) & (data["post"] == 0)]["outcome"].mean()
        did_estimate = (treated_post - treated_pre) - (control_post - control_pre)

        assert abs(did_estimate - true_effect) < 1.0

    def test_reproducibility(self):
        """Test seed produces reproducible data."""
        from diff_diff.prep import generate_event_study_data

        data1 = generate_event_study_data(seed=123)
        data2 = generate_event_study_data(seed=123)
        pd.testing.assert_frame_equal(data1, data2)

    def test_treatment_fraction(self):
        """Test that treatment_fraction is respected."""
        from diff_diff.prep import generate_event_study_data

        data = generate_event_study_data(n_units=100, treatment_fraction=0.4, seed=42)
        n_treated_units = data.groupby("unit")["treated"].first().sum()
        assert n_treated_units == 40


class TestGenerateSurveyDidData:
    """Tests for generate_survey_did_data function."""

    def test_basic_shape_and_columns(self):
        """Test output shape and expected columns."""
        from diff_diff.prep import generate_survey_did_data

        data = generate_survey_did_data(n_units=100, n_periods=4, cohort_periods=[2, 3], seed=42)
        assert len(data) == 400  # 100 units x 4 periods
        expected = {
            "unit",
            "period",
            "outcome",
            "first_treat",
            "treated",
            "true_effect",
            "stratum",
            "psu",
            "fpc",
            "weight",
        }
        assert set(data.columns) == expected

    def test_survey_columns_valid(self):
        """Test survey columns have valid values."""
        from diff_diff.prep import generate_survey_did_data

        data = generate_survey_did_data(seed=42)
        assert (data["weight"] > 0).all()
        assert (data["fpc"] > 0).all()
        assert data["stratum"].dtype in [np.int64, np.int32, int]
        assert data["psu"].dtype in [np.int64, np.int32, int]

    def test_psu_nested_within_strata(self):
        """Test each PSU appears in exactly one stratum."""
        from diff_diff.prep import generate_survey_did_data

        data = generate_survey_did_data(n_strata=5, psu_per_stratum=8, seed=42)
        psu_strata = data.groupby("psu")["stratum"].nunique()
        assert (psu_strata == 1).all(), "PSUs must be nested within strata"

    def test_weight_variation_none(self):
        """Test that weight_variation='none' gives equal weights."""
        from diff_diff.prep import generate_survey_did_data

        data = generate_survey_did_data(weight_variation="none", seed=42)
        assert data["weight"].nunique() == 1
        assert data["weight"].iloc[0] == 1.0

    def test_weight_variation_moderate(self):
        """Test moderate weight variation has reasonable CV."""
        from diff_diff.prep import generate_survey_did_data

        data = generate_survey_did_data(weight_variation="moderate", seed=42)
        unit_weights = data.groupby("unit")["weight"].first()
        cv = unit_weights.std() / unit_weights.mean()
        assert 0.05 < cv < 0.6

    def test_weight_variation_high(self):
        """Test high weight variation has larger CV than moderate."""
        from diff_diff.prep import generate_survey_did_data

        data_mod = generate_survey_did_data(weight_variation="moderate", seed=42)
        data_high = generate_survey_did_data(weight_variation="high", seed=42)
        cv_mod = data_mod.groupby("unit")["weight"].first().std()
        cv_high = data_high.groupby("unit")["weight"].first().std()
        assert cv_high > cv_mod

    def test_replicate_weights(self):
        """Test replicate weight columns are generated correctly."""
        from diff_diff.prep import generate_survey_did_data

        n_strata, psu_per = 3, 4
        data = generate_survey_did_data(
            n_strata=n_strata,
            psu_per_stratum=psu_per,
            include_replicate_weights=True,
            seed=42,
        )
        n_psu = n_strata * psu_per
        rep_cols = [c for c in data.columns if c.startswith("rep_")]
        assert len(rep_cols) == n_psu

        # Each replicate should zero out one PSU
        for r in range(n_psu):
            assert data.loc[data[f"rep_{r}"] == 0, "psu"].nunique() == 1

    def test_covariates(self):
        """Test covariate columns are added when requested."""
        from diff_diff.prep import generate_survey_did_data

        data = generate_survey_did_data(add_covariates=True, seed=42)
        assert "x1" in data.columns
        assert "x2" in data.columns

    def test_no_covariates_by_default(self):
        """Test no covariate columns by default."""
        from diff_diff.prep import generate_survey_did_data

        data = generate_survey_did_data(seed=42)
        assert "x1" not in data.columns
        assert "x2" not in data.columns

    def test_seed_reproducibility(self):
        """Test that same seed produces identical output."""
        from diff_diff.prep import generate_survey_did_data

        data1 = generate_survey_did_data(seed=123)
        data2 = generate_survey_did_data(seed=123)
        pd.testing.assert_frame_equal(data1, data2)

    def test_treatment_structure(self):
        """Test treatment cohorts match cohort_periods + never-treated."""
        from diff_diff.prep import generate_survey_did_data

        data = generate_survey_did_data(
            cohort_periods=[3, 5],
            never_treated_frac=0.3,
            seed=42,
        )
        cohorts = set(data.groupby("unit")["first_treat"].first().unique())
        assert 0 in cohorts  # never-treated
        assert 3 in cohorts
        assert 5 in cohorts

    def test_uneven_units_per_stratum(self):
        """Test that n_units not divisible by n_strata still works."""
        from diff_diff.prep import generate_survey_did_data

        # 103 units / 5 strata = 20 remainder 3
        data = generate_survey_did_data(n_units=103, n_strata=5, seed=42)
        assert len(data) == 103 * 8  # default 8 periods
        assert data["stratum"].nunique() == 5

    def test_top_level_import(self):
        """Test that generate_survey_did_data is importable from diff_diff."""
        from diff_diff import generate_survey_did_data

        data = generate_survey_did_data(n_units=10, n_periods=4, cohort_periods=[2], seed=42)
        assert len(data) == 40

    def test_jk1_minimum_psu_guard(self):
        """Test that JK1 replicates require at least 2 PSUs."""
        import pytest

        from diff_diff.prep import generate_survey_did_data

        # Configured count: 1 PSU total
        with pytest.raises(ValueError, match="at least 2 PSUs"):
            generate_survey_did_data(
                n_strata=1,
                psu_per_stratum=1,
                include_replicate_weights=True,
                seed=42,
            )

    def test_jk1_one_populated_psu_guard(self):
        """Test JK1 guard fires when only one PSU is populated."""
        import pytest

        from diff_diff.prep import generate_survey_did_data

        # 2 configured PSUs but only 1 unit -> only 1 populated PSU
        with pytest.raises(ValueError, match="at least 2 populated PSUs"):
            generate_survey_did_data(
                n_units=1,
                n_strata=1,
                psu_per_stratum=2,
                cohort_periods=[2],
                n_periods=4,
                include_replicate_weights=True,
                seed=42,
            )

    def test_repeated_cross_section(self):
        """Test panel=False generates unique unit IDs per period."""
        from diff_diff.prep import generate_survey_did_data

        data = generate_survey_did_data(
            n_units=20,
            n_periods=4,
            cohort_periods=[2],
            panel=False,
            seed=42,
        )
        assert len(data) == 80
        assert data["unit"].nunique() == 80  # unique across all periods
        # No unit appears in more than one period
        assert data.groupby("unit")["period"].nunique().max() == 1

    def test_invalid_weight_variation(self):
        """Test that invalid weight_variation raises ValueError."""
        import pytest

        from diff_diff.prep import generate_survey_did_data

        with pytest.raises(ValueError, match="weight_variation must be"):
            generate_survey_did_data(weight_variation="invalid", seed=42)

    def test_empty_cohort_periods(self):
        """Test that empty cohort_periods raises ValueError."""
        import pytest

        from diff_diff.prep import generate_survey_did_data

        with pytest.raises(ValueError, match="cohort_periods must be"):
            generate_survey_did_data(cohort_periods=[], seed=42)

    def test_cohort_period_out_of_range(self):
        """Test that out-of-range cohort periods raise ValueError."""
        import pytest

        from diff_diff.prep import generate_survey_did_data

        # g=1 is invalid: no pre-treatment period (must be >= 2)
        with pytest.raises(ValueError, match="must be between"):
            generate_survey_did_data(cohort_periods=[1], seed=42)
        # g > n_periods is invalid
        with pytest.raises(ValueError, match="must be between"):
            generate_survey_did_data(n_periods=8, cohort_periods=[9], seed=42)
        # g = n_periods is valid (last-period adoption, base period g-1 exists)
        data = generate_survey_did_data(n_periods=8, cohort_periods=[8], seed=42)
        assert len(data) == 200 * 8

    def test_cohort_period_non_integer(self):
        """Test that non-integer cohort periods raise ValueError."""
        import pytest

        from diff_diff.prep import generate_survey_did_data

        with pytest.raises(ValueError, match="must contain integers"):
            generate_survey_did_data(cohort_periods=[2.5], seed=42)

    def test_numpy_integer_cohort_periods(self):
        """Test that numpy integer cohort periods are accepted (list and array)."""
        from diff_diff.prep import generate_survey_did_data

        # As list of numpy integers
        periods = np.array([3, 5], dtype=np.int64)
        data = generate_survey_did_data(cohort_periods=list(periods), seed=42)
        assert len(data) == 200 * 8

        # As numpy array directly
        data2 = generate_survey_did_data(cohort_periods=periods, seed=42)
        assert len(data2) == 200 * 8

    def test_default_cohort_periods_small_n_periods(self):
        """Test default cohort_periods adapts to small n_periods with pre-periods."""
        from diff_diff.prep import generate_survey_did_data

        for n_per in [4, 5, 6, 7]:
            data = generate_survey_did_data(n_periods=n_per, seed=42)
            assert len(data) == 200 * n_per
            cohorts = data.groupby("unit")["first_treat"].first().unique()
            # Every treated cohort must have g >= 2 (at least one pre-period)
            for g in cohorts:
                if g > 0:
                    assert g >= 2, f"n_periods={n_per}: cohort g={g} has no pre-period"
                    assert g <= n_per, f"n_periods={n_per}: cohort g={g} > n_periods"

    def test_default_cohort_periods_too_small(self):
        """Test that n_periods < 4 with default cohort_periods raises."""
        import pytest

        from diff_diff.prep import generate_survey_did_data

        with pytest.raises(ValueError, match="too small"):
            generate_survey_did_data(n_periods=3, seed=42)

    def test_parameter_validation(self):
        """Test upfront validation for invalid parameter values."""
        import pytest

        from diff_diff.prep import generate_survey_did_data

        with pytest.raises(ValueError, match="n_units must be positive"):
            generate_survey_did_data(n_units=0, seed=42)
        with pytest.raises(ValueError, match="n_periods must be positive"):
            generate_survey_did_data(n_periods=0, seed=42)
        with pytest.raises(ValueError, match="n_strata must be positive"):
            generate_survey_did_data(n_strata=0, seed=42)
        with pytest.raises(ValueError, match="psu_per_stratum must be positive"):
            generate_survey_did_data(psu_per_stratum=0, seed=42)
        with pytest.raises(ValueError, match="never_treated_frac must be between"):
            generate_survey_did_data(never_treated_frac=-0.1, seed=42)
        with pytest.raises(ValueError, match="never_treated_frac must be between"):
            generate_survey_did_data(never_treated_frac=1.1, seed=42)
        with pytest.raises(ValueError, match="fpc_per_stratum.*must be >= psu_per_stratum"):
            generate_survey_did_data(fpc_per_stratum=3, psu_per_stratum=8, seed=42)

    def test_psu_period_factor(self):
        """Test that psu_period_factor controls time-varying PSU clustering."""
        from diff_diff.prep import generate_survey_did_data

        data_low = generate_survey_did_data(psu_period_factor=0.0, seed=42)
        data_high = generate_survey_did_data(psu_period_factor=2.0, seed=42)
        # Higher factor increases outcome variance (more PSU-period shocks)
        assert data_high["outcome"].std() > data_low["outcome"].std()
        # Same structure
        assert set(data_low.columns) == set(data_high.columns)
        assert len(data_low) == len(data_high)

    def test_psu_period_factor_deff_regression(self):
        """Verify psu_period_factor=1.0 gives DEFF > 1 for the tutorial scenario."""
        import warnings

        from diff_diff import (
            DifferenceInDifferences,
            SurveyDesign,
        )
        from diff_diff.linalg import LinearRegression
        from diff_diff.prep import generate_survey_did_data

        warnings.filterwarnings("ignore")
        df = generate_survey_did_data(
            n_units=200,
            n_periods=8,
            cohort_periods=[3, 5],
            never_treated_frac=0.3,
            treatment_effect=2.0,
            n_strata=5,
            psu_per_stratum=8,
            fpc_per_stratum=200.0,
            weight_variation="moderate",
            psu_re_sd=2.0,
            psu_period_factor=1.0,
            seed=42,
        )
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc")

        # 2x2 subset: survey SE must exceed naive SE
        c3 = df[(df["first_treat"].isin([0, 3])) & (df["period"].isin([2, 3]))].copy()
        c3["post"] = (c3["period"] == 3).astype(int)
        c3["treat"] = (c3["first_treat"] == 3).astype(int)
        did = DifferenceInDifferences()
        r_naive = did.fit(c3, outcome="outcome", treatment="treat", post="post")
        r_survey = did.fit(
            c3,
            outcome="outcome",
            treatment="treat",
            post="post",
            survey_design=sd,
        )
        assert (
            r_survey.se > r_naive.se
        ), f"Survey SE ({r_survey.se:.4f}) should exceed naive SE ({r_naive.se:.4f})"

        # DEFF for treat_x_post must be > 1
        c3["treat_x_post"] = c3["treat"] * c3["post"]
        resolved = sd.resolve(c3)
        reg = LinearRegression(include_intercept=True, survey_design=resolved)
        reg.fit(X=c3[["treat", "post", "treat_x_post"]].values, y=c3["outcome"].values)
        deff = reg.compute_deff(coefficient_names=["intercept", "treat", "post", "treat_x_post"])
        txp_deff = deff.deff[3]  # treat_x_post
        assert txp_deff > 1.0, f"DEFF for treat_x_post ({txp_deff:.2f}) should be > 1"

    def test_psu_period_factor_validation(self):
        """Test that invalid psu_period_factor values raise ValueError."""
        import math

        import pytest

        from diff_diff.prep import generate_survey_did_data

        with pytest.raises(ValueError, match="psu_period_factor"):
            generate_survey_did_data(psu_period_factor=-1.0, seed=42)
        with pytest.raises(ValueError, match="psu_period_factor"):
            generate_survey_did_data(psu_period_factor=math.nan, seed=42)
        with pytest.raises(ValueError, match="psu_period_factor"):
            generate_survey_did_data(psu_period_factor=math.inf, seed=42)


class TestSurveyDGPResearchGrade:
    """Tests for research-grade DGP parameters added to generate_survey_did_data."""

    def test_icc_parameter(self):
        """Realized ICC should be within 50% relative tolerance of target."""
        from diff_diff.prep_dgp import generate_survey_did_data

        target_icc = 0.3
        df = generate_survey_did_data(n_units=1000, icc=target_icc, seed=42)
        # ANOVA-based ICC on period 1 (pre-treatment, no TE contamination)
        p1 = df[df["period"] == 1]
        groups = p1.groupby("psu")["outcome"]
        grand_mean = p1["outcome"].mean()
        n_total = len(p1)
        n_groups = groups.ngroups
        n_bar = n_total / n_groups
        ssb = (groups.size() * (groups.mean() - grand_mean) ** 2).sum()
        msb = ssb / (n_groups - 1)
        ssw = groups.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()
        msw = ssw / (n_total - n_groups)
        realized_icc = (msb - msw) / (msb + (n_bar - 1) * msw)
        assert abs(realized_icc - target_icc) / target_icc < 0.50

    def test_icc_zero_variance_rejected(self):
        """icc with zero non-PSU variance should raise ValueError."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="non-zero non-PSU variance"):
            generate_survey_did_data(
                icc=0.3, unit_fe_sd=0, noise_sd=0, add_covariates=False, seed=42
            )

    def test_icc_and_psu_re_sd_conflict(self):
        """Cannot specify both icc and a non-default psu_re_sd."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="Cannot specify both icc"):
            generate_survey_did_data(icc=0.3, psu_re_sd=3.0, seed=42)

    def test_icc_out_of_range(self):
        """icc must be in (0, 1)."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="icc must be between"):
            generate_survey_did_data(icc=0.0, seed=42)
        with pytest.raises(ValueError, match="icc must be between"):
            generate_survey_did_data(icc=1.0, seed=42)

    def test_weight_cv_parameter(self):
        """Realized weight CV should be within 0.15 of target."""
        from diff_diff.prep_dgp import generate_survey_did_data

        target_cv = 0.5
        df = generate_survey_did_data(n_units=1000, weight_cv=target_cv, seed=42)
        weights = df.groupby("unit")["weight"].first().values
        realized_cv = weights.std() / weights.mean()
        assert abs(realized_cv - target_cv) < 0.15

    def test_weight_cv_and_weight_variation_conflict(self):
        """Cannot specify both weight_cv and a non-default weight_variation."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="Cannot specify both weight_cv"):
            generate_survey_did_data(weight_cv=0.5, weight_variation="high", seed=42)

    def test_weight_cv_nan_inf(self):
        """weight_cv must reject NaN and Inf."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="weight_cv must be finite"):
            generate_survey_did_data(weight_cv=np.nan, seed=42)
        with pytest.raises(ValueError, match="weight_cv must be finite"):
            generate_survey_did_data(weight_cv=np.inf, seed=42)

    def test_informative_sampling_panel(self):
        """Informative sampling should create weight-outcome correlation."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=1000,
            informative_sampling=True,
            weight_cv=0.5,
            seed=42,
        )
        # Period-1 outcomes: weighted mean should differ from unweighted
        p1 = df[df["period"] == 1]
        unwt_mean = p1["outcome"].mean()
        wt_mean = np.average(p1["outcome"], weights=p1["weight"])
        assert abs(wt_mean - unwt_mean) > 0.1
        # Positive correlation: higher outcome → heavier weight
        corr = np.corrcoef(p1["weight"], p1["outcome"])[0, 1]
        assert corr > 0.1

    def test_informative_sampling_default_weights(self):
        """Informative sampling preserves stratum-level weight structure."""
        from diff_diff.prep_dgp import generate_survey_did_data

        # Generate with informative_sampling but default weight_variation
        df = generate_survey_did_data(
            n_units=1000,
            informative_sampling=True,
            seed=42,
        )
        # Reference: expected stratum mean weights from weight_variation="moderate"
        # Formula: 1.0 + 1.0 * (s / max(n_strata-1, 1)) for s=0..4
        p1 = df[df["period"] == 1]
        for s in range(5):
            expected_mean = 1.0 + 1.0 * (s / 4)
            stratum_weights = p1.loc[p1["stratum"] == s, "weight"]
            assert abs(stratum_weights.mean() - expected_mean) < 0.15, (
                f"Stratum {s}: expected mean ~{expected_mean}, " f"got {stratum_weights.mean():.3f}"
            )
            # Within-stratum variation should exist (informative sampling)
            assert stratum_weights.std() > 0.01

    def test_informative_sampling_cross_section(self):
        """Cross-section informative sampling: per-period positive correlation.

        Under w_i = 1/pi_i, under-covered (high-outcome) units get heavier
        weights, so weight and outcome should be positively correlated.
        """
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=1000,
            informative_sampling=True,
            weight_cv=0.5,
            panel=False,
            seed=42,
        )
        # Check correlation for period 1
        p1 = df[df["period"] == 1]
        corr = np.corrcoef(p1["weight"], p1["outcome"])[0, 1]
        assert corr > 0.1

    def test_informative_sampling_cross_section_default_weights(self):
        """Cross-section informative sampling with default weight_variation."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=1000,
            informative_sampling=True,
            panel=False,
            seed=42,
        )
        p1 = df[df["period"] == 1]
        for s in range(5):
            expected_mean = 1.0 + 1.0 * (s / 4)
            stratum_weights = p1.loc[p1["stratum"] == s, "weight"]
            assert abs(stratum_weights.mean() - expected_mean) < 0.15
            assert stratum_weights.std() > 0.01

    def test_icc_with_covariates(self):
        """ICC calibration should account for covariate variance."""
        from diff_diff.prep_dgp import generate_survey_did_data

        target_icc = 0.3
        df = generate_survey_did_data(n_units=1000, icc=target_icc, add_covariates=True, seed=42)
        # ANOVA-based ICC on period 1
        p1 = df[df["period"] == 1]
        groups = p1.groupby("psu")["outcome"]
        grand_mean = p1["outcome"].mean()
        n_total = len(p1)
        n_groups = groups.ngroups
        n_bar = n_total / n_groups
        ssb = (groups.size() * (groups.mean() - grand_mean) ** 2).sum()
        msb = ssb / (n_groups - 1)
        ssw = groups.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()
        msw = ssw / (n_total - n_groups)
        realized_icc = (msb - msw) / (msb + (n_bar - 1) * msw)
        assert abs(realized_icc - target_icc) / target_icc < 0.50

    def test_informative_sampling_with_covariates_panel(self):
        """Informative sampling includes covariates in Y(0) ranking (panel)."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=1000,
            informative_sampling=True,
            add_covariates=True,
            seed=42,
        )
        p1 = df[df["period"] == 1]
        # Positive weight-outcome correlation preserved with covariates
        corr = np.corrcoef(p1["weight"], p1["outcome"])[0, 1]
        assert corr > 0.1
        # Covariates should be present
        assert "x1" in df.columns
        assert "x2" in df.columns

    def test_informative_sampling_with_covariates_cross_section(self):
        """Informative sampling includes covariates in Y(0) ranking (cross-section)."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=1000,
            informative_sampling=True,
            add_covariates=True,
            panel=False,
            seed=42,
        )
        p1 = df[df["period"] == 1]
        corr = np.corrcoef(p1["weight"], p1["outcome"])[0, 1]
        assert corr > 0.1
        assert "x1" in df.columns

    def test_informative_sampling_covariate_ranking_direct(self):
        """Verify covariates actually affect weight assignment in ranking.

        Use large covariate effects with tiny unit_fe_sd/psu_re_sd so
        covariates dominate Y(0). Weights with nonzero vs zero covariate
        effects should differ.
        """
        from diff_diff.prep_dgp import generate_survey_did_data

        # Covariates dominate: large beta, tiny structural variance
        df_with = generate_survey_did_data(
            n_units=200,
            informative_sampling=True,
            add_covariates=True,
            covariate_effects=(5.0, 0.0),
            unit_fe_sd=0.01,
            psu_re_sd=0.01,
            noise_sd=0.01,
            seed=42,
        )
        df_without = generate_survey_did_data(
            n_units=200,
            informative_sampling=True,
            add_covariates=True,
            covariate_effects=(0.0, 0.0),
            unit_fe_sd=0.01,
            psu_re_sd=0.01,
            noise_sd=0.01,
            seed=42,
        )
        # Weight assignments should differ when covariates dominate ranking
        w_with = df_with[df_with["period"] == 1]["weight"].values
        w_without = df_without[df_without["period"] == 1]["weight"].values
        assert not np.allclose(w_with, w_without, atol=0.01)

    def test_heterogeneous_te_by_strata(self):
        """Unweighted mean TE should differ from population ATT."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=1000,
            heterogeneous_te_by_strata=True,
            strata_sizes=[400, 200, 200, 100, 100],
            return_true_population_att=True,
            seed=42,
        )
        treated = df[df["treated"] == 1]
        unwt_mean_te = treated["true_effect"].mean()
        pop_att = df.attrs["dgp_truth"]["population_att"]
        # With unequal strata sizes + heterogeneous TE, these should differ
        assert abs(unwt_mean_te - pop_att) > 0.01

    def test_heterogeneous_te_single_stratum(self):
        """n_strata=1 with heterogeneous TE should not crash."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=50,
            n_strata=1,
            psu_per_stratum=8,
            fpc_per_stratum=200.0,
            heterogeneous_te_by_strata=True,
            seed=42,
        )
        treated = df[df["treated"] == 1]
        # All treated units should have the base treatment_effect
        assert np.allclose(treated["true_effect"].unique(), [2.0], atol=0.01)

    def test_return_true_population_att(self):
        """dgp_truth dict should have expected keys and reasonable values."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=1000,
            icc=0.3,
            return_true_population_att=True,
            seed=42,
        )
        truth = df.attrs["dgp_truth"]
        assert "population_att" in truth
        assert "deff_kish" in truth
        assert "base_stratum_effects" in truth
        assert "icc_realized" in truth
        assert truth["deff_kish"] >= 1.0
        assert truth["icc_realized"] >= 0.0
        # icc_realized should track the target ICC (ANOVA-based, same formula)
        assert abs(truth["icc_realized"] - 0.3) / 0.3 < 0.50

    def test_population_att_nan_no_treated(self):
        """population_att should be NaN when there are no treated units."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=50,
            never_treated_frac=1.0,
            return_true_population_att=True,
            seed=42,
        )
        assert np.isnan(df.attrs["dgp_truth"]["population_att"])

    def test_icc_realized_nan_no_replication(self):
        """icc_realized should be NaN when period-1 has no within-PSU replication."""
        from diff_diff.prep_dgp import generate_survey_did_data

        # 5 units across 5 strata with 8 PSUs each = 1 unit per PSU (no replication)
        df = generate_survey_did_data(
            n_units=5,
            n_strata=5,
            psu_per_stratum=8,
            return_true_population_att=True,
            seed=42,
        )
        assert np.isnan(df.attrs["dgp_truth"]["icc_realized"])

    def test_strata_sizes(self):
        """Custom strata_sizes should produce correct per-stratum counts."""
        from diff_diff.prep_dgp import generate_survey_did_data

        sizes = [60, 50, 40, 30, 20]
        df = generate_survey_did_data(n_units=200, strata_sizes=sizes, seed=42)
        for s, expected in enumerate(sizes):
            actual = df[df["period"] == 1]["stratum"].value_counts().get(s, 0)
            assert actual == expected

    def test_strata_sizes_sum_mismatch(self):
        """strata_sizes must sum to n_units."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="strata_sizes must sum"):
            generate_survey_did_data(n_units=200, strata_sizes=[50, 50, 50, 50, 49], seed=42)

    def test_strata_sizes_float_rejected(self):
        """strata_sizes must contain integers, not floats."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="strata_sizes must contain integers"):
            generate_survey_did_data(
                n_units=200, strata_sizes=[40.0, 40.0, 40.0, 40.0, 40.0], seed=42
            )

    def test_backward_compatibility(self):
        """Default params with same seed produce identical DataFrames."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df1 = generate_survey_did_data(seed=123)
        df2 = generate_survey_did_data(seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_covariate_effects_custom(self):
        """Custom covariate coefficients should change outcome variance."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df_default = generate_survey_did_data(n_units=500, add_covariates=True, seed=42)
        df_large = generate_survey_did_data(
            n_units=500,
            add_covariates=True,
            covariate_effects=(2.0, 1.0),
            seed=42,
        )
        # Larger coefficients → larger outcome variance
        assert df_large["outcome"].var() > df_default["outcome"].var()

    def test_covariate_effects_zero(self):
        """Zero covariate effects should produce same variance as no covariates."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df_no_cov = generate_survey_did_data(n_units=500, add_covariates=False, seed=42)
        df_zero = generate_survey_did_data(
            n_units=500,
            add_covariates=True,
            covariate_effects=(0.0, 0.0),
            seed=42,
        )
        # Outcome variance should be similar (covariates contribute nothing)
        assert abs(df_zero["outcome"].var() - df_no_cov["outcome"].var()) < 0.5

    def test_te_covariate_interaction(self):
        """Covariate interaction should create unit-level TE heterogeneity."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=500,
            add_covariates=True,
            te_covariate_interaction=1.0,
            seed=42,
        )
        treated = df[df["treated"] == 1]
        # true_effect should vary across treated units (not constant)
        assert treated["true_effect"].std() > 0.1

    def test_te_covariate_interaction_requires_covariates(self):
        """te_covariate_interaction without add_covariates should raise."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="te_covariate_interaction requires"):
            generate_survey_did_data(te_covariate_interaction=0.5, add_covariates=False, seed=42)

    def test_covariate_effects_validation(self):
        """covariate_effects must be length 2 and finite."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="covariate_effects must have length 2"):
            generate_survey_did_data(add_covariates=True, covariate_effects=(1.0,), seed=42)
        with pytest.raises(ValueError, match="covariate_effects must be finite"):
            generate_survey_did_data(add_covariates=True, covariate_effects=(np.nan, 0.3), seed=42)
        with pytest.raises(ValueError, match="covariate_effects must be finite"):
            generate_survey_did_data(add_covariates=True, covariate_effects=(0.5, np.inf), seed=42)

    def test_te_covariate_interaction_validation(self):
        """te_covariate_interaction must be finite."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="te_covariate_interaction must be finite"):
            generate_survey_did_data(add_covariates=True, te_covariate_interaction=np.nan, seed=42)

    # --- conditional_pt parameter tests ---

    def test_conditional_pt_requires_both_groups(self):
        """conditional_pt requires at least one ever-treated and one never-treated."""
        from diff_diff.prep_dgp import generate_survey_did_data

        # Zero never-treated (exact)
        with pytest.raises(ValueError, match="conditional_pt requires at least one"):
            generate_survey_did_data(
                add_covariates=True,
                conditional_pt=0.3,
                never_treated_frac=0.0,
                seed=42,
            )
        # Small fraction that floors to zero never-treated units
        with pytest.raises(ValueError, match="conditional_pt requires at least one"):
            generate_survey_did_data(
                n_units=50,
                add_covariates=True,
                conditional_pt=0.3,
                never_treated_frac=0.01,
                seed=42,
            )
        # All never-treated (no ever-treated units)
        with pytest.raises(ValueError, match="conditional_pt requires at least one"):
            generate_survey_did_data(
                add_covariates=True,
                conditional_pt=0.3,
                never_treated_frac=1.0,
                seed=42,
            )

    def test_conditional_pt_requires_covariates(self):
        """conditional_pt requires add_covariates=True."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="conditional_pt requires add_covariates"):
            generate_survey_did_data(conditional_pt=0.3, add_covariates=False, seed=42)

    def test_conditional_pt_nonfinite_rejected(self):
        """conditional_pt must be finite."""
        from diff_diff.prep_dgp import generate_survey_did_data

        with pytest.raises(ValueError, match="conditional_pt must be finite"):
            generate_survey_did_data(add_covariates=True, conditional_pt=np.inf, seed=42)
        with pytest.raises(ValueError, match="conditional_pt must be finite"):
            generate_survey_did_data(add_covariates=True, conditional_pt=np.nan, seed=42)

    def test_conditional_pt_x1_distribution_shift(self):
        """Treated units should have higher x1 when conditional_pt is active."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=1000,
            n_periods=4,
            add_covariates=True,
            conditional_pt=0.3,
            seed=42,
        )
        p1 = df[df["period"] == 1]
        x1_treated = p1.loc[p1["first_treat"] > 0, "x1"].values
        x1_control = p1.loc[p1["first_treat"] == 0, "x1"].values
        shift = x1_treated.mean() - x1_control.mean()
        # Expect ~1.0 SD shift; require at least 0.5
        assert shift > 0.5, f"x1 mean shift too small: {shift:.3f}"

    def test_conditional_pt_unconditional_pt_fails(self):
        """With conditional_pt active, unconditional pre-trends should differ."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=2000,
            n_periods=8,
            add_covariates=True,
            conditional_pt=0.5,
            never_treated_frac=0.5,
            seed=42,
        )
        # Compute mean outcome change (period 2 - period 1) for each group
        # before any treatment (use periods 1 and 2, treatment starts at 3+)
        p1 = df[df["period"] == 1].set_index("unit")
        p2 = df[df["period"] == 2].set_index("unit")
        common = p1.index.intersection(p2.index)
        dy = p2.loc[common, "outcome"] - p1.loc[common, "outcome"]
        is_treated = p1.loc[common, "first_treat"] > 0

        trend_treated = dy[is_treated].mean()
        trend_control = dy[~is_treated].mean()
        gap = abs(trend_treated - trend_control)
        # With conditional_pt=0.5 and 1 SD shift, expect a detectable gap
        assert gap > 0.01, f"Unconditional PT gap too small: {gap:.4f}"

    def test_conditional_pt_conditional_pt_holds(self):
        """Controlling for x1, treated/control pre-trends should be equal.

        Use low PSU noise so the conditional_pt signal dominates.
        """
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=2000,
            n_periods=8,
            add_covariates=True,
            conditional_pt=2.0,
            never_treated_frac=0.5,
            psu_re_sd=0.1,
            psu_period_factor=0.1,
            noise_sd=0.2,
            seed=42,
        )
        p1 = df[df["period"] == 1].set_index("unit")
        p2 = df[df["period"] == 2].set_index("unit")
        common = p1.index.intersection(p2.index)
        dy = p2.loc[common, "outcome"].values - p1.loc[common, "outcome"].values
        x1_vals = p1.loc[common, "x1"].values
        is_treated = (p1.loc[common, "first_treat"] > 0).values.astype(float)

        # Unconditional regression: dy ~ treated (should show large gap)
        n = len(dy)
        X_uncond = np.column_stack([np.ones(n), is_treated])
        beta_uncond = np.linalg.lstsq(X_uncond, dy, rcond=None)[0]
        uncond_gap = abs(beta_uncond[1])

        # Conditional regression: dy ~ treated + x1 (gap should shrink)
        X_cond = np.column_stack([np.ones(n), is_treated, x1_vals])
        beta_cond = np.linalg.lstsq(X_cond, dy, rcond=None)[0]
        cond_gap = abs(beta_cond[1])

        # With low noise and strong signal, controlling for x1 should
        # substantially reduce the treated coefficient
        assert uncond_gap > 0.05, f"Unconditional gap too small: {uncond_gap:.4f}"
        assert cond_gap < uncond_gap * 0.5, (
            f"Conditional gap ({cond_gap:.4f}) should be much smaller than "
            f"unconditional ({uncond_gap:.4f})"
        )

    def test_conditional_pt_crosssection_trend_did(self):
        """In cross-section mode, DID across pre-periods isolates the trend term.

        Uses group-level mean DID across periods 1 and 2 (both pre-treatment).
        This is specific to the conditional_pt * x1 * (t/T) trend - the level
        effect _beta1 * x1 cancels in the period difference.
        """
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=2000,
            n_periods=8,
            add_covariates=True,
            conditional_pt=2.0,
            never_treated_frac=0.5,
            psu_re_sd=0.1,
            psu_period_factor=0.1,
            noise_sd=0.2,
            panel=False,
            seed=42,
        )
        # Group-mean DID across pre-treatment periods 1 and 2.
        # The conditional_pt trend creates differential growth:
        # DID = conditional_pt * (E[x1|treated] - E[x1|control]) * (1/T)
        # With conditional_pt=2.0, shift=1.0, T=8: DID ≈ 0.25
        p1 = df[df["period"] == 1]
        p2 = df[df["period"] == 2]

        mean_t_p1 = p1.loc[p1["first_treat"] > 0, "outcome"].mean()
        mean_c_p1 = p1.loc[p1["first_treat"] == 0, "outcome"].mean()
        mean_t_p2 = p2.loc[p2["first_treat"] > 0, "outcome"].mean()
        mean_c_p2 = p2.loc[p2["first_treat"] == 0, "outcome"].mean()

        did = (mean_t_p2 - mean_c_p2) - (mean_t_p1 - mean_c_p1)
        # Expected DID ≈ 0.25; should be positive and detectable
        assert did > 0.05, (
            f"Cross-section DID too small: {did:.4f}. "
            f"The conditional_pt trend term may not be active."
        )

    def test_conditional_pt_crosssection_conditional_did(self):
        """In cross-section mode, x1-adjusted DID should shrink vs unconditional.

        Pools periods 1 and 2, regresses outcome on treated, post, treated*post,
        x1, and x1*post. The treated*post coefficient should shrink when x1
        interactions are included.
        """
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=2000,
            n_periods=8,
            add_covariates=True,
            conditional_pt=2.0,
            never_treated_frac=0.5,
            psu_re_sd=0.1,
            psu_period_factor=0.1,
            noise_sd=0.2,
            panel=False,
            seed=42,
        )
        pre = df[df["period"].isin([1, 2])].copy()
        y = pre["outcome"].values
        treated = (pre["first_treat"] > 0).values.astype(float)
        post = (pre["period"] == 2).values.astype(float)
        treated_post = treated * post
        x1 = pre["x1"].values
        x1_post = x1 * post
        n = len(y)

        # Unconditional DID: outcome ~ 1 + treated + post + treated*post
        X_uncond = np.column_stack([np.ones(n), treated, post, treated_post])
        beta_uncond = np.linalg.lstsq(X_uncond, y, rcond=None)[0]
        uncond_did = abs(beta_uncond[3])

        # Conditional DID: add x1 and x1*post
        X_cond = np.column_stack([np.ones(n), treated, post, treated_post, x1, x1_post])
        beta_cond = np.linalg.lstsq(X_cond, y, rcond=None)[0]
        cond_did = abs(beta_cond[3])

        assert uncond_did > 0.05, f"Unconditional DID too small: {uncond_did:.4f}"
        assert cond_did < uncond_did * 0.5, (
            f"Cross-section conditional DID ({cond_did:.4f}) should be much "
            f"smaller than unconditional ({uncond_did:.4f})"
        )

    def test_conditional_pt_backward_compatible(self):
        """conditional_pt=0.0 should produce identical output to default."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df_default = generate_survey_did_data(n_units=100, add_covariates=True, seed=99)
        df_explicit = generate_survey_did_data(
            n_units=100, add_covariates=True, conditional_pt=0.0, seed=99
        )
        pd.testing.assert_frame_equal(df_default, df_explicit)

    def test_conditional_pt_informative_sampling(self):
        """conditional_pt x1 shift should survive informative-sampling ranking."""
        from diff_diff.prep_dgp import generate_survey_did_data

        for panel_mode in [True, False]:
            df = generate_survey_did_data(
                n_units=1000,
                n_periods=4,
                add_covariates=True,
                conditional_pt=0.3,
                informative_sampling=True,
                panel=panel_mode,
                seed=42,
            )
            p1 = df[df["period"] == 1]
            x1_treated = p1.loc[p1["first_treat"] > 0, "x1"].mean()
            x1_control = p1.loc[p1["first_treat"] == 0, "x1"].mean()
            shift = x1_treated - x1_control
            assert shift > 0.5, (
                f"panel={panel_mode}: x1 shift too small after "
                f"informative sampling ranking: {shift:.3f}"
            )

    def test_conditional_pt_dgp_truth_diagnostics(self):
        """dgp_truth should include conditional_pt_active and valid ICC."""
        from diff_diff.prep_dgp import generate_survey_did_data

        df = generate_survey_did_data(
            n_units=500,
            n_periods=4,
            add_covariates=True,
            conditional_pt=0.3,
            icc=0.15,
            return_true_population_att=True,
            seed=42,
        )
        truth = df.attrs["dgp_truth"]
        assert truth["conditional_pt_active"] is True
        assert np.isfinite(truth["icc_realized"])
        assert np.isfinite(truth["population_att"])

    def test_conditional_pt_panel_and_crosssection(self):
        """conditional_pt should work in both panel and cross-section modes."""
        from diff_diff.prep_dgp import generate_survey_did_data

        for panel_mode in [True, False]:
            df = generate_survey_did_data(
                n_units=500,
                n_periods=4,
                add_covariates=True,
                conditional_pt=0.3,
                panel=panel_mode,
                seed=42,
            )
            # Basic sanity: data is produced
            assert len(df) == 500 * 4
            assert "x1" in df.columns
            # Check x1 shift exists in period 1
            p1 = df[df["period"] == 1]
            x1_treated = p1.loc[p1["first_treat"] > 0, "x1"].mean()
            x1_control = p1.loc[p1["first_treat"] == 0, "x1"].mean()
            assert x1_treated > x1_control, f"panel={panel_mode}: treated x1 not shifted"


class TestAggregateSurvey:
    """Tests for aggregate_survey function."""

    @pytest.fixture
    def micro_data(self):
        """Create simple microdata: 2 states, 2 years, with survey design."""
        rng = np.random.RandomState(42)
        n = 400
        states = np.repeat(["CA", "TX"], n // 2)
        years = np.tile(np.repeat([2019, 2020], n // 4), 2)
        strata = np.repeat(np.arange(4), n // 4)
        psu = np.arange(n) // 10  # 10 obs per PSU, 40 PSUs total
        weights = rng.uniform(0.5, 3.0, n)
        outcome = rng.normal(10, 2, n)
        # Make CA slightly higher than TX to get different means
        outcome[: n // 2] += 2.0
        covariate = rng.normal(50, 10, n)

        return pd.DataFrame(
            {
                "state": states,
                "year": years,
                "stratum": strata,
                "cluster": psu,
                "wt": weights,
                "y": outcome,
                "x": covariate,
            }
        )

    @pytest.fixture
    def design(self):
        return SurveyDesign(weights="wt", strata="stratum", psu="cluster")

    def test_basic_aggregation(self, micro_data, design):
        """Design-weighted means should differ from simple means."""
        panel, _ = aggregate_survey(
            micro_data,
            by=["state", "year"],
            outcomes="y",
            survey_design=design,
        )
        assert len(panel) == 4  # 2 states x 2 years

        # Check design-weighted mean differs from simple mean
        simple_mean = micro_data[micro_data["state"] == "CA"][micro_data["year"] == 2019][
            "y"
        ].mean()
        ca_2019 = panel[(panel["state"] == "CA") & (panel["year"] == 2019)]
        weighted_mean = ca_2019["y_mean"].iloc[0]
        # With non-uniform weights, these should differ
        assert weighted_mean != pytest.approx(simple_mean, abs=0.01)

    def test_column_naming(self, micro_data, design):
        """All expected columns should be present."""
        panel, _ = aggregate_survey(
            micro_data,
            by=["state", "year"],
            outcomes="y",
            covariates="x",
            survey_design=design,
        )
        expected = {
            "state",
            "year",
            "y_mean",
            "y_se",
            "y_n",
            "y_precision",
            "x_mean",
            "cell_n",
            "cell_n_eff",
            "cell_sum_w",
            "srs_fallback",
        }
        assert expected.issubset(set(panel.columns))

    def test_multiple_outcomes(self, micro_data, design):
        """Each outcome gets own columns; SurveyDesign uses first."""
        micro_data = micro_data.copy()
        micro_data["y2"] = micro_data["y"] * 2
        panel, stage2 = aggregate_survey(
            micro_data,
            by=["state", "year"],
            outcomes=["y", "y2"],
            survey_design=design,
        )
        assert "y_mean" in panel.columns
        assert "y2_mean" in panel.columns
        assert "y_precision" in panel.columns
        assert "y2_precision" in panel.columns
        assert stage2.weights == "y_weight"

    def test_multi_outcome_filtering_contract(self):
        """Multi-outcome filtering is based on first outcome; warns about secondary data loss."""
        rng = np.random.RandomState(33)
        data = pd.DataFrame(
            {
                "geo": np.repeat(["A", "B"], 20),
                "time": np.ones(40, dtype=int),
                "wt": np.ones(40),
                "y1": np.concatenate(
                    [[np.nan] * 20, rng.normal(10, 2, 20)]
                ),  # A: all-NaN, B: valid
                "y2": rng.normal(5, 1, 40),  # valid everywhere
            }
        )
        design = SurveyDesign(weights="wt")
        with pytest.warns(UserWarning, match="y2.*valid data in dropped"):
            panel, _ = aggregate_survey(
                data,
                by=["geo", "time"],
                outcomes=["y1", "y2"],
                survey_design=design,
            )
        # Cell A dropped (y1 non-estimable), even though y2 was valid
        assert len(panel) == 1
        assert panel["geo"].iloc[0] == "B"

    def test_covariates_mean_only(self, micro_data, design):
        """Covariates get mean column only, no SE/precision."""
        panel, _ = aggregate_survey(
            micro_data,
            by=["state", "year"],
            outcomes="y",
            covariates="x",
            survey_design=design,
        )
        assert "x_mean" in panel.columns
        assert "x_se" not in panel.columns
        assert "x_precision" not in panel.columns

    def test_returned_survey_design(self, micro_data, design):
        """Default returns pweight config with geographic clustering."""
        _, stage2 = aggregate_survey(
            micro_data,
            by=["state", "year"],
            outcomes="y",
            survey_design=design,
        )
        assert stage2.weight_type == "pweight"
        assert stage2.weights == "y_weight"
        assert stage2.psu == "state"

    def test_srs_fallback(self):
        """Cells where design-based variance fails get SRS fallback."""
        # Create data where each cell has only 1 PSU per stratum
        rng = np.random.RandomState(99)
        n = 40
        data = pd.DataFrame(
            {
                "geo": np.repeat(["A", "B"], n // 2),
                "time": np.tile(np.repeat([0, 1], n // 4), 2),
                "stratum": np.arange(n),  # every obs is its own stratum
                "psu": np.arange(n),  # every obs is its own PSU
                "wt": np.ones(n),
                "y": rng.normal(0, 1, n),
            }
        )
        design = SurveyDesign(weights="wt", strata="stratum", psu="psu", lonely_psu="remove")
        with pytest.warns(UserWarning, match="SRS fallback"):
            panel, _ = aggregate_survey(
                data,
                by=["geo", "time"],
                outcomes="y",
                survey_design=design,
            )
        assert panel["srs_fallback"].all()
        # SRS SE should be finite and positive
        assert (panel["y_se"] > 0).all()
        assert panel["y_se"].notna().all()

        # Verify SRS SE matches manual computation for one cell
        cell = data[(data["geo"] == "A") & (data["time"] == 0)]
        y_vals = cell["y"].values
        n_cell = len(y_vals)
        expected_var = np.var(y_vals, ddof=0) / n_cell * n_cell / (n_cell - 1)
        expected_se = np.sqrt(expected_var)
        actual_se = panel[(panel["geo"] == "A") & (panel["time"] == 0)]["y_se"].iloc[0]
        assert actual_se == pytest.approx(expected_se, rel=1e-10)

    def test_missing_values(self, micro_data, design):
        """Missing values reduce var_n but cell_n stays the same."""
        micro_data = micro_data.copy()
        # Set some values to NaN in CA-2019
        mask = (micro_data["state"] == "CA") & (micro_data["year"] == 2019)
        idx = micro_data[mask].index[:5]
        micro_data.loc[idx, "y"] = np.nan

        panel, _ = aggregate_survey(
            micro_data,
            by=["state", "year"],
            outcomes="y",
            survey_design=design,
        )
        ca_2019 = panel[(panel["state"] == "CA") & (panel["year"] == 2019)]
        assert ca_2019["y_n"].iloc[0] == 100 - 5  # 5 NaN
        assert ca_2019["cell_n"].iloc[0] == 100  # all respondents

    def test_zero_variance_cell(self):
        """When all values are identical, precision is NaN."""
        data = pd.DataFrame(
            {
                "geo": np.repeat(["A", "B"], 20),
                "time": np.tile(np.repeat([0, 1], 10), 2),
                "wt": np.ones(40),
                "y": np.concatenate(
                    [
                        np.full(10, 5.0),  # A-0: constant
                        np.random.RandomState(1).normal(5, 1, 10),  # A-1
                        np.random.RandomState(2).normal(5, 1, 10),  # B-0
                        np.random.RandomState(3).normal(5, 1, 10),  # B-1
                    ]
                ),
            }
        )
        design = SurveyDesign(weights="wt")
        with pytest.warns(UserWarning, match="Zero variance"):
            panel, _ = aggregate_survey(
                data,
                by=["geo", "time"],
                outcomes="y",
                survey_design=design,
            )
        a0 = panel[(panel["geo"] == "A") & (panel["time"] == 0)]
        assert a0["y_mean"].iloc[0] == pytest.approx(5.0)
        assert a0["y_se"].iloc[0] == pytest.approx(0.0)
        assert np.isnan(a0["y_precision"].iloc[0])

    def test_lonely_psu_override(self):
        """lonely_psu parameter overrides survey_design setting."""
        rng = np.random.RandomState(77)
        n = 40
        data = pd.DataFrame(
            {
                "geo": np.repeat(["A", "B"], n // 2),
                "time": np.tile(np.repeat([0, 1], n // 4), 2),
                "stratum": np.repeat(np.arange(4), n // 4),
                "psu": np.arange(n) // 5,
                "wt": np.ones(n),
                "y": rng.normal(0, 1, n),
            }
        )
        design = SurveyDesign(weights="wt", strata="stratum", psu="psu", lonely_psu="remove")
        # Override to "certainty" — different behavior for singletons
        panel_cert, _ = aggregate_survey(
            data,
            by=["geo", "time"],
            outcomes="y",
            survey_design=design,
            lonely_psu="certainty",
        )
        panel_remove, _ = aggregate_survey(
            data,
            by=["geo", "time"],
            outcomes="y",
            survey_design=design,
        )
        # Results should exist for both
        assert len(panel_cert) == 4
        assert len(panel_remove) == 4

    def test_single_by_column(self, micro_data, design):
        """Single string for by works correctly."""
        panel, stage2 = aggregate_survey(
            micro_data,
            by="state",
            outcomes="y",
            survey_design=design,
        )
        assert len(panel) == 2  # CA, TX
        assert "state" in panel.columns
        assert stage2.psu == "state"

    def test_srs_equivalence_weights_only(self):
        """With no strata/PSU, SE matches weighted SRS formula."""
        rng = np.random.RandomState(123)
        n = 100
        data = pd.DataFrame(
            {
                "geo": np.repeat(["A", "B"], n // 2),
                "time": np.ones(n, dtype=int),
                "wt": rng.uniform(0.5, 2.0, n),
                "y": rng.normal(10, 2, n),
            }
        )
        design = SurveyDesign(weights="wt")
        panel, _ = aggregate_survey(
            data,
            by=["geo", "time"],
            outcomes="y",
            survey_design=design,
        )

        # Manual full-design domain estimation for cell A:
        # psi is zero-padded to n_total; adjustment uses n_total/(n_total-1)
        w_all = data["wt"].values
        w_all_norm = w_all / w_all.mean()  # resolve() normalizes to mean=1
        cell_mask = (data["geo"] == "A").values
        y_all = data["y"].values
        w_cell = w_all_norm[cell_mask]
        y_cell = y_all[cell_mask]
        sum_w = np.sum(w_cell)
        y_bar = np.sum(w_cell * y_cell) / sum_w

        # Full-length psi with zeros outside cell
        psi_full = np.zeros(n)
        psi_full[cell_mask] = w_cell * (y_cell - y_bar) / sum_w

        # Implicit per-obs PSU with full-design adjustment
        psi_mean = psi_full.mean()
        centered = psi_full - psi_mean
        variance = (n / (n - 1)) * np.sum(centered**2)
        expected_se = np.sqrt(variance)

        actual_se = panel[panel["geo"] == "A"]["y_se"].iloc[0]
        assert actual_se == pytest.approx(expected_se, rel=1e-10)

    def test_design_effect_increases_se(self):
        """With PSU clustering, SE should be larger than without."""
        rng = np.random.RandomState(55)
        n = 200
        psu_ids = np.arange(n) // 10  # 10 obs per PSU
        # Add PSU-level random effects to create ICC
        psu_effects = rng.normal(0, 3, 20)
        y = rng.normal(0, 1, n) + psu_effects[psu_ids]

        data = pd.DataFrame(
            {
                "geo": ["A"] * n,
                "time": np.ones(n, dtype=int),
                "cluster": psu_ids,
                "wt": np.ones(n),
                "y": y,
            }
        )

        design_no_psu = SurveyDesign(weights="wt")
        design_psu = SurveyDesign(weights="wt", psu="cluster")

        panel_no_psu, _ = aggregate_survey(
            data,
            by=["geo", "time"],
            outcomes="y",
            survey_design=design_no_psu,
        )
        panel_psu, _ = aggregate_survey(
            data,
            by=["geo", "time"],
            outcomes="y",
            survey_design=design_psu,
        )

        se_no_psu = panel_no_psu["y_se"].iloc[0]
        se_psu = panel_psu["y_se"].iloc[0]
        assert se_psu > se_no_psu  # clustering increases SE

    def test_equal_weights_simple_mean(self):
        """With equal weights, design-weighted mean equals arithmetic mean."""
        rng = np.random.RandomState(88)
        n = 60
        data = pd.DataFrame(
            {
                "geo": np.repeat(["A", "B"], n // 2),
                "time": np.tile(np.repeat([0, 1], n // 4), 2),
                "wt": np.ones(n),
                "y": rng.normal(10, 2, n),
            }
        )
        design = SurveyDesign(weights="wt")
        panel, _ = aggregate_survey(
            data,
            by=["geo", "time"],
            outcomes="y",
            survey_design=design,
        )
        # Check A-0 cell
        cell = data[(data["geo"] == "A") & (data["time"] == 0)]
        assert panel[(panel["geo"] == "A") & (panel["time"] == 0)]["y_mean"].iloc[0] == (
            pytest.approx(cell["y"].mean(), rel=1e-12)
        )

    def test_pipeline_with_did(self):
        """Full pipeline: microdata → aggregate → DiD estimation."""
        from diff_diff import DifferenceInDifferences

        # Construct microdata simulating 4 states, 2 periods, ~50 obs/cell
        rng = np.random.RandomState(42)
        rows = []
        for state in range(4):
            treated = 1 if state < 2 else 0
            for period in [0, 1]:
                n_cell = rng.randint(40, 60)
                # Treatment effect in post period for treated states
                te = 3.0 if (treated and period == 1) else 0.0
                for _ in range(n_cell):
                    strat = rng.randint(0, 3)
                    psu_id = state * 100 + strat * 10 + rng.randint(0, 3)
                    rows.append(
                        {
                            "state": state,
                            "period": period,
                            "stratum": strat,
                            "psu": psu_id,
                            "wt": rng.uniform(0.5, 2.0),
                            "outcome": rng.normal(10 + te, 2),
                            "treated": treated,
                        }
                    )
        micro = pd.DataFrame(rows)

        design = SurveyDesign(weights="wt", strata="stratum", psu="psu")
        panel, stage2 = aggregate_survey(
            micro,
            by=["state", "period"],
            outcomes="outcome",
            covariates="treated",
            survey_design=design,
        )

        panel["treated_bin"] = (panel["treated_mean"] > 0.5).astype(int)

        did = DifferenceInDifferences()
        result = did.fit(
            panel,
            outcome="outcome_mean",
            treatment="treated_bin",
            post="period",
            survey_design=stage2,
        )
        assert result.att is not None
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
        # ATT should be near the true effect of 3.0
        assert abs(result.att - 3.0) < 2.0

    # --- Error tests ---

    def test_error_missing_column(self, micro_data, design):
        """Missing column raises ValueError."""
        with pytest.raises(ValueError, match="Columns not found"):
            aggregate_survey(
                micro_data,
                by=["state", "year"],
                outcomes="nonexistent",
                survey_design=design,
            )

    def test_error_invalid_survey_design(self, micro_data):
        """Non-SurveyDesign object raises TypeError."""
        with pytest.raises(TypeError, match="SurveyDesign instance"):
            aggregate_survey(
                micro_data,
                by=["state", "year"],
                outcomes="y",
                survey_design="not_a_design",
            )

    def test_error_min_n_too_small(self, micro_data, design):
        """min_n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_n must be >= 1"):
            aggregate_survey(
                micro_data,
                by=["state", "year"],
                outcomes="y",
                survey_design=design,
                min_n=0,
            )

    def test_error_empty_by(self, micro_data, design):
        """Empty by list raises ValueError."""
        with pytest.raises(ValueError, match="at least one grouping column"):
            aggregate_survey(micro_data, by=[], outcomes="y", survey_design=design)

    def test_error_empty_outcomes(self, micro_data, design):
        """Empty outcomes list raises ValueError."""
        with pytest.raises(ValueError, match="at least one outcome"):
            aggregate_survey(
                micro_data,
                by=["state", "year"],
                outcomes=[],
                survey_design=design,
            )

    def test_error_non_numeric_outcome(self, micro_data, design):
        """Non-numeric outcome column raises ValueError."""
        data = micro_data.copy()
        data["label"] = "foo"
        with pytest.raises(ValueError, match="Non-numeric column"):
            aggregate_survey(
                data,
                by=["state", "year"],
                outcomes="label",
                survey_design=design,
            )

    def test_nullable_numeric_dtypes(self):
        """Pandas nullable Int64/Float64 dtypes are accepted as numeric."""
        data = pd.DataFrame(
            {
                "geo": np.repeat(["A", "B"], 10),
                "time": np.ones(20, dtype=int),
                "wt": np.ones(20),
                "y": pd.array(np.random.RandomState(1).normal(0, 1, 20), dtype="Float64"),
            }
        )
        design = SurveyDesign(weights="wt")
        panel, _ = aggregate_survey(data, by=["geo", "time"], outcomes="y", survey_design=design)
        assert len(panel) == 2
        assert panel["y_mean"].notna().all()

    def test_error_empty_data(self, design):
        """Empty DataFrame raises ValueError."""
        empty = pd.DataFrame(columns=["state", "year", "y", "wt", "stratum", "cluster"])
        with pytest.raises(ValueError, match="data must be non-empty"):
            aggregate_survey(
                empty,
                by=["state", "year"],
                outcomes="y",
                survey_design=design,
            )

    def test_error_missing_grouping_keys(self, micro_data, design):
        """NaN in grouping columns raises ValueError."""
        data = micro_data.copy()
        data.loc[0, "state"] = np.nan
        with pytest.raises(ValueError, match="Missing values in grouping column"):
            aggregate_survey(
                data,
                by=["state", "year"],
                outcomes="y",
                survey_design=design,
            )

    def test_error_all_missing_grouping_keys(self, design):
        """All-NaN grouping column raises ValueError."""
        data = pd.DataFrame(
            {
                "state": [np.nan] * 10,
                "year": np.ones(10, dtype=int),
                "y": np.random.RandomState(1).normal(0, 1, 10),
                "wt": np.ones(10),
            }
        )
        design_simple = SurveyDesign(weights="wt")
        with pytest.raises(ValueError, match="Missing values in grouping column"):
            aggregate_survey(
                data,
                by=["state", "year"],
                outcomes="y",
                survey_design=design_simple,
            )

    def test_stage2_handoff_with_nonfinite_cells(self):
        """Non-estimable cells are dropped; stage2 works with fit()."""
        from diff_diff import DifferenceInDifferences

        rng = np.random.RandomState(99)
        rows = []
        for state in range(4):
            treated = 1 if state < 2 else 0
            for period in [0, 1]:
                te = 3.0 if (treated and period == 1) else 0.0
                n_cell = 30
                for _ in range(n_cell):
                    rows.append(
                        {
                            "state": state,
                            "period": period,
                            "wt": rng.uniform(0.5, 2.0),
                            "outcome": rng.normal(10 + te, 2),
                            "treated": treated,
                        }
                    )
        micro = pd.DataFrame(rows)
        # Make one cell all-NaN outcome → n_valid=0 → NaN mean → dropped
        mask = (micro["state"] == 0) & (micro["period"] == 0)
        micro.loc[mask, "outcome"] = np.nan

        design = SurveyDesign(weights="wt")
        with pytest.warns(UserWarning, match="non-estimable"):
            panel, stage2 = aggregate_survey(
                micro,
                by=["state", "period"],
                outcomes="outcome",
                covariates="treated",
                survey_design=design,
            )

        # Non-estimable cell should be dropped from panel
        assert len(panel) == 7  # 8 cells - 1 dropped
        assert not ((panel["state"] == 0) & (panel["period"] == 0)).any()

        # No NaN in outcome mean or weight columns
        assert panel["outcome_mean"].notna().all()
        assert panel["outcome_weight"].notna().all()

        # stage2 should work with fit()
        panel["treated_bin"] = (panel["treated_mean"] > 0.5).astype(int)
        did = DifferenceInDifferences()
        result = did.fit(
            panel,
            outcome="outcome_mean",
            treatment="treated_bin",
            post="period",
            survey_design=stage2,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)

    def test_zero_weight_psu_dropped(self):
        """Geographic units with zero total weight are dropped from panel."""
        rng = np.random.RandomState(88)
        # State 0: all cells have only 1 valid obs → NaN precision → weight=0
        # State 1-3: normal cells
        rows = []
        for state in range(4):
            for period in [0, 1]:
                if state == 0:
                    # 1 obs per cell → NaN SE → weight=0
                    rows.append(
                        {
                            "state": state,
                            "period": period,
                            "wt": 1.0,
                            "y": rng.normal(10, 2),
                        }
                    )
                else:
                    for _ in range(20):
                        rows.append(
                            {
                                "state": state,
                                "period": period,
                                "wt": 1.0,
                                "y": rng.normal(10, 2),
                            }
                        )
        data = pd.DataFrame(rows)
        design = SurveyDesign(weights="wt")
        with pytest.warns(UserWarning, match="zero total weight"):
            panel, _ = aggregate_survey(
                data,
                by=["state", "period"],
                outcomes="y",
                survey_design=design,
                second_stage_weights="aweight",
            )
        # State 0 should be entirely gone
        assert 0 not in panel["state"].values
        assert len(panel) == 6  # 3 states × 2 periods

    def test_error_all_cells_dropped(self):
        """All cells non-estimable raises ValueError."""
        data = pd.DataFrame(
            {
                "state": ["A"] * 5,
                "period": np.ones(5, dtype=int),
                "wt": np.ones(5),
                "y": [np.nan] * 5,
            }
        )
        design = SurveyDesign(weights="wt")
        with pytest.raises(ValueError, match="No estimable cells remain"):
            aggregate_survey(data, by=["state", "period"], outcomes="y", survey_design=design)

    def test_zero_weight_rows_excluded_from_n_valid(self):
        """Zero-weight rows should not count as valid observations."""
        rng = np.random.RandomState(66)
        # Cell A: 3 positive-weight obs + 7 zero-weight padding
        # n_valid should be 3, not 10
        data = pd.DataFrame(
            {
                "geo": ["A"] * 10 + ["B"] * 10,
                "time": np.ones(20, dtype=int),
                "wt": np.concatenate(
                    [
                        np.array([1.0, 1.0, 1.0] + [0.0] * 7),  # A: 3 real
                        np.ones(10),  # B: all real
                    ]
                ),
                "y": rng.normal(10, 2, 20),
            }
        )
        design = SurveyDesign(weights="wt")
        panel, _ = aggregate_survey(data, by=["geo", "time"], outcomes="y", survey_design=design)
        cell_a = panel[panel["geo"] == "A"]
        # Only 3 positive-weight obs → n_valid=3
        assert cell_a["y_n"].iloc[0] == 3

        cell_b = panel[panel["geo"] == "B"]
        # 10 positive-weight obs
        assert cell_b["y_n"].iloc[0] == 10
        assert cell_b["y_se"].iloc[0] > 0

    def test_duplicate_index(self):
        """Duplicate DataFrame indices do not break aggregation."""
        rng = np.random.RandomState(77)
        n = 40
        data = pd.DataFrame(
            {
                "geo": np.repeat(["A", "B"], n // 2),
                "time": np.tile(np.repeat([0, 1], n // 4), 2),
                "wt": np.ones(n),
                "y": rng.normal(10, 2, n),
            }
        )
        # Create duplicate indices (e.g., from concat without reset_index)
        data.index = list(range(n // 2)) * 2  # 0..19, 0..19

        design = SurveyDesign(weights="wt")
        panel_dup, _ = aggregate_survey(
            data, by=["geo", "time"], outcomes="y", survey_design=design
        )

        # Compare against clean-index version
        data_clean = data.reset_index(drop=True)
        panel_clean, _ = aggregate_survey(
            data_clean, by=["geo", "time"], outcomes="y", survey_design=design
        )

        # Results should be identical
        np.testing.assert_allclose(
            panel_dup["y_mean"].values, panel_clean["y_mean"].values, rtol=1e-12
        )
        np.testing.assert_allclose(panel_dup["y_se"].values, panel_clean["y_se"].values, rtol=1e-12)

    def test_domain_estimation_preserves_full_design(self):
        """Full-design domain estimation accounts for PSUs outside the cell.

        Stratum 0 has PSUs {0, 1}. Cell A contains only PSU 0.
        With physical subsetting, stratum 0 would be a singleton → skipped.
        With full-design domain estimation, both PSUs participate → non-zero
        stratum variance contribution and no SRS fallback.
        """
        rng = np.random.RandomState(42)
        # 2 strata, 2 PSUs each, 5 obs per PSU = 20 obs total
        # Cell A = first 10 obs (stratum 0 PSU 0 + stratum 1 PSU 2)
        # Cell B = last 10 obs (stratum 0 PSU 1 + stratum 1 PSU 3)
        data = pd.DataFrame(
            {
                "geo": ["A"] * 10 + ["B"] * 10,
                "time": np.ones(20, dtype=int),
                "stratum": np.repeat([0, 0, 1, 1], 5),
                "psu": np.repeat([0, 1, 2, 3], 5),
                "wt": np.ones(20),
                "y": rng.normal(10, 2, 20),
            }
        )
        # Reassign so cell A has only PSU 0 from stratum 0, cell B has only PSU 1
        data.loc[:4, "geo"] = "A"  # stratum 0, PSU 0
        data.loc[5:9, "geo"] = "B"  # stratum 0, PSU 1
        data.loc[10:14, "geo"] = "A"  # stratum 1, PSU 2
        data.loc[15:19, "geo"] = "B"  # stratum 1, PSU 3

        design = SurveyDesign(weights="wt", strata="stratum", psu="psu", lonely_psu="remove")
        panel, _ = aggregate_survey(
            data,
            by=["geo", "time"],
            outcomes="y",
            survey_design=design,
        )

        cell_a = panel[panel["geo"] == "A"]
        # With full-design domain estimation:
        # - Both PSUs in each stratum participate (one with zero psi)
        # - No singleton PSU → no SRS fallback needed
        assert not cell_a["srs_fallback"].iloc[0]
        assert cell_a["y_se"].iloc[0] > 0
        assert np.isfinite(cell_a["y_se"].iloc[0])
        assert np.isfinite(cell_a["y_precision"].iloc[0])

    def test_min_n_forces_srs_fallback(self):
        """min_n parameter forces SRS fallback for small cells."""
        rng = np.random.RandomState(44)
        n = 40
        data = pd.DataFrame(
            {
                "geo": np.repeat(["A", "B"], n // 2),
                "time": np.ones(n, dtype=int),
                "wt": np.ones(n),
                "y": rng.normal(10, 2, n),
            }
        )
        design = SurveyDesign(weights="wt")

        # min_n=30 → cells with 20 obs each should use SRS fallback
        with pytest.warns(UserWarning, match="SRS fallback"):
            panel_high, _ = aggregate_survey(
                data,
                by=["geo", "time"],
                outcomes="y",
                survey_design=design,
                min_n=30,
            )
        assert panel_high["srs_fallback"].all()

        # min_n=1 → should use design-based variance (no fallback)
        panel_low, _ = aggregate_survey(
            data,
            by=["geo", "time"],
            outcomes="y",
            survey_design=design,
            min_n=1,
        )
        assert not panel_low["srs_fallback"].any()

        # SEs should differ between the two
        se_high = panel_high[panel_high["geo"] == "A"]["y_se"].iloc[0]
        se_low = panel_low[panel_low["geo"] == "A"]["y_se"].iloc[0]
        assert se_high != pytest.approx(se_low, rel=1e-6)

    def test_replicate_weight_aggregation(self):
        """Aggregation with replicate weight designs produces valid SEs."""
        from diff_diff.prep_dgp import generate_survey_did_data

        micro = generate_survey_did_data(
            n_units=200,
            n_periods=4,
            cohort_periods=[3],
            n_strata=3,
            psu_per_stratum=6,
            include_replicate_weights=True,
            panel=False,
            seed=42,
        )
        # Build replicate weight column list
        rep_cols = [c for c in micro.columns if c.startswith("rep_")]
        design = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        panel, _ = aggregate_survey(
            micro,
            by=["stratum", "period"],
            outcomes="outcome",
            survey_design=design,
        )
        # All cells should have finite, positive SEs
        assert panel["outcome_se"].notna().all()
        assert (panel["outcome_se"] > 0).all()

    def test_replicate_weight_min_n_fallback(self):
        """SRS fallback works correctly under replicate-weight designs."""
        from diff_diff.prep_dgp import generate_survey_did_data

        micro = generate_survey_did_data(
            n_units=200,
            n_periods=4,
            cohort_periods=[3],
            n_strata=3,
            psu_per_stratum=6,
            include_replicate_weights=True,
            panel=False,
            seed=42,
        )
        rep_cols = [c for c in micro.columns if c.startswith("rep_")]
        design = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
        )

        # min_n high enough to force SRS fallback on all cells
        with pytest.warns(UserWarning, match="SRS fallback"):
            panel_srs, _ = aggregate_survey(
                micro,
                by=["stratum", "period"],
                outcomes="outcome",
                survey_design=design,
                min_n=9999,
            )
        assert panel_srs["srs_fallback"].all()
        assert panel_srs["outcome_se"].notna().all()
        assert (panel_srs["outcome_se"] > 0).all()

        # Default min_n → replicate-based variance (no fallback)
        panel_rep, _ = aggregate_survey(
            micro,
            by=["stratum", "period"],
            outcomes="outcome",
            survey_design=design,
        )
        assert not panel_rep["srs_fallback"].any()

        # SEs should differ between SRS fallback and replicate-based
        assert not np.allclose(
            panel_srs["outcome_se"].values,
            panel_rep["outcome_se"].values,
            rtol=1e-6,
        )

    def test_srs_fallback_scale_invariant(self):
        """SRS fallback SEs are invariant to constant weight rescaling."""
        rng = np.random.RandomState(55)
        n = 60
        data = pd.DataFrame(
            {
                "geo": np.repeat(["A", "B", "C"], n // 3),
                "time": np.ones(n, dtype=int),
                "wt": rng.uniform(0.5, 2.0, n),
                "y": rng.normal(10, 2, n),
            }
        )
        design1 = SurveyDesign(weights="wt")

        # Force SRS fallback with high min_n
        with pytest.warns(UserWarning, match="SRS fallback"):
            panel1, _ = aggregate_survey(
                data,
                by=["geo", "time"],
                outcomes="y",
                survey_design=design1,
                min_n=9999,
            )

        # Rescale weights by 5x → should give identical SEs
        data2 = data.copy()
        data2["wt"] = data2["wt"] * 5.0
        design2 = SurveyDesign(weights="wt")
        with pytest.warns(UserWarning, match="SRS fallback"):
            panel2, _ = aggregate_survey(
                data2,
                by=["geo", "time"],
                outcomes="y",
                survey_design=design2,
                min_n=9999,
            )

        np.testing.assert_allclose(panel1["y_se"].values, panel2["y_se"].values, rtol=1e-10)
        np.testing.assert_allclose(panel1["y_mean"].values, panel2["y_mean"].values, rtol=1e-10)

    def test_second_stage_weights_aweight(self, micro_data, design):
        """Opt-in aweight preserves precision-based weight behavior."""
        panel, stage2 = aggregate_survey(
            micro_data,
            by=["state", "year"],
            outcomes="y",
            survey_design=design,
            second_stage_weights="aweight",
        )
        assert stage2.weight_type == "aweight"
        assert stage2.weights == "y_weight"
        assert stage2.psu == "state"
        # Weight column should match cleaned precision (NaN/Inf -> 0.0)
        precision = panel["y_precision"].values
        expected_weight = np.where(np.isfinite(precision), precision, 0.0)
        np.testing.assert_array_equal(panel["y_weight"].values, expected_weight)

    def test_pweight_values(self, micro_data, design):
        """Pweight values are unit-constant: mean of cell_sum_w within geo unit."""
        panel, _ = aggregate_survey(
            micro_data,
            by=["state", "year"],
            outcomes="y",
            survey_design=design,
        )
        # cell_sum_w should match raw per-cell survey weight sums
        resolved = design.resolve(micro_data)
        for _, row in panel.iterrows():
            mask = (micro_data["state"] == row["state"]) & (micro_data["year"] == row["year"])
            expected_sum_w = float(np.sum(resolved.weights[mask.values]))
            assert row["cell_sum_w"] == pytest.approx(expected_sum_w, rel=1e-10)

        # y_weight should be the mean of cell_sum_w within each state (unit-constant)
        for state in panel["state"].unique():
            state_rows = panel[panel["state"] == state]
            expected_weight = state_rows["cell_sum_w"].mean()
            np.testing.assert_allclose(state_rows["y_weight"].values, expected_weight, rtol=1e-10)
            # Must be constant within unit
            assert state_rows["y_weight"].nunique() == 1

    def test_cell_sum_w_column(self, micro_data, design):
        """cell_sum_w is present in both pweight and aweight modes."""
        panel_p, _ = aggregate_survey(
            micro_data,
            by=["state", "year"],
            outcomes="y",
            survey_design=design,
            second_stage_weights="pweight",
        )
        panel_a, _ = aggregate_survey(
            micro_data,
            by=["state", "year"],
            outcomes="y",
            survey_design=design,
            second_stage_weights="aweight",
        )
        assert "cell_sum_w" in panel_p.columns
        assert "cell_sum_w" in panel_a.columns
        # cell_sum_w should be identical regardless of weight mode
        np.testing.assert_array_equal(panel_p["cell_sum_w"].values, panel_a["cell_sum_w"].values)

    def test_pweight_zero_variance_cell(self):
        """Zero-variance cells retain positive pweight but get NaN aweight."""
        data = pd.DataFrame(
            {
                "geo": np.repeat(["A", "B"], 20),
                "time": np.tile(np.repeat([0, 1], 10), 2),
                "wt": np.ones(40),
                "y": np.concatenate(
                    [
                        np.full(10, 5.0),  # A-0: constant → zero variance
                        np.random.RandomState(1).normal(5, 1, 10),  # A-1
                        np.random.RandomState(2).normal(5, 1, 10),  # B-0
                        np.random.RandomState(3).normal(5, 1, 10),  # B-1
                    ]
                ),
            }
        )
        design = SurveyDesign(weights="wt")

        # Pweight mode: zero-variance cell retains positive weight
        with pytest.warns(UserWarning, match="Zero variance"):
            panel_p, stage2_p = aggregate_survey(
                data,
                by=["geo", "time"],
                outcomes="y",
                survey_design=design,
                second_stage_weights="pweight",
            )
        a0_p = panel_p[(panel_p["geo"] == "A") & (panel_p["time"] == 0)]
        assert a0_p["y_weight"].iloc[0] > 0
        assert np.isnan(a0_p["y_precision"].iloc[0])
        assert stage2_p.weight_type == "pweight"

        # Aweight mode: zero-variance cell gets weight 0.0
        with pytest.warns(UserWarning, match="Zero variance"):
            panel_a, stage2_a = aggregate_survey(
                data,
                by=["geo", "time"],
                outcomes="y",
                survey_design=design,
                second_stage_weights="aweight",
            )
        a0_a = panel_a[(panel_a["geo"] == "A") & (panel_a["time"] == 0)]
        assert a0_a["y_weight"].iloc[0] == 0.0
        assert np.isnan(a0_a["y_precision"].iloc[0])
        assert stage2_a.weight_type == "aweight"

    def test_invalid_second_stage_weights(self, micro_data, design):
        """Invalid second_stage_weights raises ValueError."""
        with pytest.raises(ValueError, match="second_stage_weights"):
            aggregate_survey(
                micro_data,
                by=["state", "year"],
                outcomes="y",
                survey_design=design,
                second_stage_weights="invalid",
            )

    def test_pweight_callaway_santanna_integration(self):
        """Pweight geo-period workflow feeds into CallawaySantAnna.

        Builds a repeated cross-section with multiple respondents per
        geo-period cell so that cell_sum_w varies across periods within
        geographic units. The unit-constant averaging in pweight mode
        must satisfy _validate_unit_constant_survey().
        """
        from diff_diff import CallawaySantAnna

        rng = np.random.RandomState(42)
        # 6 geographic units, 4 periods, ~20 respondents per cell
        rows = []
        first_treats = {0: 0, 1: 0, 2: 3, 3: 3, 4: 4, 5: 4}
        for geo in range(6):
            for period in range(1, 5):
                n_resp = rng.randint(15, 25)  # varying respondents per cell
                te = 2.0 if (first_treats[geo] > 0 and period >= first_treats[geo]) else 0.0
                for _ in range(n_resp):
                    rows.append(
                        {
                            "geo": geo,
                            "period": period,
                            "wt": rng.uniform(0.5, 3.0),
                            "y": rng.normal(10 + te, 2),
                        }
                    )
        micro = pd.DataFrame(rows)
        design = SurveyDesign(weights="wt")
        panel, stage2 = aggregate_survey(
            micro,
            by=["geo", "period"],
            outcomes="y",
            survey_design=design,
        )
        assert stage2.weight_type == "pweight"

        # cell_sum_w should vary within geos (different respondent counts)
        # but y_weight must be unit-constant
        for geo in panel["geo"].unique():
            geo_rows = panel[panel["geo"] == geo]
            assert (
                geo_rows["y_weight"].nunique() == 1
            ), f"Geo {geo}: y_weight not constant within unit"

        panel["first_treat"] = panel["geo"].map(first_treats)
        result = CallawaySantAnna().fit(
            panel,
            outcome="y_mean",
            unit="geo",
            time="period",
            first_treat="first_treat",
            survey_design=stage2,
        )
        assert np.isfinite(result.overall_att), f"ATT not finite: {result.overall_att}"
        assert result.overall_se > 0, f"SE not positive: {result.overall_se}"

    def test_pweight_replicate_weight_design(self):
        """Pweight mode works correctly under replicate-weight survey designs."""
        from diff_diff.prep_dgp import generate_survey_did_data

        micro = generate_survey_did_data(
            n_units=200,
            n_periods=4,
            cohort_periods=[3],
            n_strata=3,
            psu_per_stratum=6,
            include_replicate_weights=True,
            panel=False,
            seed=42,
        )
        rep_cols = [c for c in micro.columns if c.startswith("rep_")]
        design = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        panel, stage2 = aggregate_survey(
            micro,
            by=["stratum", "period"],
            outcomes="outcome",
            survey_design=design,
            second_stage_weights="pweight",
        )
        assert stage2.weight_type == "pweight"
        assert "cell_sum_w" in panel.columns
        assert (panel["cell_sum_w"] > 0).all()
        # Weight column is unit-constant (mean of cell_sum_w per geo unit)
        for geo in panel["stratum"].unique():
            geo_rows = panel[panel["stratum"] == geo]
            assert geo_rows["outcome_weight"].nunique() == 1
        # All cells should have finite SEs from replicate variance
        assert panel["outcome_se"].notna().all()
        assert (panel["outcome_se"] > 0).all()

    def test_pweight_retains_zero_precision_geo(self):
        """Under pweight, a geo with NaN precision is retained (not pruned)."""
        rng = np.random.RandomState(88)
        rows = []
        for state in range(4):
            for period in [0, 1]:
                if state == 0:
                    # 1 obs per cell -> NaN SE -> NaN precision -> weight=0 under aweight
                    rows.append(
                        {"state": state, "period": period, "wt": 1.0, "y": rng.normal(10, 2)}
                    )
                else:
                    for _ in range(20):
                        rows.append(
                            {"state": state, "period": period, "wt": 1.0, "y": rng.normal(10, 2)}
                        )
        data = pd.DataFrame(rows)
        design = SurveyDesign(weights="wt")

        # Pweight mode: state 0 retained (cell_sum_w > 0 despite NaN precision)
        panel_p, _ = aggregate_survey(
            data,
            by=["state", "period"],
            outcomes="y",
            survey_design=design,
            second_stage_weights="pweight",
        )
        assert 0 in panel_p["state"].values
        assert len(panel_p) == 8  # 4 states x 2 periods

        # Aweight mode: state 0 dropped (all precision NaN -> weight 0)
        with pytest.warns(UserWarning, match="zero total weight"):
            panel_a, _ = aggregate_survey(
                data,
                by=["state", "period"],
                outcomes="y",
                survey_design=design,
                second_stage_weights="aweight",
            )
        assert 0 not in panel_a["state"].values
        assert len(panel_a) == 6  # 3 states x 2 periods


class TestAggregateSurveyScaffolding:
    """Tests for the amortized TSL variance fast path in aggregate_survey.

    Equivalence tests verify that ``_compute_if_variance_fast`` produces
    numerically identical ``_mean`` / ``_se`` / ``_precision`` outputs
    (assert_allclose atol=1e-14 rtol=1e-14) relative to the legacy
    ``compute_survey_if_variance`` path across every supported design
    mode and ``lonely_psu`` policy.  Reduction-order drift is expected
    to be sub-ULP because the formulas are identical and only the
    order of summation changes (single np.bincount vs per-stratum
    pandas groupby).
    """

    def _build_microdata(self, mode, seed=42):
        """Per-case microdata plus a SurveyDesign that exercises that mode."""
        rng = np.random.default_rng(seed)
        n_per_cell = 80
        state = np.repeat(["A", "B", "C"], 2 * n_per_cell)
        year = np.tile(np.repeat([2019, 2020], n_per_cell), 3)
        n = len(state)
        wt = rng.uniform(0.5, 2.5, n)
        y = rng.normal(5.0, 1.5, n)
        df_base = pd.DataFrame({"state": state, "year": year, "wt": wt, "y": y})

        if mode == "stratified_fpc":
            df = df_base.copy()
            df["stratum"] = rng.integers(0, 4, n)
            df["psu"] = df["stratum"] * 10 + rng.integers(0, 4, n)
            df["fpc"] = 200.0  # comfortably above per-stratum n_psu
            sd = SurveyDesign(weights="wt", strata="stratum", psu="psu", fpc="fpc")
            return df, sd

        if mode == "stratified_no_fpc":
            df = df_base.copy()
            df["stratum"] = rng.integers(0, 4, n)
            df["psu"] = df["stratum"] * 10 + rng.integers(0, 4, n)
            sd = SurveyDesign(weights="wt", strata="stratum", psu="psu")
            return df, sd

        if mode == "stratified_no_psu":
            # strata present, psu absent — each observation is its own
            # PSU within its stratum.  This is a distinct scaffolding
            # branch (survey.py:_precompute_psu_scaffolding, else clause
            # of the `if psu is not None` block).
            df = df_base.copy()
            df["stratum"] = rng.integers(0, 4, n)
            sd = SurveyDesign(weights="wt", strata="stratum")
            return df, sd

        if mode == "stratified_no_psu_fpc":
            # Same branch as above plus stratum-level FPC lookup.
            df = df_base.copy()
            df["stratum"] = rng.integers(0, 4, n)
            df["fpc"] = 1000.0  # well above per-stratum obs count
            sd = SurveyDesign(weights="wt", strata="stratum", fpc="fpc")
            return df, sd

        if mode == "psu_only":
            df = df_base.copy()
            df["psu"] = rng.integers(0, 12, n)
            sd = SurveyDesign(weights="wt", psu="psu")
            return df, sd

        if mode == "weights_only":
            return df_base.copy(), SurveyDesign(weights="wt")

        if mode.startswith("lonely_"):
            # Singleton stratum: stratum 0 has exactly one PSU; strata 1..3
            # each have 4 PSUs.  Forces every lonely_psu branch to engage.
            df = df_base.copy()
            strata = rng.integers(1, 4, n)
            psu = strata * 10 + rng.integers(0, 4, n)
            sentinel = rng.choice(n, size=n // 8, replace=False)
            strata[sentinel] = 0
            psu[sentinel] = 999
            df["stratum"] = strata
            df["psu"] = psu
            policy = mode.split("_", 1)[1]
            sd = SurveyDesign(
                weights="wt",
                strata="stratum",
                psu="psu",
                lonely_psu=policy,
            )
            return df, sd

        raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _assert_panels_equivalent(p_fast, p_legacy, outcome="y"):
        assert len(p_fast) == len(p_legacy)
        assert list(p_fast.columns) == list(p_legacy.columns)
        for suffix in ("_mean", "_se", "_precision"):
            col = f"{outcome}{suffix}"
            a = p_fast[col].to_numpy(dtype=np.float64)
            b = p_legacy[col].to_numpy(dtype=np.float64)
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            assert np.array_equal(nan_a, nan_b), f"NaN pattern mismatch in {col}"
            np.testing.assert_allclose(
                a[~nan_a],
                b[~nan_b],
                atol=1e-14,
                rtol=1e-14,
                err_msg=f"{col} diverges between fast and legacy paths",
            )

    @pytest.mark.parametrize(
        "mode",
        [
            "stratified_fpc",
            "stratified_no_fpc",
            "stratified_no_psu",
            "stratified_no_psu_fpc",
            "psu_only",
            "weights_only",
            "lonely_remove",
            "lonely_certainty",
            "lonely_adjust",
        ],
    )
    def test_fast_path_equals_legacy(self, mode, monkeypatch):
        """Fast and legacy paths produce numerically identical panels."""
        from diff_diff import prep

        data, sd = self._build_microdata(mode)
        panel_fast, _ = aggregate_survey(
            data,
            by=["state", "year"],
            outcomes="y",
            survey_design=sd,
        )
        # Force the legacy code path by disabling the scaffolding precompute.
        # _cell_mean_variance falls back to compute_survey_if_variance when
        # scaffolding is None.
        monkeypatch.setattr(
            prep,
            "_precompute_psu_scaffolding",
            lambda resolved: None,
        )
        panel_legacy, _ = aggregate_survey(
            data,
            by=["state", "year"],
            outcomes="y",
            survey_design=sd,
        )
        self._assert_panels_equivalent(panel_fast, panel_legacy)

    def test_scaffolding_stratified_shape(self):
        from diff_diff.survey import _precompute_psu_scaffolding

        data, sd = self._build_microdata("stratified_fpc")
        resolved = sd.resolve(data)
        scf = _precompute_psu_scaffolding(resolved)
        assert scf.mode == "stratified"
        assert scf.n == len(data)
        assert scf.psu_codes.shape == (len(data),)
        assert scf.psu_stratum.ndim == 1
        assert scf.n_psu_per_stratum.ndim == 1
        assert len(scf.psu_stratum) == int(scf.psu_codes.max() + 1)
        # adjustment_h is zero for any singleton stratum by construction
        if scf.singleton_strata.any():
            assert np.all(scf.adjustment_h[scf.singleton_strata] == 0.0)

    def test_scaffolding_weights_only_shape(self):
        from diff_diff.survey import _precompute_psu_scaffolding

        data, sd = self._build_microdata("weights_only")
        resolved = sd.resolve(data)
        scf = _precompute_psu_scaffolding(resolved)
        assert scf.mode == "no_strata_no_psu"
        assert scf.adjustment_direct is not None
        assert scf.psu_codes is None
        assert scf.psu_codes_only is None

    def test_scaffolding_psu_only_shape(self):
        from diff_diff.survey import _precompute_psu_scaffolding

        data, sd = self._build_microdata("psu_only")
        resolved = sd.resolve(data)
        scf = _precompute_psu_scaffolding(resolved)
        assert scf.mode == "psu_only"
        assert scf.psu_codes_only is not None
        assert scf.n_psu_only is not None and scf.n_psu_only >= 2
        assert scf.adjustment_only is not None
        assert scf.psu_codes is None
        assert scf.adjustment_direct is None

    def test_lonely_psu_certainty_counts_singletons(self):
        """Under lonely_psu='certainty', singletons contribute to legitimate_zero_count."""
        from diff_diff.survey import _precompute_psu_scaffolding

        data, sd = self._build_microdata("lonely_certainty")
        resolved = sd.resolve(data)
        scf = _precompute_psu_scaffolding(resolved)
        n_singletons = int(scf.singleton_strata.sum())
        assert n_singletons >= 1  # sanity: fixture does plant a singleton
        assert scf.legitimate_zero_count >= n_singletons

    def test_scaffolding_fpc_saturation_counts(self):
        """f_h >= 1.0 increments legitimate_zero_count independent of singletons."""
        from diff_diff.survey import _precompute_psu_scaffolding

        rng = np.random.default_rng(7)
        n = 200
        stratum = rng.integers(0, 2, n)
        # Build exactly 4 unique PSUs per stratum so FPC = n_psu exactly.
        psu = np.empty(n, dtype=np.int64)
        for h in range(2):
            idx = np.where(stratum == h)[0]
            psu[idx] = np.arange(len(idx)) % 4 + h * 10
        df = pd.DataFrame(
            {
                "wt": rng.uniform(1, 2, n),
                "stratum": stratum,
                "psu": psu,
                "y": rng.normal(size=n),
                "fpc": 4.0,  # f_h = 4/4 = 1.0
            }
        )
        sd = SurveyDesign(weights="wt", strata="stratum", psu="psu", fpc="fpc")
        resolved = sd.resolve(df)
        scf = _precompute_psu_scaffolding(resolved)
        assert scf.legitimate_zero_count >= 1
