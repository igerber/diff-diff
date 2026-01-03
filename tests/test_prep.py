"""
Tests for data preparation utility functions.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.prep import (
    make_treatment_indicator,
    make_post_indicator,
    wide_to_long,
    balance_panel,
    validate_did_data,
    summarize_did_data,
    generate_did_data,
    create_event_time,
    aggregate_to_cohorts,
)


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
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
            "y": [1, 2, 3]
        })
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
        wide_df = pd.DataFrame({
            "firm_id": [1, 2],
            "sales_2019": [100, 150],
            "sales_2020": [110, 160],
            "sales_2021": [120, 170]
        })
        result = wide_to_long(
            wide_df,
            value_columns=["sales_2019", "sales_2020", "sales_2021"],
            id_column="firm_id",
            time_name="year",
            value_name="sales"
        )
        assert len(result) == 6
        assert set(result.columns) == {"firm_id", "year", "sales"}

    def test_with_time_values(self):
        """Test with explicit time values."""
        wide_df = pd.DataFrame({
            "id": [1],
            "t1": [10],
            "t2": [20]
        })
        result = wide_to_long(
            wide_df,
            value_columns=["t1", "t2"],
            id_column="id",
            time_values=[2020, 2021]
        )
        assert result["period"].tolist() == [2020, 2021]

    def test_preserves_other_columns(self):
        """Test that other columns are preserved."""
        wide_df = pd.DataFrame({
            "id": [1, 2],
            "group": ["A", "B"],
            "t1": [10, 20],
            "t2": [15, 25]
        })
        result = wide_to_long(
            wide_df,
            value_columns=["t1", "t2"],
            id_column="id"
        )
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
        df = pd.DataFrame({
            "unit": [1, 1, 1, 2, 2, 3, 3, 3],
            "period": [1, 2, 3, 1, 2, 1, 2, 3],
            "y": [10, 11, 12, 20, 21, 30, 31, 32]
        })
        result = balance_panel(df, "unit", "period", method="inner")
        assert set(result["unit"].unique()) == {1, 3}
        assert len(result) == 6

    def test_outer_balance(self):
        """Test outer balance (include all combinations)."""
        df = pd.DataFrame({
            "unit": [1, 1, 2],
            "period": [1, 2, 1],
            "y": [10, 11, 20]
        })
        result = balance_panel(df, "unit", "period", method="outer")
        assert len(result) == 4  # 2 units x 2 periods

    def test_fill_with_value(self):
        """Test fill method with specific value."""
        df = pd.DataFrame({
            "unit": [1, 1, 2],
            "period": [1, 2, 1],
            "y": [10.0, 11.0, 20.0]
        })
        result = balance_panel(df, "unit", "period", method="fill", fill_value=0.0)
        assert len(result) == 4
        missing_row = result[(result["unit"] == 2) & (result["period"] == 2)]
        assert missing_row["y"].values[0] == 0.0

    def test_fill_forward_backward(self):
        """Test fill method with forward/backward fill."""
        df = pd.DataFrame({
            "unit": [1, 1, 1, 2, 2],
            "period": [1, 2, 3, 1, 3],  # Unit 2 missing period 2
            "y": [10.0, 11.0, 12.0, 20.0, 22.0]
        })
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
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0, 4.0],
            "treated": [0, 0, 1, 1],
            "post": [0, 1, 0, 1]
        })
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
        df = pd.DataFrame({
            "y": ["a", "b", "c", "d"],
            "treated": [0, 0, 1, 1],
            "post": [0, 1, 0, 1]
        })
        result = validate_did_data(df, "y", "treated", "post", raise_on_error=False)
        assert result["valid"] is False
        assert any("numeric" in e for e in result["errors"])

    def test_non_binary_treatment(self):
        """Test validation catches non-binary treatment."""
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0],
            "treated": [0, 1, 2],
            "post": [0, 1, 0]
        })
        result = validate_did_data(df, "y", "treated", "post", raise_on_error=False)
        assert result["valid"] is False
        assert any("binary" in e for e in result["errors"])

    def test_missing_values(self):
        """Test validation catches missing values."""
        df = pd.DataFrame({
            "y": [1.0, np.nan, 3.0, 4.0],
            "treated": [0, 0, 1, 1],
            "post": [0, 1, 0, 1]
        })
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
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0, 4.0],
            "treated": [0, 0, 1, 1],
            "post": [0, 1, 0, 1],
            "unit": [1, 1, 2, 2]
        })
        result = validate_did_data(df, "y", "treated", "post", unit="unit", raise_on_error=False)
        assert result["valid"] is True
        assert result["summary"]["n_units"] == 2


class TestSummarizeDidData:
    """Tests for summarize_did_data function."""

    def test_basic_summary(self):
        """Test basic summary statistics."""
        df = pd.DataFrame({
            "y": [10, 11, 12, 13, 20, 21, 22, 23],
            "treated": [0, 0, 1, 1, 0, 0, 1, 1],
            "post": [0, 1, 0, 1, 0, 1, 0, 1]
        })
        summary = summarize_did_data(df, "y", "treated", "post")
        assert len(summary) == 5  # 4 groups + DiD estimate

    def test_did_estimate_included(self):
        """Test that DiD estimate is calculated."""
        df = pd.DataFrame({
            "y": [10, 20, 15, 30],  # Perfect DiD = 30-15 - (20-10) = 5
            "treated": [0, 0, 1, 1],
            "post": [0, 1, 0, 1]
        })
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
            n_units=200,
            n_periods=4,
            treatment_effect=true_effect,
            noise_sd=0.5,
            seed=42
        )

        did = DifferenceInDifferences()
        results = did.fit(data, outcome="outcome", treatment="treated", time="post")

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
        df = pd.DataFrame({
            "unit": [1, 1, 1, 2, 2, 2],
            "year": [2018, 2019, 2020, 2018, 2019, 2020],
            "treatment_year": [2019, 2019, 2019, 2020, 2020, 2020]
        })
        result = create_event_time(df, "year", "treatment_year")
        assert result["event_time"].tolist() == [-1, 0, 1, -2, -1, 0]

    def test_never_treated(self):
        """Test handling of never-treated units."""
        df = pd.DataFrame({
            "unit": [1, 1, 2, 2],
            "year": [2019, 2020, 2019, 2020],
            "treatment_year": [2020, 2020, np.nan, np.nan]
        })
        result = create_event_time(df, "year", "treatment_year")
        assert result.loc[0, "event_time"] == -1
        assert result.loc[1, "event_time"] == 0
        assert pd.isna(result.loc[2, "event_time"])
        assert pd.isna(result.loc[3, "event_time"])

    def test_custom_column_name(self):
        """Test custom output column name."""
        df = pd.DataFrame({
            "year": [2019, 2020],
            "treat_time": [2020, 2020]
        })
        result = create_event_time(df, "year", "treat_time", new_column="rel_time")
        assert "rel_time" in result.columns


class TestAggregateToCohorts:
    """Tests for aggregate_to_cohorts function."""

    def test_basic_aggregation(self):
        """Test basic cohort aggregation."""
        df = pd.DataFrame({
            "unit": [1, 1, 2, 2, 3, 3, 4, 4],
            "period": [0, 1, 0, 1, 0, 1, 0, 1],
            "treated": [1, 1, 1, 1, 0, 0, 0, 0],
            "y": [10, 15, 12, 17, 8, 10, 9, 11]
        })
        result = aggregate_to_cohorts(df, "unit", "period", "treated", "y")
        assert len(result) == 4  # 2 treatment groups x 2 periods
        assert "mean_y" in result.columns
        assert "n_units" in result.columns

    def test_with_covariates(self):
        """Test aggregation with covariates."""
        df = pd.DataFrame({
            "unit": [1, 1, 2, 2],
            "period": [0, 1, 0, 1],
            "treated": [1, 1, 0, 0],
            "y": [10, 15, 8, 10],
            "x": [1.0, 1.5, 0.5, 0.8]
        })
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
            treatment_column="treated"
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
            covariates=["x1"]
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
            treated_units=[0, 1, 2]
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
            exclude_units=[15, 16, 17]
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
            n_top=5
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
            n_top=10
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
            n_treatment_candidates=5
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
            treatment_column="treated"
        )
        assert data.columns.tolist() == original_cols

    def test_error_missing_column(self):
        """Test error when column doesn't exist."""
        from diff_diff.prep import rank_control_units

        data = generate_did_data(n_units=10, n_periods=4, seed=42)

        with pytest.raises(ValueError, match="not found"):
            rank_control_units(
                data,
                unit_column="missing_col",
                time_column="period",
                outcome_column="outcome"
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
                treated_units=[0, 1]
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
                exclude_units=[5]
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
            treatment_column="treated"
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
            pre_periods=[0, 1]  # Only use first two periods
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
            covariate_weight=0.0
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
            covariate_weight=1.0
        )

        # Rankings should differ
        # (just check both work, exact comparison is data-dependent)
        assert len(result1) > 0
        assert len(result2) > 0
