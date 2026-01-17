"""
Tests for the datasets module.

These tests verify that the dataset loading functions work correctly,
including both the download/cache mechanism and the fallback data generation.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from diff_diff.datasets import (
    _CACHE_DIR,
    _construct_card_krueger_data,
    _construct_castle_doctrine_data,
    _construct_divorce_laws_data,
    _construct_mpdta_data,
    clear_cache,
    list_datasets,
    load_card_krueger,
    load_castle_doctrine,
    load_dataset,
    load_divorce_laws,
    load_mpdta,
)


class TestListDatasets:
    """Tests for list_datasets function."""

    def test_returns_dict(self):
        """list_datasets should return a dictionary."""
        result = list_datasets()
        assert isinstance(result, dict)

    def test_contains_expected_datasets(self):
        """list_datasets should contain all expected datasets."""
        result = list_datasets()
        expected = {"card_krueger", "castle_doctrine", "divorce_laws", "mpdta"}
        assert set(result.keys()) == expected

    def test_descriptions_are_strings(self):
        """All descriptions should be non-empty strings."""
        result = list_datasets()
        for name, desc in result.items():
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_by_name(self):
        """load_dataset should load datasets by name."""
        # Use fallback data to avoid network dependency
        with patch("diff_diff.datasets._download_with_cache") as mock:
            mock.side_effect = RuntimeError("No network")
            df = load_dataset("card_krueger")
            assert isinstance(df, pd.DataFrame)

    def test_invalid_name_raises(self):
        """load_dataset should raise ValueError for unknown datasets."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")


class TestCardKrueger:
    """Tests for Card-Krueger dataset."""

    def test_fallback_data_structure(self):
        """Fallback data should have expected structure."""
        df = _construct_card_krueger_data()

        # Check required columns
        required_cols = {"store_id", "state", "chain", "emp_pre", "emp_post", "treated"}
        assert required_cols.issubset(set(df.columns))

        # Check states
        assert set(df["state"].unique()) == {"NJ", "PA"}

        # Check treatment indicator
        assert df[df["state"] == "NJ"]["treated"].all() == 1
        assert df[df["state"] == "PA"]["treated"].all() == 0

        # Check chains
        expected_chains = {"bk", "kfc", "roys", "wendys"}
        assert set(df["chain"].unique()) == expected_chains

    def test_fallback_data_size(self):
        """Fallback data should have reasonable size."""
        df = _construct_card_krueger_data()
        # Should have roughly 300+ stores total
        assert 250 < len(df) < 450

    def test_fallback_data_values(self):
        """Fallback data should have reasonable values."""
        df = _construct_card_krueger_data()

        # Employment should be non-negative
        assert (df["emp_pre"] >= 0).all()
        assert (df["emp_post"] >= 0).all()

        # Wages should be reasonable (around minimum wage range)
        assert (df["wage_pre"] > 3).all()
        assert (df["wage_pre"] < 7).all()

    def test_load_uses_fallback_on_network_error(self):
        """load_card_krueger should use fallback when network fails."""
        with patch("diff_diff.datasets._download_with_cache") as mock:
            mock.side_effect = RuntimeError("Network error")
            df = load_card_krueger()
            assert isinstance(df, pd.DataFrame)
            assert "treated" in df.columns


class TestCastleDoctrine:
    """Tests for Castle Doctrine dataset."""

    def test_fallback_data_structure(self):
        """Fallback data should have expected structure."""
        df = _construct_castle_doctrine_data()

        # Check required columns
        required_cols = {"state", "year", "first_treat", "homicide_rate", "treated"}
        assert required_cols.issubset(set(df.columns))

        # Check years
        assert df["year"].min() == 2000
        assert df["year"].max() == 2010

    def test_fallback_data_treatment(self):
        """Fallback data should have correct treatment structure."""
        df = _construct_castle_doctrine_data()

        # Check that never-treated states have first_treat = 0
        never_treated = df[df["first_treat"] == 0]
        assert len(never_treated) > 0
        assert (never_treated["treated"] == 0).all()

        # Check that treated indicator matches timing
        treated_states = df[df["first_treat"] > 0]
        for _, row in treated_states.iterrows():
            expected_treated = 1 if row["year"] >= row["first_treat"] else 0
            assert row["treated"] == expected_treated

    def test_fallback_data_values(self):
        """Fallback data should have reasonable values."""
        df = _construct_castle_doctrine_data()

        # Homicide rates should be positive
        assert (df["homicide_rate"] > 0).all()
        assert (df["homicide_rate"] < 20).all()


class TestDivorceLaws:
    """Tests for Divorce Laws dataset."""

    def test_fallback_data_structure(self):
        """Fallback data should have expected structure."""
        df = _construct_divorce_laws_data()

        # Check required columns
        required_cols = {"state", "year", "first_treat", "divorce_rate", "treated"}
        assert required_cols.issubset(set(df.columns))

        # Check years
        assert df["year"].min() == 1968
        assert df["year"].max() == 1988

    def test_fallback_data_treatment(self):
        """Fallback data should have correct treatment structure."""
        df = _construct_divorce_laws_data()

        # Check that treated indicator matches timing
        for _, row in df.iterrows():
            if row["first_treat"] == 0:
                assert row["treated"] == 0
            elif row["year"] >= row["first_treat"]:
                assert row["treated"] == 1
            else:
                assert row["treated"] == 0

    def test_fallback_data_values(self):
        """Fallback data should have reasonable values."""
        df = _construct_divorce_laws_data()

        # Divorce rates should be positive
        assert (df["divorce_rate"] > 0).all()
        assert (df["divorce_rate"] < 15).all()

        # Female LFP should be between 0 and 1
        assert (df["female_lfp"] >= 0).all()
        assert (df["female_lfp"] <= 1).all()


class TestMPDTA:
    """Tests for mpdta dataset."""

    def test_fallback_data_structure(self):
        """Fallback data should have expected structure."""
        df = _construct_mpdta_data()

        # Check required columns
        required_cols = {"countyreal", "year", "lpop", "lemp", "first_treat", "treat"}
        assert required_cols.issubset(set(df.columns))

        # Check years
        assert set(df["year"].unique()) == {2003, 2004, 2005, 2006, 2007}

    def test_fallback_data_cohorts(self):
        """Fallback data should have expected cohorts."""
        df = _construct_mpdta_data()

        # Cohorts should be 0, 2004, 2006, 2007
        expected_cohorts = {0, 2004, 2006, 2007}
        assert set(df["first_treat"].unique()) == expected_cohorts

    def test_fallback_data_size(self):
        """Fallback data should have expected size."""
        df = _construct_mpdta_data()

        # 500 counties * 5 years = 2500 rows
        assert len(df) == 2500
        assert df["countyreal"].nunique() == 500


class TestClearCache:
    """Tests for cache management."""

    def test_clear_cache_creates_directory(self):
        """clear_cache should handle non-existent cache gracefully."""
        # This should not raise even if cache doesn't exist
        try:
            clear_cache()
        except Exception as e:
            pytest.fail(f"clear_cache raised unexpected exception: {e}")


class TestDatasetIntegration:
    """Integration tests verifying datasets work with estimators."""

    def test_card_krueger_with_did(self):
        """Card-Krueger data should work with DifferenceInDifferences."""
        from diff_diff import DifferenceInDifferences

        # Use fallback data
        df = _construct_card_krueger_data()

        # Reshape to long format
        df_long = df.melt(
            id_vars=["store_id", "state", "treated"],
            value_vars=["emp_pre", "emp_post"],
            var_name="period",
            value_name="employment",
        )
        df_long["post"] = (df_long["period"] == "emp_post").astype(int)
        df_long = df_long.dropna(subset=["employment"])

        # Should be able to fit DiD
        did = DifferenceInDifferences()
        results = did.fit(
            df_long, outcome="employment", treatment="treated", time="post"
        )

        assert hasattr(results, "att")
        assert hasattr(results, "se")
        assert not np.isnan(results.att)

    def test_castle_doctrine_with_cs(self):
        """Castle Doctrine data should work with CallawaySantAnna."""
        from diff_diff import CallawaySantAnna

        # Use fallback data
        df = _construct_castle_doctrine_data()

        # Should be able to fit CS
        cs = CallawaySantAnna(control_group="never_treated")
        results = cs.fit(
            df,
            outcome="homicide_rate",
            unit="state",
            time="year",
            first_treat="first_treat",
        )

        assert hasattr(results, "group_time_effects")
        assert len(results.group_time_effects) > 0

    def test_mpdta_with_cs(self):
        """mpdta data should work with CallawaySantAnna."""
        from diff_diff import CallawaySantAnna

        # Use fallback data
        df = _construct_mpdta_data()

        # Should be able to fit CS
        cs = CallawaySantAnna(control_group="never_treated")
        results = cs.fit(
            df,
            outcome="lemp",
            unit="countyreal",
            time="year",
            first_treat="first_treat",
        )

        assert hasattr(results, "group_time_effects")
        assert len(results.group_time_effects) > 0
