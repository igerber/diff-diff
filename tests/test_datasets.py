"""
Tests for the datasets module.

These tests verify that the dataset loading functions work correctly,
including both the download/cache mechanism and the fallback data generation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from diff_diff.datasets import (
    _construct_card_krueger_data,
    _construct_castle_doctrine_data,
    _construct_divorce_laws_data,
    _construct_mpdta_data,
    _construct_prop99_data,
    _construct_walmart_data,
    clear_cache,
    list_datasets,
    load_card_krueger,
    load_dataset,
    load_prop99,
    load_walmart,
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
        expected = {
            "card_krueger",
            "castle_doctrine",
            "divorce_laws",
            "mpdta",
            "prop99",
            "walmart",
        }
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

    def test_load_by_name_binary(self):
        """load_dataset should dispatch to the binary (.dta) loaders."""
        with patch("diff_diff.datasets._download_with_cache_binary") as mock:
            mock.side_effect = RuntimeError("No network")
            for name in ("prop99", "walmart"):
                with pytest.warns(UserWarning, match="SYNTHETIC"):
                    df = load_dataset(name)
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


class TestProp99:
    """Tests for California Prop 99 smoking dataset."""

    def test_fallback_data_structure(self):
        """Fallback data should have expected structure."""
        df = _construct_prop99_data()

        # Check required columns
        required_cols = {"state", "year", "first_year", "lcigsale"}
        assert required_cols.issubset(set(df.columns))

        # Check years
        assert df["year"].min() == 1970
        assert df["year"].max() == 2000

    def test_fallback_data_treatment(self):
        """Fallback data should have a single 1989 cohort and zero-coded controls."""
        df = _construct_prop99_data()

        # Exactly one treated state, cohort 1989
        treated_states = df.loc[df["first_year"] > 0, "state"].unique()
        assert len(treated_states) == 1
        assert set(df.loc[df["first_year"] > 0, "first_year"].unique()) == {1989}

        # Never-treated states coded 0
        assert df.loc[df["first_year"] == 0, "state"].nunique() == 38

    def test_fallback_data_values(self):
        """Fallback data should have reasonable log-scale values."""
        df = _construct_prop99_data()

        assert (df["lcigsale"] > 2.0).all()
        assert (df["lcigsale"] < 6.0).all()

    def test_fallback_data_size(self):
        """Fallback data should be a balanced 39 x 31 panel."""
        df = _construct_prop99_data()

        assert df["state"].nunique() == 39
        assert len(df) == 39 * 31

    def test_load_uses_fallback_on_network_error(self):
        """load_prop99 should warn and mark the frame when falling back."""
        with patch("diff_diff.datasets._download_with_cache_binary") as mock:
            mock.side_effect = RuntimeError("Network error")
            with pytest.warns(UserWarning, match="SYNTHETIC"):
                df = load_prop99()
            assert isinstance(df, pd.DataFrame)
            assert df.attrs["source"] == "synthetic_fallback"
            assert "treated" in df.columns
            assert "cohort" in df.columns
            # treated indicator consistent with first_year timing
            in_effect = (df["first_year"] > 0) & (df["year"] >= df["first_year"])
            assert (df["treated"] == in_effect.astype(int)).all()


class TestWalmart:
    """Tests for Walmart entry county panel."""

    def test_fallback_data_structure(self):
        """Fallback data should have expected structure."""
        df = _construct_walmart_data()

        # Check required columns
        required_cols = {
            "cid",
            "year",
            "first_year",
            "log_retail_emp",
            "log_wholesale_emp",
            "x1",
            "x2",
            "x3",
        }
        assert required_cols.issubset(set(df.columns))

        # Check years
        assert df["year"].min() == 1977
        assert df["year"].max() == 1999

    def test_fallback_data_treatment(self):
        """Fallback data should have staggered 1986-1999 cohorts + never-treated."""
        df = _construct_walmart_data()

        cohorts = set(df.loc[df["first_year"] > 0, "first_year"].unique())
        assert cohorts.issubset(set(range(1986, 2000)))

        # A meaningful never-treated group coded 0
        assert df.loc[df["first_year"] == 0, "cid"].nunique() > 0

    def test_fallback_data_values(self):
        """Fallback data should have reasonable values."""
        df = _construct_walmart_data()

        # Covariates are shares/rates in (0, 1)
        for col in ("x1", "x2", "x3"):
            assert (df[col] > 0).all()
            assert (df[col] < 1).all()

    def test_fallback_data_size(self):
        """Fallback data should be a balanced counties x 23-year panel."""
        df = _construct_walmart_data()

        n_counties = df["cid"].nunique()
        assert n_counties == 200
        assert len(df) == n_counties * 23

    def test_load_uses_fallback_on_network_error(self):
        """load_walmart should warn and mark the frame when falling back."""
        with patch("diff_diff.datasets._download_with_cache_binary") as mock:
            mock.side_effect = RuntimeError("Network error")
            with pytest.warns(UserWarning, match="SYNTHETIC"):
                df = load_walmart()
            assert isinstance(df, pd.DataFrame)
            assert df.attrs["source"] == "synthetic_fallback"
            assert "treated" in df.columns
            assert "cohort" in df.columns
            in_effect = (df["first_year"] > 0) & (df["year"] >= df["first_year"])
            assert (df["treated"] == in_effect.astype(int)).all()


class TestSourceValidation:
    """Tests for the downloaded-data source validators."""

    def test_prop99_valid_frame_passes(self):
        """A frame matching all source invariants should validate silently."""
        from diff_diff.datasets import _validate_prop99

        # The seeded fallback matches the real file's invariants exactly
        _validate_prop99(_construct_prop99_data())

    def test_prop99_duplicate_rows_rejected(self):
        from diff_diff.datasets import _validate_prop99

        df = _construct_prop99_data()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(RuntimeError, match="duplicate"):
            _validate_prop99(df)

    def test_prop99_nonconstant_cohort_rejected(self):
        from diff_diff.datasets import _validate_prop99

        df = _construct_prop99_data()
        df.loc[df.index[0], "first_year"] = 1975
        with pytest.raises(RuntimeError, match="not constant"):
            _validate_prop99(df)

    def test_prop99_multiple_treated_states_rejected(self):
        from diff_diff.datasets import _validate_prop99

        df = _construct_prop99_data()
        df.loc[df["state"] == "State02", "first_year"] = 1989
        with pytest.raises(RuntimeError, match="treated state count"):
            _validate_prop99(df)

    @staticmethod
    def _valid_walmart_frame():
        """Minimal frame satisfying every real-Walmart validator invariant."""
        n_counties, years = 1277, list(range(1977, 2000))
        # 391 never-treated; remaining 886 cycle through cohorts 1986-1999
        cohort_by_cid = {cid: 0 for cid in range(1, 392)}
        cohort_cycle = list(range(1986, 2000))
        for i, cid in enumerate(range(392, n_counties + 1)):
            cohort_by_cid[cid] = cohort_cycle[i % len(cohort_cycle)]
        rows = [
            {
                "year": year,
                "cid": cid,
                "first_year": fy,
                "log_retail_emp": 7.5,
                "log_wholesale_emp": 6.5,
                "x1": 0.1,
                "x2": 0.7,
                "x3": 0.2,
            }
            for cid, fy in cohort_by_cid.items()
            for year in years
        ]
        return pd.DataFrame(rows)

    def test_walmart_valid_frame_passes(self):
        from diff_diff.datasets import _validate_walmart

        _validate_walmart(self._valid_walmart_frame())

    def test_walmart_wrong_panel_rejected(self):
        from diff_diff.datasets import _validate_walmart

        # The 200-county synthetic fallback must NOT pass as the real panel
        with pytest.raises(RuntimeError, match="counties != 1277"):
            _validate_walmart(_construct_walmart_data())

    def test_walmart_duplicate_rows_rejected(self):
        from diff_diff.datasets import _validate_walmart

        df = self._valid_walmart_frame()
        df.iloc[-1, df.columns.get_loc("year")] = df.iloc[-2]["year"]
        with pytest.raises(RuntimeError, match="duplicate"):
            _validate_walmart(df)

    def test_walmart_nonconstant_cohort_rejected(self):
        from diff_diff.datasets import _validate_walmart

        df = self._valid_walmart_frame()
        df.loc[df.index[0], "first_year"] = 1990
        with pytest.raises(RuntimeError, match="not constant"):
            _validate_walmart(df)

    def test_walmart_missing_cohort_rejected(self):
        from diff_diff.datasets import _validate_walmart

        df = self._valid_walmart_frame()
        df["first_year"] = df["first_year"].replace(1986, 1987)
        with pytest.raises(RuntimeError, match="treated cohorts"):
            _validate_walmart(df)

    def test_walmart_never_treated_count_rejected(self):
        from diff_diff.datasets import _validate_walmart

        df = self._valid_walmart_frame()
        # Convert one never-treated county to a treated cohort
        df.loc[df["cid"] == 1, "first_year"] = 1990
        with pytest.raises(RuntimeError, match="never-treated county count"):
            _validate_walmart(df)


class TestBinaryDownloadIntegrity:
    """Tests for the checksum-verified binary download helper."""

    def test_checksum_mismatch_raises(self, tmp_path, monkeypatch):
        """A fresh download that fails the pinned checksum should raise."""
        import diff_diff.datasets as datasets_mod

        monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path)

        fake_response = MagicMock()
        fake_response.read.return_value = b"tampered bytes"
        fake_response.__enter__ = lambda self: self
        fake_response.__exit__ = lambda self, *a: False

        with patch("diff_diff.datasets.urlopen", return_value=fake_response):
            with pytest.raises(RuntimeError, match="Checksum mismatch"):
                datasets_mod._download_with_cache_binary(
                    "http://example.invalid/x.dta", "x", sha256="0" * 64
                )
        # Tampered bytes must not be cached
        assert not (tmp_path / "x.dta").exists()

    def test_stale_cache_triggers_redownload(self, tmp_path, monkeypatch):
        """A cached file failing the checksum should be replaced by a re-download."""
        import hashlib

        import diff_diff.datasets as datasets_mod

        monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path)
        good = b"good bytes"
        good_sha = hashlib.sha256(good).hexdigest()
        (tmp_path / "x.dta").write_bytes(b"stale bytes")

        fake_response = MagicMock()
        fake_response.read.return_value = good
        fake_response.__enter__ = lambda self: self
        fake_response.__exit__ = lambda self, *a: False

        with patch("diff_diff.datasets.urlopen", return_value=fake_response):
            content = datasets_mod._download_with_cache_binary(
                "http://example.invalid/x.dta", "x", sha256=good_sha
            )
        assert content == good
        assert (tmp_path / "x.dta").read_bytes() == good


class TestClearCache:
    """Tests for cache management."""

    def test_clear_cache_creates_directory(self):
        """clear_cache should handle non-existent cache gracefully."""
        # This should not raise even if cache doesn't exist
        try:
            clear_cache()
        except Exception as e:
            pytest.fail(f"clear_cache raised unexpected exception: {e}")

    def test_clear_cache_removes_csv_and_dta(self, tmp_path, monkeypatch):
        """clear_cache should remove both .csv and .dta cached files."""
        import diff_diff.datasets as datasets_mod

        monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path)
        csv_file = tmp_path / "dummy.csv"
        dta_file = tmp_path / "dummy.dta"
        csv_file.write_text("a,b\n1,2\n")
        dta_file.write_bytes(b"\x00\x01")

        clear_cache()

        assert not csv_file.exists()
        assert not dta_file.exists()


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
        results = did.fit(df_long, outcome="employment", treatment="treated", time="post")

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

    def test_prop99_with_did(self):
        """Prop 99 data should work with DifferenceInDifferences."""
        from diff_diff import DifferenceInDifferences

        # Use fallback data
        df = _construct_prop99_data()
        df["treated_state"] = (df["first_year"] > 0).astype(int)
        df["post"] = (df["year"] >= 1989).astype(int)

        did = DifferenceInDifferences()
        results = did.fit(df, outcome="lcigsale", treatment="treated_state", time="post")

        assert hasattr(results, "att")
        assert hasattr(results, "se")
        assert not np.isnan(results.att)
        # The synthetic DGP builds in a negative post-1989 effect for California
        assert results.att < 0

    def test_walmart_with_cs(self):
        """Walmart data should work with CallawaySantAnna."""
        from diff_diff import CallawaySantAnna

        # Use fallback data
        df = _construct_walmart_data()

        cs = CallawaySantAnna(control_group="never_treated")
        results = cs.fit(
            df,
            outcome="log_retail_emp",
            unit="cid",
            time="year",
            first_treat="first_year",
        )

        assert hasattr(results, "group_time_effects")
        assert len(results.group_time_effects) > 0
