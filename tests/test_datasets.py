"""
Tests for the datasets module.

These tests verify that the dataset loading functions work correctly,
including both the download/cache mechanism and the fallback data generation.
"""

from pathlib import Path
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
    load_castle_doctrine,
    load_dataset,
    load_divorce_laws,
    load_mpdta,
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

    def test_divorce_laws_catalogue_marks_synthetic_only(self):
        """Discovery metadata should not imply divorce_laws is canonical data."""
        result = list_datasets()
        assert "synthetic fallback only" in result["divorce_laws"]


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_by_name(self):
        """load_dataset should load datasets by name."""
        # Use fallback data to avoid network dependency
        with patch("diff_diff.datasets._download_with_cache") as mock:
            from diff_diff.datasets import _DatasetSourceError

            mock.side_effect = _DatasetSourceError("No network")
            with pytest.warns(UserWarning, match="SYNTHETIC"):
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
            from diff_diff.datasets import _DatasetSourceError

            mock.side_effect = _DatasetSourceError("Network error")
            with pytest.warns(UserWarning, match="SYNTHETIC"):
                df = load_card_krueger()
            assert isinstance(df, pd.DataFrame)
            assert df.attrs["source"] == "synthetic_fallback"
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


class TestLegacyLoaderProvenance:
    """Legacy loaders must never silently present synthetic data as canonical."""

    LOADERS = (
        (
            load_card_krueger,
            _construct_card_krueger_data,
            "_load_card_krueger_source",
            "card_krueger_public_data",
        ),
        (
            load_castle_doctrine,
            _construct_castle_doctrine_data,
            "_load_castle_doctrine_source",
            "cheng_hoekstra_castle_data",
        ),
        (
            load_divorce_laws,
            _construct_divorce_laws_data,
            None,
            None,
        ),
        (
            load_mpdta,
            _construct_mpdta_data,
            "_load_mpdta_source",
            "callaway_santanna_mpdta",
        ),
    )

    @staticmethod
    def _valid_card_source_frame():
        n = 410
        df = pd.DataFrame(
            {
                "store_id": np.arange(1, n + 1),
                "state": ["NJ"] * 331 + ["PA"] * 79,
                "chain": np.resize(["bk", "kfc", "roys", "wendys"], n),
                "emp_pre": np.full(n, 20.0),
                "emp_post": np.full(n, 21.0),
                "wage_pre": np.full(n, 4.5),
                "wage_post": np.full(n, 5.0),
            }
        )
        df.loc[:11, "emp_pre"] = np.nan
        df.loc[12:25, "emp_post"] = np.nan
        df.loc[:19, "wage_pre"] = np.nan
        df.loc[20:40, "wage_post"] = np.nan
        df["treated"] = (df["state"] == "NJ").astype(int)
        df["emp_change"] = df["emp_post"] - df["emp_pre"]
        return df

    @staticmethod
    def _valid_castle_source_frame():
        import diff_diff.datasets as datasets_mod

        states = [code for code in datasets_mod._CASTLE_STATE_BY_SID.values() if code != "_"]
        cohorts = dict(zip(states[:5], [2005, 2006, 2007, 2008, 2009]))
        rows = []
        for state in states:
            first_treat = cohorts.get(state, 0)
            for year in range(2000, 2011):
                rows.append(
                    {
                        "state": state,
                        "year": year,
                        "first_treat": first_treat,
                        "homicide_rate": 5.0,
                        "population": 1_000_000,
                        "income": 40_000,
                        "treated": int(first_treat > 0 and year >= first_treat),
                        "treatment_exposure": float(first_treat > 0 and year >= first_treat),
                        "cohort": first_treat,
                    }
                )
        return pd.DataFrame(rows)

    @pytest.mark.parametrize(("loader", "fallback", "source_loader", "_source"), LOADERS)
    def test_network_failure_warns_and_marks_synthetic_fallback(
        self, loader, fallback, source_loader, _source, monkeypatch
    ):
        import diff_diff.datasets as datasets_mod

        if source_loader is not None:
            monkeypatch.setattr(
                datasets_mod,
                source_loader,
                MagicMock(side_effect=datasets_mod._DatasetSourceError("Network error")),
            )
        with pytest.warns(UserWarning, match="SYNTHETIC") as caught:
            result = loader()

        assert len(caught) == 1
        assert caught[0].filename.endswith("test_datasets.py")
        assert result.attrs["source"] == "synthetic_fallback"
        assert result.shape == fallback().shape

    @pytest.mark.parametrize(("loader", "fallback", "source_loader", "_source"), LOADERS)
    def test_malformed_download_warns_and_uses_marked_fallback(
        self, loader, fallback, source_loader, _source, monkeypatch
    ):
        import diff_diff.datasets as datasets_mod

        if source_loader is not None:
            monkeypatch.setattr(
                datasets_mod,
                source_loader,
                lambda _force_download: pd.DataFrame({"bad": [1]}),
            )
        with pytest.warns(UserWarning, match="SYNTHETIC") as caught:
            result = loader()

        assert len(caught) == 1
        assert result.attrs["source"] == "synthetic_fallback"
        assert result.shape == fallback().shape

    @pytest.mark.parametrize(
        ("loader", "fallback", "source_loader", "source"),
        [case for case in LOADERS if case[2] is not None],
    )
    def test_verified_download_is_marked_with_source(
        self, loader, fallback, source_loader, source, monkeypatch
    ):
        import diff_diff.datasets as datasets_mod

        valid_frame = {
            load_card_krueger: self._valid_card_source_frame,
            load_castle_doctrine: self._valid_castle_source_frame,
            load_mpdta: _construct_mpdta_data,
        }[loader]()
        monkeypatch.setattr(
            datasets_mod,
            source_loader,
            lambda _force_download: valid_frame,
        )
        result = loader()

        assert result.attrs["source"] == source

    def test_verified_download_survives_cache_write_failure(self, tmp_path, monkeypatch):
        """A verified mpdta download should be returned even if caching fails."""
        import hashlib

        import diff_diff.datasets as datasets_mod

        content = _construct_mpdta_data().to_csv(index=False).encode("utf-8")
        sha256 = hashlib.sha256(content).hexdigest()
        fake_response = MagicMock()
        fake_response.read.return_value = content
        fake_response.__enter__ = lambda self: self
        fake_response.__exit__ = lambda self, *a: False

        monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path)
        monkeypatch.setattr(datasets_mod, "_MPDTA_SOURCE_SHA256", sha256)
        with (
            patch("diff_diff.datasets.urlopen", return_value=fake_response),
            patch("diff_diff.datasets.os.replace", side_effect=OSError("disk full")),
        ):
            result = load_mpdta(force_download=True)

        assert result.attrs["source"] == "callaway_santanna_mpdta"
        assert result.shape == _construct_mpdta_data().shape
        assert not (tmp_path / "mpdta.csv").exists()

    def test_verified_download_survives_unwritable_cache_directory(self, tmp_path, monkeypatch):
        """A cache-directory failure must not discard verified download bytes."""
        import hashlib

        import diff_diff.datasets as datasets_mod

        content = _construct_mpdta_data().to_csv(index=False).encode("utf-8")
        cache_dir = tmp_path / "unwritable-cache"
        sha256 = hashlib.sha256(content).hexdigest()
        fake_response = MagicMock()
        fake_response.read.return_value = content
        fake_response.__enter__ = lambda self: self
        fake_response.__exit__ = lambda self, *a: False
        original_mkdir = Path.mkdir

        def deny_cache_directory(path, *args, **kwargs):
            if path == cache_dir:
                raise PermissionError("read-only cache directory")
            return original_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(datasets_mod, "_CACHE_DIR", cache_dir)
        monkeypatch.setattr(datasets_mod, "_MPDTA_SOURCE_SHA256", sha256)
        monkeypatch.setattr(Path, "mkdir", deny_cache_directory)
        with patch("diff_diff.datasets.urlopen", return_value=fake_response):
            result = load_mpdta(force_download=True)

        assert result.attrs["source"] == "callaway_santanna_mpdta"
        assert result.shape == _construct_mpdta_data().shape
        assert not cache_dir.exists()

    def test_incomplete_download_warns_and_uses_marked_fallback(self, tmp_path, monkeypatch):
        """Truncated responses must follow the same explicit fallback path."""
        from http.client import IncompleteRead

        import diff_diff.datasets as datasets_mod

        fake_response = MagicMock()
        fake_response.read.side_effect = IncompleteRead(b"partial", 100)
        fake_response.__enter__ = lambda self: self
        fake_response.__exit__ = lambda self, *a: False

        monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path)
        with patch("diff_diff.datasets.urlopen", return_value=fake_response):
            with pytest.warns(UserWarning, match="SYNTHETIC"):
                result = load_mpdta(force_download=True)

        assert result.attrs["source"] == "synthetic_fallback"

    def test_protocol_level_http_errors_warn_and_use_marked_fallback(self, tmp_path, monkeypatch):
        """Every ``HTTPException``, not just ``IncompleteRead``, stays inside the boundary.

        ``BadStatusLine`` and its siblings derive from ``HTTPException`` rather than
        ``OSError``, so naming individual subclasses in the handler would let the rest
        escape the documented warn-and-fall-back contract.
        """
        from http.client import BadStatusLine, LineTooLong

        import diff_diff.datasets as datasets_mod

        for exc in (BadStatusLine("garbage status"), LineTooLong("header line")):
            fake_response = MagicMock()
            fake_response.read.side_effect = exc
            fake_response.__enter__ = lambda self: self
            fake_response.__exit__ = lambda self, *a: False

            monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path / type(exc).__name__)
            with patch("diff_diff.datasets.urlopen", return_value=fake_response):
                with pytest.warns(UserWarning, match="SYNTHETIC"):
                    result = load_mpdta(force_download=True)

            assert result.attrs["source"] == "synthetic_fallback"
            assert result.shape == _construct_mpdta_data().shape

    def test_failed_cache_write_leaves_no_partial_file(self, tmp_path, monkeypatch):
        """A write that fails mid-flight must not strand a ``delete=False`` temp file."""
        import diff_diff.datasets as datasets_mod

        real_ntf = datasets_mod.NamedTemporaryFile

        class FailingWrite:
            """Create the temp file for real, then fail the write (e.g. ENOSPC)."""

            def __init__(self, *args, **kwargs):
                self._f = real_ntf(*args, **kwargs)

            def __enter__(self):
                self._f.__enter__()
                return self

            def __exit__(self, *args):
                return self._f.__exit__(*args)

            @property
            def name(self):
                return self._f.name

            def write(self, data):
                raise OSError(28, "No space left on device")

        cache_path = tmp_path / "mpdta.csv"
        monkeypatch.setattr(datasets_mod, "NamedTemporaryFile", FailingWrite)

        with pytest.raises(datasets_mod._DatasetSourceError, match="Failed to cache"):
            datasets_mod._write_cache_atomically(cache_path, b"x" * 100, "mpdta")

        assert list(tmp_path.iterdir()) == [], "partial cache file was not cleaned up"

    def test_source_specific_dimensions_are_enforced(self):
        """Synthetic frames cannot pass as Card or Castle canonical data."""
        from diff_diff.datasets import (
            _validate_card_krueger_source,
            _validate_castle_doctrine_source,
        )

        with pytest.raises(RuntimeError, match="410 stores"):
            _validate_card_krueger_source(_construct_card_krueger_data())
        with pytest.raises(RuntimeError, match="50 states and 550 rows"):
            _validate_castle_doctrine_source(_construct_castle_doctrine_data())

    def test_card_source_transform_uses_fte_and_stable_duplicate_ids(self):
        """The public flat-file projection follows the published FTE formula."""
        from diff_diff.datasets import _prepare_card_krueger

        raw = pd.DataFrame(
            {
                "sheet": [407, 407],
                "state": [0, 1],
                "chain": [2, 4],
                "empft": [2.0, 5.0],
                "emppt": [10.0, 8.0],
                "nmgrs": [1.0, 2.0],
                "wage_st": [4.75, 5.75],
                "empft2": [1.0, 8.0],
                "emppt2": [12.0, 6.0],
                "nmgrs2": [2.0, 2.0],
                "wage_st2": [4.25, 5.50],
            }
        )

        result = _prepare_card_krueger(raw)

        assert result["store_id"].tolist() == [407, 408]
        assert result["state"].tolist() == ["PA", "NJ"]
        assert result["chain"].tolist() == ["kfc", "wendys"]
        assert result["emp_pre"].tolist() == [8.0, 11.0]
        assert result["emp_post"].tolist() == [9.0, 13.0]

    def test_castle_source_transform_preserves_fractional_cdl_exposure(self):
        """The binary treatment flag and fractional source exposure are distinct."""
        from diff_diff.datasets import _prepare_castle_doctrine

        raw = pd.DataFrame(
            {
                "state": ["Alabama", "Alabama"],
                "sid": [1, 1],
                "year": [2005, 2006],
                "effyear": [2006.0, 2006.0],
                "cdl": [0.0, 0.580822],
                "homicide": [7.0, 7.5],
                "population": [4_300_000, 4_350_000],
                "income": [44_000, 45_000],
            }
        )

        result = _prepare_castle_doctrine(raw)

        assert result["state"].tolist() == ["AL", "AL"]
        assert result["treated"].tolist() == [0, 1]
        assert result["treatment_exposure"].tolist() == [0.0, 0.580822]
        assert result["first_treat"].tolist() == [2006, 2006]

    def test_castle_adoption_year_has_binary_treatment_and_partial_exposure(self):
        """Real-source adoption rows retain their partial first-year exposure."""
        from diff_diff.datasets import _prepare_castle_doctrine

        raw = pd.DataFrame(
            {
                "sid": [1] * 11,
                "year": list(range(2000, 2011)),
                "effyear": [2006.0] * 11,
                "cdl": [0.0] * 6 + [0.580822] + [1.0] * 4,
                "homicide": [7.0] * 11,
                "population": [4_300_000] * 11,
                "income": [44_000] * 11,
            }
        )
        result = _prepare_castle_doctrine(raw)
        adoption_year = result.loc[result["year"] == 2006].iloc[0]

        assert adoption_year["treated"] == 1
        assert 0 < adoption_year["treatment_exposure"] < 1

    def test_card_source_rejects_unknown_chain_code(self):
        from diff_diff.datasets import _DatasetSourceError, _prepare_card_krueger

        raw = pd.DataFrame(
            {
                "sheet": [407, 407],
                "state": [0, 1],
                "chain": [2, 99],
                "empft": [2.0, 5.0],
                "emppt": [10.0, 8.0],
                "nmgrs": [1.0, 2.0],
                "wage_st": [4.75, 5.75],
                "empft2": [1.0, 8.0],
                "emppt2": [12.0, 6.0],
                "nmgrs2": [2.0, 2.0],
                "wage_st2": [4.25, 5.50],
            }
        )

        with pytest.raises(_DatasetSourceError, match="unknown chain"):
            _prepare_card_krueger(raw)

    def test_semantically_invalid_source_values_are_rejected(self):
        from diff_diff.datasets import (
            _DatasetSourceError,
            _validate_card_krueger_source,
            _validate_castle_doctrine_source,
            _validate_mpdta,
        )

        card = self._valid_card_source_frame()
        card.loc[100, "emp_change"] += 1
        with pytest.raises(_DatasetSourceError, match="emp_change"):
            _validate_card_krueger_source(card)

        card = self._valid_card_source_frame()
        card.loc[100, "emp_pre"] = np.inf
        card.loc[100, "emp_change"] = card.loc[100, "emp_post"] - np.inf
        with pytest.raises(_DatasetSourceError, match="emp_pre"):
            _validate_card_krueger_source(card)

        castle = self._valid_castle_source_frame()
        castle.loc[0, "homicide_rate"] = -1
        with pytest.raises(_DatasetSourceError, match="invalid outcome"):
            _validate_castle_doctrine_source(castle)

        mpdta = _construct_mpdta_data()
        mpdta["lemp"] = np.nan
        with pytest.raises(_DatasetSourceError, match="missing required"):
            _validate_mpdta(mpdta)


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


class TestCsvDownloadIntegrity:
    """CSV downloads receive the same trust-on-bytes contract as binary data."""

    def test_checksum_mismatch_raises_without_caching(self, tmp_path, monkeypatch):
        import diff_diff.datasets as datasets_mod

        monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path)

        fake_response = MagicMock()
        fake_response.read.return_value = b"tampered bytes"
        fake_response.__enter__ = lambda self: self
        fake_response.__exit__ = lambda self, *a: False

        with patch("diff_diff.datasets.urlopen", return_value=fake_response):
            with pytest.raises(RuntimeError, match="Checksum mismatch"):
                datasets_mod._download_with_cache(
                    "https://example.invalid/x.csv",
                    "x",
                    sha256="0" * 64,
                )

        assert not (tmp_path / "x.csv").exists()

    def test_stale_cache_triggers_verified_redownload(self, tmp_path, monkeypatch):
        import hashlib

        import diff_diff.datasets as datasets_mod

        monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path)
        good = b"a,b\n1,2\n"
        good_sha = hashlib.sha256(good).hexdigest()
        (tmp_path / "x.csv").write_bytes(b"stale bytes")

        fake_response = MagicMock()
        fake_response.read.return_value = good
        fake_response.__enter__ = lambda self: self
        fake_response.__exit__ = lambda self, *a: False

        with patch("diff_diff.datasets.urlopen", return_value=fake_response):
            content = datasets_mod._download_with_cache(
                "https://example.invalid/x.csv",
                "x",
                sha256=good_sha,
            )

        assert content == good.decode("utf-8")
        assert (tmp_path / "x.csv").read_bytes() == good

    def test_oversized_response_is_rejected_without_caching(self, tmp_path, monkeypatch):
        """A source cannot bypass checksum handling with an unbounded response."""
        import diff_diff.datasets as datasets_mod

        monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path)
        monkeypatch.setattr(datasets_mod, "_MAX_DATASET_BYTES", 4)
        fake_response = MagicMock()
        fake_response.read.return_value = b"12345"
        fake_response.__enter__ = lambda self: self
        fake_response.__exit__ = lambda self, *a: False

        with patch("diff_diff.datasets.urlopen", return_value=fake_response):
            with pytest.raises(RuntimeError, match="safety limit"):
                datasets_mod._download_with_cache(
                    "https://example.invalid/x.csv",
                    "x",
                    sha256="0" * 64,
                )

        assert not (tmp_path / "x.csv").exists()

    def test_oversized_cache_is_not_read(self, tmp_path, monkeypatch):
        """An oversized local cache cannot bypass the response-size limit."""
        import diff_diff.datasets as datasets_mod

        monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path)
        monkeypatch.setattr(datasets_mod, "_MAX_DATASET_BYTES", 4)
        (tmp_path / "x.csv").write_bytes(b"12345")

        with patch("diff_diff.datasets.urlopen", side_effect=TimeoutError("offline")):
            with pytest.raises(datasets_mod._DatasetSourceError, match="Failed to download"):
                datasets_mod._download_with_cache(
                    "https://example.invalid/x.csv",
                    "x",
                    sha256="0" * 64,
                )

    def test_failed_atomic_replace_returns_verified_download(self, tmp_path, monkeypatch):
        """An interrupted replacement must not discard verified fresh bytes."""
        import hashlib

        import diff_diff.datasets as datasets_mod

        monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path)
        good = b"a,b\n1,2\n"
        good_sha = hashlib.sha256(good).hexdigest()
        cache_path = tmp_path / "x.csv"
        cache_path.write_bytes(good)
        fake_response = MagicMock()
        fake_response.read.return_value = good
        fake_response.__enter__ = lambda self: self
        fake_response.__exit__ = lambda self, *a: False

        with (
            patch("diff_diff.datasets.urlopen", return_value=fake_response),
            patch("diff_diff.datasets.os.replace", side_effect=OSError("interrupted")),
        ):
            content = datasets_mod._download_with_cache(
                "https://example.invalid/x.csv",
                "x",
                sha256=good_sha,
                force_download=True,
            )

        assert content == good.decode("utf-8")
        assert cache_path.read_bytes() == good
        assert list(tmp_path.glob(".x.csv.*")) == []


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
