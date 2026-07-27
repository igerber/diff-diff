"""
Real-world datasets for Difference-in-Differences analysis.

This module provides functions to load classic econometrics datasets
commonly used for teaching and demonstrating DiD methods.

Canonical data are downloaded from checksum-pinned public sources and cached
locally. A download that fails verification falls back to a checksum-valid
cache entry when one exists, so verified data on disk is never displaced by
generated data. Only when no verified copy is available does the loader warn
and return an explicitly provenance-marked synthetic fallback.
"""

import hashlib
import os
import sys
import warnings
from http.client import HTTPException
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Dict, Optional, cast
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

# Cache directory for downloaded datasets
_CACHE_DIR = Path.home() / ".cache" / "diff_diff" / "datasets"
_MAX_DATASET_BYTES = 50 * 1024 * 1024

# These commit-pinned mirrors were verified byte-equivalent to their authoritative
# sources at pinning time: ``public.dat`` is byte-identical to the copy in
# ``njmin.zip`` from David Card's data archive, and ``mpdta.csv`` matches
# ``did::mpdta`` (R package 2.5.1) to CSV round-trip precision (max absolute
# difference 1.07e-14 on ``lemp``). Every cached and downloaded byte sequence is
# verified against the pinned SHA-256 below, so a later change at any mirror can
# never substitute different data: it falls back to the verified cache entry if one
# exists (with a warning naming the integrity failure), and to the loud synthetic
# fallback only when there is no verified copy to fall back to.
_CARD_KRUEGER_SOURCE_URL = (
    "https://raw.githubusercontent.com/rafiash/CardKrueger-stata-sample/"
    "07bc929f1d6552db117bd27a7cf0d881d16e9494/public.dat"
)
_CARD_KRUEGER_SOURCE_SHA256 = "04bde0cad5540980f32ce099c6dad369e2f05494698071d8a65b3e1cbe9ca53a"
_CASTLE_DOCTRINE_SOURCE_URL = (
    "https://raw.githubusercontent.com/scunning1975/mixtape/"
    "ca4279a87a6f0759f6b6f02841a53bdd68e27d3c/castle.dta"
)
_CASTLE_DOCTRINE_SOURCE_SHA256 = "804633c161827b6c0824f86f239046386d1a8266a866f83bf5ddb2aa762a5f29"
_MPDTA_SOURCE_URL = (
    "https://raw.githubusercontent.com/d2cml-ai/csdid/"
    "7ad707385354cb3924b8da94ef7e62a76bf55a4d/data/mpdta.csv"
)
_MPDTA_SOURCE_SHA256 = "2283bea1221a152420f98dfa20f633c5d054ea51d881115c8cd702a97bcd3167"

# ``sid`` follows alphabetical state order with 9 reserved for Washington, DC,
# which is absent from the source panel.
_CASTLE_STATE_BY_SID = dict(
    enumerate(
        """
        AL AK AZ AR CA CO CT DE _ FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN
        MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA
        WV WI WY
        """.split(),
        start=1,
    )
)


class _DatasetSourceError(RuntimeError):
    """Expected failure while fetching, parsing, or validating canonical data."""


def _caller_stacklevel() -> int:
    """``stacklevel`` that attributes a warning to the first frame outside this module.

    The text and binary loaders reach the download helper at different depths (the text
    path routes through ``_load_verified_dataset`` and a source adapter; the binary path
    calls it almost directly), so no fixed ``stacklevel`` points at user code for both.
    """
    level = 1
    try:
        frame = sys._getframe(1)
    except ValueError:  # pragma: no cover - defensive
        return 2
    while frame is not None:
        if frame.f_globals.get("__name__") != __name__:
            return level
        frame = frame.f_back
        level += 1
    return 2  # pragma: no cover - defensive


def _get_cache_path(name: str) -> Path:
    """Get the cache path for a dataset."""
    return _CACHE_DIR / f"{name}.csv"


def _download_with_cache(
    url: str,
    name: str,
    sha256: str,
    force_download: bool = False,
) -> str:
    """Download UTF-8 text, verify its checksum, and cache it."""
    cache_path = _get_cache_path(name)
    content = _download_verified_bytes(url, name, sha256, cache_path, force_download)
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _DatasetSourceError(
            f"Dataset '{name}' passed its byte checksum but is not valid UTF-8 text."
        ) from e


def _read_verified_cache(cache_path: Path, sha256: str) -> Optional[bytes]:
    """Return a bounded, checksum-valid cache entry or None."""
    try:
        if not cache_path.exists() or cache_path.stat().st_size > _MAX_DATASET_BYTES:
            return None
        content = cache_path.read_bytes()
    except OSError:
        return None
    if hashlib.sha256(content).hexdigest() == sha256:
        return content
    return None


def _write_cache_atomically(cache_path: Path, content: bytes, name: str) -> None:
    """Replace a cache entry only after a complete same-directory write."""
    temp_path: Optional[Path] = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            delete=False,
        ) as temp_file:
            # Bind the path before writing: ``delete=False`` means a failed write
            # still leaves the file on disk, and the handler below can only clean
            # up what it knows about.
            temp_path = Path(temp_file.name)
            temp_file.write(content)
        os.replace(temp_path, cache_path)
    except OSError as e:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise _DatasetSourceError(f"Failed to cache dataset '{name}': {e}") from e


def _download_verified_bytes(
    url: str,
    name: str,
    sha256: str,
    cache_path: Path,
    force_download: bool = False,
) -> bytes:
    """Return checksum-verified bytes from cache or a fresh download.

    The cache is read up front and retained even under ``force_download``, so that
    EVERY way a fresh download can fail verification - transport, size limit, or
    checksum - can still fall back to bytes that already passed the pinned hash.
    Falling through to the synthetic frame while verified canonical bytes sit on
    disk would be a downgrade, and on the checksum path it would let a tampered
    or moved upstream quietly replace real data with generated data.
    """
    cached = _read_verified_cache(cache_path, sha256)
    if cached is not None and not force_download:
        return cached

    def _recover(message: str, cause: Optional[BaseException] = None) -> bytes:
        """Prefer verified cached bytes over failing into the synthetic fallback."""
        if cached is not None:
            return cached
        raise _DatasetSourceError(message) from cause

    try:
        with urlopen(url, timeout=30) as response:
            content = response.read(_MAX_DATASET_BYTES + 1)
    # ``HTTPException`` is the parent of ``IncompleteRead``, ``BadStatusLine`` and the
    # rest of the protocol-level errors ``urlopen`` can surface; none of them derive
    # from ``OSError``, so catching the base class keeps the whole family inside the
    # documented warn-and-fall-back boundary rather than only the named siblings.
    except (HTTPError, HTTPException, OSError, TimeoutError, URLError) as e:
        return _recover(
            f"Failed to download dataset '{name}' from {url}: {e}\n"
            "Check your internet connection or try again later.",
            e,
        )

    if len(content) > _MAX_DATASET_BYTES:
        return _recover(
            f"Dataset '{name}' downloaded from {url} exceeds the "
            f"{_MAX_DATASET_BYTES}-byte safety limit."
        )

    if hashlib.sha256(content).hexdigest() != sha256:
        if cached is not None:
            # Canonical bytes are already on disk, so the user keeps real data - but a
            # pin mismatch is an integrity event, not a transport hiccup, and must not
            # pass unnoticed. Deliberately not a SYNTHETIC warning: nothing synthetic
            # is involved, and callers key their provenance checks on that word.
            warnings.warn(
                f"Upstream copy of dataset '{name}' at {url} no longer matches the "
                "pinned SHA-256. Returning the verified cached copy instead. Verify "
                "the source revision before updating the pinned checksum; until then "
                "treat the upstream file as untrusted.",
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )
            return cached
        raise _DatasetSourceError(
            f"Checksum mismatch for dataset '{name}' downloaded from {url}.\n"
            "The upstream file differs from the pinned SHA-256. Verify the "
            "source revision before updating the pinned checksum; otherwise "
            "treat the download as untrusted."
        )

    try:
        _write_cache_atomically(cache_path, content, name)
    except _DatasetSourceError:
        # Cache persistence is best-effort after the downloaded bytes have
        # already passed the pinned SHA-256 check.
        pass
    return content


def _get_cache_path_binary(name: str) -> Path:
    """Get the cache path for a binary (Stata .dta) dataset."""
    return _CACHE_DIR / f"{name}.dta"


def _download_with_cache_binary(
    url: str,
    name: str,
    sha256: str,
    force_download: bool = False,
) -> bytes:
    """Download a binary file (e.g. Stata .dta), verify its checksum, and cache it.

    Every byte-load (cache or fresh download) is verified against a pinned
    SHA-256. A stale or corrupt cache triggers one re-download. A checksum
    mismatch on freshly downloaded bytes falls back to a verified cache entry
    when one exists (warning that the upstream no longer matches the pin), and
    raises only when there is no verified copy to fall back to.
    """
    return _download_verified_bytes(
        url,
        name,
        sha256,
        _get_cache_path_binary(name),
        force_download,
    )


def _load_verified_dataset(
    *,
    cache_name: str,
    source: str,
    force_download: bool,
    load_source: Optional[Callable[[bool], pd.DataFrame]],
    prepare: Callable[[pd.DataFrame], pd.DataFrame],
    validate_source: Callable[[pd.DataFrame], None],
    validate_fallback: Callable[[pd.DataFrame], None],
    fallback: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Load and validate canonical data or return a loud synthetic fallback."""
    try:
        if load_source is None:
            raise _DatasetSourceError("no verified canonical source is configured")
        df = prepare(load_source(force_download))
        validate_source(df)
    except _DatasetSourceError as exc:
        warnings.warn(
            f"{cache_name} canonical data are unavailable ({exc}); returning a "
            "SYNTHETIC fallback. Check `df.attrs['source']` before treating "
            "the result as replication data.",
            UserWarning,
            stacklevel=3,
        )
        df = fallback()
        validate_fallback(df)
        df.attrs["source"] = "synthetic_fallback"
        return df

    df.attrs["source"] = source
    return df


def _load_card_krueger_source(force_download: bool) -> pd.DataFrame:
    """Load the checksum-pinned Card-Krueger public flat file."""
    content = _download_with_cache(
        _CARD_KRUEGER_SOURCE_URL,
        "card_krueger",
        _CARD_KRUEGER_SOURCE_SHA256,
        force_download,
    )
    columns = """
        sheet chain co_owned state southj centralj northj pa1 pa2 shore
        ncalls empft emppt nmgrs wage_st inctime firstinc bonus pctaff meals
        open hrsopen psoda pfry pentree nregs nregs11 type2 status2 date2
        ncalls2 empft2 emppt2 nmgrs2 wage_st2 inctime2 firstin2 special2
        meals2 open2r hrsopen2 psoda2 pfry2 pentree2 nregs2 nregs112
    """.split()
    try:
        return pd.read_csv(
            StringIO(content),
            sep=r"\s+",
            names=columns,
            na_values=".",
        )
    except (TypeError, ValueError) as e:
        raise _DatasetSourceError(f"Failed to parse Card-Krueger source data: {e}") from e


def _load_castle_doctrine_source(force_download: bool) -> pd.DataFrame:
    """Load the checksum-pinned Cheng-Hoekstra Stata data."""
    content = _download_with_cache_binary(
        _CASTLE_DOCTRINE_SOURCE_URL,
        "castle_doctrine",
        _CASTLE_DOCTRINE_SOURCE_SHA256,
        force_download,
    )
    try:
        return pd.read_stata(BytesIO(content))
    except (OSError, TypeError, ValueError) as e:
        raise _DatasetSourceError(f"Failed to parse Castle Doctrine source data: {e}") from e


def _load_mpdta_source(force_download: bool) -> pd.DataFrame:
    """Load the checksum-pinned Callaway-Sant'Anna example data."""
    content = _download_with_cache(
        _MPDTA_SOURCE_URL,
        "mpdta",
        _MPDTA_SOURCE_SHA256,
        force_download,
    )
    try:
        return pd.read_csv(StringIO(content))
    except (TypeError, ValueError) as e:
        raise _DatasetSourceError(f"Failed to parse mpdta source data: {e}") from e


def _require_columns(df: pd.DataFrame, dataset: str, columns: set) -> None:
    """Reject empty or structurally incomplete downloaded datasets."""
    if df.empty:
        raise _DatasetSourceError(f"{dataset} source is empty")
    missing = columns - set(df.columns)
    if missing:
        raise _DatasetSourceError(f"{dataset} source is missing columns: {sorted(missing)}")


def _identity_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return an already-normalized dataset unchanged."""
    return df


def _require_complete(df: pd.DataFrame, dataset: str, columns: set) -> None:
    """Reject missing values in columns whose public contract is complete."""
    missing = df[list(columns)].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise _DatasetSourceError(
            f"{dataset} source has missing required values: {missing.to_dict()}"
        )


def _require_finite(df: pd.DataFrame, dataset: str, columns: set) -> None:
    """Reject non-numeric or non-finite values in numeric contract columns."""
    try:
        values = df[list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as e:
        raise _DatasetSourceError(f"{dataset} source has non-numeric values") from e
    if not np.isfinite(values).all():
        raise _DatasetSourceError(f"{dataset} source has non-finite values in {sorted(columns)}")


def _validate_panel_keys(df: pd.DataFrame, dataset: str, unit: str) -> None:
    """Validate the common unit-time and cohort invariants for panel datasets."""
    if df.duplicated([unit, "year"]).any():
        raise _DatasetSourceError(f"{dataset} source has duplicate {unit}-year rows")
    cohort_counts = df.groupby(unit)["first_treat"].nunique(dropna=False)
    if not (cohort_counts == 1).all():
        raise _DatasetSourceError(f"{dataset} first_treat is not constant within {unit}")
    if not (df["cohort"] == df["first_treat"]).all():
        raise _DatasetSourceError(f"{dataset} cohort does not match first_treat")
    expected_treated = ((df["first_treat"] > 0) & (df["year"] >= df["first_treat"])).astype(int)
    if not (df["treated"] == expected_treated).all():
        raise _DatasetSourceError(f"{dataset} treated indicator is inconsistent with first_treat")


def _prepare_card_krueger(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Card-Krueger source frame to the public loader schema."""
    raw_columns = {
        "sheet",
        "state",
        "chain",
        "empft",
        "emppt",
        "nmgrs",
        "wage_st",
        "empft2",
        "emppt2",
        "nmgrs2",
        "wage_st2",
    }
    if raw_columns <= set(df.columns):
        store_id = df["sheet"].copy()
        _require_complete(df, "card_krueger", {"sheet", "state", "chain"})
        _require_finite(df, "card_krueger", {"sheet", "state", "chain"})
        if not set(df["state"].unique()) <= {0, 1}:
            raise _DatasetSourceError("card_krueger source has unknown state codes")
        if not set(df["chain"].unique()) <= {1, 2, 3, 4}:
            raise _DatasetSourceError("card_krueger source has unknown chain codes")
        for column in (
            "empft",
            "emppt",
            "nmgrs",
            "wage_st",
            "empft2",
            "emppt2",
            "nmgrs2",
            "wage_st2",
        ):
            converted = pd.to_numeric(df[column], errors="coerce")
            if converted.notna().sum() != df[column].notna().sum():
                raise _DatasetSourceError(f"card_krueger source has non-numeric values in {column}")
            df[column] = converted
        duplicate_407 = store_id == 407
        if (
            duplicate_407.sum() != 2
            or set(df.loc[duplicate_407, "state"]) != {0, 1}
            or (store_id == 408).any()
        ):
            raise _DatasetSourceError(
                "card_krueger source does not match the documented duplicate-407 convention"
            )
        store_id.loc[duplicate_407 & (df["state"] == 1)] = 408
        emp_pre = df["empft"] + df["nmgrs"] + 0.5 * df["emppt"]
        emp_post = df["empft2"] + df["nmgrs2"] + 0.5 * df["emppt2"]
        return pd.DataFrame(
            {
                "store_id": store_id.astype(int),
                "state": np.where(df["state"] == 1, "NJ", "PA"),
                "chain": df["chain"].map({1: "bk", 2: "kfc", 3: "roys", 4: "wendys"}),
                "emp_pre": emp_pre,
                "emp_post": emp_post,
                "wage_pre": df["wage_st"],
                "wage_post": df["wage_st2"],
                "treated": (df["state"] == 1).astype(int),
                "emp_change": emp_post - emp_pre,
            }
        )

    df = df.rename(columns={"sheet": "store_id"}).copy()
    if "state" not in df.columns and "nj" in df.columns:
        df["state"] = np.where(df["nj"] == 1, "NJ", "PA")
    if "treated" not in df.columns and "state" in df.columns:
        df["treated"] = (df["state"] == "NJ").astype(int)
    if "emp_change" not in df.columns and {"emp_post", "emp_pre"} <= set(df.columns):
        df["emp_change"] = df["emp_post"] - df["emp_pre"]
    return df


def _validate_card_krueger(df: pd.DataFrame) -> None:
    """Validate the documented Card-Krueger wide-data contract."""
    _require_columns(
        df,
        "card_krueger",
        {
            "store_id",
            "state",
            "chain",
            "emp_pre",
            "emp_post",
            "wage_pre",
            "wage_post",
            "treated",
            "emp_change",
        },
    )
    _require_complete(df, "card_krueger", {"store_id", "state", "chain", "treated"})
    _require_finite(df, "card_krueger", {"store_id", "treated"})
    if df["store_id"].duplicated().any():
        raise _DatasetSourceError("card_krueger source has duplicate store_id values")
    if set(df["state"].dropna().unique()) != {"NJ", "PA"}:
        raise _DatasetSourceError("card_krueger source must contain both NJ and PA only")
    if set(df["chain"].dropna().unique()) != {"bk", "kfc", "roys", "wendys"}:
        raise _DatasetSourceError("card_krueger source has unexpected restaurant chains")
    for column in ("emp_pre", "emp_post", "wage_pre", "wage_post"):
        values = pd.to_numeric(df[column], errors="coerce")
        if (
            values.notna().sum() != df[column].notna().sum()
            or not np.isfinite(values.dropna()).all()
            or (values.dropna() < 0).any()
        ):
            raise _DatasetSourceError(
                f"card_krueger source has invalid non-negative values in {column}"
            )
    emp_change = pd.to_numeric(df["emp_change"], errors="coerce")
    if (
        emp_change.notna().sum() != df["emp_change"].notna().sum()
        or not np.isfinite(emp_change.dropna()).all()
    ):
        raise _DatasetSourceError("card_krueger source has invalid emp_change values")
    expected_treated = (df["state"] == "NJ").astype(int)
    if not (df["treated"] == expected_treated).all():
        raise _DatasetSourceError("card_krueger treated indicator is inconsistent with state")
    expected_change = df["emp_post"] - df["emp_pre"]
    if not np.allclose(df["emp_change"], expected_change, equal_nan=True):
        raise _DatasetSourceError(
            "card_krueger emp_change is inconsistent with emp_pre and emp_post"
        )


def _validate_card_krueger_source(df: pd.DataFrame) -> None:
    """Validate source-specific Card-Krueger counts and categories."""
    _validate_card_krueger(df)
    if len(df) != 410:
        raise _DatasetSourceError("card_krueger source must contain 410 stores")
    if df.groupby("state").size().to_dict() != {"NJ": 331, "PA": 79}:
        raise _DatasetSourceError("card_krueger source has unexpected state counts")
    expected_missing = {
        "emp_pre": 12,
        "emp_post": 14,
        "wage_pre": 20,
        "wage_post": 21,
        "emp_change": 26,
    }
    if df[list(expected_missing)].isna().sum().to_dict() != expected_missing:
        raise _DatasetSourceError("card_krueger source has unexpected missing-value counts")


def _prepare_castle_doctrine(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Castle Doctrine source frame to the public loader schema."""
    df = df.copy()
    if "sid" in df.columns:
        state_codes = df["sid"].map(_CASTLE_STATE_BY_SID)
        if state_codes.notna().all() and not (state_codes == "_").any():
            df["state"] = state_codes
    if "first_treat" not in df.columns and "effyear" in df.columns:
        try:
            df["first_treat"] = df["effyear"].fillna(0).astype(int)
        except (TypeError, ValueError) as e:
            raise _DatasetSourceError("castle_doctrine source has invalid effyear values") from e
    if "cohort" not in df.columns and "first_treat" in df.columns:
        df["cohort"] = df["first_treat"]
    if {"first_treat", "year"} <= set(df.columns):
        df["treated"] = ((df["first_treat"] > 0) & (df["year"] >= df["first_treat"])).astype(int)
    if "treatment_exposure" not in df.columns and "cdl" in df.columns:
        df["treatment_exposure"] = df["cdl"]
    if "homicide_rate" not in df.columns and "homicide" in df.columns:
        df["homicide_rate"] = df["homicide"]
    if {
        "state",
        "year",
        "first_treat",
        "homicide_rate",
        "population",
        "income",
        "treated",
        "treatment_exposure",
        "cohort",
    } <= set(df.columns):
        return df[
            [
                "state",
                "year",
                "first_treat",
                "homicide_rate",
                "population",
                "income",
                "treated",
                "treatment_exposure",
                "cohort",
            ]
        ].copy()
    return df


def _validate_castle_doctrine(df: pd.DataFrame) -> None:
    """Validate the documented Castle Doctrine panel contract."""
    _require_columns(
        df,
        "castle_doctrine",
        {
            "state",
            "year",
            "first_treat",
            "homicide_rate",
            "population",
            "income",
            "treated",
            "treatment_exposure",
            "cohort",
        },
    )
    _require_complete(
        df,
        "castle_doctrine",
        {
            "state",
            "year",
            "first_treat",
            "homicide_rate",
            "population",
            "income",
            "treated",
            "cohort",
        },
    )
    _require_finite(
        df,
        "castle_doctrine",
        {
            "year",
            "first_treat",
            "homicide_rate",
            "population",
            "income",
            "treated",
            "treatment_exposure",
            "cohort",
        },
    )
    if (
        (df["homicide_rate"] < 0).any()
        or (df["population"] <= 0).any()
        or (df["income"] <= 0).any()
    ):
        raise _DatasetSourceError("castle_doctrine source has invalid outcome or covariate values")
    if not df["state"].astype(str).str.fullmatch(r"[A-Z]{2}").all():
        raise _DatasetSourceError("castle_doctrine source has invalid state abbreviations")
    if not df["treatment_exposure"].between(0, 1).all():
        raise _DatasetSourceError("castle_doctrine treatment_exposure must be between 0 and 1")
    _validate_panel_keys(df, "castle_doctrine", "state")


def _validate_castle_doctrine_source(df: pd.DataFrame) -> None:
    """Validate source-specific Castle Doctrine panel dimensions."""
    _validate_castle_doctrine(df)
    if len(df) != 550 or df["state"].nunique() != 50:
        raise _DatasetSourceError("castle_doctrine source must contain 50 states and 550 rows")
    if set(df["year"].unique()) != set(range(2000, 2011)):
        raise _DatasetSourceError("castle_doctrine source has unexpected years")
    if set(df["first_treat"].unique()) != {0, 2005, 2006, 2007, 2008, 2009}:
        raise _DatasetSourceError("castle_doctrine source has unexpected treatment cohorts")


def _validate_divorce_laws(df: pd.DataFrame) -> None:
    """Validate the documented divorce-laws panel contract."""
    _require_columns(
        df,
        "divorce_laws",
        {
            "state",
            "year",
            "first_treat",
            "divorce_rate",
            "female_lfp",
            "suicide_rate",
            "treated",
            "cohort",
        },
    )
    _require_complete(
        df,
        "divorce_laws",
        {
            "state",
            "year",
            "first_treat",
            "divorce_rate",
            "female_lfp",
            "suicide_rate",
            "treated",
            "cohort",
        },
    )
    _require_finite(
        df,
        "divorce_laws",
        {
            "year",
            "first_treat",
            "divorce_rate",
            "female_lfp",
            "suicide_rate",
            "treated",
            "cohort",
        },
    )
    if (df["divorce_rate"] < 0).any() or (df["suicide_rate"] < 0).any():
        raise _DatasetSourceError("divorce_laws source has negative outcome values")
    if not df["female_lfp"].between(0, 1).all():
        raise _DatasetSourceError("divorce_laws female_lfp must be between 0 and 1")
    _validate_panel_keys(df, "divorce_laws", "state")


def _prepare_mpdta(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an mpdta source frame to the public loader schema."""
    if "first.treat" in df.columns:
        df = df.rename(columns={"first.treat": "first_treat"})
    if "cohort" not in df.columns and "first_treat" in df.columns:
        df["cohort"] = df["first_treat"]
    columns = ["countyreal", "year", "lpop", "lemp", "first_treat", "treat", "cohort"]
    if set(columns) <= set(df.columns):
        return df[columns].copy()
    return df


def _validate_mpdta(df: pd.DataFrame) -> None:
    """Validate the canonical R did::mpdta panel structure."""
    _require_columns(
        df,
        "mpdta",
        {"countyreal", "year", "lpop", "lemp", "first_treat", "treat", "cohort"},
    )
    _require_complete(
        df,
        "mpdta",
        {"countyreal", "year", "lpop", "lemp", "first_treat", "treat", "cohort"},
    )
    _require_finite(
        df,
        "mpdta",
        {"countyreal", "year", "lpop", "lemp", "first_treat", "treat", "cohort"},
    )
    if df.duplicated(["countyreal", "year"]).any():
        raise _DatasetSourceError("mpdta source has duplicate county-year rows")
    if len(df) != 2500 or df["countyreal"].nunique() != 500:
        raise _DatasetSourceError("mpdta source must contain 500 counties and 2500 rows")
    if set(df["year"].unique()) != {2003, 2004, 2005, 2006, 2007}:
        raise _DatasetSourceError("mpdta source has unexpected years")
    if set(df["first_treat"].unique()) != {0, 2004, 2006, 2007}:
        raise _DatasetSourceError("mpdta source has unexpected treatment cohorts")
    cohort_counts = df.groupby("countyreal")["first_treat"].nunique(dropna=False)
    if not (cohort_counts == 1).all():
        raise _DatasetSourceError("mpdta first_treat is not constant within county")
    if not (df["cohort"] == df["first_treat"]).all():
        raise _DatasetSourceError("mpdta cohort does not match first_treat")
    if not (df["treat"] == (df["first_treat"] > 0).astype(int)).all():
        raise _DatasetSourceError("mpdta treat indicator is inconsistent with first_treat")


def clear_cache() -> None:
    """Clear the local dataset cache.

    Also removes any ``.<name>.<ext>.<suffix>`` scratch files left behind by an
    atomic cache write that was interrupted between creating the temporary file
    and replacing the cache entry (a hard kill, for instance). Those are hidden
    and do not match the plain ``*.csv`` / ``*.dta`` patterns, so without this
    they would accumulate and survive the documented remedy.
    """
    if _CACHE_DIR.exists():
        for pattern in ("*.csv", "*.dta", ".*.csv.*", ".*.dta.*"):
            for f in _CACHE_DIR.glob(pattern):
                if f.is_file():
                    f.unlink()
        print(f"Cleared cache at {_CACHE_DIR}")


def load_card_krueger(force_download: bool = False) -> pd.DataFrame:
    """
    Load the Card & Krueger (1994) minimum wage dataset.

    This classic dataset examines the effect of New Jersey's 1992 minimum wage
    increase on employment in fast-food restaurants, using Pennsylvania as
    a control group.

    The study is a canonical example of the Difference-in-Differences method.

    Parameters
    ----------
    force_download : bool, default=False
        If True, re-download the dataset even if cached.

    Returns
    -------
    pd.DataFrame
        Dataset with columns:
        - store_id : int - Unique store identifier
        - state : str - 'NJ' (New Jersey, treated) or 'PA' (Pennsylvania, control)
        - chain : str - Fast food chain ('bk', 'kfc', 'roys', 'wendys')
        - emp_pre : float - Full-time equivalent employment before (Feb 1992)
        - emp_post : float - Full-time equivalent employment after (Nov 1992)
        - wage_pre : float - Starting wage before
        - wage_post : float - Starting wage after
        - treated : int - 1 if NJ, 0 if PA
        - emp_change : float - Change in employment (emp_post - emp_pre)

    Notes
    -----
    The minimum wage in New Jersey increased from $4.25 to $5.05 on April 1, 1992.
    Pennsylvania's minimum wage remained at $4.25.

    Original finding: No significant negative effect of minimum wage increase
    on employment (ATT ≈ +2.8 FTE employees).

    The canonical survey is incomplete: 12 stores lack ``emp_pre``, 14 lack
    ``emp_post``, and ``wage_pre``/``wage_post`` are missing for 20 and 21
    stores. Drop the missing outcome rows before fitting, as the example below
    does; estimators reject missing outcomes rather than dropping them silently.
    The synthetic fallback frame is complete, so code that skips this step will
    work offline and fail once the canonical source is reachable.

    The canonical data are checksum-verified and returned with
    ``df.attrs["source"] == "card_krueger_public_data"``. A download failure
    falls back to a checksum-valid cache entry when one exists, returning that
    canonical data; a pin mismatch additionally warns that the upstream file has
    changed. Only when neither a verified cache entry nor a verified fresh
    download is available - or when the source fails to parse or validate - does
    the loader emit one ``UserWarning`` containing ``SYNTHETIC``
    and return ``df.attrs["source"] == "synthetic_fallback"``.

    References
    ----------
    Card, D., & Krueger, A. B. (1994). Minimum Wages and Employment: A Case Study
    of the Fast-Food Industry in New Jersey and Pennsylvania. *American Economic
    Review*, 84(4), 772-793.

    Examples
    --------
    >>> from diff_diff.datasets import load_card_krueger
    >>> from diff_diff import DifferenceInDifferences
    >>>
    >>> # Load and prepare data
    >>> ck = load_card_krueger()
    >>> ck_long = ck.melt(
    ...     id_vars=['store_id', 'state', 'treated'],
    ...     value_vars=['emp_pre', 'emp_post'],
    ...     var_name='period', value_name='employment'
    ... )
    >>> ck_long['post'] = (ck_long['period'] == 'emp_post').astype(int)
    >>>
    >>> # 26 store-waves have no employment reading in the source survey
    >>> ck_long = ck_long.dropna(subset=['employment'])
    >>>
    >>> # Estimate DiD
    >>> did = DifferenceInDifferences()
    >>> results = did.fit(ck_long, outcome='employment', treatment='treated', time='post')
    """
    return _load_verified_dataset(
        cache_name="card_krueger",
        source="card_krueger_public_data",
        force_download=force_download,
        load_source=_load_card_krueger_source,
        prepare=_prepare_card_krueger,
        validate_source=_validate_card_krueger_source,
        validate_fallback=_validate_card_krueger,
        fallback=_construct_card_krueger_data,
    )


def _construct_card_krueger_data() -> pd.DataFrame:
    """
    Construct Card-Krueger dataset from summary statistics.

    This is a fallback when the online source is unavailable.
    Uses aggregated data that preserves the key DiD estimates.
    """
    # Representative sample based on published summary statistics
    np.random.seed(1994)  # Card-Krueger publication year, for reproducibility

    stores = []
    store_id = 1

    # New Jersey stores (treated) - summary stats from paper
    # Mean emp before: 20.44, after: 21.03
    # Mean wage before: 4.61, after: 5.08
    for chain in ["bk", "kfc", "roys", "wendys"]:
        n_stores = {"bk": 85, "kfc": 62, "roys": 48, "wendys": 36}[chain]
        for _ in range(n_stores):
            emp_pre = np.random.normal(20.44, 8.5)
            emp_post = emp_pre + np.random.normal(0.59, 7.0)  # Change ≈ 0.59
            emp_pre = max(0, emp_pre)
            emp_post = max(0, emp_post)

            stores.append(
                {
                    "store_id": store_id,
                    "state": "NJ",
                    "chain": chain,
                    "emp_pre": round(emp_pre, 1),
                    "emp_post": round(emp_post, 1),
                    "wage_pre": round(np.random.normal(4.61, 0.35), 2),
                    "wage_post": round(np.random.normal(5.08, 0.12), 2),
                }
            )
            store_id += 1

    # Pennsylvania stores (control) - summary stats from paper
    # Mean emp before: 23.33, after: 21.17
    # Mean wage before: 4.63, after: 4.62
    for chain in ["bk", "kfc", "roys", "wendys"]:
        n_stores = {"bk": 30, "kfc": 20, "roys": 14, "wendys": 15}[chain]
        for _ in range(n_stores):
            emp_pre = np.random.normal(23.33, 8.2)
            emp_post = emp_pre + np.random.normal(-2.16, 7.0)  # Change ≈ -2.16
            emp_pre = max(0, emp_pre)
            emp_post = max(0, emp_post)

            stores.append(
                {
                    "store_id": store_id,
                    "state": "PA",
                    "chain": chain,
                    "emp_pre": round(emp_pre, 1),
                    "emp_post": round(emp_post, 1),
                    "wage_pre": round(np.random.normal(4.63, 0.35), 2),
                    "wage_post": round(np.random.normal(4.62, 0.35), 2),
                }
            )
            store_id += 1

    df = pd.DataFrame(stores)
    df["treated"] = (df["state"] == "NJ").astype(int)
    df["emp_change"] = df["emp_post"] - df["emp_pre"]
    return df


def load_castle_doctrine(force_download: bool = False) -> pd.DataFrame:
    """
    Load Castle Doctrine / Stand Your Ground laws dataset.

    This dataset tracks the staggered adoption of Castle Doctrine (Stand Your
    Ground) laws across U.S. states, which expanded self-defense rights.
    It's commonly used to demonstrate heterogeneous treatment timing methods
    like Callaway-Sant'Anna or Sun-Abraham.

    Parameters
    ----------
    force_download : bool, default=False
        If True, re-download the dataset even if cached.

    Returns
    -------
    pd.DataFrame
        Panel dataset with columns:
        - state : str - State abbreviation
        - year : int - Year (2000-2010)
        - first_treat : int - Year of law adoption (0 = never adopted)
        - homicide_rate : float - Homicides per 100,000 population
        - population : int - State population
        - income : float - State median income
        - treated : int - 1 if law in effect, 0 otherwise
        - treatment_exposure : float - Fraction of the year the law was in effect
        - cohort : int - Alias for first_treat

    Notes
    -----
    Castle Doctrine laws remove the duty to retreat before using deadly force
    in self-defense. States adopted these laws at different times between
    2005 and 2009, creating a staggered treatment design.

    The canonical data are checksum-verified and returned with
    ``df.attrs["source"] == "cheng_hoekstra_castle_data"``. A download failure
    falls back to a checksum-valid cache entry when one exists, returning that
    canonical data; a pin mismatch additionally warns that the upstream file has
    changed. Only when neither a verified cache entry nor a verified fresh
    download is available - or when the source fails to parse or validate - does
    the loader emit one ``UserWarning`` containing ``SYNTHETIC``
    and mark the returned frame as ``"synthetic_fallback"``.

    ``treatment_exposure`` is fractional only on canonical frames; synthetic
    fallback frames set it to a binary 0/1 copy of ``treated`` and therefore
    carry no partial-year information.

    Replicating Cheng-Hoekstra (2013) requires the paper's regressor and outcome:
    regress ``log(homicide_rate)`` on ``treatment_exposure`` (their ``CDL_it``,
    the proportion of the year the law was in effect), not the binary ``treated``.
    See "Castle Doctrine treatment coding" in ``docs/methodology/REGISTRY.md``.

    References
    ----------
    Cheng, C., & Hoekstra, M. (2013). Does Strengthening Self-Defense Law Deter
    Crime or Escalate Violence? Evidence from Expansions to Castle Doctrine.
    *Journal of Human Resources*, 48(3), 821-854.

    Examples
    --------
    >>> from diff_diff.datasets import load_castle_doctrine
    >>> from diff_diff import CallawaySantAnna
    >>>
    >>> castle = load_castle_doctrine()
    >>> cs = CallawaySantAnna(control_group="never_treated")
    >>> results = cs.fit(
    ...     castle,
    ...     outcome="homicide_rate",
    ...     unit="state",
    ...     time="year",
    ...     first_treat="first_treat"
    ... )
    """
    return _load_verified_dataset(
        cache_name="castle_doctrine",
        source="cheng_hoekstra_castle_data",
        force_download=force_download,
        load_source=_load_castle_doctrine_source,
        prepare=_prepare_castle_doctrine,
        validate_source=_validate_castle_doctrine_source,
        validate_fallback=_validate_castle_doctrine,
        fallback=_construct_castle_doctrine_data,
    )


def _construct_castle_doctrine_data() -> pd.DataFrame:
    """
    Construct Castle Doctrine dataset from documented patterns.

    This is a fallback when the online source is unavailable.
    """
    np.random.seed(2013)  # Cheng-Hoekstra publication year, for reproducibility

    # States and their Castle Doctrine adoption years
    # 0 = never adopted during the study period
    state_adoption = {
        "AL": 2006,
        "AK": 2006,
        "AZ": 2006,
        "FL": 2005,
        "GA": 2006,
        "IN": 2006,
        "KS": 2006,
        "KY": 2006,
        "LA": 2006,
        "MI": 2006,
        "MS": 2006,
        "MO": 2007,
        "MT": 2009,
        "NH": 2011,
        "NC": 2011,
        "ND": 2007,
        "OH": 2008,
        "OK": 2006,
        "PA": 2011,
        "SC": 2006,
        "SD": 2006,
        "TN": 2007,
        "TX": 2007,
        "UT": 2010,
        "WV": 2008,
        # Control states (never adopted or adopted after 2010)
        "CA": 0,
        "CO": 0,
        "CT": 0,
        "DE": 0,
        "HI": 0,
        "ID": 0,
        "IL": 0,
        "IA": 0,
        "ME": 0,
        "MD": 0,
        "MA": 0,
        "MN": 0,
        "NE": 0,
        "NV": 0,
        "NJ": 0,
        "NM": 0,
        "NY": 0,
        "OR": 0,
        "RI": 0,
        "VT": 0,
        "VA": 0,
        "WA": 0,
        "WI": 0,
        "WY": 0,
    }

    # Only include states that adopted before or during 2010, or never adopted
    state_adoption = {k: (v if v <= 2010 else 0) for k, v in state_adoption.items()}

    data = []
    for state, first_treat in state_adoption.items():
        # State-level baseline characteristics
        base_homicide = np.random.uniform(3.0, 8.0)
        pop = np.random.randint(500000, 20000000)
        base_income = np.random.uniform(30000, 50000)

        for year in range(2000, 2011):
            # Time trend
            time_effect = (year - 2005) * 0.1

            # Treatment effect (approximately +8% increase in homicide rate)
            if first_treat > 0 and year >= first_treat:
                treatment_effect = base_homicide * 0.08
            else:
                treatment_effect = 0

            homicide = max(
                0, base_homicide + time_effect + treatment_effect + np.random.normal(0, 0.5)
            )

            data.append(
                {
                    "state": state,
                    "year": year,
                    "first_treat": first_treat,
                    "homicide_rate": round(homicide, 2),
                    "population": pop + year * 10000 + np.random.randint(-5000, 5000),
                    "income": round(
                        base_income * (1 + 0.02 * (year - 2000)) + np.random.normal(0, 1000), 0
                    ),
                    "treated": int(first_treat > 0 and year >= first_treat),
                    "treatment_exposure": float(first_treat > 0 and year >= first_treat),
                }
            )

    df = pd.DataFrame(data)
    df["cohort"] = df["first_treat"]
    return df


def load_divorce_laws(force_download: bool = False) -> pd.DataFrame:
    """
    Load the synthetic-only unilateral divorce-laws dataset.

    This dataset tracks the staggered adoption of unilateral (no-fault) divorce
    laws across U.S. states. It's a classic example for studying staggered
    DiD methods and was used in Stevenson & Wolfers (2006).

    Parameters
    ----------
    force_download : bool, default=False
        Retained for API compatibility. No verified source currently satisfies
        the loader's composite schema.

    Returns
    -------
    pd.DataFrame
        Panel dataset with columns:
        - state : str - State abbreviation
        - year : int - Year
        - first_treat : int - Year unilateral divorce became available (0 = never)
        - divorce_rate : float - Divorces per 1,000 population
        - female_lfp : float - Female labor force participation rate
        - suicide_rate : float - Female suicide rate
        - treated : int - 1 if law in effect, 0 otherwise
        - cohort : int - Alias for first_treat

    Notes
    -----
    Unilateral divorce laws allow one spouse to obtain a divorce without the
    other's consent. States adopted these laws at different times, primarily
    between 1969 and 1985.

    No verified source currently reproduces all documented columns without
    deriving new variables or changing pre-panel treatment semantics. This
    loader therefore emits one ``UserWarning`` containing ``SYNTHETIC`` and
    returns ``df.attrs["source"] == "synthetic_fallback"``.

    References
    ----------
    Stevenson, B., & Wolfers, J. (2006). Bargaining in the Shadow of the Law:
    Divorce Laws and Family Distress. *Quarterly Journal of Economics*,
    121(1), 267-288.

    Wolfers, J. (2006). Did Unilateral Divorce Laws Raise Divorce Rates?
    A Reconciliation and New Results. *American Economic Review*, 96(5), 1802-1820.

    Examples
    --------
    >>> from diff_diff.datasets import load_divorce_laws
    >>> from diff_diff import CallawaySantAnna, SunAbraham
    >>>
    >>> divorce = load_divorce_laws()
    >>> cs = CallawaySantAnna(control_group="never_treated")
    >>> results = cs.fit(
    ...     divorce,
    ...     outcome="divorce_rate",
    ...     unit="state",
    ...     time="year",
    ...     first_treat="first_treat"
    ... )
    """
    return _load_verified_dataset(
        cache_name="divorce_laws",
        source="stevenson_wolfers_divorce_data",
        force_download=force_download,
        load_source=None,
        prepare=_identity_dataset,
        validate_source=_validate_divorce_laws,
        validate_fallback=_validate_divorce_laws,
        fallback=_construct_divorce_laws_data,
    )


def _construct_divorce_laws_data() -> pd.DataFrame:
    """
    Construct divorce laws dataset from documented patterns.

    This is a fallback when the online source is unavailable.
    """
    np.random.seed(2006)  # Stevenson-Wolfers publication year, for reproducibility

    # State adoption years for unilateral divorce (from Wolfers 2006)
    # 0 = never adopted or adopted before 1968
    state_adoption = {
        "AK": 1935,
        "AL": 1971,
        "AZ": 1973,
        "CA": 1970,
        "CO": 1972,
        "CT": 1973,
        "DE": 1968,
        "FL": 1971,
        "GA": 1973,
        "HI": 1973,
        "IA": 1970,
        "ID": 1971,
        "IN": 1973,
        "KS": 1969,
        "KY": 1972,
        "MA": 1975,
        "ME": 1973,
        "MI": 1972,
        "MN": 1974,
        "MO": 0,
        "MT": 1975,
        "NC": 0,
        "ND": 1971,
        "NE": 1972,
        "NH": 1971,
        "NJ": 0,
        "NM": 1973,
        "NV": 1967,
        "NY": 0,
        "OH": 0,
        "OK": 1975,
        "OR": 1971,
        "PA": 0,
        "RI": 1975,
        "SD": 1985,
        "TN": 0,
        "TX": 1970,
        "UT": 1987,
        "VA": 0,
        "WA": 1973,
        "WI": 1978,
        "WV": 1984,
        "WY": 1977,
    }

    # Filter to states with adoption dates in our range or never adopted
    state_adoption = {k: v for k, v in state_adoption.items() if v == 0 or (1968 <= v <= 1990)}

    data = []
    for state, first_treat in state_adoption.items():
        # State-level baselines
        base_divorce = np.random.uniform(2.0, 6.0)
        base_lfp = np.random.uniform(0.35, 0.55)
        base_suicide = np.random.uniform(4.0, 8.0)

        for year in range(1968, 1989):
            # Time trends
            time_trend = (year - 1978) * 0.05

            # Treatment effects (from literature)
            # Short-run increase in divorce rate, then return to trend
            if first_treat > 0 and year >= first_treat:
                years_since = year - first_treat
                # Initial spike then fade out
                if years_since <= 2:
                    divorce_effect = 0.5
                elif years_since <= 5:
                    divorce_effect = 0.3
                elif years_since <= 10:
                    divorce_effect = 0.1
                else:
                    divorce_effect = 0.0
                # Small positive effect on female LFP
                lfp_effect = 0.02
                # Reduction in female suicide
                suicide_effect = -0.5
            else:
                divorce_effect = 0
                lfp_effect = 0
                suicide_effect = 0

            data.append(
                {
                    "state": state,
                    "year": year,
                    "first_treat": first_treat if first_treat >= 1968 else 0,
                    "divorce_rate": round(
                        max(
                            0, base_divorce + time_trend + divorce_effect + np.random.normal(0, 0.3)
                        ),
                        2,
                    ),
                    "female_lfp": round(
                        min(
                            1,
                            max(
                                0,
                                base_lfp
                                + 0.01 * (year - 1968)
                                + lfp_effect
                                + np.random.normal(0, 0.02),
                            ),
                        ),
                        3,
                    ),
                    "suicide_rate": round(
                        max(0, base_suicide + suicide_effect + np.random.normal(0, 0.5)), 2
                    ),
                }
            )

    df = pd.DataFrame(data)
    df["cohort"] = df["first_treat"]
    df["treated"] = ((df["first_treat"] > 0) & (df["year"] >= df["first_treat"])).astype(int)
    return df


def load_mpdta(force_download: bool = False) -> pd.DataFrame:
    """
    Load the Minimum Wage Panel Dataset for DiD Analysis (mpdta).

    This example dataset from the R `did` package contains county-level teen
    employment data under staggered minimum wage increases. It is commonly
    used to teach the Callaway-Sant'Anna estimator.

    Parameters
    ----------
    force_download : bool, default=False
        If True, re-download the dataset even if cached.

    Returns
    -------
    pd.DataFrame
        Panel dataset with columns:
        - countyreal : int - County identifier
        - year : int - Year (2003-2007)
        - lpop : float - Log population
        - lemp : float - Log employment (outcome)
        - first_treat : int - Year of minimum wage increase (0 = never)
        - treat : int - 1 if ever treated, 0 otherwise

    Notes
    -----
    This dataset is included in the R `did` package and is commonly used
    in tutorials demonstrating the Callaway-Sant'Anna estimator.

    The canonical data are checksum-verified and returned with
    ``df.attrs["source"] == "callaway_santanna_mpdta"``. A download failure
    falls back to a checksum-valid cache entry when one exists, returning that
    canonical data; a pin mismatch additionally warns that the upstream file has
    changed. Only when neither a verified cache entry nor a verified fresh
    download is available - or when the source fails to parse or validate - does
    the loader emit one ``UserWarning`` containing ``SYNTHETIC``
    and mark the returned frame as ``"synthetic_fallback"``.

    References
    ----------
    Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with
    multiple time periods. *Journal of Econometrics*, 225(2), 200-230.

    Examples
    --------
    >>> from diff_diff.datasets import load_mpdta
    >>> from diff_diff import CallawaySantAnna
    >>>
    >>> mpdta = load_mpdta()
    >>> cs = CallawaySantAnna()
    >>> results = cs.fit(
    ...     mpdta,
    ...     outcome="lemp",
    ...     unit="countyreal",
    ...     time="year",
    ...     first_treat="first_treat"
    ... )
    """
    return _load_verified_dataset(
        cache_name="mpdta",
        source="callaway_santanna_mpdta",
        force_download=force_download,
        load_source=_load_mpdta_source,
        prepare=_prepare_mpdta,
        validate_source=_validate_mpdta,
        validate_fallback=_validate_mpdta,
        fallback=_construct_mpdta_data,
    )


def _construct_mpdta_data() -> pd.DataFrame:
    """
    Construct a synthetic stand-in for the mpdta dataset.

    Mirrors the schema and panel dimensions of the R `did` package's ``mpdta``
    (500 counties, 2003-2007, cohorts 2004/2006/2007) with generated values. It
    is NOT the canonical data and must not be used for replication.
    """
    np.random.seed(2021)  # Callaway-Sant'Anna publication year, for reproducibility

    n_counties = 500
    years = [2003, 2004, 2005, 2006, 2007]

    # Treatment cohorts: 2004, 2006, 2007, or never (0)
    cohorts = [0, 2004, 2006, 2007]
    cohort_probs = [0.4, 0.2, 0.2, 0.2]

    data = []
    for county in range(1, n_counties + 1):
        first_treat = np.random.choice(cohorts, p=cohort_probs)
        base_lpop = np.random.normal(12.0, 1.0)
        base_lemp = base_lpop - np.random.uniform(1.5, 2.5)

        for year in years:
            time_effect = (year - 2003) * 0.02

            # Treatment effect (heterogeneous by cohort)
            if first_treat > 0 and year >= first_treat:
                if first_treat == 2004:
                    te = -0.04 + (year - first_treat) * 0.01
                elif first_treat == 2006:
                    te = -0.03 + (year - first_treat) * 0.01
                else:  # 2007
                    te = -0.025
            else:
                te = 0

            data.append(
                {
                    "countyreal": county,
                    "year": year,
                    "lpop": round(base_lpop + np.random.normal(0, 0.05), 4),
                    "lemp": round(base_lemp + time_effect + te + np.random.normal(0, 0.02), 4),
                    "first_treat": first_treat,
                    "treat": int(first_treat > 0),
                }
            )

    df = pd.DataFrame(data)
    df["cohort"] = df["first_treat"]
    return df


def load_prop99(force_download: bool = False) -> pd.DataFrame:
    """
    Load the California Proposition 99 smoking dataset (Lee-Wooldridge format).

    This dataset tracks per capita cigarette sales across 39 U.S. states
    (California plus 38 never-treated donor states) from 1970 to 2000.
    California passed Proposition 99, a large tobacco tax and control
    program, effective in 1989. With a single treated unit, it is the
    canonical setting for small-sample DiD inference and synthetic
    control comparisons.

    Parameters
    ----------
    force_download : bool, default=False
        If True, re-download the dataset even if cached.

    Returns
    -------
    pd.DataFrame
        Panel dataset with columns:
        - state : str - State name
        - year : int - Year (1970-2000)
        - first_year : int - Treatment start year (1989 for California, 0 = never)
        - lcigsale : float - Log per capita cigarette sales (packs)
        - treated : int - 1 if treatment in effect, 0 otherwise
        - cohort : int - Alias for first_year

    Notes
    -----
    This is the cohort-format version of the Abadie, Diamond &
    Hainmueller (2010) California tobacco data distributed (MIT license)
    with the authors' Stata ``lwdid`` package by Hur, Lee and Wooldridge.
    The donor pool excludes states with their own tobacco programs,
    leaving exactly one treated state and 38 controls.

    Downloads are verified against a pinned SHA-256 and validated against
    the source invariants (39 states, 1970-2000, single 1989 cohort). If
    the real data cannot be obtained, a SYNTHETIC same-schema fallback is
    returned with a ``UserWarning``; check ``df.attrs["source"]``
    (``"lwdid_ssc_ancillary"`` = real data, ``"synthetic_fallback"`` =
    synthetic - never use the fallback for replication).

    References
    ----------
    Lee, S. J., & Wooldridge, J. M. (2026). Simple Approaches to Inference
    with Difference-in-Differences Estimators with Small Cross-Sectional
    Sample Sizes. SSRN Working Paper No. 5325686.

    Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control
    Methods for Comparative Case Studies: Estimating the Effect of
    California's Tobacco Control Program. *Journal of the American
    Statistical Association*, 105(490), 493-505.

    Examples
    --------
    >>> from diff_diff.datasets import load_prop99
    >>> from diff_diff import DifferenceInDifferences
    >>>
    >>> prop99 = load_prop99()
    >>> prop99["treated_state"] = (prop99["first_year"] > 0).astype(int)
    >>> prop99["post"] = (prop99["year"] >= 1989).astype(int)
    >>>
    >>> did = DifferenceInDifferences()
    >>> results = did.fit(
    ...     prop99, outcome="lcigsale", treatment="treated_state", time="post"
    ... )
    """
    url = "http://fmwww.bc.edu/repec/bocode/l/lw_smoking.dta"
    sha256 = "16c3ac1da351788817433fc890ec2f502a8bdfcb46cbc8d693653330e71d5a65"

    source = "lwdid_ssc_ancillary"
    try:
        content = _download_with_cache_binary(url, "prop99", sha256, force_download)
        df = cast(pd.DataFrame, pd.read_stata(BytesIO(content)))
    except RuntimeError as e:
        # Fallback: construct synthetic data from documented patterns - NOT the
        # real Prop 99 data; unsuitable for replication.
        warnings.warn(
            f"Could not obtain the real Prop 99 dataset ({e}). Returning a "
            "SYNTHETIC fallback panel with the same schema. Do not use it for "
            "replication; check `df.attrs['source']`.",
            UserWarning,
            stacklevel=2,
        )
        source = "synthetic_fallback"
        df = _construct_prop99_data()

    # Normalize dtypes (the .dta stores first_year as float32, 0 = never treated)
    df["state"] = df["state"].astype(str)
    df["year"] = df["year"].astype(int)
    df["first_year"] = df["first_year"].astype(int)
    df["lcigsale"] = df["lcigsale"].astype(float)

    if source == "lwdid_ssc_ancillary":
        _validate_prop99(df)

    # Add convenience columns
    if "cohort" not in df.columns:
        df["cohort"] = df["first_year"]

    if "treated" not in df.columns:
        df["treated"] = ((df["first_year"] > 0) & (df["year"] >= df["first_year"])).astype(int)

    df.attrs["source"] = source
    return df


def _validate_prop99(df: pd.DataFrame) -> None:
    """Validate the downloaded Prop 99 data against its source invariants."""
    problems = []
    if df.shape != (1209, 4):
        problems.append(f"shape {df.shape} != (1209, 4)")
    if df["state"].nunique() != 39:
        problems.append(f"{df['state'].nunique()} states != 39")
    if (df["year"].min(), df["year"].max()) != (1970, 2000):
        problems.append("year range != 1970-2000")
    if df.duplicated(["state", "year"]).any():
        problems.append("duplicate (state, year) rows")
    if not (df.groupby("state")["first_year"].nunique() == 1).all():
        problems.append("first_year not constant within state")
    if set(df.loc[df["first_year"] > 0, "first_year"].unique()) != {1989}:
        problems.append("treated cohort != {1989}")
    if df.loc[df["first_year"] > 0, "state"].nunique() != 1:
        problems.append("treated state count != 1")
    if df.loc[df["first_year"] == 0, "state"].nunique() != 38:
        problems.append("never-treated state count != 38")
    if problems:
        raise RuntimeError(
            "Downloaded Prop 99 data failed source validation: "
            + "; ".join(problems)
            + ". The upstream file may have changed - please report this."
        )


def _construct_prop99_data() -> pd.DataFrame:
    """
    Construct a synthetic Prop 99-style dataset from documented patterns.

    This is a fallback when the online source is unavailable.
    """
    rng = np.random.default_rng(2010)  # Abadie-Diamond-Hainmueller publication year

    states = ["California"] + [f"State{i:02d}" for i in range(2, 40)]

    data = []
    for state in states:
        first_year = 1989 if state == "California" else 0
        base = rng.uniform(4.3, 4.9)  # log packs per capita
        trend = rng.uniform(-0.020, -0.010)  # secular decline

        for year in range(1970, 2001):
            lcigsale = base + trend * (year - 1970) + rng.normal(0, 0.04)
            # Treatment effect: gradual decline after 1989 (~ -0.4 by 2000)
            if first_year > 0 and year >= first_year:
                lcigsale -= 0.04 * min(year - first_year + 1, 10)

            data.append(
                {
                    "state": state,
                    "year": year,
                    "first_year": first_year,
                    "lcigsale": round(lcigsale, 6),
                }
            )

    return pd.DataFrame(data)


def load_walmart(force_download: bool = False) -> pd.DataFrame:
    """
    Load the Walmart entry county panel (Lee-Wooldridge sample).

    This dataset tracks log retail and wholesale employment for 1,277
    U.S. counties from 1977 to 1999, with staggered first Walmart store
    openings between 1986 and 1999 and 391 counties never receiving a
    store. It is used to study the local labor-market effects of Walmart
    entry under staggered treatment adoption.

    Parameters
    ----------
    force_download : bool, default=False
        If True, re-download the dataset even if cached.

    Returns
    -------
    pd.DataFrame
        Panel dataset with columns:
        - cid : int - County identifier
        - year : int - Year (1977-1999)
        - first_year : int - Year of first Walmart opening (0 = never)
        - log_retail_emp : float - Log county retail employment (outcome)
        - log_wholesale_emp : float - Log county wholesale employment
        - x1 : float - County poverty rate
        - x2 : float - Share with high-school education
        - x3 : float - Manufacturing employment share
        - treated : int - 1 if a Walmart has opened, 0 otherwise
        - cohort : int - Alias for first_year

    Notes
    -----
    The panel derives from County Business Patterns data as constructed
    by Brown & Butts, and is distributed (MIT license) with the authors'
    Stata ``lwdid`` package by Hur, Lee and Wooldridge. The covariate
    labels follow the Lee & Wooldridge application.

    Downloads are verified against a pinned SHA-256 and validated against
    the source invariants (1,277 counties, 1977-1999, cohorts 1986-1999,
    391 never-treated). If the real data cannot be obtained, a SYNTHETIC
    same-schema fallback (200 counties) is returned with a
    ``UserWarning``; check ``df.attrs["source"]``
    (``"lwdid_ssc_ancillary"`` = real data, ``"synthetic_fallback"`` =
    synthetic - never use the fallback for replication).

    References
    ----------
    Lee, S. J., & Wooldridge, J. M. (2025). A Simple Transformation
    Approach to Difference-in-Differences Estimation for Panel Data.
    SSRN Working Paper No. 4516518.

    Brown, N., & Butts, K. (2025). Dynamic Treatment Effect Estimation
    with Interactive Fixed Effects and Short Panels. *Journal of
    Econometrics*.

    Examples
    --------
    >>> from diff_diff.datasets import load_walmart
    >>> from diff_diff import CallawaySantAnna
    >>>
    >>> walmart = load_walmart()
    >>> cs = CallawaySantAnna(control_group="never_treated")
    >>> results = cs.fit(
    ...     walmart,
    ...     outcome="log_retail_emp",
    ...     unit="cid",
    ...     time="year",
    ...     first_treat="first_year",
    ... )
    """
    url = "http://fmwww.bc.edu/repec/bocode/l/lw_walmart.dta"
    sha256 = "410885572143dceb9daa643a8097768f1bc3493f9437451a9e4d1d5dc1e18d14"

    source = "lwdid_ssc_ancillary"
    try:
        content = _download_with_cache_binary(url, "walmart", sha256, force_download)
        df = cast(pd.DataFrame, pd.read_stata(BytesIO(content)))
    except RuntimeError as e:
        # Fallback: construct synthetic data from documented patterns - NOT the
        # real Walmart panel (and much smaller: 200 counties vs 1,277);
        # unsuitable for replication.
        warnings.warn(
            f"Could not obtain the real Walmart dataset ({e}). Returning a "
            "SYNTHETIC fallback panel (200 counties, not the real 1,277) with "
            "the same schema. Do not use it for replication; check "
            "`df.attrs['source']`.",
            UserWarning,
            stacklevel=2,
        )
        source = "synthetic_fallback"
        df = _construct_walmart_data()

    # Normalize dtypes (the .dta stores identifiers as float32, 0 = never treated)
    df["cid"] = df["cid"].astype(int)
    df["year"] = df["year"].astype(int)
    df["first_year"] = df["first_year"].astype(int)
    for col in ("log_retail_emp", "log_wholesale_emp", "x1", "x2", "x3"):
        df[col] = df[col].astype(float)

    if source == "lwdid_ssc_ancillary":
        _validate_walmart(df)

    # Add convenience columns
    if "cohort" not in df.columns:
        df["cohort"] = df["first_year"]

    if "treated" not in df.columns:
        df["treated"] = ((df["first_year"] > 0) & (df["year"] >= df["first_year"])).astype(int)

    df.attrs["source"] = source
    return df


def _validate_walmart(df: pd.DataFrame) -> None:
    """Validate the downloaded Walmart data against its source invariants."""
    problems = []
    if df.shape != (29371, 8):
        problems.append(f"shape {df.shape} != (29371, 8)")
    if df["cid"].nunique() != 1277:
        problems.append(f"{df['cid'].nunique()} counties != 1277")
    if (df["year"].min(), df["year"].max()) != (1977, 1999):
        problems.append("year range != 1977-1999")
    if df.duplicated(["cid", "year"]).any():
        problems.append("duplicate (cid, year) rows")
    if not (df.groupby("cid")["first_year"].nunique() == 1).all():
        problems.append("first_year not constant within county")
    cohorts = set(df.loc[df["first_year"] > 0, "first_year"].unique())
    if cohorts != set(range(1986, 2000)):
        problems.append("treated cohorts != {1986, ..., 1999}")
    if df.loc[df["first_year"] == 0, "cid"].nunique() != 391:
        problems.append("never-treated county count != 391")
    if problems:
        raise RuntimeError(
            "Downloaded Walmart data failed source validation: "
            + "; ".join(problems)
            + ". The upstream file may have changed - please report this."
        )


def _construct_walmart_data() -> pd.DataFrame:
    """
    Construct a synthetic Walmart-entry-style county panel.

    This is a fallback when the online source is unavailable.
    """
    rng = np.random.default_rng(2025)  # Brown-Butts publication year, for reproducibility

    n_counties = 200
    years = range(1977, 2000)
    # Roughly 30% never treated; the rest staggered over 1986-1999
    cohorts = [0] + list(range(1986, 2000))
    cohort_probs = [0.30] + [0.05] * 14

    data = []
    for cid in range(1, n_counties + 1):
        first_year = int(rng.choice(cohorts, p=cohort_probs))
        base_retail = rng.normal(7.5, 0.8)
        base_wholesale = base_retail - rng.uniform(0.8, 1.5)
        x1 = rng.uniform(0.05, 0.30)  # poverty rate
        x2 = rng.uniform(0.50, 0.85)  # HS education share
        x3 = rng.uniform(0.05, 0.40)  # manufacturing share

        for year in years:
            trend = (year - 1977) * 0.01
            te = 0.03 if (first_year > 0 and year >= first_year) else 0.0

            data.append(
                {
                    "cid": cid,
                    "year": year,
                    "first_year": first_year,
                    "log_retail_emp": round(base_retail + trend + te + rng.normal(0, 0.05), 6),
                    "log_wholesale_emp": round(base_wholesale + trend + rng.normal(0, 0.05), 6),
                    "x1": round(x1, 6),
                    "x2": round(x2, 6),
                    "x3": round(x3, 6),
                }
            )

    return pd.DataFrame(data)


def list_datasets() -> Dict[str, str]:
    """
    List available built-in datasets.

    Returns
    -------
    dict
        Dictionary mapping dataset names to descriptions.

    Examples
    --------
    >>> from diff_diff.datasets import list_datasets
    >>> for name, desc in list_datasets().items():
    ...     print(f"{name}: {desc}")
    """
    return {
        "card_krueger": "Card & Krueger (1994) minimum wage dataset - classic 2x2 DiD",
        "castle_doctrine": "Castle Doctrine laws - staggered adoption across states",
        "divorce_laws": (
            "Unilateral divorce laws - synthetic fallback only; no verified "
            "Stevenson-Wolfers source is configured"
        ),
        "mpdta": "County teen-employment panel - Callaway-Sant'Anna example from R `did`",
        "prop99": "California Prop 99 smoking panel - single treated unit (Lee-Wooldridge format)",
        "walmart": "Walmart entry county panel - staggered adoption (Lee-Wooldridge sample)",
    }


def load_dataset(name: str, force_download: bool = False) -> pd.DataFrame:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset. Use `list_datasets()` to see available datasets.
    force_download : bool, default=False
        If True, re-download the dataset even if cached.

    Returns
    -------
    pd.DataFrame
        The requested dataset.

    Raises
    ------
    ValueError
        If the dataset name is not recognized.

    Examples
    --------
    >>> from diff_diff.datasets import load_dataset, list_datasets
    >>> print(list_datasets())
    >>> df = load_dataset("card_krueger")
    """
    loaders = {
        "card_krueger": load_card_krueger,
        "castle_doctrine": load_castle_doctrine,
        "divorce_laws": load_divorce_laws,
        "mpdta": load_mpdta,
        "prop99": load_prop99,
        "walmart": load_walmart,
    }

    if name not in loaders:
        available = ", ".join(loaders.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    return loaders[name](force_download=force_download)
