"""PyMC-Marketing interop smoke tests (schema-drift canary).

These tests exercise :func:`diff_diff.mmm.to_pymc_marketing_lift_test` against the
REAL installed pymc-marketing API - no sampling anywhere. They are skipped unless
pymc-marketing >= 1.0 is installed (it is NOT a diff-diff dependency); the dedicated
``mmm-interop.yml`` CI job installs it and runs this file, with an import-canary
step so an install failure cannot be silently skipped into a green run.

Value-retention oracle: pymc-marketing 1.0 stores an added lift test as a
``lift_measurements`` potential inside the PyMC model graph and retains no
DataFrame attribute, so retention is asserted FUNCTIONALLY - the measured values
enter the likelihood (perturbing ``delta_y``/``sigma`` changes the model's logp at
a fixed parameter point; an identical DataFrame reproduces it exactly).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip(
    "pymc_marketing",
    minversion="1.0",
    reason="pymc-marketing>=1.0 required (installed only in the mmm-interop CI job)",
)

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation  # noqa: E402

from diff_diff.mmm import to_pymc_marketing_lift_test  # noqa: E402

LIFT_COLUMNS = ["channel", "x", "delta_x", "delta_y", "sigma"]


def _national_frame(n: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=n, freq="W-MON")
    search = (1500 + 450 * (np.arange(n) > n // 2)).astype(float)
    tv = 3000 + 100 * rng.random(n)
    y = pd.Series(45000 + 2.0 * search + 0.5 * tv + rng.normal(0, 200, n), name="y")
    return pd.DataFrame({"date": dates, "search": search, "tv": tv}), y


def _build_mmm(X: pd.DataFrame, y: pd.Series, **kwargs) -> MMM:
    mmm = MMM(
        date_column="date",
        channel_columns=["search", "tv"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        **kwargs,
    )
    mmm.build_model(X, y)
    return mmm


def _model_logp(mmm: MMM) -> float:
    model = mmm.model
    return float(model.compile_logp()(model.initial_point()))


class TestLiftTestSchema:
    def test_exporter_frame_accepted(self):
        X, y = _national_frame()
        mmm = _build_mmm(X, y)
        df = to_pymc_marketing_lift_test(
            channel="search", x=1500.0, delta_x=450.0, delta_y=930.0, sigma=110.0
        )
        assert list(df.columns) == LIFT_COLUMNS
        mmm.add_lift_test_measurements(df)
        assert "lift_measurements" in mmm.model.named_vars

    def test_wrong_schema_rejected(self):
        # Regression pin: the API validates the exporter's column names - a renamed
        # column must FAIL, otherwise the schema contract is no longer exercised.
        X, y = _national_frame()
        mmm = _build_mmm(X, y)
        df = to_pymc_marketing_lift_test(
            channel="search", x=1500.0, delta_x=450.0, delta_y=930.0, sigma=110.0
        ).rename(columns={"delta_y": "lift"})
        with pytest.raises(KeyError, match="delta_y"):
            mmm.add_lift_test_measurements(df)

    def test_multiple_rows_accepted(self):
        X, y = _national_frame()
        mmm = _build_mmm(X, y)
        df = to_pymc_marketing_lift_test(
            channel=["search", "tv"],
            x=[1500.0, 3000.0],
            delta_x=[450.0, 300.0],
            delta_y=[930.0, 240.0],
            sigma=[110.0, 60.0],
        )
        mmm.add_lift_test_measurements(df)
        assert "lift_measurements" in mmm.model.named_vars


class TestLiftTestDims:
    def test_dims_row_round_trips(self):
        # dims= is a first-class exporter parameter emitting real DataFrame
        # columns; verify a dims-carrying row against the real multidimensional API.
        rng = np.random.default_rng(11)
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="W-MON")
        frames = []
        for geo in ["east", "west"]:
            search = (700 + 200 * (np.arange(n) > n // 2)).astype(float)
            tv = 1500 + 50 * rng.random(n)
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "geo": geo,
                        "search": search,
                        "tv": tv,
                        "y": 20000 + 2.0 * search + 0.5 * tv + rng.normal(0, 100, n),
                    }
                )
            )
        panel = pd.concat(frames, ignore_index=True)
        X = panel[["date", "geo", "search", "tv"]]
        y = panel["y"]
        mmm = MMM(
            date_column="date",
            channel_columns=["search", "tv"],
            dims=("geo",),
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X, y)
        df = to_pymc_marketing_lift_test(
            channel="search",
            x=700.0,
            delta_x=200.0,
            delta_y=410.0,
            sigma=55.0,
            dims={"geo": "east"},
        )
        assert list(df.columns) == ["channel", "geo", "x", "delta_x", "delta_y", "sigma"]
        mmm.add_lift_test_measurements(df)
        assert "lift_measurements" in mmm.model.named_vars


class TestLiftTestValueRetention:
    """Functional retention: the exported values actually enter the likelihood."""

    def test_identical_frames_identical_logp(self):
        X, y = _national_frame()
        df = to_pymc_marketing_lift_test(
            channel="search", x=1500.0, delta_x=450.0, delta_y=930.0, sigma=110.0
        )
        logps = []
        for _ in range(2):
            mmm = _build_mmm(X, y)
            mmm.add_lift_test_measurements(df.copy())
            logps.append(_model_logp(mmm))
        assert logps[0] == pytest.approx(logps[1], rel=0, abs=0)

    @pytest.mark.parametrize(
        "column,changed",
        [("x", 3000.0), ("delta_x", 900.0), ("delta_y", 1860.0), ("sigma", 220.0)],
    )
    def test_perturbed_values_change_logp(self, column, changed):
        X, y = _national_frame()
        base = to_pymc_marketing_lift_test(
            channel="search", x=1500.0, delta_x=450.0, delta_y=930.0, sigma=110.0
        )
        perturbed = base.copy()
        perturbed[column] = changed
        mmm_a = _build_mmm(X, y)
        mmm_a.add_lift_test_measurements(base)
        mmm_b = _build_mmm(X, y)
        mmm_b.add_lift_test_measurements(perturbed)
        assert _model_logp(mmm_a) != _model_logp(mmm_b), (
            f"changing {column} did not change the model logp - the lift value is "
            "being ignored or remapped by the framework"
        )
