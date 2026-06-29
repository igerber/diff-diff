"""Unit tests for the copy-avoiding ``within_transform`` refactor (PR-C).

Locks the ``inplace``/``suffix`` decoupling, the batch-assignment overwrite
semantics (load-bearing for TWFE's replicate refit, which re-demeans a frame that
already carries the suffix), and the absence of a pandas fragmentation warning on
the many-column path (SunAbraham can demean 100+ interaction columns).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.utils import within_transform


def _panel(nu=30, nt=5, k=2, seed=0):
    rng = np.random.default_rng(seed)
    n = nu * nt
    data = {
        "unit": np.repeat(np.arange(nu), nt),
        "time": np.tile(np.arange(nt), nu),
    }
    for j in range(k):
        data[f"v{j}"] = rng.standard_normal(n)
    return pd.DataFrame(data)


def _ref_demean(df, var):
    """Reference unweighted two-way within transform: y - y_i. - y_.t + y_.."""
    s = df[var]
    return (
        s
        - df.groupby("unit")[var].transform("mean")
        - df.groupby("time")[var].transform("mean")
        + s.mean()
    ).values


class TestWithinTransformInplaceSuffix:
    def test_inplace_false_leaves_input_untouched(self):
        df = _panel()
        orig = df.copy()
        out = within_transform(df, ["v0", "v1"], "unit", "time")
        pd.testing.assert_frame_equal(df, orig)  # default inplace=False: input not mutated
        assert out is not df
        assert "v0_demeaned" in out.columns and "v0" in out.columns
        # The unweighted path now uses iterative MAP (exact for unbalanced panels);
        # on this balanced panel it equals the closed-form additive demean to ~1 ULP.
        np.testing.assert_allclose(
            out["v0_demeaned"].values, _ref_demean(df, "v0"), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            out["v1_demeaned"].values, _ref_demean(df, "v1"), rtol=1e-12, atol=1e-12
        )

    def test_inplace_true_suffix_mutates_same_object_keeps_originals(self):
        df = _panel()
        v0_orig = df["v0"].values.copy()
        out = within_transform(df, ["v0", "v1"], "unit", "time", inplace=True)
        assert out is df  # same object, mutated in place (no copy)
        np.testing.assert_array_equal(df["v0"].values, v0_orig)  # original preserved
        np.testing.assert_allclose(
            df["v0_demeaned"].values, _ref_demean(df, "v0"), rtol=1e-12, atol=1e-12
        )

    def test_inplace_true_empty_suffix_overwrites_source(self):
        df = _panel()
        ref = _ref_demean(df, "v0")
        within_transform(df, ["v0"], "unit", "time", inplace=True, suffix="")
        assert "v0_demeaned" not in df.columns
        np.testing.assert_allclose(df["v0"].values, ref, rtol=1e-12, atol=1e-12)

    def test_redemean_existing_suffix_overwrites_no_duplicate(self):
        # The TWFE replicate scenario: a frame that already carries the suffix is
        # re-demeaned. The batch assignment must OVERWRITE the existing column
        # (single label) rather than append a duplicate — a duplicate label would
        # make ``df[col].values`` 2-D and break the downstream np.column_stack.
        df = _panel()
        within_transform(df, ["v0"], "unit", "time", inplace=True)  # adds v0_demeaned
        out = within_transform(df, ["v0"], "unit", "time", inplace=True)  # re-demean
        assert list(out.columns).count("v0_demeaned") == 1
        assert out["v0_demeaned"].values.ndim == 1

    def test_non_inplace_empty_suffix_overwrites_no_duplicate(self):
        # inplace=False + suffix="" targets the existing source column; the concat
        # path must drop the original first so the result has ONE "v0" column
        # (a duplicate label would make .values 2-D), while leaving the input frame
        # unmutated.
        df = _panel()
        ref = _ref_demean(df, "v0")
        out = within_transform(df, ["v0"], "unit", "time", suffix="")
        assert list(out.columns).count("v0") == 1
        assert out["v0"].values.ndim == 1
        np.testing.assert_allclose(out["v0"].values, ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(df["v0"].values, _panel()["v0"].values)  # input intact

    def test_non_inplace_redemean_existing_suffix_no_duplicate(self):
        # inplace=False re-demean of a frame that already carries the suffixed
        # column overwrites it (single label), not a duplicate.
        df = within_transform(_panel(), ["v0"], "unit", "time")  # has v0_demeaned
        out = within_transform(df, ["v0"], "unit", "time")  # re-demean, inplace=False
        assert list(out.columns).count("v0_demeaned") == 1
        assert out["v0_demeaned"].values.ndim == 1

    def test_weighted_inplace_matches_non_inplace(self):
        rng = np.random.default_rng(1)
        df = _panel(seed=1)
        w = rng.uniform(0.5, 2.0, len(df))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = within_transform(df.copy(), ["v0"], "unit", "time", weights=w)
            b = df.copy()
            within_transform(b, ["v0"], "unit", "time", weights=w, inplace=True)
        np.testing.assert_array_equal(a["v0_demeaned"].values, b["v0_demeaned"].values)


class TestWithinTransformManyColumns:
    def test_many_columns_no_fragmentation_warning(self):
        # SunAbraham can demean 100+ interaction columns; the default (concat)
        # path must attach them as one consolidated block and NOT trigger pandas'
        # "DataFrame is highly fragmented" PerformanceWarning that per-column
        # inserts would.
        df = _panel(nu=20, nt=4, k=150)
        cols = [f"v{j}" for j in range(150)]
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            out = within_transform(df, cols, "unit", "time")  # default inplace=False -> concat
        assert all(f"{c}_demeaned" in out.columns for c in cols)
        assert out is not df


def _unbalanced_panel(seed=0, drop=0.3):
    """Unbalanced two-way panel (some unit-time cells dropped -> non-orthogonal)."""
    rng = np.random.default_rng(seed)
    rows = [(u, t) for u in range(8) for t in range(6) if rng.random() >= drop]
    df = pd.DataFrame(rows, columns=["unit", "time"])
    df["v0"] = rng.standard_normal(len(df))
    return df


class TestWithinTransformConvergence:
    def test_unweighted_unbalanced_matches_full_dummy(self):
        # The unweighted path now iterates (MAP), so it is exact on unbalanced
        # panels. Anchor: FWL slope of v0 on a constant after demeaning must equal
        # the unit+time full-dummy residual mean (both ~0 here); more strongly, the
        # demeaned column is orthogonal to both FE spans.
        df = _unbalanced_panel(seed=2)
        out = within_transform(df, ["v0"], "unit", "time")
        v = out["v0_demeaned"].values
        for g in ("unit", "time"):
            means = pd.Series(v).groupby(df[g].values).transform("mean").values
            assert np.max(np.abs(means)) < 1e-8

    def test_unweighted_nonconvergence_warns(self):
        # A starved iteration budget on an unbalanced panel must emit the shared
        # "did not converge" UserWarning (the unweighted path could not warn before
        # the MAP refactor; silent return of the iterate is a silent failure).
        df = _unbalanced_panel(seed=3)
        with pytest.warns(UserWarning, match="did not converge"):
            within_transform(df, ["v0"], "unit", "time", max_iter=1, tol=1e-15)
