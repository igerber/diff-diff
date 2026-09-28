"""Unit tests for the private port in ``diff_diff._nprobust_port``.

These tests exercise the internal helpers directly (``kernel_W``,
``qrXXinv``, ``lprobust_bw``, etc.) so that when an end-to-end parity
test in ``tests/test_bandwidth_selector.py`` drifts, the root cause is
localised to a single helper.
"""

from __future__ import annotations

import numpy as np
import pytest

from diff_diff._nprobust_port import (
    NPROBUST_SHA,
    NPROBUST_VERSION,
    kernel_W,
    lprobust_bw,
    qrXXinv,
)

# =============================================================================
# Metadata
# =============================================================================


def test_pinned_nprobust_version():
    assert NPROBUST_VERSION == "0.5.0"
    assert NPROBUST_SHA == "36e4e532d2f7d23d4dc6e162575cca79e0927cda"


# =============================================================================
# kernel_W (W.fun port)
# =============================================================================


class TestKernelW:
    def test_epa_symmetric(self):
        u = np.array([-0.7, -0.3, 0.0, 0.3, 0.7])
        w = kernel_W(u, "epa")
        # Symmetric around 0.
        np.testing.assert_allclose(w, w[::-1], atol=1e-14, rtol=1e-14)
        # k(0) = 0.75.
        assert w[2] == pytest.approx(0.75)
        # Zero at |u| = 1.
        assert kernel_W(np.array([1.0, -1.0]), "epa")[0] == pytest.approx(0.0)
        # Zero outside.
        assert kernel_W(np.array([1.5, -1.5]), "epa")[0] == 0.0

    def test_epa_nonneg_matches_one_sided_phase1a(self):
        """For u >= 0, nprobust symmetric epa = Phase 1a one-sided epa."""
        from diff_diff.local_linear import epanechnikov_kernel

        u = np.linspace(0.0, 1.0, 20)
        assert np.allclose(kernel_W(u, "epa"), epanechnikov_kernel(u))

    def test_uniform(self):
        u = np.array([-0.9, 0.0, 0.5, 1.0, 1.1])
        w = kernel_W(u, "uni")
        assert w[0] == 0.5
        assert w[1] == 0.5
        assert w[2] == 0.5
        assert w[3] == 0.5  # |u|=1 is inside
        assert w[4] == 0.0  # |u|>1 is outside

    def test_triangular(self):
        u = np.array([-0.3, 0.0, 0.7, 1.0, 1.5])
        w = kernel_W(u, "tri")
        assert w[0] == pytest.approx(0.7)
        assert w[1] == pytest.approx(1.0)
        assert w[2] == pytest.approx(0.3)
        assert w[3] == pytest.approx(0.0)
        assert w[4] == 0.0

    def test_gaussian(self):
        from scipy.stats import norm

        u = np.array([-1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(kernel_W(u, "gau"), norm.pdf(u), atol=1e-14)

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            kernel_W(np.array([0.5]), "cosine")


# =============================================================================
# qrXXinv (chol2inv(chol(crossprod(.))) port)
# =============================================================================


class TestQrXXinv:
    def test_matches_numpy_inverse(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 5))
        expected = np.linalg.inv(X.T @ X)
        actual = qrXXinv(X)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_symmetric_output(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(30, 3))
        A = qrXXinv(X)
        np.testing.assert_allclose(A, A.T, atol=1e-14, rtol=1e-14)

    def test_identity_for_orthonormal_input(self):
        # If X has orthonormal columns, X.T @ X = I, so inverse = I.
        X = np.eye(4)[:, :3]
        np.testing.assert_allclose(qrXXinv(X), np.eye(3), atol=1e-14, rtol=1e-14)


# =============================================================================
# lprobust_bw: single-stage parity against the golden JSON
# =============================================================================


def _precompute_for_test(x):
    """Helper that sorts and builds dups/dupsid."""
    from diff_diff._nprobust_port import _precompute_nn_duplicates

    order = np.argsort(x)
    x_sorted = x[order]
    dups, dupsid = _precompute_nn_duplicates(x_sorted)
    return order, x_sorted, dups, dupsid


class TestLprobustBwStageD1:
    """First Stage-2 call (C.d1 with o=q+1=3, nu=q+1=3, o_B=q+2=4).

    Uses the DGP 1 golden values as reference. The c_bw input must
    match R's clipped value, which is the raw rule-of-thumb bandwidth
    when bwcheck does not bind (true for DGP 1 at G=2000).
    """

    def _setup(self):
        import json
        from pathlib import Path

        golden_path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "data"
            / "nprobust_mse_dpi_golden.json"
        )
        if not golden_path.exists():
            pytest.skip(
                "Golden values file not found; "
                "run: Rscript benchmarks/R/generate_nprobust_golden.R"
            )
        with golden_path.open() as f:
            golden = json.load(f)
        dgp = golden["dgp1"]
        d = np.array(dgp["d"], dtype=np.float64)
        y = np.array(dgp["y"], dtype=np.float64)
        _, d_sorted, dups, dupsid = _precompute_for_test(d)
        order = np.argsort(d)
        y_sorted = y[order]
        c_bw = float(dgp["c_bw"])
        range_ = float(d_sorted.max() - d_sorted.min())
        return d_sorted, y_sorted, dups, dupsid, c_bw, range_, dgp["stage_d1"]

    def test_V_matches(self):
        d, y, dups, dupsid, c_bw, range_, stage = self._setup()
        C_d1 = lprobust_bw(
            y,
            d,
            None,
            0.0,
            o=3,
            nu=3,
            o_B=4,
            h_V=c_bw,
            h_B1=range_,
            h_B2=range_,
            scale=0.0,
            vce="nn",
            nnmatch=3,
            kernel="epa",
            dups=dups,
            dupsid=dupsid,
        )
        assert C_d1.V == pytest.approx(stage["V"], rel=0.01)

    def test_B1_matches(self):
        d, y, dups, dupsid, c_bw, range_, stage = self._setup()
        C_d1 = lprobust_bw(
            y,
            d,
            None,
            0.0,
            o=3,
            nu=3,
            o_B=4,
            h_V=c_bw,
            h_B1=range_,
            h_B2=range_,
            scale=0.0,
            vce="nn",
            nnmatch=3,
            kernel="epa",
            dups=dups,
            dupsid=dupsid,
        )
        assert C_d1.B1 == pytest.approx(stage["B1"], rel=0.01)

    def test_B2_matches(self):
        d, y, dups, dupsid, c_bw, range_, stage = self._setup()
        C_d1 = lprobust_bw(
            y,
            d,
            None,
            0.0,
            o=3,
            nu=3,
            o_B=4,
            h_V=c_bw,
            h_B1=range_,
            h_B2=range_,
            scale=0.0,
            vce="nn",
            nnmatch=3,
            kernel="epa",
            dups=dups,
            dupsid=dupsid,
        )
        assert C_d1.B2 == pytest.approx(stage["B2"], rel=0.01)

    def test_R_is_zero_when_scale_zero(self):
        d, y, dups, dupsid, c_bw, range_, _stage = self._setup()
        C_d1 = lprobust_bw(
            y,
            d,
            None,
            0.0,
            o=3,
            nu=3,
            o_B=4,
            h_V=c_bw,
            h_B1=range_,
            h_B2=range_,
            scale=0.0,
            vce="nn",
            nnmatch=3,
            kernel="epa",
            dups=dups,
            dupsid=dupsid,
        )
        # With scale=0, BWreg is never computed -> R stays 0.
        assert C_d1.R == 0.0


# =============================================================================
# lpbwselect_mse_dpi: input validation on the advanced-use entry point
# =============================================================================


class TestLpbwselectMseDpiValidation:
    """The public wrapper is restricted to the HAD surface; the port is
    the advertised advanced-use entry point. It must enforce its own
    shape / emptiness / finiteness contract -- silently truncating a
    longer y or cluster through sort-index reindexing would be a real
    bug."""

    def test_mismatched_shapes_raise(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        x = np.array([0.1, 0.2, 0.3])
        y = np.array([1.0, 2.0, 3.0, 4.0])  # length 4 != 3
        with pytest.raises(ValueError, match="same 1-D shape"):
            lpbwselect_mse_dpi(y, x, eval_point=0.0)

    def test_longer_y_silent_truncation_rejected(self):
        """Regression: len(y) > len(x) previously got truncated via
        the sort-indexer under vce='nn'. Must now raise."""
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=100)
        y = rng.normal(size=200)  # twice the length
        with pytest.raises(ValueError, match="same 1-D shape"):
            lpbwselect_mse_dpi(y, x, eval_point=0.0, vce="nn")

    def test_cluster_wrong_length_rejected(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=100)
        y = rng.normal(size=100)
        cluster = np.arange(50)  # wrong length
        with pytest.raises(ValueError, match="cluster must have"):
            lpbwselect_mse_dpi(y, x, cluster=cluster, eval_point=0.0)

    def test_empty_direct_port_input_rejected(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="non-empty"):
            lpbwselect_mse_dpi(y, x, eval_point=0.0)

    def test_non_finite_x_rejected(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        x = np.array([0.1, np.nan, 0.3])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="x contains non-finite"):
            lpbwselect_mse_dpi(y, x, eval_point=0.0)

    def test_non_finite_y_rejected(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        x = np.array([0.1, 0.2, 0.3])
        y = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="y contains non-finite"):
            lpbwselect_mse_dpi(y, x, eval_point=0.0)

    def test_non_finite_eval_point_rejected(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=100)
        y = rng.normal(size=100)
        with pytest.raises(ValueError, match="eval_point"):
            lpbwselect_mse_dpi(y, x, eval_point=np.nan)

    def test_missing_cluster_id_rejected_nan(self):
        """Missing (NaN) cluster IDs must be rejected, not silently
        dropped from block grouping (cluster == c is False for NaN).
        Deviates from nprobust's complete-case filter -- documented
        in the module docstring."""
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=100)
        y = rng.normal(size=100)
        cluster = np.arange(100, dtype=np.float64)
        cluster[7] = np.nan
        cluster[42] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            lpbwselect_mse_dpi(y, x, cluster=cluster, eval_point=0.0)

    def test_missing_cluster_id_rejected_none(self):
        """``None`` sentinel in an object-dtype cluster array must
        also be rejected."""
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=10)
        y = rng.normal(size=10)
        cluster = np.array([1, 2, None, 3, 4, 5, 6, 7, 8, 9], dtype=object)
        with pytest.raises(ValueError, match="missing values"):
            lpbwselect_mse_dpi(y, x, cluster=cluster, eval_point=0.0)


# =============================================================================
# lprobust_vce clustered branch: port fidelity vs R source
# =============================================================================


class TestLprobustVceClustered:
    """Pin the clustered lprobust_vce result against the R source
    formula. R's lprobust.vce (npfunctions.R:165-185) returns M (the
    sum over clusters of outer products of sum-of-scores), NOT a
    finite-sample-scaled M. An earlier port mistakenly returned
    ``w * M``; this test is the regression anchor."""

    def test_clustered_meat_matches_unscaled_sum(self):
        from diff_diff._nprobust_port import lprobust_vce

        rng = np.random.default_rng(123)
        n, k = 40, 3
        RX = rng.normal(size=(n, k))
        res = rng.normal(size=(n, 1))
        cluster = np.repeat(np.arange(10), 4)

        # Manual computation: M = sum_g (X_g' r_g) outer (X_g' r_g),
        # with NO ((n-1)/(n-k))*(g/(g-1)) scaling.
        expected = np.zeros((k, k), dtype=np.float64)
        for c in np.unique(cluster):
            ind = cluster == c
            v = RX[ind].T @ res[ind].ravel()
            expected += np.outer(v, v)

        actual = lprobust_vce(RX, res, cluster)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)

    def test_clustered_is_symmetric(self):
        """The meat is a sum of outer products of rank-1 vectors and
        must be symmetric."""
        from diff_diff._nprobust_port import lprobust_vce

        rng = np.random.default_rng(7)
        n, k = 30, 4
        RX = rng.normal(size=(n, k))
        res = rng.normal(size=(n, 1))
        cluster = np.array([0] * 10 + [1] * 10 + [2] * 10)
        M = lprobust_vce(RX, res, cluster)
        np.testing.assert_allclose(M, M.T, atol=1e-14, rtol=1e-14)

    def test_clustered_end_to_end_through_lprobust_bw(self):
        """Smoke test: a clustered call through lpbwselect_mse_dpi
        completes and returns finite stage bandwidths. Pins that the
        clustered DPI path exercises lprobust_vce -> lprobust_bw ->
        driver without rank / shape / indexing crashes."""
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(20260419)
        G = 500
        d = rng.uniform(0.0, 1.0, size=G)
        y = d + d**2 + rng.normal(0, 0.3, size=G)
        # 50 clusters, balanced.
        cluster = np.repeat(np.arange(50), G // 50)
        res = lpbwselect_mse_dpi(
            y,
            d,
            cluster=cluster,
            eval_point=0.0,
            p=1,
            deriv=0,
            kernel="epa",
            vce="nn",
        )
        assert np.isfinite(res.h_mse_dpi)
        assert res.h_mse_dpi > 0.0
        assert np.isfinite(res.c_bw)
        assert res.c_bw > 0.0


# =============================================================================
# lprobust single-eval (Phase 1c: lprobust.R:177-246 port)
# =============================================================================
#
# Port-level parity tests. Exercise lprobust() directly with R-supplied
# bandwidths so drift is localized to the single-eval math (Q.q, beta.bc,
# CCT-2014 robust variance), not the bandwidth-selection pipeline. Wrapper-
# level parity lives in tests/test_bias_corrected_lprobust.py.


def _load_lprobust_golden():
    """Skip if benchmarks/ is absent (CI isolated-install path)."""
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "data"
        / "nprobust_lprobust_golden.json"
    )
    if not path.exists():
        pytest.skip(
            "Golden values file not found; "
            "run: Rscript benchmarks/R/generate_nprobust_lprobust_golden.R"
        )
    with path.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def lprobust_golden():
    return _load_lprobust_golden()


class TestLprobustSingleEval:
    """Parity and behavioral tests for the single-eval-point port of
    nprobust::lprobust (lprobust.R:177-246).

    Cluster parity here is ORTHOGONAL to the wrapper-level clustered test
    in test_bias_corrected_lprobust.py: this class exercises the port
    (lprobust_vce cluster branch -> Q.q -> V.Y.bc), while the wrapper test
    verifies the public API's cluster contract (NaN rejection, vce+cluster
    interactions).
    """

    def _dgp(self, golden, name):
        g = golden[name]
        return (
            np.asarray(g["d"], dtype=np.float64),
            np.asarray(g["y"], dtype=np.float64),
            g,
        )

    def test_lprobust_dgp1_parity(self, lprobust_golden):
        from diff_diff._nprobust_port import lprobust

        d, y, g = self._dgp(lprobust_golden, "dgp1")
        res = lprobust(
            y,
            d,
            eval_point=0.0,
            h=g["h"],
            b=g["b"],
            p=1,
            q=2,
            deriv=0,
            kernel="epa",
            vce="nn",
        )
        # Tiered tolerances per plan commit criterion #1.
        np.testing.assert_allclose(res.tau_cl, g["tau_cl"], atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(res.se_cl, g["se_cl"], atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(res.tau_bc, g["tau_bc"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(res.se_rb, g["se_rb"], atol=1e-12, rtol=1e-12)
        assert res.n_used == int(g["n_used"])

    def test_lprobust_dgp2_parity(self, lprobust_golden):
        from diff_diff._nprobust_port import lprobust

        d, y, g = self._dgp(lprobust_golden, "dgp2")
        res = lprobust(
            y,
            d,
            eval_point=0.0,
            h=g["h"],
            b=g["b"],
            p=1,
            q=2,
            deriv=0,
            kernel="epa",
            vce="nn",
        )
        np.testing.assert_allclose(res.tau_cl, g["tau_cl"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(res.se_cl, g["se_cl"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(res.tau_bc, g["tau_bc"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(res.se_rb, g["se_rb"], atol=1e-12, rtol=1e-12)

    def test_lprobust_dgp3_parity(self, lprobust_golden):
        from diff_diff._nprobust_port import lprobust

        d, y, g = self._dgp(lprobust_golden, "dgp3")
        res = lprobust(
            y,
            d,
            eval_point=0.0,
            h=g["h"],
            b=g["b"],
            p=1,
            q=2,
            deriv=0,
            kernel="epa",
            vce="nn",
        )
        np.testing.assert_allclose(res.tau_cl, g["tau_cl"], atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(res.se_cl, g["se_cl"], atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(res.tau_bc, g["tau_bc"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(res.se_rb, g["se_rb"], atol=1e-12, rtol=1e-12)

    def test_lprobust_manual_bandwidths_rho_1(self):
        """Smoke test that lprobust runs with user-supplied h == b == 0.3."""
        from diff_diff._nprobust_port import lprobust

        rng = np.random.default_rng(20260420)
        G = 1000
        d = rng.uniform(0.0, 1.0, G)
        y = d + d**2 + rng.normal(0, 0.5, G)
        res = lprobust(
            y, d, eval_point=0.0, h=0.3, b=0.3, p=1, q=2, deriv=0, kernel="epa", vce="nn"
        )
        assert np.isfinite(res.tau_cl)
        assert np.isfinite(res.tau_bc)
        assert res.se_cl > 0
        assert res.se_rb > 0
        assert res.h == 0.3
        assert res.b == 0.3

    def test_lprobust_cluster_parity(self, lprobust_golden):
        """Port-level clustered parity. DGP 4 uses manual h=b=0.3 to sidestep
        an nprobust-internal singleton-cluster bug in mse-dpi pilot fits.
        """
        from diff_diff._nprobust_port import lprobust

        g = lprobust_golden["dgp4"]
        d = np.asarray(g["d"], dtype=np.float64)
        y = np.asarray(g["y"], dtype=np.float64)
        cluster = np.asarray(g["cluster"])
        res = lprobust(
            y,
            d,
            eval_point=0.0,
            h=g["h"],
            b=g["b"],
            p=1,
            q=2,
            deriv=0,
            kernel="epa",
            vce="nn",
            cluster=cluster,
        )
        # Cluster IDs happen to match R's first-appearance order here (IDs
        # are already sequential 1..50), so bit-parity is achievable.
        np.testing.assert_allclose(res.tau_cl, g["tau_cl"], atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(res.tau_bc, g["tau_bc"], atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(res.se_cl, g["se_cl"], atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(res.se_rb, g["se_rb"], atol=1e-14, rtol=1e-14)

    def test_lprobust_bwcheck_clip_floor(self):
        """Sparse-ish data where requested h is below the 21st-NN
        distance should clip up to bw.min."""
        from diff_diff._nprobust_port import lprobust

        rng = np.random.default_rng(7)
        G = 100
        d = np.abs(rng.normal(0, 1, G))  # few obs near 0
        y = d + rng.normal(0, 0.5, G)
        # Very small h; bwcheck should float it up.
        res = lprobust(
            y,
            d,
            eval_point=0.0,
            h=0.001,
            b=0.001,
            p=1,
            q=2,
            deriv=0,
            kernel="epa",
            vce="nn",
            bwcheck=21,
        )
        # Bandwidth must have been floored — original 0.001 will not hold.
        assert res.h > 0.001
        assert res.b > 0.001
        # The floor is the 21st-NN distance to boundary 0; must be positive.
        sorted_dists = np.sort(np.abs(d - 0.0))
        expected_floor = sorted_dists[20]
        assert res.h == pytest.approx(expected_floor, rel=1e-14)

    def test_lprobust_shifted_boundary_dgp5(self, lprobust_golden):
        """Parity at a non-zero boundary (Design 1 continuous-near-d_lower)."""
        from diff_diff._nprobust_port import lprobust

        g = lprobust_golden["dgp5"]
        d = np.asarray(g["d"], dtype=np.float64)
        y = np.asarray(g["y"], dtype=np.float64)
        res = lprobust(
            y,
            d,
            eval_point=g["eval_point_override"],
            h=g["h"],
            b=g["b"],
            p=1,
            q=2,
            deriv=0,
            kernel="epa",
            vce="nn",
        )
        np.testing.assert_allclose(res.tau_cl, g["tau_cl"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(res.tau_bc, g["tau_bc"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(res.se_cl, g["se_cl"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(res.se_rb, g["se_rb"], atol=1e-12, rtol=1e-12)

    def test_lprobust_cluster_object_none_raises(self):
        """Port-level: object-dtype cluster with None sentinel is rejected.
        Mirror of the wrapper test; pins that `_cluster_has_missing` fires
        inside the direct port entry point too (CI review PR #340
        follow-up P1)."""
        from diff_diff._nprobust_port import lprobust

        rng = np.random.default_rng(0)
        G = 200
        d = rng.uniform(0.0, 1.0, G)
        y = d + rng.normal(0, 0.3, G)
        cluster = np.array([i // 10 if i != 5 else None for i in range(G)], dtype=object)
        with pytest.raises(ValueError, match="cluster contains missing"):
            lprobust(y, d, eval_point=0.0, h=0.3, b=0.3, cluster=cluster)

    def test_lprobust_cluster_object_nan_raises(self):
        """Port-level: object-dtype cluster with np.nan is rejected."""
        from diff_diff._nprobust_port import lprobust

        rng = np.random.default_rng(0)
        G = 200
        d = rng.uniform(0.0, 1.0, G)
        y = d + rng.normal(0, 0.3, G)
        cluster = np.array([i // 10 if i != 5 else np.nan for i in range(G)], dtype=object)
        with pytest.raises(ValueError, match="cluster contains missing"):
            lprobust(y, d, eval_point=0.0, h=0.3, b=0.3, cluster=cluster)

    def test_lprobust_h_gt_b_selects_h_window(self):
        """When h > b, the active window is ind.h (lprobust.R:182 conditional
        replacement), not a union of ind.h and ind.b."""
        from diff_diff._nprobust_port import lprobust

        rng = np.random.default_rng(42)
        G = 800
        d = rng.uniform(0.0, 1.0, G)
        y = d + rng.normal(0, 0.3, G)
        # h = 0.4 > b = 0.2 -> active window = ind.h (the larger window).
        # Expected n_used = count of obs within kernel support of h-window,
        # which for epa is 0 <= (d - 0)/h <= 1 => 0 <= d <= h.
        h = 0.4
        b = 0.2
        res = lprobust(
            y, d, eval_point=0.0, h=h, b=b, p=1, q=2, deriv=0, kernel="epa", vce="nn", bwcheck=None
        )
        # Post-sort d; ind.h = (|d|/h <= 1) & (d <= h) => d in [0, h].
        sorted_d = np.sort(d)
        # Relaxed: assert n_used > sum of b-window (which is smaller).
        expected_n_b = int(np.sum((np.abs(sorted_d) / b <= 1)))
        assert res.n_used > expected_n_b, (
            f"h > b should select the wider h-window: n_used={res.n_used} "
            f"should exceed b-window count ({expected_n_b})"
        )


# =============================================================================
# Weighted lprobust (Phase 4.5 survey support)
# =============================================================================


class TestWeightedLprobust:
    """Validation harness for weighted ``lprobust`` (Phase 4.5 A).

    No public weighted-CCF reference exists; the uniform-weights bit-parity
    against the unweighted path is the primary regression lock. Informative-
    weight behavior is validated via MC oracle tests in ``test_had_mc.py``
    and ``np::npreg`` partial parity in ``test_np_npreg_weighted_parity.py``.
    """

    def _panel(self, seed=42, n=200):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1.0, 1.0, n)
        y = 2.0 * x + 0.3 * x**2 + rng.normal(0.0, 0.25, n)
        return x, y

    def test_uniform_weights_bit_parity(self):
        from diff_diff._nprobust_port import lprobust

        x, y = self._panel()
        base = lprobust(y, x, eval_point=0.0, h=0.4, b=0.4)
        w1 = lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, weights=np.ones(x.size))
        np.testing.assert_allclose(w1.tau_cl, base.tau_cl, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(w1.tau_bc, base.tau_bc, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(w1.se_cl, base.se_cl, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(w1.se_rb, base.se_rb, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(w1.V_Y_cl, base.V_Y_cl, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(w1.V_Y_bc, base.V_Y_bc, atol=1e-14, rtol=1e-14)
        assert w1.n_used == base.n_used

    def test_uniform_weights_bit_parity_hc1(self):
        """vce='hc1' threads weights through the hat-matrix hii term
        (line 1276). Confirm bit-parity under w=ones for 'hc1' as well."""
        from diff_diff._nprobust_port import lprobust

        x, y = self._panel()
        base = lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, vce="hc1")
        w1 = lprobust(
            y,
            x,
            eval_point=0.0,
            h=0.4,
            b=0.4,
            vce="hc1",
            weights=np.ones(x.size),
        )
        np.testing.assert_allclose(w1.tau_bc, base.tau_bc, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(w1.se_rb, base.se_rb, atol=1e-14, rtol=1e-14)

    def test_nontrivial_weights_change_estimate(self):
        """Sanity check: under informative weights, the weighted estimate
        differs from unweighted (weight mechanism has teeth)."""
        from diff_diff._nprobust_port import lprobust

        x, y = self._panel()
        base = lprobust(y, x, eval_point=0.0, h=0.4, b=0.4)
        w = np.exp(-np.abs(x) * 2.0)  # upweight near boundary
        r_w = lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, weights=w)
        assert not np.isclose(r_w.tau_bc, base.tau_bc, atol=1e-6)

    def test_negative_weights_reject(self):
        from diff_diff._nprobust_port import lprobust

        x, y = self._panel()
        w = np.ones(x.size)
        w[0] = -0.5
        with pytest.raises(ValueError, match="non-negative"):
            lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, weights=w)

    def test_nan_weights_reject(self):
        from diff_diff._nprobust_port import lprobust

        x, y = self._panel()
        w = np.ones(x.size)
        w[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, weights=w)

    def test_inf_weights_reject(self):
        from diff_diff._nprobust_port import lprobust

        x, y = self._panel()
        w = np.ones(x.size)
        w[0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, weights=w)

    def test_zero_sum_weights_reject(self):
        from diff_diff._nprobust_port import lprobust

        x, y = self._panel()
        with pytest.raises(ValueError, match="sum to zero"):
            lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, weights=np.zeros(x.size))

    def test_weights_length_mismatch(self):
        from diff_diff._nprobust_port import lprobust

        x, y = self._panel()
        with pytest.raises(ValueError, match="weights length"):
            lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, weights=np.ones(x.size - 1))

    def test_zero_weight_observations_drop_from_window(self):
        """Observations with weights[i]=0 filter out via combined ``w>0``
        selector, matching filtering the data first and re-fitting."""
        from diff_diff._nprobust_port import lprobust

        x, y = self._panel(n=100)
        in_window = np.abs(x) < 0.3
        drop_idx = np.where(in_window)[0][:5]
        w = np.ones(x.size)
        w[drop_idx] = 0.0
        r_w = lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, weights=w)
        mask = w > 0
        r_ref = lprobust(y[mask], x[mask], eval_point=0.0, h=0.4, b=0.4)
        np.testing.assert_allclose(r_w.tau_bc, r_ref.tau_bc, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(r_w.se_rb, r_ref.se_rb, atol=1e-12, rtol=1e-12)

    def test_weights_with_cluster(self):
        """Weights + cluster are orthogonal threading paths (weights enter
        via W_h/W_b; cluster enters via variance crossproduct aggregation)."""
        from diff_diff._nprobust_port import lprobust

        x, y = self._panel(n=200)
        rng = np.random.default_rng(7)
        cluster = rng.integers(0, 20, x.size)
        w = rng.uniform(0.5, 1.5, x.size)
        r = lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, cluster=cluster, weights=w)
        assert np.isfinite(r.tau_bc) and np.isfinite(r.se_rb) and r.se_rb > 0

    def test_uniform_weights_bit_parity_with_cluster(self):
        from diff_diff._nprobust_port import lprobust

        x, y = self._panel(n=200)
        rng = np.random.default_rng(7)
        cluster = rng.integers(0, 20, x.size)
        base = lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, cluster=cluster)
        w1 = lprobust(y, x, eval_point=0.0, h=0.4, b=0.4, cluster=cluster, weights=np.ones(x.size))
        np.testing.assert_allclose(w1.tau_bc, base.tau_bc, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(w1.se_rb, base.se_rb, atol=1e-14, rtol=1e-14)
