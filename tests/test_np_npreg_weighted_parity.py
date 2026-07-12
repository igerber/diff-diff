"""Cross-language parity test for the weighted local-linear kernel fit.

Loads ``benchmarks/data/np_npreg_weighted_golden.json`` — produced by
``benchmarks/R/generate_np_npreg_weighted_golden.R`` — and verifies that
``diff_diff.local_linear.local_linear_fit`` recovers the same intercept
and slope from the same (d, y, weights) + bandwidth + Epanechnikov
kernel. The R reference is a manually-implemented weighted OLS with
one-sided Epanechnikov kernel, matching the Python formula exactly.

Purpose: regression lock on the weighted kernel composition + weighted
OLS formula that underlies ``_nprobust_port.lprobust`` when weights are
threaded. Bit-parity at ``atol=1e-12`` tolerates BLAS reduction-order
drift across Linux/macOS/Windows runners without giving up the
cross-language check.

This is NOT third-party validation of the weighted-CCF methodology —
no public weighted-CCF reference exists in any language (nprobust has
no weight argument; np::npreg's local-linear algorithm diverges from a
straightforward weighted-OLS at the intercept). See REGISTRY "Weighted
extension (Phase 4.5)" for the parity-gap acknowledgement. Methodology
confidence under informative weights comes from the uniform-weights
bit-parity lock + Monte Carlo oracle consistency, not this fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "np_npreg_weighted_golden.json"
)


@pytest.fixture(scope="module")
def weighted_golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(
            "Golden values file not found; run: "
            "Rscript benchmarks/R/generate_np_npreg_weighted_golden.R"
        )
    with GOLDEN_PATH.open() as f:
        return json.load(f)


class TestWeightedLocalLinearCrossLanguageParity:
    def _load_dgp(self, golden, name):
        g = golden[name]
        return (
            np.asarray(g["d"], dtype=np.float64),
            np.asarray(g["y"], dtype=np.float64),
            np.asarray(g["weights"], dtype=np.float64),
            g,
        )

    @pytest.mark.parametrize("dgp_name", ["dgp1", "dgp2", "dgp3", "dgp4"])
    def test_intercept_parity(self, weighted_golden, dgp_name):
        from diff_diff.local_linear import local_linear_fit

        d, y, w, g = self._load_dgp(weighted_golden, dgp_name)
        fit = local_linear_fit(
            d=d,
            y=y,
            bandwidth=float(g["h"]),
            boundary=float(g["eval_point"]),
            kernel="epanechnikov",
            weights=w,
        )
        np.testing.assert_allclose(fit.intercept, g["mu_hat"], atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize("dgp_name", ["dgp1", "dgp2", "dgp3", "dgp4"])
    def test_slope_parity(self, weighted_golden, dgp_name):
        from diff_diff.local_linear import local_linear_fit

        d, y, w, g = self._load_dgp(weighted_golden, dgp_name)
        fit = local_linear_fit(
            d=d,
            y=y,
            bandwidth=float(g["h"]),
            boundary=float(g["eval_point"]),
            kernel="epanechnikov",
            weights=w,
        )
        np.testing.assert_allclose(fit.slope, g["slope"], atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize("dgp_name", ["dgp1", "dgp2", "dgp3", "dgp4"])
    def test_n_active_parity(self, weighted_golden, dgp_name):
        """Active-window count must match exactly — the kernel-support
        boundary (``w > 0``) is a cross-language invariant."""
        from diff_diff.local_linear import local_linear_fit

        d, y, w, g = self._load_dgp(weighted_golden, dgp_name)
        fit = local_linear_fit(
            d=d,
            y=y,
            bandwidth=float(g["h"]),
            boundary=float(g["eval_point"]),
            kernel="epanechnikov",
            weights=w,
        )
        assert fit.n_effective == int(g["n_active"])

    def test_uniform_weights_matches_no_weights(self, weighted_golden):
        """dgp1 has uniform weights; passing weights=None should give the
        same result at atol=1e-14. Orthogonal cross-check of the
        weights=np.ones bit-parity test in TestWeightedLprobust."""
        from diff_diff.local_linear import local_linear_fit

        d, y, _, g = self._load_dgp(weighted_golden, "dgp1")
        fit_w = local_linear_fit(
            d=d,
            y=y,
            bandwidth=float(g["h"]),
            boundary=float(g["eval_point"]),
            kernel="epanechnikov",
            weights=np.ones(d.size),
        )
        fit_nw = local_linear_fit(
            d=d,
            y=y,
            bandwidth=float(g["h"]),
            boundary=float(g["eval_point"]),
            kernel="epanechnikov",
        )
        np.testing.assert_allclose(fit_w.intercept, fit_nw.intercept, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(fit_w.slope, fit_nw.slope, atol=1e-14, rtol=1e-14)
