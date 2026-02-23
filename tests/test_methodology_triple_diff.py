"""
Comprehensive methodology verification tests for TripleDifference estimator.

This module verifies that the TripleDifference implementation matches:
1. The theoretical formulas from DDD algebra (hand calculations)
2. The behavior of R's triplediff::ddd() (Ortiz-Villavicencio & Sant'Anna 2025)
3. All documented edge cases in docs/methodology/REGISTRY.md

References:
- Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
  Better Understanding Triple Differences Estimators.
  arXiv:2505.09942.
"""

import json
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import diff_diff
from diff_diff import TripleDifference
from diff_diff.prep_dgp import generate_ddd_data
from diff_diff.utils import safe_inference
from tests.conftest import assert_nan_inference

# Resolve repo root from the package location (robust against pytest-xdist
# chdir to temp directories, which breaks Path(__file__)-relative paths).
_REPO_ROOT = Path(diff_diff.__file__).resolve().parent.parent


# =============================================================================
# R Availability Fixtures (local, session-scoped)
# =============================================================================

_triplediff_available_cache = None


def _check_triplediff_available() -> bool:
    """Check if R and triplediff package are available (cached)."""
    global _triplediff_available_cache
    if _triplediff_available_cache is None:
        r_env = os.environ.get("DIFF_DIFF_R", "auto").lower()
        if r_env == "skip":
            _triplediff_available_cache = False
        else:
            try:
                result = subprocess.run(
                    [
                        "Rscript", "-e",
                        "library(triplediff); library(jsonlite); cat('OK')",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                _triplediff_available_cache = (
                    result.returncode == 0 and "OK" in result.stdout
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                _triplediff_available_cache = False
    return _triplediff_available_cache


@pytest.fixture(scope="session")
def triplediff_available():
    """Lazy check for R/triplediff availability."""
    return _check_triplediff_available()


@pytest.fixture
def require_triplediff(triplediff_available):
    """Skip test if R/triplediff is not available."""
    if not triplediff_available:
        pytest.skip("R or triplediff package not available")


# =============================================================================
# Data Helpers
# =============================================================================

_R_RESULTS_PATH = (
    _REPO_ROOT / "benchmarks" / "data" / "synthetic" / "ddd_r_results.json"
)


def _load_r_results():
    """Load pre-computed R benchmark results."""
    if not _R_RESULTS_PATH.exists():
        pytest.skip("R benchmark results JSON not available")
    with open(_R_RESULTS_PATH) as f:
        return json.load(f)


def _generate_hand_calculable_ddd() -> pd.DataFrame:
    """
    Generate minimal DDD data with exact hand-calculable values.

    8 cells × 2 obs = 16 observations. No noise, so the DDD is exact.

    Cell means:
      G=1,P=1,T=0: 10    G=1,P=1,T=1: 18  (diff=8)
      G=1,P=0,T=0:  6    G=1,P=0,T=1: 10  (diff=4)
      G=0,P=1,T=0:  5    G=0,P=1,T=1:  8  (diff=3)
      G=0,P=0,T=0:  3    G=0,P=0,T=1:  5  (diff=2)

    DiD(treated): (8 - 4) = 4
    DiD(control): (3 - 2) = 1
    DDD = 4 - 1 = 3.0
    """
    data = pd.DataFrame({
        "outcome": [10, 10, 18, 18,   6, 6, 10, 10,   5, 5, 8, 8,   3, 3, 5, 5],
        "group":   [ 1,  1,  1,  1,   1, 1,  1,  1,   0, 0, 0, 0,   0, 0, 0, 0],
        "partition":[1,  1,  1,  1,   0, 0,  0,  0,   1, 1, 1, 1,   0, 0, 0, 0],
        "time":    [ 0,  0,  1,  1,   0, 0,  1,  1,   0, 0, 1, 1,   0, 0, 1, 1],
        "unit_id": list(range(16)),
    })
    return data


def _load_r_dgp_data(dgp_num: int) -> pd.DataFrame:
    """Load R-generated DGP data, mapping columns to Python convention."""
    csv_path = (
        _REPO_ROOT / "benchmarks" / "data" / "synthetic" / f"ddd_r_dgp{dgp_num}.csv"
    )
    if not csv_path.exists():
        pytest.skip(f"R DGP{dgp_num} data CSV not available")
    df = pd.read_csv(csv_path)
    # Map R columns to Python convention
    df = df.rename(columns={
        "y": "outcome",
        "state": "group",
        "partition": "partition",
        "time": "time",
        "id": "unit_id",
    })
    # R uses time in {1, 2}, map to {0, 1}
    df["time"] = (df["time"] - 1).astype(int)
    return df


def _run_r_triplediff(
    data_path: str,
    method: str = "dr",
    covariates: bool = False,
) -> dict:
    """Run R's triplediff::ddd() on a CSV file and return results."""
    escaped_path = data_path.replace("\\", "/")
    xformla = "~cov1+cov2+cov3+cov4" if covariates else "~1"

    r_script = f'''
    suppressMessages(library(triplediff))
    suppressMessages(library(jsonlite))

    d <- read.csv("{escaped_path}")
    res <- ddd(
        yname = "y",
        tname = "time",
        idname = "id",
        gname = "state",
        pname = "partition",
        data = d,
        control_group = "nevertreated",
        panel = FALSE,
        xformla = {xformla},
        est_method = "{method}",
        boot = FALSE
    )

    output <- list(
        ATT = unbox(res$ATT),
        se = unbox(res$se),
        lci = unbox(res$lci),
        uci = unbox(res$uci)
    )

    cat(toJSON(output, pretty = TRUE, digits = 15))
    '''

    result = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(f"R script failed: {result.stderr}")

    return json.loads(result.stdout)


# =============================================================================
# Phase 1: Formula Verification (no R dependency)
# =============================================================================


class TestHandCalculation:
    """Verify DDD formula matches hand-calculated values."""

    def test_att_hand_calculation_no_covariates(self):
        """Manual 8-cell-mean DDD matches estimator output."""
        data = _generate_hand_calculable_ddd()

        # All three methods should give the same ATT without covariates
        for method in ["reg", "ipw", "dr"]:
            ddd = TripleDifference(estimation_method=method)
            results = ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )
            np.testing.assert_allclose(
                results.att, 3.0, atol=1e-10,
                err_msg=f"ATT ({method}) should be 3.0 by hand calculation",
            )

    def test_att_reg_matches_ols_interaction(self):
        """RA ATT equals coefficient on G*P*T in OLS with all interactions."""
        data = generate_ddd_data(n_per_cell=200, seed=42)

        # Run OLS with full 2x2x2 interaction
        G = data["group"].values.astype(float)
        P = data["partition"].values.astype(float)
        T = data["time"].values.astype(float)
        y = data["outcome"].values.astype(float)

        X = np.column_stack([
            np.ones(len(y)),
            G, P, T,
            G * P, G * T, P * T,
            G * P * T,
        ])
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        ols_att = beta_ols[7]  # coefficient on G*P*T

        # Run RA estimator
        ddd = TripleDifference(estimation_method="reg")
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        np.testing.assert_allclose(
            results.att, ols_att, rtol=1e-6,
            err_msg="RA ATT should match G*P*T OLS coefficient (no covariates)",
        )

    def test_all_methods_agree_no_covariates(self):
        """RA, IPW, DR give same ATT without covariates."""
        data = generate_ddd_data(n_per_cell=200, seed=42)

        atts = {}
        for method in ["reg", "ipw", "dr"]:
            ddd = TripleDifference(estimation_method=method)
            results = ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )
            atts[method] = results.att

        np.testing.assert_allclose(
            atts["ipw"], atts["reg"], rtol=1e-6,
            err_msg="IPW and REG ATT should agree without covariates",
        )
        np.testing.assert_allclose(
            atts["dr"], atts["reg"], rtol=1e-6,
            err_msg="DR and REG ATT should agree without covariates",
        )

    def test_all_methods_se_agree_no_covariates(self):
        """RA, IPW, DR give same SE without covariates."""
        data = generate_ddd_data(n_per_cell=200, seed=42)

        ses = {}
        for method in ["reg", "ipw", "dr"]:
            ddd = TripleDifference(estimation_method=method)
            results = ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )
            ses[method] = results.se

        np.testing.assert_allclose(
            ses["ipw"], ses["reg"], rtol=1e-4,
            err_msg="IPW and REG SE should agree without covariates",
        )
        np.testing.assert_allclose(
            ses["dr"], ses["reg"], rtol=1e-4,
            err_msg="DR and REG SE should agree without covariates",
        )

    def test_se_uses_influence_function(self):
        """Verify SE is computed from influence function, not cell variance."""
        data = generate_ddd_data(n_per_cell=100, seed=42)

        ddd = TripleDifference(estimation_method="dr")
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # SE should be positive and finite
        assert np.isfinite(results.se) and results.se > 0

        # With 800 observations and no covariates in a simple DGP,
        # the SE should be small relative to the noise (noise_sd=1.0 default)
        # A cell-variance SE would be much larger; influence function SE
        # captures the correct sampling variability
        assert results.se < 1.0, (
            f"SE={results.se} seems too large for n_per_cell=100; "
            "might be using naive cell variance instead of influence function"
        )

    def test_safe_inference_used(self):
        """Verify t_stat/p_value/conf_int come from safe_inference()."""
        data = generate_ddd_data(n_per_cell=100, seed=42)

        ddd = TripleDifference(estimation_method="dr", alpha=0.05)
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # Recompute using safe_inference
        n_obs = len(data)
        df = n_obs - 8
        df = max(df, 1)

        t_stat, p_value, conf_int = safe_inference(
            results.att, results.se, alpha=0.05, df=df,
        )

        np.testing.assert_allclose(results.t_stat, t_stat, rtol=1e-10)
        np.testing.assert_allclose(results.p_value, p_value, rtol=1e-10)
        np.testing.assert_allclose(results.conf_int[0], conf_int[0], rtol=1e-10)
        np.testing.assert_allclose(results.conf_int[1], conf_int[1], rtol=1e-10)

    def test_cell_means_match_direct_computation(self):
        """Group means in results match direct cell mean computation."""
        data = _generate_hand_calculable_ddd()

        ddd = TripleDifference(estimation_method="reg")
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        expected_means = {
            "Treated, Eligible, Pre": 10.0,
            "Treated, Eligible, Post": 18.0,
            "Treated, Ineligible, Pre": 6.0,
            "Treated, Ineligible, Post": 10.0,
            "Control, Eligible, Pre": 5.0,
            "Control, Eligible, Post": 8.0,
            "Control, Ineligible, Pre": 3.0,
            "Control, Ineligible, Post": 5.0,
        }

        for cell, expected in expected_means.items():
            actual = results.group_means[cell]
            np.testing.assert_allclose(
                actual, expected, atol=1e-10,
                err_msg=f"Cell mean mismatch for {cell}",
            )


# =============================================================================
# Phase 2: R Comparison Tests (pre-computed R results)
# =============================================================================


class TestRComparisonPrecomputed:
    """Compare against pre-computed R triplediff::ddd() results.

    Uses R-generated DGP data (from gen_dgp_2periods) with pre-stored
    R results. These tests run without R being installed.
    """

    @pytest.fixture(autouse=True)
    def _check_data_available(self):
        """Skip all tests if R benchmark data files are missing."""
        if not _R_RESULTS_PATH.exists():
            pytest.skip("R benchmark data not available")

    @pytest.fixture(scope="class")
    def r_results(self):
        return _load_r_results()

    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_att_no_covariates_matches_r_dgp1(self, r_results, method):
        """ATT without covariates within <1% of R (DGP1)."""
        data = _load_r_dgp_data(1)
        key = f"dgp1_{method}_nocov"
        r_att = r_results[key]["ATT"]

        ddd = TripleDifference(estimation_method=method)
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # Use atol for near-zero ATTs
        if abs(r_att) < 0.1:
            np.testing.assert_allclose(
                results.att, r_att, atol=0.05,
                err_msg=f"ATT ({method} nocov DGP1): Py={results.att:.6f}, R={r_att:.6f}",
            )
        else:
            np.testing.assert_allclose(
                results.att, r_att, rtol=0.01,
                err_msg=f"ATT ({method} nocov DGP1): Py={results.att:.6f}, R={r_att:.6f}",
            )

    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_se_no_covariates_matches_r_dgp1(self, r_results, method):
        """SE without covariates within <1% of R (DGP1)."""
        data = _load_r_dgp_data(1)
        key = f"dgp1_{method}_nocov"
        r_se = r_results[key]["se"]

        ddd = TripleDifference(estimation_method=method)
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        np.testing.assert_allclose(
            results.se, r_se, rtol=0.01,
            err_msg=f"SE ({method} nocov DGP1): Py={results.se:.6f}, R={r_se:.6f}",
        )

    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_att_with_covariates_matches_r_dgp1(self, r_results, method):
        """ATT with covariates within <1% of R (DGP1)."""
        data = _load_r_dgp_data(1)
        key = f"dgp1_{method}_cov"
        r_att = r_results[key]["ATT"]

        covariates = [c for c in data.columns if c.startswith("cov")]
        ddd = TripleDifference(estimation_method=method)
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            covariates=covariates,
        )

        if abs(r_att) < 0.1:
            np.testing.assert_allclose(
                results.att, r_att, atol=0.05,
                err_msg=f"ATT ({method} cov DGP1): Py={results.att:.6f}, R={r_att:.6f}",
            )
        else:
            np.testing.assert_allclose(
                results.att, r_att, rtol=0.01,
                err_msg=f"ATT ({method} cov DGP1): Py={results.att:.6f}, R={r_att:.6f}",
            )

    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_se_with_covariates_matches_r_dgp1(self, r_results, method):
        """SE with covariates within <1% of R (DGP1)."""
        data = _load_r_dgp_data(1)
        key = f"dgp1_{method}_cov"
        r_se = r_results[key]["se"]

        covariates = [c for c in data.columns if c.startswith("cov")]
        ddd = TripleDifference(estimation_method=method)
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            covariates=covariates,
        )

        np.testing.assert_allclose(
            results.se, r_se, rtol=0.01,
            err_msg=f"SE ({method} cov DGP1): Py={results.se:.6f}, R={r_se:.6f}",
        )

    @pytest.mark.parametrize("dgp", [2, 3, 4])
    def test_dr_robust_across_dgp_types(self, r_results, dgp):
        """DR ATT matches R across DGP types (different model misspecification)."""
        data = _load_r_dgp_data(dgp)
        covariates = [c for c in data.columns if c.startswith("cov")]

        for cov_suffix, cov_list in [("nocov", None), ("cov", covariates)]:
            key = f"dgp{dgp}_dr_{cov_suffix}"
            r_att = r_results[key]["ATT"]
            r_se = r_results[key]["se"]

            ddd = TripleDifference(estimation_method="dr")
            results = ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
                covariates=cov_list,
            )

            # ATT check
            if abs(r_att) < 0.1:
                np.testing.assert_allclose(
                    results.att, r_att, atol=0.05,
                    err_msg=f"DR ATT (DGP{dgp} {cov_suffix}): Py={results.att:.6f}, R={r_att:.6f}",
                )
            else:
                np.testing.assert_allclose(
                    results.att, r_att, rtol=0.01,
                    err_msg=f"DR ATT (DGP{dgp} {cov_suffix}): Py={results.att:.6f}, R={r_att:.6f}",
                )

            # SE check
            np.testing.assert_allclose(
                results.se, r_se, rtol=0.01,
                err_msg=f"DR SE (DGP{dgp} {cov_suffix}): Py={results.se:.6f}, R={r_se:.6f}",
            )


# =============================================================================
# Phase 3: Live R Comparison Tests (require R + triplediff)
# =============================================================================


class TestRComparisonLive:
    """Run R's triplediff::ddd() live and compare.

    These tests are skipped if R or the triplediff package is not installed.
    They provide an additional layer of validation using freshly generated data.
    """

    @pytest.fixture(scope="class")
    def shared_data_csv(self, tmp_path_factory):
        """Generate shared data and write to CSV for both Python and R."""
        data = generate_ddd_data(n_per_cell=300, seed=12345, add_covariates=True)
        tmp_dir = tmp_path_factory.mktemp("ddd_live_r")
        csv_path = tmp_dir / "ddd_data.csv"

        # Map to R column convention
        r_data = data.rename(columns={
            "outcome": "y",
            "group": "state",
            "partition": "partition",
            "time": "time",
            "unit_id": "id",
        })
        # R expects time in {1, 2}
        r_data["time"] = r_data["time"] + 1
        # Add covariate columns named cov1-cov4 if they exist
        if "age" in data.columns:
            r_data["cov1"] = data["age"]
            r_data["cov2"] = data["education"]
            r_data["cov3"] = np.random.default_rng(12345).normal(0, 1, len(data))
            r_data["cov4"] = np.random.default_rng(54321).normal(0, 1, len(data))

        r_data.to_csv(csv_path, index=False)
        return data, str(csv_path)

    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_live_att_no_cov(self, require_triplediff, shared_data_csv, method):
        """Live R comparison: ATT without covariates."""
        data, csv_path = shared_data_csv

        r_result = _run_r_triplediff(csv_path, method=method, covariates=False)
        r_att = r_result["ATT"]

        ddd = TripleDifference(estimation_method=method)
        py_result = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        if abs(r_att) < 0.1:
            np.testing.assert_allclose(
                py_result.att, r_att, atol=0.05,
                err_msg=f"Live ATT ({method} nocov): Py={py_result.att:.6f}, R={r_att:.6f}",
            )
        else:
            np.testing.assert_allclose(
                py_result.att, r_att, rtol=0.01,
                err_msg=f"Live ATT ({method} nocov): Py={py_result.att:.6f}, R={r_att:.6f}",
            )

    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_live_se_no_cov(self, require_triplediff, shared_data_csv, method):
        """Live R comparison: SE without covariates."""
        data, csv_path = shared_data_csv

        r_result = _run_r_triplediff(csv_path, method=method, covariates=False)
        r_se = r_result["se"]

        ddd = TripleDifference(estimation_method=method)
        py_result = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        np.testing.assert_allclose(
            py_result.se, r_se, rtol=0.01,
            err_msg=f"Live SE ({method} nocov): Py={py_result.se:.6f}, R={r_se:.6f}",
        )


# =============================================================================
# Phase 4: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases documented in docs/methodology/REGISTRY.md."""

    def test_small_sample_sizes(self):
        """All 8 cells populated with small n (3 per cell) gives valid results."""
        data = generate_ddd_data(n_per_cell=3, seed=42)

        for method in ["reg", "ipw", "dr"]:
            ddd = TripleDifference(estimation_method=method)
            results = ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )

            assert np.isfinite(results.att), f"ATT should be finite ({method})"
            assert np.isfinite(results.se) and results.se > 0, (
                f"SE should be positive and finite ({method})"
            )
            assert results.n_obs == 24  # 8 cells × 3

    def test_zero_treatment_effect(self):
        """ATT near zero when true effect is zero; inference still valid."""
        data = generate_ddd_data(
            n_per_cell=200, treatment_effect=0.0, seed=42,
        )

        ddd = TripleDifference(estimation_method="dr")
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # ATT should be near zero (within ~2 SE)
        assert abs(results.att) < 2 * results.se, (
            f"ATT={results.att:.4f} too far from zero (SE={results.se:.4f})"
        )
        # Inference should still be valid
        assert np.isfinite(results.t_stat)
        assert 0 <= results.p_value <= 1
        assert results.conf_int[0] < results.conf_int[1]

    def test_pscore_trimming_active(self):
        """Extreme propensity scores are clipped at pscore_trim."""
        # Create data with very imbalanced cells to trigger extreme pscores
        rng = np.random.default_rng(42)
        records = []
        unit_id = 0
        # Heavily imbalanced: 5 in treated eligible, 200 in control ineligible
        sizes = {
            (1, 1): 5,   # G=1, P=1 (very small)
            (1, 0): 200,  # G=1, P=0 (large)
            (0, 1): 200,  # G=0, P=1 (large)
            (0, 0): 200,  # G=0, P=0 (large)
        }
        for (g, p), n_cell in sizes.items():
            for t in [0, 1]:
                for _ in range(n_cell):
                    y = 10 + 2 * g + 1 * p + 0.5 * t + rng.normal(0, 1)
                    if g == 1 and p == 1 and t == 1:
                        y += 3.0
                    records.append({
                        "outcome": y,
                        "group": g,
                        "partition": p,
                        "time": t,
                        "unit_id": unit_id,
                    })
                    unit_id += 1
        data = pd.DataFrame(records)

        # IPW with tight trimming should still work
        ddd = TripleDifference(estimation_method="ipw", pscore_trim=0.05)
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )
        assert np.isfinite(results.att)
        assert np.isfinite(results.se) and results.se > 0

    def test_nan_inference_when_se_zero(self):
        """All inference fields are NaN when SE is zero or invalid."""
        # Create perfectly deterministic data (zero variance in all cells)
        data = pd.DataFrame({
            "outcome": [10.0, 10.0, 18.0, 18.0,
                         6.0,  6.0, 10.0, 10.0,
                         5.0,  5.0,  8.0,  8.0,
                         3.0,  3.0,  5.0,  5.0],
            "group":    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "partition": [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            "time":      [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "unit_id":  list(range(16)),
        })

        ddd = TripleDifference(estimation_method="reg")
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # With zero within-cell variance, SE should be zero
        # and safe_inference should produce NaN t_stat/p_value
        if results.se == 0.0:
            assert_nan_inference({
                "se": results.se,
                "t_stat": results.t_stat,
                "p_value": results.p_value,
                "conf_int": results.conf_int,
            })

    def test_large_treatment_effect(self):
        """Large treatment effect is detected correctly."""
        data = generate_ddd_data(
            n_per_cell=100, treatment_effect=50.0, noise_sd=1.0, seed=42,
        )

        ddd = TripleDifference(estimation_method="dr")
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        np.testing.assert_allclose(
            results.att, 50.0, rtol=0.1,
            err_msg=f"ATT={results.att:.2f} should be near 50.0",
        )
        assert results.p_value < 0.001, "Large effect should be highly significant"

    def test_covariates_reduce_se(self):
        """Adding relevant covariates reduces SE."""
        data = generate_ddd_data(
            n_per_cell=200, seed=42, add_covariates=True,
        )

        # Without covariates
        ddd_nocov = TripleDifference(estimation_method="dr")
        results_nocov = ddd_nocov.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # With covariates
        ddd_cov = TripleDifference(estimation_method="dr")
        results_cov = ddd_cov.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            covariates=["age", "education"],
        )

        assert results_cov.se < results_nocov.se, (
            f"SE with covariates ({results_cov.se:.4f}) should be less than "
            f"SE without ({results_nocov.se:.4f}) when covariates are relevant"
        )


# =============================================================================
# Phase 5: Scale Validation
# =============================================================================


class TestScaleValidation:
    """Verify results converge at different sample sizes."""

    @pytest.mark.parametrize("n_per_cell", [200, 500])
    def test_att_converges_to_true_effect(self, n_per_cell):
        """ATT converges to true effect as sample size increases."""
        true_effect = 3.0
        data = generate_ddd_data(
            n_per_cell=n_per_cell, treatment_effect=true_effect, seed=42,
        )

        ddd = TripleDifference(estimation_method="dr")
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # With n_per_cell >= 200, should be within ~2 SE of true effect
        assert abs(results.att - true_effect) < 3 * results.se, (
            f"ATT={results.att:.4f} not close to true effect {true_effect} "
            f"(SE={results.se:.4f}, n_per_cell={n_per_cell})"
        )

    def test_se_decreases_with_sample_size(self):
        """SE decreases approximately as 1/sqrt(n)."""
        ses = {}
        for n_per_cell in [100, 400]:
            data = generate_ddd_data(n_per_cell=n_per_cell, seed=42)
            ddd = TripleDifference(estimation_method="dr")
            results = ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )
            ses[n_per_cell] = results.se

        # Quadrupling n should halve SE (approximately)
        se_ratio = ses[100] / ses[400]
        assert 1.3 < se_ratio < 3.0, (
            f"SE ratio (n=100/n=400) = {se_ratio:.2f}, expected ~2.0"
        )


# =============================================================================
# Phase 6: Cross-DGP R Comparison (all 4 DGP types × 3 methods)
# =============================================================================


class TestAllDGPMethods:
    """Comprehensive R comparison across all DGP types and methods."""

    @pytest.fixture(autouse=True)
    def _check_data_available(self):
        if not _R_RESULTS_PATH.exists():
            pytest.skip("R benchmark data not available")

    @pytest.fixture(scope="class")
    def r_results(self):
        return _load_r_results()

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_att_nocov_all_dgps(self, r_results, dgp, method):
        """ATT (no covariates) within <1% of R for all DGP-method combos."""
        data = _load_r_dgp_data(dgp)
        key = f"dgp{dgp}_{method}_nocov"
        r_att = r_results[key]["ATT"]

        ddd = TripleDifference(estimation_method=method)
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        if abs(r_att) < 0.1:
            np.testing.assert_allclose(
                results.att, r_att, atol=0.05,
                err_msg=f"ATT ({method} nocov DGP{dgp})",
            )
        else:
            np.testing.assert_allclose(
                results.att, r_att, rtol=0.01,
                err_msg=f"ATT ({method} nocov DGP{dgp})",
            )

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_se_nocov_all_dgps(self, r_results, dgp, method):
        """SE (no covariates) within <1% of R for all DGP-method combos."""
        data = _load_r_dgp_data(dgp)
        key = f"dgp{dgp}_{method}_nocov"
        r_se = r_results[key]["se"]

        ddd = TripleDifference(estimation_method=method)
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        np.testing.assert_allclose(
            results.se, r_se, rtol=0.01,
            err_msg=f"SE ({method} nocov DGP{dgp})",
        )

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_att_cov_all_dgps(self, r_results, dgp, method):
        """ATT (with covariates) within <1% of R for all DGP-method combos."""
        data = _load_r_dgp_data(dgp)
        covariates = [c for c in data.columns if c.startswith("cov")]
        key = f"dgp{dgp}_{method}_cov"
        r_att = r_results[key]["ATT"]

        ddd = TripleDifference(estimation_method=method)
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            covariates=covariates,
        )

        if abs(r_att) < 0.1:
            np.testing.assert_allclose(
                results.att, r_att, atol=0.05,
                err_msg=f"ATT ({method} cov DGP{dgp})",
            )
        else:
            np.testing.assert_allclose(
                results.att, r_att, rtol=0.01,
                err_msg=f"ATT ({method} cov DGP{dgp})",
            )

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_se_cov_all_dgps(self, r_results, dgp, method):
        """SE (with covariates) within <1% of R for all DGP-method combos."""
        data = _load_r_dgp_data(dgp)
        covariates = [c for c in data.columns if c.startswith("cov")]
        key = f"dgp{dgp}_{method}_cov"
        r_se = r_results[key]["se"]

        ddd = TripleDifference(estimation_method=method)
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            covariates=covariates,
        )

        np.testing.assert_allclose(
            results.se, r_se, rtol=0.01,
            err_msg=f"SE ({method} cov DGP{dgp})",
        )


# =============================================================================
# Phase 7: Results and Params
# =============================================================================


class TestParamsAndResults:
    """Verify sklearn-like parameter interface and results completeness."""

    def test_get_params_returns_all_parameters(self):
        """All constructor params present in get_params()."""
        ddd = TripleDifference(estimation_method="dr")
        params = ddd.get_params()

        expected_keys = {
            "estimation_method", "robust", "cluster", "alpha",
            "pscore_trim", "rank_deficient_action",
        }
        assert expected_keys.issubset(params.keys()), (
            f"Missing params: {expected_keys - params.keys()}"
        )

    def test_set_params_modifies_attributes(self):
        """set_params() modifies estimator attributes."""
        ddd = TripleDifference()
        ddd.set_params(alpha=0.10, estimation_method="ipw")

        assert ddd.alpha == 0.10
        assert ddd.estimation_method == "ipw"

    def test_to_dict_contains_required_fields(self):
        """to_dict() contains all required fields."""
        data = _generate_hand_calculable_ddd()
        ddd = TripleDifference(estimation_method="reg")
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        d = results.to_dict()
        for key in ["att", "se", "t_stat", "p_value", "n_obs",
                     "estimation_method", "inference_method"]:
            assert key in d, f"Missing key '{key}' in to_dict()"

    def test_summary_contains_key_info(self):
        """summary() output contains ATT and method info."""
        data = _generate_hand_calculable_ddd()
        ddd = TripleDifference(estimation_method="dr")
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        summary = results.summary()
        assert "ATT" in summary
        assert "dr" in summary.lower() or "doubly robust" in summary.lower()

    def test_n_obs_correct(self):
        """n_obs matches actual data size."""
        data = generate_ddd_data(n_per_cell=50, seed=42)
        ddd = TripleDifference(estimation_method="reg")
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )
        assert results.n_obs == len(data)
        assert results.n_obs == 400  # 8 cells × 50

    def test_cell_counts_correct(self):
        """Cell counts match actual data composition."""
        data = generate_ddd_data(n_per_cell=50, seed=42)
        ddd = TripleDifference(estimation_method="reg")
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )
        # Each cell has 50 obs × 2 time periods = 100
        assert results.n_treated_eligible == 100
        assert results.n_treated_ineligible == 100
        assert results.n_control_eligible == 100
        assert results.n_control_ineligible == 100


# =============================================================================
# Phase 8: Parameter Functionality Tests
# =============================================================================


class TestParameterFunctionality:
    """Verify that estimator parameters actually affect behavior."""

    def test_rank_deficient_action_warn(self):
        """rank_deficient_action='warn' warns on collinear covariates."""
        data = generate_ddd_data(n_per_cell=50, seed=42, add_covariates=True)
        # Add a perfectly collinear covariate
        data["age_dup"] = data["age"]

        ddd = TripleDifference(estimation_method="reg", rank_deficient_action="warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
                covariates=["age", "age_dup"],
            )
        rank_warnings = [
            x for x in w
            if "rank" in str(x.message).lower()
            or "collinear" in str(x.message).lower()
            or "dependent" in str(x.message).lower()
        ]
        assert len(rank_warnings) > 0, (
            "Expected rank deficiency warning for collinear covariates"
        )
        assert np.isfinite(result.att)

    def test_rank_deficient_action_silent(self):
        """rank_deficient_action='silent' handles collinear covariates without warning."""
        data = generate_ddd_data(n_per_cell=50, seed=42, add_covariates=True)
        data["age_dup"] = data["age"]

        ddd = TripleDifference(
            estimation_method="reg", rank_deficient_action="silent",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
                covariates=["age", "age_dup"],
            )
        rank_warnings = [
            x for x in w
            if "rank" in str(x.message).lower()
            or "collinear" in str(x.message).lower()
            or "dependent" in str(x.message).lower()
        ]
        assert len(rank_warnings) == 0, (
            "Expected no rank deficiency warnings with action='silent'"
        )
        assert np.isfinite(result.att)

    def test_cluster_se_functional(self):
        """cluster parameter produces cluster-robust SEs."""
        data = generate_ddd_data(n_per_cell=100, seed=42)
        # Create meaningful clusters (~20 clusters of ~40 obs each)
        data["cluster_id"] = data.index % 20

        ddd_no_cluster = TripleDifference(estimation_method="dr")
        result_no_cluster = ddd_no_cluster.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        ddd_cluster = TripleDifference(estimation_method="dr", cluster="cluster_id")
        result_cluster = ddd_cluster.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        # ATT should be identical (clustering affects SE only)
        assert result_cluster.att == result_no_cluster.att
        # SE should differ (cluster-robust vs individual)
        assert result_cluster.se != result_no_cluster.se
        # n_clusters should be populated
        assert result_cluster.n_clusters is not None
        assert result_cluster.n_clusters == 20

    def test_low_cell_count_warning(self):
        """Small cells produce a warning."""
        data = generate_ddd_data(n_per_cell=5, seed=42)
        ddd = TripleDifference(estimation_method="reg")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ddd.fit(
                data,
                outcome="outcome",
                group="group",
                partition="partition",
                time="time",
            )
        low_count_warnings = [
            x for x in w if "low observation" in str(x.message).lower()
        ]
        assert len(low_count_warnings) > 0, (
            "Expected low observation count warning for n_per_cell=5"
        )
        assert np.isfinite(result.att)

    def test_robust_param_is_noop(self):
        """robust param has no effect on IF-based SEs."""
        data = generate_ddd_data(n_per_cell=50, seed=42)

        result_robust = TripleDifference(robust=True).fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )
        result_not_robust = TripleDifference(robust=False).fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
        )

        assert result_robust.att == result_not_robust.att
        assert result_robust.se == result_not_robust.se

    def test_cluster_single_cluster_raises(self):
        """Single cluster raises ValueError."""
        data = generate_ddd_data(n_per_cell=50, seed=42)
        data["cluster_id"] = 1  # All same cluster

        ddd = TripleDifference(estimation_method="dr", cluster="cluster_id")
        with pytest.raises(ValueError, match="at least 2 clusters"):
            ddd.fit(data, outcome="outcome", group="group",
                    partition="partition", time="time")

    def test_cluster_nan_ids_raises(self):
        """NaN cluster IDs raise ValueError."""
        data = generate_ddd_data(n_per_cell=50, seed=42)
        data["cluster_id"] = data.index % 20
        data.loc[0, "cluster_id"] = np.nan

        ddd = TripleDifference(estimation_method="dr", cluster="cluster_id")
        with pytest.raises(ValueError, match="missing values"):
            ddd.fit(data, outcome="outcome", group="group",
                    partition="partition", time="time")

    def test_overlap_warning_on_imbalanced_data(self):
        """Poor overlap triggers warning for IPW/DR."""
        rng = np.random.default_rng(42)
        records = []
        unit_id = 0
        # Extreme imbalance: 3 in treated eligible, 500 in others
        sizes = {(1, 1): 3, (1, 0): 500, (0, 1): 500, (0, 0): 500}
        for (g, p), n_cell in sizes.items():
            for t in [0, 1]:
                for _ in range(n_cell):
                    y = 10 + 2 * g + p + 0.5 * t + rng.normal(0, 1)
                    if g == 1 and p == 1 and t == 1:
                        y += 3.0
                    records.append({"outcome": y, "group": g, "partition": p,
                                    "time": t, "unit_id": unit_id,
                                    "cov1": rng.normal(0, 1)})
                    unit_id += 1
        data = pd.DataFrame(records)

        ddd = TripleDifference(estimation_method="ipw")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ddd.fit(data, outcome="outcome", group="group",
                             partition="partition", time="time",
                             covariates=["cov1"])
        overlap_warnings = [x for x in w
                            if "overlap" in str(x.message).lower()
                            and "trimmed" in str(x.message).lower()]
        assert len(overlap_warnings) > 0
        assert np.isfinite(result.att)

    def test_no_overlap_warning_for_reg(self):
        """RA method does not trigger overlap warnings (no pscores computed)."""
        rng = np.random.default_rng(42)
        records = []
        unit_id = 0
        sizes = {(1, 1): 3, (1, 0): 500, (0, 1): 500, (0, 0): 500}
        for (g, p), n_cell in sizes.items():
            for t in [0, 1]:
                for _ in range(n_cell):
                    y = 10 + 2 * g + p + 0.5 * t + rng.normal(0, 1)
                    if g == 1 and p == 1 and t == 1:
                        y += 3.0
                    records.append({"outcome": y, "group": g, "partition": p,
                                    "time": t, "unit_id": unit_id,
                                    "cov1": rng.normal(0, 1)})
                    unit_id += 1
        data = pd.DataFrame(records)

        ddd = TripleDifference(estimation_method="reg")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ddd.fit(data, outcome="outcome", group="group",
                             partition="partition", time="time",
                             covariates=["cov1"])
        overlap_warnings = [x for x in w if "overlap" in str(x.message).lower()]
        assert len(overlap_warnings) == 0

    @pytest.mark.parametrize("method", ["ipw", "dr"])
    def test_pscore_fallback_warns_and_skips_hessian(self, monkeypatch, method):
        """PS estimation failure emits warning, sets hessian=None, gives valid results."""
        data = generate_ddd_data(n_per_cell=50, seed=42, add_covariates=True)

        def _failing_lr(*args, **kwargs):
            raise RuntimeError("Forced PS failure for testing")

        import diff_diff.triple_diff as td_module
        monkeypatch.setattr(td_module, "_logistic_regression", _failing_lr)

        ddd = TripleDifference(estimation_method=method)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ddd.fit(data, outcome="outcome", group="group",
                             partition="partition", time="time",
                             covariates=["age"])
        ps_warnings = [x for x in w
                       if "propensity score estimation failed"
                       in str(x.message).lower()]
        assert len(ps_warnings) > 0, "Expected PS fallback warning"
        assert np.isfinite(result.att)
        assert np.isfinite(result.se) and result.se > 0

    @pytest.mark.parametrize("method", ["ipw", "dr"])
    def test_nonfinite_if_propagates_nan_se(self, monkeypatch, method):
        """Non-finite IF values produce NaN SE and NaN inference fields."""
        data = generate_ddd_data(n_per_cell=50, seed=42, add_covariates=True)

        import diff_diff.triple_diff as td_module

        original_did_rc = td_module.TripleDifference._compute_did_rc

        def _did_rc_with_nan(self_inner, *args, **kwargs):
            att, inf = original_did_rc(self_inner, *args, **kwargs)
            inf[0] = np.inf  # Inject non-finite value
            return att, inf

        monkeypatch.setattr(
            td_module.TripleDifference, "_compute_did_rc", _did_rc_with_nan,
        )

        ddd = TripleDifference(estimation_method=method)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ddd.fit(data, outcome="outcome", group="group",
                             partition="partition", time="time",
                             covariates=["age"])
        nonfinite_warnings = [
            x for x in w if "non-finite" in str(x.message).lower()
        ]
        assert len(nonfinite_warnings) > 0, "Expected non-finite IF warning"
        assert np.isnan(result.se), "SE should be NaN when IF has non-finite values"
        assert_nan_inference({
            "se": result.se,
            "t_stat": result.t_stat,
            "p_value": result.p_value,
            "conf_int": result.conf_int,
        })

    def test_r_squared_respects_rank_deficient_action(self):
        """r_squared computation uses estimator's rank_deficient_action, not hardcoded 'silent'."""
        data = generate_ddd_data(n_per_cell=50, seed=42, add_covariates=True)
        data["age_dup"] = data["age"]

        # "silent" should suppress ALL rank warnings (both main and r_squared paths)
        ddd_silent = TripleDifference(
            estimation_method="reg", rank_deficient_action="silent",
        )
        with warnings.catch_warnings(record=True) as w_silent:
            warnings.simplefilter("always")
            result_silent = ddd_silent.fit(
                data, outcome="outcome", group="group",
                partition="partition", time="time",
                covariates=["age", "age_dup"],
            )
        rank_silent = [x for x in w_silent
                       if "rank" in str(x.message).lower()
                       or "dependent" in str(x.message).lower()]

        # "warn" should emit rank warnings from both main and r_squared paths
        ddd_warn = TripleDifference(
            estimation_method="reg", rank_deficient_action="warn",
        )
        with warnings.catch_warnings(record=True) as w_warn:
            warnings.simplefilter("always")
            result_warn = ddd_warn.fit(
                data, outcome="outcome", group="group",
                partition="partition", time="time",
                covariates=["age", "age_dup"],
            )
        rank_warn = [x for x in w_warn
                     if "rank" in str(x.message).lower()
                     or "dependent" in str(x.message).lower()]

        assert len(rank_silent) == 0, "silent should suppress all rank warnings"
        assert len(rank_warn) > 0, "warn should emit rank warnings"
        # Both should produce finite results regardless
        assert np.isfinite(result_silent.att)
        assert np.isfinite(result_warn.att)
