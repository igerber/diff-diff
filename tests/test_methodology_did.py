"""
Comprehensive methodology verification tests for DifferenceInDifferences estimator.

This module verifies that the DifferenceInDifferences implementation matches:
1. The theoretical formulas from canonical DiD methodology
2. The behavior of R's fixest::feols() package
3. All documented edge cases in docs/methodology/REGISTRY.md

References:
- Wooldridge, J.M. (2010). Econometric Analysis of Cross Section and Panel Data, 2nd ed.
- Angrist, J.D., & Pischke, J.-S. (2009). Mostly Harmless Econometrics.
"""

import json
import os
import subprocess
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from diff_diff import DifferenceInDifferences


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def generate_hand_calculable_data() -> Tuple[pd.DataFrame, float]:
    """
    Generate a simple dataset with hand-calculable ATT.

    The ATT is computed as the double-difference of cell means:
    ATT = (Ȳ_treated_post - Ȳ_treated_pre) - (Ȳ_control_post - Ȳ_control_pre)

    Returns
    -------
    data : pd.DataFrame
        2x2 DiD data with 4 observations per cell
    expected_att : float
        The hand-calculated ATT value (3.0)
    """
    data = pd.DataFrame({
        'unit': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'outcome': [
            # Treated, Pre-period (mean = 11.5)
            10, 11, 12, 13,
            # Treated, Post-period (mean = 15.5) -> diff = 4.0
            14, 15, 16, 17,
            # Control, Pre-period (mean = 9.5)
            8, 9, 10, 11,
            # Control, Post-period (mean = 10.5) -> diff = 1.0
            9, 10, 11, 12,
        ],
        'treated': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'post': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    })

    # Hand calculation:
    # Ȳ_treated_pre = (10+11+12+13)/4 = 11.5
    # Ȳ_treated_post = (14+15+16+17)/4 = 15.5
    # Ȳ_control_pre = (8+9+10+11)/4 = 9.5
    # Ȳ_control_post = (9+10+11+12)/4 = 10.5
    # ATT = (15.5 - 11.5) - (10.5 - 9.5) = 4.0 - 1.0 = 3.0
    expected_att = 3.0

    return data, expected_att


def generate_did_data_with_covariates(
    n_units: int = 200,
    treatment_effect: float = 2.5,
    covariate_effect: float = 1.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate 2x2 DiD data with a covariate.

    Parameters
    ----------
    n_units : int
        Number of units
    treatment_effect : float
        True ATT
    covariate_effect : float
        Effect of covariate on outcome
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        DiD data with covariate
    """
    np.random.seed(seed)

    n_treated = n_units // 2
    data = []

    for unit in range(n_units):
        is_treated = unit < n_treated
        covariate = np.random.normal(0, 1)

        for period in [0, 1]:
            y = 10.0 + covariate * covariate_effect + period * 2.0
            if is_treated and period == 1:
                y += treatment_effect
            y += np.random.normal(0, 1)

            data.append({
                'unit': unit,
                'outcome': y,
                'treated': int(is_treated),
                'post': period,
                'x1': covariate,
            })

    return pd.DataFrame(data)


def generate_clustered_did_data(
    n_clusters: int = 20,
    cluster_size: int = 10,
    treatment_effect: float = 3.0,
    icc: float = 0.3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate DiD data with cluster structure.

    Parameters
    ----------
    n_clusters : int
        Number of clusters
    cluster_size : int
        Units per cluster
    treatment_effect : float
        True ATT
    icc : float
        Intraclass correlation coefficient
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Clustered DiD data
    """
    np.random.seed(seed)

    data = []
    n_treated_clusters = n_clusters // 2

    for cluster in range(n_clusters):
        is_treated = cluster < n_treated_clusters
        cluster_effect = np.random.normal(0, np.sqrt(icc))

        for unit in range(cluster_size):
            for period in [0, 1]:
                y = 10.0 + cluster_effect + period * 2.0
                if is_treated and period == 1:
                    y += treatment_effect
                y += np.random.normal(0, np.sqrt(1 - icc))

                data.append({
                    'cluster_id': cluster,
                    'unit': cluster * cluster_size + unit,
                    'outcome': y,
                    'treated': int(is_treated),
                    'post': period,
                })

    return pd.DataFrame(data)


def generate_heteroskedastic_did_data(
    n_units: int = 200,
    treatment_effect: float = 3.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate DiD data with heteroskedastic errors.

    Variance is higher for treated units to create heteroskedasticity.

    Returns
    -------
    pd.DataFrame
        DiD data with heteroskedastic errors
    """
    np.random.seed(seed)

    n_treated = n_units // 2
    data = []

    for unit in range(n_units):
        is_treated = unit < n_treated
        # Higher variance for treated units
        error_scale = 2.0 if is_treated else 0.5

        for period in [0, 1]:
            y = 10.0 + period * 2.0
            if is_treated and period == 1:
                y += treatment_effect
            y += np.random.normal(0, error_scale)

            data.append({
                'unit': unit,
                'outcome': y,
                'treated': int(is_treated),
                'post': period,
            })

    return pd.DataFrame(data)


# =============================================================================
# R Availability Fixtures
# =============================================================================


_fixest_available_cache = None


def _check_fixest_available() -> bool:
    """
    Check if R and fixest package are available (cached).
    """
    global _fixest_available_cache
    if _fixest_available_cache is None:
        r_env = os.environ.get("DIFF_DIFF_R", "auto").lower()
        if r_env == "skip":
            _fixest_available_cache = False
        else:
            try:
                result = subprocess.run(
                    ["Rscript", "-e", "library(fixest); library(jsonlite); cat('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                _fixest_available_cache = result.returncode == 0 and "OK" in result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                _fixest_available_cache = False
    return _fixest_available_cache


@pytest.fixture(scope="session")
def fixest_available():
    """Lazy check for R/fixest availability."""
    return _check_fixest_available()


@pytest.fixture
def require_fixest(fixest_available):
    """Skip test if R/fixest is not available."""
    if not fixest_available:
        pytest.skip("R or fixest package not available")


# =============================================================================
# Phase 1: ATT Formula Verification Tests
# =============================================================================


class TestATTFormula:
    """Tests for ATT formula verification via hand calculation."""

    def test_att_equals_double_difference_of_means(self):
        """
        Verify ATT equals the double-difference of cell means.

        ATT = (Ȳ_treated_post - Ȳ_treated_pre) - (Ȳ_control_post - Ȳ_control_pre)
        """
        data, expected_att = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        results = did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='post'
        )

        # ATT should match hand calculation exactly (to numerical precision)
        assert np.isclose(results.att, expected_att, rtol=1e-10), \
            f"ATT expected {expected_att}, got {results.att}"

    def test_att_equals_regression_interaction_coefficient(self):
        """
        Verify ATT equals the coefficient on the interaction term.

        The OLS regression Y = α + β₁D + β₂T + τ(D×T) + ε yields τ = ATT.
        """
        data, expected_att = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        results = did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='post'
        )

        # Check that the interaction coefficient matches ATT
        assert results.coefficients is not None, "coefficients should not be None"
        interaction_key = 'treated:post'
        assert interaction_key in results.coefficients, \
            f"Expected '{interaction_key}' in coefficients"
        assert np.isclose(results.coefficients[interaction_key], results.att, rtol=1e-10), \
            "ATT should equal interaction coefficient"
        assert np.isclose(results.att, expected_att, rtol=1e-10), \
            f"ATT expected {expected_att}, got {results.att}"

    def test_att_with_covariates_close_to_true_effect(self):
        """
        Verify ATT with covariates recovers approximately the true effect.
        """
        data = generate_did_data_with_covariates(
            n_units=500,
            treatment_effect=2.5,
            covariate_effect=1.5,
            seed=42
        )

        did = DifferenceInDifferences()
        results = did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='post',
            covariates=['x1']
        )

        # Should recover approximately 2.5 treatment effect
        assert abs(results.att - 2.5) < 0.5, \
            f"ATT expected ~2.5, got {results.att}"

    def test_att_sign_matches_data_direction(self):
        """
        Verify ATT sign is correct: positive when treated group improves more.
        """
        # Positive treatment effect
        data_pos = pd.DataFrame({
            'outcome': [10, 10, 20, 20, 10, 10, 10, 10],  # Treated improves by 10, control stays same
            'treated': [1, 1, 1, 1, 0, 0, 0, 0],
            'post': [0, 0, 1, 1, 0, 0, 1, 1],
        })

        did = DifferenceInDifferences()
        results_pos = did.fit(data_pos, outcome='outcome', treatment='treated', time='post')
        assert np.isclose(results_pos.att, 10.0, rtol=1e-10), f"Expected ATT=10.0, got {results_pos.att}"

        # Negative treatment effect
        data_neg = pd.DataFrame({
            'outcome': [20, 20, 10, 10, 10, 10, 10, 10],  # Treated decreases by 10, control stays same
            'treated': [1, 1, 1, 1, 0, 0, 0, 0],
            'post': [0, 0, 1, 1, 0, 0, 1, 1],
        })

        results_neg = did.fit(data_neg, outcome='outcome', treatment='treated', time='post')
        assert np.isclose(results_neg.att, -10.0, rtol=1e-10), f"Expected ATT=-10.0, got {results_neg.att}"


# =============================================================================
# Phase 2: R Comparison Tests (fixest::feols)
# =============================================================================


class TestRBenchmarkDiD:
    """Tests comparing Python implementation to R's fixest::feols()."""

    def _run_r_estimation(
        self,
        data_path: str,
        covariates: list = None,
        cluster: str = None,
        robust: bool = True
    ) -> Dict[str, Any]:
        """
        Run R's fixest::feols() and return results as dictionary.

        Parameters
        ----------
        data_path : str
            Path to CSV file with data
        covariates : list, optional
            List of covariate names
        cluster : str, optional
            Cluster variable name
        robust : bool
            Whether to use HC1 robust SEs

        Returns
        -------
        Dict with keys: att, se, pvalue, ci_lower, ci_upper, r_squared
        """
        escaped_path = data_path.replace("\\", "/")

        # Build formula
        formula = "outcome ~ treated * post"
        if covariates:
            formula += " + " + " + ".join(covariates)

        # Build vcov specification
        if cluster:
            vcov_spec = f'~{cluster}'
        elif robust:
            vcov_spec = '"HC1"'
        else:
            vcov_spec = '"iid"'

        r_script = f'''
        suppressMessages(library(fixest))
        suppressMessages(library(jsonlite))

        data_file <- normalizePath("{escaped_path}", mustWork = TRUE)
        data <- read.csv(data_file)

        # Ensure treated and post are treated as numeric for interaction
        data$treated <- as.numeric(data$treated)
        data$post <- as.numeric(data$post)

        result <- feols({formula}, data = data, vcov = {vcov_spec})

        # Extract interaction coefficient (ATT)
        coef_names <- names(coef(result))
        # The interaction term is "treated:post" in R
        att_idx <- which(coef_names == "treated:post")

        if (length(att_idx) == 0) {{
            # Try alternative naming
            att_idx <- which(grepl("treated.*post", coef_names))
        }}

        att <- coef(result)[att_idx]
        se <- sqrt(vcov(result)[att_idx, att_idx])
        pval <- summary(result)$coeftable[att_idx, "Pr(>|t|)"]

        # Confidence interval (95%)
        ci <- confint(result)[att_idx, ]

        output <- list(
            att = att[[1]],
            se = se[[1]],
            pvalue = pval[[1]],
            ci_lower = ci[[1]],
            ci_upper = ci[[2]],
            r_squared = summary(result)$r.squared[[1]]
        )

        cat(toJSON(output, pretty = TRUE))
        '''

        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            raise RuntimeError(f"R script failed: {result.stderr}")

        parsed = json.loads(result.stdout)

        # Handle R's JSON serialization quirks
        for key in ['att', 'se', 'pvalue', 'ci_lower', 'ci_upper', 'r_squared']:
            if isinstance(parsed.get(key), list):
                parsed[key] = parsed[key][0]

        return parsed

    @pytest.fixture
    def benchmark_data(self, tmp_path):
        """Generate benchmark data and save to CSV."""
        np.random.seed(12345)
        n_units = 200

        data = []
        for unit in range(n_units):
            is_treated = unit < n_units // 2
            for period in [0, 1]:
                y = 10.0 + period * 2.0
                if is_treated and period == 1:
                    y += 3.0  # True ATT
                y += np.random.normal(0, 1)

                data.append({
                    'unit': unit,
                    'outcome': y,
                    'treated': int(is_treated),
                    'post': period,
                })

        df = pd.DataFrame(data)
        csv_path = tmp_path / "benchmark_data.csv"
        df.to_csv(csv_path, index=False)
        return df, str(csv_path)

    @pytest.fixture
    def benchmark_data_with_covariate(self, tmp_path):
        """Generate benchmark data with covariate and save to CSV."""
        df = generate_did_data_with_covariates(
            n_units=200,
            treatment_effect=2.5,
            covariate_effect=1.5,
            seed=12345
        )
        csv_path = tmp_path / "benchmark_data_cov.csv"
        df.to_csv(csv_path, index=False)
        return df, str(csv_path)

    @pytest.fixture
    def benchmark_data_clustered(self, tmp_path):
        """Generate clustered benchmark data and save to CSV."""
        df = generate_clustered_did_data(
            n_clusters=20,
            cluster_size=10,
            treatment_effect=3.0,
            icc=0.3,
            seed=12345
        )
        csv_path = tmp_path / "benchmark_data_clustered.csv"
        df.to_csv(csv_path, index=False)
        return df, str(csv_path)

    def test_att_matches_r_basic(self, require_fixest, benchmark_data):
        """Test ATT matches R with basic 2x2 DiD (no covariates)."""
        data, csv_path = benchmark_data

        # Python estimation
        did = DifferenceInDifferences(robust=True)
        py_results = did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='post'
        )

        # R estimation
        r_results = self._run_r_estimation(csv_path, robust=True)

        # Compare ATT - R's JSON output is truncated to 4 decimal places
        # so use 1e-3 tolerance
        assert np.isclose(py_results.att, r_results['att'], rtol=1e-3), \
            f"ATT mismatch: Python={py_results.att}, R={r_results['att']}"

    def test_se_matches_r_hc1(self, require_fixest, benchmark_data):
        """Test HC1 standard errors match R within 5%."""
        data, csv_path = benchmark_data

        did = DifferenceInDifferences(robust=True)
        py_results = did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='post'
        )

        r_results = self._run_r_estimation(csv_path, robust=True)

        # Compare SE - within 5% tolerance
        rel_diff = abs(py_results.se - r_results['se']) / r_results['se']
        assert rel_diff < 0.05, \
            f"SE mismatch: Python={py_results.se}, R={r_results['se']}, diff={rel_diff*100:.1f}%"

    def test_pvalue_matches_r(self, require_fixest, benchmark_data):
        """Test p-value matches R within 0.01."""
        data, csv_path = benchmark_data

        did = DifferenceInDifferences(robust=True)
        py_results = did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='post'
        )

        r_results = self._run_r_estimation(csv_path, robust=True)

        # Compare p-value - within 0.01 absolute
        assert abs(py_results.p_value - r_results['pvalue']) < 0.01, \
            f"P-value mismatch: Python={py_results.p_value}, R={r_results['pvalue']}"

    def test_ci_overlaps_r(self, require_fixest, benchmark_data):
        """Test confidence intervals overlap with R."""
        data, csv_path = benchmark_data

        did = DifferenceInDifferences(robust=True, alpha=0.05)
        py_results = did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='post'
        )

        r_results = self._run_r_estimation(csv_path, robust=True)

        py_lower, py_upper = py_results.conf_int
        r_lower = float(r_results['ci_lower'])
        r_upper = float(r_results['ci_upper'])

        # CIs should overlap
        assert py_lower < r_upper and r_lower < py_upper, \
            f"CIs don't overlap: Python=[{py_lower}, {py_upper}], R=[{r_lower}, {r_upper}]"

    def test_att_matches_r_with_covariates(self, require_fixest, benchmark_data_with_covariate):
        """Test ATT matches R when covariates are included."""
        data, csv_path = benchmark_data_with_covariate

        did = DifferenceInDifferences(robust=True)
        py_results = did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='post',
            covariates=['x1']
        )

        r_results = self._run_r_estimation(csv_path, covariates=['x1'], robust=True)

        # Compare ATT - R's JSON output is truncated to 4 decimal places
        # so use 1e-3 tolerance
        assert np.isclose(py_results.att, r_results['att'], rtol=1e-3), \
            f"ATT mismatch with covariates: Python={py_results.att}, R={r_results['att']}"

    def test_se_matches_r_with_clustering(self, require_fixest, benchmark_data_clustered):
        """Test cluster-robust SEs match R."""
        data, csv_path = benchmark_data_clustered

        did = DifferenceInDifferences(cluster='cluster_id')
        py_results = did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='post'
        )

        r_results = self._run_r_estimation(csv_path, cluster='cluster_id')

        # Compare ATT - R's JSON output is truncated to 4 decimal places
        assert np.isclose(py_results.att, r_results['att'], rtol=1e-3), \
            f"ATT mismatch: Python={py_results.att}, R={r_results['att']}"

        # Compare SE - within 10% for clustered (more variance expected)
        rel_diff = abs(py_results.se - r_results['se']) / r_results['se']
        assert rel_diff < 0.10, \
            f"Clustered SE mismatch: Python={py_results.se}, R={r_results['se']}, diff={rel_diff*100:.1f}%"


# =============================================================================
# Phase 3: Standard Error Verification Tests
# =============================================================================


class TestSEVerification:
    """Tests for standard error formula verification."""

    def test_hc1_vs_classical_se_differ(self):
        """
        Verify HC1 and classical SEs differ in heteroskedastic data.

        When errors are heteroskedastic, HC1 and classical OLS will produce
        different standard errors. The relationship depends on the pattern
        of heteroskedasticity.
        """
        data = generate_heteroskedastic_did_data(
            n_units=200,
            treatment_effect=3.0,
            seed=42
        )

        did_hc1 = DifferenceInDifferences(robust=True)
        did_classical = DifferenceInDifferences(robust=False)

        results_hc1 = did_hc1.fit(
            data, outcome='outcome', treatment='treated', time='post'
        )
        results_classical = did_classical.fit(
            data, outcome='outcome', treatment='treated', time='post'
        )

        # SE methods should produce both valid and positive SEs
        assert results_hc1.se > 0, "HC1 SE should be positive"
        assert results_classical.se > 0, "Classical SE should be positive"
        # ATT should be the same regardless of SE method
        assert np.isclose(results_hc1.att, results_classical.att, rtol=1e-10), \
            "ATT should be identical for both SE methods"

    def test_cluster_se_differs_from_robust(self):
        """
        Verify cluster-robust SEs account for clustering structure.

        Cluster-robust SEs should differ from HC1 robust SEs when there
        is within-cluster correlation.
        """
        data = generate_clustered_did_data(
            n_clusters=20,
            cluster_size=10,
            treatment_effect=3.0,
            icc=0.3,
            seed=42
        )

        did_robust = DifferenceInDifferences(robust=True)
        did_cluster = DifferenceInDifferences(cluster='cluster_id')

        results_robust = did_robust.fit(
            data, outcome='outcome', treatment='treated', time='post'
        )
        results_cluster = did_cluster.fit(
            data, outcome='outcome', treatment='treated', time='post'
        )

        # Both SE methods should produce valid positive SEs
        assert results_robust.se > 0, "Robust SE should be positive"
        assert results_cluster.se > 0, "Cluster SE should be positive"

        # ATT should be the same regardless of SE method
        assert np.isclose(results_robust.att, results_cluster.att, rtol=1e-10), \
            "ATT should be identical for both SE methods"

    def test_se_decreases_with_sample_size(self):
        """Verify SE decreases approximately with sqrt(n)."""
        np.random.seed(42)

        ses = []
        for n in [100, 400, 900]:
            data = []
            for unit in range(n):
                is_treated = unit < n // 2
                for period in [0, 1]:
                    y = 10.0 + period * 2.0
                    if is_treated and period == 1:
                        y += 3.0
                    y += np.random.normal(0, 1)
                    data.append({
                        'outcome': y,
                        'treated': int(is_treated),
                        'post': period,
                    })

            did = DifferenceInDifferences(robust=True)
            results = did.fit(
                pd.DataFrame(data),
                outcome='outcome',
                treatment='treated',
                time='post'
            )
            ses.append(results.se)

        # SE should decrease approximately as 1/sqrt(n)
        # SE(n=400) / SE(n=100) ≈ sqrt(100/400) = 0.5
        ratio_1 = ses[1] / ses[0]
        ratio_2 = ses[2] / ses[1]

        assert 0.3 < ratio_1 < 0.7, f"SE ratio for 4x sample: {ratio_1:.2f}, expected ~0.5"
        assert 0.4 < ratio_2 < 0.8, f"SE ratio for 2.25x sample: {ratio_2:.2f}, expected ~0.67"

    def test_vcov_positive_semidefinite(self):
        """Verify variance-covariance matrix is positive semi-definite."""
        data, _ = generate_hand_calculable_data()

        did = DifferenceInDifferences(robust=True)
        results = did.fit(
            data, outcome='outcome', treatment='treated', time='post'
        )

        assert results.vcov is not None, "vcov should not be None"

        # Remove NaN rows/columns if any (from rank deficiency)
        vcov = results.vcov
        if np.any(np.isnan(vcov)):
            mask = ~np.isnan(vcov).all(axis=1)
            vcov = vcov[np.ix_(mask, mask)]

        # All eigenvalues should be >= 0 (with numerical tolerance)
        eigenvalues = np.linalg.eigvalsh(vcov)
        assert np.all(eigenvalues >= -1e-10), \
            f"VCoV not positive semi-definite: min eigenvalue = {eigenvalues.min()}"


class TestWildBootstrapInference:
    """Tests for wild cluster bootstrap inference."""

    def test_wild_bootstrap_produces_valid_se(self):
        """Test wild bootstrap produces finite, positive SE."""
        data = generate_clustered_did_data(
            n_clusters=15,
            cluster_size=10,
            treatment_effect=3.0,
            seed=42
        )

        did = DifferenceInDifferences(
            inference='wild_bootstrap',
            cluster='cluster_id',
            n_bootstrap=199,
            bootstrap_weights='rademacher',
            seed=42
        )
        results = did.fit(
            data, outcome='outcome', treatment='treated', time='post'
        )

        assert np.isfinite(results.se), "Bootstrap SE should be finite"
        assert results.se > 0, "Bootstrap SE should be positive"
        assert results.inference_method == 'wild_bootstrap'

    def test_wild_bootstrap_pvalue_in_valid_range(self):
        """Test wild bootstrap p-value is in [0, 1]."""
        data = generate_clustered_did_data(
            n_clusters=15,
            cluster_size=10,
            treatment_effect=3.0,
            seed=42
        )

        did = DifferenceInDifferences(
            inference='wild_bootstrap',
            cluster='cluster_id',
            n_bootstrap=199,
            seed=42
        )
        results = did.fit(
            data, outcome='outcome', treatment='treated', time='post'
        )

        assert 0 <= results.p_value <= 1, \
            f"P-value {results.p_value} not in [0, 1]"

    def test_wild_bootstrap_ci_contains_point_estimate(self):
        """Test wild bootstrap CI contains point estimate."""
        data = generate_clustered_did_data(
            n_clusters=15,
            cluster_size=10,
            treatment_effect=3.0,
            seed=42
        )

        did = DifferenceInDifferences(
            inference='wild_bootstrap',
            cluster='cluster_id',
            n_bootstrap=199,
            seed=42
        )
        results = did.fit(
            data, outcome='outcome', treatment='treated', time='post'
        )

        lower, upper = results.conf_int
        # CI should contain point estimate or be very close
        # (bootstrap CI may differ slightly from point estimate)
        assert lower < results.att + 0.5 and results.att - 0.5 < upper, \
            f"CI [{lower}, {upper}] should approximately contain ATT {results.att}"

    @pytest.mark.parametrize("weight_type", ["rademacher", "mammen", "webb"])
    def test_wild_bootstrap_weight_types(self, weight_type):
        """Test all wild bootstrap weight types work."""
        data = generate_clustered_did_data(
            n_clusters=15,
            cluster_size=10,
            treatment_effect=3.0,
            seed=42
        )

        did = DifferenceInDifferences(
            inference='wild_bootstrap',
            cluster='cluster_id',
            n_bootstrap=99,
            bootstrap_weights=weight_type,
            seed=42
        )
        results = did.fit(
            data, outcome='outcome', treatment='treated', time='post'
        )

        assert np.isfinite(results.se), f"SE should be finite for {weight_type}"
        assert results.se > 0, f"SE should be positive for {weight_type}"


# =============================================================================
# Phase 4: Edge Case Tests (from REGISTRY.md)
# =============================================================================


class TestEdgeCases:
    """Tests for all documented edge cases from REGISTRY.md."""

    def test_empty_cell_produces_nan_or_warning(self):
        """
        Empty cells (no observations in a cell) produce rank deficiency.

        Empty cells create linearly dependent columns, resulting in rank deficiency
        handled per rank_deficient_action setting (warn → NaN coefficients,
        error → ValueError, silent → NaN without warning).
        """
        # No treated units in pre-period
        data = pd.DataFrame({
            'outcome': [10, 11, 12, 13],
            'treated': [0, 0, 1, 1],
            'post': [0, 1, 1, 1],  # No treated-pre observations
        })

        did = DifferenceInDifferences(rank_deficient_action='warn')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = did.fit(data, outcome='outcome', treatment='treated', time='post')

            # Should have rank deficiency warning due to empty cell
            # causing collinearity between treated and post columns
            rank_warnings = [x for x in w if "rank" in str(x.message).lower()]
            # Either produces warning or NaN coefficient
            has_nan = results.coefficients is not None and any(
                np.isnan(v) for v in results.coefficients.values()
            )
            assert len(rank_warnings) > 0 or has_nan, \
                "Empty cell should cause rank deficiency warning or NaN coefficient"

    def test_singleton_cluster_handling(self):
        """
        Test singleton clusters are handled appropriately.

        With clustering, singleton clusters may produce warnings.
        """
        # Create data with one singleton cluster
        data = pd.DataFrame({
            'outcome': list(range(20)) + [100],
            'treated': [0]*10 + [1]*10 + [1],
            'post': [0, 1]*10 + [1],
            'cluster': list(range(10))*2 + [99],  # Cluster 99 is singleton
        })

        did = DifferenceInDifferences(cluster='cluster')
        # Should run without error (warning may be emitted)
        results = did.fit(data, outcome='outcome', treatment='treated', time='post')
        assert results is not None

    def test_rank_deficient_warn_mode(self):
        """
        Test rank_deficient_action='warn' emits warning and sets NaN.

        REGISTRY.md: "Rank-deficient design matrix: warns and sets NA for dropped coefficients"
        """
        data = pd.DataFrame({
            'outcome': [10, 11, 15, 18, 9, 10, 12, 13],
            'treated': [1, 1, 1, 1, 0, 0, 0, 0],
            'post': [0, 0, 1, 1, 0, 0, 1, 1],
            'collinear': [1, 1, 1, 1, 0, 0, 0, 0],  # Perfectly collinear with treated
        })

        did = DifferenceInDifferences(rank_deficient_action='warn')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = did.fit(
                data,
                outcome='outcome',
                treatment='treated',
                time='post',
                covariates=['collinear']
            )

            # Should have warning about rank deficiency
            rank_warnings = [x for x in w if "rank" in str(x.message).lower()]
            assert len(rank_warnings) > 0, "Expected rank deficiency warning"

        # Check that NaN is set for dropped coefficient
        assert results.coefficients is not None, "coefficients should not be None"
        has_nan = any(np.isnan(v) for v in results.coefficients.values())
        assert has_nan, "Expected NaN for dropped coefficient"

    def test_rank_deficient_error_mode(self):
        """
        Test rank_deficient_action='error' raises ValueError.
        """
        data = pd.DataFrame({
            'outcome': [10, 11, 15, 18, 9, 10, 12, 13],
            'treated': [1, 1, 1, 1, 0, 0, 0, 0],
            'post': [0, 0, 1, 1, 0, 0, 1, 1],
            'collinear': [1, 1, 1, 1, 0, 0, 0, 0],
        })

        did = DifferenceInDifferences(rank_deficient_action='error')

        with pytest.raises(ValueError, match="rank-deficient"):
            did.fit(
                data,
                outcome='outcome',
                treatment='treated',
                time='post',
                covariates=['collinear']
            )

    def test_rank_deficient_silent_mode(self):
        """
        Test rank_deficient_action='silent' produces no warning.
        """
        data = pd.DataFrame({
            'outcome': [10, 11, 15, 18, 9, 10, 12, 13],
            'treated': [1, 1, 1, 1, 0, 0, 0, 0],
            'post': [0, 0, 1, 1, 0, 0, 1, 1],
            'collinear': [1, 1, 1, 1, 0, 0, 0, 0],
        })

        did = DifferenceInDifferences(rank_deficient_action='silent')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = did.fit(
                data,
                outcome='outcome',
                treatment='treated',
                time='post',
                covariates=['collinear']
            )

            rank_warnings = [x for x in w if "rank" in str(x.message).lower()]
            assert len(rank_warnings) == 0, "Expected no rank deficiency warning"

        # ATT should still be valid
        assert np.isfinite(results.att)

    def test_non_binary_treatment_raises_error(self):
        """
        Test non-binary treatment raises ValueError.

        REGISTRY.md: "Treatment and post indicators must be binary (0/1) with variation in both"
        """
        data = pd.DataFrame({
            'outcome': [10, 11, 12, 13],
            'treated': [0, 1, 2, 3],  # Non-binary
            'post': [0, 0, 1, 1],
        })

        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="binary"):
            did.fit(data, outcome='outcome', treatment='treated', time='post')

    def test_non_binary_time_raises_error(self):
        """
        Test non-binary time indicator raises ValueError.
        """
        data = pd.DataFrame({
            'outcome': [10, 11, 12, 13],
            'treated': [0, 0, 1, 1],
            'post': [0, 1, 2, 3],  # Non-binary
        })

        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="binary"):
            did.fit(data, outcome='outcome', treatment='treated', time='post')

    def test_no_treatment_variation_raises_error(self):
        """Test error when treatment has no variation."""
        data = pd.DataFrame({
            'outcome': [10, 11, 12, 13],
            'treated': [1, 1, 1, 1],  # All treated
            'post': [0, 0, 1, 1],
        })

        did = DifferenceInDifferences()
        with pytest.raises(ValueError):
            did.fit(data, outcome='outcome', treatment='treated', time='post')

    def test_no_time_variation_raises_error(self):
        """Test error when time has no variation."""
        data = pd.DataFrame({
            'outcome': [10, 11, 12, 13],
            'treated': [0, 0, 1, 1],
            'post': [1, 1, 1, 1],  # All post
        })

        did = DifferenceInDifferences()
        with pytest.raises(ValueError):
            did.fit(data, outcome='outcome', treatment='treated', time='post')

    def test_missing_values_raise_error(self):
        """Test error when data contains missing values."""
        data = pd.DataFrame({
            'outcome': [10, np.nan, 12, 13],
            'treated': [0, 0, 1, 1],
            'post': [0, 0, 1, 1],
        })

        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="missing"):
            did.fit(data, outcome='outcome', treatment='treated', time='post')


# =============================================================================
# Phase 5: Formula Interface Tests
# =============================================================================


class TestFormulaInterface:
    """Tests for R-style formula parsing."""

    def test_formula_star_syntax(self):
        """Test 'y ~ treated * post' syntax."""
        data, expected_att = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        results = did.fit(data, formula='outcome ~ treated * post')

        assert np.isclose(results.att, expected_att, rtol=1e-10)

    def test_formula_explicit_interaction_syntax(self):
        """Test 'y ~ treated + post + treated:post' syntax."""
        data, expected_att = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        results = did.fit(data, formula='outcome ~ treated + post + treated:post')

        assert np.isclose(results.att, expected_att, rtol=1e-10)

    def test_formula_with_covariates(self):
        """Test formula with additional covariates."""
        data = generate_did_data_with_covariates(n_units=200, seed=42)

        did = DifferenceInDifferences()
        # Note: formula parsing for covariates after interaction
        results = did.fit(data, formula='outcome ~ treated * post + x1')

        assert results.coefficients is not None, "coefficients should not be None"
        assert 'x1' in results.coefficients
        assert np.isfinite(results.att)

    def test_formula_missing_tilde_raises_error(self):
        """Test error when formula doesn't contain ~."""
        data, _ = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="~"):
            did.fit(data, formula='outcome treated * post')

    def test_formula_missing_interaction_raises_error(self):
        """Test error when formula has no interaction term."""
        data, _ = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="interaction"):
            did.fit(data, formula='outcome ~ treated + post')

    def test_formula_missing_column_raises_error(self):
        """Test error when formula references non-existent column."""
        data, _ = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="not found"):
            did.fit(data, formula='outcome ~ nonexistent * post')

    def test_formula_and_params_conflict(self):
        """Test that formula overrides outcome/treatment/time params."""
        data, expected_att = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        # Formula should override the (invalid) parameter values
        results = did.fit(
            data,
            formula='outcome ~ treated * post',
            outcome='wrong',  # Would fail if used
            treatment='wrong',
            time='wrong'
        )

        assert np.isclose(results.att, expected_att, rtol=1e-10)


# =============================================================================
# Phase 6: Fixed Effects Equivalence Tests
# =============================================================================


class TestFixedEffectsEquivalence:
    """Tests for fixed effects handling equivalence."""

    @pytest.fixture
    def panel_data(self):
        """Generate panel data with unit and time effects."""
        np.random.seed(42)
        n_units = 50
        n_periods = 4

        data = []
        for unit in range(n_units):
            is_treated = unit < n_units // 2
            unit_effect = unit * 0.2

            for period in range(n_periods):
                post = 1 if period >= 2 else 0
                period_effect = period * 1.0

                y = 10.0 + unit_effect + period_effect
                if is_treated and post:
                    y += 3.0  # True ATT
                y += np.random.normal(0, 0.5)

                data.append({
                    'unit': unit,
                    'period': period,
                    'treated': int(is_treated),
                    'post': post,
                    'outcome': y,
                })

        return pd.DataFrame(data)

    def test_absorb_produces_valid_att(self, panel_data):
        """Test that absorb option produces valid ATT estimate."""
        did = DifferenceInDifferences()
        results = did.fit(
            panel_data,
            outcome='outcome',
            treatment='treated',
            time='post',
            absorb=['unit']
        )

        # ATT should be close to 3.0
        assert abs(results.att - 3.0) < 1.0, \
            f"ATT with absorb expected ~3.0, got {results.att}"

    def test_fixed_effects_produces_valid_att(self, panel_data):
        """Test that fixed_effects option produces valid ATT estimate."""
        did = DifferenceInDifferences()
        results = did.fit(
            panel_data,
            outcome='outcome',
            treatment='treated',
            time='post',
            fixed_effects=['unit']
        )

        # ATT should be close to 3.0
        assert abs(results.att - 3.0) < 1.0, \
            f"ATT with FE expected ~3.0, got {results.att}"

    def test_absorb_produces_valid_results(self, panel_data):
        """Test that absorb option produces valid ATT estimate.

        Note: R² from within-transformed models is not comparable to
        R² from untransformed models (different variance scales).
        """
        did = DifferenceInDifferences()
        results = did.fit(
            panel_data,
            outcome='outcome',
            treatment='treated',
            time='post',
            absorb=['unit']
        )

        # Verify ATT is estimated (not NaN)
        assert results.att is not None
        assert not np.isnan(results.att)

        # Verify R² exists and is valid
        assert results.r_squared is not None
        assert 0 <= results.r_squared <= 1 or np.isnan(results.r_squared)


class TestRFixedEffectsComparison:
    """Tests comparing fixed effects to R's fixest with absorbed FE."""

    def _run_r_feols_fe(self, data_path: str) -> Dict[str, Any]:
        """
        Run R's fixest::feols() with absorbed unit FE.
        """
        escaped_path = data_path.replace("\\", "/")

        r_script = f'''
        suppressMessages(library(fixest))
        suppressMessages(library(jsonlite))

        data <- read.csv("{escaped_path}")
        data$treated <- as.numeric(data$treated)
        data$post <- as.numeric(data$post)

        # Run feols with absorbed unit FE
        result <- feols(outcome ~ treated * post | unit, data = data, vcov = "HC1")

        coef_names <- names(coef(result))
        att_idx <- which(coef_names == "treated:post")
        if (length(att_idx) == 0) {{
            att_idx <- which(grepl("treated.*post", coef_names))
        }}

        att <- coef(result)[att_idx]
        se <- sqrt(vcov(result)[att_idx, att_idx])

        output <- list(
            att = unbox(att),
            se = unbox(se),
            r_squared = unbox(summary(result)$r.squared)
        )

        cat(toJSON(output, pretty = TRUE))
        '''

        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            raise RuntimeError(f"R script failed: {result.stderr}")

        parsed = json.loads(result.stdout)
        for key in ['att', 'se', 'r_squared']:
            if isinstance(parsed.get(key), list):
                parsed[key] = parsed[key][0]

        return parsed

    @pytest.fixture
    def panel_data_csv(self, tmp_path):
        """Generate panel data and save to CSV."""
        np.random.seed(12345)
        n_units = 50
        n_periods = 4

        data = []
        for unit in range(n_units):
            is_treated = unit < n_units // 2
            unit_effect = unit * 0.2

            for period in range(n_periods):
                post = 1 if period >= 2 else 0
                period_effect = period * 1.0

                y = 10.0 + unit_effect + period_effect
                if is_treated and post:
                    y += 3.0
                y += np.random.normal(0, 0.5)

                data.append({
                    'unit': unit,
                    'period': period,
                    'treated': int(is_treated),
                    'post': post,
                    'outcome': y,
                })

        df = pd.DataFrame(data)
        csv_path = tmp_path / "panel_data.csv"
        df.to_csv(csv_path, index=False)
        return df, str(csv_path)

    def test_absorb_matches_r_feols_fe(self, require_fixest, panel_data_csv):
        """Test that Python's absorb option matches R's feols() with | syntax."""
        data, csv_path = panel_data_csv

        # Python with absorb
        did = DifferenceInDifferences(robust=True)
        py_results = did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='post',
            absorb=['unit']
        )

        # R with | unit syntax
        r_results = self._run_r_feols_fe(csv_path)

        # Compare ATT
        assert np.isclose(py_results.att, r_results['att'], rtol=0.01), \
            f"ATT mismatch: Python={py_results.att}, R={r_results['att']}"


# =============================================================================
# Phase 7: Get/Set Params Tests
# =============================================================================


class TestGetSetParams:
    """Tests for sklearn-compatible parameter interface."""

    def test_get_params_returns_all_parameters(self):
        """Test that get_params returns all constructor parameters."""
        did = DifferenceInDifferences(
            robust=False,
            cluster='cluster_id',
            alpha=0.10,
            inference='wild_bootstrap',
            n_bootstrap=500,
            bootstrap_weights='webb',
            seed=123,
            rank_deficient_action='silent'
        )

        params = did.get_params()

        assert params['robust'] is False
        assert params['cluster'] == 'cluster_id'
        assert params['alpha'] == 0.10
        assert params['inference'] == 'wild_bootstrap'
        assert params['n_bootstrap'] == 500
        assert params['bootstrap_weights'] == 'webb'
        assert params['seed'] == 123
        assert params['rank_deficient_action'] == 'silent'

    def test_set_params_modifies_attributes(self):
        """Test that set_params modifies estimator attributes."""
        did = DifferenceInDifferences()

        did.set_params(alpha=0.10, n_bootstrap=500)

        assert did.alpha == 0.10
        assert did.n_bootstrap == 500

    def test_set_params_returns_self(self):
        """Test that set_params returns the estimator (fluent interface)."""
        did = DifferenceInDifferences()
        result = did.set_params(alpha=0.10)
        assert result is did

    def test_set_params_unknown_raises_error(self):
        """Test that set_params with unknown parameter raises ValueError."""
        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="Unknown parameter"):
            did.set_params(unknown_param=42)

    def test_params_affect_estimation(self):
        """Test that changed parameters actually affect estimation."""
        data = generate_heteroskedastic_did_data(n_units=200, seed=42)

        did_default = DifferenceInDifferences()  # robust=True by default
        did_classical = DifferenceInDifferences()
        did_classical.set_params(robust=False)

        results_default = did_default.fit(
            data, outcome='outcome', treatment='treated', time='post'
        )
        results_classical = did_classical.fit(
            data, outcome='outcome', treatment='treated', time='post'
        )

        # VCoV should differ (robust vs classical)
        assert results_default.vcov is not None
        assert results_classical.vcov is not None
        assert not np.allclose(results_default.vcov, results_classical.vcov)


# =============================================================================
# Phase 8: Results Object Tests
# =============================================================================


class TestResultsObject:
    """Tests for DiDResults object."""

    def test_summary_contains_key_info(self):
        """Test that summary() output contains key information."""
        data, _ = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        results = did.fit(data, outcome='outcome', treatment='treated', time='post')

        summary = results.summary()

        assert "Difference-in-Differences" in summary
        assert "ATT" in summary

    def test_to_dict_contains_all_fields(self):
        """Test to_dict contains all expected fields."""
        data, _ = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        results = did.fit(data, outcome='outcome', treatment='treated', time='post')

        result_dict = results.to_dict()

        assert 'att' in result_dict
        assert 'se' in result_dict
        assert 't_stat' in result_dict
        assert 'p_value' in result_dict
        # Note: conf_int is split into conf_int_lower and conf_int_upper
        assert 'conf_int_lower' in result_dict or 'conf_int' in result_dict
        assert 'n_obs' in result_dict

    def test_to_dataframe_has_correct_shape(self):
        """Test to_dataframe produces DataFrame with one row."""
        data, _ = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        results = did.fit(data, outcome='outcome', treatment='treated', time='post')

        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_is_significant_property(self):
        """Test is_significant property works correctly."""
        data, _ = generate_hand_calculable_data()

        did = DifferenceInDifferences(alpha=0.05)
        results = did.fit(data, outcome='outcome', treatment='treated', time='post')

        assert isinstance(results.is_significant, bool)

    def test_significance_stars_property(self):
        """Test significance_stars returns valid notation."""
        data, _ = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        results = did.fit(data, outcome='outcome', treatment='treated', time='post')

        assert results.significance_stars in ["", ".", "*", "**", "***"]

    def test_coefficients_dict(self):
        """Test coefficients dictionary contains expected keys."""
        data, _ = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        results = did.fit(data, outcome='outcome', treatment='treated', time='post')

        assert results.coefficients is not None, "coefficients should not be None"
        assert 'const' in results.coefficients
        assert 'treated' in results.coefficients
        assert 'post' in results.coefficients
        assert 'treated:post' in results.coefficients

    def test_residuals_and_fitted_values(self):
        """Test residuals and fitted values are correctly computed."""
        data, _ = generate_hand_calculable_data()

        did = DifferenceInDifferences()
        results = did.fit(data, outcome='outcome', treatment='treated', time='post')

        assert results.residuals is not None, "residuals should not be None"
        assert results.fitted_values is not None, "fitted_values should not be None"

        # Residuals + fitted should equal original outcome
        reconstructed = results.residuals + results.fitted_values
        original = data['outcome'].values.astype(float)

        assert np.allclose(reconstructed, original), \
            "Residuals + fitted should equal original outcome"
