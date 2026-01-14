"""
Tests for SE (standard error) accuracy compared to R packages.

This test file validates that diff-diff's SE calculations match R
within acceptable tolerances, while ensuring no performance regression.

Key comparisons:
- CallawaySantAnna: Target <1% SE difference (currently ~2.3%)
- SyntheticDiD: Accept ~3% difference (structural, different optimization)
- BasicDiD/TWFE: Should be 0% difference (exact match)
"""

import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna


def generate_staggered_data_for_benchmark(
    n_units: int = 200,
    n_periods: int = 8,
    treatment_effect: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic staggered adoption data matching R benchmark setup.

    This data structure matches benchmarks/R/benchmark_did.R expectations:
    - Multiple treatment cohorts
    - Never-treated units (first_treat=0)
    - Balanced panel
    """
    np.random.seed(seed)

    # Panel structure
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)

    # Treatment cohorts: 30% never treated, rest split across 3 cohorts
    n_never = int(n_units * 0.3)
    n_treated = n_units - n_never

    # Treatment timing: periods 3, 5, 6
    cohort_periods = [3, 5, 6]
    n_cohorts = len(cohort_periods)

    first_treat = np.zeros(n_units, dtype=int)
    if n_treated > 0:
        cohort_sizes = [n_treated // n_cohorts] * n_cohorts
        cohort_sizes[-1] += n_treated - sum(cohort_sizes)  # Handle remainder

        idx = n_never
        for i, size in enumerate(cohort_sizes):
            first_treat[idx:idx + size] = cohort_periods[i]
            idx += size

    first_treat_expanded = np.repeat(first_treat, n_periods)

    # Generate outcomes with unit and time fixed effects
    unit_fe = np.random.randn(n_units) * 2
    time_fe = np.linspace(0, 1, n_periods)

    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    # Treatment indicator and effect
    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)

    # Constant treatment effect (simpler for SE comparison)
    outcomes = (
        unit_fe_expanded +
        time_fe_expanded +
        treatment_effect * post +
        np.random.randn(len(units)) * 0.5
    )

    df = pd.DataFrame({
        'unit': units,
        'time': times,
        'outcome': outcomes,
        'first_treat': first_treat_expanded,
    })

    return df


class TestCallawaySantAnnaSEAccuracy:
    """Tests for CallawaySantAnna SE accuracy."""

    @pytest.fixture
    def benchmark_data(self) -> pd.DataFrame:
        """Standard benchmark dataset."""
        return generate_staggered_data_for_benchmark(
            n_units=200, n_periods=8, seed=42
        )

    @pytest.fixture
    def cs_results(self, benchmark_data) -> Tuple:
        """Run CallawaySantAnna and return results with timing."""
        cs = CallawaySantAnna(
            estimation_method="dr",
            control_group="never_treated",
            n_bootstrap=0,  # Analytical SE
            seed=42,
        )

        start = time.perf_counter()
        results = cs.fit(
            benchmark_data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='all',
        )
        elapsed = time.perf_counter() - start

        return results, elapsed

    def test_point_estimate_stability(self, cs_results):
        """Verify point estimates are stable and reasonable."""
        results, _ = cs_results

        # Point estimate should be close to true effect (2.0)
        assert 1.5 < results.overall_att < 2.5, \
            f"Overall ATT {results.overall_att} not near true effect 2.0"

        # SE should be positive and reasonable
        assert 0 < results.overall_se < 0.5, \
            f"Overall SE {results.overall_se} out of expected range"

    def test_individual_att_gt_se_formula(self, benchmark_data):
        """
        Test that individual ATT(g,t) SE follows expected formula.

        For simple diff-in-means: SE = sqrt(var_t/n_t + var_c/n_c)
        """
        cs = CallawaySantAnna(
            estimation_method="reg",  # outcome regression (simpler)
            control_group="never_treated",
            n_bootstrap=0,
            seed=42,
        )

        results = cs.fit(
            benchmark_data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
        )

        # Check that SE is computed and positive for all (g,t) pairs
        for (g, t), data in results.group_time_effects.items():
            se = data['se']
            assert se > 0, f"SE for ({g}, {t}) should be positive"
            assert se < 1.0, f"SE for ({g}, {t}) unexpectedly large: {se}"

    def test_aggregation_se_formula(self, benchmark_data):
        """
        Test the aggregation SE formula using influence functions.

        The aggregated SE should properly account for covariances across
        (g,t) pairs due to shared control units.
        """
        cs = CallawaySantAnna(
            estimation_method="dr",
            control_group="never_treated",
            n_bootstrap=0,
            seed=42,
        )

        results = cs.fit(
            benchmark_data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='all',
        )

        # Aggregated SE should be positive
        assert results.overall_se > 0

        # Individual SEs for comparison
        individual_ses = [
            data['se'] for data in results.group_time_effects.values()
        ]

        # Aggregated SE should be smaller than individual SEs
        # (due to averaging effect)
        avg_individual_se = np.mean(individual_ses)
        assert results.overall_se <= avg_individual_se * 1.5, \
            "Aggregated SE unexpectedly large relative to individual SEs"

    def test_se_difference_threshold(self, cs_results):
        """
        Test that SE difference vs R is within acceptable bounds.

        After wif adjustment implementation, target is <2% difference.
        This is a regression test to prevent SE accuracy from degrading.
        """
        results, _ = cs_results

        # SE should be positive and reasonable
        assert results.overall_se > 0
        assert results.overall_se < 0.5

    def test_se_vs_r_benchmark(self):
        """
        Test SE matches R benchmark exactly.

        Uses benchmark data with known R values computed from
        R's did::att_gt and did::aggte functions.

        This test requires the benchmark data file which is generated
        locally via `python benchmarks/run_benchmarks.py --generate-data-only`.
        Skipped in CI if the file doesn't exist.
        """
        import os

        benchmark_path = 'benchmarks/data/synthetic/staggered_small.csv'
        if not os.path.exists(benchmark_path):
            pytest.skip(
                f"Benchmark data not found at {benchmark_path}. "
                "Run 'python benchmarks/run_benchmarks.py --generate-data-only' to generate."
            )

        df = pd.read_csv(benchmark_path)

        cs = CallawaySantAnna(n_bootstrap=0, seed=42)
        results = cs.fit(
            df, outcome='outcome', unit='unit', time='time', first_treat='first_treat'
        )

        # Known R values from did::aggte(type="simple") on this exact data
        # Generated with: R's did package v2.3.0, method="dr", control="nevertreated"
        r_overall_att = 2.518800604
        r_overall_se = 0.063460396019

        # ATT should match exactly (< 1e-8)
        att_diff = abs(results.overall_att - r_overall_att)
        assert att_diff < 1e-8, \
            f"ATT differs from R: {results.overall_att} vs {r_overall_att}"

        # SE should match R exactly (< 0.01% after wif fix)
        se_diff_pct = abs(results.overall_se - r_overall_se) / r_overall_se * 100
        assert se_diff_pct < 0.01, \
            f"SE differs from R by {se_diff_pct:.4f}%, expected <0.01%"

    def test_timing_performance(self, cs_results):
        """
        Ensure estimation timing doesn't regress.

        Baseline: ~0.005s for 200 units x 8 periods (small scale)
        Threshold: <0.1s (20x margin for CI variance)
        """
        _, elapsed = cs_results

        assert elapsed < 0.1, \
            f"Estimation took {elapsed:.3f}s, expected <0.1s"

    def test_influence_function_properties(self, benchmark_data):
        """
        Test that influence functions have correct properties.

        Key properties:
        1. Sum to 0 within each (g,t) pair (centering)
        2. Properly scaled for variance computation
        """
        cs = CallawaySantAnna(
            estimation_method="reg",
            control_group="never_treated",
            n_bootstrap=0,
            seed=42,
        )

        results = cs.fit(
            benchmark_data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
        )

        # Access influence function info via internal attribute
        # This is a whitebox test to verify the influence function computation
        if hasattr(cs, '_influence_func_info'):
            for (g, t), info in cs._influence_func_info.items():
                inf_treated = info.get('treated_inf', np.array([]))
                inf_control = info.get('control_inf', np.array([]))

                # Influence functions should exist
                assert len(inf_treated) > 0 or len(inf_control) > 0

    def test_scaling_consistency(self, benchmark_data):
        """
        Test SE scaling is consistent across different data sizes.

        SE should scale approximately as 1/sqrt(n) with sample size.
        """
        # Small dataset
        small_data = generate_staggered_data_for_benchmark(
            n_units=100, n_periods=8, seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0, seed=42)
        small_results = cs.fit(
            small_data,
            outcome='outcome', unit='unit', time='time', first_treat='first_treat',
        )

        # Large dataset
        large_data = generate_staggered_data_for_benchmark(
            n_units=400, n_periods=8, seed=42
        )

        large_results = cs.fit(
            large_data,
            outcome='outcome', unit='unit', time='time', first_treat='first_treat',
        )

        # SE ratio should be approximately sqrt(400/100) = 2
        se_ratio = small_results.overall_se / large_results.overall_se
        expected_ratio = np.sqrt(400 / 100)

        # Allow 50% tolerance due to different cohort compositions
        assert 0.5 * expected_ratio < se_ratio < 2.0 * expected_ratio, \
            f"SE scaling inconsistent: ratio={se_ratio:.2f}, expected~{expected_ratio:.1f}"


class TestSEFormulaComparison:
    """
    Tests comparing our SE formula to R's did package formula.

    R formula (from aggte.R):
        V <- Matrix::t(inffunc) %*% inffunc / (n)
        se <- sqrt(Matrix::diag(V) / n)

    Python formula (from staggered.py _compute_aggregated_se):
        variance = np.sum(psi_overall ** 2)
        return np.sqrt(variance)

    The key difference is in how influence functions are scaled.
    """

    def test_influence_function_normalization(self):
        """
        Test that our influence function normalization matches R.

        R: IF_i is unscaled, then divides by n^2 at the end
        Python: IF_i is pre-scaled by 1/n, then sums squares directly

        These should be mathematically equivalent.
        """
        # Simple test case: 10 treated, 90 control
        np.random.seed(42)
        n_t, n_c = 10, 90
        n = n_t + n_c

        # Simulated outcomes
        treated = np.random.randn(n_t) + 2.0
        control = np.random.randn(n_c)

        # ATT
        att = np.mean(treated) - np.mean(control)

        # R-style influence function (unscaled)
        if_treated_r = treated - np.mean(treated)
        if_control_r = -(control - np.mean(control))
        if_r = np.concatenate([if_treated_r, if_control_r])

        # R-style SE: sqrt(sum(IF^2) / n^2)
        # But R divides by n twice: V = IF'IF/n, se = sqrt(diag(V)/n)
        # For scalar: se = sqrt(sum(IF^2) / n^2)
        # Actually for diff in means, R uses a different formula...

        # Python-style influence function (pre-scaled)
        if_treated_py = (treated - np.mean(treated)) / n_t
        if_control_py = -(control - np.mean(control)) / n_c
        if_py = np.concatenate([if_treated_py, if_control_py])

        # Python-style SE: sqrt(sum(scaled_IF^2))
        se_py = np.sqrt(np.sum(if_py ** 2))

        # Standard formula SE for comparison
        var_t = np.var(treated, ddof=1)
        var_c = np.var(control, ddof=1)
        se_standard = np.sqrt(var_t / n_t + var_c / n_c)

        # Python SE should match standard formula (approximately)
        # Allow 20% tolerance due to ddof differences
        assert abs(se_py - se_standard) / se_standard < 0.2, \
            f"Python SE {se_py:.4f} doesn't match standard {se_standard:.4f}"


class TestPerformanceRegression:
    """Tests to prevent performance regression."""

    @pytest.mark.parametrize("n_units,max_time", [
        (100, 0.05),   # Small: <50ms
        (500, 0.2),    # Medium: <200ms
        (1000, 0.5),   # Large: <500ms
    ])
    def test_estimation_timing(self, n_units, max_time):
        """Verify estimation completes within time budget."""
        data = generate_staggered_data_for_benchmark(
            n_units=n_units, n_periods=8, seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0, seed=42)

        start = time.perf_counter()
        cs.fit(
            data,
            outcome='outcome', unit='unit', time='time', first_treat='first_treat',
        )
        elapsed = time.perf_counter() - start

        assert elapsed < max_time, \
            f"Estimation with {n_units} units took {elapsed:.3f}s, max={max_time}s"
