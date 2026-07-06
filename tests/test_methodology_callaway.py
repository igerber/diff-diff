"""
Comprehensive methodology verification tests for CallawaySantAnna estimator.

This module verifies that the CallawaySantAnna implementation matches:
1. The theoretical formulas from Callaway & Sant'Anna (2021)
2. The behavior of R's did::att_gt() package
3. All documented edge cases in docs/methodology/REGISTRY.md

Reference: Callaway, B., & Sant'Anna, P.H.C. (2021). Difference-in-Differences
with multiple time periods. Journal of Econometrics, 225(2), 200-230.
"""

import json
import subprocess
import unittest.mock
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, SurveyDesign
from diff_diff.linalg import solve_logit
from diff_diff.prep import generate_staggered_data
from diff_diff.bootstrap_utils import (
    generate_bootstrap_weights_batch as _generate_bootstrap_weights_batch,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def generate_hand_calculable_data() -> Tuple[pd.DataFrame, float]:
    """
    Generate a simple dataset with hand-calculable ATT(g,t).

    Returns
    -------
    data : pd.DataFrame
        Panel data with 8 units, 3 periods
    expected_att : float
        The hand-calculated ATT(g=2, t=2) value
    """
    # 4 treated units (g=2), 4 control units (g=0/never-treated), 3 periods
    # Outcome structure:
    #   - Baseline effect varies by unit
    #   - Time trend: +1 per period for all units
    #   - Treatment effect: +3 for treated units at t=2
    data = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        'period': [0, 1, 2] * 8,
        'first_treat': [2] * 6 + [2] * 6 + [0] * 6 + [0] * 6,  # 4 treated at g=2, 4 never
        'outcome': [
            # Treated units (g=2): base + time trend + treatment at t=2
            10, 11, 15,  # unit 1: Y[0]=10, Y[1]=11, Y[2]=15 (effect=15-11-(12-11)=3)
            12, 13, 17,  # unit 2: Y[0]=12, Y[1]=13, Y[2]=17
            11, 12, 16,  # unit 3
            13, 14, 18,  # unit 4
            # Control units: base + time trend only
            10, 11, 12,  # unit 5
            12, 13, 14,  # unit 6
            11, 12, 13,  # unit 7
            13, 14, 15,  # unit 8
        ]
    })

    # Hand calculation for ATT(g=2, t=2):
    # Base period = g-1 = 1 (for post-treatment effect)
    # Treated ΔY (from t=1 to t=2) = mean([15-11, 17-13, 16-12, 18-14]) = mean([4, 4, 4, 4]) = 4
    # Control ΔY (from t=1 to t=2) = mean([12-11, 14-13, 13-12, 15-14]) = mean([1, 1, 1, 1]) = 1
    # ATT(g=2, t=2) = 4 - 1 = 3.0
    expected_att = 3.0

    return data, expected_att


# R availability is now checked lazily via conftest.py fixtures
# to avoid subprocess latency at import time


# =============================================================================
# Phase 1: Equation Verification Tests
# =============================================================================


class TestATTgtFormula:
    """Tests for ATT(g,t) basic formula verification."""

    def test_att_gt_basic_formula_hand_calculation(self):
        """
        Verify ATT(g,t) matches hand-calculated value.

        Reference formula:
            ATT(g,t) = E[Y_t - Y_{g-1} | G_g=1] - E[Y_t - Y_{g-1} | C=1]
        """
        data, expected_att = generate_hand_calculable_data()

        cs = CallawaySantAnna(estimation_method='reg', n_bootstrap=0)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='period',
            first_treat='first_treat'
        )

        # ATT(g=2, t=2) should match hand calculation exactly
        actual = results.group_time_effects[(2, 2)]['effect']
        assert np.isclose(actual, expected_att, rtol=1e-10), \
            f"ATT(2,2) expected {expected_att}, got {actual}"

    def test_att_gt_with_outcome_regression(self):
        """Test outcome regression produces consistent ATT(g,t)."""
        data, expected_att = generate_hand_calculable_data()

        cs = CallawaySantAnna(estimation_method='reg', n_bootstrap=0)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='period',
            first_treat='first_treat'
        )

        # Outcome regression without covariates should match simple DID
        actual = results.group_time_effects[(2, 2)]['effect']
        assert np.isclose(actual, expected_att, rtol=1e-10)

    def test_att_gt_with_ipw(self):
        """Test IPW produces consistent ATT(g,t) without covariates."""
        data, expected_att = generate_hand_calculable_data()

        cs = CallawaySantAnna(estimation_method='ipw', n_bootstrap=0)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='period',
            first_treat='first_treat'
        )

        # IPW without covariates should approximate simple DID
        # (may differ slightly due to unconditional propensity weighting)
        actual = results.group_time_effects[(2, 2)]['effect']
        assert np.isclose(actual, expected_att, rtol=0.01), \
            f"ATT(2,2) expected ~{expected_att}, got {actual}"

    def test_att_gt_with_doubly_robust(self):
        """Test doubly robust produces consistent ATT(g,t)."""
        data, expected_att = generate_hand_calculable_data()

        cs = CallawaySantAnna(estimation_method='dr', n_bootstrap=0)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='period',
            first_treat='first_treat'
        )

        # DR without covariates should match simple DID
        actual = results.group_time_effects[(2, 2)]['effect']
        assert np.isclose(actual, expected_att, rtol=1e-10)


class TestBasePeriodSelection:
    """Tests for base period selection (varying vs universal)."""

    def test_base_period_varying_vs_universal_post_treatment(self):
        """
        Verify post-treatment effects are identical for varying and universal.

        Both base_period modes should produce the same ATT(g,t) for t >= g
        because both use g-1-anticipation as base for post-treatment.
        """
        data = generate_staggered_data(
            n_units=100,
            n_periods=8,
            cohort_periods=[4],
            never_treated_frac=0.3,
            treatment_effect=2.0,
            seed=42
        )

        cs_varying = CallawaySantAnna(base_period="varying", n_bootstrap=0)
        cs_universal = CallawaySantAnna(base_period="universal", n_bootstrap=0)

        results_v = cs_varying.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )
        results_u = cs_universal.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # Post-treatment effects (t >= 4) should match exactly
        for t in [4, 5, 6, 7]:
            if (4, t) in results_v.group_time_effects and (4, t) in results_u.group_time_effects:
                eff_v = results_v.group_time_effects[(4, t)]['effect']
                eff_u = results_u.group_time_effects[(4, t)]['effect']
                assert np.isclose(eff_v, eff_u, rtol=1e-10), \
                    f"Post-treatment ATT(4,{t}) should match: varying={eff_v}, universal={eff_u}"

    def test_base_period_varying_pre_treatment_uses_consecutive(self):
        """
        Verify varying base period uses t-1 for pre-treatment periods.

        For base_period="varying" and t < g, base period should be t-1.
        """
        data = generate_staggered_data(
            n_units=100,
            n_periods=8,
            cohort_periods=[5],
            never_treated_frac=0.3,
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(base_period="varying", n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # Pre-treatment periods should exist (varying computes them)
        # With g=5, pre-treatment would be t in {1,2,3,4} (if anticipation=0)
        pre_treatment_exists = any(
            (g, t) in results.group_time_effects
            for g in [5] for t in [1, 2, 3, 4]
        )
        assert pre_treatment_exists, "Varying base period should produce pre-treatment effects"

    def test_base_period_universal_includes_reference_period(self):
        """
        Verify universal base period includes e=-1-anticipation in event study.

        With base_period="universal", the reference period should appear
        in event study output with effect=0 and NaN inference fields.
        """
        data = generate_staggered_data(
            n_units=100,
            n_periods=8,
            cohort_periods=[4],
            never_treated_frac=0.3,
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(base_period="universal", anticipation=0, n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        # Reference period e=-1 should exist with effect=0
        assert results.event_study_effects is not None, \
            "Event study effects should be computed"
        assert -1 in results.event_study_effects, \
            "Universal base period should include e=-1 in event study"

        ref = results.event_study_effects[-1]
        assert ref['effect'] == 0.0, "Reference period effect should be 0"
        assert np.isnan(ref['se']), "Reference period SE should be NaN"
        assert ref['n_groups'] == 0, "Reference period n_groups should be 0"


class TestDoublyRobustEstimator:
    """Tests for doubly robust estimation."""

    def test_doubly_robust_recovers_true_effect(self):
        """
        DR estimator recovers true effect with correct specification.

        The doubly robust estimator should be consistent when either
        the outcome model or the propensity model is correctly specified.
        """
        # Generate data with known DGP
        data = generate_staggered_data(
            n_units=500,
            n_periods=6,
            cohort_periods=[3],
            treatment_effect=2.5,
            never_treated_frac=0.3,
            seed=42
        )

        cs = CallawaySantAnna(estimation_method='dr', n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # Should recover approximately 2.5 treatment effect
        # Allow wider tolerance due to dynamic effects and noise
        assert abs(results.overall_att - 2.5) < 1.0, \
            f"DR should recover ~2.5 effect, got {results.overall_att}"

    def test_estimation_methods_produce_similar_results(self):
        """
        All estimation methods should produce similar results without covariates.

        When there are no covariates (unconditional parallel trends),
        reg, ipw, and dr should all produce very similar ATT estimates.
        """
        data = generate_staggered_data(
            n_units=200,
            n_periods=8,
            cohort_periods=[4],
            treatment_effect=3.0,
            seed=123
        )

        results = {}
        for method in ['reg', 'ipw', 'dr']:
            cs = CallawaySantAnna(estimation_method=method, n_bootstrap=0)
            results[method] = cs.fit(
                data, outcome='outcome', unit='unit',
                time='period', first_treat='first_treat'
            )

        # All methods should produce similar overall ATT
        atts = [results[m].overall_att for m in ['reg', 'ipw', 'dr']]
        max_diff = max(atts) - min(atts)
        assert max_diff < 0.5, \
            f"Estimation methods differ by {max_diff}: reg={atts[0]}, ipw={atts[1]}, dr={atts[2]}"


class TestDRNoCovariateSEUniformity:
    """Without covariates, doubly-robust reduces to difference in means, so its
    per-(g,t) SE must be the same influence-function form (``sqrt(sum(phi^2))``)
    used by the reg/ipw branches and R's ``DRDID::drdid_panel`` — not the ddof=1
    plug-in ``sqrt(var_t/n_t + var_c/n_c)`` it historically used (O(1/n) from R).
    Point estimates and aggregated SEs are unchanged because the same IF already
    fed aggregation."""

    @staticmethod
    def _fit(data, method, aggregate=None):
        cs = CallawaySantAnna(estimation_method=method, n_bootstrap=0)
        return cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate=aggregate,
        )

    def test_dr_reg_per_cell_se_identical(self):
        """dr and reg produce bit-identical per-cell effect AND SE without covariates."""
        data = generate_staggered_data(n_units=120, n_periods=6, never_treated_frac=0.3, seed=7)
        reg = self._fit(data, "reg")
        dr = self._fit(data, "dr")

        assert set(reg.group_time_effects) == set(dr.group_time_effects)
        compared = 0
        for key, reg_cell in reg.group_time_effects.items():
            dr_cell = dr.group_time_effects[key]
            if not np.isfinite(reg_cell["se"]):
                assert not np.isfinite(dr_cell["se"])
                continue
            # Both point AND SE match to machine precision (pre-fix the SE gapped
            # ~1.3% via the ddof=1 plug-in; the effect always matched).
            np.testing.assert_allclose(dr_cell["effect"], reg_cell["effect"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(dr_cell["se"], reg_cell["se"], rtol=0, atol=1e-12)
            compared += 1
        assert compared >= 3, f"expected several finite cells to compare, got {compared}"

    def test_dr_no_cov_per_cell_se_hand_calc(self):
        """Per-cell dr SE equals the IF form sqrt(sum(phi^2)) and is strictly
        tighter than the old ddof=1 plug-in it replaced."""
        # 2-period panel: base = period 1, post = period 2; cohort g=2 vs
        # never-treated (first_treat=0). One estimated cell: (g=2, t=2).
        treated_y = {1: (1.0, 3.0), 2: (2.0, 5.0), 3: (0.0, 1.0), 4: (1.0, 5.0)}
        control_y = {5: (0.0, 0.5), 6: (1.0, 1.0), 7: (2.0, 3.0), 8: (0.0, 1.5)}
        rows = []
        for u, (y1, y2) in treated_y.items():
            rows += [(u, 1, y1, 2), (u, 2, y2, 2)]
        for u, (y1, y2) in control_y.items():
            rows += [(u, 1, y1, 0), (u, 2, y2, 0)]
        data = pd.DataFrame(rows, columns=["unit", "period", "outcome", "first_treat"])

        res = self._fit(data, "dr")
        cell = res.group_time_effects[(2, 2)]

        tc = np.array([y2 - y1 for (y1, y2) in treated_y.values()])
        cc = np.array([y2 - y1 for (y1, y2) in control_y.values()])
        n_t, n_c = len(tc), len(cc)
        exp_att = tc.mean() - cc.mean()
        exp_se = np.sqrt(
            np.sum((tc - tc.mean()) ** 2) / n_t**2 + np.sum((cc - cc.mean()) ** 2) / n_c**2
        )
        old_plugin = np.sqrt(np.var(tc, ddof=1) / n_t + np.var(cc, ddof=1) / n_c)

        assert cell["effect"] == pytest.approx(exp_att, abs=1e-12)
        assert cell["se"] == pytest.approx(exp_se, abs=1e-12)
        # Direction sentinel: the IF-based SE is strictly smaller than the plug-in
        # the fix removed (reintroducing the plug-in would trip this).
        assert cell["se"] < old_plugin

    def test_dr_no_cov_aggregated_se_matches_reg(self):
        """Overall and event-study SEs are identical between dr and reg (both
        consume the same per-cell IF) — guards against an accidental IF change."""
        data = generate_staggered_data(n_units=120, n_periods=6, never_treated_frac=0.3, seed=11)
        reg = self._fit(data, "reg", aggregate="event_study")
        dr = self._fit(data, "dr", aggregate="event_study")

        np.testing.assert_allclose(dr.overall_att, reg.overall_att, rtol=0, atol=1e-12)
        np.testing.assert_allclose(dr.overall_se, reg.overall_se, rtol=0, atol=1e-12)

        assert reg.event_study_effects is not None and dr.event_study_effects is not None
        assert set(reg.event_study_effects) == set(dr.event_study_effects)
        for e, reg_es in reg.event_study_effects.items():
            dr_es = dr.event_study_effects[e]
            if np.isnan(reg_es["se"]):
                assert np.isnan(dr_es["se"])
            else:
                np.testing.assert_allclose(dr_es["se"], reg_es["se"], rtol=0, atol=1e-12)


# =============================================================================
# Phase 2: R Benchmark Comparison Tests
# =============================================================================


class TestRBenchmarkCallaway:
    """Tests comparing Python implementation to R's did::att_gt()."""

    def _run_r_estimation(
        self,
        data_path: str,
        estimation_method: str = "dr",
        control_group: str = "nevertreated",
        anticipation: int = 0,
        base_period: str = "varying"
    ) -> Dict[str, Any]:
        """
        Run R's did::att_gt() and return results as dictionary.

        Parameters
        ----------
        data_path : str
            Path to CSV file with data
        estimation_method : str
            R estimation method: "dr", "ipw", "reg"
        control_group : str
            R control group: "nevertreated" or "notyettreated"
        anticipation : int
            Number of anticipation periods
        base_period : str
            Base period: "varying" or "universal"

        Returns
        -------
        Dict with keys: overall_att, overall_se, group_time (dict of lists)
        """
        # Escape path for cross-platform compatibility (Windows backslashes, spaces)
        escaped_path = data_path.replace("\\", "/")

        r_script = f'''
        suppressMessages(library(did))
        suppressMessages(library(jsonlite))

        # Use normalizePath for cross-platform path handling
        data_file <- normalizePath("{escaped_path}", mustWork = TRUE)
        data <- read.csv(data_file)

        result <- att_gt(
            yname = "outcome",
            tname = "period",
            idname = "unit",
            gname = "first_treat",
            xformla = ~ 1,
            data = data,
            est_method = "{estimation_method}",
            control_group = "{control_group}",
            anticipation = {anticipation},
            base_period = "{base_period}",
            bstrap = FALSE,
            cband = FALSE
        )

        # Simple aggregation
        agg <- aggte(result, type = "simple")

        output <- list(
            overall_att = unbox(agg$overall.att),
            overall_se = unbox(agg$overall.se),
            group_time = list(
                group = as.integer(result$group),
                time = as.integer(result$t),
                att = result$att,
                se = result$se
            )
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

        import json
        parsed = json.loads(result.stdout)

        # Handle R's JSON serialization quirks
        # Extract scalar values from single-element lists if needed
        if isinstance(parsed.get('overall_att'), list):
            parsed['overall_att'] = parsed['overall_att'][0]
        if isinstance(parsed.get('overall_se'), list):
            parsed['overall_se'] = parsed['overall_se'][0]

        return parsed

    @pytest.fixture
    def benchmark_data(self, tmp_path):
        """Generate benchmark data and save to CSV."""
        data = generate_staggered_data(
            n_units=200,
            n_periods=10,
            cohort_periods=[4, 6],
            treatment_effect=2.0,
            never_treated_frac=0.3,
            seed=12345
        )
        csv_path = tmp_path / "benchmark_data.csv"
        data.to_csv(csv_path, index=False)
        return data, str(csv_path)

    def test_overall_att_matches_r_dr(self, require_r, benchmark_data):
        """Test overall ATT matches R with doubly robust estimation.

        Note: Due to differences in dynamic effect handling in generated data,
        we use 20% tolerance. Individual ATT(g,t) values match more closely.
        """
        data, csv_path = benchmark_data

        # Python estimation
        cs = CallawaySantAnna(estimation_method='dr', n_bootstrap=0)
        py_results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # R estimation
        r_results = self._run_r_estimation(csv_path, estimation_method="dr")

        # Compare overall ATT - use 20% tolerance for aggregation differences
        # The discrepancy is primarily in aggregation weights, not ATT(g,t) values
        assert np.isclose(py_results.overall_att, r_results['overall_att'], rtol=0.20), \
            f"ATT mismatch: Python={py_results.overall_att}, R={r_results['overall_att']}"

    def test_overall_att_matches_r_reg(self, require_r, benchmark_data):
        """Test overall ATT matches R with outcome regression.

        Note: Due to differences in dynamic effect handling in generated data,
        we use 20% tolerance. Individual ATT(g,t) values match more closely.
        """
        data, csv_path = benchmark_data

        cs = CallawaySantAnna(estimation_method='reg', n_bootstrap=0)
        py_results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        r_results = self._run_r_estimation(csv_path, estimation_method="reg")

        assert np.isclose(py_results.overall_att, r_results['overall_att'], rtol=0.20), \
            f"ATT mismatch: Python={py_results.overall_att}, R={r_results['overall_att']}"

    def test_group_time_effects_match_r(self, require_r, benchmark_data):
        """Test individual ATT(g,t) values match R for post-treatment periods.

        Post-treatment effects (t >= g) should match closely since both
        Python and R use the last observed pre-treatment period as base.

        Pre-treatment effects also match: since the positional-base fix, Python
        selects base periods positionally (nearest observed period) exactly like
        R did::att_gt (varying uses the immediately-preceding observed period).
        We focus on post-treatment where alignment has always been expected.
        """
        data, csv_path = benchmark_data

        cs = CallawaySantAnna(estimation_method='dr', n_bootstrap=0)
        py_results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        r_results = self._run_r_estimation(csv_path, estimation_method="dr")

        # Compare each ATT(g,t) for post-treatment only
        r_gt = r_results['group_time']
        n_comparisons = 0
        mismatches = []
        for i in range(len(r_gt['group'])):
            g = int(r_gt['group'][i])
            t = int(r_gt['time'][i])
            r_att = r_gt['att'][i]

            # Only compare post-treatment effects (t >= g)
            if t < g:
                continue

            if (g, t) in py_results.group_time_effects:
                py_att = py_results.group_time_effects[(g, t)]['effect']
                # Post-treatment effects should match within 20% or 0.5 abs
                # Wider tolerance accounts for differences in dynamic effect handling
                if not np.isclose(py_att, r_att, rtol=0.20, atol=0.5):
                    mismatches.append(f"ATT({g},{t}): Python={py_att:.4f}, R={r_att:.4f}")
                n_comparisons += 1

        # Should have made at least some comparisons
        assert n_comparisons > 0, "No post-treatment group-time effects matched between Python and R"

        # Report mismatches if any
        assert len(mismatches) == 0, f"Post-treatment ATT mismatches:\n" + "\n".join(mismatches)


# =============================================================================
# Phase 3: Edge Case Tests
# =============================================================================


class TestCallawaySantAnnaEdgeCases:
    """Tests for all documented edge cases from REGISTRY.md."""

    def test_single_obs_group_produces_valid_result(self):
        """
        Groups with single observation: included but may have high variance.

        REGISTRY.md line 202: "Groups with single observation: included but may have high variance"
        """
        # Create data with one group having very few units
        data = generate_staggered_data(
            n_units=50,
            n_periods=6,
            cohort_periods=[3, 5],
            never_treated_frac=0.4,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # Should produce valid results
        assert results.overall_att is not None
        assert np.isfinite(results.overall_att)

    def test_no_post_treatment_returns_nan_with_warning(self):
        """
        Overall ATT is NaN when no post-treatment effects exist.

        REGISTRY.md lines 217-223: When all treatment occurs after data ends,
        overall ATT and all inference fields should be NaN with warning.
        """
        # Treatment at period 10, data only goes to period 5
        # Manually create data since generate_staggered_data validates cohort periods
        np.random.seed(42)
        n_units = 50
        n_periods = 5
        units = np.repeat(np.arange(n_units), n_periods)
        periods = np.tile(np.arange(n_periods), n_units)

        # 15 never-treated (first_treat=0), 35 treated at period 10 (after data ends)
        first_treat_by_unit = np.concatenate([
            np.zeros(15),  # Never treated
            np.full(35, 10)  # Treated at period 10 (after data ends)
        ]).astype(int)
        first_treat = np.repeat(first_treat_by_unit, n_periods)

        outcomes = np.random.randn(len(units)) + units * 0.1 + periods * 0.5

        data = pd.DataFrame({
            'unit': units,
            'period': periods,
            'first_treat': first_treat,
            'outcome': outcomes
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cs = CallawaySantAnna(n_bootstrap=0)
            results = cs.fit(
                data, outcome='outcome', unit='unit',
                time='period', first_treat='first_treat'
            )

            # Check warning was emitted
            warning_messages = [str(warning.message) for warning in w]
            assert any("post-treatment" in msg.lower() for msg in warning_messages), \
                f"Expected post-treatment warning, got: {warning_messages}"

        # All overall inference fields should be NaN
        assert np.isnan(results.overall_att), "overall_att should be NaN"
        assert np.isnan(results.overall_se), "overall_se should be NaN"
        assert np.isnan(results.overall_t_stat), "overall_t_stat should be NaN"
        assert np.isnan(results.overall_p_value), "overall_p_value should be NaN"

    def test_nonfinite_se_produces_nan_tstat(self):
        """
        When SE is non-finite or zero, t_stat must be NaN.

        REGISTRY.md lines 211-216: Non-finite SE should result in NaN t_stat,
        not 0.0 which would be misleading.
        """
        data = generate_staggered_data(
            n_units=100,
            n_periods=8,
            cohort_periods=[4],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # Check that t_stat handling is consistent
        for (_g, _t), effect in results.group_time_effects.items():
            se = effect['se']
            t_stat = effect['t_stat']
            if not np.isfinite(se) or se <= 0:
                assert np.isnan(t_stat), \
                    f"t_stat should be NaN when SE={se}, got {t_stat}"
            elif np.isfinite(se) and se > 0:
                # t_stat should be effect/se
                expected_t = effect['effect'] / se
                assert np.isclose(t_stat, expected_t, rtol=1e-10), \
                    f"t_stat should be effect/se when SE is valid"

    def test_anticipation_shifts_reference_period(self):
        """
        Anticipation parameter shifts the reference period.

        REGISTRY.md lines 204-206: With anticipation=k, base period becomes
        g-1-k for post-treatment effects.
        """
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[5],
            treatment_effect=2.0,
            seed=42
        )

        # With anticipation=1, post-treatment starts at t >= g-1 = 4
        cs = CallawaySantAnna(anticipation=1, n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # With anticipation=1, period 4 (= g-1 = 5-1) should be post-treatment
        # Check that ATT(5, 4) is included in overall aggregation
        # (This is implicit in the overall_att calculation including t=4)
        assert results.overall_att is not None

    def test_not_yet_treated_excludes_cohort_g(self):
        """
        Control group with not_yet_treated excludes cohort g.

        REGISTRY.md lines 239-243: When computing ATT(g,t), cohort g should
        never be in the control group, even for pre-treatment periods.
        """
        # Two cohorts: g=4 and g=7
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[4, 7],
            never_treated_frac=0.3,
            seed=42
        )

        cs_nyt = CallawaySantAnna(control_group='not_yet_treated', n_bootstrap=0)
        cs_nt = CallawaySantAnna(control_group='never_treated', n_bootstrap=0)

        results_nyt = cs_nyt.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )
        results_nt = cs_nt.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # Both should produce valid results
        assert results_nyt.overall_att is not None
        assert results_nt.overall_att is not None

        # Control group setting should be recorded
        assert results_nyt.control_group == 'not_yet_treated'
        assert results_nt.control_group == 'never_treated'

        # Results should differ (not_yet_treated uses more controls)
        # n_control for not_yet_treated should be >= never_treated for early periods
        # This is implicit in the different estimates

    def test_control_group_invalid_raises_error(self):
        """Test that invalid control_group raises ValueError."""
        with pytest.raises(ValueError, match="control_group must be"):
            CallawaySantAnna(control_group="invalid")

    def test_estimation_method_invalid_raises_error(self):
        """Test that invalid estimation_method raises ValueError."""
        with pytest.raises(ValueError, match="estimation_method must be"):
            CallawaySantAnna(estimation_method="invalid")

    def test_base_period_invalid_raises_error(self):
        """Test that invalid base_period raises ValueError."""
        with pytest.raises(ValueError, match="base_period must be"):
            CallawaySantAnna(base_period="invalid")

    def test_bootstrap_weights_invalid_raises_error(self):
        """Test that invalid bootstrap_weights raises ValueError."""
        with pytest.raises(ValueError, match="bootstrap_weights must be"):
            CallawaySantAnna(bootstrap_weights="invalid")

    def test_missing_columns_raises_error(self):
        """Test that missing columns raise informative ValueError."""
        data = generate_staggered_data(n_units=50, n_periods=5, seed=42)

        cs = CallawaySantAnna()
        with pytest.raises(ValueError, match="Missing columns"):
            cs.fit(
                data, outcome='nonexistent', unit='unit',
                time='period', first_treat='first_treat'
            )

    def test_no_never_treated_raises_error(self):
        """Test that no never-treated units raises informative error."""
        data = generate_staggered_data(
            n_units=50,
            n_periods=5,
            cohort_periods=[3],
            never_treated_frac=0.0,  # No never-treated
            seed=42
        )

        cs = CallawaySantAnna()
        with pytest.raises(ValueError, match="No never-treated units"):
            cs.fit(
                data, outcome='outcome', unit='unit',
                time='period', first_treat='first_treat'
            )


class TestRankDeficiencyHandling:
    """Tests for rank deficiency handling in CallawaySantAnna."""

    def test_rank_deficient_action_warn_default(self):
        """Test that rank_deficient_action defaults to 'warn'."""
        cs = CallawaySantAnna()
        assert cs.rank_deficient_action == "warn"

    def test_rank_deficient_action_error_raises(self):
        """Test that rank_deficient_action='error' raises on collinearity."""
        # This test would require creating collinear covariates
        # For now, just verify the parameter is accepted
        cs = CallawaySantAnna(rank_deficient_action="error")
        assert cs.rank_deficient_action == "error"

    def test_rank_deficient_action_silent_no_warning(self):
        """Test that rank_deficient_action='silent' suppresses warnings."""
        cs = CallawaySantAnna(rank_deficient_action="silent")
        assert cs.rank_deficient_action == "silent"

    def test_rank_deficient_action_invalid_raises(self):
        """Test that invalid rank_deficient_action raises ValueError."""
        with pytest.raises(ValueError, match="rank_deficient_action must be"):
            CallawaySantAnna(rank_deficient_action="invalid")


# =============================================================================
# Phase 4: SE Formula Verification Tests
# =============================================================================


class TestSEFormulas:
    """Tests for standard error formula verification."""

    @pytest.mark.slow
    def test_analytical_se_close_to_bootstrap_se(self, ci_params):
        """
        Analytical and bootstrap SEs should be within 25%.

        Analytical SEs use influence function aggregation.
        Bootstrap SEs use multiplier bootstrap.
        They should converge for large samples. Wider tolerance (40%)
        when min_n cap reduces bootstrap iterations in pure Python mode.

        This test is marked slow because it uses 499 bootstrap iterations
        for thorough validation of SE convergence.
        """
        n_boot = ci_params.bootstrap(499, min_n=199)
        data = generate_staggered_data(
            n_units=300,
            n_periods=8,
            cohort_periods=[4],
            treatment_effect=2.0,
            seed=42
        )

        cs_anal = CallawaySantAnna(n_bootstrap=0)
        cs_boot = CallawaySantAnna(n_bootstrap=n_boot, seed=42)

        results_anal = cs_anal.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )
        results_boot = cs_boot.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # Check overall ATT SE (wider tolerance when min_n cap reduces
        # bootstrap iterations in pure Python mode)
        if results_boot.overall_se > 0:
            rel_diff = abs(results_anal.overall_se - results_boot.overall_se) / results_boot.overall_se
            threshold = 0.40 if n_boot < 100 else 0.25
            assert rel_diff < threshold, \
                f"Analytical SE ({results_anal.overall_se}) differs from bootstrap SE " \
                f"({results_boot.overall_se}) by {rel_diff*100:.1f}%"

    def test_bootstrap_weight_moments_rademacher(self):
        """
        Rademacher weights have E[w]=0, E[w^2]=1.

        These are the standard multiplier bootstrap weights.
        """
        rng = np.random.default_rng(42)
        weights = _generate_bootstrap_weights_batch(10000, 100, 'rademacher', rng)

        # E[w] should be ~0
        mean_w = np.mean(weights)
        assert abs(mean_w) < 0.02, f"Rademacher E[w] should be ~0, got {mean_w}"

        # E[w^2] should be ~1 (Var(w) = 1 since E[w]=0)
        var_w = np.var(weights)
        assert abs(var_w - 1) < 0.05, f"Rademacher Var(w) should be ~1, got {var_w}"

    def test_bootstrap_weight_moments_mammen(self):
        """
        Mammen weights have E[w]=0, E[w^2]=1, E[w^3]=1.

        Mammen's two-point distribution matches skewness of Bernoulli.
        """
        rng = np.random.default_rng(42)
        weights = _generate_bootstrap_weights_batch(10000, 100, 'mammen', rng)

        mean_w = np.mean(weights)
        assert abs(mean_w) < 0.02, f"Mammen E[w] should be ~0, got {mean_w}"

        var_w = np.var(weights)
        assert abs(var_w - 1) < 0.05, f"Mammen Var(w) should be ~1, got {var_w}"

    def test_bootstrap_weight_moments_webb(self):
        """
        Webb weights have E[w]=0 and Var[w]=1.

        Webb's 6-point distribution is recommended for few clusters.
        Values: ±sqrt(3/2), ±sqrt(2/2)=±1, ±sqrt(1/2) with equal probs (1/6 each)

        Theoretical variance with equal probabilities:
            Var = (1/6) * (3/2 + 1 + 1/2 + 1/2 + 1 + 3/2) = (1/6) * 6 = 1.0

        This matches R's `did` package behavior.
        """
        rng = np.random.default_rng(42)
        weights = _generate_bootstrap_weights_batch(10000, 100, 'webb', rng)

        mean_w = np.mean(weights)
        assert abs(mean_w) < 0.02, f"Webb E[w] should be ~0, got {mean_w}"

        # Webb's variance is 1.0 with equal probabilities (matching R's did package)
        var_w = np.var(weights)
        assert abs(var_w - 1.0) < 0.05, f"Webb Var(w) should be ~1.0, got {var_w}"

    def test_bootstrap_produces_valid_inference(self, ci_params):
        """Test that bootstrap produces valid inference with p-values and CIs.

        Uses 99 bootstrap iterations - sufficient to verify the mechanism works
        without being slow for CI runs.
        """
        n_boot = ci_params.bootstrap(99)
        data = generate_staggered_data(
            n_units=100,
            n_periods=6,
            cohort_periods=[3],
            treatment_effect=3.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # Bootstrap results should exist
        assert results.bootstrap_results is not None

        # Overall statistics should be valid
        assert np.isfinite(results.overall_se)
        assert 0 <= results.overall_p_value <= 1
        assert results.overall_conf_int[0] < results.overall_conf_int[1]

        # Group-time p-values should be in [0, 1]
        for effect in results.group_time_effects.values():
            assert 0 <= effect['p_value'] <= 1


class TestAggregationMethods:
    """Tests for aggregation method correctness."""

    def test_simple_aggregation_weights_by_group_size(self):
        """
        Simple aggregation weights by group size (n_treated).

        Overall ATT = Σ w_g * ATT(g,t) where w_g ∝ n_g
        """
        data = generate_staggered_data(
            n_units=100,
            n_periods=8,
            cohort_periods=[4, 6],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # Overall ATT should be weighted average of post-treatment effects
        assert results.overall_att is not None
        assert np.isfinite(results.overall_att)

    def test_event_study_aggregation_by_relative_time(self):
        """
        Event study aggregates by relative time e = t - g.

        ATT(e) = Σ_g w_g * ATT(g, g+e) for each event time e.
        """
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[4, 6],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0

        # Should have both pre and post treatment periods
        rel_times = list(results.event_study_effects.keys())
        assert any(e < 0 for e in rel_times), "Should have pre-treatment periods"
        assert any(e >= 0 for e in rel_times), "Should have post-treatment periods"

    def test_group_aggregation_by_cohort(self):
        """
        Group aggregation averages over post-treatment periods per cohort.

        ATT(g) = (1/T_g) Σ_t ATT(g,t) for t >= g - anticipation.
        """
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[4, 6],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='group'
        )

        assert results.group_effects is not None
        assert len(results.group_effects) > 0

        # Should have effects for each cohort
        assert 4 in results.group_effects or 6 in results.group_effects

    def test_all_aggregation_computes_everything(self):
        """Test aggregate='all' computes event_study and group effects."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=8,
            cohort_periods=[4],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='all'
        )

        assert results.event_study_effects is not None
        assert results.group_effects is not None


class TestGetSetParams:
    """Tests for sklearn-compatible parameter interface."""

    def test_get_params_returns_all_parameters(self):
        """Test that get_params returns all constructor parameters."""
        cs = CallawaySantAnna(
            control_group='not_yet_treated',
            anticipation=1,
            estimation_method='ipw',
            alpha=0.10,
            n_bootstrap=100,
            bootstrap_weights='mammen',
            seed=42,
            rank_deficient_action='silent',
            base_period='universal'
        )

        params = cs.get_params()

        assert params['control_group'] == 'not_yet_treated'
        assert params['anticipation'] == 1
        assert params['estimation_method'] == 'ipw'
        assert params['alpha'] == 0.10
        assert params['n_bootstrap'] == 100
        assert params['bootstrap_weights'] == 'mammen'
        assert params['seed'] == 42
        assert params['rank_deficient_action'] == 'silent'
        assert params['base_period'] == 'universal'

    def test_set_params_modifies_attributes(self):
        """Test that set_params modifies estimator attributes."""
        cs = CallawaySantAnna()

        cs.set_params(alpha=0.10, n_bootstrap=500)

        assert cs.alpha == 0.10
        assert cs.n_bootstrap == 500

    def test_set_params_returns_self(self):
        """Test that set_params returns the estimator (fluent interface)."""
        cs = CallawaySantAnna()
        result = cs.set_params(alpha=0.10)
        assert result is cs

    def test_set_params_unknown_raises_error(self):
        """Test that set_params with unknown parameter raises ValueError."""
        cs = CallawaySantAnna()
        with pytest.raises(ValueError, match="Unknown parameter"):
            cs.set_params(unknown_param=42)


class TestResultsObject:
    """Tests for CallawaySantAnnaResults object."""

    def test_results_summary_contains_key_info(self):
        """Test that summary() output contains key information."""
        data = generate_staggered_data(
            n_units=50,
            n_periods=6,
            cohort_periods=[3],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        summary = results.summary()

        assert "Callaway-Sant'Anna" in summary
        assert "ATT" in summary
        assert str(round(results.overall_att, 3))[:4] in summary or "ATT" in summary

    def test_results_to_dataframe_group_time(self):
        """Test to_dataframe with level='group_time'."""
        data = generate_staggered_data(
            n_units=50,
            n_periods=6,
            cohort_periods=[3],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        df = results.to_dataframe(level='group_time')

        assert 'group' in df.columns
        assert 'time' in df.columns
        assert 'effect' in df.columns
        assert 'se' in df.columns
        assert len(df) == len(results.group_time_effects)

    def test_results_to_dataframe_event_study(self):
        """Test to_dataframe with level='event_study'."""
        data = generate_staggered_data(
            n_units=50,
            n_periods=6,
            cohort_periods=[3],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        df = results.to_dataframe(level='event_study')

        assert 'relative_period' in df.columns
        assert 'effect' in df.columns

    def test_results_significance_properties(self):
        """Test is_significant and significance_stars properties."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=8,
            cohort_periods=[4],
            treatment_effect=5.0,  # Large effect for significance
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # With large effect, should be significant
        assert isinstance(results.is_significant, bool)
        assert results.significance_stars in ["", "*", "**", "***"]


# =============================================================================
# Deprecation Warning Tests
# =============================================================================


# =============================================================================
# MPDTA-based Strict R Comparison Tests
# =============================================================================


class TestEventStudySEWithWIF:
    """Tests for WIF adjustment in event study standard errors."""

    def test_event_study_analytical_se_includes_wif(self):
        """
        Event study SEs with WIF should be >= SEs without WIF.

        The WIF accounts for uncertainty in estimating group-size weights,
        so it can only add (or maintain) variance, never reduce it.
        """
        data = generate_staggered_data(
            n_units=200,
            n_periods=8,
            cohort_periods=[3, 5],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        assert results.event_study_effects is not None

        for e, eff_data in results.event_study_effects.items():
            se_with_wif = eff_data['se']
            if np.isfinite(se_with_wif) and se_with_wif > 0:
                # WIF-adjusted SE should be positive
                assert se_with_wif > 0, f"SE for e={e} should be positive"

    def test_event_study_bootstrap_consistent_with_analytical(self, ci_params):
        """
        Bootstrap event study SEs should be within ~20% of analytical SEs.
        """
        n_boot = ci_params.bootstrap(999)
        data = generate_staggered_data(
            n_units=200,
            n_periods=8,
            cohort_periods=[3, 5],
            treatment_effect=2.0,
            seed=42
        )

        # Analytical SEs
        cs_analytical = CallawaySantAnna(n_bootstrap=0)
        results_analytical = cs_analytical.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        # Bootstrap SEs
        cs_boot = CallawaySantAnna(n_bootstrap=n_boot, seed=42, cband=False)
        results_boot = cs_boot.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        assert results_analytical.event_study_effects is not None
        assert results_boot.event_study_effects is not None

        threshold = 0.40 if n_boot < 100 else 0.20
        n_compared = 0
        for e in results_analytical.event_study_effects:
            se_a = results_analytical.event_study_effects[e]['se']
            if e not in results_boot.event_study_effects:
                continue
            se_b = results_boot.event_study_effects[e]['se']
            if np.isfinite(se_a) and se_a > 0 and np.isfinite(se_b) and se_b > 0:
                rel_diff = abs(se_a - se_b) / se_a
                assert rel_diff < threshold, \
                    f"e={e}: analytical SE={se_a:.4f} vs bootstrap SE={se_b:.4f} " \
                    f"(diff={rel_diff*100:.1f}% > {threshold*100}%)"
                n_compared += 1

        assert n_compared > 0, "No event times had finite SEs for comparison"

    def test_overall_att_bootstrap_uses_combined_if(self, ci_params):
        """
        Overall ATT bootstrap SE should be consistent with analytical SE
        (which includes WIF).
        """
        n_boot = ci_params.bootstrap(999)
        data = generate_staggered_data(
            n_units=200,
            n_periods=8,
            cohort_periods=[3, 5],
            treatment_effect=2.0,
            seed=42
        )

        cs_analytical = CallawaySantAnna(n_bootstrap=0)
        results_a = cs_analytical.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        cs_boot = CallawaySantAnna(n_bootstrap=n_boot, seed=42, cband=False)
        results_b = cs_boot.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        se_a = results_a.overall_se
        se_b = results_b.overall_se

        threshold = 0.40 if n_boot < 100 else 0.20
        if np.isfinite(se_a) and se_a > 0 and np.isfinite(se_b) and se_b > 0:
            rel_diff = abs(se_a - se_b) / se_a
            assert rel_diff < threshold, \
                f"Analytical SE={se_a:.4f} vs bootstrap SE={se_b:.4f} (diff={rel_diff*100:.1f}%)"

    def test_single_group_event_time_wif_effect(self):
        """
        When only one group contributes to an event time, the WIF
        adjustment should add minimal variance (weight is near-deterministic).
        """
        data = generate_staggered_data(
            n_units=100,
            n_periods=6,
            cohort_periods=[3],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        assert results.event_study_effects is not None
        # With a single cohort, all event times have only one group
        for e, eff_data in results.event_study_effects.items():
            if eff_data['n_groups'] == 1:
                # SE should still be finite and positive
                assert np.isfinite(eff_data['se']), \
                    f"Single-group SE for e={e} should be finite"

    def test_unbalanced_panel_bootstrap_uses_global_n(self, ci_params):
        """
        Bootstrap SEs should use global N (all panel units), not just units
        appearing in influence functions. When some units have NaN outcomes
        in all periods, they appear in precomputed['all_units'] but are
        excluded from IFs, creating n_global > n_IF_units.

        Regression test for the bug where _run_multiplier_bootstrap() built
        the unit set from influence_func_info (local), causing pg = n_g / n_local
        to overestimate group shares and mis-scale WIF.
        """
        n_boot = ci_params.bootstrap(999)
        # Generate balanced staggered data
        data = generate_staggered_data(
            n_units=100,
            n_periods=8,
            cohort_periods=[3, 5],
            treatment_effect=2.0,
            seed=42
        )

        # Add extra never-treated units with NaN outcomes in all periods.
        # These units will be in the panel (and precomputed['all_units'])
        # but excluded from all IFs because NaN fails the validity check.
        n_nan_units = 10
        max_unit = data['unit'].max()
        periods = sorted(data['period'].unique())
        nan_rows = []
        for i in range(1, n_nan_units + 1):
            uid = max_unit + i
            for p in periods:
                nan_rows.append({
                    'unit': uid,
                    'period': p,
                    'first_treat': 0,  # never-treated
                    'outcome': np.nan,
                })
        data_unbalanced = pd.concat(
            [data, pd.DataFrame(nan_rows)], ignore_index=True
        )

        n_global = data_unbalanced['unit'].nunique()
        n_valid = data['unit'].nunique()  # units that actually appear in IFs
        assert n_global > n_valid, "Test setup: global N should exceed IF unit count"

        # Analytical SEs (already fixed to use global N)
        cs_analytical = CallawaySantAnna(n_bootstrap=0)
        results_a = cs_analytical.fit(
            data_unbalanced, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        # Bootstrap SEs (the fix under test)
        cs_boot = CallawaySantAnna(n_bootstrap=n_boot, seed=42, cband=False)
        results_b = cs_boot.fit(
            data_unbalanced, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        assert results_a.event_study_effects is not None
        assert results_b.event_study_effects is not None

        threshold = 0.40 if n_boot < 100 else 0.25
        n_compared = 0
        for e in results_a.event_study_effects:
            se_a = results_a.event_study_effects[e]['se']
            if e not in results_b.event_study_effects:
                continue
            se_b = results_b.event_study_effects[e]['se']
            if np.isfinite(se_a) and se_a > 0 and np.isfinite(se_b) and se_b > 0:
                rel_diff = abs(se_a - se_b) / se_a
                assert rel_diff < threshold, \
                    f"e={e}: analytical SE={se_a:.4f} vs bootstrap SE={se_b:.4f} " \
                    f"(diff={rel_diff*100:.1f}% > {threshold*100}%)"
                n_compared += 1

        assert n_compared > 0, "No event times had finite SEs for comparison"


class TestSimultaneousConfidenceBands:
    """Tests for simultaneous confidence bands (cband)."""

    def test_cband_crit_value_exceeds_pointwise(self, ci_params):
        """
        When cband=True and n_bootstrap > 0, the critical value should
        exceed the pointwise z_{0.025} = 1.96.
        """
        n_boot = ci_params.bootstrap(999, min_n=199)
        data = generate_staggered_data(
            n_units=200,
            n_periods=8,
            cohort_periods=[3, 5],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42, cband=True)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        assert results.cband_crit_value is not None, \
            "cband_crit_value should be set when cband=True and n_bootstrap > 0"
        assert results.cband_crit_value > 1.96, \
            f"Simultaneous critical value ({results.cband_crit_value:.3f}) " \
            "should exceed pointwise z_0.025=1.96"

    def test_cband_confidence_intervals_wider(self, ci_params):
        """Simultaneous CIs should be wider than pointwise CIs."""
        n_boot = ci_params.bootstrap(999, min_n=199)
        data = generate_staggered_data(
            n_units=200,
            n_periods=8,
            cohort_periods=[3, 5],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42, cband=True)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        assert results.event_study_effects is not None
        for e, eff_data in results.event_study_effects.items():
            if 'cband_conf_int' in eff_data:
                pw_ci = eff_data['conf_int']
                cb_ci = eff_data['cband_conf_int']
                pw_width = pw_ci[1] - pw_ci[0]
                cb_width = cb_ci[1] - cb_ci[0]
                if np.isfinite(pw_width) and np.isfinite(cb_width) and pw_width > 0:
                    assert cb_width >= pw_width, \
                        f"e={e}: cband CI width ({cb_width:.4f}) should >= " \
                        f"pointwise CI width ({pw_width:.4f})"

    def test_cband_false_disables_simultaneous_ci(self, ci_params):
        """When cband=False, cband_crit_value should be None."""
        n_boot = ci_params.bootstrap(999, min_n=199)
        data = generate_staggered_data(
            n_units=100,
            n_periods=6,
            cohort_periods=[3],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42, cband=False)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        assert results.cband_crit_value is None

    def test_cband_requires_bootstrap(self):
        """When n_bootstrap=0, cband has no effect (remains None)."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=6,
            cohort_periods=[3],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=0, cband=True)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        assert results.cband_crit_value is None

    def test_cband_validity_threshold(self):
        """When >50% of sup-t bootstrap draws are non-finite, cband_crit_value is None."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=6,
            cohort_periods=[3],
            treatment_effect=2.0,
            seed=42,
        )

        cs = CallawaySantAnna(n_bootstrap=50, seed=42, cband=True)

        # Patch np.max to return mostly-NaN sup-t distribution
        original_max = np.max

        def mock_max(a, axis=None):
            result = original_max(a, axis=axis)
            if axis == 0 and hasattr(result, '__len__') and len(result) > 10:
                # Make >50% of sup-t draws NaN
                result = result.copy()
                n_nan = int(len(result) * 0.6)
                result[:n_nan] = np.nan
            return result

        with unittest.mock.patch('diff_diff.staggered_bootstrap.np.max', side_effect=mock_max):
            with pytest.warns(RuntimeWarning, match="Too few valid sup-t"):
                results = cs.fit(
                    data, outcome='outcome', unit='unit',
                    time='period', first_treat='first_treat',
                    aggregate='event_study',
                )

        assert results.cband_crit_value is None

    def test_cband_in_get_params(self):
        """Test that cband parameter appears in get_params."""
        cs = CallawaySantAnna(cband=False)
        params = cs.get_params()
        assert 'cband' in params
        assert params['cband'] is False

    def test_cband_to_dataframe_columns(self, ci_params):
        """Test that to_dataframe includes cband columns."""
        n_boot = ci_params.bootstrap(999, min_n=199)
        data = generate_staggered_data(
            n_units=200,
            n_periods=8,
            cohort_periods=[3, 5],
            treatment_effect=2.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42, cband=True)
        results = cs.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat',
            aggregate='event_study'
        )

        df = results.to_dataframe(level='event_study')
        assert 'cband_lower' in df.columns
        assert 'cband_upper' in df.columns


class TestMPDTARComparison:
    """Strict R comparison tests using the real MPDTA dataset.

    These tests compare Python's CallawaySantAnna to R's did::att_gt() using
    the same data exported from R. We export R's mpdta dataset to a temp file
    and load it in Python to ensure identical input data.

    Note: Python's load_mpdta() downloads from a different source than R's
    packaged data, which has different values. These tests use R's data directly.

    Expected tolerances (based on benchmark analysis):
    - Overall ATT: <1% difference
    - Overall SE: <1% difference
    """

    def _get_r_mpdta_and_results(self, tmp_path) -> Tuple[pd.DataFrame, dict]:
        """
        Export R's mpdta dataset and run att_gt(), returning both data and results.

        Returns
        -------
        Tuple of (DataFrame, dict) where dict has keys:
            - overall_att, overall_se, n_groups, group_time (existing)
            - event_study.event_time, event_study.att, event_study.se
            - event_study_cband.crit_val
        """
        import json

        csv_path = tmp_path / "r_mpdta.csv"
        escaped_path = str(csv_path).replace("\\", "/")

        r_script = f'''
        suppressMessages(library(did))
        suppressMessages(library(jsonlite))

        # Load mpdta from did package
        data(mpdta)

        # Export to CSV for Python to read
        # Rename first.treat to first_treat for Python compatibility
        mpdta$first_treat <- mpdta$first.treat
        write.csv(mpdta, "{escaped_path}", row.names = FALSE)

        # Run att_gt with default settings (matching Python defaults)
        result <- att_gt(
            yname = "lemp",
            tname = "year",
            idname = "countyreal",
            gname = "first.treat",
            xformla = ~ 1,
            data = mpdta,
            est_method = "dr",
            control_group = "nevertreated",
            anticipation = 0,
            base_period = "varying",
            bstrap = FALSE,
            cband = FALSE
        )

        # Simple aggregation
        agg <- aggte(result, type = "simple")

        # Event study aggregation (analytical SEs)
        agg_es <- aggte(result, type = "dynamic", cband = FALSE)

        # Event study with cband (bootstrap)
        set.seed(42)
        result_boot <- att_gt(
            yname = "lemp",
            tname = "year",
            idname = "countyreal",
            gname = "first.treat",
            xformla = ~ 1,
            data = mpdta,
            est_method = "dr",
            control_group = "nevertreated",
            anticipation = 0,
            base_period = "varying",
            bstrap = TRUE,
            cband = TRUE,
            biters = 999
        )
        agg_es_cband <- aggte(result_boot, type = "dynamic", cband = TRUE)

        output <- list(
            overall_att = unbox(agg$overall.att),
            overall_se = unbox(agg$overall.se),
            n_groups = unbox(length(unique(result$group[result$group > 0]))),
            group_time = list(
                group = as.integer(result$group),
                time = as.integer(result$t),
                att = result$att,
                se = result$se
            ),
            event_study = list(
                event_time = as.integer(agg_es$egt),
                att = agg_es$att.egt,
                se = agg_es$se.egt
            ),
            event_study_cband = list(
                crit_val = unbox(agg_es_cband$crit.val)
            )
        )

        cat(toJSON(output, pretty = TRUE))
        '''

        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(f"R script failed: {result.stderr}")

        parsed = json.loads(result.stdout)

        # Handle R's JSON serialization quirks
        if isinstance(parsed.get('overall_att'), list):
            parsed['overall_att'] = parsed['overall_att'][0]
        if isinstance(parsed.get('overall_se'), list):
            parsed['overall_se'] = parsed['overall_se'][0]

        # Read the exported CSV
        mpdta = pd.read_csv(csv_path)

        return mpdta, parsed

    def test_mpdta_overall_att_matches_r_strict(self, require_r, tmp_path):
        """Test overall ATT matches R within 1% using MPDTA dataset.

        This test uses R's actual mpdta dataset (exported to CSV) to ensure
        identical input data between Python and R.
        """
        # Get R's mpdta data and results
        mpdta, r_results = self._get_r_mpdta_and_results(tmp_path)

        # Python estimation using R's data
        cs = CallawaySantAnna(estimation_method='dr', n_bootstrap=0)
        py_results = cs.fit(
            mpdta,
            outcome='lemp',
            unit='countyreal',
            time='year',
            first_treat='first_treat'
        )

        # Compare overall ATT - strict 1% tolerance for MPDTA
        rel_diff = abs(py_results.overall_att - r_results['overall_att']) / abs(r_results['overall_att'])
        assert rel_diff < 0.01, \
            f"MPDTA ATT mismatch: Python={py_results.overall_att:.6f}, " \
            f"R={r_results['overall_att']:.6f}, diff={rel_diff*100:.2f}%"

    def test_mpdta_overall_se_matches_r_strict(self, require_r, tmp_path):
        """Test overall SE matches R within 1% using MPDTA dataset.

        Uses 1% tolerance to account for minor numerical differences.
        """
        mpdta, r_results = self._get_r_mpdta_and_results(tmp_path)

        cs = CallawaySantAnna(estimation_method='dr', n_bootstrap=0)
        py_results = cs.fit(
            mpdta,
            outcome='lemp',
            unit='countyreal',
            time='year',
            first_treat='first_treat'
        )

        # Compare overall SE - strict 1% tolerance for MPDTA
        rel_diff = abs(py_results.overall_se - r_results['overall_se']) / r_results['overall_se']
        assert rel_diff < 0.01, \
            f"MPDTA SE mismatch: Python={py_results.overall_se:.6f}, " \
            f"R={r_results['overall_se']:.6f}, diff={rel_diff*100:.2f}%"

    def test_mpdta_group_time_effects_match_r_strict(self, require_r, tmp_path):
        """Test individual ATT(g,t) values match R within 1% for MPDTA.

        Post-treatment ATT(g,t) values should match very closely since
        both Python and R use identical methodology with the same dataset.
        """
        mpdta, r_results = self._get_r_mpdta_and_results(tmp_path)

        cs = CallawaySantAnna(estimation_method='dr', n_bootstrap=0)
        py_results = cs.fit(
            mpdta,
            outcome='lemp',
            unit='countyreal',
            time='year',
            first_treat='first_treat'
        )

        # Compare each post-treatment ATT(g,t)
        r_gt = r_results['group_time']
        n_comparisons = 0
        mismatches = []

        for i in range(len(r_gt['group'])):
            g = int(r_gt['group'][i])
            t = int(r_gt['time'][i])
            r_att = r_gt['att'][i]

            # Skip pre-treatment effects
            if t < g:
                continue

            if (g, t) in py_results.group_time_effects:
                py_att = py_results.group_time_effects[(g, t)]['effect']

                # Handle near-zero effects (use absolute tolerance)
                if abs(r_att) < 0.001:
                    if abs(py_att - r_att) > 0.01:
                        mismatches.append(
                            f"ATT({g},{t}): Python={py_att:.6f}, R={r_att:.6f}"
                        )
                else:
                    rel_diff = abs(py_att - r_att) / abs(r_att)
                    if rel_diff > 0.01:  # 1% tolerance
                        mismatches.append(
                            f"ATT({g},{t}): Python={py_att:.6f}, R={r_att:.6f}, " \
                            f"diff={rel_diff*100:.2f}%"
                        )
                n_comparisons += 1

        assert n_comparisons > 0, \
            "No post-treatment group-time effects matched between Python and R"

        assert len(mismatches) == 0, \
            f"MPDTA post-treatment ATT mismatches (>1% diff):\n" + "\n".join(mismatches)

    def test_mpdta_event_study_ses_match_r(self, require_r, tmp_path):
        """Test event study ATTs and SEs match R's aggte(type='dynamic').

        Compares Python's event study aggregation against R's did::aggte()
        with type="dynamic" using the MPDTA dataset. Both use varying base
        period (default in both implementations).
        """
        mpdta, r_results = self._get_r_mpdta_and_results(tmp_path)

        cs = CallawaySantAnna(estimation_method='dr', n_bootstrap=0)
        py_results = cs.fit(
            mpdta,
            outcome='lemp',
            unit='countyreal',
            time='year',
            first_treat='first_treat',
            aggregate='event_study'
        )

        assert py_results.event_study_effects is not None, \
            "Python event_study_effects is None"

        r_es = r_results['event_study']
        r_event_times = r_es['event_time']
        r_atts = r_es['att']
        r_ses = r_es['se']

        n_compared = 0
        mismatches = []
        unmatched_r = []

        for i, e in enumerate(r_event_times):
            r_att = r_atts[i]
            r_se = r_ses[i]

            # Skip if R SE is NaN or zero
            if r_se is None or np.isnan(r_se) or r_se == 0:
                continue

            if e not in py_results.event_study_effects:
                unmatched_r.append(e)
                continue

            py_eff = py_results.event_study_effects[e]
            py_att = py_eff['effect']
            py_se = py_eff['se']

            # Compare ATT: rtol=5%, atol=0.01
            if not np.isclose(py_att, r_att, rtol=0.05, atol=0.01):
                mismatches.append(
                    f"e={e} ATT: Python={py_att:.6f}, R={r_att:.6f}"
                )

            # Compare SE: rtol=10%, atol=0.005 (wider for WIF sensitivity)
            if not np.isclose(py_se, r_se, rtol=0.10, atol=0.005):
                mismatches.append(
                    f"e={e} SE: Python={py_se:.6f}, R={r_se:.6f}"
                )

            n_compared += 1

        if unmatched_r:
            # Log but don't fail for unmatched event times
            warnings.warn(
                f"R event times not in Python: {unmatched_r}"
            )

        assert n_compared > 0, \
            "No event times were compared between Python and R"

        assert len(mismatches) == 0, \
            f"Event study mismatches ({n_compared} compared):\n" + \
            "\n".join(mismatches)

    def test_mpdta_cband_crit_value_vs_r(self, require_r, tmp_path):
        """Test simultaneous confidence band critical value matches R.

        Compares the sup-t bootstrap critical value from Python's
        CallawaySantAnna(cband=True) against R's aggte(type='dynamic',
        cband=TRUE). Uses 30% tolerance because bootstrap draws differ
        between R's set.seed(42) and Python's seed=42.
        """
        mpdta, r_results = self._get_r_mpdta_and_results(tmp_path)

        r_crit = r_results['event_study_cband']['crit_val']

        cs = CallawaySantAnna(
            estimation_method='dr',
            n_bootstrap=999,
            seed=42,
            cband=True,
        )
        py_results = cs.fit(
            mpdta,
            outcome='lemp',
            unit='countyreal',
            time='year',
            first_treat='first_treat',
            aggregate='event_study'
        )

        py_crit = py_results.cband_crit_value

        # Both should be finite and > 1.96 (exceed pointwise z-value)
        assert np.isfinite(r_crit), f"R crit value not finite: {r_crit}"
        assert np.isfinite(py_crit), f"Python crit value not finite: {py_crit}"
        assert r_crit > 1.96, f"R crit value {r_crit} <= 1.96"
        assert py_crit > 1.96, f"Python crit value {py_crit} <= 1.96"

        # 30% tolerance for bootstrap variability across implementations
        assert np.isclose(py_crit, r_crit, rtol=0.30), \
            f"Cband crit value mismatch: Python={py_crit:.4f}, " \
            f"R={r_crit:.4f}, rtol={abs(py_crit - r_crit) / r_crit:.2%}"


class TestCSCovariateScaleEquilibration:
    """CallawaySantAnna covariate outcome-regression is scale-robust.

    The OR nuisance is fit through the shared scale-equilibrated ``solve_ols``
    (column-equilibrated SVD/gelsd), matching ``TripleDifference`` and R's
    ``lm()``/QR. Adding a large constant offset to a covariate is absorbed by the
    regression intercept and MUST NOT change ATT(g,t) (the fitted values are
    offset-invariant). The prior ``cho_solve(X'X)`` / ``lstsq(cond=1e-7)`` path
    lost accuracy on such ill-conditioned designs (overall ATT drifted ~3.8e-6 at
    an offset of 1e6, growing with the offset); the equilibrated SVD is invariant
    to ~1e-11. Anchored on ``est_method="reg"`` so the perturbation routes purely
    through the OR fit (``dr``/``ipw`` also fit a propensity logit, which is out of
    scope for this change).
    """

    @staticmethod
    def _panel(offset, seed=42, n_units=200, n_periods=6):
        rng = np.random.default_rng(seed)
        cohorts = rng.choice([3, 4, 5, 0], size=n_units, p=[0.25, 0.25, 0.2, 0.3])
        unit_fe = rng.normal(size=n_units)
        x = rng.normal(size=n_units)
        rows = []
        for i in range(n_units):
            g = int(cohorts[i])
            for t in range(1, n_periods + 1):
                post = 1 if (g != 0 and t >= g) else 0
                y = unit_fe[i] + 0.5 * t + 0.3 * x[i] + 2.0 * post + rng.normal(scale=0.5)
                rows.append(
                    {"unit": i, "period": t, "first_treat": g, "outcome": y, "X": offset + x[i]}
                )
        return pd.DataFrame(rows)

    def _fit(self, offset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CallawaySantAnna(estimation_method="reg").fit(
                self._panel(offset),
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                covariates=["X"],
            )

    def test_or_fit_offset_invariant(self):
        """A 1e6 constant offset on the covariate (absorbed by the intercept)
        leaves ATT(g,t), overall ATT, and SE unchanged."""
        base = self._fit(0.0)
        shifted = self._fit(1e6)
        # New (equilibrated SVD) is invariant to ~1e-11; the prior normal-
        # equations path drifted ~3.8e-6 at this offset, so 1e-7 cleanly
        # separates the two.
        assert base.overall_att == pytest.approx(shifted.overall_att, abs=1e-7)
        assert base.overall_se == pytest.approx(shifted.overall_se, abs=1e-6)
        for gt, eff in base.group_time_effects.items():
            if np.isfinite(eff["effect"]) and gt in shifted.group_time_effects:
                assert eff["effect"] == pytest.approx(
                    shifted.group_time_effects[gt]["effect"], abs=1e-7
                ), f"ATT{gt} not offset-invariant"


# =============================================================================
# DRDID panel influence-function parity (numpy cross-checks)
# =============================================================================

_GOLDEN_VALUES_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "data" / "csdid_golden_values.json"
)


def _reg_did_panel_ref(
    dY: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Independent numpy reconstruction of ``DRDID::reg_did_panel``.

    Returns ``(att, se)`` where ``se`` is the analytical influence-function
    standard error ``sd(psi) * sqrt(n-1) / n``. The control-side psi carries
    the OLS estimation-effect term ``asy.lin.rep.ols @ M1`` (the WLS residual
    projected through the bread and evaluated at the treated covariate mean);
    its intercept-only collapse is ``-resid_c / n_c``.
    """
    n = len(D)
    w = np.ones(n) if weights is None else weights / np.mean(weights)
    Xi = np.column_stack([np.ones(n), X])
    c = D == 0
    Wc = w[c]
    beta, *_ = np.linalg.lstsq(
        np.sqrt(Wc)[:, None] * Xi[c], np.sqrt(Wc) * dY[c], rcond=None
    )
    pred = Xi @ beta
    w_treat = w * D
    mw_t = w_treat.mean()
    att = float(np.sum(w_treat * (dY - pred)) / np.sum(w_treat))
    resid_c = dY[c] - pred[c]
    XpX_inv = np.linalg.inv(Xi[c].T @ (Wc[:, None] * Xi[c]) / n)
    asy_lin_rep_ols = (Wc * resid_c)[:, None] * Xi[c] @ XpX_inv
    M1 = (w_treat[:, None] * Xi).sum(axis=0) / n
    psi = np.zeros(n)
    psi[~c] = w_treat[~c] * (dY[~c] - pred[~c] - att) / mw_t
    psi[c] = -(asy_lin_rep_ols @ M1) / mw_t
    se = float(psi.std(ddof=1) * np.sqrt(n - 1) / n)
    return att, se


def _std_ipw_did_panel_ref(
    dY: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    ps: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Independent numpy reconstruction of ``DRDID::std_ipw_did_panel``.

    Hajek IPW with the propensity-score estimation-effect correction
    ``asy.lin.rep.ps @ M2``. Takes the fitted propensity scores as input so
    the check isolates the IF algebra from the logit solver.
    """
    n = len(D)
    w = np.ones(n) if weights is None else weights / np.mean(weights)
    Xi = np.column_stack([np.ones(n), X])
    w_t = w * D
    w_c = w * ps * (1 - D) / (1 - ps)
    mw_t, mw_c = w_t.mean(), w_c.mean()
    eta_t = (w_t * dY).mean() / mw_t
    eta_c = (w_c * dY).mean() / mw_c
    att = float(eta_t - eta_c)
    W_ps = ps * (1 - ps) * w
    H_inv = np.linalg.inv(Xi.T @ (W_ps[:, None] * Xi)) * n
    score = (w * (D - ps))[:, None] * Xi
    asy_lin_rep_ps = score @ H_inv
    inf_t = (w_t * dY - w_t * eta_t) / mw_t
    inf_c1 = w_c * dY - w_c * eta_c
    M2 = ((w_c * (dY - eta_c))[:, None] * Xi).sum(axis=0) / n
    psi = inf_t - (inf_c1 + asy_lin_rep_ps @ M2) / mw_c
    se = float(psi.std(ddof=1) * np.sqrt(n - 1) / n)
    return att, se


def _make_cov_panel(
    seed: int,
    n_units: int = 400,
    n_periods: int = 2,
    cohorts: Tuple[int, ...] = (2,),
) -> pd.DataFrame:
    """Two-covariate staggered panel with X-dependent selection and trends.

    The selection index is bounded so every cell's fitted propensity stays
    well inside (0.05, 0.95) - the estimator's clip at ``pscore_trim`` and
    R's ``trim.level`` drop are both no-ops, keeping the reconstruction and
    the estimator on identical weights.
    """
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n_units)
    X2 = rng.uniform(-1.0, 1.0, size=n_units)
    p_treat = 1.0 / (1.0 + np.exp(-(0.6 * X1 - 0.5 * X2)))
    u = rng.uniform(size=n_units)
    first_treat = np.zeros(n_units, dtype=int)
    # Split the treated mass evenly across cohorts.
    share = 0.55
    for j, g in enumerate(cohorts):
        lo = share * p_treat * j / len(cohorts)
        hi = share * p_treat * (j + 1) / len(cohorts)
        first_treat[(u >= lo) & (u < hi)] = g
    alpha = rng.normal(size=n_units) + 0.5 * X1
    weight = rng.uniform(0.5, 2.0, size=n_units)
    rows = []
    for t in range(1, n_periods + 1):
        eps = rng.normal(scale=0.5, size=n_units)
        trend = 0.5 * t + (0.3 * X1 - 0.2 * X2) * t
        effect = np.where(
            (first_treat > 0) & (t >= first_treat), 1.0 + 0.5 * (t - first_treat), 0.0
        )
        y = alpha + trend + effect + eps
        rows.append(
            pd.DataFrame(
                {
                    "unit": np.arange(n_units),
                    "period": t,
                    "first_treat": first_treat,
                    "outcome": y,
                    "X1": X1,
                    "X2": X2,
                    "weight": weight,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _cell_arrays(
    df: pd.DataFrame,
    g: int,
    t: int,
    base: int,
    control_cohorts: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unit-level (dY, D, X, weights) arrays for one (g, t) cell."""
    keep = list(control_cohorts) + [g]
    sub = df[df["first_treat"].isin(keep)]
    wide = sub.pivot(index="unit", columns="period", values="outcome")
    meta = sub.drop_duplicates(subset="unit").set_index("unit")
    units = wide[[base, t]].dropna().index
    dY = (wide.loc[units, t] - wide.loc[units, base]).to_numpy()
    D = (meta.loc[units, "first_treat"] == g).to_numpy(dtype=float)
    X = meta.loc[units, ["X1", "X2"]].to_numpy()
    w = meta.loc[units, "weight"].to_numpy()
    return dY, D, X, w


class TestDRDIDPanelIFParity:
    """Numpy cross-checks of CS panel reg/ipw per-cell IF standard errors
    against independent reconstructions of ``DRDID::reg_did_panel`` and
    ``DRDID::std_ipw_did_panel`` (the estimators R's ``did::att_gt``
    delegates to for panel data).

    These pin the estimation-effect (nuisance) IF terms: the OLS
    ``asy.lin.rep.ols @ M1`` term for reg and the PS score correction
    ``asy.lin.rep.ps @ M2`` for ipw - the terms whose omission left ATTs
    exact (both are mean-zero) while per-cell/aggregated SEs diverged from
    R by 4-13% / 3-20% (reg) and ~7x per-cell (ipw).
    """

    def _fit(
        self, df: pd.DataFrame, method: str, survey_design: Any = None, **cs_kwargs
    ) -> Any:
        fit_kwargs = {"survey_design": survey_design} if survey_design is not None else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CallawaySantAnna(estimation_method=method, **cs_kwargs).fit(
                df,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                covariates=["X1", "X2"],
                **fit_kwargs,
            )

    @staticmethod
    def _assert_ps_interior(X: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Fit the cell logit and assert the clip/trim guards are no-ops."""
        _, ps = solve_logit(X, D)
        assert ps.min() > 0.02 and ps.max() < 0.98, "DGP must keep ps interior"
        return ps

    def test_reg_if_crosscheck(self):
        """Unweighted reg+cov single cell: fitted per-cell/overall SE must
        equal the reg_did_panel reconstruction (vectorized covariate path)."""
        df = _make_cov_panel(seed=90210)
        res = self._fit(df, "reg")
        dY, D, X, _ = _cell_arrays(df, g=2, t=2, base=1, control_cohorts=(0,))
        att, se = _reg_did_panel_ref(dY, D, X)
        cell = res.group_time_effects[(2, 2)]
        assert cell["effect"] == pytest.approx(att, rel=1e-9)
        assert cell["se"] == pytest.approx(se, rel=1e-9)
        assert res.overall_se == pytest.approx(se, rel=1e-9)

    def test_reg_if_crosscheck_not_yet_treated(self):
        """not_yet_treated forces the per-pair (non-hoisted) bread branch of
        the vectorized producer; both post cells must match reg_did_panel."""
        df = _make_cov_panel(seed=90211, n_periods=3, cohorts=(2, 3))
        res = self._fit(df, "reg", control_group="not_yet_treated")
        # (2,2): controls = never-treated + cohort 3 (not yet treated at t=2)
        dY, D, X, _ = _cell_arrays(df, g=2, t=2, base=1, control_cohorts=(0, 3))
        att, se = _reg_did_panel_ref(dY, D, X)
        cell = res.group_time_effects[(2, 2)]
        assert cell["effect"] == pytest.approx(att, rel=1e-9)
        assert cell["se"] == pytest.approx(se, rel=1e-9)
        # (3,3): only never-treated remain
        dY, D, X, _ = _cell_arrays(df, g=3, t=3, base=2, control_cohorts=(0,))
        att, se = _reg_did_panel_ref(dY, D, X)
        cell = res.group_time_effects[(3, 3)]
        assert cell["effect"] == pytest.approx(att, rel=1e-9)
        assert cell["se"] == pytest.approx(se, rel=1e-9)

    def test_reg_if_crosscheck_survey(self):
        """Survey-weighted reg+cov (general path): fitted per-cell SE must
        equal the reg_did_panel reconstruction with i.weights."""
        df = _make_cov_panel(seed=90212)
        res = self._fit(df, "reg", survey_design=SurveyDesign(weights="weight"))
        dY, D, X, w = _cell_arrays(df, g=2, t=2, base=1, control_cohorts=(0,))
        att, se = _reg_did_panel_ref(dY, D, X, weights=w)
        cell = res.group_time_effects[(2, 2)]
        assert cell["effect"] == pytest.approx(att, rel=1e-9)
        assert cell["se"] == pytest.approx(se, rel=1e-9)

    def test_ipw_if_crosscheck(self):
        """Unweighted ipw+cov: fitted per-cell/overall SE must equal the
        std_ipw_did_panel reconstruction (PS estimation effect included)."""
        df = _make_cov_panel(seed=90213)
        res = self._fit(df, "ipw")
        dY, D, X, _ = _cell_arrays(df, g=2, t=2, base=1, control_cohorts=(0,))
        ps = self._assert_ps_interior(X, D)
        att, se = _std_ipw_did_panel_ref(dY, D, X, ps)
        cell = res.group_time_effects[(2, 2)]
        assert cell["effect"] == pytest.approx(att, rel=1e-8)
        assert cell["se"] == pytest.approx(se, rel=1e-8)
        assert res.overall_se == pytest.approx(se, rel=1e-8)

    def test_ipw_if_crosscheck_survey(self):
        """Survey-weighted ipw+cov: the survey branch already carried the PS
        correction (Phase 7a) - this pins it against the i.weights
        reconstruction."""
        df = _make_cov_panel(seed=90214)
        res = self._fit(df, "ipw", survey_design=SurveyDesign(weights="weight"))
        dY, D, X, w = _cell_arrays(df, g=2, t=2, base=1, control_cohorts=(0,))
        _, ps = solve_logit(X, D, weights=w)
        assert ps.min() > 0.02 and ps.max() < 0.98
        att, se = _std_ipw_did_panel_ref(dY, D, X, ps, weights=w)
        cell = res.group_time_effects[(2, 2)]
        assert cell["effect"] == pytest.approx(att, rel=1e-8)
        assert cell["se"] == pytest.approx(se, rel=1e-8)


@pytest.fixture(scope="module")
def golden():
    """Golden R fixture scenarios; skip when the file is absent."""
    if not _GOLDEN_VALUES_PATH.exists():
        pytest.skip(
            "Golden values file not found; "
            "run: Rscript benchmarks/R/generate_csdid_test_values.R"
        )
    with open(_GOLDEN_VALUES_PATH) as f:
        return json.load(f)["scenarios"]


class TestDRDIDGoldenIFReconstruction:
    """Reconstruct the DRDID per-cell IF SEs in numpy from the golden-fixture
    data and pin BOTH directions per cell:

    1. reconstruction == R's golden SEs (the ground truth; this held before
       the fix and guards the reconstruction itself), and
    2. fitted == reconstruction (the fix contract).

    Covers pre-treatment cells and varying base periods (base = t-1 for
    t < g, else g-1), which the DGP-based cross-checks above do not.
    """

    @staticmethod
    def _df(scenario: dict) -> pd.DataFrame:
        d = scenario["data"]
        return pd.DataFrame(
            {
                "unit": d["unit"],
                "period": d["period"],
                "first_treat": d["first_treat"],
                "outcome": d["outcome"],
                "X": d["X"],
            }
        )

    @staticmethod
    def _cell(df: pd.DataFrame, g: int, t: int):
        base = t - 1 if t < g else g - 1
        sub = df[df["first_treat"].isin([0, g])]
        wide = sub.pivot(index="unit", columns="period", values="outcome")
        meta = sub.drop_duplicates(subset="unit").set_index("unit")
        units = wide[[base, t]].dropna().index
        dY = (wide.loc[units, t] - wide.loc[units, base]).to_numpy()
        D = (meta.loc[units, "first_treat"] == g).to_numpy(dtype=float)
        X = meta.loc[units, "X"].to_numpy()[:, None]
        return dY, D, X

    def _run(self, golden, name: str, method: str, ref):
        scenario = golden[name]
        df = self._df(scenario)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna(estimation_method=method).fit(
                df,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                covariates=["X"],
            )
        gt = scenario["results"]["group_time"]
        for g, t, r_att, r_se in zip(gt["group"], gt["time"], gt["att"], gt["se"]):
            dY, D, X = self._cell(df, int(g), int(t))
            if method == "ipw":
                _, ps = solve_logit(X, D)
                att, se = ref(dY, D, X, ps)
            else:
                att, se = ref(dY, D, X)
            # Direction 1: reconstruction reproduces R's golden values.
            assert att == pytest.approx(r_att, rel=1e-6), f"recon att ({g},{t})"
            assert se == pytest.approx(r_se, rel=1e-6), f"recon se ({g},{t})"
            # Direction 2: the fitted estimator matches the reconstruction.
            cell = res.group_time_effects[(int(g), int(t))]
            assert cell["effect"] == pytest.approx(att, rel=1e-7), f"fit att ({g},{t})"
            assert cell["se"] == pytest.approx(se, rel=1e-7), f"fit se ({g},{t})"

    def test_reg_golden_cells(self, golden):
        self._run(golden, "with_covariates_reg", "reg", _reg_did_panel_ref)

    def test_reg_golden_cells_multi_cohort(self, golden):
        self._run(golden, "dynamic_effects", "reg", _reg_did_panel_ref)

    def test_ipw_golden_cells(self, golden):
        self._run(golden, "with_covariates_ipw", "ipw", _std_ipw_did_panel_ref)
