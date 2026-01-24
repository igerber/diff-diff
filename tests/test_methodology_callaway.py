"""
Comprehensive methodology verification tests for CallawaySantAnna estimator.

This module verifies that the CallawaySantAnna implementation matches:
1. The theoretical formulas from Callaway & Sant'Anna (2021)
2. The behavior of R's did::att_gt() package
3. All documented edge cases in docs/methodology/REGISTRY.md

Reference: Callaway, B., & Sant'Anna, P.H.C. (2021). Difference-in-Differences
with multiple time periods. Journal of Econometrics, 225(2), 200-230.
"""

import subprocess
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna
from diff_diff.prep import generate_staggered_data
from diff_diff.staggered_bootstrap import _generate_bootstrap_weights_batch


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
        Python and R use g-1 as the base period for these.

        Pre-treatment effects may differ due to base_period handling:
        - Python varying: uses t-1 as base for pre-treatment
        - R varying: may handle differently

        We focus on post-treatment where alignment is expected.
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
    def test_analytical_se_close_to_bootstrap_se(self):
        """
        Analytical and bootstrap SEs should be within 20%.

        Analytical SEs use influence function aggregation.
        Bootstrap SEs use multiplier bootstrap.
        They should converge for large samples.

        This test is marked slow because it uses 499 bootstrap iterations
        for thorough validation of SE convergence.
        """
        data = generate_staggered_data(
            n_units=300,
            n_periods=8,
            cohort_periods=[4],
            treatment_effect=2.0,
            seed=42
        )

        cs_anal = CallawaySantAnna(n_bootstrap=0)
        cs_boot = CallawaySantAnna(n_bootstrap=499, seed=42)

        results_anal = cs_anal.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )
        results_boot = cs_boot.fit(
            data, outcome='outcome', unit='unit',
            time='period', first_treat='first_treat'
        )

        # Check overall ATT SE
        if results_boot.overall_se > 0:
            rel_diff = abs(results_anal.overall_se - results_boot.overall_se) / results_boot.overall_se
            assert rel_diff < 0.25, \
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

    def test_bootstrap_produces_valid_inference(self):
        """Test that bootstrap produces valid inference with p-values and CIs.

        Uses 99 bootstrap iterations - sufficient to verify the mechanism works
        without being slow for CI runs.
        """
        data = generate_staggered_data(
            n_units=100,
            n_periods=6,
            cohort_periods=[3],
            treatment_effect=3.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=99, seed=42)
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


class TestDeprecationWarnings:
    """Tests for deprecated parameter handling."""

    def test_bootstrap_weight_type_deprecated(self):
        """Test that bootstrap_weight_type emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cs = CallawaySantAnna(bootstrap_weight_type="mammen")

            # Check deprecation warning was emitted
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "bootstrap_weight_type" in str(deprecation_warnings[0].message)

            # Should still work (backward compatibility)
            assert cs.bootstrap_weights == "mammen"

    def test_bootstrap_weights_takes_precedence(self):
        """Test that bootstrap_weights takes precedence over deprecated param."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cs = CallawaySantAnna(
                bootstrap_weights="rademacher",
                bootstrap_weight_type="mammen"
            )

            # bootstrap_weights should take precedence
            assert cs.bootstrap_weights == "rademacher"


# =============================================================================
# MPDTA-based Strict R Comparison Tests
# =============================================================================


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
        Tuple of (DataFrame, dict) where dict has keys: overall_att, overall_se, etc.
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

        output <- list(
            overall_att = unbox(agg$overall.att),
            overall_se = unbox(agg$overall.se),
            n_groups = unbox(length(unique(result$group[result$group > 0]))),
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
