"""
Comprehensive methodology verification tests for TwoWayFixedEffects estimator.

This module verifies that the TwoWayFixedEffects implementation matches:
1. The theoretical formulas from within-transformation algebra
2. The behavior of R's fixest::feols() with absorbed unit+time FE
3. All documented edge cases in docs/methodology/REGISTRY.md

References:
- Wooldridge, J.M. (2010). Econometric Analysis of Cross Section and Panel Data, 2nd ed.
  MIT Press, Chapter 10.
- Goodman-Bacon, A. (2021). Difference-in-Differences with variation in treatment timing.
  Journal of Econometrics, 225(2), 254-277.
"""

import json
import os
import subprocess
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from diff_diff import TwoWayFixedEffects
from diff_diff.linalg import LinearRegression
from diff_diff.utils import within_transform

# =============================================================================
# R Availability Fixtures
# =============================================================================

_fixest_available_cache = None


def _check_fixest_available() -> bool:
    """Check if R and fixest package are available (cached)."""
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
# Data Generation Helpers
# =============================================================================


def generate_twfe_panel(
    n_units: int = 20,
    n_periods: int = 4,
    treatment_effect: float = 3.0,
    noise_sd: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate panel data for TWFE testing with known ATT."""
    np.random.seed(seed)
    n_treated = n_units // 2
    data = []

    for unit in range(n_units):
        is_treated = unit < n_treated
        unit_effect = np.random.normal(0, 2)

        for period in range(n_periods):
            post = 1 if period >= n_periods // 2 else 0
            time_effect = period * 1.0

            y = 10.0 + unit_effect + time_effect
            if is_treated and post:
                y += treatment_effect
            y += np.random.normal(0, noise_sd)

            data.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": post,
                    "outcome": y,
                }
            )

    return pd.DataFrame(data)


def generate_hand_calculable_panel() -> pd.DataFrame:
    """
    Generate a minimal 2-period panel with exact hand-calculable values.

    4 units (2 treated, 2 control) × 2 periods = 8 observations.
    No noise, so ATT is exactly 3.0.
    """
    return pd.DataFrame(
        {
            "unit": [0, 0, 1, 1, 2, 2, 3, 3],
            "period": [0, 1, 0, 1, 0, 1, 0, 1],
            "treated": [1, 1, 1, 1, 0, 0, 0, 0],
            "post": [0, 1, 0, 1, 0, 1, 0, 1],
            "outcome": [
                10.0,
                15.0,  # Unit 0 (treated): pre=10, post=15 (diff=5)
                12.0,
                17.0,  # Unit 1 (treated): pre=12, post=17 (diff=5)
                8.0,
                10.0,  # Unit 2 (control): pre=8, post=10 (diff=2)
                6.0,
                8.0,  # Unit 3 (control): pre=6, post=8 (diff=2)
            ],
        }
    )
    # ATT = (mean treated diff) - (mean control diff) = 5.0 - 2.0 = 3.0


# =============================================================================
# Phase 1: Within-Transformation Algebra
# =============================================================================


class TestWithinTransformationAlgebra:
    """Verify the within-transformation (two-way demeaning) is correct."""

    def test_within_transform_hand_calculation(self):
        """Verify within-transformation matches hand calculation: y_it - ȳ_i - ȳ_t + ȳ."""
        data = generate_hand_calculable_panel()

        # Hand-calculate within-transformed outcome
        # Unit means: unit 0 = 12.5, unit 1 = 14.5, unit 2 = 9.0, unit 3 = 7.0
        # Time means: period 0 = (10+12+8+6)/4 = 9.0, period 1 = (15+17+10+8)/4 = 12.5
        # Grand mean = (10+15+12+17+8+10+6+8)/8 = 86/8 = 10.75
        unit_means = data.groupby("unit")["outcome"].transform("mean")
        time_means = data.groupby("period")["outcome"].transform("mean")
        grand_mean = data["outcome"].mean()
        expected_demeaned = data["outcome"] - unit_means - time_means + grand_mean

        # Use the library function
        result = within_transform(data, ["outcome"], "unit", "period")

        np.testing.assert_allclose(
            result["outcome_demeaned"].values,
            expected_demeaned.values,
            rtol=1e-12,
        )

    def test_within_transform_covariates_also_demeaned(self):
        """Verify covariates are demeaned (not just outcome)."""
        data = generate_twfe_panel(n_units=10, n_periods=4, seed=123)
        data["x1"] = np.random.default_rng(42).normal(0, 1, len(data))

        result = within_transform(data, ["outcome", "x1"], "unit", "period")

        # Demeaned covariates should sum to ~0 within each unit and time group
        for var in ["outcome_demeaned", "x1_demeaned"]:
            unit_sums = result.groupby("unit")[var].sum()
            time_sums = result.groupby("period")[var].sum()
            np.testing.assert_allclose(unit_sums.values, 0, atol=1e-10)
            np.testing.assert_allclose(time_sums.values, 0, atol=1e-10)

    def test_twfe_att_matches_hand_calculated_demeaned_ols(self):
        """
        Verify TWFE ATT matches manual demeaned OLS on a small panel.

        By FWL theorem, regressing demeaned Y on demeaned (D_i * Post_t) gives ATT.
        Both outcome and regressors must be within-transformed.
        """
        data = generate_hand_calculable_panel()

        # Run TWFE
        twfe = TwoWayFixedEffects()
        results = twfe.fit(data, outcome="outcome", treatment="treated", post="post", unit="unit")

        # Manual demeaned OLS: demean both y and the interaction term
        data_with_tp = data.copy()
        data_with_tp["tp"] = data["treated"] * data["post"]
        demeaned = within_transform(data_with_tp, ["outcome", "tp"], "unit", "period")
        y = demeaned["outcome_demeaned"].values
        tp = demeaned["tp_demeaned"].values
        X = np.column_stack([np.ones(len(y)), tp])
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        manual_att = coeffs[1]

        np.testing.assert_allclose(results.att, manual_att, rtol=1e-10)

    def test_twfe_att_matches_basic_did_for_two_period_design(self):
        """TWFE and basic DiD should agree on 2-period data."""
        from diff_diff import DifferenceInDifferences

        data = generate_hand_calculable_panel()

        # TWFE
        twfe = TwoWayFixedEffects()
        twfe_results = twfe.fit(
            data, outcome="outcome", treatment="treated", post="post", unit="unit"
        )

        # Basic DiD
        did = DifferenceInDifferences(cluster="unit")
        did_results = did.fit(data, outcome="outcome", treatment="treated", post="post")

        np.testing.assert_allclose(twfe_results.att, did_results.att, rtol=1e-10)

    def test_demeaned_outcome_sums_to_zero(self):
        """Within-transformed outcome sums to zero within each unit and time group."""
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=99)

        result = within_transform(data, ["outcome"], "unit", "period")

        unit_sums = result.groupby("unit")["outcome_demeaned"].sum()
        time_sums = result.groupby("period")["outcome_demeaned"].sum()

        np.testing.assert_allclose(unit_sums.values, 0, atol=1e-10)
        np.testing.assert_allclose(time_sums.values, 0, atol=1e-10)

    def test_within_transform_weighted_warns_on_nonconvergence(self):
        """Silent-failure audit axis B: within_transform weighted path must warn."""
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=99)
        weights = np.ones(len(data))

        with pytest.warns(UserWarning, match="did not converge"):
            within_transform(
                data,
                ["outcome"],
                "unit",
                "period",
                weights=weights,
                max_iter=1,
                tol=1e-15,
            )

    def test_within_transform_weighted_no_warning_on_convergence(self):
        """Silent-failure audit axis B: no warning on well-behaved convergent input."""
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=99)
        weights = np.ones(len(data))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            within_transform(data, ["outcome"], "unit", "period", weights=weights)
        assert not any("did not converge" in str(x.message) for x in w)


# =============================================================================
# Phase 2: R Comparison
# =============================================================================


def _run_r_feols_twfe(data_path: str, covariates=None) -> Dict[str, Any]:
    """Run R's fixest::feols() with absorbed unit+post FE, clustered at unit."""
    escaped_path = data_path.replace("\\", "/")

    if covariates:
        cov_str = " + ".join(covariates)
        formula = f"outcome ~ treated:post + {cov_str} | unit + post"
    else:
        formula = "outcome ~ treated:post | unit + post"

    r_script = f"""
    suppressMessages(library(fixest))
    suppressMessages(library(jsonlite))

    data <- read.csv("{escaped_path}")
    data$treated <- as.numeric(data$treated)
    data$post <- as.numeric(data$post)

    result <- feols({formula}, data = data, cluster = ~unit)

    # Use coeftable() to get fixest's own inference (SE, t-stat, p-value)
    # This ensures we use fixest's df adjustment, not a manual pt() call
    ct <- coeftable(result)
    att_row <- which(rownames(ct) == "treated:post")
    if (length(att_row) == 0) {{
        att_row <- which(grepl("treated.*post", rownames(ct)))
    }}

    att <- ct[att_row, "Estimate"]
    se_val <- ct[att_row, "Std. Error"]
    tstat <- ct[att_row, "t value"]
    pval <- ct[att_row, "Pr(>|t|)"]
    ci <- confint(result)
    ci_lower <- ci[att_row, 1]
    ci_upper <- ci[att_row, 2]

    output <- list(
        att = unbox(att),
        se = unbox(se_val),
        t_stat = unbox(tstat),
        p_value = unbox(pval),
        ci_lower = unbox(ci_lower),
        ci_upper = unbox(ci_upper),
        n_obs = unbox(result$nobs)
    )

    cat(toJSON(output, pretty = TRUE, digits = 15))
    """

    result = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(f"R script failed: {result.stderr}")

    parsed = json.loads(result.stdout)
    # Unwrap single-element lists from R's JSON encoding
    for key in parsed:
        if isinstance(parsed[key], list) and len(parsed[key]) == 1:
            parsed[key] = parsed[key][0]

    return parsed


@pytest.fixture(scope="session")
def r_benchmark_panel_data(tmp_path_factory):
    """Session-scoped panel data + CSV for R comparison (no covariate)."""
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

            data.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": post,
                    "outcome": y,
                }
            )

    df = pd.DataFrame(data)
    tmp_dir = tmp_path_factory.mktemp("r_benchmark")
    csv_path = tmp_dir / "panel_data.csv"
    df.to_csv(csv_path, index=False)
    return df, str(csv_path)


@pytest.fixture(scope="session")
def r_benchmark_panel_data_with_covariate(tmp_path_factory):
    """Session-scoped panel data + CSV for R comparison (with covariate)."""
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
            x1 = np.random.normal(0, 1) + period * 0.3

            y = 10.0 + unit_effect + period_effect + 1.5 * x1
            if is_treated and post:
                y += 3.0
            y += np.random.normal(0, 0.5)

            data.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": post,
                    "outcome": y,
                    "x1": x1,
                }
            )

    df = pd.DataFrame(data)
    tmp_dir = tmp_path_factory.mktemp("r_benchmark_cov")
    csv_path = tmp_dir / "panel_data_cov.csv"
    df.to_csv(csv_path, index=False)
    return df, str(csv_path)


@pytest.fixture(scope="session")
def r_twfe_results(fixest_available, r_benchmark_panel_data):
    """Cache R fixest results for the base panel (session-scoped)."""
    if not fixest_available:
        pytest.skip("R or fixest package not available")
    _, csv_path = r_benchmark_panel_data
    return _run_r_feols_twfe(csv_path)


@pytest.fixture(scope="session")
def r_twfe_results_with_covariate(fixest_available, r_benchmark_panel_data_with_covariate):
    """Cache R fixest results for the covariate panel (session-scoped)."""
    if not fixest_available:
        pytest.skip("R or fixest package not available")
    _, csv_path = r_benchmark_panel_data_with_covariate
    return _run_r_feols_twfe(csv_path, covariates=["x1"])


class TestRBenchmarkTWFE:
    """Compare TWFE estimates against R's fixest::feols() with absorbed FE."""

    def _run_python_twfe(self, data, covariates=None):
        """Run Python TWFE estimator."""
        twfe = TwoWayFixedEffects()
        results = twfe.fit(
            data,
            outcome="outcome",
            treatment="treated",
            post="post",
            unit="unit",
            covariates=covariates,
        )
        return results

    def test_att_matches_r_twfe(self, r_twfe_results, r_benchmark_panel_data):
        """ATT matches R's fixest at machine precision.

        Measured agreement is ~4e-16 relative (the point estimate is
        deterministic algebra on both sides); rtol=1e-12 is platform
        headroom. The prior 1e-3 band predated the D4 within-transform
        rescale work."""
        data, _ = r_benchmark_panel_data

        py_results = self._run_python_twfe(data)

        np.testing.assert_allclose(
            py_results.att,
            r_twfe_results["att"],
            rtol=1e-12,
            err_msg=f"ATT mismatch: Python={py_results.att:.6f}, R={r_twfe_results['att']:.6f}",
        )

    def test_se_matches_r_twfe(self, r_twfe_results, r_benchmark_panel_data):
        """Cluster-robust SE matches fixest under the K_reference convergence.

        Measured gap is ~1.4e-15 relative — the historical ~2.5e-3 band was
        defect D2 (the CR1 factor omitted the non-nested time FE fixest
        counts), closed in 3.9. Pinned at rtol=1e-9 rather than a
        machine-epsilon literal because this MAP-demean lane measured up to
        5.2e-11 on other fixtures (BLAS-order dependent)."""
        data, _ = r_benchmark_panel_data

        py_results = self._run_python_twfe(data)

        np.testing.assert_allclose(
            py_results.se,
            r_twfe_results["se"],
            rtol=1e-9,
            err_msg=f"SE mismatch: Python={py_results.se:.6f}, R={r_twfe_results['se']:.6f}",
        )

    def test_pvalue_matches_r_twfe(self, r_twfe_results, r_benchmark_panel_data):
        """Both p-values are numerically zero at this effect size.

        Python p ~1e-48 (t-dist at residual df=148) vs R ~4e-27 (fixest's
        cluster df G-1=49): both sides use t-distributions, but the df
        conventions differ — the documented clustered-CR1 inference-df
        deviation (REGISTRY §TwoWayFixedEffects) — so at |t|~22 the tails
        diverge by ~21 orders and a relative comparison is meaningless.
        The assert pins that both are numerically zero (atol=1e-12,
        ~15 orders of headroom above the larger tail)."""
        data, _ = r_benchmark_panel_data

        py_results = self._run_python_twfe(data)

        np.testing.assert_allclose(
            py_results.p_value,
            r_twfe_results["p_value"],
            atol=1e-12,
            err_msg=f"P-value mismatch: Python={py_results.p_value:.6f}, R={r_twfe_results['p_value']:.6f}",
        )

    def test_moderate_t_pins_residual_df_convention(self):
        """Locks WHICH df convention clustered-CR1 inference follows, in a
        regime where the conventions are distinguishable.

        The R-benchmark tests above sit at |t|~22 where both the residual-df
        and cluster-df (G-1) p-values are numerically zero, so they cannot
        tell the conventions apart. This fixture is tuned to a moderate
        |t|~1.8, where t(residual df=148) and t(G-1=49) p-values differ by
        ~5% relative: the estimator's p-value must match the residual-df
        tail exactly AND differ measurably from the cluster-df tail — the
        documented clustered-CR1 inference-df deviation (REGISTRY
        §TwoWayFixedEffects). If the default convention ever flips to
        cluster df (the planned opt-in knob's v4 default), this test fails
        loudly and must be updated alongside the REGISTRY note."""
        from scipy import stats as _stats

        rng = np.random.default_rng(42)
        n_units, n_periods = 50, 4
        rows = []
        for i in range(n_units):
            treated_unit = i < 25
            for t in range(n_periods):
                post = 1 if t >= 2 else 0
                y = (
                    1.0
                    + 0.3 * i / n_units
                    + 0.2 * t
                    + (0.25 if (treated_unit and post) else 0.0)
                    + rng.normal(0, 1.0)
                )
                rows.append(
                    {
                        "unit": i,
                        "post": post,
                        "treated": int(treated_unit and post),
                        "outcome": y,
                    }
                )
        data = pd.DataFrame(rows)

        res = self._run_python_twfe(data)

        # Guard the regime: a data-gen drift that pushes |t| into the tails
        # (where the two conventions converge to 0) would silently defang
        # the convention assertions below.
        assert 1.5 < abs(res.t_stat) < 2.5, f"|t| left the moderate regime: {res.t_stat}"

        df_residual = len(data) - (n_units + 2)  # intercept + treated + post + (n_units-1) dummies
        p_residual = 2 * _stats.t.sf(abs(res.t_stat), df_residual)
        p_cluster = 2 * _stats.t.sf(abs(res.t_stat), n_units - 1)

        np.testing.assert_allclose(
            res.p_value,
            p_residual,
            rtol=1e-10,
            err_msg="clustered-CR1 p-value no longer follows the residual-df convention",
        )
        rel_gap = abs(p_cluster - p_residual) / p_residual
        assert (
            rel_gap > 0.03
        ), f"fixture no longer distinguishes the df conventions (rel gap {rel_gap:.4f})"
        assert (
            abs(res.p_value - p_cluster) / p_cluster > 0.03
        ), "p-value matches the cluster-df convention — the documented deviation flipped"

    def test_ci_matches_r_twfe(self, r_twfe_results, r_benchmark_panel_data):
        """CI bounds within the remaining df-convention band (measured ~1.7e-3).

        Post-K_reference the SE side matches fixest at machine precision, so
        the bounds' residual gap through att ± crit*se is SOLELY the clustered
        inference-df convention (Python t(residual df=148) critical value vs
        fixest t(G−1=49); REGISTRY §TwoWayFixedEffects — PR C scope).
        Tightened 0.005 → 0.003 accordingly (measured 1.7e-3/1.4e-3). The
        ATT itself matches at machine precision."""
        data, _ = r_benchmark_panel_data

        py_results = self._run_python_twfe(data)

        np.testing.assert_allclose(
            py_results.conf_int[0],
            r_twfe_results["ci_lower"],
            rtol=0.003,
            err_msg=f"CI lower mismatch: Python={py_results.conf_int[0]:.6f}, R={r_twfe_results['ci_lower']:.6f}",
        )
        np.testing.assert_allclose(
            py_results.conf_int[1],
            r_twfe_results["ci_upper"],
            rtol=0.003,
            err_msg=f"CI upper mismatch: Python={py_results.conf_int[1]:.6f}, R={r_twfe_results['ci_upper']:.6f}",
        )

    def test_att_matches_r_with_covariate(
        self, r_twfe_results_with_covariate, r_benchmark_panel_data_with_covariate
    ):
        """ATT with demeaned covariate matches R at machine precision
        (measured ~1e-15 relative; rtol=1e-12 is platform headroom)."""
        data, _ = r_benchmark_panel_data_with_covariate

        py_results = self._run_python_twfe(data, covariates=["x1"])

        np.testing.assert_allclose(
            py_results.att,
            r_twfe_results_with_covariate["att"],
            rtol=1e-12,
            err_msg=f"ATT w/ cov mismatch: Python={py_results.att:.6f}, R={r_twfe_results_with_covariate['att']:.6f}",
        )

    def test_se_matches_r_with_covariate(
        self, r_twfe_results_with_covariate, r_benchmark_panel_data_with_covariate
    ):
        """SE with covariate matches fixest under the K_reference convergence
        (the historical ~2.5e-3 D2 band is closed; see
        test_se_matches_r_twfe for the rtol=1e-9 headroom rationale)."""
        data, _ = r_benchmark_panel_data_with_covariate

        py_results = self._run_python_twfe(data, covariates=["x1"])

        np.testing.assert_allclose(
            py_results.se,
            r_twfe_results_with_covariate["se"],
            rtol=1e-9,
            err_msg=f"SE w/ cov mismatch: Python={py_results.se:.6f}, R={r_twfe_results_with_covariate['se']:.6f}",
        )


# =============================================================================
# Phase 3: Edge Cases (from REGISTRY.md)
# =============================================================================


class TestTWFEEdgeCases:
    """Test all edge cases documented in docs/methodology/REGISTRY.md."""

    def test_staggered_treatment_warning_multiperiod_time(self):
        """Staggered treatment warning fires when `time` is multi-valued.

        This tests the multi-period `time` scenario. When `time` has actual
        period values (not binary 0/1), the staggered check can detect
        different cohorts starting treatment at different periods. We use
        `time="period"` here because the standard binary `time="post"`
        configuration cannot detect staggering (see
        test_staggered_warning_not_fired_with_binary_time).
        """
        np.random.seed(42)
        data = []
        for unit in range(20):
            # Units 0-4: treated at period 2
            # Units 5-9: treated at period 3
            # Units 10-19: never treated
            for period in range(5):
                if unit < 5:
                    treated = 1 if period >= 2 else 0
                elif unit < 10:
                    treated = 1 if period >= 3 else 0
                else:
                    treated = 0
                y = 10.0 + unit * 0.1 + period * 0.5 + treated * 3.0 + np.random.normal(0, 0.5)
                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": treated,
                        "outcome": y,
                    }
                )
        df = pd.DataFrame(data)

        twfe = TwoWayFixedEffects()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Use time="period" so staggered detection sees different first-treat times
            twfe.fit(df, outcome="outcome", treatment="treated", post="period", unit="unit")

        staggered_warnings = [x for x in w if "Staggered treatment" in str(x.message)]
        assert len(staggered_warnings) > 0, "Expected staggered treatment warning"

        # Multi-period time warning also fires (time="period" has 5 unique values)
        multiperiod_warnings = [x for x in w if "unique values" in str(x.message)]
        assert (
            len(multiperiod_warnings) > 0
        ), "Expected multi-period time warning when time='period' with 5 values"

    def test_staggered_warning_not_fired_with_binary_time(self):
        """Staggered warning does NOT fire with binary time (known limitation).

        When `time` is a binary post indicator (0/1), all treated units appear
        to start treatment at time=1, so unique_treat_times=[1] and the
        staggered check cannot distinguish cohorts. This is a documented
        limitation — users with staggered designs should use `decompose()` or
        `CallawaySantAnna` directly.
        """
        np.random.seed(42)
        data = []
        for unit in range(20):
            # Units 0-4: treated at period 2 (early cohort)
            # Units 5-9: treated at period 3 (late cohort)
            # Units 10-19: never treated
            for period in range(5):
                if unit < 5:
                    treated = 1 if period >= 2 else 0
                elif unit < 10:
                    treated = 1 if period >= 3 else 0
                else:
                    treated = 0
                post = 1 if period >= 2 else 0
                y = 10.0 + unit * 0.1 + period * 0.5 + treated * 3.0 + np.random.normal(0, 0.5)
                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "post": post,
                        "treated": treated,
                        "outcome": y,
                    }
                )
        df = pd.DataFrame(data)

        twfe = TwoWayFixedEffects()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # With binary time="post", staggering is undetectable
            twfe.fit(df, outcome="outcome", treatment="treated", post="post", unit="unit")

        staggered_warnings = [x for x in w if "Staggered treatment" in str(x.message)]
        assert (
            len(staggered_warnings) == 0
        ), "Staggered warning should NOT fire with binary time (known limitation)"

    def test_multiperiod_time_warning(self):
        """Multi-period time column triggers UserWarning advising binary post indicator."""
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=42)

        twfe = TwoWayFixedEffects()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            twfe.fit(data, outcome="outcome", treatment="treated", post="period", unit="unit")

        multiperiod_warnings = [x for x in w if "unique values" in str(x.message)]
        assert (
            len(multiperiod_warnings) > 0
        ), "Expected multi-period time warning when time has >2 unique values"
        msg = str(multiperiod_warnings[0].message)
        assert "binary" in msg, "Warning should mention binary post indicator"
        assert "post" in msg, "Warning should mention post indicator"

    def test_binary_time_no_multiperiod_warning(self):
        """Binary time column does NOT trigger multi-period time warning."""
        data = generate_hand_calculable_panel()

        twfe = TwoWayFixedEffects()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            twfe.fit(data, outcome="outcome", treatment="treated", post="post", unit="unit")

        multiperiod_warnings = [x for x in w if "unique values" in str(x.message)]
        assert (
            len(multiperiod_warnings) == 0
        ), "Multi-period time warning should NOT fire with binary time"

    def test_non_binary_time_values_warning(self):
        """Non-{0,1} binary time values emit warning but ATT is correct."""
        data = generate_hand_calculable_panel()
        data["year"] = data["post"].map({0: 2020, 1: 2021})

        twfe = TwoWayFixedEffects()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = twfe.fit(
                data, outcome="outcome", treatment="treated", post="year", unit="unit"
            )

        non_binary_warnings = [x for x in w if "instead of {0, 1}" in str(x.message)]
        assert len(non_binary_warnings) > 0, "Expected warning about non-{0,1} binary time values"
        assert np.isfinite(results.att), "ATT should be finite"
        np.testing.assert_allclose(results.att, 3.0, rtol=1e-10)

    def test_boolean_time_no_warning(self):
        """Boolean time values ({False, True}) do NOT emit non-{0,1} warning."""
        data = generate_hand_calculable_panel()
        data["post_bool"] = data["post"].astype(bool)

        twfe = TwoWayFixedEffects()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            twfe.fit(
                data,
                outcome="outcome",
                treatment="treated",
                post="post_bool",
                unit="unit",
            )

        non_binary_warnings = [x for x in w if "instead of {0, 1}" in str(x.message)]
        assert (
            len(non_binary_warnings) == 0
        ), "Boolean time values should NOT trigger non-{0,1} warning"

    def test_att_invariant_to_time_encoding(self):
        """ATT, SE, and p-value are identical for {0,1} vs {2020,2021} time encoding."""
        data = generate_hand_calculable_panel()

        # Fit with binary {0,1}
        twfe = TwoWayFixedEffects()
        results_binary = twfe.fit(
            data, outcome="outcome", treatment="treated", post="post", unit="unit"
        )

        # Fit with year encoding {2020, 2021}
        data["year"] = data["post"].map({0: 2020, 1: 2021})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results_year = twfe.fit(
                data, outcome="outcome", treatment="treated", post="year", unit="unit"
            )

        np.testing.assert_allclose(
            results_binary.att,
            results_year.att,
            rtol=1e-10,
            err_msg="ATT should be invariant to time encoding",
        )
        np.testing.assert_allclose(
            results_binary.se,
            results_year.se,
            rtol=1e-10,
            err_msg="SE should be invariant to time encoding",
        )
        np.testing.assert_allclose(
            results_binary.p_value,
            results_year.p_value,
            rtol=1e-10,
            err_msg="P-value should be invariant to time encoding",
        )

    def test_auto_clusters_at_unit_level(self):
        """SE with cluster=None (default) equals SE when explicitly passing cluster='unit'."""
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=42)

        # Default (auto-clusters at unit)
        twfe_default = TwoWayFixedEffects()
        results_default = twfe_default.fit(
            data, outcome="outcome", treatment="treated", post="post", unit="unit"
        )

        # Explicit cluster at unit
        twfe_explicit = TwoWayFixedEffects(cluster="unit")
        results_explicit = twfe_explicit.fit(
            data, outcome="outcome", treatment="treated", post="post", unit="unit"
        )

        np.testing.assert_allclose(
            results_default.se,
            results_explicit.se,
            rtol=1e-12,
        )
        # Config should be immutable
        assert twfe_default.cluster is None

    def test_df_adjustment_for_absorbed_fe(self):
        """
        Verify degrees-of-freedom adjustment for absorbed fixed effects.

        TWFE applies df_adjustment = n_units + n_times - 2 to account for
        absorbed FE. Verify the SE matches a manual LinearRegression with
        the same df adjustment.
        """
        data = generate_twfe_panel(n_units=20, n_periods=2, noise_sd=0.5, seed=42)

        # Run TWFE
        twfe = TwoWayFixedEffects()
        results = twfe.fit(data, outcome="outcome", treatment="treated", post="post", unit="unit")

        # Manual: demean both y and the interaction, then run LinearRegression
        data_with_tp = data.copy()
        data_with_tp["tp"] = data["treated"] * data["post"]
        demeaned = within_transform(data_with_tp, ["outcome", "tp"], "unit", "period")
        y = demeaned["outcome_demeaned"].values
        tp = demeaned["tp_demeaned"].values
        X = np.column_stack([np.ones(len(y)), tp])

        n_units = data["unit"].nunique()
        n_times = data["period"].nunique()
        df_adjustment = n_units + n_times - 2
        cluster_ids = data["unit"].values

        # Clustered-CR1 K_reference: the manual reference carries the same
        # non-nested absorbed rank (time | unit) as the estimator — with the
        # unit cluster, the unit FE is nested (dropped) and the time FE adds
        # its conditional rank n_times - 1.
        reg = LinearRegression(
            include_intercept=False,
            cluster_ids=cluster_ids,
            rank_deficient_action="silent",
        ).fit(X, y, df_adjustment=df_adjustment, cluster_k_adjustment=n_times - 1)
        manual_se = reg.get_inference(1).se

        np.testing.assert_allclose(
            results.se,
            manual_se,
            rtol=1e-10,
            err_msg=f"SE df-adjustment mismatch: TWFE={results.se:.8f}, manual={manual_se:.8f}",
        )

    def test_covariate_collinear_with_interaction_raises_error(self):
        """Covariate identical to treatment*post interaction causes rank deficiency.

        Adding bad_cov = treated * post duplicates the internal _treatment_post
        variable, making the demeaned design matrix rank-deficient.
        """
        data = pd.DataFrame(
            {
                "unit": [0, 0, 1, 1, 2, 2, 3, 3],
                "period": [0, 1, 0, 1, 0, 1, 0, 1],
                "treated": [1, 1, 1, 1, 0, 0, 0, 0],
                "post": [0, 1, 0, 1, 0, 1, 0, 1],
                "outcome": [10.0, 11.0, 12.0, 13.0, 8.0, 9.0, 6.0, 7.0],
            }
        )

        # bad_cov = treated * post duplicates the internal _treatment_post column
        data["bad_cov"] = data["treated"] * data["post"]

        twfe = TwoWayFixedEffects(rank_deficient_action="error")
        with pytest.raises(ValueError):
            twfe.fit(
                data,
                outcome="outcome",
                treatment="treated",
                post="post",
                unit="unit",
                covariates=["bad_cov"],
            )

    def test_covariate_collinearity_warns_not_errors(self):
        """Collinear covariate emits warning but ATT is still finite."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)
        # Add a covariate that's collinear with treatment*post
        data["bad_cov"] = data["treated"] * data["post"]

        twfe = TwoWayFixedEffects(rank_deficient_action="warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = twfe.fit(
                data,
                outcome="outcome",
                treatment="treated",
                post="post",
                unit="unit",
                covariates=["bad_cov"],
            )

        collinear_warnings = [x for x in w if "collinear" in str(x.message).lower()]
        assert len(collinear_warnings) > 0, "Expected collinearity warning"
        assert np.isfinite(results.att), "ATT should be finite despite collinearity"
        # ATT should be in reasonable range of true effect (3.0)
        assert abs(results.att - 3.0) < 1.5, f"ATT={results.att} far from true effect 3.0"

    def test_rank_deficient_action_error_raises(self):
        """rank_deficient_action='error' raises ValueError on rank-deficient data."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)
        data["bad_cov"] = data["treated"] * data["post"]

        twfe = TwoWayFixedEffects(rank_deficient_action="error")
        with pytest.raises(ValueError):
            twfe.fit(
                data,
                outcome="outcome",
                treatment="treated",
                post="post",
                unit="unit",
                covariates=["bad_cov"],
            )

    def test_rank_deficient_action_silent_no_warning(self):
        """rank_deficient_action='silent' emits no warnings."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)
        data["bad_cov"] = data["treated"] * data["post"]

        twfe = TwoWayFixedEffects(rank_deficient_action="silent")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = twfe.fit(
                data,
                outcome="outcome",
                treatment="treated",
                post="post",
                unit="unit",
                covariates=["bad_cov"],
            )

        collinear_warnings = [x for x in w if "collinear" in str(x.message).lower()]
        assert len(collinear_warnings) == 0, "Expected no collinearity warnings with silent"
        assert np.isfinite(results.att)

    def test_unbalanced_panel_produces_valid_results(self):
        """Dropping some unit-period observations still gives valid results."""
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=42)

        # Drop some observations to create unbalanced panel
        drop_indices = [3, 7, 15, 22, 45, 60]
        data = data.drop(index=drop_indices).reset_index(drop=True)

        twfe = TwoWayFixedEffects()
        results = twfe.fit(data, outcome="outcome", treatment="treated", post="post", unit="unit")

        assert np.isfinite(results.att), "ATT should be finite for unbalanced panel"
        assert results.se > 0, "SE should be positive"
        assert results.n_obs == len(data)

    def test_unit_column_missing_raises_error(self):
        """Missing unit column raises ValueError."""
        data = generate_hand_calculable_panel()

        twfe = TwoWayFixedEffects()
        with pytest.raises(ValueError, match="not found"):
            twfe.fit(
                data,
                outcome="outcome",
                treatment="treated",
                post="post",
                unit="nonexistent_unit",
            )

    def test_decompose_integration(self):
        """decompose() returns BaconDecompositionResults for staggered data."""
        from diff_diff.bacon import BaconDecompositionResults

        np.random.seed(42)
        data = []
        for unit in range(30):
            if unit < 10:
                first_treat = 3
            elif unit < 20:
                first_treat = 4
            else:
                first_treat = 0  # never treated

            for period in range(1, 6):
                treated = 1 if (first_treat > 0 and period >= first_treat) else 0
                y = 10.0 + unit * 0.1 + period * 0.5 + treated * 2.0 + np.random.normal(0, 0.5)
                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "outcome": y,
                        "first_treat": first_treat,
                    }
                )

        df = pd.DataFrame(data)

        twfe = TwoWayFixedEffects()
        decomp = twfe.decompose(
            df, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )

        assert isinstance(decomp, BaconDecompositionResults)
        assert len(decomp.comparisons) > 0


# =============================================================================
# Phase 4: SE Verification
# =============================================================================


class TestTWFESEVerification:
    """Verify standard error properties."""

    def test_cluster_se_differs_from_hc1_se(self):
        """
        Cluster-robust SE differs from HC1 SE, verifying auto-clustering is active.

        TWFE auto-clusters at unit level. We manually compute HC1 SE on the
        same demeaned data (demeaned by unit + post, matching TWFE) and verify
        the SEs are different, proving clustering changes inference.
        """
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=42)

        # TWFE: cluster-robust at unit (automatic)
        twfe = TwoWayFixedEffects()
        twfe_results = twfe.fit(
            data, outcome="outcome", treatment="treated", post="post", unit="unit"
        )

        # Manual HC1 SE on same demeaned regression (no clustering)
        # Demean by unit + post to match TWFE's within-transform
        data_with_tp = data.copy()
        data_with_tp["tp"] = data["treated"] * data["post"]
        demeaned = within_transform(data_with_tp, ["outcome", "tp"], "unit", "post")
        y = demeaned["outcome_demeaned"].values
        tp = demeaned["tp_demeaned"].values
        X = np.column_stack([np.ones(len(y)), tp])
        n_units = data["unit"].nunique()
        n_times = data["post"].nunique()
        df_adjustment = n_units + n_times - 2

        hc1_reg = LinearRegression(
            include_intercept=False,
            cluster_ids=None,  # HC1, no clustering
            rank_deficient_action="silent",
        ).fit(X, y, df_adjustment=df_adjustment)
        hc1_se = hc1_reg.get_inference(1).se

        # Verify SEs are different (auto-clustering is active)
        assert twfe_results.se != hc1_se, (
            f"Cluster SE ({twfe_results.se:.6f}) should differ from "
            f"HC1 SE ({hc1_se:.6f}) — auto-clustering must be active"
        )

        # Also verify TWFE SE matches a manually computed cluster SE. The
        # manual reference carries the clustered-CR1 K_reference increment:
        # under the unit cluster the unit FE is nested (dropped) and the
        # time FE contributes its conditional rank n_times - 1.
        cluster_reg = LinearRegression(
            include_intercept=False,
            cluster_ids=data["unit"].values,
            rank_deficient_action="silent",
        ).fit(X, y, df_adjustment=df_adjustment, cluster_k_adjustment=n_times - 1)
        manual_cluster_se = cluster_reg.get_inference(1).se

        np.testing.assert_allclose(
            twfe_results.se,
            manual_cluster_se,
            rtol=1e-10,
            err_msg="TWFE SE should match manually computed cluster SE",
        )

    def test_vcov_positive_semidefinite(self):
        """VCoV matrix should be positive semi-definite."""
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=42)

        twfe = TwoWayFixedEffects()
        results = twfe.fit(data, outcome="outcome", treatment="treated", post="post", unit="unit")

        eigenvalues = np.linalg.eigvalsh(results.vcov)
        assert np.all(
            eigenvalues >= -1e-10
        ), f"VCoV has negative eigenvalues: {eigenvalues[eigenvalues < -1e-10]}"


# =============================================================================
# Phase 5: Wild Bootstrap
# =============================================================================


class TestTWFEWildBootstrap:
    """Verify wild cluster bootstrap inference."""

    def test_wild_bootstrap_produces_valid_inference(self, ci_params):
        """Wild bootstrap produces finite SE and valid p-value."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)
        n_boot = ci_params.bootstrap(999, min_n=199)

        twfe = TwoWayFixedEffects(inference="wild_bootstrap", n_bootstrap=n_boot, seed=42)
        results = twfe.fit(data, outcome="outcome", treatment="treated", post="post", unit="unit")

        assert np.isfinite(results.se) and results.se > 0
        assert 0 <= results.p_value <= 1
        assert results.inference_method == "wild_bootstrap"

    @pytest.mark.parametrize("weight_type", ["rademacher", "mammen", "webb"])
    def test_wild_bootstrap_weight_types(self, ci_params, weight_type):
        """Each bootstrap weight type produces valid inference."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)
        n_boot = ci_params.bootstrap(199, min_n=99)

        twfe = TwoWayFixedEffects(
            inference="wild_bootstrap",
            n_bootstrap=n_boot,
            bootstrap_weights=weight_type,
            seed=42,
        )
        results = twfe.fit(data, outcome="outcome", treatment="treated", post="post", unit="unit")

        assert np.isfinite(results.se) and results.se > 0
        assert 0 <= results.p_value <= 1

    def test_inference_parameter_routing(self):
        """inference='wild_bootstrap' routes to wild bootstrap method."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)

        twfe = TwoWayFixedEffects(inference="wild_bootstrap", n_bootstrap=99, seed=42)
        results = twfe.fit(data, outcome="outcome", treatment="treated", post="post", unit="unit")

        assert results.inference_method == "wild_bootstrap"


# =============================================================================
# Phase 6: Params & Results
# =============================================================================


class TestTWFEParamsAndResults:
    """Verify sklearn-like parameter interface and results completeness."""

    def test_get_params_returns_all_parameters(self):
        """All inherited constructor params present in get_params()."""
        twfe = TwoWayFixedEffects()
        params = twfe.get_params()

        expected_keys = {
            "robust",
            "cluster",
            "alpha",
            "inference",
            "n_bootstrap",
            "bootstrap_weights",
            "seed",
            "rank_deficient_action",
        }
        assert expected_keys.issubset(
            params.keys()
        ), f"Missing params: {expected_keys - params.keys()}"

    def test_set_params_modifies_attributes(self):
        """set_params() modifies estimator attributes."""
        twfe = TwoWayFixedEffects()
        twfe.set_params(alpha=0.10, robust=False)

        assert twfe.alpha == 0.10
        assert twfe.robust is False

    def test_summary_contains_key_info(self):
        """summary() output contains ATT."""
        data = generate_hand_calculable_panel()
        twfe = TwoWayFixedEffects()
        results = twfe.fit(data, outcome="outcome", treatment="treated", post="post", unit="unit")

        summary = results.summary()
        assert "ATT" in summary

    def test_to_dict_contains_all_fields(self):
        """to_dict() contains required fields."""
        data = generate_hand_calculable_panel()
        twfe = TwoWayFixedEffects()
        results = twfe.fit(data, outcome="outcome", treatment="treated", post="post", unit="unit")

        d = results.to_dict()
        for key in ["att", "se", "t_stat", "p_value", "n_obs"]:
            assert key in d, f"Missing key '{key}' in to_dict()"

    def test_residuals_plus_fitted_equals_demeaned_outcome(self):
        """Check residuals + fitted = demeaned outcome (not raw outcome).

        TWFE demeans by unit + time (where time is the `time` parameter).
        The demeaned outcome is the within-transformed y.
        """
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=42)

        twfe = TwoWayFixedEffects()
        results = twfe.fit(data, outcome="outcome", treatment="treated", post="post", unit="unit")

        # Within-transform by unit + post (same as TWFE internally does)
        demeaned = within_transform(data, ["outcome"], "unit", "post")
        y_demeaned = demeaned["outcome_demeaned"].values

        reconstructed = results.residuals + results.fitted_values
        np.testing.assert_allclose(
            reconstructed,
            y_demeaned,
            rtol=1e-10,
            err_msg="residuals + fitted_values should equal demeaned outcome",
        )


# =============================================================================
# HC2 / HC2 Bell-McCaffrey R parity (Gate 1)
# =============================================================================


def _load_twfe_golden_scenario():
    """Load the `twfe_two_period` scenario from clubsandwich_cr2_golden.json.

    Returns the parsed scenario dict, or None if the JSON / scenario is
    missing (caller should pytest.skip).
    """
    import json
    from pathlib import Path

    golden_path = (
        Path(__file__).parent.parent / "benchmarks" / "data" / "clubsandwich_cr2_golden.json"
    )
    if not golden_path.exists():
        return None
    with open(golden_path) as f:
        golden = json.load(f)
    return golden.get("twfe_two_period")


class TestTWFEHC2RParity:
    """R parity for TwoWayFixedEffects with vcov_type in {hc2, hc2_bm}.

    These tests pin Python's ATT SE / BM DOF on the new full-dummy
    auto-route path against the R targets in
    benchmarks/data/clubsandwich_cr2_golden.json under the
    `twfe_two_period` scenario. Tolerance is atol=1e-10 (the
    same target used for the existing absorbed-FE DiD / MPD parity tests
    in tests/test_linalg_hc2_bm.py).

    Skips when the golden JSON or the scenario is missing — regenerate
    via ``Rscript benchmarks/R/generate_clubsandwich_golden.R``.
    """

    def _build_panel(self, scenario):
        return pd.DataFrame(
            {
                "unit": scenario["unit"],
                "period": scenario["period"],
                "treated": scenario["treated"],
                "post": scenario["post"],
                "y": scenario["y"],
            }
        )

    def test_twfe_hc2_se_matches_r_lm_vcovHC(self):
        """TwoWayFixedEffects(vcov_type='hc2') ATT SE matches R
        sandwich::vcovHC(lm(y ~ treat_post + factor(unit) + factor(post)),
        type='HC2') at atol=1e-10.

        Singleton-cluster CR2 trick verified separately by the BM DOF test
        below; here we pin the HC2 vcov diagonal on the ATT coefficient.
        """
        scenario = _load_twfe_golden_scenario()
        if scenario is None:
            pytest.skip(
                "twfe_two_period scenario not in golden JSON; regenerate via "
                "`Rscript benchmarks/R/generate_clubsandwich_golden.R`."
            )
        data = self._build_panel(scenario)
        res = TwoWayFixedEffects(vcov_type="hc2").fit(
            data, outcome="y", treatment="treated", post="post", unit="unit"
        )
        vcov_R = np.array(scenario["vcov_hc2"]).reshape(scenario["vcov_hc2_shape"], order="F")
        # ATT is the 2nd coef (index 1) in the R design
        # `lm(y ~ treat_post + factor(unit) + factor(post))`.
        att_idx = scenario["coef_names"].index("treat_post")
        se_R = float(np.sqrt(vcov_R[att_idx, att_idx]))
        np.testing.assert_allclose(res.se, se_R, atol=1e-10, rtol=0)

    def test_twfe_hc2_bm_dof_matches_singleton_cluster_cr2(self):
        """One-way HC2-BM DOF matches clubSandwich's singleton-cluster CR2
        Satterthwaite DOF (Pustejovsky-Tipton 2018 Section 3.3; the trick is
        that CR2 with cluster=seq_len(n) reduces to Imbens-Kolesar BM).

        Pinned via the analytical one-way HC2-BM path (no auto-cluster):
        TwoWayFixedEffects(vcov_type='hc2_bm', cluster=...) → cluster-aware
        CR2-BM (not what we want here). We invoke the one-way path by
        explicitly passing an empty cluster column, which TWFE preserves
        as-is. Actually simpler: use the linalg helper directly on the
        same X built by TWFE and compare.
        """
        scenario = _load_twfe_golden_scenario()
        if scenario is None:
            pytest.skip("twfe_two_period scenario not in golden JSON.")
        if "dof_bm_one_way" not in scenario:
            pytest.skip(
                "twfe_two_period scenario does not include dof_bm_one_way; "
                "regenerate via the R script."
            )
        data = self._build_panel(scenario)
        # Build the same full-dummy design TWFE uses internally for
        # vcov_type='hc2_bm', then call compute_robust_vcov directly to
        # extract the per-coef BM DOF (the one-way HC2-BM path).
        from diff_diff.linalg import compute_robust_vcov, solve_ols

        data_local = data.copy()
        data_local["_tp"] = data_local["treated"] * data_local["post"]
        unit_d = pd.get_dummies(
            data_local["unit"], prefix="_fe_unit", drop_first=True
        ).values.astype(np.float64)
        time_d = pd.get_dummies(
            data_local["post"], prefix="_fe_post", drop_first=True
        ).values.astype(np.float64)
        X = np.column_stack(
            [
                np.ones(len(data_local)),
                data_local["_tp"].values.astype(np.float64),
                unit_d,
                time_d,
            ]
        )
        y = data_local["y"].values.astype(np.float64)
        _, residuals, _ = solve_ols(X, y, vcov_type="hc2")
        vcov_ow, dof_bm_one_way = compute_robust_vcov(
            X, residuals, vcov_type="hc2_bm", return_dof=True
        )
        att_idx = scenario["coef_names"].index("treat_post")
        dof_R = float(scenario["dof_bm_one_way"][att_idx])
        np.testing.assert_allclose(float(dof_bm_one_way[att_idx]), dof_R, atol=1e-10, rtol=0)
        # SE-audit C2: the one-way CR2 (HC2-BM) SE was loaded but never asserted.
        if "vcov_cr2_one_way" in scenario:
            n = len(scenario["coef_names"])
            vcov_cr2_ow = np.array(scenario["vcov_cr2_one_way"]).reshape((n, n), order="F")
            se_R = float(np.sqrt(vcov_cr2_ow[att_idx, att_idx]))
            np.testing.assert_allclose(
                float(np.sqrt(vcov_ow[att_idx, att_idx])), se_R, atol=1e-10, rtol=0
            )

    def test_twfe_hc2_bm_clustered_at_unit_dof_matches_clubsandwich(self):
        """CR2-BM DOF clustered at unit matches clubSandwich
        vcovCR(cluster=unit, type='CR2') + coef_test()$df_Satt at
        atol=1e-10.

        This is the inference path triggered by
        TwoWayFixedEffects(vcov_type='hc2_bm') on its default auto-cluster
        (cluster=unit).
        """
        scenario = _load_twfe_golden_scenario()
        if scenario is None:
            pytest.skip("twfe_two_period scenario not in golden JSON.")
        if "dof_bm_unit" not in scenario:
            pytest.skip(
                "twfe_two_period scenario does not include dof_bm_unit; "
                "regenerate via the R script."
            )
        data = self._build_panel(scenario)
        from diff_diff.linalg import compute_robust_vcov, solve_ols

        data_local = data.copy()
        data_local["_tp"] = data_local["treated"] * data_local["post"]
        unit_d = pd.get_dummies(
            data_local["unit"], prefix="_fe_unit", drop_first=True
        ).values.astype(np.float64)
        time_d = pd.get_dummies(
            data_local["post"], prefix="_fe_post", drop_first=True
        ).values.astype(np.float64)
        X = np.column_stack(
            [
                np.ones(len(data_local)),
                data_local["_tp"].values.astype(np.float64),
                unit_d,
                time_d,
            ]
        )
        y = data_local["y"].values.astype(np.float64)
        cluster_ids = np.asarray(data_local["unit"].values)
        _, residuals, _ = solve_ols(X, y, vcov_type="hc2")
        _, dof_bm_unit = compute_robust_vcov(
            X,
            residuals,
            cluster_ids=cluster_ids,
            vcov_type="hc2_bm",
            return_dof=True,
        )
        att_idx = scenario["coef_names"].index("treat_post")
        dof_R = float(scenario["dof_bm_unit"][att_idx])
        np.testing.assert_allclose(float(dof_bm_unit[att_idx]), dof_R, atol=1e-10, rtol=0)

    def test_twfe_hc2_bm_se_matches_clubsandwich_cr2_unit(self):
        """The SE ``TwoWayFixedEffects(vcov_type='hc2_bm')`` reports by default
        (CR2 clustered at unit) matches clubSandwich ``vcovCR(type='CR2')`` at
        atol=1e-10. SE-audit C2: the golden's ``vcov_cr2_unit`` diagonal was
        loaded for the DOF test but its SE was never asserted."""
        scenario = _load_twfe_golden_scenario()
        if scenario is None:
            pytest.skip("twfe_two_period scenario not in golden JSON.")
        if "vcov_cr2_unit" not in scenario:
            pytest.skip("twfe_two_period scenario does not include vcov_cr2_unit.")
        data = self._build_panel(scenario)
        res = TwoWayFixedEffects(vcov_type="hc2_bm").fit(
            data, outcome="y", treatment="treated", post="post", unit="unit"
        )
        n = len(scenario["coef_names"])
        vcov_cr2 = np.array(scenario["vcov_cr2_unit"]).reshape((n, n), order="F")
        att_idx = scenario["coef_names"].index("treat_post")
        se_R = float(np.sqrt(vcov_cr2[att_idx, att_idx]))
        np.testing.assert_allclose(res.se, se_R, atol=1e-10, rtol=0)
