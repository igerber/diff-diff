"""
Equation verification and R benchmark tests for ContinuousDiD.

Phase 1: Hand-calculable cases verifying the estimator recovers known truths.
Phase 2: R `contdid` benchmarks (skipped if R not installed).
"""

import json
import os
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.continuous_did import ContinuousDiD
from diff_diff.prep_dgp import generate_continuous_did_data

# =============================================================================
# Phase 1: Hand-calculable equation verification
# =============================================================================


class TestLinearDoseResponse:
    """Two-period case with linear dose-response ATT(d) = 2d."""

    @pytest.fixture
    def linear_data(self):
        """6 treated, 4 control. True ATT(d) = 2d. No noise."""
        treated_doses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        n_control = 4

        rows = []
        # Control units: Delta Y = 0 (no treatment)
        for i in range(n_control):
            rows.append({"unit": i, "period": 1, "outcome": 0.0, "first_treat": 0, "dose": 0.0})
            rows.append({"unit": i, "period": 2, "outcome": 0.0, "first_treat": 0, "dose": 0.0})

        # Treated units: Delta Y = ATT(d) = 2*d
        for j, d in enumerate(treated_doses):
            uid = n_control + j
            rows.append({"unit": uid, "period": 1, "outcome": 0.0, "first_treat": 2, "dose": d})
            rows.append({"unit": uid, "period": 2, "outcome": 2 * d, "first_treat": 2, "dose": d})

        return pd.DataFrame(rows)

    def test_linear_att_recovery(self, linear_data):
        """With degree=1 and linear truth, ATT(d) should be exactly 2d."""
        est = ContinuousDiD(degree=1, num_knots=0, dvals=np.array([1.0, 3.0, 5.0]))
        results = est.fit(linear_data, "outcome", "unit", "period", "first_treat", "dose")
        expected = np.array([2.0, 6.0, 10.0])
        np.testing.assert_allclose(results.dose_response_att.effects, expected, atol=1e-10)

    def test_linear_acrt(self, linear_data):
        """ACRT(d) should be constant = 2 for linear truth."""
        est = ContinuousDiD(degree=1, num_knots=0, dvals=np.array([1.5, 3.0, 4.5]))
        results = est.fit(linear_data, "outcome", "unit", "period", "first_treat", "dose")
        # Derivative of 2d is 2
        np.testing.assert_allclose(results.dose_response_acrt.effects, 2.0, atol=1e-6)

    def test_att_glob_binarized(self, linear_data):
        """ATT_glob = mean(Delta_Y | treated) - mean(Delta_Y | control)."""
        treated_doses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mean_delta_treated = np.mean(2 * treated_doses)  # = 7.0
        mean_delta_control = 0.0
        expected_att_glob = mean_delta_treated - mean_delta_control

        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(linear_data, "outcome", "unit", "period", "first_treat", "dose")
        np.testing.assert_allclose(results.overall_att, expected_att_glob, atol=1e-10)

    def test_acrt_glob_plugin(self, linear_data):
        """ACRT_glob = mean(ACRT(D_i)) over treated = 2."""
        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(linear_data, "outcome", "unit", "period", "first_treat", "dose")
        np.testing.assert_allclose(results.overall_acrt, 2.0, atol=1e-6)


class TestQuadraticWithCubicBasis:
    """ATT(d) = d^2. Cubic B-spline can represent quadratic exactly."""

    @pytest.fixture
    def quadratic_data(self):
        doses = np.linspace(1, 5, 20)
        n_control = 10

        rows = []
        for i in range(n_control):
            rows.append({"unit": i, "period": 1, "outcome": 0.0, "first_treat": 0, "dose": 0.0})
            rows.append({"unit": i, "period": 2, "outcome": 0.0, "first_treat": 0, "dose": 0.0})

        for j, d in enumerate(doses):
            uid = n_control + j
            rows.append({"unit": uid, "period": 1, "outcome": 0.0, "first_treat": 2, "dose": d})
            rows.append({"unit": uid, "period": 2, "outcome": d**2, "first_treat": 2, "dose": d})

        return pd.DataFrame(rows)

    def test_quadratic_recovery(self, quadratic_data):
        """Cubic basis should recover d^2 exactly."""
        eval_grid = np.array([1.5, 2.5, 3.5, 4.5])
        est = ContinuousDiD(degree=3, num_knots=0, dvals=eval_grid)
        results = est.fit(quadratic_data, "outcome", "unit", "period", "first_treat", "dose")
        expected = eval_grid**2
        np.testing.assert_allclose(results.dose_response_att.effects, expected, atol=1e-6)


class TestMultiPeriodAggregation:
    """4 periods, 2 cohorts. Verify (g,t) cells and aggregation weights."""

    @pytest.fixture
    def staggered_data(self):
        return generate_continuous_did_data(
            n_units=200,
            n_periods=4,
            cohort_periods=[2, 3],
            seed=42,
            noise_sd=0.0,  # No noise for exact verification
            att_function="linear",
            att_slope=2.0,
            att_intercept=1.0,
        )

    def test_multiple_groups(self, staggered_data):
        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(staggered_data, "outcome", "unit", "period", "first_treat", "dose")
        assert len(results.groups) == 2
        assert 2 in results.groups
        assert 3 in results.groups

    def test_gt_cell_count(self, staggered_data):
        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(staggered_data, "outcome", "unit", "period", "first_treat", "dose")
        # Group 2: periods 1(pre-via-varying),2,3,4; Group 3: periods 2(pre),3,4
        # Exact count depends on base period logic
        assert len(results.group_time_effects) >= 4


class TestEdgeCasesMethodology:
    """Edge cases: all-same dose, single treated unit, boundary doses."""

    def test_all_same_dose(self):
        """When all treated have same dose, OLS can only recover mean effect."""
        n_control = 10
        n_treated = 5
        dose_val = 3.0
        rows = []
        for i in range(n_control):
            rows.append({"unit": i, "period": 1, "outcome": 0.0, "first_treat": 0, "dose": 0.0})
            rows.append({"unit": i, "period": 2, "outcome": 0.0, "first_treat": 0, "dose": 0.0})
        for j in range(n_treated):
            uid = n_control + j
            rows.append(
                {"unit": uid, "period": 1, "outcome": 0.0, "first_treat": 2, "dose": dose_val}
            )
            rows.append(
                {"unit": uid, "period": 2, "outcome": 5.0, "first_treat": 2, "dose": dose_val}
            )

        data = pd.DataFrame(rows)
        est = ContinuousDiD(degree=1, num_knots=0, rank_deficient_action="silent")
        with pytest.warns(UserWarning, match="[Ii]dentical"):
            results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        # ATT_glob should be 5.0
        np.testing.assert_allclose(results.overall_att, 5.0, atol=1e-10)
        # Dose-response: ATT(d) should be constant = overall_att everywhere.
        # With all-same dose, only the intercept is identified, which equals
        # mean(delta_tilde_Y) = att_glob — same quantity by both paths.
        np.testing.assert_allclose(
            results.dose_response_att.effects,
            results.overall_att,
            atol=1e-10,
        )
        # ACRT(d) should be zero everywhere (no dose variation → zero derivative)
        np.testing.assert_allclose(
            results.dose_response_acrt.effects,
            0.0,
            atol=1e-10,
        )

        # Verify bootstrap path produces finite ATT SE for rank-deficient
        # cells — regression test for P1 bootstrap fix.  Use data with
        # heterogeneous outcomes (natural sampling variance) but the same
        # dose so the design matrix is still rank-deficient.
        # ACRT SE is correctly NaN: zero dose variation → zero-variance
        # bootstrap distribution → degenerate SE → NaN by design.
        rng = np.random.default_rng(123)
        rows_hetero = []
        for i in range(n_control):
            y_pre = rng.normal(0, 0.3)
            y_post = rng.normal(0, 0.3)
            rows_hetero.append(
                {"unit": i, "period": 1, "outcome": y_pre, "first_treat": 0, "dose": 0.0}
            )
            rows_hetero.append(
                {"unit": i, "period": 2, "outcome": y_post, "first_treat": 0, "dose": 0.0}
            )
        for j in range(n_treated):
            uid = n_control + j
            y_pre = rng.normal(0, 0.3)
            rows_hetero.append(
                {"unit": uid, "period": 1, "outcome": y_pre, "first_treat": 2, "dose": dose_val}
            )
            rows_hetero.append(
                {
                    "unit": uid,
                    "period": 2,
                    "outcome": y_pre + 5.0,
                    "first_treat": 2,
                    "dose": dose_val,
                }
            )
        data_hetero = pd.DataFrame(rows_hetero)
        est_boot = ContinuousDiD(
            degree=1,
            num_knots=0,
            n_bootstrap=199,
            rank_deficient_action="silent",
            seed=42,
        )
        with pytest.warns(UserWarning, match="[Ii]dentical"):
            results_boot = est_boot.fit(
                data_hetero, "outcome", "unit", "period", "first_treat", "dose"
            )
        assert np.all(np.isfinite(results_boot.dose_response_att.se))

    def test_single_treated_unit(self):
        """Single treated unit: not enough for OLS → no valid cells → ValueError."""
        rows = []
        for i in range(5):
            rows.append({"unit": i, "period": 1, "outcome": 0.0, "first_treat": 0, "dose": 0.0})
            rows.append({"unit": i, "period": 2, "outcome": 0.0, "first_treat": 0, "dose": 0.0})
        rows.append({"unit": 5, "period": 1, "outcome": 0.0, "first_treat": 2, "dose": 2.0})
        rows.append({"unit": 5, "period": 2, "outcome": 4.0, "first_treat": 2, "dose": 2.0})

        data = pd.DataFrame(rows)
        est = ContinuousDiD(degree=1, num_knots=0, rank_deficient_action="silent")
        with pytest.raises(ValueError, match="No valid"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")


# =============================================================================
# Phase 2: R `contdid` benchmarks
# =============================================================================


def _check_r_contdid():
    """Check if R and contdid package are available."""
    try:
        result = subprocess.run(
            ["Rscript", "-e", "library(contdid); cat('OK')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() == "OK"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_HAS_R_CONTDID = _check_r_contdid()

require_contdid = pytest.mark.skipif(
    not _HAS_R_CONTDID,
    reason="R or contdid package not installed",
)


def _run_r_contdid(
    csv_path,
    degree=3,
    num_knots=0,
    control_group="nevertreated",
    aggregation="dose",
    staggered=False,
):
    """Run R's cont_did() and return results for comparison.

    For 2-period data (staggered=False): recomputes ATT(d)/ACRT(d) with consistent
    boundary knots, fixing R's contdid v0.1.0 quirk of using range(dvals) instead
    of range(dose) for the evaluation basis.

    For multi-period data (staggered=True): compares only overall ATT/ACRT, which
    are not affected by the boundary knot issue.
    """
    cg = "nevertreated" if control_group == "never_treated" else "notyettreated"

    if staggered:
        # For staggered data, compare overall ATT/ACRT only
        r_code = f"""
        library(contdid)
        library(jsonlite)

        data <- read.csv("{csv_path}")
        res_level <- cont_did(
            yname = "outcome", tname = "period", idname = "unit",
            gname = "first_treat", dname = "dose", data = data,
            target_parameter = "level", aggregation = "{aggregation}",
            treatment_type = "continuous", control_group = "{cg}",
            degree = {degree}, num_knots = {num_knots},
            bstrap = FALSE, print_details = FALSE
        )
        res_slope <- cont_did(
            yname = "outcome", tname = "period", idname = "unit",
            gname = "first_treat", dname = "dose", data = data,
            target_parameter = "slope", aggregation = "{aggregation}",
            treatment_type = "continuous", control_group = "{cg}",
            degree = {degree}, num_knots = {num_knots},
            bstrap = FALSE, print_details = FALSE
        )
        out <- list(
            overall_att = res_level$overall_att,
            overall_att_se = res_level$overall_att_se,
            overall_acrt = res_slope$overall_acrt,
            overall_acrt_se = res_slope$overall_acrt_se,
            dvals = as.numeric(res_level$dose)
        )
        cat(toJSON(out, auto_unbox = TRUE, digits = 10))
        """
    else:
        # For 2-period data, recompute dose-response with consistent knots
        r_code = f"""
        library(contdid)
        library(jsonlite)
        library(splines2)

        data <- read.csv("{csv_path}")
        res <- cont_did(
            yname = "outcome", tname = "period", idname = "unit",
            gname = "first_treat", dname = "dose", data = data,
            target_parameter = "level", aggregation = "{aggregation}",
            treatment_type = "continuous", control_group = "{cg}",
            degree = {degree}, num_knots = {num_knots},
            bstrap = FALSE, print_details = FALSE
        )
        res_slope <- cont_did(
            yname = "outcome", tname = "period", idname = "unit",
            gname = "first_treat", dname = "dose", data = data,
            target_parameter = "slope", aggregation = "{aggregation}",
            treatment_type = "continuous", control_group = "{cg}",
            degree = {degree}, num_knots = {num_knots},
            bstrap = FALSE, print_details = FALSE
        )

        dvals <- as.numeric(res$dose)
        first_period <- min(data[["period"]])
        fp_data <- data[data[["period"]] == first_period,]
        treated_doses <- fp_data[["dose"]][fp_data[["first_treat"]] > 0 & fp_data[["dose"]] > 0]
        bknots <- range(treated_doses)
        interior_knots <- as.numeric(res$pte_params$knots)

        # Rebuild OLS with consistent boundary knots
        bs_train <- bSpline(treated_doses, degree = {degree},
                            knots = interior_knots, Boundary.knots = bknots,
                            intercept = FALSE)
        post_period <- sort(unique(data[["period"]]))[2]
        pre_data <- data[data[["period"]] == first_period,]
        post_data <- data[data[["period"]] == post_period,]
        pre_data <- pre_data[order(pre_data[["unit"]]),]
        post_data <- post_data[order(post_data[["unit"]]),]
        dy <- post_data[["outcome"]] - pre_data[["outcome"]]
        dy_treated <- dy[pre_data[["first_treat"]] > 0 & pre_data[["dose"]] > 0]
        dy_control <- dy[pre_data[["first_treat"]] == 0]
        mu_0 <- mean(dy_control)

        bs_df <- as.data.frame(bs_train)
        colnames(bs_df) <- paste0("V", seq_len(ncol(bs_df)))
        bs_df$dy <- dy_treated
        reg <- lm(dy ~ ., data = bs_df)
        beta <- coef(reg)

        bs_grid <- bSpline(dvals, degree = {degree}, knots = interior_knots,
                           Boundary.knots = bknots, intercept = FALSE)
        bs_grid_df <- as.data.frame(bs_grid)
        colnames(bs_grid_df) <- paste0("V", seq_len(ncol(bs_grid_df)))
        att_d <- predict(reg, newdata = bs_grid_df) - mu_0

        dbs_grid <- dbs(dvals, degree = {degree}, knots = interior_knots,
                        Boundary.knots = bknots)
        acrt_d <- as.numeric(dbs_grid %*% beta[-1])

        out <- list(
            overall_att = res$overall_att,
            overall_att_se = res$overall_att_se,
            overall_acrt = res_slope$overall_acrt,
            overall_acrt_se = res_slope$overall_acrt_se,
            att_d = as.numeric(att_d),
            acrt_d = acrt_d,
            dvals = dvals,
            beta = as.numeric(beta)
        )
        cat(toJSON(out, auto_unbox = TRUE, digits = 10))
        """
    result = subprocess.run(
        ["Rscript", "-e", r_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"R contdid failed: {result.stderr[:500]}")
    return json.loads(result.stdout)


@require_contdid
class TestRBenchmark:
    """R `contdid` v0.1.0 benchmark comparisons."""

    def _compare_with_r(
        self,
        data,
        degree=3,
        num_knots=0,
        control_group="never_treated",
        aggregation="dose",
        staggered=False,
        att_tol=0.01,
        acrt_tol=0.02,
    ):
        """Helper: run both Python and R, compare."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            data.to_csv(f, index=False)
            csv_path = f.name

        r_out = _run_r_contdid(
            csv_path,
            degree=degree,
            num_knots=num_knots,
            control_group=control_group,
            aggregation=aggregation,
            staggered=staggered,
        )

        # Python estimation using R's dvals for exact grid match. M-025:
        # the dose curves are always computed by fit(); the event-study
        # route (R aggregation "eventstudy") is exercised POST-FIT so the
        # benchmark keeps validating that code path.
        dvals = np.array(r_out["dvals"])
        est = ContinuousDiD(
            degree=degree,
            num_knots=num_knots,
            dvals=dvals,
            control_group=control_group,
        )
        results = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
        )
        if aggregation == "eventstudy":
            results.aggregate("event_study")

        # Compare overall ATT
        r_overall_att = r_out["overall_att"]
        py_overall_att = results.overall_att
        overall_att_diff = abs(py_overall_att - r_overall_att) / (abs(r_overall_att) + 1e-10)
        assert overall_att_diff < att_tol, (
            f"Overall ATT diff: {overall_att_diff:.4f} "
            f"(R={r_overall_att:.6f}, Py={py_overall_att:.6f})"
        )

        # Compare ATT(d) and ACRT(d) only for non-staggered cases
        # (staggered cases have the R boundary knot quirk in aggregated curves)
        if not staggered:
            r_att_d = np.array(r_out["att_d"])
            py_att_d = results.dose_response_att.effects
            rel_diff_att = np.abs(py_att_d - r_att_d) / (np.abs(r_att_d) + 1e-10)
            max_att_diff = np.max(rel_diff_att)
            assert max_att_diff < att_tol, (
                f"ATT(d) max relative diff: {max_att_diff:.4f}\n"
                f"  R:  {r_att_d[:5]}...\n"
                f"  Py: {py_att_d[:5]}..."
            )

            r_acrt_d = np.array(r_out["acrt_d"])
            py_acrt_d = results.dose_response_acrt.effects
            rel_diff_acrt = np.abs(py_acrt_d - r_acrt_d) / (np.abs(r_acrt_d) + 1e-10)
            max_acrt_diff = np.max(rel_diff_acrt)
            assert max_acrt_diff < acrt_tol, (
                f"ACRT(d) max relative diff: {max_acrt_diff:.4f}\n"
                f"  R:  {r_acrt_d[:5]}...\n"
                f"  Py: {py_acrt_d[:5]}..."
            )

        return results, r_out

    def test_benchmark_1_basic_cubic(self):
        """2 periods, 1 cohort, degree=3, no knots, never_treated."""
        data = generate_continuous_did_data(
            n_units=300,
            n_periods=2,
            cohort_periods=[2],
            seed=100,
            noise_sd=0.5,
        )
        self._compare_with_r(data, degree=3, num_knots=0)

    def test_benchmark_2_linear(self):
        """2 periods, 1 cohort, degree=1 (linear), never_treated."""
        data = generate_continuous_did_data(
            n_units=300,
            n_periods=2,
            cohort_periods=[2],
            seed=101,
            noise_sd=0.5,
        )
        self._compare_with_r(data, degree=1, num_knots=0)

    def test_benchmark_3_interior_knots(self):
        """2 periods, 1 cohort, degree=3, 2 interior knots."""
        data = generate_continuous_did_data(
            n_units=300,
            n_periods=2,
            cohort_periods=[2],
            seed=102,
            noise_sd=0.5,
        )
        self._compare_with_r(data, degree=3, num_knots=2)

    def test_benchmark_4_staggered_dose(self):
        """4 periods, 3 cohorts, degree=3, dose aggregation.

        Uses R's simulate_contdid_data() to generate data compatible with
        contdid's internal aggregation. Compares overall_att and overall_acrt
        via pte_default (with consistent control_group).
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            r_code = f"""
            library(contdid)
            library(ptetools)
            library(jsonlite)

            set.seed(42)
            df <- simulate_contdid_data(
                n = 200, num_time_periods = 4, num_groups = 4,
                dose_linear_effect = 2, dose_quadratic_effect = 0.5
            )

            # Overall ACRT via cont_did (dose aggregation)
            res_slope <- cont_did(
                yname = "Y", tname = "time_period", idname = "id",
                gname = "G", dname = "D", data = df,
                target_parameter = "slope", aggregation = "dose",
                treatment_type = "continuous", control_group = "nevertreated",
                degree = 3, num_knots = 0, bstrap = FALSE, print_details = FALSE
            )

            # Overall ATT via pte_default (with matching control_group)
            att_res <- suppressWarnings(pte_default(
                yname = "Y", gname = "G", tname = "time_period",
                idname = "id", data = df, d_outcome = TRUE,
                anticipation = 0, base_period = "varying",
                control_group = "nevertreated",
                biters = 100, alp = 0.05
            ))

            write.csv(df, "{tmp_path}", row.names = FALSE)
            out <- list(
                overall_att = att_res$overall_att$overall.att,
                overall_acrt = res_slope$overall_acrt,
                dvals = as.numeric(res_slope$dose)
            )
            cat(toJSON(out, auto_unbox = TRUE, digits = 10))
            """
            result = subprocess.run(
                ["Rscript", "-e", r_code],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                pytest.skip(f"R contdid failed: {result.stderr[:500]}")
            r_out = json.loads(result.stdout)

            data = pd.read_csv(tmp_path)
            data = data.rename(
                columns={
                    "id": "unit",
                    "time_period": "period",
                    "Y": "outcome",
                    "G": "first_treat",
                    "D": "dose",
                }
            )
            dvals = np.array(r_out["dvals"])
            est = ContinuousDiD(
                degree=3,
                num_knots=0,
                dvals=dvals,
                control_group="never_treated",
            )
            results = est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "dose",
            )

            # Overall ATT
            att_diff = abs(results.overall_att - r_out["overall_att"]) / (
                abs(r_out["overall_att"]) + 1e-10
            )
            assert att_diff < 0.01, (
                f"Overall ATT diff: {att_diff:.4f} "
                f"(R={r_out['overall_att']:.6f}, Py={results.overall_att:.6f})"
            )

            # Overall ACRT
            acrt_diff = abs(results.overall_acrt - r_out["overall_acrt"]) / (
                abs(r_out["overall_acrt"]) + 1e-10
            )
            assert acrt_diff < 0.01, (
                f"Overall ACRT diff: {acrt_diff:.4f} "
                f"(R={r_out['overall_acrt']:.6f}, Py={results.overall_acrt:.6f})"
            )
        finally:
            os.unlink(tmp_path)

    def test_benchmark_5_not_yet_treated(self):
        """4 periods, 3 cohorts, not-yet-treated control."""
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            r_code = f"""
            library(contdid)
            library(ptetools)
            library(jsonlite)

            set.seed(123)
            df <- simulate_contdid_data(
                n = 200, num_time_periods = 4, num_groups = 4,
                dose_linear_effect = 1.5, dose_quadratic_effect = 0
            )

            res_slope <- cont_did(
                yname = "Y", tname = "time_period", idname = "id",
                gname = "G", dname = "D", data = df,
                target_parameter = "slope", aggregation = "dose",
                treatment_type = "continuous", control_group = "notyettreated",
                degree = 3, num_knots = 0, bstrap = FALSE, print_details = FALSE
            )

            att_res <- suppressWarnings(pte_default(
                yname = "Y", gname = "G", tname = "time_period",
                idname = "id", data = df, d_outcome = TRUE,
                anticipation = 0, base_period = "varying",
                control_group = "notyettreated",
                biters = 100, alp = 0.05
            ))

            write.csv(df, "{tmp_path}", row.names = FALSE)
            out <- list(
                overall_att = att_res$overall_att$overall.att,
                overall_acrt = res_slope$overall_acrt,
                dvals = as.numeric(res_slope$dose)
            )
            cat(toJSON(out, auto_unbox = TRUE, digits = 10))
            """
            result = subprocess.run(
                ["Rscript", "-e", r_code],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                pytest.skip(f"R contdid failed: {result.stderr[:500]}")
            r_out = json.loads(result.stdout)

            data = pd.read_csv(tmp_path)
            data = data.rename(
                columns={
                    "id": "unit",
                    "time_period": "period",
                    "Y": "outcome",
                    "G": "first_treat",
                    "D": "dose",
                }
            )
            dvals = np.array(r_out["dvals"])
            est = ContinuousDiD(
                degree=3,
                num_knots=0,
                dvals=dvals,
                control_group="not_yet_treated",
            )
            results = est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "dose",
            )

            att_diff = abs(results.overall_att - r_out["overall_att"]) / (
                abs(r_out["overall_att"]) + 1e-10
            )
            assert att_diff < 0.01, (
                f"Overall ATT diff: {att_diff:.4f} "
                f"(R={r_out['overall_att']:.6f}, Py={results.overall_att:.6f})"
            )

            acrt_diff = abs(results.overall_acrt - r_out["overall_acrt"]) / (
                abs(r_out["overall_acrt"]) + 1e-10
            )
            assert acrt_diff < 0.01, (
                f"Overall ACRT diff: {acrt_diff:.4f} "
                f"(R={r_out['overall_acrt']:.6f}, Py={results.overall_acrt:.6f})"
            )
        finally:
            os.unlink(tmp_path)

    def test_benchmark_6_event_study(self):
        """4 periods, 3 cohorts, event study aggregation (binarized ATT).

        R's event study uses ptetools::did_attgt (standard binary DiD) for
        per-cell estimation, then aggregates by relative period. We compare
        overall ATT (binarized) via pte_default with matching control_group.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            r_code = f"""
            library(contdid)
            library(ptetools)
            library(jsonlite)

            set.seed(99)
            df <- simulate_contdid_data(
                n = 200, num_time_periods = 4, num_groups = 4,
                dose_linear_effect = 2, dose_quadratic_effect = 0
            )

            # Overall ATT via pte_default (matching control_group)
            att_res <- suppressWarnings(pte_default(
                yname = "Y", gname = "G", tname = "time_period",
                idname = "id", data = df, d_outcome = TRUE,
                anticipation = 0, base_period = "varying",
                control_group = "nevertreated",
                biters = 100, alp = 0.05
            ))

            write.csv(df, "{tmp_path}", row.names = FALSE)
            out <- list(
                overall_att = att_res$overall_att$overall.att
            )
            cat(toJSON(out, auto_unbox = TRUE, digits = 10))
            """
            result = subprocess.run(
                ["Rscript", "-e", r_code],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                pytest.skip(f"R contdid failed: {result.stderr[:500]}")
            r_out = json.loads(result.stdout)

            data = pd.read_csv(tmp_path)
            data = data.rename(
                columns={
                    "id": "unit",
                    "time_period": "period",
                    "Y": "outcome",
                    "G": "first_treat",
                    "D": "dose",
                }
            )
            est = ContinuousDiD(
                degree=3,
                num_knots=0,
                control_group="never_treated",
            )
            results = est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "dose",
            )
            # M-025: keep validating the event-study code path (this
            # benchmark's documented purpose) via the post-fit route -
            # overall_att itself is aggregate-independent.
            results.aggregate("event_study")

            # Compare overall ATT (binarized)
            att_diff = abs(results.overall_att - r_out["overall_att"]) / (
                abs(r_out["overall_att"]) + 1e-10
            )
            assert att_diff < 0.01, (
                f"Overall ATT diff: {att_diff:.4f} "
                f"(R={r_out['overall_att']:.6f}, Py={results.overall_att:.6f})"
            )
        finally:
            os.unlink(tmp_path)


# =============================================================================
# Phase 3: Covariate adjustment (conditional parallel trends) — reg + dr
#
# The covariate-adjusted dose curve has NO external anchor (`contdid` v0.1.0
# hard-stops on covariates). Validation strategy:
#   - scalar overall_att + SE map EXACTLY onto DRDID reg_did_panel / drdid_panel
#     (skip-guarded, DRDID not in CI);
#   - an R-free NumPy reconstruction of the reg/dr att + SE runs IN CI (the guard
#     the p=1 reduction cannot provide for the dr propensity/augmentation terms);
#   - reg/dr ACRT(d) identity, DGP recovery, and MC coverage (R-free).
# =============================================================================


def _check_r_drdid():
    try:
        result = subprocess.run(
            ["Rscript", "-e", "library(DRDID); cat('OK')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() == "OK"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_HAS_R_DRDID = _check_r_drdid()
require_drdid = pytest.mark.skipif(not _HAS_R_DRDID, reason="R or DRDID package not installed")


def _covariate_cell_data(seed=101, n=500):
    """2-period single-cohort panel with 2 covariates and conditional PT.

    Returns (df, dY, D, X_with_intercept) so the estimator's single (g=2,t=2)
    cell overall_att maps onto one DRDID reg/dr att.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    treated = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.4 * x1 - 0.3 * x2)))).astype(int)
    dose = np.where(treated == 1, rng.uniform(0.2, 2.0, n), 0.0)
    g = np.where(treated == 1, 2, 0)
    y1 = 1.0 + 0.5 * x1 + rng.normal(0, 0.3, n)
    y2 = (
        y1 + (0.8 * x1 - 0.5 * x2) + np.where(treated == 1, 1.5 * dose, 0.0) + rng.normal(0, 0.3, n)
    )
    rows = []
    for i in range(n):
        for period, y in [(1, y1[i]), (2, y2[i])]:
            rows.append(
                {
                    "unit": i,
                    "period": period,
                    "outcome": y,
                    "first_treat": g[i],
                    "dose": dose[i],
                    "x1": x1[i],
                    "x2": x2[i],
                }
            )
    df = pd.DataFrame(rows)
    dY = y2 - y1
    D = treated.astype(float)
    X = np.column_stack([np.ones(n), x1, x2])
    return df, dY, D, X


def _numpy_reg_dr(dY, D, X):
    """Independent NumPy reconstruction of DRDID reg_did_panel / drdid_panel
    att + SE (unit weights=1, moderate overlap => no trimming). Separate
    implementation from the estimator internals — the in-CI transcription guard.
    """
    n = len(D)
    cf = D == 0
    gamma = np.linalg.lstsq(X[cf], dY[cf], rcond=None)[0]
    out = X @ gamma

    def se(inf):
        return inf.std(ddof=1) * np.sqrt(n - 1) / n

    # reg
    eta_t = (D * dY).mean() / D.mean()
    eta_c = (D * out).mean() / D.mean()
    reg_att = eta_t - eta_c
    XpX = ((1 - D)[:, None] * X).T @ X / n
    XpXi = np.linalg.inv(XpX)
    asy = ((1 - D) * (dY - out))[:, None] * X @ XpXi
    inf_t = (D * dY - D * eta_t) / D.mean()
    inf_c1 = D * out - D * eta_c
    M1 = (D[:, None] * X).mean(0)
    reg_inf = inf_t - (inf_c1 + asy @ M1) / D.mean()
    # dr
    ps = 1 / (1 + np.exp(-X @ np.linalg.lstsq(X, D, rcond=None)[0]))  # rough; refit below
    # proper logit via IRLS
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-X @ beta))
        W = mu * (1 - mu)
        z = X @ beta + (D - mu) / np.clip(W, 1e-10, None)
        beta_new = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-12:
            beta = beta_new
            break
        beta = beta_new
    ps = np.clip(1 / (1 + np.exp(-X @ beta)), 0.01, 0.99)
    wt = D
    wc = ps * (1 - D) / (1 - ps)
    drt = wt * (dY - out)
    drc = wc * (dY - out)
    et = drt.mean() / wt.mean()
    ec = drc.mean() / wc.mean()
    dr_att = et - ec
    asy_w = ((1 - D) * (dY - out))[:, None] * X @ XpXi
    Wm = ps * (1 - ps)
    Hess = np.linalg.inv(X.T @ (Wm[:, None] * X)) * n
    asy_ps = ((D - ps)[:, None] * X) @ Hess
    it = (drt - wt * et - asy_w @ (wt[:, None] * X).mean(0)) / wt.mean()
    M2 = (wc[:, None] * (dY - out - ec)[:, None] * X).mean(0)
    M3 = (wc[:, None] * X).mean(0)
    ic = (drc - wc * ec + asy_ps @ M2 - asy_w @ M3) / wc.mean()
    dr_inf = it - ic
    return {"reg_att": reg_att, "reg_se": se(reg_inf), "dr_att": dr_att, "dr_se": se(dr_inf)}


def _fit_cov(df, method, **kw):
    est = ContinuousDiD(estimation_method=method, **kw)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit(
            df,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "dose",
            covariates=["x1", "x2"],
        )


class TestCovariateReg:
    """reg + dr covariate adjustment (conditional parallel trends)."""

    @require_drdid
    def test_reg_matches_drdid(self):
        df, dY, D, X = _covariate_cell_data()
        with tempfile.TemporaryDirectory() as tmp:
            np.savetxt(f"{tmp}/dY.csv", dY, delimiter=",")
            np.savetxt(f"{tmp}/D.csv", D, delimiter=",")
            np.savetxt(f"{tmp}/X.csv", X, delimiter=",")
            r = subprocess.run(
                [
                    "Rscript",
                    "-e",
                    f"""
                suppressMessages(library(DRDID)); library(jsonlite)
                dY<-as.numeric(read.csv("{tmp}/dY.csv",header=F)[,1])
                D<-as.numeric(read.csv("{tmp}/D.csv",header=F)[,1])
                X<-as.matrix(read.csv("{tmp}/X.csv",header=F)); n<-length(D)
                o<-reg_did_panel(dY,rep(0,n),D,covariates=X)
                cat(toJSON(list(att=o$ATT,se=o$se),auto_unbox=T,digits=12))
            """,
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                pytest.skip(f"R DRDID failed: {r.stderr[:300]}")
            ref = json.loads(r.stdout)
        res = _fit_cov(df, "reg")
        assert abs(float(res.overall_att) - ref["att"]) < 1e-8
        assert abs(float(res.overall_att_se) - ref["se"]) < 1e-8

    @require_drdid
    def test_dr_matches_drdid(self):
        df, dY, D, X = _covariate_cell_data()
        with tempfile.TemporaryDirectory() as tmp:
            np.savetxt(f"{tmp}/dY.csv", dY, delimiter=",")
            np.savetxt(f"{tmp}/D.csv", D, delimiter=",")
            np.savetxt(f"{tmp}/X.csv", X, delimiter=",")
            r = subprocess.run(
                [
                    "Rscript",
                    "-e",
                    f"""
                suppressMessages(library(DRDID)); library(jsonlite)
                dY<-as.numeric(read.csv("{tmp}/dY.csv",header=F)[,1])
                D<-as.numeric(read.csv("{tmp}/D.csv",header=F)[,1])
                X<-as.matrix(read.csv("{tmp}/X.csv",header=F)); n<-length(D)
                o<-drdid_panel(dY,rep(0,n),D,covariates=X)
                cat(toJSON(list(att=o$ATT,se=o$se),auto_unbox=T,digits=12))
            """,
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                pytest.skip(f"R DRDID failed: {r.stderr[:300]}")
            ref = json.loads(r.stdout)
        res = _fit_cov(df, "dr")
        # dr att/se match DRDID (~1e-6; Python IRLS vs fastglm method=3)
        assert abs(float(res.overall_att) - ref["att"]) < 1e-5
        assert abs(float(res.overall_att_se) - ref["se"]) < 1e-5

    def test_dr_reg_numpy_crosscheck_p2(self):
        """R-free: estimator reg/dr att+se match an independent NumPy
        reconstruction at p>=2. This is the CI guard the p=1 reduction cannot
        provide for the dr propensity/augmentation terms (at p=1 dr collapses
        to reg)."""
        df, dY, D, X = _covariate_cell_data(seed=202)
        ref = _numpy_reg_dr(dY, D, X)
        reg = _fit_cov(df, "reg")
        dr = _fit_cov(df, "dr")
        assert abs(float(reg.overall_att) - ref["reg_att"]) < 1e-9
        assert abs(float(reg.overall_att_se) - ref["reg_se"]) < 1e-9
        assert abs(float(dr.overall_att) - ref["dr_att"]) < 1e-7
        assert abs(float(dr.overall_att_se) - ref["dr_se"]) < 1e-7

    def test_reg_vs_dr_acrt_identical(self):
        """reg and dr share the dose-response shape: ACRT(d) point AND SE are
        identical (the dr augmentation enters only the intercept direction,
        which dPsi annihilates). ATT(d) differs by a single constant."""
        df, _, _, _ = _covariate_cell_data(seed=303)
        reg = _fit_cov(df, "reg")
        dr = _fit_cov(df, "dr")
        np.testing.assert_allclose(
            reg.dose_response_acrt.effects, dr.dose_response_acrt.effects, atol=1e-10
        )
        np.testing.assert_allclose(reg.dose_response_acrt.se, dr.dose_response_acrt.se, atol=1e-10)
        att_diff = reg.dose_response_att.effects - dr.dose_response_att.effects
        assert np.ptp(att_diff) < 1e-9  # constant shift across the grid
        assert abs(float(reg.overall_att) - float(dr.overall_att)) > 1e-4  # levels differ

    def test_covariate_dgp_recovery(self):
        """Conditional-PT DGP where unconditional DiD is biased but covariate
        reg/dr recover the truth."""
        tau = 2.0
        rng = np.random.default_rng(7)
        n = 600
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        tr = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.7 * x1 - 0.5 * x2)))).astype(int)
        dose = np.where(tr == 1, rng.uniform(0.2, 2, n), 0.0)
        g = np.where(tr == 1, 2, 0)
        y1 = 1 + 0.5 * x1 + rng.normal(0, 0.3, n)
        y2 = y1 + (1.2 * x1 - 0.8 * x2) + np.where(tr == 1, tau, 0.0) + rng.normal(0, 0.3, n)
        rows = []
        for i in range(n):
            for period, y in [(1, y1[i]), (2, y2[i])]:
                rows.append(
                    {
                        "unit": i,
                        "period": period,
                        "outcome": y,
                        "first_treat": g[i],
                        "dose": dose[i],
                        "x1": x1[i],
                        "x2": x2[i],
                    }
                )
        df = pd.DataFrame(rows)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uncond = ContinuousDiD().fit(df, "outcome", "unit", "period", "first_treat", "dose")
        assert abs(float(uncond.overall_att) - tau) > 0.5  # unconditional biased
        for method in ("reg", "dr"):
            res = _fit_cov(df, method)
            assert abs(float(res.overall_att) - tau) < 0.1  # recovered

    @pytest.mark.slow
    def test_covariate_coverage(self):
        """Analytical SE achieves nominal coverage under conditional PT (reg &
        dr). R-free validation of the curve/scalar SE (esp. the dr SE). Slow —
        a fixed rep count is needed for a meaningful coverage estimate; the fast
        CI guard is test_dr_reg_numpy_crosscheck_p2."""
        tau = 2.0
        n = 400
        reps = 150
        import warnings

        for method in ("reg", "dr"):
            cover = 0
            total = 0
            for s in range(reps):
                rng = np.random.default_rng(10_000 + s)
                x1 = rng.normal(size=n)
                x2 = rng.normal(size=n)
                tr = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.7 * x1 - 0.5 * x2)))).astype(int)
                dose = np.where(tr == 1, rng.uniform(0.2, 2, n), 0.0)
                g = np.where(tr == 1, 2, 0)
                y1 = 1 + 0.5 * x1 + rng.normal(0, 0.3, n)
                y2 = (
                    y1 + (1.2 * x1 - 0.8 * x2) + np.where(tr == 1, tau, 0.0) + rng.normal(0, 0.3, n)
                )
                rows = []
                for i in range(n):
                    for period, y in [(1, y1[i]), (2, y2[i])]:
                        rows.append(
                            {
                                "unit": i,
                                "period": period,
                                "outcome": y,
                                "first_treat": g[i],
                                "dose": dose[i],
                                "x1": x1[i],
                                "x2": x2[i],
                            }
                        )
                df = pd.DataFrame(rows)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = _fit_cov(df, method)
                lo, hi = res.overall_att_conf_int
                cover += int(lo <= tau <= hi)
                total += 1
            rate = cover / total
            # Wide band: MC noise at 150 reps; catches a broken (too small/large) SE.
            assert 0.88 <= rate <= 0.99, f"{method} coverage {rate:.3f} off nominal"


# =============================================================================
# Discrete treatment: saturated regression (treatment_type="discrete")
# =============================================================================


def _make_discrete_panel(
    per_level_effect,
    n_per_level=40,
    n_control=80,
    control_trend=0.3,
    noise=0.5,
    seed=0,
    cohorts=(1,),
    n_periods=None,
):
    """Balanced discrete-dose panel with known per-level effects.

    ``per_level_effect`` maps dose level -> ATT(d). Each cohort in ``cohorts``
    (first_treat value) gets ``n_per_level`` treated units at every level, plus
    ``n_control`` never-treated units. Effects switch on at ``t >= first_treat``.
    """
    r = np.random.default_rng(seed)
    levels = sorted(per_level_effect)
    max_g = max(cohorts)
    if n_periods is None:
        n_periods = max_g + 1
    periods = list(range(n_periods))
    rows = []
    uid = 0

    def add_unit(ft, d):
        nonlocal uid
        base = r.normal(0, 1)
        for p in periods:
            on = ft > 0 and p >= ft
            y = base + control_trend * p + (per_level_effect[d] if on else 0.0)
            y += r.normal(0, noise)
            rows.append((uid, p, y, ft, d if ft > 0 else 0.0))
        uid += 1

    for ft in cohorts:
        for d in levels:
            for _ in range(n_per_level):
                add_unit(ft, d)
    for _ in range(n_control):
        add_unit(0, levels[0])
    return pd.DataFrame(rows, columns=["unit", "period", "outcome", "first_treat", "dose"])


def _hand_calc_discrete(df, levels):
    """Independent NumPy ATT(d_j) / ACRT / overall from a 2-period panel."""
    periods = sorted(df["period"].unique())
    p0, p1 = periods[0], periods[-1]
    wide = df.pivot(index="unit", columns="period", values="outcome")
    dy = (wide[p1] - wide[p0]).to_numpy()
    udose = df.groupby("unit")["dose"].first().to_numpy()
    uft = df.groupby("unit")["first_treat"].first().to_numpy()
    cm = uft == 0
    mu0 = dy[cm].mean()
    att = np.array([dy[udose == d].mean() - mu0 for d in levels])
    acrt = np.empty(len(levels))
    for j in range(len(levels)):
        if j == 0:
            # Backward difference to the zero-dose baseline d_0 = 0, ATT(0) = 0.
            acrt[0] = att[0] / levels[0]
        else:
            acrt[j] = (att[j] - att[j - 1]) / (levels[j] - levels[j - 1])
    overall = (dy[~cm] - mu0).mean()
    # Analytical per-level 2x2 SE in the sum-of-squares/n convention.
    n_c = int(cm.sum())
    se = np.empty(len(levels))
    for j, d in enumerate(levels):
        tmask = udose == d
        n_j = int(tmask.sum())
        it = (dy[tmask] - mu0 - att[j]) / n_j
        ic = -(dy[cm] - mu0) / n_c
        se[j] = np.sqrt((it**2).sum() + (ic**2).sum())
    return att, acrt, overall, se


def _hand_calc_lowest_dose(df, all_levels):
    """Independent NumPy lowest-dose-as-control (Remark 3.1) recomputation.

    ``all_levels`` are the sorted distinct positive doses; ``d_L = all_levels[0]``
    is the reference (control), modelled levels are ``all_levels[1:]``. Returns
    ``(modelled_levels, att, acrt, overall, overall_acrt, se)`` where
    ``att[j] = mean(dY | d_j) - mean(dY | d_L)`` (a per-level 2x2 DiD vs the
    lowest-dose group), ACRT is the backward difference with base ``d_L``, and
    ``se`` is the per-level 2x2 SE with the ``d_L`` group as control.
    """
    periods = sorted(df["period"].unique())
    p0, p1 = periods[0], periods[-1]
    wide = df.pivot(index="unit", columns="period", values="outcome")
    dy = (wide[p1] - wide[p0]).to_numpy()
    udose = df.groupby("unit")["dose"].first().to_numpy()
    d_L = all_levels[0]
    modelled = all_levels[1:]
    cm = np.abs(udose - d_L) <= 1e-9  # d_L group = control
    mu0 = dy[cm].mean()
    att = np.array([dy[np.abs(udose - d) <= 1e-9].mean() - mu0 for d in modelled])
    acrt = np.empty(len(modelled))
    for j in range(len(modelled)):
        if j == 0:
            acrt[0] = att[0] / (modelled[0] - d_L)  # backward diff to d_L
        else:
            acrt[j] = (att[j] - att[j - 1]) / (modelled[j] - modelled[j - 1])
    treated = udose - d_L > 1e-9
    overall = (dy[treated] - mu0).mean()
    # Plug-in overall ACRT: density-weighted mean over treated units' doses.
    tdose = udose[treated]
    acrt_by_level = {d: acrt[j] for j, d in enumerate(modelled)}
    overall_acrt = np.mean([acrt_by_level[min(modelled, key=lambda m: abs(m - d))] for d in tdose])
    n_c = int(cm.sum())
    se = np.empty(len(modelled))
    for j, d in enumerate(modelled):
        tmask = np.abs(udose - d) <= 1e-9
        n_j = int(tmask.sum())
        it = (dy[tmask] - mu0 - att[j]) / n_j
        ic = -(dy[cm] - mu0) / n_c
        se[j] = np.sqrt((it**2).sum() + (ic**2).sum())
    return modelled, att, acrt, overall, overall_acrt, se


class TestDiscreteSaturated:
    """Saturated regression for discrete/multi-valued treatment (CGBS 2024 Eq 4.1)."""

    _KW = dict(
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
        dose="dose",
    )

    def test_hand_calc_att_acrt_overall(self):
        """ATT(d_j) = per-level 2x2 DiD; ACRT = finite diffs; overall = mean_T."""
        levels = [1.0, 2.0, 4.0]
        df = _make_discrete_panel({1.0: 0.5, 2.0: 1.5, 4.0: 2.5}, seed=1)
        res = ContinuousDiD(treatment_type="discrete", n_bootstrap=0).fit(df, **self._KW)
        att, acrt, overall, _ = _hand_calc_discrete(df, levels)
        assert np.allclose(res.dose_response_att.dose_grid, levels)
        assert np.allclose(res.dose_response_att.effects, att, atol=1e-12)
        assert np.allclose(res.dose_response_acrt.effects, acrt, atol=1e-12)
        assert np.isclose(res.overall_att, overall, atol=1e-12)

    def test_hand_calc_analytical_se(self):
        """Saturated SE reduces exactly to the per-level 2x2 DiD SE (the gate)."""
        levels = [1.0, 2.0, 4.0]
        df = _make_discrete_panel({1.0: 0.5, 2.0: 1.5, 4.0: 2.5}, seed=2)
        res = ContinuousDiD(treatment_type="discrete", n_bootstrap=0).fit(df, **self._KW)
        _, _, _, se = _hand_calc_discrete(df, levels)
        assert np.allclose(res.dose_response_att.se, se, atol=1e-10)
        assert np.all(np.isfinite(res.dose_response_att.se))

    def test_acrt_boundary_backward_to_zero(self):
        """ACRT(d_1) = ATT(d_1)/d_1 (backward diff to d_0=0); ACRT(d_j>=2) backward."""
        levels = [1.0, 2.0, 4.0]
        df = _make_discrete_panel({1.0: 0.5, 2.0: 1.5, 4.0: 2.5}, seed=3)
        res = ContinuousDiD(treatment_type="discrete", n_bootstrap=0).fit(df, **self._KW)
        att = res.dose_response_att.effects
        acrt = res.dose_response_acrt.effects
        assert np.isclose(acrt[0], att[0] / levels[0], atol=1e-12)  # ref d_0 = 0
        assert np.isclose(acrt[1], (att[1] - att[0]) / (levels[1] - levels[0]), atol=1e-12)
        assert np.isclose(acrt[2], (att[2] - att[1]) / (levels[2] - levels[1]), atol=1e-12)

    def test_binary_single_dose_acrt_equals_att(self):
        """Single positive dose (J=1): ACRT(d_1) = ATT(d_1)/d_1; binary d=1 -> ACRT=ATT."""
        import warnings

        # d_1 = 1.0 -> ACRT = ATT (documented binary identity).
        df1 = _make_discrete_panel({1.0: 1.7}, n_per_level=60, n_control=90, seed=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = ContinuousDiD(treatment_type="discrete", n_bootstrap=0).fit(df1, **self._KW)
        assert np.isclose(
            r1.dose_response_acrt.effects[0], r1.dose_response_att.effects[0], atol=1e-12
        )
        assert np.isclose(r1.overall_acrt, r1.overall_att, atol=1e-12)
        assert np.isfinite(r1.dose_response_acrt.se[0]) and r1.dose_response_acrt.se[0] > 0
        # d_1 = 2.0 -> ACRT = ATT / 2.
        df2 = _make_discrete_panel({2.0: 3.0}, n_per_level=60, n_control=90, seed=31)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = ContinuousDiD(treatment_type="discrete", n_bootstrap=0).fit(df2, **self._KW)
        assert np.isclose(
            r2.dose_response_acrt.effects[0], r2.dose_response_att.effects[0] / 2.0, atol=1e-12
        )

    def test_dgp_recovery(self):
        """Recover heterogeneous per-level effects (no noise -> exact)."""
        effects = {1.0: 0.5, 2.0: 2.0, 4.0: 1.0}  # non-monotone ACRT
        df = _make_discrete_panel(effects, n_per_level=60, n_control=100, noise=0.0, seed=4)
        res = ContinuousDiD(treatment_type="discrete", n_bootstrap=0).fit(df, **self._KW)
        assert np.allclose(res.dose_response_att.effects, [0.5, 2.0, 1.0], atol=1e-10)
        # ACRT steps: d1 ref 0 = 0.5/1 = 0.5; bwd@d2 = (2-.5)/1 = 1.5; bwd@d3 = (1-2)/2 = -0.5
        assert np.allclose(res.dose_response_acrt.effects, [0.5, 1.5, -0.5], atol=1e-10)

    def test_staggered_shared_support(self):
        """Multi-cohort (shared dose support) discrete fit aggregates + recovers."""
        effects = {1.0: 0.5, 2.0: 1.5, 4.0: 2.5}
        df = _make_discrete_panel(
            effects, n_per_level=40, n_control=90, noise=0.0, seed=5, cohorts=(2, 3)
        )
        res = ContinuousDiD(treatment_type="discrete", n_bootstrap=0).fit(df, **self._KW)
        # Homogeneous effects across cohorts + no noise -> exact recovery.
        assert np.allclose(res.dose_response_att.effects, [0.5, 1.5, 2.5], atol=1e-9)
        assert np.all(np.isfinite(res.dose_response_att.se))

    @pytest.mark.slow
    def test_coverage(self, ci_params):
        """Analytical & bootstrap SE achieve nominal coverage for ATT(d_j)."""
        import warnings

        levels = [1.0, 2.0, 4.0]
        effects = {1.0: 0.5, 2.0: 1.5, 4.0: 2.5}
        reps = 150
        for use_boot in (False, True):
            # Bootstrap draws are scaled down in pure-Python mode; use a wider
            # coverage band there (mirrors the conftest small-n_boot tolerance
            # convention). Analytical SE has no such scaling.
            nb = ci_params.bootstrap(299, min_n=99) if use_boot else 0
            lo_band = 0.85 if (use_boot and nb < 200) else 0.88
            cover = np.zeros(len(levels))
            for s in range(reps):
                df = _make_discrete_panel(
                    effects, n_per_level=40, n_control=80, noise=1.0, seed=20_000 + s
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ContinuousDiD(treatment_type="discrete", n_bootstrap=nb, seed=s).fit(
                        df, **self._KW
                    )
                lo = res.dose_response_att.conf_int_lower
                hi = res.dose_response_att.conf_int_upper
                for j, d in enumerate(levels):
                    cover[j] += int(lo[j] <= effects[d] <= hi[j])
            rate = cover / reps
            assert np.all(
                (rate >= lo_band) & (rate <= 0.995)
            ), f"{'boot' if use_boot else 'analytic'} coverage {rate} off nominal"


class TestLowestDose:
    """Remark 3.1 lowest-dose-as-control (CGBS 2024): estimand ATT(d) - ATT(d_L)."""

    _KW = dict(
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
        dose="dose",
    )

    def test_dL_to_zero_exact_equivalence(self):
        """Relabelling the D=0 group as a tiny common dose d_L=eps reproduces the
        never_treated fit EXACTLY (same control units, same dY) — for any eps."""
        effects = {1.0: 0.5, 2.0: 1.5, 4.0: 2.5}
        df0 = _make_discrete_panel(effects, n_per_level=45, n_control=90, seed=11)
        res_nt = ContinuousDiD(control_group="never_treated", treatment_type="discrete").fit(
            df0, **self._KW
        )
        eps = 1e-6
        df_re = df0.copy()
        m = df_re["first_treat"] == 0
        df_re.loc[m, "first_treat"] = 1
        df_re.loc[m, "dose"] = eps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # tiny-boundary-gap warning
            res_ld = ContinuousDiD(control_group="lowest_dose", treatment_type="discrete").fit(
                df_re, **self._KW
            )
        # ATT and SE are EXACTLY equal (the differencing arithmetic is identical).
        assert np.allclose(
            res_ld.dose_response_att.effects, res_nt.dose_response_att.effects, atol=1e-10
        )
        assert np.allclose(res_ld.dose_response_att.se, res_nt.dose_response_att.se, atol=1e-10)
        assert res_ld.reference_dose == eps
        # Boundary ACRT references d_L=eps: ACRT(d_1) = ATT(d_1)/(d_1 - eps).
        att1 = res_ld.dose_response_att.effects[0]
        assert np.isclose(res_ld.dose_response_acrt.effects[0], att1 / (1.0 - eps), atol=1e-10)

    def test_hand_calc_att_acrt_overall(self):
        """ATT(d)=mean(dY|d)-mean(dY|d_L); ACRT backward-to-d_L; overall = mean_T."""
        all_levels = [1.0, 2.0, 4.0]
        df = _make_discrete_panel({1.0: 0.5, 2.0: 1.5, 4.0: 2.5}, n_control=0, seed=12)
        res = ContinuousDiD(control_group="lowest_dose", treatment_type="discrete").fit(
            df, **self._KW
        )
        modelled, att, acrt, overall, overall_acrt, _ = _hand_calc_lowest_dose(df, all_levels)
        assert np.allclose(res.dose_response_att.dose_grid, modelled)  # d_L excluded
        assert np.allclose(res.dose_response_att.effects, att, atol=1e-12)
        assert np.allclose(res.dose_response_acrt.effects, acrt, atol=1e-12)
        assert np.isclose(res.overall_att, overall, atol=1e-12)
        # overall_acrt (plug-in scalar) asserted explicitly, not just the rows.
        assert np.isclose(res.overall_acrt, overall_acrt, atol=1e-12)

    def test_hand_calc_analytical_se(self):
        """Lowest-dose SE == per-level 2x2 DiD SE with the d_L group as control."""
        all_levels = [1.0, 2.0, 4.0]
        df = _make_discrete_panel({1.0: 0.5, 2.0: 1.5, 4.0: 2.5}, n_control=0, seed=13)
        res = ContinuousDiD(control_group="lowest_dose", treatment_type="discrete").fit(
            df, **self._KW
        )
        _, _, _, _, _, se = _hand_calc_lowest_dose(df, all_levels)
        assert np.allclose(res.dose_response_att.se, se, atol=1e-10)
        assert np.all(np.isfinite(res.dose_response_att.se))

    def test_acrt_boundary_backward_to_dL(self):
        """ACRT(d_1) = ATT(d_1)/(d_1 - d_L); ACRT(d_j>=2) adjacent backward diffs."""
        df = _make_discrete_panel({1.0: 0.5, 2.0: 1.5, 4.0: 2.5}, n_control=0, seed=14)
        res = ContinuousDiD(control_group="lowest_dose", treatment_type="discrete").fit(
            df, **self._KW
        )
        att = res.dose_response_att.effects
        acrt = res.dose_response_acrt.effects
        modelled = res.dose_response_att.dose_grid  # [2, 4]; d_L = 1
        assert np.isclose(acrt[0], att[0] / (modelled[0] - 1.0), atol=1e-12)
        assert np.isclose(acrt[1], (att[1] - att[0]) / (modelled[1] - modelled[0]), atol=1e-12)

    def test_binary_above_reference(self):
        """Single modelled dose above d_L (J=1): ACRT(d_1) = ATT(d_1)/(d_1 - d_L)."""
        df = _make_discrete_panel({1.0: 0.4, 3.0: 2.0}, n_per_level=60, n_control=0, seed=15)
        res = ContinuousDiD(control_group="lowest_dose", treatment_type="discrete").fit(
            df, **self._KW
        )
        assert res.reference_dose == 1.0
        assert np.allclose(res.dose_response_att.dose_grid, [3.0])
        att1 = res.dose_response_att.effects[0]
        assert np.isclose(res.dose_response_acrt.effects[0], att1 / (3.0 - 1.0), atol=1e-12)

    def test_dgp_recovery_discrete(self):
        """Recover ATT(d) - ATT(d_L) with known effects (no noise -> exact)."""
        effects = {1.0: 0.5, 2.0: 2.0, 4.0: 1.0}  # non-monotone
        df = _make_discrete_panel(effects, n_per_level=60, n_control=0, noise=0.0, seed=16)
        res = ContinuousDiD(control_group="lowest_dose", treatment_type="discrete").fit(
            df, **self._KW
        )
        # d_L = 1 (effect 0.5); ATT(2)=2.0-0.5=1.5, ATT(4)=1.0-0.5=0.5.
        assert np.allclose(res.dose_response_att.effects, [1.5, 0.5], atol=1e-10)
        # ACRT: bwd@2 = 1.5/(2-1) = 1.5; bwd@4 = (0.5-1.5)/(4-2) = -0.5.
        assert np.allclose(res.dose_response_acrt.effects, [1.5, -0.5], atol=1e-10)

    def test_continuous_mass_point_recovery(self):
        """Continuous B-spline with a mass point at d_L recovers a linear slope."""
        rng = np.random.default_rng(17)
        rows, uid = [], 0
        beta = 0.7  # ATT(d) - ATT(d_L) = beta * (d - d_L)
        for d in [1.0] * 70 + list(rng.uniform(1.5, 5.0, 210)):
            base = rng.normal(0, 1)
            for p in (0, 1):
                y = base + 0.3 * p + (beta * (d - 1.0) if p >= 1 else 0.0)
                rows.append((uid, p, y, 1, d))
            uid += 1
        df = pd.DataFrame(rows, columns=["unit", "period", "outcome", "first_treat", "dose"])
        res = ContinuousDiD(control_group="lowest_dose", treatment_type="continuous", degree=1).fit(
            df, **self._KW
        )
        assert res.reference_dose == 1.0
        assert np.all(res.dose_response_att.dose_grid > 1.0)  # grid above d_L
        grid = res.dose_response_att.dose_grid
        expected = beta * (grid - 1.0)
        # B-spline + finite sample: modest tolerance on the recovered curve.
        assert np.allclose(res.dose_response_att.effects, expected, atol=0.15)

    def test_analytical_vs_bootstrap_se(self, ci_params):
        """Analytical and multiplier-bootstrap SE agree (both carry d_L variance)."""
        df = _make_discrete_panel({1.0: 0.5, 2.0: 1.5, 4.0: 2.5}, n_control=0, noise=1.0, seed=18)
        res_a = ContinuousDiD(control_group="lowest_dose", treatment_type="discrete").fit(
            df, **self._KW
        )
        nb = ci_params.bootstrap(999, min_n=199)
        res_b = ContinuousDiD(
            control_group="lowest_dose", treatment_type="discrete", n_bootstrap=nb, seed=7
        ).fit(df, **self._KW)
        thr = 0.40 if nb < 200 else 0.15
        rel = (
            np.abs(res_b.dose_response_att.se - res_a.dose_response_att.se)
            / res_a.dose_response_att.se
        )
        assert np.all(rel < thr), f"boot/analytic SE disagree: {rel}"

    def test_pre_period_placebo(self):
        """Pre-treatment cell att_glob ~ 0 (both treated and d_L control untreated)."""
        effects = {1.0: 0.5, 2.0: 1.5, 4.0: 2.5}
        df = _make_discrete_panel(
            effects, n_per_level=50, n_control=0, noise=0.0, seed=19, cohorts=(2,), n_periods=3
        )
        res = ContinuousDiD(
            control_group="lowest_dose", treatment_type="discrete", base_period="varying"
        ).fit(df, **self._KW)
        # Pre-period (t=1 < g=2) cell effects difference out to ~0.
        pre = [v for (g, t), v in res.group_time_effects.items() if t < g]
        assert pre, "expected a pre-period cell"
        assert all(abs(c["att_glob"]) < 1e-9 for c in pre)

    @pytest.mark.slow
    def test_coverage(self, ci_params):
        """Analytical & bootstrap SE achieve nominal coverage for ATT(d)-ATT(d_L)."""
        modelled = [2.0, 4.0]
        effects = {1.0: 0.5, 2.0: 1.5, 4.0: 2.5}
        truth = {d: effects[d] - effects[1.0] for d in modelled}  # ATT(d) - ATT(d_L)
        reps = 150
        for use_boot in (False, True):
            nb = ci_params.bootstrap(299, min_n=99) if use_boot else 0
            lo_band = 0.85 if (use_boot and nb < 200) else 0.88
            cover = np.zeros(len(modelled))
            for s in range(reps):
                df = _make_discrete_panel(
                    effects, n_per_level=45, n_control=0, noise=1.0, seed=40_000 + s
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ContinuousDiD(
                        control_group="lowest_dose",
                        treatment_type="discrete",
                        n_bootstrap=nb,
                        seed=s,
                    ).fit(df, **self._KW)
                lo = res.dose_response_att.conf_int_lower
                hi = res.dose_response_att.conf_int_upper
                for j, d in enumerate(modelled):
                    cover[j] += int(lo[j] <= truth[d] <= hi[j])
            rate = cover / reps
            assert np.all(
                (rate >= lo_band) & (rate <= 0.995)
            ), f"{'boot' if use_boot else 'analytic'} coverage {rate} off nominal"
