"""
Equation verification and R benchmark tests for ContinuousDiD.

Phase 1: Hand-calculable cases verifying the estimator recovers known truths.
Phase 2: R `contdid` benchmarks (skipped if R not installed).
"""

import json
import os
import subprocess
import tempfile

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
        results = est.fit(
            linear_data, "outcome", "unit", "period", "first_treat", "dose"
        )
        expected = np.array([2.0, 6.0, 10.0])
        np.testing.assert_allclose(results.dose_response_att.effects, expected, atol=1e-10)

    def test_linear_acrt(self, linear_data):
        """ACRT(d) should be constant = 2 for linear truth."""
        est = ContinuousDiD(degree=1, num_knots=0, dvals=np.array([1.5, 3.0, 4.5]))
        results = est.fit(
            linear_data, "outcome", "unit", "period", "first_treat", "dose"
        )
        # Derivative of 2d is 2
        np.testing.assert_allclose(results.dose_response_acrt.effects, 2.0, atol=1e-6)

    def test_att_glob_binarized(self, linear_data):
        """ATT_glob = mean(Delta_Y | treated) - mean(Delta_Y | control)."""
        treated_doses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mean_delta_treated = np.mean(2 * treated_doses)  # = 7.0
        mean_delta_control = 0.0
        expected_att_glob = mean_delta_treated - mean_delta_control

        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(
            linear_data, "outcome", "unit", "period", "first_treat", "dose"
        )
        np.testing.assert_allclose(results.overall_att, expected_att_glob, atol=1e-10)

    def test_acrt_glob_plugin(self, linear_data):
        """ACRT_glob = mean(ACRT(D_i)) over treated = 2."""
        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(
            linear_data, "outcome", "unit", "period", "first_treat", "dose"
        )
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
        results = est.fit(
            quadratic_data, "outcome", "unit", "period", "first_treat", "dose"
        )
        expected = eval_grid**2
        np.testing.assert_allclose(
            results.dose_response_att.effects, expected, atol=1e-6
        )


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
        results = est.fit(
            staggered_data, "outcome", "unit", "period", "first_treat", "dose"
        )
        assert len(results.groups) == 2
        assert 2 in results.groups
        assert 3 in results.groups

    def test_gt_cell_count(self, staggered_data):
        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(
            staggered_data, "outcome", "unit", "period", "first_treat", "dose"
        )
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
            rows.append({"unit": uid, "period": 1, "outcome": 0.0, "first_treat": 2, "dose": dose_val})
            rows.append({"unit": uid, "period": 2, "outcome": 5.0, "first_treat": 2, "dose": dose_val})

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
            rows_hetero.append({"unit": i, "period": 1, "outcome": y_pre, "first_treat": 0, "dose": 0.0})
            rows_hetero.append({"unit": i, "period": 2, "outcome": y_post, "first_treat": 0, "dose": 0.0})
        for j in range(n_treated):
            uid = n_control + j
            y_pre = rng.normal(0, 0.3)
            rows_hetero.append({"unit": uid, "period": 1, "outcome": y_pre, "first_treat": 2, "dose": dose_val})
            rows_hetero.append({"unit": uid, "period": 2, "outcome": y_pre + 5.0, "first_treat": 2, "dose": dose_val})
        data_hetero = pd.DataFrame(rows_hetero)
        est_boot = ContinuousDiD(
            degree=1, num_knots=0, n_bootstrap=199,
            rank_deficient_action="silent", seed=42,
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
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() == "OK"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_HAS_R_CONTDID = _check_r_contdid()

require_contdid = pytest.mark.skipif(
    not _HAS_R_CONTDID,
    reason="R or contdid package not installed",
)


def _run_r_contdid(csv_path, degree=3, num_knots=0, control_group="nevertreated",
                    aggregation="dose", staggered=False):
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
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"R contdid failed: {result.stderr[:500]}")
    return json.loads(result.stdout)


@require_contdid
class TestRBenchmark:
    """R `contdid` v0.1.0 benchmark comparisons."""

    def _compare_with_r(self, data, degree=3, num_knots=0,
                        control_group="never_treated", aggregation="dose",
                        staggered=False, att_tol=0.01, acrt_tol=0.02):
        """Helper: run both Python and R, compare."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            data.to_csv(f, index=False)
            csv_path = f.name

        r_out = _run_r_contdid(
            csv_path, degree=degree, num_knots=num_knots,
            control_group=control_group, aggregation=aggregation,
            staggered=staggered,
        )

        # Map R aggregation names to Python aggregate parameter
        py_aggregate = None
        if aggregation == "dose":
            py_aggregate = "dose"
        elif aggregation == "eventstudy":
            py_aggregate = "eventstudy"

        # Python estimation using R's dvals for exact grid match
        dvals = np.array(r_out["dvals"])
        est = ContinuousDiD(
            degree=degree, num_knots=num_knots, dvals=dvals,
            control_group=control_group,
        )
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose",
            aggregate=py_aggregate,
        )

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
            n_units=300, n_periods=2, cohort_periods=[2],
            seed=100, noise_sd=0.5,
        )
        self._compare_with_r(data, degree=3, num_knots=0)

    def test_benchmark_2_linear(self):
        """2 periods, 1 cohort, degree=1 (linear), never_treated."""
        data = generate_continuous_did_data(
            n_units=300, n_periods=2, cohort_periods=[2],
            seed=101, noise_sd=0.5,
        )
        self._compare_with_r(data, degree=1, num_knots=0)

    def test_benchmark_3_interior_knots(self):
        """2 periods, 1 cohort, degree=3, 2 interior knots."""
        data = generate_continuous_did_data(
            n_units=300, n_periods=2, cohort_periods=[2],
            seed=102, noise_sd=0.5,
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
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                pytest.skip(f"R contdid failed: {result.stderr[:500]}")
            r_out = json.loads(result.stdout)

            data = pd.read_csv(tmp_path)
            data = data.rename(columns={
                "id": "unit", "time_period": "period",
                "Y": "outcome", "G": "first_treat", "D": "dose",
            })
            dvals = np.array(r_out["dvals"])
            est = ContinuousDiD(
                degree=3, num_knots=0, dvals=dvals,
                control_group="never_treated",
            )
            results = est.fit(
                data, "outcome", "unit", "period", "first_treat", "dose",
                aggregate="dose",
            )

            # Overall ATT
            att_diff = abs(results.overall_att - r_out["overall_att"]) / (abs(r_out["overall_att"]) + 1e-10)
            assert att_diff < 0.01, (
                f"Overall ATT diff: {att_diff:.4f} "
                f"(R={r_out['overall_att']:.6f}, Py={results.overall_att:.6f})"
            )

            # Overall ACRT
            acrt_diff = abs(results.overall_acrt - r_out["overall_acrt"]) / (abs(r_out["overall_acrt"]) + 1e-10)
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
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                pytest.skip(f"R contdid failed: {result.stderr[:500]}")
            r_out = json.loads(result.stdout)

            data = pd.read_csv(tmp_path)
            data = data.rename(columns={
                "id": "unit", "time_period": "period",
                "Y": "outcome", "G": "first_treat", "D": "dose",
            })
            dvals = np.array(r_out["dvals"])
            est = ContinuousDiD(
                degree=3, num_knots=0, dvals=dvals,
                control_group="not_yet_treated",
            )
            results = est.fit(
                data, "outcome", "unit", "period", "first_treat", "dose",
                aggregate="dose",
            )

            att_diff = abs(results.overall_att - r_out["overall_att"]) / (abs(r_out["overall_att"]) + 1e-10)
            assert att_diff < 0.01, (
                f"Overall ATT diff: {att_diff:.4f} "
                f"(R={r_out['overall_att']:.6f}, Py={results.overall_att:.6f})"
            )

            acrt_diff = abs(results.overall_acrt - r_out["overall_acrt"]) / (abs(r_out["overall_acrt"]) + 1e-10)
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
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                pytest.skip(f"R contdid failed: {result.stderr[:500]}")
            r_out = json.loads(result.stdout)

            data = pd.read_csv(tmp_path)
            data = data.rename(columns={
                "id": "unit", "time_period": "period",
                "Y": "outcome", "G": "first_treat", "D": "dose",
            })
            est = ContinuousDiD(
                degree=3, num_knots=0,
                control_group="never_treated",
            )
            results = est.fit(
                data, "outcome", "unit", "period", "first_treat", "dose",
                aggregate="eventstudy",
            )

            # Compare overall ATT (binarized)
            att_diff = abs(results.overall_att - r_out["overall_att"]) / (abs(r_out["overall_att"]) + 1e-10)
            assert att_diff < 0.01, (
                f"Overall ATT diff: {att_diff:.4f} "
                f"(R={r_out['overall_att']:.6f}, Py={results.overall_att:.6f})"
            )
        finally:
            os.unlink(tmp_path)
