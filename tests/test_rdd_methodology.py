"""Methodology tests for RegressionDiscontinuity - internal-consistency
anchors that require no R, plus NaN/degenerate behavioral contracts."""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import RegressionDiscontinuity

from .conftest import assert_nan_inference


def _df(n=800, seed=42, jump=0.5):
    rng = np.random.default_rng(seed)
    x = 2 * rng.beta(2, 4, n) - 1
    y = 0.5 * x + 0.8 * x**2 + jump * (x >= 0) + rng.standard_normal(n) * 0.2
    return pd.DataFrame({"x": x, "y": y})


class TestCCT2014Remark7:
    """CCT 2014 Remark 7: with b = h and the same kernel, the bias-corrected
    order-p estimator is NUMERICALLY IDENTICAL to the (not bias-corrected)
    order-(p+1) estimator, and the robust variance equals the order-(p+1)
    conventional variance. The strongest R-free correctness anchor."""

    @pytest.mark.parametrize("kernel", ["triangular", "epanechnikov", "uniform"])
    def test_bc_equals_local_quadratic(self, kernel):
        df = _df()
        h0 = 0.35
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_p1 = RegressionDiscontinuity(p=1, q=2, h=h0, kernel=kernel).fit(
                df, "y", "x"
            )  # h alone -> b = h (rho = 1)
            fit_p2 = RegressionDiscontinuity(p=2, q=3, h=h0, kernel=kernel).fit(df, "y", "x")
        assert fit_p1.att == pytest.approx(fit_p2.att_conventional, rel=1e-10)
        assert fit_p1.se_robust == pytest.approx(fit_p2.se_conventional, rel=1e-10)

    def test_bc_at_p0_equals_local_linear(self):
        # Same identity one polynomial order down: the bias-corrected
        # local-CONSTANT fit (p=0, q=1, b=h) is numerically identical to the
        # conventional local-linear fit (p=1) - R-free anchor for the p=0
        # end of the public 0..20 order surface (golden-covered in
        # test_rdd_parity.py::p0q1).
        df = _df()
        h0 = 0.35
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_p0 = RegressionDiscontinuity(p=0, q=1, h=h0).fit(df, "y", "x")
            fit_p1 = RegressionDiscontinuity(p=1, q=2, h=h0).fit(df, "y", "x")
        assert fit_p0.att == pytest.approx(fit_p1.att_conventional, rel=1e-10)
        assert fit_p0.se_robust == pytest.approx(fit_p1.se_conventional, rel=1e-10)


class TestMasspoints:
    def test_adjust_equals_off_without_ties(self):
        df = _df(600, seed=1)  # continuous draws: no ties
        a = RegressionDiscontinuity(masspoints="adjust").fit(df, "y", "x")
        o = RegressionDiscontinuity(masspoints="off").fit(df, "y", "x")
        assert a.att == o.att
        assert a.se == o.se
        assert a.h_left == o.h_left

    def test_check_warns_on_ties(self):
        df = _df(600, seed=1)
        df["x"] = df["x"].round(2)
        with pytest.warns(UserWarning) as record:
            RegressionDiscontinuity(masspoints="check").fit(df, "y", "x")
        messages = [str(w.message) for w in record]
        assert any("Mass points detected" in m for m in messages)
        # R warns ONCE from rdrobust() itself (rdrobust.R:365-380; its
        # selection is inline, so rdbwselect's copy never stacks) - the
        # estimator must not emit a duplicate through the port.
        assert sum("Mass points detected" in m for m in messages) == 1

    def test_check_warns_on_ties_with_manual_h(self):
        # rdrobust.R:365-380 runs BEFORE the manual-bandwidth branch, so
        # the mass-point warning fires even when h= is supplied (verified
        # against installed R 4.0.0: manual h + masspoints="check" emits
        # both warning lines).
        df = _df(600, seed=1)
        df["x"] = df["x"].round(2)
        with pytest.warns(UserWarning) as record:
            RegressionDiscontinuity(h=0.5, masspoints="check").fit(df, "y", "x")
        messages = [str(w.message) for w in record]
        assert any("Mass points detected" in m for m in messages)
        assert any("masspoints='adjust'" in m for m in messages)

    def test_adjust_warns_on_ties_with_manual_h(self):
        # adjust warns too (without the "try adjust" hint); the default
        # masspoints="adjust" must not silently swallow the detection on
        # manual-bandwidth fits.
        df = _df(600, seed=1)
        df["x"] = df["x"].round(2)
        with pytest.warns(UserWarning, match="Mass points detected"):
            RegressionDiscontinuity(h=0.5, masspoints="adjust").fit(df, "y", "x")


class TestInvariances:
    def test_outcome_scale_equivariance(self):
        df = _df(500, seed=3)
        scaled = df.assign(y=df.y * 7.0)
        a = RegressionDiscontinuity().fit(df, "y", "x")
        b = RegressionDiscontinuity().fit(scaled, "y", "x")
        # Outcome scaling multiplies estimates and SEs; bandwidths are
        # selected on the standardized objective so h may shift slightly -
        # pin the manual-bandwidth case for exact equivariance.
        am = RegressionDiscontinuity(h=0.3).fit(df, "y", "x")
        bm = RegressionDiscontinuity(h=0.3).fit(scaled, "y", "x")
        assert bm.att == pytest.approx(7.0 * am.att, rel=1e-12)
        assert bm.se == pytest.approx(7.0 * am.se, rel=1e-12)
        assert np.isfinite(a.att) and np.isfinite(b.att)

    def test_running_var_translation_invariance(self):
        df = _df(500, seed=4)
        shifted = df.assign(x=df.x + 10.0)
        a = RegressionDiscontinuity(cutoff=0.0, h=0.3).fit(df, "y", "x")
        b = RegressionDiscontinuity(cutoff=10.0, h=0.3).fit(shifted, "y", "x")
        assert b.att == pytest.approx(a.att, rel=1e-9)
        assert b.se == pytest.approx(a.se, rel=1e-9)


class TestInferenceStructure:
    def test_robust_ci_wider_than_conventional_dgp_lock(self):
        # FIXTURE-SPECIFIC behavior lock, not a theorem: on this DGP the
        # robust CI is wider because se_rb adds bias-estimation
        # variability. The exact widths are golden-locked at rtol=1e-9 in
        # test_rdd_parity.py (se_cl/se_rb pins); this test only guards the
        # qualitative ordering against accidental row swaps.
        r = RegressionDiscontinuity().fit(_df(), "y", "x")
        width_rb = r.conf_int[1] - r.conf_int[0]
        width_cl = r.conf_int_conventional[1] - r.conf_int_conventional[0]
        assert width_rb > width_cl
        assert r.se_robust > r.se_conventional


class TestDegenerates:
    def test_constant_outcome_manual_h_nan_gated(self):
        # Constant y in the window: zero NN residuals -> se = 0 -> the
        # downstream inference triple gates to NaN (library contract);
        # the point estimate itself is a well-defined 0.
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.uniform(-1, 1, 100), "y": np.ones(100)})
        r = RegressionDiscontinuity(h=0.5).fit(df, "y", "x")
        assert r.att == pytest.approx(0.0, abs=1e-12)
        assert_nan_inference(
            {
                "se": r.se,
                "t_stat": r.t_stat,
                "p_value": r.p_value,
                "conf_int": r.conf_int,
            }
        )
        # All three rows gate independently:
        assert np.isnan(r.t_stat_conventional)
        assert np.isnan(r.p_value_bias_corrected)

    def test_constant_outcome_auto_bandwidth_fails_closed(self):
        # With data-driven selection, a constant outcome degenerates the
        # pilot MSE objective (0/0 -> NaN bandwidths); the estimation port
        # rejects non-finite bandwidths with a targeted error instead of
        # emitting NaN everywhere as R does (documented deviation).
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.uniform(-1, 1, 100), "y": np.ones(100)})
        with pytest.raises(ValueError, match="non-finite pilot bandwidth"):
            RegressionDiscontinuity().fit(df, "y", "x")

    def test_tiny_window_too_few_points_fails_closed(self):
        df = _df(100, seed=6)
        with pytest.raises(ValueError, match="Too few observations inside"):
            RegressionDiscontinuity(h=1e-6).fit(df, "y", "x")

    def test_valid_h_empty_b_window_fails_closed(self):
        # Valid main window but (near-)empty bias window: without the
        # per-window guard the zero W_b Gram would pinv-collapse Q_q back
        # to the conventional score and report finite "robust" inference.
        df = _df(200, seed=7)
        with pytest.raises(ValueError, match="bias bandwidth window"):
            RegressionDiscontinuity(h=0.5, b=1e-9).fit(df, "y", "x")

    def test_empty_h_window_valid_b_fails_closed(self):
        # Empty main window inside a valid bias window: the order-p fit is
        # unidentified even though the effective (wider) window is populated.
        df = _df(200, seed=7)
        with pytest.raises(ValueError, match="main bandwidth window"):
            RegressionDiscontinuity(h=1e-9, b=0.5).fit(df, "y", "x")

    def test_single_support_point_per_side_fails_closed(self):
        # Many rows but ONE distinct running-variable value per side: raw
        # counts pass any N-based check, yet the order-1 local fit is
        # unidentified (rank 1 < p+1 = 2).
        rng = np.random.default_rng(8)
        df = pd.DataFrame(
            {
                "x": np.repeat([-0.5, 0.5], 50).astype(float),
                "y": rng.normal(0.0, 1.0, 100),
            }
        )
        # The fit-level mass-point detection (rdrobust.R:365-380) fires
        # first - one distinct value per side is extreme mass - then the
        # identification guard raises.
        with pytest.warns(UserWarning, match="Mass points detected"):
            with pytest.raises(ValueError, match="distinct running-variable"):
                RegressionDiscontinuity(h=1.0).fit(df, "y", "x")

    def test_huge_rho_empty_selected_b_fails_closed(self):
        # rho without h applies b = h_selected/rho (rdrobust.R:501-504); a
        # huge rho collapses the selected bias window to empty support.
        df = _df(500, seed=9)
        with pytest.raises(ValueError, match="bias bandwidth window"):
            RegressionDiscontinuity(rho=1e12).fit(df, "y", "x")


def _fuzzy_df(n=1500, seed=9, lo=0.15, hi=0.75, effect=1.2):
    rng = np.random.default_rng(seed)
    x = 2 * rng.beta(2, 4, n) - 1
    t = (rng.uniform(size=n) < np.where(x >= 0, hi, lo)).astype(float)
    y = 0.5 * x + effect * t + rng.standard_normal(n) * 0.3
    return pd.DataFrame({"x": x, "y": y, "t": t})


class TestFuzzy:
    def test_perfect_compliance_reproduces_sharp_exactly(self):
        # T deterministic in the running variable: the first stage is
        # EXACTLY 1 (constant-0 left fit, constant-1 right fit), the ratio
        # collapses to the sharp estimate, and R selects the sharp
        # bandwidths via perf_comp - verified equal on installed 4.0.0.
        df = _fuzzy_df(800, seed=11)
        df["t"] = (df["x"] >= 0).astype(float)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            fz = RegressionDiscontinuity().fit(df, "y", "x", treatment_col="t")
        sharp = RegressionDiscontinuity().fit(df, "y", "x")
        # Bandwidths are BIT-identical (perf_comp nulls T, so selection is
        # the same sharp arithmetic); the estimates agree to the ULP - the
        # ratio divides by a first stage of 1 - O(eps) from the float
        # solve, in R exactly as here.
        assert fz.h_left == sharp.h_left and fz.b_left == sharp.b_left
        assert fz.att == pytest.approx(sharp.att, rel=1e-12)
        assert fz.se == pytest.approx(sharp.se, rel=1e-12)
        assert fz.first_stage == pytest.approx(1.0, rel=1e-12)
        assert fz.first_stage_conventional == pytest.approx(1.0, rel=1e-12)
        # Zero first-stage NN residuals -> se_T = 0 -> the first-stage
        # inference triple NaN-gates (documented deviation from R's
        # z_T = Inf, pv_T = 0)...
        assert fz.first_stage_se == 0.0
        assert_nan_inference(
            {
                "se": np.nan,  # gate is on the triple below, se itself is 0
                "t_stat": fz.first_stage_t_stat,
                "p_value": fz.first_stage_p_value,
                "conf_int": fz.first_stage_conf_int,
            }
        )
        # ...and the weak-first-stage warning must NOT fire (a perfect
        # first stage is the opposite of weak; the finite-CI gate holds).
        assert not any("Weak first stage" in str(w.message) for w in rec)

    def test_one_sided_compliance_selects_sharp_bandwidths(self):
        # var(T) == 0 on one side -> perf_comp -> bandwidth selection runs
        # on the sharp reduced-form objective (rdbwselect.R:334-346);
        # estimation stays fuzzy.
        df = _fuzzy_df(1200, seed=12)
        df.loc[df["x"] < 0, "t"] = 0.0
        fz = RegressionDiscontinuity().fit(df, "y", "x", treatment_col="t")
        sharp = RegressionDiscontinuity().fit(df, "y", "x")
        assert fz.h_left == sharp.h_left and fz.b_right == sharp.b_right
        assert fz.first_stage is not None and fz.first_stage > 0
        assert fz.att != sharp.att  # estimation is still the fuzzy ratio

    def test_sharpbw_true_selects_sharp_bandwidths(self):
        df = _fuzzy_df(1200, seed=14)
        fz_sbw = RegressionDiscontinuity(sharpbw=True).fit(df, "y", "x", treatment_col="t")
        fz_def = RegressionDiscontinuity().fit(df, "y", "x", treatment_col="t")
        sharp = RegressionDiscontinuity().fit(df, "y", "x")
        assert fz_sbw.h_left == sharp.h_left
        assert fz_sbw.h_left != fz_def.h_left  # fuzzy objective differs

    def test_outcome_scaling_scales_ratio_not_first_stage(self):
        df = _fuzzy_df(900, seed=15)
        scaled = df.assign(y=df.y * 7.0)
        a = RegressionDiscontinuity(h=0.3).fit(df, "y", "x", treatment_col="t")
        b = RegressionDiscontinuity(h=0.3).fit(scaled, "y", "x", treatment_col="t")
        assert b.att == pytest.approx(7.0 * a.att, rel=1e-12)
        assert b.se == pytest.approx(7.0 * a.se, rel=1e-12)
        # The take-up fits never see y: bit-identical first stage.
        assert b.first_stage == a.first_stage
        assert b.first_stage_se == a.first_stage_se

    def test_no_variation_no_jump_raises(self):
        # R-exact message; and the identification stop is hoisted BEFORE
        # mass-point detection (rdrobust.R:175 precedes :365-380), so no
        # mass-point warning precedes the raise even on tied data.
        df = _fuzzy_df(600, seed=16)
        df["x"] = df["x"].round(2)  # heavy ties
        df["t"] = 0.7
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="no variation and no jump"):
                RegressionDiscontinuity(masspoints="check").fit(df, "y", "x", treatment_col="t")
        assert not any("Mass points detected" in str(w.message) for w in rec)

    def test_weak_first_stage_warns(self):
        # Take-up independent of the running variable: the jump is not
        # distinguishable from zero and R is verified SILENT - we warn
        # (documented deviation; CCT 2014 Theorem 3 / FLM weak-ID).
        rng = np.random.default_rng(17)
        df = _fuzzy_df(800, seed=17)
        df["t"] = (rng.uniform(size=800) < 0.4).astype(float)
        with pytest.warns(UserWarning, match="Weak first stage"):
            RegressionDiscontinuity().fit(df, "y", "x", treatment_col="t")

    def test_strong_first_stage_no_warning(self):
        df = _fuzzy_df(1500, seed=18)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            RegressionDiscontinuity().fit(df, "y", "x", treatment_col="t")
        assert not any("Weak first stage" in str(w.message) for w in rec)

    def test_degenerate_pilot_first_stage_fails_closed(self):
        # Take-up varies only far from the cutoff (identification passes,
        # no perf_comp) but pilot windows hold zero take-up variation ->
        # pilot tau_T == 0 -> R flows Inf/NaN; the port fails closed on
        # the non-finite pilot bandwidth (documented deviation).
        rng = np.random.default_rng(21)
        n = 1000
        x = rng.uniform(-1, 1, n)
        t = np.where(np.abs(x) > 0.8, (rng.uniform(size=n) < 0.5).astype(float), 0.0)
        y = 0.4 * x + 0.9 * t + rng.standard_normal(n) * 0.2
        df = pd.DataFrame({"x": x, "y": y, "t": t})
        with pytest.raises(ValueError, match="non-finite pilot bandwidth"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RegressionDiscontinuity().fit(df, "y", "x", treatment_col="t")

    def test_manual_h_zero_first_stage_nan_gates(self):
        # Same construction with a manual h: estimation-time tau_T == 0
        # follows R's Inf/NaN flow-on and the main rows joint-NaN gate
        # (loud, not silent); the first stage itself is exactly 0 with a
        # zero SE (NaN-gated triple), so the weak-ID warning cannot fire.
        rng = np.random.default_rng(21)
        n = 1000
        x = rng.uniform(-1, 1, n)
        t = np.where(np.abs(x) > 0.8, (rng.uniform(size=n) < 0.5).astype(float), 0.0)
        y = 0.4 * x + 0.9 * t + rng.standard_normal(n) * 0.2
        df = pd.DataFrame({"x": x, "y": y, "t": t})
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            r = RegressionDiscontinuity(h=0.3).fit(df, "y", "x", treatment_col="t")
        assert np.isnan(r.att)
        assert_nan_inference(
            {"se": r.se, "t_stat": r.t_stat, "p_value": r.p_value, "conf_int": r.conf_int}
        )
        assert r.first_stage == 0.0
        assert not any("Weak first stage" in str(w.message) for w in rec)

    def test_n_below_20_fuzzy_estimates_fuzzy(self):
        # The full-range fallback still produces a FUZZY fit (first-stage
        # fields populated, resolved label "Manual").
        rng = np.random.default_rng(19)
        x = np.linspace(-1, 1, 15)
        t = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1], dtype=float)
        y = 0.5 * x + 1.0 * t + rng.standard_normal(15) * 0.1
        df = pd.DataFrame({"x": x, "y": y, "t": t})
        with pytest.warns(UserWarning, match="entire sample"):
            r = RegressionDiscontinuity().fit(df, "y", "x", treatment_col="t")
        assert r.bwselect == "Manual"
        assert r.first_stage is not None
        assert r.estimand.startswith("fuzzy")

    def test_manual_h_sharpbw_is_silent_noop(self):
        # Manual bandwidths skip selection entirely, so sharpbw is inert
        # on fuzzy fits - and silently so, matching R (the sharp-fit
        # warning is only for treatment_col=None).
        df = _fuzzy_df(900, seed=20)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            a = RegressionDiscontinuity(h=0.3, sharpbw=True).fit(df, "y", "x", treatment_col="t")
        b = RegressionDiscontinuity(h=0.3).fit(df, "y", "x", treatment_col="t")
        assert a.att == b.att and a.se == b.se
        assert not any("sharpbw" in str(w.message) for w in rec)

    def test_constant_outcome_fuzzy_manual_h(self):
        # Constant y: unlike the sharp case (exact zero residuals -> se = 0
        # -> NaN gate), the fuzzy ratio's delta vector carries a
        # tiny-but-nonzero -tau_Y/tau_T^2 component from float solve
        # roundoff, so the main rows report an O(eps) estimate with an
        # O(eps) SE - the semantically correct "no effect" conclusion, not
        # a gate (R's arithmetic behaves identically). The FIRST-STAGE
        # rows are unaffected by y and stay properly finite.
        rng = np.random.default_rng(22)
        n = 400
        x = rng.uniform(-1, 1, n)
        t = (rng.uniform(size=n) < np.where(x >= 0, 0.8, 0.2)).astype(float)
        df = pd.DataFrame({"x": x, "y": np.ones(n), "t": t})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = RegressionDiscontinuity(h=0.5).fit(df, "y", "x", treatment_col="t")
        assert r.att == pytest.approx(0.0, abs=1e-12)
        assert abs(r.se) < 1e-12  # O(eps) scale, may not be exactly zero
        assert np.isfinite(r.first_stage) and np.isfinite(r.first_stage_se)
        assert r.first_stage_se > 0
        assert np.isfinite(r.first_stage_t_stat)
