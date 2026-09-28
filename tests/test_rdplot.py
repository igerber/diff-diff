"""RDPlot tests: golden parity vs R rdrobust 4.0.0 rdplot(), R-quirk locks,
documented deviations, and API contracts.

Golden fixtures come from benchmarks/R/generate_rdplot_golden.R
(benchmarks/data/rdplot_golden.json); tests skip if the JSON is absent.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from diff_diff import RDPlot, RDPlotResult

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "rdplot_golden.json"
SENATE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "rdrobust_senate.csv"

RTOL = 1e-9

VARS_BINS_COLS = {
    "mean_bin": "rdplot_mean_bin",
    "mean_x": "rdplot_mean_x",
    "mean_y": "rdplot_mean_y",
    "min_bin": "rdplot_min_bin",
    "max_bin": "rdplot_max_bin",
    "se_y": "rdplot_se_y",
    "N": "rdplot_N",
    "ci_l": "rdplot_ci_l",
    "ci_r": "rdplot_ci_r",
}


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(
            "rdplot golden file not generated; run " "Rscript benchmarks/R/generate_rdplot_golden.R"
        )
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _senate_frame():
    if not SENATE_PATH.exists():
        pytest.skip(f"Vendored Senate CSV not found at {SENATE_PATH}")
    return pd.read_csv(SENATE_PATH)


def _frame(golden, dgp_name):
    if dgp_name == "senate":
        return _senate_frame().rename(columns={"vote": "y", "margin": "x"})
    entry = golden[dgp_name]
    cols = {"x": entry["x"], "y": entry["y"]}
    for name in ("zlong", "zb", "zdup"):
        if name in entry:
            cols[name] = entry[name]
    return pd.DataFrame(cols)


def _maybe_pair(value):
    if value is None:
        return None
    if isinstance(value, list):
        return tuple(value)
    return value


def _num_array(values):
    """jsonlite writes R NA values as the string "NA"; map them to NaN."""
    return np.array([np.nan if v is None or v == "NA" else float(v) for v in values], dtype=float)


def _build(cfg):
    a = cfg["args"]
    return RDPlot(
        cutoff=a["c"],
        p=a["p"],
        nbins=_maybe_pair(a["nbins"]),
        binselect=a["binselect"],
        scale=_maybe_pair(a["scale"]),
        kernel=a["kernel"],
        h=_maybe_pair(a["h"]),
        support=_maybe_pair(a["support"]),
        masspoints=a["masspoints"],
        ci=a["ci"],
        covs_drop=a["covs_drop"],
    )


def _fit(golden, dgp_name, cfg):
    df = _frame(golden, dgp_name)
    covariates = None
    if cfg["args"]["covs_in"]:
        names = cfg["args"]["covs_names"]
        covariates = [names] if isinstance(names, str) else list(names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _build(cfg).fit(df, "y", "x", covariates=covariates)


class TestRDPlotGoldenParity:
    """End-to-end parity against R rdplot() on every stored config."""

    def test_all_configs(self, golden):
        n_checked = 0
        for dgp_name, entry in golden.items():
            if dgp_name == "metadata":
                continue
            for cfg_name, cfg in entry["configs"].items():
                r = _fit(golden, dgp_name, cfg)
                label = f"{dgp_name}/{cfg_name}"
                # Integer/string surfaces (asserted for EVERY config,
                # including the BLAS-sensitive overflow-ladder DGP).
                assert_allclose(r.J, cfg["J"], rtol=0, atol=0, err_msg=f"{label}: J")
                assert_allclose(r.J_IMSE, cfg["J_IMSE"], rtol=0, atol=0, err_msg=f"{label}: J_IMSE")
                assert_allclose(r.J_MV, cfg["J_MV"], rtol=0, atol=0, err_msg=f"{label}: J_MV")
                assert r.N == tuple(cfg["N"]), f"{label}: N"
                assert r.N_h == tuple(cfg["N_h"]), f"{label}: N_h"
                assert r.binselect == cfg["binselect_type"], f"{label}: binselect_type"
                assert r.kernel_type == cfg["kernel_type"], f"{label}: kernel_type"
                assert len(r.vars_bins) == len(cfg["vars_bins"]["mean_y"]), f"{label}: rows"
                if dgp_name == "dgp_ladder":
                    # Float surfaces of the post-ladder k=3 fit on |x|~1e39
                    # data are dominated by catastrophic cancellation and
                    # are not comparable across BLAS builds; the integer
                    # outputs above pin the ladder behavior.
                    n_checked += 1
                    continue
                assert_allclose(r.scale, cfg["scale_out"], rtol=RTOL, err_msg=f"{label}: scale")
                assert_allclose(r.rscale, cfg["rscale"], rtol=RTOL, err_msg=f"{label}: rscale")
                assert_allclose(r.bin_avg, cfg["bin_avg"], rtol=RTOL, err_msg=f"{label}: bin_avg")
                assert_allclose(r.bin_med, cfg["bin_med"], rtol=RTOL, err_msg=f"{label}: bin_med")
                assert_allclose(r.h, cfg["h_out"], rtol=RTOL, err_msg=f"{label}: h")
                # atol floors the high-order coefficients: on x in [-100, 100]
                # the x^4 coefficient is O(1e-9) and BLAS accumulation order
                # wobbles it at O(1e-18) absolute (assert-allclose convention).
                assert_allclose(
                    r.coef["Left"],
                    cfg["coef_left"],
                    rtol=RTOL,
                    atol=1e-14,
                    err_msg=f"{label}: coef L",
                )
                assert_allclose(
                    r.coef["Right"],
                    cfg["coef_right"],
                    rtol=RTOL,
                    atol=1e-14,
                    err_msg=f"{label}: coef R",
                )
                for key, col in VARS_BINS_COLS.items():
                    got = np.asarray(r.vars_bins[col], dtype=float)
                    want = _num_array(cfg["vars_bins"][key])
                    assert_allclose(
                        got,
                        want,
                        rtol=RTOL,
                        atol=1e-12,
                        equal_nan=True,
                        err_msg=f"{label}: vars_bins.{col}",
                    )
                idx = np.asarray(cfg["vars_poly_idx"], dtype=int) - 1  # R 1-based
                assert_allclose(
                    r.vars_poly["rdplot_x"].to_numpy()[idx],
                    cfg["vars_poly_x"],
                    rtol=RTOL,
                    err_msg=f"{label}: vars_poly x",
                )
                # atol floors the curve where it CROSSES ZERO: the fitted
                # values wobble at O(1e-12) absolute across BLAS builds
                # (Accelerate vs OpenBLAS), which exceeds rtol=1e-9 only at
                # near-zero crossings (observed: dgp_small, 1 of 42 points,
                # 2.3e-12 abs = 1.03e-9 rel on the Linux pure-python leg).
                assert_allclose(
                    r.vars_poly["rdplot_y"].to_numpy()[idx],
                    cfg["vars_poly_y"],
                    rtol=RTOL,
                    atol=1e-10,
                    err_msg=f"{label}: vars_poly y",
                )
                if cfg["args"]["covs_in"]:
                    assert r.covariate_coefficients is not None, f"{label}: coef_covs"
                    assert_allclose(
                        list(r.covariate_coefficients.values()),
                        cfg["coef_covs"],
                        rtol=RTOL,
                        err_msg=f"{label}: coef_covs",
                    )
                n_checked += 1
        # 24 configurations; fail loudly if the golden shrinks.
        assert n_checked == 24

    def test_covariate_names_nchar_sorted(self, golden):
        # R sorts covariate columns by name length (order(nchar)); the
        # coefficient dict must be keyed in that order: zb before zlong.
        cfg = golden["dgp_covs"]["configs"]["covs_default"]
        r = _fit(golden, "dgp_covs", cfg)
        assert r.covariate_coefficients is not None
        assert list(r.covariate_coefficients.keys()) == ["zb", "zlong"]

    def test_collinear_covariate_dropped_with_name(self, golden):
        # The nchar sort decides WHICH of a collinear set survives: sorted
        # order is (zb, zdup, zlong), the stable QR keeps the first
        # independent columns {zb, zdup}, so zlong - although it GENERATED
        # zdup - is the one dropped (R-identical; the covs_collinear golden
        # pins the (zb, zdup) gamma).
        df = _frame(golden, "dgp_covs")
        with pytest.warns(UserWarning, match="Multicollinearity"):
            r = RDPlot().fit(df, "y", "x", covariates=["zlong", "zb", "zdup"])
        assert r.covariates_dropped == ["zlong"]
        assert r.covariate_coefficients is not None
        assert list(r.covariate_coefficients.keys()) == ["zb", "zdup"]


class TestPaperAnchors:
    """CCT 2015 supplement Figures SA-1/SA-2 report the selector outputs on
    exactly this Senate dataset; assert them independently of the JSON."""

    @pytest.mark.parametrize(
        "binselect, expected_J",
        [("esmv", (15, 35)), ("es", (8, 9)), ("qs", (21, 16)), ("qsmv", (28, 49))],
    )
    def test_senate_anchor(self, binselect, expected_J):
        df = _senate_frame()
        r = RDPlot(binselect=binselect).fit(df, "vote", "margin")
        assert r.J == tuple(float(v) for v in expected_J)
        if binselect in ("esmv", "es"):
            assert r.J_IMSE == (8.0, 9.0)

    def test_senate_wimse_weights_inverse_map(self):
        # Supplement S.1 inverse map on the default esmv fit: rscale is
        # exactly (15/8, 35/9) via the paper-anchored J values, and the
        # implied WIMSE weights are (1/(1+rscale^3), rscale^3/(1+rscale^3)),
        # summing to 1. Explicit values guard the map itself, not just the
        # identity.
        r = RDPlot().fit(_senate_frame(), "vote", "margin")
        assert r.rscale == pytest.approx((15 / 8, 35 / 9))
        assert r.wimse_variance_weight[0] == pytest.approx(1 / (1 + (15 / 8) ** 3))
        assert r.wimse_variance_weight[1] == pytest.approx(1 / (1 + (35 / 9) ** 3))
        for side in (0, 1):
            assert (r.wimse_variance_weight[side] + r.wimse_bias_weight[side]) == pytest.approx(1.0)


class TestRQuirkLocks:
    def test_empty_bins_dropped_but_lengths_cover_all(self):
        # Senate default: J = 15 + 35 = 50 bins but one left bin is empty ->
        # 49 vars_bins rows; counts still add up to the side samples.
        r = RDPlot().fit(_senate_frame(), "vote", "margin")
        assert r.J == (15.0, 35.0)
        assert len(r.vars_bins) == 49
        assert int(np.asarray(r.vars_bins["rdplot_N"]).sum()) == r.N[0] + r.N[1]

    def test_single_obs_bin_se_zero_ci_collapses(self):
        r = RDPlot().fit(_senate_frame(), "vote", "margin")
        singles = r.vars_bins[r.vars_bins["rdplot_N"] == 1]
        assert len(singles) > 0
        assert (singles["rdplot_se_y"] == 0).all()
        assert_allclose(singles["rdplot_ci_l"], singles["rdplot_mean_y"], rtol=0)
        assert_allclose(singles["rdplot_ci_r"], singles["rdplot_mean_y"], rtol=0)

    def test_qs_nbins_keeps_manual_label_but_quantile_bins(self, golden):
        # rdplot.R:452-456 sets the label "manually evenly spaced" even when
        # binselect chose quantile spacing; the bins themselves stay
        # quantile-spaced. Label AND bin edges both locked.
        cfg = golden["senate"]["configs"]["qs_nbins"]
        r = _fit(golden, "senate", cfg)
        assert r.binselect == "manually evenly spaced"
        df = _senate_frame()
        ok = df["vote"].notna() & df["margin"].notna()  # rdplot's complete-case filter
        x_l = np.sort(df.loc[ok & (df["margin"] < 0), "margin"].to_numpy())
        # hand-computed TYPE-7 quantile at prob 1/8 (R's default; numpy's
        # default linear interpolation) - locks the convention independently
        # of numpy's quantile implementation: position h = (n-1)p, linear
        # interpolation between the flanking order statistics.
        h_pos = (x_l.size - 1) * (1 / 8)
        lo = int(np.floor(h_pos))
        expected_first_edge = x_l[lo] + (h_pos - lo) * (x_l[lo + 1] - x_l[lo])
        assert r.vars_bins["rdplot_max_bin"].iloc[0] == pytest.approx(
            expected_first_edge, abs=1e-12
        )

    def test_masspoints_adjust_equals_explicit_pr_variant(self, golden):
        # adjust remaps esmv -> esmvpr; fitting esmvpr with detection off
        # must give the identical result (and no remap warning).
        df = _frame(golden, "dgp_ties")
        with pytest.warns(UserWarning, match="Mass points"):
            r_adjust = RDPlot(masspoints="adjust").fit(df, "y", "x")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r_explicit = RDPlot(binselect="esmvpr", masspoints="off").fit(df, "y", "x")
        assert r_adjust.J == r_explicit.J
        assert r_adjust.binselect == r_explicit.binselect
        pd.testing.assert_frame_equal(r_adjust.vars_bins, r_explicit.vars_bins)

    def test_masspoints_check_warns_without_remap(self, golden):
        df = _frame(golden, "dgp_ties")
        with pytest.warns(UserWarning, match="Mass points"):
            r_check = RDPlot(masspoints="check").fit(df, "y", "x")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r_off = RDPlot(masspoints="off").fit(df, "y", "x")
        # check mode detects but does NOT remap: same spacings selectors.
        assert r_check.J == r_off.J
        assert r_check.binselect == r_off.binselect

    def test_zero_variance_side_pins_J_to_one_with_inf_imse(self):
        rng = np.random.default_rng(7)
        x = np.concatenate([-rng.uniform(0.1, 1, 15), rng.uniform(0.1, 1, 15)])
        y = np.where(x < 0, 3.0, rng.normal(size=30))
        df = pd.DataFrame({"y": y, "x": x})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = RDPlot().fit(df, "y", "x")
        messages = [str(w.message) for w in caught]
        # the constant side fires BOTH the variability rescue and the
        # outcome-ties (discrete outcome) defensive warning
        assert any("variability" in m and "below" in m for m in messages)
        assert any("Heavy ties" in m for m in messages)
        assert r.J[0] == 1.0
        assert np.isinf(r.J_IMSE[0])  # R propagates Inf through J.fun
        assert r.rscale[0] == 0.0
        assert np.isfinite(r.J[1]) and r.J[1] >= 1

    def test_discrete_outcome_warns_under_spacings_selector(self):
        # Spacings selectors assume a continuously distributed outcome
        # (CCT 2015 Theorems 3-4); a binary y warns and recommends the *pr
        # sibling (R is silent - documented defensive warning, no remap).
        rng = np.random.default_rng(21)
        x = np.concatenate([-rng.uniform(0.1, 1, 40), rng.uniform(0.1, 1, 40)])
        y = (rng.uniform(size=80) < 0.4).astype(float)
        df = pd.DataFrame({"y": y, "x": x})
        with pytest.warns(UserWarning, match="esmvpr"):
            r = RDPlot().fit(df, "y", "x")
        # warn-only: the spacings selectors still ran
        assert r.binselect == "mimicking variance evenly-spaced method using spacings estimators"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RDPlot(binselect="esmvpr").fit(df, "y", "x")  # pr variant: silent

    def test_manual_h_with_zero_effective_obs_raises(self):
        rng = np.random.default_rng(22)
        x = np.concatenate([-rng.uniform(0.5, 1, 15), rng.uniform(0.5, 1, 15)])
        df = pd.DataFrame({"y": rng.normal(size=30), "x": x})
        # h below the nearest observation's distance on both sides: R would
        # silently fit a zero-weight design; we raise naming the side.
        with pytest.raises(ValueError, match="zero effective observations"):
            RDPlot(h=0.1).fit(df, "y", "x")

    def test_constant_float_outcome_side_rescued_exactly(self):
        # np.var of a constant 0.7 vector is ~1.3e-32, NOT 0 (single-pass
        # mean roundoff), while R's two-pass var() is exactly 0 - without
        # the _var0 exact-constancy routing the var==0 rescue is skipped
        # and the zero jump crashes bin construction (ZeroDivisionError;
        # caught by CI codex on the rebased head). R fires the rescue:
        # J=1 with the variability warning, Inf J_IMSE, NaN J_MV.
        rng = np.random.default_rng(7)
        x = np.concatenate([-rng.uniform(0.1, 1, 15), rng.uniform(0.1, 1, 15)])
        y = np.where(x < 0, 0.7, rng.normal(size=30))
        df = pd.DataFrame({"y": y, "x": x})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = RDPlot().fit(df, "y", "x")
        messages = [str(w.message) for w in caught]
        assert any("variability" in m and "below" in m for m in messages)
        assert r.J[0] == 1.0
        assert np.isinf(r.J_IMSE[0])
        assert np.isnan(r.J_MV[0])  # 0/0 through the MV selector, as in R
        assert np.isfinite(r.J[1]) and r.J[1] >= 1

    def test_finite_singular_selector_gram_does_not_downgrade(self):
        # Companion to the overflow-ladder golden: a finite rank-deficient
        # k=4 Gram (4 distinct x values per side) is absorbed inside
        # qrXXinv's pinv fallback at UNCHANGED k, mirroring R's internal
        # ginv fallback - the fit must succeed, not ladder down or raise.
        rng = np.random.default_rng(11)
        x = np.tile([-2.0, -1.0, 1.0, 2.0, -3.0, 3.0, -4.0, 4.0], 5)
        y = 0.5 * x + rng.normal(size=x.size)
        df = pd.DataFrame({"y": y, "x": x})
        r = RDPlot(masspoints="off").fit(df, "y", "x")
        assert np.isfinite(r.J).all() and min(r.J) >= 1


class TestDocumentedDeviations:
    """Surfaces where R 4.0.0's rdplot crashes by accident and this port
    implements the paper's formula instead (REGISTRY Deviation notes)."""

    def test_fractional_scale_applies_paper_ceiling(self):
        # R errors ("arguments imply differing number of rows"); CCT 2015
        # Eq 2 puts the ceiling over the rescaled product: 2.5*15 -> 38,
        # 2.5*35 -> 88.
        df = _senate_frame()
        r = RDPlot(scale=2.5).fit(df, "vote", "margin")
        assert r.J == (38.0, 88.0)
        assert not r.vars_bins.isna().any().any()

    def test_pair_scale_supported(self):
        # R >= 4.2 rejects a length-2 scale via its vectorized-if accident.
        df = _senate_frame()
        r = RDPlot(scale=(2, 3)).fit(df, "vote", "margin")
        assert r.J == (30.0, 105.0)

    def test_integer_scale_product_unchanged_by_ceiling(self, golden):
        cfg = golden["senate"]["configs"]["scale2"]
        r = _fit(golden, "senate", cfg)
        assert r.J == tuple(float(v) for v in cfg["J"])

    def test_ultra_small_scale_clamps_to_one_bin(self):
        # A positive scale below the ceiling's 1e-12 epsilon must not
        # produce zero bins (Eq 2's partition is at least one bin per side).
        r = RDPlot(scale=1e-14).fit(_senate_frame(), "vote", "margin")
        assert r.J == (1.0, 1.0)
        assert len(r.vars_bins) == 2  # one (non-empty) bin per side


class TestValidationAndErrors:
    def test_n_below_20_raises(self):
        rng = np.random.default_rng(3)
        x = np.concatenate([-rng.uniform(0.1, 1, 9), rng.uniform(0.1, 1, 10)])
        df = pd.DataFrame({"y": rng.normal(size=19), "x": x})
        with pytest.raises(ValueError, match="Not enough observations"):
            RDPlot().fit(df, "y", "x")

    def test_one_sided_tiny_side_raises_named(self):
        rng = np.random.default_rng(4)
        x = np.concatenate([[-0.5], rng.uniform(0.1, 1, 24)])
        df = pd.DataFrame({"y": rng.normal(size=25), "x": x})
        with pytest.raises(ValueError, match="each side"):
            RDPlot().fit(df, "y", "x")

    def test_single_support_point_side_raises(self):
        # All running values tied on one side (>= 2 obs, 1 distinct value):
        # every spacing is zero, J would be Inf, and the jump grid would
        # divide by zero (R crashes opaquely inside seq()). Clear error
        # under the default masspoints="adjust" (the remap cannot rescue a
        # single support point).
        rng = np.random.default_rng(23)
        x = np.concatenate([np.full(15, -0.5), rng.uniform(0.1, 1, 15)])
        df = pd.DataFrame({"y": rng.normal(size=30), "x": x})
        # the guard fires BEFORE masspoints detection (fail-fast)
        with pytest.raises(ValueError, match="single distinct value below"):
            RDPlot().fit(df, "y", "x")

    def test_missing_rows_dropped_with_warning(self):
        df = _senate_frame()
        n_missing = int(np.asarray(df["vote"].isna()).sum())
        assert n_missing > 0  # the vendored CSV has NA votes
        with pytest.warns(UserWarning, match=f"Dropping {n_missing} row"):
            r = RDPlot().fit(df, "vote", "margin")
        assert r.N[0] + r.N[1] == len(df) - n_missing

    def test_cutoff_outside_range_raises(self):
        df = _senate_frame()
        with pytest.raises(ValueError, match="strictly inside"):
            RDPlot(cutoff=1000.0).fit(df, "vote", "margin")

    def test_missing_column_raises(self):
        df = _senate_frame()
        with pytest.raises(ValueError, match="not found"):
            RDPlot().fit(df, "vote", "nope")

    def test_empty_covariates_list_raises(self):
        df = _senate_frame()
        with pytest.raises(ValueError, match="non-empty"):
            RDPlot().fit(df, "vote", "margin", covariates=[])

    def test_duplicate_covariate_names_raise(self, golden):
        df = _frame(golden, "dgp_covs")
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            RDPlot().fit(df, "y", "x", covariates=["zlong", "zlong"])

    def test_bare_string_covariates_raise(self):
        df = _senate_frame()
        with pytest.raises(ValueError, match="list of column names"):
            RDPlot().fit(df, "vote", "margin", covariates="margin")

    def test_covs_drop_false_strict_error_on_collinear(self, golden):
        df = _frame(golden, "dgp_covs")
        with pytest.raises(ValueError, match="collinear"):
            RDPlot(covs_drop=False).fit(df, "y", "x", covariates=["zlong", "zb", "zdup"])

    def test_covs_drop_false_keeps_user_covariate_order(self, golden):
        # R sorts covariate columns by name length ONLY inside its
        # covs_drop pipeline; with covs_drop=False the user-given order is
        # preserved (same contract as RegressionDiscontinuity). Per-name
        # gamma values are identical either way on full-rank covariates.
        df = _frame(golden, "dgp_covs")
        r_sorted = RDPlot(covs_drop=True).fit(df, "y", "x", covariates=["zlong", "zb"])
        r_strict = RDPlot(covs_drop=False).fit(df, "y", "x", covariates=["zlong", "zb"])
        assert r_sorted.covariate_coefficients is not None
        assert r_strict.covariate_coefficients is not None
        assert list(r_sorted.covariate_coefficients.keys()) == ["zb", "zlong"]
        assert list(r_strict.covariate_coefficients.keys()) == ["zlong", "zb"]
        for name in ("zb", "zlong"):
            assert r_strict.covariate_coefficients[name] == pytest.approx(
                r_sorted.covariate_coefficients[name], rel=1e-12
            )
        pd.testing.assert_frame_equal(r_strict.vars_bins, r_sorted.vars_bins)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cutoff": np.inf},
            {"p": -1},
            {"p": True},
            {"p": 1.5},
            {"nbins": 0},
            {"nbins": (10, 14, 3)},
            {"nbins": (10, True)},
            {"binselect": "esvm"},
            {"scale": 0},
            {"scale": (-1, 2)},
            {"kernel": "gaussian"},
            {"h": -5},
            {"support": (3, 1)},
            {"support": (1,)},
            {"masspoints": "on"},
            {"ci": 0},
            {"ci": 100},
            {"ci": True},
            {"covs_drop": "yes"},
        ],
    )
    def test_constructor_rejects(self, kwargs):
        with pytest.raises(ValueError):
            RDPlot(**kwargs)


class TestAPIContracts:
    def test_get_params_round_trip(self):
        rp = RDPlot(binselect="qs", ci=90, scale=(1, 2))
        params = rp.get_params()
        assert set(params) == {
            "cutoff",
            "p",
            "nbins",
            "binselect",
            "scale",
            "kernel",
            "h",
            "support",
            "masspoints",
            "ci",
            "covs_drop",
        }
        clone = RDPlot(**params)
        assert clone.get_params() == params

    def test_set_params_transactional(self):
        rp = RDPlot()
        before = rp.get_params()
        with pytest.raises(ValueError):
            rp.set_params(binselect="qs", ci=200)  # ci invalid -> nothing applied
        assert rp.get_params() == before
        with pytest.raises(ValueError):
            rp.set_params(bandwidth=3)  # unknown name
        rp.set_params(binselect="qs")
        assert rp.binselect == "qs"

    def test_fit_idempotent_on_config(self):
        df = _senate_frame()
        rp = RDPlot(ci=90)
        before = rp.get_params()
        r1 = rp.fit(df, "vote", "margin")
        r2 = rp.fit(df, "vote", "margin")
        assert rp.get_params() == before
        assert r1.J == r2.J
        pd.testing.assert_frame_equal(r1.vars_bins, r2.vars_bins)

    def test_ci_columns_always_present(self):
        df = _senate_frame()
        r_default = RDPlot().fit(df, "vote", "margin")
        assert {"rdplot_ci_l", "rdplot_ci_r"} <= set(r_default.vars_bins.columns)
        assert r_default.ci_level == 95.0 and r_default.ci_requested is False
        r_90 = RDPlot(ci=90).fit(df, "vote", "margin")
        assert r_90.ci_level == 90.0 and r_90.ci_requested is True
        # narrower level -> strictly narrower intervals on multi-obs bins
        multi = r_default.vars_bins["rdplot_N"] > 1
        width_95 = (r_default.vars_bins["rdplot_ci_r"] - r_default.vars_bins["rdplot_ci_l"])[multi]
        width_90 = (r_90.vars_bins["rdplot_ci_r"] - r_90.vars_bins["rdplot_ci_l"])[multi]
        assert (width_90 < width_95).all()

    def test_to_dataframe_returns_copy(self):
        r = RDPlot().fit(_senate_frame(), "vote", "margin")
        df1 = r.to_dataframe()
        df1.iloc[0, 0] = -999
        assert r.vars_bins.iloc[0, 0] != -999

    def test_summary_contains_r_layout_fields(self):
        r = RDPlot().fit(_senate_frame(), "vote", "margin")
        s = r.summary()
        for needle in (
            "Bins Selected",
            "IMSE-optimal bins",
            "Mimicking Variance bins",
            "Implied scale",
            "WIMSE variance weight",
            "WIMSE bias weight",
        ):
            assert needle in s

    def test_plot_importerror_without_matplotlib(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _no_mpl(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("no matplotlib")
            return real_import(name, *args, **kwargs)

        r = RDPlot().fit(_senate_frame(), "vote", "margin")
        monkeypatch.setattr(builtins, "__import__", _no_mpl)
        with pytest.raises(ImportError, match="matplotlib is required"):
            r.plot()

    def test_plot_renders_when_matplotlib_available(self):
        mpl = pytest.importorskip("matplotlib")
        mpl.use("Agg")
        r = RDPlot(ci=90).fit(_senate_frame(), "vote", "margin")
        ax = r.plot()
        assert len(ax.lines) >= 3  # two curve segments + cutoff line
        assert isinstance(r, RDPlotResult)
        # curve segments split by construction: each side's line has exactly
        # 500 points and owns its OWN x == cutoff endpoint (an x-based mask
        # would hand the right curve's cutoff point to the left segment)
        curve_lines = [ln for ln in ax.lines if len(ln.get_xdata()) == 500]
        assert len(curve_lines) == 2
        assert curve_lines[0].get_xdata()[-1] == r.cutoff
        assert curve_lines[1].get_xdata()[0] == r.cutoff
