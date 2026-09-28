"""Tests for diff_diff._rdrobust_port (sharp-RD bandwidth machinery).

Golden-file parity targets the installed CRAN rdrobust 4.0.0 (tarball
sha256 pinned in the port). Tolerance policy: the port is formula-faithful,
but R's sample SD accumulates sequentially (C double loop) while numpy uses
pairwise summation, seeding a ~2-ULP difference in ``BWp`` that the chained
d->b->h pilot power functions amplify; observed worst-case relative
disagreement on the golden configs is ~6e-12 (darwin-arm64, R 4.5.2 vs
numpy/Accelerate). Bandwidth parity therefore asserts rtol=1e-9 - far below
any methodological materiality, comfortably above cross-BLAS wobble. Head
intermediates (type-2 IQR, unique counts) are deterministic arithmetic on
identical doubles and assert much tighter.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from diff_diff._rdrobust_port import (
    BWSELECT_OPTIONS,
    KERNEL_C_C,
    RDROBUST_TARBALL_SHA256,
    RDROBUST_VERSION,
    compute_dups_dupsid,
    covs_drop_fun,
    qrXXinv,
    quantile_type2,
    rdbwselect,
    rdrobust_fit,
    rdrobust_kweight,
    rdrobust_res_nn,
    rdrobust_vander,
)

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "rdrobust_golden.json"


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(
            "Golden values file not found; run: " "Rscript benchmarks/R/generate_rdrobust_golden.R"
        )
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def test_pinned_rdrobust_version():
    assert RDROBUST_VERSION == "4.0.0"
    assert RDROBUST_TARBALL_SHA256 == (
        "78f0d6b4bdec4091cc8f42f6f1598704747f95926446d3aaee381ea1d613a36f"
    )


def test_golden_metadata_matches_pin(golden):
    assert golden["metadata"]["rdrobust_version"] == RDROBUST_VERSION
    assert golden["metadata"]["rdrobust_tarball_sha256"] == RDROBUST_TARBALL_SHA256


class TestQuantileType2:
    def test_even_n_averages_at_discontinuity(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        # n*p = 4*0.25 = 1 exactly -> average of 1st and 2nd order stats
        assert quantile_type2(x, 0.25) == 1.5
        assert quantile_type2(x, 0.75) == 3.5

    def test_non_integer_np_takes_next_order_stat(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # n*p = 1.25 -> j=1, g>0 -> 2nd order statistic
        assert quantile_type2(x, 0.25) == 2.0

    def test_matches_r_on_golden_iqr(self, golden):
        # x_iq in the golden file is R's quantile(type=2) IQR on the exact
        # same doubles; the port's arithmetic is identical (sort + average),
        # so agreement is at float64 identity level.
        for dgp in ("dgp_lee_smooth", "dgp_ties_moderate", "senate"):
            entry = golden[dgp]
            x = np.asarray(
                entry.get("x") if entry.get("x") is not None else _senate_xy(golden)[1],
                dtype=np.float64,
            )
            cfg = next(iter(entry["configs"].values()))
            got = quantile_type2(x, 0.75) - quantile_type2(x, 0.25)
            assert got == pytest.approx(cfg["x_iq"], rel=1e-15)


class TestRdrobustKweight:
    def test_triangular_values_and_h_scaling(self):
        x = np.array([-0.5, 0.0, 0.25, 1.0, 2.0])
        w = rdrobust_kweight(x, 0.0, 2.0, "triangular")
        u = x / 2.0
        expected = (1 - np.abs(u)) * (np.abs(u) <= 1) / 2.0
        np.testing.assert_allclose(w, expected, atol=1e-15)

    def test_boundary_u_equals_one_gets_zero_weight_triangular(self):
        # (1 - |u|) vanishes at |u| = 1: the observation is inside the
        # indicator but carries zero weight, so w > 0 excludes it.
        w = rdrobust_kweight(np.array([1.0]), 0.0, 1.0, "triangular")
        assert w[0] == 0.0

    def test_uniform_boundary_included(self):
        w = rdrobust_kweight(np.array([1.0, 1.0 + 1e-12]), 0.0, 1.0, "uniform")
        assert w[0] == 0.5 and w[1] == 0.0

    def test_epanechnikov_value(self):
        w = rdrobust_kweight(np.array([0.5]), 0.0, 1.0, "epanechnikov")
        assert w[0] == pytest.approx(0.75 * (1 - 0.25), rel=1e-15)

    def test_r_spellings_accepted(self):
        x = np.array([0.3])
        for full, short in [
            ("triangular", "tri"),
            ("epanechnikov", "epa"),
            ("uniform", "uni"),
        ]:
            np.testing.assert_array_equal(
                rdrobust_kweight(x, 0.0, 1.0, full),
                rdrobust_kweight(x, 0.0, 1.0, short),
            )

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="kernel must be one of"):
            rdrobust_kweight(np.array([0.0]), 0.0, 1.0, "gaussian")

    def test_kernel_constants(self):
        assert KERNEL_C_C == {
            "epanechnikov": 2.34,
            "uniform": 1.843,
            "triangular": 2.576,
        }


class TestQrXXinv:
    def test_matches_direct_inverse_on_well_conditioned(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))
        got = qrXXinv(X)
        np.testing.assert_allclose(got, np.linalg.inv(X.T @ X), rtol=1e-10)

    def test_singular_falls_back_to_pinv_not_raise(self):
        # Rank-deficient design: R's chol() fails -> MASS::ginv. The port
        # must return the pseudo-inverse, NOT raise (contrast with the
        # nprobust port, which raises by design). This fallback is
        # R-faithful for 4.0.0 - verified against the sha256-pinned CRAN
        # tarball (functions.R:128-132) AND deparse of the installed
        # namespace, both of which carry the try(chol)->ginv branch.
        # Estimator-level rank/support guards are a fit()-level concern
        # (follow-up PR), deliberately not a parity-port concern.
        X = np.ones((10, 2))  # two identical columns
        got = qrXXinv(X)
        expected = np.linalg.pinv(X.T @ X, rcond=float(np.sqrt(np.finfo(float).eps)))
        np.testing.assert_allclose(got, expected, atol=1e-12)


class TestRdrobustVander:
    def test_columns_are_successive_powers(self):
        u = np.array([-1.0, 0.5, 2.0])
        V = rdrobust_vander(u, 3)
        np.testing.assert_allclose(V[:, 0], 1.0)
        np.testing.assert_allclose(V[:, 1], u)
        np.testing.assert_allclose(V[:, 2], u * u)
        np.testing.assert_allclose(V[:, 3], u * u * u)

    def test_p_zero_returns_intercept_only(self):
        V = rdrobust_vander(np.array([2.0, 3.0]), 0)
        assert V.shape == (2, 1)
        np.testing.assert_array_equal(V, np.ones((2, 1)))


class TestDupsDupsid:
    def test_rle_semantics(self):
        x = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0])
        dups, dupsid = compute_dups_dupsid(x)
        np.testing.assert_array_equal(dups, [3, 3, 3, 1, 2, 2])
        np.testing.assert_array_equal(dupsid, [1, 2, 3, 1, 1, 2])

    def test_all_distinct(self):
        dups, dupsid = compute_dups_dupsid(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(dups, [1, 1, 1])
        np.testing.assert_array_equal(dupsid, [1, 1, 1])

    def test_empty(self):
        dups, dupsid = compute_dups_dupsid(np.array([]))
        assert dups.shape == (0,) and dupsid.shape == (0,)


class TestRdrobustResNN:
    def test_equidistant_neighbors_expand_both_sides(self):
        # x[1] = 1 sits exactly between 0 and 2: with matches=1 the exact
        # tie (functions.R:162-165) pulls BOTH neighbors in, J=2.
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 5.0, 3.0])
        dups, dupsid = compute_dups_dupsid(x)
        res = rdrobust_res_nn(x, y, 1, dups, dupsid)
        # pos=1: neighbors {0, 2}, J=2: sqrt(2/3) * (5 - (1+3)/2)
        assert res[1] == pytest.approx(np.sqrt(2 / 3) * (5.0 - 2.0), rel=1e-14)

    def test_tie_blocks_join_as_units(self):
        # A duplicate of x[0] means pos=0's first neighbor is its own tie
        # block partner (distance 0) before anything else.
        x = np.array([0.0, 0.0, 1.0])
        y = np.array([1.0, 2.0, 9.0])
        dups, dupsid = compute_dups_dupsid(x)
        res = rdrobust_res_nn(x, y, 1, dups, dupsid)
        # pos=0: dups block gives rpos=1 immediately (J=1, neighbor y=2)
        assert res[0] == pytest.approx(np.sqrt(1 / 2) * (1.0 - 2.0), rel=1e-14)

    def test_boundary_positions_expand_inward(self):
        x = np.array([0.0, 1.0, 3.0, 6.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        dups, dupsid = compute_dups_dupsid(x)
        res = rdrobust_res_nn(x, y, 3, dups, dupsid)
        # pos=0 must reach right until 3 neighbors: {1,3,6}
        assert res[0] == pytest.approx(np.sqrt(3 / 4) * (2.0 - 6.0), rel=1e-14)

    def test_j_floor_is_min_matches_n_minus_1(self):
        x = np.array([0.0, 1.0])
        y = np.array([1.0, 3.0])
        dups, dupsid = compute_dups_dupsid(x)
        res = rdrobust_res_nn(x, y, 3, dups, dupsid)  # only 1 neighbor exists
        assert res[0] == pytest.approx(np.sqrt(1 / 2) * (1.0 - 3.0), rel=1e-14)


def _senate_xy(golden):
    csv_path = Path(__file__).resolve().parents[1] / golden["senate"]["csv"]
    if not csv_path.exists():
        pytest.skip(f"Vendored Senate CSV not found at {csv_path}")
    import pandas as pd

    df = pd.read_csv(csv_path)
    ok = df[["vote", "margin"]].dropna()
    return np.asarray(ok["vote"], dtype=np.float64), np.asarray(ok["margin"], dtype=np.float64)


def _iter_golden_configs(golden):
    """Yield (label, y, x, config_dict) over every golden configuration."""
    for dgp_name, entry in golden.items():
        if dgp_name == "metadata":
            continue
        if dgp_name == "senate":
            y, x = _senate_xy(golden)
        else:
            x = np.asarray(entry["x"], dtype=np.float64)
            y = np.asarray(entry["y"], dtype=np.float64)
        for cfg_name, cfg in entry["configs"].items():
            yield f"{dgp_name}/{cfg_name}", y, x, cfg


class TestRdbwselectGoldenParity:
    """End-to-end 10-selector parity against installed rdrobust 4.0.0."""

    RTOL = 1e-9  # policy: see module docstring (observed worst ~6e-12)

    def test_all_configs_all_selectors(self, golden):
        n_checked = 0
        for label, y, x, cfg in _iter_golden_configs(golden):
            kwargs: dict = dict(
                c=float(cfg["c"]),
                p=int(cfg["p"]),
                q=int(cfg["q"]),
                deriv=int(cfg["deriv"]),
                kernel=cfg["kernel"],
                masspoints=cfg["masspoints"],
                scaleregul=float(cfg["scaleregul"]),
                bwrestrict=bool(cfg["bwrestrict"]),
                stdvars=bool(cfg["stdvars"]),
                nnmatch=int(cfg["nnmatch"]),
            )
            if cfg["bwcheck"] is not None:
                kwargs["bwcheck"] = int(cfg["bwcheck"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = rdbwselect(y, x, **kwargs)
            assert out.N == cfg["N"], label
            if cfg["masspoints"] == "off" and cfg["bwcheck"] is None:
                # Under masspoints='off' without an explicit bwcheck, R never
                # computes unique counts: the effective M stays N per side
                # (rdbwselect.R:139). The golden's M_l/M_r record the
                # unconditional DATA property instead.
                assert out.M_l == out.N_l and out.M_r == out.N_r, label
            else:
                assert out.M_l == cfg["M_l"], label
                assert out.M_r == cfg["M_r"], label
            assert out.diagnostics["x_iq"] == pytest.approx(cfg["x_iq"], rel=1e-15), label
            assert out.diagnostics["BWp"] == pytest.approx(cfg["BWp"], rel=1e-13), label
            for sel in BWSELECT_OPTIONS:
                got = np.array(out.bws[sel])
                want = np.array(cfg["bws"][sel], dtype=np.float64)
                np.testing.assert_allclose(got, want, rtol=self.RTOL, err_msg=f"{label}:{sel}")
                n_checked += 1
        # 17 configs x 10 selectors: fail loudly if the golden shrinks.
        assert n_checked == 170

    def test_smooth_dgp_adjust_equals_off(self, golden):
        """No ties -> masspoints='adjust' and 'off' must coincide exactly."""
        entry = golden["dgp_lee_smooth"]
        x = np.asarray(entry["x"], dtype=np.float64)
        y = np.asarray(entry["y"], dtype=np.float64)
        a = rdbwselect(y, x, masspoints="adjust")
        o = rdbwselect(y, x, masspoints="off")
        for sel in BWSELECT_OPTIONS:
            np.testing.assert_array_equal(a.bws[sel], o.bws[sel])

    def test_cer_reuses_mse_pilot_bandwidth(self, golden):
        """cer* shrink h only; b is inherited from the matching mse*."""
        entry = golden["dgp_lee_smooth"]
        x = np.asarray(entry["x"], dtype=np.float64)
        y = np.asarray(entry["y"], dtype=np.float64)
        out = rdbwselect(y, x)
        for mse, cer in [
            ("mserd", "cerrd"),
            ("msesum", "cersum"),
            ("msetwo", "certwo"),
            ("msecomb1", "cercomb1"),
            ("msecomb2", "cercomb2"),
        ]:
            assert out.bws[cer][2] == out.bws[mse][2]
            assert out.bws[cer][3] == out.bws[mse][3]
            cer_h = out.diagnostics["cer_h"]
            assert out.bws[cer][0] == pytest.approx(out.bws[mse][0] * cer_h, rel=1e-15)


class TestSenatePublished2017:
    """The 2017 Stata Journal paper's printed Senate bandwidths are an
    anchor independent of our own R invocation. The published numbers
    predate the masspoints option; masspoints='off' reproduces them."""

    def test_masspoints_off_matches_stata_journal_2017(self, golden):
        y, x = _senate_xy(golden)
        out = rdbwselect(y, x, masspoints="off")
        assert out.bws["mserd"][0] == pytest.approx(17.708, abs=1e-3)
        assert out.bws["mserd"][2] == pytest.approx(27.984, abs=1e-3)
        # rdbwselect ..., all output table (Stata Journal 2017, p. 400)
        assert out.bws["msetwo"][0] == pytest.approx(16.154, abs=1e-3)
        assert out.bws["msetwo"][1] == pytest.approx(18.009, abs=1e-3)
        assert out.bws["cerrd"][0] == pytest.approx(12.374, abs=1e-3)
        assert out.bws["cersum"][0] == pytest.approx(12.806, abs=1e-3)


class TestValidationAndWarnings:
    def _xy(self, n=100, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n)
        y = x + (x >= 0) + rng.standard_normal(n) * 0.1
        return y, x

    def test_n_below_20_raises(self):
        y, x = self._xy(19)
        with pytest.raises(ValueError, match="Not enough observations"):
            rdbwselect(y, x)

    def test_one_sided_data_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="one side of the cutoff"):
            rdbwselect(y, np.abs(x) + 1.0)

    def test_non_finite_raises(self):
        y, x = self._xy(50)
        y[3] = np.nan
        with pytest.raises(ValueError, match="finite and complete-case"):
            rdbwselect(y, x)

    def test_length_mismatch_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="equal length"):
            rdbwselect(y[:-1], x)

    def test_zero_variance_running_var_raises(self):
        y, _ = self._xy(50)
        with pytest.raises(ValueError, match="zero variance"):
            rdbwselect(y, np.zeros(50))

    def test_zero_variance_running_var_raises_under_stdvars(self):
        # Guard must fire BEFORE the stdvars division, not surface as a
        # divide-by-zero or a misleading one-sided-data error.
        y, _ = self._xy(50)
        with pytest.raises(ValueError, match="zero variance"):
            rdbwselect(y, np.zeros(50), stdvars=True)

    def test_zero_variance_outcome_raises_under_stdvars(self):
        _, x = self._xy(50)
        with pytest.raises(ValueError, match="outcome has zero variance"):
            rdbwselect(np.ones(50), x, stdvars=True)

    def test_two_dimensional_input_rejected(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="1-D vector"):
            rdbwselect(y.reshape(10, 5), x.reshape(10, 5))

    def test_column_vector_input_accepted(self):
        y, x = self._xy(50)
        a = rdbwselect(y, x)
        b = rdbwselect(y.reshape(-1, 1), x.reshape(-1, 1))
        for sel in BWSELECT_OPTIONS:
            np.testing.assert_array_equal(a.bws[sel], b.bws[sel])

    def test_vce_hc_raises_not_implemented(self):
        y, x = self._xy(50)
        with pytest.raises(NotImplementedError, match="vce='nn'"):
            rdbwselect(y, x, vce="hc1")

    def test_invalid_masspoints_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="masspoints"):
            rdbwselect(y, x, masspoints="on")

    def test_invalid_orders_raise(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="deriv <= p < q"):
            rdbwselect(y, x, p=2, q=2)

    def test_nnmatch_below_one_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="nnmatch"):
            rdbwselect(y, x, nnmatch=0)

    def test_bwcheck_below_one_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="bwcheck"):
            rdbwselect(y, x, bwcheck=0)

    def test_negative_scaleregul_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="scaleregul"):
            rdbwselect(y, x, scaleregul=-1.0)

    def test_non_integer_orders_raise(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="must be an integer"):
            rdbwselect(y, x, p=1.5, q=2.5)  # type: ignore[arg-type]

    def test_mass_warning_fires_on_ties(self, golden):
        entry = golden["dgp_ties_moderate"]
        x = np.asarray(entry["x"], dtype=np.float64)
        y = np.asarray(entry["y"], dtype=np.float64)
        with pytest.warns(UserWarning, match="Mass points detected"):
            rdbwselect(y, x, masspoints="adjust")

    def test_check_mode_suggests_adjust(self, golden):
        entry = golden["dgp_ties_moderate"]
        x = np.asarray(entry["x"], dtype=np.float64)
        y = np.asarray(entry["y"], dtype=np.float64)
        # check mode emits TWO warnings (mass detection + the adjust
        # suggestion); capture the full record so neither leaks into the
        # pytest warning summary.
        with pytest.warns(UserWarning) as record:
            rdbwselect(y, x, masspoints="check")
        messages = [str(w.message) for w in record]
        assert any("Mass points detected" in m for m in messages)
        assert any("masspoints='adjust'" in m for m in messages)

    def test_adjust_injects_bwcheck_10(self, golden):
        entry = golden["dgp_ties_moderate"]
        x = np.asarray(entry["x"], dtype=np.float64)
        y = np.asarray(entry["y"], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rdbwselect(y, x, masspoints="adjust")
            out_off = rdbwselect(y, x, masspoints="off")
        assert out.bwcheck_effective == 10
        assert out_off.bwcheck_effective is None

    def test_unsorted_input_matches_sorted(self, golden):
        entry = golden["dgp_ties_moderate"]
        x = np.asarray(entry["x"], dtype=np.float64)
        y = np.asarray(entry["y"], dtype=np.float64)
        rng = np.random.default_rng(5)
        perm = rng.permutation(x.shape[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = rdbwselect(y, x)
            b = rdbwselect(y[perm], x[perm])
        for sel in BWSELECT_OPTIONS:
            np.testing.assert_allclose(a.bws[sel], b.bws[sel], rtol=1e-12)


class TestRdrobustFitSharpValidation:
    """rdrobust_fit shares rdbwselect's input contract; direct
    (non-estimator) callers get targeted errors, not opaque NumPy ones."""

    def _yx(self, n=200, seed=11):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n)
        y = 0.3 * x + 0.5 * (x >= 0) + rng.normal(0, 0.1, n)
        return y, x

    def test_two_dim_input_rejected(self):
        y, x = self._yx()
        with pytest.raises(ValueError, match="1-D vector"):
            rdrobust_fit(y.reshape(50, 4), x, 0.0, 0.5, 0.5, 0.5, 0.5)

    def test_unequal_lengths_rejected(self):
        y, x = self._yx()
        with pytest.raises(ValueError, match="equal length"):
            rdrobust_fit(y[:-5], x, 0.0, 0.5, 0.5, 0.5, 0.5)

    def test_non_integer_orders_rejected(self):
        y, x = self._yx()
        with pytest.raises(ValueError, match="p must be an integer"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, p=1.0)

    def test_order_inequality_enforced(self):
        y, x = self._yx()
        with pytest.raises(ValueError, match="0 <= deriv <= p < q"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, p=2, q=2)
        with pytest.raises(ValueError, match="0 <= deriv <= p < q"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, deriv=2, p=1, q=2)

    def test_nnmatch_validated(self):
        y, x = self._yx()
        with pytest.raises(ValueError, match="nnmatch must be an integer >= 1"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, nnmatch=0)

    def test_column_vector_accepted(self):
        y, x = self._yx()
        a = rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5)
        b = rdrobust_fit(y.reshape(-1, 1), x.reshape(-1, 1), 0.0, 0.5, 0.5, 0.5, 0.5)
        assert a.tau_bc == b.tau_bc and a.se_rb == b.se_rb


ESTIMATES_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "rdrobust_estimates_golden.json"
)


@pytest.fixture(scope="module")
def estimates_golden():
    if not ESTIMATES_GOLDEN_PATH.exists():
        pytest.skip(
            "Estimates golden file not found; run: "
            "Rscript benchmarks/R/generate_rdrobust_estimates_golden.R"
        )
    with open(ESTIMATES_GOLDEN_PATH) as f:
        return json.load(f)


class TestFuzzyPortGoldenParity:
    """Port-level fuzzy parity incl. the per-side LINEARIZED biases
    (bias_side = s_Y . B_F_side, rdrobust.R:649-652), which the public
    results object does not expose - pinned here so the fuzzy bias formula
    cannot silently regress to the sharp per-component difference."""

    def test_fuzzy_configs_with_bias(self, estimates_golden):
        entry = estimates_golden["dgp_fuzzy"]
        y = np.array(entry["y"])
        n_checked = 0
        for name, cfg in entry["configs"].items():
            x = np.array(entry["x_ties"] if name == "ties_adjust" else entry["x"])
            t = np.array(entry["t_one"] if name == "one_sided" else entry["t"], dtype=np.float64)
            if cfg["h_in"] is not None:
                h_l = h_r = b_l = b_r = float(cfg["h_in"])  # h alone -> b = h
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bw = rdbwselect(
                        y,
                        x,
                        kernel=cfg["kernel"],
                        masspoints=cfg["masspoints"],
                        fuzzy=t,
                        sharpbw=bool(cfg["sharpbw"]),
                    )
                h_l, h_r, b_l, b_r = bw.bws[cfg["bwselect"]]
            fit = rdrobust_fit(y, x, 0.0, h_l, h_r, b_l, b_r, kernel=cfg["kernel"], t=t)
            for label, got, want in (
                ("h_l", h_l, cfg["h_l"]),
                ("b_l", b_l, cfg["b_l"]),
                ("tau_cl", fit.tau_cl, cfg["tau_cl"]),
                ("tau_bc", fit.tau_bc, cfg["tau_bc"]),
                ("se_cl", fit.se_cl, cfg["se_cl"]),
                ("se_rb", fit.se_rb, cfg["se_rb"]),
                ("tau_T_cl", fit.tau_T_cl, cfg["tau_T"][0]),
                ("tau_T_bc", fit.tau_T_bc, cfg["tau_T"][1]),
                ("se_T_cl", fit.se_T_cl, cfg["se_T"][0]),
                ("se_T_rb", fit.se_T_rb, cfg["se_T"][2]),
                ("bias_l", fit.bias_l, cfg["bias"][0]),
                ("bias_r", fit.bias_r, cfg["bias"][1]),
            ):
                assert got == pytest.approx(
                    want, rel=1e-9, abs=1e-12
                ), f"{name}:{label}: {got} vs {want}"
            n_checked += 1
        assert n_checked == 7


class TestFuzzyPortValidation:
    def _yxt(self, n=200, seed=13):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n)
        t = (rng.uniform(size=n) < np.where(x >= 0, 0.8, 0.2)).astype(float)
        y = 0.3 * x + t + rng.normal(0, 0.1, n)
        return y, x, t

    def test_two_dim_t_rejected(self):
        y, x, t = self._yxt()
        with pytest.raises(ValueError, match="1-D vector"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, t=t.reshape(50, 4))
        with pytest.raises(ValueError, match="1-D vector"):
            rdbwselect(y, x, fuzzy=t.reshape(50, 4))

    def test_t_length_mismatch_rejected(self):
        y, x, t = self._yxt()
        with pytest.raises(ValueError, match="length equal to x"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, t=t[:-3])
        with pytest.raises(ValueError, match="length equal to x"):
            rdbwselect(y, x, fuzzy=t[:-3])

    def test_identification_stop_reachable_directly(self):
        y, x, _ = self._yxt()
        const = np.full_like(x, 0.7)
        with pytest.raises(ValueError, match="no variation and no jump"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, t=const)
        with pytest.raises(ValueError, match="no variation and no jump"):
            rdbwselect(y, x, fuzzy=const)

    def test_column_vector_t_accepted(self):
        y, x, t = self._yxt()
        a = rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, t=t)
        b = rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, t=t.reshape(-1, 1))
        assert a.tau_bc == b.tau_bc and a.se_T_rb == b.se_T_rb


class TestCovsPortGoldenParity:
    """Port-level covariate parity incl. the per-side biases and the gamma
    matrix (R's coef_covs). The covs_msetwo / covs_cercomb2 configs pin Z
    threading through ALL THREE selector chains (the port always computes
    mserd+msetwo+msesum; partial threading would silently ship
    non-covariate-aware bandwidths for 8 of the 10 selectors), and
    covs_cercomb2 additionally locks CER-rescaling of the covariate-aware
    h. Golden covariate columns were passed to R with UNSORTED names of
    differing lengths, so parity also pins rdrobust's order(nchar) column
    sort (reproduced here by sorting the name list by length before
    building the matrix)."""

    def test_covs_configs_with_bias_and_gamma(self, estimates_golden):
        entry = estimates_golden["dgp_covs"]
        y = np.array(entry["y"])
        t_all = np.array(entry["t"], dtype=np.float64)
        cols = {
            "zlong": np.array(entry["zlong"]),
            "zb": np.array(entry["zb"], dtype=np.float64),
            "zdup": np.array(entry["zdup"]),
        }
        n_checked = 0
        for name, cfg in entry["configs"].items():
            x = np.array(entry["x_ties"] if name == "covs_ties" else entry["x"])
            t = t_all if cfg["fuzzy_in"] else None
            names_sorted = sorted(cfg["covs_names"], key=len)
            z = np.column_stack([cols[c] for c in names_sorted])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if cfg["h_in"] is not None:
                    h_l = h_r = b_l = b_r = float(cfg["h_in"])  # h alone -> b = h
                else:
                    bw = rdbwselect(
                        y,
                        x,
                        kernel=cfg["kernel"],
                        masspoints=cfg["masspoints"],
                        fuzzy=t,
                        sharpbw=bool(cfg["sharpbw"]),
                        covs=z,
                    )
                    h_l, h_r, b_l, b_r = bw.bws[cfg["bwselect"]]
                fit = rdrobust_fit(y, x, 0.0, h_l, h_r, b_l, b_r, kernel=cfg["kernel"], t=t, covs=z)
            pairs = [
                ("h_l", h_l, cfg["h_l"]),
                ("h_r", h_r, cfg["h_r"]),
                ("b_l", b_l, cfg["b_l"]),
                ("tau_cl", fit.tau_cl, cfg["tau_cl"]),
                ("tau_bc", fit.tau_bc, cfg["tau_bc"]),
                ("se_cl", fit.se_cl, cfg["se_cl"]),
                ("se_rb", fit.se_rb, cfg["se_rb"]),
                ("bias_l", fit.bias_l, cfg["bias"][0]),
                ("bias_r", fit.bias_r, cfg["bias"][1]),
            ]
            if cfg["fuzzy_in"]:
                pairs += [
                    ("tau_T_cl", fit.tau_T_cl, cfg["tau_T"][0]),
                    ("tau_T_bc", fit.tau_T_bc, cfg["tau_T"][1]),
                    ("se_T_cl", fit.se_T_cl, cfg["se_T"][0]),
                    ("se_T_rb", fit.se_T_rb, cfg["se_T"][2]),
                ]
            for label, got, want in pairs:
                assert got == pytest.approx(
                    want, rel=1e-9, abs=1e-12
                ), f"{name}:{label}: {got} vs {want}"
            gamma = np.asarray(cfg["coef_covs"], dtype=float)
            gamma = gamma.reshape(gamma.shape[0], -1)
            assert fit.gamma_p is not None
            # Row count pins WHICH columns survived the entry-point drop
            # (covs_drop_collinear: 3 passed, 2 kept).
            assert fit.gamma_p.shape == gamma.shape, f"{name}: gamma shape"
            np.testing.assert_allclose(fit.gamma_p, gamma, rtol=1e-9, err_msg=name)
            # Adjusted per-side coefficient vectors (R's beta_Y_p_*).
            np.testing.assert_allclose(
                fit.beta_p_l, np.ravel(cfg["beta_p_l"]), rtol=1e-9, err_msg=name
            )
            np.testing.assert_allclose(
                fit.beta_p_r, np.ravel(cfg["beta_p_r"]), rtol=1e-9, err_msg=name
            )
            n_checked += 1
        assert n_checked == 9


class TestCovsPortValidation:
    def _yxz(self, n=200, seed=17):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n)
        z1 = 0.5 * x + rng.normal(size=n)
        z2 = rng.binomial(1, 0.4, n).astype(float)
        y = 0.3 * x + 0.8 * (x >= 0) + 0.6 * z1 + 0.2 * z2 + rng.normal(0, 0.2, n)
        return y, x, np.column_stack([z1, z2])

    def test_three_dim_covs_rejected(self):
        y, x, z = self._yxz()
        with pytest.raises(ValueError, match="1-D vector or"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, covs=z.reshape(20, 10, 2))
        with pytest.raises(ValueError, match="1-D vector or"):
            rdbwselect(y, x, covs=z.reshape(20, 10, 2))

    def test_covs_length_mismatch_rejected(self):
        y, x, z = self._yxz()
        with pytest.raises(ValueError, match="rows to match x"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, covs=z[:-3])
        with pytest.raises(ValueError, match="rows to match x"):
            rdbwselect(y, x, covs=z[:-3])

    def test_nonfinite_covs_rejected(self):
        y, x, z = self._yxz()
        z_bad = z.copy()
        z_bad[5, 0] = np.nan
        with pytest.raises(ValueError, match="finite and complete-case"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, covs=z_bad)
        with pytest.raises(ValueError, match="finite and complete-case"):
            rdbwselect(y, x, covs=z_bad)

    def test_covs_drop_strict_bool(self):
        y, x, z = self._yxz()
        with pytest.raises(ValueError, match="covs_drop must be a bool"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, covs=z, covs_drop=1)
        with pytest.raises(ValueError, match="covs_drop must be a bool"):
            rdbwselect(y, x, covs=z, covs_drop="yes")

    def test_rank_zero_covs_fails_closed(self):
        # Deviation from R: an all-zero covariate matrix would make R
        # index a nonexistent column downstream (opaque error); the port
        # raises a targeted ValueError from both entry points.
        y, x, _ = self._yxz()
        zeros = np.zeros((y.shape[0], 2))
        with pytest.raises(ValueError, match="rank-0"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, covs=zeros)
        with pytest.raises(ValueError, match="rank-0"):
            rdbwselect(y, x, covs=zeros)

    def test_entry_drop_warns_with_r_message_and_matches_reduced(self):
        y, x, z = self._yxz()
        z_dup = np.column_stack([z, z[:, 0]])  # exact duplicate appended
        with pytest.warns(UserWarning, match="Multicollinearity issue detected"):
            fit_dup = rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, covs=z_dup)
        fit_red = rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, covs=z)
        # The appended duplicate is cycled out by the pivoted QR, leaving
        # the ORIGINAL columns - bit-identical fit.
        assert fit_dup.tau_bc == fit_red.tau_bc
        assert fit_dup.se_rb == fit_red.se_rb

    def test_covs_drop_false_collinear_raises(self):
        y, x, z = self._yxz()
        z_dup = np.column_stack([z, z[:, 0]])
        with pytest.raises(ValueError, match="covs_drop=True"):
            rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, covs=z_dup, covs_drop=False)

    def test_column_vector_covs_accepted(self):
        y, x, z = self._yxz()
        a = rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, covs=z[:, 0])
        b = rdrobust_fit(y, x, 0.0, 0.5, 0.5, 0.5, 0.5, covs=z[:, 0].reshape(-1, 1))
        assert a.tau_bc == b.tau_bc and a.se_rb == b.se_rb


class TestCovsDropFun:
    """Unit pins for the LINPACK dqrdc2 rank/pivot port against live-R
    ``qr(z, tol=1e-7)`` results (values captured from R 4.5.2 during the
    pre-implementation smoke; deterministic given the constructions).

    The load-bearing property is dqrdc2's PER-COLUMN relative rule (a
    column is negligible when its reduced norm falls below tol times its
    OWN original norm) - a small-but-independent column must never be
    dropped, while exact and near (1e-8) linear combinations must be."""

    def _base(self, n=50, seed=42):
        rng = np.random.default_rng(seed)
        return rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)

    def test_exact_duplicate_dropped(self):
        a, b, _ = self._base()
        keep, rank = covs_drop_fun(np.column_stack([a, b, a]))
        assert rank == 2 and keep.tolist() == [0, 1]

    def test_exact_combination_dropped(self):
        a, b, _ = self._base()
        keep, rank = covs_drop_fun(np.column_stack([a, b, 2 * a - 3 * b]))
        assert rank == 2 and keep.tolist() == [0, 1]

    def test_near_collinear_dropped_at_tol(self):
        a, b, noise = self._base()
        keep, rank = covs_drop_fun(np.column_stack([a, b, a + 1e-8 * noise]))
        assert rank == 2 and keep.tolist() == [0, 1]

    def test_tiny_scaled_independent_column_kept(self):
        # The |R[0,0]|-relative rule (LAPACK-style) would wrongly drop
        # this column; dqrdc2's own-norm rule keeps it, as R does.
        a, b, noise = self._base()
        keep, rank = covs_drop_fun(np.column_stack([a, b, 1e-9 * noise]))
        assert rank == 3 and keep.tolist() == [0, 1, 2]

    def test_constant_column_kept_by_qr(self):
        # No intercept in the covariate-only QR, so a constant nonzero
        # column is full-rank HERE; its collinearity with the polynomial
        # design surfaces later, in the guarded gamma solve.
        a, _, b = self._base()
        keep, rank = covs_drop_fun(np.column_stack([a, np.full(50, 0.7), b]))
        assert rank == 3 and keep.tolist() == [0, 1, 2]

    def test_zero_matrix_rank_zero(self):
        # dqrdc2's zero-norm fixup (work(j,2) = 1) makes all-zero columns
        # negligible: R gives rank 0, pivot 1:3 (verified live).
        keep, rank = covs_drop_fun(np.zeros((5, 3)))
        assert rank == 0 and keep.size == 0

    def test_zero_column_cycled_to_end(self):
        # R: qr(cbind(a, 0, b))$rank == 2, pivot == c(1, 3, 2).
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=5), rng.normal(size=5)
        keep, rank = covs_drop_fun(np.column_stack([a, np.zeros(5), b]))
        assert rank == 2 and keep.tolist() == [0, 2]


class TestCovsDegenerateGuard:
    """The guarded gamma solve (documented Deviation from R): R's
    ginv(tol=1e-20) INVERTS a float-noise singular value on
    exactly-degenerate partialled systems, making its output
    platform-noise (observed 28% cross-implementation gamma spread and
    ~0.5% tau shifts in the pre-implementation smoke). The port excludes
    per-column degeneracies, cuts set-level noise directions with an
    equilibrated (scale-invariant) pseudo-inverse, and warns."""

    def _data(self, n=400, seed=11):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n)
        z1 = 0.4 * x + rng.normal(size=n)
        y = 0.3 * x + 0.8 * (x >= 0) + 0.6 * z1 + rng.normal(0, 0.2, n)
        return y, x, z1, rng

    def test_constant_covariate_excluded_equals_fit_without_it(self):
        y, x, z1, _ = self._data()
        z = np.column_stack([z1, np.full_like(x, 0.7)])
        with pytest.warns(UserWarning, match="collinear with the local polynomial"):
            fit_c = rdrobust_fit(y, x, 0.0, 0.4, 0.4, 0.4, 0.4, covs=z)
        fit_1 = rdrobust_fit(y, x, 0.0, 0.4, 0.4, 0.4, 0.4, covs=z1)
        assert fit_c.covs_excluded is not None
        assert fit_c.covs_excluded.tolist() == [False, True]
        # gamma row zeroed -> the constant contributes exactly nothing.
        assert fit_c.gamma_p is not None and fit_c.gamma_p[1, 0] == 0.0
        # Mathematically identical; numerically only to float roundoff -
        # the (n, 3) vs (n, 2) response shapes route through different
        # BLAS matmul kernels on some platforms (CI round 1: last-ULP
        # diffs on OpenBLAS/Windows, bit-equal on Accelerate).
        assert fit_c.tau_bc == pytest.approx(fit_1.tau_bc, rel=1e-12)
        assert fit_c.se_rb == pytest.approx(fit_1.se_rb, rel=1e-12)

    def test_dummy_set_stabilized_equals_drop_one(self):
        # A full one-hot set passes the intercept-free QR (rank 3) but the
        # partialled system is rank-deficient; the stabilized cut must
        # give the SAME tau as any identified reparametrization of the
        # same span (drop one category) - the span-invariance property R's
        # noise-inverting solve does not have.
        y, x, _z1, rng = self._data()
        cat = rng.integers(0, 3, size=x.shape[0])
        dummies = np.column_stack([(cat == k).astype(float) for k in range(3)])
        with pytest.warns(UserWarning, match="rank-deficient after partialling"):
            fit3 = rdrobust_fit(y, x, 0.0, 0.4, 0.4, 0.4, 0.4, covs=dummies)
        fit2 = rdrobust_fit(y, x, 0.0, 0.4, 0.4, 0.4, 0.4, covs=dummies[:, :2])
        assert fit3.covs_set_degenerate
        assert fit3.tau_bc == pytest.approx(fit2.tau_bc, rel=1e-9)
        assert fit3.se_rb == pytest.approx(fit2.se_rb, rel=1e-9)

    def test_tiny_scaled_covariate_not_flagged(self):
        # Scale-invariance of the guard: a genuinely independent covariate
        # scaled to 1e-9 must go through the R-exact solve untouched.
        y, x, z1, rng = self._data()
        z = np.column_stack([z1, 1e-9 * rng.normal(size=x.shape[0])])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = rdrobust_fit(y, x, 0.0, 0.4, 0.4, 0.4, 0.4, covs=z)
        assert fit.covs_excluded is not None and not fit.covs_excluded.any()
        assert not fit.covs_set_degenerate
