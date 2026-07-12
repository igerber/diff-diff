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
    qrXXinv,
    quantile_type2,
    rdbwselect_sharp,
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
                out = rdbwselect_sharp(y, x, **kwargs)
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
        a = rdbwselect_sharp(y, x, masspoints="adjust")
        o = rdbwselect_sharp(y, x, masspoints="off")
        for sel in BWSELECT_OPTIONS:
            np.testing.assert_array_equal(a.bws[sel], o.bws[sel])

    def test_cer_reuses_mse_pilot_bandwidth(self, golden):
        """cer* shrink h only; b is inherited from the matching mse*."""
        entry = golden["dgp_lee_smooth"]
        x = np.asarray(entry["x"], dtype=np.float64)
        y = np.asarray(entry["y"], dtype=np.float64)
        out = rdbwselect_sharp(y, x)
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
        out = rdbwselect_sharp(y, x, masspoints="off")
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
            rdbwselect_sharp(y, x)

    def test_one_sided_data_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="one side of the cutoff"):
            rdbwselect_sharp(y, np.abs(x) + 1.0)

    def test_non_finite_raises(self):
        y, x = self._xy(50)
        y[3] = np.nan
        with pytest.raises(ValueError, match="finite and complete-case"):
            rdbwselect_sharp(y, x)

    def test_length_mismatch_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="equal length"):
            rdbwselect_sharp(y[:-1], x)

    def test_zero_variance_running_var_raises(self):
        y, _ = self._xy(50)
        with pytest.raises(ValueError, match="zero variance"):
            rdbwselect_sharp(y, np.zeros(50))

    def test_zero_variance_running_var_raises_under_stdvars(self):
        # Guard must fire BEFORE the stdvars division, not surface as a
        # divide-by-zero or a misleading one-sided-data error.
        y, _ = self._xy(50)
        with pytest.raises(ValueError, match="zero variance"):
            rdbwselect_sharp(y, np.zeros(50), stdvars=True)

    def test_zero_variance_outcome_raises_under_stdvars(self):
        _, x = self._xy(50)
        with pytest.raises(ValueError, match="outcome has zero variance"):
            rdbwselect_sharp(np.ones(50), x, stdvars=True)

    def test_two_dimensional_input_rejected(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="1-D vector"):
            rdbwselect_sharp(y.reshape(10, 5), x.reshape(10, 5))

    def test_column_vector_input_accepted(self):
        y, x = self._xy(50)
        a = rdbwselect_sharp(y, x)
        b = rdbwselect_sharp(y.reshape(-1, 1), x.reshape(-1, 1))
        for sel in BWSELECT_OPTIONS:
            np.testing.assert_array_equal(a.bws[sel], b.bws[sel])

    def test_vce_hc_raises_not_implemented(self):
        y, x = self._xy(50)
        with pytest.raises(NotImplementedError, match="vce='nn'"):
            rdbwselect_sharp(y, x, vce="hc1")

    def test_invalid_masspoints_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="masspoints"):
            rdbwselect_sharp(y, x, masspoints="on")

    def test_invalid_orders_raise(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="deriv <= p < q"):
            rdbwselect_sharp(y, x, p=2, q=2)

    def test_nnmatch_below_one_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="nnmatch"):
            rdbwselect_sharp(y, x, nnmatch=0)

    def test_bwcheck_below_one_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="bwcheck"):
            rdbwselect_sharp(y, x, bwcheck=0)

    def test_negative_scaleregul_raises(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="scaleregul"):
            rdbwselect_sharp(y, x, scaleregul=-1.0)

    def test_non_integer_orders_raise(self):
        y, x = self._xy(50)
        with pytest.raises(ValueError, match="must be an integer"):
            rdbwselect_sharp(y, x, p=1.5, q=2.5)  # type: ignore[arg-type]

    def test_mass_warning_fires_on_ties(self, golden):
        entry = golden["dgp_ties_moderate"]
        x = np.asarray(entry["x"], dtype=np.float64)
        y = np.asarray(entry["y"], dtype=np.float64)
        with pytest.warns(UserWarning, match="Mass points detected"):
            rdbwselect_sharp(y, x, masspoints="adjust")

    def test_check_mode_suggests_adjust(self, golden):
        entry = golden["dgp_ties_moderate"]
        x = np.asarray(entry["x"], dtype=np.float64)
        y = np.asarray(entry["y"], dtype=np.float64)
        # check mode emits TWO warnings (mass detection + the adjust
        # suggestion); capture the full record so neither leaks into the
        # pytest warning summary.
        with pytest.warns(UserWarning) as record:
            rdbwselect_sharp(y, x, masspoints="check")
        messages = [str(w.message) for w in record]
        assert any("Mass points detected" in m for m in messages)
        assert any("masspoints='adjust'" in m for m in messages)

    def test_adjust_injects_bwcheck_10(self, golden):
        entry = golden["dgp_ties_moderate"]
        x = np.asarray(entry["x"], dtype=np.float64)
        y = np.asarray(entry["y"], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rdbwselect_sharp(y, x, masspoints="adjust")
            out_off = rdbwselect_sharp(y, x, masspoints="off")
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
            a = rdbwselect_sharp(y, x)
            b = rdbwselect_sharp(y[perm], x[perm])
        for sel in BWSELECT_OPTIONS:
            np.testing.assert_allclose(a.bws[sel], b.bws[sel], rtol=1e-12)
