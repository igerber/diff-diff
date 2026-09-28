"""Golden-file parity for the public RegressionDiscontinuity estimator
against R rdrobust 4.0.0 (estimation blocks; bandwidth-selection parity
lives in tests/test_rdrobust_port.py).

Tolerance: rtol=1e-9 on estimates/SEs/CIs (same policy and root cause as
the bandwidth suite - see tests/test_rdrobust_port.py's module docstring;
observed worst ~5e-11 on the p=2/q=3 config whose higher-order pilots
amplify the summation-order seed most).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import RegressionDiscontinuity

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "rdrobust_estimates_golden.json"
)

RTOL = 1e-9


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(
            "Golden values file not found; run: "
            "Rscript benchmarks/R/generate_rdrobust_estimates_golden.R"
        )
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _frame(golden, dgp_name, cfg_name=None):
    entry = golden[dgp_name]
    if dgp_name == "senate":
        csv_path = Path(__file__).resolve().parents[1] / entry["csv"]
        if not csv_path.exists():
            pytest.skip(f"Vendored Senate CSV not found at {csv_path}")
        df = pd.read_csv(csv_path)[["vote", "margin"]].dropna()
        return df.rename(columns={"vote": "y", "margin": "x"})
    if dgp_name == "dgp_fuzzy":
        # Config-specific variants share the same seeded base draw:
        # ties_adjust rounds x to 2dp; one_sided zeroes take-up left of
        # the cutoff (perf_comp path).
        x = entry["x_ties"] if cfg_name == "ties_adjust" else entry["x"]
        t = entry["t_one"] if cfg_name == "one_sided" else entry["t"]
        return pd.DataFrame({"x": x, "y": entry["y"], "t": t})
    if dgp_name == "dgp_covs":
        # Covariate DGP: covs_ties reuses the 2dp-rounded running
        # variable; the covariate columns ride along for every config.
        x = entry["x_ties"] if cfg_name == "covs_ties" else entry["x"]
        return pd.DataFrame(
            {
                "x": x,
                "y": entry["y"],
                "t": entry["t"],
                "zlong": entry["zlong"],
                "zb": entry["zb"],
                "zdup": entry["zdup"],
            }
        )
    return pd.DataFrame({"x": entry["x"], "y": entry["y"]})


def _kwargs_from_config(cfg):
    kwargs = dict(
        cutoff=float(cfg["c"]),
        p=int(cfg["p"]),
        q=int(cfg["q"]),
        kernel=cfg["kernel"],
        bwselect=cfg["bwselect"],
        masspoints=cfg["masspoints"],
        sharpbw=bool(cfg["sharpbw"]),
        alpha=1 - cfg["level"] / 100.0,
    )
    if cfg["h_in"] is not None:
        kwargs["h"] = float(cfg["h_in"])
    if cfg["b_in"] is not None:
        kwargs["b"] = float(cfg["b_in"])
    if cfg["rho_in"] is not None:
        kwargs["rho"] = float(cfg["rho_in"])
    return kwargs


def _fit(golden, dgp_name, cfg, cfg_name=None):
    df = _frame(golden, dgp_name, cfg_name)
    treatment_col = "t" if cfg.get("fuzzy_in") else None
    # covs_names records the columns AS PASSED to R (unsorted, so R's
    # order(nchar) column sort is part of what parity pins).
    covariates = list(cfg["covs_names"]) if cfg.get("covs_in") else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RegressionDiscontinuity(**_kwargs_from_config(cfg)).fit(
            df,
            "y",
            "x",
            takeup=treatment_col,
            covariates=covariates,
        )


class TestEstimateGoldenParity:
    def test_all_configs(self, golden):
        n_checked = 0
        for dgp_name, entry in golden.items():
            if dgp_name == "metadata":
                continue
            for cfg_name, cfg in entry["configs"].items():
                label = f"{dgp_name}/{cfg_name}"
                r = _fit(golden, dgp_name, cfg, cfg_name)
                pairs = [
                    ("tau_cl", r.att_conventional, cfg["tau_cl"]),
                    ("tau_bc", r.att, cfg["tau_bc"]),
                    ("se_cl", r.se_conventional, cfg["se_cl"]),
                    ("se_rb", r.se, cfg["se_rb"]),
                    ("h_l", r.h_left, cfg["h_l"]),
                    ("h_r", r.h_right, cfg["h_r"]),
                    ("b_l", r.b_left, cfg["b_l"]),
                    ("b_r", r.b_right, cfg["b_r"]),
                    ("z_cl", r.t_stat_conventional, cfg["z"][0]),
                    ("z_bc", r.t_stat_bias_corrected, cfg["z"][1]),
                    ("z_rb", r.t_stat, cfg["z"][2]),
                    ("pv_cl", r.p_value_conventional, cfg["pv"][0]),
                    ("pv_bc", r.p_value_bias_corrected, cfg["pv"][1]),
                    ("pv_rb", r.p_value, cfg["pv"][2]),
                    ("ci_cl_lo", r.conf_int_conventional[0], cfg["ci_lower"][0]),
                    ("ci_cl_hi", r.conf_int_conventional[1], cfg["ci_upper"][0]),
                    ("ci_bc_lo", r.conf_int_bias_corrected[0], cfg["ci_lower"][1]),
                    ("ci_bc_hi", r.conf_int_bias_corrected[1], cfg["ci_upper"][1]),
                    ("ci_rb_lo", r.conf_int[0], cfg["ci_lower"][2]),
                    ("ci_rb_hi", r.conf_int[1], cfg["ci_upper"][2]),
                ]
                if cfg.get("fuzzy_in"):
                    # First-stage three-row block (R's tau_T/se_T/z_T/pv_T/
                    # ci_T layout: rows = Conventional/Bias-Corrected/Robust).
                    pairs += [
                        ("fs_cl", r.first_stage_conventional, cfg["tau_T"][0]),
                        ("fs_bc", r.first_stage, cfg["tau_T"][1]),
                        ("fs_se_cl", r.first_stage_se_conventional, cfg["se_T"][0]),
                        ("fs_se_rb", r.first_stage_se, cfg["se_T"][2]),
                        ("fs_z_cl", r.first_stage_t_stat_conventional, cfg["z_T"][0]),
                        ("fs_z_bc", r.first_stage_t_stat_bias_corrected, cfg["z_T"][1]),
                        ("fs_z_rb", r.first_stage_t_stat, cfg["z_T"][2]),
                        ("fs_pv_cl", r.first_stage_p_value_conventional, cfg["pv_T"][0]),
                        ("fs_pv_bc", r.first_stage_p_value_bias_corrected, cfg["pv_T"][1]),
                        ("fs_pv_rb", r.first_stage_p_value, cfg["pv_T"][2]),
                        (
                            "fs_ci_cl_lo",
                            r.first_stage_conf_int_conventional[0],
                            cfg["ci_T_lower"][0],
                        ),
                        (
                            "fs_ci_cl_hi",
                            r.first_stage_conf_int_conventional[1],
                            cfg["ci_T_upper"][0],
                        ),
                        (
                            "fs_ci_bc_lo",
                            r.first_stage_conf_int_bias_corrected[0],
                            cfg["ci_T_lower"][1],
                        ),
                        (
                            "fs_ci_bc_hi",
                            r.first_stage_conf_int_bias_corrected[1],
                            cfg["ci_T_upper"][1],
                        ),
                        ("fs_ci_rb_lo", r.first_stage_conf_int[0], cfg["ci_T_lower"][2]),
                        ("fs_ci_rb_hi", r.first_stage_conf_int[1], cfg["ci_T_upper"][2]),
                    ]
                for name, got, want in pairs:
                    assert got == pytest.approx(
                        want, rel=RTOL, abs=1e-12
                    ), f"{label}:{name}: {got} vs {want}"
                assert (r.n_h_left, r.n_h_right) == tuple(cfg["N_h"]), label
                assert (r.n_b_left, r.n_b_right) == tuple(cfg["N_b"]), label
                np.testing.assert_allclose(r.beta_p_left, cfg["beta_p_l"], rtol=RTOL, err_msg=label)
                np.testing.assert_allclose(
                    r.beta_p_right, cfg["beta_p_r"], rtol=RTOL, err_msg=label
                )
                if cfg.get("fuzzy_in"):
                    np.testing.assert_allclose(
                        r.beta_t_p_left, cfg["beta_t_p_l"], rtol=RTOL, err_msg=label
                    )
                    np.testing.assert_allclose(
                        r.beta_t_p_right, cfg["beta_t_p_r"], rtol=RTOL, err_msg=label
                    )
                if cfg.get("covs_in"):
                    # coef_covs = R's gamma over the KEPT covariates in
                    # nchar-sorted order; our name-keyed dicts preserve
                    # model order (Python dicts are insertion-ordered), so
                    # values() aligns row-for-row. Row count pins WHICH
                    # columns survived covs_drop.
                    gamma = np.asarray(cfg["coef_covs"], dtype=float)
                    gamma = gamma.reshape(gamma.shape[0], -1)
                    assert r.covariate_coefficients is not None, label
                    assert len(r.covariate_coefficients) == gamma.shape[0], label
                    np.testing.assert_allclose(
                        list(r.covariate_coefficients.values()),
                        gamma[:, 0],
                        rtol=RTOL,
                        err_msg=label,
                    )
                    if cfg.get("fuzzy_in"):
                        assert r.first_stage_covariate_coefficients is not None, label
                        np.testing.assert_allclose(
                            list(r.first_stage_covariate_coefficients.values()),
                            gamma[:, 1],
                            rtol=RTOL,
                            err_msg=label,
                        )
                n_checked += 1
        # 32 configurations; fail loudly if the golden shrinks.
        assert n_checked == 32


class TestSenatePublished2017:
    """Published 2017 Stata Journal numbers - an anchor independent of our
    own R invocation. masspoints='off' reproduces the pre-masspoints-era
    package the paper documents."""

    def test_senate_masspoints_off_matches_stata_journal_2017(self, golden):
        df = _frame(golden, "senate")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = RegressionDiscontinuity(masspoints="off").fit(df, "y", "x")
        # Stata Journal 2017 p. 392 default run (printed precision)
        assert r.att_conventional == pytest.approx(7.416, abs=1e-3)
        assert r.se_conventional == pytest.approx(1.4604, abs=1e-4)
        assert r.h_left == pytest.approx(17.708, abs=1e-3)
        assert r.b_left == pytest.approx(27.984, abs=1e-3)
        assert r.conf_int[0] == pytest.approx(4.09441, abs=1e-4)
        assert r.conf_int[1] == pytest.approx(10.9255, abs=1e-3)
        assert r.t_stat_conventional == pytest.approx(5.0782, abs=1e-3)
        assert r.t_stat == pytest.approx(4.3095, abs=1e-3)
        assert (r.n_h_left, r.n_h_right) == (359, 322)

    def test_senate_default_adjust_matches_golden(self, golden):
        cfg = golden["senate"]["configs"]["adjust"]
        r = _fit(golden, "senate", cfg)
        assert r.att_conventional == pytest.approx(cfg["tau_cl"], rel=RTOL)
        assert r.h_left == pytest.approx(17.754, abs=1e-3)


BW_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "rdrobust_golden.json"
)

BWSELECT_ALL = (
    "mserd",
    "msetwo",
    "msesum",
    "msecomb1",
    "msecomb2",
    "cerrd",
    "certwo",
    "cersum",
    "cercomb1",
    "cercomb2",
)


@pytest.fixture(scope="module")
def bw_golden():
    if not BW_GOLDEN_PATH.exists():
        pytest.skip(
            "Bandwidth golden file not found; run: "
            "Rscript benchmarks/R/generate_rdrobust_golden.R"
        )
    with open(BW_GOLDEN_PATH) as f:
        return json.load(f)


class TestPublicSelectorRouting:
    """End-to-end lock for the public ``bwselect`` surface: each of the 10
    selector options must route the estimator to the matching column of the
    rdbwselect output (bandwidths equal to the PR-1 bandwidth goldens, which
    the ``default`` config generated under the estimator's default settings)
    and complete the fit with a coherent robust row."""

    @pytest.mark.parametrize("selector", BWSELECT_ALL)
    def test_selector_routes_to_golden_bandwidths(self, bw_golden, selector):
        entry = bw_golden["dgp_lee_smooth"]
        cfg = entry["configs"]["default"]
        # The golden 'default' config was generated at the estimator's
        # defaults; guard the premise so a golden regeneration can't
        # silently invalidate this routing lock.
        assert (cfg["p"], cfg["q"], cfg["kernel"]) == (1, 2, "tri")
        assert (cfg["masspoints"], cfg["bwrestrict"]) == ("adjust", True)
        df = pd.DataFrame({"x": entry["x"], "y": entry["y"]})
        r = RegressionDiscontinuity(bwselect=selector).fit(df, "y", "x")
        h_l, h_r, b_l, b_r = cfg["bws"][selector]
        assert r.h_left == pytest.approx(h_l, rel=RTOL)
        assert r.h_right == pytest.approx(h_r, rel=RTOL)
        assert r.b_left == pytest.approx(b_l, rel=RTOL)
        assert r.b_right == pytest.approx(b_r, rel=RTOL)
        assert r.bwselect == selector
        assert np.isfinite(r.att) and np.isfinite(r.se)
        assert r.t_stat == pytest.approx(r.att / r.se, rel=1e-14)
