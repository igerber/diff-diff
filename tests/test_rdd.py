"""API, validation, and behavioral tests for RegressionDiscontinuity."""

import warnings

import numpy as np
import pandas as pd
import pytest

import diff_diff
from diff_diff import RDD, RegressionDiscontinuity, RegressionDiscontinuityResults


def _df(n=200, seed=0, jump=1.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 0.5 * x + jump * (x >= 0) + rng.standard_normal(n) * 0.2
    return pd.DataFrame({"x": x, "y": y})


class TestConstructor:
    def test_constructor_defaults_match_rdrobust(self):
        rd = RegressionDiscontinuity()
        params = rd.get_params()
        assert params == {
            "cutoff": 0.0,
            "p": 1,
            "q": None,
            "kernel": "triangular",
            "bwselect": "mserd",
            "h": None,
            "b": None,
            "rho": None,
            "vcov_type": "nn",
            "nnmatch": 3,
            "masspoints": "adjust",
            "bwcheck": None,
            "bwrestrict": True,
            "scaleregul": 1.0,
            "alpha": 0.05,
        }

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="kernel must be one of"):
            RegressionDiscontinuity(kernel="gaussian")

    def test_invalid_bwselect_raises(self):
        with pytest.raises(ValueError, match="bwselect"):
            RegressionDiscontinuity(bwselect="ik")

    def test_vcov_type_hc_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="vcov_type='nn'"):
            RegressionDiscontinuity(vcov_type="hc1")

    def test_invalid_masspoints_raises(self):
        with pytest.raises(ValueError, match="masspoints"):
            RegressionDiscontinuity(masspoints="on")

    def test_q_le_p_raises(self):
        with pytest.raises(ValueError, match="q must be None"):
            RegressionDiscontinuity(p=2, q=2)

    def test_p_q_bounds_match_rdrobust(self):
        # rdrobust.R:47-57: p and q are single integers in 0:20 with q > p;
        # p=0 (local-constant RD) is inside R's accepted surface.
        RegressionDiscontinuity(p=0)  # q resolves to 1 at fit time
        RegressionDiscontinuity(p=0, q=1)
        with pytest.raises(ValueError, match="p must be an integer in 0..20"):
            RegressionDiscontinuity(p=-1)
        with pytest.raises(ValueError, match="p must be an integer in 0..20"):
            RegressionDiscontinuity(p=21)
        with pytest.raises(ValueError, match="q must be None"):
            RegressionDiscontinuity(p=1, q=21)

    def test_p20_default_q_resolution_matches_r_quirk(self):
        # rdrobust.R:53-57 resolves a NULL q to p+1 BEFORE its validation
        # and never re-checks the default: R accepts p=20 (q -> 21) while
        # rejecting an explicit q=21. Deliberately mirrored (REGISTRY p/q
        # surface note).
        RegressionDiscontinuity(p=20)  # accepted; q resolves to 21 at fit
        with pytest.raises(ValueError, match="q must be None"):
            RegressionDiscontinuity(p=20, q=21)

    def test_bool_rejected_for_integer_knobs(self):
        # bool is an int subclass; p=True must not silently fit p=1.
        for kwargs in (
            {"p": True},
            {"p": 0, "q": True},
            {"nnmatch": True},
            {"bwcheck": True},
        ):
            with pytest.raises(ValueError):
                RegressionDiscontinuity(**kwargs)

    def test_non_numeric_scalars_raise_value_error(self):
        # Strings (even numeric-looking ones) and other non-real types must
        # fail with the estimator's ValueError, not NumPy's TypeError.
        for kwargs in (
            {"cutoff": "0"},
            {"h": "0.5"},
            {"b": "0.5", "h": 0.5},
            {"rho": "2"},
            {"scaleregul": "1"},
            {"alpha": "0.05"},
            {"cutoff": True},
        ):
            with pytest.raises(ValueError):
                RegressionDiscontinuity(**kwargs)

    def test_non_bool_bwrestrict_raises(self):
        # A string like "False" must not silently coerce to truthy and
        # flip bandwidth restriction ON (no-silent-failures policy).
        with pytest.raises(ValueError, match="bwrestrict must be a bool"):
            RegressionDiscontinuity(bwrestrict="False")
        with pytest.raises(ValueError, match="bwrestrict must be a bool"):
            RegressionDiscontinuity(bwrestrict=1)

    def test_nonpositive_h_raises(self):
        with pytest.raises(ValueError, match="h must be None or finite"):
            RegressionDiscontinuity(h=0.0)

    def test_alpha_bounds(self):
        with pytest.raises(ValueError, match="alpha"):
            RegressionDiscontinuity(alpha=1.5)

    def test_kernel_r_spellings_accepted(self):
        df = _df()
        a = RegressionDiscontinuity(kernel="tri").fit(df, "y", "x")
        b = RegressionDiscontinuity(kernel="triangular").fit(df, "y", "x")
        assert a.att == b.att
        assert a.kernel == "triangular"


class TestParamsPlumbing:
    def test_set_params_atomic_on_invalid(self):
        rd = RegressionDiscontinuity()
        with pytest.raises(ValueError):
            rd.set_params(kernel="uniform", alpha=7.0)
        # Dry-run failed -> NOTHING mutated, including the valid kernel.
        assert rd.kernel == "triangular"
        assert rd.alpha == 0.05

    def test_set_params_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            RegressionDiscontinuity().set_params(bandwidth=1.0)

    def test_get_params_roundtrip_q_none(self):
        rd = RegressionDiscontinuity(q=None)
        clone = RegressionDiscontinuity(**rd.get_params())
        assert clone.get_params() == rd.get_params()
        assert clone.q is None

    def test_sklearn_clone(self):
        pytest.importorskip("sklearn")
        from sklearn.base import clone

        rd = RegressionDiscontinuity(cutoff=2.0, kernel="uniform", rho=2.0)
        cloned = clone(rd)
        assert cloned.get_params() == rd.get_params()

    def test_fit_leaves_estimator_stateless(self):
        rd = RegressionDiscontinuity()
        before = rd.get_params()
        rd.fit(_df(), "y", "x")
        assert rd.get_params() == before
        assert set(vars(rd).keys()) == set(before.keys())

    def test_repeat_fit_identical(self):
        rd = RegressionDiscontinuity()
        df = _df()
        r1 = rd.fit(df, "y", "x")
        r2 = rd.fit(df, "y", "x")
        assert r1.att == r2.att and r1.se == r2.se


class TestFitValidation:
    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            RegressionDiscontinuity().fit(_df(), "y", "score")

    def test_nan_rows_warn_and_drop(self):
        df = _df(100)
        df.loc[3, "y"] = np.nan
        df.loc[7, "x"] = np.nan
        with pytest.warns(UserWarning, match="Dropping 2 row"):
            r = RegressionDiscontinuity().fit(df, "y", "x")
        assert r.n_obs == 98
        assert r.n_dropped == 2

    def test_all_obs_one_side_raises(self):
        df = _df(100)
        df["x"] = np.abs(df["x"]) + 1.0
        with pytest.raises(ValueError, match="outside the observed"):
            RegressionDiscontinuity().fit(df, "y", "x")

    def test_cutoff_outside_support_raises(self):
        with pytest.raises(ValueError, match="outside the observed"):
            RegressionDiscontinuity(cutoff=5.0).fit(_df(), "y", "x")

    def test_b_without_h_warns_and_ignored(self):
        df = _df()
        with pytest.warns(UserWarning, match="b= was supplied without h="):
            r = RegressionDiscontinuity(b=0.5).fit(df, "y", "x")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = RegressionDiscontinuity().fit(df, "y", "x")
        assert r.att == ref.att
        assert r.b_left == ref.b_left

    def test_h_with_b_and_rho_warns_rho_wins(self):
        df = _df()
        with pytest.warns(UserWarning, match="rho takes"):
            r = RegressionDiscontinuity(h=0.5, b=0.9, rho=2.0).fit(df, "y", "x")
        assert r.b_left == pytest.approx(0.25)

    def test_rho_without_h_applies_to_selected_bandwidths(self):
        df = _df(500, seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = RegressionDiscontinuity().fit(df, "y", "x")
            r = RegressionDiscontinuity(rho=2.0).fit(df, "y", "x")
        assert r.h_left == base.h_left
        assert r.b_left == pytest.approx(base.h_left / 2.0)

    def test_manual_h_alone_sets_b_equal_h(self):
        r = RegressionDiscontinuity(h=0.4).fit(_df(), "y", "x")
        assert r.h_left == 0.4 and r.b_left == 0.4
        assert r.bwselect == "Manual"

    def test_n_below_20_warns_and_overrides_manual_h(self):
        # rdrobust.R:303-307: the full-range override runs AFTER manual-h
        # resolution and therefore overrides even a user-supplied h.
        df = _df(15, seed=1)
        with pytest.warns(UserWarning, match="entire sample"):
            r = RegressionDiscontinuity(h=0.1).fit(df, "y", "x")
        full = max(abs(df.x.min() - 0.0), abs(df.x.max() - 0.0))
        assert r.h_left == pytest.approx(full)
        assert r.b_left == pytest.approx(full)
        assert r.bwselect == "Manual"

    def test_x_equal_cutoff_assigned_treated_side(self):
        df = _df(200, seed=5)
        df.loc[0, "x"] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = RegressionDiscontinuity().fit(df, "y", "x")
        assert r.n_left + r.n_right == 200
        assert r.n_right == int(np.sum(df.x.values >= 0))

    def test_unsorted_input_matches_sorted(self):
        df = _df(300, seed=7)
        shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)
        a = RegressionDiscontinuity().fit(df, "y", "x")
        b = RegressionDiscontinuity().fit(shuffled, "y", "x")
        assert a.att == pytest.approx(b.att, rel=1e-12)
        assert a.se == pytest.approx(b.se, rel=1e-12)

    def test_int_dtype_running_var(self):
        rng = np.random.default_rng(9)
        x = rng.integers(-50, 50, 400)
        y = 0.02 * x + (x >= 0) + rng.standard_normal(400) * 0.3
        df = pd.DataFrame({"x": x, "y": y})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = RegressionDiscontinuity().fit(df, "y", "x")
        assert np.isfinite(r.att)


class TestResults:
    def test_canonical_identities_hold(self):
        r = RegressionDiscontinuity().fit(_df(500, seed=2), "y", "x")
        assert r.t_stat == pytest.approx(r.att / r.se, rel=1e-14)
        lo, hi = r.conf_int
        assert (lo + hi) / 2 == pytest.approx(r.att, rel=1e-12)
        assert r.se == r.se_robust

    def test_att_is_bias_corrected_not_conventional(self):
        r = RegressionDiscontinuity().fit(_df(500, seed=2), "y", "x")
        # The two estimates differ by the (nonzero) estimated bias.
        assert r.att != r.att_conventional

    def test_summary_contains_three_rows(self):
        r = RegressionDiscontinuity().fit(_df(), "y", "x")
        text = r.summary()
        for token in (
            "Conventional",
            "Bias-Corrected",
            "Robust",
            "att_conventional",
        ):
            assert token in text

    def test_to_dict_ci_split(self):
        r = RegressionDiscontinuity().fit(_df(), "y", "x")
        d = r.to_dict()
        assert d["conf_int_lower"] == r.conf_int[0]
        assert d["conf_int_upper"] == r.conf_int[1]
        assert d["conf_int_conventional_lower"] == r.conf_int_conventional[0]
        assert all(not isinstance(v, (np.floating, np.integer)) for v in d.values())

    def test_to_dataframe_single_row(self):
        r = RegressionDiscontinuity().fit(_df(), "y", "x")
        frame = r.to_dataframe()
        assert frame.shape[0] == 1
        assert frame.loc[0, "att"] == r.att

    def test_results_type(self):
        assert isinstance(
            RegressionDiscontinuity().fit(_df(), "y", "x"),
            RegressionDiscontinuityResults,
        )

    def test_config_echoes_reconstruct_fit(self):
        # Every constructor knob that affects fitting is echoed on the
        # results object (and in to_dict), so a saved result suffices to
        # reconstruct the fit configuration. h/b/rho echo RAW inputs
        # (resolved per-side bandwidths live in h_left/right, b_left/right).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = RegressionDiscontinuity(
                h=0.3, rho=1.5, bwcheck=5, bwrestrict=False, scaleregul=0.5
            ).fit(_df(), "y", "x")
        assert r.h_input == 0.3
        assert r.b_input is None
        assert r.rho_input == 1.5
        assert r.bwcheck == 5
        assert r.bwrestrict is False
        assert r.scaleregul == 0.5
        assert r.bwselect == "Manual"  # resolved label (manual h supplied)
        assert r.b_left == pytest.approx(0.3 / 1.5)  # rho applied
        d = r.to_dict()
        for key in (
            "h_input",
            "b_input",
            "rho_input",
            "bwcheck",
            "bwrestrict",
            "scaleregul",
        ):
            assert key in d
        assert d["b_input"] is None
        assert d["rho_input"] == 1.5

    def test_config_echoes_data_driven_defaults(self):
        r = RegressionDiscontinuity().fit(_df(), "y", "x")
        assert r.h_input is None and r.b_input is None and r.rho_input is None
        assert r.bwcheck is None
        assert r.bwrestrict is True
        assert r.scaleregul == 1.0
        assert r.bwselect == "mserd"


class TestAliasAndExports:
    def test_alias_rdd_identity(self):
        assert RDD is RegressionDiscontinuity

    def test_all_exports(self):
        for name in ("RegressionDiscontinuity", "RegressionDiscontinuityResults", "RDD"):
            assert name in diff_diff.__all__
            assert hasattr(diff_diff, name)
