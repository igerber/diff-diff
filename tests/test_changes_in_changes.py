"""Unit and edge-case tests for ChangesInChanges (CiC) and QDiD.

Covers constructor validation, the sklearn-style param surface (round-trip,
transactional set_params, clone), fit input validation (formula/kwargs,
panel hygiene, NA handling), the NaN-inference contract with n_bootstrap=0,
bootstrap seeding/failure gating, all diagnostic warnings, and the results
API. Methodology verification lives in test_methodology_changes_in_changes.py
and R parity in test_changes_in_changes_parity.py.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import ChangesInChanges, ChangesInChangesResults, QDiD, QDiDResults
from tests.conftest import assert_nan_inference

BOTH = pytest.mark.parametrize("cls", [ChangesInChanges, QDiD], ids=["cic", "qdid"])


def make_2x2(n_treated=60, n_control=80, seed=0, effect=1.0):
    """Full-overlap continuous 2x2 panel (long format, one row per unit-period)."""
    rng = np.random.default_rng(seed)
    n = n_treated + n_control
    treat = np.repeat([1, 0], [n_treated, n_control])
    u = rng.normal(0, 1, n)
    y_pre = u + rng.normal(0, 0.3, n)
    y_post = u + 0.5 + rng.normal(0, 0.3, n) + treat * effect
    return pd.DataFrame(
        {
            "id": np.tile(np.arange(n), 2),
            "post": np.repeat([0, 1], n),
            "treated": np.tile(treat, 2),
            "y": np.concatenate([y_pre, y_post]),
        }
    )


def make_cov_2x2(n_treated=40, n_control=40, seed=0, effect=1.0, x_shift=0.4):
    """2x2 with covariates x1 (relevant, group-shifted composition) and x2 (noise).

    Both the level and the noise scale of y depend on x1, so the conditional
    (quantile-regression) path genuinely differs from the unconditional one.
    Covariates are time-invariant; cells stay small enough that the ~200 LP
    solves per covariate fit keep the suite fast.
    """
    rng = np.random.default_rng(seed)
    n = n_treated + n_control
    treat = np.repeat([1, 0], [n_treated, n_control])
    x1 = rng.uniform(0, 2, n) + x_shift * treat
    x2 = rng.normal(1, 0.5, n)
    u = rng.normal(0, 1, n)
    y_pre = 0.5 + 0.8 * x1 + (0.4 + 0.2 * x1) * u
    y_post = (
        0.8 + 1.1 * x1 + (0.5 + 0.2 * x1) * (0.7 * u + 0.5 * rng.normal(0, 1, n)) + treat * effect
    )
    return pd.DataFrame(
        {
            "id": np.tile(np.arange(n), 2),
            "post": np.repeat([0, 1], n),
            "treated": np.tile(treat, 2),
            "x1": np.tile(x1, 2),
            "x2": np.tile(x2, 2),
            "y": np.concatenate([y_pre, y_post]),
        }
    )


def fit_quiet(est, df, **kwargs):
    """Fit while swallowing the (expected, tested-elsewhere) diagnostic warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit(df, outcome="y", treatment="treated", time="post", **kwargs)


# =============================================================================
# Constructor validation
# =============================================================================


@BOTH
class TestConstructorValidation:
    def test_defaults(self, cls):
        est = cls()
        assert est.quantiles is None
        assert est.n_bootstrap == 200
        assert est.alpha == 0.05
        assert est.panel is False
        assert est.seed is None
        assert est.is_fitted_ is False
        assert est.results_ is None

    @pytest.mark.parametrize("bad", [[], [0.0, 0.5], [0.5, 1.0], [0.5, np.nan], "mid"])
    def test_bad_quantiles(self, cls, bad):
        with pytest.raises((ValueError, TypeError)):
            cls(quantiles=bad)

    @pytest.mark.parametrize("bad", [-1, 2.5, True, "many"])
    def test_bad_n_bootstrap(self, cls, bad):
        with pytest.raises(ValueError, match="n_bootstrap"):
            cls(n_bootstrap=bad)

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1, "small", True])
    def test_bad_alpha(self, cls, bad):
        with pytest.raises(ValueError, match="alpha"):
            cls(alpha=bad)

    @pytest.mark.parametrize("bad", [1, "yes", None])
    def test_bad_panel(self, cls, bad):
        with pytest.raises(ValueError, match="panel"):
            cls(panel=bad)

    @pytest.mark.parametrize("bad", [-1, 1.5, True, "seed"])
    def test_bad_seed(self, cls, bad):
        with pytest.raises(ValueError, match="seed"):
            cls(seed=bad)

    def test_error_message_echoes_value(self, cls):
        with pytest.raises(ValueError, match="got '-3'"):
            cls(n_bootstrap=-3)


# =============================================================================
# get_params / set_params / clone
# =============================================================================


@BOTH
class TestParamSurface:
    def test_get_params_round_trips_init(self, cls):
        est = cls(quantiles=[0.25, 0.5, 0.75], n_bootstrap=10, alpha=0.1, panel=True, seed=3)
        clone = cls(**est.get_params())
        assert clone.get_params() == est.get_params()

    def test_get_params_preserves_raw_none_quantiles(self, cls):
        assert cls().get_params()["quantiles"] is None

    def test_set_params_returns_self(self, cls):
        est = cls()
        assert est.set_params(n_bootstrap=5) is est
        assert est.n_bootstrap == 5

    def test_set_params_unknown_key(self, cls):
        with pytest.raises(ValueError, match="Unknown parameter: iters"):
            cls().set_params(iters=100)

    def test_set_params_transactional(self, cls):
        est = cls(n_bootstrap=100, alpha=0.05)
        with pytest.raises(ValueError):
            est.set_params(n_bootstrap=50, alpha=2.0)
        # The failing batch must not have mutated anything.
        assert est.n_bootstrap == 100
        assert est.alpha == 0.05

    def test_fit_revalidates_after_direct_mutation(self, cls):
        est = cls()
        est.alpha = 5.0  # bypass set_params
        with pytest.raises(ValueError, match="alpha"):
            fit_quiet(est, make_2x2())

    def test_get_params_deep_flag(self, cls):
        # sklearn.base.clone calls get_params(deep=False); the flag is accepted
        # and ignored (no nested estimators) - HAD precedent.
        est = cls(n_bootstrap=9)
        assert est.get_params() == est.get_params(deep=True) == est.get_params(deep=False)

    def test_sklearn_clone_if_available(self, cls):
        sklearn_base = pytest.importorskip("sklearn.base")
        est = cls(quantiles=[0.5], n_bootstrap=7, alpha=0.1, panel=True, seed=11)
        clone = sklearn_base.clone(est)
        assert clone is not est
        assert clone.get_params() == est.get_params()
        assert type(clone) is type(est)


# =============================================================================
# fit() input validation
# =============================================================================


@BOTH
class TestFitValidation:
    def test_requires_formula_or_columns(self, cls):
        with pytest.raises(ValueError, match="formula"):
            cls(n_bootstrap=0).fit(make_2x2(), outcome="y", treatment="treated")

    def test_formula_equals_kwargs(self, cls):
        df = make_2x2(seed=5)
        r_kw = fit_quiet(cls(n_bootstrap=0), df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_f = cls(n_bootstrap=0).fit(df, formula="y ~ treated * post")
            r_f2 = cls(n_bootstrap=0).fit(df, formula="y ~ treated + post + treated:post")
        assert r_f.att == r_kw.att
        assert r_f2.att == r_kw.att
        np.testing.assert_array_equal(
            r_f.quantile_effects["qte"].to_numpy(), r_kw.quantile_effects["qte"].to_numpy()
        )

    def test_formula_interaction_order_invariant(self, cls):
        # 'treated:post' and 'post:treated' are algebraically the same formula;
        # roles come from the main-effect order, so both must give identical
        # results (CiC/QDiD are NOT symmetric in treatment/time).
        df = make_2x2(seed=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_a = cls(n_bootstrap=0).fit(df, formula="y ~ treated + post + treated:post")
            r_b = cls(n_bootstrap=0).fit(df, formula="y ~ treated + post + post:treated")
            r_kw = cls(n_bootstrap=0).fit(df, outcome="y", treatment="treated", time="post")
        assert r_a.att == r_b.att == r_kw.att
        np.testing.assert_array_equal(
            r_a.quantile_effects["qte"].to_numpy(), r_b.quantile_effects["qte"].to_numpy()
        )

    def test_formula_covariates_equal_kwarg(self, cls):
        # Trailing formula terms are covariates and must reproduce the kwarg
        # route exactly (both '*' and ':' grammars).
        df = make_cov_2x2(seed=5)
        r_kw = fit_quiet(cls(n_bootstrap=0), df, covariates=["x1", "x2"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_star = cls(n_bootstrap=0).fit(df, formula="y ~ treated * post + x1 + x2")
            r_colon = cls(n_bootstrap=0).fit(
                df, formula="y ~ treated + post + x1 + x2 + treated:post"
            )
        assert r_star.att == r_kw.att == r_colon.att
        np.testing.assert_array_equal(
            r_star.quantile_effects["qte"].to_numpy(), r_kw.quantile_effects["qte"].to_numpy()
        )
        np.testing.assert_array_equal(
            r_colon.quantile_effects["qte"].to_numpy(), r_kw.quantile_effects["qte"].to_numpy()
        )
        assert r_star.covariates == r_kw.covariates == r_colon.covariates == ["x1", "x2"]

    def test_formula_with_explicit_kwargs_raises(self, cls):
        # Uniform strictness (deliberately stricter than DifferenceInDifferences,
        # which silently lets the formula win): formula + ANY explicit column
        # argument is ambiguous and raises.
        df = make_cov_2x2()
        with pytest.raises(ValueError, match="not both"):
            cls(n_bootstrap=0).fit(df, formula="y ~ treated * post", covariates=["x1"])
        with pytest.raises(ValueError, match="not both"):
            cls(n_bootstrap=0).fit(df, formula="y ~ treated * post", outcome="y")

    def test_colon_form_requires_pair_as_mains(self, cls):
        # 'y ~ treated + treated:post + x1' must not silently fall back to
        # interaction-order role assignment (v1 role-order invariant).
        with pytest.raises(ValueError, match="main effects"):
            cls(n_bootstrap=0).fit(make_cov_2x2(), formula="y ~ treated + treated:post + x1")

    def test_formula_missing_interaction(self, cls):
        with pytest.raises(ValueError, match="interaction"):
            cls(n_bootstrap=0).fit(make_2x2(), formula="y ~ treated + post")

    def test_missing_column(self, cls):
        with pytest.raises(ValueError, match="not found"):
            cls(n_bootstrap=0).fit(make_2x2(), outcome="wage", treatment="treated", time="post")

    def test_non_binary_treatment(self, cls):
        df = make_2x2()
        df.loc[0, "treated"] = 2
        with pytest.raises(ValueError, match="binary"):
            fit_quiet(cls(n_bootstrap=0), df)

    def test_non_binary_time(self, cls):
        df = make_2x2()
        df.loc[0, "post"] = 3
        with pytest.raises(ValueError, match="binary"):
            fit_quiet(cls(n_bootstrap=0), df)

    def test_empty_cell_raises(self, cls):
        # Remove the entire treated pre-period cell (Assumption 5.1(ii)).
        df = make_2x2()
        df = df[~((df["treated"] == 1) & (df["post"] == 0))]
        with pytest.raises(ValueError, match="Assumption 5.1"):
            fit_quiet(cls(n_bootstrap=0), df)

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_nonfinite_outcome_raises(self, cls, bad):
        # dropna() keeps inf; it must be rejected explicitly, never silently
        # corrupt CDFs/quantiles/bootstrap (local review P1).
        df = make_2x2()
        df.loc[3, "y"] = bad
        with pytest.raises(ValueError, match="non-finite"):
            fit_quiet(cls(n_bootstrap=0), df)

    def test_na_rows_dropped_with_warning(self, cls):
        df = make_2x2()
        df.loc[:4, "y"] = np.nan
        with pytest.warns(UserWarning, match="Dropped 5 row"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                res = cls(n_bootstrap=0).fit(df, outcome="y", treatment="treated", time="post")
        assert res.n_obs == len(df) - 5


@BOTH
class TestPanelHygiene:
    def test_panel_requires_unit(self, cls):
        with pytest.raises(ValueError, match="unit"):
            fit_quiet(cls(n_bootstrap=0, panel=True), make_2x2())

    def test_duplicate_unit_period_raises(self, cls):
        df = make_2x2()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            fit_quiet(cls(n_bootstrap=0, panel=True), df, unit="id")

    def test_time_varying_group_raises(self, cls):
        df = make_2x2()
        # Flip one unit's group indicator in the post period only.
        idx = df[(df["id"] == 0) & (df["post"] == 1)].index
        df.loc[idx, "treated"] = 1 - df.loc[idx, "treated"]
        with pytest.raises(ValueError, match="constant within unit"):
            fit_quiet(cls(n_bootstrap=0, panel=True), df, unit="id")

    def test_unbalanced_units_dropped_with_warning(self, cls):
        df = make_2x2()
        df = df.drop(df[(df["id"] == 3) & (df["post"] == 1)].index)
        with pytest.warns(UserWarning, match="balanced-panel"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                res = cls(n_bootstrap=0, panel=True).fit(
                    df, outcome="y", treatment="treated", time="post", unit="id"
                )
        assert res.n_obs == len(make_2x2()) - 2

    def test_unit_ignored_when_rcs(self, cls):
        # Documented-ignore: unit= is a no-op with panel=False (qte idname precedent).
        df = make_2x2(seed=9)
        r1 = fit_quiet(cls(n_bootstrap=0), df, unit="id")
        r2 = fit_quiet(cls(n_bootstrap=0), df)
        assert r1.att == r2.att

    def test_panel_and_rcs_identical_points(self, cls):
        # The point estimator uses only the four marginal cell distributions;
        # panel mode changes the bootstrap, never the estimate.
        df = make_2x2(seed=11)
        r_panel = fit_quiet(cls(n_bootstrap=0, panel=True), df, unit="id")
        r_rcs = fit_quiet(cls(n_bootstrap=0), df)
        assert r_panel.att == r_rcs.att
        np.testing.assert_array_equal(
            r_panel.quantile_effects["qte"].to_numpy(),
            r_rcs.quantile_effects["qte"].to_numpy(),
        )


# =============================================================================
# Inference contract
# =============================================================================


@BOTH
class TestInferenceContract:
    def test_no_bootstrap_nan_inference(self, cls):
        res = fit_quiet(cls(n_bootstrap=0), make_2x2())
        assert np.isfinite(res.att)
        assert_nan_inference(
            {
                "se": res.se,
                "t_stat": res.t_stat,
                "p_value": res.p_value,
                "conf_int": res.conf_int,
            }
        )
        qe = res.quantile_effects
        assert np.all(np.isfinite(qe["qte"]))
        for _, row in qe.iterrows():
            assert_nan_inference(
                {
                    "se": row["se"],
                    "t_stat": row["t_stat"],
                    "p_value": row["p_value"],
                    "conf_int": (row["conf_low"], row["conf_high"]),
                }
            )
        assert np.isnan(res.sup_t_crit)

    def test_bootstrap_produces_finite_inference(self, cls):
        res = fit_quiet(cls(n_bootstrap=60, seed=1), make_2x2(seed=2))
        assert np.isfinite(res.se) and res.se > 0
        assert np.isfinite(res.p_value)
        assert res.conf_int[0] < res.att < res.conf_int[1]
        assert np.isfinite(res.sup_t_crit) and res.sup_t_crit > 0
        assert res.n_bootstrap_valid == 60

    def test_seed_determinism(self, cls):
        df = make_2x2(seed=3)
        r1 = fit_quiet(cls(n_bootstrap=40, seed=7), df)
        r2 = fit_quiet(cls(n_bootstrap=40, seed=7), df)
        r3 = fit_quiet(cls(n_bootstrap=40, seed=8), df)
        assert r1.se == r2.se
        np.testing.assert_array_equal(
            r1.quantile_effects["se"].to_numpy(), r2.quantile_effects["se"].to_numpy()
        )
        assert r1.se != r3.se

    def test_bootstrap_ci_level_follows_alpha(self, cls):
        df = make_2x2(seed=4)
        r05 = fit_quiet(cls(n_bootstrap=50, seed=1, alpha=0.05), df)
        r20 = fit_quiet(cls(n_bootstrap=50, seed=1, alpha=0.20), df)
        assert r05.se == r20.se  # same replicates
        width05 = r05.conf_int[1] - r05.conf_int[0]
        width20 = r20.conf_int[1] - r20.conf_int[0]
        assert width20 < width05


class TestCiCInteriorRangeGuard:
    def test_exterior_quantiles_keep_points_nan_inference(self):
        # Treated-pre support wider than control-pre: eq. 17 interior shrinks.
        rng = np.random.default_rng(0)
        n = 120
        df = pd.DataFrame(
            {
                "id": np.tile(np.arange(n), 2),
                "post": np.repeat([0, 1], n),
                "treated": np.tile(np.repeat([1, 0], [60, 60]), 2),
            }
        )
        y_pre = np.where(df["treated"][:n] == 1, rng.normal(0, 3, n), rng.normal(0, 1, n))
        y_post = y_pre + 0.5 + rng.normal(0, 0.2, n)
        df["y"] = np.concatenate([y_pre, y_post])

        est = ChangesInChanges(n_bootstrap=30, seed=5)
        with pytest.warns(UserWarning, match="interior range"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                res = est.fit(df, outcome="y", treatment="treated", time="post")
        qe = res.quantile_effects
        exterior = ~((qe["quantile"] > res.q_lower) & (qe["quantile"] < res.q_upper))
        assert exterior.any(), "test DGP must produce exterior quantiles"
        # Points survive (qte parity), inference is NaN outside the interior.
        assert np.all(np.isfinite(qe["qte"]))
        assert qe.loc[exterior, "se"].isna().all()
        assert qe.loc[exterior, "p_value"].isna().all()
        assert qe.loc[~exterior, "se"].notna().all()
        # sup_t_crit is computed over ALL columns before the overwrite.
        assert np.isfinite(res.sup_t_crit)

    def test_qdid_has_no_guard(self):
        res = fit_quiet(QDiD(n_bootstrap=0), make_2x2())
        assert np.isnan(res.q_lower) and np.isnan(res.q_upper)


# =============================================================================
# Diagnostic warnings
# =============================================================================


class TestWarnings:
    def test_support_violation_warns_cic(self):
        df = make_2x2(seed=1)
        df.loc[(df["treated"] == 1) & (df["post"] == 0), "y"] += 10  # shift out of support
        with pytest.warns(UserWarning, match="support"):
            ChangesInChanges(n_bootstrap=0).fit(df, outcome="y", treatment="treated", time="post")

    @BOTH
    def test_ties_warn(self, cls):
        df = make_2x2(seed=2)
        df["y"] = np.round(df["y"])  # heavy ties
        with pytest.warns(UserWarning, match="ties"):
            cls(n_bootstrap=0).fit(df, outcome="y", treatment="treated", time="post")

    def test_qdid_non_monotone_warns(self):
        # Crossing quantile curves: control shrinks spread strongly over time
        # while the treated-pre distribution is wide.
        rng = np.random.default_rng(3)
        n = 200
        treat = np.repeat([1, 0], n // 2)
        y_pre = np.where(treat == 1, rng.normal(0, 4, n), rng.normal(0, 3, n))
        y_post = np.where(treat == 1, rng.normal(1, 4, n), rng.normal(0.5, 0.1, n))
        df = pd.DataFrame(
            {
                "post": np.repeat([0, 1], n),
                "treated": np.tile(treat, 2),
                "y": np.concatenate([y_pre, y_post]),
            }
        )
        with pytest.warns(UserWarning, match="non-monotone"):
            QDiD(n_bootstrap=0).fit(df, outcome="y", treatment="treated", time="post")

    def test_clean_data_no_unexpected_warnings(self):
        df = make_2x2(n_treated=150, n_control=150, seed=12)
        # Deterministically shrink the treated pre-period sample strictly inside
        # the control pre-period support so the (legitimate, tested-above)
        # support warning cannot fire on this random draw.
        pre_t = (df["treated"] == 1) & (df["post"] == 0)
        pre_c = (df["treated"] == 0) & (df["post"] == 0)
        lo, hi = df.loc[pre_c, "y"].min(), df.loc[pre_c, "y"].max()
        center = 0.5 * (lo + hi)
        y10 = df.loc[pre_t, "y"]
        scale = 0.4 * (hi - lo) / (y10.max() - y10.min())
        df.loc[pre_t, "y"] = center + (y10 - y10.mean()) * scale
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ChangesInChanges(n_bootstrap=0).fit(df, outcome="y", treatment="treated", time="post")


class TestDegenerateBootstrap:
    def test_zero_spread_replicates_nan_everywhere(self):
        # Constant outcome within every cell: every bootstrap replicate is
        # identical, so replicate SDs are exactly 0. The joint-NaN contract
        # requires the STORED se to be NaN too (not 0.0) so that
        # uniform_bands() cannot emit finite zero-width bands while pointwise
        # inference is NaN (local review P0).
        n = 40
        df = pd.DataFrame(
            {
                "post": np.repeat([0, 1], n),
                "treated": np.tile(np.repeat([1, 0], n // 2), 2),
            }
        )
        df["y"] = 1.0 * df["treated"] + 2.0 * df["post"]  # constant within each cell
        res = fit_quiet(QDiD(n_bootstrap=50, seed=3), df)
        qe = res.quantile_effects
        assert qe["se"].isna().all()
        assert qe["t_stat"].isna().all()
        assert np.isnan(res.se)
        bands = res.uniform_bands()
        assert bands["band_low"].isna().all()
        assert bands["band_high"].isna().all()


class TestBootstrapFailureGate:
    def test_tiny_cells_gate_to_nan(self):
        # 2 obs per cell: pooled RCS resampling frequently empties a cell.
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "post": [0, 0, 1, 1] * 2,
                "treated": [0] * 4 + [1] * 4,
                "y": rng.normal(0, 1, 8),
            }
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = ChangesInChanges(n_bootstrap=200, seed=0).fit(
                df, outcome="y", treatment="treated", time="post"
            )
        messages = [str(w.message) for w in rec]
        assert any("bootstrap iterations succeeded" in m for m in messages)
        assert res.n_bootstrap_valid < 200
        assert np.isfinite(res.att)  # point estimate unaffected


# =============================================================================
# Covariates (qte xformla parity path)
# =============================================================================


@BOTH
class TestCovariates:
    def test_happy_path(self, cls):
        res = fit_quiet(cls(n_bootstrap=0), make_cov_2x2(), covariates=["x1"])
        assert np.isfinite(res.att)
        assert np.all(np.isfinite(res.quantile_effects["qte"]))
        assert res.covariates == ["x1"]

    def test_covariates_change_the_estimate(self, cls):
        # x1 is genuinely relevant (group-shifted composition), so the
        # conditional and unconditional estimates must differ.
        df = make_cov_2x2(seed=8)
        r_cov = fit_quiet(cls(n_bootstrap=0), df, covariates=["x1"])
        r_unc = fit_quiet(cls(n_bootstrap=0), df)
        assert r_cov.att != r_unc.att

    def test_empty_list_behaves_as_none(self, cls):
        df = make_cov_2x2(seed=9)
        r_empty = fit_quiet(cls(n_bootstrap=0), df, covariates=[])
        r_none = fit_quiet(cls(n_bootstrap=0), df)
        assert r_empty.att == r_none.att
        assert r_empty.covariates is None

    def test_missing_covariate_column(self, cls):
        with pytest.raises(ValueError, match="Covariate column 'age' not found"):
            fit_quiet(cls(n_bootstrap=0), make_cov_2x2(), covariates=["age"])

    def test_bare_string_covariates_rejected(self, cls):
        # covariates="x1" would iterate character-wise (["x", "1"]) and could
        # silently fit the wrong covariate set if such columns exist.
        with pytest.raises(ValueError, match="bare string"):
            fit_quiet(cls(n_bootstrap=0), make_cov_2x2(), covariates="x1")

    def test_non_numeric_covariate(self, cls):
        df = make_cov_2x2()
        df["region"] = np.where(df["x1"] > 1, "north", "south")
        with pytest.raises(ValueError, match="dummy-encode"):
            fit_quiet(cls(n_bootstrap=0), df, covariates=["region"])

    def test_reserved_name_collision(self, cls):
        with pytest.raises(ValueError, match="collide"):
            fit_quiet(cls(n_bootstrap=0), make_cov_2x2(), covariates=["y"])

    def test_duplicate_covariates(self, cls):
        with pytest.raises(ValueError, match="duplicate"):
            fit_quiet(cls(n_bootstrap=0), make_cov_2x2(), covariates=["x1", "x1"])

    def test_na_in_covariate_rows_dropped_with_warning(self, cls):
        df = make_cov_2x2()
        df.loc[:2, "x1"] = np.nan
        with pytest.warns(UserWarning, match="Dropped 3 row"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                res = cls(n_bootstrap=0).fit(
                    df, outcome="y", treatment="treated", time="post", covariates=["x1"]
                )
        assert res.n_obs == len(df) - 3

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_nonfinite_covariate_raises(self, cls, bad):
        df = make_cov_2x2()
        df.loc[3, "x1"] = bad
        with pytest.raises(ValueError, match="non-finite"):
            fit_quiet(cls(n_bootstrap=0), df, covariates=["x1"])

    def test_too_few_obs_per_qr_cell(self, cls):
        # 3 rows in a quantile-regression cell with 2 covariates: below k+2.
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "post": [0, 0, 0, 1, 1, 1] * 2,
                "treated": [0] * 6 + [1] * 6,
                "x1": rng.normal(0, 1, 12),
                "x2": rng.normal(0, 1, 12),
                "y": rng.normal(0, 1, 12),
            }
        )
        with pytest.raises(ValueError, match="Too few observations for quantile regression"):
            fit_quiet(cls(n_bootstrap=0), df, covariates=["x1", "x2"])

    def test_constant_covariate_runs_finite(self, cls):
        # Zero-variance covariate: the LP still solves (collinear direction is
        # flat, on-span predictions stay determined - qte/rq behave the same);
        # no proactive error, results finite.
        df = make_cov_2x2(seed=10)
        df["x1"] = 1.5
        res = fit_quiet(cls(n_bootstrap=0), df, covariates=["x1"])
        assert np.isfinite(res.att)
        assert np.all(np.isfinite(res.quantile_effects["qte"]))

    def test_interaction_order_invariant_with_covariates(self, cls):
        df = make_cov_2x2(seed=11)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_a = cls(n_bootstrap=0).fit(df, formula="y ~ treated + post + x1 + treated:post")
            r_b = cls(n_bootstrap=0).fit(df, formula="y ~ treated + post + x1 + post:treated")
        assert r_a.att == r_b.att
        assert r_a.covariates == r_b.covariates == ["x1"]

    def test_panel_and_rcs_identical_points(self, cls):
        # Panel mode changes only the bootstrap, never the point estimate -
        # also true on the covariate path (qte parity: identical ate observed
        # in the golden generator for cov scenarios).
        df = make_cov_2x2(seed=12)
        r_panel = fit_quiet(cls(n_bootstrap=0, panel=True), df, unit="id", covariates=["x1"])
        r_rcs = fit_quiet(cls(n_bootstrap=0), df, covariates=["x1"])
        assert r_panel.att == r_rcs.att
        np.testing.assert_array_equal(
            r_panel.quantile_effects["qte"].to_numpy(),
            r_rcs.quantile_effects["qte"].to_numpy(),
        )

    def test_seed_determinism_with_covariates(self, cls):
        df = make_cov_2x2(n_treated=25, n_control=25, seed=13)
        r1 = fit_quiet(cls(n_bootstrap=12, seed=7), df, covariates=["x1"])
        r2 = fit_quiet(cls(n_bootstrap=12, seed=7), df, covariates=["x1"])
        assert r1.se == r2.se
        np.testing.assert_array_equal(
            r1.quantile_effects["se"].to_numpy(), r2.quantile_effects["se"].to_numpy()
        )

    def test_treated_post_covariates_unused(self, cls):
        # Neither estimator consumes x11 (CiC: QR in c00/c01, ranks at c10;
        # QDiD: QR in c00/c01/c10): perturbing treated post-period covariate
        # values must leave everything unchanged. Locks the cells contract.
        df = make_cov_2x2(seed=14)
        r_base = fit_quiet(cls(n_bootstrap=0), df, covariates=["x1"])
        df2 = df.copy()
        mask = (df2["treated"] == 1) & (df2["post"] == 1)
        df2.loc[mask, "x1"] = df2.loc[mask, "x1"] + 100.0
        r_pert = fit_quiet(cls(n_bootstrap=0), df2, covariates=["x1"])
        assert r_base.att == r_pert.att
        np.testing.assert_array_equal(
            r_base.quantile_effects["qte"].to_numpy(),
            r_pert.quantile_effects["qte"].to_numpy(),
        )

    def test_time_varying_covariates_panel_ok(self, cls):
        # Each (group, period) cell uses its own rows' covariate values (qte
        # parity), so time-varying covariates are legal in panel mode.
        df = make_cov_2x2(seed=15)
        rng = np.random.default_rng(15)
        post_mask = df["post"] == 1
        df.loc[post_mask, "x1"] = df.loc[post_mask, "x1"] + rng.normal(0, 0.2, post_mask.sum())
        res = fit_quiet(cls(n_bootstrap=0, panel=True), df, unit="id", covariates=["x1"])
        assert np.isfinite(res.att)

    def test_bootstrap_failure_gate_with_covariates(self, cls):
        # Tiny cells + pooled RCS resampling: replicates frequently empty a
        # cell or degenerate the QR - they must NaN-row through the gate,
        # never raise.
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "post": [0, 0, 0, 1, 1, 1] * 2,
                "treated": [0] * 6 + [1] * 6,
                "x1": rng.normal(0, 1, 12),
                "y": rng.normal(0, 1, 12),
            }
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = cls(n_bootstrap=100, seed=0).fit(
                df, outcome="y", treatment="treated", time="post", covariates=["x1"]
            )
        assert np.isfinite(res.att)
        assert res.n_bootstrap_valid <= 100
        del rec  # gate warning may or may not fire depending on failure share

    def test_no_bootstrap_nan_inference_with_covariates(self, cls):
        res = fit_quiet(cls(n_bootstrap=0), make_cov_2x2(), covariates=["x1"])
        assert np.isfinite(res.att)
        assert_nan_inference(
            {
                "se": res.se,
                "t_stat": res.t_stat,
                "p_value": res.p_value,
                "conf_int": res.conf_int,
            }
        )

    def test_results_surface(self, cls):
        res = fit_quiet(cls(n_bootstrap=0), make_cov_2x2(), covariates=["x1", "x2"])
        assert res.covariates == ["x1", "x2"]
        assert res.to_dict()["covariates"] == ["x1", "x2"]
        assert "Covariates: x1, x2" in res.summary()
        # Unconditional fits keep covariates=None (and no summary line).
        res_unc = fit_quiet(cls(n_bootstrap=0), make_cov_2x2())
        assert res_unc.covariates is None
        assert res_unc.to_dict()["covariates"] is None
        assert "Covariates:" not in res_unc.summary()

    def test_get_params_has_no_covariates_key(self, cls):
        # covariates is fit-scope (like unit), not a hyperparameter.
        assert "covariates" not in cls().get_params()

    def test_sklearn_clone_then_fit_with_covariates(self, cls):
        sklearn_base = pytest.importorskip("sklearn.base")
        est = cls(n_bootstrap=0)
        clone = sklearn_base.clone(est)
        res = fit_quiet(clone, make_cov_2x2(), covariates=["x1"])
        assert np.isfinite(res.att)


class TestCovariateDiagnostics:
    def test_no_unconditional_guard_under_covariates(self):
        # LOCK TEST for the NaN-exterior hazard: with covariates,
        # q_lower/q_upper are NaN and the unconditional interior-range
        # overwrite must NOT run - otherwise the all-True NaN mask would
        # silently NaN every quantile's inference.
        df = make_cov_2x2(n_treated=30, n_control=30, seed=16)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = ChangesInChanges(n_bootstrap=20, seed=2).fit(
                df, outcome="y", treatment="treated", time="post", covariates=["x1"]
            )
        assert np.isnan(res.q_lower) and np.isnan(res.q_upper)
        assert not any("interior range" in str(w.message) for w in rec)
        qe = res.quantile_effects
        assert qe["se"].notna().all()
        assert qe["t_stat"].notna().all()
        assert qe["p_value"].notna().all()

    def test_envelope_warning_fires_on_support_failure(self):
        # Treated pre-period outcomes shifted far outside the conditional
        # control pre-period distribution -> MS Assumption-4 diagnostic.
        df = make_cov_2x2(seed=17)
        mask = (df["treated"] == 1) & (df["post"] == 0)
        df.loc[mask, "y"] = df.loc[mask, "y"] + 25.0
        with pytest.warns(UserWarning, match="conditional quantile envelope"):
            ChangesInChanges(n_bootstrap=0).fit(
                df, outcome="y", treatment="treated", time="post", covariates=["x1"]
            )

    def test_unconditional_support_warning_not_used_under_covariates(self):
        # The same shifted DGP must NOT emit the unconditional Assumption-3.4
        # message on the covariate path (the conditional diagnostic replaces it).
        df = make_cov_2x2(seed=17)
        mask = (df["treated"] == 1) & (df["post"] == 0)
        df.loc[mask, "y"] = df.loc[mask, "y"] + 25.0
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ChangesInChanges(n_bootstrap=0).fit(
                df, outcome="y", treatment="treated", time="post", covariates=["x1"]
            )
        messages = [str(w.message) for w in rec]
        # "Corollary 3.1" appears only in the UNCONDITIONAL support warning
        # (the conditional-envelope message references Assumption 3.4 as its
        # analogue, so matching on that string would be ambiguous).
        assert not any("Corollary 3.1" in m for m in messages)
        assert any("conditional quantile envelope" in m for m in messages)

    def test_clean_covariate_fit_no_unexpected_warnings(self):
        # Full-overlap covariate DGP (no group composition shift): none of the
        # diagnostics may fire. ~2% envelope share is expected by construction;
        # the 10% threshold must not be trigger-happy on clean data.
        df = make_cov_2x2(n_treated=150, n_control=150, seed=18, x_shift=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ChangesInChanges(n_bootstrap=0).fit(
                df, outcome="y", treatment="treated", time="post", covariates=["x1"]
            )

    def test_ties_warning_still_fires_with_covariates(self):
        df = make_cov_2x2(seed=19)
        df["y"] = np.round(df["y"])
        with pytest.warns(UserWarning, match="ties"):
            ChangesInChanges(n_bootstrap=0).fit(
                df, outcome="y", treatment="treated", time="post", covariates=["x1"]
            )

    def test_qdid_monotonicity_check_skipped_with_covariates(self):
        # The unconditional-QDiD footnote-21 warning diagnoses the additive
        # quantile model, which is not the object estimated under covariates
        # (the imputed counterfactual's quantile curve is monotone by
        # construction) - it must not fire.
        rng = np.random.default_rng(3)
        n = 200
        treat = np.repeat([1, 0], n // 2)
        x1 = rng.uniform(0, 1, n)
        y_pre = np.where(treat == 1, rng.normal(0, 4, n), rng.normal(0, 3, n))
        y_post = np.where(treat == 1, rng.normal(1, 4, n), rng.normal(0.5, 0.1, n))
        df = pd.DataFrame(
            {
                "post": np.repeat([0, 1], n),
                "treated": np.tile(treat, 2),
                "x1": np.tile(x1, 2),
                "y": np.concatenate([y_pre, y_post]),
            }
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            QDiD(n_bootstrap=0).fit(
                df, outcome="y", treatment="treated", time="post", covariates=["x1"]
            )
        assert not any("non-monotone" in str(w.message) for w in rec)


# =============================================================================
# Results API
# =============================================================================


@BOTH
class TestResultsAPI:
    def test_results_types_and_fields(self, cls):
        res = fit_quiet(cls(n_bootstrap=25, seed=1), make_2x2())
        assert isinstance(res, ChangesInChangesResults)
        assert res.estimator == ("cic" if cls is ChangesInChanges else "qdid")
        assert res.cell_sizes == {
            "control_pre": 80,
            "control_post": 80,
            "treated_pre": 60,
            "treated_post": 60,
        }
        assert len(res.quantile_effects) == 19  # default grid

    def test_summary_smoke(self, cls):
        res = fit_quiet(cls(n_bootstrap=25, seed=1), make_2x2())
        text = res.summary()
        assert "ATT" in text
        assert "Quantile treatment effects" in text

    def test_to_dict_keys(self, cls):
        d = fit_quiet(cls(n_bootstrap=0), make_2x2()).to_dict()
        for key in ("att", "se", "conf_int_lower", "conf_int_upper", "estimator", "panel"):
            assert key in d
        assert d["inference_method"] == "none"

    def test_to_dataframe_levels(self, cls):
        res = fit_quiet(cls(n_bootstrap=0), make_2x2())
        assert len(res.to_dataframe("quantiles")) == 19
        assert len(res.to_dataframe("att")) == 1
        with pytest.raises(ValueError, match="level"):
            res.to_dataframe("horizons")

    def test_uniform_bands(self, cls):
        res = fit_quiet(cls(n_bootstrap=40, seed=2), make_2x2())
        bands = res.uniform_bands()
        finite = res.quantile_effects["se"].notna()
        # Sup-t bands are at least as wide as pointwise CIs on finite rows.
        assert np.all(
            bands.loc[finite, "band_low"].to_numpy()
            <= res.quantile_effects.loc[finite, "conf_low"].to_numpy() + 1e-12
        )

    def test_fitted_state(self, cls):
        est = cls(n_bootstrap=0)
        assert not est.is_fitted_
        res = fit_quiet(est, make_2x2())
        assert est.is_fitted_
        assert est.results_ is res

    def test_custom_quantile_grid(self, cls):
        res = fit_quiet(cls(quantiles=[0.25, 0.5, 0.75], n_bootstrap=0), make_2x2())
        np.testing.assert_array_equal(
            res.quantile_effects["quantile"].to_numpy(), [0.25, 0.5, 0.75]
        )


def test_qdid_results_alias():
    assert QDiDResults is ChangesInChangesResults
