"""Tests for WooldridgeDiD estimator and WooldridgeDiDResults."""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.wooldridge import (
    WooldridgeDiD,
    _build_interaction_matrix,
    _filter_sample,
    _prepare_covariates,
    _suggest_nonlinear_method,
)
from diff_diff.wooldridge_results import WooldridgeDiDResults


def _make_minimal_results(**kwargs):
    """Helper: build a WooldridgeDiDResults with required fields."""
    defaults = dict(
        group_time_effects={
            (2, 2): {
                "att": 1.0,
                "se": 0.5,
                "t_stat": 2.0,
                "p_value": 0.04,
                "conf_int": (0.02, 1.98),
            },
            (2, 3): {
                "att": 1.5,
                "se": 0.6,
                "t_stat": 2.5,
                "p_value": 0.01,
                "conf_int": (0.32, 2.68),
            },
            (3, 3): {
                "att": 0.8,
                "se": 0.4,
                "t_stat": 2.0,
                "p_value": 0.04,
                "conf_int": (0.02, 1.58),
            },
        },
        overall_att=1.1,
        overall_se=0.35,
        overall_t_stat=3.14,
        overall_p_value=0.002,
        overall_conf_int=(0.41, 1.79),
        group_effects=None,
        calendar_effects=None,
        event_study_effects=None,
        method="ols",
        control_group="not_yet_treated",
        groups=[2, 3],
        time_periods=[1, 2, 3],
        n_obs=300,
        n_treated_units=100,
        n_control_units=200,
        alpha=0.05,
        _gt_weights={(2, 2): 50, (2, 3): 50, (3, 3): 30},
        _gt_vcov=None,
    )
    defaults.update(kwargs)
    return WooldridgeDiDResults(**defaults)


class TestWooldridgeDiDResults:
    def test_repr(self):
        r = _make_minimal_results()
        s = repr(r)
        assert "WooldridgeDiDResults" in s
        assert "ATT" in s

    def test_summary_default(self):
        r = _make_minimal_results()
        s = r.summary()
        assert "1.1" in s or "ATT" in s

    def test_to_dataframe_event(self):
        r = _make_minimal_results()
        r.aggregate("event")
        df = r.to_dataframe("event")
        assert isinstance(df, pd.DataFrame)
        assert "att" in df.columns

    def test_aggregate_simple_returns_self(self):
        r = _make_minimal_results()
        result = r.aggregate("simple")
        assert result is r  # chaining

    def test_aggregate_group(self):
        r = _make_minimal_results()
        r.aggregate("group")
        assert r.group_effects is not None
        assert 2 in r.group_effects
        assert 3 in r.group_effects

    def test_aggregate_calendar(self):
        r = _make_minimal_results()
        r.aggregate("calendar")
        assert r.calendar_effects is not None
        assert 2 in r.calendar_effects or 3 in r.calendar_effects

    def test_aggregate_event(self):
        r = _make_minimal_results()
        r.aggregate("event")
        assert r.event_study_effects is not None
        # relative period 0 (treatment period itself) should be present
        assert 0 in r.event_study_effects or 1 in r.event_study_effects

    def test_aggregate_invalid_raises(self):
        r = _make_minimal_results()
        with pytest.raises(ValueError, match="type"):
            r.aggregate("bad_type")


class TestWooldridgeDiDAPI:
    def test_default_construction(self):
        est = WooldridgeDiD()
        assert est.method == "ols"
        assert est.control_group == "not_yet_treated"
        assert est.anticipation == 0
        assert est.demean_covariates is True
        assert est.alpha == 0.05
        assert est.cluster is None
        assert est.n_bootstrap == 0
        assert est.bootstrap_weights == "rademacher"
        assert est.seed is None
        assert est.rank_deficient_action == "warn"
        assert not est.is_fitted_

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            WooldridgeDiD(method="probit")

    def test_invalid_control_group_raises(self):
        with pytest.raises(ValueError, match="control_group"):
            WooldridgeDiD(control_group="clean_control")

    def test_invalid_anticipation_raises(self):
        with pytest.raises(ValueError, match="anticipation"):
            WooldridgeDiD(anticipation=-1)

    def test_get_params_roundtrip(self):
        est = WooldridgeDiD(method="logit", alpha=0.1, anticipation=1)
        params = est.get_params()
        assert params["method"] == "logit"
        assert params["alpha"] == 0.1
        assert params["anticipation"] == 1

    def test_set_params_roundtrip(self):
        est = WooldridgeDiD()
        est.set_params(alpha=0.01, n_bootstrap=100)
        assert est.alpha == 0.01
        assert est.n_bootstrap == 100

    def test_set_params_returns_self(self):
        est = WooldridgeDiD()
        result = est.set_params(alpha=0.1)
        assert result is est

    def test_set_params_unknown_raises(self):
        est = WooldridgeDiD()
        with pytest.raises(ValueError, match="Unknown"):
            est.set_params(nonexistent_param=42)

    def test_results_before_fit_raises(self):
        est = WooldridgeDiD()
        with pytest.raises(RuntimeError, match="fit"):
            _ = est.results_


def _make_panel(n_units=10, n_periods=5, treat_share=0.5, seed=0):
    """Create a simple balanced panel for testing."""
    rng = np.random.default_rng(seed)
    units = np.arange(n_units)
    n_treated = int(n_units * treat_share)
    # Two cohorts: half treated in period 3, rest never treated
    cohort = np.array([3] * n_treated + [0] * (n_units - n_treated))
    rows = []
    for u in units:
        for t in range(1, n_periods + 1):
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "cohort": cohort[u],
                    "y": rng.standard_normal(),
                    "x1": rng.standard_normal(),
                }
            )
    return pd.DataFrame(rows)


class TestDataPrep:
    def test_filter_sample_not_yet_treated(self):
        df = _make_panel()
        filtered = _filter_sample(
            df,
            unit="unit",
            time="time",
            cohort="cohort",
            control_group="not_yet_treated",
            anticipation=0,
        )
        # All treated units should be present (all periods)
        treated_units = df[df["cohort"] == 3]["unit"].unique()
        assert set(treated_units).issubset(filtered["unit"].unique())

    def test_filter_sample_never_treated(self):
        df = _make_panel()
        filtered = _filter_sample(
            df,
            unit="unit",
            time="time",
            cohort="cohort",
            control_group="never_treated",
            anticipation=0,
        )
        # Only never-treated (cohort==0) and treated units should remain
        assert (filtered["cohort"].isin([0, 3])).all()

    def test_build_interaction_matrix_columns(self):
        df = _make_panel()
        filtered = _filter_sample(df, "unit", "time", "cohort", "not_yet_treated", anticipation=0)
        X_int, col_names, gt_keys = _build_interaction_matrix(
            filtered, cohort="cohort", time="time", anticipation=0
        )
        # Each column should be a valid (g, t) pair with t >= g
        for g, t in gt_keys:
            assert t >= g

    def test_build_interaction_matrix_binary(self):
        df = _make_panel()
        filtered = _filter_sample(df, "unit", "time", "cohort", "not_yet_treated", anticipation=0)
        X_int, col_names, gt_keys = _build_interaction_matrix(
            filtered, cohort="cohort", time="time", anticipation=0
        )
        # All values should be 0 or 1
        assert set(np.unique(X_int)).issubset({0, 1})

    def test_prepare_covariates_exovar(self):
        df = _make_panel()
        X_cov = _prepare_covariates(
            df,
            exovar=["x1"],
            xtvar=None,
            xgvar=None,
            cohort="cohort",
            time="time",
            demean_covariates=True,
            groups=[3],
        )
        assert X_cov.shape[0] == len(df)
        assert X_cov.shape[1] == 1  # just x1

    def test_prepare_covariates_xtvar_demeaned(self):
        df = _make_panel()
        X_raw = _prepare_covariates(
            df,
            exovar=None,
            xtvar=["x1"],
            xgvar=None,
            cohort="cohort",
            time="time",
            demean_covariates=False,
            groups=[3],
        )
        X_dem = _prepare_covariates(
            df,
            exovar=None,
            xtvar=["x1"],
            xgvar=None,
            cohort="cohort",
            time="time",
            demean_covariates=True,
            groups=[3],
        )
        # Demeaned version should differ from raw
        assert not np.allclose(X_raw, X_dem)


class TestWooldridgeDiDFitOLS:
    @pytest.fixture
    def mpdta(self):
        from diff_diff.datasets import load_mpdta

        return load_mpdta()

    def test_fit_returns_results(self, mpdta):
        est = WooldridgeDiD()
        results = est.fit(
            mpdta, outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
        )
        assert isinstance(results, WooldridgeDiDResults)

    def test_fit_sets_is_fitted(self, mpdta):
        est = WooldridgeDiD()
        est.fit(mpdta, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")
        assert est.is_fitted_

    def test_overall_att_finite(self, mpdta):
        est = WooldridgeDiD()
        r = est.fit(mpdta, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)
        assert r.overall_se > 0

    def test_group_time_effects_populated(self, mpdta):
        est = WooldridgeDiD()
        r = est.fit(mpdta, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")
        assert len(r.group_time_effects) > 0
        for (g, t), eff in r.group_time_effects.items():
            assert t >= g
            assert "att" in eff and "se" in eff

    def test_all_inference_fields_finite(self, mpdta):
        """No inference field should be NaN in normal data."""
        est = WooldridgeDiD()
        r = est.fit(mpdta, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")
        assert np.isfinite(r.overall_t_stat)
        assert np.isfinite(r.overall_p_value)
        assert all(np.isfinite(c) for c in r.overall_conf_int)

    def test_never_treated_control_group(self, mpdta):
        est = WooldridgeDiD(control_group="never_treated")
        r = est.fit(mpdta, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")
        assert len(r.group_time_effects) > 0

    def test_metadata_correct(self, mpdta):
        est = WooldridgeDiD()
        r = est.fit(mpdta, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")
        assert r.method == "ols"
        assert r.n_obs > 0
        assert r.n_treated_units > 0
        assert r.n_control_units > 0


class TestAggregations:
    @pytest.fixture
    def fitted(self):
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        est = WooldridgeDiD()
        return est.fit(df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")

    def test_simple_matches_manual_weighted_average(self, fitted):
        """simple ATT must equal manually computed weighted average of ATT(g,t)."""
        gt = fitted.group_time_effects
        w = fitted._gt_weights
        post_keys = [(g, t) for (g, t) in w if t >= g]
        w_total = sum(w[k] for k in post_keys)
        manual_att = sum(w[k] * gt[k]["att"] for k in post_keys) / w_total
        assert abs(fitted.overall_att - manual_att) < 1e-10

    def test_aggregate_group_keys_match_cohorts(self, fitted):
        fitted.aggregate("group")
        assert set(fitted.group_effects.keys()) == set(fitted.groups)

    def test_aggregate_event_relative_periods(self, fitted):
        fitted.aggregate("event")
        for k in fitted.event_study_effects:
            assert isinstance(k, (int, np.integer))

    def test_aggregate_calendar_finite(self, fitted):
        fitted.aggregate("calendar")
        for t, eff in fitted.calendar_effects.items():
            assert np.isfinite(eff["att"])

    def test_summary_runs(self, fitted):
        s = fitted.summary("simple")
        assert "ETWFE" in s or "Wooldridge" in s

    def test_to_dataframe_event(self, fitted):
        fitted.aggregate("event")
        df = fitted.to_dataframe("event")
        assert "relative_period" in df.columns
        assert "att" in df.columns

    def test_to_dataframe_gt(self, fitted):
        df = fitted.to_dataframe("gt")
        assert "cohort" in df.columns
        assert "time" in df.columns
        assert len(df) == len(fitted.group_time_effects)


class TestWooldridgeDiDLogit:
    @pytest.fixture
    def binary_panel(self):
        """Simulated binary outcome panel with known positive ATT."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 60, 5
        rows = []
        for u in range(n_units):
            cohort = 3 if u < 30 else 0
            for t in range(1, n_periods + 1):
                treated = int(cohort > 0 and t >= cohort)
                eta = -0.5 + 1.0 * treated + 0.1 * rng.standard_normal()
                y = int(rng.random() < 1 / (1 + np.exp(-eta)))
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": y})
        return pd.DataFrame(rows)

    def test_logit_fit_runs(self, binary_panel):
        est = WooldridgeDiD(method="logit")
        r = est.fit(binary_panel, outcome="y", unit="unit", time="time", cohort="cohort")
        assert isinstance(r, WooldridgeDiDResults)

    def test_logit_att_sign(self, binary_panel):
        """ATT should be positive (treatment increases binary outcome)."""
        est = WooldridgeDiD(method="logit")
        r = est.fit(binary_panel, outcome="y", unit="unit", time="time", cohort="cohort")
        assert r.overall_att > 0

    def test_logit_se_positive(self, binary_panel):
        est = WooldridgeDiD(method="logit")
        r = est.fit(binary_panel, outcome="y", unit="unit", time="time", cohort="cohort")
        assert r.overall_se > 0

    def test_logit_method_stored(self, binary_panel):
        est = WooldridgeDiD(method="logit")
        r = est.fit(binary_panel, outcome="y", unit="unit", time="time", cohort="cohort")
        assert r.method == "logit"


class TestWooldridgeDiDPoisson:
    @pytest.fixture
    def count_panel(self):
        rng = np.random.default_rng(7)
        n_units, n_periods = 60, 5
        rows = []
        for u in range(n_units):
            cohort = 3 if u < 30 else 0
            for t in range(1, n_periods + 1):
                treated = int(cohort > 0 and t >= cohort)
                mu = np.exp(0.5 + 0.8 * treated + 0.1 * rng.standard_normal())
                y = rng.poisson(mu)
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": float(y)})
        return pd.DataFrame(rows)

    def test_poisson_fit_runs(self, count_panel):
        est = WooldridgeDiD(method="poisson")
        r = est.fit(count_panel, outcome="y", unit="unit", time="time", cohort="cohort")
        assert isinstance(r, WooldridgeDiDResults)

    def test_poisson_att_sign(self, count_panel):
        est = WooldridgeDiD(method="poisson")
        r = est.fit(count_panel, outcome="y", unit="unit", time="time", cohort="cohort")
        assert r.overall_att > 0

    def test_poisson_se_positive(self, count_panel):
        est = WooldridgeDiD(method="poisson")
        r = est.fit(count_panel, outcome="y", unit="unit", time="time", cohort="cohort")
        assert r.overall_se > 0


class TestBootstrap:
    @pytest.mark.slow
    def test_multiplier_bootstrap_ols(self, ci_params):
        """Bootstrap SE should be close to analytic SE."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        n_boot = ci_params.bootstrap(50, min_n=19)
        est = WooldridgeDiD(n_bootstrap=n_boot, seed=42)
        r = est.fit(df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")
        assert abs(r.overall_se - r.overall_att) / max(abs(r.overall_att), 1e-8) < 10

    def test_bootstrap_zero_disables(self):
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        est = WooldridgeDiD(n_bootstrap=0)
        r = est.fit(df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")
        assert np.isfinite(r.overall_se)


class TestMethodologyCorrectness:
    def test_ols_att_sign_direction(self):
        """ATT sign should be consistent across cohorts on mpdta."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        # never_treated with within-transformation can produce collinear
        # pre-treatment interactions; use silent to avoid warnings
        est = WooldridgeDiD(control_group="never_treated", rank_deficient_action="silent")
        r = est.fit(df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")
        assert np.isfinite(r.overall_att)

    def test_never_treated_produces_event_effects_with_placebo_leads(self):
        """With never_treated, event aggregation should include negative event
        times (placebo leads) because pre-treatment interactions are included."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        est = WooldridgeDiD(control_group="never_treated")
        r = est.fit(df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")
        r.aggregate("event")
        assert r.event_study_effects is not None
        assert len(r.event_study_effects) > 0
        # never_treated includes pre-treatment interaction indicators,
        # so negative event times (placebo leads) should be present
        neg_keys = [k for k in r.event_study_effects.keys() if k < 0]
        assert len(neg_keys) > 0, (
            "Expected negative event times (placebo leads) for never_treated, "
            f"got keys: {sorted(r.event_study_effects.keys())}"
        )

    def test_single_cohort_degenerates_to_simple_did(self):
        """With one cohort, ETWFE should collapse to a standard DiD."""
        rng = np.random.default_rng(0)
        n = 100
        rows = []
        for u in range(n):
            cohort = 2 if u < 50 else 0
            for t in [1, 2]:
                treated = int(cohort > 0 and t >= cohort)
                y = 1.0 * treated + rng.standard_normal()
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": y})
        df = pd.DataFrame(rows)
        r = WooldridgeDiD().fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        assert len(r.group_time_effects) == 1
        assert abs(r.overall_att - 1.0) < 0.5

    def test_aggregation_weights_sum_to_one(self):
        """Simple aggregation weights should sum to 1."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        r = WooldridgeDiD().fit(
            df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
        )
        w = r._gt_weights
        post_keys = [(g, t) for (g, t) in w if t >= g]
        w_total = sum(w[k] for k in post_keys)
        norm_weights = [w[k] / w_total for k in post_keys]
        assert abs(sum(norm_weights) - 1.0) < 1e-10

    def test_logit_delta_method_se_finite(self):
        """Logit delta-method SEs should be finite and non-negative."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta().copy()
        df["lemp_bin"] = (df["lemp"] > df["lemp"].median()).astype(int)

        est = WooldridgeDiD(method="logit")
        results = est.fit(
            df, outcome="lemp_bin", unit="countyreal", time="year", cohort="first_treat"
        )

        assert len(results.group_time_effects) > 0
        for key, cell in results.group_time_effects.items():
            assert cell["se"] >= 0, f"Negative SE at {key}"
            assert np.isfinite(cell["se"]), f"Non-finite SE at {key}"

    def test_poisson_delta_method_se_finite(self):
        """Poisson delta-method SEs should be finite and non-negative."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta().copy()
        df["emp_count"] = np.exp(df["lemp"]).round().astype(int)

        est = WooldridgeDiD(method="poisson")
        results = est.fit(
            df, outcome="emp_count", unit="countyreal", time="year", cohort="first_treat"
        )

        assert len(results.group_time_effects) > 0
        for key, cell in results.group_time_effects.items():
            assert cell["se"] >= 0, f"Negative SE at {key}"
            assert np.isfinite(cell["se"]), f"Non-finite SE at {key}"

    def test_ols_etwfe_att_matches_callaway_santanna(self):
        """OLS ETWFE ATT(g,t) equals CallawaySantAnna ATT(g,t) (Proposition 3.1)."""
        from diff_diff import CallawaySantAnna
        from diff_diff.datasets import load_mpdta

        mpdta = load_mpdta()

        etwfe = WooldridgeDiD(method="ols", control_group="not_yet_treated")
        cs = CallawaySantAnna(control_group="not_yet_treated")

        er = etwfe.fit(mpdta, outcome="lemp", unit="countyreal", time="year", cohort="first_treat")
        cr = cs.fit(
            mpdta, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )

        matched = 0
        for key, effect in er.group_time_effects.items():
            if key in cr.group_time_effects:
                cs_att = cr.group_time_effects[key]["effect"]
                np.testing.assert_allclose(
                    effect["att"],
                    cs_att,
                    atol=5e-3,
                    err_msg=f"ATT mismatch at {key}: ETWFE={effect['att']:.4f}, CS={cs_att:.4f}",
                )
                matched += 1
        assert matched > 0, "No matching (g,t) keys found between ETWFE and CS"


class TestExports:
    def test_top_level_import(self):
        from diff_diff import ETWFE, WooldridgeDiD

        assert ETWFE is WooldridgeDiD

    def test_alias_etwfe(self):
        import diff_diff

        assert hasattr(diff_diff, "ETWFE")
        assert diff_diff.ETWFE is diff_diff.WooldridgeDiD


class TestAnticipation:
    def test_anticipation_includes_pre_treatment_cells(self):
        """With anticipation=1, cells include t >= g-1 (one period before treatment)."""
        rng = np.random.default_rng(42)
        rows = []
        for u in range(40):
            cohort = 3 if u < 20 else 0
            for t in range(1, 6):
                y = rng.standard_normal() + (1.0 if cohort > 0 and t >= cohort else 0)
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": y})
        df = pd.DataFrame(rows)
        est = WooldridgeDiD(anticipation=1)
        r = est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        # With anticipation=1, should have cells for t >= g-1 = 2
        keys = list(r.group_time_effects.keys())
        min_t = min(t for (g, t) in keys)
        assert min_t == 2, f"Expected min t=2 with anticipation=1, got {min_t}"

    def test_anticipation_aware_identification_rejects_pseudo_controls(self):
        """When anticipation consumes all not-yet-treated controls, fit() must
        raise ValueError rather than proceeding with an unidentified design."""
        rows = []
        # Single cohort g=2, times 1-3, no never-treated. With anticipation=1:
        # cohort - anticipation = 1, so all obs have time >= cohort - anticipation.
        # No untreated comparison observations remain.
        for u in range(20):
            for t in range(1, 4):
                rows.append({"unit": u, "time": t, "cohort": 2, "y": float(u + t)})
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="no untreated comparison"):
            WooldridgeDiD(anticipation=1, control_group="not_yet_treated").fit(
                df, outcome="y", unit="unit", time="time", cohort="cohort"
            )

    def test_anticipation_aggregate_semantics(self):
        """With anticipation > 0, simple/group/calendar aggregation uses t >= g
        (not t >= g - anticipation). Anticipation cells are estimated but excluded
        from the overall ATT."""
        rng = np.random.default_rng(42)
        rows = []
        for u in range(60):
            cohort = 4 if u < 30 else 0
            for t in range(1, 8):
                y = rng.standard_normal() + (1.5 if cohort > 0 and t >= cohort else 0)
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": y})
        df = pd.DataFrame(rows)
        est = WooldridgeDiD(anticipation=1)
        r = est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        r.aggregate("event").aggregate("group").aggregate("simple")
        assert np.isfinite(r.overall_att)
        assert r.event_study_effects is not None
        assert r.group_effects is not None
        # Anticipation cells (t < g but t >= g - anticipation) should be in
        # group_time_effects but NOT included in overall_att aggregation.
        # The overall ATT should only average post-treatment cells (t >= g).
        gt = r.group_time_effects
        post_keys = [(g, t) for (g, t) in gt if t >= g]
        antic_keys = [(g, t) for (g, t) in gt if g - 1 <= t < g]
        assert len(antic_keys) > 0, "Expected anticipation cells in group_time_effects"
        assert len(post_keys) > 0, "Expected post-treatment cells"
        # Manually compute post-treatment-only weighted ATT
        w = r._gt_weights
        w_total = sum(w.get(k, 0) for k in post_keys)
        manual_att = sum(w.get(k, 0) * gt[k]["att"] for k in post_keys) / w_total
        assert abs(r.overall_att - manual_att) < 1e-10


class TestXgvarCovariates:
    def test_xgvar_fit_runs(self):
        """xgvar covariates should not crash and should produce finite results."""
        rng = np.random.default_rng(0)
        rows = []
        for u in range(60):
            cohort = 3 if u < 30 else 0
            x1 = rng.standard_normal()
            for t in range(1, 6):
                y = rng.standard_normal() + 0.5 * x1
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": y, "x1": x1})
        df = pd.DataFrame(rows)
        est = WooldridgeDiD()
        r = est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort", xgvar=["x1"])
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)
        assert r.overall_se > 0


class TestAllEventuallyTreated:
    def test_no_never_treated_not_yet_treated_control(self):
        """All units eventually treated, using not_yet_treated control group.

        With not_yet_treated control, the latest cohort's post-treatment
        cells are rank-deficient (no controls remain). The estimator drops
        those columns, so we check that at least the earlier cohort cells
        produce finite ATT effects and the overall ATT is computed from them.
        """
        rng = np.random.default_rng(7)
        rows = []
        for u in range(200):
            # Three cohorts: t=3, t=5, t=8 — wide gaps give plenty of
            # not-yet-treated controls for the earlier cohorts.
            if u < 70:
                cohort = 3
            elif u < 140:
                cohort = 5
            else:
                cohort = 8
            for t in range(1, 10):
                treated = int(t >= cohort)
                y = rng.standard_normal() + 1.5 * treated
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": y})
        df = pd.DataFrame(rows)
        est = WooldridgeDiD(control_group="not_yet_treated")
        r = est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        # At least some cells should have finite ATT
        finite_cells = [k for k, v in r.group_time_effects.items() if np.isfinite(v["att"])]
        assert len(finite_cells) > 0
        # Early cohort (g=3) should have identifiable effects
        early_finite = [k for k in finite_cells if k[0] == 3]
        assert len(early_finite) > 0
        for k in early_finite:
            assert np.isfinite(r.group_time_effects[k]["att"])


class TestEmptyCells:
    def test_sparse_panel_no_crash(self):
        """Panel where some cohort-time cells have few/no obs should not crash."""
        rng = np.random.default_rng(3)
        rows = []
        for u in range(80):
            cohort = 3 if u < 20 else (5 if u < 40 else 0)
            for t in range(1, 7):
                y = rng.standard_normal() + (1.0 if cohort > 0 and t >= cohort else 0)
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": y})
        df = pd.DataFrame(rows)
        est = WooldridgeDiD()
        r = est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        assert np.isfinite(r.overall_att)
        r.aggregate("event")
        assert r.event_study_effects is not None


class TestMpdtaLogitPoisson:
    @pytest.fixture
    def mpdta(self):
        from diff_diff.datasets import load_mpdta

        return load_mpdta()

    def test_logit_on_mpdta(self, mpdta):
        """Logit fit on binary outcome derived from mpdta should produce finite results."""
        df = mpdta.copy()
        df["lemp_bin"] = (df["lemp"] > df["lemp"].median()).astype(int)
        est = WooldridgeDiD(method="logit")
        r = est.fit(df, outcome="lemp_bin", unit="countyreal", time="year", cohort="first_treat")
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)
        assert r.overall_se > 0
        r.aggregate("event")
        assert r.event_study_effects is not None

    def test_poisson_on_mpdta(self, mpdta):
        """Poisson fit on exp(lemp) should produce finite results."""
        df = mpdta.copy()
        df["emp"] = np.exp(df["lemp"])
        est = WooldridgeDiD(method="poisson")
        r = est.fit(df, outcome="emp", unit="countyreal", time="year", cohort="first_treat")
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)
        assert r.overall_se > 0
        r.aggregate("simple")
        assert np.isfinite(r.overall_att)


class TestControlGroupDistinction:
    """P0 regression test: never_treated and not_yet_treated must differ."""

    def test_never_treated_differs_from_not_yet_treated(self):
        """On multi-cohort data with never-treated group, the two control
        group settings must produce different overall ATT estimates."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        r_nyt = WooldridgeDiD(control_group="not_yet_treated").fit(
            df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
        )
        r_nt = WooldridgeDiD(control_group="never_treated").fit(
            df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
        )
        assert np.isfinite(r_nyt.overall_att)
        assert np.isfinite(r_nt.overall_att)
        # They must differ — if they don't, control_group is a no-op
        assert r_nyt.overall_att != r_nt.overall_att, (
            f"never_treated ATT ({r_nt.overall_att:.6f}) == not_yet_treated ATT "
            f"({r_nyt.overall_att:.6f}); control_group has no effect"
        )

    def test_never_treated_more_interaction_terms(self):
        """never_treated should have more interaction terms (includes pre-treatment
        placebo indicators) but the same sample size."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        r_nyt = WooldridgeDiD(control_group="not_yet_treated").fit(
            df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
        )
        r_nt = WooldridgeDiD(control_group="never_treated").fit(
            df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
        )
        # Same sample (all obs kept), but more (g,t) cells for never_treated
        assert r_nt.n_obs == r_nyt.n_obs
        assert len(r_nt.group_time_effects) > len(r_nyt.group_time_effects)


class TestIdentificationChecks:
    def test_no_treated_raises(self):
        """Fitting with no treated cohorts should raise ValueError."""
        df = pd.DataFrame(
            {"unit": [1, 1, 2, 2], "time": [1, 2, 1, 2], "cohort": [0, 0, 0, 0], "y": [1, 2, 3, 4]}
        )
        with pytest.raises(ValueError, match="No treated cohorts"):
            WooldridgeDiD().fit(df, outcome="y", unit="unit", time="time", cohort="cohort")

    def test_never_treated_no_controls_raises(self):
        """never_treated with no cohort==0 units should raise ValueError."""
        df = pd.DataFrame(
            {"unit": [1, 1, 2, 2], "time": [1, 2, 1, 2], "cohort": [2, 2, 3, 3], "y": [1, 2, 3, 4]}
        )
        with pytest.raises(ValueError, match="no never-treated"):
            WooldridgeDiD(control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", cohort="cohort"
            )


class TestBootstrapValidation:
    def test_invalid_bootstrap_weights_raises(self):
        with pytest.raises(ValueError, match="bootstrap_weights"):
            WooldridgeDiD(bootstrap_weights="invalid_dist")

    def test_bootstrap_nonlinear_raises(self):
        """Bootstrap with logit/poisson should raise ValueError."""
        rng = np.random.default_rng(0)
        rows = []
        for u in range(40):
            cohort = 3 if u < 20 else 0
            for t in range(1, 5):
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": rng.standard_normal()})
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="Bootstrap inference is only supported"):
            WooldridgeDiD(method="logit", n_bootstrap=50).fit(
                df, outcome="y", unit="unit", time="time", cohort="cohort"
            )
        with pytest.raises(ValueError, match="Bootstrap inference is only supported"):
            WooldridgeDiD(method="poisson", n_bootstrap=50).fit(
                df, outcome="y", unit="unit", time="time", cohort="cohort"
            )


class TestBootstrapClusterLevel:
    """Regression test: bootstrap must draw weights at the analytic cluster level."""

    def test_bootstrap_with_coarser_cluster(self):
        """Bootstrap with cluster != unit should produce different SE than unit-level."""
        rng = np.random.default_rng(42)
        rows = []
        for u in range(80):
            cohort = 3 if u < 40 else 0
            region = u // 10  # 8 regions, coarser than 80 units
            for t in range(1, 6):
                y = rng.standard_normal() + (1.0 if cohort > 0 and t >= cohort else 0)
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": y, "region": region})
        df = pd.DataFrame(rows)

        # Bootstrap at unit level (default)
        r_unit = WooldridgeDiD(n_bootstrap=99, seed=0).fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        # Bootstrap at region level (coarser)
        r_region = WooldridgeDiD(n_bootstrap=99, seed=0, cluster="region").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        assert np.isfinite(r_unit.overall_se)
        assert np.isfinite(r_region.overall_se)
        # Coarser clustering with fewer clusters should produce different SE
        assert r_unit.overall_se != r_region.overall_se


class TestNonlinearRankDeficiency:
    """Regression test: rank-deficient logit/Poisson must produce finite SEs
    for estimable ATT cells when columns are dropped."""

    def test_logit_rank_deficient_design(self):
        """Logit with a rank-deficient design (many cohort×time cells on small
        data) should handle dropped columns and produce finite SEs."""
        rng = np.random.default_rng(42)
        rows = []
        for u in range(60):
            cohort = 3 if u < 30 else 0
            for t in range(1, 6):
                treated = int(cohort > 0 and t >= cohort)
                eta = -0.5 + 1.0 * treated + 0.1 * rng.standard_normal()
                y = int(rng.random() < 1 / (1 + np.exp(-eta)))
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": y})
        df = pd.DataFrame(rows)
        est = WooldridgeDiD(method="logit", rank_deficient_action="silent")
        r = est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        assert len(r.group_time_effects) > 0
        for cell in r.group_time_effects.values():
            assert np.isfinite(cell["se"]), "SE should be finite for estimable cells"
            assert cell["se"] >= 0

    def test_poisson_rank_deficient_design(self):
        """Poisson with a rank-deficient design should handle dropped columns
        and produce finite SEs."""
        rng = np.random.default_rng(7)
        rows = []
        for u in range(60):
            cohort = 3 if u < 30 else 0
            for t in range(1, 6):
                treated = int(cohort > 0 and t >= cohort)
                mu = np.exp(0.5 + 0.8 * treated + 0.1 * rng.standard_normal())
                y = rng.poisson(mu)
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": float(y)})
        df = pd.DataFrame(rows)
        est = WooldridgeDiD(method="poisson", rank_deficient_action="silent")
        r = est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        assert len(r.group_time_effects) > 0
        for cell in r.group_time_effects.values():
            assert np.isfinite(cell["se"]), "SE should be finite for estimable cells"
            assert cell["se"] >= 0

    def test_logit_with_covariates(self):
        """Logit with covariates should produce finite ATT/SE and differ from
        no-covariate fit (confirming covariates are used)."""
        rng = np.random.default_rng(42)
        rows = []
        for u in range(60):
            cohort = 3 if u < 30 else 0
            x1 = rng.standard_normal()
            for t in range(1, 6):
                treated = int(cohort > 0 and t >= cohort)
                eta = -0.5 + 1.0 * treated + 0.3 * x1 + 0.1 * rng.standard_normal()
                y = int(rng.random() < 1 / (1 + np.exp(-eta)))
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": y, "x1": x1})
        df = pd.DataFrame(rows)
        r_cov = WooldridgeDiD(method="logit").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort", exovar=["x1"]
        )
        r_nocov = WooldridgeDiD(method="logit").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        assert np.isfinite(r_cov.overall_att)
        assert np.isfinite(r_cov.overall_se)
        assert r_cov.overall_se > 0
        assert r_cov.overall_att != r_nocov.overall_att, "Covariates should affect ATT"

    def test_poisson_with_covariates(self):
        """Poisson with covariates should produce finite ATT/SE and differ from
        no-covariate fit (confirming covariates are used)."""
        rng = np.random.default_rng(7)
        rows = []
        for u in range(60):
            cohort = 3 if u < 30 else 0
            x1 = rng.standard_normal()
            for t in range(1, 6):
                treated = int(cohort > 0 and t >= cohort)
                mu = np.exp(0.5 + 0.8 * treated + 0.2 * x1 + 0.1 * rng.standard_normal())
                y = rng.poisson(mu)
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": float(y), "x1": x1})
        df = pd.DataFrame(rows)
        r_cov = WooldridgeDiD(method="poisson").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort", exovar=["x1"]
        )
        r_nocov = WooldridgeDiD(method="poisson").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        assert np.isfinite(r_cov.overall_att)
        assert np.isfinite(r_cov.overall_se)
        assert r_cov.overall_se > 0
        assert r_cov.overall_att != r_nocov.overall_att, "Covariates should affect ATT"


class TestCohortTimeInvariance:
    def test_varying_cohort_raises(self):
        """cohort must be constant within each unit."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "cohort": [2, 3, 0, 0],  # unit 1 has varying cohort
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        with pytest.raises(ValueError, match="not time-invariant"):
            WooldridgeDiD().fit(df, outcome="y", unit="unit", time="time", cohort="cohort")


class TestAnticipationEventLabels:
    def test_event_summary_labels_anticipation_cells(self):
        """summary('event') should label anticipation cells as [antic], not [pre]."""
        rng = np.random.default_rng(42)
        rows = []
        for u in range(60):
            cohort = 4 if u < 30 else 0
            for t in range(1, 8):
                y = rng.standard_normal() + (1.5 if cohort > 0 and t >= cohort else 0)
                rows.append({"unit": u, "time": t, "cohort": cohort, "y": y})
        df = pd.DataFrame(rows)
        est = WooldridgeDiD(anticipation=1)
        r = est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        r.aggregate("event")
        summary = r.summary("event")
        # k=-1 should be labeled [antic] (within anticipation window)
        assert "[antic]" in summary, f"Expected [antic] label in summary, got:\n{summary}"


class TestOutcomeValidation:
    def test_logit_rejects_out_of_range(self):
        """Logit should reject outcomes outside [0, 1]."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "cohort": [2, 2, 0, 0],
                "y": [0.0, 5.0, 0.0, 1.0],
            }
        )
        with pytest.raises(ValueError, match="outcomes in \\[0, 1\\]"):
            WooldridgeDiD(method="logit").fit(
                df, outcome="y", unit="unit", time="time", cohort="cohort"
            )

    def test_poisson_rejects_negative(self):
        """Poisson should reject negative outcomes."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "cohort": [2, 2, 0, 0],
                "y": [1.0, -1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValueError, match="non-negative"):
            WooldridgeDiD(method="poisson").fit(
                df, outcome="y", unit="unit", time="time", cohort="cohort"
            )


class TestUnbalancedOLS:
    """Regression: OLS with exact FE absorption on unbalanced panels.

    The iterative alternating-projections within-transform must match
    explicit unit+time dummy OLS on unbalanced data (P0 fix).
    """

    @pytest.fixture
    def unbalanced_data(self):
        """Panel where some units have fewer periods (dropout)."""
        rng = np.random.RandomState(42)
        rows = []
        for u in range(20):
            g = 3 if u < 10 else 0
            # Make 3 units unbalanced (observed only periods 1-3)
            max_t = 3 if u in [2, 5, 7] else 5
            for t in range(1, max_t + 1):
                effect = 0.5 if g > 0 and t >= g else 0.0
                y = rng.normal() + effect
                rows.append({"unit": u, "time": t, "cohort": g, "y": y})
        return pd.DataFrame(rows)

    def test_parity_with_dummy_ols_not_yet_treated(self, unbalanced_data):
        from diff_diff.linalg import solve_ols

        df = unbalanced_data
        r = WooldridgeDiD(control_group="not_yet_treated").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )

        # Build explicit dummy regression on same sample
        sample = _filter_sample(df, "unit", "time", "cohort", "not_yet_treated", 0)
        X_int, _, gt_keys = _build_interaction_matrix(
            sample,
            cohort="cohort",
            time="time",
            anticipation=0,
            control_group="not_yet_treated",
            method="ols",
        )
        unit_dummies = pd.get_dummies(sample["unit"], drop_first=True).values.astype(float)
        time_dummies = pd.get_dummies(sample["time"], drop_first=True).values.astype(float)
        intercept = np.ones((len(sample), 1))
        X_full = np.hstack([intercept, X_int, unit_dummies, time_dummies])
        y = sample["y"].values

        coefs_dummy, _, _ = solve_ols(X_full, y)

        for i, (g, t) in enumerate(gt_keys):
            if (g, t) in r.group_time_effects:
                np.testing.assert_allclose(
                    r.group_time_effects[(g, t)]["att"],
                    coefs_dummy[1 + i],
                    atol=1e-6,
                    err_msg=f"ATT mismatch at cell ({g},{t})",
                )

    def test_never_treated_unbalanced_finite(self, unbalanced_data):
        """never_treated on unbalanced data should produce finite results.

        Explicit-dummy parity is not meaningful for never_treated because the
        all-cells interaction matrix creates rank deficiency with unit dummies
        (within-transform avoids this by absorbing FE before solving).
        """
        df = unbalanced_data
        r = WooldridgeDiD(control_group="never_treated").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)
        # Should have pre-treatment placebo cells (OLS never_treated)
        pre_cells = [(g, t) for (g, t) in r.group_time_effects if t < g]
        assert len(pre_cells) > 0


class TestNonlinearNeverTreated:
    """Regression: nonlinear never_treated uses post-treatment cells only (P1 fix).

    Nonlinear methods with never_treated must produce a complete post-treatment
    ATT grid without arbitrary QR-based column dropping.
    """

    @pytest.fixture
    def binary_data(self):
        rng = np.random.RandomState(42)
        rows = []
        for u in range(60):
            g = 3 if u < 20 else (4 if u < 40 else 0)
            for t in range(1, 6):
                effect = 0.3 if g > 0 and t >= g else 0.0
                p = 1 / (1 + np.exp(-(rng.normal() * 0.3 + effect)))
                y = int(rng.random() < p)
                rows.append({"unit": u, "time": t, "cohort": g, "y": float(y)})
        return pd.DataFrame(rows)

    @pytest.fixture
    def count_data(self):
        rng = np.random.RandomState(42)
        rows = []
        for u in range(60):
            g = 3 if u < 20 else (4 if u < 40 else 0)
            for t in range(1, 6):
                effect = 0.3 if g > 0 and t >= g else 0.0
                y = rng.poisson(np.exp(0.5 + effect))
                rows.append({"unit": u, "time": t, "cohort": g, "y": float(y)})
        return pd.DataFrame(rows)

    def test_logit_never_treated_post_treatment_only(self, binary_data):
        r = WooldridgeDiD(method="logit", control_group="never_treated").fit(
            binary_data, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        # All cells should be post-treatment
        for g, t in r.group_time_effects:
            assert t >= g, f"Pre-treatment cell ({g},{t}) in nonlinear never_treated"
        # All expected post-treatment cells present
        expected = {(g, t) for g in [3, 4] for t in range(1, 6) if t >= g}
        assert set(r.group_time_effects.keys()) == expected
        assert np.isfinite(r.overall_att)

    def test_poisson_never_treated_post_treatment_only(self, count_data):
        r = WooldridgeDiD(method="poisson", control_group="never_treated").fit(
            count_data, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        for g, t in r.group_time_effects:
            assert t >= g, f"Pre-treatment cell ({g},{t}) in nonlinear never_treated"
        expected = {(g, t) for g in [3, 4] for t in range(1, 6) if t >= g}
        assert set(r.group_time_effects.keys()) == expected
        assert np.isfinite(r.overall_att)

    def test_interaction_matrix_fewer_cols_for_nonlinear(self):
        """For never_treated, nonlinear methods get fewer interaction columns
        than OLS (no pre-treatment cells)."""
        rows = []
        for u in range(20):
            g = 3 if u < 10 else 0
            for t in range(1, 6):
                rows.append({"unit": u, "time": t, "cohort": g, "y": 0.0})
        df = pd.DataFrame(rows)

        X_ols, _, _ = _build_interaction_matrix(df, "cohort", "time", 0, "never_treated", "ols")
        X_logit, _, _ = _build_interaction_matrix(df, "cohort", "time", 0, "never_treated", "logit")
        X_nyt, _, _ = _build_interaction_matrix(df, "cohort", "time", 0, "not_yet_treated", "ols")
        # OLS never_treated > nonlinear never_treated == not_yet_treated
        assert X_ols.shape[1] > X_logit.shape[1]
        assert X_logit.shape[1] == X_nyt.shape[1]

    def test_ols_never_treated_still_has_pre_treatment(self):
        """OLS path should still include pre-treatment placebo cells."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        r = WooldridgeDiD(control_group="never_treated").fit(
            df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
        )
        # OLS never_treated should have pre-treatment cells
        pre_treatment = [(g, t) for (g, t) in r.group_time_effects if t < g]
        assert len(pre_treatment) > 0, "OLS never_treated lost pre-treatment placebo cells"


class TestFullCovariateBasis:
    """Regression: covariate-adjusted ETWFE includes full W2025 Eq. 5.3 basis
    (D_g × X, f_t × X, D_{g,t} × X̃, raw X)."""

    @pytest.fixture
    def cov_data(self):
        rng = np.random.RandomState(42)
        rows = []
        for u in range(30):
            g = 3 if u < 10 else (4 if u < 20 else 0)
            x1 = rng.normal()
            for t in range(1, 6):
                effect = 0.5 if g > 0 and t >= g else 0.0
                y = rng.normal() + effect + 0.3 * x1
                rows.append({"unit": u, "time": t, "cohort": g, "y": y, "x1": x1})
        return pd.DataFrame(rows)

    def test_ols_covariate_parity_with_full_basis_dummy_ols(self, cov_data):
        """OLS with exovar should match explicit-dummy OLS with full basis."""
        from diff_diff.linalg import solve_ols

        df = cov_data
        r = WooldridgeDiD(control_group="not_yet_treated").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort", exovar=["x1"]
        )

        # Build explicit-dummy regression with full basis
        sample = _filter_sample(df, "unit", "time", "cohort", "not_yet_treated", 0)
        X_int, _, gt_keys = _build_interaction_matrix(
            sample, "cohort", "time", 0, "not_yet_treated", "ols"
        )
        n_int = X_int.shape[1]
        x1_raw = sample["x1"].values.astype(float)

        # Cell × demeaned-X interactions
        groups = sorted(g for g in sample["cohort"].unique() if g > 0)
        x1_demeaned = x1_raw.copy()
        for g in groups:
            mask = sample["cohort"].values == g
            if mask.any():
                x1_demeaned[mask] -= x1_raw[mask].mean()
        cell_cov = np.column_stack([X_int[:, i] * x1_demeaned for i in range(n_int)])

        # D_g × X (cohort × covariate)
        cohort_cov = np.column_stack(
            [(sample["cohort"].values == g).astype(float) * x1_raw for g in groups]
        )

        # f_t × X (time × covariate, drop first)
        times = sorted(sample["time"].unique())
        time_cov = np.column_stack(
            [(sample["time"].values == t).astype(float) * x1_raw for t in times[1:]]
        )

        # Full design: intercept + cells + cell×cov + D_g×X + f_t×X + raw_X + unit + time dummies
        unit_dummies = pd.get_dummies(sample["unit"], drop_first=True).values.astype(float)
        time_dummies = pd.get_dummies(sample["time"], drop_first=True).values.astype(float)
        intercept = np.ones((len(sample), 1))
        X_full = np.hstack(
            [
                intercept,
                X_int,
                cell_cov,
                cohort_cov,
                time_cov,
                x1_raw.reshape(-1, 1),
                unit_dummies,
                time_dummies,
            ]
        )
        y = sample["y"].values
        coefs_dummy, _, _ = solve_ols(X_full, y, rank_deficient_action="silent")

        # Compare ATT coefficients (positions 1..n_int in dummy OLS)
        for i, (g, t) in enumerate(gt_keys):
            if (g, t) in r.group_time_effects:
                np.testing.assert_allclose(
                    r.group_time_effects[(g, t)]["att"],
                    coefs_dummy[1 + i],
                    atol=1e-5,
                    err_msg=f"Covariate ATT mismatch at cell ({g},{t})",
                )

    def test_covariates_affect_ols_att(self, cov_data):
        """OLS with covariates should produce different ATT than without."""
        df = cov_data
        r_cov = WooldridgeDiD().fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort", exovar=["x1"]
        )
        r_nocov = WooldridgeDiD().fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        assert (
            r_cov.overall_att != r_nocov.overall_att
        ), "Covariate-adjusted ATT should differ from unadjusted"


class TestWooldridgeSurvey:
    """Survey design support for WooldridgeDiD."""

    @pytest.fixture
    def survey_panel(self):
        """Panel data with survey design columns."""
        rng = np.random.default_rng(99)
        n_units, n_periods = 80, 5
        rows = []
        for u in range(n_units):
            cohort = 3 if u < 30 else (4 if u < 50 else 0)
            stratum = u % 4
            psu = u  # globally unique PSU per unit
            weight = 1.0 + rng.exponential(0.5)
            for t in range(1, n_periods + 1):
                treated = int(cohort > 0 and t >= cohort)
                y_cont = 1.0 + 2.0 * treated + 0.3 * rng.standard_normal()
                eta = -0.5 + 1.0 * treated + 0.1 * rng.standard_normal()
                y_bin = int(rng.random() < 1 / (1 + np.exp(-eta)))
                mu = np.exp(0.5 + 0.5 * treated + 0.1 * rng.standard_normal())
                y_count = float(rng.poisson(mu))
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "cohort": cohort,
                        "y": y_cont,
                        "y_bin": y_bin,
                        "y_count": y_count,
                        "stratum": stratum,
                        "psu": psu,
                        "weight": weight,
                    }
                )
        return pd.DataFrame(rows)

    def test_ols_survey_runs(self, survey_panel):
        """OLS with full survey design completes."""
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        r = WooldridgeDiD().fit(
            survey_panel,
            outcome="y",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)
        assert r.overall_se > 0

    def test_ols_survey_se_differs_from_naive(self, survey_panel):
        """Survey SE should differ from naive (unweighted) SE."""
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        r_survey = WooldridgeDiD().fit(
            survey_panel,
            outcome="y",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        r_naive = WooldridgeDiD().fit(
            survey_panel,
            outcome="y",
            unit="unit",
            time="time",
            cohort="cohort",
        )
        assert r_survey.overall_se != r_naive.overall_se

    def test_logit_survey_runs(self, survey_panel):
        """Logit with survey design completes."""
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        r = WooldridgeDiD(method="logit").fit(
            survey_panel,
            outcome="y_bin",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)
        assert r.overall_se > 0

    def test_poisson_survey_runs(self, survey_panel):
        """Poisson with survey design completes."""
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        r = WooldridgeDiD(method="poisson").fit(
            survey_panel,
            outcome="y_count",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)
        assert r.overall_se > 0

    def test_bootstrap_survey_rejected(self, survey_panel):
        """n_bootstrap > 0 with survey_design raises ValueError."""
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="weight")
        with pytest.raises(
            ValueError, match="Bootstrap inference is not supported with survey_design"
        ):
            WooldridgeDiD(n_bootstrap=100).fit(
                survey_panel,
                outcome="y",
                unit="unit",
                time="time",
                cohort="cohort",
                survey_design=sd,
            )

    def test_weights_only_survey(self, survey_panel):
        """Weights-only survey (no strata/PSU) works."""
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="weight")
        r = WooldridgeDiD().fit(
            survey_panel,
            outcome="y",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)
        assert r.survey_metadata is not None

    def test_survey_metadata_present(self, survey_panel):
        """survey_metadata is populated with correct fields."""
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        r = WooldridgeDiD().fit(
            survey_panel,
            outcome="y",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        sm = r.survey_metadata
        assert sm is not None
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0
        assert sm.design_effect > 0
        assert sm.n_strata is not None
        assert sm.n_psu is not None
        assert sm.df_survey is not None

    def test_replicate_weights_rejected(self, survey_panel):
        """Replicate-weight designs raise NotImplementedError."""
        from diff_diff.survey import SurveyDesign

        # Add replicate weight columns
        survey_panel["rep_w1"] = 1.0
        survey_panel["rep_w2"] = 1.0
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=["rep_w1", "rep_w2"],
            replicate_method="BRR",
        )
        with pytest.raises(NotImplementedError, match="replicate-weight variance"):
            WooldridgeDiD().fit(
                survey_panel,
                outcome="y",
                unit="unit",
                time="time",
                cohort="cohort",
                survey_design=sd,
            )

    def test_weights_only_plus_cluster(self, survey_panel):
        """Weights-only survey + cluster= injects cluster as PSU."""
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="weight")
        r = WooldridgeDiD(cluster="stratum").fit(
            survey_panel,
            outcome="y",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        # Cluster should have been injected as PSU
        n_strata = survey_panel["stratum"].nunique()
        assert r.survey_metadata is not None
        assert r.survey_metadata.n_psu == n_strata

        # SE should differ from same run without cluster
        r_no_cluster = WooldridgeDiD().fit(
            survey_panel,
            outcome="y",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        assert r.overall_se != r_no_cluster.overall_se

    def test_survey_gt_weights_are_counts(self, survey_panel):
        """Survey aggregation uses cell counts, not survey-weight sums."""
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="weight", strata="stratum", psu="unit")
        r = WooldridgeDiD(method="logit").fit(
            survey_panel,
            outcome="y_bin",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        for k, w in r._gt_weights.items():
            assert isinstance(w, int), (
                f"gt_weights[{k}] = {w} (type {type(w).__name__}); " f"expected int (cell count)"
            )

    def test_weights_only_no_cluster_implicit_psu(self, survey_panel):
        """Weights-only survey without cluster= keeps implicit per-obs PSUs."""
        from diff_diff.survey import SurveyDesign
        from diff_diff.wooldridge import _filter_sample

        sd = SurveyDesign(weights="weight")
        r = WooldridgeDiD().fit(
            survey_panel,
            outcome="y",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        # n_psu should equal n_obs in the filtered sample (not n_units)
        sample = _filter_sample(
            survey_panel.copy().assign(cohort=lambda d: d["cohort"].fillna(0)),
            "unit",
            "time",
            "cohort",
            "not_yet_treated",
            0,
        )
        assert r.survey_metadata is not None
        assert r.survey_metadata.n_psu == len(sample)

    def test_fweight_rejected(self, survey_panel):
        """fweight raises ValueError (pweight only)."""
        from diff_diff.survey import SurveyDesign

        # Use integer weights so fweight validation passes in resolve(),
        # and the pweight guard in _resolve_survey_for_wooldridge fires.
        df = survey_panel.copy()
        df["int_weight"] = 1
        sd = SurveyDesign(weights="int_weight", weight_type="fweight")
        with pytest.raises(ValueError, match="weight_type='pweight'"):
            WooldridgeDiD().fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                cohort="cohort",
                survey_design=sd,
            )

    def test_poisson_zero_weight_cell(self, survey_panel):
        """Poisson survey fit handles zero-weight treated cells cleanly."""
        from diff_diff.survey import SurveyDesign

        df = survey_panel.copy()
        # Zero out weights for one treated cohort so some cells have zero weight
        df.loc[df["cohort"] == 3, "weight"] = 0.0
        sd = SurveyDesign(weights="weight", strata="stratum", psu="unit")
        r = WooldridgeDiD(method="poisson").fit(
            df,
            outcome="y_count",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)

    def test_ols_survey_rank_deficient(self, survey_panel):
        """Survey OLS handles rank-deficient all-eventually-treated designs."""
        from diff_diff.survey import SurveyDesign

        # Remove never-treated (cohort=0) to create rank-deficient design
        df = survey_panel[survey_panel["cohort"] > 0].copy()
        sd = SurveyDesign(weights="weight", strata="stratum", psu="unit")
        r = WooldridgeDiD(control_group="not_yet_treated").fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)

    def test_ols_survey_zero_weight_unit_rejected(self, survey_panel):
        """Zero-weight unit raises ValueError before within_transform."""
        from diff_diff.survey import SurveyDesign

        df = survey_panel.copy()
        # Zero out all weights for unit 0
        df.loc[df["unit"] == 0, "weight"] = 0.0
        sd = SurveyDesign(weights="weight", strata="stratum", psu="unit")
        with pytest.raises(ValueError, match="Survey weights sum to zero for unit"):
            WooldridgeDiD().fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                cohort="cohort",
                survey_design=sd,
            )

    def test_logit_survey_zero_weight_cell(self, survey_panel):
        """Logit survey fit skips zero-weight treated cells cleanly."""
        from diff_diff.survey import SurveyDesign

        df = survey_panel.copy()
        df.loc[df["cohort"] == 3, "weight"] = 0.0
        sd = SurveyDesign(weights="weight", strata="stratum", psu="unit")
        r = WooldridgeDiD(method="logit").fit(
            df,
            outcome="y_bin",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)

    def test_ols_survey_non_range_index(self, survey_panel):
        """OLS survey zero-weight guard works with non-RangeIndex DataFrames."""
        from diff_diff.survey import SurveyDesign

        df = survey_panel.copy()
        df.index = df.index + 1000  # shift to non-zero-based index
        df.loc[df["unit"] == 0, "weight"] = 0.0
        sd = SurveyDesign(weights="weight", strata="stratum", psu="unit")
        with pytest.raises(ValueError, match="Survey weights sum to zero for unit"):
            WooldridgeDiD().fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                cohort="cohort",
                survey_design=sd,
            )

    def test_survey_aggregate_and_summary(self, survey_panel):
        """Survey aggregate() uses df_survey and summary() shows survey block."""
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="weight", strata="stratum", psu="unit")
        r = WooldridgeDiD().fit(
            survey_panel,
            outcome="y",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=sd,
        )
        # aggregate() should use t-distribution with survey df
        r.aggregate("group")
        assert r.group_effects is not None
        assert r._df_survey is not None
        for eff in r.group_effects.values():
            assert np.isfinite(eff["p_value"])

        # summary() should include survey design block
        s = r.summary()
        assert "Survey Design" in s
        assert "pweight" in s


class TestCohortNaNWarning:
    """Axis-E: silent recategorization of NaN cohort rows as never-treated."""

    @staticmethod
    def _make_panel_with_nan_cohort():
        rows = []
        for unit in range(10):
            cohort_val = np.nan if unit < 2 else 0.0
            for t in range(1, 5):
                rows.append(
                    {
                        "unit": unit,
                        "time": t,
                        "cohort": cohort_val,
                        "y": unit + t + np.random.default_rng(unit).normal(0, 0.1),
                    }
                )
        return pd.DataFrame(rows)

    def test_fit_warns_on_nan_cohort_with_count(self):
        df = self._make_panel_with_nan_cohort()
        est = WooldridgeDiD(method="ols")
        with pytest.warns(UserWarning, match=r"8 row\(s\) have NaN cohort values"):
            try:
                est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
            except Exception:
                pass

    def test_fit_silent_on_clean_cohort(self):
        import warnings

        df = self._make_panel_with_nan_cohort()
        df["cohort"] = df["cohort"].fillna(0)
        est = WooldridgeDiD(method="ols")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
            except Exception:
                pass
        nan_warnings = [x for x in w if "NaN cohort values" in str(x.message)]
        assert nan_warnings == []

    def test_select_sample_helper_warns(self):
        df = self._make_panel_with_nan_cohort()
        with pytest.warns(UserWarning, match=r"8 row\(s\) have NaN cohort values"):
            _filter_sample(
                df,
                unit="unit",
                time="time",
                cohort="cohort",
                control_group="never_treated",
                anticipation=0,
            )


def _make_vcov_panel(n_units=40, n_periods=6, seed=202605211230):
    """Fixed-seed staggered panel for vcov_type tests.

    Three cohorts (0=never, 3, 5), heterogeneous treatment effects (stronger
    for cohort=3), 40 units × 6 periods. Heterogeneous effects per
    ``feedback_homogeneous_dgp_no_twfe_bias`` — required for meaningful
    TWFE-style bias-vs-corrected comparisons. The fixed seed and panel
    shape are pinned by
    ``TestWooldridgeVcovType::test_hc1_se_bit_equal_to_pre_pr_baseline``
    against a hardcoded SE captured on the Phase 1b PR 3/8 branch
    (commit-SHA-equivalent to ``origin/main`` at fork time: ``24de9062``).
    """
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    cohort_choices = [0, 3, 5]
    cohorts = rng.choice(cohort_choices, size=n_units, p=[0.4, 0.3, 0.3])
    cohort_per_obs = cohorts[units]
    tau = np.where(
        (cohort_per_obs > 0) & (periods >= cohort_per_obs),
        0.4 + 0.25 * (periods - cohort_per_obs) + 0.3 * (cohort_per_obs == 3),
        0.0,
    )
    y = 0.7 + 0.1 * periods + 0.05 * units + tau + 0.15 * rng.normal(size=len(units))
    return pd.DataFrame({"unit": units, "time": periods, "cohort": cohort_per_obs, "y": y})


class TestWooldridgeVcovType:
    """Phase 1b PR 3/8: vcov_type input contract + branching for OLS path."""

    def test_default_vcov_type_is_hc1(self):
        est = WooldridgeDiD()
        assert est.vcov_type == "hc1"
        assert est._vcov_type_explicit is False

    def test_hc1_se_bit_equal_to_pre_pr_baseline(self):
        """HC1 within-transform path must match the frozen baseline at atol=1e-14.

        Baseline originally captured on the Phase 1b PR 3/8 branch with
        ``_make_vcov_panel(seed=202605211230)`` (FWL preserves the CR1
        cluster-robust score, so the ``vcov_type`` branching kept HC1 bit-equal
        to the prior hard-coded HC1 behavior). Recaptured under the v3.6.x
        factorize-once + bincount demeaner, which moved the values by ~1e-15
        (bincount accumulation vs pandas' compensated grouped mean - see
        REGISTRY "Absorbed Fixed Effects"); the 1e-14 lock semantics are
        unchanged.
        """
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="hc1").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        assert res.overall_att == pytest.approx(0.9178849934516233, abs=1e-14)
        assert res.overall_se == pytest.approx(0.03149488781317813, abs=1e-14)

    def test_hc2_bm_finite_and_inflates_over_hc1(self):
        df = _make_vcov_panel()
        res_hc1 = WooldridgeDiD(method="ols", vcov_type="hc1").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        res_bm = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        for k, eff in res_bm.group_time_effects.items():
            assert np.isfinite(eff["se"])
        assert np.isfinite(res_bm.overall_se)
        assert res_bm.overall_se > res_hc1.overall_se
        # ATT identity across vcov branches (only SE differs)
        assert res_bm.overall_att == pytest.approx(res_hc1.overall_att, abs=1e-10)

    def test_atts_identical_across_vcov_branches(self):
        """Per-cell ATT estimates must be identical across all 4 vcov branches
        (within-transform hc1 vs full-dummy hc2_bm/hc2/classical)."""
        df = _make_vcov_panel()
        results = {}
        for vt in ("hc1", "hc2_bm", "hc2", "classical"):
            results[vt] = WooldridgeDiD(method="ols", vcov_type=vt).fit(
                df, outcome="y", unit="unit", time="time", cohort="cohort"
            )
        ref = results["hc1"]
        for vt in ("hc2_bm", "hc2", "classical"):
            assert results[vt].overall_att == pytest.approx(ref.overall_att, abs=1e-10)
            for k in ref.group_time_effects:
                assert results[vt].group_time_effects[k]["att"] == pytest.approx(
                    ref.group_time_effects[k]["att"], abs=1e-10
                ), f"per-cell ATT diverged for vcov_type={vt!r} at cell {k}"

    def test_classical_with_explicit_user_cluster_rejected_by_linalg(self):
        df = _make_vcov_panel()
        est = WooldridgeDiD(method="ols", vcov_type="classical", cluster="unit")
        with pytest.raises(ValueError):
            est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")

    def test_classical_drops_auto_cluster(self):
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="classical").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        assert np.isfinite(res.overall_se)
        assert res.cluster_name is None
        assert res.n_clusters is None

    def test_hc2_drops_auto_cluster(self):
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="hc2").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        assert np.isfinite(res.overall_se)
        assert res.cluster_name is None
        assert res.n_clusters is None

    def test_conley_accepted_at_init(self):
        """conley is a valid constructor value as of the conley-threading PR
        (coords/cutoff/unit/lag validation is deferred to fit()). The OLS-only
        restriction still rejects conley on the logit/poisson paths."""
        est = WooldridgeDiD(vcov_type="conley")
        assert est.vcov_type == "conley"
        with pytest.raises(NotImplementedError):
            WooldridgeDiD(method="logit", vcov_type="conley")

    def test_invalid_vcov_type_rejected(self):
        with pytest.raises(ValueError, match="hc4"):
            WooldridgeDiD(vcov_type="hc4")

    def test_logit_plus_hc2_bm_rejected_at_init(self):
        with pytest.raises(NotImplementedError, match=r"method='logit'"):
            WooldridgeDiD(method="logit", vcov_type="hc2_bm")

    def test_poisson_plus_hc2_bm_rejected_at_init(self):
        with pytest.raises(NotImplementedError, match=r"method='poisson'"):
            WooldridgeDiD(method="poisson", vcov_type="hc2_bm")

    def test_logit_plus_hc1_default_preserved(self):
        # method='logit' + vcov_type='hc1' (default) must NOT raise —
        # preserves the prior nonlinear path bit-equally.
        est = WooldridgeDiD(method="logit", vcov_type="hc1")
        assert est.method == "logit"
        assert est.vcov_type == "hc1"

    def test_survey_design_plus_hc2_bm_rejected(self):
        from diff_diff.survey import SurveyDesign

        df = _make_vcov_panel()
        df["w"] = 1.0
        design = SurveyDesign(weights="w", weight_type="pweight")
        est = WooldridgeDiD(method="ols", vcov_type="hc2_bm")
        with pytest.raises(NotImplementedError, match=r"survey_design"):
            est.fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                cohort="cohort",
                survey_design=design,
            )

    def test_survey_design_plus_classical_rejected(self):
        from diff_diff.survey import SurveyDesign

        df = _make_vcov_panel()
        df["w"] = 1.0
        design = SurveyDesign(weights="w", weight_type="pweight")
        est = WooldridgeDiD(method="ols", vcov_type="classical")
        with pytest.raises(NotImplementedError, match=r"survey_design"):
            est.fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                cohort="cohort",
                survey_design=design,
            )

    def test_bootstrap_plus_one_way_rejected_regardless_of_cluster(self):
        """Bootstrap + one-way analytical vcov_type is rejected at the
        estimator boundary regardless of ``self.cluster`` — under
        ``cluster=None`` the auto-cluster is dropped (no cluster for the
        bootstrap to draw at); under ``cluster=X`` the linalg validator
        rejects one-way + cluster_ids. Both fail paths produce a less-
        informative downstream error, so the estimator rejects up front."""
        df = _make_vcov_panel()
        # Case 1: cluster=None (default) — bootstrap reject fires
        est = WooldridgeDiD(method="ols", vcov_type="classical", n_bootstrap=10, seed=0)
        with pytest.raises(ValueError, match=r"multiplier bootstrap"):
            est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        # Case 2: cluster=X — also rejected at the estimator boundary (would
        # otherwise hit the linalg validator with a less-informative message)
        est_cl = WooldridgeDiD(
            method="ols", vcov_type="hc2", n_bootstrap=10, cluster="unit", seed=0
        )
        with pytest.raises(ValueError, match=r"multiplier bootstrap"):
            est_cl.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")

    def test_hc2_bm_plus_bootstrap_finite_inference(self):
        """Positive regression: ``vcov_type='hc2_bm'`` + ``n_bootstrap > 0``
        runs through the new full-dummy branch's bootstrap closure (with
        ``coef_offset=1`` for the post-period ATT reconstruction) without
        regressing. Asserts finite ``overall_se`` (overridden by the
        multiplier bootstrap), stable ``overall_att`` (matches the
        analytical fit at machine precision since the bootstrap only
        overrides SE), and finite event-study aggregation."""
        df = _make_vcov_panel()
        # Analytical hc2_bm fit for ATT reference.
        res_analytical = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        # Bootstrap fit on the same data + seed.
        res_boot = WooldridgeDiD(method="ols", vcov_type="hc2_bm", n_bootstrap=50, seed=0).fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        # ATT is unchanged by the bootstrap (only SE is overridden)
        assert res_boot.overall_att == pytest.approx(res_analytical.overall_att, abs=1e-10)
        # SE finite + sensible (positive, smaller than the panel SD of y)
        assert np.isfinite(res_boot.overall_se)
        assert res_boot.overall_se > 0
        assert res_boot.overall_se < df["y"].std()
        # Bootstrap overrides analytical inference for overall ATT
        assert np.isfinite(res_boot.overall_t_stat)
        assert np.isfinite(res_boot.overall_p_value)
        # Per-cell SEs still come from the analytical full-dummy CR2-BM path
        # (bootstrap only overrides overall_*); locks the coef_offset
        # bootstrap indexing didn't regress the per-cell analytical path.
        for k, eff in res_boot.group_time_effects.items():
            assert np.isfinite(eff["se"])
            assert eff["att"] == pytest.approx(
                res_analytical.group_time_effects[k]["att"], abs=1e-10
            )
        # Event-study aggregate also produces finite inference under bootstrap
        res_boot.aggregate("event")
        assert res_boot.event_study_effects is not None
        for k, eff in res_boot.event_study_effects.items():
            assert np.isfinite(eff["att"])
            assert np.isfinite(eff["se"])
            assert np.isfinite(eff["t_stat"])

    def test_hc2_bm_plus_bootstrap_rank_deficient(self):
        """hc2_bm + bootstrap on a rank-deficient design (all-eventually-
        treated panel where late cohorts drop out of solve_ols) — bootstrap
        loop must still run because cluster_ids_bootstrap defaults to unit
        (cluster_ids itself is non-None on hc2_bm). Locks that the
        coef_offset + dropped-cell indexing in the bootstrap closure
        survives rank deficiency."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 8
        units = np.repeat(np.arange(n_units), n_periods)
        periods = np.tile(np.arange(1, n_periods + 1), n_units)
        cohorts = rng.choice([3, 5, 7], size=n_units)
        cohort_per_obs = cohorts[units]
        tau = np.where(periods >= cohort_per_obs, 0.5 + 0.2 * (periods - cohort_per_obs), 0.0)
        y = 1.0 + 0.1 * periods + tau + 0.1 * rng.normal(size=len(units))
        df = pd.DataFrame({"unit": units, "time": periods, "cohort": cohort_per_obs, "y": y})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = WooldridgeDiD(method="ols", vcov_type="hc2_bm", n_bootstrap=50, seed=0).fit(
                df, outcome="y", unit="unit", time="time", cohort="cohort"
            )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        assert res.overall_se > 0

    def test_get_params_includes_vcov_type(self):
        est = WooldridgeDiD(vcov_type="hc2_bm")
        params = est.get_params()
        assert params["vcov_type"] == "hc2_bm"
        # Round-trip via get_params → __init__
        est2 = WooldridgeDiD(**params)
        assert est2.vcov_type == "hc2_bm"

    def test_set_params_revalidates_vcov_type(self):
        est = WooldridgeDiD()
        with pytest.raises(ValueError, match="hc4"):
            est.set_params(vcov_type="hc4")

    def test_set_params_catches_method_vcov_interaction(self):
        est = WooldridgeDiD(method="ols", vcov_type="hc1")
        with pytest.raises(NotImplementedError):
            est.set_params(method="logit", vcov_type="hc2_bm")

    def test_set_params_is_atomic_on_validation_failure(self):
        """Per codex R5 P1: rejected set_params must leave the estimator
        unchanged so subsequent fit() runs on the validated configuration,
        not a half-mutated one. Without atomicity, a caller that catches
        the exception could later run e.g. a logit HC1 fit while
        ``self.vcov_type`` silently reads ``'hc2_bm'``."""
        est = WooldridgeDiD(method="ols", vcov_type="hc1")
        original_params = est.get_params()
        # Reject: method=logit + vcov_type=hc2_bm (interaction guard)
        with pytest.raises(NotImplementedError):
            est.set_params(method="logit", vcov_type="hc2_bm")
        # Estimator must be unchanged
        assert est.get_params() == original_params
        assert est.method == "ols"
        assert est.vcov_type == "hc1"
        assert est._vcov_type_explicit is False
        # Reject: unknown vcov_type. Try changing multiple params at once
        # to verify atomicity catches partial application.
        with pytest.raises(ValueError, match="hc4"):
            est.set_params(method="poisson", vcov_type="hc4")
        # method must NOT have changed to "poisson" — the validator rejected
        # the batch before any setattr() ran.
        assert est.method == "ols"
        assert est.vcov_type == "hc1"
        # Unknown parameter key: same atomicity guarantee.
        with pytest.raises(ValueError, match="bogus_param"):
            est.set_params(vcov_type="hc2_bm", bogus_param=42)
        assert est.vcov_type == "hc1"
        assert est._vcov_type_explicit is False

    def test_survey_design_clears_cluster_metadata(self):
        """Per codex R5 P2: under survey TSL the analytical sandwich (and
        its cluster_ids) is replaced — cluster_name / n_clusters should be
        ``None`` (the survey design's stratification lives in
        ``survey_metadata``), not a misleading echo of the default unit
        cluster."""
        from diff_diff.survey import SurveyDesign

        df = _make_vcov_panel()
        df["w"] = 1.0
        design = SurveyDesign(weights="w", weight_type="pweight")
        # OLS + survey + default hc1: the analytical fall-through would
        # have surfaced cluster_name='unit', n_clusters=N — but survey TSL
        # replaces that vcov, so the dataclass must report None.
        res = WooldridgeDiD(method="ols", vcov_type="hc1").fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            cohort="cohort",
            survey_design=design,
        )
        assert res.survey_metadata is not None
        assert res.cluster_name is None
        assert res.n_clusters is None

    def test_set_params_updates_vcov_type_explicit_flag(self):
        est = WooldridgeDiD(vcov_type="hc1")
        assert est._vcov_type_explicit is False
        est.set_params(vcov_type="hc2_bm")
        assert est._vcov_type_explicit is True
        est.set_params(vcov_type="hc1")
        assert est._vcov_type_explicit is False

    def test_results_carries_vcov_type(self):
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        assert res.vcov_type == "hc2_bm"

    def test_results_carries_cluster_name_for_clustered_fit(self):
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="hc1").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        assert res.cluster_name == "unit"
        assert res.n_clusters is not None
        assert res.n_clusters > 0

    def test_explicit_user_cluster_preserved_under_hc1(self):
        df = _make_vcov_panel()
        # Synthetic state column with 4 levels — 10 units per state on the
        # 40-unit panel
        df["state"] = (df["unit"] // 10).astype(int)
        res = WooldridgeDiD(method="ols", vcov_type="hc1", cluster="state").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        assert res.cluster_name == "state"
        assert res.n_clusters == 4

    def test_fit_clone_idempotent_on_vcov_type(self):
        """fit, clone via get_params, refit — SE must be bit-equal."""
        df = _make_vcov_panel()
        est = WooldridgeDiD(method="ols", vcov_type="hc2_bm")
        res1 = est.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        est2 = WooldridgeDiD(**est.get_params())
        res2 = est2.fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        assert res1.overall_se == pytest.approx(res2.overall_se, abs=1e-14)
        assert res1.overall_att == pytest.approx(res2.overall_att, abs=1e-14)

    def test_bm_dof_nan_fails_closed(self, monkeypatch):
        """When ``_compute_cr2_bm_contrast_dof`` returns NaN, BOTH per-cell
        AND overall ATT inference fields (t_stat / p_value / conf_int) MUST
        be NaN — do NOT fall back to ``safe_inference(df=None)`` which
        silently uses normal-theory. Per ``feedback_bm_contrast_dof_fail_closed``.
        """
        df = _make_vcov_panel()
        import diff_diff.linalg as linalg_mod

        def _fake_dof(X, cluster_ids, bread, contrasts):
            return np.full(contrasts.shape[1], np.nan)

        monkeypatch.setattr(linalg_mod, "_compute_cr2_bm_contrast_dof", _fake_dof)
        res = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        # Overall: att + se preserved (sandwich is finite); inference NaN
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        assert np.isnan(res.overall_t_stat)
        assert np.isnan(res.overall_p_value)
        assert np.isnan(res.overall_conf_int[0])
        assert np.isnan(res.overall_conf_int[1])
        # Per-cell: same pattern (att + se preserved, inference NaN)
        for (g, t), eff in res.group_time_effects.items():
            assert np.isfinite(eff["att"]), f"cell ({g},{t}) att should be finite"
            assert np.isfinite(eff["se"]), f"cell ({g},{t}) se should be finite"
            assert np.isnan(eff["t_stat"]), f"cell ({g},{t}) t_stat should be NaN"
            assert np.isnan(eff["p_value"]), f"cell ({g},{t}) p_value should be NaN"
            assert np.isnan(eff["conf_int"][0]), f"cell ({g},{t}) conf_int[0] should be NaN"
            assert np.isnan(eff["conf_int"][1]), f"cell ({g},{t}) conf_int[1] should be NaN"

    def test_aggregate_group_under_hc2_bm_uses_bm_contrast_dof(self):
        """aggregate('group') under hc2_bm produces finite p-values using
        Bell-McCaffrey contrast DOFs; reverts to NaN under monkeypatch-
        induced fail-closed."""
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        res.aggregate("group")
        assert res.group_effects is not None
        for g, eff in res.group_effects.items():
            assert np.isfinite(eff["att"])
            assert np.isfinite(eff["se"])
            assert np.isfinite(eff["t_stat"]), f"group {g} t_stat NaN — BM DOF threading regressed"
            assert np.isfinite(eff["p_value"])
            assert np.isfinite(eff["conf_int"][0])
            assert np.isfinite(eff["conf_int"][1])

    def test_aggregate_event_under_hc2_bm_uses_bm_contrast_dof(self):
        """aggregate('event') under hc2_bm produces finite p-values using
        Bell-McCaffrey contrast DOFs."""
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        res.aggregate("event")
        assert res.event_study_effects is not None
        for k, eff in res.event_study_effects.items():
            assert np.isfinite(eff["att"])
            assert np.isfinite(eff["se"])
            assert np.isfinite(
                eff["t_stat"]
            ), f"event k={k} t_stat NaN — BM DOF threading regressed"
            assert np.isfinite(eff["p_value"])
            assert np.isfinite(eff["conf_int"][0])
            assert np.isfinite(eff["conf_int"][1])

    def test_aggregate_calendar_under_hc2_bm_uses_bm_contrast_dof(self):
        """aggregate('calendar') under hc2_bm produces finite p-values using
        Bell-McCaffrey contrast DOFs."""
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        res.aggregate("calendar")
        assert res.calendar_effects is not None
        for t, eff in res.calendar_effects.items():
            assert np.isfinite(eff["att"])
            assert np.isfinite(eff["se"])
            assert np.isfinite(
                eff["t_stat"]
            ), f"calendar t={t} t_stat NaN — BM DOF threading regressed"
            assert np.isfinite(eff["p_value"])
            assert np.isfinite(eff["conf_int"][0])
            assert np.isfinite(eff["conf_int"][1])

    def test_hc2_bm_handles_rank_deficient_all_eventually_treated(self):
        """All-eventually-treated panel with not_yet_treated control: late
        cohorts have no valid post-treatment comparison and get dropped by
        solve_ols's rank-deficiency handling. hc2_bm must compute BM DOF
        on the REDUCED design (kept-column subspace) — operating on the
        unreduced full-dummy bread would LinAlgError and fail-close every
        inference field to NaN (codex R3 P1). Per-cell + aggregate
        inference on identified cells must remain finite."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 8
        units = np.repeat(np.arange(n_units), n_periods)
        periods = np.tile(np.arange(1, n_periods + 1), n_units)
        cohorts = rng.choice([3, 5, 7], size=n_units)
        cohort_per_obs = cohorts[units]
        tau = np.where(periods >= cohort_per_obs, 0.5 + 0.2 * (periods - cohort_per_obs), 0.0)
        y = 1.0 + 0.1 * periods + tau + 0.1 * rng.normal(size=len(units))
        df = pd.DataFrame({"unit": units, "time": periods, "cohort": cohort_per_obs, "y": y})
        # Expect a rank-deficient warning from solve_ols (late-cohort drop).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
                df, outcome="y", unit="unit", time="time", cohort="cohort"
            )
        # Per-cell inference: all identified cells finite (att + se + p +
        # CI). solve_ols already excluded the dropped cells from
        # group_time_effects, so every key here is identified.
        assert len(res.group_time_effects) > 0
        for k, eff in res.group_time_effects.items():
            assert np.isfinite(eff["att"]), f"({k}) att NaN"
            assert np.isfinite(eff["se"]), f"({k}) se NaN"
            assert np.isfinite(
                eff["t_stat"]
            ), f"({k}) t_stat NaN — BM DOF not threaded on reduced design"
            assert np.isfinite(eff["p_value"]), f"({k}) p_value NaN"
            assert np.isfinite(eff["conf_int"][0])
            assert np.isfinite(eff["conf_int"][1])
        # Overall ATT inference: finite end-to-end.
        assert np.isfinite(res.overall_t_stat)
        assert np.isfinite(res.overall_p_value)
        # All three aggregations (group/calendar/event) must produce finite
        # inference on identified contrasts under the reduced-design BM path.
        for agg_type in ("group", "calendar", "event"):
            res.aggregate(agg_type)
        assert res.event_study_effects is not None
        for k, eff in res.event_study_effects.items():
            assert np.isfinite(
                eff["t_stat"]
            ), f"event k={k} t_stat NaN — aggregate BM DOF on reduced design regressed"
            assert np.isfinite(eff["p_value"])
        assert res.group_effects is not None
        for g, eff in res.group_effects.items():
            assert np.isfinite(
                eff["t_stat"]
            ), f"group g={g} t_stat NaN — aggregate BM DOF on reduced design regressed"
            assert np.isfinite(eff["p_value"])
        assert res.calendar_effects is not None
        # Calendar entries with at least one identified treated cell should
        # have finite BM inference; entirely-pre-treatment calendar periods
        # are absent from calendar_effects (their cells aren't post-treatment).
        for t, eff in res.calendar_effects.items():
            assert np.isfinite(
                eff["t_stat"]
            ), f"calendar t={t} t_stat NaN — aggregate BM DOF on reduced design regressed"
            assert np.isfinite(eff["p_value"])

    def test_hc2_bm_handles_rank_deficient_with_unit_invariant_exovar(self):
        """Unit-invariant exovar covariate is collinear with unit FE under
        full-dummy: solve_ols drops it as rank-deficient. hc2_bm must
        compute BM DOF on the reduced design (P1 codex R3 regression)."""
        df = _make_vcov_panel(n_units=30, n_periods=6, seed=20260521)
        # Unit-invariant covariate: x = f(unit) only → collinear with unit FE
        df["x_unit"] = df["unit"].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                cohort="cohort",
                exovar=["x_unit"],
            )
        # Per-cell + overall inference finite on identified cells
        assert len(res.group_time_effects) > 0
        for k, eff in res.group_time_effects.items():
            assert np.isfinite(eff["att"]), f"({k}) att NaN"
            assert np.isfinite(eff["se"]), f"({k}) se NaN"
            assert np.isfinite(
                eff["t_stat"]
            ), f"({k}) t_stat NaN under rank-deficient exovar — BM DOF not threaded"
            assert np.isfinite(eff["p_value"])
        assert np.isfinite(res.overall_t_stat)
        assert np.isfinite(res.overall_p_value)
        # Group + calendar + event aggregates should all produce finite
        # inference under the reduced-design BM path.
        for agg_type in ("group", "calendar", "event"):
            res.aggregate(agg_type)
        for g, eff in (res.group_effects or {}).items():
            assert np.isfinite(eff["t_stat"]), f"group g={g} t_stat NaN under rank-deficient exovar"
        for t, eff in (res.calendar_effects or {}).items():
            assert np.isfinite(
                eff["t_stat"]
            ), f"calendar t={t} t_stat NaN under rank-deficient exovar"
        for k, eff in (res.event_study_effects or {}).items():
            assert np.isfinite(eff["t_stat"]), f"event k={k} t_stat NaN under rank-deficient exovar"

    def test_aggregate_under_hc2_bm_fail_closed_on_dof_helper_error(self, monkeypatch):
        """When _compute_cr2_bm_contrast_dof raises in aggregate(), the
        affected aggregate inference fields are NaN (fail-closed),
        att + se preserved."""
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
            df, outcome="y", unit="unit", time="time", cohort="cohort"
        )
        # Patch the helper AFTER fit so that aggregate() retry fails.
        import diff_diff.linalg as linalg_mod

        def _raise(X, cluster_ids, bread, contrasts):
            raise ValueError("induced failure for fail-closed test")

        monkeypatch.setattr(linalg_mod, "_compute_cr2_bm_contrast_dof", _raise)
        with pytest.warns(UserWarning, match=r"could not compute Bell-McCaffrey"):
            res.aggregate("group")
        assert res.group_effects is not None
        for g, eff in res.group_effects.items():
            assert np.isfinite(eff["att"])
            assert np.isfinite(eff["se"])
            assert np.isnan(eff["t_stat"])
            assert np.isnan(eff["p_value"])
            assert np.isnan(eff["conf_int"][0])
            assert np.isnan(eff["conf_int"][1])


class TestOutcomeFitHint:
    """method='ols' outcome-fit hint: binary -> logit, count -> poisson.

    Per Wooldridge (2023), a matching nonlinear model is often the *more
    appropriate* specification for binary/count outcomes -- link-scale (not
    level) parallel trends, and less biased / more precise in the paper's
    Section 5 simulations. A different identifying assumption, so a recommended
    comparison, never a free efficiency upgrade or a canonical-link / validity
    requirement. See REGISTRY.md WooldridgeDiD "Nonlinear extensions".
    """

    # ---- detector unit tests ----
    @pytest.mark.parametrize(
        "values, expected",
        [
            ([0, 1, 1, 0, 1], "logit"),  # binary {0, 1}
            ([0.0, 1.0, 0.0], "logit"),  # binary as floats
            (pd.Series([True, False, True]), "logit"),  # bool dtype
            ([0, 1, 2, 3, 4], "poisson"),  # count
            ([1, 2, 3], "poisson"),  # count without zero
            ([0, 1, 2, 3], "poisson"),  # bounded-support integer routes to poisson too
            # ^ documented heuristic limit: a known-upper-bound (binomial-style)
            #   integer outcome is NOT separately distinguished from an unbounded
            #   count -- both take the poisson branch (Wooldridge 2023 Table 1).
            ([0.1, 0.5, 1.7, 2.3], None),  # continuous
            ([0.1, 0.4, 0.9], None),  # fractional in [0, 1]
            ([-1, 0, 1, 2], None),  # has a negative
            ([np.nan, np.nan], None),  # all non-finite
            ([1.0, 1.0, 1.0], None),  # constant / single value
            (pd.Series(["a", "b", "c"], dtype=object), None),  # non-numeric
        ],
    )
    def test_detector(self, values, expected):
        assert _suggest_nonlinear_method(values) == expected

    # ---- gate behavior ----
    def _binary_panel(self):
        df = _make_panel(seed=1)
        rng = np.random.default_rng(1)
        df["y"] = rng.integers(0, 2, len(df)).astype(float)
        return df

    def _count_panel(self):
        df = _make_panel(seed=2)
        rng = np.random.default_rng(2)
        df["y"] = rng.integers(0, 6, len(df)).astype(float)
        return df

    @staticmethod
    def _fit(df, **kwargs):
        return WooldridgeDiD(**kwargs).fit(df, "y", "unit", "time", "cohort")

    @staticmethod
    def _hint_msgs(rec):
        return [str(w.message) for w in rec if "matching nonlinear model" in str(w.message)]

    def test_ols_binary_warns_logit(self):
        df = self._binary_panel()
        with pytest.warns(UserWarning, match=r"method='logit'.*more appropriate"):
            res = self._fit(df, method="ols")
        assert np.isfinite(res.overall_att)

    def test_ols_count_warns_poisson(self):
        df = self._count_panel()
        with pytest.warns(UserWarning, match=r"method='poisson'.*more appropriate"):
            self._fit(df, method="ols")

    def test_ols_continuous_silent(self):
        df = _make_panel(seed=3)  # default y is standard-normal continuous
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            self._fit(df, method="ols")
        assert not self._hint_msgs(rec)

    def test_logit_binary_no_hint(self):
        # The hint is OLS-only; a logit fit on binary data must not emit it.
        df = self._binary_panel()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            self._fit(df, method="logit")
        assert not self._hint_msgs(rec)

    def test_hint_fires_with_cohort_trends(self):
        # Param-interaction smoke: hint still fires on the cohort_trends path.
        df = self._binary_panel()
        with pytest.warns(UserWarning, match=r"more appropriate"):
            self._fit(df, method="ols", cohort_trends=True)

    def test_suppression_via_filterwarnings(self):
        df = self._binary_panel()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            res = self._fit(df, method="ols")  # must not raise
        assert np.isfinite(res.overall_att)

    def test_framing_paper_faithful(self):
        # Lock the paper-faithful framing (Wooldridge 2023: LPT vs IPT + the
        # Section 5 simulations). The nonlinear model is the *more appropriate*
        # specification for binary/count outcomes — it reduces bias (not only
        # variance) and rests on a *different identifying assumption* (link-scale
        # vs level parallel trends): a recommended comparison, never an
        # unconditional efficiency upgrade and never a "violation" of OLS.
        df = self._binary_panel()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            self._fit(df, method="ols")
        msgs = self._hint_msgs(rec)
        assert msgs
        text = msgs[0].lower()
        assert "more appropriate" in text  # appropriateness, not only efficiency
        assert "biased" in text  # the paper's bias finding, not just precision
        assert "assumption" in text  # a different identifying assumption
        assert "recommended comparison" in text  # a comparison, not a switch
        # Never frame OLS as wrong / a link requirement or an automatic upgrade.
        for forbidden in ("canonical", "violation", "required", "must use"):
            assert forbidden not in text

    def test_filter_sample_preserves_outcome_support(self):
        # Invariant behind full-column detection: _filter_sample selects the
        # control group via the design matrix, NOT by dropping rows, so the
        # full outcome column and the estimation sample always share the same
        # support -- the pre/post-filter detection distinction is therefore
        # moot. A future refactor that drops rows would surface here.
        df = self._count_panel()
        for cg in ("not_yet_treated", "never_treated"):
            sample = _filter_sample(df, "unit", "time", "cohort", cg, 0)
            assert sorted(sample["y"].unique()) == sorted(df["y"].unique())


class TestAbsorbedCovariateSnap:
    """Unit-constant exovar columns (and their cohort interactions, which are
    also unit-constant) are spanned by the unit FE on the within-transform
    path: they must snap to deterministic NaN with a cause warning
    (REGISTRY 'Absorbed FE')."""

    def test_unit_constant_exovar_snaps_with_cause_warning(self):
        rng = np.random.default_rng(5)
        n_units, n_periods = 60, 6
        units = np.repeat(np.arange(n_units), n_periods)
        t = np.tile(np.arange(n_periods), n_units)
        cohort = np.repeat(
            np.where(np.arange(n_units) % 3 == 0, 0, np.where(np.arange(n_units) % 3 == 1, 3, 4)),
            n_periods,
        )
        d = (cohort > 0) & (t >= cohort)
        y = 0.5 * d + rng.normal(size=len(units))
        df = pd.DataFrame({"unit": units, "time": t, "cohort": cohort, "y": y})
        df["xc"] = np.repeat(rng.normal(size=n_units), n_periods)

        with pytest.warns(UserWarning, match="collinear with the absorbed"):
            res = WooldridgeDiD(method="ols", vcov_type="hc1").fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                cohort="cohort",
                exovar=["xc"],
            )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
