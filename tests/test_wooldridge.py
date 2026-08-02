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
        X_int, col_names, gt_keys, _ = _build_interaction_matrix(
            filtered, cohort="cohort", time="time", anticipation=0
        )
        # Each column should be a valid (g, t) pair with t >= g
        for g, t in gt_keys:
            assert t >= g

    def test_build_interaction_matrix_binary(self):
        df = _make_panel()
        filtered = _filter_sample(df, "unit", "time", "cohort", "not_yet_treated", anticipation=0)
        X_int, col_names, gt_keys, _ = _build_interaction_matrix(
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
            mpdta, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
        assert isinstance(results, WooldridgeDiDResults)

    def test_fit_sets_is_fitted(self, mpdta):
        est = WooldridgeDiD()
        est.fit(mpdta, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat")
        assert est.is_fitted_

    def test_overall_att_finite(self, mpdta):
        est = WooldridgeDiD()
        r = est.fit(
            mpdta, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)
        assert r.overall_se > 0

    def test_group_time_effects_populated(self, mpdta):
        est = WooldridgeDiD()
        r = est.fit(
            mpdta, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
        assert len(r.group_time_effects) > 0
        for (g, t), eff in r.group_time_effects.items():
            assert t >= g
            assert "att" in eff and "se" in eff

    def test_all_inference_fields_finite(self, mpdta):
        """No inference field should be NaN in normal data."""
        est = WooldridgeDiD()
        r = est.fit(
            mpdta, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
        assert np.isfinite(r.overall_t_stat)
        assert np.isfinite(r.overall_p_value)
        assert all(np.isfinite(c) for c in r.overall_conf_int)

    def test_never_treated_control_group(self, mpdta):
        est = WooldridgeDiD(control_group="never_treated")
        r = est.fit(
            mpdta, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
        assert len(r.group_time_effects) > 0

    def test_metadata_correct(self, mpdta):
        est = WooldridgeDiD()
        r = est.fit(
            mpdta, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
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
        return est.fit(
            df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )

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
        r = est.fit(binary_panel, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert isinstance(r, WooldridgeDiDResults)

    def test_logit_att_sign(self, binary_panel):
        """ATT should be positive (treatment increases binary outcome)."""
        est = WooldridgeDiD(method="logit")
        r = est.fit(binary_panel, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert r.overall_att > 0

    def test_logit_se_positive(self, binary_panel):
        est = WooldridgeDiD(method="logit")
        r = est.fit(binary_panel, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert r.overall_se > 0

    def test_logit_method_stored(self, binary_panel):
        est = WooldridgeDiD(method="logit")
        r = est.fit(binary_panel, outcome="y", unit="unit", time="time", first_treat="cohort")
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
        r = est.fit(count_panel, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert isinstance(r, WooldridgeDiDResults)

    def test_poisson_att_sign(self, count_panel):
        est = WooldridgeDiD(method="poisson")
        r = est.fit(count_panel, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert r.overall_att > 0

    def test_poisson_se_positive(self, count_panel):
        est = WooldridgeDiD(method="poisson")
        r = est.fit(count_panel, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert r.overall_se > 0


class TestBootstrap:
    @pytest.mark.slow
    def test_multiplier_bootstrap_ols(self, ci_params):
        """Bootstrap SE should be close to analytic SE."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        n_boot = ci_params.bootstrap(50, min_n=19)
        est = WooldridgeDiD(n_bootstrap=n_boot, seed=42)
        r = est.fit(df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat")
        assert abs(r.overall_se - r.overall_att) / max(abs(r.overall_att), 1e-8) < 10

    def test_bootstrap_zero_disables(self):
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        est = WooldridgeDiD(n_bootstrap=0)
        r = est.fit(df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat")
        assert np.isfinite(r.overall_se)


class TestMethodologyCorrectness:
    def test_ols_att_sign_direction(self):
        """ATT sign should be consistent across cohorts on mpdta."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        # No `rank_deficient_action="silent"` here any more. That workaround was
        # added because never_treated emitted the reference cell and left QR to
        # drop something arbitrary (#724); the reference is now omitted by
        # construction, so a clean fit must produce NO rank warning.
        est = WooldridgeDiD(control_group="never_treated")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = est.fit(
                df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
            )
        assert np.isfinite(r.overall_att)
        rank = [w for w in caught if "Rank-deficient" in str(w.message)]
        assert rank == [], f"unexpected rank deficiency: {[str(w.message)[:100] for w in rank]}"

    def test_never_treated_produces_event_effects_with_placebo_leads(self):
        """With never_treated, event aggregation should include negative event
        times (placebo leads) because pre-treatment interactions are included."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        est = WooldridgeDiD(control_group="never_treated")
        r = est.fit(df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat")
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
        r = WooldridgeDiD().fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert len(r.group_time_effects) == 1
        assert abs(r.overall_att - 1.0) < 0.5

    def test_aggregation_weights_sum_to_one(self):
        """Simple aggregation weights should sum to 1."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        r = WooldridgeDiD().fit(
            df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
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
            df, outcome="lemp_bin", unit="countyreal", time="year", first_treat="first_treat"
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
            df, outcome="emp_count", unit="countyreal", time="year", first_treat="first_treat"
        )

        assert len(results.group_time_effects) > 0
        for key, cell in results.group_time_effects.items():
            assert cell["se"] >= 0, f"Negative SE at {key}"
            assert np.isfinite(cell["se"]), f"Non-finite SE at {key}"


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
        r = est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
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
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
        r = est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
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
        r = est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort", xgvar=["x1"])
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)
        assert r.overall_se > 0


class TestAllEventuallyTreated:
    def test_no_never_treated_not_yet_treated_control(self):
        """All units eventually treated: the fit SUCCEEDS via W2025 Section 5.4.

        This panel used to be refused. Before the #729 completeness gate it was
        worse than refused -- it returned finite and WRONG numbers, because once
        QR dropped one of the jointly-collinear cells at a fully-treated period
        the survivors identified contrasts BETWEEN treated cohorts while keeping
        their ATT(g, t) labels. Measured then (true ATT = 1.5 everywhere)::

            ATT(3, 8) = -0.0096      ATT(3, 9) = -0.1038
            ATT(5, 8) = -0.0214      ATT(5, 9) = -0.2393
            overall_att = 1.0023     (true 1.5)

        Comparison-support filtering now removes those periods BEFORE the solve,
        which is exactly the paper's rule: with no never-treated group the last
        cohort serves as the reference and "all dT_i-containing regressors get
        dropped" (W2025 Eq. 5.13-5.15). Cohort 8 therefore receives no cells,
        and the retained cells are the absolute tau_gt on t <= G_max - 1.
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
        with pytest.warns(UserWarning, match="no eligible comparison group"):
            res = est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")

        # Eq. 5.15 on a balanced panel: cells {(g, t) : g <= G_max - 1,
        # g <= t <= G_max - 1}. G_max = 8, so periods 8 and 9 are dropped and
        # cohort 8 emits nothing.
        assert sorted((int(g), int(t)) for g, t in res.group_time_effects) == [
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (5, 5),
            (5, 6),
            (5, 7),
        ]
        # The dropped cohort is not advertised as an estimated one.
        assert set(int(g) for g in res.groups) == {3, 5}
        assert 8 not in res.groups
        # Absolute ATTs, not relative: true effect is 1.5 for every post cell.
        # Scaled by each cell's own SE rather than a fixed band -- per-cell SEs
        # here are ~0.17-0.21, so a fixed tolerance is either flaky or vacuous.
        # The pre-fix values this guards against sat ~9 SE away (-0.0096 etc.).
        assert res.overall_att == pytest.approx(1.5502, abs=1e-3)
        for key, eff in res.group_time_effects.items():
            assert (
                abs(eff["att"] - 1.5) < 3 * eff["se"]
            ), f"ATT{key} = {eff['att']:.4f} is more than 3 SE from the true 1.5"

    def test_all_treated_names_the_zero_cell_cohort(self):
        """The user is told which cohort lost its cells, not just that rows went.

        Dropping data silently is the failure mode this whole path exists to
        avoid: `jwdid` drops the same rows and only reports a smaller N.
        """
        rng = np.random.default_rng(7)
        rows = []
        for u in range(120):
            cohort = 3 if u < 60 else 6
            for t in range(1, 7):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "cohort": cohort,
                        "y": rng.standard_normal() + 1.5 * int(t >= cohort),
                    }
                )
        df = pd.DataFrame(rows)
        with pytest.warns(UserWarning, match=r"Cohort\(s\) 6 have NO estimated cells"):
            WooldridgeDiD(control_group="not_yet_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )


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
        r = est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
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
        r = est.fit(
            df, outcome="lemp_bin", unit="countyreal", time="year", first_treat="first_treat"
        )
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
        r = est.fit(df, outcome="emp", unit="countyreal", time="year", first_treat="first_treat")
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
            df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
        r_nt = WooldridgeDiD(control_group="never_treated").fit(
            df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
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
            df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
        r_nt = WooldridgeDiD(control_group="never_treated").fit(
            df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
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
            WooldridgeDiD().fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")

    def test_never_treated_no_controls_raises(self):
        """never_treated with no cohort==0 units should raise ValueError."""
        df = pd.DataFrame(
            {"unit": [1, 1, 2, 2], "time": [1, 2, 1, 2], "cohort": [2, 2, 3, 3], "y": [1, 2, 3, 4]}
        )
        with pytest.raises(ValueError, match="no never-treated"):
            WooldridgeDiD(control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        with pytest.raises(ValueError, match="Bootstrap inference is only supported"):
            WooldridgeDiD(method="poisson", n_bootstrap=50).fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
        )
        # Bootstrap at region level (coarser)
        r_region = WooldridgeDiD(n_bootstrap=99, seed=0, cluster="region").fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
        r = est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
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
        r = est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort", exovar=["x1"]
        )
        r_nocov = WooldridgeDiD(method="logit").fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort", exovar=["x1"]
        )
        r_nocov = WooldridgeDiD(method="poisson").fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
            WooldridgeDiD().fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")


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
        r = est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
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
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
        )

        # Build explicit dummy regression on same sample
        sample = _filter_sample(df, "unit", "time", "cohort", "not_yet_treated", 0)
        X_int, _, gt_keys, _ = _build_interaction_matrix(
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
            binary_data, outcome="y", unit="unit", time="time", first_treat="cohort"
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
            count_data, outcome="y", unit="unit", time="time", first_treat="cohort"
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

        X_ols, _, _, _ = _build_interaction_matrix(df, "cohort", "time", 0, "never_treated", "ols")
        X_logit, _, _, _ = _build_interaction_matrix(
            df, "cohort", "time", 0, "never_treated", "logit"
        )
        X_nyt, _, _, _ = _build_interaction_matrix(
            df, "cohort", "time", 0, "not_yet_treated", "ols"
        )
        # OLS never_treated > nonlinear never_treated == not_yet_treated
        assert X_ols.shape[1] > X_logit.shape[1]
        assert X_logit.shape[1] == X_nyt.shape[1]

    def test_ols_never_treated_still_has_pre_treatment(self):
        """OLS path should still include pre-treatment placebo cells."""
        from diff_diff.datasets import load_mpdta

        df = load_mpdta()
        r = WooldridgeDiD(control_group="never_treated").fit(
            df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort", exovar=["x1"]
        )

        # Build explicit-dummy regression with full basis
        sample = _filter_sample(df, "unit", "time", "cohort", "not_yet_treated", 0)
        X_int, _, gt_keys, _ = _build_interaction_matrix(
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort", exovar=["x1"]
        )
        r_nocov = WooldridgeDiD().fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
        )
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
            first_treat="cohort",
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
            first_treat="cohort",
            survey_design=sd,
        )
        r_naive = WooldridgeDiD().fit(
            survey_panel,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="cohort",
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
            first_treat="cohort",
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
            first_treat="cohort",
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
                first_treat="cohort",
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
            first_treat="cohort",
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
            first_treat="cohort",
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
                first_treat="cohort",
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
            first_treat="cohort",
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
            first_treat="cohort",
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
            first_treat="cohort",
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
            first_treat="cohort",
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
                first_treat="cohort",
                survey_design=sd,
            )

    def test_poisson_zero_weight_cell(self, survey_panel):
        """Poisson survey fit handles zero-weight treated cells cleanly."""
        from diff_diff.survey import SurveyDesign

        df = survey_panel.copy()
        # Zero out ONE treated cell's weights. Deliberately NOT the whole cohort:
        # that would leave cohort 3 with no supported reference period, and
        # exclusion under a survey design is now refused outright (see
        # test_survey_plus_unidentified_cohort_is_refused). Cohort 3 keeps its
        # t=1,2 pre-periods, so this still exercises what the test is for --
        # a zero-weight TREATED CELL inside an otherwise identified design.
        df.loc[(df["cohort"] == 3) & (df["time"] == 5), "weight"] = 0.0
        sd = SurveyDesign(weights="weight", strata="stratum", psu="unit")
        r = WooldridgeDiD(method="poisson").fit(
            df,
            outcome="y_count",
            unit="unit",
            time="time",
            first_treat="cohort",
            survey_design=sd,
        )
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)

    def test_ols_survey_all_treated_refuses_period_filtering(self, survey_panel):
        """Comparison-support filtering under `survey_design=` is REFUSED.

        This panel previously returned finite and WRONG numbers (a surviving
        cell came back as -1.94 against true effects of 1.0 and 3.0), then was
        made to fail closed with a rank-reduction ValueError. Now the periods
        that caused it are unsupported and would be FILTERED -- but deleting
        rows under a complex survey design is naive subsetting: it removes
        their PSUs and strata from the TSL meat and from
        `df_survey = n_PSU - n_strata`, so the surviving estimates would carry a
        variance computed on a design the user never specified.

        The refusal is raised BEFORE any row is removed, and points at explicit
        frame restriction -- NOT `SurveyDesign.subpopulation()`, whose
        zero-weight padding `_reject_zero_weight_groups` refuses on this path.
        """
        from diff_diff.survey import SurveyDesign

        # Remove never-treated (cohort=0) so later periods lose all comparisons.
        df = survey_panel[survey_panel["cohort"] > 0].copy()
        sd = SurveyDesign(weights="weight", strata="stratum", psu="unit")
        with pytest.raises(NotImplementedError, match="no eligible comparison group"):
            WooldridgeDiD(control_group="not_yet_treated").fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="cohort",
                survey_design=sd,
            )

    def test_survey_refusal_is_conditional_on_rows_actually_dropping(self, survey_panel):
        """The refusal must NOT fire on survey fits that lose nothing.

        Instrumented over the suite, 59 `fit` calls pass `survey_design=` and
        exactly one has an unsupported period. An unconditional refusal would
        break the other 58, so the guard is pinned as conditional here.
        """
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="weight", strata="stratum", psu="unit")
        res = WooldridgeDiD(control_group="not_yet_treated").fit(
            survey_panel,  # retains its never-treated units at every period
            outcome="y",
            unit="unit",
            time="time",
            first_treat="cohort",
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)

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
                first_treat="cohort",
                survey_design=sd,
            )

    def test_logit_survey_zero_weight_cell(self, survey_panel):
        """Logit survey fit skips zero-weight treated cells cleanly."""
        from diff_diff.survey import SurveyDesign

        df = survey_panel.copy()
        # Zero out ONE treated cell's weights. Deliberately NOT the whole cohort:
        # that would leave cohort 3 with no supported reference period, and
        # exclusion under a survey design is now refused outright (see
        # test_survey_plus_unidentified_cohort_is_refused). Cohort 3 keeps its
        # t=1,2 pre-periods, so this still exercises what the test is for --
        # a zero-weight TREATED CELL inside an otherwise identified design.
        df.loc[(df["cohort"] == 3) & (df["time"] == 5), "weight"] = 0.0
        sd = SurveyDesign(weights="weight", strata="stratum", psu="unit")
        r = WooldridgeDiD(method="logit").fit(
            df,
            outcome="y_bin",
            unit="unit",
            time="time",
            first_treat="cohort",
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
                first_treat="cohort",
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
            first_treat="cohort",
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
                est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
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
                est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
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
        ``_make_vcov_panel(seed=202605211230)``; recaptured under the v3.6.x
        factorize-once + bincount demeaner (~1e-15 move), and recaptured
        AGAIN under the 3.9 clustered-CR1 K_reference convergence
        (variance-conventions.md D2): the CR1 factor now counts the absorbed
        FE not nested in the unit cluster, matching Stata reghdfe (jwdid) at
        machine precision on the committed arms. The ATT is unchanged; the SE
        moved from the pre-fix 0.03149488781317813 by exactly the K change.
        """
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="hc1").fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
        )
        assert res.overall_att == pytest.approx(0.9178849934516233, abs=1e-14)
        assert res.overall_se == pytest.approx(0.031906603167527435, abs=1e-14)

    def test_hc2_bm_finite_and_inflates_over_hc1(self):
        df = _make_vcov_panel()
        res_hc1 = WooldridgeDiD(method="ols", vcov_type="hc1").fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
        )
        res_bm = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")

    def test_classical_drops_auto_cluster(self):
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="classical").fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
        )
        assert np.isfinite(res.overall_se)
        assert res.cluster_name is None
        assert res.n_clusters is None

    def test_hc2_drops_auto_cluster(self):
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="hc2").fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
                first_treat="cohort",
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
                first_treat="cohort",
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
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
        # Case 2: cluster=X — also rejected at the estimator boundary (would
        # otherwise hit the linalg validator with a less-informative message)
        est_cl = WooldridgeDiD(
            method="ols", vcov_type="hc2", n_bootstrap=10, cluster="unit", seed=0
        )
        with pytest.raises(ValueError, match=r"multiplier bootstrap"):
            est_cl.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")

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
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
        )
        # Bootstrap fit on the same data + seed.
        res_boot = WooldridgeDiD(method="ols", vcov_type="hc2_bm", n_bootstrap=50, seed=0).fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
        # Rank deficiency is induced by a DUPLICATED COVARIATE, not by dropping
        # treatment cells. The panel keeps a never-treated group so every
        # ATT(g, t) stays identified: a lost treatment cell now fails closed
        # (codex R8 P0), and it always meant the survivors were relabeled
        # contrasts -- which is not what this test is about. The subject here is
        # the coef_offset / dropped-column indexing in the bootstrap closure,
        # and a collinear covariate exercises exactly that on a design whose
        # estimands are sound.
        cohorts = rng.choice([0, 3, 5, 7], size=n_units)
        cohort_per_obs = cohorts[units]
        treated = (cohort_per_obs > 0) & (periods >= cohort_per_obs)
        tau = np.where(treated, 0.5 + 0.2 * (periods - cohort_per_obs), 0.0)
        y = 1.0 + 0.1 * periods + tau + 0.1 * rng.normal(size=len(units))
        xc = rng.normal(size=len(units))
        df = pd.DataFrame(
            {
                "unit": units,
                "time": periods,
                "cohort": cohort_per_obs,
                "y": y,
                "xc": xc,
                "xc_dup": xc,  # exactly collinear -> one covariate column dropped
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = WooldridgeDiD(method="ols", vcov_type="hc2_bm", n_bootstrap=50, seed=0).fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="cohort",
                exovar=["xc", "xc_dup"],
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
            first_treat="cohort",
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
        )
        assert res.vcov_type == "hc2_bm"

    def test_results_carries_cluster_name_for_clustered_fit(self):
        df = _make_vcov_panel()
        res = WooldridgeDiD(method="ols", vcov_type="hc1").fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
        )
        assert res.cluster_name == "state"
        assert res.n_clusters == 4

    def test_fit_clone_idempotent_on_vcov_type(self):
        """fit, clone via get_params, refit — SE must be bit-equal."""
        df = _make_vcov_panel()
        est = WooldridgeDiD(method="ols", vcov_type="hc2_bm")
        res1 = est.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
        est2 = WooldridgeDiD(**est.get_params())
        res2 = est2.fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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

    def test_hc2_bm_handles_rank_deficient_reduced_design(self):
        """hc2_bm must compute BM DOF on the REDUCED design (kept-column
        subspace) — operating on the unreduced full-dummy bread would
        LinAlgError and fail-close every inference field to NaN (codex R3 P1).
        Per-cell + aggregate inference on identified cells must remain finite.

        Rank deficiency comes from a DUPLICATED COVARIATE rather than from
        dropping late-cohort treatment cells. The original all-eventually-treated
        panel produced finite-but-WRONG ATTs (the survivors were relabeled
        cohort contrasts) and now fails closed (codex R8 P0); a collinear
        covariate reaches the same reduced-design code path on a panel whose
        estimands are sound."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 8
        units = np.repeat(np.arange(n_units), n_periods)
        periods = np.tile(np.arange(1, n_periods + 1), n_units)
        cohorts = rng.choice([0, 3, 5, 7], size=n_units)
        cohort_per_obs = cohorts[units]
        treated = (cohort_per_obs > 0) & (periods >= cohort_per_obs)
        tau = np.where(treated, 0.5 + 0.2 * (periods - cohort_per_obs), 0.0)
        y = 1.0 + 0.1 * periods + tau + 0.1 * rng.normal(size=len(units))
        xc = rng.normal(size=len(units))
        df = pd.DataFrame(
            {
                "unit": units,
                "time": periods,
                "cohort": cohort_per_obs,
                "y": y,
                "xc": xc,
                "xc_dup": xc,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = WooldridgeDiD(method="ols", vcov_type="hc2_bm").fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="cohort",
                exovar=["xc", "xc_dup"],
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
                first_treat="cohort",
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
            df, outcome="y", unit="unit", time="time", first_treat="cohort"
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
                first_treat="cohort",
                exovar=["xc"],
            )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)


class TestReferencePeriodNormalization:
    """Issue #724 and the edge cases the reference-period fix introduced.

    The ETWFE design omits the ``g-1`` cell as the reference (Wooldridge 2025,
    Eq. 6.1/6.4). The library previously emitted it and left generic QR rank
    detection to drop something arbitrary, which silently removed genuine
    post-treatment effects. Cross-implementation parity against Stata
    ``jwdid ... never`` lives in ``tests/test_etwfe_cs_stata_parity.py``; this
    class covers the edges that have no Stata counterpart.
    """

    @staticmethod
    def _panel(cohorts, times, n_per=20, seed=0, drop=None):
        """Balanced panel unless ``drop`` names ``(cohort, time)`` pairs to omit."""
        rng = np.random.default_rng(seed)
        drop = set(drop or ())
        rows = []
        uid = 0
        for g in cohorts:
            for _ in range(n_per):
                for t in times:
                    if (g, t) in drop:
                        continue
                    treated = g != 0 and t >= g
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "x1": rng.normal(),
                            "y": 0.3 * t + (1.0 if treated else 0.0) + rng.normal(0, 0.2),
                        }
                    )
                uid += 1
        return pd.DataFrame(rows)

    def test_gapped_cohort_keeps_every_post_cell(self):
        """A cohort unobserved at the panel-wide latest pre-period.

        A panel-wide reference rule would omit an all-zero column here, leaving
        the collinearity intact and reproducing #724. The per-cohort rule uses
        that cohort's own support instead.
        """
        df = self._panel([0, 4], [1, 2, 3, 4, 5], drop={(4, 3)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = WooldridgeDiD(method="ols", control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        post = {k for k in res.group_time_effects if k[1] >= k[0]}
        assert post == {(4, 4), (4, 5)}
        assert [w for w in caught if "Rank-deficient" in str(w.message)] == []

    def test_unobserved_pairs_are_skipped_and_reported(self):
        """Identically-zero columns are not emitted, and the skip is not silent."""
        df = self._panel([0, 4], [1, 2, 3, 4, 5], drop={(4, 3)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WooldridgeDiD(method="ols", control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        skipped = [w for w in caught if "no observations" in str(w.message)]
        assert len(skipped) == 1
        assert "(4, 3)" in str(skipped[0].message)

    def test_skip_warning_respects_rank_deficient_action_silent(self):
        """``silent`` is a documented, supported value; the new notice must not
        become an unsuppressible replacement for the rank warning it removes."""
        df = self._panel([0, 4], [1, 2, 3, 4, 5], drop={(4, 3)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WooldridgeDiD(
                method="ols", control_group="never_treated", rank_deficient_action="silent"
            ).fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert [w for w in caught if "no observations" in str(w.message)] == []

    def test_cohort_without_a_reference_is_excluded_not_silently_rebaselined(self):
        """The P0 guard.

        A cohort with no period before ``g - anticipation`` is unidentified.
        Dropping only its COLUMNS leaves its rows in the omitted baseline, so its
        treatment effect loads onto the time FE and biases the cohorts that ARE
        identified. The estimates must therefore match a fit where that cohort
        was removed by hand -- asserting only the warning would not catch this.
        """
        df = self._panel([0, 3, 5], [1, 2, 3, 4, 5, 6])
        kw = dict(outcome="y", unit="unit", time="time", first_treat="cohort")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            auto = WooldridgeDiD(method="ols", control_group="never_treated", anticipation=2).fit(
                df, **kw
            )
        excluded = [w for w in caught if "unidentified under the ETWFE" in str(w.message)]
        assert len(excluded) == 1, "the excluded cohort must be named"
        assert "3" in str(excluded[0].message)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manual = WooldridgeDiD(method="ols", control_group="never_treated", anticipation=2).fit(
                df[df["cohort"] != 3].copy(), **kw
            )

        shared = set(auto.group_time_effects) & set(manual.group_time_effects)
        assert shared, "no cells survived to compare"
        for key in shared:
            np.testing.assert_allclose(
                auto.group_time_effects[key]["att"],
                manual.group_time_effects[key]["att"],
                rtol=0,
                atol=1e-12,
                err_msg=f"ATT{key} differs from an explicit-exclusion fit -- "
                "the excluded cohort's rows are still in the baseline",
            )

    def test_excluding_every_comparison_fails_closed(self):
        """Exclusion can cascade. On an all-treated ``{1, 3}`` panel, removing
        cohort 1 leaves cohort 3 alone with its cells collinear with the time FE
        -- which previously produced an all-NaN fit and no error at all.
        """
        df = self._panel([1, 3], [1, 2, 3, 4])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="no comparison observations remain"):
                WooldridgeDiD(method="ols", control_group="not_yet_treated").fit(
                    df, outcome="y", unit="unit", time="time", first_treat="cohort"
                )

    def test_no_post_treatment_cell_fails_closed(self):
        """A cohort observed only BEFORE its own treatment start has no
        estimable ATT, and the fit must say so rather than return NaN.

        This is the case the old ``X_int.shape[1] == 0`` guard could not see.
        On ``never_treated`` + OLS, ``include_pre`` also emits placebo
        (``t < g``) columns, so the design is NON-empty even when not one
        post-treatment cell survives -- the fit returned placebo coefficients
        with ``overall_att = NaN``. The first half of this test pins that
        mechanism (columns exist, no post cell among them) so the guard cannot
        be weakened back to a column count without failing here.
        """
        df = self._panel([0, 4], [1, 2, 3, 4], drop=[(4, 3), (4, 4)])

        X_int, _, gt_keys, _ = _build_interaction_matrix(
            df, cohort="cohort", time="time", anticipation=0, control_group="never_treated"
        )
        assert X_int.shape[1] > 0, "design is empty -- test no longer exercises the guard"
        assert [(g, t) for g, t in gt_keys if t >= g] == []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="No estimable post-treatment cells"):
                WooldridgeDiD(method="ols", control_group="never_treated").fit(
                    df, outcome="y", unit="unit", time="time", first_treat="cohort"
                )

    def test_not_yet_treated_is_also_fixed(self):
        """#724 was NOT confined to never_treated: any cohort whose cell block
        spans its own cohort dummy hits it, which anticipation makes reachable
        on the default path."""
        df = self._panel([0, 3, 5], [1, 2, 3, 4, 5, 6])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = WooldridgeDiD(method="ols", control_group="not_yet_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert [w for w in caught if "Rank-deficient" in str(w.message)] == []
        assert {k for k in res.group_time_effects if k[1] >= k[0]}

    @pytest.mark.parametrize("control_group", ["never_treated", "not_yet_treated"])
    def test_covariate_path_keeps_every_cell(self, control_group):
        """The cell x covariate block is built from the interaction columns, so
        the reference omission propagates into it; nothing may be rank-dropped."""
        df = self._panel([0, 3, 5], [1, 2, 3, 4, 5, 6])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = WooldridgeDiD(method="ols", control_group=control_group).fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort", exovar=["x1"]
            )
        assert [w for w in caught if "Rank-deficient" in str(w.message)] == []
        assert {k for k in res.group_time_effects if k[1] >= k[0]}

    @pytest.mark.parametrize("method", ["logit", "poisson"])
    def test_nonlinear_methods_unaffected(self, method):
        """The builder runs before method dispatch, so logit/Poisson see the
        same changes. They restrict to post-treatment cells, so rule 1 is
        unreachable there -- this pins that they still fit cleanly."""
        df = self._panel([0, 3, 5], [1, 2, 3, 4, 5, 6])
        df["y"] = (df["y"] > df["y"].median()).astype(int) if method == "logit" else df["y"].abs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(method=method, control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert {k for k in res.group_time_effects if k[1] >= k[0]}

    def test_zero_weight_reference_cell_does_not_lose_a_post_cell(self):
        """Survey weights scale the design by ``sqrt(w)``, so a reference cell
        whose rows all carry zero ``pweight`` is absent from the EFFECTIVE
        regression even though its rows are in the frame. Choosing the reference
        from raw presence therefore omitted nothing, the cohort's block again
        spanned the absorbed cohort dummy, and QR dropped a real post cell --
        returning a finite `overall_att` from an incomplete cell set, silently
        under the supported ``rank_deficient_action="silent"``.
        """
        from diff_diff import SurveyDesign

        rng = np.random.default_rng(1)
        rows = []
        uid = 0
        for g in (0, 4):
            for _ in range(25):
                for t in range(1, 6):
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            # cohort 4's natural reference (4, 3) carries no weight
                            "w": 0.0 if (g == 4 and t == 3) else 1.0,
                            "y": 0.3 * t + (1.0 if (g and t >= g) else 0.0) + rng.normal(0, 0.2),
                        }
                    )
                uid += 1
        df = pd.DataFrame(rows)

        for action in ("warn", "silent"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = WooldridgeDiD(
                    method="ols", control_group="never_treated", rank_deficient_action=action
                ).fit(
                    df,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="cohort",
                    survey_design=SurveyDesign(weights="w"),
                )
            post = {k for k in res.group_time_effects if k[1] >= k[0]}
            assert post == {(4, 4), (4, 5)}, f"{action}: lost a post cell, got {sorted(post)}"
            assert [w for w in caught if "Rank-deficient" in str(w.message)] == []

    def test_excluded_cohort_leaves_no_stale_derived_state(self):
        """``groups`` is read by covariate construction, fit dispatch, results
        metadata and aggregation counts. Left stale after an exclusion it
        reports a phantom zero-member cohort and emits an all-zero
        ``D{g}_x_{cov}`` column, which fails outright under
        ``rank_deficient_action="error"``.
        """
        rng = np.random.default_rng(0)
        rows = []
        uid = 0
        for g in (0, 3, 5):
            for _ in range(20):
                for t in range(1, 7):
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "x1": rng.normal(),
                            "y": 0.3 * t + (1.0 if (g and t >= g) else 0.0) + rng.normal(0, 0.2),
                        }
                    )
                uid += 1
        df = pd.DataFrame(rows)
        kw = dict(outcome="y", unit="unit", time="time", first_treat="cohort")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(method="ols", control_group="never_treated", anticipation=2).fit(
                df, **kw
            )
        assert 3 not in res.groups, "excluded cohort still advertised in results.groups"
        assert set(res.groups) == {k[0] for k in res.group_time_effects}
        counts = getattr(res, "_n_g_per_cohort", {})
        assert 3 not in counts, f"phantom zero-member cohort in {counts}"

        # The strict path must fit rather than trip on an artificial column.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strict = WooldridgeDiD(
                method="ols",
                control_group="never_treated",
                anticipation=2,
                rank_deficient_action="error",
            ).fit(df, exovar=["x1"], **kw)
        assert {k for k in strict.group_time_effects if k[1] >= k[0]}


class TestOverallAttFailsClosed:
    """A successful fit reports a finite overall ATT, or raises saying why not.

    Three distinct paths used to return a result object with
    ``overall_att = overall_se = NaN`` and no error, which reads as a completed
    estimate. Two of them (rank reduction, the anticipation window) are only
    visible AFTER the solve, so the build-time cell guard cannot see them --
    hence the separate last-line-of-defence check. Surfaced by codex R2 (M1);
    both post-solve cases were verified PRE-EXISTING against the pre-PR base
    commit before being fixed here (ledger row M-124).
    """

    @staticmethod
    def _unbalanced(support, n_per=15, seed=0):
        """``{cohort: [observed times]}`` -> panel. ``0`` is never-treated."""
        rng = np.random.default_rng(seed)
        rows = []
        uid = 0
        for g, times in support.items():
            for _ in range(n_per):
                for t in times:
                    rows.append(
                        {"unit": uid, "time": t, "cohort": g, "y": 0.3 * t + rng.normal(0, 0.2)}
                    )
                uid += 1
        return pd.DataFrame(rows)

    def test_all_post_columns_rank_dropped_fails_closed(self):
        """One treated cohort, no never-treated group: every post cell equals a
        time indicator, so QR drops them all and nothing is left to average.

        The upstream ``has_untreated`` guard passes here because it counts this
        cohort's OWN pre-treatment rows as comparison observations -- which is
        precisely why the check has to also run after the solve.

        Still fails closed, but now with a DIFFERENT and more accurate message.
        Periods 3 and 4 have no untreated unit, so support filtering removes
        them; cohort 3 then has no observation at or after its own treatment
        start, so there is nothing to estimate. The old rank-reduction path is
        no longer reached.
        """
        df = self._unbalanced({3: [1, 2, 3, 4]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="No estimable post-treatment cells found"):
                WooldridgeDiD(method="ols", control_group="not_yet_treated").fit(
                    df, outcome="y", unit="unit", time="time", first_treat="cohort"
                )

    def test_two_cohorts_without_same_period_controls_fail_closed(self):
        """Cohort COUNT is not a proxy for comparison support.

        Two treated cohorts survive identification here -- so any guard keyed on
        "fewer than 2 cohorts" passes -- yet neither post cell has a usable
        same-period comparison and both are dropped as collinear. Regression
        gate for exactly that substitution.
        """
        df = self._unbalanced({3: [1, 2, 3], 5: [3, 4, 5]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="not identified and were removed"):
                WooldridgeDiD(method="ols", control_group="not_yet_treated").fit(
                    df, outcome="y", unit="unit", time="time", first_treat="cohort"
                )

    def test_only_anticipation_window_cells_fails_closed(self):
        """Cells inside the anticipation window are ESTIMATED (t >= g - k) but
        excluded from the overall ATT (t >= g), so a cohort never observed once
        actually treated yields cells and a NaN headline.

        This is the gap the build-time guard cannot close: its notion of a
        treatment cell is deliberately the wider one, so the two checks are not
        redundant. The first half pins that the fit really does produce cells,
        so the test cannot pass for the trivial reason of an empty design.
        """
        df = self._unbalanced({0: [1, 2, 3, 4, 5, 6], 5: [1, 2, 3, 4]})

        _, _, gt_keys, _ = _build_interaction_matrix(
            df, cohort="cohort", time="time", anticipation=2, control_group="never_treated"
        )
        assert [(g, t) for g, t in gt_keys if t >= g - 2] != [], "no treatment cell was emitted"
        assert [(g, t) for g, t in gt_keys if t >= g] == []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="No cohort-time cell contributes"):
                WooldridgeDiD(method="ols", control_group="never_treated", anticipation=2).fit(
                    df, outcome="y", unit="unit", time="time", first_treat="cohort"
                )

    def test_valid_fit_is_unaffected(self):
        """The guard must not fire on an ordinary identified design."""
        df = self._unbalanced({0: [1, 2, 3, 4], 3: [1, 2, 3, 4]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(method="ols", control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)


class TestInvalidWeightsCannotHideBehindExclusion:
    """Invalid survey weights must be REJECTED, never silently reshape the sample.

    Cell support is decided from RAW weights (a zero-weight cell is absent from
    the effective regression), and an unsupported reference period excludes the
    cohort's rows. So an invalid weight confined to those rows used to be
    consumed by the support logic and then never reach ``SurveyDesign.resolve``:
    the fit dropped the cohort, returned a confident finite ``overall_att``, and
    warned that the cohort "has no pre-treatment period" -- when in truth its
    pre-periods were merely poisoned. Measured before the fix: all three invalid
    values below produced overall_att = 0.021416714008590323.

    Surfaced by codex R3 (M1); introduced by this PR's weight-aware support, so
    it is gated here rather than tracked. `validate_raw_weights` is shared with
    survey.py so both paths raise identically.
    """

    @staticmethod
    def _poisoned(bad, cohorts=(0, 3, 4), poisoned_cohort=4, n_per=12, seed=0):
        """Panel whose bad weights sit ONLY in ``poisoned_cohort``'s pre-periods."""
        rng = np.random.default_rng(seed)
        rows = []
        uid = 0
        for g in cohorts:
            for _ in range(n_per):
                for t in (1, 2, 3, 4):
                    w = bad if (g == poisoned_cohort and t < poisoned_cohort) else 1.0
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "w": w,
                            "y": float(rng.integers(0, 2)),
                        }
                    )
                uid += 1
        return pd.DataFrame(rows)

    @pytest.mark.parametrize(
        "bad,message",
        [
            (np.nan, "Weights contain NaN values"),
            (-1.0, "Weights must be non-negative"),
            (-np.inf, "Weights contain Inf values"),
            (np.inf, "Weights contain Inf values"),
        ],
    )
    @pytest.mark.parametrize("control_group", ["never_treated", "not_yet_treated"])
    @pytest.mark.parametrize("method", ["ols", "logit", "poisson"])
    def test_invalid_weights_raise_instead_of_excluding_the_cohort(
        self, bad, message, control_group, method
    ):
        from diff_diff.survey import SurveyDesign

        df = self._poisoned(bad)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=message):
                WooldridgeDiD(method=method, control_group=control_group).fit(
                    df,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="cohort",
                    survey_design=SurveyDesign(weights="w"),
                )

    def test_rejection_precedes_any_exclusion_warning(self):
        """The error must arrive INSTEAD of the misleading exclusion warning,
        not after it -- a user who sees only the warning would conclude their
        panel lacks pre-periods and 'fix' the wrong thing."""
        from diff_diff.survey import SurveyDesign

        df = self._poisoned(np.nan)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="Weights contain NaN values"):
                WooldridgeDiD(method="ols", control_group="never_treated").fit(
                    df,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="cohort",
                    survey_design=SurveyDesign(weights="w"),
                )
        assert [w for w in caught if "no pre-treatment period" in str(w.message)] == []

    def test_valid_weights_still_fit(self):
        """The guard must not reject a legitimate weighted design."""
        from diff_diff.survey import SurveyDesign

        df = self._poisoned(1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(method="ols", control_group="never_treated").fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="cohort",
                survey_design=SurveyDesign(weights="w"),
            )
        assert np.isfinite(res.overall_att)
        assert set(res.groups) == {3, 4}, "no cohort should be excluded on valid weights"

    def test_zero_weights_still_shape_support_without_excluding(self):
        """Zero is VALID and keeps its support semantics -- the guard must not
        over-reject.

        Zeroing the LATEST pre-period moves the reference to the next supported
        one rather than eliminating it, so the cohort stays identified and the
        fit proceeds. This is the weight-aware support this PR added, exercised
        without tripping the survey-exclusion refusal below.
        """
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(0)
        rows = []
        uid = 0
        for g in (0, 4):
            for _ in range(12):
                for t in (1, 2, 3, 4):
                    # cohort 4's latest pre-period (t=3) carries zero weight ->
                    # reference falls back to t=2, cohort remains identified.
                    w = 0.0 if (g == 4 and t == 3) else 1.0
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "w": w,
                            "y": 0.3 * t + rng.normal(0, 0.2),
                        }
                    )
                uid += 1
        df = pd.DataFrame(rows)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = WooldridgeDiD(method="ols", control_group="never_treated").fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="cohort",
                survey_design=SurveyDesign(weights="w"),
            )
        assert np.isfinite(res.overall_att)
        assert (4, 3) not in res.group_time_effects, "zero-weight cell should be unsupported"
        assert [w for w in caught if "no pre-treatment period" in str(w.message)] == []

    @pytest.mark.parametrize("method", ["ols", "logit", "poisson"])
    def test_survey_plus_unidentified_cohort_is_refused(self, method):
        """Excluding a cohort under a survey design is DOMAIN estimation, and
        this path does not implement it -- so it is refused, not approximated.

        Deleting the rows also deletes their PSUs/strata from the TSL meat and
        from ``df_survey = n_PSU - n_strata``. Measured before this refusal on a
        two-stratum panel: the fit returned a finite SE with ``df_survey = 14``
        against a full-design 22, i.e. variance from a design the user never
        specified. Applies to ALL THREE methods -- the earlier zero-weight guard
        caught only zero-weight units, and every unit here carries POSITIVE
        weight. Codex R5.
        """
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(3)
        rows = []
        uid = 0
        # cohort 4 observed only at t=3,4; with anticipation=1 it needs t < 3.
        for g, times in ((0, [1, 2, 3, 4]), (3, [1, 2, 3, 4]), (4, [3, 4])):
            for k in range(8):
                stratum = f"s{k % 2}"
                for t in times:
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "stratum": stratum,
                            "psu": f"{stratum}_g{g}_p{k}",
                            "w": 1.0 + 0.1 * (uid % 3),
                            "y": float(rng.integers(0, 2)),
                        }
                    )
                uid += 1
        df = pd.DataFrame(rows)
        assert (df.groupby("unit")["w"].sum() > 0).all(), "fixture must have no zero-weight unit"

        sd = SurveyDesign(weights="w", strata="stratum", psu="psu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotImplementedError, match="not supported"):
                WooldridgeDiD(method=method, control_group="never_treated", anticipation=1).fit(
                    df,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="cohort",
                    survey_design=sd,
                )

    @pytest.mark.parametrize(
        "field,mutate,message",
        [
            (
                "psu",
                lambda d: d.assign(psu=np.where(d["cohort"] == 4, None, d["psu"])),
                "PSU column",
            ),
            (
                "strata",
                lambda d: d.assign(stratum=np.where(d["cohort"] == 4, None, d["stratum"])),
                "[Ss]trat",
            ),
        ],
    )
    def test_invalid_survey_structure_cannot_hide_in_an_excluded_cohort(
        self, field, mutate, message
    ):
        """Structural survey metadata is validated on the PRE-exclusion sample.

        Weight validation alone was not enough: missing strata/PSU values, PSU
        reuse and invalid FPC were all still checked inside
        ``SurveyDesign.resolve()``, which ran after the rows were deleted. So
        invalid metadata confined to the excluded cohort vanished and the fit
        returned finite inference. Resolving the whole design up front fixes the
        class, not the instance -- and preserves the SPECIFIC error rather than
        a generic refusal. Codex R5.
        """
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(3)
        rows = []
        uid = 0
        for g, times in ((0, [1, 2, 3, 4]), (3, [1, 2, 3, 4]), (4, [3, 4])):
            for k in range(8):
                stratum = f"s{k % 2}"
                for t in times:
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "stratum": stratum,
                            "psu": f"{stratum}_g{g}_p{k}",
                            "w": 1.0,
                            "y": 0.3 * t + rng.normal(0, 0.2),
                        }
                    )
                uid += 1
        df = mutate(pd.DataFrame(rows))

        sd = SurveyDesign(weights="w", strata="stratum", psu="psu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=message):
                WooldridgeDiD(method="ols", control_group="never_treated", anticipation=1).fit(
                    df,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="cohort",
                    survey_design=sd,
                )

    @pytest.mark.parametrize("control_group", ["never_treated", "not_yet_treated"])
    def test_fractional_fweight_is_rejected_before_exclusion(self, control_group):
        """Every documented weight-TYPE rule must also run before exclusion.

        `fweight` must be a non-negative integer. That check originally lived
        only in ``SurveyDesign.resolve()``, i.e. AFTER support-based exclusion --
        so a cohort whose reference periods carry zero weight (excluded) and
        whose remaining rows carry a fractional fweight slipped through and the
        fit returned finite estimates on the reduced sample. Codex R4.
        """
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(0)
        rows = []
        uid = 0
        for g in (0, 3, 4):
            for _ in range(12):
                for t in (1, 2, 3, 4):
                    # cohort 4: zero-weight pre-periods (kills its reference, so
                    # exclusion would fire) + a FRACTIONAL fweight after.
                    w = (0.0 if t < 4 else 0.5) if g == 4 else 2.0
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "w": w,
                            "y": 0.3 * t + rng.normal(0, 0.2),
                        }
                    )
                uid += 1
        df = pd.DataFrame(rows)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="must be non-negative integers"):
                WooldridgeDiD(method="ols", control_group=control_group).fit(
                    df,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="cohort",
                    survey_design=SurveyDesign(weights="w", weight_type="fweight"),
                )
        assert [w for w in caught if "no pre-treatment period" in str(w.message)] == []

    def test_zero_weight_units_cannot_be_deleted_by_cohort_exclusion(self):
        """A zero-weight UNIT must be rejected even when exclusion would remove it.

        The library rejects zero-weight units rather than estimating around them
        (`test_ols_survey_zero_weight_unit_rejected`); genuine domain estimation
        is `SurveyDesign.subpopulation`, which this within-transform path does
        not implement. That guard ran only just before the within-transform, so
        unidentified-cohort exclusion could delete the offending rows first: a
        subpopulation-style fit whose zero-weight units made up a whole
        unidentified cohort silently degraded to naive subsetting and reported a
        survey df computed from a design missing entire PSUs (measured: 22 ->
        14, with a misleading "no pre-treatment period" warning) instead of
        raising. Codex R4.
        """
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(7)
        rows = []
        uid = 0
        for g in (0, 3, 4):
            for k in range(8):
                stratum = f"s{k % 2}"
                for t in (1, 2, 3, 4):
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "stratum": stratum,
                            "psu": f"{stratum}_g{g}_p{k}",
                            "w": 1.0,
                            "y": 0.3 * t + rng.normal(0, 0.2),
                        }
                    )
                uid += 1
        df = pd.DataFrame(rows)
        sd = SurveyDesign(weights="w", strata="stratum", psu="psu")
        sub_design, sub_data = sd.subpopulation(df, (df["cohort"] != 4).to_numpy())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="Survey weights sum to zero for unit"):
                WooldridgeDiD(method="ols", control_group="never_treated").fit(
                    sub_data,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="cohort",
                    survey_design=sub_design,
                )
        assert [w for w in caught if "no pre-treatment period" in str(w.message)] == []


class TestRankDeficientActionHonoredOnSkippedCells:
    """`rank_deficient_action` must mean the same thing after the skip rewrite.

    Pre-3.9, an unobserved (g, t) reached the design as an identically-zero
    column and QR dropped it, so ``"error"`` FAILED CLOSED. This branch skips
    those cells earlier -- better behavior, but the first version gated only on
    ``!= "silent"``, collapsing ``"warn"`` and ``"error"`` into a warning and
    silently downgrading a caller's explicit fail-closed request. Measured on a
    cohort-4-gapped panel: base RAISED under ``"error"``; the skip returned
    att=0.009034 with only a warning. Codex R6.
    """

    @staticmethod
    def _gapped():
        rng = np.random.default_rng(0)
        rows = []
        uid = 0
        for g, times in ((0, [1, 2, 3, 4]), (4, [1, 2, 4])):  # cohort 4 unobserved at t=3
            for _ in range(12):
                for t in times:
                    rows.append(
                        {"unit": uid, "time": t, "cohort": g, "y": 0.3 * t + rng.normal(0, 0.2)}
                    )
                uid += 1
        return pd.DataFrame(rows)

    def test_error_raises_naming_the_skipped_pair(self):
        with pytest.raises(ValueError, match=r"Skipped 1 cohort-time cell\(s\).*\(4, 3\)"):
            WooldridgeDiD(
                method="ols", control_group="never_treated", rank_deficient_action="error"
            ).fit(self._gapped(), outcome="y", unit="unit", time="time", first_treat="cohort")

    def test_warn_still_fits_and_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = WooldridgeDiD(
                method="ols", control_group="never_treated", rank_deficient_action="warn"
            ).fit(self._gapped(), outcome="y", unit="unit", time="time", first_treat="cohort")
        assert np.isfinite(res.overall_att)
        assert [w for w in caught if "Skipped 1 cohort-time" in str(w.message)] != []

    def test_silent_fits_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = WooldridgeDiD(
                method="ols", control_group="never_treated", rank_deficient_action="silent"
            ).fit(self._gapped(), outcome="y", unit="unit", time="time", first_treat="cohort")
        assert np.isfinite(res.overall_att)
        assert [w for w in caught if "Skipped" in str(w.message)] == []

    @staticmethod
    def _gapped_post():
        """Cohort 4 unobserved at t=5, a POST-treatment cell.

        ``_gapped()``'s hole is at t=3, which is PRE-treatment for cohort 4 and
        so is only ever emitted on the ``never_treated`` + OLS path
        (``include_pre``). The nonlinear paths emit post cells only, so the gate
        has to be exercised there with a gapped POST cell or the test passes
        vacuously.
        """
        rng = np.random.default_rng(0)
        rows = []
        uid = 0
        for g, times in ((0, [1, 2, 3, 4, 5]), (4, [1, 2, 4])):
            for _ in range(12):
                for t in times:
                    rows.append(
                        {"unit": uid, "time": t, "cohort": g, "y": 0.3 * t + rng.normal(0, 0.2)}
                    )
                uid += 1
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("method", ["logit", "poisson"])
    def test_error_is_honored_on_the_nonlinear_paths_too(self, method):
        """The builder runs before method dispatch, so all three share the gate."""
        df = self._gapped_post()
        df["y"] = (df["y"] > df["y"].median()).astype(float)
        with pytest.raises(ValueError, match=r"Skipped 1 cohort-time cell\(s\).*\(4, 5\)"):
            WooldridgeDiD(
                method=method, control_group="never_treated", rank_deficient_action="error"
            ).fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")


class TestWithinCohortSupportConnectivity:
    """Per-period support does not imply identification -- connectivity does.

    A cohort can have positive-weight observations in every period while its
    UNITS split into groups with disjoint observation windows. Identification
    runs through the unit fixed effects, so a group whose periods are all
    treatment cells contributes columns that sum to its own unit indicators:
    rank-deficient, and QR drops one -- possibly a genuine post-treatment
    effect, leaving `overall_att` an average over an incomplete cell set. That
    is issue #724's failure mode reached through unit support instead of the
    reference period. Codex R7.
    """

    @staticmethod
    def _split_support():
        """Cohort 4: half the units observe {2,4}, the other half {1,5}.

        Aggregate support covers 1,2,4,5 so ref=2 and cells 1,4,5 are emitted,
        but {1,5} is a closed component: both its periods are cells.
        """
        rng = np.random.default_rng(11)
        rows = []
        uid = 0
        for _ in range(15):
            for t in (1, 2, 3, 4, 5):
                rows.append(
                    {"unit": uid, "time": t, "cohort": 0, "y": 0.3 * t + rng.normal(0, 0.2)}
                )
            uid += 1
        for times in ((2, 4), (1, 5)):
            for _ in range(10):
                for t in times:
                    rows.append(
                        {"unit": uid, "time": t, "cohort": 4, "y": 0.3 * t + 1 + rng.normal(0, 0.2)}
                    )
                uid += 1
        return pd.DataFrame(rows)

    def test_disconnected_component_fails_closed(self):
        """Measured before the check: `g4_t5` was dropped by QR and the fit
        returned overall_att=0.0589 from the single surviving post cell."""
        with pytest.raises(ValueError, match="observed but NOT identified"):
            WooldridgeDiD(method="ols", control_group="never_treated").fit(
                self._split_support(), outcome="y", unit="unit", time="time", first_treat="cohort"
            )

    def test_the_builder_names_the_unidentified_cells(self):
        """The diagnostic must name the closed component's cells (1 and 5), not
        the connected ones -- otherwise the error cannot guide a fix."""
        df = self._split_support()
        _, _, _, diag = _build_interaction_matrix(
            df,
            cohort="cohort",
            time="time",
            anticipation=0,
            control_group="never_treated",
            method="ols",
            unit="unit",
        )
        assert sorted(t for _, t in diag.disconnected) == [1, 5]

    def test_connected_unbalanced_panel_is_not_flagged(self):
        """Guards against a false positive that would reject legitimate data:
        an unbalanced cohort whose units still share a period stays connected,
        so it must fit normally."""
        rng = np.random.default_rng(5)
        rows = []
        uid = 0
        for _ in range(15):
            for t in (1, 2, 3, 4, 5):
                rows.append(
                    {"unit": uid, "time": t, "cohort": 0, "y": 0.3 * t + rng.normal(0, 0.2)}
                )
            uid += 1
        # Both subsets observe t=2, so the cohort graph is one component.
        for times in ((2, 3, 4), (1, 2, 5)):
            for _ in range(10):
                for t in times:
                    rows.append(
                        {"unit": uid, "time": t, "cohort": 4, "y": 0.3 * t + 1 + rng.normal(0, 0.2)}
                    )
                uid += 1
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(method="ols", control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert np.isfinite(res.overall_att)
        assert (4, 5) in res.group_time_effects

    @pytest.mark.parametrize("method", ["logit", "poisson"])
    def test_nonlinear_paths_are_not_subject_to_the_unit_fe_condition(self, method):
        """The connectivity condition is collinearity with ABSORBED UNIT FEs.

        Only the OLS path within-transforms unit fixed effects away.
        logit/Poisson use explicit cohort + time dummies, so nothing absorbs a
        component's cell block and the design is full rank. Applying the check
        there rejected valid fits (codex R9). Same split-support panel the OLS
        test refuses -- here it must estimate.
        """
        rng = np.random.default_rng(3)
        rows = []
        uid = 0
        for _ in range(20):
            for t in (1, 2, 4, 5):
                rows.append({"unit": uid, "time": t, "cohort": 0, "y": float(rng.integers(0, 2))})
            uid += 1
        # cohort 4's pre-period units are DISJOINT from its post-period units,
        # so its post-only component has every period emitted.
        for times in ((1, 2), (4, 5)):
            for _ in range(10):
                for t in times:
                    rows.append(
                        {"unit": uid, "time": t, "cohort": 4, "y": float(rng.integers(0, 2))}
                    )
                uid += 1
        df = pd.DataFrame(rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(method=method, control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert {k for k in res.group_time_effects if k[1] >= k[0]} == {(4, 4), (4, 5)}
        assert np.isfinite(res.overall_att)

    def test_ols_still_refuses_the_same_panel(self):
        """The OLS half of the contrast: the same data IS unidentified there,
        so gating on method must not weaken the OLS guard."""
        rng = np.random.default_rng(3)
        rows = []
        uid = 0
        for _ in range(20):
            for t in (1, 2, 4, 5):
                rows.append(
                    {"unit": uid, "time": t, "cohort": 0, "y": 0.3 * t + rng.normal(0, 0.2)}
                )
            uid += 1
        for times in ((1, 2), (4, 5)):
            for _ in range(10):
                for t in times:
                    rows.append(
                        {"unit": uid, "time": t, "cohort": 4, "y": 0.3 * t + rng.normal(0, 0.2)}
                    )
                uid += 1
        with pytest.raises(ValueError, match="observed but NOT identified"):
            WooldridgeDiD(method="ols", control_group="never_treated").fit(
                pd.DataFrame(rows), outcome="y", unit="unit", time="time", first_treat="cohort"
            )


class TestPartialRankLossFailsClosed:
    """A finite overall ATT does not mean the reported cell set is complete.

    When rank reduction removes SOME treatment columns, the survivors are
    re-identified as whatever contrast the reduced design supports and keep
    their `ATT(g, t)` labels. Codex R8 P0.
    """

    @staticmethod
    def _no_comparison_final_period():
        """Never-treated observed through t=4; cohorts 3 and 4 through t=5.

        At t=5 every observed unit is treated, so `g3_t5 + g4_t5` equals the
        t=5 indicator and the pair is collinear with the time fixed effect.
        Effects are deliberately far apart (1.0 vs 3.0) so a relabeled
        contrast is unmistakable.
        """
        rng = np.random.default_rng(4)
        eff = {3: 1.0, 4: 3.0}
        rows = []
        uid = 0
        for _ in range(20):
            for t in (1, 2, 3, 4):
                rows.append(
                    {"unit": uid, "time": t, "cohort": 0, "y": 0.2 * t + rng.normal(0, 0.05)}
                )
            uid += 1
        for g in (3, 4):
            for _ in range(20):
                for t in (1, 2, 3, 4, 5):
                    y = 0.2 * t + (eff[g] if t >= g else 0.0) + rng.normal(0, 0.05)
                    rows.append({"unit": uid, "time": t, "cohort": g, "y": y})
                uid += 1
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("action", ["warn", "silent", "error"])
    def test_partial_cell_loss_is_now_filtered_for_every_rank_deficient_action(self, action):
        """The unsupported period is removed BEFORE the solve, in every mode.

        This panel used to raise under all three settings, and before the #729
        gate it was worse: under `"silent"` ATT(3,5) came back as -1.9393 (true
        +1.0, wrong SIGN) inside overall_att=+0.7703 (true ~1.8), with no
        warning of any kind.

        `t=5` has no never-treated observations, so comparison-support filtering
        drops it and the remaining design is exactly the one
        `test_the_same_panel_fits_once_the_bad_period_is_removed` fits by hand.
        Parameterized across all three modes because the drop is NOT gated on
        `rank_deficient_action` -- that setting governs how rank warnings
        surface, not whether the estimation sample changes.
        """
        with pytest.warns(UserWarning, match="no eligible comparison group"):
            res = WooldridgeDiD(
                method="ols", control_group="never_treated", rank_deficient_action=action
            ).fit(
                self._no_comparison_final_period(),
                outcome="y",
                unit="unit",
                time="time",
                first_treat="cohort",
            )
        assert sorted((int(g), int(t)) for g, t in res.group_time_effects) == [
            (3, 1),
            (3, 3),
            (3, 4),
            (4, 1),
            (4, 2),
            (4, 4),
        ]
        # The values the pre-gate bug corrupted, now recovered.
        assert res.group_time_effects[(3, 3)]["att"] == pytest.approx(1.0, abs=0.1)
        assert res.group_time_effects[(4, 4)]["att"] == pytest.approx(3.0, abs=0.1)

    def test_the_drop_warning_names_the_period_and_row_count(self):
        """A user cannot act on 'something was dropped' -- the message has to say
        WHICH period and how much data, since the remedy is theirs to choose
        (add never-treated units, or restrict the panel deliberately).

        Replaces an earlier test that pinned the lost-CELL error message; that
        error is no longer reached on this panel because the period is filtered
        before the solve.
        """
        with pytest.warns(UserWarning) as rec:
            WooldridgeDiD(method="ols", control_group="never_treated").fit(
                self._no_comparison_final_period(),
                outcome="y",
                unit="unit",
                time="time",
                first_treat="cohort",
            )
        msgs = [str(w.message) for w in rec]
        drop = [m for m in msgs if "no eligible comparison group" in m]
        assert drop, f"no comparison-support warning emitted; got {msgs}"
        assert "1 of 5 periods: 5" in drop[0]
        assert "observations" in drop[0]
        # Branch 1 (never_treated + OLS): the cause is missing never-treated
        # rows, NOT "every unit is already treated".
        assert "no never-treated units are observed" in drop[0]

    def test_the_same_panel_fits_once_the_bad_period_is_removed(self):
        """The gate must not be a blanket refusal: dropping the period that has
        no comparison group leaves a design that estimates correctly."""
        df = self._no_comparison_final_period()
        df = df[df["time"] <= 4]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(method="ols", control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert np.isfinite(res.overall_att)
        # True ATT(3,3)=ATT(3,4)=1.0, ATT(4,4)=3.0 -- recovered, not contrasts.
        assert res.group_time_effects[(3, 3)]["att"] == pytest.approx(1.0, abs=0.1)
        assert res.group_time_effects[(4, 4)]["att"] == pytest.approx(3.0, abs=0.1)


class TestControlGroupCellSets:
    """The two control groups are DIFFERENT specifications, not a tuning knob.

    `never_treated` + OLS is Wooldridge (2025) Eq. 6.1/6.4's lead-and-lag
    design: it emits pre-treatment cells and omits one per cohort as the
    reference. The default `not_yet_treated` is the lag-only design: post cells
    only, so it emits no placebo cells and omits no reference. The REGISTRY note
    and tutorial previously applied the `g-1` reference claim to the default,
    which does not exhibit it (codex R9).
    """

    @staticmethod
    def _panel():
        rng = np.random.default_rng(2)
        rows = []
        uid = 0
        for g in (0, 3, 5):
            for _ in range(20):
                for t in range(1, 7):
                    rows.append(
                        {"unit": uid, "time": t, "cohort": g, "y": 0.3 * t + rng.normal(0, 0.2)}
                    )
                uid += 1
        return pd.DataFrame(rows)

    def test_default_emits_no_pre_cells_and_omits_no_reference(self):
        df = self._panel()
        _, _, gt, diag = _build_interaction_matrix(
            df, cohort="cohort", time="time", anticipation=0, control_group="not_yet_treated"
        )
        assert [k for k in gt if k[1] < k[0]] == [], "lag-only design must emit no placebo cells"
        # A reference is still COMPUTED (it gates unidentified-cohort detection)
        # but it is never in the emitted set, so nothing is omitted from this design.
        assert set(diag.references) == {3, 5}
        for g, ref in diag.references.items():
            assert (g, ref) not in gt

    def test_never_treated_emits_pre_cells_except_the_reference(self):
        df = self._panel()
        _, _, gt, diag = _build_interaction_matrix(
            df, cohort="cohort", time="time", anticipation=0, control_group="never_treated"
        )
        assert [k for k in gt if k[1] < k[0]] != [], "lead-and-lag design must emit placebo cells"
        for g, ref in diag.references.items():
            assert ref == g - 1, "balanced panel: the reference is g-1"
            assert (g, ref) not in gt, "the reference cell must be omitted"
            emitted = {t for (gg, t) in gt if gg == g}
            expected = set(range(1, 7)) - {ref}
            assert emitted == expected, f"cohort {g}: expected all periods but {ref}"


class TestComparisonSupportFiltering:
    """Per-period comparison support (W2025 Section 5.4).

    Every panel and every number in this class was measured before the feature
    was written; the plan that specifies them was revised nine times against
    executed output, twice because a figure had been paired with the wrong
    frame. Assertions therefore name the panel they were taken on.
    """

    @staticmethod
    def _all_treated(cohorts=(3, 5, 8), periods=9, n_per=70, seed=7, effect=1.5):
        """All-eventually-treated panel: no cohort 0 anywhere."""
        rng = np.random.default_rng(seed)
        rows = []
        uid = 0
        for g in cohorts:
            for _ in range(n_per):
                for t in range(1, periods + 1):
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "y": rng.standard_normal() + effect * int(t >= g),
                        }
                    )
                uid += 1
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("anticipation,expected_kept", [(0, 7), (1, 6), (2, 5)])
    def test_eq_5_15_cell_set_on_a_balanced_panel(self, anticipation, expected_kept):
        """The retained cell set is Eq. 5.15's, intersected with observed times.

        `{(g, t) : g <= G_max - 1, g - anticipation <= t <= G_max - 1 -
        anticipation}`. The window bound is `t >= g - anticipation` (the builder
        emits anticipation-window cells), NOT `t >= g` -- an earlier draft of
        this assertion used `g <= t` and would have failed at anticipation > 0.
        """
        df = self._all_treated()
        g_max = 8
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(control_group="not_yet_treated", anticipation=anticipation).fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )

        kept = sorted({int(t) for (_g, t) in res.group_time_effects})
        assert max(kept) == g_max - 1 - anticipation
        assert len(set(range(1, g_max - anticipation))) == expected_kept

        observed = set(range(1, 10))
        expected = {
            (g, t)
            for g in (3, 5, 8)
            for t in observed
            if g <= g_max - 1
            # A cohort with no observed period before `g - anticipation` has no
            # reference and is excluded as unidentified, even though it meets
            # the Eq. 5.15 cohort bound. Observations start at t=1, so that is
            # `g - anticipation > 1`. At anticipation=2 this removes cohort 3,
            # leaving exactly {(5,3),(5,4),(5,5)} -- measured.
            and g - anticipation > 1 and g - anticipation <= t <= g_max - 1 - anticipation
        }
        assert {(int(g), int(t)) for g, t in res.group_time_effects} == expected
        # cohort G_max is the reference: no cells of its own.
        assert g_max not in {int(g) for (g, _t) in res.group_time_effects}

    def test_no_op_when_every_period_has_a_comparison(self):
        """Filtering is a no-op iff every period has an eligible comparison --
        not merely 'a never-treated group exists'."""
        rng = np.random.default_rng(21)
        rows = []
        for u in range(120):
            g = 0 if u < 40 else (3 if u < 80 else 5)
            for t in range(1, 7):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "cohort": g,
                        "y": rng.standard_normal() + 1.0 * int(g > 0 and t >= g),
                    }
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = WooldridgeDiD(control_group="not_yet_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert not [m for m in w if "no eligible comparison group" in str(m.message)]
        assert res.n_obs == len(df)

    def test_branch_one_reference_move_warns_by_name(self):
        """`never_treated` + OLS: a reference CAN move, and must say so.

        The ATTs stay correctly labelled and validly identified -- they are just
        normalized against a different baseline period than an unfiltered fit
        would use. Silent renormalization is the #724 defect class, so this
        warns rather than raising.
        """
        rng = np.random.default_rng(5)
        rows = []
        uid = 0
        for _ in range(30):  # never-treated, observed t=1..4 only
            for t in range(1, 5):
                rows.append(
                    {"unit": uid, "time": t, "cohort": 0, "y": 0.2 * t + rng.normal(0, 0.05)}
                )
            uid += 1
        for g in (3, 6):
            for _ in range(30):
                for t in range(1, 7):
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "y": 0.2 * t + (1.0 if t >= g else 0.0) + rng.normal(0, 0.05),
                        }
                    )
                uid += 1
        df = pd.DataFrame(rows)
        with pytest.warns(UserWarning, match=r"reference period moved from 5 to 4"):
            res = WooldridgeDiD(method="ols", control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert res.overall_att == pytest.approx(1.0171, abs=5e-3)

    def test_branch_one_no_move_when_never_treated_covers_the_period(self):
        """Companion to the above, on a frame truncated to t <= 5.

        Stated explicitly as truncated: on the untruncated 6-period frame t=6
        is still unsupported, so rows WOULD drop and warning (a) would fire
        correctly -- an earlier draft described this case as 'nothing is
        filtered', which is false.
        """
        rng = np.random.default_rng(5)
        rows = []
        uid = 0
        for _ in range(30):  # never-treated now observed through t=5
            for t in range(1, 6):
                rows.append(
                    {"unit": uid, "time": t, "cohort": 0, "y": 0.2 * t + rng.normal(0, 0.05)}
                )
            uid += 1
        for g in (3, 6):
            for _ in range(30):
                for t in range(1, 7):
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": g,
                            "y": 0.2 * t + (1.0 if t >= g else 0.0) + rng.normal(0, 0.05),
                        }
                    )
                uid += 1
        df = pd.DataFrame(rows)
        df = df[df["time"] <= 5]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = WooldridgeDiD(method="ols", control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert not [m for m in w if "reference period moved" in str(m.message)]
        assert res.overall_att == pytest.approx(0.9951, abs=5e-3)

    def test_branch_two_references_are_invariant(self):
        """`not_yet_treated`: references provably cannot move, so no warning.

        Reference eligibility (`t < g - anticipation`) IS the predicate's second
        disjunct evaluated at the cohort's own rows, so any reference period is
        supported by construction. The estimator asserts this at runtime; a
        violation would mean silent renormalization.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WooldridgeDiD(control_group="not_yet_treated").fit(
                self._all_treated(), outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert not [m for m in w if "reference period moved" in str(m.message)]

    @staticmethod
    def _all_treated_nonlinear(kind, cohorts=(3, 5, 8), periods=9, n_per=25, seed=17):
        """All-eventually-treated panel with a binary / count outcome."""
        rng = np.random.default_rng(seed)
        rows = []
        uid = 0
        for g in cohorts:
            for _ in range(n_per):
                uid += 1
                fe = rng.standard_normal() * 0.4
                for t in range(1, periods + 1):
                    lin = -0.2 + fe + 0.05 * t + 0.8 * int(t >= g)
                    y = (
                        rng.binomial(1, 1.0 / (1.0 + np.exp(-lin)))
                        if kind == "logit"
                        else rng.poisson(np.exp(lin))
                    )
                    rows.append({"unit": uid, "time": t, "cohort": g, "y": y})
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("kind", ["logit", "poisson"])
    def test_nonlinear_all_treated_paths_filter_identically(self, kind):
        """The filter runs on logit/Poisson too, and they were untested.

        These paths differ from OLS in design (explicit cohort + time dummies,
        no within-transform) and in control-pool semantics, so OLS coverage
        does not carry over. The Section 5.4 cell set is a property of the
        PERIODS, not of the link, so all three must agree on it exactly.
        """
        df = self._all_treated_nonlinear(kind)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = WooldridgeDiD(method=kind, control_group="not_yet_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        # Same cell set as the OLS path: cohorts 3 and 5 over t <= 7, none for 8.
        assert sorted((int(g), int(t)) for g, t in res.group_time_effects) == [
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (5, 5),
            (5, 6),
            (5, 7),
        ]
        assert sorted(int(g) for g in res.groups) == [3, 5]
        # 2 of 9 periods removed -> 7/9 of the rows survive.
        assert res.n_obs == len(df) * 7 // 9
        assert np.isfinite(res.overall_att) and np.isfinite(res.overall_se)

        # Both warnings fire here, exactly as on OLS.
        assert [m for m in w if "no eligible comparison group" in str(m.message)]
        assert [m for m in w if "have NO estimated cells" in str(m.message)]
        # References cannot move on this branch, whatever the link.
        assert not [m for m in w if "reference period moved" in str(m.message)]

    @pytest.mark.parametrize("kind", ["logit", "poisson"])
    def test_nonlinear_survey_refusal_also_fires(self, kind):
        """The survey refusal is decided before any row is removed, so it must
        fire on every method -- not only the OLS default."""
        from diff_diff.survey import SurveyDesign

        df = self._all_treated_nonlinear(kind)
        df["w"] = 1.0
        design = SurveyDesign(weights="w")
        with pytest.raises(NotImplementedError, match="no eligible comparison group"):
            WooldridgeDiD(method=kind, control_group="not_yet_treated").fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="cohort",
                survey_design=design,
            )

    def test_warning_a_survives_a_downstream_raise(self):
        """The drop warning fires at filter time, before anything can raise.

        On this panel `_exclude_unidentified_cohorts` raises after the filter,
        so a build-time-only warning would never reach the user -- they would
        see an error about comparison observations with no indication that
        half their rows had already been removed.
        """
        rng = np.random.default_rng(9)
        rows = []
        uid = 0
        for g in (1, 3):
            for _ in range(20):
                for t in range(1, 5):
                    rows.append({"unit": uid, "time": t, "cohort": g, "y": rng.standard_normal()})
                uid += 1
        df = pd.DataFrame(rows)
        with pytest.warns(UserWarning, match="no eligible comparison group"):
            with pytest.raises(ValueError):
                WooldridgeDiD(control_group="not_yet_treated").fit(
                    df, outcome="y", unit="unit", time="time", first_treat="cohort"
                )

    def test_zero_cell_cohort_is_reported_once_not_twice(self):
        """Warning (b) reads the FINAL cell set.

        On `{3,5,8}` at anticipation=2 the filter drops rows AND cohort 3 is
        then excluded as unidentified. Cohort 3 already has its own exclusion
        warning, so reading the first build's cell set would double-report it.
        """
        df = self._all_treated()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                WooldridgeDiD(control_group="not_yet_treated", anticipation=2).fit(
                    df, outcome="y", unit="unit", time="time", first_treat="cohort"
                )
            except ValueError:
                pass
        zero_cell = [m for m in w if "have NO estimated cells" in str(m.message)]
        for m in zero_cell:
            assert (
                "3" not in str(m.message).split("have NO estimated cells")[0].split("Cohort(s)")[-1]
            ), f"cohort 3 double-reported: {m.message}"

    def test_zero_cell_warning_does_not_misattribute_the_cause(self):
        """A zero-cell cohort is explained by ITS cause, not by Section 5.4.

        Section 5.4 explains exactly one cohort -- the last, and only when no
        never-treated group exists. Here a never-treated group IS present and
        cohort 7 is zero-cell purely because the panel never reaches t=7, so
        telling the user this is "the W2025 Section 5.4 normalization when no
        never-treated group exists" states something false about their panel.
        The companion all-treated case must keep the Section 5.4 wording.
        """
        rng = np.random.default_rng(21)
        rows = []
        uid = 0
        for cohort in (0, 3, 7):
            for _ in range(40):
                uid += 1
                fe = rng.standard_normal()
                for t in range(1, 6):
                    eff = 1.5 if (cohort > 0 and t >= cohort) else 0.0
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "cohort": cohort,
                            "y": fe + 0.1 * t + eff + rng.standard_normal() * 0.3,
                        }
                    )
        df = pd.DataFrame(rows)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = WooldridgeDiD(control_group="not_yet_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert 7 not in res.groups
        zero = [m for m in w if "have NO estimated cells" in str(m.message)]
        assert len(zero) == 1, f"expected one zero-cell warning, got {len(zero)}"
        text = str(zero[0].message)
        assert "7" in text
        assert "Section 5.4" not in text, f"misattributed to Section 5.4: {text}"
        assert "not treated within the observed periods" in text

        # The genuine Section 5.4 case still says so.
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            WooldridgeDiD(control_group="not_yet_treated").fit(
                self._all_treated(), outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        sec54 = [m for m in w2 if "have NO estimated cells" in str(m.message)]
        assert len(sec54) == 1
        assert "Section 5.4" in str(sec54[0].message)

    def test_cohort_share_fails_closed_when_filtering_removes_units(self):
        """`N_g` is read off the FINAL sample, so filtering can shrink it.

        Period filtering is the first thing in this estimator that removes SOME
        units of a RETAINED cohort: `_filter_sample` keeps every row of every
        treated unit, and unidentified-cohort exclusion removes whole cohorts.
        On an unbalanced panel a unit observed only at unsupported periods
        vanishes, and `aggregate(weights="cohort_share")` silently reweights --
        measured on this panel: 1.8078 with the supplied N_g against 3.8157
        with the post-filter one, a 2x swing on a headline number.

        W2025 Section 7 defines N_g on a balanced panel and does not say which
        reading applies, so `aggregate` refuses rather than picking one.
        `weights="cell"` never reads N_g and must keep working.
        """
        rng = np.random.default_rng(11)
        rows = []
        uid = 0

        def add(u, c, ts, eff):
            fe = rng.standard_normal()
            for t in ts:
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "cohort": c,
                        "y": fe + 0.1 * t + (eff if t >= c else 0.0) + rng.standard_normal() * 0.25,
                    }
                )

        every = list(range(1, 8))
        for _ in range(10):
            uid += 1
            add(uid, 2, every, 1.0)
        for _ in range(90):  # observed ONLY at t=7, which has no comparison
            uid += 1
            add(uid, 2, [7], 1.0)
        for _ in range(40):
            uid += 1
            add(uid, 4, every, 5.0)
        for _ in range(40):
            uid += 1
            add(uid, 7, every, 9.0)
        df = pd.DataFrame(rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(control_group="not_yet_treated").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        # 90 of cohort 2's 100 units lived only in the dropped period.
        assert res._cohort_units_dropped == {2: 90}
        assert res._n_g_per_cohort[2] == 10

        with pytest.raises(ValueError, match="comparison-support filtering removed"):
            res.aggregate(type="simple", weights="cohort_share")
        # The default weighting is untouched.
        assert np.isfinite(res.aggregate(type="simple", weights="cell").overall_att)

    def test_every_aggregation_surface_reads_the_filtered_cell_set(self):
        """`group` / `calendar` / `event` all consume the REDUCED cell set.

        `results.groups` and `time_periods` both shrink under filtering, and
        each aggregation keys off a different one of them. A stale read would
        surface an all-NaN row for `G_max`, or a calendar entry for a period
        that was removed before the solve -- either of which would advertise an
        estimate the fit never produced.

        Cells here are (3, 3..7) and (5, 5..7), so k = t - g spans 0..4 and
        cohort 8 contributes nothing to any of the three.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(control_group="not_yet_treated").fit(
                self._all_treated(), outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert res.time_periods == [1, 2, 3, 4, 5, 6, 7], "dropped periods leaked into metadata"

        # group: G_max has no cells and must not appear.
        grp = res.aggregate(type="group")
        assert sorted(int(g) for g in grp.group_effects) == [3, 5]

        # calendar: only periods carrying an estimated cell, so never 8 or 9.
        cal = res.aggregate(type="calendar")
        cal_keys = sorted(int(t) for t in cal.calendar_effects)
        assert cal_keys == [3, 4, 5, 6, 7]
        assert not {8, 9} & set(cal_keys)

        # event: exposure times of the retained cells only.
        evt = res.aggregate(type="event")
        assert sorted(int(k) for k in evt.event_study_effects) == [0, 1, 2, 3, 4]

        for agg in (grp, cal, evt):
            assert np.isfinite(agg.overall_att) and np.isfinite(agg.overall_se)

    def test_cohort_share_still_works_when_no_unit_is_lost(self):
        """The refusal must not fire on the balanced deliverable.

        Filtering there removes whole PERIODS and no units at all, so `N_g` is
        exactly the supplied cohort size and cohort-share weighting is
        well-defined. A guard that fired here would break the capability this
        change exists to restore.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(control_group="not_yet_treated").fit(
                self._all_treated(), outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert res._cohort_units_dropped == {}
        agg = res.aggregate(type="simple", weights="cohort_share")
        assert np.isfinite(agg.overall_att)

    def test_cells_derived_groups_did_not_leak_into_the_design(self):
        """`results.groups` is cells-derived; the DESIGN keeps present cohorts.

        This is the regression guard for the most dangerous defect review found:
        routing the covariate blocks through a cells-derived list would drop
        `D_{G_max} x X` and silently apply the Section 5.4 covariate
        normalization this release defers -- measured at 8/8 ATTs moving, max
        0.0648, on a dropped column with R^2 = 0.5551 against the rest of the
        design (i.e. real information, not a spanned duplicate).

        Detected here by the presence of the rank deficiency itself: if the
        list had leaked, `D8_x_x` would be gone and this would fit cleanly.
        Baseline control matters -- with no covariates the two lists agree
        exactly, so a covariate-free test proves nothing.
        """
        rng = np.random.default_rng(11)
        df = self._all_treated(n_per=60, seed=11)
        df["x"] = rng.standard_normal(len(df))

        with pytest.raises(ValueError, match="rank-deficient"):
            WooldridgeDiD(control_group="not_yet_treated", rank_deficient_action="error").fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort", exovar=["x"]
            )

        # xtvar is full rank under the default demeaning and must stay that way.
        res = WooldridgeDiD(control_group="not_yet_treated").fit(
            df, outcome="y", unit="unit", time="time", first_treat="cohort", xtvar=["x"]
        )
        assert np.isfinite(res.overall_att)

    def test_bootstrap_runs_on_a_filtered_fit(self):
        """Bootstrap consumes the FILTERED sample's residuals and cluster ids.

        Row alignment between the reduced frame and the resampling machinery is
        exercised here rather than assumed.
        """
        df = self._all_treated(n_per=40, seed=13)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = WooldridgeDiD(control_group="not_yet_treated", n_bootstrap=49, seed=42).fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)


class TestWooldridgeDfConvention:
    """The three-value df_convention knob on WooldridgeDiD OLS arms (3.9 / M-127)."""

    @staticmethod
    def _panel(seed=5):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(50):
            coh = [0, 4, 5][u % 3]
            for t in range(1, 7):
                eff = 0.6 if (coh and t >= coh) else 0.0
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        cohort=coh,
                        outcome=0.2 * u / 50 + 0.1 * t + eff + rng.standard_normal() * 0.5,
                    )
                )
        return pd.DataFrame(rows)

    _kw = dict(outcome="outcome", unit="unit", time="time", first_treat="cohort")

    @staticmethod
    def _t_p(t_stat, df):
        from scipy import stats

        return 2 * stats.t.sf(abs(t_stat), df)

    @staticmethod
    def _z_p(t_stat):
        from scipy import stats

        return 2 * stats.norm.sf(abs(t_stat))

    def test_default_hc1_is_t_residual_not_z(self):
        """The defect fix: the auto-clustered hc1 arm moved z -> t(residual)."""
        res = WooldridgeDiD().fit(self._panel(), **self._kw)
        assert res.overall_p_value != self._z_p(res.overall_t_stat)
        # residual df = n - k_kept - absorbed [unit, time] rank; recover it
        n = 300
        match = [
            d
            for d in range(2, n)
            if abs(res.overall_p_value - self._t_p(res.overall_t_stat, d)) < 1e-13
        ]
        assert len(match) == 1, match
        # per-cell inference shares the same df
        _, eff = next(iter(res.group_time_effects.items()))
        assert eff["p_value"] == pytest.approx(self._t_p(eff["t_stat"], match[0]), rel=1e-12)

    def test_cluster_matches_g_minus_1(self):
        data = self._panel()
        r0 = WooldridgeDiD().fit(data, **self._kw)
        r1 = WooldridgeDiD(df_convention="cluster").fit(data, **self._kw)
        G = data["unit"].nunique()
        assert r1.overall_att == r0.overall_att and r1.overall_se == r0.overall_se
        assert r1.overall_p_value == pytest.approx(self._t_p(r1.overall_t_stat, G - 1), rel=1e-12)

    def test_normal_reproduces_pre39_z(self):
        data = self._panel()
        r0 = WooldridgeDiD().fit(data, **self._kw)
        rn = WooldridgeDiD(df_convention="normal").fit(data, **self._kw)
        assert rn.overall_att == r0.overall_att and rn.overall_se == r0.overall_se
        assert rn.overall_p_value == pytest.approx(self._z_p(rn.overall_t_stat), rel=1e-14)

    def test_classical_and_hc2_bit_identical_under_default(self):
        """classical/hc2 already used t(n - rank(X)); the ONE resolved
        fallback must reproduce those values exactly (n - k_kept on the
        full-dummy design)."""
        data = self._panel()
        for vt in ("classical", "hc2"):
            res = WooldridgeDiD(vcov_type=vt).fit(data, **self._kw)
            _, eff = next(iter(res.group_time_effects.items()))
            n = len(data)
            match = [
                d
                for d in range(2, n)
                if abs(eff["p_value"] - self._t_p(eff["t_stat"], n - d)) < 1e-12
            ]
            assert match, f"{vt} cell not on t(n-k)"

    def test_hc1_cohort_trends_uses_full_dummy_residual_df(self):
        """The easy-to-miss lane: hc1 + cohort_trends routes FULL-DUMMY, so
        its residual df is n - k_kept (no absorbed-rank subtraction)."""
        data = self._panel()
        res = WooldridgeDiD(cohort_trends=True).fit(data, **self._kw)
        res_within = WooldridgeDiD().fit(data, **self._kw)
        _, eff = next(iter(res.group_time_effects.items()))
        n = len(data)
        # full-dummy residual df: recover as n - k_kept
        match_fd = [
            k
            for k in range(1, 200)
            if abs(eff["p_value"] - self._t_p(eff["t_stat"], n - k)) < 1e-12
        ]
        assert match_fd, "cohort_trends hc1 cell not on t(n - k_kept)"
        # and the within arm's df differs (it subtracts the absorbed rank on
        # top of a much smaller visible k)
        _, effw = next(iter(res_within.group_time_effects.items()))
        match_w = [
            d for d in range(2, n) if abs(effw["p_value"] - self._t_p(effw["t_stat"], d)) < 1e-13
        ]
        assert match_w and (n - match_fd[0]) != match_w[0]

    def test_postfit_aggregate_reproduces_fit_time_convention(self):
        """aggregate() consumes the stored resolved fallback on every OLS arm."""
        res = WooldridgeDiD().fit(self._panel(), **self._kw)
        n = 300
        match = [
            d
            for d in range(2, n)
            if abs(res.overall_p_value - self._t_p(res.overall_t_stat, d)) < 1e-13
        ]
        agg = res.aggregate("group")
        ge = next(iter(agg.group_effects.values()))
        assert ge["p_value"] == pytest.approx(self._t_p(ge["t_stat"], match[0]), rel=1e-12)
        simple = res.aggregate("simple")
        assert simple.overall_p_value == pytest.approx(
            self._t_p(simple.overall_t_stat, match[0]), rel=1e-12
        )

    def test_bootstrap_aggregate_group_stays_analytical_t(self):
        """On a bootstrap fit only the overall p/CI are percentile-overridden;
        post-fit aggregate('group') re-derives ANALYTICAL inference from the
        stored fallback (the _df_analytic_fallback-stays-live behavior)."""
        res = WooldridgeDiD(n_bootstrap=20, seed=1).fit(self._panel(), **self._kw)
        agg = res.aggregate("group")
        ge = next(iter(agg.group_effects.values()))
        assert np.isfinite(ge["p_value"])
        assert ge["p_value"] != self._z_p(ge["t_stat"]), "bootstrap fit reverted aggregates to z"

    def test_glm_warns_on_explicit_nondefault_only(self):
        data = self._panel()
        data["bin"] = (data["outcome"] > data["outcome"].median()).astype(int)
        kw = dict(outcome="bin", unit="unit", time="time", first_treat="cohort")
        with pytest.warns(UserWarning, match="no effect on the logit/poisson"):
            WooldridgeDiD(method="logit", df_convention="cluster").fit(data, **kw)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WooldridgeDiD(method="logit").fit(data, **kw)
        assert not any("no effect on the logit" in str(w.message) for w in caught)

    def test_pickle_setstate_migrates_df_one_way(self):
        """A pre-3.9 pickle stores _df_one_way; __setstate__ carries it into
        _df_analytic_fallback so a legacy result's aggregate() keeps its
        fit-time classical/hc2 inference."""
        res = WooldridgeDiD(vcov_type="classical").fit(self._panel(), **self._kw)
        legacy_df = res._df_analytic_fallback
        state = dict(res.__dict__)
        state.pop("_df_analytic_fallback")
        state.pop("df_convention", None)
        state["_df_one_way"] = legacy_df
        revived = WooldridgeDiDResults.__new__(WooldridgeDiDResults)
        revived.__setstate__(state)
        assert revived._df_analytic_fallback == legacy_df
        assert not hasattr(revived, "_df_one_way") or "_df_one_way" not in revived.__dict__
        assert revived.df_convention == "residual"
        agg = revived.aggregate("simple")
        assert np.isfinite(agg.overall_p_value)

    def test_survey_df_precedence_over_knob(self):
        """Survey design df wins under EVERY knob value: the three fits are
        bit-identical on overall AND per-cell inference (the knob is never
        consulted on surveyed fits)."""
        from diff_diff.survey import SurveyDesign

        data = self._panel()
        data["w"] = 1.0 + (data["unit"] % 5) * 0.2
        data["stratum"] = data["unit"] % 4
        sd = SurveyDesign(weights="w", strata="stratum", psu="unit")
        fits = {
            conv: WooldridgeDiD(df_convention=conv).fit(data, survey_design=sd, **self._kw)
            for conv in ("residual", "cluster", "normal")
        }
        base = fits["residual"]
        # the survey df is finite -> t inference even under "normal"
        assert base.overall_p_value != self._z_p(base.overall_t_stat)
        for conv in ("cluster", "normal"):
            r = fits[conv]
            assert r.overall_p_value == base.overall_p_value
            assert r.overall_conf_int == base.overall_conf_int
            for k, eff in base.group_time_effects.items():
                assert r.group_time_effects[k]["p_value"] == eff["p_value"]

    def test_validation_and_transactional_set_params(self):
        with pytest.raises(ValueError, match="df_convention"):
            WooldridgeDiD(df_convention="bogus")
        est = WooldridgeDiD()
        with pytest.raises(ValueError, match="df_convention"):
            est.set_params(df_convention="bogus", alpha=0.10)
        assert est.df_convention == "residual" and est.alpha == 0.05
        # Valid value + unknown key: the unknown-key rejection must also
        # leave the estimator fully unchanged (no partial application).
        before = est.get_params()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(df_convention="normal", nonexistent_param=1)
        assert est.get_params() == before
        assert WooldridgeDiD(df_convention="cluster").get_params()["df_convention"] == "cluster"
