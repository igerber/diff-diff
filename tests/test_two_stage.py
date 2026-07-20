"""
Tests for Gardner (2022) Two-Stage DiD estimator.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.two_stage import (
    TwoStageBootstrapResults,
    TwoStageDiD,
    TwoStageDiDResults,
    two_stage_did,
)

# =============================================================================
# Shared test data generation
# =============================================================================


def generate_test_data(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_effect: float = 2.0,
    never_treated_frac: float = 0.3,
    dynamic_effects: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic staggered adoption data for testing."""
    rng = np.random.default_rng(seed)

    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    n_never = int(n_units * never_treated_frac)
    n_treated = n_units - n_never

    cohort_periods = np.array([3, 5, 7])
    first_treat = np.zeros(n_units, dtype=int)
    if n_treated > 0:
        cohort_assignments = rng.choice(len(cohort_periods), size=n_treated)
        first_treat[n_never:] = cohort_periods[cohort_assignments]

    first_treat_expanded = np.repeat(first_treat, n_periods)

    unit_fe = rng.standard_normal(n_units) * 2.0
    time_fe = np.linspace(0, 1, n_periods)

    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
    relative_time = times - first_treat_expanded

    if dynamic_effects:
        dynamic_mult = 1 + 0.1 * np.maximum(relative_time, 0)
    else:
        dynamic_mult = np.ones_like(relative_time, dtype=float)

    effect = treatment_effect * dynamic_mult

    outcomes = (
        unit_fe_expanded + time_fe_expanded + effect * post + rng.standard_normal(len(units)) * 0.5
    )

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcomes,
            "first_treat": first_treat_expanded,
        }
    )


# =============================================================================
# TestTwoStageDiDBasic
# =============================================================================


class TestTwoStageDiDBasic:
    """Tests for basic TwoStageDiD functionality."""

    def test_basic_fit(self):
        """Test basic model fitting."""
        data = generate_test_data()

        est = TwoStageDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert est.is_fitted_
        assert isinstance(results, TwoStageDiDResults)

    def test_att_accuracy(self):
        """Test that ATT recovers true treatment effect."""
        data = generate_test_data(treatment_effect=2.0, dynamic_effects=False, seed=123)

        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Should recover ~2.0 with reasonable tolerance
        assert abs(results.overall_att - 2.0) < 0.3

    def test_se_positive_finite(self):
        """Test that SEs are positive and finite."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.overall_se > 0
        assert np.isfinite(results.overall_se)

    def test_ci_contains_point_estimate(self):
        """Test that confidence interval contains the point estimate."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.overall_conf_int[0] <= results.overall_att
        assert results.overall_att <= results.overall_conf_int[1]

    def test_t_stat_and_p_value(self):
        """Test that t-stat and p-value are consistent."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert np.isfinite(results.overall_t_stat)
        assert 0 <= results.overall_p_value <= 1

        # t-stat should equal ATT / SE
        expected_t = results.overall_att / results.overall_se
        assert abs(results.overall_t_stat - expected_t) < 1e-10

    def test_event_study(self):
        """Test event study specification."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0

        # Check reference period is present
        ref_period = -1
        assert ref_period in results.event_study_effects
        assert results.event_study_effects[ref_period]["effect"] == 0.0

        # Post-treatment effects should be positive (treatment_effect=2.0)
        post_effects = {h: e for h, e in results.event_study_effects.items() if h >= 0}
        assert len(post_effects) > 0
        for h, eff in post_effects.items():
            assert eff["effect"] > 0, f"Post-treatment effect at h={h} should be positive"
            assert eff["se"] > 0, f"SE at h={h} should be positive"
            assert np.isfinite(eff["t_stat"])
            assert 0 <= eff["p_value"] <= 1

    def test_group_effects(self):
        """Test group (cohort) effects."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        assert results.group_effects is not None
        # Should have 3 groups (cohorts 3, 5, 7)
        assert len(results.group_effects) == 3
        for g, eff in results.group_effects.items():
            assert eff["effect"] > 0
            assert eff["se"] > 0
            assert np.isfinite(eff["t_stat"])

    def test_all_aggregation(self):
        """Test aggregate='all' produces both event study and group effects."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )

        assert results.event_study_effects is not None
        assert results.group_effects is not None

    def test_summary_text(self):
        """Test that summary produces expected header text."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        text = results.summary()
        assert "Two-Stage DiD Estimator Results (Gardner 2022)" in text
        assert "ATT" in text
        assert "Overall Average Treatment Effect" in text

    def test_to_dataframe_event_study(self):
        """Test to_dataframe with event_study level."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        df = results.to_dataframe("event_study")
        assert isinstance(df, pd.DataFrame)
        assert "relative_period" in df.columns
        assert "effect" in df.columns
        assert "se" in df.columns
        assert len(df) > 0

    def test_to_dataframe_group(self):
        """Test to_dataframe with group level."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        df = results.to_dataframe("group")
        assert isinstance(df, pd.DataFrame)
        assert "group" in df.columns
        assert len(df) == 3

    def test_to_dataframe_observation(self):
        """Test to_dataframe with observation level."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        df = results.to_dataframe("observation")
        assert isinstance(df, pd.DataFrame)
        assert "tau_hat" in df.columns
        assert "weight" in df.columns
        assert "unit" in df.columns
        assert "time" in df.columns
        assert len(df) == results.n_treated_obs

    def test_to_dataframe_invalid_level(self):
        """Test to_dataframe with invalid level raises."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        with pytest.raises(ValueError, match="Unknown level"):
            results.to_dataframe("invalid")

    def test_to_dataframe_no_event_study(self):
        """Test to_dataframe raises when event study not computed."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        with pytest.raises(ValueError, match="Event study effects not computed"):
            results.to_dataframe("event_study")

    def test_repr(self):
        """Test __repr__ contains expected elements."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        repr_str = repr(results)
        assert "TwoStageDiDResults" in repr_str
        assert "ATT=" in repr_str
        assert "SE=" in repr_str

    def test_is_significant_property(self):
        """Test is_significant property."""
        data = generate_test_data(treatment_effect=2.0)
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert isinstance(results.is_significant, bool)
        # Strong treatment effect should be significant
        assert results.is_significant

    def test_significance_stars_property(self):
        """Test significance_stars property."""
        data = generate_test_data(treatment_effect=2.0)
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        stars = results.significance_stars
        assert isinstance(stars, str)
        # Strong effect should have at least one star
        assert len(stars.strip()) > 0

    def test_metadata_fields(self):
        """Test that metadata fields are populated correctly."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.n_obs == len(data)
        assert results.n_treated_obs > 0
        assert results.n_untreated_obs > 0
        assert results.n_treated_obs + results.n_untreated_obs == results.n_obs
        assert results.n_treated_units > 0
        assert results.n_control_units > 0
        assert len(results.groups) == 3
        assert len(results.time_periods) == 10


# =============================================================================
# TestTwoStageDiDEquivalence
# =============================================================================


class TestTwoStageDiDEquivalence:
    """Test that TwoStageDiD point estimates match ImputationDiD."""

    def test_overall_att_matches_imputation(self):
        """Overall ATT should match ImputationDiD to machine precision."""
        from diff_diff.imputation import ImputationDiD

        data = generate_test_data()

        ts_results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        imp_results = ImputationDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert abs(ts_results.overall_att - imp_results.overall_att) < 1e-10

    def test_event_study_effects_match_imputation(self):
        """Event study point estimates should match ImputationDiD."""
        from diff_diff.imputation import ImputationDiD

        data = generate_test_data()

        ts_results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        imp_results = ImputationDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Both should have the same horizons
        ts_horizons = set(ts_results.event_study_effects.keys())
        imp_horizons = set(imp_results.event_study_effects.keys())
        assert ts_horizons == imp_horizons

        # Point estimates should match
        for h in ts_horizons:
            ts_eff = ts_results.event_study_effects[h]["effect"]
            imp_eff = imp_results.event_study_effects[h]["effect"]
            if np.isfinite(ts_eff) and np.isfinite(imp_eff):
                assert (
                    abs(ts_eff - imp_eff) < 1e-8
                ), f"Effect mismatch at h={h}: TS={ts_eff:.10f}, Imp={imp_eff:.10f}"

    def test_group_effects_match_imputation(self):
        """Group point estimates should match ImputationDiD."""
        from diff_diff.imputation import ImputationDiD

        data = generate_test_data()

        ts_results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )
        imp_results = ImputationDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        assert set(ts_results.group_effects.keys()) == set(imp_results.group_effects.keys())

        for g in ts_results.group_effects:
            ts_eff = ts_results.group_effects[g]["effect"]
            imp_eff = imp_results.group_effects[g]["effect"]
            assert abs(ts_eff - imp_eff) < 1e-8

    def test_ses_differ_from_imputation(self):
        """GMM SEs should differ from conservative (Theorem 3) SEs."""
        from diff_diff.imputation import ImputationDiD

        data = generate_test_data()

        ts_results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        imp_results = ImputationDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # SEs should differ (different variance estimators)
        assert abs(ts_results.overall_se - imp_results.overall_se) > 1e-6


# =============================================================================
# TestTwoStageDiDVariance
# =============================================================================


class TestTwoStageDiDVariance:
    """Tests for GMM sandwich variance estimator."""

    def test_gmm_se_differs_from_naive(self):
        """GMM SE should differ from naive Stage 2 OLS SE."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # The GMM SE accounts for first-stage estimation uncertainty
        assert results.overall_se > 0
        assert np.isfinite(results.overall_se)

    def test_event_study_se_positive(self):
        """Event study SEs should all be positive."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        for h, eff in results.event_study_effects.items():
            if eff.get("n_obs", 0) > 0:
                assert eff["se"] > 0, f"SE at h={h} should be positive"
                assert np.isfinite(eff["se"])

    def test_sparse_factorized_dense_fallback_emits_warning(self):
        """Silent-failure audit axis C: when sparse factorization of Stage 1's
        normal-equations matrix fails and the GMM sandwich falls back to sparse
        LSMR, a UserWarning must surface so callers know SE came from the
        degraded path rather than the fast sparse path.

        Also verifies the LSMR fallback still yields finite, usable SEs so
        that a future regression in the fallback control flow cannot keep the
        warning while breaking the degraded path."""
        import unittest.mock

        data = generate_test_data()

        with unittest.mock.patch(
            "diff_diff.two_stage.sparse_factorized",
            side_effect=RuntimeError("test failure"),
        ):
            with pytest.warns(
                UserWarning, match="sparse factorization.*falling back to sparse LSMR"
            ):
                results = TwoStageDiD().fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                )

        # LSMR fallback must still produce a usable SE.
        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0

    def test_lsmr_fallback_matches_dense_lstsq_oracle(self):
        """Consumer-level parity on a genuinely singular Stage-1 Gram: any
        least-squares solutions differ only by null(X'X) = null(X_10)
        components, and every gamma_hat consumer is an X_10-range
        functional (Psi = X_10 @ gamma; score correction c_g'gamma with
        c_g in rowspace(X_10)) — so the LSMR fallback and a dense-lstsq
        oracle must agree on X_10 @ solution even where the raw
        coefficient vectors differ. (theta_exact's treated-row consumer is
        covered separately by min-norm agreement; see the forced-fallback
        fit-level test.)"""
        import scipy.sparse as sp

        from diff_diff.two_stage import _lsmr_certified_normal_solve

        rng = np.random.default_rng(3)
        n, p = 120, 8
        X = rng.normal(size=(n, p))
        X[:, p - 1] = X[:, 0] + X[:, 1]  # rank-deficient by construction
        gram = sp.csc_matrix(X.T @ X)
        rhs = X.T @ rng.normal(size=(n, 3))  # multi-RHS, in rowspace(X)
        z_lsmr = _lsmr_certified_normal_solve(gram, rhs)
        z_dense = np.linalg.lstsq(gram.toarray(), rhs, rcond=None)[0]
        # consumer functional X @ z is invariant across LS solutions
        np.testing.assert_allclose(X @ z_lsmr, X @ z_dense, atol=1e-8)
        # 1-d rhs round-trips with the right shape
        z1 = _lsmr_certified_normal_solve(gram, rhs[:, 0])
        np.testing.assert_allclose(X @ z1.ravel(), X @ z_dense[:, 0], atol=1e-8)

    def test_lsmr_fallback_never_densifies(self):
        """The whole point of the swap: the fallback path must not call
        .toarray() on the Stage-1 Gram (the O((U+T+K)^2) OOM risk the
        TODO row tracked)."""
        import unittest.mock

        data = generate_test_data()

        def _no_lstsq(*a, **k):
            raise AssertionError(
                "np.linalg.lstsq reached — the Stage-1 fallback must solve "
                "via sparse LSMR, never a dense lstsq on the Gram"
            )

        with unittest.mock.patch(
            "diff_diff.two_stage.sparse_factorized",
            side_effect=RuntimeError("test failure"),
        ):
            with unittest.mock.patch("numpy.linalg.lstsq", _no_lstsq):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = TwoStageDiD().fit(
                        data,
                        outcome="outcome",
                        unit="unit",
                        time="time",
                        first_treat="first_treat",
                    )
        assert np.isfinite(results.overall_se)

    def test_uncertified_lsmr_fails_closed_to_nan(self):
        """A finite-but-uncertified LSMR result (istop outside {0,1,2,4,5}
        on both attempts) must NOT feed the GMM sandwich: the solve raises
        _LSMRUnconvergedError and the variance boundary reports NaN
        inference."""
        import unittest.mock

        data = generate_test_data()

        def _fake_lsmr(A, b, **kwargs):
            x = np.zeros(A.shape[1])
            return (x, 7, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        with unittest.mock.patch(
            "diff_diff.two_stage.sparse_factorized",
            side_effect=RuntimeError("test failure"),
        ):
            with unittest.mock.patch("scipy.sparse.linalg.lsmr", _fake_lsmr):
                with pytest.warns(UserWarning, match="did not converge"):
                    results = TwoStageDiD().fit(
                        data,
                        outcome="outcome",
                        unit="unit",
                        time="time",
                        first_treat="first_treat",
                    )
        assert np.isnan(results.overall_se)
        assert np.isnan(results.overall_p_value)

    def test_weighted_survey_lsmr_fallback_no_densify(self):
        """The weighted/survey GMM-sandwich site (X'_10 W X_10) takes the
        same LSMR fallback: finite SEs, and no Gram densification."""
        import unittest.mock

        from diff_diff import SurveyDesign

        data = generate_test_data()
        rng = np.random.default_rng(5)
        unit_w = {
            u: w
            for u, w in zip(
                data["unit"].unique(), rng.uniform(0.5, 2.0, size=data["unit"].nunique())
            )
        }
        data["w"] = data["unit"].map(unit_w)

        def _no_lstsq(*a, **k):
            raise AssertionError(
                "np.linalg.lstsq reached — the Stage-1 fallback must solve "
                "via sparse LSMR, never a dense lstsq on the Gram"
            )

        with unittest.mock.patch(
            "diff_diff.two_stage.sparse_factorized",
            side_effect=RuntimeError("test failure"),
        ):
            with unittest.mock.patch("numpy.linalg.lstsq", _no_lstsq):
                with pytest.warns(UserWarning, match="falling back to sparse LSMR"):
                    res = TwoStageDiD().fit(
                        data,
                        outcome="outcome",
                        unit="unit",
                        time="time",
                        first_treat="first_treat",
                        survey_design=SurveyDesign(weights="w"),
                    )
        assert np.isfinite(res.overall_se)

    def test_forced_fallback_fit_level_lsmr_vs_dense_oracle(self):
        """Fit-level parity on the FORCED fallback path (the fixture itself
        is full-rank; sparse factorization is mocked to fail): LSMR vs a
        dense-lstsq oracle must agree on overall ATT and SE. Covers the one
        consumer the range-functional argument does not (the bootstrap
        exact-residual helper's X_1 @ theta on treated rows) via min-norm
        agreement — both solvers return the min-norm least-squares solution.
        This test does NOT construct a singular fit-level design; the
        genuinely singular Gram is covered by the helper-level oracle test
        above, and the singular fit-level path by the methodology suite's
        singular-Omega_0 regression."""
        import unittest.mock

        import diff_diff.two_stage as ts

        data = generate_test_data()

        def _fit():
            return TwoStageDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

        with unittest.mock.patch(
            "diff_diff.two_stage.sparse_factorized",
            side_effect=RuntimeError("test failure"),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_lsmr = _fit()

            def _dense_oracle(gram_csc, rhs):
                out = np.linalg.lstsq(
                    gram_csc.toarray(), np.asarray(rhs, dtype=np.float64), rcond=None
                )[0]
                return out.reshape(gram_csc.shape[0], -1)

            with unittest.mock.patch.object(ts, "_lsmr_certified_normal_solve", _dense_oracle):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res_dense = _fit()

        np.testing.assert_allclose(res_lsmr.overall_att, res_dense.overall_att, rtol=1e-8)
        np.testing.assert_allclose(res_lsmr.overall_se, res_dense.overall_se, rtol=1e-6)

    def test_sparse_factorized_bootstrap_dense_fallback_emits_warning(self):
        """Silent-failure audit axis C: the TwoStage bootstrap path has the
        same sparse->LSMR fallback and must also emit a UserWarning.

        Also verifies the bootstrap LSMR fallback still yields finite,
        usable SEs."""
        import unittest.mock

        data = generate_test_data()

        with unittest.mock.patch(
            "diff_diff.two_stage_bootstrap.sparse_factorized",
            side_effect=RuntimeError("test failure"),
        ):
            with pytest.warns(
                UserWarning, match="sparse factorization.*falling back to sparse LSMR"
            ):
                results = TwoStageDiD(n_bootstrap=4, seed=42).fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                )

        # Bootstrap LSMR fallback must still produce a usable SE.
        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0


# =============================================================================
# TestTwoStageDiDEdgeCases
# =============================================================================


class TestTwoStageDiDEdgeCases:
    """Tests for edge cases and error handling."""

    def test_always_treated_excluded_with_warning(self):
        """Always-treated units should be excluded with a warning."""
        data = generate_test_data()

        # Add an always-treated unit (first_treat = 0 means treated at time 0)
        always_treated = pd.DataFrame(
            {
                "unit": np.repeat(999, 10),
                "time": np.arange(10),
                "outcome": np.random.default_rng(42).standard_normal(10),
                "first_treat": np.repeat(-1, 10),  # treated before sample starts
            }
        )
        data_with_always = pd.concat([data, always_treated], ignore_index=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = TwoStageDiD().fit(
                data_with_always,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
            always_treated_warns = [
                x for x in w if "treated in all observed periods" in str(x.message)
            ]
            assert len(always_treated_warns) > 0

        # Verify unit was excluded (total obs should be less)
        assert results.n_obs == len(data)

    def test_no_never_treated_works(self):
        """Estimation should work without never-treated units."""
        data = generate_test_data(never_treated_frac=0.0)

        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.overall_att > 0
        assert results.overall_se > 0

    def test_single_cohort(self):
        """Should work with a single treatment cohort."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 50, 8
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        first_treat = np.zeros(n_units, dtype=int)
        first_treat[15:] = 4  # single cohort at period 4

        ft_exp = np.repeat(first_treat, n_periods)
        post = (times >= ft_exp) & (ft_exp > 0)
        outcomes = (
            rng.standard_normal(n_units)[np.repeat(np.arange(n_units), n_periods)]
            + 2.0 * post
            + rng.standard_normal(len(units)) * 0.5
        )

        data = pd.DataFrame(
            {"unit": units, "time": times, "outcome": outcomes, "first_treat": ft_exp}
        )

        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert abs(results.overall_att - 2.0) < 0.5
        assert len(results.groups) == 1

    def test_anticipation_shifts_timing(self):
        """Anticipation parameter should shift effective treatment timing."""
        data = generate_test_data(seed=123)

        results_no_ant = TwoStageDiD(anticipation=0).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        results_with_ant = TwoStageDiD(anticipation=1).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # With anticipation, more obs are treated -> different ATT
        assert results_with_ant.n_treated_obs > results_no_ant.n_treated_obs
        assert abs(results_no_ant.overall_att - results_with_ant.overall_att) > 0.01

    def test_rank_deficiency_warning(self):
        """Rank deficiency should emit warning in 'warn' mode."""
        # Create data where some treated units have no untreated periods
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # All units treated at period 0 (except never-treated)
        first_treat = np.zeros(n_units, dtype=int)
        first_treat[5:] = 0  # never treated (first_treat=0)
        first_treat[:5] = 1  # treated at period 1

        ft_exp = np.repeat(first_treat, n_periods)
        outcomes = rng.standard_normal(len(units))

        data = pd.DataFrame(
            {"unit": units, "time": times, "outcome": outcomes, "first_treat": ft_exp}
        )

        # Should work without error in warn mode
        results = TwoStageDiD(rank_deficient_action="warn").fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert isinstance(results, TwoStageDiDResults)

    def test_rank_deficiency_error(self):
        """Rank deficiency should raise in 'error' mode when violated."""
        # Create data where a treated unit has NO untreated periods at all
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # Some units treated at period 0 (no pre-treatment)
        first_treat = np.zeros(n_units, dtype=int)
        first_treat[10:] = 0  # first_treat at the first time period
        first_treat[:5] = 0  # never treated
        first_treat[5:10] = 0  # Make all units at period 0 as treated
        # Actually let's have some treated at period 0 so they fail rank check
        first_treat[5:10] = 0  # All these are coded as never-treated (first_treat=0)

        ft_exp = np.repeat(first_treat, n_periods)
        outcomes = rng.standard_normal(len(units))
        data = pd.DataFrame(
            {"unit": units, "time": times, "outcome": outcomes, "first_treat": ft_exp}
        )

        # All units are never-treated, so no treated obs -> ValueError
        with pytest.raises(ValueError, match="No treated observations"):
            TwoStageDiD(rank_deficient_action="error").fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

    def test_nan_propagation(self):
        """NaN SE should propagate to t_stat, p_value, conf_int."""
        from tests.conftest import assert_nan_inference

        # Use never_treated_frac=0.0 to trigger Proposition 5 NaN horizons
        data = generate_test_data(never_treated_frac=0.0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = TwoStageDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )

        # Proposition 5 horizons should have NaN inference fields
        assert results.event_study_effects, "Event study should be computed"
        nan_horizons_found = 0
        for h, eff in results.event_study_effects.items():
            if np.isnan(eff["effect"]):
                nan_horizons_found += 1
                assert_nan_inference(
                    {
                        "se": eff["se"],
                        "t_stat": eff["t_stat"],
                        "p_value": eff["p_value"],
                        "conf_int": eff["conf_int"],
                    }
                )
        assert nan_horizons_found > 0, "Should have at least one Prop 5 NaN horizon"

        # Normal results should have finite values
        assert np.isfinite(results.overall_t_stat)
        assert np.isfinite(results.overall_p_value)

    def test_covariates(self):
        """Estimation with covariates should work."""
        data = generate_test_data()
        rng = np.random.default_rng(99)
        data["x1"] = rng.standard_normal(len(data))
        data["x2"] = rng.standard_normal(len(data))

        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
        )

        assert results.overall_att > 0
        assert results.overall_se > 0
        assert np.isfinite(results.overall_se)

    def test_missing_column_error(self):
        """Missing required columns should raise ValueError."""
        data = generate_test_data()

        with pytest.raises(ValueError, match="Missing columns"):
            TwoStageDiD().fit(
                data,
                outcome="nonexistent",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_no_treated_obs_error(self):
        """Should raise when no treated observations exist."""
        rng = np.random.default_rng(42)
        n = 100
        data = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(10), 10),
                "time": np.tile(np.arange(10), 10),
                "outcome": rng.standard_normal(n),
                "first_treat": 0,  # all never-treated
            }
        )

        with pytest.raises(ValueError, match="No treated"):
            TwoStageDiD().fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

    def test_horizon_max(self):
        """horizon_max should limit event study horizons."""
        data = generate_test_data()
        results = TwoStageDiD(horizon_max=2).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # All horizons should have |h| <= 2
        for h in results.event_study_effects:
            if results.event_study_effects[h].get("n_obs", 0) > 0:
                assert abs(h) <= 2

    def test_always_treated_warning_lists_unit_ids(self):
        """Always-treated warning should include affected unit IDs."""
        data = generate_test_data()

        # Add two always-treated units (first_treat before min_time=0)
        always_treated = pd.DataFrame(
            {
                "unit": np.repeat([997, 998], 10),
                "time": np.tile(np.arange(10), 2),
                "outcome": np.random.default_rng(42).standard_normal(20),
                "first_treat": np.repeat([-1, -2], 10),
            }
        )
        data_with_always = pd.concat([data, always_treated], ignore_index=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TwoStageDiD().fit(
                data_with_always,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
            always_warns = [x for x in w if "treated in all observed periods" in str(x.message)]
            assert len(always_warns) == 1
            msg = str(always_warns[0].message)
            assert "997" in msg
            assert "998" in msg

    def test_bootstrap_with_nan_y_tilde(self, ci_params):
        """Bootstrap should handle NaN y_tilde from unidentified FEs."""
        # No never-treated units: cohorts 3, 5, 7 on periods 0-9 means
        # periods 7-9 have zero untreated obs -> NaN y_tilde
        data = generate_test_data(never_treated_frac=0.0)
        n_boot = ci_params.bootstrap(20)

        results = TwoStageDiD(n_bootstrap=n_boot).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert np.isfinite(results.overall_att)
        assert results.overall_se > 0

    def test_balance_e_empty_cohorts_warns(self):
        """Unreasonably large balance_e should warn when no cohorts qualify."""
        data = generate_test_data()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = TwoStageDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
                balance_e=100,  # No cohort can satisfy this
            )
            balance_warns = [x for x in w if "No cohorts satisfy" in str(x.message)]
            assert len(balance_warns) > 0

        # Event study should contain only the reference period
        assert len(results.event_study_effects) == 1
        ref_key = list(results.event_study_effects.keys())[0]
        assert results.event_study_effects[ref_key]["n_obs"] == 0

    def test_proposition_5_nan_for_long_run_horizons(self):
        """Prop 5 horizons have n_obs > 0 but NaN inference (unidentified)."""
        # No never-treated: cohorts 3, 5, 7; periods 0-9.
        # h_bar = max(groups) - min(groups) = 7 - 3 = 4.
        # Horizons 0-3: identified, valid effects.
        # Horizons 4, 5, 6: Prop 5 unidentified — treated obs exist but
        # counterfactual is unidentified without never-treated units.
        data = generate_test_data(never_treated_frac=0.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = TwoStageDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )

        assert results.event_study_effects is not None

        # Check Prop 5 warning was emitted
        prop5_warnings = [x for x in w if "not identified without never-treated" in str(x.message)]
        assert len(prop5_warnings) > 0, "Proposition 5 warning should be emitted"

        # Horizons 0-3 should have observations and finite effects
        for h in range(0, 4):
            eff = results.event_study_effects[h]
            assert eff["n_obs"] > 0, f"Horizon {h} should have observations"
            assert np.isfinite(eff["effect"]), f"Horizon {h} effect should be finite"

        # Horizons 4, 5, 6: Prop 5 — n_obs > 0 but NaN inference
        for h in [4, 5, 6]:
            eff = results.event_study_effects[h]
            assert eff["n_obs"] > 0, f"Horizon {h} should have n_obs > 0 (Prop 5)"
            assert np.isnan(eff["effect"]), f"Horizon {h} effect should be NaN"
            assert np.isnan(eff["se"]), f"Horizon {h} SE should be NaN"
            assert np.isnan(eff["t_stat"]), f"Horizon {h} t_stat should be NaN"
            assert np.isnan(eff["p_value"]), f"Horizon {h} p_value should be NaN"
            assert np.isnan(eff["conf_int"][0]), f"Horizon {h} CI lower should be NaN"

    def test_group_effects_nan_for_all_nan_cohort(self):
        """Cohort with all NaN y_tilde produces NaN group effect."""
        # No never-treated units: cohorts 3, 5, 7; periods 0-9.
        # Periods 7, 8, 9 have zero untreated obs (all 3 cohorts treated by t=7).
        # Cohort 7: treated at periods 7-9, all have NaN y_tilde -> n_obs=0.
        # Cohorts 3, 5: have some valid treated periods -> n_obs > 0.
        data = generate_test_data(never_treated_frac=0.0)
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        assert results.group_effects is not None

        # Cohorts 3 and 5 should have valid effects
        for g in [3, 5]:
            eff = results.group_effects[g]
            assert eff["n_obs"] > 0, f"Cohort {g} should have observations"
            assert np.isfinite(eff["effect"]), f"Cohort {g} effect should be finite"

        # Cohort 7: all treated obs have NaN y_tilde -> zero obs -> NaN
        eff_7 = results.group_effects[7]
        assert eff_7["n_obs"] == 0, "Cohort 7 should have 0 observations"
        assert np.isnan(eff_7["effect"]), "Cohort 7 effect should be NaN"
        assert np.isnan(eff_7["se"]), "Cohort 7 SE should be NaN"


# =============================================================================
# TestTwoStageDiDParameters
# =============================================================================


class TestTwoStageDiDParameters:
    """Tests for parameter handling."""

    def test_get_params(self):
        """get_params should include all __init__ params."""
        est = TwoStageDiD(anticipation=1, alpha=0.1, n_bootstrap=100, seed=42, horizon_max=5)
        params = est.get_params()

        assert params["anticipation"] == 1
        assert params["alpha"] == 0.1
        assert params["n_bootstrap"] == 100
        assert params["seed"] == 42
        assert params["horizon_max"] == 5
        assert params["rank_deficient_action"] == "warn"
        assert params["cluster"] is None
        assert params["bootstrap_weights"] == "rademacher"

    def test_bootstrap_weights_in_get_set_params(self):
        """bootstrap_weights should appear in get_params and be settable."""
        est = TwoStageDiD(bootstrap_weights="mammen")
        assert est.get_params()["bootstrap_weights"] == "mammen"
        est.set_params(bootstrap_weights="webb")
        assert est.bootstrap_weights == "webb"

    def test_bootstrap_weights_invalid_raises(self):
        """Invalid bootstrap_weights value should raise ValueError."""
        with pytest.raises(ValueError, match="bootstrap_weights"):
            TwoStageDiD(bootstrap_weights="invalid")

    def test_set_params(self):
        """set_params should modify attributes."""
        est = TwoStageDiD()
        est.set_params(anticipation=2, alpha=0.1)

        assert est.anticipation == 2
        assert est.alpha == 0.1

    def test_set_params_returns_self(self):
        """set_params should return self for chaining."""
        est = TwoStageDiD()
        result = est.set_params(anticipation=1)
        assert result is est

    def test_set_params_unknown_raises(self):
        """set_params with unknown param should raise."""
        est = TwoStageDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(nonexistent_param=42)

    def test_rank_deficient_action_validation(self):
        """Invalid rank_deficient_action should raise."""
        with pytest.raises(ValueError, match="rank_deficient_action"):
            TwoStageDiD(rank_deficient_action="invalid")

    def test_cluster_changes_ses(self):
        """Different cluster variable should change SEs."""
        data = generate_test_data()
        # Add a cluster variable with fewer clusters than units
        data["cluster"] = data["unit"] % 10

        results_unit = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        results_cluster = TwoStageDiD(cluster="cluster").fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Point estimates should be the same
        assert abs(results_unit.overall_att - results_cluster.overall_att) < 1e-10
        # SEs should differ
        assert abs(results_unit.overall_se - results_cluster.overall_se) > 1e-6

    def test_horizon_max_limits_horizons(self):
        """horizon_max should limit event study horizons."""
        data = generate_test_data()

        results_full = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        results_limited = TwoStageDiD(horizon_max=2).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        full_horizons = set(results_full.event_study_effects.keys())
        limited_horizons = set(results_limited.event_study_effects.keys())

        assert len(limited_horizons) <= len(full_horizons)


# =============================================================================
# TestTwoStageDiDBootstrap
# =============================================================================


class TestTwoStageDiDBootstrap:
    """Tests for bootstrap inference."""

    def test_bootstrap_runs(self, ci_params):
        """Bootstrap should complete and produce results."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        results = TwoStageDiD(n_bootstrap=n_boot, seed=42).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.bootstrap_results is not None
        assert isinstance(results.bootstrap_results, TwoStageBootstrapResults)

    def test_bootstrap_structure(self, ci_params):
        """Bootstrap results should have correct structure."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        results = TwoStageDiD(n_bootstrap=n_boot, seed=42).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        br = results.bootstrap_results
        assert br.n_bootstrap == n_boot
        assert br.weight_type == "rademacher"
        assert br.overall_att_se > 0
        assert br.overall_att_ci[0] < br.overall_att_ci[1]
        assert 0 < br.overall_att_p_value <= 1

    def test_bootstrap_updates_inference(self, ci_params):
        """Bootstrap should update the main results inference."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)

        results_analytical = TwoStageDiD(seed=42).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        results_bootstrap = TwoStageDiD(n_bootstrap=n_boot, seed=42).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Point estimates should match
        assert abs(results_analytical.overall_att - results_bootstrap.overall_att) < 1e-10
        # SEs should differ (analytical GMM vs bootstrap)
        assert abs(results_analytical.overall_se - results_bootstrap.overall_se) > 1e-6

    def test_bootstrap_event_study(self, ci_params):
        """Bootstrap should work with event study specification."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        results = TwoStageDiD(n_bootstrap=n_boot, seed=42).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.event_study_ses is not None
        for h, se in results.bootstrap_results.event_study_ses.items():
            assert se > 0

    def test_bootstrap_weights_mammen(self, ci_params):
        """Bootstrap with mammen weights should produce valid results."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        results = TwoStageDiD(n_bootstrap=n_boot, bootstrap_weights="mammen", seed=42).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        br = results.bootstrap_results
        assert br is not None
        assert br.weight_type == "mammen"
        assert br.overall_att_se > 0
        assert np.isfinite(br.overall_att_p_value)

    def test_bootstrap_weights_webb(self, ci_params):
        """Bootstrap with webb weights should produce valid results."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        results = TwoStageDiD(n_bootstrap=n_boot, bootstrap_weights="webb", seed=42).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        br = results.bootstrap_results
        assert br is not None
        assert br.weight_type == "webb"
        assert br.overall_att_se > 0
        assert np.isfinite(br.overall_att_p_value)

    def test_bootstrap_weights_event_study(self, ci_params):
        """Bootstrap with non-default weights should work for event study aggregation."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        results = TwoStageDiD(n_bootstrap=n_boot, bootstrap_weights="mammen", seed=42).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        br = results.bootstrap_results
        assert br is not None
        assert br.weight_type == "mammen"
        assert br.event_study_ses is not None
        assert len(br.event_study_ses) > 0
        for h, se in br.event_study_ses.items():
            assert se > 0, f"Non-positive SE at horizon {h}"

    def test_bootstrap_weights_group(self, ci_params):
        """Bootstrap with non-default weights should work for group aggregation."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        results = TwoStageDiD(n_bootstrap=n_boot, bootstrap_weights="mammen", seed=42).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        br = results.bootstrap_results
        assert br is not None
        assert br.weight_type == "mammen"
        assert br.group_ses is not None
        assert len(br.group_ses) > 0
        for g, se in br.group_ses.items():
            assert se > 0, f"Non-positive SE for group {g}"


# =============================================================================
# TestTwoStageDiDConvenience
# =============================================================================


class TestTwoStageDiDConvenience:
    """Tests for convenience function."""

    def test_convenience_function_returns_results(self):
        """Convenience function should return TwoStageDiDResults."""
        data = generate_test_data()
        results = two_stage_did(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert isinstance(results, TwoStageDiDResults)
        assert results.overall_att > 0

    def test_convenience_function_kwargs(self):
        """Constructor kwargs should be forwarded."""
        data = generate_test_data()
        results = two_stage_did(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            anticipation=1,
            alpha=0.1,
        )

        assert isinstance(results, TwoStageDiDResults)
        assert results.alpha == 0.1

    def test_convenience_function_aggregate(self):
        """Convenience function should support aggregate parameter."""
        data = generate_test_data()
        results = two_stage_did(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None

    def test_estimator_summary_before_fit_raises(self):
        """Calling summary() before fit() should raise."""
        est = TwoStageDiD()
        with pytest.raises(RuntimeError, match="fitted"):
            est.summary()

    def test_print_summary(self, capsys):
        """print_summary should print to stdout."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        results.print_summary()
        captured = capsys.readouterr()
        assert "Two-Stage DiD" in captured.out

    def test_sparse_fallback_path(self):
        """Size guard falls back to per-column path and produces same results."""
        import diff_diff.two_stage as ts_mod

        data = generate_test_data(n_units=50, n_periods=6, seed=42)

        # Run with normal (high) threshold — uses dense path
        result_dense = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Patch threshold to 1 to force per-column path on all data
        orig = ts_mod._SPARSE_DENSE_THRESHOLD
        try:
            ts_mod._SPARSE_DENSE_THRESHOLD = 1
            result_sparse = TwoStageDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        finally:
            ts_mod._SPARSE_DENSE_THRESHOLD = orig

        np.testing.assert_allclose(result_dense.overall_att, result_sparse.overall_att, rtol=1e-10)
        np.testing.assert_allclose(result_dense.overall_se, result_sparse.overall_se, rtol=1e-10)


class TestSilentWarningAudit:
    """Tests for UserWarning emissions added by the silent warning audit."""

    def test_item2_nan_ytilde_masking_warning(self):
        """Item 2: Warn when NaN y_tilde values are masked."""
        # never_treated_frac=0 forces some periods without untreated obs
        data = generate_test_data(n_units=50, n_periods=10, never_treated_frac=0.0, seed=42)
        ts = TwoStageDiD()
        with pytest.warns(UserWarning, match="non-finite imputed outcomes"):
            ts.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_item3_always_treated_survey_weight_note(self):
        """Item 3: Enhanced always-treated warning mentions survey weights."""
        data = generate_test_data(n_units=50, n_periods=10, never_treated_frac=0.3, seed=42)
        # Shift time so min_time > 0, then set some units always-treated
        data["time"] = data["time"] + 1  # now min_time = 1
        min_time = data["time"].min()
        # Pick treated units and make them always-treated (first_treat=1 <= min_time=1)
        treated_units = data.loc[data["first_treat"] > 0, "unit"].unique()[:3]
        data.loc[data["unit"].isin(treated_units), "first_treat"] = min_time

        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(42)
        unit_weights = {u: rng.uniform(0.5, 2.0) for u in data["unit"].unique()}
        data["sw"] = data["unit"].map(unit_weights)
        survey = SurveyDesign(weights="sw")

        ts = TwoStageDiD()
        with pytest.warns(UserWarning, match="survey weights"):
            ts.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=survey,
            )

    def test_item3_always_treated_no_survey_note_without_weights(self):
        """Item 3 negative: Without survey weights, no survey note."""
        data = generate_test_data(n_units=50, n_periods=10, never_treated_frac=0.3, seed=42)
        data["time"] = data["time"] + 1
        min_time = data["time"].min()
        treated_units = data.loc[data["first_treat"] > 0, "unit"].unique()[:3]
        data.loc[data["unit"].isin(treated_units), "first_treat"] = min_time

        ts = TwoStageDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ts.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        survey_notes = [x for x in w if "survey weights" in str(x.message)]
        assert len(survey_notes) == 0, f"Unexpected survey note: {survey_notes}"

    def test_item2_nan_ytilde_event_study(self):
        """Item 2: y_tilde warning fires for aggregate='event_study'."""
        data = generate_test_data(n_units=50, n_periods=10, never_treated_frac=0.0, seed=42)
        ts = TwoStageDiD()
        with pytest.warns(UserWarning, match="non-finite imputed outcomes"):
            ts.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )

    def test_item2_nan_ytilde_group(self):
        """Item 2: y_tilde warning fires for aggregate='group'."""
        data = generate_test_data(n_units=50, n_periods=10, never_treated_frac=0.0, seed=42)
        ts = TwoStageDiD()
        with pytest.warns(UserWarning, match="non-finite imputed outcomes"):
            ts.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="group",
            )

    def test_iterative_fe_warns_on_nonconvergence(self):
        """Silent-failure audit axis B: _iterative_fe must warn when max_iter exhausts."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 8, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        y = rng.standard_normal(n_units * n_periods)
        idx = pd.RangeIndex(len(y))
        est = TwoStageDiD()

        with pytest.warns(UserWarning, match="did not converge"):
            est._iterative_fe(y, units, times, idx, max_iter=1, tol=1e-15)

    def test_iterative_fe_no_warning_on_convergence(self):
        """Silent-failure audit axis B: no warning on well-behaved convergent input."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 8, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        y = rng.standard_normal(n_units * n_periods)
        idx = pd.RangeIndex(len(y))
        est = TwoStageDiD()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est._iterative_fe(y, units, times, idx)
        assert not any("did not converge" in str(x.message) for x in w)

    # NOTE (intentional coverage narrowing): the direct `_iterative_demean`
    # non-convergence warn tests were retired with the method itself - the
    # covariate within-transform now routes through the shared MAP engine
    # with max_iter=10_000 hardcoded at the call site, so demean
    # non-convergence is no longer forceable THROUGH this estimator.
    # Engine-level warning coverage lives in
    # tests/test_utils.py::TestDemeanByGroups; the `_iterative_fe` warn
    # tests above still exercise this estimator's FE-solver warning path.

    def test_iterative_fe_zero_weight_unit_gets_nan_fe(self):
        """A unit whose rows ALL carry zero weight surfaces as NaN FE.

        Locks the shared-solver zero-weight contract (spillover precedent:
        never a silent finite 0.0) AND clean convergence - the historical
        pandas loop divided 0/0 there and burned max_iter iterations.
        """
        rng = np.random.default_rng(42)
        n_units, n_periods = 8, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        y = rng.standard_normal(n_units * n_periods)
        w = np.ones(n_units * n_periods)
        w[units == 2] = 0.0
        idx = pd.RangeIndex(len(y))
        est = TwoStageDiD()

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            unit_fe, time_fe = est._iterative_fe(y, units, times, idx, weights=w)
        assert not any("did not converge" in str(x.message) for x in rec)

        assert np.isnan(unit_fe[2])  # zero-weight unit: NaN, key retained
        assert all(np.isfinite(v) for u, v in unit_fe.items() if u != 2)
        assert all(np.isfinite(v) for v in time_fe.values())


# ---------------------------------------------------------------------------
# Silent-failure audit PR #9: sibling of finding #17 — both the analytical
# TSL variance (`two_stage.py`) and the multiplier-bootstrap bread
# (`two_stage_bootstrap.py`) previously fell back to `np.linalg.lstsq`
# silently when the Stage-2 `X'_2 W X_2` matrix was singular. They now
# emit a `UserWarning` with the same message shape as STD #17.
# ---------------------------------------------------------------------------


class TestTwoStageStage2BreadWarning:
    """Sibling of STD finding #17: the TwoStage Stage-2 bread (`X'_2 W X_2`)
    inversion was silent on a singular design and garbage on a *near*-singular
    one. It now routes through the shared `_rank_guarded_inv`, which rank-reduces
    to a finite SE on the identified subspace and warns. X_2 is built from
    treatment/event-time/group indicators (not user covariates), so we force the
    rank-deficiency at the `_rank_guarded_inv` seam rather than via data crafting,
    per the PR #334 CI review guidance."""

    def test_analytical_bread_rank_reduces_and_warns(self):
        """When the Stage-2 bread is rank-deficient, the analytical TSL path
        rank-reduces via `_rank_guarded_inv`, warns, and still returns a finite
        variance."""
        from unittest.mock import patch

        import diff_diff.two_stage as ts_mod

        data = generate_test_data(n_units=80, n_periods=6, seed=77)
        est = TwoStageDiD()

        real_rgi = ts_mod._rank_guarded_inv

        def force_drop(A, **kwargs):
            # Force a rank-deficiency *report* (finite inverse, n_dropped=1) to
            # exercise the rank-reduce warning path deterministically, with an
            # empty dropped mask so the identified ATT SE stays finite (the
            # NaN-for-dropped behavior is covered by the conley direct-call and
            # _rank_guarded_inv unit tests). Mirrors the helper's return arity.
            inv, _, rank = real_rgi(A)
            if kwargs.get("return_dropped"):
                return inv, 1, rank, np.zeros(A.shape[0], dtype=bool)
            return inv, 1, rank

        with patch.object(ts_mod, "_rank_guarded_inv", side_effect=force_drop):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = est.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                )
        fallback = [w for w in caught if "TwoStageDiD TSL variance" in str(w.message)]
        assert len(fallback) >= 1, (
            "Expected TSL-variance rank-reduce warning when the Stage-2 bread is "
            f"rank-deficient; got warnings: {[str(w.message) for w in caught]}"
        )
        msg = str(fallback[0].message)
        assert "rank-reducing" in msg
        assert "X_2'WX_2" in msg
        # rank-reduced bread must still produce a finite SE.
        assert np.isfinite(result.overall_se)

    def test_bootstrap_bread_rank_reduces_and_warns(self):
        """Same contract for the multiplier-bootstrap bread path (the
        cross-surface twin in two_stage_bootstrap.py)."""
        from unittest.mock import patch

        import diff_diff.two_stage_bootstrap as tsb_mod

        data = generate_test_data(n_units=80, n_periods=6, seed=77)
        est = TwoStageDiD(n_bootstrap=10, seed=0)

        real_rgi = tsb_mod._rank_guarded_inv

        def force_drop(A, **kwargs):
            # n_dropped=1 (warning fires), empty dropped mask so the identified
            # bootstrap SEs stay finite. Mirrors the helper's return arity.
            inv, _, rank = real_rgi(A)
            if kwargs.get("return_dropped"):
                return inv, 1, rank, np.zeros(A.shape[0], dtype=bool)
            return inv, 1, rank

        with patch.object(tsb_mod, "_rank_guarded_inv", side_effect=force_drop):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                est.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                )
        fallback = [w for w in caught if "TwoStageDiD multiplier bootstrap bread" in str(w.message)]
        assert len(fallback) >= 1, (
            "Expected bootstrap-bread rank-reduce warning when the bread is "
            f"rank-deficient; got warnings: {[str(w.message) for w in caught]}"
        )
        msg = str(fallback[0].message)
        assert "rank-reducing" in msg
        assert "X_2'WX_2" in msg

    @staticmethod
    def _drop_last_rgi(real_rgi):
        """Mock factory: genuinely drop the last Stage-2 coordinate (zero-fill its
        row/col in the inverse + report it dropped) so the caller NaNs it."""

        def drop_last(A, **kwargs):
            inv, _, rank = real_rgi(A)
            inv = np.array(inv, dtype=float)
            inv[-1, :] = 0.0
            inv[:, -1] = 0.0
            k = A.shape[0]
            dropped = np.zeros(k, dtype=bool)
            dropped[-1] = True
            if kwargs.get("return_dropped"):
                return inv, 1, k - 1, dropped
            return inv, 1, k - 1

        return drop_last

    @pytest.mark.parametrize(
        "aggregate,attr",
        [("event_study", "event_study_effects"), ("group", "group_effects")],
    )
    def test_dropped_coefficient_propagates_nan_inference(self, aggregate, attr):
        """A dropped (unidentified) Stage-2 coefficient must report NaN se / t_stat
        / p_value / conf_int at the ESTIMATOR level (not the zero-filled se=0) — the
        per-coefficient propagation the rank-guard enables (CI codex P2 test-depth).
        Covers the analytical event-study AND group surfaces."""
        from unittest.mock import patch

        import diff_diff.two_stage as ts_mod

        data = generate_test_data(n_units=80, n_periods=6, seed=77)
        with patch.object(
            ts_mod, "_rank_guarded_inv", side_effect=self._drop_last_rgi(ts_mod._rank_guarded_inv)
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = TwoStageDiD().fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    aggregate=aggregate,
                )
        effects = getattr(res, attr)
        nan_k = [k for k, e in effects.items() if np.isnan(e["se"])]
        fin_k = [k for k, e in effects.items() if np.isfinite(e["se"]) and e["se"] > 0]
        assert nan_k, (
            f"a dropped Stage-2 coordinate should yield a NaN-se {aggregate} effect; "
            f"got {{k: e['se'] for k, e in effects.items()}}"
        )
        assert fin_k, "identified effects should keep finite SE"
        # A dropped coefficient's FULL inference tuple must be NaN, not just se.
        for k in nan_k:
            e = effects[k]
            assert np.isnan(e["t_stat"]) and np.isnan(e["p_value"])
            assert all(np.isnan(c) for c in e["conf_int"])

    def test_bootstrap_dropped_coefficient_propagates_nan_inference(self):
        """Same estimator-level NaN propagation on the multiplier-bootstrap surface:
        a dropped Stage-2 coordinate NaNs the bootstrap coefficient column, so the
        affected event-time SE is NaN while identified horizons stay finite."""
        from unittest.mock import patch

        import diff_diff.two_stage as ts_mod
        import diff_diff.two_stage_bootstrap as tsb_mod

        data = generate_test_data(n_units=80, n_periods=6, seed=77)
        with patch.object(
            tsb_mod, "_rank_guarded_inv", side_effect=self._drop_last_rgi(tsb_mod._rank_guarded_inv)
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ts_mod.TwoStageDiD(n_bootstrap=25, seed=0).fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    aggregate="all",
                )
        es = res.event_study_effects
        nan_h = [h for h, e in es.items() if np.isnan(e["se"])]
        fin_h = [h for h, e in es.items() if np.isfinite(e["se"]) and e["se"] > 0]
        assert nan_h, (
            "a dropped Stage-2 coordinate should yield a NaN-se bootstrap horizon; "
            "got {h: e['se'] for h, e in es.items()}"
        )
        assert fin_h, "identified bootstrap horizons should keep finite SE"


# =============================================================================
# TestTwoStageDiDWaveE3ParityAlwaysTreated
# =============================================================================


def _build_parity_panel(
    sharp_psu0: bool = False,
    include_always_treated: bool = True,
    seed: int = 17,
) -> pd.DataFrame:
    """Build a 6-PSU x 4-period staggered panel for Wave E.3 parity tests.

    PSU layout:
      - PSUs 0, 1, 2 in stratum 0; PSUs 3, 4, 5 in stratum 1.
      - Each PSU contains 2 units by default; each unit has 4 observations
        (one per period).
      - Unit 0 (PSU 0): always-treated (first_treat=1) when
        ``include_always_treated=True``; never-treated otherwise.
      - Other units: alternating never-treated / staggered-onset.

    When ``sharp_psu0=True``, drop unit 1 (PSU 0) so the always-treated
    unit is the sole occupant of PSU 0. Under pre-PR TwoStageDiD this
    triggers the design-subset bug: dropping unit 0 from PSU 0 also drops
    PSU 0 from `resolved_survey.psu` so the reported `n_psu` falls from 6
    to 5. Wave E.3 parity contract: `n_psu` remains 6.
    """
    rng = np.random.default_rng(seed)
    rows = []
    unit_id = 0
    for psu in range(6):
        stratum = 0 if psu < 3 else 1
        N_h = 100  # FPC: hypothetical stratum size
        for _ in range(2):
            if unit_id == 0:
                first_treat = 1 if include_always_treated else 0
            elif unit_id % 2 == 0:
                first_treat = 0  # never-treated
            else:
                # Staggered onsets at period 2 or 3 (within observed periods 1-4)
                first_treat = 2 + (unit_id % 2)
            for t in range(1, 5):
                y = (
                    1.5 * (unit_id % 4)
                    + 0.3 * t
                    + (1.0 if first_treat > 0 and t >= first_treat else 0.0)
                    + rng.normal(0, 0.5)
                )
                rows.append(
                    dict(
                        id=unit_id,
                        t=t,
                        y=y,
                        g=first_treat if first_treat > 0 else 0,
                        psu=psu,
                        stratum=stratum,
                        N_h=N_h,
                        w=1.0,
                    )
                )
            unit_id += 1
    df = pd.DataFrame(rows)
    if sharp_psu0:
        # Drop unit 1 (also in PSU 0) so PSU 0's only resident is the
        # always-treated unit 0. Post-drop fit sample loses PSU 0 entirely.
        df = df[~((df["psu"] == 0) & (df["id"] == 1))].copy()
    return df


class TestTwoStageDiDWaveE3ParityAlwaysTreated:
    """Wave E.3 parity contract: TwoStageDiD's always-treated drop retains
    the FULL-DOMAIN survey design (n_psu, n_strata, df_survey, strata, fpc).

    Mirrors PR #482 SpilloverDiD Wave E.3 (merge 24de9062) which established
    the same invariant for SpilloverDiD's finite_mask / subpopulation drops.
    Adopts the R `survey::svyrecvar(subset())` convention (Lumley 2010 §2.5)
    and the in-library precedents at `imputation.py:2175-2183`
    (PreTrendsImputation) and `prep.py:1401-1432` (DCDH cell variance).

    Scope: this PR tests only `vcov_type` paths reachable from TwoStageDiD's
    public API — stratified-PSU meat via `_compute_stratified_meat_from_psu_scores`
    (with or without FPC) and unstratified `S.T @ S`. TwoStageDiD does NOT
    currently expose `vcov_type="conley"`; that follow-up is tracked
    separately at DEFERRED.md.
    """

    def test_a_no_always_treated_baseline_survey_path(self):
        """Sanity: no-always-treated fit reports n_psu reflecting the data's
        full PSU set (no artificial reduction). Locks the zero-pad-of-all-True
        mask = no-op invariant; the parity code path runs but is a no-op."""
        from diff_diff.survey import SurveyDesign

        data = _build_parity_panel(sharp_psu0=False, include_always_treated=False)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = TwoStageDiD()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = est.fit(
                data,
                outcome="y",
                unit="id",
                time="t",
                first_treat="g",
                survey_design=design,
            )
        assert result.survey_metadata is not None
        # Full-domain PSU count from data
        n_psu_data = int(data["psu"].nunique())
        n_strata_data = int(data["stratum"].nunique())
        assert result.survey_metadata.n_psu == n_psu_data
        assert result.survey_metadata.df_survey == n_psu_data - n_strata_data
        # ATT + SE finite
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)

    def test_b_full_domain_df_survey_under_always_treated_drop(self):
        """Wave E.3 parity contract: when the always-treated drop removes a
        PSU entirely from the fit sample, reported df_survey reflects the
        FULL-DOMAIN n_psu - n_strata, NOT the post-drop count."""
        from diff_diff.survey import SurveyDesign

        data = _build_parity_panel(sharp_psu0=True, include_always_treated=True)
        n_psu_full = int(data["psu"].nunique())  # = 6 (PSU 0 still in data via unit 0)
        n_strata_full = int(data["stratum"].nunique())  # = 2

        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = TwoStageDiD()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = est.fit(
                data,
                outcome="y",
                unit="id",
                time="t",
                first_treat="g",
                survey_design=design,
            )
        assert result.survey_metadata is not None
        # Wave E.3 parity contract: post-drop fit sample is missing PSU 0
        # (always-treated unit was its sole occupant), but full-domain count
        # is retained.
        assert result.survey_metadata.df_survey == n_psu_full - n_strata_full, (
            f"Wave E.3 parity: df_survey should reflect full domain "
            f"({n_psu_full - n_strata_full}); got {result.survey_metadata.df_survey}"
        )
        # Defensive: gate that the always-treated drop did not remove all
        # treated units (front-door check per
        # `feedback_front_door_gate_active_sample_mirror`).
        assert result.n_treated_obs > 0

    def test_c_full_domain_n_psu_reporting(self):
        """Companion to (b): reported n_psu reflects the FULL-DOMAIN count
        even when the always-treated drop empties a PSU from the fit sample."""
        from diff_diff.survey import SurveyDesign

        data = _build_parity_panel(sharp_psu0=True, include_always_treated=True)
        n_psu_full = int(data["psu"].nunique())  # = 6

        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = TwoStageDiD()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = est.fit(
                data,
                outcome="y",
                unit="id",
                time="t",
                first_treat="g",
                survey_design=design,
            )
        assert result.survey_metadata is not None
        assert result.survey_metadata.n_psu == n_psu_full, (
            f"Wave E.3 parity: n_psu should reflect full domain ({n_psu_full}); "
            f"got {result.survey_metadata.n_psu}"
        )

    def test_d_zero_pad_psu_score_spy(self):
        """Mock-spy on `_compute_stratified_meat_from_psu_scores`: capture the
        per-PSU score matrix and assert the row corresponding to the
        drop-only PSU (PSU 0) is exactly zero (zero-padded by the parity path).
        Locks the score-zero-pad invariant directly at the meat boundary."""
        from unittest.mock import patch

        import diff_diff.survey as survey_mod
        from diff_diff.survey import SurveyDesign

        data = _build_parity_panel(sharp_psu0=True, include_always_treated=True)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")

        captured = {}
        real_helper = survey_mod._compute_stratified_meat_from_psu_scores

        def spy(*, psu_scores, psu_strata, fpc_per_psu, lonely_psu):
            captured["psu_scores"] = np.asarray(psu_scores).copy()
            captured["psu_strata"] = np.asarray(psu_strata).copy()
            return real_helper(
                psu_scores=psu_scores,
                psu_strata=psu_strata,
                fpc_per_psu=fpc_per_psu,
                lonely_psu=lonely_psu,
            )

        with patch.object(survey_mod, "_compute_stratified_meat_from_psu_scores", side_effect=spy):
            est = TwoStageDiD()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(
                    data,
                    outcome="y",
                    unit="id",
                    time="t",
                    first_treat="g",
                    survey_design=design,
                )

        psu_scores = captured["psu_scores"]
        # Per-PSU score matrix has shape (G_full, k); G_full = 6 (full domain),
        # not 5 (post-drop fit sample).
        assert psu_scores.shape[0] == 6, (
            f"Wave E.3 parity: stratified meat should receive full-domain "
            f"G_full=6 per-PSU scores; got {psu_scores.shape[0]}"
        )
        # The score row for PSU 0 (the drop-only PSU) is exactly zero — its
        # only resident was the always-treated unit, dropped from stage-1/2.
        # PSU labels are sorted by np.unique → PSU 0 is row 0.
        assert np.allclose(psu_scores[0], 0.0), (
            "Wave E.3 parity: drop-only PSU row should be zero-padded; "
            f"got psu_scores[0]={psu_scores[0]}"
        )

    def test_e_subpopulation_plus_always_treated_composition(self):
        """Two zero-pad mechanisms compose cleanly: (i) SurveyDesign.subpopulation()
        excludes some rows via zero weights, (ii) always-treated drop removes
        the unit physically. Both should preserve full-domain n_psu / df_survey."""
        from diff_diff.survey import SurveyDesign

        data = _build_parity_panel(sharp_psu0=True, include_always_treated=True)
        n_psu_full = int(data["psu"].nunique())
        n_strata_full = int(data["stratum"].nunique())

        base_design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        # Subpopulation: exclude PSU 5's two units (mask=False there);
        # always-treated unit 0 is INSIDE the subpopulation domain.
        subpop_mask = data["psu"] != 5
        subpop_design, data_subpop = base_design.subpopulation(data, subpop_mask)

        est = TwoStageDiD()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = est.fit(
                data_subpop,
                outcome="y",
                unit="id",
                time="t",
                first_treat="g",
                survey_design=subpop_design,
            )

        # Always-treated warning must still fire (composition does not silence it).
        at_warnings = [w for w in caught if "treated in all observed periods" in str(w.message)]
        assert len(at_warnings) >= 1, (
            f"Expected always-treated warning under subpop+always-treated; "
            f"got: {[str(w.message) for w in caught]}"
        )

        assert result.survey_metadata is not None
        # Full-domain n_psu / df_survey retained (subpopulation = zero-weight
        # padding, doesn't reduce design dimension).
        assert result.survey_metadata.n_psu == n_psu_full
        assert result.survey_metadata.df_survey == n_psu_full - n_strata_full

    def test_f_cluster_as_psu_plus_always_treated(self):
        """Cluster-injection path (user-specified `cluster=` without explicit
        survey_design.psu): cluster column is injected as effective PSU.
        Wave E.3 parity must preserve full-domain n_psu count even when
        always-treated drop removes a cluster from the fit sample."""
        from diff_diff.survey import SurveyDesign

        data = _build_parity_panel(sharp_psu0=True, include_always_treated=True)
        n_psu_full = int(data["psu"].nunique())  # PSU column used as cluster
        n_strata_full = int(data["stratum"].nunique())

        # Survey design has strata + fpc but NO explicit psu — cluster=
        # injects "psu" column as effective PSU.
        design = SurveyDesign(weights="w", strata="stratum", fpc="N_h")
        est = TwoStageDiD(cluster="psu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = est.fit(
                data,
                outcome="y",
                unit="id",
                time="t",
                first_treat="g",
                survey_design=design,
            )
        assert result.survey_metadata is not None
        # Cluster-as-PSU injection produces full-domain n_psu (post-injection
        # count includes drop-only PSUs).
        assert result.survey_metadata.n_psu == n_psu_full, (
            f"Wave E.3 parity (cluster-injection): n_psu should reflect "
            f"full domain ({n_psu_full}); got {result.survey_metadata.n_psu}"
        )
        assert result.survey_metadata.df_survey == n_psu_full - n_strata_full

    def test_g_no_survey_path_unchanged_under_always_treated(self):
        """Pure unweighted path: always-treated drop with `survey_design=None`
        must produce IDENTICAL results to a fit on data that excludes the
        always-treated unit upstream (the parity zero-pad path is gated on
        `resolved_survey is not None` so the unweighted path is unaffected)."""
        data_with_at = _build_parity_panel(sharp_psu0=True, include_always_treated=True)
        data_without_at = data_with_at[data_with_at["id"] != 0].copy()

        est = TwoStageDiD()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_with_at = est.fit(
                data_with_at,
                outcome="y",
                unit="id",
                time="t",
                first_treat="g",
            )
            result_without_at = est.fit(
                data_without_at,
                outcome="y",
                unit="id",
                time="t",
                first_treat="g",
            )
        # Always-treated drop equivalence on the unweighted path: fitting
        # with the always-treated unit (which gets dropped internally) is
        # equivalent to fitting on the pre-filtered dataset.
        np.testing.assert_allclose(
            result_with_at.overall_att, result_without_at.overall_att, rtol=1e-10
        )
        np.testing.assert_allclose(
            result_with_at.overall_se, result_without_at.overall_se, rtol=1e-10
        )

    def test_h_psu_entirely_always_treated_unidentified_gate(self):
        """Optional sharper case: a PSU containing ONLY always-treated units
        is dropped entirely from the fit sample. Verify (i) the variance
        computation proceeds (zero-padded PSU 0 + 5 active PSUs gives G=6,
        sufficient for stratified-PSU variance identification; the meat row
        for PSU 0 is zero but its existence preserves the n_psu count),
        (ii) reported n_psu = full domain."""
        from diff_diff.survey import SurveyDesign

        data = _build_parity_panel(sharp_psu0=True, include_always_treated=True)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = TwoStageDiD()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = est.fit(
                data,
                outcome="y",
                unit="id",
                time="t",
                first_treat="g",
                survey_design=design,
            )
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None
        assert result.survey_metadata.n_psu == 6


# =============================================================================
# Phase 1b interstitial #5 (final): vcov_type threading on TwoStageDiD
# =============================================================================

from diff_diff import SurveyDesign  # noqa: E402


def _add_survey_cols(data, n_rep=8):
    """Add a constant pweight 'w' + unit-constant JK1 replicate-weight columns.

    Each replicate zeroes one block of units and rescales survivors by
    n_rep/(n_rep-1) (JK1 convention), broadcast panel-constant per unit.
    """
    d = data.copy()
    d["w"] = 1.0
    units = np.sort(d["unit"].unique())
    n_units = len(units)
    unit_pos = {u: i for i, u in enumerate(units)}
    rows = d["unit"].map(unit_pos).values
    units_per_rep = max(n_units // n_rep, 1)
    rep_cols = []
    for r in range(n_rep):
        w_r = np.ones(n_units)
        start = r * units_per_rep
        end = min((r + 1) * units_per_rep, n_units)
        w_r[start:end] = 0.0
        nz = w_r > 0
        w_r[nz] = w_r[nz] * n_rep / (n_rep - 1)
        d[f"rep_{r}"] = w_r[rows]
        rep_cols.append(f"rep_{r}")
    return d, rep_cols


def _fit_ts(est, data, **kw):
    """Fit helper suppressing convergence/bootstrap-size warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            **kw,
        )


def _assert_results_bit_equal(r0, r1):
    """Assert two TwoStageDiDResults are numerically identical (NaN-aware)."""

    def _eq(a, b):
        return a == b or (np.isnan(a) and np.isnan(b))

    assert _eq(r0.overall_att, r1.overall_att)
    assert _eq(r0.overall_se, r1.overall_se)
    for attr in ("event_study_effects", "group_effects"):
        e0 = getattr(r0, attr)
        e1 = getattr(r1, attr)
        if e0 is None:
            assert e1 is None
            continue
        assert set(e0) == set(e1)
        for k in e0:
            for f in ("effect", "se"):
                assert _eq(e0[k][f], e1[k][f]), f"{attr}[{k}][{f}] differs"


def _assert_full_bootstrap_nan(r):
    """Assert the FULL public NaN-propagation contract under a degenerate
    bootstrap (n_clusters<2 / n_psu<2): every overall + per-horizon + per-group
    inference field is NaN, not just the SE (REGISTRY NaN-inference contract).
    """
    # Overall inference fields
    assert np.isnan(r.overall_se)
    assert np.isnan(r.overall_t_stat)
    assert np.isnan(r.overall_p_value)
    assert all(np.isnan(x) for x in r.overall_conf_int)
    assert np.isnan(r.coef_var)
    # Bootstrap payload
    b = r.bootstrap_results
    assert np.isnan(b.overall_att_se)
    assert np.isnan(b.overall_att_p_value)
    assert all(np.isnan(x) for x in b.overall_att_ci)
    if b.event_study_ses:
        assert all(np.isnan(v) for v in b.event_study_ses.values())
    if b.group_ses:
        assert all(np.isnan(v) for v in b.group_ses.values())
    # Per-horizon event-study inference fields (skip reference-period markers,
    # which carry n_obs == 0 and are not real effects).
    for eff in (r.event_study_effects or {}).values():
        if eff.get("n_obs", 1) == 0:
            continue
        assert np.isnan(eff["se"])
        assert np.isnan(eff["t_stat"])
        assert np.isnan(eff["p_value"])
        assert all(np.isnan(x) for x in eff["conf_int"])
    # Per-group inference fields
    for eff in (r.group_effects or {}).values():
        assert np.isnan(eff["se"])
        assert np.isnan(eff["t_stat"])
        assert np.isnan(eff["p_value"])
        assert all(np.isnan(x) for x in eff["conf_int"])


class TestTwoStageDiDVcovType:
    """Phase 1b interstitial #5 (final): vcov_type input contract on TwoStageDiD.

    TwoStageDiD's variance is the Gardner (2022) two-stage GMM cluster-sandwich
    (``V = bread @ (S' S) @ bread``; always clusters, default at the unit
    column). ``vcov_type`` is permanently narrow to ``{"hc1"}``;
    analytical-sandwich families ``{classical, hc2, hc2_bm}`` and ``conley`` are
    rejected with GMM-meat-specific messages. Mirrors the ImputationDiD
    interstitial #3 template
    (``tests/test_imputation.py::TestImputationDiDVcovType``).
    """

    # ---- introspection / defaults ----

    def test_default_vcov_type(self):
        est = TwoStageDiD()
        assert est.vcov_type == "hc1"
        assert est.get_params()["vcov_type"] == "hc1"

    def test_results_carry_vcov_metadata(self):
        data = generate_test_data(n_units=80, seed=1)
        r = _fit_ts(TwoStageDiD(), data, aggregate="all")
        assert r.vcov_type == "hc1"
        assert r.cluster_name == "unit"
        assert r.n_clusters == 80

    def test_to_dict_carries_vcov(self):
        data = generate_test_data(n_units=80, seed=1)
        d = _fit_ts(TwoStageDiD(), data).to_dict()
        assert d["vcov_type"] == "hc1"
        assert d["cluster_name"] == "unit"
        assert d["n_clusters"] == 80
        assert d["inference_method"] == "cluster"

    def test_convenience_function_threads_vcov_type(self):
        data = generate_test_data(n_units=60, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = two_stage_did(data, "outcome", "unit", "time", "first_treat", vcov_type="hc1")
        assert r.vcov_type == "hc1"
        with pytest.raises(ValueError):
            two_stage_did(data, "outcome", "unit", "time", "first_treat", vcov_type="classical")

    def test_fit_clone_idempotence(self):
        data = generate_test_data(n_units=60, seed=3)
        est = TwoStageDiD()
        r1 = _fit_ts(est, data)
        clone = TwoStageDiD(**est.get_params())
        r2 = _fit_ts(clone, data)
        assert clone.vcov_type == "hc1"
        assert r1.overall_att == pytest.approx(r2.overall_att)
        assert r1.overall_se == pytest.approx(r2.overall_se)

    # ---- rejections ----

    @pytest.mark.parametrize("bad", ["classical", "hc2", "hc2_bm", "conley", "garbage"])
    def test_invalid_vcov_type_rejected_at_init(self, bad):
        with pytest.raises(ValueError, match=bad):
            TwoStageDiD(vcov_type=bad)

    @pytest.mark.parametrize("bad", ["classical", "hc2", "hc2_bm", "conley"])
    def test_fit_revalidates_after_set_params(self, bad):
        data = generate_test_data(n_units=60, seed=4)
        est = TwoStageDiD()
        est.set_params(vcov_type=bad)  # sklearn mutate-then-validate-at-use
        assert est.vcov_type == bad
        with pytest.raises(ValueError, match=bad):
            est.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

    def test_rejection_messages_are_methodology_specific(self):
        with pytest.raises(ValueError, match="GMM"):
            TwoStageDiD(vcov_type="hc2")
        with pytest.raises(ValueError, match="Conley|spatial"):
            TwoStageDiD(vcov_type="conley")

    # ---- summary labels ----

    def test_summary_default_renders_unit_cluster_cr1(self):
        data = generate_test_data(n_units=80, seed=5)
        s = _fit_ts(TwoStageDiD(), data).summary()
        assert "CR1 cluster-robust at unit" in s
        assert "HC1 heteroskedasticity" not in s

    def test_summary_explicit_cluster_renders_named_cr1(self):
        data = generate_test_data(n_units=80, seed=6)
        data["st"] = data["unit"] % 6
        r = _fit_ts(TwoStageDiD(cluster="st"), data)
        assert r.cluster_name == "st" and r.n_clusters == 6
        assert "CR1 cluster-robust at st, G=6" in r.summary()

    def test_summary_suppresses_variance_label_under_bootstrap(self):
        data = generate_test_data(n_units=80, seed=7)
        s = _fit_ts(TwoStageDiD(n_bootstrap=199, seed=7), data).summary()
        assert "Inference method:" in s and "bootstrap" in s
        assert "Variance estimator:" not in s

    def test_cluster_name_suppressed_under_tsl_survey(self):
        data, _ = _add_survey_cols(generate_test_data(n_units=80, seed=8))
        r = _fit_ts(TwoStageDiD(), data, survey_design=SurveyDesign(weights="w"))
        assert r.cluster_name is None
        assert r.n_clusters is None

    def test_cluster_name_suppressed_under_replicate_survey(self):
        data, rep_cols = _add_survey_cols(generate_test_data(n_units=80, seed=9))
        design = SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1")
        r = _fit_ts(TwoStageDiD(), data, survey_design=design)
        assert r.cluster_name is None
        assert r.n_clusters is None

    # ---- bit-equality regression guards (vcov_type='hc1' is a pure no-op) ----

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group", "all"])
    def test_default_path_bit_equal(self, aggregate):
        data = generate_test_data(n_units=80, seed=10)
        r0 = _fit_ts(TwoStageDiD(), data, aggregate=aggregate)
        r1 = _fit_ts(TwoStageDiD(vcov_type="hc1"), data, aggregate=aggregate)
        _assert_results_bit_equal(r0, r1)

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group", "all"])
    def test_cluster_path_bit_equal(self, aggregate):
        data = generate_test_data(n_units=80, seed=11)
        data["st"] = data["unit"] % 7
        r0 = _fit_ts(TwoStageDiD(cluster="st"), data, aggregate=aggregate)
        r1 = _fit_ts(TwoStageDiD(cluster="st", vcov_type="hc1"), data, aggregate=aggregate)
        _assert_results_bit_equal(r0, r1)

    def test_tsl_survey_path_bit_equal(self):
        data, _ = _add_survey_cols(generate_test_data(n_units=80, seed=12))
        design = SurveyDesign(weights="w")
        r0 = _fit_ts(TwoStageDiD(), data, survey_design=design, aggregate="all")
        r1 = _fit_ts(TwoStageDiD(vcov_type="hc1"), data, survey_design=design, aggregate="all")
        _assert_results_bit_equal(r0, r1)

    def test_replicate_survey_path_bit_equal(self):
        data, rep_cols = _add_survey_cols(generate_test_data(n_units=80, seed=13))
        design = SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1")
        r0 = _fit_ts(TwoStageDiD(), data, survey_design=design, aggregate="event_study")
        r1 = _fit_ts(
            TwoStageDiD(vcov_type="hc1"), data, survey_design=design, aggregate="event_study"
        )
        _assert_results_bit_equal(r0, r1)

    def test_bootstrap_path_bit_equal(self):
        data = generate_test_data(n_units=80, seed=14)
        r0 = _fit_ts(TwoStageDiD(n_bootstrap=199, seed=99), data, aggregate="all")
        r1 = _fit_ts(TwoStageDiD(n_bootstrap=199, seed=99, vcov_type="hc1"), data, aggregate="all")
        _assert_results_bit_equal(r0, r1)
        b0, b1 = r0.bootstrap_results, r1.bootstrap_results
        assert b0.overall_att_se == b1.overall_att_se
        if b0.event_study_ses:
            assert b0.event_study_ses == b1.event_study_ses
        if b0.group_ses:
            assert b0.group_ses == b1.group_ses

    # ---- cluster + replicate rejection ----

    def test_cluster_plus_replicate_weights_rejected(self):
        data, rep_cols = _add_survey_cols(generate_test_data(n_units=80, seed=15))
        data["st"] = data["unit"] % 5
        design = SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1")
        with pytest.raises(NotImplementedError, match="replicate"):
            TwoStageDiD(cluster="st").fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )

    # ---- bootstrap G<2 NaN guard (both entry paths) ----

    def test_bootstrap_single_cluster_returns_nan(self):
        data = generate_test_data(n_units=80, seed=16)
        data["solo"] = 1  # single cluster
        est = TwoStageDiD(cluster="solo", n_bootstrap=199, seed=3)
        with pytest.warns(UserWarning, match="n_clusters=1"):
            r = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="all",
            )
        _assert_full_bootstrap_nan(r)

    def test_bootstrap_single_psu_survey_returns_nan(self):
        data, _ = _add_survey_cols(generate_test_data(n_units=80, seed=17))
        data["onepsu"] = 0  # single PSU
        design = SurveyDesign(weights="w", psu="onepsu")
        est = TwoStageDiD(n_bootstrap=199, seed=3)
        with pytest.warns(UserWarning):
            r = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
                aggregate="event_study",
            )
        _assert_full_bootstrap_nan(r)

    # ---- metadata reflects the post-drop fit sample (codex P2) ----

    def test_n_clusters_reflects_post_drop_fit_sample(self):
        """Always-treated units are dropped before estimation, so reported
        n_clusters must equal the POST-DROP effective cluster count the GMM
        sandwich uses (cluster_ids = df[cluster_var] on the post-drop df), not
        the full-input cluster count. Regression for the codex P2 metadata bug.
        """
        # Restrict to time >= 3 so the first_treat=3 cohort becomes
        # always-treated (treated in every observed period) and is dropped.
        data = generate_test_data(n_units=80, seed=20)
        data = data[data["time"] >= 3].copy()
        first_by_unit = data.groupby("unit")["first_treat"].first()
        min_t = data["time"].min()
        always_treated = first_by_unit[(first_by_unit > 0) & (first_by_unit <= min_t)].index
        assert len(always_treated) > 0, "fixture should contain always-treated units"
        full_units = data["unit"].nunique()
        expected_g = full_units - len(always_treated)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = TwoStageDiD().fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )
        assert r.cluster_name == "unit"
        assert r.n_clusters == expected_g, (
            f"n_clusters should be the post-drop count {expected_g}, "
            f"got {r.n_clusters} (full input has {full_units})"
        )
        assert r.n_clusters < full_units  # would equal full_units under the bug
        # to_dict mirrors the corrected count
        assert r.to_dict()["n_clusters"] == expected_g

    def test_n_clusters_counts_nan_cluster_like_the_variance(self):
        """The GMM sandwich counts clusters via np.unique(cluster_ids), which
        keeps a single NaN group; Series.nunique() would drop NaN. n_clusters
        metadata must match the variance so a `cluster=` column with missing IDs
        cannot make the reported G undercount the SE's actual cluster count.
        Regression for the codex round-3 P2.
        """
        data = generate_test_data(n_units=80, seed=21)
        data["cl"] = (data["unit"] % 6).astype(float)
        data.loc[data["unit"].isin([0, 1, 2]), "cl"] = np.nan
        # No always-treated drop here (cohorts start at t=3, min_time=0), so
        # df == data; count clusters the way the variance does.
        expected_g = int(np.unique(data["cl"].values).size)
        n_valid = int(data.loc[data["cl"].notna(), "cl"].nunique())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = TwoStageDiD(cluster="cl").fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )
        assert r.cluster_name == "cl"
        assert r.n_clusters == expected_g
        # NaN is counted as a cluster (Series.nunique() would have dropped it),
        # so G strictly exceeds the distinct non-NaN cluster count.
        assert r.n_clusters > n_valid
        assert r.to_dict()["n_clusters"] == expected_g


# =============================================================================
# TestZeroWeightGroups — shared-engine migration behavioral locks
# =============================================================================


class TestZeroWeightGroups:
    """Zero-total-weight groups on the Stage-1 paths (shared-engine migration).

    JK1/plain-BRR replicate weights zero whole PSUs and reach Stage 1
    unmasked (keep_mask only drops always-treated units). Before the
    shared-engine migration the pandas loops divided 0/0 there: with
    covariates, y_dm/X_dm NaN-poisoned, EVERY replicate refit failed inside
    solve_ols(check_finite=True), and the fit returned NaN SEs after a
    non-convergence warning storm. These tests lock the fixed contract,
    including the warn_nan=False suppression of the per-replicate
    "non-finite imputed outcomes" warning (main-fit warning unchanged).
    """

    @staticmethod
    def _with_covariates(data, seed=7):
        rng = np.random.default_rng(seed)
        d = data.copy()
        x = rng.standard_normal((len(d), 2))
        d["x1"], d["x2"] = x[:, 0], x[:, 1]
        d["outcome"] = d["outcome"] + x @ np.array([0.6, -0.3])
        return d

    def test_replicate_covariates_zero_weight_psus_finite_se(self):
        """Covariates + JK1 zeroed-PSU replicates -> finite SE, no warning storm."""
        data, rep_cols = _add_survey_cols(generate_test_data(n_units=60, seed=21))
        data = self._with_covariates(data)
        design = SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            r = TwoStageDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
                survey_design=design,
            )
        messages = [str(w.message) for w in rec]
        assert not any("replicate refits failed" in m for m in messages)
        assert not any("did not converge" in m for m in messages)
        # Main-fit weights are all positive here, and the per-replicate
        # nan-ytilde warning is suppressed (warn_nan=False): zero copies.
        assert not any("non-finite imputed outcomes" in m for m in messages)
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se) and r.overall_se > 0

    def test_main_fit_zero_weight_unit_warns_once_and_fits(self):
        """Main-fit zero-weight treated unit: nan-ytilde warning UNCHANGED.

        The warn_nan=False suppression applies ONLY inside replicate-refit
        closures - a main fit that produces NaN y_tilde (zero-weight unit ->
        NaN FE) must still surface the practitioner-facing warning, and the
        fit must succeed (before the migration it raised an opaque
        ValueError from solve_ols on the NaN-poisoned demeaned design).
        """
        data = self._with_covariates(generate_test_data(n_units=40, seed=22))
        data["w"] = 1.0
        treated_units = data.loc[data["first_treat"] > 0, "unit"].unique()
        data.loc[data["unit"] == treated_units[0], "w"] = 0.0
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            r = TwoStageDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
                survey_design=SurveyDesign(weights="w"),
            )
        nan_ytilde = [
            m for m in (str(w.message) for w in rec) if "non-finite imputed outcomes" in m
        ]
        assert len(nan_ytilde) >= 1  # main-fit warning still fires
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se) and r.overall_se > 0


class TestEventStudyVcovPersistence:
    """M-092 follow-up: the full Gardner-GMM ES VCV + df provenance are
    persisted on TwoStageDiDResults, gated by inference mode."""

    @staticmethod
    def _panel(seed=7):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(90):
            g = [4, 6, 0][u % 3]
            for t in range(1, 11):
                y = 1.0 + 0.1 * t + u * 0.01 + (1.2 if g and t >= g else 0.0) + rng.normal(0, 0.3)
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "outcome": y,
                        "first_treat": g if g else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    def test_analytical_fit_persists_vcov_diag_matches_ses(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TwoStageDiD().fit(
                self._panel(),
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )
        assert res.event_study_vcov is not None
        assert res.event_study_vcov_index is not None
        ses = {h: d["se"] for h, d in res.event_study_effects.items()}
        diag = np.sqrt(np.maximum(np.diag(res.event_study_vcov), 0.0))
        for i, h in enumerate(res.event_study_vcov_index):
            np.testing.assert_allclose(diag[i], ses[h], rtol=1e-14)
        # Non-survey GMM inference is normal-theory: no df provenance.
        assert res.event_study_df is None

    def test_bootstrap_clears_vcov_and_df(self):
        # Bootstrap replaces the stored ES se/p/CI with percentile values:
        # the analytical matrix's diagonal no longer matches the stored SEs
        # and no df governed the stored inference - both must be None.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TwoStageDiD(n_bootstrap=20, seed=1).fit(
                self._panel(),
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )
        assert res.bootstrap_results is not None
        assert res.event_study_vcov is None
        assert res.event_study_vcov_index is None
        assert res.event_study_df is None

    def test_replicate_weight_survey_clears_vcov_and_threads_final_df(self):
        # Replicate-weight designs: reported ES SEs come from the replicate
        # VCV's mixed [overall, ES, groups] layout, so the analytical GMM
        # matrix must NOT be persisted; the final (possibly
        # dropped-replicate-tightened) survey df that every recomputed ES
        # row's safe_inference used IS threaded.
        data, rep_cols = _add_survey_cols(self._panel(), n_rep=8)
        design = SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TwoStageDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
                survey_design=design,
            )
        assert res.event_study_vcov is None
        assert res.event_study_vcov_index is None
        assert res.event_study_df is not None and res.event_study_df > 0
        assert res.event_study_df == float(res.survey_metadata.df_survey)
        from diff_diff.results_base import build_event_study_surface

        surface = build_event_study_surface(res)
        finite_p = np.isfinite(surface.p_value)
        assert finite_p.any()
        assert set(surface.df[finite_p].tolist()) == {float(res.event_study_df)}

    def test_group_only_aggregate_has_no_es_vcov(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TwoStageDiD().fit(
                self._panel(),
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="group",
            )
        assert res.event_study_effects is None
        assert res.event_study_vcov is None
        assert res.event_study_vcov_index is None
        assert res.event_study_df is None
