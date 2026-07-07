"""Tests for HAD survey_design= consolidation + deprecation cycle.

As of 3.7.0, ``HeterogeneousAdoptionDiD.fit`` no longer accepts ``survey=`` /
``weights=`` (they raise ``TypeError``; ``survey_design=`` is the sole weighting
entry and ``cband`` is keyword-only). The 7 pretest helpers (did_had_pretest_workflow
+ 4 array-in pretests + 2 data-in joint wrappers) still accept the deprecated
``survey=`` / ``weights=`` aliases pending a follow-up removal.

Each pretest-helper surface gets:

1. survey_design= positive smoke (new kwarg accepted, finite output).
2. weights= deprecation warning (DeprecationWarning emitted; back-compat
   numerics preserved).
3. survey= deprecation warning (DeprecationWarning emitted; back-compat
   numerics preserved).
4. Numerical parity legacy ≡ new at atol=0 (skipped on qug_test, which
   raises NotImplementedError on all paths).
5. Three-way mutex ValueError (any 2-of-3 combo).

The ``TestHADFitDeprecation`` class instead pins the fit()-surface removal
(``survey=`` / ``weights=`` → ``TypeError``; ``cband`` keyword-only).

Plus surface-spanning tests:
- make_pweight_design importable from diff_diff top-level.
- make_pweight_design ≡ _make_trivial_resolved (private alias).
- Array-in helpers reject SurveyDesign (TypeError).
- Bit-exact normalization-order invariant (scale-invariance).
- qug_test surface symmetry (signature consistent with siblings).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    HeterogeneousAdoptionDiD,
    SurveyDesign,
    did_had_pretest_workflow,
    joint_homogeneity_test,
    joint_pretrends_test,
    make_pweight_design,
    qug_test,
    stute_joint_pretest,
    stute_test,
    yatchew_hr_test,
)
from diff_diff.survey import ResolvedSurveyDesign

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def array_in_data():
    """Simple (d, dy) arrays for the 3 numeric array-in helpers."""
    rng = np.random.default_rng(0)
    G = 30
    d = rng.uniform(0, 1, size=G)
    dy = 0.5 + 1.5 * d + rng.normal(0, 0.3, size=G)
    return d, dy


@pytest.fixture
def array_in_doses():
    """Just doses for qug_test (single-array)."""
    return np.array([0.1, 0.3, 0.5, 0.7, 0.9])


@pytest.fixture
def two_period_panel():
    """Two-period panel for HAD.fit + did_had_pretest_workflow on
    aggregate='overall'. G=200 units, T=2 periods, dose constant within unit,
    Beta(0.5, 1) draws so d.min() approaches 0 (boundary at 0 satisfied for
    Design 1' continuous_at_zero)."""
    rng = np.random.default_rng(1)
    G = 200
    # Beta(0.5, 1) puts mass near 0; d.min() will be very small relative to
    # median, satisfying the Design 1' boundary heuristic.
    d = rng.beta(0.5, 1.0, size=G)
    rows = []
    for g in range(G):
        for t in (0, 1):
            y = 0.0 if t == 0 else d[g] * 1.2 + rng.normal(0, 0.1)
            rows.append({"unit": g, "time": t, "y": y, "d": (0.0 if t == 0 else d[g])})
    df = pd.DataFrame(rows)
    df["w"] = 1.0  # uniform weight column for SurveyDesign(weights="w")
    return df


@pytest.fixture
def event_study_panel():
    """Multi-period panel for joint_pretrends/joint_homogeneity workflows."""
    rng = np.random.default_rng(2)
    G = 30
    rows = []
    F = 2
    for g in range(G):
        d_g = rng.uniform(0.0, 1.0)
        for t in range(4):
            d_t = 0.0 if t < F else d_g
            y = (0.0 if t < F else d_t * 1.5) + rng.normal(0, 0.15)
            rows.append({"unit": g, "time": t, "y": y, "d": d_t})
    df = pd.DataFrame(rows)
    df["w"] = 1.0
    return df


# =============================================================================
# 1. Surface-spanning tests
# =============================================================================


class TestPublicHelpers:
    def test_make_pweight_design_export(self):
        """make_pweight_design is importable from the diff_diff top level."""
        from diff_diff import make_pweight_design as mpd

        assert mpd is make_pweight_design

    def test_make_pweight_design_returns_resolved(self):
        w = np.array([1.0, 2.0, 3.0, 4.0])
        resolved = make_pweight_design(w)
        assert isinstance(resolved, ResolvedSurveyDesign)
        assert resolved.weight_type == "pweight"
        assert resolved.strata is None
        assert resolved.psu is None
        assert resolved.fpc is None
        assert resolved.replicate_weights is None
        assert resolved.n_strata == 0
        assert resolved.n_psu == 4
        assert np.array_equal(resolved.weights, w.astype(np.float64))

    def test_make_pweight_design_eq_underscore_alias(self):
        """Permanent private alias _make_trivial_resolved IS make_pweight_design."""
        from diff_diff.survey import _make_trivial_resolved

        assert _make_trivial_resolved is make_pweight_design

    def test_make_pweight_design_rejects_scalar(self):
        """PR #376 R3 P1: scalar / 0-D inputs raise a clear front-door
        ValueError instead of bubbling a low-level numpy or dataclass
        exception (was: `1.0` would fail at `int(w.shape[0])` with
        `IndexError: tuple index out of range`)."""
        with pytest.raises(ValueError, match="weights must be 1-dimensional"):
            make_pweight_design(1.0)

    def test_make_pweight_design_rejects_zero_d_array(self):
        """PR #376 R3 P1: `np.array(1.0)` (0-D ndarray) raises ValueError."""
        with pytest.raises(ValueError, match="weights must be 1-dimensional"):
            make_pweight_design(np.array(1.0))

    def test_make_pweight_design_rejects_column_vector(self):
        """PR #376 R3 P1: `(n, 1)` column vectors raise ValueError pointing
        users to `df['w'].to_numpy()` instead of `df[['w']].to_numpy()`."""
        with pytest.raises(ValueError, match="weights must be 1-dimensional"):
            make_pweight_design(np.ones((5, 1)))

    def test_array_in_helpers_legacy_weights_scalar_raises_value_error(self, array_in_data):
        """PR #376 R3 P1: deprecated `weights=scalar` on array-in helpers
        also raises ValueError (the shim routes through make_pweight_design,
        which catches scalars at its front door)."""
        d, dy = array_in_data
        with pytest.raises(ValueError, match="weights must be 1-dimensional"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                stute_test(d, dy, weights=1.0, n_bootstrap=199, seed=0)
        with pytest.raises(ValueError, match="weights must be 1-dimensional"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                yatchew_hr_test(d, dy, weights=1.0)

    def test_qug_test_legacy_weights_scalar_raises_value_error(self, array_in_doses):
        """PR #376 R3 P1: deprecated `weights=scalar` on qug_test also raises
        ValueError (the shim routes through make_pweight_design before the
        NotImplementedError gate)."""
        with pytest.raises(ValueError, match="weights must be 1-dimensional"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                qug_test(array_in_doses, weights=1.0)


class TestArrayInTypeGuard:
    """Array-in helpers reject SurveyDesign (cannot resolve column names).

    Both the canonical `survey_design=SurveyDesign(...)` form AND the
    deprecated `survey=SurveyDesign(...)` alias trigger the same TypeError
    (PR #376 R1 P1: alias must behave identically to the canonical kwarg).
    """

    def test_stute_test_rejects_SurveyDesign(self, array_in_data):
        d, dy = array_in_data
        with pytest.raises(TypeError, match="make_pweight_design"):
            stute_test(d, dy, survey_design=SurveyDesign(weights="w"), n_bootstrap=199, seed=0)

    def test_stute_test_rejects_SurveyDesign_via_legacy_alias(self, array_in_data):
        """PR #376 R1 P1: `survey=SurveyDesign(...)` (deprecated alias) must
        trigger the same TypeError as `survey_design=SurveyDesign(...)`."""
        d, dy = array_in_data
        with pytest.raises(TypeError, match="make_pweight_design"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                stute_test(d, dy, survey=SurveyDesign(weights="w"), n_bootstrap=199, seed=0)

    def test_yatchew_hr_test_rejects_SurveyDesign(self, array_in_data):
        d, dy = array_in_data
        with pytest.raises(TypeError, match="make_pweight_design"):
            yatchew_hr_test(d, dy, survey_design=SurveyDesign(weights="w"))

    def test_yatchew_hr_test_rejects_SurveyDesign_via_legacy_alias(self, array_in_data):
        """PR #376 R1 P1: alias parity with canonical kwarg."""
        d, dy = array_in_data
        with pytest.raises(TypeError, match="make_pweight_design"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                yatchew_hr_test(d, dy, survey=SurveyDesign(weights="w"))

    def test_stute_joint_pretest_rejects_SurveyDesign(self):
        rng = np.random.default_rng(3)
        G = 30
        d = rng.uniform(0, 1, size=G)
        residuals = {0: rng.normal(0, 0.1, G)}
        fitted = {0: np.zeros(G)}
        X = np.column_stack([np.ones(G), d])
        with pytest.raises(TypeError, match="make_pweight_design"):
            stute_joint_pretest(
                residuals_by_horizon=residuals,
                fitted_by_horizon=fitted,
                doses=d,
                design_matrix=X,
                survey_design=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )

    def test_stute_joint_pretest_rejects_SurveyDesign_via_legacy_alias(self):
        """PR #376 R1 P1: alias parity with canonical kwarg."""
        rng = np.random.default_rng(3)
        G = 30
        d = rng.uniform(0, 1, size=G)
        residuals = {0: rng.normal(0, 0.1, G)}
        fitted = {0: np.zeros(G)}
        X = np.column_stack([np.ones(G), d])
        with pytest.raises(TypeError, match="make_pweight_design"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                stute_joint_pretest(
                    residuals_by_horizon=residuals,
                    fitted_by_horizon=fitted,
                    doses=d,
                    design_matrix=X,
                    survey=SurveyDesign(weights="w"),
                    n_bootstrap=199,
                    seed=0,
                )


class TestScaleInvariance:
    """Bit-exact normalization-order invariant (Stability invariant #7).

    The legacy weights= deprecation shim binds
    `survey_design = make_pweight_design(weights_unnormalized)` and lets
    the unified survey_design= path apply the mean=1 normalization step
    EXACTLY ONCE downstream. If the shim pre-normalized AND the unified
    path also normalized, the test statistic would scale differently
    under multiplicative weight rescaling.
    """

    def test_stute_weights_alias_scale_invariant(self, array_in_data):
        d, dy = array_in_data
        w = np.random.default_rng(4).uniform(0.5, 1.5, size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r1 = stute_test(d, dy, weights=w, n_bootstrap=199, seed=0)
            r2 = stute_test(d, dy, weights=w * 100.0, n_bootstrap=199, seed=0)
        # Use atol/rtol=1e-14 (per `feedback_assert_allclose_numerical_parity`):
        # the mean=1 normalization step `w * G/sum(w)` produces results that
        # agree to ~16 significant figures but not bit-exactly across
        # multiplicative rescaling (FP rounding in the renormalization step).
        np.testing.assert_allclose(r1.cvm_stat, r2.cvm_stat, atol=1e-14, rtol=1e-14)

    def test_yatchew_weights_alias_scale_invariant(self, array_in_data):
        d, dy = array_in_data
        w = np.random.default_rng(5).uniform(0.5, 1.5, size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r1 = yatchew_hr_test(d, dy, weights=w)
            r2 = yatchew_hr_test(d, dy, weights=w * 100.0)
        np.testing.assert_allclose(r1.t_stat_hr, r2.t_stat_hr, atol=1e-14, rtol=1e-14)


# =============================================================================
# 2. Per-surface deprecation + parity tests
# =============================================================================


class TestQUGTestDeprecation:
    """qug_test (array-in, gated): all paths raise NotImplementedError;
    consolidation tests focus on the deprecation/mutex cascade."""

    def test_survey_design_kwarg_raises_notimpl(self, array_in_doses):
        with pytest.raises(NotImplementedError, match="QUG"):
            qug_test(array_in_doses, survey_design=make_pweight_design(np.ones(5)))

    def test_weights_emits_deprecation_warning(self, array_in_doses):
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            with pytest.raises(NotImplementedError):
                qug_test(array_in_doses, weights=np.ones(5))

    def test_survey_emits_deprecation_warning(self, array_in_doses):
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            with pytest.raises(NotImplementedError):
                qug_test(array_in_doses, survey=SurveyDesign(weights="w"))

    def test_three_way_mutex_design_plus_survey(self, array_in_doses):
        with pytest.raises(ValueError, match="at most one of"):
            qug_test(
                array_in_doses,
                survey_design=make_pweight_design(np.ones(5)),
                survey=SurveyDesign(weights="w"),
            )

    def test_three_way_mutex_design_plus_weights(self, array_in_doses):
        with pytest.raises(ValueError, match="at most one of"):
            qug_test(
                array_in_doses,
                survey_design=make_pweight_design(np.ones(5)),
                weights=np.ones(5),
            )

    def test_three_way_mutex_all_three(self, array_in_doses):
        with pytest.raises(ValueError, match="at most one of"):
            qug_test(
                array_in_doses,
                survey_design=make_pweight_design(np.ones(5)),
                survey=SurveyDesign(weights="w"),
                weights=np.ones(5),
            )


class TestStuteTestDeprecation:
    def test_survey_design_kwarg_smoke(self, array_in_data):
        d, dy = array_in_data
        w = np.ones(30)
        r = stute_test(d, dy, survey_design=make_pweight_design(w), n_bootstrap=199, seed=0)
        assert np.isfinite(r.cvm_stat)
        assert 0.0 <= r.p_value <= 1.0

    def test_weights_emits_deprecation_warning(self, array_in_data):
        d, dy = array_in_data
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            stute_test(d, dy, weights=np.ones(30), n_bootstrap=199, seed=0)

    def test_survey_emits_deprecation_warning(self, array_in_data):
        d, dy = array_in_data
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            stute_test(
                d,
                dy,
                survey=make_pweight_design(np.ones(30)),
                n_bootstrap=199,
                seed=0,
            )

    def test_numerical_parity_weights_legacy_eq_new(self, array_in_data):
        d, dy = array_in_data
        w = np.random.default_rng(7).uniform(0.5, 1.5, size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = stute_test(d, dy, weights=w, n_bootstrap=199, seed=0)
        r_new = stute_test(d, dy, survey_design=make_pweight_design(w), n_bootstrap=199, seed=0)
        assert r_legacy.cvm_stat == r_new.cvm_stat
        assert r_legacy.p_value == r_new.p_value

    def test_numerical_parity_survey_legacy_eq_new(self, array_in_data):
        d, dy = array_in_data
        w = np.random.default_rng(8).uniform(0.5, 1.5, size=30)
        resolved = make_pweight_design(w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = stute_test(d, dy, survey=resolved, n_bootstrap=199, seed=0)
        r_new = stute_test(d, dy, survey_design=resolved, n_bootstrap=199, seed=0)
        assert r_legacy.cvm_stat == r_new.cvm_stat
        assert r_legacy.p_value == r_new.p_value

    def test_three_way_mutex_design_plus_survey(self, array_in_data):
        d, dy = array_in_data
        w = np.ones(30)
        with pytest.raises(ValueError, match="at most one of"):
            stute_test(
                d,
                dy,
                survey_design=make_pweight_design(w),
                survey=make_pweight_design(w),
                n_bootstrap=199,
                seed=0,
            )

    def test_three_way_mutex_all_three(self, array_in_data):
        d, dy = array_in_data
        w = np.ones(30)
        with pytest.raises(ValueError, match="at most one of"):
            stute_test(
                d,
                dy,
                survey_design=make_pweight_design(w),
                survey=make_pweight_design(w),
                weights=w,
                n_bootstrap=199,
                seed=0,
            )


class TestYatchewHRTestDeprecation:
    def test_survey_design_kwarg_smoke(self, array_in_data):
        d, dy = array_in_data
        w = np.ones(30)
        r = yatchew_hr_test(d, dy, survey_design=make_pweight_design(w))
        assert np.isfinite(r.t_stat_hr)

    def test_weights_emits_deprecation_warning(self, array_in_data):
        d, dy = array_in_data
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            yatchew_hr_test(d, dy, weights=np.ones(30))

    def test_survey_emits_deprecation_warning(self, array_in_data):
        d, dy = array_in_data
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            yatchew_hr_test(d, dy, survey=make_pweight_design(np.ones(30)))

    def test_numerical_parity_weights_legacy_eq_new(self, array_in_data):
        d, dy = array_in_data
        w = np.random.default_rng(9).uniform(0.5, 1.5, size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = yatchew_hr_test(d, dy, weights=w)
        r_new = yatchew_hr_test(d, dy, survey_design=make_pweight_design(w))
        assert r_legacy.t_stat_hr == r_new.t_stat_hr
        assert r_legacy.p_value == r_new.p_value

    def test_three_way_mutex_design_plus_weights(self, array_in_data):
        d, dy = array_in_data
        with pytest.raises(ValueError, match="at most one of"):
            yatchew_hr_test(
                d,
                dy,
                survey_design=make_pweight_design(np.ones(30)),
                weights=np.ones(30),
            )


class TestStuteJointPretestDeprecation:
    def _setup(self):
        rng = np.random.default_rng(10)
        G = 30
        d = rng.uniform(0, 1, size=G)
        residuals = {0: rng.normal(0, 0.1, G), 1: rng.normal(0, 0.1, G)}
        fitted = {0: np.zeros(G), 1: np.zeros(G)}
        X = np.column_stack([np.ones(G), d])
        return d, residuals, fitted, X

    def test_survey_design_kwarg_smoke(self):
        d, residuals, fitted, X = self._setup()
        w = np.ones(30)
        r = stute_joint_pretest(
            residuals_by_horizon=residuals,
            fitted_by_horizon=fitted,
            doses=d,
            design_matrix=X,
            survey_design=make_pweight_design(w),
            n_bootstrap=199,
            seed=0,
        )
        assert np.isfinite(r.cvm_stat_joint)

    def test_weights_emits_deprecation_warning(self):
        d, residuals, fitted, X = self._setup()
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            stute_joint_pretest(
                residuals_by_horizon=residuals,
                fitted_by_horizon=fitted,
                doses=d,
                design_matrix=X,
                weights=np.ones(30),
                n_bootstrap=199,
                seed=0,
            )

    def test_survey_emits_deprecation_warning(self):
        d, residuals, fitted, X = self._setup()
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            stute_joint_pretest(
                residuals_by_horizon=residuals,
                fitted_by_horizon=fitted,
                doses=d,
                design_matrix=X,
                survey=make_pweight_design(np.ones(30)),
                n_bootstrap=199,
                seed=0,
            )

    def test_numerical_parity_weights_legacy_eq_new(self):
        d, residuals, fitted, X = self._setup()
        w = np.random.default_rng(11).uniform(0.5, 1.5, size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = stute_joint_pretest(
                residuals_by_horizon=residuals,
                fitted_by_horizon=fitted,
                doses=d,
                design_matrix=X,
                weights=w,
                n_bootstrap=199,
                seed=0,
            )
        r_new = stute_joint_pretest(
            residuals_by_horizon=residuals,
            fitted_by_horizon=fitted,
            doses=d,
            design_matrix=X,
            survey_design=make_pweight_design(w),
            n_bootstrap=199,
            seed=0,
        )
        assert r_legacy.cvm_stat_joint == r_new.cvm_stat_joint
        assert r_legacy.p_value == r_new.p_value

    def test_three_way_mutex_all_three(self):
        d, residuals, fitted, X = self._setup()
        w = np.ones(30)
        with pytest.raises(ValueError, match="at most one of"):
            stute_joint_pretest(
                residuals_by_horizon=residuals,
                fitted_by_horizon=fitted,
                doses=d,
                design_matrix=X,
                survey_design=make_pweight_design(w),
                survey=make_pweight_design(w),
                weights=w,
                n_bootstrap=199,
                seed=0,
            )


class TestJointPretrendsTestDeprecation:
    def test_survey_design_kwarg_smoke(self, event_study_panel):
        df = event_study_panel
        r = joint_pretrends_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[0],
            base_period=1,
            survey_design=SurveyDesign(weights="w"),
            n_bootstrap=199,
            seed=0,
        )
        assert np.isfinite(r.cvm_stat_joint)

    def test_weights_emits_deprecation_warning(self, event_study_panel):
        df = event_study_panel
        n = len(df)
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[0],
                base_period=1,
                weights=np.ones(n),
                n_bootstrap=199,
                seed=0,
            )

    def test_survey_emits_deprecation_warning(self, event_study_panel):
        df = event_study_panel
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[0],
                base_period=1,
                survey=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )

    def test_three_way_mutex_design_plus_survey(self, event_study_panel):
        df = event_study_panel
        n = len(df)
        with pytest.raises(ValueError, match="at most one of"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[0],
                base_period=1,
                survey_design=SurveyDesign(weights="w"),
                weights=np.ones(n),
                n_bootstrap=199,
                seed=0,
            )

    def test_legacy_alias_parity_survey(self, event_study_panel):
        """PR #376 R9 P3: deprecated `survey=SurveyDesign(...)` ≡ canonical
        `survey_design=SurveyDesign(...)` on joint_pretrends_test (locks
        rebinding parity)."""
        df = event_study_panel
        sd = SurveyDesign(weights="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[0],
                base_period=1,
                survey=sd,
                n_bootstrap=199,
                seed=0,
            )
        r_new = joint_pretrends_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[0],
            base_period=1,
            survey_design=sd,
            n_bootstrap=199,
            seed=0,
        )
        assert r_legacy.cvm_stat_joint == r_new.cvm_stat_joint
        assert r_legacy.p_value == r_new.p_value

    def test_legacy_alias_parity_weights(self, event_study_panel):
        """PR #376 R10 P3: deprecated `weights=np.ones(n)` ≡ canonical
        `survey_design=SurveyDesign(weights="w")` (uniform 1.0 column) on
        joint_pretrends_test."""
        df = event_study_panel
        n = len(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[0],
                base_period=1,
                weights=np.ones(n),
                n_bootstrap=199,
                seed=0,
            )
        r_new = joint_pretrends_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[0],
            base_period=1,
            survey_design=SurveyDesign(weights="w"),
            n_bootstrap=199,
            seed=0,
        )
        assert r_legacy.cvm_stat_joint == r_new.cvm_stat_joint
        assert r_legacy.p_value == r_new.p_value


class TestJointHomogeneityTestDeprecation:
    def test_survey_design_kwarg_smoke(self, event_study_panel):
        df = event_study_panel
        r = joint_homogeneity_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            post_periods=[2, 3],
            base_period=1,
            survey_design=SurveyDesign(weights="w"),
            n_bootstrap=199,
            seed=0,
        )
        assert np.isfinite(r.cvm_stat_joint)

    def test_weights_emits_deprecation_warning(self, event_study_panel):
        df = event_study_panel
        n = len(df)
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[2, 3],
                base_period=1,
                weights=np.ones(n),
                n_bootstrap=199,
                seed=0,
            )

    def test_survey_emits_deprecation_warning(self, event_study_panel):
        df = event_study_panel
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[2, 3],
                base_period=1,
                survey=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )

    def test_legacy_alias_parity_survey(self, event_study_panel):
        """PR #376 R9 P3: deprecated `survey=SurveyDesign(...)` ≡ canonical
        `survey_design=SurveyDesign(...)` on joint_homogeneity_test."""
        df = event_study_panel
        sd = SurveyDesign(weights="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = joint_homogeneity_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[2, 3],
                base_period=1,
                survey=sd,
                n_bootstrap=199,
                seed=0,
            )
        r_new = joint_homogeneity_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            post_periods=[2, 3],
            base_period=1,
            survey_design=sd,
            n_bootstrap=199,
            seed=0,
        )
        assert r_legacy.cvm_stat_joint == r_new.cvm_stat_joint
        assert r_legacy.p_value == r_new.p_value

    def test_legacy_alias_parity_weights(self, event_study_panel):
        """PR #376 R10 P3: deprecated `weights=np.ones(n)` ≡ canonical
        `survey_design=SurveyDesign(weights="w")` on
        joint_homogeneity_test."""
        df = event_study_panel
        n = len(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = joint_homogeneity_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[2, 3],
                base_period=1,
                weights=np.ones(n),
                n_bootstrap=199,
                seed=0,
            )
        r_new = joint_homogeneity_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            post_periods=[2, 3],
            base_period=1,
            survey_design=SurveyDesign(weights="w"),
            n_bootstrap=199,
            seed=0,
        )
        assert r_legacy.cvm_stat_joint == r_new.cvm_stat_joint
        assert r_legacy.p_value == r_new.p_value


class TestHADFitDeprecation:
    def test_survey_design_kwarg_smoke(self, two_period_panel):
        df = two_period_panel
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        r = est.fit(df, "y", "d", "time", "unit", survey_design=SurveyDesign(weights="w"))
        assert np.isfinite(r.att)

    def test_weights_kwarg_removed(self, two_period_panel):
        """`weights=` was removed on HAD.fit() in 3.7.0 (survey_design= only)."""
        df = two_period_panel
        n = len(df)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.raises(TypeError, match="unexpected keyword argument 'weights'"):
            est.fit(df, "y", "d", "time", "unit", weights=np.ones(n))

    def test_survey_kwarg_removed(self, two_period_panel):
        """`survey=` was removed on HAD.fit() in 3.7.0 (use survey_design=)."""
        df = two_period_panel
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.raises(TypeError, match="unexpected keyword argument 'survey'"):
            est.fit(df, "y", "d", "time", "unit", survey=SurveyDesign(weights="w"))

    def test_cband_is_keyword_only(self, two_period_panel):
        """`cband` became keyword-only in 3.7.0 (it followed the removed
        positional survey/weights slots)."""
        df = two_period_panel
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.raises(TypeError):
            # positional through aggregate, then a positional cband -- now
            # keyword-only, so the 8th positional argument is rejected.
            est.fit(df, "y", "d", "time", "unit", None, "overall", True)
        r = est.fit(df, "y", "d", "time", "unit", cband=True)
        assert np.isfinite(r.att)

    def test_fit_rejects_pre_resolved_design_overall(self, two_period_panel):
        """PR #376 R8 P1: HAD.fit() data-in surface must reject a
        pre-resolved ResolvedSurveyDesign with TypeError pointing users to
        `SurveyDesign(weights='col_name', ...)`. Mirrors the array-in
        helpers' rejection of SurveyDesign — the data-in/array-in surface
        split is symmetric."""
        df = two_period_panel
        n = len(df)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        # survey_design=ResolvedSurveyDesign should raise TypeError.
        with pytest.raises(TypeError, match=r"`survey_design=` accepts a SurveyDesign"):
            est.fit(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey_design=make_pweight_design(np.ones(n // 2)),
            )

    def test_fit_rejects_pre_resolved_design_event_study(self, event_study_continuous_panel):
        """PR #376 R8 P1: same TypeError on aggregate='event_study'."""
        df = event_study_continuous_panel
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", n_bootstrap=99, seed=0)
        with pytest.raises(TypeError, match=r"`survey_design=` accepts a SurveyDesign"):
            est.fit(
                df,
                "y",
                "d",
                "time",
                "unit",
                aggregate="event_study",
                survey_design=make_pweight_design(np.ones(200)),
            )


class TestDidHadPretestWorkflowDeprecation:
    def test_survey_design_kwarg_smoke(self, two_period_panel):
        df = two_period_panel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # QUG-skip warning
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey_design=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )
        assert report.qug is None  # skipped under survey path
        assert report.stute is not None

    def test_weights_emits_deprecation_warning(self, two_period_panel):
        df = two_period_panel
        n = len(df)
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # We still need to allow the DeprecationWarning to propagate
                # to the outer pytest.warns; only filter UserWarning.
                warnings.simplefilter("always", DeprecationWarning)
                did_had_pretest_workflow(
                    df,
                    "y",
                    "d",
                    "time",
                    "unit",
                    weights=np.ones(n),
                    n_bootstrap=199,
                    seed=0,
                )

    def test_survey_emits_deprecation_warning(self, two_period_panel):
        df = two_period_panel
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("always", DeprecationWarning)
                did_had_pretest_workflow(
                    df,
                    "y",
                    "d",
                    "time",
                    "unit",
                    survey=SurveyDesign(weights="w"),
                    n_bootstrap=199,
                    seed=0,
                )

    def test_three_way_mutex_all_three(self, two_period_panel):
        df = two_period_panel
        n = len(df)
        with pytest.raises(ValueError, match="at most one of"):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey_design=SurveyDesign(weights="w"),
                survey=SurveyDesign(weights="w"),
                weights=np.ones(n),
                n_bootstrap=199,
                seed=0,
            )

    def test_legacy_alias_parity_survey_overall(self, two_period_panel):
        """PR #376 R9 P3: deprecated `survey=SurveyDesign(...)` ≡ canonical
        `survey_design=SurveyDesign(...)` on
        did_had_pretest_workflow(aggregate='overall'). Locks rebinding
        parity on the workflow's overall-path data-in surface."""
        df = two_period_panel
        sd = SurveyDesign(weights="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # QUG-skip warning
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey=sd,
                n_bootstrap=199,
                seed=0,
            )
            r_new = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey_design=sd,
                n_bootstrap=199,
                seed=0,
            )
        assert r_legacy.stute.cvm_stat == r_new.stute.cvm_stat
        assert r_legacy.stute.p_value == r_new.stute.p_value
        assert r_legacy.yatchew.t_stat_hr == r_new.yatchew.t_stat_hr

    def test_legacy_alias_parity_weights_overall(self, two_period_panel):
        """PR #376 R10 P3: deprecated `weights=np.ones(n)` ≡ canonical
        `survey_design=SurveyDesign(weights="w")` on
        did_had_pretest_workflow(aggregate='overall'). Closes the data-in
        rebinding-parity gap on the weights= shortcut path."""
        df = two_period_panel
        n = len(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                weights=np.ones(n),
                n_bootstrap=199,
                seed=0,
            )
            r_new = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey_design=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )
        assert r_legacy.stute.cvm_stat == r_new.stute.cvm_stat
        assert r_legacy.stute.p_value == r_new.stute.p_value
        assert r_legacy.yatchew.t_stat_hr == r_new.yatchew.t_stat_hr


# =============================================================================
# 3. PR #376 R2 P1: extended dispatch-matrix coverage on the new front door
# =============================================================================
#
# Reviewer flagged that the canonical `survey_design=` kwarg was added across
# all HAD design × aggregate combinations but only directly tested on the
# two-period continuous_at_zero / overall path. These tests cover the
# weighted mass_point overall path, the weighted continuous event-study
# path, and the workflow event-study path — each with both a
# `survey_design=` smoke and a legacy-alias parity check.


@pytest.fixture
def mass_point_panel():
    """Two-period panel with a continuous mass-point at d_lower=0.05.

    G=200 units, fraction `0.06 > 0.02` modal at d_lower triggers the
    mass-point heuristic in HAD's auto-detection. Used to exercise
    `design="mass_point"` survey_design= forwarding through the weighted
    2SLS sandwich.
    """
    rng = np.random.default_rng(13)
    G = 200
    n_modal = int(0.06 * G)  # 12 units at d_lower
    d_modal = np.full(n_modal, 0.05)
    d_continuous = rng.uniform(0.06, 1.0, size=G - n_modal)
    d = np.concatenate([d_modal, d_continuous])
    rng.shuffle(d)
    rows = []
    for g in range(G):
        for t in (0, 1):
            y = 0.0 if t == 0 else d[g] * 1.2 + rng.normal(0, 0.1)
            rows.append({"unit": g, "time": t, "y": y, "d": (0.0 if t == 0 else d[g])})
    df = pd.DataFrame(rows)
    df["w"] = 1.0
    return df


@pytest.fixture
def event_study_continuous_panel():
    """Multi-period continuous_at_zero panel for HAD.fit aggregate='event_study'.

    G=200 units, T=3 periods (t=0 pre, t=1 base, t=2 post), Beta(0.5, 1)
    doses so d.min() approaches 0 (Design 1' boundary heuristic satisfied),
    F=2 (treatment starts at t=2)."""
    rng = np.random.default_rng(14)
    G = 200
    d = rng.beta(0.5, 1.0, size=G)
    rows = []
    F = 2
    for g in range(G):
        for t in range(3):
            d_t = 0.0 if t < F else d[g]
            y = (0.0 if t < F else d_t * 1.2) + rng.normal(0, 0.1)
            rows.append({"unit": g, "time": t, "y": y, "d": d_t})
    df = pd.DataFrame(rows)
    df["w"] = 1.0
    return df


class TestHADFitMassPointSurveyDesign:
    """PR #376 R2 P1: cover `design='mass_point'` + survey_design= path.

    Mass-point + survey requires vcov_type='hc1' (not the classical default)
    per the documented Phase 4.5 B deviation: the survey path composes
    Binder-TSL on the HC1-scale IF.
    """

    def test_survey_design_kwarg_smoke(self, mass_point_panel):
        df = mass_point_panel
        est = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # mass-point methodology warning
            r = est.fit(df, "y", "d", "time", "unit", survey_design=SurveyDesign(weights="w"))
        assert np.isfinite(r.att)
        assert np.isfinite(r.se)


class TestHADFitEventStudySurveyDesign:
    """PR #376 R2 P1: cover aggregate='event_study' + cband=True + survey_design=."""

    def test_survey_design_kwarg_smoke(self, event_study_continuous_panel):
        df = event_study_continuous_panel
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", n_bootstrap=99, seed=0)
        r = est.fit(
            df,
            "y",
            "d",
            "time",
            "unit",
            aggregate="event_study",
            survey_design=SurveyDesign(weights="w"),
            cband=True,
        )
        # Event-study returns HeterogeneousAdoptionDiDEventStudyResults
        assert r.att.shape[0] >= 1
        assert np.all(np.isfinite(r.att))
        assert r.cband_low is not None
        assert r.cband_high is not None


class TestDidHadPretestWorkflowEventStudySurveyDesign:
    """PR #376 R2 P1: cover did_had_pretest_workflow(aggregate='event_study',
    survey_design=...)."""

    def test_survey_design_kwarg_smoke(self, event_study_panel):
        df = event_study_panel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # QUG-skip + staggered
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                aggregate="event_study",
                survey_design=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )
        assert report.qug is None  # skipped under survey path
        assert report.homogeneity_joint is not None

    def test_legacy_alias_parity_survey(self, event_study_panel):
        """survey=SurveyDesign(...) (deprecated) ≡ survey_design=SurveyDesign(...)
        on workflow event-study path."""
        df = event_study_panel
        sd = SurveyDesign(weights="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                aggregate="event_study",
                survey=sd,
                n_bootstrap=199,
                seed=0,
            )
            r_new = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                aggregate="event_study",
                survey_design=sd,
                n_bootstrap=199,
                seed=0,
            )
        # Joint Stute on the event-study path is bootstrap-driven; both calls
        # use the same seed=0 + same survey design → identical bootstrap
        # multiplier draws → identical p-values + statistics.
        assert r_legacy.homogeneity_joint.cvm_stat_joint == r_new.homogeneity_joint.cvm_stat_joint
        assert r_legacy.homogeneity_joint.p_value == r_new.homogeneity_joint.p_value

    def test_legacy_alias_parity_weights(self, event_study_panel):
        """weights=arr (deprecated) ≡ survey_design=SurveyDesign(weights='w')
        with uniform 1.0 weights on the workflow event-study path. Locks the
        nested-DeprecationWarning suppression: the user-facing warning fires
        ONCE at the workflow front door, no extra warnings from the joint
        wrappers when survey/weights are forwarded internally."""
        df = event_study_panel
        n = len(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with warnings.catch_warnings(record=True) as w_record:
                warnings.simplefilter("always", DeprecationWarning)
                r_legacy = did_had_pretest_workflow(
                    df,
                    "y",
                    "d",
                    "time",
                    "unit",
                    aggregate="event_study",
                    weights=np.ones(n),
                    n_bootstrap=199,
                    seed=0,
                )
            # PR #376 R2 P3 fix: workflow event-study weights= path emits
            # exactly ONE DeprecationWarning (not three — joint wrappers'
            # nested warnings are suppressed since the user-facing one
            # already fired at the workflow's front door).
            n_dep_warnings = sum(1 for w in w_record if issubclass(w.category, DeprecationWarning))
            assert n_dep_warnings == 1, (
                f"expected 1 DeprecationWarning at workflow front door, got " f"{n_dep_warnings}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_new = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                aggregate="event_study",
                survey_design=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )
        assert r_legacy.homogeneity_joint.cvm_stat_joint == r_new.homogeneity_joint.cvm_stat_joint
        assert r_legacy.homogeneity_joint.p_value == r_new.homogeneity_joint.p_value
