"""Tests for the duck-typed learner protocol and native learners (PR-B0).

No sklearn import anywhere in this file — the duck-typed acceptance test uses
a tiny local class, proving the protocol carries no dependency.
"""

import pickle
import warnings

import numpy as np
import pytest

from diff_diff._learners import (
    LinearLearner,
    LogitLearner,
    RidgeLearner,
    _validate_predictions,
    make_learner,
    validate_learner,
)
from diff_diff.linalg import solve_logit, solve_ols, solve_ridge


def _reg_data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = 1.0 + X @ np.array([0.5, -1.0, 2.0]) + rng.normal(scale=0.2, size=n)
    return X, y


def _clf_data(n=200, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    p = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))
    y = (rng.uniform(size=n) < p).astype(float)
    return X, y


class TestMakeLearner:
    def test_string_enum_roundtrip(self):
        assert isinstance(make_learner("linear", kind="regressor"), LinearLearner)
        assert isinstance(make_learner("ridge", kind="regressor"), RidgeLearner)
        assert isinstance(make_learner("logit", kind="classifier"), LogitLearner)

    def test_unknown_name_raises_listing_valid(self):
        with pytest.raises(ValueError, match=r"valid names.*linear.*ridge"):
            make_learner("forest", kind="regressor")

    def test_wrong_kind_pairing_raises(self):
        with pytest.raises(ValueError, match=r"valid names.*logit"):
            make_learner("ridge", kind="classifier")
        with pytest.raises(ValueError, match=r"valid names.*linear"):
            make_learner("logit", kind="regressor")

    def test_options_forwarded(self):
        learner = make_learner("ridge", kind="regressor", alpha=2.0)
        assert learner.alpha == 2.0

    def test_duck_typed_object_accepted(self):
        class TinyMeanRegressor:
            def fit(self, X, y, sample_weight=None):
                self.mean_ = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), self.mean_)

        obj = TinyMeanRegressor()
        assert make_learner(obj, kind="regressor") is obj


class TestValidateLearner:
    def test_missing_fit_raises(self):
        class NoFit:
            def predict(self, X):
                return X

        with pytest.raises(TypeError, match="missing the required method 'fit'"):
            validate_learner(NoFit(), kind="regressor", param_name="outcome_learner")

    def test_missing_predict_proba_raises(self):
        class NoProba:
            def fit(self, X, y, sample_weight=None):
                return self

        with pytest.raises(TypeError, match="'predict_proba'"):
            validate_learner(NoProba(), kind="classifier", param_name="propensity_learner")

    def test_uncallable_attr_raises(self):
        class BadFit:
            fit = "not callable"

            def predict(self, X):
                return X

        with pytest.raises(TypeError, match="not callable"):
            validate_learner(BadFit(), kind="regressor", param_name="outcome_learner")


class TestValidatePredictions:
    def test_shape_column_extraction(self):
        two_col = np.column_stack([np.full(5, 0.3), np.full(5, 0.7)])
        out = _validate_predictions(two_col, 5, kind="classifier", context="t")
        np.testing.assert_allclose(out, 0.7)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            _validate_predictions(np.ones((5, 3)), 5, kind="regressor", context="t")

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            _validate_predictions(np.ones(4), 5, kind="regressor", context="t")

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            _validate_predictions(np.array([1.0, np.nan]), 2, kind="regressor", context="t")

    def test_probability_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _validate_predictions(np.array([0.5, 1.2]), 2, kind="classifier", context="t")

    def test_no_clipping(self):
        out = _validate_predictions(np.array([0.0, 1.0]), 2, kind="classifier", context="t")
        np.testing.assert_array_equal(out, [0.0, 1.0])


class TestNativeLearnersMatchSolvers:
    def test_linear_matches_solve_ols(self):
        X, y = _reg_data()
        pred = LinearLearner().fit(X, y).predict(X)
        Xi = np.column_stack([np.ones(len(y)), X])
        coefs, _, _ = solve_ols(Xi, y, return_vcov=False)
        np.testing.assert_allclose(pred, Xi @ coefs, atol=1e-12)

    def test_ridge_matches_solve_ridge(self):
        X, y = _reg_data()
        learner = RidgeLearner(alpha=2.0).fit(X, y)
        coefs = solve_ridge(X, y, alpha=2.0)
        np.testing.assert_allclose(learner.intercept_, coefs[0], atol=1e-12)
        np.testing.assert_allclose(learner.coef_, coefs[1:], atol=1e-12)

    def test_logit_matches_solve_logit_proba(self):
        X, y = _clf_data()
        proba = LogitLearner().fit(X, y).predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
        _, ps = solve_logit(X, y)
        np.testing.assert_allclose(proba[:, 1], ps, atol=1e-10)

    def test_logit_predict_proba_overflow_stable(self):
        # Extreme linear predictors (|eta| ~ 1000) must not emit overflow
        # warnings; probabilities saturate cleanly at 0/1.
        X, y = _clf_data()
        learner = LogitLearner().fit(X, y)
        X_extreme = np.array([[1000.0, 0.0], [-1000.0, 0.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any overflow warning fails the test
            proba = learner.predict_proba(X_extreme)
        assert np.all(np.isfinite(proba))
        assert np.all((proba >= 0) & (proba <= 1))

    def test_sample_weight_forwarded(self):
        X, y = _reg_data(n=30)
        w = np.ones(len(y))
        w[:10] = 3.0
        pred_w = LinearLearner().fit(X, y, sample_weight=w).predict(X)
        Xi = np.column_stack([np.ones(len(y)), X])
        coefs, _, _ = solve_ols(Xi, y, return_vcov=False, weights=w)
        np.testing.assert_allclose(pred_w, Xi @ coefs, atol=1e-12)


class TestLearnersInterceptContract:
    def test_regressors_y_shift_moves_only_intercept(self):
        X, y = _reg_data()
        for cls in (LinearLearner, lambda: RidgeLearner(alpha=1.0)):
            base = cls().fit(X, y)
            shifted = cls().fit(X, y + 50.0)
            np.testing.assert_allclose(shifted.intercept_, base.intercept_ + 50.0, atol=1e-7)
            np.testing.assert_allclose(shifted.coef_, base.coef_, atol=1e-8)

    def test_logit_covariate_shift_translation_invariance(self):
        X, y = _clf_data()
        Xs = X.copy()
        Xs[:, 0] += 5.0
        base = LogitLearner().fit(X, y).predict_proba(X)
        shifted = LogitLearner().fit(Xs, y).predict_proba(Xs)
        np.testing.assert_allclose(shifted, base, atol=1e-7)


class TestRankDeficientPrediction:
    def test_linear_learner_fails_closed_on_rank_deficiency(self):
        # An unpenalized rank-deficient fit has no well-defined out-of-sample
        # prediction (the retained collinear column can impersonate the
        # intercept in-fold and produce arbitrary held-out variation).
        X, y = _reg_data(n=40)
        Xc = np.column_stack([X, np.full(len(y), 3.0)])  # constant column
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # solver's rank warning
            with pytest.raises(ValueError, match="rank-deficient"):
                LinearLearner().fit(Xc, y)

    def test_logit_learner_fails_closed_on_rank_deficiency(self):
        X, y = _clf_data()
        Xc = np.column_stack([X, np.full(len(y), 3.0)])  # constant column
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="rank-deficient"):
                LogitLearner().fit(Xc, y)

    def test_ridge_learner_identified_columns_prediction(self):
        # RidgeLearner stays well-posed: solve_ridge preserves the intercept
        # structurally and NaN-drops only zero-variance columns.
        X, y = _reg_data(n=40)
        Xc = np.column_stack([X, np.full(len(y), 3.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner = RidgeLearner(alpha=1.0).fit(Xc, y)
        pred = learner.predict(Xc)
        assert np.isfinite(pred).all()
        assert np.isnan(learner.coef_).any()
        assert np.isfinite(learner.intercept_)


class TestFitResetAndPickle:
    def test_fit_fully_reinitializes(self):
        X, y = _reg_data()
        learner = LinearLearner().fit(X, y + 100.0)
        learner.fit(X, y)  # second fit must fully replace fitted state
        fresh = LinearLearner().fit(X, y)
        np.testing.assert_allclose(learner.predict(X), fresh.predict(X), atol=1e-12)

    def test_native_learners_picklable(self):
        X, y = _reg_data()
        Xc, yc = _clf_data()
        for learner, Xp in (
            (LinearLearner().fit(X, y), X),
            (RidgeLearner(alpha=1.0).fit(X, y), X),
            (LogitLearner().fit(Xc, yc), Xc),
        ):
            clone = pickle.loads(pickle.dumps(learner))
            if hasattr(learner, "predict"):
                np.testing.assert_allclose(clone.predict(Xp), learner.predict(Xp))
            else:
                np.testing.assert_allclose(clone.predict_proba(Xp), learner.predict_proba(Xp))

    def test_unfitted_predict_raises(self):
        with pytest.raises(ValueError, match="not fitted"):
            LinearLearner().predict(np.ones((2, 2)))
