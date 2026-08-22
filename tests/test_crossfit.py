"""Tests for unit-level K-fold cross-fitting (PR-B0)."""

import pickle

import numpy as np
import pytest

from diff_diff._crossfit import (
    CrossFitResult,
    DegenerateFoldError,
    FoldAssignment,
    assign_folds,
    cross_fit_predict,
)
from diff_diff._learners import LinearLearner, LogitLearner
from diff_diff.linalg import solve_logit, solve_ols


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestAssignFolds:
    def test_partition_and_balance(self):
        folds = assign_folds(23, 5, rng=_rng())
        assert np.all((folds.fold_ids >= 0) & (folds.fold_ids < 5))
        counts = folds.counts()
        assert counts.sum() == 23
        assert counts.max() - counts.min() <= 1

    def test_deterministic_from_state(self):
        a = assign_folds(50, 4, rng=_rng(3))
        b = assign_folds(50, 4, rng=_rng(3))
        np.testing.assert_array_equal(a.fold_ids, b.fold_ids)

    def test_replay_reproduces_after_extra_draws(self):
        rng = _rng(7)
        folds = assign_folds(40, 4, rng=rng, stratify=np.repeat([0, 1], 20))
        rng.standard_normal(100)  # keep drawing from the live generator
        replayed = folds.replay()
        np.testing.assert_array_equal(replayed.fold_ids, folds.fold_ids)

    def test_replay_immune_to_state_dict_mutation(self):
        folds = assign_folds(30, 3, rng=_rng(13))
        state = folds.bitgen_state
        if isinstance(state.get("state"), dict):  # PCG64 layout
            state["state"]["state"] = 0  # in-place mutation of retrieved state
        replayed = folds.replay()  # reads the construction-time snapshot
        np.testing.assert_array_equal(replayed.fold_ids, folds.fold_ids)

    def test_replay_unknown_bitgen_raises(self):
        folds = assign_folds(10, 2, rng=_rng())
        broken = FoldAssignment(
            n_folds=folds.n_folds,
            n_units=folds.n_units,
            fold_ids=folds.fold_ids,
            bitgen_state=folds.bitgen_state,
            bitgen_name="NotABitGenerator",
        )
        with pytest.raises(ValueError, match="unknown bit generator"):
            broken.replay()

    def test_non_generator_rng_raises(self):
        with pytest.raises(TypeError, match="numpy.random.Generator"):
            assign_folds(10, 2, rng=np.random.RandomState(0))

    def test_picklable(self):
        folds = assign_folds(12, 3, rng=_rng(), stratify=np.repeat([0, 1], 6))
        clone = pickle.loads(pickle.dumps(folds))
        np.testing.assert_array_equal(clone.fold_ids, folds.fold_ids)
        np.testing.assert_array_equal(clone.replay().fold_ids, folds.fold_ids)

    def test_stratified_balance_within_stratum(self):
        strata = np.repeat([0, 1, 2], [9, 7, 11])
        folds = assign_folds(27, 4, rng=_rng(1), stratify=strata)
        for s in (0, 1, 2):
            per_fold = np.bincount(folds.fold_ids[strata == s], minlength=4)
            assert per_fold.max() - per_fold.min() <= 1
        overall = folds.counts()
        assert overall.max() - overall.min() <= 1  # global cursor carries over

    def test_no_empty_fold_multi_small_strata(self):
        # strata sizes 2 + 3 with n_folds=4: per-stratum dealing without a
        # carried cursor would leave a fold empty; the global cursor cannot.
        strata = np.repeat([0, 1], [2, 3])
        folds = assign_folds(5, 4, rng=_rng(2), stratify=strata)
        assert np.all(folds.counts() > 0)

    def test_size_two_stratum_in_every_complement(self):
        strata = np.array([0, 0] + [1] * 8)
        folds = assign_folds(10, 5, rng=_rng(4), stratify=strata)
        for k in range(5):
            assert np.any(strata[folds.train_mask(k)] == 0)

    def test_singleton_stratum_raises(self):
        strata = np.array([0] + [1] * 9)
        with pytest.raises(ValueError, match="stratum 0.*singleton|only 1 member"):
            assign_folds(10, 3, rng=_rng(), stratify=strata)

    def test_cluster_units_share_fold(self):
        clusters = np.repeat(np.arange(6), 4)
        folds = assign_folds(24, 3, rng=_rng(5), cluster_ids=clusters)
        for c in range(6):
            assert np.unique(folds.fold_ids[clusters == c]).shape[0] == 1
        assert folds.counts().sum() == 6  # member counts = clusters

    def test_cluster_stratum_conflict_raises(self):
        clusters = np.repeat([0, 1], 5)
        strata = np.arange(10) % 2  # varies within each cluster
        with pytest.raises(ValueError, match="varies within cluster"):
            assign_folds(10, 2, rng=_rng(), stratify=strata, cluster_ids=clusters)

    def test_n_folds_bounds(self):
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            assign_folds(10, 1, rng=_rng())
        with pytest.raises(ValueError, match="exceeds the number of assignment members"):
            assign_folds(3, 4, rng=_rng())
        clusters = np.repeat([0, 1], 10)
        with pytest.raises(ValueError, match="clusters"):
            assign_folds(20, 3, rng=_rng(), cluster_ids=clusters)

    def test_mixed_type_labels_targeted_error(self):
        mixed = np.array([1, "a"] + [2] * 8, dtype=object)
        with pytest.raises(ValueError, match="non-comparable mixed-type"):
            assign_folds(10, 2, rng=_rng(), cluster_ids=mixed)
        with pytest.raises(ValueError, match="non-comparable mixed-type"):
            assign_folds(10, 2, rng=_rng(), stratify=mixed)

    def test_non_integer_counts_rejected(self):
        with pytest.raises(ValueError, match="n_folds must be an integer"):
            assign_folds(10, 2.5, rng=_rng())
        with pytest.raises(ValueError, match="n_folds must be an integer"):
            assign_folds(10, True, rng=_rng())
        with pytest.raises(ValueError, match="n_units must be an integer"):
            assign_folds(10.0, 2, rng=_rng())

    def test_input_validation(self):
        with pytest.raises(ValueError, match="stratify has length"):
            assign_folds(10, 2, rng=_rng(), stratify=np.zeros(9))
        with pytest.raises(ValueError, match="missing values"):
            assign_folds(10, 2, rng=_rng(), stratify=np.array([np.nan] + [0.0] * 9))
        with pytest.raises(ValueError, match="1-dimensional"):
            assign_folds(10, 2, rng=_rng(), cluster_ids=np.zeros((10, 1)))

    def test_missing_labels_rejected_dtype_independent(self):
        import pandas as pd

        # Object-dtype np.nan cluster ids would silently split one "missing"
        # cluster across folds if they reached np.unique; reject up front.
        obj_nan = np.array(["a", np.nan, "b", np.nan] + ["c"] * 6, dtype=object)
        with pytest.raises(ValueError, match="cluster_ids contains missing"):
            assign_folds(10, 2, rng=_rng(), cluster_ids=obj_nan)
        pd_na = np.array(["a", pd.NA] + ["b"] * 8, dtype=object)
        with pytest.raises(ValueError, match="cluster_ids contains missing"):
            assign_folds(10, 2, rng=_rng(), cluster_ids=pd_na)
        with pytest.raises(ValueError, match="stratify contains missing"):
            assign_folds(10, 2, rng=_rng(), stratify=np.array([None] + [0] * 9, dtype=object))

    def test_fold_assignment_invariants_enforced(self):
        good = assign_folds(10, 2, rng=_rng())
        with pytest.raises(ValueError, match="fold_ids values must lie"):
            FoldAssignment(2, 10, np.array([0, 1, 2] + [0] * 7), {}, "PCG64")
        with pytest.raises(ValueError, match="own no units"):
            FoldAssignment(3, 10, np.array([0, 1] * 5), {}, "PCG64")
        with pytest.raises(ValueError, match="integer array"):
            FoldAssignment(2, 10, np.zeros(10), {}, "PCG64")
        with pytest.raises(ValueError, match="1-dimensional"):
            FoldAssignment(2, 10, np.zeros((10, 1), dtype=np.int64), {}, "PCG64")
        # Arrays are frozen: post-construction mutation is impossible.
        with pytest.raises(ValueError):
            good.fold_ids[0] = 1
        # Cluster cohesion: a hand-built assignment splitting a cluster is
        # rejected (cluster leakage across train/test).
        with pytest.raises(ValueError, match="spans multiple folds"):
            FoldAssignment(
                2,
                4,
                np.array([0, 1, 0, 1]),
                {},
                "PCG64",
                cluster_ids=np.array(["a", "a", "b", "b"], dtype=object),
            )
        with pytest.raises(ValueError, match="cluster_ids must be 1-dimensional"):
            FoldAssignment(2, 4, np.array([0, 1, 0, 1]), {}, "PCG64", cluster_ids=np.zeros(3))

    def test_iter_folds_contract(self):
        folds = assign_folds(9, 3, rng=_rng())
        seen = []
        for k, train, test in folds.iter_folds():
            assert train.dtype.kind == "i" and test.dtype.kind == "i"
            assert np.intersect1d(train, test).size == 0
            assert train.size + test.size == 9
            seen.extend(test.tolist())
        assert sorted(seen) == list(range(9))


def _reg_setup(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y = 1.0 + X @ np.array([1.0, -0.5]) + rng.normal(scale=0.2, size=n)
    return X, y


class TestCrossFitPredict:
    def test_matches_hand_fit_two_fold(self):
        X, y = _reg_setup()
        folds = assign_folds(len(y), 2, rng=_rng(1))
        res = cross_fit_predict(LinearLearner(), X, y, folds)
        for k, train, test in folds.iter_folds():
            Xi = np.column_stack([np.ones(train.size), X[train]])
            coefs, _, _ = solve_ols(Xi, y[train], return_vcov=False)
            hand = np.column_stack([np.ones(test.size), X[test]]) @ coefs
            np.testing.assert_allclose(res.oof_predictions[test], hand, atol=1e-12)

    def test_fold_losses_hand_checked_unweighted(self):
        X, y = _reg_setup()
        folds = assign_folds(len(y), 2, rng=_rng(2))
        res = cross_fit_predict(LinearLearner(), X, y, folds)
        for k in range(2):
            test = folds.test_mask(k)
            mse = np.mean((y[test] - res.oof_predictions[test]) ** 2)
            np.testing.assert_allclose(res.fold_losses[k], mse, atol=1e-14)

    def test_fold_losses_weighted_and_zero_weight_sentinel(self):
        X, y = _reg_setup(n=30)
        folds = assign_folds(len(y), 3, rng=_rng(3))
        w = np.ones(len(y))
        w[folds.test_mask(1)] = 0.0  # fold 1's TEST weight mass is zero
        w[folds.test_mask(2)] *= 2.0
        res = cross_fit_predict(LinearLearner(), X, y, folds, sample_weight=w)
        assert np.isnan(res.fold_losses[1])  # sentinel, no error
        test2 = folds.test_mask(2)
        errs = (y[test2] - res.oof_predictions[test2]) ** 2
        np.testing.assert_allclose(
            res.fold_losses[2], np.sum(w[test2] * errs) / np.sum(w[test2]), atol=1e-14
        )

    def test_predict_proba_happy_path(self):
        rng = np.random.default_rng(9)
        n = 120
        X = rng.standard_normal((n, 2))
        p = 1.0 / (1.0 + np.exp(-(1.2 * X[:, 0])))
        y = (rng.uniform(size=n) < p).astype(float)
        folds = assign_folds(n, 2, rng=_rng(4), stratify=y)
        res = cross_fit_predict(LogitLearner(), X, y, folds, predict_method="predict_proba")
        assert np.all((res.oof_predictions >= 0) & (res.oof_predictions <= 1))
        assert np.isfinite(res.fold_losses).all()
        for k, train, test in folds.iter_folds():
            beta, _ = solve_logit(X[train], y[train])
            eta = np.column_stack([np.ones(test.size), X[test]]) @ beta
            np.testing.assert_allclose(
                res.oof_predictions[test], 1 / (1 + np.exp(-eta)), atol=1e-10
            )

    def test_log_loss_finite_with_saturated_probability(self):
        # A perfectly separable-in-fold DGP can produce fitted probabilities
        # of ~1.0; the loss-only clip keeps log-loss finite.
        rng = np.random.default_rng(12)
        n = 40
        X = np.concatenate([rng.normal(-8, 0.1, n // 2), rng.normal(8, 0.1, n // 2)])
        X = X.reshape(-1, 1)
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        folds = assign_folds(n, 2, rng=_rng(5), stratify=y)
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")  # separation warnings expected
            res = cross_fit_predict(LogitLearner(), X, y, folds, predict_method="predict_proba")
        assert np.isfinite(res.fold_losses).all()
        # oof-unclipped invariant: saturated outputs survive un-clipped.
        assert res.oof_predictions.max() > 1.0 - 1e-9

    def test_oof_unclipped_matches_direct_learner_predictions(self):
        X, y = _reg_setup()
        folds = assign_folds(len(y), 2, rng=_rng(6))
        res = cross_fit_predict(LinearLearner(), X, y, folds)
        for k, train, test in folds.iter_folds():
            direct = LinearLearner().fit(X[train], y[train]).predict(X[test])
            np.testing.assert_array_equal(res.oof_predictions[test], direct)

    def test_sample_weight_forwarded_to_fit(self):
        X, y = _reg_setup()
        folds = assign_folds(len(y), 2, rng=_rng(7))
        w = np.ones(len(y))
        w[:10] = 5.0
        res = cross_fit_predict(LinearLearner(), X, y, folds, sample_weight=w)
        k0_train = np.flatnonzero(folds.train_mask(0))
        Xi = np.column_stack([np.ones(k0_train.size), X[k0_train]])
        coefs, _, _ = solve_ols(Xi, y[k0_train], return_vcov=False, weights=w[k0_train])
        test0 = np.flatnonzero(folds.test_mask(0))
        hand = np.column_stack([np.ones(test0.size), X[test0]]) @ coefs
        np.testing.assert_allclose(res.oof_predictions[test0], hand, atol=1e-12)

    def test_fit_mask_untreated_only_still_predicts_treated(self):
        X, y = _reg_setup()
        treated = np.zeros(len(y), dtype=bool)
        treated[:15] = True
        folds = assign_folds(len(y), 2, rng=_rng(8), stratify=treated)
        res = cross_fit_predict(LinearLearner(), X, y, folds, fit_mask=~treated)
        assert np.isfinite(res.oof_predictions[treated]).all()
        assert res.n_fit_per_fold.sum() == (~treated).sum()

    def test_nonfinite_X_y_rejected(self):
        X, y = _reg_setup(n=20)
        folds = assign_folds(20, 2, rng=_rng(11))
        Xn = X.copy()
        Xn[0, 0] = np.nan
        with pytest.raises(ValueError, match="X contains NaN"):
            cross_fit_predict(LinearLearner(), Xn, y, folds)
        yn = y.copy()
        yn[0] = np.inf
        with pytest.raises(ValueError, match="y contains NaN"):
            cross_fit_predict(LinearLearner(), X, yn, folds)

    def test_stateful_learner_cannot_leak_across_folds(self):
        # Each fold fits a DEEP COPY of the never-fit template, so an
        # accumulating warm-start-style learner cannot carry state (and data
        # from a previous complement) into later folds.
        class AccumulatingMean:
            def __init__(self, bias=0.0):
                self.bias = bias
                self._seen_y = []

            def get_params(self):
                return {"bias": self.bias}

            def fit(self, X, y, sample_weight=None):
                self._seen_y.extend(y.tolist())  # accumulates across fits!
                self.mean_ = float(np.mean(self._seen_y)) + self.bias
                return self

            def predict(self, X):
                return np.full(len(X), self.mean_)

        n = 20
        X = np.zeros((n, 1))
        y = np.zeros(n)
        folds = assign_folds(n, 2, rng=_rng(20))
        y[folds.test_mask(0)] = 1.0  # fold 0's outcomes differ from fold 1's
        res = cross_fit_predict(AccumulatingMean(), X, y, folds)
        # Each fold's prediction equals a FRESH fit on its complement alone.
        for k, train, test in folds.iter_folds():
            expected = float(np.mean(y[train]))
            np.testing.assert_allclose(res.oof_predictions[test], expected)

    def test_composite_learner_isolated_per_fold(self):
        # The per-fold deep copy isolates composites too: the template (and
        # its nested inner learner) is never fit.
        class Composite:
            def __init__(self, inner=None):
                self.inner = inner if inner is not None else LinearLearner()
                self.fit_count = 0

            def get_params(self, deep=True):
                params = {"inner": self.inner}
                if deep:
                    params["inner__nonconstructor"] = 1  # would break **kwargs
                return params

            def fit(self, X, y, sample_weight=None):
                self.fit_count += 1
                self.inner.fit(X, y, sample_weight=sample_weight)
                return self

            def predict(self, X):
                return self.inner.predict(X)

        X, y = _reg_setup()
        folds = assign_folds(len(y), 2, rng=_rng(21))
        template = Composite()
        res = cross_fit_predict(template, X, y, folds)
        assert template.fit_count == 0  # every fold used a fresh reconstruction
        assert np.isfinite(res.oof_predictions).all()

    def test_nested_stateful_learner_cannot_leak(self):
        # A composite whose INNER learner accumulates must also be safe: the
        # per-fold deep copy gives every fold a fresh nested object, not a
        # shared reference.
        class AccumulatingInner:
            def __init__(self):
                self._seen_y = []

            def get_params(self, deep=False):
                return {}

            def fit(self, X, y, sample_weight=None):
                self._seen_y.extend(y.tolist())
                self.mean_ = float(np.mean(self._seen_y))
                return self

            def predict(self, X):
                return np.full(len(X), self.mean_)

        class Outer:
            def __init__(self, inner=None):
                self.inner = inner if inner is not None else AccumulatingInner()

            def get_params(self, deep=False):
                return {"inner": self.inner}

            def fit(self, X, y, sample_weight=None):
                self.inner.fit(X, y, sample_weight=sample_weight)
                return self

            def predict(self, X):
                return self.inner.predict(X)

        n = 20
        X = np.zeros((n, 1))
        y = np.zeros(n)
        folds = assign_folds(n, 2, rng=_rng(23))
        y[folds.test_mask(0)] = 1.0
        template = Outer()
        res = cross_fit_predict(template, X, y, folds)
        assert template.inner._seen_y == []  # template's inner never touched
        for k, train, test in folds.iter_folds():
            np.testing.assert_allclose(res.oof_predictions[test], float(np.mean(y[train])))

    def test_ridge_overflow_becomes_degenerate_fold_error(self):
        from diff_diff._learners import RidgeLearner

        n = 20
        X = np.arange(n * 2, dtype=float).reshape(n, 2)
        y = np.full(n, 1e308)  # finite but overflow-inducing
        folds = assign_folds(n, 2, rng=_rng(24))
        with pytest.raises(DegenerateFoldError, match="learner error in fold"):
            cross_fit_predict(
                RidgeLearner(alpha=1.0),
                X,
                y,
                folds,
                sample_weight=np.full(n, 1e10),
            )

    def test_copy_failure_warns_loudly(self):
        class Undeepcopyable:
            def __deepcopy__(self, memo):
                raise TypeError("cannot deep-copy this learner")

            def fit(self, X, y, sample_weight=None):
                self.mean_ = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), self.mean_)

        X, y = _reg_setup()
        folds = assign_folds(len(y), 2, rng=_rng(22))
        with pytest.warns(UserWarning, match="could not deep-copy"):
            res = cross_fit_predict(Undeepcopyable(), X, y, folds)
        assert np.isfinite(res.oof_predictions).all()

    def test_result_picklable(self):
        X, y = _reg_setup()
        folds = assign_folds(len(y), 2, rng=_rng(10))
        res = cross_fit_predict(LinearLearner(), X, y, folds)
        clone = pickle.loads(pickle.dumps(res))
        np.testing.assert_array_equal(clone.oof_predictions, res.oof_predictions)
        assert isinstance(clone, CrossFitResult)


class TestFoldErrorContract:
    def test_empty_fit_subset_degenerate_fold_error(self):
        X, y = _reg_setup(n=20)
        folds = assign_folds(20, 2, rng=_rng(1))
        mask = folds.test_mask(1)  # fold 0's complement is entirely masked out
        with pytest.raises(DegenerateFoldError, match="fold 0.*fit subset is empty"):
            cross_fit_predict(LinearLearner(), X, y, folds, fit_mask=~mask & folds.test_mask(0))

    def test_zero_positive_weight_fold_exact_type(self):
        X, y = _reg_setup(n=20)
        folds = assign_folds(20, 2, rng=_rng(2))
        w = np.ones(20)
        w[folds.train_mask(0)] = 0.0
        with pytest.raises(DegenerateFoldError, match="zero sample_weight"):
            cross_fit_predict(LinearLearner(), X, y, folds, sample_weight=w)

    def test_single_class_fold_degenerate(self):
        rng = np.random.default_rng(3)
        n = 20
        X = rng.standard_normal((n, 2))
        y = np.zeros(n)
        y[:2] = 1.0
        folds = FoldAssignment(
            n_folds=2,
            n_units=n,
            fold_ids=np.array([1] * 2 + [0] * 9 + [1] * 9),  # both 1s in fold 1
            bitgen_state={},
            bitgen_name="PCG64",
        )
        with pytest.raises(DegenerateFoldError, match="single .*class"):
            cross_fit_predict(LogitLearner(), X, y, folds, predict_method="predict_proba")

    def test_wrapped_learner_error_chained(self):
        # LinearLearner on a fold with n_fit < p+1 trips solve_ols's own
        # "Fewer observations" ValueError -> DegenerateFoldError, chained.
        rng = np.random.default_rng(4)
        n = 12
        X = rng.standard_normal((n, 6))
        y = rng.standard_normal(n)
        folds = assign_folds(n, 2, rng=_rng(3))
        fit_mask = np.zeros(n, dtype=bool)
        fit_mask[np.flatnonzero(folds.train_mask(0))[:3]] = True  # 3 rows < 7 params
        with pytest.raises(DegenerateFoldError, match="learner error in fold") as ei:
            cross_fit_predict(LinearLearner(), X, y, folds, fit_mask=fit_mask)
        assert ei.value.__cause__ is not None
        assert "Fewer observations" in str(ei.value)

    def test_context_label_prefixes_fold_errors(self):
        X, y = _reg_setup(n=20)
        folds = assign_folds(20, 2, rng=_rng(5))
        with pytest.raises(DegenerateFoldError, match="propensity g: fold"):
            cross_fit_predict(
                LinearLearner(),
                X,
                y,
                folds,
                fit_mask=np.zeros(20, dtype=bool),
                context_label="propensity g",
            )

    def test_argument_validation_is_plain_value_error(self):
        X, y = _reg_setup(n=20)
        folds = assign_folds(20, 2, rng=_rng(6))
        cases = [
            dict(fit_mask=np.ones(19, dtype=bool)),
            dict(fit_mask=np.ones(20, dtype=float)),  # non-bool dtype
            dict(sample_weight=np.ones((20, 1))),
            dict(sample_weight=np.full(20, np.inf)),
            dict(sample_weight=-np.ones(20)),
        ]
        for kwargs in cases:
            with pytest.raises(ValueError) as ei:
                cross_fit_predict(LinearLearner(), X, y, folds, **kwargs)
            assert not isinstance(ei.value, DegenerateFoldError), kwargs

    def test_non_binary_y_with_predict_proba_raises(self):
        X, y = _reg_setup(n=20)
        yb = np.zeros(20)
        yb[:10] = 2.0  # {0, 2} encoding must be rejected up front
        folds = assign_folds(20, 2, rng=_rng(7))
        with pytest.raises(ValueError, match="strictly binary 0/1"):
            cross_fit_predict(LogitLearner(), X, yb, folds, predict_method="predict_proba")

    def test_reversed_classes_column_selection(self):
        # A classifier with classes_ == [1, 0] must have its P(y=1) column
        # (column 0 there) selected — not blindly column 1.
        class ReversedClasses:
            classes_ = np.array([1.0, 0.0])

            def fit(self, X, y, sample_weight=None):
                return self

            def predict_proba(self, X):
                p1 = np.full(len(X), 0.8)
                return np.column_stack([p1, 1.0 - p1])  # col 0 = P(y=1)

        rng = np.random.default_rng(15)
        n = 20
        X = rng.standard_normal((n, 2))
        y = (rng.uniform(size=n) < 0.5).astype(float)
        folds = assign_folds(n, 2, rng=_rng(15), stratify=y)
        res = cross_fit_predict(ReversedClasses(), X, y, folds, predict_method="predict_proba")
        np.testing.assert_allclose(res.oof_predictions, 0.8)

    def test_invalid_classes_rejected(self):
        class BadClasses:
            classes_ = np.array([1.0, 2.0])

            def fit(self, X, y, sample_weight=None):
                return self

            def predict_proba(self, X):
                return np.full((len(X), 2), 0.5)

        rng = np.random.default_rng(16)
        n = 20
        X = rng.standard_normal((n, 2))
        y = (rng.uniform(size=n) < 0.5).astype(float)
        folds = assign_folds(n, 2, rng=_rng(16), stratify=y)
        with pytest.raises(DegenerateFoldError, match="classes_ must be exactly"):
            cross_fit_predict(BadClasses(), X, y, folds, predict_method="predict_proba")

    def test_multi_output_regressor_rejected(self):
        # A regressor returning (n, 2) must be rejected (silently taking one
        # column would feed the wrong target as a nuisance).
        class TwoOutputRegressor:
            def fit(self, X, y, sample_weight=None):
                return self

            def predict(self, X):
                return np.ones((len(X), 2))

        X, y = _reg_setup(n=20)
        folds = assign_folds(20, 2, rng=_rng(9))
        with pytest.raises(DegenerateFoldError, match="regressor learner output") as ei:
            cross_fit_predict(TwoOutputRegressor(), X, y, folds)
        assert isinstance(ei.value.__cause__, ValueError)

    def test_invalid_predict_method_raises(self):
        X, y = _reg_setup(n=10)
        folds = assign_folds(10, 2, rng=_rng(8))
        with pytest.raises(ValueError, match="predict_method"):
            cross_fit_predict(LinearLearner(), X, y, folds, predict_method="proba")
