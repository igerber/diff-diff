"""Unit tests for the entropy-balancing solver (diff_diff/balancing.py).

Entropy balancing (Hainmueller 2012) must achieve exact first-moment balance to the
target whenever the target is in the interior of the control covariate hull, and must
fail loudly (BalanceError) when it is not.
"""

import numpy as np
import pytest

from diff_diff.balancing import BalanceError, entropy_balance


def _treated_control(seed, n_control=200, n_treated=80, k=3, shift=0.6):
    """Two groups whose covariate means differ (treated shifted), so balancing is
    non-trivial but feasible (treated mean stays inside the control spread)."""
    rng = np.random.default_rng(seed)
    Xc = rng.normal(0.0, 1.0, size=(n_control, k))
    Xt = rng.normal(shift, 1.0, size=(n_treated, k))
    return Xc, Xt


class TestEntropyBalance:
    def test_exact_first_moment_balance(self):
        Xc, Xt = _treated_control(seed=1)
        target = Xt.mean(axis=0)
        # before balancing the control means differ from the target
        assert np.max(np.abs(Xc.mean(axis=0) - target)) > 0.1
        w, info = entropy_balance(Xc, target, tol=1e-10)
        assert info["converged"]
        # weighted control means match the treated means to tolerance
        np.testing.assert_allclose(w @ Xc, target, atol=1e-9)
        # weights are a valid nonnegative distribution
        assert np.all(w >= 0)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert info["max_residual"] < 1e-10
        assert 0 < info["ess"] <= Xc.shape[0]

    def test_trivial_target_is_near_uniform(self):
        Xc, _ = _treated_control(seed=2)
        target = Xc.mean(axis=0)  # already balanced
        w, info = entropy_balance(Xc, target, tol=1e-12)
        assert info["converged"]
        np.testing.assert_allclose(w, np.full(Xc.shape[0], 1.0 / Xc.shape[0]), atol=1e-8)

    def test_base_weights_respected(self):
        Xc, Xt = _treated_control(seed=3)
        target = Xt.mean(axis=0)
        rng = np.random.default_rng(99)
        q = rng.uniform(0.5, 2.0, size=Xc.shape[0])
        w, info = entropy_balance(Xc, target, base_weights=q, tol=1e-10)
        assert info["converged"]
        np.testing.assert_allclose(w @ Xc, target, atol=1e-9)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)

    def test_collinear_covariates_ridge(self):
        # duplicate a column -> singular Hessian; ridge/lstsq must still balance the
        # identified moments without blowing up.
        Xc, Xt = _treated_control(seed=4, k=2)
        Xc = np.column_stack([Xc, Xc[:, 0]])  # 3rd col == 1st
        Xt = np.column_stack([Xt, Xt[:, 0]])
        target = Xt.mean(axis=0)
        w, info = entropy_balance(Xc, target, tol=1e-8)
        assert info["converged"]
        np.testing.assert_allclose(w @ Xc, target, atol=1e-7)

    def test_infeasible_target_raises(self):
        Xc, _ = _treated_control(seed=5, k=2)
        # target far outside the control hull on covariate 0 -> no finite balancing λ
        target = np.array([Xc[:, 0].max() + 5.0, 0.0])
        with pytest.raises(BalanceError) as exc:
            entropy_balance(Xc, target, tol=1e-8, max_iter=100)
        assert exc.value.max_residual >= 1e-8
        assert exc.value.residuals.shape == (2,)

    def test_input_validation(self):
        Xc, _ = _treated_control(seed=6, k=3)
        with pytest.raises(ValueError):
            entropy_balance(Xc, np.zeros(2))  # wrong target length
        with pytest.raises(ValueError):
            entropy_balance(Xc, np.zeros(3), base_weights=np.ones(5))  # wrong q length
        with pytest.raises(ValueError):
            entropy_balance(Xc[:, 0], np.zeros(3))  # 1-D X
