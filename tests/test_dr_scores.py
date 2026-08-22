"""Oracle tests pinning the DR-score relocation (PR-B0).

ORACLE PROVENANCE
-----------------
Captured from the UNMODIFIED tree at commit 4bd5d6b1 (the commit before the
``_dr_scores.py`` lift) with::

    DIFF_DIFF_BACKEND=python python tests/_capture_dml_b0_dr_oracles.py

on a clean working tree (macOS arm64, python backend forced). The literals
below pin the numeric behavior of ``ContinuousDiD._dr_cell_inf_func`` and the
``ContinuousDiD`` DR covariate path across the relocation of the influence
function into ``diff_diff/_dr_scores.py`` — the same-process parity between
two callers of moved code is blind to the move, so the pinned values are the
relocation gate. Regenerate ONLY from a tree without the lift.

Tolerances: function tier atol=1e-10, estimator tier atol=1e-8 (the CI matrix
spans four platforms / BLAS builds; do not tighten to the capture platform's
precision).
"""

import os

import numpy as np
import pytest

from tests._capture_dml_b0_dr_oracles import (
    estimator_tier_fit,
    function_tier_inputs,
)

# The oracles are python-backend literals; the rust leg agrees only to ~1e-8
# and runs the whole suite under DIFF_DIFF_BACKEND=rust (module-local guard,
# same pattern as tests/test_v4_merge_ddd.py).
_BACKEND_IS_PYTHON = os.environ.get("DIFF_DIFF_BACKEND", "").lower() == "python"
requires_python_backend = pytest.mark.skipif(
    not _BACKEND_IS_PYTHON,
    reason="oracle literals captured under DIFF_DIFF_BACKEND=python",
)

FUNCTION_TIER_IF = np.array(
    [
        -1.3770806502333335,
        0.10922482099945141,
        0.43694261678454366,
        -0.9269214384042687,
        0.7731898959810042,
        1.3440297602752902,
        0.6198773966586693,
        -0.43364134433221746,
        2.238379349286304,
        -0.37762629548315796,
        -0.6711431829669507,
        -1.6145482835756453,
        0.3782937194507452,
        0.8142930333269399,
        -0.7355205254870829,
        0.2533172541623486,
    ]
)

ESTIMATOR_TIER_OVERALL_ATT = 4.459287706210213
ESTIMATOR_TIER_OVERALL_ATT_SE = 0.21458964510376213
ESTIMATOR_TIER_OVERALL_ACRT = 1.9764778160087522
ESTIMATOR_TIER_OVERALL_ACRT_SE = 0.131747670707081


def _dr_inf_func(dY, D, X, gamma, ps):
    """The function under oracle: the DRDID panel influence function.

    Commit 1 targeted the pre-lift ``ContinuousDiD._dr_cell_inf_func`` method
    (captured at 4bd5d6b1); this now targets the relocated
    ``_dr_scores.drdid_panel_inf_func`` with the SAME literals.
    """
    from diff_diff._dr_scores import drdid_panel_inf_func

    return drdid_panel_inf_func(dY, D, X, gamma, ps)


class TestDrScoresOracles:
    @requires_python_backend
    def test_inf_func_matches_committed_oracle(self):
        dY, D, X, gamma, ps = function_tier_inputs()
        inf = _dr_inf_func(dY, D, X, gamma, ps)
        np.testing.assert_allclose(inf, FUNCTION_TIER_IF, atol=1e-10, rtol=0)

    @requires_python_backend
    def test_continuous_did_dr_matches_committed_oracle(self):
        res = estimator_tier_fit()
        np.testing.assert_allclose(res.overall_att, ESTIMATOR_TIER_OVERALL_ATT, atol=1e-8, rtol=0)
        np.testing.assert_allclose(
            res.overall_att_se, ESTIMATOR_TIER_OVERALL_ATT_SE, atol=1e-8, rtol=0
        )
        np.testing.assert_allclose(res.overall_acrt, ESTIMATOR_TIER_OVERALL_ACRT, atol=1e-8, rtol=0)
        np.testing.assert_allclose(
            res.overall_acrt_se, ESTIMATOR_TIER_OVERALL_ACRT_SE, atol=1e-8, rtol=0
        )

    def test_inf_func_ordering_contract(self):
        """Consistently permuting all inputs permutes the output identically."""
        dY, D, X, gamma, ps = function_tier_inputs()
        base = _dr_inf_func(dY, D, X, gamma, ps)
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(D))
        permuted = _dr_inf_func(dY[perm], D[perm], X[perm], gamma, ps[perm])
        np.testing.assert_allclose(permuted, base[perm], atol=1e-12, rtol=0)


class TestChangScoreValidation:
    """Both sides of every stated domain raise targeted errors."""

    def _inputs(self, n=20):
        rng = np.random.default_rng(3)
        D = (rng.uniform(size=n) < 0.5).astype(float)
        dY = rng.normal(size=n)
        m_hat = rng.normal(size=n)
        ps = np.clip(rng.uniform(size=n), 0.1, 0.9)
        return dY, D, m_hat, ps

    def test_ps_upper_bound_raises(self):
        from diff_diff._dr_scores import chang_panel_score

        dY, D, m_hat, ps = self._inputs()
        ps[0] = 1.0
        with pytest.raises(ValueError, match=r"ps must lie in \[0, 1\)"):
            chang_panel_score(dY, D, m_hat, ps, 0.5)

    def test_ps_lower_bound_raises(self):
        from diff_diff._dr_scores import chang_panel_score

        dY, D, m_hat, ps = self._inputs()
        ps[0] = -0.01
        with pytest.raises(ValueError, match=r"ps must lie in \[0, 1\)"):
            chang_panel_score(dY, D, m_hat, ps, 0.5)

    def test_drdid_inf_func_validation(self):
        from diff_diff._dr_scores import drdid_panel_inf_func

        dY, D, X, gamma, ps = function_tier_inputs()
        with pytest.raises(ValueError, match="empty"):
            drdid_panel_inf_func(np.empty(0), np.empty(0), np.empty((0, 3)), gamma, np.empty(0))
        with pytest.raises(ValueError, match="both treated and control"):
            drdid_panel_inf_func(dY, np.zeros_like(D), X, gamma, ps)
        with pytest.raises(ValueError, match="columns but gamma"):
            drdid_panel_inf_func(dY, D, X, gamma[:-1], ps)
        with pytest.raises(ValueError, match="row counts"):
            drdid_panel_inf_func(dY, D[:-1], X, gamma, ps)
        bad_ps = ps.copy()
        bad_ps[0] = 1.0
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            drdid_panel_inf_func(dY, D, X, gamma, bad_ps)
        bad_dY = dY.copy()
        bad_dY[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            drdid_panel_inf_func(bad_dY, D, X, gamma, ps)

    def test_drdid_zero_comparison_mass_raises(self):
        from diff_diff._dr_scores import drdid_panel_inf_func

        dY, D, X, gamma, ps = function_tier_inputs()
        with pytest.raises(ValueError, match="control-odds mass is zero"):
            drdid_panel_inf_func(dY, D, X, gamma, np.zeros_like(ps))

    def test_scalar_input_targeted_error(self):
        from diff_diff._dr_scores import chang_panel_score

        with pytest.raises(ValueError, match="1-dimensional"):
            chang_panel_score(1.0, 1.0, 1.0, 0.5, 0.5)

    def test_empty_inputs_raise(self):
        from diff_diff._dr_scores import chang_panel_score, chang_panel_score_augmented

        empty = np.empty(0)
        with pytest.raises(ValueError, match="empty"):
            chang_panel_score(empty, empty, empty, empty, 0.5)
        with pytest.raises(ValueError, match="empty"):
            chang_panel_score_augmented(empty, empty, 1.0, 0.5)

    @pytest.mark.parametrize("p_hat", [0.0, -0.2, 1.0, 1.5, np.nan])
    def test_p_hat_domain_raises(self, p_hat):
        from diff_diff._dr_scores import chang_panel_score

        dY, D, m_hat, ps = self._inputs()
        with pytest.raises(ValueError, match="p_hat"):
            chang_panel_score(dY, D, m_hat, ps, p_hat)

    def test_non_binary_D_raises(self):
        from diff_diff._dr_scores import chang_panel_score

        dY, D, m_hat, ps = self._inputs()
        D[0] = 2.0
        with pytest.raises(ValueError, match="strictly binary"):
            chang_panel_score(dY, D, m_hat, ps, 0.5)

    def test_shape_mismatch_raises(self):
        from diff_diff._dr_scores import chang_panel_score

        dY, D, m_hat, ps = self._inputs()
        with pytest.raises(ValueError, match="length"):
            chang_panel_score(dY, D, m_hat[:-1], ps, 0.5)

    def test_nonfinite_input_raises(self):
        from diff_diff._dr_scores import chang_panel_score

        dY, D, m_hat, ps = self._inputs()
        dY[0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            chang_panel_score(dY, D, m_hat, ps, 0.5)

    def test_augmented_validation(self):
        from diff_diff._dr_scores import chang_panel_score_augmented

        summand = np.zeros(4)
        D = np.array([0.0, 1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="p_hat"):
            chang_panel_score_augmented(summand, D, 1.0, 0.0)
        with pytest.raises(ValueError, match="p_hat"):
            chang_panel_score_augmented(summand, D, 1.0, 1.5)
        with pytest.raises(ValueError, match="theta"):
            chang_panel_score_augmented(summand, D, np.nan, 0.5)
        with pytest.raises(ValueError, match="strictly binary"):
            chang_panel_score_augmented(summand, np.array([0.0, 2.0, 0.0, 1.0]), 1.0, 0.5)
        with pytest.raises(ValueError, match="length"):
            chang_panel_score_augmented(summand[:-1], D, 1.0, 0.5)
