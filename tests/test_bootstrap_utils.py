"""Tests for bootstrap utility edge cases (NaN propagation)."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pytest

from diff_diff.bootstrap_utils import (
    apply_stratum_centering,
    compute_effect_bootstrap_stats,
    compute_effect_bootstrap_stats_batch,
    stratified_bootstrap_indices,
    warn_bootstrap_failure_rate,
)
from diff_diff.survey import ResolvedSurveyDesign


class TestBootstrapStatsNaNPropagation:
    """Regression tests for compute_effect_bootstrap_stats NaN guard."""

    def test_bootstrap_stats_single_valid_sample(self):
        """Single valid sample: ddof=1 produces NaN SE -> all NaN."""
        boot_dist = np.array([1.5])
        with pytest.warns(RuntimeWarning, match="non-finite or zero"):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=1.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_all_nonfinite(self):
        """All non-finite samples: fails 50% validity check -> all NaN."""
        boot_dist = np.array([np.nan, np.nan, np.inf])
        with pytest.warns(RuntimeWarning):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=1.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_identical_values(self):
        """All identical values: se=0 -> all NaN."""
        boot_dist = np.array([2.0] * 100)
        with pytest.warns(RuntimeWarning, match="non-finite or zero"):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=2.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_mostly_valid_but_identical(self):
        """67% valid (passes 50% check) but identical values: se=0 -> all NaN."""
        boot_dist = np.array([2.0, 2.0, np.nan])
        with pytest.warns(RuntimeWarning, match="non-finite or zero"):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=2.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_original_effect_with_finite_boot_dist(self, bad_value):
        """Non-finite original_effect must return all-NaN even with finite boot_dist."""
        boot_dist = np.arange(100.0)
        se, ci, p_value = compute_effect_bootstrap_stats(
            original_effect=bad_value, boot_dist=boot_dist
        )
        assert np.isnan(se)
        assert np.isnan(ci[0]) and np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_normal_case(self):
        """Normal case with varied values: all fields finite."""
        boot_dist = np.arange(100.0)
        se, ci, p_value = compute_effect_bootstrap_stats(original_effect=50.0, boot_dist=boot_dist)
        assert np.isfinite(se)
        assert se > 0
        assert np.isfinite(ci[0])
        assert np.isfinite(ci[1])
        assert ci[0] < ci[1]
        assert np.isfinite(p_value)
        assert 0 < p_value <= 1


class TestBatchBootstrapStatsWarnings:
    """Tests for warning emission in compute_effect_bootstrap_stats_batch."""

    def test_batch_warns_insufficient_valid_samples(self):
        """Batch function should warn when >50% of bootstrap samples are NaN."""
        rng = np.random.default_rng(42)
        n_bootstrap = 100
        n_effects = 3
        # Column 1 has >50% NaN -> should trigger warning
        matrix = rng.normal(size=(n_bootstrap, n_effects))
        matrix[:60, 1] = np.nan  # 60% NaN

        effects = np.array([1.0, 2.0, 3.0])
        with pytest.warns(RuntimeWarning, match="too few valid"):
            ses, ci_lo, ci_hi, pvals = compute_effect_bootstrap_stats_batch(effects, matrix)
        # Effect 1 (index 1) should be NaN
        assert np.isnan(ses[1])
        # Other effects should be finite
        assert np.isfinite(ses[0])
        assert np.isfinite(ses[2])

    def test_batch_warns_zero_se(self):
        """Batch function should warn when bootstrap SE is zero (identical values)."""
        n_bootstrap = 100
        n_effects = 2
        matrix = np.ones((n_bootstrap, n_effects)) * 5.0  # All identical -> SE=0

        effects = np.array([5.0, 5.0])
        with pytest.warns(RuntimeWarning, match="non-finite or zero"):
            ses, ci_lo, ci_hi, pvals = compute_effect_bootstrap_stats_batch(effects, matrix)
        assert np.isnan(ses[0])
        assert np.isnan(ses[1])

    def test_batch_no_warning_for_normal_case(self):
        """Batch function should not warn when all values are normal."""
        rng = np.random.default_rng(42)
        n_bootstrap = 200
        n_effects = 3
        matrix = rng.normal(size=(n_bootstrap, n_effects))
        effects = np.array([0.5, -0.3, 1.0])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ses, ci_lo, ci_hi, pvals = compute_effect_bootstrap_stats_batch(effects, matrix)


class TestWarnBootstrapFailureRate:
    """Proportional failure-rate guard for replicate loops (axis-D)."""

    def test_warns_above_threshold(self):
        """11/200 successes = 94.5% failure rate — must warn."""
        with pytest.warns(UserWarning, match=r"11/200 bootstrap iterations"):
            warn_bootstrap_failure_rate(n_success=11, n_attempted=200, context="test case")

    def test_warning_message_includes_context(self):
        """Context label must appear verbatim in the warning."""
        with pytest.warns(UserWarning, match="TROP global bootstrap") as rec:
            warn_bootstrap_failure_rate(
                n_success=50,
                n_attempted=200,
                context="TROP global bootstrap",
            )
        assert len(rec) == 1
        assert "75.0% failure rate" in str(rec[0].message)

    def test_silent_below_threshold(self):
        """Default threshold=0.05 — 4% failure is below and must not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warn_bootstrap_failure_rate(n_success=960, n_attempted=1000, context="test case")

    def test_silent_on_full_success(self):
        """No warning when every replicate succeeded."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warn_bootstrap_failure_rate(n_success=200, n_attempted=200, context="test case")

    def test_silent_when_n_attempted_zero(self):
        """Degenerate empty call must not divide by zero."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warn_bootstrap_failure_rate(n_success=0, n_attempted=0, context="test case")

    def test_custom_threshold(self):
        """Higher threshold suppresses the 50% case."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warn_bootstrap_failure_rate(
                n_success=100,
                n_attempted=200,
                context="test case",
                threshold=0.75,
            )

        with pytest.warns(UserWarning, match="50.0% failure rate"):
            warn_bootstrap_failure_rate(
                n_success=100,
                n_attempted=200,
                context="test case",
                threshold=0.25,
            )

    def test_all_failed_warns(self):
        """0/N replicates succeeded — caller handles NaN return, but the warning fires."""
        with pytest.warns(UserWarning, match=r"0/50 bootstrap iterations"):
            warn_bootstrap_failure_rate(n_success=0, n_attempted=50, context="test case")


class TestStratifiedBootstrapIndices:
    """Shared stratified-bootstrap index helper used by TROP Rust + Python paths.

    Pinning these invariants matters because both TROP backends now consume
    the helper's output directly; any drift in shape, dtype, or draw order
    would silently break backend parity (silent-failures audit finding #23).
    """

    def test_shapes_and_dtype(self):
        rng = np.random.default_rng(0)
        ctrl, trt = stratified_bootstrap_indices(rng, n_control=5, n_treated=3, n_bootstrap=7)
        assert ctrl.shape == (7, 5)
        assert trt.shape == (7, 3)
        assert ctrl.dtype == np.int64
        assert trt.dtype == np.int64

    def test_value_range(self):
        rng = np.random.default_rng(123)
        ctrl, trt = stratified_bootstrap_indices(rng, n_control=4, n_treated=6, n_bootstrap=50)
        assert ctrl.min() >= 0 and ctrl.max() < 4
        assert trt.min() >= 0 and trt.max() < 6

    def test_determinism(self):
        ctrl_a, trt_a = stratified_bootstrap_indices(np.random.default_rng(42), 3, 2, 5)
        ctrl_b, trt_b = stratified_bootstrap_indices(np.random.default_rng(42), 3, 2, 5)
        np.testing.assert_array_equal(ctrl_a, ctrl_b)
        np.testing.assert_array_equal(trt_a, trt_b)

    def test_prefix_invariance(self):
        """n_bootstrap=N prefix must match first N rows of n_bootstrap=M>N.

        Pins the sequential-per-replicate consumption law: one rng advances
        through all replicates in order, so extending the loop only appends.
        """
        ctrl_short, trt_short = stratified_bootstrap_indices(np.random.default_rng(7), 4, 3, 10)
        ctrl_long, trt_long = stratified_bootstrap_indices(np.random.default_rng(7), 4, 3, 100)
        np.testing.assert_array_equal(ctrl_short, ctrl_long[:10])
        np.testing.assert_array_equal(trt_short, trt_long[:10])

    def test_value_pin_default_rng_42(self):
        """Hard-coded byte-level pin. Catches silent draw-order drift.

        Any refactor that reorders the draws (e.g. treated-then-control,
        vectorized single call, or a new rng primitive) will break this.
        """
        rng = np.random.default_rng(42)
        ctrl, trt = stratified_bootstrap_indices(rng, n_control=3, n_treated=2, n_bootstrap=5)
        expected_ctrl = np.array(
            [[0, 2, 1], [2, 0, 2], [1, 2, 2], [2, 1, 0], [1, 1, 0]],
            dtype=np.int64,
        )
        expected_trt = np.array(
            [[0, 0], [0, 0], [1, 1], [1, 0], [1, 1]],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(ctrl, expected_ctrl)
        np.testing.assert_array_equal(trt, expected_trt)

    def test_empty_control_pool(self):
        rng = np.random.default_rng(1)
        ctrl, trt = stratified_bootstrap_indices(rng, n_control=0, n_treated=3, n_bootstrap=4)
        assert ctrl.shape == (4, 0)
        assert trt.shape == (4, 3)
        assert trt.min() >= 0 and trt.max() < 3

    def test_empty_treated_pool(self):
        rng = np.random.default_rng(1)
        ctrl, trt = stratified_bootstrap_indices(rng, n_control=3, n_treated=0, n_bootstrap=4)
        assert ctrl.shape == (4, 3)
        assert trt.shape == (4, 0)
        assert ctrl.min() >= 0 and ctrl.max() < 3


def _make_resolved(
    *,
    n_obs: int,
    psu: Optional[np.ndarray],
    strata: Optional[np.ndarray],
    fpc: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    lonely_psu: str = "remove",
) -> ResolvedSurveyDesign:
    """Test-local builder for a minimal ResolvedSurveyDesign."""
    if weights is None:
        weights = np.ones(n_obs, dtype=np.float64)
    n_strata = 1 if strata is None else int(len(np.unique(strata)))
    if psu is None:
        n_psu = n_obs
    else:
        n_psu = int(len(np.unique(psu)))
    return ResolvedSurveyDesign(
        weights=weights.astype(np.float64),
        weight_type="pweight",
        strata=None if strata is None else strata.astype(np.int64),
        psu=None if psu is None else psu.astype(np.int64),
        fpc=None if fpc is None else fpc.astype(np.float64),
        n_strata=n_strata,
        n_psu=n_psu,
        lonely_psu=lonely_psu,
    )


def _reference_stratum_centering(
    tensor: np.ndarray,
    resolved_survey: ResolvedSurveyDesign,
    psu_ids: np.ndarray,
) -> np.ndarray:
    """Pre-refactor reference implementation (literal copy of the inline
    block at the original ``had.py:2172-2204``). Used as the ground
    truth for the bit-parity regression test against
    :func:`apply_stratum_centering` at ``psu_axis=0``.

    Mutates ``tensor`` in place. Caller is responsible for axis=0
    layout (n_psu rows × any number of columns)."""
    n_psu = int(tensor.shape[0])
    if resolved_survey.strata is not None:
        strata = np.asarray(resolved_survey.strata)
        psu_stratum = np.empty(n_psu, dtype=strata.dtype)
        psu_id_to_col = {int(p): c for c, p in enumerate(psu_ids)}
        if resolved_survey.psu is not None:
            seen = np.zeros(n_psu, dtype=bool)
            unit_psu = np.asarray(resolved_survey.psu)
            for i in range(len(unit_psu)):
                col = psu_id_to_col[int(unit_psu[i])]
                if not seen[col]:
                    psu_stratum[col] = strata[i]
                    seen[col] = True
        else:
            psu_stratum = strata.copy()

        for h in np.unique(psu_stratum):
            mask_h = psu_stratum == h
            n_h = int(mask_h.sum())
            if n_h < 2:
                continue
            tensor[mask_h] -= tensor[mask_h].mean(axis=0, keepdims=True)
            tensor[mask_h] *= np.sqrt(n_h / (n_h - 1))
    else:
        if n_psu >= 2:
            tensor -= tensor.mean(axis=0, keepdims=True)
            tensor *= np.sqrt(n_psu / (n_psu - 1))
    return tensor


class TestApplyStratumCentering:
    """Unit tests for ``apply_stratum_centering`` — the shared
    within-stratum demean + sqrt(n_h/(n_h-1)) Bessel rescale used by
    both the HAD sup-t event-study bootstrap (``had._sup_t_multiplier_bootstrap``,
    psu_axis=0 on the PSU influence tensor) AND the HAD Stute
    survey-bootstrap family (``stute_test`` / ``stute_joint_pretest``,
    psu_axis=1 on the multiplier matrix).

    Eight cases:
    1. Single-implicit-stratum with n_psu=1 (degenerate, unchanged).
    2. Single-implicit-stratum with n_psu>=2 (Bessel correction applied).
    3. Multiple strata, balanced PSUs per stratum.
    4. Multiple strata, unbalanced PSUs per stratum.
    5. Singleton stratum under lonely_psu='remove' (centering skipped).
    6. Singleton stratum under lonely_psu='certainty' (centering skipped).
    7. FPC pre-baked: helper does NOT re-scale by FPC.
    8. Bit-parity vs the pre-refactor inline implementation at
       psu_axis=0 (regression for the HAD sup-t refactor).
    """

    def test_single_implicit_stratum_n_psu_one_unchanged(self):
        """Degenerate n_psu=1: skip centering (sqrt(1/0) divide-by-zero)."""
        tensor = np.array([[2.0, 3.0]], dtype=np.float64)  # (1, 2)
        original = tensor.copy()
        resolved = _make_resolved(n_obs=1, psu=None, strata=None)
        psu_ids = np.array([0], dtype=np.int64)
        out = apply_stratum_centering(tensor, resolved, psu_ids, psu_axis=0)
        np.testing.assert_array_equal(out, original)

    def test_single_implicit_stratum_n_psu_two_centered_and_rescaled(self):
        """Implicit single-stratum case with n_psu=2: demean across all
        PSUs, rescale by sqrt(2/1) = sqrt(2)."""
        tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)  # (n_psu=2, H=2)
        resolved = _make_resolved(n_obs=2, psu=None, strata=None)
        psu_ids = np.array([0, 1], dtype=np.int64)
        out = apply_stratum_centering(tensor, resolved, psu_ids, psu_axis=0)
        # Means before centering: [2.0, 3.0]; after centering: [[-1, -1], [1, 1]];
        # after sqrt(2) rescale: [[-sqrt(2), -sqrt(2)], [sqrt(2), sqrt(2)]].
        expected = np.array(
            [[-np.sqrt(2), -np.sqrt(2)], [np.sqrt(2), np.sqrt(2)]], dtype=np.float64
        )
        np.testing.assert_allclose(out, expected, atol=1e-14, rtol=1e-14)

    def test_multiple_strata_balanced_psus(self):
        """4 strata, 2 PSUs per stratum. Each within-stratum pair is
        demeaned independently and rescaled by sqrt(2)."""
        # Manually constructed (n_psu=8, H=1) tensor with arbitrary values.
        tensor = np.array([[1.0], [3.0], [10.0], [20.0], [-5.0], [5.0], [0.0], [4.0]])
        psu = np.arange(8)
        strata = np.repeat(np.arange(4), 2)
        resolved = _make_resolved(n_obs=8, psu=psu, strata=strata)
        psu_ids = np.arange(8, dtype=np.int64)
        out = apply_stratum_centering(tensor, resolved, psu_ids, psu_axis=0)
        # Stratum 0: [1, 3] -> demeaned to [-1, 1], scaled by sqrt(2): [-sqrt(2), sqrt(2)].
        # Stratum 1: [10, 20] -> [-5, 5] -> [-5*sqrt(2), 5*sqrt(2)].
        # Stratum 2: [-5, 5] -> [-5, 5] -> [-5*sqrt(2), 5*sqrt(2)].
        # Stratum 3: [0, 4] -> [-2, 2] -> [-2*sqrt(2), 2*sqrt(2)].
        expected = np.array(
            [
                [-np.sqrt(2)],
                [np.sqrt(2)],
                [-5 * np.sqrt(2)],
                [5 * np.sqrt(2)],
                [-5 * np.sqrt(2)],
                [5 * np.sqrt(2)],
                [-2 * np.sqrt(2)],
                [2 * np.sqrt(2)],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(out, expected, atol=1e-14, rtol=1e-14)

    def test_multiple_strata_unbalanced_psus(self):
        """Strata with different n_h apply different Bessel factors."""
        # Stratum 0: 3 PSUs (sqrt(3/2)); Stratum 1: 4 PSUs (sqrt(4/3)).
        tensor = np.array([[1.0], [4.0], [7.0], [10.0], [20.0], [30.0], [40.0]], dtype=np.float64)
        psu = np.arange(7)
        strata = np.array([0, 0, 0, 1, 1, 1, 1])
        resolved = _make_resolved(n_obs=7, psu=psu, strata=strata)
        psu_ids = np.arange(7, dtype=np.int64)
        out = apply_stratum_centering(tensor, resolved, psu_ids, psu_axis=0)
        # Stratum 0: mean=4 -> [-3,0,3] -> *sqrt(3/2).
        # Stratum 1: mean=25 -> [-15,-5,5,15] -> *sqrt(4/3).
        f0 = np.sqrt(3 / 2)
        f1 = np.sqrt(4 / 3)
        expected = np.array(
            [
                [-3 * f0],
                [0.0 * f0],
                [3 * f0],
                [-15 * f1],
                [-5 * f1],
                [5 * f1],
                [15 * f1],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(out, expected, atol=1e-14, rtol=1e-14)

    def test_singleton_stratum_remove_centering_skipped(self):
        """lonely_psu='remove': singleton-stratum entries are left
        unchanged by the helper (caller is responsible for zeroing the
        corresponding multipliers via
        ``generate_survey_multiplier_weights_batch``; the helper just
        avoids the divide-by-zero on sqrt(1/0))."""
        tensor = np.array([[1.0], [3.0], [99.0]], dtype=np.float64)  # singleton at idx 2
        psu = np.arange(3)
        strata = np.array([0, 0, 1])  # Stratum 1 has only 1 PSU
        resolved = _make_resolved(n_obs=3, psu=psu, strata=strata, lonely_psu="remove")
        psu_ids = np.arange(3, dtype=np.int64)
        out = apply_stratum_centering(tensor, resolved, psu_ids, psu_axis=0)
        # Stratum 0: centered + rescaled. Stratum 1 (singleton): skipped, value preserved.
        assert out[2, 0] == 99.0  # Singleton preserved
        np.testing.assert_allclose(out[:2, 0], [-np.sqrt(2), np.sqrt(2)], atol=1e-14)

    def test_singleton_stratum_certainty_centering_skipped(self):
        """lonely_psu='certainty': same skip semantics as 'remove' at
        the helper level (the lonely_psu policy distinction lives in
        ``generate_survey_multiplier_weights_batch``, not here)."""
        tensor = np.array([[2.0], [4.0], [77.0]], dtype=np.float64)
        psu = np.arange(3)
        strata = np.array([0, 0, 1])
        resolved = _make_resolved(n_obs=3, psu=psu, strata=strata, lonely_psu="certainty")
        psu_ids = np.arange(3, dtype=np.int64)
        out = apply_stratum_centering(tensor, resolved, psu_ids, psu_axis=0)
        assert out[2, 0] == 77.0
        np.testing.assert_allclose(out[:2, 0], [-np.sqrt(2), np.sqrt(2)], atol=1e-14)

    def test_fpc_baked_in_helper_is_fpc_agnostic(self):
        """FPC scaling is the responsibility of
        ``generate_survey_multiplier_weights_batch`` (it bakes
        ``sqrt(1 - f_h)`` into the multipliers). The helper here does
        NOT re-scale by FPC — it just applies the demean + Bessel
        rescale, regardless of whether ``resolved_survey.fpc`` is
        populated or not. Locks the contract that the two corrections
        compose multiplicatively (FPC at draw time + Bessel at
        centering time)."""
        # Same input tensor, same strata layout; one resolved survey with
        # fpc=None, one with fpc populated. Helper output is IDENTICAL.
        tensor_a = np.array([[1.0], [3.0], [10.0], [20.0]], dtype=np.float64)
        tensor_b = tensor_a.copy()
        psu = np.arange(4)
        strata = np.repeat(np.arange(2), 2)
        resolved_no_fpc = _make_resolved(n_obs=4, psu=psu, strata=strata, fpc=None)
        resolved_with_fpc = _make_resolved(
            n_obs=4,
            psu=psu,
            strata=strata,
            fpc=np.array([100.0, 100.0, 100.0, 100.0]),
        )
        psu_ids = np.arange(4, dtype=np.int64)
        apply_stratum_centering(tensor_a, resolved_no_fpc, psu_ids, psu_axis=0)
        apply_stratum_centering(tensor_b, resolved_with_fpc, psu_ids, psu_axis=0)
        np.testing.assert_array_equal(tensor_a, tensor_b)

    def test_bit_parity_vs_pre_refactor_inline_block(self):
        """Regression for the HAD sup-t refactor at ``had.py:2264-2317``.
        Locks ``apply_stratum_centering(psu_axis=0)`` bit-exactly against
        the pre-refactor inline implementation on a 200-row × 4-horizon
        fixture spanning balanced + unbalanced + singleton strata under
        ``lonely_psu='remove'``."""
        rng = np.random.default_rng(20260514)
        n_psu = 200
        n_horizons = 4
        tensor_helper = rng.normal(size=(n_psu, n_horizons)).astype(np.float64)
        tensor_reference = tensor_helper.copy()
        # Mix of strata sizes: some balanced, some unbalanced, one singleton.
        strata = np.concatenate(
            [
                np.repeat(0, 60),  # n_h=60
                np.repeat(1, 50),
                np.repeat(2, 40),
                np.repeat(3, 30),
                np.repeat(4, 19),  # unbalanced
                np.repeat(5, 1),  # singleton
            ]
        ).astype(np.int64)
        assert strata.shape == (n_psu,)
        psu = np.arange(n_psu, dtype=np.int64)
        resolved = _make_resolved(n_obs=n_psu, psu=psu, strata=strata, lonely_psu="remove")
        psu_ids = np.arange(n_psu, dtype=np.int64)
        apply_stratum_centering(tensor_helper, resolved, psu_ids, psu_axis=0)
        _reference_stratum_centering(tensor_reference, resolved, psu_ids)
        np.testing.assert_allclose(tensor_helper, tensor_reference, atol=1e-14, rtol=1e-14)


class TestApplyStratumCenteringMultiplierLayout:
    """Verify ``apply_stratum_centering(psu_axis=1)`` produces the same
    algebra as ``psu_axis=0`` after a transpose. Locks that the helper
    is consistent across the two layouts used in production: PSU
    influence tensors (HAD sup-t, axis 0) and PSU multiplier matrices
    (Stute, axis 1)."""

    def test_psu_axis_1_matches_psu_axis_0_after_transpose(self):
        rng = np.random.default_rng(99)
        n_bootstrap = 50
        n_psu = 20
        psu_mults = rng.normal(size=(n_bootstrap, n_psu)).astype(np.float64)
        # Two strata, balanced.
        strata = np.repeat(np.arange(2), 10).astype(np.int64)
        psu = np.arange(n_psu, dtype=np.int64)
        resolved = _make_resolved(n_obs=n_psu, psu=psu, strata=strata)
        psu_ids = np.arange(n_psu, dtype=np.int64)

        # Apply at psu_axis=1 on the (B, n_psu) multiplier matrix.
        out_axis1 = psu_mults.copy()
        apply_stratum_centering(out_axis1, resolved, psu_ids, psu_axis=1)

        # Apply at psu_axis=0 on the transposed (n_psu, B) layout.
        out_axis0 = psu_mults.copy().T
        apply_stratum_centering(out_axis0, resolved, psu_ids, psu_axis=0)
        np.testing.assert_allclose(out_axis1, out_axis0.T, atol=1e-14, rtol=1e-14)

    def test_psu_axis_1_strata_none_implicit_single_stratum(self):
        """Multiplier-layout (psu_axis=1) strata-None path applies the
        single-implicit-stratum correction across all PSUs in each
        replicate row."""
        rng = np.random.default_rng(123)
        n_bootstrap = 30
        n_psu = 10
        psu_mults = rng.normal(size=(n_bootstrap, n_psu)).astype(np.float64)
        resolved = _make_resolved(n_obs=n_psu, psu=None, strata=None)
        psu_ids = np.arange(n_psu, dtype=np.int64)
        out = psu_mults.copy()
        apply_stratum_centering(out, resolved, psu_ids, psu_axis=1)
        # Each row should have within-row mean = 0 and per-row sum-of-squares
        # rescaled by n_psu/(n_psu-1).
        np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-14)
        # Variance scaling: sum of squares should equal original * n/(n-1).
        original_centered = psu_mults - psu_mults.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(
            out, original_centered * np.sqrt(n_psu / (n_psu - 1)), atol=1e-14
        )
