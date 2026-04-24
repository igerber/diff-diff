"""Tests for bootstrap utility edge cases (NaN propagation)."""

import warnings

import numpy as np
import pytest

from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats,
    compute_effect_bootstrap_stats_batch,
    generate_bootstrap_weights_batch,
    generate_bootstrap_weights_batch_numpy,
    stratified_bootstrap_indices,
    warn_bootstrap_failure_rate,
)


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


class TestBootstrapWeightsBatchDistributions:
    """Distributional coverage for the numpy multiplier-weight helper.

    Ported from the deleted Rust-backend tests when the Rust
    ``generate_bootstrap_weights_batch`` binding was removed (silent-failures
    audit follow-up). The numpy helper is now the sole code path, so these
    tests cover shape, support, and statistical moments for each of the
    Rademacher, Mammen, and Webb multiplier distributions.
    """

    def test_alias_is_same_function(self):
        """``generate_bootstrap_weights_batch_numpy`` is a back-compat alias."""
        assert generate_bootstrap_weights_batch_numpy is generate_bootstrap_weights_batch

    def test_invalid_weight_type_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="weight_type must be"):
            generate_bootstrap_weights_batch(5, 3, "unknown", rng)

    def test_rademacher_shape_and_support(self):
        rng = np.random.default_rng(42)
        weights = generate_bootstrap_weights_batch(100, 50, "rademacher", rng)
        assert weights.shape == (100, 50)
        assert set(np.unique(weights)) == {-1.0, 1.0}

    def test_rademacher_mean_and_variance(self):
        rng = np.random.default_rng(42)
        weights = generate_bootstrap_weights_batch(10000, 100, "rademacher", rng)
        assert abs(weights.mean()) < 0.02, f"mean={weights.mean()}"
        assert abs(weights.var() - 1.0) < 0.02, f"var={weights.var()}"

    def test_mammen_moments(self):
        rng = np.random.default_rng(42)
        weights = generate_bootstrap_weights_batch(10000, 100, "mammen", rng)
        # Mammen: E[w]=0, E[w^2]=1, E[w^3]=1
        assert abs(weights.mean()) < 0.02, f"E[w]={weights.mean()}"
        assert abs((weights**2).mean() - 1.0) < 0.02, f"E[w^2]={(weights ** 2).mean()}"
        assert abs((weights**3).mean() - 1.0) < 0.1, f"E[w^3]={(weights ** 3).mean()}"
        # Support is exactly two values
        unique = np.unique(weights)
        assert len(unique) == 2, f"Mammen has two-point support, got {len(unique)}"

    def test_webb_support_and_moments(self):
        rng = np.random.default_rng(42)
        weights = generate_bootstrap_weights_batch(10000, 100, "webb", rng)
        # Webb: six-point support, E[w]=0, Var[w]=1
        unique = np.unique(weights)
        assert len(unique) == 6, f"Webb has six-point support, got {len(unique)}"
        assert abs(weights.mean()) < 0.05, f"E[w]={weights.mean()}"
        assert abs(weights.var() - 1.0) < 0.05, f"Var[w]={weights.var()}"

    def test_reproducibility_same_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        w1 = generate_bootstrap_weights_batch(100, 50, "rademacher", rng1)
        w2 = generate_bootstrap_weights_batch(100, 50, "rademacher", rng2)
        np.testing.assert_array_equal(w1, w2)

    def test_different_seeds_differ(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        w1 = generate_bootstrap_weights_batch(100, 50, "rademacher", rng1)
        w2 = generate_bootstrap_weights_batch(100, 50, "rademacher", rng2)
        assert not np.array_equal(w1, w2)

    def test_value_pin_default_rng_42_rademacher(self):
        """Byte-level pin of ``default_rng(42)`` Rademacher output.

        Catches silent numpy-RNG stream drift across numpy versions. Mirror
        of the ``TestStratifiedBootstrapIndices::test_value_pin_default_rng_42``
        pattern.
        """
        rng = np.random.default_rng(42)
        w = generate_bootstrap_weights_batch(5, 3, "rademacher", rng)
        expected = np.array(
            [
                [-1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(w, expected)

    def test_value_pin_default_rng_42_mammen(self):
        """Byte-level pin of ``default_rng(42)`` Mammen output."""
        rng = np.random.default_rng(42)
        w = generate_bootstrap_weights_batch(5, 3, "mammen", rng)
        sqrt5 = np.sqrt(5.0)
        neg = -(sqrt5 - 1.0) / 2.0
        pos = (sqrt5 + 1.0) / 2.0
        expected = np.array(
            [
                [pos, neg, pos],
                [neg, neg, pos],
                [pos, pos, neg],
                [neg, neg, pos],
                [neg, pos, neg],
            ]
        )
        np.testing.assert_allclose(w, expected, atol=1e-14, rtol=1e-14)

    def test_value_pin_default_rng_42_webb(self):
        """Byte-level pin of ``default_rng(42)`` Webb output."""
        rng = np.random.default_rng(42)
        w = generate_bootstrap_weights_batch(5, 3, "webb", rng)
        v3 = np.sqrt(3.0 / 2.0)
        v1 = np.sqrt(1.0 / 2.0)
        expected = np.array(
            [
                [-v3, 1.0, v1],
                [-v1, -v1, v3],
                [-v3, 1.0, -1.0],
                [-v3, v1, v3],
                [1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_allclose(w, expected, atol=1e-14, rtol=1e-14)
