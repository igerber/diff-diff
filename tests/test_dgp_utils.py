"""Bit-identity and unit tests for synthetic DGP factories in tests/_dgp_utils.py.

These tests guard against silent drift in the Butts (2021) DGP factories used as
identification anchors throughout the spillover-DiD test suite. Even a single
floating-point change in the DGP would shift downstream MC-tolerance pass/fail
boundaries in test_spillover.py.

The bit-identity hashes were captured against the staggered/non-staggered DGP
factories as of the spillover-conley Wave B merge (PR #446, commit 71f84d0e).
The Wave C event-study extension adds optional callable kwargs to
``generate_butts_staggered_dgp`` (``tau_per_event_time``,
``delta_per_ring_per_event_time``); when both default to ``None`` the output
must remain bit-identical to the Wave B baseline.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tests._dgp_utils import (
    generate_butts_nonstaggered_dgp,
    generate_butts_staggered_dgp,
)

_STAGGERED_BASELINE_HASHES = {
    0: "aa6e5e8f28668423a8f09c3e50a9554bd546bcc5e323de371ad82c41cbc3037f",
    42: "f73726161914acd4b5bf50e0cb9a848479cdf54e7da53a457bd06caea7af9b2a",
    100: "d38dd8cce14733cb833bd1da873be19d1c82ffcd83ab41a0c592e297a182006b",
}

_NONSTAGGERED_BASELINE_HASHES = {
    0: "4fbf9240ebc69e02021af6549d8fe917d866cbac6b79e94b05fa55f7fb751cce",
    42: "ff30a9ea0f3af253ce0bb620ed85338351f4cc6647d8865c60403f61715fe3e7",
    100: "3173ad3eeb5c6194b51314dc053574bd8701e2c236fec92a63696c3e57132700",
}


def _sha256_of_y(df) -> str:
    y = np.asarray(df["y"].values, dtype=np.float64)
    return hashlib.sha256(y.tobytes()).hexdigest()


class TestButtsStaggeredDgpBitIdentity:
    """Verify ``generate_butts_staggered_dgp`` outputs remain bit-identical.

    Captured at Wave B (PR #446). Any Wave C+ refactor that touches the DGP
    must preserve this baseline with the new kwargs defaulting to ``None``.
    """

    @pytest.mark.parametrize("seed", [0, 42, 100])
    def test_y_hash_matches_baseline(self, seed: int) -> None:
        df = generate_butts_staggered_dgp(seed=seed)
        assert _sha256_of_y(df) == _STAGGERED_BASELINE_HASHES[seed], (
            f"generate_butts_staggered_dgp(seed={seed}) output differs from "
            "pinned baseline. If this is intentional (e.g. DGP semantics "
            "changed), update _STAGGERED_BASELINE_HASHES."
        )

    def test_row_count_stable(self) -> None:
        df = generate_butts_staggered_dgp(seed=0)
        assert len(df) == 1080, "Row count drift in default staggered DGP"


class TestButtsNonStaggeredDgpBitIdentity:
    """Verify ``generate_butts_nonstaggered_dgp`` outputs remain bit-identical."""

    @pytest.mark.parametrize("seed", [0, 42, 100])
    def test_y_hash_matches_baseline(self, seed: int) -> None:
        df = generate_butts_nonstaggered_dgp(seed=seed)
        assert _sha256_of_y(df) == _NONSTAGGERED_BASELINE_HASHES[seed], (
            f"generate_butts_nonstaggered_dgp(seed={seed}) output differs " "from pinned baseline."
        )

    def test_row_count_stable(self) -> None:
        df = generate_butts_nonstaggered_dgp(seed=0)
        assert len(df) == 400, "Row count drift in default non-staggered DGP"


class TestButtsStaggeredDgpCallableKwargs:
    """Verify the Wave C callable kwargs on generate_butts_staggered_dgp."""

    def test_constant_tau_callable_matches_scalar(self) -> None:
        """A constant-callable tau is bit-identical to the scalar default."""
        df_scalar = generate_butts_staggered_dgp(seed=7, tau_total=-0.07)
        df_callable = generate_butts_staggered_dgp(
            seed=7,
            tau_per_event_time=lambda k: -0.07,
        )
        np.testing.assert_array_equal(df_scalar["y"].values, df_callable["y"].values)

    def test_constant_delta_callable_matches_scalar(self) -> None:
        """A constant-callable delta is bit-identical to the scalar default."""
        df_scalar = generate_butts_staggered_dgp(seed=7, delta_1=-0.04)
        df_callable = generate_butts_staggered_dgp(
            seed=7,
            delta_per_ring_per_event_time=lambda j, k: -0.04,
        )
        np.testing.assert_array_equal(df_scalar["y"].values, df_callable["y"].values)

    def test_per_event_time_tau_recovers_exact_y_with_zero_noise(self) -> None:
        """With ``error_sd=0`` and a known tau callable, treated row Y values
        exactly match the closed-form ``y = mu_i + lambda_t + tau_fn(k)``.

        Strengthened per PR #456 R1 review: the previous version only
        checked that k=0 treated rows existed without verifying the formula.
        """

        def tau_fn(k):
            return -0.05 - 0.01 * k

        df = generate_butts_staggered_dgp(
            seed=11,
            error_sd=0.0,
            tau_per_event_time=tau_fn,
            delta_per_ring_per_event_time=lambda j, k: 0.0,
        )
        # The DGP sets y = mu_i + lambda_t + effect (effect=0 pre-treatment).
        # For each treated unit, derive mu_i + lambda_t at a pre-treatment
        # observation, then verify each post-treatment row matches the
        # closed-form expectation y = (mu_i + lambda_pre) + (lambda_t -
        # lambda_pre) + tau_fn(k).
        treated_mask = df["D"] == 1
        treated_df = df[treated_mask].copy()
        treated_df["k"] = (treated_df["time"] - treated_df["first_treat"]).astype(int)

        n_checked = 0
        for u in treated_df["unit"].unique():
            unit_rows = df[df["unit"] == u].sort_values("time")
            pre_rows = unit_rows[unit_rows["D"] == 0]
            if pre_rows.empty:
                continue
            pre = pre_rows.iloc[0]
            y_pre = pre["y"]
            t_pre = pre["time"]
            unit_treated = treated_df[treated_df["unit"] == u]
            for _, row in unit_treated.iterrows():
                k = int(row["k"])
                # lambda_t spec from generate_butts_staggered_dgp: 0.05 * t.
                lambda_diff = 0.05 * (row["time"] - t_pre)
                expected_y = y_pre + lambda_diff + tau_fn(k)
                np.testing.assert_allclose(
                    row["y"],
                    expected_y,
                    atol=1e-12,
                    err_msg=(
                        f"unit={u}, t={row['time']}, k={k}: "
                        f"y={row['y']:.10f}, expected={expected_y:.10f}"
                    ),
                )
                n_checked += 1
        assert n_checked > 0, "DGP produced no checkable post-treatment rows"

    def test_per_ring_event_time_delta_invokes_with_ring_zero(self) -> None:
        """Spillover rows invoke the delta callable with ring_idx=0 (DGP convention)."""
        seen_ring_indices: set = set()

        def delta_fn(j: int, k: int) -> float:
            seen_ring_indices.add(j)
            return -0.03

        generate_butts_staggered_dgp(
            seed=3,
            delta_per_ring_per_event_time=delta_fn,
        )
        assert seen_ring_indices == {0}, (
            f"Expected only ring_idx=0 invocations (one-cohort-one-cluster DGP); "
            f"got {seen_ring_indices}"
        )

    def test_callable_kwargs_independent(self) -> None:
        """Supplying only one of the callable kwargs leaves the other on its scalar path."""
        df_tau_only = generate_butts_staggered_dgp(
            seed=5,
            tau_per_event_time=lambda k: -0.10,  # different from default tau_total
            # delta defaults to scalar delta_1 = -0.04
        )
        df_scalar = generate_butts_staggered_dgp(seed=5, tau_total=-0.10)
        np.testing.assert_array_equal(df_tau_only["y"].values, df_scalar["y"].values)
