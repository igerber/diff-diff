"""
Tests for generate_reversible_did_data — the reversible-treatment data
generator added in Phase 1 of the de Chaisemartin-D'Haultfoeuille (dCDH)
estimator implementation.

This generator is the only one in the library that produces panel data
with treatment that can switch on and off over time. It is used by the
dCDH test suite (Phase 1 of the dCDH rollout, see ROADMAP.md).
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.prep import generate_reversible_did_data

# =============================================================================
# Shape and column tests
# =============================================================================


class TestGeneratorShape:
    """Verify the generator produces a balanced panel with the expected columns."""

    def test_balanced_panel(self):
        """One row per (group, period) cell, no duplicates."""
        df = generate_reversible_did_data(n_groups=10, n_periods=5, seed=1)
        assert len(df) == 10 * 5
        # Each (group, period) appears exactly once
        assert df.duplicated(subset=["group", "period"]).sum() == 0

    def test_expected_columns(self):
        df = generate_reversible_did_data(n_groups=5, n_periods=3, seed=1)
        expected = {
            "group",
            "period",
            "treatment",
            "outcome",
            "true_effect",
            "d_lag",
            "switcher_type",
        }
        assert set(df.columns) == expected

    def test_group_ids_zero_indexed(self):
        df = generate_reversible_did_data(n_groups=7, n_periods=4, seed=1)
        assert sorted(df["group"].unique()) == list(range(7))

    def test_period_ids_zero_indexed(self):
        df = generate_reversible_did_data(n_groups=5, n_periods=4, seed=1)
        assert sorted(df["period"].unique()) == list(range(4))

    def test_treatment_is_binary(self):
        df = generate_reversible_did_data(
            n_groups=20, n_periods=6, pattern="random", p_switch=0.5, seed=1
        )
        assert set(df["treatment"].unique()).issubset({0, 1})


# =============================================================================
# Pattern correctness
# =============================================================================


class TestSingleSwitchPattern:
    """The default pattern: each group switches exactly once."""

    def test_default_pattern_is_single_switch(self):
        # Confirm the default really is single_switch (not 'random')
        df = generate_reversible_did_data(n_groups=20, n_periods=6, seed=42)
        # Each group has exactly one switch from period to period
        for g in df["group"].unique():
            grp = df[df["group"] == g].sort_values("period")
            n_switches = (grp["treatment"].to_numpy()[1:] != grp["treatment"].to_numpy()[:-1]).sum()
            assert n_switches == 1, f"group {g}: expected 1 switch, got {n_switches}"

    def test_single_switch_no_multi_switch_groups(self):
        df = generate_reversible_did_data(n_groups=50, n_periods=8, pattern="single_switch", seed=7)
        for g in df["group"].unique():
            grp = df[df["group"] == g].sort_values("period")
            n_switches = (grp["treatment"].to_numpy()[1:] != grp["treatment"].to_numpy()[:-1]).sum()
            assert n_switches <= 1


class TestJoinersOnlyPattern:
    """Pure staggered adoption: every group starts at 0, switches to 1 once."""

    def test_all_groups_start_untreated(self):
        df = generate_reversible_did_data(n_groups=20, n_periods=5, pattern="joiners_only", seed=1)
        assert set(df.query("period == 0")["treatment"].unique()) == {0}

    def test_each_group_has_at_most_one_switch_up(self):
        df = generate_reversible_did_data(n_groups=15, n_periods=6, pattern="joiners_only", seed=2)
        for g in df["group"].unique():
            grp = df[df["group"] == g].sort_values("period")["treatment"].to_numpy()
            # Once a 1 appears, it should stay 1 (absorbing)
            first_one = np.argmax(grp == 1) if (grp == 1).any() else len(grp)
            if first_one < len(grp):
                assert (grp[first_one:] == 1).all(), f"group {g}: not absorbing"


class TestLeaversOnlyPattern:
    """Mirror of joiners_only: every group starts at 1, switches to 0 once."""

    def test_all_groups_start_treated(self):
        df = generate_reversible_did_data(n_groups=20, n_periods=5, pattern="leavers_only", seed=1)
        assert set(df.query("period == 0")["treatment"].unique()) == {1}

    def test_each_group_has_at_most_one_switch_down(self):
        df = generate_reversible_did_data(n_groups=15, n_periods=6, pattern="leavers_only", seed=2)
        for g in df["group"].unique():
            grp = df[df["group"] == g].sort_values("period")["treatment"].to_numpy()
            # Once a 0 appears, it should stay 0
            if (grp == 0).any():
                first_zero = np.argmax(grp == 0)
                assert (grp[first_zero:] == 0).all()


class TestMixedSingleSwitchPattern:
    """Deterministic 50/50 mix of joiners and leavers."""

    def test_first_half_are_joiners(self):
        df = generate_reversible_did_data(
            n_groups=20, n_periods=5, pattern="mixed_single_switch", seed=1
        )
        # First 10 groups are joiners (start at 0)
        first_half_t0 = df.query("group < 10 and period == 0")["treatment"]
        assert set(first_half_t0.unique()) == {0}

    def test_second_half_are_leavers(self):
        df = generate_reversible_did_data(
            n_groups=20, n_periods=5, pattern="mixed_single_switch", seed=1
        )
        # Last 10 groups are leavers (start at 1)
        second_half_t0 = df.query("group >= 10 and period == 0")["treatment"]
        assert set(second_half_t0.unique()) == {1}

    def test_no_multi_switch_groups(self):
        df = generate_reversible_did_data(
            n_groups=20, n_periods=8, pattern="mixed_single_switch", seed=3
        )
        for g in df["group"].unique():
            grp = df[df["group"] == g].sort_values("period")["treatment"].to_numpy()
            n_switches = (grp[1:] != grp[:-1]).sum()
            assert n_switches == 1


class TestRandomPattern:
    """Random flip pattern — produces multi-switch groups for n_periods >= 4."""

    def test_p_switch_zero_means_no_switches(self):
        df = generate_reversible_did_data(
            n_groups=20, n_periods=6, pattern="random", p_switch=0.0, seed=1
        )
        # With p_switch=0, every group keeps its initial state
        for g in df["group"].unique():
            grp = df[df["group"] == g].sort_values("period")["treatment"].to_numpy()
            assert len(set(grp)) == 1

    def test_random_produces_some_switches(self):
        df = generate_reversible_did_data(
            n_groups=100, n_periods=8, pattern="random", p_switch=0.4, seed=42
        )
        # With p_switch=0.4 and n_periods=8, the expected number of multi-switch
        # groups is high. We just assert at least one switch happens somewhere.
        total_switches = 0
        for g in df["group"].unique():
            grp = df[df["group"] == g].sort_values("period")["treatment"].to_numpy()
            total_switches += (grp[1:] != grp[:-1]).sum()
        assert total_switches > 0


class TestCyclesPattern:
    """Deterministic on/off cycles — guaranteed multi-switch."""

    def test_cycle_length_2_exact_sequence(self):
        df = generate_reversible_did_data(
            n_groups=4, n_periods=6, pattern="cycles", cycle_length=2, seed=1
        )
        # First two groups: phase = (t // 2) % 2 → [0,0,1,1,0,0]
        # Last two groups: opposite → [1,1,0,0,1,1]
        first = df[df["group"] == 0].sort_values("period")["treatment"].tolist()
        last = df[df["group"] == 3].sort_values("period")["treatment"].tolist()
        assert first == [0, 0, 1, 1, 0, 0]
        assert last == [1, 1, 0, 0, 1, 1]

    def test_cycles_produces_multi_switch_groups(self):
        df = generate_reversible_did_data(
            n_groups=10, n_periods=8, pattern="cycles", cycle_length=2, seed=1
        )
        for g in df["group"].unique():
            grp = df[df["group"] == g].sort_values("period")["treatment"].to_numpy()
            n_switches = (grp[1:] != grp[:-1]).sum()
            # cycles always produces > 1 switch when n_periods > 2 * cycle_length
            assert n_switches >= 2


class TestMarketingPattern:
    """Seasonal '2 on, 1 off' pattern — guaranteed multi-switch."""

    def test_marketing_pattern_exact_sequence(self):
        df = generate_reversible_did_data(n_groups=5, n_periods=9, pattern="marketing", seed=1)
        # Pattern: t % 3 != 2 → on, else → off
        # → [1, 1, 0, 1, 1, 0, 1, 1, 0]
        for g in df["group"].unique():
            grp = df[df["group"] == g].sort_values("period")["treatment"].tolist()
            assert grp == [1, 1, 0, 1, 1, 0, 1, 1, 0]

    def test_marketing_all_groups_identical(self):
        df = generate_reversible_did_data(n_groups=8, n_periods=6, pattern="marketing", seed=1)
        first = df[df["group"] == 0].sort_values("period")["treatment"].tolist()
        for g in df["group"].unique():
            grp = df[df["group"] == g].sort_values("period")["treatment"].tolist()
            assert grp == first


# =============================================================================
# Reproducibility
# =============================================================================


class TestSeedReproducibility:
    def test_same_seed_same_data_random(self):
        df1 = generate_reversible_did_data(n_groups=20, n_periods=6, pattern="random", seed=42)
        df2 = generate_reversible_did_data(n_groups=20, n_periods=6, pattern="random", seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_same_seed_same_data_single_switch(self):
        df1 = generate_reversible_did_data(n_groups=15, n_periods=5, seed=7)
        df2 = generate_reversible_did_data(n_groups=15, n_periods=5, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_data(self):
        df1 = generate_reversible_did_data(n_groups=20, n_periods=6, seed=1)
        df2 = generate_reversible_did_data(n_groups=20, n_periods=6, seed=2)
        # The treatment matrices should differ
        assert not df1["treatment"].equals(df2["treatment"])


# =============================================================================
# True effect column (ground truth for downstream tests)
# =============================================================================


class TestTrueEffectColumn:
    def test_zero_on_untreated_cells(self):
        df = generate_reversible_did_data(n_groups=20, n_periods=6, seed=1)
        untreated = df[df["treatment"] == 0]
        assert (untreated["true_effect"] == 0.0).all()

    def test_constant_treatment_effect_homogeneous(self):
        df = generate_reversible_did_data(
            n_groups=20,
            n_periods=6,
            treatment_effect=3.5,
            heterogeneous_effects=False,
            seed=1,
        )
        treated = df[df["treatment"] == 1]
        assert (treated["true_effect"] == 3.5).all()

    def test_heterogeneous_effects_vary(self):
        df = generate_reversible_did_data(
            n_groups=50,
            n_periods=6,
            treatment_effect=2.0,
            heterogeneous_effects=True,
            effect_sd=0.5,
            seed=42,
        )
        treated = df[df["treatment"] == 1]
        # With heterogeneous effects, the std should be approximately effect_sd
        assert treated["true_effect"].std() > 0.1


# =============================================================================
# Switcher type classification
# =============================================================================


class TestSwitcherTypeColumn:
    def test_period_zero_is_initial(self):
        df = generate_reversible_did_data(n_groups=10, n_periods=5, seed=1)
        period_zero = df[df["period"] == 0]
        assert (period_zero["switcher_type"] == "initial").all()

    def test_d_lag_nan_at_period_zero(self):
        df = generate_reversible_did_data(n_groups=10, n_periods=5, seed=1)
        period_zero = df[df["period"] == 0]
        assert period_zero["d_lag"].isna().all()

    def test_d_lag_finite_after_period_zero(self):
        df = generate_reversible_did_data(n_groups=10, n_periods=5, seed=1)
        non_zero = df[df["period"] > 0]
        assert non_zero["d_lag"].notna().all()

    def test_switcher_type_matches_lag_diff_joiners_only(self):
        df = generate_reversible_did_data(n_groups=10, n_periods=5, pattern="joiners_only", seed=1)
        # In joiners_only, period 0 is "initial", and within each group
        # there is exactly one "joiner" cell (the switch) followed by
        # "stable_1" cells. The remaining post-period-0 cells before
        # the switch are "stable_0".
        for g in df["group"].unique():
            grp = df[df["group"] == g].sort_values("period")
            types = grp["switcher_type"].tolist()
            assert types[0] == "initial"
            joiner_count = sum(1 for t in types if t == "joiner")
            leaver_count = sum(1 for t in types if t == "leaver")
            assert joiner_count <= 1  # at most one joiner cell per group
            assert leaver_count == 0  # never a leaver in joiners_only

    def test_switcher_type_classification_explicit(self):
        """Build a known panel and verify each cell's switcher_type."""
        # Use single_switch with seed to control the panel
        # Check specific (treatment, d_lag) combinations are classified correctly.
        df = generate_reversible_did_data(n_groups=20, n_periods=6, seed=42)
        # Period 0 cells must be "initial" with NaN d_lag
        p0 = df[df["period"] == 0]
        assert (p0["switcher_type"] == "initial").all()
        # Period > 0 cells: classification matches (treatment, d_lag)
        post = df[df["period"] > 0]
        joiners = post[(post["d_lag"] == 0) & (post["treatment"] == 1)]
        leavers = post[(post["d_lag"] == 1) & (post["treatment"] == 0)]
        stable_0 = post[(post["d_lag"] == 0) & (post["treatment"] == 0)]
        stable_1 = post[(post["d_lag"] == 1) & (post["treatment"] == 1)]
        assert (joiners["switcher_type"] == "joiner").all()
        assert (leavers["switcher_type"] == "leaver").all()
        assert (stable_0["switcher_type"] == "stable_0").all()
        assert (stable_1["switcher_type"] == "stable_1").all()


# =============================================================================
# Validation errors
# =============================================================================


class TestValidationErrors:
    def test_invalid_pattern_raises(self):
        with pytest.raises(ValueError, match="pattern must be one of"):
            generate_reversible_did_data(pattern="bogus", seed=1)

    def test_n_groups_zero_raises(self):
        with pytest.raises(ValueError, match="n_groups must be positive"):
            generate_reversible_did_data(n_groups=0, seed=1)

    def test_n_periods_one_raises(self):
        with pytest.raises(ValueError, match="n_periods must be at least 2"):
            generate_reversible_did_data(n_groups=10, n_periods=1, seed=1)

    def test_initial_treat_frac_out_of_range_raises(self):
        with pytest.raises(ValueError, match="initial_treat_frac must be in"):
            generate_reversible_did_data(initial_treat_frac=1.5, seed=1)

    def test_p_switch_out_of_range_raises(self):
        with pytest.raises(ValueError, match="p_switch must be in"):
            generate_reversible_did_data(p_switch=-0.1, seed=1)

    def test_negative_cycle_length_raises(self):
        with pytest.raises(ValueError, match="cycle_length must be positive"):
            generate_reversible_did_data(cycle_length=0, seed=1)

    def test_negative_noise_sd_raises(self):
        with pytest.raises(ValueError, match="noise_sd must be non-negative"):
            generate_reversible_did_data(noise_sd=-1.0, seed=1)
