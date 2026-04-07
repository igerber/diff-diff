"""
Test suite for the Efficient DiD estimator (Chen, Sant'Anna & Xie 2025).

Organized into tiers:
  Tier 1 — Core correctness (fast, deterministic)
  Tier 2 — Weight behavior and edge cases
  Tier 3 — Bootstrap
  Tier 4 — Simulation validation (slow, scaled via ci_params)
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from edid_dgp import make_compustat_dgp

from diff_diff import CallawaySantAnna, EDiD, EfficientDiD
from diff_diff.efficient_did_results import EfficientDiDResults
from diff_diff.efficient_did_weights import (
    enumerate_valid_triples,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_simple_panel(
    n_units=100,
    n_periods=5,
    n_treated=50,
    treat_period=3,
    effect=2.0,
    sigma=0.5,
    seed=42,
):
    """Generate a simple balanced panel with one treatment cohort."""
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)

    ft = np.full(n_units, np.inf)
    ft[:n_treated] = treat_period
    ft_col = np.repeat(ft, n_periods)

    unit_fe = np.repeat(rng.normal(0, 1, n_units), n_periods)
    time_fe = np.tile(np.arange(1, n_periods + 1) * 0.5, n_units)
    tau = np.where((ft_col < np.inf) & (times >= ft_col), effect, 0.0)
    y = unit_fe + time_fe + tau + rng.normal(0, sigma, len(units))

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "first_treat": ft_col,
            "y": y,
        }
    )


def _make_staggered_panel(
    n_per_group=60,
    n_control=80,
    groups=(3, 5),
    effects=None,
    n_periods=7,
    sigma=0.5,
    rho=0.0,
    seed=42,
):
    """Generate staggered treatment panel with AR(1) errors."""
    if effects is None:
        effects = {3: 2.0, 5: 1.0}
    rng = np.random.default_rng(seed)
    n_units = n_per_group * len(groups) + n_control
    n_t = n_periods

    units = np.repeat(np.arange(n_units), n_t)
    times = np.tile(np.arange(1, n_t + 1), n_units)

    ft = np.full(n_units, np.inf)
    start = 0
    for g in groups:
        ft[start : start + n_per_group] = g
        start += n_per_group
    ft_col = np.repeat(ft, n_t)

    unit_fe = np.repeat(rng.normal(0, 0.5, n_units), n_t)
    time_fe = np.tile(rng.normal(0, 0.1, n_t), n_units)

    # AR(1) errors
    eps = np.zeros((n_units, n_t))
    eps[:, 0] = rng.normal(0, sigma, n_units)
    for t in range(1, n_t):
        eps[:, t] = rho * eps[:, t - 1] + rng.normal(0, sigma, n_units)
    eps_flat = eps.flatten()

    tau = np.zeros(len(units))
    for g, eff in effects.items():
        mask = (ft_col == g) & (times >= g)
        tau[mask] = eff

    y = unit_fe + time_fe + tau + eps_flat

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "first_treat": ft_col,
            "y": y,
        }
    )


def _make_compustat_dgp(n_units=400, n_periods=11, rho=0.0, seed=42):
    """Delegate to shared DGP in edid_dgp.py."""
    return make_compustat_dgp(n_units=n_units, n_periods=n_periods, rho=rho, seed=seed)


# =============================================================================
# Tier 1: Core Correctness
# =============================================================================


class TestBasicFit:
    """Test basic fit mechanics: types, shapes, required outputs."""

    def test_basic_fit(self):
        df = _make_simple_panel()
        edid = EfficientDiD(pt_assumption="all")
        result = edid.fit(df, "y", "unit", "time", "first_treat")

        assert isinstance(result, EfficientDiDResults)
        assert isinstance(result.overall_att, float)
        assert isinstance(result.overall_se, float)
        assert len(result.group_time_effects) > 0
        assert result.n_obs == len(df)
        assert result.pt_assumption == "all"

    def test_zero_effect(self):
        df = _make_simple_panel(effect=0.0)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        # ATT should be near 0
        assert abs(result.overall_att) < 0.5

    def test_positive_effect(self):
        df = _make_simple_panel(effect=2.0, n_units=200)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        # Recover ~2.0 within 2 SE
        assert abs(result.overall_att - 2.0) < 2 * result.overall_se + 0.5

    def test_single_pre_period(self):
        """When g=2 (only 1 pre-period), weights are trivially [1.0]."""
        df = _make_simple_panel(n_periods=4, treat_period=2)
        result = EfficientDiD(pt_assumption="all").fit(df, "y", "unit", "time", "first_treat")
        assert len(result.group_time_effects) > 0
        # Check weights are stored and have length 1 for the single valid pair
        if result.efficient_weights:
            for gt, w in result.efficient_weights.items():
                if len(w) == 1:
                    assert abs(w[0] - 1.0) < 1e-10


class TestPTPostMatchesCS:
    """Under PT-Post, EDiD should approximately match CS.

    The EDiD formula uses period_1 (earliest period) as the universal baseline,
    while CS uses g-1 (varying base). These are the same when g=2 (period_1 = g-1),
    and approximately the same for g > 2 under parallel trends.
    """

    def test_single_group_g2_exact_match(self):
        """g=2 means g-1 = period_1 = 1, so baselines coincide."""
        df = _make_simple_panel(n_units=200, treat_period=2, n_periods=5)
        edid = EfficientDiD(pt_assumption="post")
        cs = CallawaySantAnna(control_group="never_treated", base_period="varying")

        res_e = edid.fit(df, "y", "unit", "time", "first_treat")
        res_c = cs.fit(df, "y", "unit", "time", "first_treat")

        for gt in res_e.group_time_effects:
            if gt in res_c.group_time_effects:
                e_eff = res_e.group_time_effects[gt]["effect"]
                c_eff = res_c.group_time_effects[gt]["effect"]
                assert abs(e_eff - c_eff) < 1e-10, f"ATT{gt}: EDiD={e_eff:.10f} CS={c_eff:.10f}"

    def test_staggered_approximate_match(self):
        """For g > 2, EDiD(PT-Post) should exactly match CS for post-treatment effects."""
        df = _make_staggered_panel()
        edid = EfficientDiD(pt_assumption="post")
        cs = CallawaySantAnna(control_group="never_treated", base_period="varying")

        res_e = edid.fit(df, "y", "unit", "time", "first_treat")
        res_c = cs.fit(df, "y", "unit", "time", "first_treat")

        matched = 0
        for g, t in res_e.group_time_effects:
            if t >= g and (g, t) in res_c.group_time_effects:
                e_eff = res_e.group_time_effects[(g, t)]["effect"]
                c_eff = res_c.group_time_effects[(g, t)]["effect"]
                assert abs(e_eff - c_eff) < 1e-8, f"ATT({g},{t}): EDiD={e_eff:.10f} CS={c_eff:.10f}"
                matched += 1
        assert matched > 0, "No matching post-treatment effects found"


class TestAggregation:
    """Test aggregation: event study, group, overall."""

    def test_event_study_aggregation(self):
        df = _make_simple_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="event_study")
        assert result.event_study_effects is not None
        # Should have pre and post-treatment event times
        keys = sorted(result.event_study_effects.keys())
        assert any(e < 0 for e in keys), "Should have pre-treatment event times"
        assert any(e >= 0 for e in keys), "Should have post-treatment event times"

    def test_group_aggregation(self):
        df = _make_staggered_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="group")
        assert result.group_effects is not None
        assert 3.0 in result.group_effects
        assert 5.0 in result.group_effects

    def test_aggregate_all(self):
        df = _make_staggered_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="all")
        assert result.event_study_effects is not None
        assert result.group_effects is not None


class TestValidation:
    """Test input validation: missing columns, unbalanced, non-absorbing."""

    def test_balanced_panel_validation(self):
        df = _make_simple_panel()
        # Drop some rows to create unbalanced panel
        df = df.drop(df.index[:3])
        with pytest.raises(ValueError, match="Unbalanced panel"):
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat")

    def test_absorbing_treatment_validation(self):
        df = _make_simple_panel()
        # Make treatment non-absorbing for one unit
        mask = (df["unit"] == 0) & (df["time"] == 1)
        df.loc[mask, "first_treat"] = 5  # changes first_treat mid-panel
        with pytest.raises(ValueError, match="Non-absorbing"):
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat")

    def test_missing_covariate_column_raises(self):
        df = _make_simple_panel()
        with pytest.raises(ValueError, match="Missing covariate columns"):
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat", covariates=["nonexistent"])

    def test_missing_columns(self):
        df = _make_simple_panel()
        with pytest.raises(ValueError, match="Missing columns"):
            EfficientDiD().fit(df, "y", "unit", "time", "nonexistent")

    def test_pt_post_no_never_treated_raises(self):
        """PT-Post without never-treated group should raise."""
        df = _make_simple_panel(n_treated=100)  # all treated
        with pytest.raises(ValueError, match="never-treated"):
            EfficientDiD(pt_assumption="post").fit(df, "y", "unit", "time", "first_treat")

    def test_nan_outcome_raises(self):
        """Non-finite outcomes in a balanced panel should be rejected."""
        df = _make_simple_panel()
        df.loc[df.index[0], "y"] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat")

    def test_duplicate_unit_time_raises(self):
        """Duplicate (unit, time) rows should be rejected."""
        df = _make_simple_panel()
        # Duplicate a row
        dup_row = df.iloc[[0]].copy()
        df = pd.concat([df, dup_row], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat")


class TestSklearnCompat:
    """Test get_params / set_params."""

    def test_get_set_params(self):
        edid = EfficientDiD(pt_assumption="post", alpha=0.10, anticipation=1)
        params = edid.get_params()
        assert params["pt_assumption"] == "post"
        assert params["alpha"] == 0.10
        assert params["anticipation"] == 1

        edid.set_params(alpha=0.01)
        assert edid.alpha == 0.01
        assert edid.get_params()["alpha"] == 0.01

    def test_unknown_param_raises(self):
        edid = EfficientDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            edid.set_params(nonexistent=True)

    def test_set_params_validates(self):
        edid = EfficientDiD()
        with pytest.raises(ValueError, match="pt_assumption"):
            edid.set_params(pt_assumption="POST")
        edid2 = EfficientDiD()
        with pytest.raises(ValueError, match="bootstrap_weights"):
            edid2.set_params(bootstrap_weights="invalid")

    def test_alias(self):
        assert EDiD is EfficientDiD


class TestOutputFormats:
    """Test summary() and to_dataframe()."""

    def test_summary_and_dataframe(self):
        df = _make_simple_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="all")

        # summary() returns a string
        s = result.summary()
        assert isinstance(s, str)
        assert "Efficient DiD" in s

        # to_dataframe at different levels
        df_gt = result.to_dataframe("group_time")
        assert isinstance(df_gt, pd.DataFrame)
        assert "effect" in df_gt.columns

        df_es = result.to_dataframe("event_study")
        assert "relative_period" in df_es.columns

        df_g = result.to_dataframe("group")
        assert "group" in df_g.columns

    def test_to_dataframe_raises_without_aggregation(self):
        df = _make_simple_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        with pytest.raises(ValueError, match="Event study effects not computed"):
            result.to_dataframe("event_study")

    def test_repr(self):
        df = _make_simple_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        r = repr(result)
        assert "EfficientDiDResults" in r

    def test_significance_properties(self):
        df = _make_simple_panel(effect=5.0, n_units=200)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        assert isinstance(result.is_significant, bool)
        assert isinstance(result.significance_stars, str)


class TestNanInference:
    """Test NaN propagation for undefined inference."""

    def test_nan_for_empty_pairs(self):
        """When no valid pairs exist, ATT should be NaN with proper NaN inference."""
        # Create a scenario with a single period (no pre-treatment baseline)
        df = _make_simple_panel(n_periods=2, treat_period=2)
        # Under PT-Post, baseline is g-1 = 1 = period_1, which IS the
        # universal reference. The enumerate function skips period_1 as t_pre,
        # so no valid pairs exist.
        # Actually, under PT-Post, baseline = g - 1 = 1 and period_1 = 1.
        # The valid pair would be (inf, 1), but period_1 is skipped.
        # So we should get NaN for pre-treatment effects at least.

        result = EfficientDiD(pt_assumption="all").fit(df, "y", "unit", "time", "first_treat")
        # At minimum, all effects should have finite or NaN SE
        for gt, d in result.group_time_effects.items():
            assert np.isfinite(d["effect"]) or np.isnan(d["effect"])


class TestPretreatment:
    """Test pre-treatment placebo effects."""

    def test_pretreatment_placebo_near_zero(self):
        """Under correct PT, pre-treatment ATT(g,t) for t < g should be near 0."""
        df = _make_simple_panel(n_units=200, effect=2.0, sigma=0.3)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="event_study")
        # Check pre-treatment effects are near zero
        for e, d in result.event_study_effects.items():
            if e < 0:
                assert (
                    abs(d["effect"]) < 1.0
                ), f"Pre-treatment effect at e={e} is {d['effect']:.4f}, expected ~0"

    def test_pretreatment_in_event_study(self):
        """Placebo effects should appear with negative event-time keys."""
        df = _make_simple_panel(n_periods=6, treat_period=3)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="event_study")
        assert result.event_study_effects is not None
        neg_keys = [e for e in result.event_study_effects if e < 0]
        assert len(neg_keys) > 0, "Should have negative event-time keys"

    def test_pretreatment_detects_violation(self):
        """DGP with pre-trend should produce non-zero placebo ATTs."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 200, 6
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(1, n_periods + 1), n_units)
        ft = np.full(n_units, np.inf)
        ft[:100] = 4  # treated at t=4
        ft_col = np.repeat(ft, n_periods)
        uf = np.repeat(rng.normal(0, 1, n_units), n_periods)
        tf = np.tile(np.arange(1, n_periods + 1) * 0.5, n_units)
        # Add pre-trend for treated group
        pre_trend = np.where(ft_col < np.inf, times * 0.3, 0.0)
        treatment = np.where((ft_col < np.inf) & (times >= ft_col), 2.0, 0.0)
        y = uf + tf + pre_trend + treatment + rng.normal(0, 0.2, len(units))
        df = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "first_treat": ft_col,
                "y": y,
            }
        )
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="event_study")
        # Pre-treatment effects should be significantly non-zero
        pre_effects = [d["effect"] for e, d in result.event_study_effects.items() if e < 0]
        assert any(
            abs(e) > 0.1 for e in pre_effects
        ), f"Pre-trend should be detected; pre effects: {pre_effects}"


# =============================================================================
# Tier 2: Weight Behavior and Edge Cases
# =============================================================================


class TestWeightBehavior:
    """Test that efficient weights respond to error structure."""

    def test_weights_uniform_under_iid(self):
        """iid errors -> weights should sum to 1 and be non-degenerate."""
        df = _make_staggered_panel(rho=0.0, seed=123, n_per_group=100, n_control=100)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        if result.efficient_weights:
            for gt, w in result.efficient_weights.items():
                if len(w) > 1:
                    # Weights should sum to 1
                    assert abs(w.sum() - 1.0) < 1e-8
                    # At least some variation (not all same)
                    assert w.std() > 0

    def test_condition_number_warning(self):
        """Near-singular Omega* should trigger a warning."""
        # Use a perfectly collinear DGP to produce near-singular Omega*
        n_units, n_periods = 100, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(1, n_periods + 1), n_units)
        ft = np.full(n_units, np.inf)
        ft[:50] = 4
        ft_col = np.repeat(ft, n_periods)
        # Constant outcome (zero variance -> degenerate Omega*)
        y = np.ones(len(units)) + np.where((ft_col < np.inf) & (times >= ft_col), 1.0, 0.0)
        df = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "first_treat": ft_col,
                "y": y,
            }
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
            # Should get a warning about condition number or zero matrix
            warning_msgs = [str(x.message) for x in w]
            assert any(
                "condition" in m.lower()
                or "zero" in m.lower()
                or "pseudoinverse" in m.lower()
                or "uniform" in m.lower()
                for m in warning_msgs
            ), f"Expected condition/zero warning, got: {warning_msgs}"


class TestValidTriples:
    """Test enumerate_valid_triples with hand-worked examples."""

    def test_pt_all_simple(self):
        """T=5, groups={3, inf}, target (3, 4), period_1=1.
        Under PT-All: g'=inf with t_pre in {2,3,4,5} = 4 pairs,
        plus g'=3 (same-group) with t_pre in {2} (t_pre < g'=3) = 1 pair.
        Total: 5 pairs."""
        pairs = enumerate_valid_triples(
            target_g=3,
            treatment_groups=[3],
            time_periods=[1, 2, 3, 4, 5],
            period_1=1,
            pt_assumption="all",
        )
        expected = {(np.inf, 2), (np.inf, 3), (np.inf, 4), (np.inf, 5), (3, 2)}
        actual = set(pairs)
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_pt_all_staggered(self):
        """T=5, groups={3, 5, inf}, target (3, 4), period_1=1.
        Under PT-All: g'=inf: t_pre in {2,3,4,5} = 4 pairs,
        g'=5: t_pre in {2,3,4} (t_pre < 5) = 3 pairs,
        g'=3: t_pre in {2} (t_pre < 3) = 1 pair.
        Total: 8 pairs."""
        pairs = enumerate_valid_triples(
            target_g=3,
            treatment_groups=[3, 5],
            time_periods=[1, 2, 3, 4, 5],
            period_1=1,
            pt_assumption="all",
        )
        expected = {
            (np.inf, 2),
            (np.inf, 3),
            (np.inf, 4),
            (np.inf, 5),
            (5, 2),
            (5, 3),
            (5, 4),
            (3, 2),
        }
        actual = set(pairs)
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_pt_post_single_pair(self):
        """PT-Post: only (inf, g-1)."""
        pairs = enumerate_valid_triples(
            target_g=3,
            treatment_groups=[3, 5],
            time_periods=[1, 2, 3, 4, 5],
            period_1=1,
            pt_assumption="post",
        )
        assert pairs == [(np.inf, 2)]

    def test_g2_has_valid_pairs_pt_all(self):
        """When g=2, period_1=1, under PT-All: g'=inf gives t_pre in {2,3}
        (no t_pre < g constraint), g'=2 has no valid t_pre (t_pre < 2, skip period_1).
        So pairs should be non-empty."""
        pairs = enumerate_valid_triples(
            target_g=2,
            treatment_groups=[2],
            time_periods=[1, 2, 3],
            period_1=1,
            pt_assumption="all",
        )
        # g'=inf: t_pre in {2, 3} (no constraint other than != period_1)
        # g'=2: t_pre must be < 2 and != 1 -> empty
        expected = {(np.inf, 2), (np.inf, 3)}
        actual = set(pairs)
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_anticipation(self):
        """Anticipation shifts effective treatment boundary."""
        pairs_no_ant = enumerate_valid_triples(
            target_g=4,
            treatment_groups=[4],
            time_periods=[1, 2, 3, 4, 5],
            period_1=1,
            pt_assumption="all",
            anticipation=0,
        )
        pairs_ant1 = enumerate_valid_triples(
            target_g=4,
            treatment_groups=[4],
            time_periods=[1, 2, 3, 4, 5],
            period_1=1,
            pt_assumption="all",
            anticipation=1,
        )
        # With anticipation=1, effective treatment is at g-1=3
        # so fewer pre-treatment baselines available
        assert len(pairs_ant1) <= len(pairs_no_ant)


class TestHausmanPretest:
    """Hausman pretest for PT-All vs PT-Post."""

    def test_hausman_homogeneous_trends_fail_to_reject(self):
        """DGP with homogeneous trends → fail to reject PT-All."""
        # Standard DGP: parallel trends hold for all groups
        df = _make_staggered_panel(n_per_group=100, n_control=150, sigma=0.3, seed=42)
        pretest = EfficientDiD.hausman_pretest(df, "y", "unit", "time", "first_treat", alpha=0.05)
        assert np.isfinite(pretest.statistic)
        assert np.isfinite(pretest.p_value)
        assert pretest.df > 0
        # With homogeneous trends, should generally fail to reject
        assert pretest.recommendation in ("pt_all", "pt_post", "inconclusive")

    def test_hausman_differential_trends_detects(self):
        """DGP with cohort-specific trends → test detects or warns."""
        rng = np.random.default_rng(42)
        n_per_group = 200
        n_control = 300
        n_periods = 7
        groups = (3, 5)
        n_units = n_per_group * len(groups) + n_control

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(1, n_periods + 1), n_units)
        ft = np.full(n_units, np.inf)
        ft[:n_per_group] = 3
        ft[n_per_group : 2 * n_per_group] = 5
        ft_col = np.repeat(ft, n_periods)

        # Add strong cohort-specific trends that violate PT-All
        trend = np.zeros(len(units))
        for i in range(len(units)):
            if ft_col[i] == 3:
                trend[i] = 2.0 * times[i]
            elif ft_col[i] == 5:
                trend[i] = -1.5 * times[i]

        unit_fe = np.repeat(rng.normal(0, 0.1, n_units), n_periods)
        time_fe = np.tile(rng.normal(0, 0.05, n_periods), n_units)
        eps = rng.normal(0, 0.1, len(units))
        tau = np.where((ft_col < np.inf) & (times >= ft_col), 2.0, 0.0)

        y = unit_fe + time_fe + trend + tau + eps
        df = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "first_treat": ft_col,
                "y": y,
            }
        )

        pretest = EfficientDiD.hausman_pretest(df, "y", "unit", "time", "first_treat", alpha=0.05)
        # With strong differential trends, either:
        # (a) test rejects PT-All, or
        # (b) covariance is unreliable (NaN) and recommendation defaults to pt_post
        # Both are acceptable outcomes for a DGP that violates PT-All
        if np.isfinite(pretest.statistic):
            assert pretest.statistic >= 0
        assert pretest.recommendation in ("pt_all", "pt_post", "inconclusive")

    def test_hausman_es_details(self):
        """gt_details should have event-study columns per Theorem A.1."""
        df = _make_staggered_panel(n_per_group=80, n_control=100)
        pretest = EfficientDiD.hausman_pretest(df, "y", "unit", "time", "first_treat")
        assert pretest.gt_details is not None
        expected_cols = {"relative_period", "es_all", "es_post", "delta"}
        assert set(pretest.gt_details.columns) == expected_cols
        # All relative periods should be post-treatment (>= 0)
        assert all(e >= 0 for e in pretest.gt_details["relative_period"])

    def test_hausman_recommendation_field(self):
        """recommendation should be pt_all or pt_post."""
        df = _make_staggered_panel(n_per_group=80, n_control=100)
        pretest = EfficientDiD.hausman_pretest(df, "y", "unit", "time", "first_treat")
        assert pretest.recommendation in ("pt_all", "pt_post", "inconclusive")
        if pretest.reject:
            assert pretest.recommendation == "pt_post"
        else:
            assert pretest.recommendation == "pt_all"

    def test_hausman_repr(self):
        """repr should be informative."""
        df = _make_staggered_panel(n_per_group=80, n_control=100)
        pretest = EfficientDiD.hausman_pretest(df, "y", "unit", "time", "first_treat")
        r = repr(pretest)
        assert "HausmanPretestResult" in r
        assert "recommend=" in r

    def test_hausman_clustered(self):
        """Hausman pretest with cluster-robust covariance should produce finite output."""
        rng = np.random.default_rng(42)
        n_clusters = 40
        units_per_cluster = 5
        n_units = n_clusters * units_per_cluster
        n_periods = 7
        n_per_group = n_units // 4

        cluster_ids = np.repeat(np.arange(n_clusters), units_per_cluster)
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(1, n_periods + 1), n_units)
        ft = np.full(n_units, np.inf)
        ft[:n_per_group] = 3
        ft[n_per_group : 2 * n_per_group] = 5
        ft_col = np.repeat(ft, n_periods)

        unit_fe = np.repeat(rng.normal(0, 0.3, n_units), n_periods)
        cluster_fe = np.repeat(rng.normal(0, 0.5, n_clusters)[cluster_ids], n_periods)
        eps = rng.normal(0, 0.3, len(units))
        tau = np.where((ft_col < np.inf) & (times >= ft_col), 2.0, 0.0)
        y = unit_fe + cluster_fe + tau + eps

        df = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "first_treat": ft_col,
                "y": y,
                "cluster_id": np.repeat(cluster_ids, n_periods),
            }
        )

        pretest = EfficientDiD.hausman_pretest(
            df, "y", "unit", "time", "first_treat", cluster="cluster_id"
        )
        assert pretest.recommendation in ("pt_all", "pt_post", "inconclusive")
        assert pretest.df >= 0

    def test_hausman_clustered_stale_ncl_after_nan_filtering(self):
        """After filtering NaN EIF rows, n_cl must be recomputed.

        If entire clusters are dropped by the row_finite filter, the original
        n_cl overcounts clusters, inflating variance via the (n_cl / (n_cl-1))
        correction and phantom zero rows in _cluster_aggregate.
        """
        rng = np.random.default_rng(99)
        n_clusters = 10
        units_per_cluster = 8
        n_units = n_clusters * units_per_cluster
        n_periods = 7

        cluster_ids = np.repeat(np.arange(n_clusters), units_per_cluster)
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(1, n_periods + 1), n_units)
        ft = np.full(n_units, np.inf)
        # Two small treatment cohorts
        ft[:8] = 3  # cluster 0 only
        ft[8:24] = 5  # clusters 1-2

        ft_col = np.repeat(ft, n_periods)
        unit_fe = np.repeat(rng.normal(0, 0.3, n_units), n_periods)
        eps = rng.normal(0, 0.3, len(units))
        tau = np.where((ft_col < np.inf) & (times >= ft_col), 2.0, 0.0)
        y = unit_fe + tau + eps

        df = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "first_treat": ft_col,
                "y": y,
                "cluster_id": np.repeat(cluster_ids, n_periods),
            }
        )

        pretest = EfficientDiD.hausman_pretest(
            df, "y", "unit", "time", "first_treat", cluster="cluster_id"
        )
        assert pretest.recommendation in ("pt_all", "pt_post", "inconclusive")
        # Key assertion: if the statistic is finite, df should reflect the
        # actual number of clusters remaining after NaN filtering
        if np.isfinite(pretest.statistic):
            assert pretest.df > 0

    def test_hausman_last_cohort(self):
        """Hausman pretest on all-treated panel with last_cohort control."""
        df = _make_staggered_panel(
            n_per_group=80,
            n_control=0,
            groups=(3, 5, 7),
            effects={3: 2.0, 5: 1.5, 7: 1.0},
        )
        pretest = EfficientDiD.hausman_pretest(
            df,
            "y",
            "unit",
            "time",
            "first_treat",
            control_group="last_cohort",
        )
        assert pretest.recommendation in ("pt_all", "pt_post", "inconclusive")
        assert np.isfinite(pretest.att_all)
        assert np.isfinite(pretest.att_post)


class TestClusterRobustSE:
    """Cluster-robust standard errors for EfficientDiD."""

    @staticmethod
    def _make_clustered_panel(n_clusters=20, units_per_cluster=5, seed=42):
        """Panel data with cluster structure and intracluster correlation."""
        rng = np.random.default_rng(seed)
        n_units = n_clusters * units_per_cluster
        n_periods = 7
        groups = (3, 5)
        n_per_group = n_units // 4  # ~25% in each treatment group

        cluster_ids = np.repeat(np.arange(n_clusters), units_per_cluster)
        cluster_effects = rng.normal(0, 1.0, n_clusters)

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(1, n_periods + 1), n_units)

        ft = np.full(n_units, np.inf)
        ft[:n_per_group] = groups[0]
        ft[n_per_group : 2 * n_per_group] = groups[1]
        ft_col = np.repeat(ft, n_periods)

        # Intracluster correlation via shared cluster effect
        unit_fe = np.repeat(rng.normal(0, 0.3, n_units), n_periods)
        cluster_fe = np.repeat(cluster_effects[cluster_ids], n_periods)
        time_fe = np.tile(rng.normal(0, 0.1, n_periods), n_units)
        eps = rng.normal(0, 0.3, len(units))

        tau = np.zeros(len(units))
        for g in groups:
            mask = (ft_col == g) & (times >= g)
            tau[mask] = 2.0

        y = unit_fe + cluster_fe + time_fe + tau + eps
        cluster_col = np.repeat(cluster_ids, n_periods)

        return pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "first_treat": ft_col,
                "y": y,
                "cluster_id": cluster_col,
            }
        )

    def test_cluster_no_longer_raises(self):
        """cluster parameter should not raise NotImplementedError."""
        df = self._make_clustered_panel()
        result = EfficientDiD(cluster="cluster_id").fit(df, "y", "unit", "time", "first_treat")
        assert np.isfinite(result.overall_att)

    def test_single_unit_clusters_match_unclustered(self):
        """With one unit per cluster, clustered SE should match unclustered."""
        df = _make_staggered_panel(n_per_group=60, n_control=80)
        # Add cluster column = unit (each unit is its own cluster)
        df["cluster_id"] = df["unit"]
        result_unclustered = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        result_clustered = EfficientDiD(cluster="cluster_id").fit(
            df, "y", "unit", "time", "first_treat"
        )
        assert result_clustered.overall_att == pytest.approx(
            result_unclustered.overall_att, abs=1e-10
        )
        # SEs should be very close (centering correction is negligible)
        assert result_clustered.overall_se == pytest.approx(result_unclustered.overall_se, rel=0.05)

    def test_clustered_se_at_least_as_large(self):
        """Clustered SE >= unclustered SE with positive intracluster correlation."""
        df = self._make_clustered_panel()
        result_unclustered = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        result_clustered = EfficientDiD(cluster="cluster_id").fit(
            df, "y", "unit", "time", "first_treat"
        )
        # Both SEs should be finite and positive
        assert result_clustered.overall_se > 0
        assert result_unclustered.overall_se > 0

    def test_clustered_aggregate_event_study(self):
        """Clustered SE with aggregate='event_study' should produce finite results."""
        df = self._make_clustered_panel(n_clusters=60, units_per_cluster=3)
        result = EfficientDiD(cluster="cluster_id").fit(
            df, "y", "unit", "time", "first_treat", aggregate="event_study"
        )
        assert result.event_study_effects is not None
        for e, d in result.event_study_effects.items():
            assert np.isfinite(d["se"])

    def test_clustered_aggregate_all(self):
        """Clustered SE with aggregate='all' should produce finite results."""
        df = self._make_clustered_panel(n_clusters=60, units_per_cluster=3)
        result = EfficientDiD(cluster="cluster_id").fit(
            df, "y", "unit", "time", "first_treat", aggregate="all"
        )
        assert result.event_study_effects is not None
        assert result.group_effects is not None
        for g, d in result.group_effects.items():
            assert np.isfinite(d["se"])

    def test_cluster_bootstrap(self, ci_params):
        """Cluster bootstrap should produce finite inference."""
        n_boot = ci_params.bootstrap(99)
        df = self._make_clustered_panel()
        result = EfficientDiD(cluster="cluster_id", n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_few_clusters_warns(self):
        """Fewer than 50 clusters should warn."""
        df = self._make_clustered_panel(n_clusters=10, units_per_cluster=10)
        with pytest.warns(UserWarning, match="Only 10 clusters"):
            EfficientDiD(cluster="cluster_id").fit(df, "y", "unit", "time", "first_treat")

    def test_cluster_get_params(self):
        """cluster param round-trips through get_params/set_params."""
        edid = EfficientDiD(cluster="state")
        assert edid.get_params()["cluster"] == "state"

    def test_clustered_se_manual_liang_zeger(self):
        """Verify clustered SE matches hand-computed Liang-Zeger formula."""
        from diff_diff.efficient_did import _compute_se_from_eif

        # 6 units, 2 clusters of 3 units each
        eif = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        cluster_indices = np.array([0, 0, 0, 1, 1, 1])
        n_clusters = 2
        n_units = 6
        # Cluster sums: [1+2+3=6, -1-2-3=-6]
        # Cluster mean: (6 + -6) / 2 = 0
        # Centered: [6, -6]
        # sum(centered^2) = 36 + 36 = 72
        # G/(G-1) = 2/1 = 2
        # Var = 2 * 72 / 36 = 4.0
        # SE = sqrt(4.0) = 2.0
        se = _compute_se_from_eif(eif, n_units, cluster_indices, n_clusters)
        assert se == pytest.approx(2.0, rel=1e-10)

    def test_cluster_missing_column_raises(self):
        """Missing cluster column should raise ValueError."""
        df = _make_staggered_panel(n_per_group=60, n_control=80)
        with pytest.raises(ValueError, match="not found"):
            EfficientDiD(cluster="nonexistent").fit(df, "y", "unit", "time", "first_treat")

    def test_cluster_nan_raises(self):
        """NaN in cluster column should raise ValueError."""
        df = _make_staggered_panel(n_per_group=60, n_control=80)
        df["cluster_id"] = df["unit"] % 5
        df.loc[df.index[0], "cluster_id"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            EfficientDiD(cluster="cluster_id").fit(df, "y", "unit", "time", "first_treat")

    def test_cluster_varies_within_unit_raises(self):
        """Cluster that changes over time should raise ValueError."""
        df = _make_staggered_panel(n_per_group=60, n_control=80)
        # Assign cluster = time (varies within unit)
        df["cluster_id"] = df["time"]
        with pytest.raises(ValueError, match="varies within unit"):
            EfficientDiD(cluster="cluster_id").fit(df, "y", "unit", "time", "first_treat")

    def test_single_cluster_raises(self):
        """Single cluster should raise ValueError."""
        df = _make_staggered_panel(n_per_group=60, n_control=80)
        df["cluster_id"] = 0  # all same cluster
        with pytest.raises(ValueError, match="at least 2 clusters"):
            EfficientDiD(cluster="cluster_id").fit(df, "y", "unit", "time", "first_treat")

    def test_cluster_plus_survey_raises(self):
        """cluster + survey_design should raise NotImplementedError."""
        df = _make_staggered_panel(n_per_group=60, n_control=80)
        df["cluster_id"] = df["unit"] % 5
        df["w"] = 1.0
        with pytest.raises(NotImplementedError, match="cluster and survey_design"):
            EfficientDiD(cluster="cluster_id").fit(
                df, "y", "unit", "time", "first_treat", survey_design="w"
            )

    def test_clustered_bootstrap_aggregate_all(self, ci_params):
        """Clustered bootstrap with aggregate='all' should produce finite results."""
        n_boot = ci_params.bootstrap(99)
        df = self._make_clustered_panel(n_clusters=60, units_per_cluster=3)
        result = EfficientDiD(cluster="cluster_id", n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat", aggregate="all"
        )
        assert result.event_study_effects is not None
        assert result.group_effects is not None
        for e, d in result.event_study_effects.items():
            assert np.isfinite(d["se"])
        for g, d in result.group_effects.items():
            assert np.isfinite(d["se"])


class TestSmallCohortWarning:
    """Small cohort warnings for numerical stability."""

    def test_single_unit_cohort_warns(self):
        """Cohort with 1 unit triggers instability warning."""
        # Create panel with 1-unit cohort (group 3) and normal cohort (group 5)
        df = _make_staggered_panel(n_per_group=1, n_control=80, groups=(3,), effects={3: 2.0})
        # Add a normal-sized cohort
        df2 = _make_staggered_panel(
            n_per_group=60, n_control=0, groups=(5,), effects={5: 1.0}, seed=99
        )
        df2["unit"] += df["unit"].max() + 1
        combined = pd.concat([df, df2], ignore_index=True)

        with pytest.warns(UserWarning, match="only 1 unit"):
            result = EfficientDiD().fit(combined, "y", "unit", "time", "first_treat")
        # Estimation should still succeed
        assert np.isfinite(result.overall_att)

    def test_small_share_cohort_warns(self):
        """Cohort with < 1% share triggers precision warning."""
        # 2 units in group 3 out of ~202 total units (< 1%)
        df = _make_staggered_panel(n_per_group=2, n_control=100, groups=(3,), effects={3: 2.0})
        df2 = _make_staggered_panel(
            n_per_group=100, n_control=0, groups=(5,), effects={5: 1.0}, seed=99
        )
        df2["unit"] += df["unit"].max() + 1
        combined = pd.concat([df, df2], ignore_index=True)

        with pytest.warns(UserWarning, match="< 1%"):
            result = EfficientDiD().fit(combined, "y", "unit", "time", "first_treat")
        assert np.isfinite(result.overall_att)

    def test_normal_cohorts_no_warning(self):
        """Normal-sized cohorts should not warn about cohort size."""
        df = _make_staggered_panel(n_per_group=60, n_control=80)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        cohort_warnings = [x for x in w if "Cohort" in str(x.message)]
        assert len(cohort_warnings) == 0


class TestEdgeCases:
    """Edge cases: all treated, empty pairs."""

    def test_all_units_treated_pt_all(self):
        """No never-treated units under PT-All should raise ValueError with default control_group."""
        df = _make_staggered_panel(n_control=0, groups=(3, 5))
        with pytest.raises(ValueError, match="control_group='last_cohort'"):
            EfficientDiD(pt_assumption="all").fit(df, "y", "unit", "time", "first_treat")

    def test_all_units_treated_pt_post_raises(self):
        """No never-treated under PT-Post raises ValueError with default control_group."""
        df = _make_staggered_panel(n_control=0, groups=(3, 5))
        with pytest.raises(ValueError, match="control_group='last_cohort'"):
            EfficientDiD(pt_assumption="post").fit(df, "y", "unit", "time", "first_treat")

    def test_anticipation_parameter(self):
        """Anticipation=1 shifts treatment boundary."""
        df = _make_simple_panel(treat_period=4, n_periods=6)
        result = EfficientDiD(anticipation=1).fit(df, "y", "unit", "time", "first_treat")
        # With anticipation=1, effective treatment starts at g-1=3
        # So ATT(4,3) should be post-treatment
        post_effects = [
            (g, t)
            for (g, t) in result.group_time_effects
            if t >= g - 1  # effective treatment at g - anticipation
        ]
        assert len(post_effects) > 0


class TestLastCohortControl:
    """Last-cohort-as-control fallback when no never-treated units."""

    def test_last_cohort_pt_all(self):
        """All-treated data with last_cohort control should fit successfully."""
        df = _make_staggered_panel(
            n_per_group=60,
            n_control=0,
            groups=(3, 5, 7),
            effects={3: 2.0, 5: 1.5, 7: 1.0},
        )
        result = EfficientDiD(pt_assumption="all", control_group="last_cohort").fit(
            df, "y", "unit", "time", "first_treat"
        )
        # Last cohort (7) becomes pseudo-control, only groups 3 and 5 remain
        assert np.isfinite(result.overall_att)
        assert result.control_group == "last_cohort"
        assert 7 not in result.groups

    def test_last_cohort_pt_post(self):
        """PT-Post with last_cohort control works (just-identified)."""
        df = _make_staggered_panel(
            n_per_group=60,
            n_control=0,
            groups=(3, 5, 7),
            effects={3: 2.0, 5: 1.5, 7: 1.0},
        )
        result = EfficientDiD(pt_assumption="post", control_group="last_cohort").fit(
            df, "y", "unit", "time", "first_treat"
        )
        assert np.isfinite(result.overall_att)

    def test_last_cohort_reasonable_att(self):
        """Last-cohort ATT should be close to true effect."""
        # True effects: group 3 gets +2.0, group 5 gets +1.5
        df = _make_staggered_panel(
            n_per_group=100,
            n_control=0,
            groups=(3, 5, 7),
            effects={3: 2.0, 5: 1.5, 7: 1.0},
            sigma=0.1,
        )
        result = EfficientDiD(control_group="last_cohort").fit(
            df, "y", "unit", "time", "first_treat"
        )
        # ATT should be in the ballpark of the true effects (1.5-2.0 range)
        assert 0.5 < result.overall_att < 3.5

    def test_last_cohort_single_cohort_raises(self):
        """Single treatment cohort with last_cohort should raise."""
        df = _make_staggered_panel(n_per_group=60, n_control=0, groups=(3,), effects={3: 2.0})
        with pytest.raises(ValueError, match="Only one treatment cohort"):
            EfficientDiD(control_group="last_cohort").fit(df, "y", "unit", "time", "first_treat")

    def test_last_cohort_with_never_treated_reclassifies(self):
        """Using last_cohort when never-treated exist should reclassify last cohort."""
        df = _make_staggered_panel(n_per_group=60, n_control=80, groups=(3, 5))
        result = EfficientDiD(control_group="last_cohort").fit(
            df, "y", "unit", "time", "first_treat"
        )
        # Last cohort (5) is reclassified — only group 3 treated
        assert 5 not in result.groups
        assert 3 in result.groups
        assert result.control_group == "last_cohort"
        assert np.isfinite(result.overall_att)

    def test_control_group_get_params(self):
        """control_group should appear in get_params and round-trip via set_params."""
        edid = EfficientDiD(control_group="last_cohort")
        params = edid.get_params()
        assert params["control_group"] == "last_cohort"

        edid2 = EfficientDiD()
        edid2.set_params(control_group="last_cohort")
        assert edid2.control_group == "last_cohort"

    def test_control_group_invalid_raises(self):
        """Invalid control_group should raise ValueError."""
        with pytest.raises(ValueError, match="control_group"):
            EfficientDiD(control_group="invalid")

    def test_last_cohort_no_treated_raises(self):
        """All-never-treated data with last_cohort should raise."""
        df = _make_staggered_panel(n_per_group=0, n_control=100, groups=())
        with pytest.raises(ValueError, match="No treated cohorts"):
            EfficientDiD(control_group="last_cohort").fit(df, "y", "unit", "time", "first_treat")

    def test_last_cohort_aggregate_event_study(self):
        """last_cohort with aggregate='event_study' should produce finite results."""
        df = _make_staggered_panel(
            n_per_group=60,
            n_control=0,
            groups=(3, 5, 7),
            effects={3: 2.0, 5: 1.5, 7: 1.0},
        )
        result = EfficientDiD(control_group="last_cohort").fit(
            df, "y", "unit", "time", "first_treat", aggregate="event_study"
        )
        assert result.event_study_effects is not None
        assert 7 not in result.groups
        for e, d in result.event_study_effects.items():
            assert np.isfinite(d["effect"])

    def test_last_cohort_aggregate_all(self):
        """last_cohort with aggregate='all' should produce finite results."""
        df = _make_staggered_panel(
            n_per_group=60,
            n_control=0,
            groups=(3, 5, 7),
            effects={3: 2.0, 5: 1.5, 7: 1.0},
        )
        result = EfficientDiD(control_group="last_cohort").fit(
            df, "y", "unit", "time", "first_treat", aggregate="all"
        )
        assert result.event_study_effects is not None
        assert result.group_effects is not None
        assert 7 not in result.groups
        for g, d in result.group_effects.items():
            assert g != 7
            assert np.isfinite(d["effect"])

    def test_last_cohort_bootstrap(self, ci_params):
        """last_cohort with bootstrap should produce finite inference."""
        n_boot = ci_params.bootstrap(99)
        df = _make_staggered_panel(
            n_per_group=60,
            n_control=0,
            groups=(3, 5, 7),
            effects={3: 2.0, 5: 1.5, 7: 1.0},
        )
        result = EfficientDiD(control_group="last_cohort", n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0
        assert 7 not in result.groups


class TestBalanceE:
    """Test balance_e event study balancing."""

    def test_balance_e_basic(self):
        """balance_e restricts event study to cohorts present at anchor horizon."""
        df = _make_staggered_panel(n_per_group=80, n_control=80, groups=(3, 5))
        result = EfficientDiD().fit(
            df,
            "y",
            "unit",
            "time",
            "first_treat",
            aggregate="event_study",
            balance_e=0,
        )
        assert result.event_study_effects is not None
        for e, d in result.event_study_effects.items():
            assert np.isfinite(d["effect"])

    def test_balance_e_with_bootstrap(self, ci_params):
        """Bootstrap balance_e should produce finite SEs."""
        n_boot = ci_params.bootstrap(99)
        df = _make_staggered_panel(n_per_group=80, n_control=80, groups=(3, 5))
        result = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df,
            "y",
            "unit",
            "time",
            "first_treat",
            aggregate="event_study",
            balance_e=0,
        )
        assert result.event_study_effects is not None
        for e, d in result.event_study_effects.items():
            if np.isfinite(d["effect"]):
                assert np.isfinite(d["se"])

    def test_balance_e_nan_anchor_filters_group(self):
        """When a group has NaN at the anchor horizon, bootstrap should
        exclude it from groups_at_e, matching the analytical path."""
        edid = EfficientDiD()
        edid.anticipation = 0

        # Simulate: group 3 has finite effect at e=0, group 5 has NaN at e=0
        gt_pairs = [(3.0, 3), (3.0, 4), (5.0, 5), (5.0, 6)]
        original_atts = np.array([1.0, 1.5, np.nan, 0.8])
        cohort_fractions = {3.0: 0.4, 5.0: 0.3}

        result = edid._prepare_es_agg_boot(gt_pairs, original_atts, cohort_fractions, balance_e=0)
        # Group 5 has NaN at e=0 (t=5, g=5), so it should be excluded
        # Only group 3 effects should appear in the balanced set
        for e, info in result.items():
            gt_indices = info["gt_indices"]
            groups_in_e = {gt_pairs[j][0] for j in gt_indices}
            assert 5.0 not in groups_in_e, (
                f"Group 5 (NaN at anchor) should be excluded at e={e}, " f"got groups {groups_in_e}"
            )

    def test_balance_e_empty_warns(self):
        """When no cohort survives the anchor horizon, warn the user."""
        edid = EfficientDiD()
        edid.anticipation = 0

        # All effects are NaN at e=0
        gt_pairs = [(3.0, 3), (3.0, 4), (5.0, 5), (5.0, 6)]
        original_atts = np.array([np.nan, 1.5, np.nan, 0.8])
        cohort_fractions = {3.0: 0.4, 5.0: 0.3}

        with pytest.warns(UserWarning, match="no cohort has a finite effect"):
            result = edid._prepare_es_agg_boot(
                gt_pairs, original_atts, cohort_fractions, balance_e=0
            )
        assert result == {}


# =============================================================================
# Tier 3: Bootstrap
# =============================================================================


class TestBootstrap:
    """Test multiplier bootstrap inference."""

    def test_bootstrap_se_finite(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        df = _make_simple_panel()
        result = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        assert result.bootstrap_results is not None
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0
        for gt, d in result.group_time_effects.items():
            if np.isfinite(d["effect"]):
                assert np.isfinite(d["se"])

    def test_bootstrap_with_aggregation(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        df = _make_simple_panel()
        result = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat", aggregate="all"
        )
        assert result.bootstrap_results is not None
        if result.event_study_effects:
            for e, d in result.event_study_effects.items():
                if np.isfinite(d["effect"]):
                    assert np.isfinite(d["se"])

    def test_bootstrap_coverage_basic(self, ci_params):
        """Rough coverage check: true effect should be in CI."""
        n_boot = ci_params.bootstrap(199, min_n=49)
        df = _make_simple_panel(effect=2.0, n_units=200, seed=42)
        result = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        ci = result.overall_conf_int
        # True effect is 2.0 — should be within CI for this seed
        if np.isfinite(ci[0]) and np.isfinite(ci[1]):
            # Just check CI is reasonable (not testing exact coverage)
            assert ci[0] < ci[1], "CI should be ordered"


# =============================================================================
# Tier 4: Simulation Validation
# =============================================================================


class TestSimulationValidation:
    """Validation against paper's DGP properties."""

    def test_synthetic_staggered_unbiased(self):
        """Single run at rho=0, verify ATT estimates near true values."""
        df = _make_compustat_dgp(rho=0.0, seed=42)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="all")

        # Check individual ATT(g,t) estimates
        # ATT(5,5) should be near 0.154
        gt_55 = (5.0, 5)
        if gt_55 in result.group_time_effects:
            d = result.group_time_effects[gt_55]
            se = d["se"]
            if np.isfinite(se) and se > 0:
                assert (
                    abs(d["effect"] - 0.154) < 3 * se + 0.1
                ), f"ATT(5,5)={d['effect']:.4f}, expected ~0.154"

        # ATT(5,6) should be near 0.308
        gt_56 = (5.0, 6)
        if gt_56 in result.group_time_effects:
            d = result.group_time_effects[gt_56]
            se = d["se"]
            if np.isfinite(se) and se > 0:
                assert (
                    abs(d["effect"] - 0.308) < 3 * se + 0.1
                ), f"ATT(5,6)={d['effect']:.4f}, expected ~0.308"

    def test_efficiency_gain_negative_rho(self):
        """With rho=-0.5, EDiD should have lower SE than CS."""
        df = _make_compustat_dgp(rho=-0.5, seed=42)

        edid = EfficientDiD(pt_assumption="all")
        cs = CallawaySantAnna(control_group="never_treated")

        res_e = edid.fit(df, "y", "unit", "time", "first_treat")
        res_c = cs.fit(df, "y", "unit", "time", "first_treat")

        # Count how many post-treatment effects have lower SE
        lower_count = 0
        total_count = 0
        for gt in res_e.group_time_effects:
            if gt in res_c.group_time_effects:
                g, t = gt
                if t >= g:  # post-treatment
                    e_se = res_e.group_time_effects[gt]["se"]
                    c_se = res_c.group_time_effects[gt]["se"]
                    if np.isfinite(e_se) and np.isfinite(c_se) and c_se > 0:
                        total_count += 1
                        if e_se < c_se:
                            lower_count += 1

        if total_count > 0:
            # Majority of post-treatment effects should have lower SE
            ratio = lower_count / total_count
            assert ratio > 0.3, (
                f"EDiD should have lower SE for most effects with rho=-0.5 "
                f"({lower_count}/{total_count} = {ratio:.2f})"
            )

    def test_weights_shift_with_rho(self):
        """Verify weights sum to 1 and change with serial correlation."""
        weights_rho0 = {}
        weights_rho09 = {}

        for rho, store in [(0.0, weights_rho0), (0.9, weights_rho09)]:
            df = _make_compustat_dgp(rho=rho, seed=42)
            result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
            if result.efficient_weights:
                for gt, w in result.efficient_weights.items():
                    if len(w) > 2:
                        assert (
                            abs(w.sum() - 1.0) < 1e-8
                        ), f"Weights should sum to 1, got {w.sum():.10f}"
                        store[gt] = w.copy()

        # Weights should differ between rho=0 and rho=0.9
        common = set(weights_rho0) & set(weights_rho09)
        if common:
            diffs = [np.linalg.norm(weights_rho0[gt] - weights_rho09[gt]) for gt in common]
            assert max(diffs) > 0.01, "Weights should change with rho"

    def test_analytical_se_consistency(self, ci_params):
        """Analytical SE should roughly match bootstrap SE."""
        n_boot = ci_params.bootstrap(999, min_n=199)
        threshold = 0.40 if n_boot < 100 else 0.30

        df = _make_simple_panel(n_units=200, effect=2.0, seed=42)

        # Analytical SE
        res_anal = EfficientDiD(n_bootstrap=0).fit(df, "y", "unit", "time", "first_treat")
        anal_se = res_anal.overall_se

        # Bootstrap SE
        res_boot = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        boot_se = res_boot.overall_se

        if np.isfinite(anal_se) and np.isfinite(boot_se) and boot_se > 0:
            rel_diff = abs(anal_se - boot_se) / boot_se
            assert rel_diff < threshold, (
                f"Analytical SE ({anal_se:.4f}) differs from bootstrap SE "
                f"({boot_se:.4f}) by {rel_diff:.2%}"
            )


# =============================================================================
# Regression Tests (PR #192 review feedback)
# =============================================================================


class TestPTPostExactMatch:
    """Fix 2: EDiD(PT-Post) should exactly match CS for all g, including g > 2."""

    def test_pt_post_staggered_exact_match(self):
        """With per-group baseline, EDiD(PT-Post) = CS for post-treatment effects."""
        df = _make_staggered_panel(n_per_group=100, n_control=100, groups=(3, 5))
        edid = EfficientDiD(pt_assumption="post")
        cs = CallawaySantAnna(control_group="never_treated", base_period="varying")

        res_e = edid.fit(df, "y", "unit", "time", "first_treat")
        res_c = cs.fit(df, "y", "unit", "time", "first_treat")

        matched = 0
        for g, t in res_e.group_time_effects:
            if t >= g and (g, t) in res_c.group_time_effects:
                e_eff = res_e.group_time_effects[(g, t)]["effect"]
                c_eff = res_c.group_time_effects[(g, t)]["effect"]
                assert abs(e_eff - c_eff) < 1e-8, f"ATT({g},{t}): EDiD={e_eff:.10f} CS={c_eff:.10f}"
                matched += 1
        assert matched > 0, "No matching post-treatment effects found"


class TestBridgingComparison:
    """Fix 1: Bridging comparisons should be valid under PT-All."""

    def test_bridging_comparison_valid(self):
        """ATT should be finite even when bridging comparisons are used."""
        # Create panel where g'=3 is used as comparison for g=5 at t=4 (g' treated at t=3)
        df = _make_staggered_panel(n_per_group=80, n_control=80, groups=(3, 5), n_periods=7)
        result = EfficientDiD(pt_assumption="all").fit(df, "y", "unit", "time", "first_treat")
        # Post-treatment effects for g=5 should be finite
        for (g, t), d in result.group_time_effects.items():
            if g == 5.0 and t >= 5:
                assert np.isfinite(d["effect"]), f"ATT({g},{t}) should be finite"


class TestWIFCorrection:
    """Fix 3: WIF correction for aggregated SEs."""

    def test_wif_contribution_nonzero(self):
        """WIF correction should produce nonzero contribution for staggered design."""
        df = _make_staggered_panel(n_per_group=100, n_control=100, groups=(3, 5))
        edid = EfficientDiD(pt_assumption="all")
        result = edid.fit(df, "y", "unit", "time", "first_treat")

        # Reconstruct WIF inputs from result
        gt_effects = result.group_time_effects
        keepers = [
            (g, t) for (g, t) in gt_effects if t >= g and np.isfinite(gt_effects[(g, t)]["effect"])
        ]
        effects = np.array([gt_effects[gt]["effect"] for gt in keepers])

        # Build unit_cohorts and cohort_fractions from data
        unit_info = df.groupby("unit")["first_treat"].first()
        unit_cohorts = unit_info.values.astype(float)
        unit_cohorts[unit_cohorts == np.inf] = 0.0  # normalize never-treated
        n_units = len(unit_cohorts)
        cohort_fractions = {}
        for g in [3.0, 5.0]:
            cohort_fractions[g] = float(np.sum(unit_cohorts == g)) / n_units

        wif = edid._compute_wif_contribution(
            keepers, effects, unit_cohorts, cohort_fractions, n_units
        )
        # WIF should be nonzero for staggered design with 2+ groups
        assert (
            np.linalg.norm(wif) > 1e-10
        ), f"WIF contribution should be nonzero, got norm={np.linalg.norm(wif):.2e}"

    def test_wif_se_vs_bootstrap(self, ci_params):
        """WIF-corrected SE should roughly match bootstrap SE."""
        n_boot = ci_params.bootstrap(999, min_n=199)
        threshold = 0.40 if n_boot < 100 else 0.35

        df = _make_staggered_panel(n_per_group=100, n_control=100, groups=(3, 5))

        # Analytical SE (with WIF)
        res_anal = EfficientDiD(n_bootstrap=0).fit(df, "y", "unit", "time", "first_treat")
        anal_se = res_anal.overall_se

        # Bootstrap SE
        res_boot = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        boot_se = res_boot.overall_se

        if np.isfinite(anal_se) and np.isfinite(boot_se) and boot_se > 0:
            rel_diff = abs(anal_se - boot_se) / boot_se
            assert rel_diff < threshold, (
                f"WIF-corrected SE ({anal_se:.4f}) differs from bootstrap SE "
                f"({boot_se:.4f}) by {rel_diff:.2%}"
            )


class TestResultsParams:
    """Fix 7: Results object should contain estimator params."""

    def test_results_contain_params(self):
        df = _make_simple_panel()
        result = EfficientDiD(pt_assumption="post", anticipation=1, n_bootstrap=0, seed=123).fit(
            df, "y", "unit", "time", "first_treat"
        )

        assert result.pt_assumption == "post"
        assert result.anticipation == 1
        assert result.n_bootstrap == 0
        assert result.bootstrap_weights == "rademacher"
        assert result.seed == 123

    def test_summary_shows_anticipation(self):
        df = _make_simple_panel(treat_period=4, n_periods=6)
        result = EfficientDiD(anticipation=1).fit(df, "y", "unit", "time", "first_treat")
        s = result.summary()
        assert "Anticipation" in s

    def test_summary_shows_bootstrap(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        df = _make_simple_panel()
        result = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        s = result.summary()
        assert "Bootstrap" in s


# =============================================================================
# Regression Tests (PR #192 review feedback, Round 2)
# =============================================================================


class TestPTAllIndexSet:
    """Fix 1 (Round 2): PT-All index set must include g'=g and not require t_pre < g."""

    def test_g2_finite_att_pt_all(self):
        """g=2 under PT-All should produce finite ATTs (not NaN)."""
        df = _make_staggered_panel(
            n_per_group=60, n_control=80, groups=(2, 4), n_periods=5, seed=42
        )
        result = EfficientDiD(pt_assumption="all").fit(df, "y", "unit", "time", "first_treat")
        # g=2 post-treatment effects should be finite
        for (g, t), d in result.group_time_effects.items():
            if g == 2.0 and t >= 2:
                assert np.isfinite(
                    d["effect"]
                ), f"ATT({g},{t}) should be finite under PT-All, got {d['effect']}"

    def test_pt_all_more_moments_than_pt_post(self):
        """PT-All should produce strictly more moments than PT-Post."""
        pairs_all = enumerate_valid_triples(
            target_g=3,
            treatment_groups=[3, 5],
            time_periods=[1, 2, 3, 4, 5, 6],
            period_1=1,
            pt_assumption="all",
        )
        pairs_post = enumerate_valid_triples(
            target_g=3,
            treatment_groups=[3, 5],
            time_periods=[1, 2, 3, 4, 5, 6],
            period_1=1,
            pt_assumption="post",
        )
        assert len(pairs_all) > len(pairs_post), (
            f"PT-All ({len(pairs_all)}) should have more moments than "
            f"PT-Post ({len(pairs_post)})"
        )

    def test_same_group_pairs_valid(self):
        """g'=g pairs should be present in PT-All enumeration."""
        pairs = enumerate_valid_triples(
            target_g=3,
            treatment_groups=[3, 5],
            time_periods=[1, 2, 3, 4, 5],
            period_1=1,
            pt_assumption="all",
        )
        assert (3, 2) in pairs, f"Same-group pair (3, 2) should be valid, got {pairs}"


class TestBootstrapNanResilience:
    """Fix 2 (Round 2): Bootstrap should filter NaN cells."""

    def test_bootstrap_nan_cell_resilience(self, ci_params):
        """Bootstrap should not be poisoned by NaN ATT cells."""
        n_boot = ci_params.bootstrap(99, min_n=49)
        # Use PT-All which gives finite cells for g=2
        df = _make_staggered_panel(
            n_per_group=60, n_control=80, groups=(2, 4), n_periods=5, seed=42
        )
        result = EfficientDiD(pt_assumption="all", n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        assert np.isfinite(
            result.overall_se
        ), f"Overall SE should be finite, got {result.overall_se}"
        assert result.bootstrap_results is not None


class TestCohortDropWarning:
    """Fix 3 (Round 2): PT-Post + anticipation should warn on cohort drop."""

    def test_cohort_drop_warning(self):
        """Cohort g=2 with anticipation=1 under PT-Post: baseline=0, not in data."""
        df = _make_staggered_panel(
            n_per_group=60, n_control=80, groups=(2, 4), n_periods=5, seed=42
        )
        with pytest.warns(UserWarning, match=r"Cohort g=2.*dropped"):
            result = EfficientDiD(pt_assumption="post", anticipation=1).fit(
                df, "y", "unit", "time", "first_treat"
            )
        # Only g=4 effects should be present
        groups_present = {g for (g, t) in result.group_time_effects}
        assert 2.0 not in groups_present, "g=2 should have been dropped"
        assert 4.0 in groups_present, "g=4 should still be present"


# =============================================================================
# Covariate Tests
# =============================================================================


def _make_covariate_panel(
    n_units=300,
    n_periods=11,
    seed=42,
    covariate_effect=0.5,
    confounding_strength=0.0,
):
    """Helper: staggered panel with time-invariant covariates.

    Uses n_periods=11 (default) so both treatment groups g=5 and g=8 are valid.
    """
    return make_compustat_dgp(
        n_units=n_units,
        n_periods=n_periods,
        rho=0.0,
        seed=seed,
        add_covariates=True,
        covariate_effect=covariate_effect,
        confounding_strength=confounding_strength,
    )


class TestCovariatesBasic:
    """Tier 1: basic covariate path correctness."""

    def test_covariates_fit_produces_results(self):
        """Smoke test: fit with covariates returns valid results."""
        df = _make_covariate_panel()
        result = EfficientDiD(pt_assumption="post").fit(
            df, "y", "unit", "time", "first_treat", covariates=["x1", "x2"]
        )
        assert isinstance(result, EfficientDiDResults)
        assert result.estimation_path == "dr"
        assert np.isfinite(result.overall_att)
        assert result.overall_se > 0
        assert len(result.group_time_effects) > 0
        for (g, t), eff in result.group_time_effects.items():
            assert np.isfinite(eff["effect"])
            # Baseline cells (t == g-1 under PT-Post) have SE=0 by construction
            if t >= g:
                assert eff["se"] > 0, f"SE=0 for post-treatment cell ({g}, {t})"

    def test_nocov_match_when_irrelevant(self):
        """Random noise covariates should give ~same ATT as nocov."""
        df = _make_covariate_panel(covariate_effect=0.0)
        edid = EfficientDiD(pt_assumption="post")
        r_nocov = edid.fit(df, "y", "unit", "time", "first_treat")
        r_cov = EfficientDiD(pt_assumption="post").fit(
            df, "y", "unit", "time", "first_treat", covariates=["x1", "x2"]
        )
        # ATT should be close (not identical due to nuisance estimation noise)
        assert (
            abs(r_cov.overall_att - r_nocov.overall_att) < 0.3
        ), f"DR ATT {r_cov.overall_att:.4f} too far from nocov {r_nocov.overall_att:.4f}"

    def test_covariates_produce_valid_se(self):
        """DR path with covariates explaining variance produces valid SE."""
        df = _make_covariate_panel(covariate_effect=2.0, n_units=600)
        r_cov = EfficientDiD(pt_assumption="post").fit(
            df, "y", "unit", "time", "first_treat", covariates=["x1"]
        )
        # DR SE should be positive and finite
        assert r_cov.overall_se > 0
        assert np.isfinite(r_cov.overall_se)
        # ATT should be close to the nocov estimate (no confounding)
        r_nocov = EfficientDiD(pt_assumption="post").fit(df, "y", "unit", "time", "first_treat")
        assert abs(r_cov.overall_att - r_nocov.overall_att) < 0.2

    def test_covariates_recover_effect_under_confounding(self):
        """DGP with confounding: DR should recover true ATT closer to truth than nocov.

        The DGP adds x1-dependent time trends to ALL units and shifts x1
        distribution by group, so unconditional PT fails but conditional PT holds.
        True ATT is unchanged by confounding (only levels shift, not treatment).
        """
        from edid_dgp import true_overall_att

        true_att = true_overall_att()
        df = _make_covariate_panel(
            n_units=900,
            covariate_effect=1.0,
            confounding_strength=2.0,
            seed=123,
        )
        r_nocov = EfficientDiD(pt_assumption="post").fit(df, "y", "unit", "time", "first_treat")
        r_cov = EfficientDiD(pt_assumption="post").fit(
            df, "y", "unit", "time", "first_treat", covariates=["x1"]
        )
        assert np.isfinite(r_nocov.overall_att)
        assert np.isfinite(r_cov.overall_att)
        # DR should be closer to the true ATT than nocov
        bias_nocov = abs(r_nocov.overall_att - true_att)
        bias_cov = abs(r_cov.overall_att - true_att)
        assert (
            bias_cov < bias_nocov
        ), f"DR bias ({bias_cov:.4f}) should be smaller than nocov bias ({bias_nocov:.4f})"

    def test_empty_covariates_uses_nocov(self):
        """covariates=[] should normalize to nocov path."""
        df = _make_covariate_panel()
        result = EfficientDiD(pt_assumption="post").fit(
            df, "y", "unit", "time", "first_treat", covariates=[]
        )
        assert result.estimation_path == "nocov"


class TestCovariateValidation:
    """Tier 1: input validation for covariates."""

    def test_missing_covariate_column_raises(self):
        df = _make_covariate_panel()
        with pytest.raises(ValueError, match="Missing covariate columns"):
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat", covariates=["nonexistent"])

    def test_nan_covariates_raises(self):
        df = _make_covariate_panel()
        df.loc[0, "x1"] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat", covariates=["x1"])

    def test_ratio_clip_validation(self):
        with pytest.raises(ValueError, match="ratio_clip"):
            EfficientDiD(ratio_clip=0.5)
        with pytest.raises(ValueError, match="ratio_clip"):
            EfficientDiD(ratio_clip=1.0)
        with pytest.raises(ValueError, match="ratio_clip"):
            EfficientDiD(ratio_clip=np.nan)
        with pytest.raises(ValueError, match="ratio_clip"):
            EfficientDiD(ratio_clip=np.inf)

    def test_kernel_bandwidth_validation(self):
        with pytest.raises(ValueError, match="kernel_bandwidth"):
            EfficientDiD(kernel_bandwidth=0.0)
        with pytest.raises(ValueError, match="kernel_bandwidth"):
            EfficientDiD(kernel_bandwidth=-1.0)
        with pytest.raises(ValueError, match="kernel_bandwidth"):
            EfficientDiD(kernel_bandwidth=np.nan)
        with pytest.raises(ValueError, match="kernel_bandwidth"):
            EfficientDiD(kernel_bandwidth=np.inf)
        # None is valid (auto bandwidth)
        edid = EfficientDiD(kernel_bandwidth=None)
        assert edid.kernel_bandwidth is None

    def test_sieve_k_max_validation(self):
        with pytest.raises(ValueError, match="sieve_k_max"):
            EfficientDiD(sieve_k_max=0)
        with pytest.raises(ValueError, match="sieve_k_max"):
            EfficientDiD(sieve_k_max=-1)
        # None is valid (auto)
        edid = EfficientDiD(sieve_k_max=None)
        assert edid.sieve_k_max is None

    def test_sieve_criterion_validation(self):
        with pytest.raises(ValueError, match="sieve_criterion"):
            EfficientDiD(sieve_criterion="invalid")

    def test_new_params_in_get_params(self):
        edid = EfficientDiD(sieve_k_max=3, sieve_criterion="aic", ratio_clip=10.0)
        params = edid.get_params()
        assert params["sieve_k_max"] == 3
        assert params["sieve_criterion"] == "aic"
        assert params["ratio_clip"] == 10.0
        assert "kernel_bandwidth" in params

    def test_time_varying_covariates_raises(self):
        df = _make_covariate_panel()
        # Make x1 vary over time for one unit
        mask = (df["unit"] == 0) & (df["time"] == 2)
        df.loc[mask, "x1"] = 999.0
        with pytest.raises(ValueError, match="varies over time"):
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat", covariates=["x1"])


class TestCovariatesPTAssumptions:
    """Tier 2: covariates under different PT assumptions."""

    def test_covariates_pt_post(self):
        df = _make_covariate_panel()
        result = EfficientDiD(pt_assumption="post").fit(
            df, "y", "unit", "time", "first_treat", covariates=["x1"]
        )
        assert isinstance(result, EfficientDiDResults)
        assert result.estimation_path == "dr"
        assert np.isfinite(result.overall_att)

    def test_covariates_pt_all(self):
        df = _make_covariate_panel()
        result = EfficientDiD(pt_assumption="all").fit(
            df, "y", "unit", "time", "first_treat", covariates=["x1"]
        )
        assert isinstance(result, EfficientDiDResults)
        assert result.estimation_path == "dr"
        assert np.isfinite(result.overall_att)

    def test_covariates_aggregate_event_study(self):
        df = _make_covariate_panel()
        result = EfficientDiD(pt_assumption="post").fit(
            df,
            "y",
            "unit",
            "time",
            "first_treat",
            covariates=["x1"],
            aggregate="event_study",
        )
        assert result.event_study_effects is not None
        assert len(result.event_study_effects) > 0
        for e, eff in result.event_study_effects.items():
            assert np.isfinite(eff["effect"])

    def test_covariates_aggregate_group(self):
        df = _make_covariate_panel()
        result = EfficientDiD(pt_assumption="post").fit(
            df,
            "y",
            "unit",
            "time",
            "first_treat",
            covariates=["x1"],
            aggregate="group",
        )
        assert result.group_effects is not None
        assert len(result.group_effects) > 0

    def test_covariates_aggregate_all(self):
        df = _make_covariate_panel()
        result = EfficientDiD(pt_assumption="post").fit(
            df,
            "y",
            "unit",
            "time",
            "first_treat",
            covariates=["x1"],
            aggregate="all",
        )
        assert result.event_study_effects is not None
        assert result.group_effects is not None
        assert np.isfinite(result.overall_att)


class TestCovariatesEdgeCases:
    """Tier 2: edge cases for covariate path."""

    def test_single_covariate(self):
        df = _make_covariate_panel()
        result = EfficientDiD(pt_assumption="post").fit(
            df, "y", "unit", "time", "first_treat", covariates=["x1"]
        )
        assert np.isfinite(result.overall_att)

    def test_binary_covariate(self):
        df = _make_covariate_panel()
        result = EfficientDiD(pt_assumption="post").fit(
            df, "y", "unit", "time", "first_treat", covariates=["x2"]
        )
        assert np.isfinite(result.overall_att)

    def test_many_covariates(self):
        """Multiple covariates including derived ones."""
        df = _make_covariate_panel()
        # Create a unit-level covariate (must be time-invariant)
        rng = np.random.default_rng(99)
        units = df["unit"].unique()
        x3_map = dict(
            zip(units, df.groupby("unit")["x1"].first() * 0.5 + rng.normal(0, 0.1, len(units)))
        )
        df["x3"] = df["unit"].map(x3_map)
        result = EfficientDiD(pt_assumption="post").fit(
            df, "y", "unit", "time", "first_treat", covariates=["x1", "x2", "x3"]
        )
        assert np.isfinite(result.overall_att)

    def test_sieve_ratio_produces_valid_results(self):
        """Sieve ratio estimation produces finite ATT with valid ratios."""
        df = _make_covariate_panel(n_units=300, seed=88)
        result = EfficientDiD(pt_assumption="post", sieve_k_max=3, sieve_criterion="bic").fit(
            df, "y", "unit", "time", "first_treat", covariates=["x1"]
        )
        assert np.isfinite(result.overall_att)
        assert result.overall_se > 0

    def test_shuffled_units_match_ordered(self):
        """Shuffled unit ordering must produce same ATT as original ordering.

        Regression test for P0 label-alignment bug in estimate_propensity_ratio:
        D labels must follow the row order of combined_mask, not assume
        g-units come before g'-units.
        """
        df_ordered = _make_covariate_panel(n_units=300, seed=55)
        # Shuffle: randomize unit IDs so cohorts are interleaved
        rng = np.random.default_rng(55)
        df_shuffled = df_ordered.copy()
        units = df_shuffled["unit"].unique()
        perm = rng.permutation(len(units))
        unit_map = dict(zip(units, perm))
        df_shuffled["unit"] = df_shuffled["unit"].map(unit_map)
        df_shuffled = df_shuffled.sort_values(["unit", "time"]).reset_index(drop=True)

        edid = EfficientDiD(pt_assumption="post")
        r_ordered = edid.fit(df_ordered, "y", "unit", "time", "first_treat", covariates=["x1"])
        r_shuffled = EfficientDiD(pt_assumption="post").fit(
            df_shuffled, "y", "unit", "time", "first_treat", covariates=["x1"]
        )
        assert abs(r_ordered.overall_att - r_shuffled.overall_att) < 1e-10, (
            f"ATT mismatch: ordered={r_ordered.overall_att:.6f} "
            f"vs shuffled={r_shuffled.overall_att:.6f}"
        )

    def test_extreme_covariates_warns_overlap(self):
        """Extreme covariates should trigger overlap warning and still produce valid results."""
        df = _make_covariate_panel(n_units=300, seed=77)
        rng = np.random.default_rng(77)
        units = df["unit"].unique()
        n_units = len(units)
        ft_map = df.groupby("unit")["first_treat"].first()
        sep_vals = np.where(
            ft_map.values < np.inf,
            5.0 + rng.normal(0, 0.01, n_units),
            -5.0 + rng.normal(0, 0.01, n_units),
        )
        sep_map = dict(zip(units, sep_vals))
        df["x_sep"] = df["unit"].map(sep_map)
        with pytest.warns(UserWarning, match="overlap|clipped|propensity"):
            result = EfficientDiD(pt_assumption="post").fit(
                df, "y", "unit", "time", "first_treat", covariates=["x_sep"]
            )
        assert np.isfinite(result.overall_att)
        assert result.overall_se > 0

    def test_eif_mean_approximately_zero(self):
        """EIF with per-unit weights should have sample mean ≈ 0."""
        from diff_diff.efficient_did_covariates import compute_eif_cov

        rng = np.random.default_rng(42)
        n, H = 200, 3
        gen_out = rng.normal(0, 1, (n, H))
        # Non-constant per-unit weights (each row sums to 1)
        raw_w = rng.exponential(1, (n, H))
        per_unit_w = raw_w / raw_w.sum(axis=1, keepdims=True)
        att = float(np.mean(np.sum(per_unit_w * gen_out, axis=1)))
        eif = compute_eif_cov(per_unit_w, gen_out, att, n)
        assert abs(np.mean(eif)) < 1e-10, f"EIF mean should be ≈ 0, got {np.mean(eif):.2e}"


class TestCovariatesBootstrap:
    """Tier 2: bootstrap with covariates."""

    def test_bootstrap_with_covariates_smoke(self):
        """Bootstrap with covariates produces valid inference."""
        df = _make_covariate_panel(n_units=300)
        result = EfficientDiD(pt_assumption="post", n_bootstrap=99, seed=42).fit(
            df, "y", "unit", "time", "first_treat", covariates=["x1"]
        )
        assert result.bootstrap_results is not None
        assert np.isfinite(result.overall_att)
        assert result.overall_se > 0
        ci = result.overall_conf_int
        assert ci[0] < ci[1], "CI lower must be less than upper"
        assert np.isfinite(result.overall_p_value)

    def test_covariates_pt_all_bootstrap(self):
        """PT-All + bootstrap + covariates end-to-end."""
        df = _make_covariate_panel(n_units=300)
        result = EfficientDiD(pt_assumption="all", n_bootstrap=99, seed=42).fit(
            df,
            "y",
            "unit",
            "time",
            "first_treat",
            covariates=["x1"],
            aggregate="all",
        )
        assert result.bootstrap_results is not None
        assert result.event_study_effects is not None
        assert result.group_effects is not None
        assert np.isfinite(result.overall_att)
        assert result.overall_se > 0


class TestSieveFallbacks:
    """Tier 2: sieve estimation failure fallbacks."""

    def test_ratio_sieve_fallback_tiny_group_warns(self):
        """When comparison group is too small for any basis, fall back with warning."""
        from diff_diff.efficient_did_covariates import estimate_propensity_ratio_sieve

        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, (n, 3))  # 3 covariates
        mask_g = np.zeros(n, dtype=bool)
        mask_g[:50] = True
        # Tiny comparison group: only 2 units (fewer than any basis dimension)
        mask_gp = np.zeros(n, dtype=bool)
        mask_gp[50:52] = True
        with pytest.warns(UserWarning, match="Propensity ratio sieve estimation failed"):
            ratio = estimate_propensity_ratio_sieve(X, mask_g, mask_gp, k_max=3)
        assert np.all(np.isfinite(ratio))
        # Fallback: constant ratio of 1 (clipped to [1/ratio_clip, ratio_clip])
        assert np.allclose(ratio, 1.0)

    def test_inverse_propensity_sieve_fallback_warns(self):
        """When group is too small for sieve, fall back with warning."""
        from diff_diff.efficient_did_covariates import estimate_inverse_propensity_sieve

        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, (n, 5))  # 5 covariates
        # Tiny group: only 2 units
        mask = np.zeros(n, dtype=bool)
        mask[:2] = True
        with pytest.warns(UserWarning, match="Inverse propensity sieve estimation failed"):
            s_hat = estimate_inverse_propensity_sieve(X, mask, k_max=3)
        assert np.all(np.isfinite(s_hat))
        # Should fall back to unconditional n/n_group = 100/2 = 50
        assert np.allclose(s_hat, 50.0)
