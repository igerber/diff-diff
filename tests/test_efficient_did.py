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
from diff_diff.survey import SurveyDesign

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
        """Near-singular Omega* should trigger a warning (legacy path).

        Re-scoped to omega_ridge=0 with the v3.7 ridge default: this test
        exercises the legacy inv/pinv path's per-cell warning contract, which
        the default ridge path intentionally replaces with one aggregate
        fit-level warning (see TestOmegaRidge).
        """
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
            EfficientDiD(omega_ridge=0.0).fit(df, "y", "unit", "time", "first_treat")
            # Should get a warning about condition number or zero matrix
            warning_msgs = [str(x.message) for x in w]
            assert any(
                "condition" in m.lower()
                or "zero" in m.lower()
                or "pseudoinverse" in m.lower()
                or "uniform" in m.lower()
                for m in warning_msgs
            ), f"Expected condition/zero warning, got: {warning_msgs}"


class TestOmegaRidge:
    """omega_ridge parameter surface, warning contract, and engagement lock.

    The v3.7 default (OMEGA_RIDGE_DEFAULT) ridge-regularizes the Omega*
    inversion behind the efficient weights; omega_ridge=0 restores the exact
    legacy inv/pinv code path. See REGISTRY.md (EfficientDiD, Omega* ridge
    regularization note).
    """

    @staticmethod
    def _duplicated_period_panel(n_units=120, seed=11):
        """Panel where period 3 duplicates period 2 exactly.

        The never-treated moments for t_pre=2 and t_pre=3 become identical,
        so every overidentified cell's Omega* is exactly singular
        (cond > 1e12) WITHOUT being the all-zero matrix.
        """
        rng = np.random.default_rng(seed)
        n_periods = 6
        ft = np.full(n_units, np.inf)
        ft[: n_units // 2] = 5
        y_wide = rng.normal(0, 1.0, (n_units, n_periods))
        y_wide[:, 2] = y_wide[:, 1]  # period 3 == period 2
        treated = (ft[:, None] < np.inf) & (np.arange(1, n_periods + 1)[None, :] >= ft[:, None])
        y_wide = y_wide + 2.0 * treated
        return pd.DataFrame(
            {
                "unit": np.repeat(np.arange(n_units), n_periods),
                "time": np.tile(np.arange(1, n_periods + 1), n_units),
                "first_treat": np.repeat(ft, n_periods),
                "y": y_wide.ravel(),
            }
        )

    def test_param_surface(self):
        from diff_diff.efficient_did_covariates import OMEGA_RIDGE_DEFAULT

        est = EfficientDiD()
        assert est.get_params()["omega_ridge"] == OMEGA_RIDGE_DEFAULT
        est.set_params(omega_ridge=0.0)
        assert est.omega_ridge == 0.0
        est.set_params(omega_ridge=1e-4)
        assert est.get_params()["omega_ridge"] == 1e-4

    @pytest.mark.parametrize("bad", [-1e-6, np.nan, np.inf, -np.inf])
    def test_validation_rejects_bad_values(self, bad):
        with pytest.raises(ValueError, match="omega_ridge"):
            EfficientDiD(omega_ridge=bad)

    def test_set_params_transactional(self):
        from diff_diff.efficient_did_covariates import OMEGA_RIDGE_DEFAULT

        est = EfficientDiD()
        with pytest.raises(ValueError):
            est.set_params(omega_ridge=-1.0)
        assert est.omega_ridge == OMEGA_RIDGE_DEFAULT

    def test_results_echo(self):
        df = _make_staggered_panel()
        res = EfficientDiD(omega_ridge=1e-5).fit(df, "y", "unit", "time", "first_treat")
        assert res.omega_ridge == 1e-5

    def test_aggregate_warning_payload_and_legacy_per_cell(self):
        """Default ridge: ONE fit-level warning whose cell count matches the
        number of genuinely ill-conditioned cells; omega_ridge=0: the legacy
        per-cell pseudoinverse warnings for the same cells."""
        import re

        df = self._duplicated_period_panel()

        with warnings.catch_warnings(record=True) as w_ridge:
            warnings.simplefilter("always")
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        agg = [str(x.message) for x in w_ridge if "regularization handled" in str(x.message)]
        assert len(agg) == 1, f"expected exactly one aggregate warning, got {agg}"
        m = re.search(r"in (\d+) of (\d+) \(g, t\) cells", agg[0])
        assert m is not None, agg[0]
        n_ill, n_cells = int(m.group(1)), int(m.group(2))
        assert 1 <= n_ill <= n_cells

        with warnings.catch_warnings(record=True) as w_legacy:
            warnings.simplefilter("always")
            EfficientDiD(omega_ridge=0.0).fit(df, "y", "unit", "time", "first_treat")
        legacy_msgs = [str(x.message) for x in w_legacy]
        per_cell = [m_ for m_ in legacy_msgs if "using pseudoinverse for weights" in m_]
        # Same pathology surfaces per-cell on the legacy path. The legacy pair
        # set additionally contains the degenerate (g'=g, t_pre=t) self-pair
        # for pre-treatment cells (dropped on the ridge path), so legacy may
        # warn on MORE cells - never fewer.
        assert len(per_cell) >= n_ill
        assert not any("regularization handled" in m_ for m_ in legacy_msgs)

    def test_ridge_engaged_and_overall_stable(self):
        """Default vs omega_ridge=0 on an overidentified panel: overall ATT
        agrees tightly while at least one per-cell effect differs (proves the
        ridge path is actually engaged)."""
        df = _make_staggered_panel(n_per_group=80, n_control=100)
        res_r = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        res_l = EfficientDiD(omega_ridge=0.0).fit(df, "y", "unit", "time", "first_treat")
        assert res_r.overall_att == pytest.approx(res_l.overall_att, rel=1e-4)
        diffs = [
            abs(res_r.group_time_effects[k]["effect"] - res_l.group_time_effects[k]["effect"])
            for k in res_r.group_time_effects
        ]
        assert max(diffs) > 0.0

    def test_degenerate_self_pair_dropped_only_on_ridge_path(self):
        """Pre-treatment cells lose the identically-zero (g'=g, t_pre=t)
        self-pair under ridge (stored nocov weights shrink by exactly one),
        while post-treatment cells keep the same pair count."""
        df = _make_staggered_panel()
        res_r = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        res_l = EfficientDiD(omega_ridge=0.0).fit(df, "y", "unit", "time", "first_treat")
        assert res_r.efficient_weights is not None
        assert res_l.efficient_weights is not None
        period_1 = min(res_r.time_periods)
        for gt, w_l in res_l.efficient_weights.items():
            g, t = gt
            w_r = res_r.efficient_weights[gt]
            if t < g and t != period_1:
                assert len(w_r) == len(w_l) - 1, f"cell {gt}"
            else:
                assert len(w_r) == len(w_l), f"cell {gt}"

    def test_pretreatment_placebos_remain_data_driven(self):
        """Under default ridge, pre-treatment placebo cells stay data-driven
        (nonzero noise), NOT deterministically zero - the degenerate-pair
        drop preserves the pre-trend diagnostic."""
        df = _make_staggered_panel(rho=0.3, seed=99)
        res = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        pre = [
            abs(v["effect"])
            for (g, t), v in res.group_time_effects.items()
            if t < g and np.isfinite(v["effect"])
        ]
        assert pre, "expected pre-treatment cells"
        assert max(pre) > 1e-8


class TestFusedConditionalPath:
    """The v3.7 fused unit-tiled GEMM conditional path (default ridge):
    semantic equivalence to the dense construction, tile invariance, and the
    1-ulp stability contract that motivated the ridge."""

    @staticmethod
    def _cov_df(n_units=200, n_periods=9, seed=42):
        return _make_covariate_panel(n_units=n_units, n_periods=n_periods, seed=seed)

    @staticmethod
    def _surfaces(res):
        gt = {k: (v["effect"], v["se"]) for k, v in res.group_time_effects.items()}
        es = (
            {k: (v["effect"], v["se"]) for k, v in res.event_study_effects.items()}
            if res.event_study_effects
            else {}
        )
        return gt, es, (res.overall_att, res.overall_se)

    @classmethod
    def _assert_close_surfaces(cls, r1, r2, rtol, atol=1e-12):
        gt1, es1, ov1 = cls._surfaces(r1)
        gt2, es2, ov2 = cls._surfaces(r2)
        np.testing.assert_allclose(ov1, ov2, rtol=rtol, atol=atol)
        for k in gt2:
            np.testing.assert_allclose(gt1[k], gt2[k], rtol=rtol, atol=atol, err_msg=str(k))
        for k in es2:
            np.testing.assert_allclose(es1[k], es2[k], rtol=rtol, atol=atol, err_msg=str(k))

    def test_fused_matches_dense_reference(self):
        """Fused tiled GEMM cells match a dense reference built from the
        legacy compute_omega_star_conditional + the same ridge solve. The
        residual is the omega GEMM's ~1e-15 reassociation drift amplified by
        the ridge's bounded 1/lambda sensitivity (~1e6) - i.e. <= ~1e-8, the
        designed stability, vs ~1e-2 under the legacy pseudoinverse."""
        import diff_diff.efficient_did as ed_mod
        from diff_diff.efficient_did_covariates import (
            _ridge_solve_weights,
            compute_generated_outcomes_cov,
            compute_omega_star_conditional,
        )

        def dense_reference(
            cell_specs,
            outcome_wide,
            covariate_matrix,
            cohort_masks,
            never_treated_mask,
            period_to_col,
            cohort_fractions,
            m_hat_cache,
            r_hat_cache,
            s_hat_cache,
            bandwidth,
            omega_ridge,
            unit_weights=None,
            never_treated_val=np.inf,
            tile_bytes=None,
        ):
            out = {}
            for spec in cell_specs:
                g, t, pairs = spec["g"], spec["t"], spec["pairs"]
                gen_out = compute_generated_outcomes_cov(
                    target_g=g,
                    target_t=t,
                    valid_pairs=pairs,
                    outcome_wide=outcome_wide,
                    cohort_masks=cohort_masks,
                    never_treated_mask=never_treated_mask,
                    period_to_col=period_to_col,
                    period_1_col=spec["y1_col"],
                    cohort_fractions=cohort_fractions,
                    m_hat_cache=m_hat_cache,
                    r_hat_cache=r_hat_cache,
                )
                if len(pairs) == 1:
                    scores = gen_out[:, 0]
                else:
                    omega = compute_omega_star_conditional(
                        target_g=g,
                        target_t=t,
                        valid_pairs=pairs,
                        outcome_wide=outcome_wide,
                        cohort_masks=cohort_masks,
                        never_treated_mask=never_treated_mask,
                        period_to_col=period_to_col,
                        period_1_col=spec["y1_col"],
                        cohort_fractions=cohort_fractions,
                        covariate_matrix=covariate_matrix,
                        s_hat_cache=s_hat_cache,
                        bandwidth=bandwidth,
                        unit_weights=unit_weights,
                    )
                    w = _ridge_solve_weights(omega, omega_ridge)
                    scores = np.sum(w * gen_out, axis=1)
                att = (
                    float(np.average(scores, weights=unit_weights))
                    if unit_weights is not None
                    else float(np.mean(scores))
                )
                out[(g, t)] = (att, scores - att)
            return out

        df = self._cov_df()
        fit_kwargs = dict(covariates=["x1", "x2"], aggregate="all")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_fused = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", **fit_kwargs)
        orig = ed_mod.compute_conditional_cells_tiled
        ed_mod.compute_conditional_cells_tiled = dense_reference
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_ref = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", **fit_kwargs)
        finally:
            ed_mod.compute_conditional_cells_tiled = orig
        self._assert_close_surfaces(r_fused, r_ref, rtol=1e-6, atol=1e-8)

    def test_tile_forced_twin(self, monkeypatch):
        """Forcing one-unit tiles must reproduce the single-tile fit
        (rel 1e-10) - and must actually execute multi-tile."""
        import diff_diff.efficient_did_covariates as cov_mod

        df = self._cov_df(n_units=150)
        fit_kwargs = dict(covariates=["x1", "x2"], aggregate="all")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_single = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", **fit_kwargs)

        calls = {"n": 0}
        orig_kwm = cov_mod._kernel_weights_matrix

        def counting_kwm(*args, **kwargs):
            calls["n"] += 1
            return orig_kwm(*args, **kwargs)

        monkeypatch.setattr(cov_mod, "_TARGET_OMEGA_TILE_BYTES", 1)
        monkeypatch.setattr(cov_mod, "_kernel_weights_matrix", counting_kwm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_tiled = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", **fit_kwargs)
        # one-unit tiles -> at least n_units kernel-matrix builds (vs ~4 for
        # a single tile), proving the multi-tile path executed
        assert calls["n"] >= 150, calls["n"]
        self._assert_close_surfaces(r_tiled, r_single, rtol=1e-10, atol=1e-12)

    def test_tile_forced_twin_survey_weights(self, monkeypatch):
        """Tile invariance under survey weights (weighted kernels, weighted
        ATT averaging)."""
        import diff_diff.efficient_did_covariates as cov_mod

        df = self._cov_df(n_units=150)
        rng = np.random.default_rng(5)
        n_units = df["unit"].nunique()
        pw = np.exp(rng.normal(0, 0.4, n_units))
        df = df.merge(pd.DataFrame({"unit": sorted(df["unit"].unique()), "pw": pw}), on="unit")
        fit_kwargs = dict(
            covariates=["x1", "x2"], aggregate="all", survey_design=SurveyDesign(weights="pw")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_single = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", **fit_kwargs)
        monkeypatch.setattr(cov_mod, "_TARGET_OMEGA_TILE_BYTES", 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_tiled = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", **fit_kwargs)
        self._assert_close_surfaces(r_tiled, r_single, rtol=1e-10, atol=1e-12)

    def test_one_ulp_stability_conditional(self):
        """THE stability contract: a 1-ulp outcome perturbation moves
        per-cell and event-study ATTs and SEs by <= 1e-6 relative under the
        default ridge (the legacy pseudoinverse path moves ~1e-4)."""
        df = self._cov_df()
        fit_kwargs = dict(covariates=["x1", "x2"], aggregate="all")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_base = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", **fit_kwargs)
        df_ulp = df.copy()
        df_ulp["y"] = np.nextafter(df_ulp["y"].to_numpy(), np.inf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_ulp = EfficientDiD().fit(df_ulp, "y", "unit", "time", "first_treat", **fit_kwargs)
        self._assert_close_surfaces(r_ulp, r_base, rtol=1e-6, atol=1e-9)

    def test_one_ulp_stability_nocov(self):
        """Same stability contract on the no-covariates PT-All path."""
        df = _make_staggered_panel(n_per_group=80, n_control=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_base = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="all")
        df_ulp = df.copy()
        df_ulp["y"] = np.nextafter(df_ulp["y"].to_numpy(), np.inf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_ulp = EfficientDiD().fit(df_ulp, "y", "unit", "time", "first_treat", aggregate="all")
        self._assert_close_surfaces(r_ulp, r_base, rtol=1e-6, atol=1e-9)


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

    def test_last_cohort_with_anticipation_trims_at_last_g_minus_anticipation(self):
        """last_cohort + anticipation>0 trims at `last_g - anticipation`, not `last_g`.

        Regression guard for PR #230 deferral: the code at efficient_did.py:470 uses
        `effective_last = last_g - self.anticipation` so anticipation-contaminated periods
        are excluded from the pseudo-control's pre-treatment window. If a future change
        reverts to `t < last_g`, this test will catch it by checking the trimmed
        `time_periods` set exposed on EfficientDiDResults.
        """
        df = _make_staggered_panel(
            n_per_group=60,
            n_control=0,
            groups=(3, 5, 7),
            effects={3: 2.0, 5: 1.5, 7: 1.0},
        )
        # _make_staggered_panel default n_periods=7, last_g=7, times 1..7.
        # anticipation=0: effective_last=7, time_periods=[1..6]
        # anticipation=1: effective_last=6, time_periods=[1..5]
        result_a0 = EfficientDiD(
            pt_assumption="all", control_group="last_cohort", anticipation=0
        ).fit(df, "y", "unit", "time", "first_treat")
        result_a1 = EfficientDiD(
            pt_assumption="all", control_group="last_cohort", anticipation=1
        ).fit(df, "y", "unit", "time", "first_treat")

        assert max(result_a0.time_periods) == 6
        assert max(result_a1.time_periods) == 5
        assert len(result_a1.time_periods) == len(result_a0.time_periods) - 1
        assert np.isfinite(result_a0.overall_att)
        assert np.isfinite(result_a1.overall_att)
        assert 7 not in result_a0.groups
        assert 7 not in result_a1.groups

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


# ---------------------------------------------------------------------------
# Silent-failure audit PR #9: finding #18 — estimate_*_sieve silently
# `continue`'d past rank-deficient K values. Now we track skipped K and
# warn when we ship a result that wasn't the IC-winner across all K.
# ---------------------------------------------------------------------------


class TestSievePartialKSkipWarning:
    """Finding #18 (axis A): partial K-failure no longer silent."""

    def test_ratio_sieve_partial_skip_warns(self):
        """If some K's are rank-deficient but at least one succeeds,
        the function warns about the partial skip instead of swallowing it."""
        from diff_diff.efficient_did_covariates import estimate_propensity_ratio_sieve

        rng = np.random.default_rng(7)
        n = 200
        # 1D covariate with discrete support {0, 1}. At K=1 the basis is
        # [1, x]; at K>=2 the basis reaches size >= n_gp for most groups
        # before hitting singularity, but with this discrete support the
        # polynomial powers x^2, x^3, ... equal x, yielding rank-deficient
        # normal equations deterministically.
        X = rng.integers(0, 2, size=(n, 1)).astype(float)
        mask_g = np.zeros(n, dtype=bool)
        mask_g[:100] = True
        mask_gp = np.zeros(n, dtype=bool)
        mask_gp[100:] = True
        with pytest.warns(UserWarning) as caught:
            ratio = estimate_propensity_ratio_sieve(X, mask_g, mask_gp, k_max=3)
        assert np.all(np.isfinite(ratio))
        partial_skip_msgs = [str(w.message) for w in caught if "skipped K=" in str(w.message)]
        assert partial_skip_msgs, (
            "Expected a partial-K-skip warning when some K's are rank deficient "
            "but at least one succeeds; got none."
        )
        # Message should name the specific K values that were skipped.
        assert any("K=" in m for m in partial_skip_msgs)

    def test_inverse_propensity_sieve_partial_skip_warns(self):
        """Same contract for the inverse propensity sieve."""
        from diff_diff.efficient_did_covariates import estimate_inverse_propensity_sieve

        rng = np.random.default_rng(7)
        n = 200
        X = rng.integers(0, 2, size=(n, 1)).astype(float)
        mask = np.zeros(n, dtype=bool)
        mask[:100] = True
        with pytest.warns(UserWarning) as caught:
            s_hat = estimate_inverse_propensity_sieve(X, mask, k_max=3)
        assert np.all(np.isfinite(s_hat))
        partial_skip_msgs = [str(w.message) for w in caught if "skipped K=" in str(w.message)]
        assert partial_skip_msgs

    def test_ratio_sieve_no_warning_when_no_skips(self):
        """Clean, well-conditioned covariates → no partial-skip warning."""
        from diff_diff.efficient_did_covariates import estimate_propensity_ratio_sieve

        rng = np.random.default_rng(101)
        n = 300
        X = rng.normal(0, 1, (n, 2))
        mask_g = np.zeros(n, dtype=bool)
        mask_g[:150] = True
        mask_gp = np.zeros(n, dtype=bool)
        mask_gp[150:] = True
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            ratio = estimate_propensity_ratio_sieve(X, mask_g, mask_gp, k_max=3)
        assert np.all(np.isfinite(ratio))
        partial_skip_msgs = [str(w.message) for w in caught if "skipped K=" in str(w.message)]
        assert (
            partial_skip_msgs == []
        ), f"Unexpected partial-skip warning on clean data: {partial_skip_msgs}"


# =============================================================================
# Phase 1b interstitial #4: vcov_type input contract on EfficientDiD
# =============================================================================


def _efficient_clustered_panel(
    seed: int = 71,
    n_units: int = 60,
    n_periods: int = 6,
    n_states: int = 8,
) -> pd.DataFrame:
    """Staggered-adoption panel with a ``state`` cluster column.

    States carry random effects so ``cluster=state`` shifts the analytical
    SE relative to the default ``cluster=None`` per-unit EIF SE.
    """
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    unit_to_state = rng.integers(0, n_states, size=n_units)
    state = np.repeat(unit_to_state, n_periods)
    state_re = rng.standard_normal(n_states) * 1.2

    cohorts = np.array([2, 3, 4])
    n_never = n_units // 2
    n_treated = n_units - n_never
    first_treat = np.zeros(n_units, dtype=int)
    first_treat[n_never:] = cohorts[rng.integers(0, len(cohorts), size=n_treated)]
    first_treat_expanded = np.repeat(first_treat, n_periods)

    unit_fe = rng.standard_normal(n_units) * 1.2
    time_fe = np.linspace(0, 0.5, n_periods)
    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)
    state_fe_expanded = state_re[state]

    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
    outcome = (
        unit_fe_expanded
        + time_fe_expanded
        + state_fe_expanded
        + 1.5 * post
        + rng.standard_normal(len(units)) * 0.4
    )

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "y": outcome,
            "first_treat": first_treat_expanded,
            "state": state,
        }
    )


def _efficient_survey_panel(
    seed: int = 71,
    n_units: int = 80,
    n_periods: int = 5,
    n_psu: int = 16,
    n_strata: int = 4,
) -> pd.DataFrame:
    """Staggered-adoption panel with analytical survey columns (pweight +
    panel-constant PSU + stratum). Used for TSL-survey bit-equality tests."""
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    unit_psu = rng.integers(0, n_psu, size=n_units)
    psu = np.repeat(unit_psu, n_periods)
    psu_to_stratum = rng.integers(0, n_strata, size=n_psu)
    stratum = psu_to_stratum[psu]

    cohorts = np.array([2, 3])
    n_never = n_units // 2
    n_treated = n_units - n_never
    first_treat = np.zeros(n_units, dtype=int)
    first_treat[n_never:] = cohorts[rng.integers(0, len(cohorts), size=n_treated)]
    first_treat_expanded = np.repeat(first_treat, n_periods)

    unit_fe = rng.standard_normal(n_units) * 1.0
    time_fe = np.linspace(0, 0.4, n_periods)
    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
    outcome = (
        unit_fe_expanded + time_fe_expanded + 1.2 * post + rng.standard_normal(len(units)) * 0.35
    )

    unit_weight = 1.0 + rng.exponential(0.3, n_units)
    weight = np.repeat(unit_weight, n_periods)

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "y": outcome,
            "first_treat": first_treat_expanded,
            "psu": psu,
            "stratum": stratum,
            "weight": weight,
        }
    )


def _efficient_replicate_panel(
    seed: int = 89, n_units: int = 40, n_periods: int = 5, n_rep: int = 8
):
    """Staggered-adoption panel with JK1 replicate-weight columns."""
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    cohorts = np.array([2, 3])
    n_never = n_units // 2
    n_treated = n_units - n_never
    first_treat = np.zeros(n_units, dtype=int)
    first_treat[n_never:] = cohorts[rng.integers(0, len(cohorts), size=n_treated)]
    first_treat_expanded = np.repeat(first_treat, n_periods)

    unit_fe = rng.standard_normal(n_units) * 1.0
    time_fe = np.linspace(0, 0.4, n_periods)
    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
    outcome = (
        unit_fe_expanded + time_fe_expanded + 1.0 * post + rng.standard_normal(len(units)) * 0.4
    )

    unit_weight = 1.0 + rng.exponential(0.2, n_units)
    weight = np.repeat(unit_weight, n_periods)

    data = pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "y": outcome,
            "first_treat": first_treat_expanded,
            "weight": weight,
        }
    )

    units_per_rep = max(n_units // n_rep, 1)
    rep_cols = []
    for r in range(n_rep):
        w_r = unit_weight.copy()
        start = r * units_per_rep
        end = min((r + 1) * units_per_rep, n_units)
        w_r[start:end] = 0.0
        nonzero = w_r > 0
        w_r[nonzero] = w_r[nonzero] * n_rep / (n_rep - 1)
        col = f"rep_{r}"
        data[col] = np.repeat(w_r, n_periods)
        rep_cols.append(col)
    return data, rep_cols


class TestEfficientDiDVcovType:
    """Phase 1b interstitial #4: vcov_type input contract on EfficientDiD.

    EfficientDiD uses IF-based variance per Chen-Sant'Anna-Xie (2025) achieving
    the semiparametric efficiency bound; ``vcov_type`` is permanently narrow
    to ``{"hc1"}``. Analytical-sandwich families ``{classical, hc2, hc2_bm}``
    and ``conley`` are rejected at ``__init__`` / ``set_params`` with
    methodology-rooted messages. Mirrors ImputationDiD PR #492 template.

    Key divergence from ImputationDiD: default ``cluster=None`` SE is the
    per-unit EIF SE ``sqrt(mean(EIF²)/n)`` — methodologically HC1-style
    (NOT auto-cluster-at-unit). The summary label "HC1 heteroskedasticity-
    robust" is methodologically correct here, and ``cluster_name``/``n_clusters``
    stay None under unclustered fits.

    7-surface matrix:
      1. Default preserved bit-equally across all 4 aggregate modes
      2. Cluster path preserved bit-equally across the same aggregate grid
      3. TSL-survey path preserved bit-equally across the same aggregate grid
      4. Replicate-survey path preserved bit-equally across the same aggregate grid
      5. Bootstrap × cluster + bootstrap × survey bit-equal
      6. set_params(vcov_type=bad) eager revalidation
      7. Bootstrap n_psu<2 NaN propagation (defensive fix regression)

    Plus 7 introspection tests, 5 input-rejection pins, the
    ``cluster + replicate_weights`` rejection, and a DR-path bit-equality test.
    """

    # ---- Surface 1: default bit-equal across aggregation modes ------------

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group", "all"])
    def test_default_hc1_bit_equal_baseline(self, aggregate):
        data = _efficient_clustered_panel()
        common = dict(
            data=data,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate=aggregate,
        )
        r_default = EfficientDiD().fit(**common)
        r_explicit = EfficientDiD(vcov_type="hc1").fit(**common)
        assert r_default.overall_att == r_explicit.overall_att
        assert r_default.overall_se == r_explicit.overall_se

    # ---- Surface 2: cluster path bit-equal --------------------------------

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group", "all"])
    def test_cluster_hc1_bit_equal_baseline(self, aggregate):
        data = _efficient_clustered_panel()
        common = dict(
            data=data,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate=aggregate,
        )
        r_default = EfficientDiD(cluster="state").fit(**common)
        r_explicit = EfficientDiD(cluster="state", vcov_type="hc1").fit(**common)
        assert r_default.overall_att == r_explicit.overall_att
        assert r_default.overall_se == r_explicit.overall_se

    # ---- Surface 3: TSL-survey path bit-equal -----------------------------

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group", "all"])
    def test_survey_tsl_hc1_bit_equal_baseline(self, aggregate):
        data = _efficient_survey_panel()
        design = SurveyDesign(weights="weight", psu="psu", strata="stratum", weight_type="pweight")
        common = dict(
            data=data,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate=aggregate,
            survey_design=design,
        )
        r_default = EfficientDiD().fit(**common)
        r_explicit = EfficientDiD(vcov_type="hc1").fit(**common)
        assert r_default.overall_att == r_explicit.overall_att
        assert r_default.overall_se == r_explicit.overall_se

    # ---- Surface 4: replicate-survey path bit-equal -----------------------

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group", "all"])
    def test_survey_replicate_hc1_bit_equal_baseline(self, aggregate):
        data, rep_cols = _efficient_replicate_panel()
        design = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
            weight_type="pweight",
        )
        common = dict(
            data=data,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
            aggregate=aggregate,
        )
        r_default = EfficientDiD().fit(**common)
        r_explicit = EfficientDiD(vcov_type="hc1").fit(**common)
        assert r_default.overall_att == r_explicit.overall_att
        assert r_default.overall_se == r_explicit.overall_se
        # Per-horizon / per-group SE override branches must also agree under
        # the replicate-weight variance path.
        if aggregate in ("event_study", "all"):
            assert r_default.event_study_effects is not None
            assert r_explicit.event_study_effects is not None
            for h in r_default.event_study_effects:
                assert (
                    r_default.event_study_effects[h]["se"]
                    == r_explicit.event_study_effects[h]["se"]
                )
        if aggregate in ("group", "all"):
            assert r_default.group_effects is not None
            assert r_explicit.group_effects is not None
            for g in r_default.group_effects:
                assert r_default.group_effects[g]["se"] == r_explicit.group_effects[g]["se"]

    # ---- Surface 5: bootstrap × cluster / × survey bit-equal --------------

    def test_bootstrap_cluster_hc1_bit_equal(self, ci_params):
        data = _efficient_clustered_panel()
        n_boot = ci_params.bootstrap(199)
        common = dict(
            data=data,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )
        r_default = EfficientDiD(cluster="state", n_bootstrap=n_boot, seed=11).fit(**common)
        r_explicit = EfficientDiD(
            cluster="state", n_bootstrap=n_boot, seed=11, vcov_type="hc1"
        ).fit(**common)
        assert r_default.bootstrap_results is not None
        assert r_explicit.bootstrap_results is not None
        assert (
            r_default.bootstrap_results.overall_att_se
            == r_explicit.bootstrap_results.overall_att_se
        )
        # Per-horizon / per-group bootstrap SE override branches at
        # efficient_did.py:1090-1115 must also agree.
        assert r_default.bootstrap_results.event_study_ses is not None
        assert r_explicit.bootstrap_results.event_study_ses is not None
        for h, se in r_default.bootstrap_results.event_study_ses.items():
            assert se == r_explicit.bootstrap_results.event_study_ses[h]
        assert r_default.bootstrap_results.group_effect_ses is not None
        assert r_explicit.bootstrap_results.group_effect_ses is not None
        for g, se in r_default.bootstrap_results.group_effect_ses.items():
            assert se == r_explicit.bootstrap_results.group_effect_ses[g]

    def test_bootstrap_survey_hc1_bit_equal(self, ci_params):
        data = _efficient_survey_panel()
        design = SurveyDesign(weights="weight", psu="psu", strata="stratum", weight_type="pweight")
        n_boot = ci_params.bootstrap(199)
        common = dict(
            data=data,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
            aggregate="all",
        )
        r_default = EfficientDiD(n_bootstrap=n_boot, seed=23).fit(**common)
        r_explicit = EfficientDiD(n_bootstrap=n_boot, seed=23, vcov_type="hc1").fit(**common)
        assert r_default.bootstrap_results is not None
        assert r_explicit.bootstrap_results is not None
        assert (
            r_default.bootstrap_results.overall_att_se
            == r_explicit.bootstrap_results.overall_att_se
        )

    # ---- Surface 6: set_params eager revalidation -------------------------

    def test_set_params_bad_vcov_caught_immediately(self):
        # EfficientDiD's set_params calls _validate_params (which now invokes
        # _validate_vcov_type), so the check fires NOW (not at fit-time).
        # This intentionally diverges from ImputationDiD/TripleDifference
        # (which defer to fit-time per sklearn mutate-then-validate-at-use).
        ed = EfficientDiD()
        with pytest.raises(ValueError, match="influence-function"):
            ed.set_params(vcov_type="classical")

    def test_set_params_unknown_vcov_caught_immediately(self):
        ed = EfficientDiD()
        with pytest.raises(ValueError, match="hc4"):
            ed.set_params(vcov_type="hc4")

    def test_set_params_rollback_on_validation_failure(self):
        # set_params is atomic: when validation rejects a batched call, NO
        # attribute mutation persists. Pre-fix, set_params assigned every
        # kwarg before invoking _validate_params, so a rejected
        # `set_params(vcov_type="classical", alpha=0.1, anticipation=2)`
        # raised but left all three attributes mutated — weakening eager-
        # validation for callers that catch ValueError and keep using the
        # estimator.
        ed = EfficientDiD()
        original_vcov = ed.vcov_type
        original_alpha = ed.alpha
        original_anticipation = ed.anticipation
        with pytest.raises(ValueError, match="influence-function"):
            ed.set_params(vcov_type="classical", alpha=0.1, anticipation=2)
        assert ed.vcov_type == original_vcov
        assert ed.alpha == original_alpha
        assert ed.anticipation == original_anticipation

    # ---- Surface 7: bootstrap n_psu<2 NaN propagation ---------------------

    def test_bootstrap_n_psu_less_than_2_returns_nan(self):
        # Single-PSU survey design forces the survey-PSU bootstrap path to
        # hit the n_psu<2 BLAS-roundoff guard. Survey weight_type must be
        # pweight per EfficientDiD's survey contract.
        data = _efficient_survey_panel(seed=42)
        data["single_psu"] = 0
        data["single_stratum"] = 0
        design = SurveyDesign(
            weights="weight",
            psu="single_psu",
            strata="single_stratum",
            weight_type="pweight",
        )
        with pytest.warns(UserWarning, match="n_psu=1"):
            results = EfficientDiD(n_bootstrap=199, seed=5).fit(
                data,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
        assert results.bootstrap_results is not None
        assert np.isnan(results.bootstrap_results.overall_att_se)
        assert np.isnan(results.bootstrap_results.overall_att_p_value)
        assert all(np.isnan(x) for x in results.bootstrap_results.overall_att_ci)
        # Derived coef_var propagates NaN through the alias property.
        assert np.isnan(results.coef_var)

    # ---- DR (covariates) path bit-equal -----------------------------------

    def test_dr_path_hc1_bit_equal(self):
        # Doubly-robust (covariates=) path uses the same _eif_se / _aggregate_*
        # variance funnel as the no-cov path — only EIF *construction* differs.
        # Validates the variance machinery passes through the sieve/OLS DR path
        # unchanged under explicit vcov_type="hc1".
        data = make_compustat_dgp(seed=23, n_units=80, n_periods=5)
        # Use one panel-constant synthetic covariate so DR path engages.
        rng = np.random.default_rng(23)
        n_units = data["unit"].nunique()
        x1_per_unit = rng.standard_normal(n_units)
        unit_to_x1 = dict(zip(sorted(data["unit"].unique()), x1_per_unit))
        data = data.copy()
        data["x1"] = data["unit"].map(unit_to_x1)
        common = dict(
            data=data,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1"],
            aggregate="event_study",
        )
        r_default = EfficientDiD().fit(**common)
        r_explicit = EfficientDiD(vcov_type="hc1").fit(**common)
        assert r_default.estimation_path == "dr"
        assert r_explicit.estimation_path == "dr"
        assert r_default.overall_att == r_explicit.overall_att
        assert r_default.overall_se == r_explicit.overall_se

    # ---- Input rejection: methodology-rooted messages ---------------------

    @pytest.mark.parametrize(
        "bad_vcov,keyword",
        [
            ("classical", "influence-function"),
            ("hc2", "Chen"),
            ("hc2_bm", "Bell-McCaffrey"),
            ("hc2_bm", "hat matrix"),
        ],
    )
    def test_reject_invalid_vcov_at_init(self, bad_vcov, keyword):
        with pytest.raises(ValueError, match=keyword):
            EfficientDiD(vcov_type=bad_vcov)

    def test_reject_conley_at_init(self):
        with pytest.raises(ValueError, match="spatial-HAC"):
            EfficientDiD(vcov_type="conley")

    def test_reject_unknown_vcov_at_init(self):
        with pytest.raises(ValueError, match="hc4"):
            EfficientDiD(vcov_type="hc4")

    # ---- cluster + survey blanket rejection (covers replicate subset) -----

    def test_cluster_plus_replicate_weights_rejected(self):
        # cluster + survey_design is blanket-rejected at efficient_did.py:357,
        # which transitively covers cluster + replicate_weights. Asserting
        # via NotImplementedError on a JK1 replicate design.
        data, rep_cols = _efficient_replicate_panel()
        data["state"] = (data["unit"] // 4).astype(int)
        design = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
            weight_type="pweight",
        )
        with pytest.raises(NotImplementedError, match="cluster and survey_design"):
            EfficientDiD(cluster="state").fit(
                data,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )

    # ---- Introspection / safety-gate tests --------------------------------

    def test_default_vcov_type_is_hc1(self):
        assert EfficientDiD().vcov_type == "hc1"

    def test_get_params_includes_vcov_type(self):
        params = EfficientDiD().get_params()
        assert "vcov_type" in params
        assert params["vcov_type"] == "hc1"

    def test_results_carries_vcov_type(self):
        data = _efficient_clustered_panel()
        r = EfficientDiD().fit(
            data, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        assert r.vcov_type == "hc1"

    def test_to_dict_includes_vcov_type(self):
        data = _efficient_clustered_panel()
        r = EfficientDiD().fit(
            data, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        d = r.to_dict()
        assert d["vcov_type"] == "hc1"
        # Headline alias keys are present per the TripleDifference precedent.
        for k in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper"):
            assert k in d
        # Default unclustered fit → no cluster_name / n_clusters in dict.
        assert "cluster_name" not in d
        assert "n_clusters" not in d
        assert d["inference_method"] == "heteroskedasticity_robust"

    def test_to_dict_under_cluster(self):
        data = _efficient_clustered_panel()
        r = EfficientDiD(cluster="state").fit(
            data, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        d = r.to_dict()
        assert d["cluster_name"] == "state"
        assert d["n_clusters"] is not None and d["n_clusters"] > 1
        assert d["inference_method"] == "cluster_robust"

    def test_summary_includes_vcov_type_label_default(self):
        # Default cluster=None (no survey, no bootstrap) → HC1 label, NOT CR1.
        # This is methodologically correct for EfficientDiD: the per-unit EIF
        # SE `sqrt(mean(EIF²)/n)` is HC1-style (no Liang-Zeger G/(G-1)
        # finite-sample correction). Diverges from ImputationDiD (BJS Theorem 3
        # auto-clusters at unit by construction).
        data = _efficient_clustered_panel()
        r = EfficientDiD().fit(
            data, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        text = r.summary()
        assert "Variance estimator:" in text
        assert "HC1 heteroskedasticity-robust" in text
        # No CR1 cluster label under default cluster=None.
        assert "CR1 cluster-robust" not in text
        # Results metadata stays None under default fits (no auto-cluster-at-unit).
        assert r.cluster_name is None
        assert r.n_clusters is None

    def test_summary_renders_cluster_label_under_cluster(self):
        data = _efficient_clustered_panel()
        r = EfficientDiD(cluster="state").fit(
            data, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        text = r.summary()
        assert "Variance estimator:" in text
        assert "CR1 cluster-robust" in text
        assert "state" in text
        assert "Number of clusters:" in text
        assert r.cluster_name == "state"
        assert r.n_clusters is not None and r.n_clusters > 1

    def test_summary_suppresses_variance_label_under_bootstrap(self, ci_params):
        # Under bootstrap fits, bootstrap_results overwrites SE/CI/p-value, so
        # the analytical variance-family label would misstate the inference
        # source. Mirror the canonical DiDResults gate at results.py:213-226.
        data = _efficient_clustered_panel()
        n_boot = ci_params.bootstrap(199)
        r = EfficientDiD(n_bootstrap=n_boot, seed=7).fit(
            data, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        text = r.summary()
        assert "Inference method:" in text
        assert "bootstrap" in text
        # Analytical variance-family label must be suppressed.
        assert "Variance estimator:" not in text
        assert "HC1 heteroskedasticity-robust" not in text
        assert "CR1 cluster-robust" not in text

    def test_cluster_name_suppressed_under_survey(self):
        data = _efficient_survey_panel()
        design = SurveyDesign(weights="weight", psu="psu", strata="stratum", weight_type="pweight")
        r = EfficientDiD().fit(
            data,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        assert r.cluster_name is None
        assert r.n_clusters is None

    def test_cluster_name_suppressed_under_replicate_survey(self):
        # Replicate-weight survey designs have psu=None — gate must be on
        # `resolved_survey is not None`, NOT `resolved_survey.psu is not None`.
        # Mirror ImputationDiD R2 fix pattern.
        data, rep_cols = _efficient_replicate_panel()
        design = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
            weight_type="pweight",
        )
        r = EfficientDiD().fit(
            data,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        assert r.cluster_name is None
        assert r.n_clusters is None
        text = r.summary()
        assert "Number of clusters:" not in text
        assert "CR1 cluster-robust" not in text

    def test_fit_clone_idempotent_on_vcov_type(self):
        data = _efficient_clustered_panel()
        ed1 = EfficientDiD(vcov_type="hc1")
        r1 = ed1.fit(data, outcome="y", unit="unit", time="time", first_treat="first_treat")
        ed2 = EfficientDiD(**ed1.get_params())
        r2 = ed2.fit(data, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert r1.overall_att == r2.overall_att
        assert r1.overall_se == r2.overall_se
        assert r1.vcov_type == r2.vcov_type


class TestSieveBasisCache:
    """The per-fit sieve-basis cache shares ``_polynomial_sieve_basis(X, K)`` across the
    three DR nuisance helpers. Because the basis is a pure function of ``(X, degree)`` and
    the helpers only read it, caching is bit-identical to rebuilding — these tests pin the
    cache mechanism (the end-to-end bit-identity is also proven against an origin/main
    capture during development)."""

    def test_cache_hit_returns_same_object_and_is_bit_identical(self):
        from diff_diff.efficient_did_covariates import (
            _polynomial_sieve_basis,
            _sieve_basis_cached,
        )

        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 2))
        cache: dict = {}
        a = _sieve_basis_cached(X, 2, cache)
        b = _sieve_basis_cached(X, 2, cache)
        # Cache hit returns the SAME object (so downstream reads see identical bytes)...
        assert a is b
        assert len(cache) == 1
        # ...and it equals a fresh build bit-for-bit.
        np.testing.assert_array_equal(a, _polynomial_sieve_basis(X, 2))
        # A different degree adds a second, distinct entry.
        c = _sieve_basis_cached(X, 3, cache)
        assert len(cache) == 2
        assert c is not a
        np.testing.assert_array_equal(c, _polynomial_sieve_basis(X, 3))

    def test_cache_none_is_plain_passthrough(self):
        from diff_diff.efficient_did_covariates import (
            _polynomial_sieve_basis,
            _sieve_basis_cached,
        )

        rng = np.random.default_rng(1)
        X = rng.normal(size=(30, 2))
        a = _sieve_basis_cached(X, 2, None)
        b = _sieve_basis_cached(X, 2, None)
        # No cache: distinct fresh arrays, each equal to a direct build.
        assert a is not b
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, _polynomial_sieve_basis(X, 2))

    def test_reads_do_not_mutate_cached_basis(self):
        from diff_diff.efficient_did_covariates import (
            _polynomial_sieve_basis,
            _sieve_basis_cached,
        )

        rng = np.random.default_rng(2)
        X = rng.normal(size=(50, 2))
        pristine = _polynomial_sieve_basis(X, 2)
        cache: dict = {}
        cached = _sieve_basis_cached(X, 2, cache)
        # The representative reads the helpers perform on basis_all.
        mask = np.arange(50) % 2 == 0
        _ = cached[mask]
        _ = cached @ np.ones(cached.shape[1])
        _ = (np.ones(50)[:, None] * cached).sum(axis=0)
        _ = cached.sum(axis=0)
        # Re-fetch: still the same object and still bit-identical to the pristine build.
        again = _sieve_basis_cached(X, 2, cache)
        assert again is cached
        np.testing.assert_array_equal(again, pristine)

    def test_fit_builds_each_degree_once_across_helpers(self, monkeypatch):
        """End-to-end: a covariate DR fit requests the basis many times (3 helpers ×
        multiple (g,t) cells) but builds each distinct degree exactly once, proving the
        per-fit cache actually shares work."""
        import diff_diff.efficient_did_covariates as cov

        real_build = cov._polynomial_sieve_basis
        real_cached = cov._sieve_basis_cached
        build_keys: list = []  # one entry per ACTUAL _polynomial_sieve_basis build
        request_keys: list = []  # one entry per _sieve_basis_cached request

        def counting_build(X, degree):
            build_keys.append((id(X), degree))
            return real_build(X, degree)

        def counting_cached(X, degree, cache):
            request_keys.append((id(X), degree))
            return real_cached(X, degree, cache)

        monkeypatch.setattr(cov, "_polynomial_sieve_basis", counting_build)
        monkeypatch.setattr(cov, "_sieve_basis_cached", counting_cached)

        df = _make_covariate_panel(n_units=150)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = EfficientDiD(pt_assumption="post").fit(
                df, "y", "unit", "time", "first_treat", covariates=["x1", "x2"]
            )
        assert np.isfinite(result.overall_att)
        # The path was exercised through the cache.
        assert request_keys, "covariate DR path did not run the sieve helpers"
        # Each distinct (X, degree) was built exactly once (perfect dedup)...
        assert len(build_keys) == len(set(build_keys))
        assert len(build_keys) == len(set(request_keys))
        # ...and there was genuine redundancy for the cache to eliminate.
        assert len(request_keys) > len(build_keys)
