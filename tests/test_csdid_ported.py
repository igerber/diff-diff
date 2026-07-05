"""
Tests ported from R's ``did`` package (github.com/bcallaway11/did).

Tier 1 tests use Python-generated data and assert against known true effects
with loose tolerance (±0.5), matching R's own test philosophy.

Tier 2 tests use R-generated data from ``benchmarks/data/csdid_golden_values.json``
and assert strict tolerance against R's exact output.

Each test docstring cites the original R source.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna
from tests.helpers.csdid_dgp import build_csdid_sim_data, build_two_group_sim_data

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOLDEN_VALUES_PATH = Path(__file__).parents[1] / "benchmarks" / "data" / "csdid_golden_values.json"


@pytest.fixture(scope="module")
def golden_values():
    """Load R golden values. Skip if file absent."""
    if not GOLDEN_VALUES_PATH.exists():
        pytest.skip(
            "Golden values file not found; "
            "run: Rscript benchmarks/R/generate_csdid_test_values.R"
        )
    with open(GOLDEN_VALUES_PATH) as f:
        return json.load(f)["scenarios"]


def _golden_to_df(data_dict: dict) -> pd.DataFrame:
    """Reconstruct a DataFrame from the golden values data export."""
    df = pd.DataFrame(
        {
            "unit": data_dict["unit"],
            "period": data_dict["period"],
            "first_treat": data_dict["first_treat"],
            "outcome": data_dict["outcome"],
        }
    )
    if data_dict.get("X") is not None:
        df["X"] = data_dict["X"]
    if data_dict.get("cluster") is not None:
        df["cluster"] = data_dict["cluster"]
    # R uses 0 for never-treated; diff-diff also uses 0
    return df


# ---------------------------------------------------------------------------
# Tier 1: Always-run tests (Python DGP, loose tolerance)
# ---------------------------------------------------------------------------


class TestCSDIDCoreEstimation:
    """Core ATT recovery tests ported from R's test-att_gt.R."""

    def test_att_recovery_no_covariates(self, ci_params):
        """Ported from R did::test-att_gt.R: 'no covariates case'

        Default DGP with bett=betu=0 (no covariate effects).
        ATT(g,t) should be approximately 1.
        """
        (n,) = ci_params.grid([5000])
        data = build_csdid_sim_data(
            n=n,
            time_periods=4,
            te=1.0,
            bett=np.zeros(4),
            betu=np.zeros(4),
            seed=914202401,
        )
        cs = CallawaySantAnna(estimation_method="dr")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        # Check first post-treatment ATT ≈ 1
        gt_effects = results.group_time_effects
        first_key = sorted(gt_effects.keys())[0]
        assert abs(gt_effects[first_key]["effect"] - 1.0) < 0.5, (
            f"ATT(g={first_key[0]}, t={first_key[1]}) = "
            f"{gt_effects[first_key]['effect']:.3f}, expected ≈ 1.0"
        )

    def test_att_recovery_with_covariates_dr(self, ci_params):
        """Ported from R did::test-att_gt.R: 'att_gt works w/o dynamics'

        DGP with ipw=False (DR/REG compatible). DR with covariates.
        """
        (n,) = ci_params.grid([5000])
        data = build_csdid_sim_data(
            n=n,
            time_periods=4,
            ipw=False,
            te=1.0,
            seed=914202402,
        )
        cs = CallawaySantAnna(estimation_method="dr")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
        )
        gt_effects = results.group_time_effects
        first_key = sorted(gt_effects.keys())[0]
        assert abs(gt_effects[first_key]["effect"] - 1.0) < 0.5

    def test_att_recovery_ipw(self, ci_params):
        """Ported from R did::test-att_gt.R: 'att_gt works using ipw'

        DGP with reg=False (IPW compatible). R uses xformla=~1 for IPW here
        (intercept-only propensity), but the DR estimator uses xformla=~X.
        We test both: DR with covariates (should recover ATT≈1) and
        IPW without covariates (should also recover ATT≈1).
        """
        (n,) = ci_params.grid([5000])
        data = build_csdid_sim_data(
            n=n,
            time_periods=4,
            reg=False,
            te=1.0,
            seed=914202403,
        )
        # DR with covariates on the IPW-compatible DGP
        cs_dr = CallawaySantAnna(estimation_method="dr")
        results_dr = cs_dr.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
        )
        gt_dr = results_dr.group_time_effects
        first_key = sorted(gt_dr.keys())[0]
        assert abs(gt_dr[first_key]["effect"] - 1.0) < 0.5

        # IPW without covariates
        cs_ipw = CallawaySantAnna(estimation_method="ipw")
        results_ipw = cs_ipw.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        gt_ipw = results_ipw.group_time_effects
        first_key_ipw = sorted(gt_ipw.keys())[0]
        assert abs(gt_ipw[first_key_ipw]["effect"] - 1.0) < 0.5

    def test_two_period_all_aggregations(self, ci_params):
        """Ported from R did::test-att_gt.R: 'two period case'

        time.periods=2, n=10000. All aggregation types should recover ATT ≈ 1.
        """
        (n,) = ci_params.grid([10000])
        data = build_csdid_sim_data(
            n=n,
            time_periods=2,
            ipw=False,
            te=1.0,
            seed=914202404,
        )
        cs = CallawaySantAnna(estimation_method="reg")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="all",
        )
        assert (
            abs(results.overall_att - 1.0) < 0.5
        ), f"Simple ATT = {results.overall_att:.3f}, expected ≈ 1.0"
        # Event study: post-treatment effect ≈ 1
        if results.event_study_effects:
            post_effects = {e: v for e, v in results.event_study_effects.items() if e >= 0}
            for e, eff in post_effects.items():
                assert abs(eff["effect"] - 1.0) < 0.5, f"Event study e={e}: {eff['effect']:.3f}"

    def test_dynamic_effects_dose_response(self, ci_params):
        """Ported from R did::test-att_gt.R: 'aggregations' (dynamic)

        te=0, te.e=1:T. Event study ATT(e=k) should be approximately k.
        """
        (n,) = ci_params.grid([5000])
        T = 4
        data = build_csdid_sim_data(
            n=n,
            time_periods=T,
            te=0.0,
            te_e=np.arange(1, T + 1, dtype=float),
            seed=914202405,
        )
        cs = CallawaySantAnna(estimation_method="reg")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )
        assert results.event_study_effects is not None
        # Post-treatment event study effects should be positive and increasing.
        # R: panel=FALSE gives att(e=2) ≈ 2. Panel=TRUE may give ≈ 3
        # (different aggregation weights). Check monotonic increase.
        post_effects = sorted(
            [(e, v["effect"]) for e, v in results.event_study_effects.items() if e >= 0],
            key=lambda x: x[0],
        )
        if len(post_effects) >= 2:
            for i in range(1, len(post_effects)):
                assert post_effects[i][1] > post_effects[i - 1][1] - 0.5, (
                    f"Expected increasing: e={post_effects[i][0]} "
                    f"({post_effects[i][1]:.3f}) >= "
                    f"e={post_effects[i-1][0]} ({post_effects[i-1][1]:.3f})"
                )

    def test_aggregation_group(self, ci_params):
        """Ported from R did::test-att_gt.R: 'aggregations' (group)

        te=0, te.bet.ind=1:T, reg=False. Group-specific effects.
        R expects agg_group$att.egt[2] ≈ 2*2 = 4.
        """
        (n,) = ci_params.grid([5000])
        T = 4
        data = build_csdid_sim_data(
            n=n,
            time_periods=T,
            te=0.0,
            reg=False,
            te_bet_ind=np.arange(1, T + 1, dtype=float),
            seed=914202406,
        )
        cs = CallawaySantAnna(estimation_method="ipw", control_group="not_yet_treated")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="group",
        )
        assert results.group_effects is not None
        # At least one group should have a sizeable effect
        any_large = any(abs(v["effect"]) > 1.0 for v in results.group_effects.values())
        assert any_large, "Expected group-specific heterogeneity"

    def test_aggregation_balance_e(self, ci_params):
        """Ported from R did::test-att_gt.R: 'aggregations' (balance_e)

        balance_e=1: balanced event study at e=1.
        R expects consecutive e differences ≈ 1.
        """
        (n,) = ci_params.grid([5000])
        T = 4
        data = build_csdid_sim_data(
            n=n,
            time_periods=T,
            te=0.0,
            te_e=np.arange(1, T + 1, dtype=float),
            te_bet_ind=np.arange(1, T + 1, dtype=float),
            seed=914202407,
        )
        cs = CallawaySantAnna(estimation_method="dr")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
            balance_e=1,
        )
        assert results.event_study_effects is not None
        es = results.event_study_effects
        if 0 in es and 1 in es:
            diff = es[1]["effect"] - es[0]["effect"]
            assert abs(diff - 1.0) < 0.5, f"balance_e consecutive diff = {diff:.3f}, expected ≈ 1.0"


class TestCSDIDNonStandardConfigs:
    """Non-standard configurations from R's test-att_gt.R."""

    def test_non_consecutive_time_periods(self, ci_params):
        """Ported from R did::test-att_gt.R: 'unequally spaced groups'

        Periods [1,2,5,7], te.e=1:8. Event study at e=2 should ≈ 3.
        """
        (n,) = ci_params.grid([5000])
        T = 8
        data = build_csdid_sim_data(
            n=n,
            time_periods=T,
            te=0.0,
            te_e=np.arange(1, T + 1, dtype=float),
            seed=914202408,
        )
        keep_periods = [1, 2, 5, 7]
        data = data[
            (data["first_treat"].isin([0] + keep_periods)) & (data["period"].isin(keep_periods))
        ].reset_index(drop=True)

        cs = CallawaySantAnna(estimation_method="reg")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )
        assert results.event_study_effects is not None
        # R: agg_dynamic$att.egt[egt==2] ≈ 3
        if 2 in results.event_study_effects:
            att_e2 = results.event_study_effects[2]["effect"]
            assert abs(att_e2 - 3.0) < 1.0, f"Non-consecutive e=2: {att_e2:.3f}, expected ≈ 3"

    def test_anticipation_with_and_without(self, ci_params):
        """Ported from R did::test-att_gt.R: 'anticipation'

        With anticipation=1 and shifted groups, dynamic e=2 should ≈ 2.
        Without anticipation (ignore it), e=2 should ≈ 3 (overstated).
        """
        (n,) = ci_params.grid([5000])
        T = 5
        data = build_csdid_sim_data(
            n=n,
            time_periods=T,
            te=0.0,
            te_e=np.arange(-1, T - 1, dtype=float),
            seed=914202409,
        )
        # Shift groups: G = G + 1 for treated, drop G > T
        data["first_treat"] = data["first_treat"].apply(lambda g: 0 if g == 0 else g + 1)
        data = data[data["first_treat"] <= T].reset_index(drop=True)

        # With correct anticipation=1
        cs = CallawaySantAnna(estimation_method="dr", anticipation=1)
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )
        if results.event_study_effects and 2 in results.event_study_effects:
            att_e2 = results.event_study_effects[2]["effect"]
            assert abs(att_e2 - 2.0) < 0.5, f"Anticipation=1 e=2: {att_e2:.3f}, expected ≈ 2"

        # Without anticipation (incorrectly ignoring it)
        cs_no = CallawaySantAnna(estimation_method="dr", anticipation=0)
        results_no = cs_no.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )
        if results_no.event_study_effects and 2 in results_no.event_study_effects:
            att_e2_no = results_no.event_study_effects[2]["effect"]
            assert abs(att_e2_no - 3.0) < 0.5, f"No anticipation e=2: {att_e2_no:.3f}, expected ≈ 3"

    def test_varying_vs_universal_pretreatment(self, ci_params):
        """Ported from R did::test-att_gt.R: 'varying or universal base period'

        Pre-treatment effects differ: varying ≈ 1, universal ≈ -2.
        """
        (n,) = ci_params.grid([5000])
        T = 8
        data = build_csdid_sim_data(
            n=n,
            time_periods=T,
            te=0.0,
            te_e=np.arange(1, T + 1, dtype=float),
            seed=914202410,
        )
        # Keep only groups <= 5 (plus never-treated), shift G by +3
        data = data[(data["first_treat"] <= 5) | (data["first_treat"] == 0)].reset_index(drop=True)
        data["first_treat"] = data["first_treat"].apply(lambda g: 0 if g == 0 else g + 3)

        # Varying base period
        cs_v = CallawaySantAnna(estimation_method="dr", base_period="varying")
        res_v = cs_v.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )

        # Universal base period
        cs_u = CallawaySantAnna(estimation_method="dr", base_period="universal")
        res_u = cs_u.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )

        # R: varying at e=-3 ≈ 1, universal at e=-3 ≈ -2
        if res_v.event_study_effects and -3 in res_v.event_study_effects:
            att_v = res_v.event_study_effects[-3]["effect"]
            assert abs(att_v - 1.0) < 0.5, f"Varying e=-3: {att_v:.3f}, expected ≈ 1"
        if res_u.event_study_effects and -3 in res_u.event_study_effects:
            att_u = res_u.event_study_effects[-3]["effect"]
            assert abs(att_u - (-2.0)) < 0.5, f"Universal e=-3: {att_u:.3f}, expected ≈ -2"

    def test_small_groups_warning(self):
        """Ported from R did::test-att_gt.R: 'small groups'

        One observation in group 2. Other groups should still produce
        valid estimates. diff-diff may or may not warn for small groups;
        the key behavior is that estimation still succeeds.
        """
        data = build_csdid_sim_data(n=5000, time_periods=4, seed=914202411)

        # Keep only 1 unit from group 2
        g2_units = data[data["first_treat"] == 2]["unit"].unique()
        if len(g2_units) > 0:
            keep_unit = g2_units[0]
            data = data[(data["first_treat"] != 2) | (data["unit"] == keep_unit)].reset_index(
                drop=True
            )

        cs = CallawaySantAnna(estimation_method="dr")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

        # Group 3 should still have valid estimates ≈ 1
        # R: expect_equal(res_dr$att[idx], 1, tol=.5) where group==3, t==3
        gt = results.group_time_effects
        g3_effects = {k: v for k, v in gt.items() if k[0] == 3 and k[1] >= 3}
        for (g, t), eff in g3_effects.items():
            assert abs(eff["effect"] - 1.0) < 1.0, f"ATT(g={g}, t={t}) = {eff['effect']:.3f}"

    def test_small_comparison_group(self):
        """Ported from R did::test-att_gt.R: 'small comparison group'

        1 never-treated unit: nevertreated should error or produce
        degenerate results; notyettreated should succeed for some (g,t).
        """
        data = build_csdid_sim_data(n=5000, time_periods=4, seed=914202412)

        # Keep only 1 never-treated unit
        g0_units = data[data["first_treat"] == 0]["unit"].unique()
        if len(g0_units) > 1:
            keep_unit = g0_units[0]
            data = data[(data["first_treat"] != 0) | (data["unit"] == keep_unit)].reset_index(
                drop=True
            )

        # nevertreated with 1 control unit: R errors, diff-diff may
        # error, warn, or produce results with limited precision.
        cs_nt = CallawaySantAnna(control_group="never_treated")
        import warnings

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results_nt = cs_nt.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="period",
                    first_treat="first_treat",
                )
            # If it succeeds, verify at least some results exist
            assert len(results_nt.group_time_effects) > 0
        except (ValueError, RuntimeError):
            pass  # Also acceptable — R errors here too

        # notyettreated should succeed for some cells
        cs_nyt = CallawaySantAnna(control_group="not_yet_treated")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs_nyt.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        assert len(results.group_time_effects) > 0
        some_finite = any(np.isfinite(v["effect"]) for v in results.group_time_effects.values())
        assert some_finite, "Expected some finite ATT(g,t) with not_yet_treated"

    def test_not_yet_treated_control(self, ci_params):
        """Ported from R did::test-att_gt.R: 'not yet treated comparison group'

        Not-yet-treated control should recover ATT ≈ 1.
        """
        (n,) = ci_params.grid([5000])
        data = build_csdid_sim_data(
            n=n,
            time_periods=4,
            reg=False,
            te=1.0,
            seed=914202413,
        )
        cs = CallawaySantAnna(
            estimation_method="dr",
            control_group="not_yet_treated",
        )
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
        )
        gt = results.group_time_effects
        first_key = sorted(gt.keys())[0]
        assert abs(gt[first_key]["effect"] - 1.0) < 0.5

    @pytest.mark.slow
    def test_significance_level_and_cbands(self, ci_params):
        """Ported from R did::test-att_gt.R: 'significance level...'

        CI ordering: alp=0.05 narrower than alp=0.01.
        cband CI wider than pointwise CI.
        """
        n_boot = ci_params.bootstrap(499, min_n=199)
        data = build_csdid_sim_data(n=5000, time_periods=4, seed=914202414)

        cs05 = CallawaySantAnna(
            estimation_method="dr",
            alpha=0.05,
            n_bootstrap=n_boot,
            cband=True,
            seed=1234,
        )
        res05 = cs05.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )

        cs01 = CallawaySantAnna(
            estimation_method="dr",
            alpha=0.01,
            n_bootstrap=n_boot,
            cband=True,
            seed=1234,
        )
        res01 = cs01.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )

        # 5% CI should be narrower than (or equal to) 1% CI
        if (
            res05.event_study_effects
            and res01.event_study_effects
            and res05.cband_crit_value
            and res01.cband_crit_value
        ):
            assert res05.cband_crit_value <= res01.cband_crit_value + 0.01

    def test_some_units_treated_first_period(self):
        """Ported from R did::test-att_gt.R: 'some units treated in first period'

        When data starts at period 2 but G=2 exists, units are treated in
        the first observed period. R warns; diff-diff should either warn
        or handle by dropping those units (G=2 has no pre-treatment period).
        """
        data = build_csdid_sim_data(n=5000, time_periods=4, seed=914202415)
        # Drop period 1 so that G=2 units are "treated in first period"
        data = data[data["period"] >= 2].reset_index(drop=True)

        cs = CallawaySantAnna(estimation_method="reg")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        # G=2 is treated in the first observed period, so it has no valid base
        # period -> all its (g,t) cells are non-estimable. They are now materialized
        # as NaN entries (skip_reason="missing_period") rather than omitted, so G=2
        # contributes no FINITE estimate (the prior "excluded from the analysis"
        # intent: it is not silently dropped, but it is never estimated).
        g2_cells = [v for (g, t), v in results.group_time_effects.items() if g == 2]
        assert g2_cells, "G=2 cells should be materialized (as NaN), not silently dropped"
        assert all(
            np.isnan(v["effect"]) and v["skip_reason"] == "missing_period" for v in g2_cells
        ), "G=2 (no pre-treatment period) must be all-NaN (missing_period), never estimated"


class TestCSDIDBugFixRegressions:
    """Bug-fix regression tests from R's test-user_bug_fixes.R."""

    def test_column_named_t1_no_crash(self):
        """Ported from R did::test-user_bug_fixes.R: 'column named t1...'

        Extra columns with confusing names should not crash the estimator.
        """
        data = build_csdid_sim_data(n=1000, time_periods=4, seed=914202416)
        data["t1"] = 1  # R bug: extra column named t1

        cs = CallawaySantAnna(estimation_method="reg", control_group="not_yet_treated")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results is not None
        assert len(results.group_time_effects) > 0

    def test_missing_covariates_warning(self):
        """Ported from R did::test-user_bug_fixes.R: 'missing covariates'

        NaN in a covariate column should warn but still run.
        """
        data = build_csdid_sim_data(n=2000, time_periods=4, seed=914202417)
        data.loc[0, "X"] = np.nan

        cs = CallawaySantAnna(estimation_method="reg", control_group="not_yet_treated")
        with pytest.warns(UserWarning, match="Missing values in covariates"):
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                covariates=["X"],
            )
        assert results is not None

    def test_fewer_time_periods_than_groups(self, ci_params):
        """Ported from R did::test-user_bug_fixes.R: 'fewer time periods than groups'

        Drop periods 2 and 5 from a 6-period DGP.
        Aggregations should have no NaN.
        """
        (n,) = ci_params.grid([5000])
        T = 6
        data = build_csdid_sim_data(
            n=n,
            time_periods=T,
            te=0.0,
            te_e=np.arange(1, T + 1, dtype=float),
            seed=914202418,
        )
        data = data[~data["period"].isin([2, 5])].reset_index(drop=True)

        cs = CallawaySantAnna(estimation_method="dr")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="all",
        )

        # Dynamic aggregation should have no NaN ATTs
        if results.event_study_effects:
            for e, eff in results.event_study_effects.items():
                assert np.isfinite(eff["effect"]), f"NaN in event_study at e={e}"

        # Group aggregation should have no NaN ATTs
        if results.group_effects:
            for g, eff in results.group_effects.items():
                assert np.isfinite(eff["effect"]), f"NaN in group_effects at g={g}"

    def test_zero_pretreatment_outcomes(self):
        """Ported from R did::test-user_bug_fixes.R: '0 pre-treatment estimates...'

        All pre-treatment outcomes are 0 for both treated and control groups.
        DiD of zeros minus zeros should be exactly 0 for pre-treatment ATT.
        R checks ATT(g=9, t=7) == 0 with exact equality.

        R's test uses not_yet_treated with NO never-treated units, which also
        validates that not_yet_treated works without a never-treated group.
        """
        rng = np.random.default_rng(914202419)
        n_per_group = 200
        # No never-treated group — matches R's original test
        groups = [8, 9, 10]
        periods = list(range(6, 11))

        rows = []
        uid = 0
        for g in groups:
            for _ in range(n_per_group):
                uid += 1
                for t in periods:
                    if t < g:
                        y = 0.0
                    else:
                        y = 2.0 + rng.standard_normal() * 0.5
                    rows.append(
                        {
                            "unit": uid,
                            "period": t,
                            "first_treat": g,
                            "outcome": y,
                        }
                    )
        data = pd.DataFrame(rows)

        cs = CallawaySantAnna(
            control_group="not_yet_treated",
            base_period="universal",
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

        # Pre-treatment effects: DiD of 0-0 vs 0-0 should be 0
        gt = results.group_time_effects
        pre_effects = {k: v for k, v in gt.items() if k[1] < k[0]}
        for (g, t), eff in pre_effects.items():
            # Non-estimable pre-cells are now materialized as NaN (e.g. the last
            # cohort under not_yet_treated has no controls); skip them. Finite
            # pre-treatment cells (DiD of 0-0 vs 0-0) must still be ~0.
            if np.isnan(eff["effect"]):
                continue
            assert abs(eff["effect"]) < 0.01, (
                f"Pre-treatment ATT(g={g}, t={t}) = {eff['effect']:.4f}, " "expected 0"
            )

    def test_anticipation_window_classification(self):
        """Ported from R did::test-user_bug_fixes.R: 'groups treated after max(t)...'

        With anticipation=2, group 6 (treated after max(t)=5) should NOT be
        coerced to never-treated, since anticipation starts at period 4.
        With anticipation=0, group 6 should be treated as never-treated
        (no post-treatment periods observed).

        Note: diff-diff may still include group 6 in results with ant=0 if
        it keeps groups treated after max(t) rather than coercing them.
        The key test is that ant=2 includes group 6 as treated.
        """
        rng = np.random.default_rng(20250228)
        n_units = 600
        periods = list(range(1, 6))

        rows = []
        for uid in range(1, n_units + 1):
            if uid <= 200:
                g = 0
            elif uid <= 400:
                g = 4
            else:
                g = 6
            for t in periods:
                rows.append(
                    {
                        "unit": uid,
                        "period": t,
                        "first_treat": g,
                        "outcome": rng.standard_normal(),
                    }
                )
        data = pd.DataFrame(rows)

        # With anticipation=2: group 6 anticipates at period 4, should be treated
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs2 = CallawaySantAnna(anticipation=2)
            results2 = cs2.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        groups2 = set(k[0] for k in results2.group_time_effects.keys())
        assert 6 in groups2, f"ant=2 groups: {groups2}, expected 6 in set"

        # With anticipation=0: group 6 beyond max(t)=5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs0 = CallawaySantAnna(anticipation=0)
            results0 = cs0.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        groups0 = set(k[0] for k in results0.group_time_effects.keys())
        # Group 4 must be present
        assert 4 in groups0, f"ant=0 groups: {groups0}, expected 4 in set"
        # With ant=2 there should be more groups than ant=0
        assert len(groups2) >= len(
            groups0
        ), f"ant=2 should have >= groups: {len(groups2)} vs {len(groups0)}"


class TestCSDIDTwoGroupValidation:
    """Two-group DGP from R's test_sim_data_2_groups.R."""

    def test_two_group_known_effect(self, ci_params):
        """Ported from R did::test_sim_data_2_groups.R: 'att_gt works with 2 groups'

        G ∈ {0, 3}, true ATT ≈ 3 at period 3.
        4 specifications should agree within 0.001.
        """
        (n,) = ci_params.grid([5000])
        data = build_two_group_sim_data(n=n, seed=20241017)

        specs = [
            ("never_treated", "varying"),
            ("never_treated", "universal"),
            ("not_yet_treated", "varying"),
            ("not_yet_treated", "universal"),
        ]
        att_values = []
        for cg, bp in specs:
            cs = CallawaySantAnna(
                estimation_method="reg",
                control_group=cg,
                base_period=bp,
            )
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
            gt = results.group_time_effects
            # ATT(g=3, t=3) should ≈ 3
            if (3, 3) in gt:
                att = gt[(3, 3)]["effect"]
                att_values.append(att)
                assert abs(att - 3.0) < 0.5, f"{cg}/{bp}: ATT(3,3) = {att:.3f}, expected ≈ 3"

        # Cross-spec consistency: all 4 should be within 0.1 of each other
        if len(att_values) >= 2:
            spread = max(att_values) - min(att_values)
            assert spread < 0.1, f"Cross-spec spread = {spread:.4f}, expected < 0.1"


# ---------------------------------------------------------------------------
# Tier 2: Golden value tests (R-generated data, strict tolerance)
# ---------------------------------------------------------------------------


class TestCSDIDGoldenValues:
    """Compare against R ``did::att_gt()`` exact output."""

    def _run_scenario(
        self,
        golden_values,
        scenario_name,
        covariates=None,
        est_method="dr",
        control_group="never_treated",
        base_period="varying",
        anticipation=0,
    ):
        """Helper to run a golden value scenario."""
        scenario = golden_values[scenario_name]
        data = _golden_to_df(scenario["data"])
        if covariates is None and "X" in data.columns:
            xformla = scenario.get("params", {}).get("xformla", "~1")
            covariates = ["X"] if xformla == "~X" else None

        cs = CallawaySantAnna(
            estimation_method=est_method,
            control_group=control_group,
            base_period=base_period,
            anticipation=anticipation,
        )
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=covariates,
        )
        return results, scenario["results"]

    def test_golden_default_dgp_dr(self, golden_values):
        """ATT(g,t) from default no-covariate DGP match R within 1e-2."""
        results, expected = self._run_scenario(
            golden_values,
            "default_no_covariates_dr",
            covariates=None,
            est_method="dr",
        )
        r_gt = expected["group_time"]
        for i, (g, t) in enumerate(zip(r_gt["group"], r_gt["time"])):
            g, t = int(g), int(t)
            if (g, t) in results.group_time_effects:
                py_att = results.group_time_effects[(g, t)]["effect"]
                r_att = r_gt["att"][i]
                assert (
                    abs(py_att - r_att) < 0.02
                ), f"ATT(g={g}, t={t}): Python={py_att:.6f}, R={r_att:.6f}"

    def test_golden_default_dgp_reg(self, golden_values):
        """Regression method with covariates matches R - ATT AND per-cell SE.

        The SE assertion pins the ``DRDID::reg_did_panel`` influence function
        (including the OLS estimation-effect term ``asy.lin.rep.ols @ M1``).
        Before that term was added the per-cell SEs sat 4-13% from these
        fixtures; observed agreement is now ~5e-12 (deterministic OLS), so
        the 1e-8 tolerance is pure platform headroom.
        """
        if "with_covariates_reg" not in golden_values:
            pytest.skip("Scenario not in golden values")
        results, expected = self._run_scenario(
            golden_values,
            "with_covariates_reg",
            est_method="reg",
        )
        r_gt = expected["group_time"]
        for i, (g, t) in enumerate(zip(r_gt["group"], r_gt["time"])):
            g, t = int(g), int(t)
            if (g, t) in results.group_time_effects:
                eff = results.group_time_effects[(g, t)]
                r_att = r_gt["att"][i]
                assert (
                    abs(eff["effect"] - r_att) < 0.02
                ), f"REG ATT(g={g}, t={t}): Py={eff['effect']:.6f}, R={r_att:.6f}"
                assert (
                    abs(eff["se"] - r_gt["se"][i]) < 1e-8
                ), f"REG SE(g={g}, t={t}): Py={eff['se']:.10f}, R={r_gt['se'][i]:.10f}"

    def test_golden_default_dgp_ipw(self, golden_values):
        """IPW method matches R - ATT AND per-cell SE.

        The SE assertion pins the ``DRDID::std_ipw_did_panel`` influence
        function (Hajek terms + the PS estimation-effect correction
        ``asy.lin.rep.ps @ M2``) and the IF-based per-cell SE. Before the
        fix the per-cell SEs were ~7x these fixtures (a weighted population
        variance was never scaled by an effective sample size). Observed
        agreement is now ~5e-12; 1e-6 leaves headroom for IRLS-vs-glm logit
        convergence differences across platforms/BLAS.
        """
        if "with_covariates_ipw" not in golden_values:
            pytest.skip("Scenario not in golden values")
        results, expected = self._run_scenario(
            golden_values,
            "with_covariates_ipw",
            est_method="ipw",
        )
        r_gt = expected["group_time"]
        for i, (g, t) in enumerate(zip(r_gt["group"], r_gt["time"])):
            g, t = int(g), int(t)
            if (g, t) in results.group_time_effects:
                eff = results.group_time_effects[(g, t)]
                r_att = r_gt["att"][i]
                assert (
                    abs(eff["effect"] - r_att) < 0.05
                ), f"IPW ATT(g={g}, t={t}): Py={eff['effect']:.6f}, R={r_att:.6f}"
                assert (
                    abs(eff["se"] - r_gt["se"][i]) < 1e-6
                ), f"IPW SE(g={g}, t={t}): Py={eff['se']:.10f}, R={r_gt['se'][i]:.10f}"

    def test_golden_default_dgp_dr_covariates(self, golden_values):
        """DR method WITH covariates matches R `did` att AND se.

        Fills the previously-unasserted ``with_covariates_dr`` golden scenario
        (``test_att_recovery_with_covariates_dr`` checks recovery of the TRUE
        effect, not parity vs R). The covariate outcome-regression nuisance is
        now fit through the shared scale-equilibrated ``solve_ols`` (matching
        TripleDifference + R's ``lm()``/QR); this pins both the ATT(g,t) and the
        analytical IF SE against R. Tolerances are empirical: DR ATT(g,t) agrees
        with R to ~1e-3 (DR small-sample numerics) and the DR SE to ~1e-4.
        (``reg``/``ipw`` per-cell SEs are pinned in their own golden tests
        above, now that both carry the DRDID estimation-effect IF terms.)
        """
        if "with_covariates_dr" not in golden_values:
            pytest.skip("Scenario not in golden values")
        results, expected = self._run_scenario(
            golden_values,
            "with_covariates_dr",
            est_method="dr",
        )
        r_gt = expected["group_time"]
        for i, (g, t) in enumerate(zip(r_gt["group"], r_gt["time"])):
            g, t = int(g), int(t)
            if (g, t) in results.group_time_effects:
                eff = results.group_time_effects[(g, t)]
                assert (
                    abs(eff["effect"] - r_gt["att"][i]) < 2e-3
                ), f"DR ATT(g={g}, t={t}): Py={eff['effect']:.6f}, R={r_gt['att'][i]:.6f}"
                assert (
                    abs(eff["se"] - r_gt["se"][i]) < 2e-3
                ), f"DR SE(g={g}, t={t}): Py={eff['se']:.6f}, R={r_gt['se'][i]:.6f}"

    def test_golden_dynamic_effects(self, golden_values):
        """Event study aggregation ATTs match R."""
        if "dynamic_effects" not in golden_values:
            pytest.skip("Scenario not in golden values")
        scenario = golden_values["dynamic_effects"]
        data = _golden_to_df(scenario["data"])

        cs = CallawaySantAnna(estimation_method="reg")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )
        r_dyn = scenario["results"]["dynamic"]
        if results.event_study_effects:
            for i, e in enumerate(r_dyn["egt"]):
                e = int(e)
                if e in results.event_study_effects:
                    py_att = results.event_study_effects[e]["effect"]
                    r_att = r_dyn["att"][i]
                    assert (
                        abs(py_att - r_att) < 0.05
                    ), f"Dynamic e={e}: Py={py_att:.4f}, R={r_att:.4f}"

    def test_golden_non_consecutive_periods(self, golden_values):
        """Non-consecutive periods event study matches R."""
        if "non_consecutive_periods" not in golden_values:
            pytest.skip("Scenario not in golden values")
        scenario = golden_values["non_consecutive_periods"]
        data = _golden_to_df(scenario["data"])

        cs = CallawaySantAnna(estimation_method="reg")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )
        r_dyn = scenario["results"]["dynamic"]
        if results.event_study_effects:
            for i, e in enumerate(r_dyn["egt"]):
                e = int(e)
                if e in results.event_study_effects:
                    py_att = results.event_study_effects[e]["effect"]
                    r_att = r_dyn["att"][i]
                    # Non-consecutive periods with small n have higher variance
                    assert (
                        abs(py_att - r_att) < 0.5
                    ), f"Nonconsec e={e}: Py={py_att:.4f}, R={r_att:.4f}"

    def test_golden_anticipation(self, golden_values):
        """Anticipation=1 matches R."""
        if "anticipation" not in golden_values:
            pytest.skip("Scenario not in golden values")
        scenario = golden_values["anticipation"]
        data = _golden_to_df(scenario["data"])

        cs = CallawaySantAnna(
            estimation_method="dr",
            anticipation=1,
        )
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )
        r_dyn = scenario["results"]["dynamic"]
        if results.event_study_effects:
            for i, e in enumerate(r_dyn["egt"]):
                e = int(e)
                if e in results.event_study_effects:
                    py_att = results.event_study_effects[e]["effect"]
                    r_att = r_dyn["att"][i]
                    assert (
                        abs(py_att - r_att) < 0.1
                    ), f"Anticipation e={e}: Py={py_att:.4f}, R={r_att:.4f}"

    def test_golden_varying_vs_universal(self, golden_values):
        """Both base period modes match R."""
        if "varying_vs_universal" not in golden_values:
            pytest.skip("Scenario not in golden values")
        scenario = golden_values["varying_vs_universal"]
        data = _golden_to_df(scenario["data"])

        for bp in ["varying", "universal"]:
            cs = CallawaySantAnna(estimation_method="dr", base_period=bp)
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                covariates=["X"],
            )
            r_gt = scenario["results"][f"{bp}_gt"]
            for i, (g, t) in enumerate(zip(r_gt["group"], r_gt["time"])):
                g, t = int(g), int(t)
                if (g, t) in results.group_time_effects:
                    py_att = results.group_time_effects[(g, t)]["effect"]
                    r_att = r_gt["att"][i]
                    assert (
                        abs(py_att - r_att) < 0.05
                    ), f"{bp} ATT(g={g},t={t}): Py={py_att:.4f}, R={r_att:.4f}"

    def test_golden_two_group(self, golden_values):
        """Two-group 4-spec comparison matches R within 1e-2."""
        if "two_group" not in golden_values:
            pytest.skip("Scenario not in golden values")
        scenario = golden_values["two_group"]
        data = _golden_to_df(scenario["data"])

        spec_map = {
            "nt_varying": ("never_treated", "varying"),
            "nt_universal": ("never_treated", "universal"),
            "nyt_varying": ("not_yet_treated", "varying"),
            "nyt_universal": ("not_yet_treated", "universal"),
        }
        for spec_name, (cg, bp) in spec_map.items():
            if spec_name not in scenario["results"]:
                continue
            cs = CallawaySantAnna(
                estimation_method="reg",
                control_group=cg,
                base_period=bp,
            )
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
            r_gt = scenario["results"][spec_name]
            for i, (g, t) in enumerate(zip(r_gt["group"], r_gt["time"])):
                g, t = int(g), int(t)
                if (g, t) in results.group_time_effects:
                    py_att = results.group_time_effects[(g, t)]["effect"]
                    r_att = r_gt["att"][i]
                    assert abs(py_att - r_att) < 0.02, (
                        f"{spec_name} ATT(g={g},t={t}): " f"Py={py_att:.6f}, R={r_att:.6f}"
                    )

    def test_golden_fewer_periods(self, golden_values):
        """Fewer-periods-than-groups matches R."""
        if "fewer_periods" not in golden_values:
            pytest.skip("Scenario not in golden values")
        scenario = golden_values["fewer_periods"]
        data = _golden_to_df(scenario["data"])

        cs = CallawaySantAnna(estimation_method="dr")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="all",
        )
        r_gt = scenario["results"]["group_time"]
        for i, (g, t) in enumerate(zip(r_gt["group"], r_gt["time"])):
            g, t = int(g), int(t)
            if (g, t) in results.group_time_effects:
                py_att = results.group_time_effects[(g, t)]["effect"]
                # Skip cells we materialize as non-estimable (e.g. a gapped panel
                # where the base period g-1 is not observed -> missing_period). R
                # falls back to an available base and reports a value where our
                # impl does not; compare only cells both estimate (R-parity on the
                # finite cells, which is what this golden test pins).
                if not np.isfinite(py_att):
                    continue
                r_att = r_gt["att"][i]
                assert abs(py_att - r_att) < 0.05, (
                    f"Fewer periods ATT(g={g},t={t}): " f"Py={py_att:.4f}, R={r_att:.4f}"
                )

    def test_golden_zero_pretreatment(self, golden_values):
        """Zero pre-treatment outcomes match R.

        R's test uses not_yet_treated with no never-treated units.
        """
        if "zero_pretreatment" not in golden_values:
            pytest.skip("Scenario not in golden values")
        scenario = golden_values["zero_pretreatment"]
        data = _golden_to_df(scenario["data"])

        for bp in ["universal", "varying"]:
            cs = CallawaySantAnna(
                control_group="not_yet_treated",
                base_period=bp,
            )
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
            r_gt = scenario["results"][f"{bp}_gt"]
            for i, (g, t) in enumerate(zip(r_gt["group"], r_gt["time"])):
                g, t = int(g), int(t)
                if (g, t) in results.group_time_effects:
                    py_att = results.group_time_effects[(g, t)]["effect"]
                    r_att = r_gt["att"][i]
                    assert abs(py_att - r_att) < 0.05, (
                        f"ZeroPre {bp} ATT(g={g},t={t}): " f"Py={py_att:.4f}, R={r_att:.4f}"
                    )

    def test_golden_simple_aggregation_se(self, golden_values):
        """Simple (overall) aggregated ATT and SE match R ``aggte(type="simple")``.

        Enables the previously-unasserted ``simple`` block of the
        ``default_no_covariates_dr`` golden scenario (analytical SEs,
        ``bstrap=FALSE``). Observed agreement is ~1e-12; the 2e-3 tolerance
        matches the per-cell DR SE precedent above.
        """
        if "default_no_covariates_dr" not in golden_values:
            pytest.skip("Scenario not in golden values")
        scenario = golden_values["default_no_covariates_dr"]
        data = _golden_to_df(scenario["data"])

        cs = CallawaySantAnna(estimation_method="dr")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="simple",
        )
        r_simple = scenario["results"]["simple"]
        assert (
            abs(results.att - r_simple["att"]) < 2e-3
        ), f"Simple ATT: Py={results.att:.6f}, R={r_simple['att']:.6f}"
        assert (
            abs(results.se - r_simple["se"]) < 2e-3
        ), f"Simple SE: Py={results.se:.6f}, R={r_simple['se']:.6f}"

    def test_golden_event_study_aggregation_se(self, golden_values):
        """Event-study aggregated ATT and SE match R ``aggte(type="dynamic")``
        under BOTH base-period modes.

        Enables the previously-unasserted ``varying_dynamic`` /
        ``universal_dynamic`` blocks of the ``varying_vs_universal`` golden
        scenario (analytical SEs). R reports NA at the universal reference
        period e=-1 (identically-zero estimate); NA entries are skipped -
        this pins parity on the estimated horizons (observed agreement
        ~7e-6).

        The ``two_period`` and ``dynamic_effects`` (reg) aggregated SEs -
        previously 3-20% off because the reg IF lacked the OLS
        estimation-effect term - are now enabled in
        ``test_golden_reg_aggregation_se`` below.
        """
        if "varying_vs_universal" not in golden_values:
            pytest.skip("Scenario not in golden values")
        scenario = golden_values["varying_vs_universal"]
        data = _golden_to_df(scenario["data"])

        for bp in ["varying", "universal"]:
            cs = CallawaySantAnna(estimation_method="dr", base_period=bp)
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                covariates=["X"],
                aggregate="event_study",
            )
            blk = scenario["results"][f"{bp}_dynamic"]
            n_checked = 0
            for e, r_att, r_se in zip(blk["egt"], blk["att"], blk["se"]):
                if r_att == "NA" or r_se == "NA":
                    continue
                es = results.event_study_effects.get(int(e))
                assert es is not None, f"{bp}: missing event time e={e}"
                assert (
                    abs(es["effect"] - float(r_att)) < 2e-3
                ), f"{bp} dyn ATT(e={e}): Py={es['effect']:.6f}, R={r_att}"
                assert (
                    abs(es["se"] - float(r_se)) < 2e-3
                ), f"{bp} dyn SE(e={e}): Py={es['se']:.6f}, R={r_se}"
                n_checked += 1
            assert n_checked >= 8, f"{bp}: too few comparable horizons"

    def test_golden_reg_group_time_se(self, golden_values):
        """Per-cell reg SEs match R across the remaining reg golden scenarios.

        Complements ``test_golden_default_dgp_reg``: ``two_period`` (single
        2x2 cell) and ``dynamic_effects`` (3 cohorts x 3 periods, including
        PRE-treatment cells whose varying base period is t-1). These pin the
        ``DRDID::reg_did_panel`` IF on cell shapes the default scenario does
        not cover. Observed agreement ~5e-12 (deterministic OLS); 1e-8 is
        platform headroom.
        """
        for name in ("two_period", "dynamic_effects"):
            if name not in golden_values:
                pytest.skip("Scenario not in golden values")
            scenario = golden_values[name]
            data = _golden_to_df(scenario["data"])
            cs = CallawaySantAnna(estimation_method="reg")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                covariates=["X"],
            )
            r_gt = scenario["results"]["group_time"]
            groups = np.atleast_1d(r_gt["group"])
            times = np.atleast_1d(r_gt["time"])
            atts = np.atleast_1d(r_gt["att"])
            ses = np.atleast_1d(r_gt["se"])
            for g, t, r_att, r_se in zip(groups, times, atts, ses):
                eff = results.group_time_effects[(int(g), int(t))]
                assert (
                    abs(eff["effect"] - r_att) < 1e-8
                ), f"{name} ATT(g={g},t={t}): Py={eff['effect']:.10f}, R={r_att:.10f}"
                assert (
                    abs(eff["se"] - r_se) < 1e-8
                ), f"{name} SE(g={g},t={t}): Py={eff['se']:.10f}, R={r_se:.10f}"

    def test_golden_reg_aggregation_se(self, golden_values):
        """Aggregated reg ATTs and SEs match R ``aggte`` (simple/group/dynamic).

        Enables the previously-unasserted aggregation blocks of the two reg
        golden scenarios: ``two_period`` (simple + group + dynamic of a
        single cell) and ``dynamic_effects`` (dynamic, all horizons incl.
        pre-treatment). Before the reg IF carried the OLS estimation-effect
        term these SEs sat 3-20% from the fixtures while the ATTs matched at
        1e-11. Observed post-fix agreement is ~5e-12 (deterministic OLS +
        the aggregation machinery already validated at 1e-12 on the dr
        scenarios); 1e-8 is platform headroom.
        """
        if "two_period" not in golden_values or "dynamic_effects" not in golden_values:
            pytest.skip("Scenario not in golden values")

        # two_period: one (2,2) cell, so simple == group == dynamic(e=0)
        scenario = golden_values["two_period"]
        data = _golden_to_df(scenario["data"])
        cs = CallawaySantAnna(estimation_method="reg")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="all",
        )
        r = scenario["results"]
        assert abs(results.overall_att - r["simple"]["att"]) < 1e-8
        assert abs(results.overall_se - r["simple"]["se"]) < 1e-8
        g_att = np.atleast_1d(r["group"]["att"])[0]
        g_se = np.atleast_1d(r["group"]["se"])[0]
        grp = results.group_effects[2]
        assert abs(grp["effect"] - g_att) < 1e-8
        assert abs(grp["se"] - g_se) < 1e-8
        d_att = np.atleast_1d(r["dynamic"]["att"])[0]
        d_se = np.atleast_1d(r["dynamic"]["se"])[0]
        es0 = results.event_study_effects[0]
        assert abs(es0["effect"] - d_att) < 1e-8
        assert abs(es0["se"] - d_se) < 1e-8

        # dynamic_effects: full event study, pre and post horizons
        scenario = golden_values["dynamic_effects"]
        data = _golden_to_df(scenario["data"])
        cs = CallawaySantAnna(estimation_method="reg")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="event_study",
        )
        r_dyn = scenario["results"]["dynamic"]
        for i, e in enumerate(r_dyn["egt"]):
            es = results.event_study_effects.get(int(e))
            assert es is not None, f"missing event time e={e}"
            assert (
                abs(es["effect"] - r_dyn["att"][i]) < 1e-8
            ), f"dyn ATT(e={e}): Py={es['effect']:.10f}, R={r_dyn['att'][i]:.10f}"
            assert (
                abs(es["se"] - r_dyn["se"][i]) < 1e-8
            ), f"dyn SE(e={e}): Py={es['se']:.10f}, R={r_dyn['se'][i]:.10f}"

    def test_golden_ipw_aggregation_se_vs_r_did_251(self, golden_values):
        """Aggregated ipw SEs match R did 2.5.1 ``aggte`` on the golden data.

        The golden JSON carries no aggregation blocks for the ipw scenario,
        so the expected values are hardcoded from a fresh R run (did 2.5.1,
        DRDID 1.3.0, ``bstrap=FALSE``, computed 2026-07-05 on the EXACT
        ``with_covariates_ipw`` fixture data; see
        benchmarks/R/generate_csdid_test_values.R which will fold these into
        the JSON on its next regeneration). Discriminating: the pre-fix
        Python simple SE was 0.32125722 (missing PS estimation-effect
        correction, ~2.4% off). Observed post-fix agreement ~1e-10; 1e-6
        covers the fixtures' 8-decimal quantization + IRLS-vs-glm headroom.
        """
        if "with_covariates_ipw" not in golden_values:
            pytest.skip("Scenario not in golden values")
        scenario = golden_values["with_covariates_ipw"]
        data = _golden_to_df(scenario["data"])
        cs = CallawaySantAnna(estimation_method="ipw")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["X"],
            aggregate="all",
        )
        # R did 2.5.1: aggte(type="simple", bstrap=FALSE, cband=FALSE)
        assert abs(results.overall_att - 0.68857256) < 1e-6
        assert abs(results.overall_se - 0.31382230) < 1e-6
        # aggte(type="dynamic"): (att.egt, se.egt) per event time
        r_dynamic = {
            -2: (-0.58336476, 0.27944371),
            -1: (-0.40369753, 0.22022753),
            0: (0.68317989, 0.20153325),
            1: (0.69816671, 0.38903747),
            2: (0.68521672, 0.64844936),
        }
        for e, (r_att, r_se) in r_dynamic.items():
            es = results.event_study_effects[e]
            assert abs(es["effect"] - r_att) < 1e-6, f"ipw dyn ATT(e={e})"
            assert abs(es["se"] - r_se) < 1e-6, f"ipw dyn SE(e={e})"
        # aggte(type="group"): (att.egt, se.egt) per cohort
        r_group = {
            2: (0.91085897, 0.44399703),
            3: (0.60976619, 0.33782836),
            4: (0.24708101, 0.30682305),
        }
        for g, (r_att, r_se) in r_group.items():
            grp = results.group_effects[g]
            assert abs(grp["effect"] - r_att) < 1e-6, f"ipw group ATT(g={g})"
            assert abs(grp["se"] - r_se) < 1e-6, f"ipw group SE(g={g})"
