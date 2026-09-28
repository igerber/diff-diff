"""
Tests for Borusyak-Jaravel-Spiess (2024) imputation DiD estimator.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.imputation import (
    ImputationBootstrapResults,
    ImputationDiD,
    ImputationDiDResults,
    imputation_did,
)
from diff_diff.survey import SurveyDesign

# ---------------------------------------------------------------------------
# Rows M-021/M-022 (+ M-118/M-119): ImputationDiD / TwoStageDiD
# ``fit(aggregate=, balance_e=)`` is deprecated (3.9, removed 4.0) and warns on
# ANY supplied value. The deprecated fit-time route is kept DELIBERATELY here:
# these tests pin FIT-TIME surface behaviour (bit-equality grids, bootstrap
# aggregation, R/Stata parity, replicate overrides, native effect dicts) that
# the post-fit ``results.aggregate(...)`` container route does not reproduce
# shape-for-shape. The shim warning is therefore filtered BY MESSAGE, scoped to
# these two estimators only - every other FutureWarning (including the other
# estimators' aggregate() shims) still surfaces.
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.filterwarnings(
    r"ignore:(ImputationDiD|TwoStageDiD)\.fit\((aggregate=|balance_e=|aggregate= / balance_e=)\):FutureWarning"
)

# =============================================================================
# Shared test data generation
# =============================================================================


def generate_test_data(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_effect: float = 2.0,
    never_treated_frac: float = 0.3,
    dynamic_effects: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic staggered adoption data for testing."""
    rng = np.random.default_rng(seed)

    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    n_never = int(n_units * never_treated_frac)
    n_treated = n_units - n_never

    cohort_periods = np.array([3, 5, 7])
    first_treat = np.zeros(n_units, dtype=int)
    if n_treated > 0:
        cohort_assignments = rng.choice(len(cohort_periods), size=n_treated)
        first_treat[n_never:] = cohort_periods[cohort_assignments]

    first_treat_expanded = np.repeat(first_treat, n_periods)

    unit_fe = rng.standard_normal(n_units) * 2.0
    time_fe = np.linspace(0, 1, n_periods)

    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
    relative_time = times - first_treat_expanded

    if dynamic_effects:
        dynamic_mult = 1 + 0.1 * np.maximum(relative_time, 0)
    else:
        dynamic_mult = np.ones_like(relative_time, dtype=float)

    effect = treatment_effect * dynamic_mult

    outcomes = (
        unit_fe_expanded + time_fe_expanded + effect * post + rng.standard_normal(len(units)) * 0.5
    )

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcomes,
            "first_treat": first_treat_expanded,
        }
    )


# =============================================================================
# TestImputationDiD
# =============================================================================


class TestImputationDiD:
    """Tests for ImputationDiD estimator."""

    def test_basic_fit(self):
        """Test basic model fitting."""
        data = generate_test_data()

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert est.is_fitted_
        assert isinstance(results, ImputationDiDResults)
        assert results.overall_att is not None
        assert results.overall_se > 0
        assert results.n_treated_obs > 0
        assert results.n_untreated_obs > 0
        assert results.n_treated_units > 0
        assert results.n_control_units > 0
        assert len(results.groups) == 3
        assert len(results.time_periods) == 10

    def test_positive_treatment_effect(self):
        """Test recovery of positive treatment effect."""
        data = generate_test_data(treatment_effect=3.0, seed=123)

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_att > 0
        # Effect should be close to 3.0 (dynamic effects add some)
        assert abs(results.overall_att - 3.0) < 2 * results.overall_se + 1.5

    def test_zero_treatment_effect(self):
        """Test with no treatment effect."""
        data = generate_test_data(treatment_effect=0.0, seed=456)

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert abs(results.overall_att) < 3 * results.overall_se + 0.5

    def test_aggregate_simple(self):
        """Test that default aggregate computes overall ATT."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_att is not None
        assert results.overall_se > 0
        assert results.event_study_effects is None
        assert results.group_effects is None

    def test_aggregate_event_study(self):
        """Test event study aggregation."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0
        assert results.group_effects is None

        for h, eff in results.event_study_effects.items():
            assert "effect" in eff
            assert "se" in eff
            assert "t_stat" in eff
            assert "p_value" in eff
            assert "conf_int" in eff
            assert "n_obs" in eff

    def test_aggregate_group(self):
        """Test group aggregation."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        assert results.group_effects is not None
        assert len(results.group_effects) == 3  # 3 cohorts
        assert results.event_study_effects is None

        for g, eff in results.group_effects.items():
            assert "effect" in eff
            assert "se" in eff
            assert eff["se"] > 0

    def test_aggregate_all(self):
        """Test 'all' aggregation computes both event study and group."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )

        assert results.event_study_effects is not None
        assert results.group_effects is not None

    def test_covariates(self):
        """Test estimation with covariates."""
        data = generate_test_data()
        rng = np.random.default_rng(99)
        data["x1"] = rng.standard_normal(len(data))
        data["x2"] = rng.standard_normal(len(data))

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
        )

        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_anticipation(self):
        """Test anticipation parameter."""
        data = generate_test_data()

        est0 = ImputationDiD(anticipation=0)
        results0 = est0.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        est1 = ImputationDiD(anticipation=1)
        results1 = est1.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # With anticipation=1, more obs are treated, fewer untreated
        assert results1.n_treated_obs > results0.n_treated_obs

        # Reference period changes
        ref0 = [h for h, e in results0.event_study_effects.items() if e.get("n_obs", 1) == 0]
        ref1 = [h for h, e in results1.event_study_effects.items() if e.get("n_obs", 1) == 0]
        assert -1 in ref0
        assert -2 in ref1

    def test_balance_e(self):
        """Test balance_e restricts event study to balanced cohorts."""
        data = generate_test_data()

        est = ImputationDiD()
        results_unbal = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        results_bal = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            balance_e=2,
        )

        # Balanced should have same or fewer horizons
        assert len(results_bal.event_study_effects) <= len(results_unbal.event_study_effects) + 5

    def test_horizon_max(self):
        """Test horizon_max caps event study horizons."""
        data = generate_test_data()

        est = ImputationDiD(horizon_max=3)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        for h in results.event_study_effects:
            if results.event_study_effects[h].get("n_obs", 0) > 0:
                assert abs(h) <= 3

    def test_summary(self):
        """Test summary output."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )

        summary = results.summary()
        assert "Imputation DiD" in summary
        assert "ATT" in summary
        assert "Event Study" in summary
        assert "Group" in summary

    def test_to_dataframe_observation(self):
        """Test to_dataframe at observation level."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        df = results.to_dataframe("observation")
        assert "tau_hat" in df.columns
        assert "weight" in df.columns
        assert len(df) == results.n_treated_obs

    def test_to_dataframe_event_study(self):
        """Test to_dataframe at event study level."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        df = results.to_dataframe("event_study")
        assert "relative_period" in df.columns
        assert "effect" in df.columns
        assert "se" in df.columns

    def test_to_dataframe_group(self):
        """Test to_dataframe at group level."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        df = results.to_dataframe("group")
        assert "group" in df.columns
        assert len(df) == 3

    def test_to_dataframe_errors(self):
        """Test to_dataframe raises on invalid level."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        with pytest.raises(ValueError, match="Unknown level"):
            results.to_dataframe("invalid")

        with pytest.raises(ValueError, match="Event study effects not computed"):
            results.to_dataframe("event_study")

    def test_get_params(self):
        """Test get_params returns all constructor parameters."""
        est = ImputationDiD(
            anticipation=1,
            alpha=0.10,
            n_bootstrap=100,
            seed=42,
            horizon_max=5,
            aux_partition="cohort",
        )
        params = est.get_params()

        assert params["anticipation"] == 1
        assert params["alpha"] == 0.10
        assert params["n_bootstrap"] == 100
        assert params["seed"] == 42
        assert params["horizon_max"] == 5
        assert params["aux_partition"] == "cohort"
        assert params["cluster"] is None
        assert params["rank_deficient_action"] == "warn"
        assert params["bootstrap_weights"] == "rademacher"

    def test_bootstrap_weights_in_get_set_params(self):
        """bootstrap_weights should appear in get_params and be settable."""
        est = ImputationDiD(bootstrap_weights="mammen")
        assert est.get_params()["bootstrap_weights"] == "mammen"
        est.set_params(bootstrap_weights="webb")
        assert est.bootstrap_weights == "webb"

    def test_bootstrap_weights_invalid_raises(self):
        """Invalid bootstrap_weights value should raise ValueError."""
        with pytest.raises(ValueError, match="bootstrap_weights"):
            ImputationDiD(bootstrap_weights="invalid")

    def test_set_params(self):
        """Test set_params modifies attributes."""
        est = ImputationDiD()
        est.set_params(alpha=0.10, anticipation=2)

        assert est.alpha == 0.10
        assert est.anticipation == 2

    def test_set_params_unknown(self):
        """Test set_params raises on unknown parameter."""
        est = ImputationDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(nonexistent=True)

    def test_missing_columns(self):
        """Test error on missing columns."""
        data = generate_test_data()

        est = ImputationDiD()
        with pytest.raises(ValueError, match="Missing columns"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="nonexistent",
            )

    def test_significance_properties(self):
        """Test is_significant and significance_stars properties."""
        data = generate_test_data(treatment_effect=5.0)
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.is_significant
        assert results.significance_stars in ("***", "**", "*", ".")

    def test_repr(self):
        """Test string representation."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        r = repr(results)
        assert "ImputationDiDResults" in r
        assert "ATT=" in r

    def test_convenience_function(self):
        """KEEP (2(d) PR-A, M-070): the deprecated wrapper still works, and
        warns - here BOTH warnings fire (wrapper + forwarded aggregate=)."""
        data = generate_test_data()
        with pytest.warns(FutureWarning, match=r"imputation_did\(\) is deprecated"):
            results = imputation_did(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                aggregate="event_study",
            )

        assert isinstance(results, ImputationDiDResults)
        assert results.event_study_effects is not None

    def test_convenience_function_kwargs(self):
        """KEEP (M-070): wrapper ctor-kwarg forwarding."""
        data = generate_test_data()
        with pytest.warns(FutureWarning, match=r"imputation_did\(\) is deprecated"):
            results = imputation_did(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                alpha=0.10,
            )

        assert results.alpha == 0.10

    def test_unbalanced_panel(self):
        """Test with unbalanced panel (some units missing periods)."""
        data = generate_test_data(seed=99)
        rng = np.random.default_rng(99)

        # Drop some observations randomly
        keep = rng.random(len(data)) > 0.1
        data_unbal = data[keep].reset_index(drop=True)

        est = ImputationDiD()
        results = est.fit(
            data_unbal,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_balance_e_checks_pre_treatment_periods(self):
        """balance_e should drop cohorts missing pre-treatment observations."""
        # Cohort A (first_treat=4): units 0-4, all periods 0-7
        #   rel_times: -4, -3, -2, -1, 0, 1, 2, 3
        # Cohort B (first_treat=6): units 5-9, all periods 0-7 EXCEPT time=4
        #   rel_times: -6, -5, -4, -3, -1, 0, 1  (missing -2)
        # Never-treated: units 10-14, all periods
        #
        # horizon_max=1 caps post-treatment to {0,1} so both cohorts can
        # cover the required post-treatment range. Without it, the union of
        # all_horizons includes h=2,3 which cohort B can't reach (max h=1).
        rows = []
        rng = np.random.default_rng(123)

        # Cohort A: complete panel
        for u in range(5):
            for t in range(8):
                y = u * 0.5 + t * 0.1 + (3.0 if t >= 4 else 0.0)
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "first_treat": 4,
                        "outcome": y + rng.normal(0, 0.01),
                    }
                )

        # Cohort B: missing time=4 (which is rel_time = 4 - 6 = -2)
        for u in range(5, 10):
            for t in range(8):
                if t == 4:
                    continue  # drop => missing rel_time=-2
                y = u * 0.5 + t * 0.1 + (3.0 if t >= 6 else 0.0)
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "first_treat": 6,
                        "outcome": y + rng.normal(0, 0.01),
                    }
                )

        # Never-treated
        for u in range(10, 15):
            for t in range(8):
                y = u * 0.5 + t * 0.1
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "first_treat": 0,
                        "outcome": y + rng.normal(0, 0.01),
                    }
                )

        data = pd.DataFrame(rows)
        est = ImputationDiD(horizon_max=1)

        # balance_e=2, horizon_max=1: required = {-2,-1,0,1}
        # Cohort B missing -2 => should be dropped
        results_bal2 = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            balance_e=2,
        )

        # balance_e=1, horizon_max=1: required = {-1,0,1}
        # Both cohorts have -1 => both kept
        results_bal1 = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            balance_e=1,
        )

        # Cohort B dropped at balance_e=2 => fewer obs at horizon 0
        n_obs_bal2_h0 = results_bal2.event_study_effects[0]["n_obs"]
        n_obs_bal1_h0 = results_bal1.event_study_effects[0]["n_obs"]
        assert n_obs_bal2_h0 < n_obs_bal1_h0, (
            f"balance_e=2 should drop cohort B (missing rel_time=-2), "
            f"got n_obs={n_obs_bal2_h0} vs {n_obs_bal1_h0}"
        )


# =============================================================================
# TestImputationDiDResults
# =============================================================================


class TestImputationDiDResults:
    """Tests for ImputationDiDResults."""

    def test_pretrend_test(self):
        """Test pre-trend test on data with parallel trends."""
        data = generate_test_data(dynamic_effects=False, seed=77, n_units=200)
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        pt = results.pretrend_test()
        assert "f_stat" in pt
        assert "p_value" in pt
        assert "n_leads" in pt
        assert pt["n_leads"] > 0

        # Under parallel trends, should not reject
        assert pt["p_value"] > 0.01

    def test_pretrend_with_violation(self):
        """Test pre-trend test detects trend violation."""
        data = generate_test_data(seed=88, n_units=200)

        # Add a pre-treatment trend for treated units
        for idx in data.index:
            if data.loc[idx, "first_treat"] > 0:
                t = data.loc[idx, "time"]
                ft = data.loc[idx, "first_treat"]
                if t < ft:
                    data.loc[idx, "outcome"] += 0.5 * (t - ft)

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        pt = results.pretrend_test()
        # With pre-trend violation, should reject (low p-value)
        assert pt["p_value"] < 0.10

    def test_pretrend_unbalanced_panel(self):
        """Test pretrend_test uses iterative demeaning for unbalanced panels."""
        data = generate_test_data(dynamic_effects=False, seed=77, n_units=200)
        # Make unbalanced by dropping ~15% of observations
        rng = np.random.default_rng(77)
        keep = rng.random(len(data)) > 0.15
        data_unbal = data[keep].reset_index(drop=True)

        est = ImputationDiD()
        results = est.fit(
            data_unbal,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        pt = results.pretrend_test()
        assert pt["n_leads"] > 0
        # Under parallel trends, should not reject
        assert pt["p_value"] > 0.01

    def test_pretrend_n_leads(self):
        """Test pre-trend test with specified number of leads."""
        data = generate_test_data(n_units=200, seed=55)
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        pt = results.pretrend_test(n_leads=2)
        assert pt["n_leads"] == 2


# =============================================================================
# TestImputationVariance
# =============================================================================


class TestImputationVariance:
    """Tests for conservative variance estimation (Theorem 3)."""

    def test_se_positive(self):
        """Test that SE is positive."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_se > 0

    def test_se_positive_event_study(self):
        """Test that event study SEs are positive."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        for h, eff in results.event_study_effects.items():
            if eff.get("n_obs", 0) > 0 and np.isfinite(eff["se"]):
                assert eff["se"] > 0

    def test_aux_partition_cohort_horizon(self):
        """Test cohort_horizon partition produces valid SEs."""
        data = generate_test_data()
        est = ImputationDiD(aux_partition="cohort_horizon")
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.overall_se > 0

    def test_aux_partition_cohort(self):
        """Test cohort partition produces valid SEs."""
        data = generate_test_data()
        est = ImputationDiD(aux_partition="cohort")
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.overall_se > 0

    def test_aux_partition_horizon(self):
        """Test horizon partition produces valid SEs."""
        data = generate_test_data()
        est = ImputationDiD(aux_partition="horizon")
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.overall_se > 0

    def test_cohort_partition_coincides_on_balanced_uniform_panel(self):
        """On a BALANCED panel with uniform weights the cohort partition is an
        arithmetic identity with the default cohort_horizon: only v != 0 rows
        contribute to a group's Eq. 8 aggregate, and the uniform-weight overall
        makes the cohort mean equal the mean of cell means. Coarser is therefore
        "typically, not guaranteed" more conservative - see REGISTRY
        ## ImputationDiD, Note (deviation from R). The identity is the strongest
        pin on this DGP (the old ordering assertion was a one-sided band around
        this exact equality); genuine divergence is asserted by the companion
        test below on an unbalanced subsample.
        """
        data = generate_test_data(n_units=200, seed=42)

        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        results_fine = ImputationDiD(aux_partition="cohort_horizon").fit(data, **kwargs)
        results_coarse = ImputationDiD(aux_partition="cohort").fit(data, **kwargs)

        # rtol=0: numpy's default rtol=1e-7 would mask the identity pin.
        np.testing.assert_allclose(
            results_coarse.overall_se, results_fine.overall_se, rtol=0, atol=1e-12
        )

    def test_coarser_partition_diverges_on_unbalanced_panel(self):
        """Companion to the identity test: on an unbalanced subsample the coarse
        partitions genuinely diverge from the default (a unit contributes several
        observations to a group with non-uniform effective weighting). Margins are
        measurement-derived on this DGP: cohort/fine = 1.0159, horizon/fine =
        1.0051 (non-LOO; under leave_one_out=True the horizon ratio inverts to
        0.9986 - the "coarser => more conservative" heuristic is typical, not
        guaranteed). No golden dependency: this is the divergence coverage that
        runs even where benchmarks/data/ is absent.
        """
        data = generate_test_data(n_units=200, seed=42)
        # Drop the last 3 periods (t >= 7 on the t=0..9 panel) for every 4th unit;
        # a 2-period drop does not clear the 1.01 margin (measured 1.0092).
        sub = data.loc[(data["unit"] % 4 != 0) | (data["time"] < 7)]
        assert len(sub) < len(data)

        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        se_fine = ImputationDiD(aux_partition="cohort_horizon").fit(sub, **kwargs).overall_se
        se_cohort = ImputationDiD(aux_partition="cohort").fit(sub, **kwargs).overall_se
        se_horizon = ImputationDiD(aux_partition="horizon").fit(sub, **kwargs).overall_se

        assert se_cohort > se_fine * 1.01
        assert se_horizon > se_fine * 1.002

    def test_invalid_aux_partition(self):
        """Test that invalid aux_partition raises ValueError."""
        with pytest.raises(ValueError, match="aux_partition"):
            ImputationDiD(aux_partition="invalid")

    def test_sparse_solver_matches_dense(self):
        """Test that sparse solver produces finite SEs with covariates."""
        data = generate_test_data(n_units=100, n_periods=10, seed=42)
        rng = np.random.default_rng(42)
        data["x1"] = rng.standard_normal(len(data))

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1"],
        )

        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0

    def test_sparse_solver_lsmr_fallback(self):
        """Test that the LSMR fallback produces finite SE when the sparse
        factorization fails."""
        import unittest.mock

        data = generate_test_data(n_units=80, n_periods=8, seed=42)
        rng = np.random.default_rng(42)
        data["x1"] = rng.standard_normal(len(data))

        est = ImputationDiD()

        # Monkey-patch the sparse factorization to force the LSMR fallback.
        with unittest.mock.patch(
            "diff_diff.imputation_aggregation.sparse_factorized",
            side_effect=RuntimeError("test failure"),
        ):
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1"],
            )

        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0

    def test_sparse_solver_lsmr_fallback_emits_warning(self):
        """Silent-failure audit axis C: the sparse-factorization -> LSMR
        fallback must emit a UserWarning so callers are informed that variance
        estimates come from the degraded path."""
        import unittest.mock

        data = generate_test_data(n_units=80, n_periods=8, seed=42)
        rng = np.random.default_rng(42)
        data["x1"] = rng.standard_normal(len(data))

        est = ImputationDiD()

        with unittest.mock.patch(
            "diff_diff.imputation_aggregation.sparse_factorized",
            side_effect=RuntimeError("test failure"),
        ):
            with pytest.warns(
                UserWarning, match="sparse factorization.*falling back to a sparse LSMR"
            ):
                est.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    covariates=["x1"],
                )


# =============================================================================
# TestImputationBootstrap
# =============================================================================


class TestImputationBootstrap:
    """Tests for bootstrap inference."""

    def test_basic_bootstrap(self, ci_params):
        """Test basic bootstrap inference."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.bootstrap_results is not None
        assert isinstance(results.bootstrap_results, ImputationBootstrapResults)
        assert results.bootstrap_results.n_bootstrap == n_boot
        assert results.bootstrap_results.overall_att_se > 0

    def test_bootstrap_reproducibility(self, ci_params):
        """Test that same seed gives same results."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)

        est1 = ImputationDiD(n_bootstrap=n_boot, seed=42)
        r1 = est1.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        est2 = ImputationDiD(n_bootstrap=n_boot, seed=42)
        r2 = est2.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        assert r1.overall_se == r2.overall_se

    def test_bootstrap_different_seeds(self, ci_params):
        """Test that different seeds give different results."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)

        est1 = ImputationDiD(n_bootstrap=n_boot, seed=42)
        r1 = est1.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        est2 = ImputationDiD(n_bootstrap=n_boot, seed=99)
        r2 = est2.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        # Results should differ (at least slightly)
        assert r1.overall_se != r2.overall_se

    def test_bootstrap_event_study(self, ci_params):
        """Test bootstrap with event study aggregation."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        br = results.bootstrap_results
        assert br.event_study_ses is not None
        assert len(br.event_study_ses) > 0

    def test_bootstrap_group(self, ci_params):
        """Test bootstrap with group aggregation."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        br = results.bootstrap_results
        assert br.group_ses is not None
        assert len(br.group_ses) == 3

    def test_bootstrap_balance_e_consistency(self, ci_params):
        """Test bootstrap event study respects balance_e filtering."""
        data = generate_test_data(n_units=150, seed=42)
        n_boot = ci_params.bootstrap(50)

        # Run WITH balance_e
        est_bal = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results_bal = est_bal.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            balance_e=2,
        )

        # Run WITHOUT balance_e
        est_nobal = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results_nobal = est_nobal.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results_bal.bootstrap_results is not None
        assert results_bal.bootstrap_results.event_study_ses is not None

        # Verify SEs are finite
        for h in results_bal.event_study_effects:
            eff = results_bal.event_study_effects[h]
            if eff.get("n_obs", 0) > 0 and np.isfinite(eff["effect"]):
                if h in results_bal.bootstrap_results.event_study_ses:
                    assert np.isfinite(results_bal.bootstrap_results.event_study_ses[h])

        # Verify balance_e changed bootstrap SEs at some horizon
        if results_nobal.bootstrap_results is not None:
            bal_ses = results_bal.bootstrap_results.event_study_ses
            nobal_ses = results_nobal.bootstrap_results.event_study_ses
            shared_h = set(bal_ses.keys()) & set(nobal_ses.keys())
            any_different = any(
                not np.isclose(bal_ses[h], nobal_ses[h], rtol=0.05)
                for h in shared_h
                if np.isfinite(bal_ses[h]) and np.isfinite(nobal_ses[h])
            )
            assert any_different, "balance_e should change bootstrap SEs for at least one horizon"

    def test_bootstrap_p_value_significance(self, ci_params):
        """Test bootstrap p-value for significant effect."""
        data = generate_test_data(treatment_effect=5.0, n_units=200)
        n_boot = ci_params.bootstrap(199, min_n=99)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Strong effect should be significant
        assert results.overall_p_value < 0.05

    def test_bootstrap_zero_noise_near_zero_se(self, ci_params):
        """Bootstrap SE ~ 0 when influence function is zero (constant effect, no noise)."""
        n_units, n_periods = 40, 8
        true_effect = 3.0
        rows = []
        for i in range(n_units):
            ft = 4 if i < 20 else 0
            unit_fe = i * 0.5
            for t in range(n_periods):
                y = unit_fe + t * 0.1  # exact FE, no noise
                if ft > 0 and t >= ft:
                    y += true_effect
                rows.append({"unit": i, "time": t, "outcome": y, "first_treat": ft})
        data = pd.DataFrame(rows)

        n_boot = ci_params.bootstrap(99)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert abs(results.overall_att - true_effect) < 1e-8
        assert results.bootstrap_results is not None
        # With zero noise, influence function sums are ~0, so SE should be ~0
        assert results.bootstrap_results.overall_att_se < 0.01

    def test_bootstrap_percentile_ci(self, ci_params):
        """Test that bootstrap CIs use percentile method, not normal approx."""
        data = generate_test_data(dynamic_effects=False, seed=42)
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        br = results.bootstrap_results
        assert br is not None

        # Verify CIs match percentile of bootstrap distribution
        dist = br.bootstrap_distribution
        expected_lower = float(np.percentile(dist, 2.5))
        expected_upper = float(np.percentile(dist, 97.5))
        np.testing.assert_allclose(br.overall_att_ci[0], expected_lower, rtol=1e-10)
        np.testing.assert_allclose(br.overall_att_ci[1], expected_upper, rtol=1e-10)

    def test_bootstrap_weights_mammen(self, ci_params):
        """Bootstrap with mammen weights should produce valid results."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, bootstrap_weights="mammen", seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        br = results.bootstrap_results
        assert br is not None
        assert br.weight_type == "mammen"
        assert br.overall_att_se > 0
        assert np.isfinite(br.overall_att_p_value)

    def test_bootstrap_weights_webb(self, ci_params):
        """Bootstrap with webb weights should produce valid results."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, bootstrap_weights="webb", seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        br = results.bootstrap_results
        assert br is not None
        assert br.weight_type == "webb"
        assert br.overall_att_se > 0
        assert np.isfinite(br.overall_att_p_value)

    def test_bootstrap_weights_event_study(self, ci_params):
        """Bootstrap with non-default weights should work for event study aggregation."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, bootstrap_weights="mammen", seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        br = results.bootstrap_results
        assert br is not None
        assert br.weight_type == "mammen"
        assert br.event_study_ses is not None
        assert len(br.event_study_ses) > 0
        for h, se in br.event_study_ses.items():
            assert se > 0, f"Non-positive SE at horizon {h}"

    def test_bootstrap_weights_group(self, ci_params):
        """Bootstrap with non-default weights should work for group aggregation."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, bootstrap_weights="mammen", seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        br = results.bootstrap_results
        assert br is not None
        assert br.weight_type == "mammen"
        assert br.group_ses is not None
        assert len(br.group_ses) > 0
        for g, se in br.group_ses.items():
            assert se > 0, f"Non-positive SE for group {g}"

    def test_bootstrap_with_covariates(self, ci_params):
        """Bootstrap should work with covariates."""
        data = generate_test_data()
        # Add a covariate
        rng = np.random.default_rng(123)
        data["x1"] = rng.normal(0, 1, len(data))
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1"],
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.overall_att_se > 0
        assert np.isfinite(results.bootstrap_results.overall_att_p_value)


# =============================================================================
# TestImputationVsOtherEstimators
# =============================================================================


class TestImputationVsOtherEstimators:
    """Cross-validation with CallawaySantAnna and SunAbraham."""

    def test_similar_point_estimates_vs_cs(self):
        """Test that point estimates are similar to CallawaySantAnna."""
        from diff_diff import CallawaySantAnna

        data = generate_test_data(n_units=200, treatment_effect=2.0, seed=42, dynamic_effects=False)

        imp_est = ImputationDiD()
        imp_results = imp_est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        cs = CallawaySantAnna()
        cs_results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Point estimates should be reasonably close
        cs_att = cs_results.overall_att
        imp_att = imp_results.overall_att
        assert abs(imp_att - cs_att) < 1.0

    def test_similar_point_estimates_vs_sa(self):
        """Test that point estimates are similar to SunAbraham."""
        from diff_diff import SunAbraham

        data = generate_test_data(n_units=200, treatment_effect=2.0, seed=42, dynamic_effects=False)

        imp_est = ImputationDiD()
        imp_results = imp_est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        sa = SunAbraham()
        sa_results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Point estimates should be reasonably close
        assert abs(imp_results.overall_att - sa_results.overall_att) < 1.0

    def test_shorter_cis_under_homogeneous_effects(self):
        """Monte Carlo: imputation CIs shorter than CS and SA under homogeneous effects.

        Validates carousel claims:
        - ~50% shorter CIs than Callaway-Sant'Anna
        - 2-3.5x shorter than Sun-Abraham
        Theory: Borusyak et al. (2024, Theorem 1) - semi-parametric efficiency bound.
        """
        from diff_diff import CallawaySantAnna, SunAbraham

        n_trials = 10
        imp_vs_cs_ratios = []
        imp_vs_sa_ratios = []

        for seed in range(n_trials):
            data = generate_test_data(
                n_units=200,
                treatment_effect=2.0,
                seed=seed,
                dynamic_effects=False,
            )

            imp_r = ImputationDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
            cs_r = CallawaySantAnna().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
            sa_r = SunAbraham().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

            imp_w = imp_r.overall_conf_int[1] - imp_r.overall_conf_int[0]
            cs_w = cs_r.overall_conf_int[1] - cs_r.overall_conf_int[0]
            sa_w = sa_r.overall_conf_int[1] - sa_r.overall_conf_int[0]

            imp_vs_cs_ratios.append(imp_w / cs_w)
            imp_vs_sa_ratios.append(imp_w / sa_w)

        median_vs_cs = np.median(imp_vs_cs_ratios)
        median_vs_sa = np.median(imp_vs_sa_ratios)

        # Imputation CIs should be meaningfully shorter than CS
        # Carousel claims ~50% shorter; use conservative 0.85 threshold
        assert (
            median_vs_cs < 0.85
        ), f"Imputation CIs not shorter than CS: median ratio={median_vs_cs:.3f}"

        # Imputation CIs should be meaningfully shorter than SA
        # Carousel claims 2-3.5x shorter; use conservative 0.85 threshold
        assert (
            median_vs_sa < 0.85
        ), f"Imputation CIs not shorter than SA: median ratio={median_vs_sa:.3f}"


# =============================================================================
# TestImputationEdgeCases
# =============================================================================


class TestImputationEdgeCases:
    """Tests for edge cases."""

    def test_single_cohort(self):
        """Test with a single treatment cohort."""
        rng = np.random.default_rng(42)
        n_units = 50
        n_periods = 8

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        first_treat = np.zeros(n_units, dtype=int)
        first_treat[25:] = 4  # Single cohort at period 4

        first_treat_exp = np.repeat(first_treat, n_periods)
        post = (times >= first_treat_exp) & (first_treat_exp > 0)

        outcomes = (
            np.repeat(rng.standard_normal(n_units) * 2, n_periods)
            + np.tile(np.linspace(0, 1, n_periods), n_units)
            + 2.0 * post
            + rng.standard_normal(len(units)) * 0.5
        )

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_exp,
            }
        )

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert len(results.groups) == 1
        assert results.overall_se > 0
        assert abs(results.overall_att - 2.0) < 1.0

    def test_no_never_treated(self):
        """Test with no never-treated units (Proposition 5)."""
        data = generate_test_data(never_treated_frac=0.0, seed=42)

        est = ImputationDiD()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )

        # Should still estimate
        assert results.overall_att is not None
        assert results.overall_se > 0

        # Proposition 5: long-run horizons should be NaN
        prop5_nans = [
            h
            for h, eff in results.event_study_effects.items()
            if np.isnan(eff["effect"]) and eff.get("n_obs", 0) > 0
        ]
        assert len(prop5_nans) > 0, "Should have Prop 5 NaN horizons"

        # Check all inference fields are NaN for Prop 5 horizons
        for h in prop5_nans:
            eff = results.event_study_effects[h]
            assert np.isnan(eff["se"])
            assert np.isnan(eff["t_stat"])
            assert np.isnan(eff["p_value"])
            assert np.isnan(eff["conf_int"][0])
            assert np.isnan(eff["conf_int"][1])

    def test_two_periods(self):
        """Test with just two periods (basic 2x2 DiD)."""
        rng = np.random.default_rng(42)
        n_units = 60
        n_periods = 2

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        first_treat = np.zeros(n_units, dtype=int)
        first_treat[30:] = 1  # Treated in period 1

        first_treat_exp = np.repeat(first_treat, n_periods)
        post = (times >= first_treat_exp) & (first_treat_exp > 0)

        outcomes = (
            np.repeat(rng.standard_normal(n_units) * 2, n_periods)
            + 3.0 * post
            + rng.standard_normal(len(units)) * 0.5
        )

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_exp,
            }
        )

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert abs(results.overall_att - 3.0) < 1.0

    def test_rank_deficiency_warn(self):
        """Test rank_deficient_action='warn' doesn't error."""
        data = generate_test_data()
        est = ImputationDiD(rank_deficient_action="warn")
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.overall_se > 0

    def test_rank_deficiency_error(self):
        """Test rank_deficient_action='error' works."""
        est = ImputationDiD(rank_deficient_action="error")
        # Should work fine on good data
        data = generate_test_data()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.overall_se > 0

    def test_invalid_rank_deficient_action(self):
        """Test invalid rank_deficient_action raises ValueError."""
        with pytest.raises(ValueError, match="rank_deficient_action"):
            ImputationDiD(rank_deficient_action="ignore")

    def test_always_treated_warning(self):
        """Test warning for units treated in all periods."""
        rng = np.random.default_rng(42)
        n_units = 40
        n_periods = 6

        units = np.repeat(np.arange(n_units), n_periods)

        # To trigger the always-treated check we need first_treat > 0 but
        # <= min(time). first_treat == 0 means never-treated in the code, so
        # with times starting at 0 that is impossible — start times at 1.
        times_shifted = np.tile(np.arange(1, n_periods + 1), n_units)

        first_treat_3 = np.zeros(n_units, dtype=int)
        first_treat_3[:10] = 0  # never treated
        first_treat_3[10:15] = 1  # treated from the very beginning (always treated)
        first_treat_3[15:] = 4

        first_treat_exp_3 = np.repeat(first_treat_3, n_periods)
        post_3 = (times_shifted >= first_treat_exp_3) & (first_treat_exp_3 > 0)

        outcomes_3 = (
            np.repeat(rng.standard_normal(n_units) * 2, n_periods)
            + 2.0 * post_3
            + rng.standard_normal(len(units)) * 0.5
        )

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times_shifted,
                "outcome": outcomes_3,
                "first_treat": first_treat_exp_3,
            }
        )

        est = ImputationDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

        # Should have issued a warning about always-treated
        always_treated_warnings = [
            x for x in w if "treated in all observed periods" in str(x.message)
        ]
        assert len(always_treated_warnings) > 0

    def test_no_treated_units(self):
        """Test error when no treated units."""
        data = generate_test_data()
        data["first_treat"] = 0  # All never-treated

        est = ImputationDiD()
        with pytest.raises(ValueError, match="No treated"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_nan_propagation_all_nan_horizon(self):
        """Test NaN propagation when all tau_hat at a horizon are NaN."""
        data = generate_test_data(never_treated_frac=0.0, seed=42)

        est = ImputationDiD()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )

        # Check that NaN horizons have all-NaN inference
        for h, eff in results.event_study_effects.items():
            if eff.get("n_obs", 0) > 0 and np.isnan(eff["effect"]):
                assert np.isnan(eff["se"])
                assert np.isnan(eff["t_stat"])
                assert np.isnan(eff["p_value"])
                assert np.isnan(eff["conf_int"][0])
                assert np.isnan(eff["conf_int"][1])

    def test_summary_not_fitted(self):
        """Test error when calling summary before fit."""
        est = ImputationDiD()
        with pytest.raises(RuntimeError, match="must be fitted"):
            est.summary()

    def test_rank_condition_missing_untreated_period(self):
        """Test warning when a post-treatment period has no untreated units."""
        # Construct data where ALL units are treated from period 2 onward,
        # so periods 2+ have no untreated observations
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 5
        rows = []
        for i in range(n_units):
            ft = 2  # all units treated at period 2
            for t in range(n_periods):
                y = rng.standard_normal() + i * 0.1 + t * 0.05
                if t >= ft:
                    y += 1.0  # treatment effect
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "first_treat": ft,
                    }
                )
        data = pd.DataFrame(rows)

        est = ImputationDiD(rank_deficient_action="warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )
            rank_warnings = [x for x in w if "Rank condition" in str(x.message)]
            assert len(rank_warnings) > 0, "Should warn about rank condition violation"

        # Affected horizons should have NaN effects (periods with no untreated units)
        if results.event_study_effects:
            nan_effects = [
                h
                for h, d in results.event_study_effects.items()
                if np.isnan(d["effect"]) and d.get("n_obs", 1) > 0
            ]
            assert len(nan_effects) > 0, "Some horizons should have NaN effects"

    def test_rank_condition_error_mode(self):
        """Test error raised when rank condition fails with action='error'."""
        # Same setup as test_rank_condition_missing_untreated_period
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 5
        rows = []
        for i in range(n_units):
            ft = 2
            for t in range(n_periods):
                y = rng.standard_normal() + i * 0.1 + t * 0.05
                if t >= ft:
                    y += 1.0
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "first_treat": ft,
                    }
                )
        data = pd.DataFrame(rows)

        est = ImputationDiD(rank_deficient_action="error")
        with pytest.raises(ValueError, match="Rank condition"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_bootstrap_cluster_not_unit(self, ci_params):
        """Test bootstrap uses cluster column when cluster != unit."""
        data = generate_test_data(n_units=100, n_periods=8, seed=42)
        # Create cluster column grouping every 5 units
        unit_to_cluster = {u: u // 5 for u in data["unit"].unique()}
        data["cluster_id"] = data["unit"].map(unit_to_cluster)

        n_boot = ci_params.bootstrap(99, min_n=49)
        est = ImputationDiD(cluster="cluster_id", n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.bootstrap_results is not None
        assert results.bootstrap_results.overall_att_se > 0

        # Bootstrap SE with cluster should differ from unit-level bootstrap
        est_unit = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results_unit = est_unit.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert (
            results.bootstrap_results.overall_att_se
            != results_unit.bootstrap_results.overall_att_se
        )

    def test_bootstrap_invalid_cluster_column(self):
        """Test error when cluster column doesn't exist."""
        data = generate_test_data(n_units=50, seed=42)
        est = ImputationDiD(cluster="nonexistent_col")
        with pytest.raises(ValueError, match="not found"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_plot_reference_with_anticipation(self):
        """Test event study plot detects reference period with anticipation."""
        data = generate_test_data(n_units=100, n_periods=10, seed=42)
        est = ImputationDiD(anticipation=1)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        # Reference should be at -2 (= -1 - anticipation)
        assert -2 in results.event_study_effects
        assert results.event_study_effects[-2]["n_obs"] == 0  # reference marker

        # Test that plot_event_study auto-detects it
        pytest.importorskip("matplotlib")
        from diff_diff import plot_event_study

        fig = plot_event_study(results)
        assert fig is not None

    def test_overall_se_with_partial_nan_tau_hat(self):
        """Test overall SE uses finite-only weights when some tau_hat are NaN."""
        # Create staggered data: cohort A treated at t=2, cohort B never-treated
        # but drop all never-treated obs at t=5, so t=5 time FE is unidentified
        # -> tau_hat for (cohort A, t=5) will be NaN
        rng = np.random.default_rng(42)
        n_units, n_periods = 40, 6
        rows = []
        for i in range(n_units):
            if i < 20:
                ft = 2  # early-treated
            else:
                ft = 99  # never-treated
            for t in range(n_periods):
                # Drop never-treated at t=5 to create unidentified time FE
                if ft == 99 and t == 5:
                    continue
                y = rng.standard_normal() + i * 0.1 + t * 0.05
                if t >= ft:
                    y += 1.0
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "first_treat": ft,
                    }
                )
        data = pd.DataFrame(rows)

        est = ImputationDiD(rank_deficient_action="silent")
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        tau_hat = results.treatment_effects["tau_hat"]
        n_nan = tau_hat.isna().sum()
        n_finite = tau_hat.notna().sum()

        # Verify the scenario actually produces partial NaN
        assert n_nan > 0, "Expected some NaN tau_hat (missing time FE at t=5)"
        assert n_finite > 0, "Expected some finite tau_hat"

        # Partial NaN case: SE should be finite (computed from finite-only weights)
        assert np.isfinite(
            results.overall_se
        ), f"overall_se should be finite with {n_finite} finite and {n_nan} NaN tau_hat"
        assert np.isfinite(results.overall_att)

    def test_covariate_delta_matches_full_dummy_ols(self):
        """Step-A/B covariate coefficients match explicit unit+time dummy OLS.

        Estimator-level lock on the shared-engine within-transform path
        (replaces the direct `_iterative_demean` balanced-one-pass test; the
        private per-estimator demean loops were consolidated into
        `diff_diff.utils.demean_by_groups`).
        """
        rng = np.random.default_rng(42)
        n_units, n_periods = 15, 6
        rows = []
        for i in range(n_units):
            for t in range(n_periods):
                if rng.random() < 0.2:  # unbalanced
                    continue
                rows.append({"unit": i, "time": t})
        df = pd.DataFrame(rows)
        n = len(df)
        x = rng.standard_normal((n, 2))
        df["x1"], df["x2"] = x[:, 0], x[:, 1]
        u_fe = rng.standard_normal(n_units)
        t_fe = np.linspace(0, 1, n_periods)
        df["outcome"] = (
            u_fe[df["unit"]]
            + t_fe[df["time"]]
            + x @ np.array([0.7, -0.4])
            + rng.standard_normal(n) * 0.1
        )
        df["first_treat"] = 0  # all never-treated -> omega_0 = everything

        est = ImputationDiD()
        omega_0 = pd.Series(True, index=df.index)
        _, _, _, delta_hat, _ = est._fit_untreated_model(
            df, "outcome", "unit", "time", ["x1", "x2"], omega_0
        )

        # Explicit full-dummy OLS: [all unit dummies, time dummies (drop 1), X]
        u_d = pd.get_dummies(df["unit"]).values.astype(float)
        t_d = pd.get_dummies(df["time"]).values.astype(float)[:, 1:]
        X_full = np.column_stack([u_d, t_d, x])
        coef = np.linalg.lstsq(X_full, df["outcome"].values, rcond=None)[0]
        np.testing.assert_allclose(delta_hat, coef[-2:], atol=1e-8)

    def test_unbalanced_panel_fe_correctness(self):
        """Test FE estimates match OLS for unbalanced panel."""
        # Create small unbalanced panel with known FE structure
        rng = np.random.default_rng(42)
        n_units, n_periods = 8, 5
        unit_fe_true = rng.standard_normal(n_units) * 2.0
        time_fe_true = np.linspace(0, 1, n_periods)

        rows = []
        for i in range(n_units):
            for t in range(n_periods):
                # Drop ~20% of obs to make unbalanced
                if rng.random() < 0.2:
                    continue
                y = unit_fe_true[i] + time_fe_true[t] + rng.standard_normal() * 0.01
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "first_treat": n_periods,  # all never-treated -> Omega_0
                    }
                )

        df_0 = pd.DataFrame(rows)

        # Compute FE via iterative method (what we're testing)
        est = ImputationDiD()
        unit_fe_iter, time_fe_iter = est._iterative_fe(
            df_0["outcome"].values,
            df_0["unit"].values,
            df_0["time"].values,
            df_0.index,
        )

        # Compute exact OLS FE via lstsq with dummy variables
        unique_units = sorted(df_0["unit"].unique())
        unique_times = sorted(df_0["time"].unique())
        n = len(df_0)
        n_u = len(unique_units)
        n_t = len(unique_times)
        u_map = {u: i for i, u in enumerate(unique_units)}
        t_map = {t: i for i, t in enumerate(unique_times)}

        X = np.zeros((n, 1 + (n_u - 1) + (n_t - 1)))
        X[:, 0] = 1.0  # intercept
        for j in range(n):
            uid = u_map[df_0["unit"].iloc[j]]
            tid = t_map[df_0["time"].iloc[j]]
            if uid > 0:
                X[j, uid] = 1.0
            if tid > 0:
                X[j, n_u + tid - 1] = 1.0

        beta_ols = np.linalg.lstsq(X, df_0["outcome"].values, rcond=None)[0]

        # Reconstruct OLS fitted values
        intercept = beta_ols[0]
        unit_fe_ols = {unique_units[0]: intercept}
        for i in range(1, n_u):
            unit_fe_ols[unique_units[i]] = intercept + beta_ols[i]
        time_fe_ols = {unique_times[0]: 0.0}
        for i in range(1, n_t):
            time_fe_ols[unique_times[i]] = beta_ols[n_u + i - 1]

        # Compare fitted values (parameterization-invariant check)
        for j in range(n):
            u = df_0["unit"].iloc[j]
            t = df_0["time"].iloc[j]
            y_hat_iter = unit_fe_iter[u] + time_fe_iter[t]
            y_hat_ols = unit_fe_ols[u] + time_fe_ols[t]
            assert abs(y_hat_iter - y_hat_ols) < 1e-6, (
                f"Fitted values differ at unit={u}, time={t}: "
                f"iterative={y_hat_iter:.8f} vs OLS={y_hat_ols:.8f}"
            )

    def test_non_constant_first_treat_warning(self):
        """Warn when first_treat varies within a unit (violates absorbing treatment)."""
        data = generate_test_data(dynamic_effects=False, seed=42)
        # Corrupt first_treat for unit 0: make it vary across rows
        bad_unit = data["unit"].unique()[0]
        mask = data["unit"] == bad_unit
        data.loc[mask & (data["time"] >= 5), "first_treat"] = 99

        est = ImputationDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
            absorbing_warnings = [x for x in w if "non-constant" in str(x.message)]
            assert len(absorbing_warnings) >= 1, "Expected warning about non-constant first_treat"
            # Verify warning mentions the unit count and example
            msg = str(absorbing_warnings[0].message)
            assert "1 unit(s)" in msg
            assert str(bad_unit) in msg

        # Behavioral assertion: estimator still produces results (warns, doesn't crash)
        assert isinstance(results, ImputationDiDResults)
        assert np.isfinite(results.overall_att)

        # Behavioral assertion: coercion applied — first_treat is now constant per unit
        fit_df = est._fit_data["df"]
        bad_rows = fit_df[fit_df["unit"] == bad_unit]
        ft_vals = bad_rows["first_treat"].unique()
        assert (
            len(ft_vals) == 1
        ), f"first_treat should be coerced to single value per unit, got {ft_vals}"

    def test_treatment_effects_weight_nan_consistency(self):
        """Test that treatment_effects weights are 0 for NaN tau_hat and 1/n_valid for finite."""
        # Reuse the partial-NaN scenario from test_overall_se_with_partial_nan_tau_hat
        rng = np.random.default_rng(42)
        n_units, n_periods = 40, 6
        rows = []
        for i in range(n_units):
            if i < 20:
                ft = 2  # early-treated
            else:
                ft = 99  # never-treated
            for t in range(n_periods):
                # Drop never-treated at t=5 to create unidentified time FE
                if ft == 99 and t == 5:
                    continue
                y = rng.standard_normal() + i * 0.1 + t * 0.05
                if t >= ft:
                    y += 1.0
                rows.append({"unit": i, "time": t, "outcome": y, "first_treat": ft})
        data = pd.DataFrame(rows)

        est = ImputationDiD(rank_deficient_action="silent")
        results = est.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        te = results.treatment_effects
        nan_rows = te[te["tau_hat"].isna()]
        finite_rows = te[te["tau_hat"].notna()]

        # Verify scenario produces partial NaN
        assert len(nan_rows) > 0
        assert len(finite_rows) > 0

        # NaN tau_hat rows have weight 0
        assert (nan_rows["weight"] == 0.0).all(), "NaN tau_hat rows should have weight 0"

        # Finite weights sum to ~1.0
        assert abs(finite_rows["weight"].sum() - 1.0) < 1e-10, "Finite weights should sum to 1"

        # Each finite weight equals 1/n_finite
        n_finite = len(finite_rows)
        expected_weight = 1.0 / n_finite
        np.testing.assert_allclose(finite_rows["weight"].values, expected_weight, rtol=1e-10)

    def test_rank_deficient_covariates_excluded_from_variance(self):
        """Rank-deficient covariates are excluded from variance design matrices."""
        data = generate_test_data(n_units=80, n_periods=8, seed=42)
        rng = np.random.default_rng(42)
        data["x1"] = rng.standard_normal(len(data))
        data["x2"] = 2.0 * data["x1"]  # perfectly collinear

        est = ImputationDiD(rank_deficient_action="silent")
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
        )

        # SE should be finite (not blown up by singular design matrix)
        assert np.isfinite(results.overall_se), "SE should be finite with rank-deficient covariates"
        assert results.overall_se > 0

        # Verify kept_cov_mask is stored and has one True + one False
        mask = est._fit_data["kept_cov_mask"]
        assert mask is not None
        assert mask.sum() == 1, f"Expected 1 kept covariate, got {mask.sum()}"
        assert len(mask) == 2
        assert (~mask).sum() == 1, "Expected 1 dropped covariate"

    def test_bootstrap_psi_precomputation_failure_warning(self, ci_params):
        """Warning emitted and bootstrap skipped when psi precomputation fails."""
        data = generate_test_data(dynamic_effects=False, seed=42)
        n_boot = ci_params.bootstrap(99)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)

        # Monkey-patch to force failure
        def failing_precompute(*args, **kwargs):
            raise RuntimeError("test failure")

        est._precompute_bootstrap_psi = failing_precompute

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
            psi_warnings = [x for x in w if "Bootstrap pre-computation failed" in str(x.message)]
            assert len(psi_warnings) >= 1

        # Behavioral assertion: bootstrap_results is None
        assert results.bootstrap_results is None
        # Analytical SE still present
        assert results.overall_se > 0

    def test_event_study_empty_after_filtering(self):
        """Warn when balance_e/horizon_max filter out all treated horizons."""
        data = generate_test_data(dynamic_effects=False, seed=42)
        # balance_e=100 requires cohorts to span [-100, max_h+1], which none do.
        # All cohorts fail the balanced check, so all horizons have n_h=0.
        est = ImputationDiD()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
                balance_e=100,
            )
            empty_warnings = [x for x in w if "no horizons with observations" in str(x.message)]
            assert len(empty_warnings) >= 1, "Expected warning about empty event study"

        # Only reference period should remain
        ref_period = -1
        assert ref_period in results.event_study_effects
        real_effects = {
            h: v
            for h, v in results.event_study_effects.items()
            if h != ref_period and v.get("n_obs", 0) > 0
        }
        assert len(real_effects) == 0

    def test_balanced_cohort_mask_requires_negative_horizons(self):
        """_compute_balanced_cohort_mask must check negative relative times."""
        cohort_rel_times = {
            5: {-2, -1, 0, 1, 2},
            7: {-1, 0, 1, 2},  # missing -2
        }
        df_treated = pd.DataFrame({"first_treat": [5, 5, 5, 7, 7, 7]})
        all_horizons = [0, 1, 2]

        # balance_e=2 requires {-2,-1,0,1,2}: only cohort 5 passes
        mask2 = ImputationDiD._compute_balanced_cohort_mask(
            df_treated, "first_treat", all_horizons, 2, cohort_rel_times
        )
        assert mask2.tolist() == [True, True, True, False, False, False]

        # balance_e=1 requires {-1,0,1,2}: both pass
        mask1 = ImputationDiD._compute_balanced_cohort_mask(
            df_treated, "first_treat", all_horizons, 1, cohort_rel_times
        )
        assert all(mask1)

    def test_iterative_fe_warns_on_nonconvergence(self):
        """Silent-failure audit axis B: _iterative_fe must warn when max_iter exhausts."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 8, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        y = rng.standard_normal(n_units * n_periods)
        idx = pd.RangeIndex(len(y))
        est = ImputationDiD()

        with pytest.warns(UserWarning, match="did not converge"):
            est._iterative_fe(y, units, times, idx, max_iter=1, tol=1e-15)

    def test_iterative_fe_no_warning_on_convergence(self):
        """Silent-failure audit axis B: no warning on well-behaved convergent input."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 8, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        y = rng.standard_normal(n_units * n_periods)
        idx = pd.RangeIndex(len(y))
        est = ImputationDiD()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est._iterative_fe(y, units, times, idx)
        assert not any("did not converge" in str(x.message) for x in w)

    # NOTE (intentional coverage narrowing): the direct `_iterative_demean`
    # non-convergence warn tests were retired with the method itself - the
    # covariate within-transform now routes through the shared MAP engine
    # with max_iter=10_000 hardcoded at the call sites, so demean
    # non-convergence is no longer forceable THROUGH this estimator.
    # Engine-level warning coverage lives in
    # tests/test_utils.py::TestDemeanByGroups; the `_iterative_fe` warn
    # tests above still exercise this estimator's FE-solver warning path.

    def test_iterative_fe_zero_weight_unit_gets_nan_fe(self):
        """A unit whose rows ALL carry zero weight surfaces as NaN FE.

        Locks the shared-solver zero-weight contract (spillover precedent:
        never a silent finite 0.0) AND that the solver still converges
        cleanly - the historical pandas loop divided 0/0 there and burned
        max_iter iterations before warning.
        """
        rng = np.random.default_rng(42)
        n_units, n_periods = 8, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        y = rng.standard_normal(n_units * n_periods)
        w = np.ones(n_units * n_periods)
        w[units == 3] = 0.0  # zero out one whole unit
        idx = pd.RangeIndex(len(y))
        est = ImputationDiD()

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            unit_fe, time_fe = est._iterative_fe(y, units, times, idx, weights=w)
        assert not any("did not converge" in str(x.message) for x in rec)

        assert np.isnan(unit_fe[3])  # zero-weight unit: NaN, key retained
        assert all(np.isfinite(v) for u, v in unit_fe.items() if u != 3)
        assert all(np.isfinite(v) for v in time_fe.values())


# =============================================================================
# TestImputationDiDVcovType  (Phase 1b interstitial #3)
# =============================================================================


def _imputation_clustered_panel(
    seed: int = 53,
    n_units: int = 60,
    n_periods: int = 6,
    n_states: int = 12,
) -> pd.DataFrame:
    """Staggered-adoption panel with a `state` cluster column drawn from a
    finite set of states with intra-state random effects. Used for
    ``cluster=state`` bit-equality tests on the vcov_type contract.
    """
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    # Assign units to states; states carry random effects so cluster=state
    # actually shifts SE relative to cluster=None (cluster=unit default).
    unit_to_state = rng.integers(0, n_states, size=n_units)
    state = np.repeat(unit_to_state, n_periods)
    state_re = rng.standard_normal(n_states) * 1.5

    # Half never-treated, rest assigned to one of three treatment cohorts.
    cohorts = np.array([2, 3, 4])
    n_never = n_units // 2
    n_treated = n_units - n_never
    first_treat = np.zeros(n_units, dtype=int)
    first_treat[n_never:] = cohorts[rng.integers(0, len(cohorts), size=n_treated)]
    first_treat_expanded = np.repeat(first_treat, n_periods)

    unit_fe = rng.standard_normal(n_units) * 1.5
    time_fe = np.linspace(0, 0.5, n_periods)
    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)
    state_fe_expanded = state_re[state]

    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
    outcome = (
        unit_fe_expanded
        + time_fe_expanded
        + state_fe_expanded
        + 2.0 * post
        + rng.standard_normal(len(units)) * 0.5
    )

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcome,
            "first_treat": first_treat_expanded,
            "state": state,
        }
    )


def _imputation_survey_panel(
    seed: int = 71,
    n_units: int = 60,
    n_periods: int = 4,
    n_psu: int = 12,
    n_strata: int = 3,
) -> pd.DataFrame:
    """Staggered-adoption panel with analytical survey columns (pweight +
    panel-constant PSU + stratum). Used for TSL-survey bit-equality tests
    on the vcov_type contract."""
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

    unit_fe = rng.standard_normal(n_units) * 1.2
    time_fe = np.linspace(0, 0.5, n_periods)
    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
    outcome = (
        unit_fe_expanded + time_fe_expanded + 1.5 * post + rng.standard_normal(len(units)) * 0.4
    )

    # Panel-constant weights (per-unit).
    unit_weight = 1.0 + rng.exponential(0.3, n_units)
    weight = np.repeat(unit_weight, n_periods)

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcome,
            "first_treat": first_treat_expanded,
            "psu": psu,
            "stratum": stratum,
            "weight": weight,
        }
    )


def _imputation_replicate_panel(
    seed: int = 89, n_units: int = 40, n_periods: int = 4, n_rep: int = 8
):
    """Staggered-adoption panel with JK1 replicate-weight columns. Mirrors
    the pattern from ``test_triple_diff._ddd_replicate_panel`` but uses a
    panel layout suitable for ImputationDiD's fit signature."""
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
        unit_fe_expanded + time_fe_expanded + 1.2 * post + rng.standard_normal(len(units)) * 0.4
    )

    unit_weight = 1.0 + rng.exponential(0.2, n_units)
    weight = np.repeat(unit_weight, n_periods)

    data = pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcome,
            "first_treat": first_treat_expanded,
            "weight": weight,
        }
    )

    # JK1 jackknife replicates: zero out one PSU (block of units) per replicate
    # and rescale survivors. Panel-constant per unit.
    units_per_rep = max(n_units // n_rep, 1)
    rep_cols = []
    for r in range(n_rep):
        w_r = unit_weight.copy()
        start = r * units_per_rep
        end = min((r + 1) * units_per_rep, n_units)
        w_r[start:end] = 0.0
        nonzero = w_r > 0
        # JK1 scaling (n_rep / (n_rep - 1)) applied to survivors.
        w_r[nonzero] = w_r[nonzero] * n_rep / (n_rep - 1)
        col = f"rep_{r}"
        data[col] = np.repeat(w_r, n_periods)
        rep_cols.append(col)
    return data, rep_cols


class TestImputationDiDVcovType:
    """Phase 1b interstitial #3: vcov_type input contract on ImputationDiD.

    ImputationDiD uses IF-based variance per Borusyak-Jaravel-Spiess (2024)
    Theorem 3; vcov_type is permanently narrow to {"hc1"}.
    Analytical-sandwich families {classical, hc2, hc2_bm} and conley are
    rejected at __init__ with methodology-rooted messages. Mirrors CS
    PR #487 (`tests/test_staggered.py`) and TD PR #488
    (`tests/test_triple_diff.py::TestTripleDifferenceVcovType`) templates.

    7-surface matrix:
      1. Default preserved bit-equally across `aggregate ∈ {None, event_study, group}`
      2. Cluster path preserved bit-equally across the same aggregate grid
      3. TSL-survey path preserved bit-equally across the same aggregate grid
      4. Replicate-survey path preserved bit-equally (event_study only; pretrends rejection limits the grid)
      5. Bootstrap × cluster + bootstrap × survey bit-equal
      6. fit()-time revalidation after `set_params(vcov_type=bad)`
      7. Bootstrap n_psu<2 / n_clusters<2 NaN propagation (defensive fix regression)

    Plus 8 introspection + safety-gate tests, 5 input-rejection pins, the
    `cluster + replicate_weights` rejection, and a `pretrends=True` ×
    `vcov_type='hc1'` × cluster bit-equality lock.
    """

    # ---- Surface 1: default bit-equal across aggregation modes ------------

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group"])
    def test_default_hc1_bit_equal_baseline(self, aggregate):
        data = generate_test_data(seed=53, n_units=80, n_periods=8)
        common = dict(
            data=data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate=aggregate,
        )
        r_default = ImputationDiD().fit(**common)
        r_explicit = ImputationDiD(vcov_type="hc1").fit(**common)
        assert r_default.overall_att == r_explicit.overall_att
        assert r_default.overall_se == r_explicit.overall_se

    # ---- Surface 2: cluster path bit-equal --------------------------------

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group"])
    def test_cluster_hc1_bit_equal_baseline(self, aggregate):
        data = _imputation_clustered_panel()
        common = dict(
            data=data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate=aggregate,
        )
        r_default = ImputationDiD(cluster="state").fit(**common)
        r_explicit = ImputationDiD(cluster="state", vcov_type="hc1").fit(**common)
        assert r_default.overall_att == r_explicit.overall_att
        assert r_default.overall_se == r_explicit.overall_se

    # ---- Surface 3: TSL-survey path bit-equal -----------------------------

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group"])
    def test_survey_tsl_hc1_bit_equal_baseline(self, aggregate):
        data = _imputation_survey_panel()
        design = SurveyDesign(
            weights="weight",
            psu="psu",
            strata="stratum",
            weight_type="pweight",
        )
        common = dict(
            data=data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate=aggregate,
            survey_design=design,
        )
        r_default = ImputationDiD().fit(**common)
        r_explicit = ImputationDiD(vcov_type="hc1").fit(**common)
        assert r_default.overall_att == r_explicit.overall_att
        assert r_default.overall_se == r_explicit.overall_se

    # ---- Surface 4: replicate-survey path bit-equal -----------------------

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group"])
    def test_survey_replicate_hc1_bit_equal_baseline(self, aggregate):
        data, rep_cols = _imputation_replicate_panel()
        design = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
            weight_type="pweight",
        )
        common = dict(
            data=data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
            aggregate=aggregate,
        )
        r_default = ImputationDiD().fit(**common)
        r_explicit = ImputationDiD(vcov_type="hc1").fit(**common)
        assert r_default.overall_att == r_explicit.overall_att
        assert r_default.overall_se == r_explicit.overall_se
        # Per-horizon / per-group SE override branches must also agree under
        # the replicate-weight variance path.
        if aggregate == "event_study":
            assert r_default.event_study_effects is not None
            assert r_explicit.event_study_effects is not None
            for h in r_default.event_study_effects:
                assert (
                    r_default.event_study_effects[h]["se"]
                    == r_explicit.event_study_effects[h]["se"]
                )
        if aggregate == "group":
            assert r_default.group_effects is not None
            assert r_explicit.group_effects is not None
            for g in r_default.group_effects:
                assert r_default.group_effects[g]["se"] == r_explicit.group_effects[g]["se"]

    # ---- Surface 5: bootstrap × cluster / × survey bit-equal --------------

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group"])
    def test_bootstrap_cluster_hc1_bit_equal(self, ci_params, aggregate):
        data = _imputation_clustered_panel()
        n_boot = ci_params.bootstrap(199)
        common = dict(
            data=data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate=aggregate,
        )
        r_default = ImputationDiD(cluster="state", n_bootstrap=n_boot, seed=11).fit(**common)
        r_explicit = ImputationDiD(
            cluster="state", n_bootstrap=n_boot, seed=11, vcov_type="hc1"
        ).fit(**common)
        assert r_default.bootstrap_results is not None
        assert r_explicit.bootstrap_results is not None
        assert (
            r_default.bootstrap_results.overall_att_se
            == r_explicit.bootstrap_results.overall_att_se
        )
        # Per-horizon / per-group bootstrap SE override branches at
        # imputation_aggregation.py::_replicate_override_aggregates must also agree.
        if aggregate == "event_study":
            assert r_default.bootstrap_results.event_study_ses is not None
            assert r_explicit.bootstrap_results.event_study_ses is not None
            for h, se in r_default.bootstrap_results.event_study_ses.items():
                assert se == r_explicit.bootstrap_results.event_study_ses[h]
        if aggregate == "group":
            assert r_default.bootstrap_results.group_ses is not None
            assert r_explicit.bootstrap_results.group_ses is not None
            for g, se in r_default.bootstrap_results.group_ses.items():
                assert se == r_explicit.bootstrap_results.group_ses[g]

    @pytest.mark.parametrize("aggregate", [None, "event_study", "group"])
    def test_bootstrap_survey_hc1_bit_equal(self, ci_params, aggregate):
        data = _imputation_survey_panel()
        design = SurveyDesign(
            weights="weight",
            psu="psu",
            strata="stratum",
            weight_type="pweight",
        )
        n_boot = ci_params.bootstrap(199)
        common = dict(
            data=data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
            aggregate=aggregate,
        )
        r_default = ImputationDiD(n_bootstrap=n_boot, seed=23).fit(**common)
        r_explicit = ImputationDiD(n_bootstrap=n_boot, seed=23, vcov_type="hc1").fit(**common)
        assert r_default.bootstrap_results is not None
        assert r_explicit.bootstrap_results is not None
        assert (
            r_default.bootstrap_results.overall_att_se
            == r_explicit.bootstrap_results.overall_att_se
        )
        if aggregate == "event_study":
            assert r_default.bootstrap_results.event_study_ses is not None
            assert r_explicit.bootstrap_results.event_study_ses is not None
            for h, se in r_default.bootstrap_results.event_study_ses.items():
                assert se == r_explicit.bootstrap_results.event_study_ses[h]
        if aggregate == "group":
            assert r_default.bootstrap_results.group_ses is not None
            assert r_explicit.bootstrap_results.group_ses is not None
            for g, se in r_default.bootstrap_results.group_ses.items():
                assert se == r_explicit.bootstrap_results.group_ses[g]

    # ---- Surface 6: eager transactional validation (BaseEstimator) --------

    def test_set_params_bad_vcov_raises_eagerly_classical(self):
        # set_params validates via constructor probe (transactional per the
        # locked v4 rule): the bad value raises at set_params and the
        # estimator is unchanged.
        imp = ImputationDiD()
        with pytest.raises(ValueError, match="influence-function"):
            imp.set_params(vcov_type="classical")
        assert imp.vcov_type == "hc1"

    def test_set_params_bad_vcov_raises_eagerly_unknown(self):
        imp = ImputationDiD()
        with pytest.raises(ValueError, match="hc4"):
            imp.set_params(vcov_type="hc4")
        assert imp.vcov_type == "hc1"

    # ---- Surface 7: bootstrap n_psu/n_clusters<2 NaN propagation ----------

    def test_bootstrap_n_clusters_less_than_2_returns_nan(self):
        # Construct a panel where the cluster column has exactly 1 unique
        # value so the analytical-cluster bootstrap path hits the n<2 guard.
        data = generate_test_data(seed=7, n_units=40, n_periods=6)
        data["single_cluster"] = 1
        with pytest.warns(UserWarning, match="n_clusters=1"):
            results = ImputationDiD(cluster="single_cluster", n_bootstrap=199, seed=3).fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        assert results.bootstrap_results is not None
        assert np.isnan(results.bootstrap_results.overall_att_se)
        assert np.isnan(results.bootstrap_results.overall_att_p_value)
        assert all(np.isnan(x) for x in results.bootstrap_results.overall_att_ci)
        # Derived coef_var propagates NaN through the alias property.
        assert np.isnan(results.coef_var)

    def test_bootstrap_n_psu_less_than_2_returns_nan(self):
        # Construct a panel with a single PSU so the survey-PSU bootstrap
        # path hits the n_psu<2 BLAS-roundoff guard. Survey weight_type
        # must be pweight per the ImputationDiD survey contract.
        data = _imputation_survey_panel(seed=42)
        data["single_psu"] = 0
        data["single_stratum"] = 0
        design = SurveyDesign(
            weights="weight",
            psu="single_psu",
            strata="single_stratum",
            weight_type="pweight",
        )
        with pytest.warns(UserWarning, match="n_psu=1"):
            results = ImputationDiD(n_bootstrap=199, seed=5).fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
        assert results.bootstrap_results is not None
        assert np.isnan(results.bootstrap_results.overall_att_se)
        assert np.isnan(results.bootstrap_results.overall_att_p_value)
        assert all(np.isnan(x) for x in results.bootstrap_results.overall_att_ci)
        assert np.isnan(results.coef_var)

    # ---- Input rejection: methodology-rooted messages ---------------------

    @pytest.mark.parametrize(
        "bad_vcov,keyword",
        [
            ("classical", "influence-function"),
            ("hc2", "Borusyak"),
            ("hc2_bm", "Bell-McCaffrey"),
            ("hc2_bm", "hat matrix"),
        ],
    )
    def test_reject_invalid_vcov_at_init(self, bad_vcov, keyword):
        with pytest.raises(ValueError, match=keyword):
            ImputationDiD(vcov_type=bad_vcov)

    def test_reject_conley_at_init(self):
        with pytest.raises(ValueError, match="spatial-HAC"):
            ImputationDiD(vcov_type="conley")

    def test_reject_unknown_vcov_at_init(self):
        with pytest.raises(ValueError, match="hc4"):
            ImputationDiD(vcov_type="hc4")

    # ---- cluster + replicate_weights fail-closed --------------------------

    def test_cluster_plus_replicate_weights_rejected(self):
        data, rep_cols = _imputation_replicate_panel()
        # Synthesize a state column for the bare cluster= argument.
        data["state"] = (data["unit"] // 4).astype(int)
        design = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
            weight_type="pweight",
        )
        with pytest.raises(NotImplementedError, match="replicate-weight"):
            ImputationDiD(cluster="state").fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )

    # ---- pretrends × cluster × explicit hc1 bit-equality ------------------

    def test_pretrends_hc1_bit_equal_with_cluster(self):
        data = _imputation_clustered_panel()
        common = dict(
            data=data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        r_default = ImputationDiD(cluster="state", pretrends=True).fit(**common)
        r_explicit = ImputationDiD(cluster="state", pretrends=True, vcov_type="hc1").fit(**common)
        # pretrend_test() (the explicit Wald-F lead-coefficient routine) uses
        # the same Theorem 3 variance machinery — values across default vs
        # explicit hc1 must agree to within iterative-FE-solver convergence
        # tolerance. Sub-ULP differences come from BLAS non-associativity in
        # the shared MAP demean engine (`demean_by_groups`) across distinct
        # estimator instances and are not methodological divergence under
        # the narrow vcov_type contract.
        pt_default = r_default.pretrend_test()
        pt_explicit = r_explicit.pretrend_test()
        assert np.isclose(pt_default["f_stat"], pt_explicit["f_stat"], rtol=0, atol=1e-12)
        assert np.isclose(pt_default["p_value"], pt_explicit["p_value"], rtol=0, atol=1e-12)
        # Event-study SE (computed during fit() via Theorem 3 machinery on
        # within-transformed residuals; pretrends=True path includes the
        # pre-period horizons). Sub-ULP differences come from BLAS
        # non-associativity in the shared MAP demean engine
        # (`demean_by_groups`) across distinct estimator instances and are
        # not methodological divergence under the narrow vcov_type contract.
        assert r_default.event_study_effects is not None
        assert r_explicit.event_study_effects is not None
        for h in r_default.event_study_effects:
            assert np.isclose(
                r_default.event_study_effects[h]["se"],
                r_explicit.event_study_effects[h]["se"],
                rtol=0,
                atol=1e-12,
                equal_nan=True,
            )

    # ---- Introspection / safety-gate tests --------------------------------

    def test_default_vcov_type_is_hc1(self):
        assert ImputationDiD().vcov_type == "hc1"

    def test_get_params_includes_vcov_type(self):
        params = ImputationDiD().get_params()
        assert "vcov_type" in params
        assert params["vcov_type"] == "hc1"

    def test_results_carries_vcov_type(self):
        data = generate_test_data(seed=11)
        r = ImputationDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert r.vcov_type == "hc1"

    def test_to_dict_includes_vcov_type(self):
        data = generate_test_data(seed=11)
        r = ImputationDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        d = r.to_dict()
        assert d["vcov_type"] == "hc1"
        # Headline alias keys are present per the TripleDifference precedent.
        for k in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper"):
            assert k in d

    def test_summary_includes_vcov_type_label_default(self):
        # cluster=None still routes the Theorem 3 variance through
        # cluster_var=unit at imputation.py:418, so summary should render
        # the unit-clustered CR1 label rather than generic HC1.
        data = generate_test_data(seed=11)
        r = ImputationDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        text = r.summary()
        assert "Variance estimator:" in text
        assert "CR1 cluster-robust" in text
        assert "unit" in text
        assert r.cluster_name == "unit"
        assert r.n_clusters == data["unit"].nunique()

    def test_summary_suppresses_variance_label_under_bootstrap(self, ci_params):
        # Under bootstrap fits, fit() overwrites the reported SE/CI/p-value
        # with bootstrap_results, so the analytical variance-family label
        # would misstate the inference source. Mirror the canonical
        # DiDResults gate at diff_diff/results.py:213-226.
        data = generate_test_data(seed=11)
        n_boot = ci_params.bootstrap(199)
        r = ImputationDiD(n_bootstrap=n_boot, seed=7).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        text = r.summary()
        assert "Inference method:" in text
        assert "bootstrap" in text
        # Analytical variance-family label must be suppressed.
        assert "Variance estimator:" not in text
        assert "CR1 cluster-robust" not in text
        assert "HC1 heteroskedasticity-robust" not in text

    def test_summary_includes_vcov_type_label_cluster(self):
        data = _imputation_clustered_panel()
        r = ImputationDiD(cluster="state").fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        text = r.summary()
        assert "Variance estimator:" in text
        # Cluster path renders the CR1 cluster-robust label per _format_vcov_label.
        assert "CR1 cluster-robust" in text
        assert "state" in text
        assert r.cluster_name == "state"
        assert r.n_clusters is not None and r.n_clusters > 1

    def test_cluster_name_suppressed_under_survey(self):
        data = _imputation_survey_panel()
        design = SurveyDesign(
            weights="weight",
            psu="psu",
            strata="stratum",
            weight_type="pweight",
        )
        r = ImputationDiD(cluster="psu").fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        # Under survey designs, Results.cluster_name and n_clusters are
        # suppressed so they can't misreport the bare cluster argument
        # when the resolver picks the survey PSU as the effective cluster.
        assert r.cluster_name is None
        assert r.n_clusters is None

    def test_cluster_name_suppressed_under_replicate_survey(self):
        # Replicate-weight survey designs have psu=None but still must
        # suppress cluster_name/n_clusters: replicate variance is computed
        # by replicate reweighting (BRR / Fay / JK1 / JKn / SDR) and
        # ignores PSU/cluster entirely, so populating cluster_name="unit"
        # and n_clusters=n_units would misreport the inference source.
        # Summary must also omit the "Number of clusters:" line and the
        # CR1 cluster-robust label.
        data, rep_cols = _imputation_replicate_panel()
        design = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
            weight_type="pweight",
        )
        r = ImputationDiD().fit(
            data,
            outcome="outcome",
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
        data = generate_test_data(seed=11)
        imp1 = ImputationDiD(vcov_type="hc1")
        r1 = imp1.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        imp2 = ImputationDiD(**imp1.get_params())
        r2 = imp2.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert r1.overall_se == r2.overall_se
        assert r1.vcov_type == r2.vcov_type

    def test_imputation_did_convenience_func_rejects_bad_vcov(self):
        data = generate_test_data(seed=11)
        with pytest.warns(FutureWarning, match=r"imputation_did\(\) is deprecated"):
            with pytest.raises(ValueError, match="influence-function"):
                imputation_did(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    vcov_type="classical",
                )

    def test_imputation_did_convenience_func_threads_vcov_type(self):
        data = generate_test_data(seed=11)
        with pytest.warns(FutureWarning, match=r"imputation_did\(\) is deprecated"):
            r = imputation_did(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                vcov_type="hc1",
            )
        assert r.vcov_type == "hc1"


# =============================================================================
# TestZeroWeightGroups — shared-engine migration behavioral locks
# =============================================================================


class TestZeroWeightGroups:
    """Zero-total-weight groups on the Step-1 paths (shared-engine migration).

    JK1/plain-BRR replicate weights zero whole PSUs and reach Step 1 unmasked.
    Before the shared-engine migration the pandas loops divided 0/0 there:
    with covariates, y_dm/X_dm NaN-poisoned, EVERY replicate refit failed
    inside solve_ols(check_finite=True), and the fit returned NaN SEs after a
    non-convergence warning storm. These tests lock the fixed contract.
    """

    @staticmethod
    def _with_covariates(data, seed=7):
        rng = np.random.default_rng(seed)
        d = data.copy()
        x = rng.standard_normal((len(d), 2))
        d["x1"], d["x2"] = x[:, 0], x[:, 1]
        d["outcome"] = d["outcome"] + x @ np.array([0.6, -0.3])
        return d

    def test_replicate_covariates_zero_weight_psus_finite_se(self):
        """Covariates + JK1 zeroed-PSU replicates -> finite SE, no warning storm."""
        data, rep_cols = _imputation_replicate_panel()
        data = self._with_covariates(data)
        design = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
            weight_type="pweight",
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            r = ImputationDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
                survey_design=design,
            )
        messages = [str(w.message) for w in rec]
        assert not any("replicate refits failed" in m for m in messages)
        assert not any("did not converge" in m for m in messages)
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se) and r.overall_se > 0

    def test_main_fit_zero_weight_treated_unit_covariates(self):
        """Main fit with a zero-weight treated unit + covariates.

        Before: opaque ValueError from solve_ols (NaN in demeaned design).
        After: fit succeeds; the zero-weight unit's FE is NaN, its cohort
        cell (it is the ONLY cohort-2 unit) goes NaN across ALL inference
        fields, and the overall ATT stays finite.
        """
        rng = np.random.default_rng(11)
        n_units, n_periods = 30, 6
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        first_treat = np.zeros(n_units, dtype=int)
        first_treat[0] = 2  # the ONLY cohort-2 unit — will carry zero weight
        first_treat[10:20] = 3
        ft = np.repeat(first_treat, n_periods)
        x = rng.standard_normal((len(units), 2))
        post = (ft > 0) & (times >= ft)
        outcome = (
            np.repeat(rng.standard_normal(n_units), n_periods)
            + 0.2 * times
            + 1.5 * post
            + x @ np.array([0.6, -0.3])
            + rng.standard_normal(len(units)) * 0.3
        )
        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcome,
                "first_treat": ft,
                "x1": x[:, 0],
                "x2": x[:, 1],
                "w": np.where(units == 0, 0.0, 1.0),
            }
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = ImputationDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
                survey_design=SurveyDesign(weights="w"),
                aggregate="group",
            )
        from tests.conftest import assert_nan_inference

        assert np.isfinite(r.overall_att)
        assert r.group_effects is not None
        # Cohort 2 contains only the zero-weight unit: NaN across ALL fields.
        assert_nan_inference(r.group_effects[2])
        assert np.isnan(r.group_effects[2]["effect"])
        # Cohort 3 is unaffected.
        assert np.isfinite(r.group_effects[3]["effect"])


class TestLeadSnapAbsorbed:
    """FE-spanned lead indicators on the pretrends path are SNAPPED to exact
    zero (deterministic NaN coefficient + cause-specific warning) instead of
    reaching the solver as numerical junk — the snap_absorbed_regressors
    adoption on _compute_lead_coefficients (TODO row: lead columns are the
    most plausible FE-spanned regressors)."""

    @staticmethod
    def _panel(never_treated_last_period):
        rng = np.random.default_rng(5)
        rows = []
        for i in range(30):
            ft = 7 if i < 15 else 0
            periods = range(1, 8) if ft else range(1, never_treated_last_period + 1)
            for t in periods:
                y = (
                    1.0
                    + 0.1 * i
                    + 0.2 * t
                    + (1.0 if (ft and t >= ft) else 0.0)
                    + rng.normal(0, 0.1)
                )
                rows.append({"unit": i, "time": t, "first_treat": ft, "y": y})
        return pd.DataFrame(rows)

    def test_spanned_lead_snaps_to_nan_with_cause_warning(self):
        # Never-treated units end at t=4, so Omega_0 at t=5 contains ONLY the
        # g=7 cohort: lead[-2] == 1{t==5} on Omega_0 — exactly in the span of
        # the absorbed time FE.
        df = self._panel(never_treated_last_period=4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = ImputationDiD(pretrends=True).fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )
        assert res.event_study_effects is not None
        eff = res.event_study_effects
        # The spanned lead is deterministically NaN — the FULL inference
        # tuple (review P3: assert every field, not just effect/se).
        assert np.isnan(eff[-2]["effect"]) and np.isnan(eff[-2]["se"])
        assert np.isnan(eff[-2]["t_stat"]) and np.isnan(eff[-2]["p_value"])
        assert np.all(np.isnan(np.asarray(eff[-2]["conf_int"], dtype=float)))
        # Of the remaining leads {-6,-5,-4,-3}, the leads-sum dummy trap costs
        # exactly ONE more column — but WHICH one the rank handler drops is
        # pivoted-QR/BLAS-order dependent (observed: -4 on macOS/Accelerate,
        # -3 on linux-arm py3.11, -6 on the pure-python CI backend). Assert
        # the count and the health of the survivors, not specific horizons.
        others = [-6, -5, -4, -3]
        finite = [h for h in others if np.isfinite(eff[h]["effect"])]
        assert len(finite) == len(others) - 1, f"finite leads: {finite}"
        for h in finite:
            assert np.isfinite(eff[h]["se"]) and eff[h]["se"] > 0, f"h={h}"
        # Cause-specific snap warning names the display label, not the raw column.
        snap_msgs = [
            str(x.message)
            for x in w
            if "collinear with the absorbed fixed effects" in str(x.message)
        ]
        assert any("lead[-2]" in m and "pretrends lead model" in m for m in snap_msgs), snap_msgs

    def test_identified_leads_unchanged_no_snap_warning(self):
        # Balanced never-treated span: every lead period has never-treated
        # rows in Omega_0 -> nothing is FE-spanned; the snap is a no-op and
        # no cause-specific warning fires.
        df = self._panel(never_treated_last_period=7)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = ImputationDiD(pretrends=True).fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )
        assert res.event_study_effects is not None
        finite_leads = [
            h for h, e in res.event_study_effects.items() if h < -1 and np.isfinite(e["effect"])
        ]
        assert len(finite_leads) >= 4
        assert not any("collinear with the absorbed fixed effects" in str(x.message) for x in w)


class TestLSMRFallbackParity:
    """The sparse LSMR fallback replaces dense lstsq on the (possibly
    singular) normal equations. Solver choice cannot change the estimator:
    least-squares solutions differ only by null(A_0'[W]A_0) = null(sqrt(W)A_0)
    components, which the projection v = -[W_0] A_0 z annihilates. Lock the
    projection parity against a dense-lstsq oracle on a genuinely singular
    system."""

    def test_singular_system_projection_matches_dense_oracle(self):
        import scipy.sparse as sp

        from diff_diff.imputation import _lsmr_minnorm_normal_solve

        rng = np.random.default_rng(3)
        n, p = 200, 12
        A0_dense = rng.normal(size=(n, p))
        A0_dense[:, -1] = A0_dense[:, 0]  # exact collinearity -> singular normal eqs
        A_0 = sp.csr_matrix(A0_dense)
        A0tA0 = sp.csc_matrix(A_0.T @ A_0)
        rhs = rng.normal(size=p)

        z_lsmr = _lsmr_minnorm_normal_solve(A0tA0, rhs)
        z_dense = np.linalg.lstsq(A0tA0.toarray(), rhs, rcond=None)[0]
        assert np.all(np.isfinite(z_lsmr))
        # The z's may differ by a null-space component; the PROJECTION A_0 z
        # (what the estimator consumes) must agree.
        np.testing.assert_allclose(A_0 @ z_lsmr, A_0 @ z_dense, rtol=0, atol=1e-8)

    def test_weighted_singular_system_projection_matches_dense_oracle(self):
        """Weighted variant (CI-review D1): the production path solves
        (A_0'[W]A_0) z = rhs with survey weights W. Null-space components of
        the weighted normal equations live in null(sqrt(W) A_0), so the
        WEIGHTED projection W_0 A_0 z — what the weighted estimator
        consumes — must agree across solvers even where the unweighted
        projection A_0 z need not."""
        import scipy.sparse as sp

        from diff_diff.imputation import _lsmr_minnorm_normal_solve

        rng = np.random.default_rng(9)
        n, p = 180, 10
        A0_dense = rng.normal(size=(n, p))
        A0_dense[:, -1] = 2.0 * A0_dense[:, 1]  # exact collinearity
        w = rng.uniform(0.2, 3.0, size=n)
        w[:12] = 0.0  # zero-weight rows (subpopulation) stay inert
        A_0 = sp.csr_matrix(A0_dense)
        A0tWA0 = sp.csc_matrix((A_0.T.multiply(w)) @ A_0)
        rhs = rng.normal(size=p)

        z_lsmr = _lsmr_minnorm_normal_solve(A0tWA0, rhs)
        z_dense = np.linalg.lstsq(A0tWA0.toarray(), rhs, rcond=None)[0]
        assert np.all(np.isfinite(z_lsmr))
        np.testing.assert_allclose(w * (A_0 @ z_lsmr), w * (A_0 @ z_dense), rtol=0, atol=1e-8)

    def test_no_dense_materialization_on_fallback(self, monkeypatch):
        """The singular-build fallback path must never call .toarray() on the
        normal matrix (the O((U+T+K)^2) OOM risk this closes)."""
        import unittest.mock

        import diff_diff.imputation_aggregation as imp

        data = generate_test_data(n_units=60, n_periods=6, seed=7)

        with unittest.mock.patch(
            "diff_diff.imputation_aggregation.sparse_factorized", side_effect=RuntimeError("forced")
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                orig_lsmr = imp._lsmr_minnorm_normal_solve
                calls = []

                def _spy(mat, rhs):
                    calls.append(mat.shape)
                    mat.toarray = None  # densifying would now raise
                    return orig_lsmr(mat, rhs)

                monkeypatch.setattr(imp, "_lsmr_minnorm_normal_solve", _spy)
                res = ImputationDiD().fit(
                    data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
                )
        assert calls, "fallback path did not route through the LSMR solver"
        assert np.isfinite(res.overall_att)
        assert any("sparse LSMR" in str(x.message) for x in w)

    def test_unconverged_lsmr_fails_closed_to_nan(self, monkeypatch):
        """CI-review P1 regression: a finite-but-uncertified LSMR result
        (istop outside {0,1,2,4,5} on both attempts) must NOT feed the
        variance; the solve raises _LSMRUnconvergedError and the variance
        boundary returns NaN, so inference degrades to NaN."""
        import scipy.sparse as sp

        import diff_diff.imputation as imp

        def _fake_lsmr(A, b, **kwargs):
            # finite vector, but istop=7 (max-iteration exhaustion)
            return (np.ones(A.shape[0]), 7, 5, 1.0, 1.0, 1.0, 1.0, 1.0)

        monkeypatch.setattr("scipy.sparse.linalg.lsmr", _fake_lsmr)
        A0tA0 = sp.csc_matrix(np.eye(4))
        with pytest.warns(UserWarning, match="did not converge"):
            with pytest.raises(imp._LSMRUnconvergedError):
                imp._lsmr_minnorm_normal_solve(A0tA0, np.ones(4))

    def test_unconverged_lsmr_fit_level_nan_inference(self, monkeypatch):
        """CI-review P0 regression: a globally failed solve must NOT be
        laundered into finite inference by the missing-FE nan_to_num — the
        full inference tuple degrades to NaN at the variance boundary."""
        import unittest.mock

        def _fake_lsmr(A, b, **kwargs):
            return (np.ones(A.shape[0]), 7, 5, 1.0, 1.0, 1.0, 1.0, 1.0)

        data = generate_test_data(n_units=60, n_periods=6, seed=7)
        monkeypatch.setattr("scipy.sparse.linalg.lsmr", _fake_lsmr)
        with unittest.mock.patch(
            "diff_diff.imputation_aggregation.sparse_factorized", side_effect=RuntimeError("forced")
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ImputationDiD().fit(
                    data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
                )
        assert np.isfinite(res.overall_att)  # point estimate unaffected
        assert np.isnan(res.overall_se)
        assert np.isnan(res.overall_t_stat)
        assert np.isnan(res.overall_p_value)
        assert np.all(np.isnan(np.asarray(res.overall_conf_int, dtype=float)))

    def test_machine_precision_istop_accepted(self, monkeypatch):
        """CI-review P1 regression: istop 4/5 (machine-precision analogues of
        1/2 per SciPy) are certified — no retry, no failure handling."""
        import scipy.sparse as sp2

        import diff_diff.imputation as imp2

        calls = []

        def _fake_lsmr(A, b, **kwargs):
            calls.append(kwargs)
            return (np.full(A.shape[0], 2.0), 4, 5, 1.0, 1.0, 1.0, 1.0, 1.0)

        monkeypatch.setattr("scipy.sparse.linalg.lsmr", _fake_lsmr)
        z = imp2._lsmr_minnorm_normal_solve(sp2.csc_matrix(np.eye(3)), np.ones(3))
        assert len(calls) == 1  # accepted on the first attempt
        np.testing.assert_array_equal(z, np.full(3, 2.0))


class TestImputationDfConvention:
    """The three-value df_convention knob on ImputationDiD's pretrends lead
    regression (3.9 / M-127) — the one ImputationDiD surface on the shared
    clustered CR1 sandwich. BJS overall/post inference and the joint pretrend
    Wald F are knob-independent.
    """

    @staticmethod
    def _panel(seed=13):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(50):
            ft = [0, 5, 7][u % 3]
            for t in range(1, 10):
                eff = 0.5 if (ft and t >= ft) else 0.0
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        first_treat=ft,
                        outcome=0.3 * u / 50 + 0.15 * t + eff + rng.standard_normal() * 0.5,
                    )
                )
        return pd.DataFrame(rows)

    _kw = dict(
        outcome="outcome",
        unit="unit",
        time="time",
        first_treat="first_treat",
        aggregate="event_study",
    )

    @staticmethod
    def _t_p(t_stat, df):
        from scipy import stats

        return 2 * stats.t.sf(abs(t_stat), df)

    @staticmethod
    def _z_p(t_stat):
        from scipy import stats

        return 2 * stats.norm.sf(abs(t_stat))

    def _lead(self, res):
        h = min(k for k in res.event_study_effects if k < -1)
        return h, res.event_study_effects[h]

    def test_leads_are_t_residual_not_z(self):
        res = ImputationDiD(pretrends=True).fit(self._panel(), **self._kw)
        h, e = self._lead(res)
        assert e["p_value"] != self._z_p(e["t_stat"])
        match = [d for d in range(2, 600) if abs(e["p_value"] - self._t_p(e["t_stat"], d)) < 1e-13]
        assert len(match) == 1

    def test_cluster_matches_g_minus_1_on_leads_only(self):
        data = self._panel()
        r0 = ImputationDiD(pretrends=True).fit(data, **self._kw)
        rc = ImputationDiD(pretrends=True, df_convention="cluster").fit(data, **self._kw)
        h, e0 = self._lead(r0)
        ec = rc.event_study_effects[h]
        G = data["unit"].nunique()
        assert ec["effect"] == e0["effect"] and ec["se"] == e0["se"]
        assert ec["p_value"] == pytest.approx(self._t_p(ec["t_stat"], G - 1), rel=1e-12)
        # post rows are BJS (knob-independent)
        hp = min(k for k in r0.event_study_effects if k >= 0)
        assert r0.event_study_effects[hp]["p_value"] == rc.event_study_effects[hp]["p_value"]

    def test_normal_reproduces_pre39_z_on_leads(self):
        data = self._panel()
        r0 = ImputationDiD(pretrends=True).fit(data, **self._kw)
        rn = ImputationDiD(pretrends=True, df_convention="normal").fit(data, **self._kw)
        h, e0 = self._lead(r0)
        en = rn.event_study_effects[h]
        assert en["effect"] == e0["effect"] and en["se"] == e0["se"]
        assert en["p_value"] == pytest.approx(self._z_p(en["t_stat"]), rel=1e-14)

    def test_pretrend_wald_f_is_knob_independent(self):
        data = self._panel()
        pt0 = ImputationDiD(pretrends=True).fit(data, **self._kw).pretrend_test()
        ptc = (
            ImputationDiD(pretrends=True, df_convention="cluster")
            .fit(data, **self._kw)
            .pretrend_test()
        )
        assert pt0["p_value"] == ptc["p_value"]
        assert pt0["f_stat"] == ptc["f_stat"]

    def test_survey_df_precedence_on_leads(self):
        """Survey design df wins on the pretrends leads under EVERY knob
        value (the knob is never consulted on surveyed fits)."""
        data = self._panel()
        data["weight"] = 1.0 + (data["unit"] % 5) * 0.2
        data["stratum"] = data["unit"] % 4
        data["psu"] = data["unit"]
        design = SurveyDesign(weights="weight", psu="psu", strata="stratum", weight_type="pweight")
        fits = {
            conv: ImputationDiD(pretrends=True, df_convention=conv).fit(
                data, survey_design=design, **self._kw
            )
            for conv in ("residual", "cluster", "normal")
        }
        h, e0 = self._lead(fits["residual"])
        for conv in ("cluster", "normal"):
            e = fits[conv].event_study_effects[h]
            assert e["p_value"] == e0["p_value"]
            assert e["conf_int"] == e0["conf_int"]

    def test_inert_config_warns_on_explicit_nondefault(self):
        """M-127 REACHABILITY predicate (revised with the M-021 post-fit
        migration): warn iff the per-lead inference is unreachable on every
        route for this fit config — pretrends=False always warns;
        pretrends=True analytical fits never warn (post-fit
        results.aggregate('event_study') reaches the leads regardless of
        the deprecated fit-time aggregate value); pretrends=True with
        n_bootstrap>0 and aggregate unset warns (no ES surface is built
        and post-fit aggregate() fails closed); pretrends=True with the
        deprecated fit-time ES supplied never warns, bootstrap included
        (fit-time leads use analytical inference)."""
        data = self._panel()
        base = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        with pytest.warns(UserWarning, match="affects only the pretrends"):
            ImputationDiD(df_convention="cluster").fit(data, **base)
        # pretrends=True + deprecated aggregate='group': the knob is NO
        # LONGER inert (post-fit ES reaches the leads) — no inert-config
        # warning (the FutureWarning from aggregate= is separate).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ImputationDiD(pretrends=True, df_convention="cluster").fit(
                data, aggregate="group", **base
            )
        assert not any("affects only the pretrends" in str(w.message) for w in caught)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ImputationDiD(pretrends=True, df_convention="cluster").fit(data, **self._kw)
        assert not any("affects only the pretrends" in str(w.message) for w in caught)
        # pretrends=True + bootstrap, aggregate unset: unreachable → warns.
        with pytest.warns(UserWarning, match="affects only the pretrends"):
            ImputationDiD(pretrends=True, df_convention="cluster", n_bootstrap=9, seed=1).fit(
                data, **base
            )
        # pretrends=True + bootstrap + deprecated fit-time ES: reachable
        # (analytical lead inference rides the fit-time surface) → no warn.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ImputationDiD(pretrends=True, df_convention="cluster", n_bootstrap=9, seed=1).fit(
                data, aggregate="event_study", **base
            )
        assert not any("affects only the pretrends" in str(w.message) for w in caught)

    def test_validation_and_transactional_set_params(self):
        with pytest.raises(ValueError, match="df_convention"):
            ImputationDiD(df_convention="bogus")
        est = ImputationDiD()
        with pytest.raises(ValueError, match="df_convention"):
            est.set_params(df_convention="bogus", alpha=0.10)
        assert est.df_convention == "residual" and est.alpha == 0.05
        # Valid value + unknown key: the unknown-key rejection must also
        # leave the estimator fully unchanged (no partial application).
        before = est.get_params()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(df_convention="normal", nonexistent_param=1)
        assert est.get_params() == before
        assert ImputationDiD(df_convention="normal").get_params()["df_convention"] == "normal"


@pytest.fixture(scope="module")
def alpha_fitted():
    data = generate_test_data()
    return ImputationDiD().fit(
        data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
    )


class TestSummaryAlphaContract:
    """summary(alpha=...) never recomputes stored inference.

    Family-wide guard (results_base._require_fit_alpha): a non-fit alpha
    raises instead of silently relabeling the confidence-interval header
    over fit-time stored intervals; alpha=0.0 (previously swallowed by the
    falsy `alpha or self.alpha` idiom) now raises too.
    """

    @pytest.mark.parametrize("bad_alpha", [0.10, 0.0])
    def test_summary_rejects_non_fit_alpha(self, alpha_fitted, bad_alpha):
        with pytest.raises(ValueError, match="never recomputes"):
            alpha_fitted.summary(alpha=bad_alpha)

    def test_summary_accepts_fit_alpha(self, alpha_fitted):
        assert alpha_fitted.summary(alpha=alpha_fitted.alpha) == alpha_fitted.summary()
