"""
Tests for Stacked DiD estimator (Wing, Freedman & Hollingsworth 2024).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import StackedDiD, StackedDiDResults
from diff_diff.prep_dgp import generate_staggered_data
from diff_diff.results_base import EVENT_STUDY_SCHEMA

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def staggered_data():
    """Standard staggered adoption data for testing."""
    return generate_staggered_data(
        n_units=200,
        n_periods=12,
        cohort_periods=[4, 6, 8],
        never_treated_frac=0.3,
        treatment_effect=5.0,
        dynamic_effects=True,
        seed=42,
    )


@pytest.fixture
def constant_effect_data():
    """Staggered data with constant treatment effect (no dynamics)."""
    return generate_staggered_data(
        n_units=200,
        n_periods=12,
        cohort_periods=[4, 6, 8],
        never_treated_frac=0.3,
        treatment_effect=5.0,
        dynamic_effects=False,
        seed=42,
    )


@pytest.fixture
def no_never_treated_data():
    """Staggered data without never-treated units."""
    return generate_staggered_data(
        n_units=200,
        n_periods=12,
        cohort_periods=[4, 6, 8],
        never_treated_frac=0.0,
        treatment_effect=5.0,
        dynamic_effects=True,
        seed=42,
    )


# =============================================================================
# TestStackedDiDBasic
# =============================================================================


class TestStackedDiDBasic:
    """Basic functionality tests."""

    def test_basic_fit(self, staggered_data):
        """Default parameters produce valid results."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results, StackedDiDResults)
        assert np.isfinite(results.att)
        assert np.isfinite(results.se)
        assert results.se > 0
        assert results.n_stacked_obs > 0
        assert results.n_sub_experiments > 0

    def test_event_study(self, staggered_data):
        """Event study aggregation populates event_study_effects."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.event_study_effects is not None
        assert -1 in results.event_study_effects  # reference period
        # Reference period effect should be zero
        ref = results.event_study_effects[-1]
        assert ref["effect"] == 0.0
        assert ref["is_reference"] is True and "n_obs" not in ref  # 4.0 marking (M-093)

        # Post-treatment periods should have effects
        for h in range(0, 3):
            if h in results.event_study_effects:
                assert results.event_study_effects[h]["n_obs"] > 0

    def test_fit_time_aggregate_kwarg_removed(self, staggered_data):
        """4.0: the fit-time ``aggregate=`` keyword is gone (native TypeError);
        the event study is always computed."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        with pytest.raises(TypeError, match="unexpected keyword argument 'aggregate'"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="group",
            )

    def test_group_aggregation_unsupported(self, staggered_data):
        """StackedDiD exposes no group aggregation: the post-fit
        ``aggregate("group")`` raises the mixin's unsupported-type error."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            results.aggregate("group")

    def test_simple_att(self, staggered_data):
        """Plain fit: overall ATT plus the always-computed surface (M-024)."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert np.isfinite(results.att)
        # The event-study surface is ALWAYS materialized on 3.9+ fits.
        assert results.event_study_effects is not None
        assert results.group_effects is None

    def test_known_constant_effect(self, constant_effect_data):
        """With constant treatment effect, estimated ATT should be close."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            constant_effect_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        # Treatment effect is 5.0; allow generous tolerance
        assert (
            abs(results.att - 5.0) < 1.5
        ), f"Estimated ATT {results.att:.2f} too far from true effect 5.0"

    def test_dynamic_effects(self, staggered_data):
        """With dynamic effects, post-treatment coefficients should increase."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.event_study_effects is not None
        # Post-treatment effects should generally increase
        post_effects = [
            results.event_study_effects[h]["effect"]
            for h in sorted(results.event_study_effects.keys())
            if h >= 0 and results.event_study_effects[h]["n_obs"] > 0
        ]
        if len(post_effects) >= 2:
            # Last post should be larger than first post (dynamic growth)
            assert post_effects[-1] > post_effects[0]


# =============================================================================
# TestTrimming
# =============================================================================


class TestTrimming:
    """Tests for IC1/IC2 trimming logic."""

    def test_ic1_window_trimming(self, staggered_data):
        """Events outside the observation window are trimmed."""
        # With very large kappa, early/late events should be trimmed
        est = StackedDiD(kappa_pre=4, kappa_post=4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        # With kappa_pre=4, kappa_post=4 on 12 periods, some events should trim
        if len(results.trimmed_groups) > 0:
            assert any("Trimmed" in str(wi.message) for wi in w)

    def test_ic2_no_controls_trimming(self, no_never_treated_data):
        """Events without clean controls are trimmed with never_treated mode."""
        est = StackedDiD(kappa_pre=1, kappa_post=1, control_group="never_treated")
        # No never-treated units exist → all events should be trimmed
        with pytest.raises(ValueError, match="All.*adoption events were trimmed"):
            est.fit(
                no_never_treated_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_trimmed_groups_reported(self, staggered_data):
        """Trimmed groups are reported in results."""
        est = StackedDiD(kappa_pre=5, kappa_post=5)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                results = est.fit(
                    staggered_data,
                    outcome="outcome",
                    unit="unit",
                    time="period",
                    first_treat="first_treat",
                )
                # If some groups survive, check trimmed_groups
                assert isinstance(results.trimmed_groups, list)
            except ValueError:
                # All trimmed — expected for large kappa
                pass

    def test_all_trimmed_raises(self, staggered_data):
        """ValueError when all events are eliminated by trimming."""
        est = StackedDiD(kappa_pre=10, kappa_post=10)
        with pytest.raises(ValueError, match="All.*adoption events were trimmed"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_wider_window_more_trimming(self, staggered_data):
        """Larger kappa values should trim more (or equal) events."""
        est1 = StackedDiD(kappa_pre=1, kappa_post=1)
        results1 = est1.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )

        est2 = StackedDiD(kappa_pre=2, kappa_post=2)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results2 = est2.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

        assert len(results2.trimmed_groups) >= len(results1.trimmed_groups)


# =============================================================================
# TestQWeights
# =============================================================================


class TestQWeights:
    """Tests for Q-weight computation."""

    def test_treated_weight_is_one(self, staggered_data):
        """All treated observations have Q=1 under the default aggregate weighting."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        treated_weights = results.stacked_data.loc[results.stacked_data["_D_sa"] == 1, "_Q_weight"]
        assert np.allclose(treated_weights, 1.0)

    def test_aggregate_weighting_formula(self, staggered_data):
        """Q-weights match R's observation-count formula at (event_time, sub_exp) level."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, weighting="aggregate")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        sd = results.stacked_data

        # Compute expected Q per R formula at (event_time, sub_exp) level
        for et in sd["_event_time"].unique():
            et_data = sd[sd["_event_time"] == et]
            stack_treat_n = (et_data["_D_sa"] == 1).sum()
            stack_control_n = (et_data["_D_sa"] == 0).sum()
            for sub_exp in results.groups:
                sub_et = et_data[et_data["_sub_exp"] == sub_exp]
                sub_treat_n = (sub_et["_D_sa"] == 1).sum()
                sub_control_n = (sub_et["_D_sa"] == 0).sum()
                if sub_control_n > 0 and stack_treat_n > 0 and stack_control_n > 0:
                    expected_q = (sub_treat_n / stack_treat_n) / (sub_control_n / stack_control_n)
                    actual_q = sub_et.loc[sub_et["_D_sa"] == 0, "_Q_weight"].iloc[0]
                    assert (
                        abs(actual_q - expected_q) < 1e-10
                    ), f"Sub-exp {sub_exp}, et={et}: expected Q={expected_q:.6f}, got {actual_q:.6f}"

    def test_sample_share_weighting(self, staggered_data):
        """Verify sample_share Q formula."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, weighting="sample_share")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        sd = results.stacked_data

        # All weights should be positive and finite
        assert np.all(sd["_Q_weight"] > 0)
        assert np.all(np.isfinite(sd["_Q_weight"]))

    def test_weights_positive(self, staggered_data):
        """All Q-weights should be positive."""
        for w in ["aggregate", "sample_share"]:
            est = StackedDiD(kappa_pre=2, kappa_post=2, weighting=w)
            results = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
            assert np.all(results.stacked_data["_Q_weight"] > 0)


# =============================================================================
# TestCleanControl
# =============================================================================


class TestCleanControl:
    """Tests for clean control group definitions."""

    def test_not_yet_treated_default(self, staggered_data):
        """Default includes not-yet-treated and never-treated as controls."""
        est = StackedDiD(kappa_pre=1, kappa_post=1, control_group="not_yet_treated")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_control_units > 0

    def test_strict_excludes_more(self, staggered_data):
        """Strict mode should have fewer (or equal) controls than not_yet_treated."""
        est_nyt = StackedDiD(kappa_pre=2, kappa_post=2, control_group="not_yet_treated")
        results_nyt = est_nyt.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )

        est_strict = StackedDiD(kappa_pre=2, kappa_post=2, control_group="strict")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                results_strict = est_strict.fit(
                    staggered_data,
                    outcome="outcome",
                    unit="unit",
                    time="period",
                    first_treat="first_treat",
                )
                # Strict should have fewer or equal stacked obs
                assert results_strict.n_stacked_obs <= results_nyt.n_stacked_obs
            except ValueError:
                # Strict may trim all events — that's valid behavior
                pass

    def test_never_treated_only(self, staggered_data):
        """never_treated mode only uses never-treated as controls."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, control_group="never_treated")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        sd = results.stacked_data
        # All control units should have first_treat = inf
        control_ft = sd.loc[sd["_D_sa"] == 0, "first_treat"].unique()
        assert all(np.isinf(ft) for ft in control_ft)

    def test_never_treated_no_nevertreated_raises(self, no_never_treated_data):
        """Error when no never-treated units exist with never_treated mode."""
        est = StackedDiD(kappa_pre=1, kappa_post=1, control_group="never_treated")
        with pytest.raises(ValueError, match="All.*adoption events were trimmed"):
            est.fit(
                no_never_treated_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )


# =============================================================================
# TestCleanControlAnticipation
# =============================================================================


def _anticipation_panel(cohorts, lead, n_periods, effect=2.0, n_per_group=30):
    """Noiseless balanced panel: equal-sized groups adopting at ``cohorts``
    (``np.inf`` = never treated), with a constant effect that starts ``lead``
    periods BEFORE the recorded adoption date."""
    rows = []
    for unit in range(n_per_group * len(cohorts)):
        g = cohorts[unit % len(cohorts)]
        for t in range(n_periods):
            responding = np.isfinite(g) and t >= g - lead
            rows.append((unit, t, g, 0.5 * t + (unit % 7) + (effect if responding else 0.0)))
    return pd.DataFrame(rows, columns=["unit", "period", "first_treat", "outcome"])


def _control_cohorts(results):
    stacked = results.stacked_data
    return {
        float(a): sorted(float(g) for g in sub.loc[sub["_D_sa"] == 0, "first_treat"].unique())
        for a, sub in stacked.groupby("_sub_exp")
    }


_ANT_FIT = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


class TestCleanControlAnticipation:
    """A later adopter responds from ``A_s - anticipation``, so it is a clean
    control for event ``a`` only if ``A_s - anticipation > a + kappa_post``."""

    def test_anticipating_later_cohort_is_not_a_control(self):
        """Cohort 6 responds from period 5, inside cohort 3's window [0, 5].
        Using it as a control biased event time 2 to 1.5 and the overall ATT
        to 1.875 where the truth is 2 at every included horizon."""
        data = _anticipation_panel([3, 6, np.inf], lead=1, n_periods=9)
        results = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1).fit(data, **_ANT_FIT)
        assert _control_cohorts(results) == {3.0: [np.inf], 6.0: [np.inf]}
        es = results.aggregate("event_study").to_dataframe().set_index("event_time")
        assert es.loc[2, "att"] == pytest.approx(2.0, abs=1e-10)
        # anticipation periods count as post-treatment: h in {-1, 0, 1, 2}
        np.testing.assert_allclose(es.loc[[-1, 0, 1, 2], "att"].to_numpy(), 2.0, atol=1e-10)
        assert es.loc[-3, "att"] == pytest.approx(0.0, abs=1e-10)
        assert results.att == pytest.approx(2.0, abs=1e-10)

    def test_later_cohort_far_enough_away_stays_a_control(self):
        """Cohort 7 responds from period 6, after cohort 3's window closes."""
        data = _anticipation_panel([3, 7, np.inf], lead=1, n_periods=10)
        results = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1).fit(data, **_ANT_FIT)
        assert _control_cohorts(results) == {3.0: [7.0, np.inf], 7.0: [np.inf]}
        assert results.att == pytest.approx(2.0, abs=1e-10)

    def test_no_anticipation_keeps_the_paper_rule(self):
        """``anticipation=0``: A_s > a + kappa_post, so cohort 6 is a control
        for cohort 3 (Wing et al. 2024, Appendix A)."""
        data = _anticipation_panel([3, 6, np.inf], lead=0, n_periods=9)
        results = StackedDiD(kappa_pre=2, kappa_post=2).fit(data, **_ANT_FIT)
        assert _control_cohorts(results) == {3.0: [6.0, np.inf], 6.0: [np.inf]}
        assert results.att == pytest.approx(2.0, abs=1e-10)

    def test_strict_adds_its_separation_on_top(self):
        """``strict`` with anticipation longer than ``kappa_pre``: cohort 8
        passes A_s > a + kappa_post + kappa_pre = 7 but responds from period 6,
        inside cohort 4's window [1, 6]."""
        data = _anticipation_panel([4, 8, np.inf], lead=2, n_periods=12)
        results = StackedDiD(kappa_pre=1, kappa_post=2, anticipation=2, control_group="strict").fit(
            data, **_ANT_FIT
        )
        assert _control_cohorts(results) == {4.0: [np.inf], 8.0: [np.inf]}
        assert results.att == pytest.approx(2.0, abs=1e-10)

    def test_inclusion_check_uses_the_same_rule(self):
        """IC2 and the stack builder agree. With no never-treated units, cohort
        3's only candidate control is the later cohort: at 7 it qualifies and
        the last cohort is trimmed; at 6 nothing qualifies and the fit raises."""
        kept = _anticipation_panel([3, 7], lead=1, n_periods=10)
        with pytest.warns(UserWarning, match="Trimmed 1 adoption event"):
            results = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1).fit(kept, **_ANT_FIT)
        assert [float(g) for g in results.groups] == [3.0]
        assert [float(g) for g in results.trimmed_groups] == [7.0]
        assert _control_cohorts(results) == {3.0: [7.0]}
        assert results.att == pytest.approx(2.0, abs=1e-10)

        none_left = _anticipation_panel([3, 6], lead=1, n_periods=9)
        with pytest.raises(ValueError, match="adoption events were trimmed"):
            StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1).fit(none_left, **_ANT_FIT)


class TestCleanControlAnticipationSurvey:
    """Under ``survey_design=`` the design is re-resolved on the stacked rows.
    If the anticipation term takes a later cohort out of every sub-experiment,
    its PSUs and strata would leave the variance, so the fit is refused."""

    @staticmethod
    def _census(lead):
        """Stratified census: 3 strata x cohorts (3, 8, never) x 2 single-unit
        PSUs, FPC = 6 per stratum. Cohort 3's effects alternate 1.5 / 2.5 within
        a stratum and start ``lead`` periods before adoption."""
        rows, unit = [], 0
        for stratum in range(3):
            for g in (3, 8, np.inf):
                for k in range(2):
                    effect = 1.5 if k == 0 else 2.5
                    for t in range(9):
                        responding = g == 3 and t >= g - lead
                        y = 0.5 * t + unit % 5 + (effect if responding else 0.0)
                        rows.append((unit, t, g, y, stratum, unit, 1.0, 6))
                    unit += 1
        return pd.DataFrame(
            rows,
            columns=["unit", "period", "first_treat", "outcome", "stratum", "psu", "w", "N"],
        )

    @staticmethod
    def _design():
        from diff_diff import SurveyDesign

        return SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N")

    def test_refused_when_the_cutoff_removes_a_cohort_from_the_stack(self):
        """strict, kappa 2/2, anticipation=1: cohort 8 is trimmed as an event
        (its window ends at 10) and the adjusted cutoff (> 8) also drops it as
        a control for cohort 3, so its six PSUs would leave the design."""
        data = self._census(lead=1)
        est = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1, control_group="strict")
        with pytest.raises(NotImplementedError, match=r"cohort 8: 6"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(data, survey_design=self._design(), **_ANT_FIT)
        # the unweighted fit proceeds and excludes cohort 8 from the controls
        with pytest.warns(UserWarning, match="Trimmed 1 adoption event"):
            plain = est.fit(data, **_ANT_FIT)
        assert _control_cohorts(plain) == {3.0: [np.inf]}
        assert plain.att == pytest.approx(2.0, abs=1e-10)

    def test_supported_when_no_unit_is_lost(self):
        """anticipation=0: cohort 8 passes the strict cutoff (> 7) and stays a
        control, so every PSU is in the stack. A census has zero design
        variance: 18 PSUs, 15 survey df, SE 0 and NaN inference."""
        data = self._census(lead=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = StackedDiD(kappa_pre=2, kappa_post=2, control_group="strict").fit(
                data, survey_design=self._design(), **_ANT_FIT
            )
        assert _control_cohorts(results) == {3.0: [8.0, np.inf]}
        assert results.survey_metadata.n_psu == 18
        assert results.survey_metadata.df_survey == 15
        assert results.att == pytest.approx(2.0, abs=1e-10)
        assert results.se == pytest.approx(0.0, abs=1e-12)
        assert np.isnan(results.t_stat) and np.isnan(results.p_value)
        assert all(np.isnan(b) for b in results.conf_int)

    def test_not_refused_when_the_cutoff_changes_nothing(self):
        """anticipation=1 with a later cohort far enough away (and with
        never-treated controls): no unit leaves the stack, so the survey fit
        runs."""
        rows, unit = [], 0
        for stratum in range(3):
            for g in (3, np.inf):
                for k in range(3):
                    for t in range(9):
                        y = 0.5 * t + unit % 5 + (2.0 if (g == 3 and t >= 2) else 0.0) + 0.1 * k * t
                        rows.append((unit, t, g, y, stratum, unit, 1.0, 12))
                    unit += 1
        data = pd.DataFrame(
            rows,
            columns=["unit", "period", "first_treat", "outcome", "stratum", "psu", "w", "N"],
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1)
        assert (
            est._units_lost_to_anticipation_cutoff(
                [3], 0, 8, pd.DataFrame({"_first_treat": [3.0, np.inf]})
            )
            == {}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = est.fit(data, survey_design=self._design(), **_ANT_FIT)
        assert results.survey_metadata.n_psu == 18
        assert np.isfinite(results.att)

    @staticmethod
    def _sampled_panel():
        """3 strata x cohorts (3, 6, never) x 2 units, FPC = 8 per stratum (a
        sample of 6 from 8). Effects of 1.5 / 2.5 within each cohort and
        stratum, starting one period before adoption."""
        rows, unit = [], 0
        for stratum in range(3):
            for g in (3, 6, np.inf):
                for k in range(2):
                    effect = 1.5 if k == 0 else 2.5
                    for t in range(9):
                        responding = np.isfinite(g) and t >= g - 1
                        y = 0.5 * t + unit % 5 + (effect if responding else 0.0)
                        rows.append((unit, t, g, y, stratum, unit, 1.0, 8))
                    unit += 1
        return pd.DataFrame(
            rows,
            columns=["unit", "period", "first_treat", "outcome", "stratum", "unit_id", "w", "N"],
        )

    @pytest.mark.parametrize(
        "cluster, psu, warning",
        [
            ("unit", None, None),
            ("unit_subexp", None, "uses each unit as its own PSU"),
            ("unit_subexp", "unit_id", "Both survey_design.psu and cluster"),
        ],
    )
    def test_implicit_survey_psu_is_the_original_unit(self, cluster, psu, warning):
        """With no ``psu`` column each ORIGINAL unit is the PSU, for either
        ``cluster``. (unit, sub-experiment) pairs are not sampling units: here
        they number 8 per stratum, which read against FPC = 8 as a census and
        gave SE 0 with 21 df. Cohort 6 leaves cohort 3's control pool under
        the anticipation cutoff but stays in the stack as a treated cohort, so
        nothing is refused."""
        from diff_diff import SurveyDesign

        data = self._sampled_panel()
        design = SurveyDesign(
            weights="w", strata="stratum", fpc="N", **({"psu": psu} if psu else {})
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1, cluster=cluster)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = est.fit(data, survey_design=design, **_ANT_FIT)
        messages = [str(w.message) for w in caught]
        if warning is None:
            assert not any("PSU" in m for m in messages)
        else:
            assert any(warning in m for m in messages)
        assert _control_cohorts(results) == {3.0: [np.inf], 6.0: [np.inf]}
        assert results.stacked_data["unit"].nunique() == 18
        assert results.survey_metadata.n_psu == 18
        assert results.survey_metadata.df_survey == 15
        assert results.att == pytest.approx(2.0, abs=1e-10)
        assert results.se == pytest.approx(0.07906, abs=1e-5)
        assert np.isfinite(results.t_stat) and np.isfinite(results.p_value)

    @staticmethod
    def _small_panel(strata, fpc=None):
        """Six units (two treated at 3, four never treated), nine periods."""
        cohorts = [3, 3, np.inf, np.inf, np.inf, np.inf]
        rows = []
        for unit, g in enumerate(cohorts):
            for t in range(9):
                effect = 2.0 + 0.5 * unit if (np.isfinite(g) and t >= g) else 0.0
                y = 0.5 * t + unit + effect + 0.3 * ((unit * 7 + t * 3) % 5)
                rows.append((unit, t, g, y, strata[unit], 1.0, fpc[unit] if fpc else 0.0, unit))
        return pd.DataFrame(
            rows,
            columns=["unit", "period", "first_treat", "outcome", "stratum", "w", "N", "unit_id"],
        )

    @pytest.mark.parametrize("cluster", ["unit", "unit_subexp"])
    def test_zero_survey_df_gives_nan_inference(self, cluster):
        """One PSU per stratum under lonely_psu="adjust": df_survey = 0, the
        variance is not identified. The df is no longer clamped to 1 (which
        reported t(1) p-values); estimates and SEs stay, inference is NaN."""
        from diff_diff import SurveyDesign

        data = self._small_panel(strata=list(range(6)))
        design = SurveyDesign(weights="w", strata="stratum", lonely_psu="adjust")
        est = StackedDiD(kappa_pre=2, kappa_post=2, cluster=cluster)
        with pytest.warns(UserWarning, match="zero degrees of freedom"):
            results = est.fit(data, survey_design=design, **_ANT_FIT)
        assert results.survey_metadata.df_survey == 0
        assert np.isfinite(results.att) and np.isfinite(results.se) and results.se > 0
        # the SE is identified as a number but not for inference
        assert np.isnan(results.t_stat) and np.isnan(results.p_value)
        assert all(np.isnan(bound) for bound in results.conf_int)
        es = results.aggregate("event_study").to_dataframe()
        rows = es[~es["is_reference"]]
        assert np.isfinite(rows["se"]).all()
        assert rows[["t_stat", "p_value", "conf_int_lower", "conf_int_upper"]].isna().all().all()

    @pytest.mark.parametrize("psu", [None, "unit_id"])
    def test_implicit_unit_psus_are_validated_like_explicit_ones(self, psu):
        """A stratum holding one unit warns that it is excluded from variance
        estimation, and an FPC below a stratum's PSU count raises, whether the
        unit PSU is implicit or named. Both fits give the same SE."""
        from diff_diff import SurveyDesign

        strata = [0, 1, 0, 1, 0, 2]  # the last unit is alone in its stratum
        extra = {"psu": psu} if psu else {}
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        with pytest.warns(UserWarning, match="Stratum 2 has only 1 PSU"):
            results = est.fit(
                self._small_panel(strata),
                survey_design=SurveyDesign(weights="w", strata="stratum", **extra),
                **_ANT_FIT,
            )
        assert results.se == pytest.approx(0.3307, abs=1e-4)
        assert results.survey_metadata.df_survey == 3
        with pytest.raises(ValueError, match=r"FPC \(0\.5\) is less than the number of PSUs"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(
                    self._small_panel(strata, fpc=[10, 10, 10, 10, 10, 0.5]),
                    survey_design=SurveyDesign(weights="w", strata="stratum", fpc="N", **extra),
                    **_ANT_FIT,
                )

    @pytest.mark.parametrize("psu", [None, "psu"])
    def test_fpc_is_checked_against_the_full_sample_not_the_stack(self, psu):
        """Six sampled units per stratum, FPC = 4. Cohort 8 is trimmed (its
        window does not fit) and, with never-treated controls, is no one's
        control, so only four units per stratum reach the stack. The implicit
        unit PSUs must be validated on the supplied sample, as an explicit
        ``psu`` is: the fit raises instead of passing the check on the trimmed
        stack and reporting SE 0."""
        from diff_diff import SurveyDesign

        data = self._census(lead=0)
        data["N"] = 4
        design = SurveyDesign(
            weights="w", strata="stratum", fpc="N", **({"psu": psu} if psu else {})
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2, control_group="never_treated")
        with pytest.raises(ValueError, match=r"FPC \(4\.0\) is less than the number of PSUs \(6\)"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(data, survey_design=design, **_ANT_FIT)

    def test_lost_units_helper(self):
        info = pd.DataFrame({"_first_treat": [3.0] * 2 + [6.0] * 3 + [8.0] * 4 + [np.inf] * 2})
        nyt = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1)
        # cohort 6 was a control for cohort 3 (6 > 5) and is not one now
        # (6 > 6 fails); it is still in the stack as a treated cohort
        # (window [3, 8] fits in [0, 10]), so nothing is lost
        assert nyt._units_lost_to_anticipation_cutoff([3, 6, 8], 0, 10, info) == {}
        # with periods 0..8 cohort 6's own window [3, 8] still fits, cohort 8's
        # does not: cohort 8 stays a control for cohort 3 (8 > 6), nothing lost
        assert nyt._units_lost_to_anticipation_cutoff([3, 6, 8], 0, 8, info) == {}
        strict = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1, control_group="strict")
        # strict: cohort 8 passed 8 > 7 and fails 8 > 8; its own window does
        # not fit in [0, 8], so its four units are in no sub-experiment
        assert strict._units_lost_to_anticipation_cutoff([3, 8], 0, 8, info) == {8.0: 4}
        never = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1, control_group="never_treated")
        assert never._units_lost_to_anticipation_cutoff([3, 6, 8], 0, 8, info) == {}
        assert (
            StackedDiD(kappa_pre=2, kappa_post=2)._units_lost_to_anticipation_cutoff(
                [3, 6, 8], 0, 8, info
            )
            == {}
        )


# =============================================================================
# TestClustering
# =============================================================================


class TestClustering:
    """Tests for clustering standard errors."""

    def test_unit_clustering(self, staggered_data):
        """Default unit clustering produces finite SEs."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, cluster="unit")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert np.isfinite(results.se)
        assert results.se > 0

    def test_unit_subexp_clustering(self, staggered_data):
        """unit_subexp clustering produces finite SEs."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, cluster="unit_subexp")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert np.isfinite(results.se)
        assert results.se > 0


# =============================================================================
# TestStackedData
# =============================================================================


class TestStackedData:
    """Tests for the stacked dataset."""

    def test_stacked_data_accessible(self, staggered_data):
        """results.stacked_data is a DataFrame."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results.stacked_data, pd.DataFrame)

    def test_required_columns(self, staggered_data):
        """Stacked data has _sub_exp, _event_time, _D_sa, _Q_weight."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        required = {"_sub_exp", "_event_time", "_D_sa", "_Q_weight"}
        assert required.issubset(results.stacked_data.columns)

    def test_event_time_range(self, staggered_data):
        """Event times span [-kappa_pre, ..., kappa_post]."""
        kp, kq = 2, 2
        est = StackedDiD(kappa_pre=kp, kappa_post=kq)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        et = results.stacked_data["_event_time"]
        # Event times should include the reference period -1
        assert et.min() <= -kp
        assert et.max() >= kq


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_cohort(self):
        """Works with only one adoption event."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[5],
            never_treated_frac=0.5,
            treatment_effect=3.0,
            dynamic_effects=False,
            seed=99,
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_sub_experiments == 1
        assert np.isfinite(results.att)

    def test_anticipation_reference_period(self):
        """anticipation=1 shifts reference period to e=-2."""
        data = generate_staggered_data(
            n_units=200,
            n_periods=12,
            cohort_periods=[5, 7],
            never_treated_frac=0.3,
            treatment_effect=5.0,
            seed=42,
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )

        # Reference period is -2 (not -1)
        assert -2 in results.event_study_effects
        assert results.event_study_effects[-2]["effect"] == 0.0
        assert results.event_study_effects[-2]["is_reference"] is True  # explicit marker (M-093)
        assert "n_obs" not in results.event_study_effects[-2]

        # -1 is NOT the reference; it should have a non-zero estimated effect
        assert -1 in results.event_study_effects
        assert results.event_study_effects[-1]["n_obs"] > 0

        # Extra pre-period -3 should have a dummy
        assert -3 in results.event_study_effects
        assert results.event_study_effects[-3]["n_obs"] > 0

        # Post-treatment includes anticipation period (-1)
        # Overall ATT averages h in {-1, 0, 1, 2}
        assert np.isfinite(results.att)

    def test_unbalanced_panel(self):
        """Missing observations inside a window trim the unit from that
        sub-experiment (complete-window rule), with a warning."""
        data = generate_staggered_data(
            n_units=200,
            n_periods=12,
            cohort_periods=[4, 6, 8],
            never_treated_frac=0.3,
            treatment_effect=5.0,
            seed=42,
        )
        # Remove some random rows to create unbalanced panel
        rng = np.random.default_rng(42)
        drop_idx = rng.choice(len(data), size=50, replace=False)
        data = data.drop(data.index[drop_idx]).reset_index(drop=True)

        est = StackedDiD(kappa_pre=2, kappa_post=2)
        with pytest.warns(UserWarning, match="not observed at every"):
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        sd = results.stacked_data
        assert np.isfinite(results.att)
        assert np.all(sd["_Q_weight"] > 0)
        assert np.all(np.isfinite(sd["_Q_weight"]))
        # Every unit left in a sub-experiment covers the whole window.
        rows = sd.groupby(["_sub_exp", "unit"])["_event_time"].agg(["size", "nunique"])
        assert (rows["size"] == 5).all() and (rows["nunique"] == 5).all()
        assert results.n_stacked_obs < len(
            StackedDiD(kappa_pre=2, kappa_post=2)
            .fit(
                generate_staggered_data(
                    n_units=200,
                    n_periods=12,
                    cohort_periods=[4, 6, 8],
                    never_treated_frac=0.3,
                    treatment_effect=5.0,
                    seed=42,
                ),
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
            .stacked_data
        )

    def test_nan_inference(self):
        """Degenerate case with NaN inference fields."""
        # Create small data where estimation might degenerate.
        # Need n > k to avoid division by zero in cluster-robust VCV:
        # Design matrix has 4 columns (intercept, D_sa, lambda_0, delta_0),
        # so we need > 4 observations (3 units × 2 periods = 6).
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2, 3, 3],
                "period": [1, 2, 1, 2, 1, 2],
                "outcome": [1.0, 2.0, 1.0, 2.0, 1.5, 2.5],
                "first_treat": [2, 2, 0, 0, 0, 0],
            }
        )
        est = StackedDiD(kappa_pre=1, kappa_post=0)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        # Should produce finite results or NaN (not crash)
        assert isinstance(results, StackedDiDResults)

    def test_never_treated_encoding_zero(self):
        """first_treat=0 treated same as first_treat=inf (never-treated)."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[5],
            never_treated_frac=0.5,
            treatment_effect=5.0,
            seed=42,
        )
        # The generator uses 0 for never-treated
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_control_units > 0

    def test_never_treated_encoding_inf(self):
        """first_treat=inf works for never-treated units."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[5],
            never_treated_frac=0.5,
            treatment_effect=5.0,
            seed=42,
        )
        # Replace 0 with inf for never-treated
        data["first_treat"] = data["first_treat"].replace(0, np.inf)

        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_control_units > 0


# =============================================================================
# TestSklearnInterface
# =============================================================================


class TestSklearnInterface:
    """Tests for sklearn-compatible API."""

    def test_get_params(self):
        """All init params present in get_params."""
        est = StackedDiD(
            kappa_pre=3,
            kappa_post=2,
            weighting="population",
            control_group="strict",
            cluster="unit_subexp",
            alpha=0.10,
            anticipation=1,
            rank_deficient_action="error",
        )
        params = est.get_params()
        assert params["kappa_pre"] == 3
        assert params["kappa_post"] == 2
        assert params["weighting"] == "population"
        # get_params returns the RAW args; control_group was passed
        # explicitly (M-043), clean_control stays at its sentinel.
        assert params["control_group"] == "strict"
        assert params["cluster"] == "unit_subexp"
        assert params["alpha"] == 0.10
        assert params["anticipation"] == 1
        assert params["rank_deficient_action"] == "error"

    def test_set_params(self):
        """set_params modifies attributes correctly."""
        est = StackedDiD()
        est.set_params(kappa_pre=5, weighting="sample_share")
        assert est.kappa_pre == 5
        assert est.weighting == "sample_share"

    def test_set_params_unknown_raises(self):
        """set_params raises on unknown parameter."""
        est = StackedDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(nonexistent_param=42)


# =============================================================================
# TestResultsMethods
# =============================================================================


class TestResultsMethods:
    """Tests for StackedDiDResults methods."""

    def test_summary(self, staggered_data):
        """summary() returns formatted string."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        summary = results.summary()
        assert "Stacked DiD" in summary
        assert "ATT" in summary

    def test_to_dataframe_event_study(self, staggered_data):
        """to_dataframe(level='event_study') returns DataFrame."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        df = results.to_dataframe(level="event_study")
        assert isinstance(df, pd.DataFrame)
        # 4.0 (M-093 8b): the unified EVENT_STUDY_SCHEMA frame, bit-equal to
        # the aggregate() container's.
        assert list(df.columns) == list(EVENT_STUDY_SCHEMA)
        pd.testing.assert_frame_equal(df, results.aggregate("event_study").to_dataframe())

    def test_to_dataframe_group_raises(self, staggered_data):
        """to_dataframe(level='group') raises ValueError."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        with pytest.raises(ValueError, match="Group aggregation is not supported"):
            results.to_dataframe(level="group")

    def test_to_dataframe_event_study_total_on_plain_fit(self, staggered_data):
        """Plain fits carry the surface (M-024): to_dataframe is total."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        df = results.to_dataframe(level="event_study")
        assert sorted(df["event_time"].tolist()) == [-2, -1, 0, 1, 2]
        assert df.loc[df["is_reference"], "event_time"].tolist() == [-1]

    def test_to_dataframe_legacy_pickle_none_branch(self, staggered_data):
        """The effects-None branch survives for pre-3.9 pickles only."""
        import dataclasses

        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        legacy = dataclasses.replace(results, event_study_effects=None)
        with pytest.raises(ValueError, match=r"diff-diff >= 3\.9"):
            legacy.to_dataframe(level="event_study")

    def test_summary_prints_event_study_table_on_plain_fit(self, staggered_data):
        """summary() renders the ES block on every 3.9+ fit (M-024)."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert "Event Study (Dynamic) Effects" in results.summary()

    def test_is_significant(self, staggered_data):
        """is_significant property works."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results.is_significant, bool)

    def test_significance_stars(self, staggered_data):
        """significance_stars property works."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results.significance_stars, str)

    def test_repr(self, staggered_data):
        """__repr__ returns formatted string."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        r = repr(results)
        assert "StackedDiDResults" in r
        assert "ATT=" in r


# =============================================================================
# TestValidation
# =============================================================================


class TestValidation:
    """Tests for input validation."""

    def test_missing_columns(self, staggered_data):
        """Raises on missing required columns."""
        est = StackedDiD()
        with pytest.raises(ValueError, match="Missing columns"):
            est.fit(
                staggered_data,
                outcome="nonexistent",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_invalid_weighting(self):
        """Raises on invalid weighting parameter."""
        with pytest.raises(ValueError, match="weighting"):
            StackedDiD(weighting="invalid")

    def test_invalid_control_group_and_removed_clean_control(self):
        """Raises on an invalid control-group value; the pre-3.9
        ``clean_control`` spelling is no longer a parameter (M-043, removed 4.0)."""
        with pytest.raises(ValueError, match="control_group"):
            StackedDiD(control_group="invalid")
        with pytest.raises(TypeError, match="unexpected keyword argument 'clean_control'"):
            StackedDiD(clean_control="invalid")

    def test_invalid_cluster(self):
        """Raises on invalid cluster parameter."""
        with pytest.raises(ValueError, match="cluster"):
            StackedDiD(cluster="invalid")

    def test_invalid_aggregate(self, staggered_data):
        """The fit-time ``aggregate=`` keyword is gone (4.0): any value is a
        native TypeError; an unknown post-fit aggregation type raises."""
        est = StackedDiD()
        with pytest.raises(TypeError, match="unexpected keyword argument 'aggregate'"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="invalid",
            )
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            results.aggregate("invalid")

    def test_population_required_for_population_weighting(self, staggered_data):
        """Raises when population col not specified with weighting='population'."""
        est = StackedDiD(weighting="population")
        with pytest.raises(ValueError, match="population"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_no_treated_units(self):
        """Raises when no treated units exist."""
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "period": [1, 2, 1, 2],
                "outcome": [1.0, 2.0, 1.0, 2.0],
                "first_treat": [0, 0, 0, 0],
            }
        )
        est = StackedDiD()
        with pytest.raises(ValueError, match="No treated units"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )


# =============================================================================
# TestStackedDiDVcovType — Phase 1b 2/8: vcov_type threading
# =============================================================================


@pytest.fixture
def baseline_panel():
    """Fixed-seed panel matching the captured pre-PR HC1 SE baseline.

    Captured on commit 955aa4be0887c71defb8cdab402e955f7c36e48d (PR #475
    clubSandwich port merge) BEFORE Phase 1b 2/8 source edits. Used by
    `test_hc1_se_bit_equal_to_pre_pr_baseline` to lock the bake-Q-into-X
    -> explicit-weights= switch as bit-equal on the hc1 path.
    """
    return generate_staggered_data(
        n_units=50,
        n_periods=8,
        cohort_periods=[3, 5, 7],
        never_treated_frac=0.3,
        treatment_effect=2.0,
        dynamic_effects=False,
        seed=20260521,
    )


class TestStackedDiDVcovType:
    """Phase 1b 2/8: vcov_type input contract, reject paths, and bit-equality.

    Mirrors the SunAbraham Phase 1b 1/8 test pattern. The plan locks 19
    sub-tests covering: default behavior, hc1 bit-equality vs prior bake-Q
    pattern, hc2_bm functional check, six reject paths, surface contract
    (get_params/set_params/results.vcov_type), clone idempotency, and the
    replicate-refit closure smoke.
    """

    def test_default_vcov_type_is_hc1(self):
        assert StackedDiD().vcov_type == "hc1"

    def test_default_bit_equal_to_explicit_hc1(self, staggered_data):
        """Default fit and explicit vcov_type='hc1' fit produce identical SE."""
        est_default = StackedDiD(kappa_pre=2, kappa_post=2)
        est_explicit = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc1")
        kwargs = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
        res_default = est_default.fit(staggered_data, **kwargs)
        res_explicit = est_explicit.fit(staggered_data, **kwargs)
        np.testing.assert_allclose(res_default.se, res_explicit.se, atol=1e-15)
        np.testing.assert_allclose(res_default.att, res_explicit.att, atol=1e-15)

    def test_hc1_se_bit_equal_to_pre_pr_baseline(self, baseline_panel):
        """HC1 SE matches the captured pre-PR baseline at machine precision.

        Baseline captured on commit 955aa4be (PR #475 clubSandwich merge),
        BEFORE switching from bake-Q-into-X to explicit weights= pattern in
        the StackedDiD solve_ols call. Locks the WLS-CR1 invariance claim
        (HC1 score is identical between the two algebraic forms; only
        multiplication ordering differs at ULP scale).

        Tolerance: atol=1e-13. The plan originally targeted atol=1e-14 but
        empirically the bake-w vs explicit-weights paths drift by ~2 ULPs at
        SE scale due to NumPy internal multiplication ordering — well within
        the "no methodologically significant drift" band.

        Panel descriptor (regenerate if test fails):
            generate_staggered_data(n_units=50, n_periods=8, cohort_periods=[3, 5, 7],
                                    never_treated_frac=0.3, treatment_effect=2.0,
                                    dynamic_effects=False, seed=20260521)
            StackedDiD(kappa_pre=2, kappa_post=2)  # defaults otherwise
        """
        BASELINE_OVERALL_ATT = 2.08078331612939
        BASELINE_OVERALL_SE = 0.15699149429146309
        est = StackedDiD(kappa_pre=2, kappa_post=2)  # default hc1
        res = est.fit(
            baseline_panel, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        np.testing.assert_allclose(
            res.att,
            BASELINE_OVERALL_ATT,
            atol=1e-13,
            err_msg="HC1 overall_att drifted from pre-PR baseline",
        )
        np.testing.assert_allclose(
            res.se,
            BASELINE_OVERALL_SE,
            atol=1e-13,
            err_msg="HC1 overall_se drifted from pre-PR baseline",
        )

    def test_hc2_bm_finite_and_att_identical_to_hc1(self, staggered_data):
        """hc2_bm produces finite SE; ATT identical to hc1 (only the variance
        family changes, not the point estimate). The relationship hc2_bm SE
        vs hc1 SE depends on cluster leverage and is NOT monotonic — both
        smaller and larger values are valid depending on the design."""
        kwargs = dict(
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        res_hc1 = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc1").fit(
            staggered_data, **kwargs
        )
        res_hc2bm = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm").fit(
            staggered_data, **kwargs
        )
        np.testing.assert_allclose(
            res_hc1.att,
            res_hc2bm.att,
            atol=1e-13,
            err_msg="ATT must be identical across vcov_type",
        )
        assert np.isfinite(res_hc2bm.se) and res_hc2bm.se > 0
        # Per-event-time SE also all finite under hc2_bm
        for h, eff in res_hc2bm.event_study_effects.items():
            if h == -1:
                continue  # reference period (SE=0 by construction)
            assert np.isfinite(eff["se"]) and eff["se"] > 0, f"event_time {h} hc2_bm SE not finite"
        # And hc2_bm event-study SE differs from hc1 event-study SE on at least one
        # event-time (proves the leverage/DOF adjustment actually fired).
        hc1_es = {h: eff["se"] for h, eff in res_hc1.event_study_effects.items() if h != -1}
        hc2_es = {h: eff["se"] for h, eff in res_hc2bm.event_study_effects.items() if h != -1}
        diffs = [abs(hc1_es[h] - hc2_es[h]) for h in hc1_es]
        assert (
            max(diffs) > 1e-6
        ), "hc2_bm event-study SEs identical to hc1 — leverage adjustment didn't fire"

    def test_classical_rejected_at_init(self):
        with pytest.raises(ValueError, match="clusters intrinsically"):
            StackedDiD(vcov_type="classical")

    def test_hc2_rejected_at_init(self):
        with pytest.raises(ValueError, match="clusters intrinsically"):
            StackedDiD(vcov_type="hc2")

    def test_conley_rejected_at_init_with_deferral(self):
        with pytest.raises(ValueError, match="conley"):
            StackedDiD(vcov_type="conley")

    def test_invalid_vcov_type_rejected(self):
        with pytest.raises(ValueError, match="hc1.*hc2_bm|hc2_bm.*hc1"):
            StackedDiD(vcov_type="hc4")

    def test_survey_design_plus_hc2_bm_rejected(self, staggered_data):
        """survey_design + non-hc1 vcov_type raises NotImplementedError.

        Reject order locked: fweight/aweight check fires first (per
        stacked_did.py:309), then the survey + non-hc1 vcov check.
        """
        from diff_diff.survey import SurveyDesign

        # Add a uniform pweight column to satisfy SurveyDesign
        data = staggered_data.copy()
        data["w"] = 1.0
        design = SurveyDesign(weights="w", weight_type="pweight")
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        with pytest.raises(NotImplementedError, match="survey TSL"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                survey_design=design,
            )

    def test_survey_design_plus_classical_rejected(self, staggered_data):
        """The classical reject fires at __init__ (before fit), so this test
        verifies the symmetric path — that a survey-design fit with the
        already-rejected classical vcov_type fails on the __init__ guard."""
        with pytest.raises(ValueError, match="clusters intrinsically"):
            StackedDiD(vcov_type="classical")

    def test_get_params_includes_vcov_type(self):
        params = StackedDiD(vcov_type="hc2_bm").get_params()
        assert "vcov_type" in params
        assert params["vcov_type"] == "hc2_bm"

    def test_set_params_updates_vcov_type(self):
        est = StackedDiD(vcov_type="hc1")
        est.set_params(vcov_type="hc2_bm")
        assert est.vcov_type == "hc2_bm"

    def test_set_params_revalidates_vcov_type(self):
        """Per local codex R4 P3: `set_params(vcov_type=...)` must re-validate
        via the same estimator-level guard as `__init__`, not silently accept
        any value and fail later in the linalg layer with a different message.
        """
        est = StackedDiD(vcov_type="hc1")
        # All three reject paths must fire from set_params too:
        with pytest.raises(ValueError, match="clusters intrinsically"):
            est.set_params(vcov_type="classical")
        assert est.vcov_type == "hc1"  # estimator state unchanged after reject
        with pytest.raises(ValueError, match="clusters intrinsically"):
            est.set_params(vcov_type="hc2")
        with pytest.raises(ValueError, match="conley"):
            est.set_params(vcov_type="conley")
        with pytest.raises(ValueError, match="hc1.*hc2_bm|hc2_bm.*hc1"):
            est.set_params(vcov_type="hc4")
        # Valid mutation still works
        est.set_params(vcov_type="hc2_bm")
        assert est.vcov_type == "hc2_bm"

    def test_results_carries_vcov_type(self, staggered_data):
        for vcov in ["hc1", "hc2_bm"]:
            est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type=vcov)
            res = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
            assert res.vcov_type == vcov

    def test_unit_subexp_cluster_plus_hc2_bm_finite(self, staggered_data):
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm", cluster="unit_subexp")
        res = est.fit(
            staggered_data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        assert np.isfinite(res.se) and res.se > 0
        assert res.vcov_type == "hc2_bm"

    def test_replicate_refit_smoke_with_default_hc1(self, staggered_data):
        """Replicate-weight survey + default hc1 fits cleanly through the
        _refit_stacked closure (which now uses explicit weights= per
        stacked_did.py post-PR). Smoke test only — variance correctness is
        covered by separate replicate-refit tests."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(2026)
        data = staggered_data.copy()
        # Generate 20 replicate weight columns (simulating jackknife/BRR-style)
        n_units = data["unit"].nunique()
        rep_w = rng.uniform(0.5, 1.5, size=(n_units, 20))
        rep_w_cols = [f"rep_w{i}" for i in range(20)]
        unit_to_rep = {u: rep_w[i] for i, u in enumerate(sorted(data["unit"].unique()))}
        for j, col in enumerate(rep_w_cols):
            data[col] = data["unit"].map(lambda u, j=j: unit_to_rep[u][j])
        data["w"] = 1.0
        design = SurveyDesign(
            weights="w",
            weight_type="pweight",
            replicate_weights=rep_w_cols,
            replicate_method="JK1",
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc1")
        res = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            survey_design=design,
        )
        assert np.isfinite(res.se) and res.se > 0

    def test_fit_clone_idempotent_on_vcov_type(self, staggered_data):
        """Per `feedback_fit_does_not_mutate_config`: fit, clone the
        estimator config via get_params/set_params, refit, assert SE
        bit-equal. Locks that fit() doesn't mutate self.vcov_type."""
        kwargs = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
        est_a = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        res_a = est_a.fit(staggered_data, **kwargs)
        # Clone via get_params
        est_b = StackedDiD(**est_a.get_params())
        assert est_a.vcov_type == est_b.vcov_type == "hc2_bm"
        res_b = est_b.fit(staggered_data, **kwargs)
        np.testing.assert_allclose(res_a.se, res_b.se, atol=1e-15)
        # And refitting est_a doesn't mutate its config
        _ = est_a.fit(staggered_data, **kwargs)
        assert est_a.vcov_type == "hc2_bm"

    def test_aweight_plus_hc2_bm_rejected_by_stacked_did_level_guard(self, staggered_data):
        """Per review MEDIUM #1: the existing fweight/aweight reject at
        stacked_did.py:309 fires BEFORE the new vcov_type=non-hc1 reject.
        This locks the order so a future refactor swapping the checks would
        silently change error messaging from 'Q-weight ratio semantics' to
        'survey TSL' on the same input.
        """
        from diff_diff.survey import SurveyDesign

        data = staggered_data.copy()
        data["w"] = 1.0
        design = SurveyDesign(weights="w", weight_type="aweight")
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        # Expect the Q-weight semantics error (stacked_did.py:309), NOT the
        # survey TSL vcov error. The match pattern checks for the Q-weight
        # phrasing specifically.
        with pytest.raises(ValueError, match="weight_type='aweight'.*Q-weight"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                survey_design=design,
            )

    def test_survey_design_plus_hc2_bm_rejected_unit_subexp_cluster(self, staggered_data):
        """Per review MEDIUM #2: the survey+non-hc1 reject must fire
        regardless of cluster level."""
        from diff_diff.survey import SurveyDesign

        data = staggered_data.copy()
        data["w"] = 1.0
        design = SurveyDesign(weights="w", weight_type="pweight")
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm", cluster="unit_subexp")
        with pytest.raises(NotImplementedError, match="survey TSL"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                survey_design=design,
            )

    def test_replicate_refit_coef_bit_equal_vs_bake_w_baseline(self, baseline_panel):
        """Per review Q2: pin the replicate-refit overall_att/SE on default
        hc1 + replicate-weight survey at atol=1e-13. Catches any float64
        multiplication-ordering drift introduced by the bake-Q-into-X ->
        explicit weights= switch in the _refit_stacked closure.

        Baseline panel + estimator config matches
        test_hc1_se_bit_equal_to_pre_pr_baseline. The replicate weights
        below are seeded deterministically — same seed must produce same
        result pre- and post-switch.
        """
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(20260521)
        data = baseline_panel.copy()
        n_units = data["unit"].nunique()
        rep_w = rng.uniform(0.5, 1.5, size=(n_units, 16))
        rep_w_cols = [f"rep_w{i}" for i in range(16)]
        unit_to_rep = {u: rep_w[i] for i, u in enumerate(sorted(data["unit"].unique()))}
        for j, col in enumerate(rep_w_cols):
            data[col] = data["unit"].map(lambda u, j=j: unit_to_rep[u][j])
        data["w"] = 1.0
        design = SurveyDesign(
            weights="w",
            weight_type="pweight",
            replicate_weights=rep_w_cols,
            replicate_method="JK1",
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc1")
        res = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            survey_design=design,
        )
        # Drift lock: hardcoded baseline values captured on the post-PR
        # branch with the explicit weights= pattern. If multiplication
        # ordering ever shifts (e.g., a future linalg refactor changes the
        # bake-w internals in solve_ols), this test catches it. Coef is
        # pinned at atol=1e-13 (per-replicate drift typically 1-2 ULPs);
        # the variance compounds the per-replicate drift through
        # compute_replicate_refit_variance's squared-difference
        # aggregation, so SE gets atol=1e-10. Both bands match
        # similar "no methodologically significant drift" tolerances
        # used elsewhere in the project.
        # The ATT is the full-sample fit (w = 1.0, default weighting), whose
        # Q-weights are untouched. The SE moved from 0.14697256517699428 when
        # the replicate refits began recomputing the corrective weights from
        # each replicate's own (non-uniform) weights; the new value is the one
        # TestStackedDiDCohortShares::test_replicate_refits_recompute_the_shares
        # justifies against independent per-replicate fits.
        BASELINE_ATT = 2.0807833161293945
        BASELINE_SE = 0.1447117933599559
        np.testing.assert_allclose(res.att, BASELINE_ATT, atol=1e-13)
        np.testing.assert_allclose(res.se, BASELINE_SE, atol=1e-10)

    def test_hc2_bm_uses_t_distribution_not_normal_on_well_conditioned_design(self, staggered_data):
        """Smoke-test on a well-conditioned design: hc2_bm CIs use t(BM DOF),
        not normal-theory z=1.96. End-to-end check that BM DOF threading
        from PR R1 fix is wired through to safe_inference."""
        from scipy.stats import norm

        z_975 = norm.ppf(0.975)  # ~1.96

        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        # No fallback warning on a well-conditioned design.
        fallback_warns = [w for w in caught if "Bell-McCaffrey contrast DOF" in str(w.message)]
        assert len(fallback_warns) == 0, (
            f"hc2_bm should not warn on a well-conditioned design; "
            f"got {len(fallback_warns)} warning(s)"
        )
        # CI half-width / SE > z_0.975 for at least one post-treatment effect
        # AND for the overall ATT (BM DOF inflates the critical value).
        any_t_dist_used = False
        for h in [0, 1, 2]:
            if h in res.event_study_effects:
                es = res.event_study_effects[h]
                if es["n_obs"] == 0 or es["se"] == 0:
                    continue
                half_width = es["conf_int"][1] - es["effect"]
                t_crit = half_width / es["se"]
                if t_crit > z_975 * 1.0001:
                    any_t_dist_used = True
                    break
        assert any_t_dist_used, (
            "hc2_bm CIs should use t(BM DOF), not normal — at least one "
            "post-treatment event_study CI half-width must exceed z_0.975 * SE"
        )
        overall_half_width = res.conf_int[1] - res.att
        overall_t_crit = overall_half_width / res.se
        assert overall_t_crit > z_975 * 1.0001

    def test_hc2_bm_rank_deficient_keeps_bm_dof_on_identified_contrasts(
        self, staggered_data, monkeypatch
    ):
        """Per local codex R2 P1 (and R5 P3 test-coverage clarification):
        when solve_ols's rank-deficient handler drops a column (returns
        NaN in coef[j] for that column), the SURVIVING delta_h coefficients
        should still get BM DOF (t-distribution). The dropped coefficient
        gets NaN inference; identified ones keep finite t-distribution CIs.

        Forces rank deficiency by monkeypatching solve_ols to return a coef
        vector with one NaN entry at a non-target column (specifically: the
        intercept column, which is NOT one of the event-study delta_h
        targets). This exercises the reduced-design code path at
        stacked_did.py:529-577 added in the R2 P1 fix.

        Verifies:
          1. The fit doesn't emit the fallback warning (BM DOF computed
             successfully on the reduced design).
          2. event_study_effects[h] for identified delta_h have finite
             inference using t-distribution (CI half-width > z_0.975 * SE).
          3. The dropped coefficient (intercept) gets NaN inference IF the
             estimator surfaces it — for StackedDiD it doesn't, since only
             delta_h coefficients are exposed via event_study_effects.
        """
        import importlib

        from scipy.stats import norm

        # Use importlib to get the MODULE (not the same-name function exported
        # at the diff_diff package level).
        sd_module = importlib.import_module("diff_diff.stacked_did")

        z_975 = norm.ppf(0.975)

        # Wrap solve_ols to inject NaN at the intercept column (index 0).
        # The intercept is NOT a target delta_h, so dropping it shouldn't
        # affect event-study identification — just exercises the
        # rank-deficient code path.
        _orig_solve_ols = sd_module.solve_ols

        def _fake_solve_ols(*args, **kwargs):
            coef, residuals, vcov = _orig_solve_ols(*args, **kwargs)
            # Inject NaN at intercept (column 0) to simulate rank deficiency
            coef_modified = coef.copy()
            coef_modified[0] = np.nan
            # Also NaN-out vcov row/col 0 to match what _expand_vcov_with_nan
            # would produce for a dropped column.
            if vcov is not None:
                vcov_modified = vcov.copy()
                vcov_modified[0, :] = np.nan
                vcov_modified[:, 0] = np.nan
            else:
                vcov_modified = vcov
            return coef_modified, residuals, vcov_modified

        monkeypatch.setattr(sd_module, "solve_ols", _fake_solve_ols)

        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        # The reduced-design code path should fire without emitting the
        # fallback warning (the helper succeeds on the kept-column subset).
        fallback_warns = [w for w in caught if "Bell-McCaffrey contrast DOF" in str(w.message)]
        assert len(fallback_warns) == 0, (
            f"Rank-deficient handling should subset to kept columns and "
            f"succeed; got {len(fallback_warns)} fallback warning(s): "
            f"{[str(w.message) for w in fallback_warns]}"
        )
        # All event-study delta_h coefficients (none dropped here) should
        # have finite t-distribution inference.
        for h, eff in res.event_study_effects.items():
            if h == -1:
                continue
            if eff["n_obs"] == 0 or eff["se"] == 0:
                continue
            assert np.isfinite(eff["effect"]), f"delta_{h} effect should survive"
            assert np.isfinite(eff["se"])
            assert np.isfinite(eff["t_stat"]), f"delta_{h} t_stat should be finite (t-dist)"
            half_width = eff["conf_int"][1] - eff["effect"]
            t_crit = half_width / eff["se"]
            assert t_crit > z_975 * 1.0001, (
                f"delta_{h} CI must use t(BM DOF) on rank-deficient design; " f"got t_crit={t_crit}"
            )
        # Overall ATT also keeps t-distribution
        if np.isfinite(res.se) and res.se > 0:
            overall_t_crit = (res.conf_int[1] - res.att) / res.se
            assert overall_t_crit > z_975 * 1.0001

    def test_hc2_bm_nan_dof_fails_closed_with_all_nan_inference(self, staggered_data, monkeypatch):
        """Per local codex R3 P1: when BM contrast DOF returns NaN (noise-
        floor guard fires from PR #475), StackedDiD must emit all-NaN
        inference fields on the affected contrast — NOT mixed finite-t /
        NaN-p (which would happen if NaN df flowed through safe_inference)
        and NOT normal-theory fallback (which would silently produce wrong
        small-sample CIs).

        Forces NaN DOF by monkeypatching `_compute_cr2_bm_contrast_dof` to
        return a NaN-only vector. Verifies that:
          - All event_study_effects entries on the hc2_bm path have NaN
            t_stat, p_value, conf_int.
          - overall_* fields are all NaN.
          - effect and se themselves remain finite (only inference is
            suppressed, mirroring the LinearRegression.get_inference
            pattern from PR #475 R7).
        """

        def _fake_contrast_dof(*args, **kwargs):
            # Match the m-dimensional output shape expected by callers.
            n_contrasts = args[3].shape[1] if len(args) >= 4 else kwargs["contrasts"].shape[1]
            return np.full(n_contrasts, np.nan)

        # Patch the import inside stacked_did's module namespace + the
        # source module (in case of dynamic re-import).
        from diff_diff import linalg as _linalg_mod

        monkeypatch.setattr(_linalg_mod, "_compute_cr2_bm_contrast_dof", _fake_contrast_dof)

        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        res = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        # All event_study_effects (except the ref period) should have NaN
        # inference but finite effect+se.
        for h, eff in res.event_study_effects.items():
            if h == -1:
                continue  # ref period: SE=0 by construction
            assert np.isfinite(eff["effect"]), f"effect at h={h} should be finite"
            assert np.isfinite(eff["se"]) and eff["se"] > 0, f"se at h={h} should be finite > 0"
            assert np.isnan(eff["t_stat"]), (
                f"t_stat at h={h} must be NaN when BM DOF NaN-guarded; "
                f"got {eff['t_stat']} (silent wrong inference)"
            )
            assert np.isnan(eff["p_value"]), f"p_value at h={h} must be NaN; got {eff['p_value']}"
            assert all(
                np.isnan(b) for b in eff["conf_int"]
            ), f"conf_int at h={h} must be all-NaN; got {eff['conf_int']}"
        # Overall ATT: same fail-closed expectation
        assert np.isfinite(res.att) and np.isfinite(res.se)
        assert np.isnan(res.t_stat), f"overall_t_stat must be NaN under NaN DOF; got {res.t_stat}"
        assert np.isnan(res.p_value)
        assert all(np.isnan(b) for b in res.conf_int)

    def test_hc2_bm_helper_raises_fails_closed_with_all_nan_inference(
        self, staggered_data, monkeypatch
    ):
        """Per local codex R3 P1: when `_compute_cr2_bm_contrast_dof` raises
        (e.g., genuine singularity on the identified design), the estimator
        must emit all-NaN inference rather than fall back to normal-theory
        CIs/p-values."""
        from diff_diff import linalg as _linalg_mod

        def _fake_raises(*args, **kwargs):
            raise np.linalg.LinAlgError("forced linalg failure for test")

        monkeypatch.setattr(_linalg_mod, "_compute_cr2_bm_contrast_dof", _fake_raises)

        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        # Warning fired (informational that DOF was unavailable)
        warning_msgs = [str(w.message) for w in caught]
        assert any(
            "Bell-McCaffrey contrast DOF" in m for m in warning_msgs
        ), f"Should warn on helper failure; got: {warning_msgs}"
        # Inference NaN-closed for all event-study + overall
        for h, eff in res.event_study_effects.items():
            if h == -1:
                continue
            assert np.isnan(eff["t_stat"]) and np.isnan(
                eff["p_value"]
            ), f"event_time h={h} inference must NaN-close, not fall back to normal"
        assert np.isnan(res.t_stat), "overall_t_stat must NaN-close on helper failure"
        assert np.isnan(res.p_value)

    def test_anticipation_plus_hc2_bm_threads_bm_dof(self, staggered_data):
        """Per local codex R7 P1: the BM-DOF code path uses anticipation in
        post-period selection (`h >= -self.anticipation` at stacked_did.py:
        546-548 and 754-758). Verify hc2_bm + anticipation=1 still threads
        BM DOF into both event-study and overall inference (CI half-width >
        z_0.975 * SE), the shifted reference period is honored, and the
        overall ATT contrast includes h=-1 (per anticipation extension)."""
        from scipy.stats import norm

        z_975 = norm.ppf(0.975)

        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm", anticipation=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        fallback_warns = [w for w in caught if "Bell-McCaffrey contrast DOF" in str(w.message)]
        assert (
            len(fallback_warns) == 0
        ), f"hc2_bm + anticipation should not warn; got {len(fallback_warns)}"
        # Reference period under anticipation=1 shifts to e=-2 (=-1-1)
        # so event_study_effects[-2] should have SE=0 (reference).
        assert -2 in res.event_study_effects
        assert res.event_study_effects[-2]["se"] == 0.0, (
            f"Reference period should shift to -2 under anticipation=1; "
            f"got SE={res.event_study_effects[-2]['se']}"
        )
        # Overall ATT: includes h=-1 per anticipation extension; verify
        # the overall CI uses t(BM DOF), not normal.
        if np.isfinite(res.se) and res.se > 0:
            overall_t_crit = (res.conf_int[1] - res.att) / res.se
            assert overall_t_crit > z_975 * 1.0001, (
                f"Overall ATT CI under anticipation=1 must use t(BM DOF); "
                f"got t_crit={overall_t_crit}"
            )
        # At least one event-study CI uses t-distribution
        any_t = False
        for h, eff in res.event_study_effects.items():
            if h == -2 or eff["n_obs"] == 0 or eff["se"] == 0:
                continue
            tc = (eff["conf_int"][1] - eff["effect"]) / eff["se"]
            if tc > z_975 * 1.0001:
                any_t = True
                break
        assert any_t, "hc2_bm + anticipation=1 event-study CIs should use t(BM DOF)"

    def test_population_weighting_plus_hc2_bm_finite_threads_bm_dof(self, staggered_data):
        """Per local codex R7 P1: weighting='population' uses a different
        Q-weight formula (`Q_sa = (Pop^D_a / Pop^D) / (N^C_a / N^C)` for
        controls per StackedDiD registry); verify that hc2_bm still threads
        BM DOF correctly when composed_weights changes formula."""
        from scipy.stats import norm

        z_975 = norm.ppf(0.975)

        # Need a population column. Add one (constant for simplicity).
        data = staggered_data.copy()
        data["pop"] = 1.0  # Equal populations ⇒ same as aggregate weighting

        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm", weighting="population")
        res = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            population="pop",
        )
        assert np.isfinite(res.att) and np.isfinite(res.se)
        assert res.se > 0
        # Verify BM DOF threaded (t-distribution used)
        if np.isfinite(res.se) and res.se > 0:
            overall_t_crit = (res.conf_int[1] - res.att) / res.se
            assert (
                overall_t_crit > z_975 * 1.0001
            ), "hc2_bm + weighting='population' overall CI must use t(BM DOF)"

    def test_sample_share_weighting_plus_hc2_bm_finite_threads_bm_dof(self, staggered_data):
        """Per local codex R7 P1: weighting='sample_share' uses a third
        Q-weight formula (`Q_sa = ((N^D_a + N^C_a) / (N^D + N^C)) / (N^C_a / N^C)`
        for controls per StackedDiD registry); verify hc2_bm threads BM DOF."""
        from scipy.stats import norm

        z_975 = norm.ppf(0.975)

        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm", weighting="sample_share")
        res = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert np.isfinite(res.att) and np.isfinite(res.se)
        assert res.se > 0
        if np.isfinite(res.se) and res.se > 0:
            overall_t_crit = (res.conf_int[1] - res.att) / res.se
            assert (
                overall_t_crit > z_975 * 1.0001
            ), "hc2_bm + weighting='sample_share' overall CI must use t(BM DOF)"

    def test_hc1_vs_hc2_bm_differ_under_anticipation(self, staggered_data):
        """Sanity check: hc1 and hc2_bm produce different SEs under
        anticipation=1 (proves the BM-DOF logic actually fires under
        non-default params, not just defaults)."""
        kwargs = dict(
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        res_hc1 = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc1", anticipation=1).fit(
            staggered_data, **kwargs
        )
        res_hc2bm = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm", anticipation=1).fit(
            staggered_data, **kwargs
        )
        # ATT identical (vcov-only change); SE differs on event-study
        np.testing.assert_allclose(res_hc1.att, res_hc2bm.att, atol=1e-13)
        hc1_es = {
            h: e["se"] for h, e in res_hc1.event_study_effects.items() if h != -2 and e["se"] > 0
        }
        hc2_es = {
            h: e["se"] for h, e in res_hc2bm.event_study_effects.items() if h != -2 and e["se"] > 0
        }
        diffs = [abs(hc1_es[h] - hc2_es[h]) for h in hc1_es if h in hc2_es]
        assert max(diffs) > 1e-6, (
            "Under anticipation=1, hc1 vs hc2_bm SEs must differ — "
            "leverage/DOF adjustment didn't fire"
        )

    def test_summary_renders_clustered_variance_label(self, staggered_data):
        """Per CI codex R2 P2: summary() must render the clustered CR1/CR2-BM
        label ('CR1 cluster-robust at unit, G=N' / 'CR2 Bell-McCaffrey
        cluster-robust at unit, G=N'), NOT the one-way label ('HC1
        heteroskedasticity-robust' / 'HC2 + Bell-McCaffrey DOF (one-way)').
        StackedDiD is intrinsically clustered."""
        kwargs = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
        for vcov, expected_substr in [
            ("hc1", "CR1 cluster-robust at unit"),
            ("hc2_bm", "CR2 Bell-McCaffrey cluster-robust at unit"),
        ]:
            est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type=vcov, cluster="unit")
            res = est.fit(staggered_data, **kwargs)
            s = res.summary()
            assert "Variance:" in s, f"summary() missing Variance: line for {vcov}"
            variance_lines = [line for line in s.split("\n") if "Variance:" in line]
            assert any(expected_substr in line for line in variance_lines), (
                f"summary() variance label for vcov_type={vcov} should contain "
                f"'{expected_substr}', got: {variance_lines}"
            )
            # MUST NOT render as one-way (would mislead users)
            assert not any(
                "(one-way" in line for line in variance_lines
            ), f"vcov_type={vcov} mislabeled as one-way: {variance_lines}"
            # G count present
            assert any(
                "G=" in line for line in variance_lines
            ), f"summary() should include cluster count G= for {vcov}: {variance_lines}"
        # Also test unit_subexp cluster level
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm", cluster="unit_subexp")
        res = est.fit(staggered_data, **kwargs)
        s = res.summary()
        variance_lines = [line for line in s.split("\n") if "Variance:" in line]
        assert any(
            "at unit_subexp" in line for line in variance_lines
        ), f"cluster='unit_subexp' should render in label: {variance_lines}"


# =============================================================================
# Covariate balancing (CBWSDID, Ustyuzhanin 2026)
# =============================================================================


def _balance_panel(
    seed=5,
    *,
    hetero_x=True,
    n_per=20,
    n_never=60,
    const_x=False,
    treated_x_mean=0.7,
    infeasible=False,
):
    """Staggered panel with a covariate x; treated cohorts optionally selected on x."""
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for ft, n in [(4, n_per), (5, n_per), (6, n_per), (np.inf, n_never)]:
        treated = np.isfinite(ft)
        if const_x:
            xm, xs = 1.0, 0.0
        elif infeasible and treated:
            xm, xs = 12.0, 0.1  # treated x far outside the control hull
        else:
            xm, xs = (treated_x_mean if (treated and hetero_x) else 0.0), 1.0
        for _ in range(n):
            uid += 1
            x = xm + xs * rng.normal()
            alpha = rng.normal()
            slope = 0.3 * x + rng.normal(0, 0.02)
            for t in range(1, 11):
                y = (
                    alpha
                    + slope * t
                    + rng.normal(0, 0.04)
                    + (2.0 if (treated and t >= ft) else 0.0)
                )
                rows.append((uid, t, ft if treated else 0, x, y))
    return pd.DataFrame(rows, columns=["unit", "time", "first_treat", "x", "y"])


def _cb_fit(df, **kw):
    params = dict(
        kappa_pre=2,
        kappa_post=2,
        weighting="aggregate",
        control_group="never_treated",
        cluster="unit",
    )
    params.update(
        {k: v for k, v in kw.items() if k in ("weighting", "balance", "cluster", "vcov_type")}
    )
    est = StackedDiD(**params)
    return est.fit(
        df,
        outcome="y",
        unit="unit",
        time="time",
        first_treat="first_treat",
        covariates=kw.get("covariates"),
        survey_design=kw.get("survey_design"),
        population=kw.get("population"),
    )


class TestStackedDiDCovariateBalance:
    """CBWSDID covariate-balancing path on StackedDiD."""

    # ---- validation + scope guards ----
    def test_invalid_balance_value_raises(self):
        with pytest.raises(ValueError, match="balance must be"):
            StackedDiD(balance="match")

    def test_set_params_revalidates_balance(self):
        est = StackedDiD()
        with pytest.raises(ValueError, match="balance must be"):
            est.set_params(balance="ipw")

    def test_balance_without_covariates_raises(self):
        df = _balance_panel()
        with pytest.raises(ValueError, match="requires a non-empty covariates"):
            _cb_fit(df, balance="entropy", covariates=None)

    def test_covariates_without_balance_raises(self):
        df = _balance_panel()
        with pytest.raises(ValueError, match="balance='none'"):
            _cb_fit(df, balance="none", covariates=["x"])

    def test_balance_with_population_weighting_raises(self):
        df = _balance_panel()
        df["pop"] = 1.0
        with pytest.raises(NotImplementedError, match="weighting='aggregate'"):
            _cb_fit(
                df, balance="entropy", covariates=["x"], weighting="population", population="pop"
            )

    def test_missing_covariate_column_raises(self):
        df = _balance_panel()
        with pytest.raises(ValueError, match="not found in data"):
            _cb_fit(df, balance="entropy", covariates=["nonexistent"])

    def test_infeasible_cohort_raises_named(self):
        df = _balance_panel(infeasible=True)
        with pytest.raises(ValueError, match="balancing failed for cohort"):
            _cb_fit(df, balance="entropy", covariates=["x"])

    # ---- correctness / properties ----
    def test_balance_constraint_satisfied(self):
        df = _balance_panel(seed=2)
        res = _cb_fit(df, balance="entropy", covariates=["x"])
        assert res.balance == "entropy"
        assert res.covariates == ["x"]
        assert res.balance_diagnostics is not None
        for a, d in res.balance_diagnostics.items():
            # entropy balancing achieves exact first-moment balance
            assert d["max_imbalance_post"] < 1e-7
            assert d["max_imbalance_pre"] > d["max_imbalance_post"]
        # stacked_data carries the raw design weights
        assert "_b_sa" in res.stacked_data.columns

    def test_constant_covariate_reduces_to_plain_stacked(self):
        # constant covariate -> uniform entropy weights -> identical to plain StackedDiD
        df = _balance_panel(seed=4, const_x=True)
        cb = _cb_fit(df, balance="entropy", covariates=["x"])
        plain = _cb_fit(df, balance="none")
        for h in plain.event_study_effects:
            if plain.event_study_effects[h].get("n_obs", 0) > 0:
                assert (
                    abs(
                        cb.event_study_effects[h]["effect"] - plain.event_study_effects[h]["effect"]
                    )
                    < 1e-6
                )

    def test_covariate_scale_invariance(self):
        # entropy balancing is invariant to covariate rescaling: x vs 2*x give the
        # same weights and hence the same estimate.
        df = _balance_panel(seed=6)
        df["x2"] = 2.0 * df["x"]
        a = _cb_fit(df, balance="entropy", covariates=["x"])
        b = _cb_fit(df, balance="entropy", covariates=["x2"])
        assert abs(a.att - b.att) < 1e-8

    def test_balance_on_plain_fit(self):
        # balancing works on a plain fit; since M-024 the surface (and the
        # hc2_bm BM-DOF machinery) no longer branches on fit-time
        # aggregate, so this is the same code path as every _cb_fit case.
        df = _balance_panel(seed=8)
        res = _cb_fit(df, balance="entropy", covariates=["x"])
        assert np.isfinite(res.att)
        assert res.balance == "entropy"
        assert res.event_study_effects is not None

    # ---- sklearn surface ----
    def test_get_params_includes_balance_and_clone(self):
        est = StackedDiD(balance="entropy")
        assert est.get_params()["balance"] == "entropy"
        clone = StackedDiD(**est.get_params())  # round-trip
        assert clone.balance == "entropy"

    def test_ragged_event_window_trims(self):
        # A treated unit missing one event-time row is trimmed from its
        # sub-experiment (complete-window rule), under balance="entropy" exactly
        # as under balance="none"; the fit equals the fit without that unit.
        df = _balance_panel(seed=10, const_x=True)
        u = int(df.loc[df["first_treat"] == 4, "unit"].iloc[0])
        drop = (df["unit"] == u) & (df["time"] == 5)  # a post-period row (keeps t=a-1 ref)
        dfr = df[~drop].reset_index(drop=True)
        dfu = df[df["unit"] != u].reset_index(drop=True)
        for balance, cov in (("entropy", ["x"]), ("none", None)):
            with pytest.warns(UserWarning, match="not observed at every"):
                ragged = _cb_fit(dfr, balance=balance, covariates=cov)
            removed = _cb_fit(dfu, balance=balance, covariates=cov)
            np.testing.assert_allclose(ragged.att, removed.att, rtol=1e-12)
            np.testing.assert_allclose(ragged.se, removed.se, rtol=1e-12)
            assert ragged.n_treated_units == removed.n_treated_units
        assert ragged.n_treated_units == df.loc[df["first_treat"] > 0, "unit"].nunique() - 1
        entropy = _cb_fit(dfr, balance="entropy", covariates=["x"])
        assert entropy.balance_diagnostics[4]["n_treated"] == 19  # 20 per cohort, one trimmed

    def test_zero_row_eligible_control_is_trimmed(self):
        # An eligible (never-treated) control with ZERO rows in a cohort's window is
        # not in that sub-experiment; with rows missing in every cohort's window it
        # is in none, and the balanced fit equals the fit without the unit.
        df = _balance_panel(seed=11, const_x=True)
        u = int(df.loc[df["first_treat"] == 0, "unit"].iloc[0])  # a never-treated unit
        drop = (df["unit"] == u) & (df["time"].between(2, 6))  # cohort-4 window [2,6]
        dfz = df[~drop].reset_index(drop=True)
        with pytest.warns(UserWarning, match="not observed at every"):
            ragged = _cb_fit(dfz, balance="entropy", covariates=["x"])
        removed = _cb_fit(
            df[df["unit"] != u].reset_index(drop=True), balance="entropy", covariates=["x"]
        )
        np.testing.assert_allclose(ragged.att, removed.att, rtol=1e-12)
        assert u not in set(ragged.stacked_data["unit"])
        for a, diag in ragged.balance_diagnostics.items():
            assert diag["n_control"] == removed.balance_diagnostics[a]["n_control"]

    def test_duplicate_unit_time_rows_raise(self):
        # Duplicate (unit, time) rows are rejected at fit(), whatever the balance
        # mode: a duplicated event time would be double-counted in its cell mean.
        df = _balance_panel(seed=12, const_x=True)
        u = int(df.loc[df["first_treat"] == 4, "unit"].iloc[0])
        dupe = df[(df["unit"] == u) & (df["time"] == 3)]  # event_time -1 for cohort 4
        df2 = df[~((df["unit"] == u) & (df["time"] == 5))]  # drop event_time +1
        dfd = pd.concat([df2, dupe], ignore_index=True)  # same row count, ragged coverage
        with pytest.raises(ValueError, match=r"sharing a \(unit, time\) pair"):
            _cb_fit(dfd, balance="entropy", covariates=["x"])
        with pytest.raises(ValueError, match=r"sharing a \(unit, time\) pair"):
            _cb_fit(dfd, balance="none")

    def test_string_covariate_raises_typeerror(self):
        df = _balance_panel(seed=13, const_x=True)
        with pytest.raises(TypeError, match="must be a list"):
            _cb_fit(df, balance="entropy", covariates="x")


class TestStackedDfConvention:
    """The three-value df_convention knob on StackedDiD (3.9 / M-127)."""

    @staticmethod
    def _panel(seed=9, n_units=60):
        from diff_diff import generate_staggered_data

        return generate_staggered_data(n_units=n_units, n_periods=10, seed=seed)

    _kw = dict(
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
    )

    @staticmethod
    def _t_p(t_stat, df):
        from scipy import stats

        return 2 * stats.t.sf(abs(t_stat), df)

    @staticmethod
    def _z_p(t_stat):
        from scipy import stats

        return 2 * stats.norm.sf(abs(t_stat))

    def test_default_is_t_g_minus_1_with_pinned_df(self):
        """The 4.0 default (M-130): the hc1 lane uses t(G-1). Assert the
        RESOLVED df value directly (a p-movement assertion is vacuous on
        stacked designs for the residual arm: that df is large, t ~= z)."""
        res = StackedDiD(kappa_pre=2, kappa_post=2).fit(self._panel(), **self._kw)
        assert res.inference_df == res.n_clusters - 1
        assert res.p_value == pytest.approx(self._t_p(res.t_stat, res.n_clusters - 1), rel=1e-12)
        finite = {v for v in res.event_study_df.values() if np.isfinite(v)}
        assert finite == {float(res.n_clusters - 1)}

    def test_residual_is_t_with_pinned_df(self):
        """The 3.9 defect fix, now behind explicit "residual": the hc1 lane
        uses t(pooled residual df), asserted through the RESOLVED df value
        (that df is large, so t ~= z and a p-movement assertion is vacuous)."""
        res = StackedDiD(kappa_pre=2, kappa_post=2, df_convention="residual").fit(
            self._panel(), **self._kw
        )
        dfres = res.inference_df
        assert dfres is not None and dfres > res.n_clusters
        assert res.p_value == pytest.approx(self._t_p(res.t_stat, dfres), rel=1e-12)
        finite = {v for v in res.event_study_df.values() if np.isfinite(v)}
        assert finite == {dfres}

    def test_small_n_default_moves_off_z(self):
        """On a SMALL stacked design the z -> t movement is measurable."""
        res = StackedDiD(kappa_pre=1, kappa_post=1).fit(self._panel(n_units=12), **self._kw)
        assert np.isfinite(res.p_value)
        assert res.p_value != self._z_p(res.t_stat)

    def test_cluster_matches_g_minus_1_effective_count(self):
        data = self._panel()
        r0 = StackedDiD(kappa_pre=2, kappa_post=2, df_convention="residual").fit(data, **self._kw)
        r1 = StackedDiD(kappa_pre=2, kappa_post=2, df_convention="cluster").fit(data, **self._kw)
        G = r1.n_clusters
        assert r1.att == r0.att and r1.se == r0.se
        assert r1.p_value == pytest.approx(self._t_p(r1.t_stat, G - 1), rel=1e-12)
        assert r1.inference_df == G - 1

    def test_cluster_g_uses_effective_count_not_raw_unique(self):
        """A cluster whose TOTAL composed weight is zero is excluded from the
        'cluster' G (effective_cluster_count) even though results.n_clusters
        keeps the raw unique count. Discriminated by zeroing one unit's rows
        via a zero survey weight is unsupported here, so instead simulate at
        the resolver level with the estimator's own ingredients: verify that
        on an ordinary fit the two counts AGREE (the documented divergence is
        unreachable without zero-weight clusters) and that the df equals
        effective-count-minus-1."""
        from diff_diff.linalg import effective_cluster_count

        data = self._panel()
        r1 = StackedDiD(kappa_pre=2, kappa_post=2, df_convention="cluster").fit(data, **self._kw)
        # Reconstruct the effective count from the stacked frame the fit stored
        stacked = r1.stacked_data
        cluster_ids = stacked["unit"].values
        # composed weights are strictly positive on this fixture -> equality
        g_eff = effective_cluster_count(cluster_ids, None)
        assert r1.inference_df == g_eff - 1 == r1.n_clusters - 1

    def test_normal_reproduces_pre39_z(self):
        data = self._panel()
        r0 = StackedDiD(kappa_pre=2, kappa_post=2).fit(data, **self._kw)
        rn = StackedDiD(kappa_pre=2, kappa_post=2, df_convention="normal").fit(data, **self._kw)
        assert rn.att == r0.att and rn.se == r0.se
        assert rn.p_value == pytest.approx(self._z_p(rn.t_stat), rel=1e-14)
        assert rn.inference_df is None

    def test_bm_dof_precedence_over_knob(self):
        data = self._panel()
        r_res = StackedDiD(
            kappa_pre=2, kappa_post=2, vcov_type="hc2_bm", df_convention="residual"
        ).fit(data, **self._kw)
        r_clu = StackedDiD(
            kappa_pre=2, kappa_post=2, vcov_type="hc2_bm", df_convention="cluster"
        ).fit(data, **self._kw)
        assert r_res.p_value == r_clu.p_value

    def test_survey_df_precedence_over_knob(self):
        """Survey design df wins under EVERY knob value: the three fits are
        bit-identical on inference and share the finite survey-df provenance
        (the knob is never consulted on surveyed fits)."""
        from diff_diff.survey import SurveyDesign

        data = self._panel()
        data["w"] = 1.0 + (data["unit"] % 5) * 0.2
        data["stratum"] = data["unit"] % 4
        design = SurveyDesign(weights="w", strata="stratum", psu="unit", weight_type="pweight")
        fits = {
            conv: StackedDiD(kappa_pre=2, kappa_post=2, df_convention=conv).fit(
                data, survey_design=design, **self._kw
            )
            for conv in ("residual", "cluster", "normal")
        }
        base = fits["residual"]
        assert base.inference_df is not None and np.isfinite(base.inference_df)
        for conv in ("cluster", "normal"):
            r = fits[conv]
            assert r.p_value == base.p_value
            assert r.conf_int == base.conf_int
            assert r.inference_df == base.inference_df

    def test_validation_and_transactional_set_params(self):
        with pytest.raises(ValueError, match="df_convention"):
            StackedDiD(df_convention="bogus")
        est = StackedDiD()
        with pytest.raises(ValueError, match="df_convention"):
            est.set_params(df_convention="bogus", alpha=0.10)
        assert est.df_convention == "cluster" and est.alpha == 0.05
        # Valid value + unknown key: the unknown-key rejection must also
        # leave the estimator fully unchanged (no partial application).
        before = est.get_params()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(df_convention="normal", nonexistent_param=1)
        assert est.get_params() == before
        assert StackedDiD(df_convention="normal").get_params()["df_convention"] == "normal"


class TestPlainFitPlotting:
    def test_plot_event_study_renders_on_plain_fit(self, staggered_data):
        """M-024: plot_event_study flips from TypeError to rendering on a
        plain fit - the duck-typed effects branch now matches because the
        surface is always populated."""
        matplotlib = pytest.importorskip("matplotlib")

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from diff_diff import plot_event_study

        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        ax = plot_event_study(results, show=False)
        assert ax is not None
        plt.close("all")


class TestReferenceSupportGuard:
    """M-024 follow-up (CI review): the omitted reference cell must be
    populated - a gapped panel where every retained cohort's calendar
    period a-1-anticipation is absent would otherwise rank-drop and
    silently re-normalize while the surface certifies delta_0 = 0 at the
    reference (and the container's reference_event_times provenance
    would then mislead HonestDiD/PreTrendsPower)."""

    @staticmethod
    def _gapped_panel(periods, cohorts, n_units=90, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(n_units):
            g = cohorts[u % len(cohorts)]
            for t in periods:
                d = 1 if (g and t >= g) else 0
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": u * 0.01 + 0.2 * t + 1.5 * d + rng.normal(0, 0.3),
                        "first_treat": g,
                    }
                )
        return pd.DataFrame(rows)

    def test_stack_wide_missing_reference_fails_closed(self):
        # Calendar period 4 absent; sole cohort a=5 has ref a-1=4. IC1 passes
        # (min/max only), but no unit can cover the window [3, 7], so the
        # complete-window rule leaves no sub-experiment and fit() raises,
        # naming the gap in the time grid (before the complete-window rule
        # the reference-support guard caught this panel).
        panel = self._gapped_panel([1, 2, 3, 5, 6, 7], [0, 5])
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        with pytest.raises(ValueError, match=r"No sub-experiment has both.*period\(s\) \[4\]"):
            est.fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_partial_reference_support_still_fits(self):
        # Cohort 5's own ref period (4) is absent for every unit, so its
        # window [3, 7] cannot be covered: the event is dropped with a
        # warning and listed in trimmed_groups. Cohort 7's window [5, 9] is
        # complete, so the fit proceeds on it and the normalization at e=-1
        # is real.
        panel = self._gapped_panel([1, 2, 3, 5, 6, 7, 8, 9], [0, 5, 7])
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = est.fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        assert not [w for w in caught if "Rank-deficient" in str(w.message)]
        dropped = [w for w in caught if "adoption event(s) were dropped" in str(w.message)]
        assert len(dropped) == 1 and "[4]" in str(dropped[0].message)
        assert [float(g) for g in res.groups] == [7.0]
        assert [float(g) for g in res.trimmed_groups] == [5.0]
        assert res.n_sub_experiments == 1
        assert res.event_study_effects is not None
        surf = res.aggregate("event_study")
        assert surf.reference_event_times == (-1,)
        assert np.isfinite(res.att)


class TestReferenceSupportGuardEffectiveWeights:
    """CI review R2: Equation 3 is Q-WEIGHTED WLS, so reference support
    means positive COMPOSED weight in both baseline cells - raw row
    presence is not enough."""

    def test_ragged_reference_cells_leave_no_sub_experiment(self):
        # Raw rows exist in both reference cells, but cohort 5's controls
        # all lack its reference period (4) and cohort 7's treated units
        # all lack theirs (6). Under the complete-window rule neither
        # sub-experiment keeps both a treated unit and a clean control, so
        # fit() raises (before that rule the aggregate Q-weights zeroed the
        # control reference cell and the reference-support guard fired).
        rng = np.random.default_rng(3)
        rows = []
        for u in range(120):
            kind = ["treatA", "ctrlX", "treatB", "ctrlY"][u % 4]
            g = {"treatA": 5, "ctrlX": 0, "treatB": 7, "ctrlY": 0}[kind]
            for t in range(1, 11):
                if kind == "ctrlX" and t == 4:
                    continue  # A's controls gapped at A's ref (cal 4)
                if kind == "treatB" and t == 6:
                    continue  # B's treated gapped at B's ref (cal 6)
                if kind == "ctrlY" and t == 4:
                    continue  # Y gapped at cal 4 too
                d = 1 if (g and t >= g) else 0
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": u * 0.01 + 0.2 * t + 1.5 * d + rng.normal(0, 0.3),
                        "first_treat": g,
                    }
                )
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        with pytest.raises(ValueError, match="No sub-experiment has both") as exc:
            est.fit(
                pd.DataFrame(rows),
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        assert "no clean control is observed at every event time" in str(exc.value)
        assert "no treated unit is observed at every event time" in str(exc.value)

    @staticmethod
    def _balanced_panel(pop_treated=100.0, seed=3):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(90):
            g = [0, 5, 7][u % 3]
            for t in range(1, 11):
                d = 1 if (g and t >= g) else 0
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": u * 0.01 + 0.2 * t + 1.5 * d + rng.normal(0, 0.3),
                        "first_treat": g,
                        "pop": pop_treated if g else 50.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_kappa_pre_zero_reference_outside_window_fails_closed(self):
        # kappa_pre = 0: the window [0, kappa_post] does not contain the
        # reference e = -1, so there is no reference cell at all.
        with pytest.raises(ValueError, match="positive weight.*fabricated") as exc:
            StackedDiD(kappa_pre=0, kappa_post=2).fit(
                self._balanced_panel(),
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        assert "kappa_pre = 0" in str(exc.value)

    def test_zero_treated_population_fails_closed(self):
        # weighting="population" with zero treated population zeroes every
        # control Q-weight, so the control reference cell has no mass.
        with pytest.raises(ValueError, match="positive weight.*fabricated") as exc:
            StackedDiD(kappa_pre=2, kappa_post=2, weighting="population").fit(
                self._balanced_panel(pop_treated=0.0),
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                population="pop",
            )
        assert "zero treated population" in str(exc.value)

    def test_survey_zeroed_reference_cell_fails_closed(self):
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(3)
        rows = []
        for u in range(90):
            g = [0, 5, 7][u % 3]
            for t in range(1, 11):
                d = 1 if (g and t >= g) else 0
                # every control carries zero pweight (unit-constant, as the
                # survey columns must be) -> no sub-experiment has control
                # mass, every corrective weight is zero, and the guard
                # reports the empty treated reference cell
                w = 0.0 if g == 0 else 1.0
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": u * 0.01 + 0.2 * t + 1.5 * d + rng.normal(0, 0.3),
                        "first_treat": g,
                        "w": w,
                    }
                )
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="positive weight.*fabricated"):
                est.fit(
                    pd.DataFrame(rows),
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    survey_design=SurveyDesign(weights="w"),
                )


# =============================================================================
# Complete event windows (ragged panels)
# =============================================================================

_CW_EFFECTS = {4: (2.0, 0.5), 6: (3.0, 0.25)}  # cohort -> (level, slope in event time)


def _cw_panel(noise=0.0, seed=0, n=90, n_periods=10, hetero=True):
    """Cohorts 4 and 6 plus never-treated; unit levels differ by cohort, and
    (with ``hetero``) so do the effects, so the trimmed aggregate ATT depends on
    which treated units each sub-experiment holds."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n):
        g = (0, 4, 6)[u % 3]
        level = rng.normal(0, 1) + {0: 0.0, 4: 0.8, 6: -0.5}[g]
        for t in range(1, n_periods + 1):
            e = t - g
            base, slope = _CW_EFFECTS[g] if (g and hetero) else (2.0, 0.5)
            tau = (base + slope * e) if (g and e >= 0) else 0.0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": level + 0.3 * t + 0.02 * t * t + tau + rng.normal(0, noise),
                    "first_treat": g,
                    # population differs by cohort, so population shares are not
                    # the unit shares
                    "pop": 100.0 + 10.0 * (u % 7) + (150.0 if g == 6 else 0.0),
                    "w": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _cw_ragged(name, d):
    """The ragged panels of the complete-window fix, by name."""
    nt = d["first_treat"] == 0
    c6 = d["first_treat"] == 6
    if name == "cohort6_missing_ref":
        out = d[~(c6 & (d["time"] == 5))]
    elif name == "cohort6_no_pre":
        out = d[~(c6 & (d["time"] < 6))]
    elif name == "half_cohort6_missing_ref":
        out = d[~(c6 & (d["time"] == 5) & (d["unit"] % 2 == 0))]
    elif name == "one_nt_obs_missing":
        u0 = d.loc[nt, "unit"].iloc[0]
        out = d[~((d["unit"] == u0) & (d["time"] == 5))]
    elif name == "random_5pct":
        out = d.drop(np.random.default_rng(1).choice(len(d), len(d) // 20, replace=False))
    elif name == "period7_absent":
        out = d[d["time"] != 7]
    elif name == "nt_missing_7_8":
        out = d[~(nt & d["time"].between(7, 8))]
    elif name == "period5_absent":
        out = d[d["time"] != 5]
    elif name == "nt_missing_4_8":
        out = d[~(nt & d["time"].between(4, 8))]
    else:  # pragma: no cover
        raise KeyError(name)
    return out.reset_index(drop=True)


def _cw_complete_units(d, a, kappa_pre=2, kappa_post=2, anticipation=0):
    """Units with an observation at EVERY period of event a's window, found
    independently of the estimator (wide pivot, no missing cell)."""
    window = list(range(a - kappa_pre - anticipation, a + kappa_post + 1))
    wide = (
        d[d["time"].isin(window)]
        .pivot(index="unit", columns="time", values="y")
        .reindex(columns=window)
    )
    return set(wide.index[wide.notna().all(axis=1)])


def _cw_cells(d, kappa_pre=2, kappa_post=2, anticipation=0):
    """Per realized event: (complete treated units, complete clean controls)
    under the default not-yet-treated control rule."""
    ft = d.groupby("unit")["first_treat"].first()
    cells = {}
    for a in sorted(g for g in ft.unique() if g > 0):
        complete = _cw_complete_units(d, a, kappa_pre, kappa_post, anticipation)
        treated = complete & set(ft.index[ft == a])
        control = complete & set(ft.index[(ft == 0) | (ft > a + kappa_post)])
        if treated and control:
            cells[a] = (treated, control)
    return cells


def _cw_truth(d, hetero=True, kappa_pre=2, kappa_post=2):
    """True trimmed aggregate ATT per event time: sum_a (N_a^D / N^D) tau_a(e),
    N_a^D counting the complete treated units of each realized event."""
    cells = _cw_cells(d, kappa_pre, kappa_post)
    n_total = sum(len(t) for t, _ in cells.values())
    truth = {}
    for e in range(-kappa_pre, kappa_post + 1):
        val = 0.0
        for a, (treated, _) in cells.items():
            base, slope = _CW_EFFECTS[a] if hetero else (2.0, 0.5)
            val += len(treated) / n_total * ((base + slope * e) if e >= 0 else 0.0)
        truth[e] = val
    return truth, cells


def _cw_fit(d, survey_design=None, population=None, **kw):
    kw.setdefault("kappa_pre", 2)
    kw.setdefault("kappa_post", 2)
    return StackedDiD(**kw).fit(
        d,
        outcome="y",
        unit="unit",
        time="time",
        first_treat="first_treat",
        survey_design=survey_design,
        population=population,
    )


def _cw_quiet_fit(d, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _cw_fit(d, **kw)


_CW_RECOVERABLE = [
    "cohort6_missing_ref",
    "cohort6_no_pre",
    "half_cohort6_missing_ref",
    "one_nt_obs_missing",
    "random_5pct",
    "period7_absent",
    "nt_missing_7_8",
]


class TestStackedDiDCompleteWindows:
    """A unit enters a sub-experiment only if it is observed at every event
    time of the window (Wing et al. 2024, Section 5.1). Before this rule a
    ragged panel returned shifted estimates with no warning: the regression
    contrasts event-time cell means, and unit levels leaked in."""

    @pytest.mark.parametrize("name", _CW_RECOVERABLE)
    def test_exact_recovery_on_ragged_panels(self, name):
        d = _cw_ragged(name, _cw_panel())
        truth, cells = _cw_truth(d)
        res = _cw_quiet_fit(d)
        assert [float(g) for g in res.groups] == [float(a) for a in cells]
        for e, val in truth.items():
            assert res.event_study_effects[e]["effect"] == pytest.approx(val, abs=1e-9), (name, e)
        assert res.att == pytest.approx(np.mean([truth[e] for e in (0, 1, 2)]), abs=1e-9)

    def test_balanced_panel_recovers_and_warns_nothing(self):
        d = _cw_panel()
        truth, cells = _cw_truth(d)
        assert sorted(cells) == [4, 6]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = _cw_fit(d)
        for e, val in truth.items():
            assert res.event_study_effects[e]["effect"] == pytest.approx(val, abs=1e-9)
        assert res.trimmed_groups == []

    def test_equals_manual_aggregate_of_cohort_contrasts(self):
        # Independent oracle (paper Eq. 2 / footnote 5): per-cohort DID
        # contrasts from plain means over the complete units, weighted by the
        # complete treated-unit shares. No regression, no Q-weights.
        d = _cw_ragged("random_5pct", _cw_panel(noise=0.5, seed=4))
        cells = _cw_cells(d)
        assert len(cells) == 2
        n_total = sum(len(t) for t, _ in cells.values())
        res = _cw_quiet_fit(d)
        for e in (-2, 0, 1, 2):
            manual = 0.0
            for a, (treated, control) in cells.items():

                def _mean(units, t):
                    rows = d[d["unit"].isin(units) & (d["time"] == t)]
                    assert len(rows) == len(units)
                    return rows["y"].mean()

                did = (_mean(treated, a + e) - _mean(treated, a - 1)) - (
                    _mean(control, a + e) - _mean(control, a - 1)
                )
                manual += len(treated) / n_total * did
            np.testing.assert_allclose(res.event_study_effects[e]["effect"], manual, rtol=1e-10)

    def test_equals_the_fit_without_the_incomplete_units(self):
        # Each incomplete unit here is incomplete in EVERY sub-experiment it
        # enters (a treated unit sits in one; the never-treated unit misses a
        # period common to both windows), so removing the units outright must
        # give the same fit.
        d = _cw_panel(noise=0.5, seed=7)
        c6_units = sorted(d.loc[d["first_treat"] == 6, "unit"].unique())[::3]
        nt_unit = d.loc[d["first_treat"] == 0, "unit"].iloc[0]
        gone = (d["unit"].isin(c6_units) & (d["time"] == 7)) | (
            (d["unit"] == nt_unit) & (d["time"] == 5)
        )
        with pytest.warns(UserWarning, match="not observed at every"):
            ragged = _cw_fit(d[~gone].reset_index(drop=True))
        removed = _cw_fit(d[~d["unit"].isin(list(c6_units) + [nt_unit])].reset_index(drop=True))
        np.testing.assert_allclose(ragged.att, removed.att, rtol=1e-12)
        np.testing.assert_allclose(ragged.se, removed.se, rtol=1e-12)
        np.testing.assert_allclose(ragged.event_study_vcov, removed.event_study_vcov, rtol=1e-12)
        for e, row in removed.event_study_effects.items():
            np.testing.assert_allclose(
                ragged.event_study_effects[e]["effect"], row["effect"], rtol=1e-12
            )
            np.testing.assert_allclose(ragged.event_study_effects[e]["se"], row["se"], rtol=1e-12)
        assert ragged.n_treated_units == removed.n_treated_units
        assert ragged.n_control_units == removed.n_control_units
        assert ragged.n_stacked_obs == removed.n_stacked_obs

    def test_unit_is_trimmed_per_sub_experiment(self):
        # Period 2 lies in cohort 4's window [2, 6] only: the unit leaves that
        # sub-experiment and stays, with all five rows, in cohort 6's.
        d = _cw_panel(noise=0.3)
        nt_unit = d.loc[d["first_treat"] == 0, "unit"].iloc[0]
        res = _cw_quiet_fit(d[~((d["unit"] == nt_unit) & (d["time"] == 2))].reset_index(drop=True))
        sd = res.stacked_data
        assert nt_unit not in set(sd.loc[sd["_sub_exp"] == 4, "unit"])
        kept = sd[(sd["_sub_exp"] == 6) & (sd["unit"] == nt_unit)]
        assert sorted(kept["_event_time"]) == [-2, -1, 0, 1, 2]

    def test_q_weights_are_the_unit_count_formula(self):
        d = _cw_ragged("random_5pct", _cw_panel(noise=0.3))
        cells = _cw_cells(d)
        n_d = sum(len(t) for t, _ in cells.values())
        n_c = sum(len(c) for _, c in cells.values())
        sd = _cw_quiet_fit(d).stacked_data
        assert (sd.loc[sd["_D_sa"] == 1, "_Q_weight"] == 1.0).all()
        for a, (treated, control) in cells.items():
            q = sd.loc[(sd["_sub_exp"] == a) & (sd["_D_sa"] == 0), "_Q_weight"]
            assert len(q) == 5 * len(control)
            np.testing.assert_allclose(
                q.to_numpy(), (len(treated) / n_d) / (len(control) / n_c), rtol=1e-13
            )

    def test_trim_warning_reports_the_counts(self):
        d = _cw_ragged("half_cohort6_missing_ref", _cw_panel())
        with pytest.warns(UserWarning) as rec:
            res = _cw_fit(d)
        msgs = [str(w.message) for w in rec]
        assert len(msgs) == 1
        assert "15 treated and 0 control unit-by-sub-experiment entries" in msgs[0]
        assert "15 of 30 treated, 30 of 30 control" in msgs[0]
        assert "trimmed aggregate ATT over the units observed across their whole window" in msgs[0]
        assert res.n_treated_units == 45 and res.n_control_units == 30

    @pytest.mark.parametrize(
        "name, reason",
        [
            ("cohort6_missing_ref", "no treated unit is observed at every event time (0 of 30)"),
            ("nt_missing_7_8", "no clean control is observed at every event time (0 of 30)"),
            ("period7_absent", "calendar period(s) [7] of its window are not observed"),
        ],
    )
    def test_event_without_complete_units_is_dropped(self, name, reason):
        d = _cw_ragged(name, _cw_panel(noise=0.3))
        with pytest.warns(UserWarning) as rec:
            res = _cw_fit(d)
        dropped = [
            str(w.message) for w in rec if "adoption event(s) were dropped" in str(w.message)
        ]
        assert len(dropped) == 1 and reason in dropped[0]
        assert [float(g) for g in res.groups] == [4.0]
        assert [float(g) for g in res.trimmed_groups] == [6.0]
        assert res.n_sub_experiments == 1
        assert set(res.stacked_data["_sub_exp"]) == {4}
        assert np.isfinite(res.att) and np.isfinite(res.se)

    def test_no_event_left_raises(self):
        d = _cw_panel()
        with pytest.raises(ValueError, match="No sub-experiment has both") as exc:
            _cw_fit(_cw_ragged("period5_absent", d))
        assert "period(s) [5] are not observed for any unit" in str(exc.value)
        with pytest.raises(ValueError, match="No sub-experiment has both") as exc:
            _cw_fit(_cw_ragged("nt_missing_4_8", d))
        assert "no clean control is observed at every event time (0 of 30)" in str(exc.value)
        assert "time grid has gaps" not in str(exc.value)

    def test_biennial_grid_raises_naming_the_missing_periods(self):
        d = _cw_panel()
        with pytest.raises(ValueError, match="No sub-experiment has both") as exc:
            _cw_fit(d[d["time"] % 2 == 0].reset_index(drop=True))
        assert "period(s) [3, 5, 7, 9] are not observed for any unit" in str(exc.value)
        assert "re-index time to consecutive integers" in str(exc.value)

    def test_datetime_coded_time_reports_the_span(self):
        # A datetime column coerces to epoch integers a day apart by ~1e11
        # or more, so every window misses. The message must describe the
        # span, not enumerate the missing integers.
        d = _cw_panel(n=30)
        dates = pd.Series(pd.date_range("2020-01-01", periods=10, freq="D"))
        epoch = pd.to_numeric(dates)
        d["time"] = d["time"].map(dict(zip(range(1, 11), dates)))
        d["first_treat"] = d["first_treat"].map(
            {0: 0, 4: int(epoch.iloc[3]), 6: int(epoch.iloc[5])}
        )
        with pytest.raises(ValueError, match="No sub-experiment has both") as exc:
            _cw_fit(d)
        assert "holds only 10 distinct periods" in str(exc.value)

    def test_input_validation(self):
        d = _cw_panel(n=30)
        dup = pd.concat([d, d.iloc[[5]]], ignore_index=True)
        with pytest.raises(ValueError, match=r"sharing a \(unit, time\) pair"):
            _cw_fit(dup)
        nan_time = d.astype({"time": float})
        nan_time.loc[3, "time"] = np.nan
        with pytest.raises(ValueError, match="missing or non-finite"):
            _cw_fit(nan_time)
        frac = d.astype({"time": float})
        frac.loc[3, "time"] = 4.5
        with pytest.raises(ValueError, match="must be integer-valued"):
            _cw_fit(frac)
        frac_ft = d.astype({"first_treat": float})
        frac_ft.loc[frac_ft["first_treat"] == 6, "first_treat"] = 5.5
        with pytest.raises(ValueError, match="integer adoption periods"):
            _cw_fit(frac_ft)
        with pytest.raises(TypeError):
            _cw_fit(d, kappa_pre=1.5)

    def test_never_treated_encodings_pass_validation(self):
        d = _cw_panel(n=30, noise=0.2)
        base = _cw_fit(d)
        as_inf = d.astype({"first_treat": float})
        as_inf.loc[as_inf["first_treat"] == 0, "first_treat"] = np.inf
        np.testing.assert_allclose(_cw_fit(as_inf).att, base.att, rtol=1e-12)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"control_group": "strict"},
            {"control_group": "never_treated"},
            {"vcov_type": "hc2_bm"},
            {"cluster": "unit_subexp"},
        ],
        ids=lambda k: "-".join(f"{a}={b}" for a, b in k.items()),
    )
    def test_exact_recovery_across_configurations(self, kwargs):
        # Homogeneous effects, so the truth is 2 + 0.5 e whatever the controls.
        d = _cw_ragged("random_5pct", _cw_panel(hetero=False))
        res = _cw_quiet_fit(d, **kwargs)
        assert len(res.groups) == 2
        for e in (-2, 0, 1, 2):
            want = (2.0 + 0.5 * e) if e >= 0 else 0.0
            assert res.event_study_effects[e]["effect"] == pytest.approx(want, abs=1e-9)

    def test_exact_recovery_with_anticipation(self):
        # anticipation=1 widens the window to [-3, 2] and moves the reference
        # to e = -2; the trim uses the widened window.
        d = _cw_ragged("random_5pct", _cw_panel(hetero=False))
        res = _cw_quiet_fit(d, anticipation=1)
        sd = res.stacked_data
        assert (sd.groupby(["_sub_exp", "unit"]).size() == 6).all()
        for e in (-3, -1, 0, 1, 2):
            want = (2.0 + 0.5 * e) if e >= 0 else 0.0
            assert res.event_study_effects[e]["effect"] == pytest.approx(want, abs=1e-9)

    def test_random_missingness_never_returns_a_wrong_number(self):
        n_fit = 0
        for seed in range(30):
            rng = np.random.default_rng(100 + seed)
            d = _cw_panel(seed=seed)
            frac = rng.uniform(0.02, 0.12)
            d = d.drop(rng.choice(len(d), int(frac * len(d)), replace=False)).reset_index(drop=True)
            truth, cells = _cw_truth(d)
            if not cells:
                with pytest.raises(ValueError, match="No sub-experiment has both"):
                    _cw_quiet_fit(d)
                continue
            res = _cw_quiet_fit(d)
            n_fit += 1
            assert [float(g) for g in res.groups] == [float(a) for a in cells], seed
            for e, val in truth.items():
                assert res.event_study_effects[e]["effect"] == pytest.approx(val, abs=1e-9), seed
        assert n_fit >= 25


class TestStackedDiDCompleteWindowsSurvey:
    """Under survey_design= the unit-level trim is refused: removing units
    would drop their PSUs and strata from the variance and df_survey."""

    @staticmethod
    def _designs(d):
        from diff_diff.survey import SurveyDesign

        d = d.copy()
        d["psu"] = d["unit"]
        units = sorted(d["unit"].unique())
        cols = []
        n_rep = 10
        for r in range(n_rep):
            deleted = set(units[r::n_rep])
            col = f"rep_{r}"
            d[col] = np.where(d["unit"].isin(deleted), 0.0, d["w"] * n_rep / (n_rep - 1))
            cols.append(col)
        return d, {
            "tsl": SurveyDesign(weights="w", psu="psu"),
            "replicate": SurveyDesign(weights="w", replicate_weights=cols, replicate_method="JK1"),
        }

    @pytest.mark.parametrize("design", ["tsl", "replicate"])
    @pytest.mark.parametrize(
        "name", ["half_cohort6_missing_ref", "cohort6_missing_ref", "nt_missing_4_8"]
    )
    def test_ragged_panel_is_refused(self, design, name):
        d, designs = self._designs(_cw_ragged(name, _cw_panel(noise=0.3)))
        with pytest.raises(NotImplementedError, match="survey_design=") as exc:
            _cw_fit(d, survey_design=designs[design])
        assert "not domain estimation" in str(exc.value)

    @pytest.mark.parametrize("design", ["tsl", "replicate"])
    def test_balanced_panel_fits(self, design):
        d, designs = self._designs(_cw_panel(noise=0.3))
        res = _cw_quiet_fit(d, survey_design=designs[design])
        plain = _cw_fit(_cw_panel(noise=0.3))
        np.testing.assert_allclose(res.att, plain.att, rtol=1e-10)  # uniform weights
        assert np.isfinite(res.se) and res.se > 0
        assert [float(g) for g in res.groups] == [4.0, 6.0]

    @pytest.mark.parametrize("design", ["tsl", "replicate"])
    def test_dropped_event_is_refused(self, design):
        # Period 7 is missing for EVERY unit, so cohort 6 cannot be realized.
        # Dropping it would take the cohort's PSUs out of the design (60 of
        # 90 PSUs left), so the survey fit refuses instead of proceeding on
        # cohort 4 as the unweighted fit does.
        d, designs = self._designs(_cw_ragged("period7_absent", _cw_panel(noise=0.3)))
        with pytest.raises(NotImplementedError, match="cannot drop adoption event") as exc:
            _cw_fit(d, survey_design=designs[design])
        assert "calendar period(s) [7]" in str(exc.value)
        assert _cw_quiet_fit(d).n_sub_experiments == 1  # the unweighted fit proceeds

    @pytest.mark.parametrize("col", ["w", "psu"])
    def test_survey_columns_must_be_constant_within_unit(self, col):
        # A weight that changes over a unit's rows changes each event-time
        # cell's weighted composition just as a missing row does: positive
        # time-varying weights returned 2.6174 for a true 2.5 on this panel.
        d, designs = self._designs(_cw_panel(hetero=False))
        half = (d["first_treat"] == 6) & (d["unit"] % 2 == 0)
        if col == "w":
            d.loc[half & (d["time"] < 6), "w"] = 1.5
            d.loc[half & (d["time"] >= 6), "w"] = 0.5
        else:
            d.loc[half & (d["time"] >= 6), "psu"] = -1
        with pytest.raises(ValueError, match=f"Survey column '{col}' varies within units"):
            _cw_fit(d, survey_design=designs["tsl"])

    def test_zero_weight_on_one_row_is_rejected(self):
        d, designs = self._designs(_cw_panel(hetero=False))
        d.loc[(d["first_treat"] == 6) & (d["time"] == 5), "w"] = 0.0
        with pytest.raises(ValueError, match="varies within units"):
            _cw_fit(d, survey_design=designs["tsl"])

    def test_gapped_grid_diagnosis_takes_precedence(self):
        d, designs = self._designs(_cw_panel(noise=0.3))
        with pytest.raises(ValueError, match="No sub-experiment has both"):
            _cw_fit(d[d["time"] % 2 == 0].reset_index(drop=True), survey_design=designs["tsl"])


# =============================================================================
# Cohort shares: both arms aggregate sub-experiments with the same shares
# =============================================================================

_CW_SCHEMES = ["aggregate", "population", "sample_share"]


def _cw_unequal(d):
    """Balanced panel with unequal cohort sizes: every second cohort-6 unit removed."""
    return d[~((d["first_treat"] == 6) & (d["unit"] % 2 == 0))].reset_index(drop=True)


def _cw_shares(d, scheme, weight_col=None, kappa_post=2):
    """Target cohort shares pi_a of a weighting scheme, computed from the input
    panel independently of the estimator, plus each event's complete unit sets
    and the unit weights. Events without positive treated and control mass, or
    with a zero target, are left out."""
    w = (
        d.groupby("unit")[weight_col].first()
        if weight_col
        else pd.Series(1.0, index=sorted(d["unit"].unique()))
    )
    pop = d.groupby("unit")["pop"].first()
    cells = {
        a: (sorted(t), sorted(c))
        for a, (t, c) in _cw_cells(d, kappa_post=kappa_post).items()
        if w[sorted(t)].sum() > 0 and w[sorted(c)].sum() > 0
    }
    if scheme == "aggregate":
        raw = {a: w[t].sum() for a, (t, c) in cells.items()}
    elif scheme == "population":
        raw = {a: (w[t] * pop[t]).sum() for a, (t, c) in cells.items()}
    else:
        raw = {a: w[t].sum() + w[c].sum() for a, (t, c) in cells.items()}
    raw = {a: v for a, v in raw.items() if v > 0}
    total = sum(raw.values())
    return {a: v / total for a, v in raw.items()}, cells, w


def _cw_share_truth(shares, event_times=(-2, 0, 1, 2)):
    return {
        e: sum(
            share * ((_CW_EFFECTS[a][0] + _CW_EFFECTS[a][1] * e) if e >= 0 else 0.0)
            for a, share in shares.items()
        )
        for e in event_times
    }


def _cw_manual_aggregate(d, shares, cells, w, e):
    """sum_a pi_a DID_{a,e} from (survey-weighted) cell means: no regression."""

    def _mean(units, t):
        rows = d[d["unit"].isin(units) & (d["time"] == t)].set_index("unit")["y"]
        assert len(rows) == len(units)
        return float(np.average(rows[units], weights=w[units]))

    total = 0.0
    for a, share in shares.items():
        treated = [u for u in cells[a][0] if w[u] > 0]
        control = [u for u in cells[a][1] if w[u] > 0]
        did = (_mean(treated, a + e) - _mean(treated, a - 1)) - (
            _mean(control, a + e) - _mean(control, a - 1)
        )
        total += share * did
    return total


def _cw_weighted(d, kind, seed=5):
    """Unit-constant survey weights in column ``w``."""
    d = d.copy()
    ft = d.groupby("unit")["first_treat"].first()
    if kind == "by_cohort":
        unit_w = ft.map({4: 1.5, 6: 0.5, 0: 1.0})
    else:  # random
        rng = np.random.default_rng(seed)
        unit_w = pd.Series(rng.uniform(0.2, 3.0, len(ft)), index=ft.index)
    d["w"] = d["unit"].map(unit_w)
    return d


def _cw_survey():
    from diff_diff.survey import SurveyDesign

    return SurveyDesign(weights="w")


class TestStackedDiDCohortShares:
    """Wing et al. (2024, Section 5.3): the stacked regression recovers a convex
    combination of sub-experiment ATTs only if the treated arm and the control
    arm aggregate sub-experiments with the SAME shares. Before the weights were
    unified, ``population`` and ``sample_share`` set the treated weight to 1
    (the paper's Table 1 has a treatment weight for both), and survey fits used
    raw unit counts for the corrective weights; untreated trends then failed to
    cancel on every panel with a nonlinear time trend."""

    @pytest.mark.parametrize("scheme", _CW_SCHEMES)
    @pytest.mark.parametrize("panel", ["unequal", "random_5pct"])
    def test_truth_recovery_under_every_scheme(self, scheme, panel):
        d = _cw_unequal(_cw_panel()) if panel == "unequal" else _cw_ragged(panel, _cw_panel())
        shares, _, _ = _cw_shares(d, scheme)
        assert len(shares) == 2
        truth = _cw_share_truth(shares)
        res = _cw_quiet_fit(
            d, weighting=scheme, population="pop" if scheme == "population" else None
        )
        for e, val in truth.items():
            assert res.event_study_effects[e]["effect"] == pytest.approx(val, abs=1e-9), e

    @pytest.mark.parametrize("scheme", _CW_SCHEMES)
    def test_q_weights_are_table_1(self, scheme):
        d = _cw_unequal(_cw_panel(noise=0.3))
        shares, cells, _ = _cw_shares(d, scheme)
        n_d = sum(len(t) for t, _ in cells.values())
        n_c = sum(len(c) for _, c in cells.values())
        sd = _cw_fit(
            d, weighting=scheme, population="pop" if scheme == "population" else None
        ).stacked_data
        for a, (treated, control) in cells.items():
            in_a = sd["_sub_exp"] == a
            np.testing.assert_allclose(
                sd.loc[in_a & (sd["_D_sa"] == 1), "_Q_weight"].to_numpy(),
                shares[a] / (len(treated) / n_d),
                rtol=1e-13,
            )
            np.testing.assert_allclose(
                sd.loc[in_a & (sd["_D_sa"] == 0), "_Q_weight"].to_numpy(),
                shares[a] / (len(control) / n_c),
                rtol=1e-13,
            )
        if scheme == "aggregate":
            assert (sd.loc[sd["_D_sa"] == 1, "_Q_weight"] == 1.0).all()
        else:
            assert sd.loc[sd["_D_sa"] == 1, "_Q_weight"].nunique() == 2  # not all 1

    @pytest.mark.parametrize("scheme", _CW_SCHEMES)
    def test_equals_manual_aggregate_under_every_scheme(self, scheme):
        d = _cw_ragged("random_5pct", _cw_unequal(_cw_panel(noise=0.5, seed=4)))
        shares, cells, w = _cw_shares(d, scheme)
        res = _cw_quiet_fit(
            d, weighting=scheme, population="pop" if scheme == "population" else None
        )
        for e in (-2, 0, 1, 2):
            np.testing.assert_allclose(
                res.event_study_effects[e]["effect"],
                _cw_manual_aggregate(d, shares, cells, w, e),
                rtol=1e-10,
            )

    @pytest.mark.parametrize("scheme", _CW_SCHEMES)
    @pytest.mark.parametrize("kind", ["by_cohort", "random"])
    def test_survey_weighted_truth_recovery(self, scheme, kind):
        # Survey-weighted cohort shares on both arms. With raw-count corrective
        # weights the by-cohort case returned an ATT of 2.46 for a true 2.50
        # (homogeneous effects) on the balanced panel.
        d = _cw_weighted(_cw_panel(), kind)
        shares, _, _ = _cw_shares(d, scheme, weight_col="w")
        truth = _cw_share_truth(shares)
        res = _cw_quiet_fit(
            d,
            weighting=scheme,
            population="pop" if scheme == "population" else None,
            survey_design=_cw_survey(),
        )
        for e, val in truth.items():
            assert res.event_study_effects[e]["effect"] == pytest.approx(val, abs=1e-9), e

    @pytest.mark.parametrize("scheme", _CW_SCHEMES)
    def test_survey_fit_equals_weighted_manual_aggregate(self, scheme):
        d = _cw_weighted(_cw_unequal(_cw_panel(noise=0.5, seed=4)), "random")
        shares, cells, w = _cw_shares(d, scheme, weight_col="w")
        res = _cw_quiet_fit(
            d,
            weighting=scheme,
            population="pop" if scheme == "population" else None,
            survey_design=_cw_survey(),
        )
        for e in (-2, 0, 1, 2):
            np.testing.assert_allclose(
                res.event_study_effects[e]["effect"],
                _cw_manual_aggregate(d, shares, cells, w, e),
                rtol=1e-10,
            )

    def test_survey_weights_are_scale_invariant(self):
        d = _cw_weighted(_cw_panel(noise=0.4), "random")
        base = _cw_quiet_fit(d, survey_design=_cw_survey())
        scaled = _cw_quiet_fit(d.assign(w=d["w"] * 7.0), survey_design=_cw_survey())
        np.testing.assert_allclose(scaled.att, base.att, rtol=1e-12)
        np.testing.assert_allclose(scaled.se, base.se, rtol=1e-12)

    def test_zero_weight_units_define_a_subpopulation(self):
        # Half of cohort 6 at weight 0: the estimate is the truth over the
        # positive-weight units (raw-count corrective weights returned 2.4733
        # for a true 2.50 on the homogeneous-effects version of this panel).
        d = _cw_weighted(_cw_panel(), "random")
        d.loc[(d["first_treat"] == 6) & (d["unit"] % 2 == 0), "w"] = 0.0
        shares, _, _ = _cw_shares(d, "aggregate", weight_col="w")
        truth = _cw_share_truth(shares)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = _cw_fit(d, survey_design=_cw_survey())
        # Both cohorts still carry mass, so no sub-experiment is reported empty.
        assert not [w for w in caught if "carry zero weight" in str(w.message)]
        for e, val in truth.items():
            assert res.event_study_effects[e]["effect"] == pytest.approx(val, abs=1e-9)

    def test_cohort_at_zero_weight_contributes_nothing(self):
        d = _cw_weighted(_cw_panel(), "by_cohort")
        d.loc[d["first_treat"] == 6, "w"] = 0.0
        with pytest.warns(UserWarning, match="carry zero weight") as rec:
            res = _cw_fit(d, survey_design=_cw_survey(), control_group="never_treated")
        msg = next(str(w.message) for w in rec if "carry zero weight" in str(w.message))
        assert "event 6: no treated unit with positive weight" in msg
        truth = _cw_share_truth({4: 1.0})
        for e, val in truth.items():
            assert res.event_study_effects[e]["effect"] == pytest.approx(val, abs=1e-9)
        # The rows stay in the stack (the design is unchanged), at zero weight.
        sd = res.stacked_data
        assert set(sd["_sub_exp"]) == {4, 6}
        assert (sd.loc[sd["_sub_exp"] == 6, "_Q_weight"] == 0.0).all()
        assert [float(g) for g in res.groups] == [4.0, 6.0]

    def test_event_without_control_mass_is_left_out_of_the_shares(self):
        # kappa_post=1: event 6's clean controls are the never-treated units
        # only, event 4's also include cohort 6 (not yet treated through 5).
        # With every never-treated unit at weight 0, event 6 has no control
        # mass; its TREATED mass must not enter the totals event 4 is scaled by.
        d = _cw_weighted(_cw_panel(), "random")
        d.loc[d["first_treat"] == 0, "w"] = 0.0
        with pytest.warns(UserWarning, match="carry zero weight") as rec:
            res = _cw_fit(d, survey_design=_cw_survey(), kappa_post=1)
        msg = next(str(w.message) for w in rec if "carry zero weight" in str(w.message))
        assert "event 6: no clean control with positive weight" in msg
        for e in (-2, 0, 1):
            want = (_CW_EFFECTS[4][0] + _CW_EFFECTS[4][1] * e) if e >= 0 else 0.0
            assert res.event_study_effects[e]["effect"] == pytest.approx(want, abs=1e-9)
        sd = res.stacked_data
        in_4 = sd["_sub_exp"] == 4
        np.testing.assert_allclose(sd.loc[in_4 & (sd["_D_sa"] == 1), "_Q_weight"], 1.0)
        np.testing.assert_allclose(sd.loc[in_4 & (sd["_D_sa"] == 0), "_Q_weight"], 1.0)

    def test_cohort_with_zero_treated_population_warns(self):
        d = _cw_panel()
        d.loc[d["first_treat"] == 6, "pop"] = 0.0
        with pytest.warns(UserWarning, match="carry zero weight") as rec:
            res = _cw_fit(d, weighting="population", population="pop")
        msg = next(str(w.message) for w in rec if "carry zero weight" in str(w.message))
        assert "event 6: zero treated population" in msg
        for e, val in _cw_share_truth({4: 1.0}).items():
            assert res.event_study_effects[e]["effect"] == pytest.approx(val, abs=1e-9)

    def test_population_input_is_validated_on_treated_units(self):
        d = _cw_panel(n=30).astype({"pop": float})
        kw = dict(weighting="population", population="pop")
        # A cohort-4 row inside its window [2, 6]: validation reads the stack.
        treated_row = d.index[(d["first_treat"] == 4) & (d["time"] == 4)][0]
        nan_pop = d.copy()
        nan_pop.loc[treated_row, "pop"] = np.nan
        with pytest.raises(ValueError, match="missing or non-finite"):
            _cw_fit(nan_pop, **kw)
        neg_pop = d.copy()
        neg_pop.loc[d["first_treat"] == 4, "pop"] = -1.0
        with pytest.raises(ValueError, match="negative value"):
            _cw_fit(neg_pop, **kw)
        varying = d.copy()
        varying.loc[treated_row, "pop"] = varying.loc[treated_row, "pop"] + 5.0
        with pytest.raises(ValueError, match="varies within 1 treated unit"):
            _cw_fit(varying, **kw)
        # Control units' population, and treated rows outside every window,
        # are never read.
        unread = d.copy()
        unread.loc[d["first_treat"] == 0, "pop"] = np.nan
        unread.loc[(d["first_treat"] == 4) & (d["time"] == 10), "pop"] = -3.0
        base = _cw_fit(d, **kw)
        np.testing.assert_allclose(_cw_fit(unread, **kw).att, base.att, rtol=1e-12)
        # ... whatever they hold: an object column with pd.NA on the controls.
        as_object = d.astype({"pop": object})
        as_object.loc[d["first_treat"] == 0, "pop"] = pd.NA
        res = _cw_fit(as_object, **kw)
        np.testing.assert_allclose(res.att, base.att, rtol=1e-12)
        np.testing.assert_allclose(res.se, base.se, rtol=1e-12)

    @staticmethod
    def _jk1(d, n_rep=10):
        d = d.copy()
        units = sorted(d["unit"].unique())
        cols = []
        for r in range(n_rep):
            col = f"rep_{r}"
            d[col] = np.where(
                d["unit"].isin(set(units[r::n_rep])), 0.0, d["w"] * n_rep / (n_rep - 1)
            )
            cols.append(col)
        return d, cols

    @pytest.mark.parametrize("mse", [False, True])
    def test_replicate_refits_recompute_the_shares(self, mse):
        # Each replicate estimate must be the estimate under THAT replicate's
        # weights, whose cohort shares differ from the full sample's. The
        # oracle is independent of the estimator: weighted cell means under
        # each replicate column, aggregated with that replicate's shares.
        # The weights vary by unit: with weights constant within a cohort this
        # jackknife removes the same mass from every cohort, the shares never
        # move, and refits that reused the full-sample corrective weights
        # would pass as well.
        from diff_diff.survey import SurveyDesign

        d, cols = self._jk1(_cw_weighted(_cw_panel(noise=0.5, seed=4), "random"))
        res = _cw_quiet_fit(
            d,
            survey_design=SurveyDesign(
                weights="w", replicate_weights=cols, replicate_method="JK1", mse=mse
            ),
        )
        events = (-2, 0, 1, 2)
        theta = np.array([res.event_study_effects[e]["effect"] for e in events])
        reps, rep_shares = [], []
        for col in cols:
            shares, cells, w = _cw_shares(d, "aggregate", weight_col=col)
            rep_shares.append(shares[4])
            reps.append([_cw_manual_aggregate(d, shares, cells, w, e) for e in events])
        reps = np.array(reps)
        full_share = _cw_shares(d, "aggregate", weight_col="w")[0][4]
        assert max(abs(s - full_share) for s in rep_shares) > 0.01
        centre = theta if mse else reps.mean(axis=0)
        n_rep = len(cols)
        oracle_var = (n_rep - 1) / n_rep * ((reps - centre) ** 2).sum(axis=0)
        reported = np.array([res.event_study_effects[e]["se"] for e in events])
        np.testing.assert_allclose(reported, np.sqrt(oracle_var), rtol=1e-9)

    def test_replicate_without_control_mass_is_dropped(self):
        from diff_diff.survey import SurveyDesign

        d = _cw_weighted(_cw_panel(noise=0.5, seed=4), "by_cohort")
        d = d[d["first_treat"] != 6].reset_index(drop=True)  # one cohort, never-treated controls
        d, cols = self._jk1(d)
        d.loc[d["first_treat"] == 0, cols[0]] = 0.0  # replicate 0: no control at all
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = _cw_fit(
                d,
                survey_design=SurveyDesign(
                    weights="w", replicate_weights=cols, replicate_method="JK1"
                ),
            )
        assert any("replicate" in str(w.message).lower() for w in caught)
        assert np.isfinite(res.se) and res.se > 0


@pytest.fixture(scope="module")
def alpha_fitted():
    data = generate_staggered_data(
        n_units=120,
        n_periods=10,
        cohort_periods=[4, 6],
        never_treated_frac=0.3,
        treatment_effect=5.0,
        seed=42,
    )
    return StackedDiD(kappa_pre=2, kappa_post=2).fit(
        data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
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
