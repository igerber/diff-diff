"""Tests for the practitioner guidance module."""

import numpy as np
import pytest

from diff_diff import (
    BaconDecomposition,
    CallawaySantAnna,
    ChangesInChanges,
    DifferenceInDifferences,
    HeterogeneousAdoptionDiDEventStudyResults,
    HeterogeneousAdoptionDiDResults,
    MultiPeriodDiD,
    QDiD,
    generate_did_data,
    generate_staggered_data,
)
from diff_diff.changes_in_changes_results import ChangesInChangesResults
from diff_diff.continuous_did_results import ContinuousDiDResults
from diff_diff.efficient_did_results import EfficientDiDResults
from diff_diff.imputation_results import ImputationDiDResults
from diff_diff.practitioner import STEPS, practitioner_next_steps
from diff_diff.results import DiDResults, SyntheticDiDResults
from diff_diff.stacked_did_results import StackedDiDResults
from diff_diff.sun_abraham import SunAbrahamResults
from diff_diff.synthetic_control_results import SyntheticControlResults
from diff_diff.triple_diff import TripleDifferenceResults
from diff_diff.trop_results import TROPResults
from diff_diff.two_stage_results import TwoStageDiDResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def did_data():
    return generate_did_data(n_units=50, treatment_effect=3.0, seed=42)


@pytest.fixture(scope="session")
def staggered_data():
    return generate_staggered_data(n_units=60, n_periods=8, treatment_effect=2.0, seed=42)


@pytest.fixture(scope="session")
def did_results(did_data):
    did = DifferenceInDifferences()
    return did.fit(did_data, outcome="outcome", treatment="treated", post="post")


@pytest.fixture(scope="session")
def multi_period_results(did_data):
    es = MultiPeriodDiD()
    return es.fit(did_data, outcome="outcome", unit="unit", time="period", treatment="treated")


@pytest.fixture(scope="session")
def cs_results(staggered_data):
    cs = CallawaySantAnna()
    return cs.fit(
        staggered_data,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
    )


@pytest.fixture(scope="session")
def bacon_results(staggered_data):
    bacon = BaconDecomposition()
    return bacon.fit(
        staggered_data,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
    )


# ---------------------------------------------------------------------------
# Mock result fixtures for expensive estimators
# ---------------------------------------------------------------------------
def _mock_result(cls, **overrides):
    """Create a minimal mock of a results dataclass."""
    # Provide default fields that most result types share
    defaults = dict(
        att=0.5,
        se=0.1,
        t_stat=5.0,
        p_value=0.001,
        conf_int=(0.3, 0.7),
        n_obs=100,
        n_treated=50,
        n_control=50,
    )
    defaults.update(overrides)
    try:
        return cls(**defaults)
    except TypeError:
        # Some result classes have different required fields
        return cls.__new__(cls)


@pytest.fixture
def mock_synth_results():
    r = SyntheticDiDResults.__new__(SyntheticDiDResults)
    r.att = 1.0
    r.se = 0.3
    return r


@pytest.fixture
def mock_trop_results():
    r = TROPResults.__new__(TROPResults)
    r.att = 0.8
    r.se = 0.2
    return r


@pytest.fixture
def mock_scm_results():
    r = SyntheticControlResults.__new__(SyntheticControlResults)
    r.att = 1.2
    r.se = np.nan
    r.placebo_p_value = np.nan
    return r


@pytest.fixture
def mock_efficient_results():
    r = EfficientDiDResults.__new__(EfficientDiDResults)
    r.overall_att = 0.6
    r.overall_se = 0.15
    return r


@pytest.fixture
def mock_continuous_results():
    r = ContinuousDiDResults.__new__(ContinuousDiDResults)
    r.overall_att = 0.4
    # Canonical SE field on ContinuousDiDResults is overall_att_se (the ATT side).
    # `overall_se` is a read-only property alias since the v3.3.3 inference-field
    # alias surface; assigning to it raises AttributeError.
    r.overall_att_se = 0.1
    return r


@pytest.fixture
def mock_triple_results():
    r = TripleDifferenceResults.__new__(TripleDifferenceResults)
    r.att = 0.7
    r.se = 0.2
    return r


@pytest.fixture
def mock_sa_results():
    r = SunAbrahamResults.__new__(SunAbrahamResults)
    r.overall_att = 0.5
    r.overall_se = 0.1
    return r


@pytest.fixture
def mock_imputation_results():
    r = ImputationDiDResults.__new__(ImputationDiDResults)
    r.overall_att = 0.5
    r.overall_se = 0.1
    return r


@pytest.fixture
def mock_two_stage_results():
    r = TwoStageDiDResults.__new__(TwoStageDiDResults)
    r.overall_att = 0.5
    r.overall_se = 0.1
    return r


@pytest.fixture
def mock_stacked_results():
    r = StackedDiDResults.__new__(StackedDiDResults)
    r.overall_att = 0.5
    r.overall_se = 0.1
    return r


@pytest.fixture
def mock_had_results():
    r = HeterogeneousAdoptionDiDResults.__new__(HeterogeneousAdoptionDiDResults)
    r.att = 0.5
    return r


@pytest.fixture
def mock_had_event_study_results():
    r = HeterogeneousAdoptionDiDEventStudyResults.__new__(HeterogeneousAdoptionDiDEventStudyResults)
    # 5 horizons: e in {-3, -2, 0, 1, 2}
    r.att = np.array([0.01, -0.02, 0.30, 0.45, 0.50])
    r.event_times = np.array([-3, -2, 0, 1, 2])
    return r


@pytest.fixture
def mock_had_results_nan_att():
    r = HeterogeneousAdoptionDiDResults.__new__(HeterogeneousAdoptionDiDResults)
    r.att = float("nan")
    return r


@pytest.fixture
def mock_had_event_study_results_all_nan():
    r = HeterogeneousAdoptionDiDEventStudyResults.__new__(HeterogeneousAdoptionDiDEventStudyResults)
    r.att = np.full(5, np.nan)
    return r


@pytest.fixture
def mock_had_event_study_results_partial_nan():
    r = HeterogeneousAdoptionDiDEventStudyResults.__new__(HeterogeneousAdoptionDiDEventStudyResults)
    r.att = np.array([0.5, np.nan, 0.3])
    return r


# ---------------------------------------------------------------------------
# Tests: return schema
# ---------------------------------------------------------------------------
class TestReturnSchema:
    def test_has_expected_keys(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert "estimator" in output
        assert "completed" in output
        assert "next_steps" in output
        assert "warnings" in output

    def test_estimator_name(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert output["estimator"] == "DifferenceInDifferences"

    def test_estimation_always_completed(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert "estimation" in output["completed"]

    def test_steps_1_and_2_always_emitted(self, did_results):
        """Steps 1 (target parameter) and 2 (assumptions) should always appear."""
        output = practitioner_next_steps(did_results, verbose=False)
        baker_steps = [s["baker_step"] for s in output["next_steps"]]
        assert 1 in baker_steps, "Step 1 (target parameter) missing"
        assert 2 in baker_steps, "Step 2 (assumptions) missing"

    def test_steps_1_and_2_filterable(self, did_results):
        """Agents can filter Steps 1-2 via completed_steps."""
        output = practitioner_next_steps(
            did_results,
            completed_steps=["target_parameter", "assumptions"],
            verbose=False,
        )
        baker_steps = [s["baker_step"] for s in output["next_steps"]]
        assert 1 not in baker_steps
        assert 2 not in baker_steps

    def test_next_steps_are_dicts(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        for step in output["next_steps"]:
            assert "baker_step" in step
            assert "label" in step
            assert "why" in step
            assert "code" in step
            assert "priority" in step

    def test_warnings_are_strings(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        for w in output["warnings"]:
            assert isinstance(w, str)


# ---------------------------------------------------------------------------
# Tests: each result type produces guidance
# ---------------------------------------------------------------------------
class TestResultTypeDispatch:
    def test_did_results(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert len(output["next_steps"]) > 0

    def test_multi_period_results(self, multi_period_results):
        output = practitioner_next_steps(multi_period_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "MultiPeriodDiD (Event Study)"

    def test_cs_results(self, cs_results):
        output = practitioner_next_steps(cs_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "CallawaySantAnna"

    def test_bacon_results(self, bacon_results):
        output = practitioner_next_steps(bacon_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "BaconDecomposition"
        # Bacon should suggest switching to a robust estimator
        labels = [s["label"] for s in output["next_steps"]]
        assert any("heterogeneity-robust" in lbl for lbl in labels)

    def test_sa_results(self, mock_sa_results):
        output = practitioner_next_steps(mock_sa_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "SunAbraham"
        # SA guidance should use to_dataframe, NOT aggregate='group'
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "aggregate=" not in all_code or "to_dataframe" in all_code

    def test_imputation_results(self, mock_imputation_results):
        output = practitioner_next_steps(mock_imputation_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # ImputationDiD has no control_group parameter — code snippets must not use it
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "control_group" not in all_code

    def test_two_stage_results(self, mock_two_stage_results):
        output = practitioner_next_steps(mock_two_stage_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # TwoStageDiD has no control_group parameter — code snippets must not use it
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "control_group" not in all_code

    def test_stacked_results(self, mock_stacked_results):
        output = practitioner_next_steps(mock_stacked_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # StackedDiD uses clean_control, not control_group
        all_text = " ".join(s.get("code", "") + s.get("why", "") for s in output["next_steps"])
        assert "not_yet_treated" not in all_text or "control_group" in all_text

    def test_stacked_balance_step_uses_distinct_step_name(self, mock_stacked_results):
        # M-024: "Check sub-experiment balance" must NOT reuse
        # step_name="heterogeneity" - since the ES surface is always
        # populated, DiagnosticReport's heterogeneity check runs on every
        # plain fit and a shared key silently dropped this unrelated
        # advice from next_steps via _filter_steps.
        from diff_diff.practitioner import STEPS, _handle_stacked

        steps, _ = _handle_stacked(mock_stacked_results)
        balance = [s for s in steps if s["label"] == "Check sub-experiment balance"]
        assert len(balance) == 1
        assert balance[0]["_step_name"] == "sub_experiment_balance"
        # SURVIVAL: completing heterogeneity (what DiagnosticReport does on
        # every surface-populated fit) must no longer drop the balance step.
        completed = practitioner_next_steps(
            mock_stacked_results, verbose=False, completed_steps=["heterogeneity"]
        )
        labels = [s["label"] for s in completed["next_steps"]]
        assert "Check sub-experiment balance" in labels
        # Like "loo_jackknife", the key deliberately stays OUT of the STEPS
        # completion vocabulary: no diagnostic ever completes it.
        assert "sub_experiment_balance" not in STEPS

    def test_synth_results(self, mock_synth_results):
        output = practitioner_next_steps(mock_synth_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "SyntheticDiD"
        # SDiD handler steps (exclude generic Steps 1-2) should NOT use staggered knobs
        handler_steps = [s for s in output["next_steps"] if s["baker_step"] > 2]
        all_code = " ".join(s.get("code", "") for s in handler_steps)
        assert "control_group" not in all_code
        assert "anticipation" not in all_code

    def test_trop_results(self, mock_trop_results):
        output = practitioner_next_steps(mock_trop_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # TROP handler steps (exclude generic Steps 1-2) should NOT use staggered knobs
        handler_steps = [s for s in output["next_steps"] if s["baker_step"] > 2]
        all_code = " ".join(s.get("code", "") for s in handler_steps)
        assert "control_group" not in all_code
        assert "anticipation" not in all_code

    def test_synthetic_control_results(self, mock_scm_results):
        output = practitioner_next_steps(mock_scm_results, verbose=False)
        assert output["estimator"] == "SyntheticControl"
        assert len(output["next_steps"]) > 0
        # The in-space placebo step must surface (it is SCM's significance test)
        # and must not be auto-suppressed as the completed estimation step.
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        all_labels = " ".join(s.get("label", "") for s in output["next_steps"]).lower()
        assert "in_space_placebo" in all_code
        assert "placebo" in all_labels
        # The ADH-2015 robustness steps also surface (opt-in diagnostics, non-STEPS
        # tags so a caller's completed_steps can never suppress them).
        assert "leave_one_out" in all_code and "in_time_placebo" in all_code
        # SCM is not a staggered DiD: no control-group / anticipation knobs.
        handler_steps = [s for s in output["next_steps"] if s["baker_step"] > 2]
        handler_code = " ".join(s.get("code", "") for s in handler_steps)
        assert "control_group" not in handler_code and "anticipation" not in handler_code

    def test_efficient_results(self, mock_efficient_results):
        output = practitioner_next_steps(mock_efficient_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # EfficientDiD uses never_treated/last_cohort — code must not suggest not_yet_treated
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "not_yet_treated" not in all_code

    def test_continuous_results(self, mock_continuous_results):
        output = practitioner_next_steps(mock_continuous_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # ContinuousDiD should NOT emit check_parallel_trends
        all_text = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "check_parallel_trends" not in all_text

    def test_triple_results(self, mock_triple_results):
        output = practitioner_next_steps(mock_triple_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # DDD should NOT claim "requires PT along two dimensions"
        all_text = " ".join(s.get("why", "") for s in output["next_steps"])
        assert "two dimensions" not in all_text
        assert "check_parallel_trends" not in " ".join(
            s.get("code", "") for s in output["next_steps"]
        )


# ---------------------------------------------------------------------------
# Tests: completed_steps filtering
# ---------------------------------------------------------------------------
class TestCompletedSteps:
    def test_filter_parallel_trends(self, cs_results):
        full = practitioner_next_steps(cs_results, verbose=False)
        filtered = practitioner_next_steps(
            cs_results, completed_steps=["parallel_trends"], verbose=False
        )
        assert len(filtered["next_steps"]) < len(full["next_steps"])
        # No step should have baker_step 3 about parallel trends
        for s in filtered["next_steps"]:
            if s["baker_step"] == 3:
                assert "parallel trends" not in s["label"].lower()

    def test_filter_sensitivity(self, cs_results):
        full = practitioner_next_steps(cs_results, verbose=False)
        filtered = practitioner_next_steps(
            cs_results, completed_steps=["sensitivity"], verbose=False
        )
        assert len(filtered["next_steps"]) < len(full["next_steps"])

    def test_filter_all_steps(self, cs_results):
        output = practitioner_next_steps(cs_results, completed_steps=list(STEPS), verbose=False)
        assert len(output["next_steps"]) == 0

    def test_invalid_step_name_raises(self, did_results):
        with pytest.raises(ValueError, match="Unknown step names"):
            practitioner_next_steps(did_results, completed_steps=["invalid_step"], verbose=False)


# ---------------------------------------------------------------------------
# Tests: verbose output
# ---------------------------------------------------------------------------
class TestVerboseOutput:
    def test_verbose_prints(self, did_results, capsys):
        practitioner_next_steps(did_results, verbose=True)
        captured = capsys.readouterr()
        assert "Practitioner Guidance" in captured.out
        assert "Baker et al." in captured.out
        assert "DifferenceInDifferences" in captured.out

    def test_no_print_when_silent(self, did_results, capsys):
        practitioner_next_steps(did_results, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Tests: NaN handling
# ---------------------------------------------------------------------------
class TestNaNHandling:
    def test_nan_att_produces_warning(self):
        r = DiDResults(
            att=float("nan"),
            se=float("nan"),
            t_stat=float("nan"),
            p_value=float("nan"),
            conf_int=(float("nan"), float("nan")),
            n_obs=100,
            n_treated=50,
            n_control=50,
        )
        output = practitioner_next_steps(r, verbose=False)
        assert len(output["warnings"]) > 0
        assert any("NaN" in w for w in output["warnings"])

    def test_nan_avg_att_multi_period(self):
        """MultiPeriodDiDResults uses avg_att, not att."""
        from diff_diff.results import MultiPeriodDiDResults

        r = MultiPeriodDiDResults.__new__(MultiPeriodDiDResults)
        r.avg_att = float("nan")
        output = practitioner_next_steps(r, verbose=False)
        assert any("NaN" in w for w in output["warnings"])


# ---------------------------------------------------------------------------
# Tests: Bacon handler warnings
# ---------------------------------------------------------------------------
class TestBaconWarnings:
    def test_forbidden_comparison_warning(self, bacon_results):
        output = practitioner_next_steps(bacon_results, verbose=False)
        # Real Bacon results from staggered data should have forbidden comparisons
        weight = getattr(bacon_results, "total_weight_later_vs_earlier", 0)
        if weight > 0.01:
            assert any("contaminated" in w for w in output["warnings"])

    def test_bacon_with_high_forbidden_weight(self):
        """Mock Bacon results with high forbidden comparison weight."""
        from diff_diff.bacon import BaconDecompositionResults

        r = BaconDecompositionResults.__new__(BaconDecompositionResults)
        r.overall_att = 0.5
        r.total_weight_later_vs_earlier = 0.4
        r.comparisons = []
        output = practitioner_next_steps(r, verbose=False)
        assert any("contaminated" in w for w in output["warnings"])
        assert any("40%" in w for w in output["warnings"])


# ---------------------------------------------------------------------------
# Tests: EfficientDiD handler path
# ---------------------------------------------------------------------------
class TestEfficientDiDHandler:
    def test_hausman_pretest_in_guidance(self, mock_efficient_results):
        output = practitioner_next_steps(mock_efficient_results, verbose=False)
        labels = [s["label"] for s in output["next_steps"]]
        assert any("hausman" in lbl.lower() or "Hausman" in lbl for lbl in labels)

    def test_hausman_snippet_uses_classmethod(self, mock_efficient_results):
        output = practitioner_next_steps(mock_efficient_results, verbose=False)
        hausman_steps = [
            s
            for s in output["next_steps"]
            if "hausman" in s["label"].lower() or "Hausman" in s["label"]
        ]
        assert len(hausman_steps) > 0
        assert "hausman_pretest" in hausman_steps[0]["code"]

    def _agg_step(self, output):
        return [
            s
            for s in output["next_steps"]
            if "Aggregate treatment-effect heterogeneity" in s["label"]
        ]

    def test_aggregation_step_post_fit_branch(self, mock_efficient_results):
        # Analytical fit (no bootstrap_results attr on the mock -> None
        # branch): the guidance recommends post-fit aggregate() (M-023).
        output = practitioner_next_steps(mock_efficient_results, verbose=False)
        steps = self._agg_step(output)
        assert len(steps) == 1
        assert "results.aggregate('group')" in steps[0]["code"]
        assert "no refit needed" in steps[0]["why"]

    def test_aggregation_step_bootstrap_branch(self, mock_efficient_results):
        # Bootstrapped fit: post-fit aggregate() fails closed, so the
        # guidance routes through the deprecated fit-time aggregation.
        mock_efficient_results.bootstrap_results = object()
        output = practitioner_next_steps(mock_efficient_results, verbose=False)
        steps = self._agg_step(output)
        assert len(steps) == 1
        assert "BOOTSTRAPPED" in steps[0]["why"]
        assert "aggregate='all'" in steps[0]["code"]

    def test_aggregation_step_name_is_non_steps_key(self, mock_efficient_results):
        # The "aggregation" key is deliberately OUTSIDE the STEPS
        # completion vocabulary (the M-024 step-name-collision lesson):
        # no DiagnosticReport check can auto-suppress this guidance, and
        # a completed heterogeneity check must not swallow it.
        from diff_diff.practitioner import STEPS

        assert "aggregation" not in STEPS
        output = practitioner_next_steps(
            mock_efficient_results,
            completed_steps=["heterogeneity"],
            verbose=False,
        )
        assert len(self._agg_step(output)) == 1


class _AggregationStepMixin:
    """Shared pins for the post-fit aggregation guidance (M-021/M-022,
    mirroring the EfficientDiD M-023 pins above)."""

    row_id = ""
    fit_var = ""

    @staticmethod
    def _agg_step(output):
        return [
            s
            for s in output["next_steps"]
            if "Aggregate treatment-effect heterogeneity" in s["label"]
        ]

    def _results(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def test_aggregation_step_post_fit_branch(self):
        output = practitioner_next_steps(self._results(), verbose=False)
        steps = self._agg_step(output)
        assert len(steps) == 1
        assert "results.aggregate('group')" in steps[0]["code"]
        assert "no refit needed" in steps[0]["why"]
        assert self.row_id in steps[0]["why"]

    def test_aggregation_step_bootstrap_branch(self):
        r = self._results()
        r.bootstrap_results = object()
        output = practitioner_next_steps(r, verbose=False)
        steps = self._agg_step(output)
        assert len(steps) == 1
        assert "BOOTSTRAPPED" in steps[0]["why"]
        assert "aggregate='all'" in steps[0]["code"]

    def test_aggregation_step_name_is_non_steps_key(self):
        from diff_diff.practitioner import STEPS

        assert "aggregation" not in STEPS
        output = practitioner_next_steps(
            self._results(), completed_steps=["heterogeneity"], verbose=False
        )
        assert len(self._agg_step(output)) == 1


class TestImputationAggregationStep(_AggregationStepMixin):
    row_id = "M-021"

    def _results(self):
        r = ImputationDiDResults.__new__(ImputationDiDResults)
        r.overall_att = 0.6
        r.overall_se = 0.15
        return r


class TestTwoStageAggregationStep(_AggregationStepMixin):
    row_id = "M-022"

    def _results(self):
        r = TwoStageDiDResults.__new__(TwoStageDiDResults)
        r.overall_att = 0.6
        r.overall_se = 0.15
        return r


class TestContinuousAggregationStep:
    """M-025 post-fit aggregation guidance for ContinuousDiD.

    Deliberately NOT on ``_AggregationStepMixin``: that mixin pins
    ``aggregate('group')`` (not in ContinuousDiD's supported set),
    simulates bootstrap via a ``bootstrap_results`` field
    (ContinuousDiDResults keys on ``n_bootstrap`` and has no such field),
    and expects an ``aggregate='all'`` fallback (never a valid
    ContinuousDiD value). ContinuousDiD's step is a single unconditional
    step whose wording carries the bootstrap carve-out.
    """

    @staticmethod
    def _agg_step(output):
        return [s for s in output["next_steps"] if "Aggregate post-fit" in s["label"]]

    def _results(self):
        r = ContinuousDiDResults.__new__(ContinuousDiDResults)
        r.overall_att = 0.6
        r.overall_att_se = 0.15
        return r

    def test_aggregation_step_present_with_all_routes(self):
        output = practitioner_next_steps(self._results(), verbose=False)
        steps = self._agg_step(output)
        assert len(steps) == 1
        assert "M-025" in steps[0]["why"]
        assert "results.aggregate('event_study')" in steps[0]["code"]
        assert "results.aggregate('dose')" in steps[0]["code"]
        assert "results.aggregate('simple')" in steps[0]["code"]
        # The bootstrap carve-out is carried in the wording.
        assert "n_bootstrap=0" in steps[0]["why"]

    def test_aggregation_step_name_is_non_steps_key(self):
        from diff_diff.practitioner import STEPS

        assert "aggregation" not in STEPS
        output = practitioner_next_steps(
            self._results(), completed_steps=["heterogeneity"], verbose=False
        )
        assert len(self._agg_step(output)) == 1


# ---------------------------------------------------------------------------
# Tests: unknown result type fallback
# ---------------------------------------------------------------------------
class TestFallback:
    def test_unknown_type(self):
        class FakeResults:
            att = 1.0
            se = 0.5

        output = practitioner_next_steps(FakeResults(), verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "FakeResults"


# ---------------------------------------------------------------------------
# Tests: HeterogeneousAdoptionDiD (HAD) handler dispatch
# ---------------------------------------------------------------------------
class TestHADDispatch:
    def test_had_results_dispatch(self, mock_had_results):
        output = practitioner_next_steps(mock_had_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "HeterogeneousAdoptionDiD (HAD)"

    def test_had_event_study_dispatch(self, mock_had_event_study_results):
        output = practitioner_next_steps(mock_had_event_study_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "HeterogeneousAdoptionDiD (Event Study)"

    def test_had_pretest_workflow_referenced(self, mock_had_results):
        output = practitioner_next_steps(mock_had_results, verbose=False)
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "did_had_pretest_workflow" in all_code

    def test_had_event_study_pretest_workflow_referenced(self, mock_had_event_study_results):
        output = practitioner_next_steps(mock_had_event_study_results, verbose=False)
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "did_had_pretest_workflow" in all_code
        # M-027/M-139: the guidance must NOT teach the deprecated mode kwarg
        # - the workflow selects the event-study battery from the panel
        # shape, and the snippet shows a plain call.
        assert "aggregate=" not in all_code

    def test_had_bandwidth_diagnostics_referenced(self, mock_had_results):
        output = practitioner_next_steps(mock_had_results, verbose=False)
        all_text = " ".join(
            (s.get("code", "") + " " + s.get("why", "")) for s in output["next_steps"]
        )
        assert "bandwidth_diagnostics" in all_text

    def test_had_event_study_simultaneous_bands_referenced(self, mock_had_event_study_results):
        output = practitioner_next_steps(mock_had_event_study_results, verbose=False)
        all_text = " ".join(
            (s.get("code", "") + " " + s.get("why", "")) for s in output["next_steps"]
        )
        assert "cband" in all_text
        # Either "sup-t" wording or "simultaneous" wording is acceptable.
        assert ("sup-t" in all_text) or ("simultaneous" in all_text)

    def test_had_no_comparison_group_framing(self, mock_had_results, mock_had_event_study_results):
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            all_text = " ".join(
                (s.get("code", "") + " " + s.get("why", "") + " " + s.get("label", ""))
                for s in output["next_steps"]
            )
            all_text += " ".join(output["warnings"])
            assert "no comparison group" not in all_text.lower()
            assert "missing comparison" not in all_text.lower()

    def test_had_nan_warning_scalar(self, mock_had_results_nan_att):
        output = practitioner_next_steps(mock_had_results_nan_att, verbose=False)
        warnings = " ".join(output["warnings"])
        assert "NaN" in warnings or "nan" in warnings.lower()

    def test_had_event_study_nan_warning_array(self, mock_had_event_study_results_all_nan):
        output = practitioner_next_steps(mock_had_event_study_results_all_nan, verbose=False)
        warnings = " ".join(output["warnings"])
        assert "per-horizon" in warnings or "All" in warnings

    def test_had_partial_nan_array_no_warning(self, mock_had_event_study_results_partial_nan):
        # Partial-NaN arrays are legitimate event-study output (some
        # horizons may collapse on degenerate-design grounds while others
        # remain finite). The all-NaN warning must NOT fire here.
        output = practitioner_next_steps(mock_had_event_study_results_partial_nan, verbose=False)
        # No "per-horizon" or "All ... NaN" warning string should appear.
        warnings = " ".join(output["warnings"])
        assert "per-horizon" not in warnings
        assert "All " not in warnings

    def test_had_step_4_estimator_selection_present(
        self, mock_had_results, mock_had_event_study_results
    ):
        # Step-4 must surface the WAS-vs-ATT(d) estimand difference (not
        # a blanket "if untreated → not HAD" rule which would contradict
        # REGISTRY § HeterogeneousAdoptionDiD edge cases lines ~2403/2408).
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
            assert len(step_4_steps) >= 1
            all_text = " ".join(
                (s.get("code", "") + " " + s.get("why", "") + " " + s.get("label", ""))
                for s in step_4_steps
            )
            # Routing nudge must name ContinuousDiD as the estimand
            # alternative; framing must center on WAS vs ATT(d) (the
            # actual estimand differentiator), NOT on whether untreated
            # units exist.
            assert "ContinuousDiD" in all_text
            assert "WAS" in all_text
            assert "ATT(d)" in all_text

    def test_had_step_4_does_not_misframe_untreated_unit_routing(
        self, mock_had_results, mock_had_event_study_results
    ):
        # Per REGISTRY: HAD is compatible with a small share of
        # never-treated units (paper edge case), and on staggered
        # event-study panels never-treated units are explicitly RETAINED
        # (Appendix B.2 / had.py:1432). The Step-4 routing must NOT
        # carry the wrong "if untreated → not HAD" framing.
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
            all_text = " ".join(
                (s.get("code", "") + " " + s.get("why", "") + " " + s.get("label", ""))
                for s in step_4_steps
            ).lower()
            forbidden_phrases = (
                "switch away from had",
                "had's was divisor under-weights",
                "drop untreated",
                "must drop never-treated",
            )
            for phrase in forbidden_phrases:
                assert phrase not in all_text, (
                    f"HAD Step-4 must not carry the phrase {phrase!r}: "
                    f"per REGISTRY § HeterogeneousAdoptionDiD edge cases, "
                    f"HAD is compatible with a small share of never-treated "
                    f"units and explicitly retains them on staggered "
                    f"event-study panels."
                )

    def test_had_step_4_accepts_first_treat_inf_encoding(
        self, mock_had_results, mock_had_event_study_results
    ):
        """The HAD -> ContinuousDiD inverse handoff at Step 4 must
        accommodate both `first_treat == 0` and `first_treat == inf`
        as never-treated encodings (ContinuousDiD normalizes
        `inf -> 0` internally). Without this, a HAD user contemplating
        the switch to ContinuousDiD could misclassify an `inf`-encoded
        panel as ineligible. Mirror of
        `test_handle_continuous_step_4_accepts_first_treat_inf_encoding`.
        """
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
            assert len(step_4_steps) >= 1
            text = " ".join(
                (s.get("why", "") + " " + s.get("code", "")) for s in step_4_steps
            ).lower()
            # Either the explicit `inf` encoding is mentioned, or the
            # encoding-agnostic "dose == 0 throughout" framing is used.
            assert "inf" in text or "dose == 0 throughout" in text, (
                "HAD Step-4 ContinuousDiD-handoff rationale must accommodate "
                "both first_treat == 0 and first_treat == inf as never-treated "
                "encodings, or mention 'dose == 0 throughout' framing."
            )

    def test_handle_continuous_step_4_routes_to_had(self, mock_continuous_results):
        # Symmetric pair: ContinuousDiD users with no untreated units
        # should be routed to HeterogeneousAdoptionDiD.
        output = practitioner_next_steps(mock_continuous_results, verbose=False)
        step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
        assert len(step_4_steps) >= 1
        all_text = " ".join((s.get("code", "") + " " + s.get("why", "")) for s in step_4_steps)
        assert "HeterogeneousAdoptionDiD" in all_text

    def test_handle_generic_ndarray_att_triggers_warning(self):
        # Cross-handler regression: a future estimator that returns
        # ndarray att and falls through to _handle_generic must produce
        # the same all-NaN warning as the dedicated HAD event-study path.
        class FutureNdarrayAttResults:
            att: np.ndarray

        r = FutureNdarrayAttResults()
        r.att = np.full(3, np.nan)
        output = practitioner_next_steps(r, verbose=False)
        warnings = " ".join(output["warnings"])
        assert "per-horizon" in warnings or "All" in warnings

    def test_had_handlers_string_only_no_attribute_reads(
        self, mock_had_results, mock_had_event_study_results
    ):
        # Stability invariant #7: handlers are STRING-ONLY at runtime.
        # The fixtures construct results with ONLY .att (and event_times
        # on the event-study fixture); confirm no AttributeError is
        # raised when the handlers run. Protects against a future
        # refactor that starts reading result.<some_field> inside a
        # handler and silently breaks the minimal-fixture contract.
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            assert isinstance(output, dict)
            assert "next_steps" in output

    def test_had_handler_snippets_are_valid_python_syntax(
        self, mock_had_results, mock_had_event_study_results
    ):
        # Snippet smoke test: every code block emitted by the HAD
        # handlers must parse as valid Python. Catches the failure mode
        # where snippets reference undefined names with placeholder
        # syntax that doesn't compile (e.g. `survey_design=design` with
        # no `design` defined in scope, or attribute typos that break
        # copy/paste).
        import ast

        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            for step in output["next_steps"]:
                code = step.get("code", "")
                if not code.strip():
                    continue
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    pytest.fail(
                        f"Step {step['baker_step']} ({step['label']!r}) "
                        f"emits a code snippet that does not parse as "
                        f"valid Python: {e}\n\nSnippet:\n{code}"
                    )

    def test_handle_continuous_step_4_snippet_is_valid_python(self, mock_continuous_results):
        # Same syntax check on the symmetric Step-4 in _handle_continuous.
        import ast

        output = practitioner_next_steps(mock_continuous_results, verbose=False)
        step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
        for step in step_4_steps:
            code = step.get("code", "")
            if code.strip():
                ast.parse(code)  # raises SyntaxError on failure

    def test_handle_continuous_step_4_honors_had_panel_shape_contract(
        self, mock_continuous_results
    ):
        """The ContinuousDiD -> HAD Step-4 handoff must respect HAD's
        panel-shape contract: `aggregate="overall"` (default) is
        two-period only; multi-period panels MUST set
        `aggregate="event_study"`; staggered panels yield last-
        cohort-only WAS. Without surfacing those distinctions, a
        copy-paste on a valid multi-period ContinuousDiD result
        either raises at fit time or silently shifts estimand.
        """
        output = practitioner_next_steps(mock_continuous_results, verbose=False)
        step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
        assert len(step_4_steps) >= 1
        had_step = next(
            (s for s in step_4_steps if "HeterogeneousAdoptionDiD" in s.get("code", "")),
            None,
        )
        assert had_step is not None, "Step 4 must include a HAD handoff snippet for ContinuousDiD"
        text = (had_step.get("why", "") + " " + had_step.get("code", "")).lower()
        # Snippet or rationale must call out the multi-period event-study
        # path in the POST-M-027 vocabulary: fit() selects the mode from the
        # panel shape, so the handoff must NOT teach the deprecated kwarg
        # (copy-paste on a multi-period panel now just works).
        assert "event-study" in text or "event_study" in text
        assert "aggregate=" not in text, (
            "ContinuousDiD -> HAD handoff must not teach the deprecated "
            "fit(aggregate=) kwarg (M-027: the mode is panel-inferred)."
        )
        # And the staggered last-cohort-only caveat must be surfaced.
        assert "last-cohort" in text or "last cohort" in text, (
            "Step-4 rationale must surface HAD's last-cohort-only event-study "
            "behavior on staggered panels so agents understand the estimand "
            "shift before recommending the handoff."
        )
        # And the ChaisemartinDHaultfoeuille pointer for full multi-cohort
        # staggered continuous support must be present.
        assert "chaisemartindhaultfoeuille" in text, (
            "Step-4 rationale must point at ChaisemartinDHaultfoeuille as the "
            "alternative when full multi-cohort staggered continuous support "
            "is required."
        )

    def test_handle_continuous_step_4_accepts_first_treat_inf_encoding(
        self, mock_continuous_results
    ):
        """`ContinuousDiD.fit` accepts both `first_treat == 0` and
        `first_treat == inf` (the latter is normalized to 0 internally).
        Step-4 guidance must not hard-code "first_treat == 0" as the
        only never-treated encoding, or agents will misclassify
        inf-encoded panels.
        """
        output = practitioner_next_steps(mock_continuous_results, verbose=False)
        step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
        had_step = next(
            (s for s in step_4_steps if "HeterogeneousAdoptionDiD" in s.get("code", "")),
            None,
        )
        assert had_step is not None
        text = (had_step.get("why", "") + " " + had_step.get("code", "")).lower()
        # Either the explicit `inf` encoding is mentioned, or the
        # encoding-agnostic "dose == 0 throughout" framing is used.
        assert "inf" in text or "dose = 0" in text or "dose == 0" in text, (
            "Step-4 rationale must accommodate both first_treat == 0 and "
            "first_treat == inf as never-treated encodings, or mention the "
            "encoding-agnostic 'dose == 0 throughout' framing."
        )

    def test_handle_continuous_step_4_recodes_first_treat_inf_for_had(
        self, mock_continuous_results
    ):
        """R10 P1: the ContinuousDiD -> HAD Step-4 handoff must include
        an explicit `first_treat=inf -> 0` recode in the emitted code
        snippet. ContinuousDiD silently normalizes `inf` to `0`, but
        HAD's _validate_had_panel rejects any first_treat value outside
        {0, t_post} at the front door (had.py:1208-1214). Without the
        recode, a copy-paste of the advertised handoff on a valid
        inf-encoded ContinuousDiD panel raises
        `ValueError: first_treat='first_treat' contains value(s)
        [inf] outside the allowed set {0, t_post}`.
        """
        output = practitioner_next_steps(mock_continuous_results, verbose=False)
        step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
        had_step = next(
            (s for s in step_4_steps if "HeterogeneousAdoptionDiD" in s.get("code", "")),
            None,
        )
        assert had_step is not None
        code = had_step.get("code", "")
        # The snippet must show the recode BEFORE the had.fit() call.
        # Accept any of: pandas `.replace({np.inf: 0})`, mask-style
        # `.where(~np.isinf(...), 0)`, or an explicit comment naming
        # the requirement on the panel preparation step.
        recode_present = (
            "np.inf: 0" in code or "np.isinf" in code or "first_treat=0" in code.replace(" ", "")
        )
        assert recode_present, (
            "Step-4 ContinuousDiD->HAD snippet must include an explicit "
            "first_treat=inf -> 0 recode (HAD requires first_treat in "
            "{0, t_post}; ContinuousDiD's inf-encoding is rejected at the "
            f"HAD front door). Snippet:\n{code}"
        )
        # And the snippet must mention the underlying contract so users
        # who recode by other means know why.
        why = had_step.get("why", "") + " " + code
        assert "first_treat" in why and ("0" in why) and ("inf" in why.lower()), (
            "Step-4 rationale/code must reference both first_treat=0 "
            "and first_treat=inf so the recode step is self-explanatory."
        )

    def test_had_event_study_sup_t_snippet_uses_hc1_for_mass_point_survey_compatibility(
        self, mock_had_event_study_results
    ):
        # Per had.py:3646-3658 the mass-point design rejects the
        # default classical vcov family on the survey_design= path
        # (NotImplementedError). The Step-6 sup-t snippet shows a
        # generic weighted event-study fit; if it uses the default
        # vcov_type a copy/paste on a mass-point panel raises at
        # fit time. Snippet must either use vcov_type='hc1' /
        # robust=True OR explicitly note the requirement so agents
        # can adapt.
        output = practitioner_next_steps(mock_had_event_study_results, verbose=False)
        step_6_steps = [s for s in output["next_steps"] if s["baker_step"] == 6]
        assert len(step_6_steps) >= 1
        # Find the sup-t / cband step (sensitivity step).
        sup_t = next(
            (s for s in step_6_steps if "cband" in s.get("code", "")),
            None,
        )
        assert sup_t is not None, "sup-t / cband step not found at baker_step=6"
        snippet = sup_t.get("code", "")
        # Either the snippet itself uses vcov_type='hc1' / robust=True
        # OR it documents the requirement inline (so agents adapting
        # the snippet on a mass-point panel know to add it).
        ok = (
            "vcov_type='hc1'" in snippet
            or 'vcov_type="hc1"' in snippet
            or "robust=True" in snippet
            or ("mass-point" in snippet and "vcov_type" in snippet)
            or ("mass_point" in snippet and "vcov_type" in snippet)
        )
        assert ok, (
            "Sup-t / cband snippet must either use vcov_type='hc1' / "
            "robust=True or surface the mass-point + survey vcov "
            "requirement inline. Per had.py:3646-3658 the default "
            "classical sandwich raises NotImplementedError on the "
            "mass-point + survey path; the example as written would "
            "fail at fit time on a mass-point panel."
        )

    def test_had_results_to_dict_docstring_matches_weighted_mass_point_contract(self):
        # Parallel to the dataclass-field-docstring regression below:
        # PR #402 R8 P3 caught that HeterogeneousAdoptionDiDResults.to_dict()
        # docstring still described variance_formula as continuous-only
        # / "pweight" or "survey_binder_tsl", contradicting the field
        # docstrings (fixed in R5) and llms-full.txt (fixed in R3).
        # Lock the to_dict() docstring against drift back.
        from diff_diff.had import HeterogeneousAdoptionDiDResults

        doc = HeterogeneousAdoptionDiDResults.to_dict.__doc__ or ""
        for label in (
            "survey_binder_tsl",
            "survey_binder_tsl_2sls",
        ):
            assert label in doc, (
                f"HeterogeneousAdoptionDiDResults.to_dict() docstring "
                f"must enumerate the {label!r} variance_formula label - "
                f"weighted mass-point fits populate survey_binder_tsl_2sls "
                f"per had.py. The to_dict() docstring is a public "
                f"source-of-truth surface and must match the field "
                f"docstrings + llms-full.txt HAD section."
            )
        # The pweight / pweight_2sls labels were removed with the weights=
        # kwarg in 3.7.0 and must not reappear.
        assert "pweight" not in doc, (
            "HeterogeneousAdoptionDiDResults.to_dict() docstring must not "
            "mention the removed pweight labels after the 3.7.0 consolidation."
        )
        # effective_dose_mean: must mention mass-point Wald-IV semantics.
        assert "mass_point" in doc or "mass-point" in doc, (
            "HeterogeneousAdoptionDiDResults.to_dict() docstring must "
            "describe the mass-point effective_dose_mean semantics; "
            "weighted mass-point fits populate it as the weighted "
            "Wald-IV dose gap per had.py:3793-3811."
        )
        assert "Wald-IV" in doc or "Z=1" in doc, (
            "HeterogeneousAdoptionDiDResults.to_dict() docstring must "
            "describe the weighted Wald-IV dose gap semantics for "
            "mass-point fits."
        )

    def test_had_results_dataclass_docstrings_match_weighted_mass_point_contract(self):
        # PR #402 R3 fixed the llms-full.txt field descriptions to
        # acknowledge that weighted mass-point fits populate
        # variance_formula in {"pweight_2sls", "survey_binder_tsl_2sls"}
        # and effective_dose_mean as the weighted Wald-IV dose gap (per
        # had.py:3736-3811). PR #402 R5 P3 caught that the dataclass
        # field docstrings still said those fields were continuous-only
        # / None on mass-point - leaving two source-of-truth surfaces
        # disagreeing about the same public result object. Lock the
        # dataclass docstrings against drift back to the continuous-only
        # framing.
        import inspect

        from diff_diff.had import HeterogeneousAdoptionDiDResults

        # Field docstrings live as raw __doc__ on the FieldDescriptor /
        # in __dataclass_fields__'s metadata; read them via the type's
        # source-level docstring attached to the class via the field's
        # `__doc__` after assignment in the class body.
        # Easier: read the class source via inspect.getsource() and check
        # the field-docstring blocks we care about.
        src = inspect.getsource(HeterogeneousAdoptionDiDResults)
        # variance_formula docstring must enumerate the 2 Binder-TSL labels
        # (the pweight / pweight_2sls labels were removed with the weights=
        # kwarg in the 3.7.0 consolidation).
        assert "survey_binder_tsl_2sls" in src, (
            "HeterogeneousAdoptionDiDResults.variance_formula docstring "
            "must mention `survey_binder_tsl_2sls` (weighted mass-point "
            "Binder-TSL label)."
        )
        assert "survey_binder_tsl" in src, (
            "HeterogeneousAdoptionDiDResults.variance_formula docstring "
            "must mention `survey_binder_tsl` (weighted continuous "
            "Binder-TSL label)."
        )
        # effective_dose_mean docstring must mention mass-point Wald-IV.
        assert "mass_point" in src or "mass-point" in src, (
            "HeterogeneousAdoptionDiDResults.effective_dose_mean "
            "docstring must mention mass-point semantics; weighted "
            "mass-point fits populate it as the weighted Wald-IV dose "
            "gap per had.py:3793-3811."
        )
        assert "Wald-IV" in src or "Z=1" in src, (
            "HeterogeneousAdoptionDiDResults.effective_dose_mean "
            "docstring must describe the weighted Wald-IV dose gap "
            "semantics (or the underlying Z=1/Z=0 subgroup-mean form) "
            "for mass-point fits."
        )

    def test_had_step_3_documents_earlier_pre_period_precondition_for_step_2(
        self, mock_had_results, mock_had_event_study_results
    ):
        # Per docs/methodology/REGISTRY.md HeterogeneousAdoptionDiD
        # § "Assumption 7 / step 2 closure" + had_pretests.py:4738-4756 +
        # 2769: aggregate="event_study" closes step 2 ONLY IF the panel
        # carries at least one earlier placebo pre-period beyond the
        # base F-1. With only F-1 available the workflow sets
        # pretrends_joint=None, all_pass=False, and the verdict carries
        # 'joint pre-trends skipped (no earlier pre-period)'. Both HAD
        # handler variants must surface this precondition - otherwise
        # agents reading the guidance can think any multi-period
        # event-study fit closes step 2 when it does not.
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            step_3_steps = [s for s in output["next_steps"] if s["baker_step"] == 3]
            assert len(step_3_steps) == 1
            text = (step_3_steps[0].get("why", "") + " " + step_3_steps[0].get("code", "")).lower()
            # Must mention "earlier" pre-period / placebo precondition.
            assert "earlier" in text and ("pre-period" in text or "placebo" in text), (
                "Step-3 text must mention the 'earlier pre-period' "
                "precondition for closing Assumption 7 / step 2 on the "
                "event-study path. With only the base F-1 pre-period "
                "the workflow returns pretrends_joint=None and the "
                "verdict carries 'joint pre-trends skipped (no earlier "
                "pre-period)' - step 2 stays uncovered."
            )
            # Must mention the skip-fallback verdict so agents know
            # what to expect when the precondition fails.
            assert "skipped" in text or "pretrends_joint=none" in text, (
                "Step-3 text must surface the 'joint pre-trends skipped' "
                "/ pretrends_joint=None fallback when no earlier "
                "pre-period exists - otherwise agents cannot tell "
                "whether step 2 was actually covered on a minimal "
                "event-study fit."
            )

    def test_had_step_3_flags_qug_under_survey_deferral(
        self, mock_had_results, mock_had_event_study_results
    ):
        # Per diff_diff/had_pretests.py:4488-4495 + REGISTRY § "QUG Null
        # Test" Note (Phase 4.5 C0): when survey_design= / survey= /
        # weights= is supplied, did_had_pretest_workflow skips the QUG
        # step with a UserWarning and returns a linearity-conditional
        # verdict only. Both HAD handler variants must surface this
        # caveat so agents do not assume step 1 / Design 1' vs Design 1
        # was checked on weighted fits when the library deliberately
        # cannot check it there.
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            step_3_steps = [s for s in output["next_steps"] if s["baker_step"] == 3]
            assert len(step_3_steps) == 1
            text = (step_3_steps[0].get("why", "") + " " + step_3_steps[0].get("code", "")).lower()
            # Must mention that survey-weighted fits skip QUG.
            assert "skip" in text and "qug" in text, (
                "Step-3 text must explicitly say survey-weighted fits "
                "skip QUG (Phase 4.5 C0 deferral). Without this caveat "
                "agents may assume step 1 / Design 1' vs Design 1 was "
                "checked on weighted fits when the library deliberately "
                "does not check it there."
            )
            # Must mention "linearity-conditional" verdict OR equivalent
            # framing so agents know the weighted verdict is conditional
            # on QUG holding by assumption.
            assert (
                "linearity-conditional" in text
                or "linearity conditional" in text
                or "qug holding by assumption" in text
            ), (
                "Step-3 text must describe the weighted verdict as "
                "linearity-conditional / conditional on QUG holding by "
                "assumption."
            )

    def test_had_step_3_qualifies_supported_survey_scope(
        self, mock_had_results, mock_had_event_study_results
    ):
        # Per diff_diff/had_pretests.py:1725-1740 + :1927-1940, only
        # pweight + PSU/FPC survey designs are supported on HAD
        # pretests. Stratified (SurveyDesign(strata=...)) and
        # replicate-weight (BRR/Fay/JK1/JKn/SDR) designs raise
        # NotImplementedError on the linearity kernels. Both HAD
        # handlers' Step-3 text must call out the supported subset
        # and the deferred regimes so agents don't generate
        # `practitioner_next_steps` outputs that overstate what the
        # workflow will run on a given survey design.
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            step_3_steps = [s for s in output["next_steps"] if s["baker_step"] == 3]
            assert len(step_3_steps) == 1
            text = step_3_steps[0].get("why", "").lower()
            # Supported subset must be named explicitly.
            assert "pweight" in text and "psu" in text and "fpc" in text, (
                "Step-3 text must name the supported survey-pretest scope "
                "(pweight + PSU/FPC) so agents do not assume any "
                "survey_design= path is supported."
            )
            # Deferred regimes must be flagged explicitly so agents
            # know not to attempt them.
            assert "stratif" in text, (
                "Step-3 text must explicitly note that stratified "
                "(SurveyDesign(strata=...)) survey designs are not yet "
                "supported on HAD pretests."
            )
            assert "replicate" in text, (
                "Step-3 text must explicitly note that replicate-weight "
                "(BRR/Fay/JK1/JKn/SDR) survey designs are not yet "
                "supported on HAD pretests."
            )
            assert "notimplementederror" in text, (
                "Step-3 text must name the actual exception raised "
                "(NotImplementedError) so agents can match it in "
                "error-handling paths."
            )

    def test_had_step_3_pretest_assumption_labels_correct(self, mock_had_results):
        # Per docs/methodology/REGISTRY.md and diff_diff/had_pretests.py
        # docstrings:
        #   - did_had_pretest_workflow(aggregate="overall") covers paper
        #     Section 4.2 steps 1 + 3 ONLY; step 2 (Assumption 7
        #     pre-trends) is explicitly NOT covered on the overall path.
        #   - qug_test = support-infimum test (H0: d_lower = 0),
        #     NOT "Assumption 5" (Design 1 sign identification, which is
        #     not testable per registry).
        #   - stute_test = Assumption 8 linearity, NOT Assumption 7
        #     mean-independence.
        # The single-period Step-3 guidance must not mislabel these.
        output = practitioner_next_steps(mock_had_results, verbose=False)
        step_3_steps = [s for s in output["next_steps"] if s["baker_step"] == 3]
        assert len(step_3_steps) == 1
        why = step_3_steps[0].get("why", "")
        # Must NOT call QUG an "Assumption 5" test.
        assert "QUG (Assumption 5" not in why, (
            "Step-3 why-text must not call QUG an 'Assumption 5' test - "
            "QUG tests H_0: d_lower = 0 (paper Theorem 4); Assumption 5 "
            "is the Design 1 sign-identification condition and is NOT "
            "testable per registry."
        )
        # Must NOT claim Stute is Assumption 7 mean-independence.
        forbidden = (
            "Stute (Assumption 7",
            "Stute / Yatchew-HR Assumption 7",
            "Assumption 7 mean-independence",
        )
        for phrase in forbidden:
            assert phrase not in why, (
                f"Step-3 why-text must not carry the phrase {phrase!r} - "
                f"stute_test / yatchew_hr_test are Assumption 8 linearity "
                f"tests (paper Section 4.2 step 3); Assumption 7 (pre-trends) "
                f"is paper step 2 and is NOT covered on the overall workflow "
                f"path - the workflow's verdict explicitly flags that gap."
            )
        # Must positively acknowledge the Assumption 7 / step 2 gap on
        # the overall path (not silently imply it's covered).
        assert "Assumption 7" in why or "step 2" in why, (
            "Step-3 why-text must explicitly mention Assumption 7 / step 2 "
            "to acknowledge the gap on the overall workflow path - "
            "agents reading the guidance must not assume the workflow "
            "covers what it does not cover."
        )


# ---------------------------------------------------------------------------
# ChangesInChanges / QDiD handler fixtures
# ---------------------------------------------------------------------------
def _make_cic_2x2_panel(n=60, seed=42):
    """Balanced 2x2 panel designed to fit warning-free.

    Continuous draws (no ties); treated pre-period outcomes strictly
    inside the INTERIOR (6%-94% quantiles) of the control pre-period
    distribution, so both the unconditional support check and the
    conditional 99-tau envelope check (which spans conditional quantiles
    0.01-0.99 only) pass; the control post-period is an exact +0.3 shift
    of the pre-period, so the QDiD counterfactual quantile curve
    Q7(y10) + Q7(y01) - Q7(y00) = Q7(y10) + 0.3 is monotone by
    construction. One independent numeric covariate for the
    conditional-fit fixtures.
    """
    rng = np.random.default_rng(seed)
    import pandas as pd

    y00 = rng.normal(0.0, 1.0, n)
    y01 = y00 + 0.3
    lo, hi = np.quantile(y00, [0.06, 0.94])
    y10 = np.linspace(lo, hi, n)
    y11 = np.linspace(lo, hi, n) + 0.5
    rows = []
    for i in range(n):
        rows.append({"unit": i, "treated": 0, "post": 0, "y": y00[i]})
        rows.append({"unit": i, "treated": 0, "post": 1, "y": y01[i]})
        rows.append({"unit": 1000 + i, "treated": 1, "post": 0, "y": y10[i]})
        rows.append({"unit": 1000 + i, "treated": 1, "post": 1, "y": y11[i]})
    df = pd.DataFrame(rows)
    df["x1"] = rng.normal(0.0, 1.0, len(df))
    return df


@pytest.fixture(scope="module")
def cic_2x2_data():
    return _make_cic_2x2_panel()


@pytest.fixture(scope="module")
def cic_fit_results(cic_2x2_data):
    # Panel unit-resampling cannot empty a (group, period) cell, so all
    # 20 seeded replicates are valid and the fixture is warning-free.
    est = ChangesInChanges(n_bootstrap=20, panel=True, seed=42)
    return est.fit(cic_2x2_data, outcome="y", treatment="treated", time="post", unit="unit")


@pytest.fixture(scope="module")
def qdid_fit_results(cic_2x2_data):
    est = QDiD(n_bootstrap=20, panel=True, seed=42)
    return est.fit(cic_2x2_data, outcome="y", treatment="treated", time="post", unit="unit")


@pytest.fixture(scope="module")
def cic_cov_fit_results(cic_2x2_data):
    # n_bootstrap=0 keeps the ~4k bootstrap quantile-regression LPs out
    # of the default suite (only the ~200 point-fit LPs run); the
    # disabled-inference practitioner warning is expected and asserted.
    est = ChangesInChanges(n_bootstrap=0)
    return est.fit(cic_2x2_data, outcome="y", treatment="treated", time="post", covariates=["x1"])


@pytest.fixture(scope="module")
def qdid_cov_fit_results(cic_2x2_data):
    est = QDiD(n_bootstrap=0)
    return est.fit(cic_2x2_data, outcome="y", treatment="treated", time="post", covariates=["x1"])


def _mock_cic(**fields):
    r = ChangesInChangesResults.__new__(ChangesInChangesResults)
    for k, v in fields.items():
        setattr(r, k, v)
    return r


class TestCiCHandler:
    """Dedicated ChangesInChanges / QDiD handler (shared results class)."""

    def _labels(self, output):
        return [s["label"] for s in output["next_steps"]]

    def _all_text(self, output):
        return " ".join(
            s["label"] + " " + s["why"] + " " + s.get("code", "") for s in output["next_steps"]
        )

    def test_cic_dispatch_and_display(self, cic_fit_results):
        output = practitioner_next_steps(cic_fit_results, verbose=False)
        assert output["estimator"] == "ChangesInChanges (CiC)"
        assert any("distributional identifying assumptions" in lbl for lbl in self._labels(output))

    def test_qdid_dispatch_and_display(self, qdid_fit_results):
        output = practitioner_next_steps(qdid_fit_results, verbose=False)
        assert output["estimator"] == "QDiD"

    def test_bare_mock_falls_to_cic_unconditional_branch(self):
        # Unknown/missing `estimator` kind: display falls to the static
        # map entry and the step set defaults to the CiC-unconditional
        # branch (locked via its marker step - the interior-range step
        # exists on no other branch).
        r = ChangesInChangesResults.__new__(ChangesInChangesResults)
        output = practitioner_next_steps(r, verbose=False)
        assert output["estimator"] == "ChangesInChanges / QDiD"
        assert any("interior point-identification range" in lbl for lbl in self._labels(output))

    def test_interior_range_step_only_on_unconditional_cic(
        self, cic_fit_results, cic_cov_fit_results, qdid_fit_results, qdid_cov_fit_results
    ):
        marker = "interior point-identification range"
        assert any(
            marker in lbl
            for lbl in self._labels(practitioner_next_steps(cic_fit_results, verbose=False))
        )
        for other in (cic_cov_fit_results, qdid_fit_results, qdid_cov_fit_results):
            output = practitioner_next_steps(other, verbose=False)
            assert not any(marker in lbl for lbl in self._labels(output))

    def test_envelope_step_only_on_covariate_cic(
        self, cic_fit_results, cic_cov_fit_results, qdid_fit_results, qdid_cov_fit_results
    ):
        # The conditional-envelope diagnostic exists on the CiC covariate
        # path only - the QDiD covariate path has no support diagnostic,
        # so its guidance must not claim one.
        marker = "envelope diagnostic"
        assert any(
            marker in lbl
            for lbl in self._labels(practitioner_next_steps(cic_cov_fit_results, verbose=False))
        )
        for other in (cic_fit_results, qdid_fit_results, qdid_cov_fit_results):
            output = practitioner_next_steps(other, verbose=False)
            assert not any(marker in lbl for lbl in self._labels(output))

    def test_prefer_cic_step_only_on_qdid(
        self, cic_fit_results, cic_cov_fit_results, qdid_fit_results, qdid_cov_fit_results
    ):
        marker = "Prefer ChangesInChanges over QDiD"
        for qdid_res in (qdid_fit_results, qdid_cov_fit_results):
            output = practitioner_next_steps(qdid_res, verbose=False)
            assert any(marker in lbl for lbl in self._labels(output))
        for cic_res in (cic_fit_results, cic_cov_fit_results):
            output = practitioner_next_steps(cic_res, verbose=False)
            assert not any(marker in lbl for lbl in self._labels(output))

    def test_qdid_monotonicity_moot_clause_present_under_covariates(self, qdid_cov_fit_results):
        # The footnote-21 check is unconditional-only; the covariate
        # branch guidance must say the check is moot there, not imply
        # it ran.
        output = practitioner_next_steps(qdid_cov_fit_results, verbose=False)
        assert "moot" in self._all_text(output)

    def test_covariate_comparison_direction(
        self, cic_fit_results, cic_cov_fit_results, qdid_fit_results, qdid_cov_fit_results
    ):
        # Covariate fits get "drop the covariates and compare";
        # unconditional fits get "add covariates if composition changed".
        for cov_res in (cic_cov_fit_results, qdid_cov_fit_results):
            labels = self._labels(practitioner_next_steps(cov_res, verbose=False))
            assert any("Report with and without covariates" in lbl for lbl in labels)
            assert not any("Re-estimate with covariates" in lbl for lbl in labels)
        for uncond_res in (cic_fit_results, qdid_fit_results):
            labels = self._labels(practitioner_next_steps(uncond_res, verbose=False))
            assert any("Re-estimate with covariates" in lbl for lbl in labels)
            assert not any("Report with and without covariates" in lbl for lbl in labels)

    def test_honest_did_never_recommended(
        self, cic_fit_results, cic_cov_fit_results, qdid_fit_results, qdid_cov_fit_results
    ):
        # compute_honest_did requires event-study effects, which this
        # results type does not carry - recommending it would send users
        # into a TypeError.
        for res in (cic_fit_results, cic_cov_fit_results, qdid_fit_results, qdid_cov_fit_results):
            text = self._all_text(practitioner_next_steps(res, verbose=False))
            assert "HonestDiD" not in text
            assert "compute_honest_did" not in text

    def test_snippets_are_valid_python_syntax(
        self, cic_fit_results, cic_cov_fit_results, qdid_fit_results, qdid_cov_fit_results
    ):
        import ast

        for res in (cic_fit_results, cic_cov_fit_results, qdid_fit_results, qdid_cov_fit_results):
            output = practitioner_next_steps(res, verbose=False)
            for step in output["next_steps"]:
                code = step.get("code", "")
                if not code.strip():
                    continue
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    pytest.fail(
                        f"Step {step['baker_step']} ({step['label']!r}) "
                        f"emits a code snippet that does not parse as "
                        f"valid Python: {e}\n\nSnippet:\n{code}"
                    )

    def test_healthy_fit_no_warnings(self, cic_fit_results):
        output = practitioner_next_steps(cic_fit_results, verbose=False)
        assert output["warnings"] == []

    def test_n_bootstrap_zero_warning(self, cic_cov_fit_results):
        output = practitioner_next_steps(cic_cov_fit_results, verbose=False)
        assert len(output["warnings"]) == 1
        assert "n_bootstrap=0" in output["warnings"][0]

    def test_failed_replicates_warning(self):
        r = _mock_cic(
            att=0.5, estimator="cic", covariates=None, n_bootstrap=200, n_bootstrap_valid=100
        )
        output = practitioner_next_steps(r, verbose=False)
        joined = " ".join(output["warnings"])
        assert "100 of 200" in joined
        assert "50%" in joined

    def test_minor_replicate_failures_below_threshold_no_warning(self):
        # 4/200 = 2% failed, below the 5% fit-time materiality threshold
        # (warn_bootstrap_failure_rate) that this surface mirrors.
        r = _mock_cic(
            att=0.5, estimator="cic", covariates=None, n_bootstrap=200, n_bootstrap_valid=196
        )
        output = practitioner_next_steps(r, verbose=False)
        assert output["warnings"] == []

    def test_nan_att_warning(self):
        r = _mock_cic(
            att=float("nan"),
            estimator="cic",
            covariates=None,
            n_bootstrap=200,
            n_bootstrap_valid=200,
        )
        output = practitioner_next_steps(r, verbose=False)
        assert any("NaN ATT" in w for w in output["warnings"])

    def test_completed_placebo_filters_placebo_step(self, cic_fit_results):
        output = practitioner_next_steps(
            cic_fit_results, completed_steps=["placebo"], verbose=False
        )
        assert not any("Placebo" in lbl for lbl in self._labels(output))
        # Other steps survive the filter
        assert any("interior point-identification range" in lbl for lbl in self._labels(output))

    def test_empty_list_covariates_mock_takes_unconditional_branch(self):
        # fit() normalizes covariates=[] to None, but hand-built results
        # may carry the empty list - the branch predicate is truthiness.
        r = _mock_cic(
            att=0.5, estimator="cic", covariates=[], n_bootstrap=200, n_bootstrap_valid=200
        )
        output = practitioner_next_steps(r, verbose=False)
        labels = [s["label"] for s in output["next_steps"]]
        assert any("interior point-identification range" in lbl for lbl in labels)
        assert not any("envelope diagnostic" in lbl for lbl in labels)


@pytest.fixture(scope="module")
def cic_cov_panel_fit_results(cic_2x2_data):
    # Panel AND covariates together: locks the combined snippet path
    # (panel=True + unit= + covariates= all mirrored). n_bootstrap=0
    # keeps the LP cost to the point fit.
    est = ChangesInChanges(n_bootstrap=0, panel=True)
    return est.fit(
        cic_2x2_data,
        outcome="y",
        treatment="treated",
        time="post",
        unit="unit",
        covariates=["x1"],
    )


class TestCiCHandlerSpecificationPropagation:
    """Refit snippets and Step 2 must mirror the fit's actual design."""

    def _steps(self, results, **kwargs):
        return practitioner_next_steps(results, verbose=False, **kwargs)["next_steps"]

    def _step_by_label(self, results, label_fragment):
        matches = [s for s in self._steps(results) if label_fragment in s["label"]]
        assert matches, f"no step with label containing {label_fragment!r}"
        return matches[0]

    def test_step2_override_cic(self, cic_fit_results):
        step2 = [s for s in self._steps(cic_fit_results) if s["baker_step"] == 2][0]
        assert "distributional" in step2["label"]
        assert "not a mean parallel-trends variant" in step2["why"]
        assert "monotone outcome model" in step2["why"]
        assert "parallel trends variant you are invoking" not in step2["why"]

    def test_step2_override_qdid(self, qdid_fit_results):
        step2 = [s for s in self._steps(qdid_fit_results) if s["baker_step"] == 2][0]
        assert "not a mean parallel-trends variant" in step2["why"]
        assert "FOUR" in step2["why"]

    def test_step2_generic_untouched_for_other_estimators(self, did_results):
        step2 = [s for s in self._steps(did_results) if s["baker_step"] == 2][0]
        assert "parallel trends variant" in step2["why"]

    def test_step2_override_still_filterable(self, cic_fit_results):
        steps = self._steps(cic_fit_results, completed_steps=["assumptions"])
        assert not any(s["baker_step"] == 2 for s in steps)

    def test_placebo_snippet_mirrors_covariates(self, cic_cov_fit_results):
        code = self._step_by_label(cic_cov_fit_results, "Placebo")["code"]
        assert "covariates=['x1']" in code

    def test_placebo_snippet_mirrors_panel(self, cic_fit_results):
        code = self._step_by_label(cic_fit_results, "Placebo")["code"]
        assert "panel=True" in code
        assert "unit=" in code

    def test_placebo_snippet_rcs_unconditional_has_neither(self, cic_cov_fit_results):
        # The covariate fixture is repeated cross-section: no panel args.
        code = self._step_by_label(cic_cov_fit_results, "Placebo")["code"]
        assert "panel=True" not in code

    def test_placebo_snippet_mirrors_panel_and_covariates_together(self, cic_cov_panel_fit_results):
        code = self._step_by_label(cic_cov_panel_fit_results, "Placebo")["code"]
        assert "covariates=['x1']" in code
        assert "panel=True" in code
        assert "unit=" in code

    def test_prefer_cic_snippet_mirrors_specification(self, qdid_cov_fit_results):
        # "Fit CiC as the primary" must run the SAME specification the
        # QDiD fit ran, not silently drop the covariates.
        code = self._step_by_label(qdid_cov_fit_results, "Prefer ChangesInChanges")["code"]
        assert "covariates=['x1']" in code

    def test_with_without_covariates_snippet_is_unconditional_by_design(self, cic_cov_fit_results):
        code = self._step_by_label(cic_cov_fit_results, "Report with and without")["code"]
        assert "covariates=" not in code
        assert "UNCONDITIONAL" in code

    def test_cross_estimator_snippet_mirrors_covariates(self, cic_cov_fit_results):
        code = self._step_by_label(cic_cov_fit_results, "Compare with QDiD")["code"]
        # Both the QDiD leg and the mean-DiD anchor carry the covariates.
        assert code.count("covariates=['x1']") == 2

    def test_qdid_uncond_anchor_keeps_population_equivalence(self, qdid_fit_results):
        step = self._step_by_label(qdid_fit_results, "Anchor against mean DiD")
        assert "population equivalence" in step["label"]
        assert "matches standard" in step["why"]

    def test_qdid_cov_anchor_drops_population_equivalence_claim(self, qdid_cov_fit_results):
        # The p. 447 equivalence is established for the unconditional
        # estimator only - the covariate branch must not inherit it.
        labels = [s["label"] for s in self._steps(qdid_cov_fit_results)]
        assert not any("population equivalence" in lbl for lbl in labels)
        step = self._step_by_label(qdid_cov_fit_results, "covariate-adjusted mean DiD")
        assert "descriptive anchor" in step["why"]
        assert "covariates=['x1']" in step["code"]
        assert "flags small cells" not in step["why"]

    def test_snippets_document_inference_normalization(self, cic_fit_results):
        # quantiles=/alpha= are intentionally NOT mirrored into refit
        # snippets (a 19-value grid inlined into guidance is unreadable;
        # neither changes the identifying specification) - the snippet
        # must SAY so rather than silently normalize, and must not call
        # seed=42 a "default" (the constructor default is seed=None).
        code = self._step_by_label(cic_fit_results, "Placebo")["code"]
        assert "carry over quantiles=/alpha= if you customized them" in code
        assert "seed=42 is illustrative" in code

    def test_cic_step3_does_not_require_mean_parallel_trends(self, cic_fit_results):
        # CI review R1 P1 lock: under a nonlinear outcome model, group
        # mean trends need NOT be parallel in a valid CiC design - the
        # guidance must not frame a means check as a screen CiC has to
        # pass.
        step3 = self._step_by_label(cic_fit_results, "distributional identifying assumptions")
        text = step3["why"] + " " + step3["code"]
        assert "necessary" not in text.lower()
        assert "NOT by itself evidence against CiC" in step3["why"]
        assert "descriptive" in step3["why"]

    def test_qdid_step3_keeps_meaningful_means_screen(self, qdid_fit_results):
        # QDiD's additive quantile model moves cell means additively
        # (population mean equivalence with DiD), so for QDiD a
        # pre-period mean-trend break IS evidence against the model.
        step3 = self._step_by_label(qdid_fit_results, "distributional identifying assumptions")
        assert "IS evidence against QDiD's model" in step3["why"]
        assert "meaningful" in step3["why"]

    def test_n_bootstrap_one_warning(self):
        # n_bootstrap=1 passes the disabled-inference check but cannot
        # clear the >= 2 valid-replicate SE gate: all inference is NaN.
        r = _mock_cic(att=0.5, estimator="cic", covariates=None, n_bootstrap=1, n_bootstrap_valid=1)
        output = practitioner_next_steps(r, verbose=False)
        assert any("n_bootstrap=1 cannot produce inference" in w for w in output["warnings"])

    def test_bootstrap_warnings_accept_numpy_scalars(self):
        r = _mock_cic(
            att=0.5,
            estimator="cic",
            covariates=None,
            n_bootstrap=np.int64(200),
            n_bootstrap_valid=np.int64(100),
        )
        output = practitioner_next_steps(r, verbose=False)
        assert any("100 of 200" in w for w in output["warnings"])
        r0 = _mock_cic(att=0.5, estimator="cic", covariates=None, n_bootstrap=np.int64(0))
        output0 = practitioner_next_steps(r0, verbose=False)
        assert any("n_bootstrap=0" in w for w in output0["warnings"])


class TestEmittedGuidanceCanonicalNames:
    """2(d) PR-A per-site pins: emitted recommendation strings use full
    class names (user decision 2026-08-06); the ``_ESTIMATOR_NAMES``
    display map keeps kept-alias parentheticals but drops the DYING
    Gardner one. Per-site assertions only - method/citation prose
    (e.g. "Gardner 2022", "Stacked DiD") is deliberately untouched, so
    no module-wide alias ban is asserted.
    """

    def test_compare_step_alternatives_use_class_names(self):
        # Every _robustness_compare_step call site was migrated to full
        # class names; the constructor embeds `alternatives` verbatim in
        # the emitted label and code payloads.
        import inspect

        import diff_diff.practitioner as practitioner_mod

        source = inspect.getsource(practitioner_mod)
        migrated = [
            '_robustness_compare_step("CallawaySantAnna, SunAbraham, or ImputationDiD")',
            '_robustness_compare_step("SunAbraham, ImputationDiD, or TwoStageDiD")',
            '_robustness_compare_step("CallawaySantAnna, ImputationDiD, or TwoStageDiD")',
            '_robustness_compare_step("CallawaySantAnna, SunAbraham, or TwoStageDiD")',
            '_robustness_compare_step("CallawaySantAnna, ImputationDiD, or SunAbraham")',
            '_robustness_compare_step("SyntheticDiD or CallawaySantAnna")',
        ]
        for call in migrated:
            assert call in source, f"missing migrated compare-step call: {call}"
        # The old shorthand rosters are gone from compare-step calls.
        import re

        for stale in re.findall(r"_robustness_compare_step\(\"([^\"]+)\"\)", source):
            for tok in ("CS", "SA", "BJS", "Gardner"):
                assert not re.search(
                    rf"\b{tok}\b", stale
                ), f"compare-step roster still uses alias shorthand: {stale!r}"

    def test_emitted_compare_step_label_uses_class_names(self):
        # Behavioral arm: the constructed step embeds the roster in the
        # emitted label/code strings.
        from diff_diff.practitioner import _robustness_compare_step

        step = _robustness_compare_step("CallawaySantAnna, SunAbraham, or ImputationDiD")
        assert "CallawaySantAnna, SunAbraham, or ImputationDiD" in step["label"]
        assert "CallawaySantAnna, SunAbraham, or ImputationDiD" in step["code"]

    def test_falsification_why_payloads_use_class_names(self):
        import inspect

        import diff_diff.practitioner as practitioner_mod

        source = inspect.getsource(practitioner_mod)
        assert "compare with CS/SA as" not in source
        assert source.count("compare with CallawaySantAnna/") == 2  # BJS + Gardner handler twins
        assert "Use CS, SA, BJS, or another" not in source
        assert "Use CallawaySantAnna, SunAbraham, " in source

    def test_staggered_comparison_label_uses_class_names(self):
        import inspect

        import diff_diff.practitioner as practitioner_mod

        source = inspect.getsource(practitioner_mod)
        assert 'label="Compare with staggered estimators (CallawaySantAnna, SunAbraham)"' in source
        assert "(CS, SA)" not in source

    def test_estimator_display_map_drops_dying_gardner_parenthetical(self):
        # Tier (ii): the display map keeps KEPT-alias parentheticals
        # (DDD/HAD/CiC - pinned verbatim elsewhere) but carries no DYING
        # alias token.
        import re

        from diff_diff.practitioner import _ESTIMATOR_NAMES

        assert _ESTIMATOR_NAMES["TwoStageDiDResults"] == "TwoStageDiD"
        for value in _ESTIMATOR_NAMES.values():
            assert "Gardner" not in value, value
            assert "CDiD" not in value, value
            # `Stacked` only as part of StackedDiD (no "(Stacked)" parenthetical).
            assert not re.search(r"\bStacked\b(?! ?DiD)", value.replace("StackedDiD", "")), value
        # Kept parentheticals stay (deliberate kept-alias documentation).
        assert _ESTIMATOR_NAMES["TripleDifferenceResults"] == "TripleDifference (DDD)"
        assert _ESTIMATOR_NAMES["HeterogeneousAdoptionDiDResults"] == (
            "HeterogeneousAdoptionDiD (HAD)"
        )
