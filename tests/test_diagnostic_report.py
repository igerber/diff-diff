"""Tests for ``diff_diff.diagnostic_report.DiagnosticReport``.

Covers:
- Schema contract: every top-level key always present, stable enum values.
- Applicability matrix: per-estimator ``applicable_checks`` property.
- JSON round-trip.
- ``precomputed=`` passthrough (sensitivity).
- Pre-trends verdict thresholds (three bins).
- Power-aware tier thresholds (three bins + fallback).
- DEFF reads from ``survey_metadata`` when present.
- EfficientDiD ``hausman_pretest`` pathway.
- SDiD / TROP native diagnostics.
- Error-doesn't-break-report (diagnostic raises -> section records error).
"""

from __future__ import annotations

import json
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import diff_diff as dd
from diff_diff import (
    CallawaySantAnna,
    DiagnosticReport,
    DiagnosticReportResults,
    DifferenceInDifferences,
    EfficientDiD,
    MultiPeriodDiD,
    SyntheticDiD,
    generate_did_data,
    generate_factor_data,
    generate_staggered_data,
    synthetic_control,
)
from diff_diff.diagnostic_report import (
    DIAGNOSTIC_REPORT_SCHEMA_VERSION,
    _power_tier,
    _pt_verdict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_TOP_LEVEL_KEYS = {
    "schema_version",
    "estimator",
    "headline_metric",
    "target_parameter",
    "parallel_trends",
    "pretrends_power",
    "sensitivity",
    "placebo",
    "bacon",
    "design_effect",
    "heterogeneity",
    "epv",
    "estimator_native_diagnostics",
    "skipped",
    "warnings",
    "overall_interpretation",
    "next_steps",
}

_STATUS_ENUM = {
    "ran",
    "skipped",
    "error",
    "not_applicable",
    "not_run",
    "computed",
}


@pytest.fixture(scope="module")
def did_fit():
    warnings.filterwarnings("ignore")
    df = generate_did_data(n_units=80, n_periods=4, treatment_effect=1.5, seed=7)
    did = DifferenceInDifferences().fit(df, outcome="outcome", treatment="treated", post="post")
    return did, df


@pytest.fixture(scope="module")
def multi_period_fit():
    warnings.filterwarnings("ignore")
    df = generate_did_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
    es = MultiPeriodDiD().fit(
        df,
        outcome="outcome",
        treatment="treated",
        time="period",
        unit="unit",
        reference_period=3,
    )
    return es, df


@pytest.fixture(scope="module")
def cs_fit():
    warnings.filterwarnings("ignore")
    sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
    # Use base_period='universal' so HonestDiD sensitivity can run on this
    # fixture. CS's default is 'varying', which DR now skips with a
    # methodology-critical reason (Rambachan-Roth bounds are not valid for
    # interpretation on consecutive-comparison pre-periods). See the
    # round-5 CI review on PR #318.
    cs = CallawaySantAnna(base_period="universal").fit(
        sdf,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
        aggregate="event_study",
    )
    return cs, sdf


@pytest.fixture(scope="module")
def edid_fit():
    warnings.filterwarnings("ignore")
    sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
    edid = EfficientDiD().fit(
        sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
    )
    return edid, sdf


@pytest.fixture(scope="module")
def sdid_fit():
    warnings.filterwarnings("ignore")
    fdf = generate_factor_data(n_units=25, n_pre=8, n_post=4, n_treated=4, seed=11)
    sdid = SyntheticDiD().fit(fdf, outcome="outcome", unit="unit", time="period", treatment="treat")
    return sdid, fdf


def _scm_panel(n_donors=4, T=8, T0=6, effect=3.0, seed=0):
    """Small balanced SCM panel: treated = convex mix of two donors + post effect."""
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + T))
    donors = {
        j: rng.normal(10, 2) + rng.normal(0, 0.3) * np.arange(T) + rng.normal(0, 0.15, T)
        for j in range(n_donors)
    }
    treated = 0.6 * donors[0] + 0.4 * donors[1] + rng.normal(0, 0.08, T)
    treated[T0:] += effect
    rows = []
    for j in range(n_donors):
        for t in range(T):
            rows.append({"unit": f"d{j}", "year": years[t], "y": donors[j][t], "treated": 0})
    for t in range(T):
        rows.append({"unit": "treated", "year": years[t], "y": treated[t], "treated": int(t >= T0)})
    return pd.DataFrame(rows)


@pytest.fixture  # function-scoped: some tests run in_space_placebo(), which mutates the result
def scm_fit():
    df = _scm_panel(n_donors=4)
    # Cheap optimizer settings keep the pure-Python fixture fast; the report path,
    # not solver accuracy, is what these tests exercise. Scope the warning
    # suppression to this call via catch_warnings() so it does not mutate the
    # global filter state for later tests in the same worker.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            seed=0,
            n_starts=1,
            optimizer_options={"maxiter": 50},
            inner_min_decrease=1e-3,
        )
    return res, df


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
class TestSchemaContract:
    """The AI-legible schema is the public promise. These tests lock it down."""

    def test_every_top_level_key_present_did(self, did_fit):
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df, outcome="outcome", treatment="treated", time="post")
        schema = dr.to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_every_top_level_key_present_multiperiod(self, multi_period_fit):
        fit, _ = multi_period_fit
        schema = DiagnosticReport(fit).to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_every_top_level_key_present_cs(self, cs_fit):
        fit, sdf = cs_fit
        schema = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        ).to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_every_top_level_key_present_sdid(self, sdid_fit):
        fit, _ = sdid_fit
        schema = DiagnosticReport(fit).to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_schema_version_constant(self, multi_period_fit):
        fit, _ = multi_period_fit
        schema = DiagnosticReport(fit).to_dict()
        assert schema["schema_version"] == DIAGNOSTIC_REPORT_SCHEMA_VERSION
        assert DIAGNOSTIC_REPORT_SCHEMA_VERSION == "2.0"

    def test_all_statuses_use_closed_enum(self, cs_fit):
        fit, sdf = cs_fit
        schema = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        ).to_dict()
        for key in [
            "parallel_trends",
            "pretrends_power",
            "sensitivity",
            "placebo",
            "bacon",
            "design_effect",
            "heterogeneity",
            "epv",
            "estimator_native_diagnostics",
        ]:
            section = schema.get(key)
            assert isinstance(section, dict), f"{key} missing"
            assert (
                section.get("status") in _STATUS_ENUM
            ), f"{key}.status = {section.get('status')!r} not in {_STATUS_ENUM}"

    def test_json_round_trip_multiperiod(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        dumped = json.dumps(dr.to_dict())
        assert len(dumped) > 0
        round = json.loads(dumped)
        assert round["schema_version"] == DIAGNOSTIC_REPORT_SCHEMA_VERSION

    def test_json_round_trip_sdid(self, sdid_fit):
        fit, _ = sdid_fit
        dumped = json.dumps(DiagnosticReport(fit).to_dict())
        assert len(dumped) > 0


# ---------------------------------------------------------------------------
# Applicability matrix
# ---------------------------------------------------------------------------
class TestApplicabilityMatrix:
    """Per-estimator applicability set filtered by instance state + options."""

    def test_did_without_data_skips_pt(self, did_fit):
        fit, _ = did_fit
        dr = DiagnosticReport(fit)  # no data
        assert "parallel_trends" not in dr.applicable_checks
        assert "parallel_trends" in dr.skipped_checks
        reason = dr.skipped_checks["parallel_trends"]
        assert "data" in reason.lower()

    def test_did_with_data_runs_pt(self, did_fit):
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df, outcome="outcome", treatment="treated", time="post")
        assert "parallel_trends" in dr.applicable_checks

    def test_did_with_data_but_no_column_kwargs_skips_pt(self, did_fit):
        """Round-11 regression: ``applicable_checks`` must match the
        runner's full argument contract. 2x2 PT needs ``data`` AND
        ``outcome`` / ``time`` / ``treatment`` — not just ``data``."""
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df)  # missing column kwargs
        assert "parallel_trends" not in dr.applicable_checks
        reason = dr.skipped_checks["parallel_trends"]
        assert "outcome" in reason
        assert "time" in reason
        assert "treatment" in reason

    def test_bacon_applicability_requires_all_column_kwargs(self, cs_fit):
        """Round-11 regression: Bacon needs the full ``outcome`` / ``time``
        / ``unit`` / ``first_treat`` contract from ``bacon_decompose``."""
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            first_treat="first_treat",
            # intentionally omit outcome / time / unit
        )
        assert "bacon" not in dr.applicable_checks
        reason = dr.skipped_checks["bacon"]
        assert "outcome" in reason or "time" in reason or "unit" in reason

    def test_multiperiod_runs_pt_and_power_and_sensitivity(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        applicable = set(dr.applicable_checks)
        assert "parallel_trends" in applicable
        assert "pretrends_power" in applicable
        assert "sensitivity" in applicable

    def test_cs_runs_heterogeneity(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        applicable = set(dr.applicable_checks)
        assert "heterogeneity" in applicable
        assert "bacon" in applicable
        assert "parallel_trends" in applicable

    def test_sdid_has_estimator_native(self, sdid_fit):
        fit, _ = sdid_fit
        dr = DiagnosticReport(fit)
        assert "estimator_native" in dr.applicable_checks

    def test_run_opt_outs_move_checks_to_skipped(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit, run_sensitivity=False)
        assert "sensitivity" not in dr.applicable_checks
        assert dr.skipped_checks["sensitivity"].startswith("run_sensitivity=False")

    def test_placebo_is_reserved_and_skipped(self, did_fit):
        """Placebo is always in _CHECK_NAMES, always skipped in MVP.

        Round-26 P3: tightened from ``status in {"skipped",
        "not_applicable"}`` to exact ``status == "skipped"`` because
        both REPORTING.md §MVP scope and the implementation
        (``_compute_applicable_checks`` always seeds ``"placebo"`` into
        ``skipped``) now pin the MVP contract to a single value.
        """
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df, outcome="outcome", treatment="treated", time="post")
        placebo_section = dr.to_dict()["placebo"]
        assert placebo_section["status"] == "skipped"
        assert isinstance(placebo_section.get("reason"), str) and placebo_section["reason"]


# ---------------------------------------------------------------------------
# Precomputed passthrough
# ---------------------------------------------------------------------------
class TestPrecomputed:
    def test_precomputed_sensitivity_is_used_verbatim(self, multi_period_fit):
        fit, _ = multi_period_fit

        # Construct a minimal SensitivityResults-shaped object the formatter recognizes.
        class _FakeSens:
            M_values = np.array([0.5, 1.0])
            bounds = [(0.1, 2.0), (-0.2, 2.5)]
            robust_cis = [(0.05, 2.1), (-0.3, 2.6)]
            breakdown_M = 0.75
            method = "relative_magnitude"
            original_estimate = 1.0
            original_se = 0.2
            alpha = 0.05

        fake = _FakeSens()
        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis") as mock:
            dr = DiagnosticReport(fit, precomputed={"sensitivity": fake})
            dr.to_dict()
            mock.assert_not_called()
        schema = dr.to_dict()
        assert schema["sensitivity"]["status"] == "ran"
        assert schema["sensitivity"]["breakdown_M"] == 0.75

    def test_precomputed_pretrends_power_parity_with_default_path(self, cs_fit):
        """Round-20 P1 regression: ``precomputed={"pretrends_power": ...}``
        must apply the same covariance-source annotation and conservative
        diagonal-fallback downgrade as ``_check_pretrends_power``. Otherwise
        the same fit can be labeled ``well_powered`` through the precomputed
        path and ``moderately_powered`` through the default path.
        """
        from diff_diff.pretrends import compute_pretrends_power

        fit, data = cs_fit

        # Precompute the power result from the same fit. The compute function
        # populates ``original_results`` on the output so DR's precomputed
        # adapter can inspect the source fit's event_study_vcov.
        pp = compute_pretrends_power(fit, alpha=0.05, target_power=0.80, violation_type="linear")
        assert getattr(pp, "original_results", None) is fit

        dr_default = DiagnosticReport(fit, data=data).to_dict()
        dr_precomputed = DiagnosticReport(
            fit, data=data, precomputed={"pretrends_power": pp}
        ).to_dict()

        default_block = dr_default["pretrends_power"]
        precomp_block = dr_precomputed["pretrends_power"]

        # Both paths are "ran"; the precomputed path flags itself with
        # ``precomputed=True`` while the default path sets ``method=
        # compute_pretrends_power``.
        assert default_block["status"] == "ran"
        assert precomp_block["status"] == "ran"
        assert precomp_block.get("precomputed") is True

        # Tier and covariance_source must agree across paths so downstream
        # BR prose does not diverge based on which path produced the block.
        assert default_block["tier"] == precomp_block["tier"]
        assert default_block["covariance_source"] == precomp_block["covariance_source"]

    def test_precomputed_pretrends_power_persisted_full_vcov_no_downgrade(self):
        """PR-B R3+R4 regression: a precomputed ``PreTrendsPowerResults``
        carrying ``covariance_source='full_pre_period_vcov'`` (the value
        ``compute_pretrends_power`` records post-PR-B) must NOT be
        downgraded by ``DiagnosticReport``. Locks the new contract that
        full-VCV CS / SA fits keep their ``well_powered`` tier.
        """
        from diff_diff.pretrends import PreTrendsPowerResults

        class _CSStub:
            overall_att = 1.0
            overall_se = 0.25
            overall_t_stat = 4.0
            overall_p_value = 0.001
            overall_conf_int = (0.5, 1.5)
            alpha = 0.05
            n_obs = 400
            n_treated = 80
            n_control = 320
            survey_metadata = None
            event_study_effects = None
            event_study_vcov = np.eye(3)
            event_study_vcov_index = {-2: 0, -1: 1, 0: 2}

        stub = _CSStub()
        stub.__class__.__name__ = "CallawaySantAnnaResults"

        pp = PreTrendsPowerResults(
            power=0.80,
            mdv=0.1,
            violation_magnitude=0.1,
            violation_type="linear",
            alpha=0.05,
            target_power=0.80,
            n_pre_periods=2,
            test_statistic=np.nan,
            critical_value=1.96,
            noncentrality=np.nan,
            pre_period_effects=np.zeros(2),
            pre_period_ses=np.ones(2),
            vcov=np.eye(2),
            original_results=stub,
            covariance_source="full_pre_period_vcov",
        )

        dr = DiagnosticReport(stub, precomputed={"pretrends_power": pp})
        block = dr.to_dict()["pretrends_power"]
        assert block["status"] == "ran"
        assert block["covariance_source"] == "full_pre_period_vcov"
        assert block["tier"] == "well_powered"

    def test_precomputed_pretrends_power_legacy_missing_field_still_downgraded(self):
        """R4 regression: legacy ``PreTrendsPowerResults`` pre-PR-B has no
        ``covariance_source`` field. We cannot tell from the source-fit
        object whether the stored power was computed from full Σ_22 or
        from the diag fallback (PR-A behavior was diag even when
        ``event_study_vcov`` was attached). The adapter MUST treat the
        missing-field case as legacy-ambiguous and apply the conservative
        downgrade — otherwise an old serialized CS result silently
        upgrades to ``well_powered``.

        Pairs with the
        ``test_precomputed_pretrends_power_persisted_full_vcov_no_downgrade``
        positive case to lock both legs of the legacy-fallback contract.
        """

        # Minimal CS-shaped stub with full vcov populated.
        class _CSStub:
            overall_att = 1.0
            overall_se = 0.25
            overall_t_stat = 4.0
            overall_p_value = 0.001
            overall_conf_int = (0.5, 1.5)
            alpha = 0.05
            n_obs = 400
            n_treated = 80
            n_control = 320
            survey_metadata = None
            event_study_effects = None
            event_study_vcov = np.eye(3)
            event_study_vcov_index = {-2: 0, -1: 1, 0: 2}

        stub = _CSStub()
        stub.__class__.__name__ = "CallawaySantAnnaResults"

        class _LegacyPPStub:
            mdv = 0.1
            violation_type = "linear"
            alpha = 0.05
            target_power = 0.80
            violation_magnitude = 0.1
            power = 0.80
            n_pre_periods = 2
            original_results = stub
            # No covariance_source attribute — simulates an old serialized
            # PreTrendsPowerResults from a pre-PR-B fit.

        dr = DiagnosticReport(stub, precomputed={"pretrends_power": _LegacyPPStub()})
        block = dr.to_dict()["pretrends_power"]
        assert block["status"] == "ran"
        # Legacy-ambiguous → conservative sentinel + downgrade applies.
        assert block["covariance_source"] == "diag_fallback_available_full_vcov_unused"
        assert block["tier"] == "moderately_powered"

    def test_precomputed_pretrends_power_legacy_mpd_without_interaction_indices_reports_diag(
        self,
    ):
        """PR-B R5 regression: ``MultiPeriodDiDResults`` legacy fits without
        ``interaction_indices`` truly take the ``np.diag(ses**2)`` fallback
        inside ``pretrends.py:_extract_pre_period_params`` MPD branch. The
        report-layer's ``_infer_cov_source`` fallback must surface that
        accurately as ``"diag_fallback"`` rather than overclaiming
        ``"full_pre_period_vcov"`` (MPD is not in the event-study type set,
        so the previous non-event-study branch unconditionally returned
        ``"full_pre_period_vcov"`` — wrong for MPD without interaction
        indices).
        """

        class _LegacyMPDStub:
            avg_att = 1.0
            avg_se = 0.25
            avg_t_stat = 4.0
            avg_p_value = 0.001
            avg_conf_int = (0.5, 1.5)
            alpha = 0.05
            n_obs = 400
            n_treated = 80
            n_control = 320
            survey_metadata = None
            # No interaction_indices, no full vcov — pretrends.py MPD
            # branch falls through to diag(ses**2).
            vcov = None
            interaction_indices = None

        stub = _LegacyMPDStub()
        stub.__class__.__name__ = "MultiPeriodDiDResults"

        class _LegacyPPStub:
            mdv = 0.1
            violation_type = "linear"
            alpha = 0.05
            target_power = 0.80
            violation_magnitude = 0.1
            power = 0.80
            n_pre_periods = 2
            original_results = stub
            # Legacy — no covariance_source field set.

        dr = DiagnosticReport(stub, precomputed={"pretrends_power": _LegacyPPStub()})
        block = dr.to_dict()["pretrends_power"]
        assert block["status"] == "ran"
        # Legacy MPD without interaction_indices reports diag_fallback —
        # the conservative downgrade does NOT fire (this isn't an
        # "available but unused" case, just a normal fallback).
        assert block["covariance_source"] == "diag_fallback"
        # No downgrade applies on "diag_fallback" (vs the sentinel label).
        assert block["tier"] == "well_powered"

    def test_precomputed_pretrends_power_consumes_persisted_cov_source(self):
        """PR-B R3 regression: the precomputed adapter must prefer the
        ``covariance_source`` recorded on ``PreTrendsPowerResults`` over
        the legacy type-based inference. Demonstrates the architectural
        fix the R3 codex review called out (provenance should be recorded
        on the result, not re-inferred from result type each time).

        Constructs a stub fit whose source-side type-based inference would
        produce the LEGACY conservative downgrade label
        ``diag_fallback_available_full_vcov_unused`` — and verifies that
        the explicit persisted ``full_pre_period_vcov`` label wins,
        keeping the ``well_powered`` tier. The legacy fallback only
        activates when the persisted field is missing or ``"unknown"``.
        """
        from diff_diff.pretrends import PreTrendsPowerResults

        class _CSStub:
            overall_att = 1.0
            overall_se = 0.25
            overall_t_stat = 4.0
            overall_p_value = 0.001
            overall_conf_int = (0.5, 1.5)
            alpha = 0.05
            n_obs = 400
            n_treated = 80
            n_control = 320
            survey_metadata = None
            event_study_effects = None
            event_study_vcov = np.eye(3)
            event_study_vcov_index = {-2: 0, -1: 1, 0: 2}

        stub = _CSStub()
        stub.__class__.__name__ = "CallawaySantAnnaResults"

        pp = PreTrendsPowerResults(
            power=0.80,
            mdv=0.1,
            violation_magnitude=0.1,
            violation_type="linear",
            alpha=0.05,
            target_power=0.80,
            n_pre_periods=2,
            test_statistic=np.nan,
            critical_value=1.96,
            noncentrality=np.nan,
            pre_period_effects=np.zeros(2),
            pre_period_ses=np.ones(2),
            vcov=np.eye(2),
            original_results=stub,
            covariance_source="full_pre_period_vcov",
        )

        dr = DiagnosticReport(stub, precomputed={"pretrends_power": pp})
        block = dr.to_dict()["pretrends_power"]
        assert block["covariance_source"] == "full_pre_period_vcov"
        assert block["tier"] == "well_powered"

    def test_precomputed_parallel_trends_bypasses_applicability_gate(self, cs_fit):
        """Round-22 P1 regression: ``precomputed["parallel_trends"]`` was
        documented as supported but ``_instance_skip_reason`` skipped the
        PT check on applicability grounds (missing raw panel / columns
        for the event-study replay, non-replayable EfficientDiD fits,
        etc.) BEFORE the precomputed runner could fire. The fix
        short-circuits the gate when the precomputed key is present so
        advertised passthroughs actually land on the runner.
        """
        fit, _ = cs_fit
        precomputed_pt = {
            "status": "ran",
            "method": "event_study",
            "joint_p_value": 0.42,
            "n_pre_periods": 3,
            "verdict": "no_detected_violation",
        }

        # Without passing ``data`` + column kwargs, the applicability
        # gate would previously have marked PT as skipped. With the
        # precomputed override, it must land on the formatter instead.
        dr = DiagnosticReport(fit, precomputed={"parallel_trends": precomputed_pt})
        pt_block = dr.to_dict()["parallel_trends"]
        assert pt_block["status"] == "ran", (
            f"precomputed parallel_trends must bypass the applicability gate. "
            f"Got status={pt_block.get('status')}, reason={pt_block.get('reason')}"
        )

    def test_precomputed_parallel_trends_preserves_schema_shaped_joint_p(self, cs_fit):
        """Round-23 P1 regression: schema-shaped PT dicts with
        ``joint_p_value`` (the key emitted by the default DR path and
        the shape users are most likely to replay from one DR to
        another) must land on ``joint_p_value`` in the output, not
        silently fall through to ``None``. Prior formatter read only
        ``p_value``, so a dict with ``joint_p_value=0.42`` was
        degraded to ``joint_p_value=None`` / ``verdict="inconclusive"``.
        """
        fit, _ = cs_fit
        dr = DiagnosticReport(
            fit,
            precomputed={
                "parallel_trends": {
                    "joint_p_value": 0.42,
                    "test_statistic": 5.6,
                    "df": 3,
                    "method": "hausman",
                }
            },
        )
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "hausman"
        assert (
            pt["joint_p_value"] == 0.42
        ), f"joint_p_value must survive formatting; got {pt.get('joint_p_value')}"
        assert pt["test_statistic"] == 5.6
        assert pt["df"] == 3
        # Verdict must be derived from the surviving p-value, not None.
        assert pt["verdict"] != "inconclusive"

    def test_precomputed_parallel_trends_accepts_native_hausman_result(self, cs_fit):
        """Round-23 P1 regression: ``_pt_hausman`` tells users with
        non-replayable EfficientDiD fits to pass a precomputed pretest
        result, but the formatter previously rejected non-dict inputs
        outright. The ``HausmanPretestResult`` dataclass — the exact
        object ``EfficientDiD.hausman_pretest(...)`` returns — must
        now pass through with ``statistic`` / ``p_value`` / ``df``
        preserved on the schema.
        """
        from types import SimpleNamespace

        fit, _ = cs_fit
        # Mirror HausmanPretestResult: the key fields are ``statistic``,
        # ``p_value``, ``df``. Uses SimpleNamespace so the test does
        # not need EfficientDiD's construction path.
        hausman = SimpleNamespace(
            statistic=7.2,
            p_value=0.065,
            df=3,
            reject=False,
            alpha=0.05,
            att_all=1.0,
            att_post=1.05,
            recommendation="pt_all",
        )
        dr = DiagnosticReport(fit, precomputed={"parallel_trends": hausman})
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran", (
            f"Native HausmanPretestResult must be accepted; got "
            f"status={pt.get('status')}, reason={pt.get('reason')}"
        )
        assert pt["joint_p_value"] == 0.065
        # ``statistic`` on the source object maps to ``test_statistic``
        # in the emitted schema (matches the default ``_pt_hausman``
        # path that also exposes it as ``test_statistic``).
        assert pt["test_statistic"] == 7.2
        assert pt["df"] == 3

    def test_precomputed_pt_infers_slope_difference_method_for_raw_2x2_dict(self, cs_fit):
        """Round-26 P2 regression: a raw ``utils.check_parallel_trends()``
        dict (no ``method`` key, has ``trend_difference`` / p_value) must
        be recognized as the slope-difference 2x2 path and render with
        the single-statistic ``p`` label, not the generic ``joint p``
        wording that ``"precomputed"`` falls through to.
        """
        from diff_diff import BusinessReport
        from diff_diff.diagnostic_report import DiagnosticReportResults

        fit, _ = cs_fit
        raw_2x2 = {
            "treated_trend": 0.1,
            "treated_trend_se": 0.05,
            "control_trend": 0.08,
            "control_trend_se": 0.04,
            "trend_difference": 0.02,
            "trend_difference_se": 0.06,
            "t_statistic": 0.33,
            "p_value": 0.40,
            "parallel_trends_plausible": True,
        }
        dr = DiagnosticReport(fit, precomputed={"parallel_trends": raw_2x2})
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "slope_difference", (
            f"Raw check_parallel_trends dict must infer "
            f"method='slope_difference'; got {pt.get('method')!r}"
        )

        # Markdown prose must use the single-statistic ``p`` label
        # (not ``joint p``, which is Wald / Bonferroni-specific).
        br_dr = DiagnosticReportResults(
            schema=dr.to_dict(),
            interpretation="",
            applicable_checks=("parallel_trends",),
            skipped_checks={},
            warnings=(),
        )
        md = BusinessReport(fit, diagnostics=br_dr).full_report()
        pt_section = md.split("## Pre-Trends", 1)[1].split("\n## ", 1)[0]
        assert "joint p" not in pt_section
        assert "p = 0.4" in pt_section

    def test_precomputed_pt_infers_hausman_method_for_native_object(self, cs_fit):
        """Round-26 P2 regression: a native Hausman-like object without
        an explicit ``method`` tag (``HausmanPretestResult`` shape:
        ``statistic`` + ``att_all`` / ``att_post`` / ``recommendation``)
        must be recognized as the Hausman path and render with the
        single-statistic ``p`` label, not ``joint p``.
        """
        from types import SimpleNamespace

        from diff_diff import BusinessReport
        from diff_diff.diagnostic_report import DiagnosticReportResults

        fit, _ = cs_fit
        hausman_like = SimpleNamespace(
            statistic=4.5,
            p_value=0.21,
            df=3,
            reject=False,
            alpha=0.05,
            att_all=1.0,
            att_post=1.1,
            recommendation="pt_all",
            # Note: no ``method`` attribute — tests the inference path.
        )
        dr = DiagnosticReport(fit, precomputed={"parallel_trends": hausman_like})
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "hausman", (
            f"Native Hausman-like object must infer method='hausman'; " f"got {pt.get('method')!r}"
        )
        assert pt["test_statistic"] == 4.5
        assert pt["joint_p_value"] == 0.21

        # Markdown prose must use the single-statistic ``p`` label.
        br_dr = DiagnosticReportResults(
            schema=dr.to_dict(),
            interpretation="",
            applicable_checks=("parallel_trends",),
            skipped_checks={},
            warnings=(),
        )
        md = BusinessReport(fit, diagnostics=br_dr).full_report()
        pt_section = md.split("## Pre-Trends", 1)[1].split("\n## ", 1)[0]
        assert "joint p" not in pt_section
        assert "p = 0.21" in pt_section

    def test_precomputed_pt_explicit_method_wins_over_inference(self, cs_fit):
        """Explicit ``method`` in the input must never be overridden by
        the heuristic inference (defensive: e.g., a user passes a
        schema-shaped dict labeled ``method='event_study'`` where the
        ``trend_difference`` markers would otherwise suggest
        slope_difference).
        """
        fit, _ = cs_fit
        spoofed = {
            "method": "event_study",
            "joint_p_value": 0.42,
            "trend_difference": 0.02,  # would otherwise trigger slope_difference inference
        }
        dr = DiagnosticReport(fit, precomputed={"parallel_trends": spoofed})
        assert dr.to_dict()["parallel_trends"]["method"] == "event_study"

    def test_precomputed_parallel_trends_rejects_input_without_p_value(self, cs_fit):
        """Inputs without any recognized p-value field (neither
        ``joint_p_value`` nor ``p_value``) must surface a clear error,
        not silently land on ``joint_p_value=None``. Keeps the formatter
        permissive about absent ``test_statistic`` / ``df`` (2x2 PT has
        neither) while catching obviously-wrong inputs.
        """
        fit, _ = cs_fit
        dr = DiagnosticReport(fit, precomputed={"parallel_trends": {"method": "event_study"}})
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "error"
        assert "joint_p_value" in pt["reason"] or "p_value" in pt["reason"]

    def test_precomputed_bacon_bypasses_applicability_gate(self, cs_fit):
        """Round-22 P1 regression: ``precomputed["bacon"]`` was
        documented as supported but ``_instance_skip_reason`` skipped
        Bacon on applicability grounds (``data`` / column kwargs missing)
        before the runner could fire. Users with an already-computed
        ``BaconDecompositionResults`` must be able to pass it through
        without re-supplying the raw panel.
        """
        from types import SimpleNamespace

        fit, _ = cs_fit
        precomputed_bacon = SimpleNamespace(
            weights=None,
            att=1.2,
            comparison_types={},
            total_weight_later_vs_earlier=0.02,
        )

        dr = DiagnosticReport(fit, precomputed={"bacon": precomputed_bacon})
        bacon_block = dr.to_dict()["bacon"]
        assert bacon_block["status"] == "ran", (
            f"precomputed bacon must bypass the applicability gate. "
            f"Got status={bacon_block.get('status')}, reason={bacon_block.get('reason')}"
        )

    def test_precomputed_single_m_sensitivity_exposes_original_estimate_and_se(self, cs_fit):
        """Pre-emptive audit regression: ``_format_precomputed_sensitivity``
        used to drop ``original_estimate`` and ``original_se`` on the
        single-M ``HonestDiDResults`` branch, even though both
        ``SensitivityResults`` and ``HonestDiDResults`` carry those fields.
        The grid branch surfaces them via ``_format_sensitivity_results``,
        so dropping them on the single-M branch made the schema shape
        dependent on which object type the user passed. Parity fix: the
        single-M branch now carries the same fields.
        """
        from types import SimpleNamespace

        fit, _ = cs_fit
        single_m = SimpleNamespace(
            lb=0.3,
            ub=1.8,
            ci_lb=0.15,
            ci_ub=1.95,
            M=1.0,
            method="relative_magnitude",
            original_estimate=1.05,
            original_se=0.22,
            alpha=0.05,
        )

        block = DiagnosticReport(fit, precomputed={"sensitivity": single_m}).to_dict()[
            "sensitivity"
        ]
        assert block["status"] == "ran"
        assert block["conclusion"] == "single_M_precomputed"
        # Parity with the grid branch: these fields must be present and
        # reflect the passed object's values.
        assert block["original_estimate"] == 1.05
        assert block["original_se"] == 0.22


# ---------------------------------------------------------------------------
# Verdict / tier helpers
# ---------------------------------------------------------------------------
class TestJointWaldAlignment:
    """Cover the event-study PT joint-Wald vs Bonferroni fallback paths.

    These tests address the correctness-sensitive codepath in
    ``_pt_event_study`` where pre-period coefficient keys must align with
    ``interaction_indices`` before the joint Wald statistic can be indexed
    into the right vcov rows/columns. When alignment fails, the code must
    fall back to Bonferroni rather than compute a Wald statistic on the
    wrong rows.
    """

    @staticmethod
    def _stub_result(pre_effects, interaction_indices, vcov, **extra):
        """Build a minimal MultiPeriodDiDResults-shaped stub for PT tests.

        ``pre_effects`` is an iterable of ``(period_key, effect, se, p_value)``
        tuples. Returns an object whose class name is ``MultiPeriodDiDResults``
        so DR's name-keyed dispatch routes it to the event-study PT path.
        """
        from types import SimpleNamespace

        pre_map = {
            k: SimpleNamespace(effect=eff, se=se, p_value=p) for (k, eff, se, p) in pre_effects
        }

        class MultiPeriodDiDResults:  # noqa: D401 — test stub that mimics the real class name
            pass

        obj = MultiPeriodDiDResults()
        obj.pre_period_effects = pre_map
        obj.interaction_indices = interaction_indices
        obj.vcov = np.asarray(vcov, dtype=float) if vcov is not None else None
        obj.avg_att = 1.0
        obj.avg_se = 0.1
        obj.avg_p_value = 0.001
        obj.avg_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 50
        obj.n_control = 50
        obj.survey_metadata = None
        for k, v in extra.items():
            setattr(obj, k, v)
        return obj

    def test_joint_wald_runs_when_keys_align(self):
        """With aligned pre_effects + interaction_indices + vcov, Wald runs
        and the computed chi-squared statistic matches the closed form."""
        pre = [(-3, 0.0, 0.5, 0.99), (-2, 0.0, 0.5, 0.99), (-1, 0.0, 0.5, 0.99)]
        interaction_indices = {-3: 0, -2: 1, -1: 2, 0: 3}  # maps period -> vcov row
        vcov = np.diag([0.25, 0.25, 0.25, 0.25])  # SE = 0.5 for each pre-period
        stub = self._stub_result(pre, interaction_indices, vcov)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert (
            pt["method"] == "joint_wald"
        ), f"Expected joint_wald with aligned keys; got {pt.get('method')}"
        # beta=0 across all periods -> test_statistic = 0 -> p = 1.0
        assert pt["test_statistic"] == pytest.approx(0.0)
        assert pt["joint_p_value"] == pytest.approx(1.0)
        assert pt["df"] == 3

    def test_joint_wald_computes_expected_statistic(self):
        """Verify the Wald statistic matches a known closed-form value."""
        # beta = [1.0, -0.5, 0.2]; vcov diagonal with variances [0.25, 0.25, 0.16]
        # -> test_statistic = 1.0^2/0.25 + 0.5^2/0.25 + 0.2^2/0.16
        #                   = 4.0 + 1.0 + 0.25 = 5.25
        pre = [(-3, 1.0, 0.5, 0.04), (-2, -0.5, 0.5, 0.30), (-1, 0.2, 0.4, 0.61)]
        interaction_indices = {-3: 0, -2: 1, -1: 2}
        vcov = np.diag([0.25, 0.25, 0.16])
        stub = self._stub_result(pre, interaction_indices, vcov)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["method"] == "joint_wald"
        assert pt["test_statistic"] == pytest.approx(5.25, rel=1e-6)

    def test_falls_back_to_bonferroni_without_interaction_indices(self):
        pre = [(-2, 1.0, 0.5, 0.04), (-1, 0.2, 0.5, 0.69)]
        stub = self._stub_result(pre, interaction_indices=None, vcov=np.diag([0.25, 0.25]))
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "bonferroni", (
            "Missing interaction_indices must force Bonferroni fallback, "
            "never attempt a Wald statistic on misaligned rows."
        )
        # Bonferroni: min(per-period p) * n = 0.04 * 2 = 0.08 (< 1)
        assert pt["joint_p_value"] == pytest.approx(0.08, rel=1e-6)

    def test_falls_back_to_bonferroni_when_keys_misaligned(self):
        """pre_effects has keys [-2, -1] but interaction_indices uses [2019, 2020]."""
        pre = [(-2, 1.0, 0.5, 0.04), (-1, 0.2, 0.5, 0.69)]
        interaction_indices = {2019: 0, 2020: 1}  # deliberately different namespace
        vcov = np.diag([0.25, 0.25])
        stub = self._stub_result(pre, interaction_indices, vcov)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "bonferroni", (
            "Misaligned interaction_indices must force Bonferroni fallback — "
            "the len(keys_in_vcov) == df guard should prevent the Wald path."
        )

    def test_falls_back_to_bonferroni_when_vcov_missing(self):
        pre = [(-2, 1.0, 0.5, 0.04), (-1, 0.2, 0.5, 0.69)]
        interaction_indices = {-2: 0, -1: 1}
        stub = self._stub_result(pre, interaction_indices, vcov=None)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["method"] == "bonferroni"

    def test_joint_wald_uses_F_reference_when_survey_df_is_finite(self):
        """Round-27 P1 regression: event-study PT on a survey-backed fit
        must use an F reference distribution with denominator df =
        ``survey_metadata.df_survey`` rather than the chi-square
        reference. Chi-square over-rejects under a finite-sample
        correction; the design-based SE already reflects the effective
        sample size and the PT test must match.
        """
        from types import SimpleNamespace

        from scipy.stats import chi2
        from scipy.stats import f as f_dist

        # Same fixture as ``test_joint_wald_runs_when_keys_align`` but with
        # a survey_metadata carrying a finite df_survey.
        pre = [(-3, 1.0, 1.0, 0.32), (-2, 1.0, 1.0, 0.32), (-1, 1.0, 1.0, 0.32)]
        interaction_indices = {-3: 0, -2: 1, -1: 2, 0: 3}
        vcov = np.eye(4)
        stub = self._stub_result(
            pre,
            interaction_indices,
            vcov,
            survey_metadata=SimpleNamespace(df_survey=20.0),
        )

        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]

        # With beta = [1,1,1] and V = I, the Wald statistic is 3.0.
        assert pt["status"] == "ran"
        assert pt["test_statistic"] == pytest.approx(3.0, rel=1e-6)
        assert pt["df"] == 3

        # Method tag surfaces the survey branch so BR / DR prose can
        # flag the finite-sample correction. Denominator df is exposed
        # on the schema for downstream consumers.
        assert pt["method"].endswith("_survey")
        assert pt["df_denom"] == pytest.approx(20.0)

        # F statistic = W / k = 3.0 / 3 = 1.0; survey p-value uses
        # F(3, 20) instead of chi-square(3).
        expected_p_survey = float(1.0 - f_dist.cdf(1.0, dfn=3, dfd=20.0))
        expected_p_chi2 = float(1.0 - chi2.cdf(3.0, df=3))
        assert pt["joint_p_value"] == pytest.approx(expected_p_survey, rel=1e-6)
        # Chi-square would be noticeably more confident (smaller p) than
        # F under finite df; confirm the survey path isn't degenerating
        # back to chi-square.
        assert expected_p_survey > expected_p_chi2

    def test_precomputed_survey_pt_replay_preserves_df_denom(self, cs_fit):
        """Round-28 P3 regression: a schema-shaped PT block carrying the
        survey ``df_denom`` and ``_survey`` method suffix must round-trip
        through ``precomputed={"parallel_trends": ...}`` without losing
        the finite-sample provenance. Previously ``_format_precomputed_pt``
        dropped ``df_denom``, so replaying a survey-aware DR block
        silently demoted it to a chi-square-style passthrough.
        """
        fit, _ = cs_fit
        survey_pt = {
            "method": "joint_wald_event_study_survey",
            "joint_p_value": 0.18,
            "test_statistic": 5.2,
            "df": 3,
            "df_denom": 20.0,
        }
        dr = DiagnosticReport(fit, precomputed={"parallel_trends": survey_pt})
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "joint_wald_event_study_survey"
        assert pt["df_denom"] == 20.0
        assert pt["df"] == 3

    def test_dr_prose_uses_event_study_subject_for_survey_pt(self):
        """Round-29 P3 regression: DR's own ``_pt_subject_phrase`` /
        ``_pt_stat_label`` helpers previously didn't recognize the
        ``_survey`` variants, so summary / full_report prose fell
        through to the generic "Pre-treatment data" wording — BR's
        helpers were fixed last round but DR's were not. The survey
        variants must render with the event-study subject and the
        ``joint p`` label; the F-reference correction is a different
        reference distribution, not a different test.
        """
        from diff_diff.diagnostic_report import (
            _pt_stat_label,
            _pt_subject_phrase,
        )

        for method in (
            "joint_wald_survey",
            "joint_wald_event_study_survey",
        ):
            assert _pt_subject_phrase(method) == "Pre-treatment event-study coefficients", (
                f"DR subject for {method!r} must match the non-survey "
                f"event-study phrasing; got "
                f"{_pt_subject_phrase(method)!r}"
            )
            assert _pt_stat_label(method) == "joint p"

    def test_joint_wald_ignores_non_finite_survey_df(self):
        """If ``df_survey`` is NaN / inf / non-positive, fall back to
        chi-square (no finite-sample correction available).
        """
        from types import SimpleNamespace

        pre = [(-3, 1.0, 1.0, 0.32), (-2, 1.0, 1.0, 0.32), (-1, 1.0, 1.0, 0.32)]
        interaction_indices = {-3: 0, -2: 1, -1: 2, 0: 3}
        vcov = np.eye(4)
        stub = self._stub_result(
            pre,
            interaction_indices,
            vcov,
            survey_metadata=SimpleNamespace(df_survey=float("nan")),
        )
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        # Non-finite df_survey must not taint the method tag.
        assert not pt["method"].endswith("_survey")
        assert "df_denom" not in pt


class TestNarrowedApplicabilityAndPlaceboSchema:
    """Regressions for the round-3 CI-review findings.

    * ``pretrends_power`` and ``sensitivity`` are now restricted to the
      result families that their backing helpers actually support, so
      default reports no longer land in ``error`` for SA / Imputation /
      Stacked / EfficientDiD / StaggeredTripleDiff / Wooldridge.
    * ``placebo`` is always ``status="skipped"`` in MVP regardless of
      estimator, matching the ``REPORTING.md`` contract.
    """

    def test_placebo_is_always_skipped_not_not_applicable(self, did_fit):
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df, outcome="outcome", treatment="treated", time="post")
        placebo = dr.to_dict()["placebo"]
        assert placebo["status"] == "skipped", (
            f"placebo must always be status='skipped' per REPORTING.md; "
            f"got {placebo['status']!r}"
        )

    def test_placebo_skipped_for_multiperiod_fit(self, multi_period_fit):
        fit, _ = multi_period_fit
        placebo = DiagnosticReport(fit).to_dict()["placebo"]
        assert placebo["status"] == "skipped"

    def test_placebo_skipped_for_sdid_fit(self, sdid_fit):
        fit, _ = sdid_fit
        placebo = DiagnosticReport(fit).to_dict()["placebo"]
        assert placebo["status"] == "skipped"

    def test_sun_abraham_sensitivity_not_applicable(self):
        """SA is not in HonestDiD's adapter list; DR must not try to run it."""
        import warnings

        from diff_diff import SunAbraham, generate_staggered_data

        warnings.filterwarnings("ignore")
        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        fit = SunAbraham().fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        dr = DiagnosticReport(fit)
        applicable = set(dr.applicable_checks)
        sensitivity = dr.to_dict()["sensitivity"]
        assert "sensitivity" not in applicable, (
            "SunAbrahamResults has no HonestDiD adapter; sensitivity must not "
            "be marked applicable"
        )
        assert sensitivity["status"] == "not_applicable"

    def test_n_obs_zero_reference_marker_filtered(self):
        """Stacked / TwoStage / Imputation reference markers use n_obs=0
        (not n_groups=0). ``_collect_pre_period_coefs`` must filter both."""
        import numpy as np

        from diff_diff.diagnostic_report import _collect_pre_period_coefs

        class StackedDiDResults:
            pass

        obj = StackedDiDResults()
        obj.event_study_effects = {
            -2: {"effect": 0.1, "se": 0.3, "p_value": 0.74, "n_obs": 50},
            -1: {
                "effect": 0.0,
                "se": np.nan,
                "p_value": np.nan,
                "n_obs": 0,  # synthetic reference marker
            },
            0: {"effect": 1.5, "se": 0.2, "p_value": 0.0001, "n_obs": 50},
        }
        coefs, _ = _collect_pre_period_coefs(obj)
        keys = [k for (k, _, _, _) in coefs]
        assert -1 not in keys, "n_obs==0 row must be filtered out"
        assert -2 in keys


class TestReferenceMarkerAndNaNFiltering:
    """Regression for the P0 finding that reference markers + NaN pre-periods
    were being swept into Bonferroni / Wald PT as real evidence.

    Universal-base CS / SA / ImputationDiD / Stacked event-study output
    injects a synthetic reference-period row (``effect=0``, ``se=NaN``,
    ``p_value=NaN``, ``n_groups=0``). Treating that row as valid
    pre-period evidence would inflate the Bonferroni denominator and
    collapse all-NaN fallbacks to a false-clean verdict.
    """

    @staticmethod
    def _cs_stub_with_reference_marker():
        import numpy as np

        class CallawaySantAnnaResults:
            pass

        obj = CallawaySantAnnaResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 200
        obj.n_treated = 40
        obj.n_control = 160
        obj.survey_metadata = None
        # Two real pre-period rows + one universal-base reference marker (n_groups=0).
        obj.event_study_effects = {
            -3: {"effect": 0.1, "se": 0.3, "p_value": 0.74, "n_groups": 5},
            -2: {"effect": -0.2, "se": 0.3, "p_value": 0.51, "n_groups": 5},
            -1: {
                "effect": 0.0,
                "se": np.nan,
                "p_value": np.nan,
                "conf_int": (np.nan, np.nan),
                "n_groups": 0,
            },
            0: {"effect": 1.5, "se": 0.2, "p_value": 0.0001, "n_groups": 5},
        }
        obj.vcov = None
        obj.interaction_indices = None
        obj.event_study_vcov = None
        obj.event_study_vcov_index = None
        return obj

    def test_reference_marker_excluded_from_pt_collection(self):
        from diff_diff.diagnostic_report import _collect_pre_period_coefs

        obj = self._cs_stub_with_reference_marker()
        coefs, _ = _collect_pre_period_coefs(obj)
        keys = [k for (k, _, _, _) in coefs]
        assert -1 not in keys, (
            "Universal-base reference marker (n_groups=0) must not appear "
            "as a valid pre-period coefficient"
        )
        assert -3 in keys and -2 in keys
        # Every returned SE must be finite.
        for _k, _eff, se, _p in coefs:
            assert np.isfinite(se), f"Non-finite SE leaked through: {se}"

    def test_all_nan_pre_periods_do_not_produce_clean_verdict(self):
        """If *every* pre-period row is a reference marker / NaN, the PT
        check must return inconclusive / skipped — never a clean p_value=1.0.
        """
        import numpy as np

        class CallawaySantAnnaResults:
            pass

        obj = CallawaySantAnnaResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 200
        obj.n_treated = 40
        obj.n_control = 160
        obj.survey_metadata = None
        obj.event_study_effects = {
            -1: {
                "effect": 0.0,
                "se": np.nan,
                "p_value": np.nan,
                "n_groups": 0,
            },
            0: {"effect": 1.5, "se": 0.2, "p_value": 0.0001, "n_groups": 5},
        }
        obj.vcov = None
        obj.interaction_indices = None
        obj.event_study_vcov = None
        obj.event_study_vcov_index = None
        dr = DiagnosticReport(obj, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        # All pre-period rows were reference markers → no valid data → skipped.
        assert pt["status"] == "skipped"
        # Verdict must not falsely say "no detected violation" when the only
        # "data" was a reference marker.
        assert pt.get("verdict") != "no_detected_violation"

    def test_undefined_pre_period_inference_yields_inconclusive_not_shrunken_bonferroni(self):
        """Round-33 P0 regression: when any pre-period has undefined
        inference (non-finite effect / SE or ``se <= 0``), the Bonferroni
        fallback must NOT silently shrink the test family on the
        remaining subset and publish a clean joint p-value. Per the
        ``safe_inference`` contract (``utils.py`` line 175), undefined
        SE yields NaN downstream; the joint PT test must be explicitly
        inconclusive so BR prose does not render a stakeholder-facing
        "parallel trends hold" verdict from a partially-undefined
        pre-period surface.
        """
        from types import SimpleNamespace

        import numpy as np

        class MultiPeriodDiDResults:
            pass

        obj = MultiPeriodDiDResults()
        # One valid row + one row whose p-value is NaN (the ``se`` here
        # is finite / positive; the NaN p models an exotic fit where
        # the inference pipeline could not produce a p-value even with
        # a valid SE).
        obj.pre_period_effects = {
            -2: SimpleNamespace(effect=1.0, se=0.5, p_value=0.04),
            -1: SimpleNamespace(effect=0.5, se=0.0, p_value=np.nan),
        }
        obj.vcov = None
        obj.interaction_indices = None
        obj.event_study_vcov = None
        obj.event_study_vcov_index = None
        obj.avg_att = 1.0
        obj.avg_se = 0.1
        obj.avg_p_value = 0.001
        obj.avg_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 50
        obj.n_control = 50
        obj.survey_metadata = None

        dr = DiagnosticReport(obj, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]

        # Method flagged inconclusive; joint_p None; verdict inconclusive.
        assert pt["method"] == "inconclusive"
        assert pt["joint_p_value"] is None
        assert pt["verdict"] == "inconclusive"
        # Metadata records how many pre-periods were dropped and why.
        assert pt["n_dropped_undefined"] == 1
        assert "undefined inference" in pt["reason"]

    def test_nan_headline_yields_estimation_failure_prose_not_did_not_change(self):
        """Round-36 P0 regression: a non-finite headline effect
        (``NaN`` ATT from a failed fit) previously passed the ``val is
        not None`` guard in ``_render_overall_interpretation``. Since
        ``NaN > 0`` and ``NaN < 0`` are both false, the directional
        branch fell through to "did not change" and rendered
        "did not change ... by nan (p = nan, 95% CI: nan to nan)" —
        misleading stakeholder prose on a failed fit.

        Both ``DiagnosticReport.summary()`` and
        ``to_dict()["overall_interpretation"]`` must now emit an
        explicit estimation-failure sentence instead.
        """

        class DiDResults:
            pass

        stub = DiDResults()
        stub.att = float("nan")
        stub.se = float("nan")
        stub.t_stat = float("nan")
        stub.p_value = float("nan")
        stub.conf_int = (float("nan"), float("nan"))
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 50
        stub.n_control = 50
        stub.survey_metadata = None

        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        summary = dr.summary()
        interp = dr.to_dict()["overall_interpretation"]

        for label, prose in [("summary", summary), ("overall_interpretation", interp)]:
            lower = prose.lower()
            # Must NOT render directional / numeric prose on a NaN fit.
            assert (
                "did not change" not in lower
            ), f"{label} rendered 'did not change' on a NaN fit; got: {prose!r}"
            assert (
                "nan" not in lower
            ), f"{label} rendered 'nan' in the stakeholder-facing prose; got: {prose!r}"
            assert "by nan" not in lower
            assert "ci: nan" not in lower
            # Must name the non-finite state explicitly.
            assert (
                "non-finite" in lower or "did not produce" in lower
            ), f"{label} must emit an estimation-failure sentence; got: {prose!r}"

    def test_summary_prose_surfaces_inconclusive_pt_explicitly(self):
        """Round-35 P1 regression: when pre-trends is inconclusive
        (undefined pre-period inference), both ``BusinessReport.summary()``
        and ``DiagnosticReport.summary()`` must emit explicit inconclusive
        prose — not merely omit the PT sentence. A missing sentence was
        indistinguishable from "PT did not run" and would silently drop
        the identifying-assumption diagnostic from stakeholder output.
        """
        from diff_diff import BusinessReport

        class StackedDiDResults:
            pass

        obj = StackedDiDResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.2
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.6, 1.4)
        obj.alpha = 0.05
        obj.n_obs = 400
        obj.n_treated_units = 100
        obj.n_control_units = 300
        obj.survey_metadata = None
        obj.event_study_effects = {
            -2: {"effect": 0.1, "se": 0.2, "p_value": 0.62, "n_obs": 400},
            -1: {"effect": 0.05, "se": 0.3, "p_value": float("nan"), "n_obs": 400},
        }

        dr_summary = DiagnosticReport(obj, run_sensitivity=False, run_bacon=False).summary()
        br_summary = BusinessReport(obj).summary()

        # Both summaries must explicitly name the inconclusive state.
        for label, prose in [("DR", dr_summary), ("BR", br_summary)]:
            assert "inconclusive" in prose.lower(), (
                f"{label}.summary() must surface the inconclusive PT "
                f"state explicitly; got: {prose!r}"
            )
            # And must not offer false-clean "do not reject" wording.
            assert "do not reject parallel trends" not in prose.lower()
            assert "consistent with parallel trends" not in prose.lower()

    def test_design_effect_deff_below_95_uses_improves_precision_wording(self):
        """Round-35 P2 regression: ``deff < 0.95`` is a precision-
        improving survey design — effective N is LARGER than nominal
        N. DR emits ``band_label="improves_precision"`` and BR narrates
        "improves effective sample size" instead of "reduces".
        """
        from types import SimpleNamespace

        from diff_diff import BusinessReport

        class CallawaySantAnnaResults:
            pass

        obj = CallawaySantAnnaResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.2
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.6, 1.4)
        obj.alpha = 0.05
        obj.n_obs = 500
        obj.n_treated = 100
        obj.n_control_units = 400
        obj.event_study_effects = None
        obj.survey_metadata = SimpleNamespace(
            design_effect=0.80,
            effective_n=625.0,
            weight_type="pweight",
            n_strata=None,
            n_psu=None,
            df_survey=None,
            replicate_method=None,
        )

        # Schema: band_label surfaces the precision-improving state.
        deff_block = DiagnosticReport(obj).to_dict()["design_effect"]
        assert deff_block["band_label"] == "improves_precision"

        # Prose: BR says "improves", not "reduces".
        summary = BusinessReport(obj).summary().lower()
        assert "improves effective sample size" in summary
        assert "reduces effective sample size" not in summary

    def test_finite_se_nan_p_value_yields_inconclusive_on_bonferroni_only_surface(self):
        """Round-34 P0 regression: replicate-weight survey fits can emit
        event-study rows with finite ``effect`` / ``se`` but
        ``p_value=NaN`` when ``safe_inference`` sees ``df <= 0`` — the
        design-based SE is still defined but inference fields collapse
        to NaN per ``utils.py`` line 175. The round-33 collector filter
        (``se > 0``) lets such rows through; the Bonferroni fallback
        previously excluded NaN p-values and scaled by the reduced
        family, producing a clean joint PT verdict that BR rendered as
        "do not reject parallel trends" prose.

        Use a ``StackedDiDResults`` stub (Bonferroni-only surface: no
        ``vcov`` / ``event_study_vcov``) with one finite-inference row
        and one finite-SE / NaN-p row, and assert DR emits inconclusive.
        """
        from diff_diff import BusinessReport

        class StackedDiDResults:
            pass

        obj = StackedDiDResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.2
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.6, 1.4)
        obj.alpha = 0.05
        obj.n_obs = 400
        obj.n_treated_units = 100
        obj.n_control_units = 300
        obj.survey_metadata = None
        obj.event_study_effects = {
            -2: {"effect": 0.1, "se": 0.2, "p_value": 0.62, "n_obs": 400},
            # Finite SE but NaN p-value — models the replicate-weight
            # collapsed-df case. Previously stayed in the family but
            # was dropped from the Bonferroni denominator.
            -1: {"effect": 0.05, "se": 0.3, "p_value": float("nan"), "n_obs": 400},
        }

        dr = DiagnosticReport(obj, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]

        assert pt["method"] == "inconclusive", (
            f"Bonferroni-only surface with NaN per-period p-value must "
            f"return inconclusive; got method={pt.get('method')!r} with "
            f"joint_p={pt.get('joint_p_value')!r}"
        )
        assert pt["verdict"] == "inconclusive"
        assert pt["joint_p_value"] is None
        assert pt["n_dropped_undefined"] == 1

        # And BR must not turn that into "do not reject" / "consistent
        # with parallel trends" wording.
        br_summary = BusinessReport(obj).summary().lower()
        assert "do not reject parallel trends" not in br_summary
        assert "consistent with parallel trends" not in br_summary

    def test_zero_se_pre_period_yields_inconclusive(self):
        """Round-33 P0 regression: a pre-period row whose SE is
        zero/negative is undefined inference per the ``safe_inference``
        contract and must push the event-study PT to inconclusive.
        """
        from types import SimpleNamespace

        class MultiPeriodDiDResults:
            pass

        obj = MultiPeriodDiDResults()
        obj.pre_period_effects = {
            -2: SimpleNamespace(effect=1.0, se=0.5, p_value=0.04),
            -1: SimpleNamespace(effect=0.5, se=0.0, p_value=0.99),
        }
        obj.vcov = None
        obj.interaction_indices = None
        obj.event_study_vcov = None
        obj.event_study_vcov_index = None
        obj.avg_att = 1.0
        obj.avg_se = 0.1
        obj.avg_p_value = 0.001
        obj.avg_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 50
        obj.n_control = 50
        obj.survey_metadata = None

        pt = DiagnosticReport(obj, run_sensitivity=False, run_bacon=False).to_dict()[
            "parallel_trends"
        ]
        assert pt["verdict"] == "inconclusive"
        assert pt["method"] == "inconclusive"
        assert pt["n_dropped_undefined"] >= 1

    def test_all_pre_periods_undefined_yields_inconclusive_not_skipped(self):
        """Round-42 P1 regression: the twin of the partially-undefined
        case. When every pre-period row is dropped by the collector
        for undefined inference (all ``se <= 0`` or non-finite effect/SE),
        ``_collect_pre_period_coefs`` returns ``([], n_dropped_undefined > 0)``.
        The prior behavior routed through the empty-coefs ``skipped``
        path ("No pre-period event-study coefficients available"),
        which let BR drop the identifying-assumption warning and render
        a silent-PT-absent narrative. That violates the inconclusive
        contract documented in REPORTING.md: when any pre-row is
        dropped for undefined inference, the joint PT test is
        inconclusive, not skipped.
        """
        from diff_diff import BusinessReport

        class StackedDiDResults:
            pass

        obj = StackedDiDResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.2
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.6, 1.4)
        obj.alpha = 0.05
        obj.n_obs = 400
        obj.n_treated_units = 100
        obj.n_control_units = 300
        obj.survey_metadata = None
        # All pre-rows have ``se == 0`` — undefined inference per the
        # safe-inference contract (``utils.py:175``). The collector's
        # ``se > 0`` filter drops all of them, leaving pre_coefs=[]
        # with n_dropped_undefined=2 (the R42 all-undefined case).
        obj.event_study_effects = {
            -2: {
                "effect": 0.1,
                "se": 0.0,
                "p_value": 1.0,
                "n_obs": 400,
            },
            -1: {
                "effect": 0.05,
                "se": 0.0,
                "p_value": 1.0,
                "n_obs": 400,
            },
        }

        dr = DiagnosticReport(obj, run_sensitivity=False, run_bacon=False)
        # Applicability gate: PT must be marked applicable (runs as
        # inconclusive), not skipped with "no coefficients available".
        assert "parallel_trends" in dr.applicable_checks, (
            "All-undefined pre-period case must keep PT applicable so "
            "the inconclusive runner can emit the explicit "
            "n_dropped_undefined provenance. Current skipped reasons: "
            f"{dr.skipped_checks}"
        )
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran", pt
        assert pt["method"] == "inconclusive", (
            f"All-undefined pre-period family must route to the "
            f"inconclusive runner, not 'skipped'. Got status="
            f"{pt.get('status')!r}, method={pt.get('method')!r}, "
            f"reason={pt.get('reason')!r}"
        )
        assert pt["verdict"] == "inconclusive"
        assert pt["joint_p_value"] is None
        # All-undefined: n_dropped_undefined equals attempted pre-period
        # count (2 rows here), and the valid subset is empty.
        assert pt["n_dropped_undefined"] == 2
        assert pt["n_pre_periods"] == 0

        # BR must surface this as an inconclusive identifying-
        # assumption warning, not silently omit PT. The "inconclusive"
        # verdict phrasing is the load-bearing contract for
        # stakeholders.
        br_summary = BusinessReport(obj).summary().lower()
        assert "inconclusive" in br_summary, (
            f"All-undefined PT must surface 'inconclusive' in BR " f"summary. Got: {br_summary!r}"
        )
        # And must not claim PT was untested / no-coefs.
        assert "no pre-period event-study coefficients" not in br_summary
        assert "consistent with parallel trends" not in br_summary

    def test_pretrends_power_adapter_filters_zero_se_cs(self):
        """Round-33 P0 regression: CS / SA ``compute_pretrends_power``
        adapters also use the ``se > 0`` filter alongside
        ``np.isfinite(se)`` so the power analysis never includes rows
        whose per-period SE collapsed.
        """

        import numpy as np

        from diff_diff.pretrends import compute_pretrends_power
        from diff_diff.staggered import CallawaySantAnnaResults

        obj = object.__new__(CallawaySantAnnaResults)
        obj.anticipation = 0
        # Three pre-periods: two valid, one with zero SE. The valid
        # two are enough to run power analysis; the zero-SE row must
        # NOT slip into the `ses` vector and divide-by-zero.
        obj.event_study_effects = {
            -3: {"effect": 0.1, "se": 0.2, "p_value": 0.7, "n_groups": 1},
            -2: {"effect": 0.0, "se": 0.0, "p_value": float("nan"), "n_groups": 1},
            -1: {"effect": 0.0, "se": 0.2, "p_value": 0.99, "n_groups": 1},
            0: {"effect": 1.0, "se": 0.2, "p_value": 0.0, "n_groups": 1},
        }
        obj.overall_att = 1.0
        obj.alpha = 0.05

        pp = compute_pretrends_power(obj, alpha=0.05, target_power=0.80, violation_type="linear")
        # Zero-SE row must not appear in pre_period_ses.
        assert len(pp.pre_period_ses) == 2
        assert np.all(pp.pre_period_ses > 0)


class TestPrecomputedValidation:
    """Regression for the P1 finding that ``precomputed=`` silently accepted
    keys that were never implemented. Unsupported keys now raise."""

    def test_unsupported_precomputed_key_raises(self, multi_period_fit):
        fit, _ = multi_period_fit
        with pytest.raises(ValueError, match="not implemented"):
            DiagnosticReport(fit, precomputed={"design_effect": object()})

    def test_supported_precomputed_keys_accepted(self, multi_period_fit):
        fit, _ = multi_period_fit
        # The four implemented keys should not raise at construction.
        DiagnosticReport(fit, precomputed={"parallel_trends": {"p_value": 0.5}})

    def test_mixed_supported_and_unsupported_raises(self, multi_period_fit):
        fit, _ = multi_period_fit
        with pytest.raises(ValueError, match="epv"):
            DiagnosticReport(fit, precomputed={"sensitivity": None, "epv": object()})


class TestSingleMSensitivityPrecomputed:
    """Single-M HonestDiDResults must NOT be narrated as full-grid robustness.

    Regression for the P0 CI-review finding that ``conclusion='single_M_precomputed'``
    was being swallowed because both renderers checked ``breakdown_M is None`` and
    fell through to the "robust across the full grid" phrasing.
    """

    def _fake_single_m(self, M=1.5, ci_lb=1.0, ci_ub=3.0):
        from types import SimpleNamespace

        return SimpleNamespace(
            M=M,
            lb=ci_lb,
            ub=ci_ub,
            ci_lb=ci_lb,
            ci_ub=ci_ub,
            method="relative_magnitude",
            alpha=0.05,
        )

    def test_dr_schema_preserves_single_m_marker(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit, precomputed={"sensitivity": self._fake_single_m()})
        sens = dr.to_dict()["sensitivity"]
        assert sens["status"] == "ran"
        assert sens["conclusion"] == "single_M_precomputed"
        assert sens["breakdown_M"] is None
        assert len(sens["grid"]) == 1

    def test_dr_summary_does_not_claim_full_grid_robustness(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit, precomputed={"sensitivity": self._fake_single_m()})
        summary = dr.summary()
        assert "across the entire HonestDiD grid" not in summary
        assert "robust across the grid" not in summary
        # It should narrate the single-M check honestly.
        assert "single point checked" in summary
        assert "not a breakdown" in summary or "not a grid" in summary

    def test_br_summary_does_not_claim_full_grid_robustness(self, multi_period_fit):
        """BR via honest_did_results= passthrough must not oversell a point check."""
        from diff_diff import BusinessReport

        fit, _ = multi_period_fit
        br = BusinessReport(fit, honest_did_results=self._fake_single_m())
        summary = br.summary()
        assert "full grid" not in summary
        assert "single point checked" in summary


class TestEPVDictBacked:
    """EPV diagnostics on fits that use the dict-of-dicts convention.

    Regression for the P0 CI-review finding that ``_check_epv`` assumed
    ``low_epv_cells`` / ``min_epv`` attributes but the library stores
    ``epv_diagnostics`` as ``{(g, t): {"is_low": ..., "epv": ...}}``.
    """

    def _make_cs_stub(self, epv_diag, threshold=10.0):
        class CallawaySantAnnaResults:
            pass

        obj = CallawaySantAnnaResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 200
        obj.n_treated = 40
        obj.n_control = 160
        obj.survey_metadata = None
        obj.event_study_effects = None
        obj.epv_diagnostics = epv_diag
        obj.epv_threshold = threshold
        return obj

    def test_low_epv_cells_counted_from_is_low_flag(self):
        epv = {
            (2020, 1): {"is_low": True, "epv": 4.5},
            (2020, 2): {"is_low": False, "epv": 18.0},
            (2021, 1): {"is_low": True, "epv": 2.0},
            (2021, 2): {"is_low": False, "epv": 22.0},
        }
        stub = self._make_cs_stub(epv, threshold=10.0)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        section = dr.to_dict()["epv"]
        assert section["status"] == "ran"
        assert section["n_cells_low"] == 2
        assert section["n_cells_total"] == 4
        assert section["min_epv"] == pytest.approx(2.0)
        assert section["threshold"] == pytest.approx(10.0)

    def test_no_low_cells_reports_clean(self):
        epv = {(2020, 1): {"is_low": False, "epv": 15.0}}
        stub = self._make_cs_stub(epv, threshold=10.0)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        section = dr.to_dict()["epv"]
        assert section["n_cells_low"] == 0
        assert section["min_epv"] == pytest.approx(15.0)

    def test_threshold_read_from_results_not_hardcoded(self):
        """Pass a non-default epv_threshold and confirm DR echoes it."""
        epv = {(2020, 1): {"is_low": True, "epv": 7.0}}
        stub = self._make_cs_stub(epv, threshold=8.5)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        assert dr.to_dict()["epv"]["threshold"] == pytest.approx(8.5)


class TestCSEventStudyVCovSupport:
    """CS sensitivity + pretrends_power must not be skipped for absence of results.vcov.

    Regression for the P1 CI-review finding that the applicability gate required
    ``results.vcov`` but CS exposes ``event_study_vcov`` / ``event_study_vcov_index``.
    """

    def test_cs_sensitivity_runs_on_aggregated_fit(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert (
            "sensitivity" in dr.applicable_checks
        ), "CS fit with event_study aggregation must not skip sensitivity"
        sens = dr.to_dict()["sensitivity"]
        # It may run successfully or emit an error depending on data shape,
        # but it must NOT be skipped for "results.vcov not available".
        assert sens["status"] in {"ran", "error"}, sens

    def test_cs_pretrends_power_runs_on_aggregated_fit(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert (
            "pretrends_power" in dr.applicable_checks
        ), "CS fit with event_study aggregation must not skip pretrends_power"


class TestCSJointWaldViaEventStudyVCov:
    """CS PT should use joint_wald via event_study_vcov when interaction_indices is absent.

    Regression for the P1 CI-review finding that CS always fell back to Bonferroni
    even though ``event_study_vcov`` + ``event_study_vcov_index`` were available.
    """

    def _make_cs_stub_with_es_vcov(self):
        class CallawaySantAnnaResults:
            pass

        obj = CallawaySantAnnaResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 200
        obj.n_treated = 40
        obj.n_control = 160
        obj.survey_metadata = None
        # Pre-period event-study entries with known coefficients + vcov.
        obj.event_study_effects = {
            -3: {"effect": 0.5, "se": 0.5, "p_value": 0.32},
            -2: {"effect": -0.5, "se": 0.5, "p_value": 0.32},
            -1: {"effect": 0.2, "se": 0.4, "p_value": 0.62},
            0: {"effect": 2.0, "se": 0.3, "p_value": 0.0001},
            1: {"effect": 2.5, "se": 0.3, "p_value": 0.0001},
        }
        obj.event_study_vcov = np.diag([0.25, 0.25, 0.16, 0.09, 0.09])
        obj.event_study_vcov_index = [-3, -2, -1, 0, 1]
        obj.vcov = None  # CS convention
        obj.interaction_indices = None
        return obj

    def test_cs_pt_uses_event_study_vcov_wald(self):
        stub = self._make_cs_stub_with_es_vcov()
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert (
            pt["method"] == "joint_wald_event_study"
        ), f"Expected event-study-backed Wald; got method={pt.get('method')!r}"
        # Closed-form: 0.5^2/0.25 + (-0.5)^2/0.25 + 0.2^2/0.16 = 1 + 1 + 0.25 = 2.25
        assert pt["test_statistic"] == pytest.approx(2.25, rel=1e-6)
        assert pt["df"] == 3


class TestContinuousDiDHeadline:
    """ContinuousDiDResults exposes overall_att_se/p_value/conf_int, not overall_se/…

    Regression for the P1 CI-review finding that both report classes missed
    ContinuousDiDResults inference fields.
    """

    def test_extract_scalar_headline_resolves_continuous_did_aliases(self):
        from diff_diff.diagnostic_report import _extract_scalar_headline

        class ContinuousDiDResults:
            pass

        obj = ContinuousDiDResults()
        obj.overall_att = 2.5
        obj.overall_att_se = 0.4
        obj.overall_att_p_value = 0.00001
        obj.overall_att_conf_int = (1.7, 3.3)
        obj.alpha = 0.05

        result = _extract_scalar_headline(obj)
        assert result is not None
        name, value, se, p, ci, alpha = result
        assert name == "overall_att"
        assert value == pytest.approx(2.5)
        assert se == pytest.approx(0.4)
        assert p == pytest.approx(0.00001)
        assert ci == [pytest.approx(1.7), pytest.approx(3.3)]
        assert alpha == pytest.approx(0.05)


class TestVerdictsAndTiers:
    def test_pt_verdict_three_bins(self):
        assert _pt_verdict(0.001) == "clear_violation"
        assert _pt_verdict(0.049) == "clear_violation"
        assert _pt_verdict(0.10) == "some_evidence_against"
        assert _pt_verdict(0.29) == "some_evidence_against"
        assert _pt_verdict(0.30) == "no_detected_violation"
        assert _pt_verdict(0.99) == "no_detected_violation"
        assert _pt_verdict(None) == "inconclusive"
        assert _pt_verdict(float("nan")) == "inconclusive"

    def test_power_tier_three_bins_plus_unknown(self):
        assert _power_tier(0.1) == "well_powered"
        assert _power_tier(0.24) == "well_powered"
        assert _power_tier(0.25) == "moderately_powered"
        assert _power_tier(0.99) == "moderately_powered"
        assert _power_tier(1.0) == "underpowered"
        assert _power_tier(5.0) == "underpowered"
        assert _power_tier(None) == "unknown"
        assert _power_tier(float("nan")) == "unknown"


# ---------------------------------------------------------------------------
# EfficientDiD hausman pathway
# ---------------------------------------------------------------------------
class TestEfficientDiDHausman:
    def test_hausman_pretest_runs_with_data_kwargs(self, edid_fit):
        fit, sdf = edid_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "hausman"

    def test_hausman_skipped_without_data_kwargs(self, edid_fit):
        """Without the raw panel kwargs, PT is now gated at the
        applicability level (round-10 CI review) — no method field on
        the skip section, but ``applicable_checks`` excludes
        ``parallel_trends`` and ``skipped_checks`` names it with the
        missing-kwargs reason."""
        fit, _ = edid_fit
        dr = DiagnosticReport(fit)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "skipped"
        assert "parallel_trends" not in dr.applicable_checks
        assert "parallel_trends" in dr.skipped_checks
        assert "hausman_pretest" in dr.skipped_checks["parallel_trends"]


# ---------------------------------------------------------------------------
# SDiD native
# ---------------------------------------------------------------------------
class TestSDiDNative:
    def test_sdid_pt_uses_synthetic_fit_method(self, sdid_fit):
        fit, _ = sdid_fit
        pt = DiagnosticReport(fit).to_dict()["parallel_trends"]
        assert pt["method"] == "synthetic_fit"
        assert pt["verdict"] == "design_enforced_pt"
        assert isinstance(pt.get("pre_treatment_fit_rmse"), float)

    def test_sdid_native_section_populated(self, sdid_fit):
        fit, _ = sdid_fit
        native = DiagnosticReport(fit).to_dict()["estimator_native_diagnostics"]
        assert native["status"] == "ran"
        assert native["estimator"] == "SyntheticDiD"
        assert "weight_concentration" in native
        assert "in_time_placebo" in native
        assert "zeta_sensitivity" in native

    def test_sdid_does_not_call_honest_did(self, sdid_fit):
        """HonestDiD sensitivity should NOT run on SDiD (native path used instead)."""
        fit, _ = sdid_fit
        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis") as mock:
            DiagnosticReport(fit).to_dict()
            mock.assert_not_called()


class TestSCMNative:
    """Classic SCM routes to the fit-based PT analogue + native diagnostics."""

    def test_scm_applicable_checks(self, scm_fit):
        res, _ = scm_fit
        applicable = DiagnosticReport(res).applicable_checks
        assert "estimator_native" in applicable
        assert "parallel_trends" in applicable  # design-enforced fit analogue
        # SCM is not HonestDiD/heterogeneity territory (one treated unit).
        assert "sensitivity" not in applicable
        assert "heterogeneity" not in applicable

    def test_scm_pt_uses_scm_fit_method(self, scm_fit):
        res, _ = scm_fit
        pt = DiagnosticReport(res).to_dict()["parallel_trends"]
        assert pt["method"] == "scm_fit"
        assert pt["verdict"] == "design_enforced_pt"
        assert isinstance(pt.get("pre_treatment_fit_rmse"), float)

    def test_scm_native_section_populated(self, scm_fit):
        res, _ = scm_fit
        native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
        assert native["status"] == "ran"
        assert native["estimator"] == "SyntheticControl"
        assert isinstance(native["pre_rmspe"], float)
        assert "herfindahl" in native["weight_concentration"]
        # Placebo is opt-in: NOT auto-run inside the report.
        assert native["in_space_placebo"]["status"] == "not_run"
        # The ADH-2015 diagnostics are also opt-in: surfaced as "not_run" stubs until
        # the user calls leave_one_out() / in_time_placebo().
        assert native["leave_one_out"]["status"] == "not_run"
        assert native["in_time_placebo"]["status"] == "not_run"

    def test_scm_native_surfaces_placebo_after_optin_run(self, scm_fit):
        res, _ = scm_fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.in_space_placebo()
            native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
        block = native["in_space_placebo"]
        assert block["n_placebos"] == res.n_placebos
        assert block["placebo_p_value"] == pytest.approx(res.placebo_p_value)

    def test_scm_native_surfaces_leave_one_out_after_optin_run(self, scm_fit):
        res, _ = scm_fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.leave_one_out()
            native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
        block = native["leave_one_out"]
        assert block["status"] == "ran"
        assert block["att_range"] is not None and len(block["att_range"]) == 2

    def test_scm_native_surfaces_in_time_placebo_after_optin_run(self, scm_fit):
        res, _ = scm_fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.in_time_placebo()
            native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
        block = native["in_time_placebo"]
        assert block["status"] == "ran" and block["n_dates"] >= 1

    def test_scm_native_confidence_set_not_run_stub(self, scm_fit):
        # Firpo-Possebom test-inversion confidence set is opt-in, like the placebos.
        res, _ = scm_fit
        native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
        assert native["confidence_set"]["status"] == "not_run"

    def test_scm_native_surfaces_confidence_set_after_optin_run(self, scm_fit):
        res, _ = scm_fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.confidence_set(family="constant", gamma=0.34)  # J=4 -> gamma > 1/(J+1)
            native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
        block = native["confidence_set"]
        assert block["status"] in ("ran", "empty", "unbounded")
        assert block["family"] == "constant" and block["parameter"] == "c"
        assert block["gamma"] == pytest.approx(0.34)

    def test_scm_native_conformal_not_run_stub(self, scm_fit):
        # CWZ (2021) conformal inference is opt-in, like the placebos / confidence set.
        res, _ = scm_fit
        native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
        assert native["conformal_inference"]["status"] == "not_run"

    def test_scm_native_surfaces_conformal_after_optin_run(self, scm_fit):
        res, _ = scm_fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.conformal_test(0.0, scheme="moving_block")
            native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
        block = native["conformal_inference"]
        assert block["status"] == "ran"
        assert block["kind"] == "joint"
        assert block["scheme"] == "moving_block"
        assert 0.0 < block["p_value"] <= 1.0

    def test_scm_native_surfaces_pointwise_unbounded_count(self, scm_fit):
        # moving-block pointwise -> |Pi|=T0+1=7; alpha=0.05 < 1/7 -> every period unbounded.
        # The native block must surface n_unbounded so consumers see the granularity state.
        res, _ = scm_fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.conformal_confidence_intervals(alpha=0.05, scheme="moving_block")
            native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
        block = native["conformal_inference"]
        assert block["kind"] == "pointwise"
        assert block["n_unbounded"] == res.n_post_periods

    def test_scm_does_not_call_honest_did(self, scm_fit):
        """HonestDiD sensitivity should NOT run on SCM (fit-based / native path)."""
        res, _ = scm_fit
        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis") as mock:
            DiagnosticReport(res).to_dict()
            mock.assert_not_called()

    def test_scm_significance_not_marked_done_until_placebo_run(self, scm_fit):
        """SCM significance is the OPT-IN in-space placebo, which DR never runs.

        Unlike SDiD/TROP (whose native block IS the sensitivity work), SCM's
        native block only reports the placebo as ``not_run``, so the in-space
        placebo next-step must persist — significance is not complete merely
        because the native diagnostics ran.
        """
        res, _ = scm_fit  # the fixture does NOT run the placebo
        schema = DiagnosticReport(res).to_dict()
        labels = " ".join(s.get("label", "") for s in schema.get("next_steps", [])).lower()
        # Target the IN-SPACE step specifically (the in-time-placebo step's label also
        # contains "placebo" but is a different, always-on ADH-2015 recommendation).
        assert "in-space placebo" in labels  # the significance recommendation surfaces

    def test_scm_placebo_step_completes_after_run(self, scm_fit):
        """Once the opt-in in-space placebo has been run, DR stops recommending it."""
        res, _ = scm_fit
        before = DiagnosticReport(res).to_dict()["next_steps"]
        assert any("in-space placebo" in s.get("label", "").lower() for s in before)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.in_space_placebo()  # opt-in significance procedure now done
        after = DiagnosticReport(res).to_dict()["next_steps"]
        # The in-space step is suppressed; the (differently-tagged) in-time-placebo
        # and leave-one-out recommendations are unaffected.
        assert not any("in-space placebo" in s.get("label", "").lower() for s in after)

    def test_scm_rejects_precomputed_parallel_trends_and_sensitivity(self, scm_fit):
        # Like SDiD/TROP, SCM computes its PT verdict internally (the scm_fit
        # design-enforced pre-fit), so a precomputed parallel_trends p-value payload
        # is methodology-incompatible and rejected — as is sensitivity (no HonestDiD
        # analogue). SCM's PT still runs natively (scm_fit) without a precomputed input.
        res, _ = scm_fit
        with pytest.raises(ValueError, match="estimator_native_diagnostics"):
            DiagnosticReport(res, precomputed={"parallel_trends": {"joint_p_value": 0.5}})
        with pytest.raises(ValueError, match="estimator_native_diagnostics"):
            DiagnosticReport(res, precomputed={"sensitivity": {"breakdown_M": 1.0}})
        # Without a precomputed input, the native scm_fit PT verdict still renders.
        assert DiagnosticReport(res).to_dict()["parallel_trends"]["method"] == "scm_fit"

    def test_scm_pt_prose_is_scm_specific_not_sdid(self, scm_fit):
        # The design_enforced_pt prose must describe SCM's donor-weighted, single-
        # treated-unit, placebo-based identification — NOT SDiD's weighted-PT analogue.
        res, _ = scm_fit
        interp = DiagnosticReport(res).to_dict()["overall_interpretation"].lower()
        assert "in-space placebo" in interp or "donor-weighted" in interp
        assert "sdid" not in interp

    def test_scm_target_parameter_is_named(self, scm_fit):
        res, _ = scm_fit
        tp = DiagnosticReport(res).to_dict()["target_parameter"]
        assert "unrecognized" not in tp["name"].lower()
        assert "scm" in tp["name"].lower() or "synthetic control" in tp["name"].lower()

    def test_scm_generic_placebo_section_points_to_native(self, scm_fit):
        # The generic ``placebo`` section must not claim placebo is "not yet
        # implemented" for SCM — its in-space placebo IS implemented and surfaced
        # under estimator_native_diagnostics (avoids contradictory report state).
        res, _ = scm_fit
        reason = (DiagnosticReport(res).to_dict()["placebo"].get("reason") or "").lower()
        assert "not yet implemented" not in reason
        assert "in_space_placebo" in reason or "estimator_native" in reason

    def test_scm_native_marks_infeasible_placebo(self):
        # An attempted-but-infeasible placebo (here J<2) surfaces an explicit
        # status="infeasible" + reason in _scm_native, not a bare NaN p-value.
        rng = np.random.default_rng(0)
        T, T0 = 8, 6
        years = list(range(2000, 2000 + T))
        d0 = rng.normal(10, 2, T)
        treated = d0 + rng.normal(0, 0.1, T)
        treated[T0:] += 3.0
        rows = [{"unit": "d0", "year": years[t], "y": d0[t], "treated": 0} for t in range(T)]
        rows += [
            {"unit": "treated", "year": years[t], "y": treated[t], "treated": int(t >= T0)}
            for t in range(T)
        ]
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = synthetic_control(
                df, "y", "treated", "unit", "year", seed=0, n_starts=1, inner_min_decrease=1e-3
            )
            res.in_space_placebo()  # only 1 donor -> infeasible
            native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
        block = native["in_space_placebo"]
        assert block["status"] == "infeasible"
        assert np.isnan(block["placebo_p_value"]) and "reason" in block


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_sensitivity_error_does_not_break_report(self, multi_period_fit):
        """A failing diagnostic records its error in the section; the report still renders."""
        fit, _ = multi_period_fit

        def _raise(*args, **kwargs):
            raise RuntimeError("synthetic test failure")

        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis", side_effect=_raise):
            dr = DiagnosticReport(fit)
            schema = dr.to_dict()
            sens = schema["sensitivity"]
            assert sens["status"] == "error"
            assert "synthetic test failure" in sens["reason"]
            # Other sections still ran.
            assert schema["parallel_trends"]["status"] == "ran"


# ---------------------------------------------------------------------------
# Overall prose
# ---------------------------------------------------------------------------
class TestOverallInterpretation:
    def test_overall_interpretation_nonempty_for_fit(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        prose = dr.summary()
        assert isinstance(prose, str)
        assert len(prose) > 50  # a real paragraph

    def test_full_report_has_headers(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        md = dr.full_report()
        assert "# Diagnostic Report" in md
        assert "## Overall Interpretation" in md
        assert "## Parallel trends" in md
        assert "## HonestDiD sensitivity" in md


class TestDRFragilePhrasingIsGridAware:
    """CI review on PR #341 R1: DR's ``overall_interpretation``
    fragile-sensitivity sentence must be gated on the actual
    evaluated grid, not just on ``breakdown_M``. ``breakdown_M`` is
    the interpolated threshold between grid points; "smallest grid
    point fails" is only a valid claim when the smallest actually-
    evaluated M has ``robust_to_zero == False``. Mirrors the BR test
    class ``TestCanonicalValidationSurfaceFixes``.
    """

    @staticmethod
    def _grid(breakdown_m, grid_rows):
        """Build a sensitivity block with a populated grid, matching
        the schema ``_check_sensitivity`` emits."""
        return {
            "status": "ran",
            "method": "relative_magnitude",
            "breakdown_M": breakdown_m,
            "conclusion": "fragile",
            "grid": [
                {
                    "M": row["M"],
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "bound_lower": 0.0,
                    "bound_upper": 0.0,
                    "robust_to_zero": row["robust_to_zero"],
                }
                for row in grid_rows
            ],
        }

    def _render(self, sens_block):
        """Call the DR overall-interpretation renderer on a minimal
        schema that has our sensitivity block and otherwise skipped
        sections (so the fragile-sensitivity branch fires alone)."""
        from diff_diff.diagnostic_report import _render_overall_interpretation

        schema = {
            "schema_version": "2.0",
            "estimator": {"class_name": "CallawaySantAnnaResults", "display_name": "CS"},
            "headline_metric": {
                "status": "ran",
                "effect": 0.5,
                "se": 0.1,
                "p_value": 0.0,
                "ci_lower": 0.3,
                "ci_upper": 0.7,
                "is_significant": True,
                "sign": "positive",
                "alpha": 0.05,
            },
            "parallel_trends": {"status": "skipped", "reason": "stub"},
            "pretrends_power": {"status": "skipped", "reason": "stub"},
            "sensitivity": sens_block,
            "placebo": {"status": "skipped", "reason": "stub"},
            "bacon": {"status": "skipped", "reason": "stub"},
            "design_effect": {"status": "skipped", "reason": "stub"},
            "heterogeneity": {"status": "skipped", "reason": "stub"},
            "epv": {"status": "skipped", "reason": "stub"},
            "estimator_native_diagnostics": {"status": "not_applicable"},
            "skipped": {},
            "warnings": [],
            "next_steps": [],
        }
        return _render_overall_interpretation(schema, {})

    def test_dr_smallest_grid_m_fails_uses_smallest_m_wording(self):
        """Castle Doctrine pattern: grid ``[0.5, 1.0, ...]`` with M=0.5
        already non-robust. DR emits "smallest M evaluated (M = 0.5)".
        """
        sens = self._grid(
            breakdown_m=0.0,
            grid_rows=[
                {"M": 0.5, "robust_to_zero": False},
                {"M": 1.0, "robust_to_zero": False},
            ],
        )
        prose = self._render(sens)
        assert "smallest M evaluated on the sensitivity grid" in prose
        assert "M = 0.5" in prose
        assert "0x the pre-period variation" not in prose

    def test_dr_smallest_grid_m_robust_falls_through_to_multiplier(self):
        """Grid starting at M=0 with smallest point still robust.
        ``breakdown_M=0.03`` is the interpolated threshold between
        M=0 and M=0.25; DR must NOT claim the smallest grid point
        failed (it didn't) and must use the multiplier wording
        instead.
        """
        sens = self._grid(
            breakdown_m=0.03,
            grid_rows=[
                {"M": 0.0, "robust_to_zero": True},
                {"M": 0.25, "robust_to_zero": False},
            ],
        )
        prose = self._render(sens)
        assert "smallest M evaluated on the sensitivity grid" not in prose
        assert "0.03x" in prose

    def test_dr_normal_fragile_keeps_multiplier(self):
        """Normal fragile value (e.g., 0.3) still quotes the multiplier."""
        sens = self._grid(
            breakdown_m=0.3,
            grid_rows=[
                {"M": 0.5, "robust_to_zero": True},
                {"M": 1.0, "robust_to_zero": True},
            ],
        )
        prose = self._render(sens)
        assert "0.3x" in prose
        assert "smallest M evaluated on the sensitivity grid" not in prose


# ---------------------------------------------------------------------------
# Public result class
# ---------------------------------------------------------------------------
class TestDiagnosticReportResults:
    def test_run_all_returns_dataclass(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        results = dr.run_all()
        assert isinstance(results, DiagnosticReportResults)
        assert isinstance(results.applicable_checks, tuple)
        assert isinstance(results.schema, dict)

    def test_run_all_is_idempotent(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        a = dr.run_all()
        b = dr.run_all()
        assert a is b  # cached


class TestDCDHParallelTrendsViaPlaceboEventStudy:
    """Regression for the round-6 P1 finding that dCDH was advertised as
    PT-applicable but ``_collect_pre_period_coefs`` never read
    ``placebo_event_study``, so the PT check was silently skipped even
    on fits with valid placebo horizons.
    """

    def _stub(self, with_placebo: bool):
        class ChaisemartinDHaultfoeuilleResults:
            pass

        stub = ChaisemartinDHaultfoeuilleResults()
        stub.att = 1.0
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.event_study_effects = None
        if with_placebo:
            stub.placebo_event_study = {
                -3: {
                    "effect": 0.05,
                    "se": 0.1,
                    "p_value": 0.62,
                    "conf_int": (-0.15, 0.25),
                    "n_obs": 40,
                },
                -2: {
                    "effect": -0.08,
                    "se": 0.09,
                    "p_value": 0.38,
                    "conf_int": (-0.26, 0.10),
                    "n_obs": 45,
                },
                -1: {
                    "effect": 0.04,
                    "se": 0.10,
                    "p_value": 0.69,
                    "conf_int": (-0.16, 0.24),
                    "n_obs": 50,
                },
            }
        else:
            stub.placebo_event_study = None
        return stub

    def test_pt_check_reads_placebo_event_study(self):
        stub = self._stub(with_placebo=True)
        dr = DiagnosticReport(stub).run_all()
        pt = dr.schema["parallel_trends"]
        assert (
            pt["status"] == "ran"
        ), f"dCDH PT check must run on a fit with placebo_event_study; got {pt}"
        # Per-period rows should come from the placebo keys (negative horizons).
        per_period = pt.get("per_period") or pt.get("periods") or []
        assert per_period, "PT output must include per-period rows"
        periods = [row.get("period") for row in per_period]
        assert all(
            isinstance(p, int) and p < 0 for p in periods
        ), f"dCDH PT must use negative placebo horizons; got {periods}"

    def test_pt_check_skips_when_no_placebo_event_study(self):
        stub = self._stub(with_placebo=False)
        dr = DiagnosticReport(stub).run_all()
        pt = dr.schema["parallel_trends"]
        assert (
            pt["status"] == "skipped"
        ), f"dCDH PT must skip when placebo_event_study is missing; got {pt}"


class TestHeterogeneityPostTreatmentOnly:
    """Regression for the round-6 P1 finding that ``_check_heterogeneity``
    was mixing pre- and post-treatment coefficients into the CV / range /
    sign-consistency summary.
    """

    def test_collector_prefers_post_period_effects_over_period_effects(self):
        """On a MultiPeriod-shaped stub, ``_collect_effect_scalars`` must read
        ``post_period_effects`` (post-treatment only), not ``period_effects``
        (which mixes pre- and post-treatment coefficients). If the pre-period
        value leaked in, sign_consistency would flip and the range would span
        a much larger interval."""
        from diff_diff.diagnostic_report import DiagnosticReport

        class MultiPeriodDiDResults:
            pass

        stub = MultiPeriodDiDResults()
        pe_pre = type("PeriodEffect", (), {"effect": -1.0, "se": 0.2})()
        pe_post_1 = type("PeriodEffect", (), {"effect": 1.0, "se": 0.2})()
        pe_post_2 = type("PeriodEffect", (), {"effect": 3.0, "se": 0.2})()
        stub.period_effects = {-1: pe_pre, 0: pe_post_1, 1: pe_post_2}
        stub.post_period_effects = {0: pe_post_1, 1: pe_post_2}
        stub.pre_period_effects = {-1: pe_pre}
        stub.avg_att = 2.0
        stub.avg_se = 0.1
        stub.avg_p_value = 0.001
        stub.avg_conf_int = (1.8, 2.2)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None

        # Bypass the applicability-matrix gate by constructing the report
        # object and calling the extractor directly: the fix is in the
        # extractor, and MultiPeriod's applicability matrix may or may
        # not include heterogeneity at any given release.
        dr = DiagnosticReport(stub)
        effects = sorted(dr._collect_effect_scalars())
        assert effects == [1.0, 3.0], (
            f"Extractor must return only post-treatment effects "
            f"(no pre-period -1.0); got {effects}"
        )
        assert dr._heterogeneity_source() == "post_period_effects"

    def test_event_study_filters_pre_period_and_reference_markers(self):
        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        # Event study: pre horizons (rel<0), reference marker (n_groups=0),
        # non-finite row, and two valid post rows.
        stub.event_study_effects = {
            -2: {"effect": -3.0, "se": 0.2, "n_groups": 15},
            -1: {"effect": 0.0, "se": float("nan"), "n_groups": 0},  # reference marker
            0: {"effect": 1.0, "se": 0.2, "n_groups": 15},
            1: {"effect": 2.0, "se": 0.2, "n_groups": 12},
            2: {"effect": float("nan"), "se": 0.2, "n_groups": 5},  # non-finite
        }
        stub.overall_att = 1.5
        stub.overall_se = 0.1
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (1.3, 1.7)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.base_period = "universal"

        dr = DiagnosticReport(
            stub,
            run_parallel_trends=False,
            run_sensitivity=False,
            run_bacon=False,
        ).run_all()
        het = dr.schema["heterogeneity"]
        assert het["status"] == "ran"
        assert het["source"] == "event_study_effects_post"
        # Only rel>=0, finite, non-reference rows: {1.0, 2.0}.
        assert het["n_effects"] == 2
        assert het["min"] == pytest.approx(1.0)
        assert het["max"] == pytest.approx(2.0)
        assert het["sign_consistent"] is True


# ---------------------------------------------------------------------------
# Round-40 P1: survey-design threading for fit-faithful replay
# ---------------------------------------------------------------------------
class TestSurveyDesignThreading:
    """Round-40 P1 CI review on PR #318: when a fitted result carries
    ``survey_metadata``, Goodman-Bacon and the simple 2x2 PT helper
    cannot be faithfully replayed without the original ``SurveyDesign``.

    DR must:
      * accept a ``survey_design`` kwarg;
      * thread it to ``bacon_decompose(survey_design=...)`` when the
        user supplies it;
      * skip Bacon with an explicit reason when ``survey_metadata`` is
        set but ``survey_design`` is not supplied;
      * skip the simple 2x2 PT check with an explicit reason on
        survey-backed ``DiDResults`` (the helper has no
        ``survey_design`` parameter).
    """

    def _did_with_survey(self):
        from types import SimpleNamespace

        class DiDResults:
            pass

        obj = DiDResults()
        obj.att = 1.0
        obj.se = 0.2
        obj.t_stat = 5.0
        obj.p_value = 0.001
        obj.conf_int = (0.6, 1.4)
        obj.alpha = 0.05
        obj.n_obs = 400
        obj.n_treated = 100
        obj.n_control = 300
        obj.survey_metadata = SimpleNamespace(
            design_effect=1.25,
            effective_n=320.0,
            weight_type="pweight",
            n_strata=None,
            n_psu=None,
            df_survey=20.0,
            replicate_method=None,
        )
        obj.inference_method = "analytical"
        return obj

    def _staggered_stub_with_survey(self):
        """Lightweight CS-like stub carrying survey_metadata for Bacon gating."""
        from types import SimpleNamespace

        class CallawaySantAnnaResults:
            pass

        obj = CallawaySantAnnaResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.2
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.6, 1.4)
        obj.alpha = 0.05
        obj.n_obs = 600
        obj.n_treated = 200
        obj.n_control_units = 400
        obj.survey_metadata = SimpleNamespace(
            design_effect=1.5,
            effective_n=400.0,
            weight_type="pweight",
            n_strata=None,
            n_psu=None,
            df_survey=30.0,
            replicate_method=None,
        )
        obj.event_study_effects = None
        return obj

    def test_survey_backed_did_skips_2x2_pt_with_reason(self):
        """Survey-backed ``DiDResults`` must skip the 2x2 PT helper
        (``utils.check_parallel_trends`` is unweighted) and produce a
        skip reason naming the survey-design replay requirement.
        """
        obj = self._did_with_survey()
        import pandas as pd

        panel = pd.DataFrame(
            {
                "outcome": [1.0, 2.0, 1.1, 2.2],
                "post": [0, 1, 0, 1],
                "treated": [0, 0, 1, 1],
            }
        )
        dr = DiagnosticReport(
            obj,
            data=panel,
            outcome="outcome",
            time="post",
            treatment="treated",
        )
        assert "parallel_trends" not in dr.applicable_checks
        reason = dr.skipped_checks["parallel_trends"]
        assert "survey design" in reason.lower()
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "skipped"

    def test_survey_backed_did_skips_2x2_pt_even_when_survey_design_supplied(self):
        """Round-41 P3 regression: supplying ``survey_design`` does NOT
        unlock the simple 2x2 PT helper. ``utils.check_parallel_trends``
        has no survey-aware variant, so the helper cannot consume the
        design even when it is available; the check is skipped
        unconditionally on a survey-backed ``DiDResults`` and the skip
        reason must point the user at the precomputed-PT opt-in rather
        than imply that ``survey_design`` would have helped.
        """
        import pandas as pd

        obj = self._did_with_survey()
        panel = pd.DataFrame(
            {
                "outcome": [1.0, 2.0, 1.1, 2.2],
                "post": [0, 1, 0, 1],
                "treated": [0, 0, 1, 1],
            }
        )
        sentinel_design = object()
        dr = DiagnosticReport(
            obj,
            data=panel,
            outcome="outcome",
            time="post",
            treatment="treated",
            survey_design=sentinel_design,
        )
        # Supplying survey_design does not unlock 2x2 PT.
        assert "parallel_trends" not in dr.applicable_checks
        reason = dr.skipped_checks["parallel_trends"]
        # Reason must point at the precomputed-PT opt-in and must not
        # claim ``survey_design`` fixes this path.
        assert "precomputed" in reason.lower()
        assert "parallel_trends" in reason.lower()
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "skipped"

    def test_survey_backed_did_with_precomputed_pt_runs(self):
        """When the user supplies ``precomputed={'parallel_trends': ...}``
        on a survey-backed DiDResults, DR must honor the override rather
        than skip with the survey-design reason.
        """
        obj = self._did_with_survey()
        precomputed_pt = {
            "p_value": 0.42,
            "treated_trend": 0.05,
            "control_trend": 0.04,
            "trend_difference": 0.01,
            "t_statistic": 0.8,
        }
        dr = DiagnosticReport(
            obj,
            precomputed={"parallel_trends": precomputed_pt},
        )
        assert "parallel_trends" in dr.applicable_checks
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"

    def test_survey_backed_staggered_skips_bacon_without_survey_design(self):
        """CS-like survey-backed fit: Bacon replay must skip with a
        reason naming the survey-design requirement rather than produce
        an unweighted decomposition for a weighted estimate.
        """
        obj = self._staggered_stub_with_survey()
        import pandas as pd

        panel = pd.DataFrame(
            {
                "outcome": [1.0, 2.0, 1.1, 2.2, 1.2, 2.3, 1.3, 2.4],
                "unit": [1, 1, 2, 2, 3, 3, 4, 4],
                "period": [1, 2, 1, 2, 1, 2, 1, 2],
                "first_treat": [0, 0, 0, 0, 2, 2, 2, 2],
            }
        )
        dr = DiagnosticReport(
            obj,
            data=panel,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert "bacon" not in dr.applicable_checks
        reason = dr.skipped_checks["bacon"]
        assert "survey design" in reason.lower()
        assert "survey_design" in reason or "SurveyDesign" in reason
        bacon = dr.to_dict()["bacon"]
        assert bacon["status"] == "skipped"

    def test_survey_backed_staggered_threads_survey_design_to_bacon(self):
        """When ``survey_design`` is supplied, Bacon applicability flips
        back to runnable and ``bacon_decompose`` is invoked with the
        survey design. Assert via ``unittest.mock.patch`` that the
        kwarg is forwarded.
        """
        from unittest.mock import MagicMock, patch

        obj = self._staggered_stub_with_survey()
        import pandas as pd

        panel = pd.DataFrame(
            {
                "outcome": [1.0, 2.0, 1.1, 2.2, 1.2, 2.3, 1.3, 2.4],
                "unit": [1, 1, 2, 2, 3, 3, 4, 4],
                "period": [1, 2, 1, 2, 1, 2, 1, 2],
                "first_treat": [0, 0, 0, 0, 2, 2, 2, 2],
            }
        )

        sentinel_design = object()
        fake_decomp = MagicMock()
        fake_decomp.total_weight_treated_vs_never = 0.9
        fake_decomp.total_weight_earlier_vs_later = 0.05
        fake_decomp.total_weight_later_vs_earlier = 0.05
        fake_decomp.twfe_estimate = 1.1
        fake_decomp.n_timing_groups = 2

        with patch("diff_diff.bacon.bacon_decompose", return_value=fake_decomp) as m:
            dr = DiagnosticReport(
                obj,
                data=panel,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                survey_design=sentinel_design,
            )
            # Applicability gate passes since survey_design is supplied.
            assert "bacon" in dr.applicable_checks
            bacon = dr.to_dict()["bacon"]
            assert bacon["status"] == "ran"
            # The survey_design must be threaded through to
            # bacon_decompose as a kwarg so the replayed decomposition
            # matches the fitted design.
            assert m.called, "bacon_decompose was not called"
            _, kwargs = m.call_args
            assert kwargs.get("survey_design") is sentinel_design


# ---------------------------------------------------------------------------
# SpilloverDiD routing (Butts 2021 ring-indicator spillover-aware DiD)
# ---------------------------------------------------------------------------
class TestSpilloverDiDDiagnosticReportRouting:
    """``SpilloverDiDResults`` registration in ``_APPLICABILITY`` / ``_PT_METHOD``.

    Applicable set is TwoStageDiD's MINUS ``bacon``: ``parallel_trends``
    (event-study joint test on the per-event-time DIRECT-effect dynamics),
    ``design_effect`` (survey-weighted fits), ``heterogeneity``. ``bacon`` is
    excluded because SpilloverDiD identifies off far-away controls, not TWFE
    2x2 comparisons; ``pretrends_power`` / ``sensitivity`` / ``epv`` /
    ``estimator_native`` are likewise not applicable.
    """

    @staticmethod
    def _staggered_df():
        from tests._dgp_utils import generate_butts_staggered_dgp

        return generate_butts_staggered_dgp(seed=42)

    @staticmethod
    def _fit(df, *, event_study=True, survey_design=None):
        from diff_diff import SpilloverDiD

        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
            vcov_type="hc1",
            event_study=event_study,
            horizon_max=2 if event_study else None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return est.fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=survey_design,
            )

    def _report(self, res, df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return DiagnosticReport(
                res, data=df, outcome="y", unit="unit", time="time", first_treat="first_treat"
            )

    def test_event_study_fit_routes_pt_and_heterogeneity(self):
        df = self._staggered_df()
        res = self._fit(df)
        dr = self._report(res, df)
        schema = dr.to_dict()
        # Non-survey event-study fit: exactly PT + heterogeneity apply.
        assert set(dr.applicable_checks) == {"parallel_trends", "heterogeneity"}
        assert schema["parallel_trends"]["status"] == "ran"
        assert schema["heterogeneity"]["status"] == "ran"
        # Excluded checks (bacon deliberately excluded; the rest have no adapter).
        for chk in ("bacon", "pretrends_power", "sensitivity", "epv"):
            assert chk not in dr.applicable_checks
            assert schema[chk]["status"] == "not_applicable"

    def test_survey_fit_routes_design_effect(self):
        from diff_diff import SurveyDesign

        df = self._staggered_df().copy()
        df["w"] = 1.0
        units_sorted = sorted(df["unit"].unique())
        unit_to_psu = {u: i for i, u in enumerate(units_sorted)}
        df["psu"] = df["unit"].map(unit_to_psu)
        res = self._fit(df, survey_design=SurveyDesign(weights="w", psu="psu"))
        dr = self._report(res, df)
        schema = dr.to_dict()
        assert "design_effect" in dr.applicable_checks
        assert schema["design_effect"]["status"] == "ran"

    def test_aggregate_only_fit_skips_pt_with_estimator_aware_message(self):
        df = self._staggered_df()
        res = self._fit(df, event_study=False)
        dr = self._report(res, df)
        skipped = dr.skipped_checks
        # SpilloverDiD's event-study switch is the ``event_study=True``
        # CONSTRUCTOR kwarg, not the generic ``aggregate='event_study'`` the
        # shared staggered-estimator message points at.
        assert "parallel_trends" in skipped
        assert "event_study=True" in skipped["parallel_trends"]
        assert "aggregate='event_study'" not in skipped["parallel_trends"]
        assert "parallel_trends" not in dr.applicable_checks
        assert "heterogeneity" in skipped
        assert "group/event-study effects" in skipped["heterogeneity"]

    def test_business_report_smoke(self):
        from diff_diff import BusinessReport

        df = self._staggered_df()
        res = self._fit(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            br = BusinessReport(
                res, data=df, outcome="y", unit="unit", time="time", first_treat="first_treat"
            )
            schema = br.to_dict()
        # Registration flows through to the target-parameter block; BR no
        # longer routes spillover to an empty diagnostic set.
        assert schema["target_parameter"]["aggregation"] == "spillover"
        assert schema["target_parameter"]["headline_attribute"] == "att"


# ---------------------------------------------------------------------------
# Public API exposure
# ---------------------------------------------------------------------------
def test_public_api_exports():
    for name in ("DiagnosticReport", "DiagnosticReportResults", "DIAGNOSTIC_REPORT_SCHEMA_VERSION"):
        assert hasattr(dd, name), f"diff_diff must export {name}"


# ---------------------------------------------------------------------------
# ES-VCV consumers: Stacked/TwoStage now expose event_study_vcov (M-092
# follow-up), upgrading the parallel-trends check from the Bonferroni
# fallback to the joint-Wald path. Intended behavior change, CHANGELOG'd.
# ---------------------------------------------------------------------------
class TestStackedTwoStageJointWaldUpgrade:
    @staticmethod
    def _panel(seed=42):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(60):
            g = [4, 6, 0][u % 3]
            for t in range(1, 11):
                y = 1.0 + 0.1 * t + u * 0.01 + (1.5 if g and t >= g else 0.0) + rng.normal(0, 0.3)
                rows.append({"unit": u, "time": t, "outcome": y, "first_treat": g})
        return pd.DataFrame(rows)

    def test_stacked_pt_check_uses_joint_wald(self):
        from diff_diff import StackedDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = StackedDiD(kappa_pre=2, kappa_post=2).fit(
                self._panel(),
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
            rep = DiagnosticReport(res).run_all()
        pt = rep.to_dict()["parallel_trends"]
        assert pt["method"] == "joint_wald_event_study", (
            f"StackedDiD persists its full ES VCV; the PT check must take the "
            f"joint-Wald path, got {pt.get('method')}"
        )
        assert np.isfinite(pt["joint_p_value"])

    def test_two_stage_nan_vcov_row_excluded_from_pt_family(self):
        # Induce REAL Path-B horizons (two_stage.py all-filtered branch):
        # balance_e=4 drops cohort 7 (unobserved through e=4) AFTER the
        # horizon grid is built from all rows, so its exclusive deep
        # pre-horizons (-6, -5) stay estimated Stage-2 columns whose rows
        # are all balance-filtered -> NaN effect/SE and rank-guard NaN
        # rows/cols in the persisted VCV. The PT check must EXCLUDE those
        # horizons from the tested family and run the joint Wald on the
        # finite pre-period sub-block - never compute a statistic THROUGH
        # the NaN covariance entries.
        from diff_diff.two_stage import TwoStageDiD

        rng = np.random.default_rng(9)
        rows = []
        for u in range(90):
            g = [5, 7, 0][u % 3]
            for t in range(1, 11):
                y = 1.0 + 0.1 * t + u * 0.01 + (1.0 if g and t >= g else 0.0) + rng.normal(0, 0.3)
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "outcome": y,
                        "first_treat": g if g else np.nan,
                    }
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TwoStageDiD(pretrends=True).fit(
                pd.DataFrame(rows),
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
                balance_e=4,
            )
        # Path-B precondition: -6 is an estimated column with NaN inference
        # and a NaN-filled VCV row; finite pre-periods {-4, -3, -2} remain.
        assert res.event_study_effects[-6]["n_obs"] == 0
        assert not np.isfinite(res.event_study_effects[-6]["se"])
        idx = res.event_study_vcov_index
        assert -6 in idx and -4 in idx
        pos = idx.index(-6)
        assert np.isnan(res.event_study_vcov[pos, :]).all()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = DiagnosticReport(res).run_all()
        pt = rep.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        # Joint Wald ran on the FINITE pre-period family only: the Path-B
        # horizons (NaN se) are excluded by the collector, so the tested
        # block is finite and the p-value is a real number, never
        # NaN-contaminated by the excluded rows of the matrix.
        assert pt["method"] == "joint_wald_event_study"
        assert np.isfinite(pt["joint_p_value"])
        tested = {row["period"] for row in pt["per_period"]}
        assert tested == {-4, -3, -2}
        assert pt["df"] == 3

    @staticmethod
    def _hc2bm_fit(monkeypatch=None):
        from diff_diff import StackedDiD

        rng = np.random.default_rng(42)
        rows = []
        for u in range(60):
            g = [4, 6, 0][u % 3]
            for t in range(1, 11):
                y = 1.0 + 0.1 * t + u * 0.01 + (1.5 if g and t >= g else 0.0) + rng.normal(0, 0.3)
                rows.append({"unit": u, "time": t, "outcome": y, "first_treat": g})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm").fit(
                pd.DataFrame(rows),
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_stacked_hc2_bm_fail_closed_pt_is_inconclusive(self, monkeypatch):
        # Provenance-laundering regression: when the BM contrast
        # DOF is unavailable, StackedDiD fails closed (finite effect/SE,
        # deliberately-NaN t/p/CI) but the persisted covariance stays
        # finite. The joint-Wald path consumes only effects + covariance,
        # so without the provenance guard it would publish a finite joint
        # p-value for inference the estimator refused to produce.
        import diff_diff.linalg as _linalg

        def _broken(*args, **kwargs):
            raise ValueError("forced BM failure")

        monkeypatch.setattr(_linalg, "_compute_cr2_bm_contrast_dof", _broken)
        res = self._hc2bm_fit()
        # Fail-closed precondition: finite effect/SE, NaN p, finite vcov.
        pre = res.event_study_effects[-2]
        assert np.isfinite(pre["effect"]) and np.isfinite(pre["se"])
        assert not np.isfinite(pre["p_value"])
        assert res.event_study_vcov is not None
        assert np.isfinite(np.diag(res.event_study_vcov)).all()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pt = DiagnosticReport(res).run_all().to_dict()["parallel_trends"]
        assert pt["method"] == "inconclusive"
        assert pt["verdict"] == "inconclusive"
        assert pt["joint_p_value"] is None
        assert pt["n_dropped_undefined"] >= 1

    def test_stacked_hc2_bm_routes_to_bonferroni_not_chi_square_wald(self):
        # Healthy hc2_bm fit: every pre-row's p-value is BM-adjusted. The
        # generic chi-square joint Wald would discard that small-sample
        # correction, so the PT check must use Bonferroni over the
        # BM-adjusted per-row p-values despite the persisted covariance
        # (REPORTING.md "hc2_bm parallel-trends policy"; AHT/HTZ tracked in
        # DEFERRED.md).
        res = self._hc2bm_fit()
        assert res.event_study_vcov is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pt = DiagnosticReport(res).run_all().to_dict()["parallel_trends"]
        assert pt["method"] == "bonferroni"
        assert np.isfinite(pt["joint_p_value"])

    def test_vcov_backed_nan_p_rows_force_inconclusive(self):
        # Construction-level twin of the hc2_bm case for the collapsed
        # replicate-survey df mode: retained pre-rows with finite
        # effect/SE, NaN p (safe_inference df<=0 sentinel), and a persisted
        # finite covariance. The joint-Wald provenance guard must return
        # inconclusive, never a finite p through the covariance.
        class _FakeCovBacked:
            alpha = 0.05
            vcov_type = "hc1"
            event_study_effects = {
                -3: {
                    "effect": 0.10,
                    "se": 0.05,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": 40,
                },
                -2: {
                    "effect": 0.05,
                    "se": 0.04,
                    "t_stat": 1.25,
                    "p_value": 0.21,
                    "conf_int": (-0.03, 0.13),
                    "n_obs": 40,
                },
                -1: {
                    "effect": 0.0,
                    "se": 0.0,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": 0,
                },
                0: {
                    "effect": 0.9,
                    "se": 0.1,
                    "t_stat": 9.0,
                    "p_value": 0.0,
                    "conf_int": (0.7, 1.1),
                    "n_obs": 40,
                },
            }
            event_study_vcov = np.array(
                [[0.0025, 0.0002, 0.0001], [0.0002, 0.0016, 0.0002], [0.0001, 0.0002, 0.01]]
            )
            event_study_vcov_index = [-3, -2, 0]

        from diff_diff.diagnostic_report import DiagnosticReport as _DR

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pt = _DR(_FakeCovBacked())._pt_event_study()
        assert pt["method"] == "inconclusive"
        assert pt["joint_p_value"] is None
        assert pt["n_dropped_undefined"] == 1

    @staticmethod
    def _low_cluster_panel():
        # 12 clusters, one cohort at t=14 with a 12-lead pre window: the
        # cluster-sandwich covariance has rank bounded by the cluster-score
        # dimension, so the retained pre-period block is SINGULAR (rank <
        # n_pre) - the normal low-G/high-lead edge case.
        rng = np.random.default_rng(3)
        rows = []
        for u in range(12):
            g = 14 if u % 2 == 0 else 0
            for t in range(1, 18):
                y = 1.0 + 0.05 * t + u * 0.05 + (1.0 if g and t >= g else 0.0) + rng.normal(0, 0.3)
                rows.append({"unit": u, "time": t, "outcome": y, "first_treat": g})
        return pd.DataFrame(rows)

    def test_stacked_singular_vcov_downgrades_to_bonferroni(self):
        from diff_diff import StackedDiD

        data = self._low_cluster_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = StackedDiD(kappa_pre=12, kappa_post=2).fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        # Precondition: the retained pre-period block is genuinely singular.
        V = res.event_study_vcov
        idx = res.event_study_vcov_index
        pre_idx = [i for i, h in enumerate(idx) if h < -1]
        sub = V[np.ix_(pre_idx, pre_idx)]
        sub = 0.5 * (sub + sub.T)
        assert np.linalg.matrix_rank(sub) < len(pre_idx)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pt = DiagnosticReport(res).run_all().to_dict()["parallel_trends"]
        # Rank guard: never a Wald statistic through the singular block -
        # the unguarded quadratic form here is ~ -2.5e14 with chi2 p=1.0.
        assert pt["method"] == "bonferroni"
        assert pt["test_statistic"] is None
        assert np.isfinite(pt["joint_p_value"])

    def test_two_stage_singular_vcov_downgrades_to_bonferroni(self):
        from diff_diff.two_stage import TwoStageDiD

        data = self._low_cluster_panel()
        data["first_treat"] = data["first_treat"].replace(0, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TwoStageDiD(pretrends=True).fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )
        V = res.event_study_vcov
        idx = res.event_study_vcov_index
        pre_idx = [i for i, h in enumerate(idx) if h < -1 and np.isfinite(V[i, i])]
        sub = V[np.ix_(pre_idx, pre_idx)]
        sub = 0.5 * (sub + sub.T)
        assert np.linalg.matrix_rank(sub) < len(pre_idx)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pt = DiagnosticReport(res).run_all().to_dict()["parallel_trends"]
        assert pt["method"] == "bonferroni"
        assert pt["test_statistic"] is None
        assert np.isfinite(pt["joint_p_value"])

    def test_joint_wald_never_negative_statistic(self):
        # Belt-and-braces contract: a Wald statistic is a PSD quadratic
        # form. Feed a singular-but-finite covariance directly through the
        # event-study path and assert the guard rejects it rather than
        # reporting a negative statistic as chi-square p=1.0.
        rng = np.random.default_rng(0)
        A = rng.normal(size=(3, 2))
        V = A @ A.T  # PSD, rank 2 < 3

        class _FakeSingular:
            alpha = 0.05
            vcov_type = "hc1"
            event_study_effects = {
                -4: {
                    "effect": 0.10,
                    "se": float(np.sqrt(V[0, 0])),
                    "t_stat": 1.0,
                    "p_value": 0.3,
                    "conf_int": (-0.1, 0.3),
                    "n_obs": 40,
                },
                -3: {
                    "effect": 0.05,
                    "se": float(np.sqrt(V[1, 1])),
                    "t_stat": 0.5,
                    "p_value": 0.6,
                    "conf_int": (-0.1, 0.2),
                    "n_obs": 40,
                },
                -2: {
                    "effect": -0.07,
                    "se": float(np.sqrt(V[2, 2])),
                    "t_stat": -0.7,
                    "p_value": 0.5,
                    "conf_int": (-0.2, 0.1),
                    "n_obs": 40,
                },
                -1: {
                    "effect": 0.0,
                    "se": 0.0,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": 0,
                },
                0: {
                    "effect": 0.9,
                    "se": 0.1,
                    "t_stat": 9.0,
                    "p_value": 0.0,
                    "conf_int": (0.7, 1.1),
                    "n_obs": 40,
                },
            }
            event_study_vcov = V
            event_study_vcov_index = [-4, -3, -2]

        from diff_diff.diagnostic_report import DiagnosticReport as _DR2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pt = _DR2(_FakeSingular())._pt_event_study()
        assert pt["method"] == "bonferroni"
        stat = pt["test_statistic"]
        assert stat is None or stat >= 0.0


class TestStackedAlwaysComputedSurfaceCheckFlips:
    """M-024: the always-materialized ES surface flips two DR checks on
    PLAIN StackedDiD fits (executed-verified during the plan review):
    heterogeneity always, parallel_trends at kappa_pre >= 2."""

    @staticmethod
    def _panel(seed=42):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(60):
            g = [4, 6, 0][u % 3]
            for t in range(1, 11):
                y = 1.0 + 0.1 * t + u * 0.01 + (1.5 if g and t >= g else 0.0) + rng.normal(0, 0.3)
                rows.append({"unit": u, "time": t, "outcome": y, "first_treat": g})
        return pd.DataFrame(rows)

    def _fit(self, kappa_pre):
        from diff_diff import StackedDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return StackedDiD(kappa_pre=kappa_pre, kappa_post=2).fit(
                self._panel(),
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def _report(self, res):
        from diff_diff.diagnostic_report import DiagnosticReport

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return DiagnosticReport(
                res,
                data=self._panel(),
                unit="unit",
                time="time",
                outcome="outcome",
                first_treat="first_treat",
            ).to_dict()

    def test_kappa_pre_2_runs_heterogeneity_and_parallel_trends(self):
        d = self._report(self._fit(kappa_pre=2))
        assert d["parallel_trends"]["status"] == "ran"
        assert d["heterogeneity"]["status"] == "ran"
        assert d["heterogeneity"]["source"] == "event_study_effects_post"

    def test_kappa_pre_1_default_heterogeneity_only(self):
        # The default kappa_pre=1 grid minus the reference has ZERO
        # estimated pre-periods, so PT stays skipped - with the M-024
        # per-type remediation, NOT the generic (now-inert for Stacked)
        # aggregate='event_study' advice.
        d = self._report(self._fit(kappa_pre=1))
        assert d["heterogeneity"]["status"] == "ran"
        pt = d["parallel_trends"]
        assert pt["status"] == "skipped"
        reason = pt.get("reason", "") + " ".join(str(v) for v in pt.values() if isinstance(v, str))
        assert "kappa_pre >= 2" in reason
        assert "aggregate='event_study'" not in reason
