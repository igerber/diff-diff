"""Tests for ``diff_diff.business_report.BusinessReport``.

Covers the expanded test list from the approved plan:
- Schema contract across result types.
- JSON round-trip.
- BR-DR integration (auto, explicit, False).
- ``honest_did_results=`` passthrough (no re-computation).
- Unit-label behavior (pp vs $ differ; column-name fallback).
- Log-points unit policy (no arithmetic translation; informational caveat).
- Significance-chasing guard boundary.
- Pre-trends verdict thresholds (three bins routed through BR phrasing).
- Power-aware phrasing (three tiers + underpowered fallback).
- NaN ATT surfaces a caveat and does not crash.
- ``include_appendix`` toggle.
- ``BusinessReport(BaconDecompositionResults)`` raises TypeError.
- Survey metadata passthrough to schema + phrasing.
- Single-knob alpha drives both CI level and phrasing.
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
    BusinessContext,
    BusinessReport,
    CallawaySantAnna,
    DiagnosticReport,
    DifferenceInDifferences,
    MultiPeriodDiD,
    SyntheticDiD,
    bacon_decompose,
    generate_did_data,
    generate_factor_data,
    generate_staggered_data,
    synthetic_control,
)
from diff_diff.business_report import BUSINESS_REPORT_SCHEMA_VERSION

warnings.filterwarnings("ignore")

_BR_TOP_LEVEL_KEYS = {
    "schema_version",
    "estimator",
    "context",
    "headline",
    "target_parameter",
    "assumption",
    "pre_trends",
    "sensitivity",
    "sample",
    "heterogeneity",
    "robustness",
    "diagnostics",
    "next_steps",
    "caveats",
    "references",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def did_fit():
    df = generate_did_data(n_units=80, n_periods=4, treatment_effect=1.5, seed=7)
    did = DifferenceInDifferences().fit(df, outcome="outcome", treatment="treated", post="post")
    return did, df


@pytest.fixture(scope="module")
def event_study_fit():
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
    sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
    # base_period='universal' so DR's sensitivity check can run without
    # hitting the round-5 methodology-critical skip (Rambachan-Roth bounds
    # are not interpretable on consecutive-comparison pre-periods).
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
def sdid_fit():
    fdf = generate_factor_data(n_units=25, n_pre=8, n_post=4, n_treated=4, seed=11)
    sdid = SyntheticDiD().fit(fdf, outcome="outcome", unit="unit", time="period", treatment="treat")
    return sdid, fdf


@pytest.fixture  # function-scoped: some tests run in_space_placebo(), which mutates the result
def scm_fit():
    rng = np.random.default_rng(0)
    T, T0, n = 8, 6, 4
    years = list(range(2000, 2000 + T))
    donors = {
        j: rng.normal(10, 2) + rng.normal(0, 0.3) * np.arange(T) + rng.normal(0, 0.15, T)
        for j in range(n)
    }
    treated = 0.6 * donors[0] + 0.4 * donors[1] + rng.normal(0, 0.08, T)
    treated[T0:] += 3.0
    rows = []
    for j in range(n):
        for t in range(T):
            rows.append({"unit": f"d{j}", "year": years[t], "y": donors[j][t], "treated": 0})
    for t in range(T):
        rows.append({"unit": "treated", "year": years[t], "y": treated[t], "treated": int(t >= T0)})
    df = pd.DataFrame(rows)
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


@pytest.fixture(scope="module")
def edid_fit():
    from diff_diff import EfficientDiD

    sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
    edid = EfficientDiD().fit(
        sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
    )
    return edid, sdf


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
class TestSchemaContract:
    def test_top_level_keys(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=False)
        assert set(br.to_dict().keys()) == _BR_TOP_LEVEL_KEYS

    def test_schema_version(self, event_study_fit):
        fit, _ = event_study_fit
        assert (
            BusinessReport(fit, auto_diagnostics=False).to_dict()["schema_version"]
            == BUSINESS_REPORT_SCHEMA_VERSION
        )

    def test_json_round_trip(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(
            fit,
            outcome_label="sales",
            outcome_unit="$",
            treatment_label="the policy",
        )
        dumped = json.dumps(br.to_dict())
        assert len(dumped) > 0
        assert json.loads(dumped)["schema_version"] == BUSINESS_REPORT_SCHEMA_VERSION

    def test_json_round_trip_sdid(self, sdid_fit):
        fit, _ = sdid_fit
        br = BusinessReport(fit, outcome_label="revenue", outcome_unit="$")
        dumped = json.dumps(br.to_dict())
        assert len(dumped) > 0


# ---------------------------------------------------------------------------
# BR ↔ DR integration
# ---------------------------------------------------------------------------
class TestDiagnosticsIntegration:
    def test_auto_diagnostics_true_populates_diagnostics_block(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=True)
        d = br.to_dict()
        assert d["diagnostics"]["status"] == "ran"
        assert "schema" in d["diagnostics"]

    def test_auto_diagnostics_false_skips(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=False)
        d = br.to_dict()
        assert d["diagnostics"]["status"] == "skipped"
        assert "auto_diagnostics=False" in d["diagnostics"]["reason"]

    def test_explicit_diagnostics_results_takes_precedence(self, event_study_fit):
        fit, _ = event_study_fit
        dr = DiagnosticReport(fit)
        dr_results = dr.run_all()
        br = BusinessReport(fit, diagnostics=dr_results)
        d = br.to_dict()
        assert d["diagnostics"]["status"] == "ran"
        # Same dict identity shows the supplied results were used verbatim.
        assert d["diagnostics"]["schema"] is dr_results.schema

    def test_explicit_diagnostics_report_runs(self, event_study_fit):
        fit, _ = event_study_fit
        dr = DiagnosticReport(fit)
        br = BusinessReport(fit, diagnostics=dr)
        assert br.to_dict()["diagnostics"]["status"] == "ran"

    def test_diagnostics_wrong_type_raises(self, event_study_fit):
        fit, _ = event_study_fit
        with pytest.raises(TypeError):
            BusinessReport(fit, diagnostics="not a DR")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HonestDiD passthrough
# ---------------------------------------------------------------------------
class TestHonestDiDPassthrough:
    def test_supplied_sensitivity_is_not_recomputed(self, event_study_fit):
        fit, _ = event_study_fit

        class _FakeSens:
            M_values = np.array([0.5, 1.0])
            bounds = [(0.1, 2.0), (-0.2, 2.5)]
            robust_cis = [(0.05, 2.1), (-0.3, 2.6)]
            breakdown_M = 1.5
            method = "relative_magnitude"
            original_estimate = 1.0
            original_se = 0.2
            alpha = 0.05

        fake = _FakeSens()
        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis") as mock:
            br = BusinessReport(fit, honest_did_results=fake)
            schema = br.to_dict()
            mock.assert_not_called()
        sens = schema["sensitivity"]
        assert sens["status"] == "computed"
        assert sens["breakdown_M"] == 1.5


# ---------------------------------------------------------------------------
# Unit labels and policy
# ---------------------------------------------------------------------------
class TestUnitLabels:
    def test_dollar_unit_formats_currency(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(fit, outcome_label="sales", outcome_unit="$", auto_diagnostics=False)
        headline = br.headline()
        assert "$" in headline

    def test_pp_unit_formats_percentage_points(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(
            fit, outcome_label="awareness", outcome_unit="pp", auto_diagnostics=False
        )
        headline = br.headline()
        assert "pp" in headline

    def test_zero_config_falls_back_to_generic_label(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(fit, auto_diagnostics=False)
        d = br.to_dict()
        assert d["context"]["outcome_label"] == "the outcome"
        assert d["context"]["treatment_label"] == "the treatment"

    def test_log_points_emits_unit_policy_caveat(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(fit, outcome_unit="log_points", auto_diagnostics=False)
        caveats = br.caveats()
        topics = {c.get("topic") for c in caveats}
        assert "unit_policy" in topics


# ---------------------------------------------------------------------------
# Significance phrasing
# ---------------------------------------------------------------------------
class TestOutcomeDirection:
    """outcome_direction selects value-laden vs neutral verbs."""

    def test_higher_is_better_positive_effect_uses_lifted(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(
            fit,
            outcome_label="sales",
            outcome_unit="$",
            outcome_direction="higher_is_better",
            treatment_label="the policy",
            auto_diagnostics=False,
        )
        headline = br.headline()
        assert "lifted" in headline
        assert "increased" not in headline

    def test_lower_is_better_positive_effect_uses_worsened(self, cs_fit):
        fit, _ = cs_fit  # CS has a positive effect on this seed
        br = BusinessReport(
            fit,
            outcome_label="churn",
            outcome_unit="%",
            outcome_direction="lower_is_better",
            treatment_label="the change",
            auto_diagnostics=False,
        )
        headline = br.headline()
        assert "worsened" in headline

    def test_direction_none_uses_neutral_verb(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(
            fit,
            outcome_label="sales",
            outcome_unit="$",
            auto_diagnostics=False,
        )
        headline = br.headline()
        assert "increased" in headline
        assert "lifted" not in headline


class TestWarningsPassthrough:
    """Broad exception handling still records provenance in schema.warnings."""

    def test_diagnostic_error_surfaces_as_top_level_warning(self, event_study_fit):
        fit, _ = event_study_fit

        def _raise(*args, **kwargs):
            raise RuntimeError("synthetic test failure")

        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis", side_effect=_raise):
            br = BusinessReport(fit, auto_diagnostics=True)
            schema = br.to_dict()
            inner = schema["diagnostics"]["schema"]
            # The error is recorded at the section level...
            assert inner["sensitivity"]["status"] == "error"
            # ...AND surfaced at the top level for quick scanning.
            assert any("sensitivity:" in w for w in inner["warnings"])
            assert any("synthetic test failure" in w for w in inner["warnings"])


class TestSignificancePhrasing:
    def test_high_significance_produces_strong_language(self, cs_fit):
        """CS on this seed has p ~ 1e-56 (very strong) -> 'strongly supported'."""
        fit, _ = cs_fit
        br = BusinessReport(fit, outcome_label="sales", outcome_unit="$")
        summary = br.summary()
        assert "strongly supported" in summary

    def test_near_threshold_caveat(self, event_study_fit):
        """Fabricate a p-value near 0.05 to exercise the significance-chasing guard."""
        fit, _ = event_study_fit
        # Monkey-patch the result to land p_value in (0.04, 0.051).
        original = fit.avg_p_value
        try:
            fit.avg_p_value = 0.045
            br = BusinessReport(fit, auto_diagnostics=False)
            caveats = br.caveats()
            topics = {c.get("topic") for c in caveats}
            assert "near_significance" in topics
        finally:
            fit.avg_p_value = original

    def test_far_from_threshold_no_near_caveat(self, event_study_fit):
        fit, _ = event_study_fit
        original = fit.avg_p_value
        try:
            fit.avg_p_value = 0.010
            br = BusinessReport(fit, auto_diagnostics=False)
            topics = {c.get("topic") for c in br.caveats()}
            assert "near_significance" not in topics
        finally:
            fit.avg_p_value = original


# ---------------------------------------------------------------------------
# Pre-trends verdict + power tier phrasing
# ---------------------------------------------------------------------------
class TestPreTrendsVerdictPhrasing:
    """Verdict and tier should flow through into schema AND phrasing."""

    def test_verdict_and_tier_surface_in_schema(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=True)
        pt = br.to_dict()["pre_trends"]
        # This fixture has a clear violation and an underpowered test — both set.
        assert pt["status"] == "computed"
        assert pt["verdict"] in {
            "no_detected_violation",
            "some_evidence_against",
            "clear_violation",
        }

    def test_clear_violation_phrased_tentatively(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=True)
        if br.to_dict()["pre_trends"].get("verdict") == "clear_violation":
            summary = br.summary()
            assert "tentative" in summary or "reject parallel trends" in summary

    def test_underpowered_phrasing_uses_hedge_language(self, cs_fit):
        """CS fit on this seed typically produces 'no_detected_violation' + underpowered."""
        fit, sdf = cs_fit
        # Force the CS fit through our BR pipeline.
        br = BusinessReport(
            fit,
            outcome_label="sales",
            outcome_unit="$",
            diagnostics=DiagnosticReport(
                fit,
                data=sdf,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            ),
        )
        pt = br.to_dict()["pre_trends"]
        if pt.get("verdict") == "no_detected_violation":
            summary = br.summary()
            # One of the three tier-specific phrases should appear.
            assert (
                "limited power" in summary
                or "moderately informative" in summary
                or "well-powered" in summary
                or "likely have been detected" in summary
            )


# ---------------------------------------------------------------------------
# NaN ATT
# ---------------------------------------------------------------------------
class TestNaNATT:
    def test_nan_att_produces_caveat_and_does_not_crash(self, event_study_fit):
        fit, _ = event_study_fit
        original = fit.avg_att
        try:
            fit.avg_att = float("nan")
            br = BusinessReport(fit, auto_diagnostics=False)
            summary = br.summary()
            caveats = br.caveats()
            assert isinstance(summary, str)
            assert any(c.get("topic") == "estimation_failure" for c in caveats)
            assert br.to_dict()["headline"]["sign"] == "undefined"
        finally:
            fit.avg_att = original


# ---------------------------------------------------------------------------
# include_appendix toggle
# ---------------------------------------------------------------------------
class TestAppendix:
    def test_include_appendix_true_embeds_summary(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=False, include_appendix=True)
        md = br.full_report()
        assert "## Technical Appendix" in md

    def test_include_appendix_false_omits(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=False, include_appendix=False)
        md = br.full_report()
        assert "## Technical Appendix" not in md

    def test_appendix_summary_failure_is_visible(self, event_study_fit):
        fit, _ = event_study_fit
        with patch.object(fit, "summary", side_effect=RuntimeError("internal path /tmp/private")):
            br = BusinessReport(fit, auto_diagnostics=False, include_appendix=True)
            md = br.full_report()

        assert "## Technical Appendix" in md
        assert "Technical appendix unavailable" in md
        assert "RuntimeError" in md

    def test_appendix_summary_failure_does_not_leak_exception_details(self, event_study_fit):
        fit, _ = event_study_fit
        with patch.object(fit, "summary", side_effect=RuntimeError("secret-token-123")):
            br = BusinessReport(fit, auto_diagnostics=False, include_appendix=True)
            md = br.full_report()

        assert "secret-token-123" not in md
        assert "Traceback" not in md

    def test_appendix_stringification_failure_is_visible(self, event_study_fit):
        class _BadSummary:
            def __str__(self):
                raise ValueError("sensitive-render-context")

        fit, _ = event_study_fit
        with patch.object(fit, "summary", return_value=_BadSummary()):
            br = BusinessReport(fit, auto_diagnostics=False, include_appendix=True)
            md = br.full_report()

        assert "## Technical Appendix" in md
        assert "Technical appendix unavailable" in md
        assert "ValueError" in md
        assert "sensitive-render-context" not in md


# ---------------------------------------------------------------------------
# BaconDecompositionResults
# ---------------------------------------------------------------------------
class TestBaconTypeError:
    def test_br_on_bacon_raises(self):
        sdf = generate_staggered_data(n_units=30, n_periods=6, treatment_effect=1.5, seed=7)
        bacon = bacon_decompose(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        with pytest.raises(TypeError, match="BaconDecompositionResults is a diagnostic"):
            BusinessReport(bacon)


# ---------------------------------------------------------------------------
# Survey metadata passthrough
# ---------------------------------------------------------------------------
class TestSurveyPassthrough:
    def test_survey_absent_yields_null_survey_block(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(fit, auto_diagnostics=False)
        d = br.to_dict()
        assert d["sample"]["survey"] is None

    def test_survey_present_populates_block(self, event_study_fit):
        """Synthetically attach a survey_metadata shim and verify BR surfaces it."""
        fit, _ = event_study_fit

        class _ShimMeta:
            weight_type = "pweight"
            effective_n = 120.0
            design_effect = 2.5
            sum_weights = 200.0
            n_strata = 8
            n_psu = 20
            df_survey = 18
            replicate_method = None

        original = fit.survey_metadata
        try:
            fit.survey_metadata = _ShimMeta()
            br = BusinessReport(fit, auto_diagnostics=False)
            survey = br.to_dict()["sample"]["survey"]
            assert survey is not None
            assert survey["weight_type"] == "pweight"
            assert survey["design_effect"] == 2.5
            assert survey["is_trivial"] is False

            summary = br.summary()
            # When DEFF >= 1.5 we inject a caveat or a summary sentence.
            assert (
                "design effect" in summary.lower()
                or "effective sample size" in summary.lower()
                or any(c.get("topic") == "design_effect" for c in br.caveats())
            )
        finally:
            fit.survey_metadata = original


# ---------------------------------------------------------------------------
# Single-knob alpha
# ---------------------------------------------------------------------------
class TestAlphaKnob:
    def test_alpha_equal_to_result_alpha_drives_ci_level(self, event_study_fit):
        """When caller's alpha matches the fit's native alpha, ``ci_level``
        reflects that alpha (e.g., alpha=0.05 -> 95% CI)."""
        fit, _ = event_study_fit
        br = BusinessReport(fit, alpha=0.05, auto_diagnostics=False)
        assert br.to_dict()["headline"]["ci_level"] == 95

    def test_alpha_mismatch_preserves_fitted_ci_at_native_level(self, event_study_fit):
        """Round-7 regression: a caller alpha that differs from the fit's
        native alpha must NOT recompute a z-based CI (the fit used t-based
        inference with a finite ``df`` that BR cannot reproduce from
        ``(att, se)`` alone). The displayed CI stays at the fit's native
        level, while significance phrasing uses the caller's alpha. A
        caveat records the override.
        """
        import math

        fit, _ = event_study_fit
        br95 = BusinessReport(fit, alpha=0.05, auto_diagnostics=False)
        br90 = BusinessReport(fit, alpha=0.10, auto_diagnostics=False)
        h95 = br95.to_dict()["headline"]
        h90 = br90.to_dict()["headline"]
        if h95["effect"] is not None and math.isfinite(h95["effect"]):
            # Bounds must match between the two: the alpha=0.10 call
            # preserves the fit's 95% CI rather than recomputing a 90% z-CI.
            assert h90["ci_lower"] == pytest.approx(h95["ci_lower"])
            assert h90["ci_upper"] == pytest.approx(h95["ci_upper"])
        # ``ci_level`` stays at the fit's native level in both cases.
        assert h95["ci_level"] == 95
        assert h90["ci_level"] == 95
        # Override is surfaced as an info-level caveat.
        topics = {c.get("topic") for c in br90.caveats()}
        assert "alpha_override_preserved" in topics, (
            "Alpha mismatch must surface a caveat documenting the preserved "
            "native CI level; topics seen: " + str(topics)
        )


class TestAlphaOverrideBootstrapAndFiniteDF:
    """Alpha override preserves the fitted CI on any inference contract
    that cannot be reproduced from point-estimate + SE alone (bootstrap /
    wild cluster bootstrap / percentile / jackknife / placebo / finite-df
    survey / undefined-d.f. replicate / analytical t-quantile). The
    displayed CI stays at the fit's native level; significance phrasing
    still uses the caller's alpha; an informational caveat records the
    override.
    """

    class _BootstrapResultStub:
        """Minimal stub shaped like a bootstrap-inferred result."""

        def __init__(self):
            self.att = 1.0
            self.se = 0.5
            self.p_value = 0.04
            # Original 95% CI from the bootstrap distribution.
            self.conf_int = (0.05, 1.95)
            self.alpha = 0.05
            self.n_obs = 100
            self.n_treated = 40
            self.n_control = 60
            self.inference_method = "bootstrap"
            self.survey_metadata = None
            # Presence of a bootstrap distribution triggers the preserve path.
            import numpy as np

            self.bootstrap_distribution = np.random.default_rng(0).normal(1.0, 0.5, 200)

    def test_bootstrap_fit_preserves_fitted_ci_on_alpha_mismatch(self):
        stub = self._BootstrapResultStub()
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        # Native fit was at 95%; requested 90% should NOT be reflected in the label.
        assert h["ci_level"] == 95, (
            "Bootstrap fit must preserve fitted CI level (95) when caller "
            f"requests a different alpha; got {h['ci_level']}"
        )
        # Bounds should match the stored bootstrap interval, not a normal-z
        # recomputation at 90%.
        assert h["ci_lower"] == pytest.approx(0.05)
        assert h["ci_upper"] == pytest.approx(1.95)
        # A caveat records the override.
        caveat_topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in caveat_topics

    class _FiniteDfSurveyStub:
        def __init__(self):
            from types import SimpleNamespace

            self.att = 2.0
            self.se = 0.4
            self.p_value = 0.001
            self.conf_int = (1.22, 2.78)  # 95% via survey t-quantile
            self.alpha = 0.05
            self.n_obs = 120
            self.n_treated = 50
            self.n_control = 70
            self.inference_method = "analytical"
            # Finite survey d.f. triggers the preserve path — normal approx
            # would widen / narrow incorrectly.
            self.survey_metadata = SimpleNamespace(
                weight_type="pweight",
                effective_n=110.0,
                design_effect=1.2,
                sum_weights=120.0,
                n_strata=4,
                n_psu=12,
                df_survey=8,
                replicate_method=None,
            )

    def test_finite_df_fit_preserves_fitted_ci_on_alpha_mismatch(self):
        stub = self._FiniteDfSurveyStub()
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(1.22)
        assert h["ci_upper"] == pytest.approx(2.78)
        caveat_topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in caveat_topics


class TestWildBootstrapAlphaOverride:
    """Regression for the round-4 P0 finding that ``inference='wild_bootstrap'``
    results were falling through to a normal-approximation recomputation."""

    def test_wild_bootstrap_preserves_fitted_ci(self):
        class _WildBootstrapStub:
            def __init__(self):
                self.att = 1.0
                self.se = 0.5
                self.p_value = 0.04
                # 95% CI produced by the wild cluster bootstrap surface.
                self.conf_int = (0.10, 1.90)
                self.alpha = 0.05
                self.n_obs = 100
                self.n_treated = 40
                self.n_control = 60
                self.inference_method = "wild_bootstrap"
                self.survey_metadata = None
                # Wild-boot fits don't necessarily carry a raw distribution;
                # the inference_method string alone must be enough.
                self.bootstrap_distribution = None

        stub = _WildBootstrapStub()
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95, (
            "Wild cluster bootstrap must preserve fitted CI level on alpha "
            f"mismatch; got {h['ci_level']}"
        )
        assert h["ci_lower"] == pytest.approx(0.10)
        assert h["ci_upper"] == pytest.approx(1.90)
        caveats = br.caveats()
        assert any(c.get("topic") == "alpha_override_preserved" for c in caveats)
        # Caveat message should call out wild cluster bootstrap specifically.
        preserved_msg = next(
            c["message"] for c in caveats if c.get("topic") == "alpha_override_preserved"
        )
        assert "wild cluster bootstrap" in preserved_msg


class TestAssumptionBlockSourceFaithful:
    """Regression for the round-4 P1 finding that ``_describe_assumption``
    was producing generic DiD PT text for ContinuousDiD, TripleDifference,
    and StaggeredTripleDifference — all of which have different identifying
    logic per the Methodology Registry."""

    def _stub(self, class_name):
        cls = type(class_name, (), {})
        obj = cls()
        obj.att = 1.0
        obj.se = 0.1
        obj.p_value = 0.001
        obj.conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 40
        obj.n_control = 60
        obj.survey_metadata = None
        obj.event_study_effects = None
        obj.inference_method = "analytical"
        return obj

    def test_continuous_did_assumption_uses_two_level_pt(self):
        br = BusinessReport(self._stub("ContinuousDiDResults"), auto_diagnostics=False)
        assumption = br.to_dict()["assumption"]
        assert assumption["parallel_trends_variant"] == "dose_pt_or_strong_pt"
        desc = assumption["description"]
        # Registry-backed language: PT vs Strong PT + ACRT mention.
        assert "Strong Parallel Trends" in desc or "SPT" in desc
        assert "ATT(d" in desc or "ACRT" in desc
        assert "Callaway" in desc  # attribution to CGBS 2024

    def test_triple_difference_assumption_uses_ddd_decomposition(self):
        class TripleDifferenceResults:
            pass

        obj = TripleDifferenceResults()
        obj.att = 1.0
        obj.se = 0.1
        obj.p_value = 0.001
        obj.conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 40
        obj.n_control = 60
        obj.survey_metadata = None
        obj.inference_method = "analytical"

        br = BusinessReport(obj, auto_diagnostics=False)
        assumption = br.to_dict()["assumption"]
        assert assumption["parallel_trends_variant"] == "triple_difference_cancellation"
        desc = assumption["description"]
        assert "DDD" in desc
        assert "Ortiz-Villavicencio" in desc or "2025" in desc


class TestSCMBusinessReport:
    """Classic SCM routes through BusinessReport with the fit-based assumption
    block, native-diagnostics robustness, and an ADH 2010 attribution."""

    def test_scm_assumption_block_is_fit_based(self, scm_fit):
        res, _ = scm_fit
        assumption = BusinessReport(res, auto_diagnostics=False).to_dict()["assumption"]
        # Distinct from SDiD's "synthetic_fit" — classic SCM is a donor-weighted match.
        assert assumption["parallel_trends_variant"] == "scm_fit"
        desc = assumption["description"].lower()
        assert "placebo" in desc  # significance is placebo-based, not analytical SE
        assert "donor-weighted" in desc or "donor" in desc

    def test_scm_report_is_json_serializable(self, scm_fit):
        # SCM's analytical p_value is NaN; the headline is_significant must be a
        # plain bool (not numpy bool_) so the AI-legible schema serializes.
        res, _ = scm_fit
        schema = BusinessReport(res, auto_diagnostics=True).to_dict()
        json.dumps(schema)  # must not raise
        assert isinstance(schema["headline"]["is_significant"], bool)

    def test_scm_robustness_uses_native_diagnostics(self, scm_fit):
        # SCM omits HonestDiD "sensitivity", so robustness must come from the
        # estimator-native diagnostics block, which must be populated and ran.
        res, _ = scm_fit
        diag = BusinessReport(res, auto_diagnostics=True).to_dict()["diagnostics"]
        assert diag["status"] == "ran"
        native = diag["schema"]["estimator_native_diagnostics"]
        assert native["estimator"] == "SyntheticControl"
        assert isinstance(native["pre_rmspe"], float)

    def test_scm_appendix_attributes_adh_2010(self, scm_fit):
        res, _ = scm_fit
        blob = json.dumps(BusinessReport(res, auto_diagnostics=False).to_dict()).lower()
        assert "abadie" in blob and "2010" in blob

    def test_scm_rejects_honest_did_passthrough(self, scm_fit):
        res, _ = scm_fit
        with pytest.raises(ValueError, match="estimator_native_diagnostics"):
            BusinessReport(res, honest_did_results=object(), auto_diagnostics=False)

    def test_scm_target_parameter_is_named(self, scm_fit):
        # BR must name SCM's headline estimand (single-unit mean post-treatment gap),
        # not fall back to "ATT (unrecognized result class)".
        res, _ = scm_fit
        tp = BusinessReport(res, auto_diagnostics=False).to_dict()["target_parameter"]
        assert "unrecognized" not in tp["name"].lower()
        assert "scm" in tp["name"].lower() or "synthetic control" in tp["name"].lower()

    def test_scm_robustness_block_surfaces_native_fields(self, scm_fit):
        # The top-level robustness block must surface SCM-native content (pre_rmspe,
        # weight concentration, the placebo result) rather than the SDiD-shaped
        # pre_treatment_fit=None left over from the generic lift.
        res, _ = scm_fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.in_space_placebo()
            rob = BusinessReport(res, auto_diagnostics=True).to_dict()["robustness"]
        native = rob["estimator_native"]
        assert native["estimator"] == "SyntheticControl"
        assert isinstance(native["pre_rmspe"], float)
        assert "weight_concentration" in native
        assert native["in_space_placebo"]["n_placebos"] == res.n_placebos

    def test_scm_robustness_block_surfaces_adh2015_diagnostics(self, scm_fit):
        # After running the opt-in ADH-2015 diagnostics, the robustness block must
        # carry their native sub-blocks (lifted from estimator_native_diagnostics).
        res, _ = scm_fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.leave_one_out()
            res.in_time_placebo()
            rob = BusinessReport(res, auto_diagnostics=True).to_dict()["robustness"]
        native = rob["estimator_native"]
        assert native["leave_one_out"]["status"] == "ran"
        assert native["in_time_placebo"]["status"] == "ran"

    def test_staggered_triple_diff_assumption_uses_ddd_not_generic_pt(self):
        class StaggeredTripleDiffResults:
            pass

        obj = StaggeredTripleDiffResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 40
        obj.n_control = 60
        obj.survey_metadata = None
        obj.event_study_effects = None
        obj.inference_method = "analytical"

        br = BusinessReport(obj, auto_diagnostics=False)
        assumption = br.to_dict()["assumption"]
        assert assumption["parallel_trends_variant"] == "triple_difference_cancellation"
        desc = assumption["description"]
        assert "triple-difference" in desc.lower() or "DDD" in desc
        # Must NOT be the generic group-time PT text.
        assert "group-time ATT" not in desc

    def test_imputation_did_assumption_uses_untreated_fe_model(self):
        """Round-42 P1 regression: BJS (2024) identifies through the
        untreated-outcome FE model (Step 1 estimates FE on ``Omega_0``
        = never-treated + not-yet-treated observations, Assumption 1
        parallel trends applies to ``E[Y_it(0)]``). The old generic
        "group-time ATT" wording misstated this: the identifying
        restriction is on the UNTREATED outcome's additive FE
        structure, not on cohort-time ATT equality. REGISTRY.md
        §ImputationDiD lines 1000-1013 and Assumption 1/2.
        """

        class ImputationDiDResults:
            pass

        obj = ImputationDiDResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 40
        obj.n_control = 60
        obj.survey_metadata = None
        obj.event_study_effects = None
        obj.inference_method = "analytical"
        obj.anticipation = 0

        br = BusinessReport(obj, auto_diagnostics=False)
        assumption = br.to_dict()["assumption"]
        assert assumption["parallel_trends_variant"] == "untreated_outcome_fe_model"
        desc = assumption["description"]
        # Registry-backed: Borusyak-Jaravel-Spiess attribution.
        assert "Borusyak" in desc or "BJS" in desc or "2024" in desc
        # Load-bearing source detail: untreated-observation FE model.
        assert "untreated" in desc.lower()
        assert "Omega_0" in desc or "fixed effect" in desc.lower()
        # Must NOT render the pre-R42 generic group-time-ATT template
        # that grouped BJS in with CS / SA.
        assert (
            "parallel trends across treatment cohorts and time periods (group-time ATT)" not in desc
        ), (
            "ImputationDiD identifies via untreated-outcome FE modelling "
            "(BJS 2024 Assumption 1), not generic group-time ATT PT. The "
            f"assumption description must not use the pre-R42 template. Got: {desc!r}"
        )

    def test_two_stage_did_assumption_uses_untreated_fe_model(self):
        """Round-42 P1 regression: Gardner (2022) two-stage DiD shares
        BJS's untreated-outcome FE identification (REGISTRY.md explicitly
        states "Parallel trends (same as ImputationDiD)" and the point
        estimates are algebraically equivalent). Stage 1 fits FE on
        untreated observations, Stage 2 residualizes treated observations.
        The old generic "group-time ATT" wording dropped the untreated-
        subset detail. REGISTRY.md §TwoStageDiD lines 1113-1128.
        """

        class TwoStageDiDResults:
            pass

        obj = TwoStageDiDResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 40
        obj.n_control = 60
        obj.survey_metadata = None
        obj.event_study_effects = None
        obj.inference_method = "analytical"
        obj.anticipation = 0

        br = BusinessReport(obj, auto_diagnostics=False)
        assumption = br.to_dict()["assumption"]
        assert assumption["parallel_trends_variant"] == "untreated_outcome_fe_model"
        desc = assumption["description"]
        # Registry-backed: Gardner 2022 attribution.
        assert "Gardner" in desc or "2022" in desc
        # Load-bearing: Stage 1 operates on untreated observations.
        assert "untreated" in desc.lower()
        assert "Stage 1" in desc or "stage 1" in desc.lower()
        # Must mention the two-stage procedure.
        assert "two-stage" in desc.lower() or "Two-Stage" in desc
        # Must NOT render the pre-R42 generic group-time-ATT template
        # that grouped Gardner in with CS / SA.
        assert (
            "parallel trends across treatment cohorts and time periods (group-time ATT)" not in desc
        ), (
            "TwoStageDiD identifies via the same untreated-outcome FE "
            "model as ImputationDiD (Gardner 2022); the assumption "
            f"description must not use the pre-R42 template. Got: {desc!r}"
        )


class TestEfficientDiDAssumptionPtAllPtPost:
    """Round-8 regression: EfficientDiD has two distinct PT regimes
    (PT-All and PT-Post, per Chen-Sant'Anna-Xie 2025 Corollary 3.2 and
    Lemma 2.1). The old generic group-time PT text was source-unfaithful;
    the assumption block must now read ``results.pt_assumption`` and
    branch on it.
    """

    def _stub(self, pt_assumption: str, control_group: str = "never_treated"):
        """Build an EfficientDiD-shaped stub. ``control_group`` defaults to
        ``"never_treated"`` (the estimator's actual default); the only other
        accepted value is ``"last_cohort"`` (pseudo-never-treated). The
        earlier ``"not_yet_treated"`` default was invalid for this estimator
        and was flagged in round-10 CI review."""

        class EfficientDiDResults:
            pass

        stub = EfficientDiDResults()
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
        stub.inference_method = "analytical"
        stub.pt_assumption = pt_assumption
        stub.control_group = control_group
        return stub

    def test_pt_all_uses_pt_all_language(self):
        br = BusinessReport(self._stub("all"), auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a["parallel_trends_variant"] == "pt_all"
        assert "PT-All" in a["description"]
        assert "Hausman" in a["description"]
        # Must NOT be the old generic group-time PT text.
        assert "group-time ATT" not in a["description"]

    def test_pt_post_uses_pt_post_language(self):
        br = BusinessReport(self._stub("post"), auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a["parallel_trends_variant"] == "pt_post"
        assert "PT-Post" in a["description"]
        assert "Corollary 3.2" in a["description"] or "single-baseline" in a["description"]

    def test_pt_post_never_treated_names_never_treated(self):
        """Default control_group: description must say never-treated."""
        br = BusinessReport(self._stub("post", "never_treated"), auto_diagnostics=False)
        desc = br.to_dict()["assumption"]["description"]
        assert "never-treated" in desc
        assert "latest treated cohort" not in desc

    def test_pt_post_last_cohort_branch_describes_pseudo_control(self):
        """Round-10 regression: ``control_group='last_cohort'`` must not be
        narrated with generic never-treated language. The description must
        describe the pseudo-never-treated latest-cohort design (REGISTRY.md
        §EfficientDiD line 908)."""
        br = BusinessReport(self._stub("post", "last_cohort"), auto_diagnostics=False)
        desc = br.to_dict()["assumption"]["description"]
        assert "latest treated cohort" in desc
        assert "pseudo-never-treated" in desc
        assert "dropped" in desc

    def test_pt_all_last_cohort_branch_describes_pseudo_control(self):
        br = BusinessReport(self._stub("all", "last_cohort"), auto_diagnostics=False)
        desc = br.to_dict()["assumption"]["description"]
        assert "latest treated cohort" in desc
        assert "pseudo-never-treated" in desc

    def test_control_group_is_reflected_in_block(self):
        br = BusinessReport(self._stub("all", "last_cohort"), auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a.get("control_group") == "last_cohort"


class TestMethodAwarePTProse:
    """Round-8 regression: BR and DR summary prose must branch on the
    ``parallel_trends.method`` field. Generic "pre-treatment event-study
    coefficients" wording is wrong for the 2x2 ``slope_difference`` path
    and for EfficientDiD's ``hausman`` PT-All vs PT-Post pretest.
    """

    def test_br_summary_uses_slope_difference_wording_for_simple_did(self):
        """Use a stub DR schema with a known slope_difference verdict so
        the test is deterministic across pre-period counts. The real
        2x2 fit can produce NaN verdicts when there is only one
        pre-period, so we don't rely on a real DR here."""

        class DiDResults:
            pass

        stub = DiDResults()
        stub.att = 1.0
        stub.se = 0.2
        stub.p_value = 0.001
        stub.conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.inference_method = "analytical"

        # Hand-crafted DR schema with ``method = "slope_difference"``.
        from diff_diff.diagnostic_report import DiagnosticReportResults

        fake_schema = {
            "schema_version": "2.0",
            "estimator": "DiDResults",
            "headline_metric": {"name": "att", "value": 1.0},
            "parallel_trends": {
                "status": "ran",
                "method": "slope_difference",
                "joint_p_value": 0.40,
                "verdict": "no_detected_violation",
            },
            "pretrends_power": {"status": "not_applicable"},
            "sensitivity": {"status": "not_applicable"},
            "placebo": {"status": "skipped", "reason": "opt-in"},
            "bacon": {"status": "not_applicable"},
            "design_effect": {"status": "not_applicable"},
            "heterogeneity": {"status": "not_applicable"},
            "epv": {"status": "not_applicable"},
            "estimator_native_diagnostics": {"status": "not_applicable"},
            "skipped": {},
            "warnings": [],
            "overall_interpretation": "",
            "next_steps": [],
        }
        fake_dr_results = DiagnosticReportResults(
            schema=fake_schema,
            interpretation="",
            applicable_checks=("parallel_trends",),
            skipped_checks={},
            warnings=(),
        )
        br = BusinessReport(stub, diagnostics=fake_dr_results)
        summary = br.summary()
        pt_method = br.to_dict()["pre_trends"].get("method")
        assert pt_method == "slope_difference"
        # Must NOT use the generic event-study wording.
        assert "event-study coefficients" not in summary
        # Must use the slope-difference subject phrase.
        assert "slope-difference" in summary

    def test_dr_summary_uses_hausman_wording_for_efficient_did(self, edid_fit):
        from diff_diff import DiagnosticReport

        fit, sdf = edid_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        summary = dr.summary()
        pt = dr.to_dict()["parallel_trends"]
        # EfficientDiD's PT check routes through hausman_pretest.
        assert pt.get("method") == "hausman"
        # The generic event-study wording must not appear for this path.
        assert "event-study coefficients" not in summary


class TestFullReportMethodAwarePTLabel:
    """Round-25 P2 CI review on PR #318: ``BusinessReport.full_report()``
    previously hard-coded ``joint p = ...`` in the Pre-Trends section,
    which mislabels the 2x2 ``slope_difference`` and EfficientDiD
    ``hausman`` single-statistic tests and invents a nonexistent
    ``joint p`` label for design-enforced SDiD / TROP paths that have
    no p-value at all. The markdown path must use the same
    method-aware label helper the summary path already uses
    (``_pt_method_stat_label``).
    """

    @staticmethod
    def _stub_result_with_method(method: str):
        from diff_diff.diagnostic_report import DiagnosticReportResults

        class DiDResults:
            pass

        stub = DiDResults()
        stub.att = 1.0
        stub.se = 0.2
        stub.p_value = 0.001
        stub.conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.inference_method = "analytical"

        pt_block: dict = {
            "status": "ran",
            "method": method,
            "verdict": "no_detected_violation",
        }
        # SDiD's synthetic_fit path has no p-value by design; the other
        # methods do.
        if method != "synthetic_fit":
            pt_block["joint_p_value"] = 0.40

        fake_schema = {
            "schema_version": "2.0",
            "estimator": "DiDResults",
            "headline_metric": {"name": "att", "value": 1.0},
            "parallel_trends": pt_block,
            "pretrends_power": {"status": "not_applicable"},
            "sensitivity": {"status": "not_applicable"},
            "placebo": {"status": "skipped", "reason": "opt-in"},
            "bacon": {"status": "not_applicable"},
            "design_effect": {"status": "not_applicable"},
            "heterogeneity": {"status": "not_applicable"},
            "epv": {"status": "not_applicable"},
            "estimator_native_diagnostics": {"status": "not_applicable"},
            "skipped": {},
            "warnings": [],
            "overall_interpretation": "",
            "next_steps": [],
        }
        fake_dr = DiagnosticReportResults(
            schema=fake_schema,
            interpretation="",
            applicable_checks=("parallel_trends",),
            skipped_checks={},
            warnings=(),
        )
        return stub, fake_dr

    def _pt_section(self, md: str) -> str:
        # The Pre-Trends section is delimited by the next ``##`` heading.
        after = md.split("## Pre-Trends", 1)[1]
        return after.split("\n## ", 1)[0]

    def test_full_report_slope_difference_uses_single_p_label(self):
        stub, fake_dr = self._stub_result_with_method("slope_difference")
        md = BusinessReport(stub, diagnostics=fake_dr).full_report()
        section = self._pt_section(md)
        assert "joint p" not in section, (
            f"2x2 slope_difference is a single-statistic test and must "
            f"not be labeled ``joint p`` in the markdown. Got: {section!r}"
        )
        # The single-statistic label ``p = ...`` must be present.
        assert "p = 0.4" in section

    def test_full_report_hausman_uses_single_p_label(self):
        stub, fake_dr = self._stub_result_with_method("hausman")
        section = self._pt_section(BusinessReport(stub, diagnostics=fake_dr).full_report())
        assert "joint p" not in section, (
            f"EfficientDiD Hausman is a single-statistic test and must "
            f"not be labeled ``joint p`` in the markdown. Got: {section!r}"
        )
        assert "p = 0.4" in section

    def test_full_report_synthetic_fit_omits_p_label(self):
        stub, fake_dr = self._stub_result_with_method("synthetic_fit")
        section = self._pt_section(BusinessReport(stub, diagnostics=fake_dr).full_report())
        # No p-value of any kind for design-enforced SDiD PT analogue.
        assert "joint p" not in section
        assert "p = " not in section
        # Verdict must still render.
        assert "Verdict:" in section


class TestHausmanPretestPropagatesFitDesign:
    """Round-9 regression: ``_pt_hausman`` must propagate the fitted
    result's ``control_group`` and ``anticipation`` into
    ``EfficientDiD.hausman_pretest`` so the pretest diagnoses the same
    design as the estimate being summarized. Rerunning with defaults
    would silently change the identification regime.
    """

    def _real_edid_fit(self):
        from diff_diff import EfficientDiD

        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        edid = EfficientDiD().fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        # Force non-default design knobs on the result so the regression
        # exercises propagation even when the constructor used defaults.
        edid.control_group = "last_cohort"
        edid.anticipation = 1
        return edid, sdf

    def test_hausman_pretest_receives_control_group_and_anticipation(self):
        from diff_diff import DiagnosticReport

        fit, sdf = self._real_edid_fit()
        captured: dict = {}

        def _fake_hausman(*args, **kwargs):
            captured.update(kwargs)

            class _Result:
                statistic = 0.0
                p_value = 0.5
                df = 1

            return _Result()

        with patch(
            "diff_diff.efficient_did.EfficientDiD.hausman_pretest",
            side_effect=_fake_hausman,
        ):
            DiagnosticReport(
                fit,
                data=sdf,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            ).run_all()

        assert (
            captured.get("control_group") == "last_cohort"
        ), f"control_group must propagate from the fit; got {captured}"
        assert (
            captured.get("anticipation") == 1
        ), f"anticipation must propagate from the fit; got {captured}"


class TestHausmanFitFaithfulSkip:
    """Round-10 regression: DR / survey-weighted EfficientDiD fits cannot
    replay the Hausman pretest from ``(data, outcome, unit, time,
    first_treat)`` alone because the result does not expose ``covariates``,
    ``cluster``, nuisance kwargs, or the full survey design. DR must skip
    with an explicit reason rather than rerunning defaults.
    """

    def _make_fit(self, *, estimation_path="nocov", survey_metadata=None):
        from diff_diff import EfficientDiD

        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        edid = EfficientDiD().fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        edid.estimation_path = estimation_path
        edid.survey_metadata = survey_metadata
        return edid, sdf

    def test_dr_covariate_path_skipped_with_reason(self):
        from diff_diff import DiagnosticReport

        fit, sdf = self._make_fit(estimation_path="dr")
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert "parallel_trends" not in dr.applicable_checks
        reason = dr.skipped_checks.get("parallel_trends", "")
        assert "doubly-robust" in reason

    def test_survey_weighted_fit_skipped_with_reason(self):
        from types import SimpleNamespace

        from diff_diff import DiagnosticReport

        fake_survey = SimpleNamespace(
            weight_type="pweight",
            effective_n=80.0,
            design_effect=1.25,
            sum_weights=100.0,
            n_strata=None,
            n_psu=None,
            df_survey=40,
        )
        fit, sdf = self._make_fit(survey_metadata=fake_survey)
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert "parallel_trends" not in dr.applicable_checks
        reason = dr.skipped_checks.get("parallel_trends", "")
        assert "survey design" in reason


class TestHausmanPretestPropagatesCluster:
    """Round-11 regression: ``EfficientDiDResults`` now persists the
    ``cluster_name`` column used at fit time, and ``_pt_hausman`` forwards
    it to ``EfficientDiD.hausman_pretest``. Without this, clustered
    fits would be replayed under unclustered inference, silently
    publishing an H statistic / p-value for the wrong design.
    """

    def test_hausman_pretest_receives_cluster_kwarg(self):
        import pandas as pd

        from diff_diff import DiagnosticReport, EfficientDiD

        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        # Add a cluster column (e.g., region) to the panel.
        sdf = pd.DataFrame(sdf).copy()
        sdf["cluster_col"] = sdf["unit"] % 10

        edid = EfficientDiD(cluster="cluster_col").fit(
            sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        # Confirm persistence landed.
        assert getattr(edid, "cluster_name", None) == "cluster_col"

        captured: dict = {}

        def _fake_hausman(*args, **kwargs):
            captured.update(kwargs)

            class _Result:
                statistic = 0.0
                p_value = 0.5
                df = 1

            return _Result()

        with patch(
            "diff_diff.efficient_did.EfficientDiD.hausman_pretest",
            side_effect=_fake_hausman,
        ):
            DiagnosticReport(
                edid,
                data=sdf,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            ).run_all()

        assert (
            captured.get("cluster") == "cluster_col"
        ), f"cluster column must propagate from fit to Hausman pretest; got {captured}"


class TestAnticipationPersistsOnRealResults:
    """Round-19 P1 regression: ``CallawaySantAnnaResults``,
    ``SunAbrahamResults``, and ``StaggeredTripleDiffResults`` must
    persist the ``anticipation`` field so the anticipation-aware
    reporting code (round-15/17) actually fires on real fits. Stub-
    only regressions had hidden that the result constructors were
    dropping the value.
    """

    def test_cs_fit_persists_anticipation(self):
        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        cs = CallawaySantAnna(base_period="universal", anticipation=1).fit(
            sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert getattr(cs, "anticipation", None) == 1
        br = BusinessReport(cs, auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        # Round-17 assumption-aware block now fires on a real fit.
        assert a["no_anticipation"] is False
        assert a["anticipation_periods"] == 1
        assert "not strict no-anticipation" in a["description"]

    def test_sun_abraham_fit_persists_anticipation(self):
        from diff_diff import SunAbraham

        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        sa = SunAbraham(anticipation=1).fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        assert getattr(sa, "anticipation", None) == 1
        br = BusinessReport(sa, auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a["no_anticipation"] is False
        assert a["anticipation_periods"] == 1

    def test_imputation_fit_persists_anticipation(self):
        from diff_diff import ImputationDiD

        sdf = generate_staggered_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
        im = ImputationDiD(anticipation=1).fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        assert getattr(im, "anticipation", None) == 1
        br = BusinessReport(im, auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a["no_anticipation"] is False
        assert a["anticipation_periods"] == 1
        assert "not strict no-anticipation" in a["description"]

    def test_two_stage_fit_persists_anticipation(self):
        from diff_diff import TwoStageDiD

        sdf = generate_staggered_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
        ts = TwoStageDiD(anticipation=2).fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        assert getattr(ts, "anticipation", None) == 2
        br = BusinessReport(ts, auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a["no_anticipation"] is False
        assert a["anticipation_periods"] == 2
        assert "2 periods" in a["description"]

    def test_stacked_fit_persists_anticipation(self):
        from diff_diff import StackedDiD

        sdf = generate_staggered_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
        st = StackedDiD(anticipation=1).fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        assert getattr(st, "anticipation", None) == 1
        br = BusinessReport(st, auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a["no_anticipation"] is False
        assert a["anticipation_periods"] == 1


class TestInconclusivePTProvenancePreservedOnBRSchema:
    """Round-39 P3 CI review on PR #318: DR's ``_pt_event_study`` emits
    ``n_dropped_undefined`` and a detailed ``reason`` on the
    inconclusive PT block (undefined pre-period inference — NaN
    per-period p-value or zero / negative SE). BR's ``_lift_pre_trends``
    was dropping both fields at the lift boundary, so the BR schema
    and BR's summary renderer lost the provenance DR had already
    computed. Preserve both so BR consumers see the exact count of
    undefined rows and the same reason without re-consulting the DR
    schema.
    """

    def test_n_dropped_undefined_and_reason_land_on_br_pre_trends(self):
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

        br = BusinessReport(obj)
        pt = br.to_dict()["pre_trends"]
        # Status and verdict reflect the inconclusive outcome.
        assert pt["verdict"] == "inconclusive"
        # The provenance fields are present on the BR schema.
        assert pt["n_dropped_undefined"] == 1
        assert isinstance(pt.get("reason"), str) and pt["reason"]
        # And the summary renderer quotes the count (the existing
        # inconclusive branch in ``_render_summary`` reads
        # ``pt.get("n_dropped_undefined")``; before this fix that lookup
        # returned ``None`` because the lift had dropped it).
        summary = br.summary()
        assert "1 pre-period row had undefined inference" in summary


class TestStaggeredTripleDiffNeverTreatedFixedComparison:
    """Round-37 P1 CI review on PR #318: ``StaggeredTripleDiffResults``
    stores ``n_control_units`` as a composite total that also includes
    the eligibility-denied cohorts. The valid fixed comparison under
    ``control_group="never_treated"`` is the never-enabled cohort
    (``staggered_triple_diff.py:384``, REGISTRY.md §StaggeredTripleDifference
    line 1730). BR was previously narrating the composite total as
    "control" on the ``nevertreated`` mode; the fix surfaces
    ``n_never_enabled`` as the fixed comparison count on that path
    too (the dynamic ``notyettreated`` path was already correct).
    """

    @staticmethod
    def _stub(control_group: str):
        class StaggeredTripleDiffResults:
            pass

        stub = StaggeredTripleDiffResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 800
        stub.n_treated = 100
        stub.n_control_units = 500  # composite total
        stub.n_never_enabled = 300  # fixed never-enabled subset
        stub.event_study_effects = None
        stub.survey_metadata = None
        stub.control_group = control_group
        return stub

    def test_never_treated_mode_surfaces_never_enabled_not_composite_total(self):
        sample = BusinessReport(self._stub("never_treated"), auto_diagnostics=False).to_dict()[
            "sample"
        ]
        # Composite total must not be surfaced as the fixed control
        # count on the ``nevertreated`` path.
        assert sample["n_control"] is None, (
            f"n_control must not carry the composite n_control_units "
            f"total on StaggeredTripleDiff(control_group='never_treated'); "
            f"got sample={sample!r}"
        )
        assert sample["n_never_enabled"] == 300

    def test_never_treated_mode_summary_renders_never_enabled_count(self):
        """Round-38 P3 strengthened regression: the summary must
        POSITIVELY surface the valid fixed comparison cohort
        (``300 never-enabled``), not merely avoid the wrong
        ``500 control`` phrasing.
        """
        import re

        summary = BusinessReport(self._stub("never_treated"), auto_diagnostics=False).summary()
        # Old wrong phrasing absent.
        assert not re.search(r"\b500\s+control", summary), summary
        # New fixed cohort present.
        assert "300 never-enabled" in summary, (
            f"BR summary must render the valid fixed never-enabled "
            f"comparison cohort on StaggeredTripleDiff(control_group="
            f"'never_treated'); got: {summary!r}"
        )
        # And the generic no-comparison fallback must not fire.
        assert "Sample: 800 observations." not in summary

    def test_never_treated_mode_full_report_renders_never_enabled_count(self):
        md = BusinessReport(self._stub("never_treated"), auto_diagnostics=False).full_report()
        sample_section = md.split("## Sample", 1)[1].split("\n## ", 1)[0]
        assert "never-enabled" in sample_section.lower()
        assert "300" in sample_section
        # No bare "- Control: 500" line (composite total) should appear
        # on this path.
        assert "- Control: 500" not in sample_section


class TestBRHeadlineOmitsBrokenCIOnUndefinedInference:
    """Round-37 P1 CI review on PR #318: ``_extract_headline`` preserves
    the fit's native CI even when it is undefined (e.g., survey-df
    collapse produces finite ATT but NaN CI endpoints). The renderer
    previously gated on ``isinstance(lo, (int, float))``, which accepts
    ``NaN`` (a float) and rendered ``95% CI: undefined to undefined``.
    Gate on ``np.isfinite`` instead, and emit an explicit
    "inference unavailable" trailer when at least one bound is
    non-finite. DR's own headline renderer already handled this
    correctly (round-36 fix).
    """

    @staticmethod
    def _stub_nan_ci():
        class DiDResults:
            pass

        stub = DiDResults()
        stub.att = 1.0
        stub.se = float("nan")
        stub.t_stat = float("nan")
        stub.p_value = float("nan")
        stub.conf_int = (float("nan"), float("nan"))
        stub.alpha = 0.05
        stub.n_obs = 200
        stub.n_treated = 100
        stub.n_control = 100
        stub.survey_metadata = None
        return stub

    def test_summary_does_not_render_undefined_ci_interval(self):
        summary = BusinessReport(self._stub_nan_ci(), auto_diagnostics=False).summary()
        lower = summary.lower()
        # Must not render the broken CI interval fragment.
        assert "undefined to undefined" not in lower, summary
        assert "95% ci: nan" not in lower
        # Must explicitly flag that inference is unavailable.
        assert "inference unavailable" in lower

    def test_full_report_does_not_render_undefined_ci_interval(self):
        md = BusinessReport(self._stub_nan_ci(), auto_diagnostics=False).full_report()
        lower = md.lower()
        assert "undefined to undefined" not in lower
        assert "95% ci: nan" not in lower
        assert "inference unavailable" in lower


class TestStackedCleanControlSurfacesInSampleBlock:
    """Pre-emptive audit regression: ``StackedDiD`` exposes its control-
    group choice as ``clean_control`` (the public Wing-Freedman-
    Hollingsworth-2024 kwarg name), not ``control_group``. The BR sample
    block must normalize the key so downstream agents see a consistent
    ``control_group`` field across estimators.

    ``n_control_units`` on ``StackedDiDResults`` is documented as
    "distinct control units across the trimmed set" (stacked_did_results
    L59-62). Under ``clean_control="not_yet_treated"`` the trimmed set
    admits future-treated controls by construction, so the count is
    NOT a never-treated tally and must not be relabeled as
    ``n_never_treated`` — round-21 P1 CI review on PR #318 flagged the
    prior relabeling as a semantic-contract violation because it can
    fabricate never-treated support that does not exist (e.g., in an
    all-eventually-treated panel).
    """

    def test_stacked_not_yet_treated_surfaces_as_dynamic_without_never_treated_relabel(self):
        """``clean_control='not_yet_treated'`` is a dynamic, sub-
        experiment-specific comparison set (``A_s > a + kappa_post``);
        ``n_control`` is cleared (not a fixed tally), ``n_never_treated``
        is NOT relabeled, and the distinct-controls tally is surfaced
        under the dedicated ``n_distinct_controls_trimmed`` key.
        """
        from diff_diff import StackedDiD

        sdf = generate_staggered_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
        st = StackedDiD(control_group="not_yet_treated").fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        assert getattr(st, "clean_control", None) == "not_yet_treated"
        sample = BusinessReport(st, auto_diagnostics=False).to_dict()["sample"]
        assert sample["control_group"] == "not_yet_treated"
        assert sample["dynamic_control"] is True
        assert sample["n_never_treated"] is None, (
            "StackedDiDResults.n_control_units is the distinct-control-"
            "units tally of the trimmed set (includes future-treated "
            "controls); it must not be surfaced as n_never_treated."
        )
        # Round-22 correction: ``n_control`` must be cleared under
        # dynamic modes so the report does not narrate a fixed control
        # tally. The underlying count is surfaced under the dedicated
        # Stacked key.
        assert sample["n_control"] is None
        assert sample["n_distinct_controls_trimmed"] == int(st.n_control_units)

    def test_stacked_strict_clean_control_surfaces_as_dynamic(self):
        """``clean_control='strict'`` (``A_s > a + kappa_post + kappa_pre``)
        is also a sub-experiment-specific rule — stricter than
        ``not_yet_treated`` but still NOT a fixed never-treated pool
        (round-22 P1 CI review on PR #318).
        """
        from diff_diff import StackedDiD

        sdf = generate_staggered_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
        st = StackedDiD(control_group="strict").fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        sample = BusinessReport(st, auto_diagnostics=False).to_dict()["sample"]
        assert sample["control_group"] == "strict"
        assert sample["dynamic_control"] is True, (
            "clean_control='strict' is sub-experiment-specific (rule "
            "A_s > a + kappa_post + kappa_pre) and must be marked dynamic "
            "so the report does not claim a fixed never-treated control "
            "pool."
        )
        assert sample["n_control"] is None
        assert sample["n_never_treated"] is None

    def test_stacked_never_treated_surfaces_as_fixed_control(self):
        from diff_diff import StackedDiD

        sdf = generate_staggered_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
        st = StackedDiD(control_group="never_treated").fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        sample = BusinessReport(st, auto_diagnostics=False).to_dict()["sample"]
        assert sample["control_group"] == "never_treated"
        assert sample["dynamic_control"] is False

    def test_stacked_all_eventually_treated_panel_does_not_fabricate_never_treated(self):
        """All-eventually-treated stacked panel with
        ``clean_control="not_yet_treated"`` must not claim any
        never-treated units, because every unit is eventually treated
        (the round-21 reviewer example).
        """

        from diff_diff import StackedDiD

        # Every unit is eventually treated (no never-treated).
        # Multiple cohorts so Stacked has something to stack against.
        sdf = generate_staggered_data(
            n_units=80,
            n_periods=10,
            never_treated_frac=0.0,
            treatment_effect=1.5,
            seed=7,
        )
        # Sanity: the fixture has no never-treated units.
        assert sdf[sdf["first_treat"] == 0].empty

        st = StackedDiD(control_group="not_yet_treated", kappa_pre=1, kappa_post=1).fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        sample = BusinessReport(st, auto_diagnostics=False).to_dict()["sample"]
        assert sample["n_never_treated"] is None, (
            "All-eventually-treated panel under clean_control='not_yet_treated' "
            "must not surface any never-treated count; the trimmed stack "
            "contains only future-treated controls."
        )


class TestStackedDiDAssumptionBlock:
    """Round-22 P1 regression: ``StackedDiDResults`` must get a
    dedicated assumption description reflecting Wing-Freedman-
    Hollingsworth (2024) identification — sub-experiment common trends
    plus IC1 (event window fits) and IC2 (clean controls exist) — not
    the generic "group-time ATT" clause used for CS / SA / etc. The
    active ``clean_control`` rule must be named in the description.
    """

    @staticmethod
    def _stub(clean_control: str):
        class StackedDiDResults:
            pass

        stub = StackedDiDResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 400
        stub.n_treated = 50
        stub.n_control_units = 300
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.control_group = clean_control
        return stub

    def test_not_yet_treated_names_subexperiment_contract(self):
        br = BusinessReport(self._stub("not_yet_treated"), auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a["parallel_trends_variant"] == "stacked_sub_experiment"
        desc = a["description"]
        assert "Wing, Freedman & Hollingsworth 2024" in desc
        assert "sub-experiment" in desc
        assert "IC1" in desc and "IC2" in desc
        assert "A_s > a + kappa_post" in desc
        assert "not_yet_treated" not in desc or "``A_s > a + kappa_post``" in desc
        # The active control-group rule is carried on the block explicitly
        # for consumers that want structured access - under BOTH keys
        # through the 3.9 shim window (row M-095).
        assert a["control_group"] == "not_yet_treated"
        assert a["clean_control"] == "not_yet_treated"

    def test_strict_names_strict_rule(self):
        desc = BusinessReport(self._stub("strict"), auto_diagnostics=False).to_dict()["assumption"][
            "description"
        ]
        assert "A_s > a + kappa_post + kappa_pre" in desc

    def test_never_treated_names_fixed_pool(self):
        desc = BusinessReport(self._stub("never_treated"), auto_diagnostics=False).to_dict()[
            "assumption"
        ]["description"]
        assert "never treated" in desc.lower()
        assert "A_s = infinity" in desc


class TestStackedRenderingNarratesDynamicControl:
    """Round-22 P1 regression: BR ``summary()`` / ``full_report()`` must
    narrate Stacked dynamic clean-control designs as sub-experiment-
    specific comparisons, not as fixed "N treated / M control" samples.
    Previously the ``n_control`` branch fired first and misrendered both
    ``clean_control='not_yet_treated'`` and ``'strict'``.
    """

    def test_summary_does_not_narrate_stacked_dynamic_as_fixed_control(self):
        from diff_diff import StackedDiD

        sdf = generate_staggered_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
        st = StackedDiD(control_group="not_yet_treated").fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        summary = BusinessReport(st, auto_diagnostics=False).summary()
        # Must NOT render a "X treated, Y control" clause (that narration
        # implies a fixed comparison pool).
        import re

        assert not re.search(r"\d[\d,]*\s+treated,\s+\d[\d,]*\s+control", summary), (
            f"Stacked with dynamic clean-control must not be narrated "
            f"as fixed treated/control counts. Got: {summary!r}"
        )
        # Must narrate the sub-experiment-specific clean-control contract.
        assert "sub-experiment-specific clean-control" in summary
        assert "control_group='not_yet_treated'" in summary

    def test_full_report_names_sub_experiment_comparison_for_stacked_strict(self):
        from diff_diff import StackedDiD

        sdf = generate_staggered_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
        st = StackedDiD(control_group="strict").fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        md = BusinessReport(st, auto_diagnostics=False).full_report()
        # Must NOT emit a bare "Control: N" line.
        assert (
            "- Control:" not in md or "- Control: " not in md.split("## Sample")[1].split("##")[0]
        ), (
            "Stacked with dynamic clean-control must not render a fixed "
            "'- Control: N' line in the Sample section."
        )
        assert "sub-experiment-specific clean controls" in md
        assert "control_group='strict'" in md


class TestDCDHPhase3AssumptionClause:
    """Pre-emptive audit regression: ``ChaisemartinDHaultfoeuilleResults``
    populates ``covariate_residuals`` when ``controls`` is set in fit,
    ``linear_trends_effects`` when ``trends_linear=True``, and
    ``heterogeneity_effects`` when ``heterogeneity`` is set. Each change
    modifies the identifying contract and the estimand label
    (``DID^X_l`` / ``DID^{fd}_l`` / ``DID^{X,fd}_l``). The BR assumption
    description must surface the active configuration so the prose does
    not misrepresent the identifying assumption on a Phase-3 fit.
    """

    def test_dcdh_base_case_has_no_phase3_clause(self):
        from diff_diff.business_report import _describe_assumption

        class Stub:
            covariate_residuals = None
            linear_trends_effects = None
            heterogeneity_effects = None

        block = _describe_assumption("ChaisemartinDHaultfoeuilleResults", Stub())
        assert "Phase-3 configuration" not in block["description"]

    def test_dcdh_controls_only_surfaces_did_x(self):
        import pandas as pd

        from diff_diff.business_report import _describe_assumption

        class Stub:
            covariate_residuals = pd.DataFrame({"theta_hat": [0.1]})
            linear_trends_effects = None
            heterogeneity_effects = None

        desc = _describe_assumption("ChaisemartinDHaultfoeuilleResults", Stub())["description"]
        assert "Phase-3 configuration" in desc
        assert "DID^X_l" in desc
        assert "first-stage residualization" in desc
        assert "DID^{fd}_l" not in desc

    def test_dcdh_trends_linear_only_surfaces_did_fd(self):
        from diff_diff.business_report import _describe_assumption

        class Stub:
            covariate_residuals = None
            linear_trends_effects = {1: {"effect": 0.1}}
            heterogeneity_effects = None

        desc = _describe_assumption("ChaisemartinDHaultfoeuilleResults", Stub())["description"]
        assert "Phase-3 configuration" in desc
        assert "DID^{fd}_l" in desc
        assert "group-specific linear pre-trends" in desc

    def test_dcdh_controls_and_trends_surfaces_combined_estimand(self):
        import pandas as pd

        from diff_diff.business_report import _describe_assumption

        class Stub:
            covariate_residuals = pd.DataFrame({"theta_hat": [0.1]})
            linear_trends_effects = {1: {"effect": 0.1}}
            heterogeneity_effects = {1: {}}

        desc = _describe_assumption("ChaisemartinDHaultfoeuilleResults", Stub())["description"]
        assert "DID^{X,fd}_l" in desc
        assert "heterogeneity tests" in desc
        assert "beta^{het}_l" in desc


class TestAnticipationStripsStrictNoAnticipationClause:
    """Round-30 P1 CI review on PR #318: ``_apply_anticipation_to_assumption``
    previously only appended an anticipation clause. Several base
    descriptions already say "plus no anticipation" or "Also assumes
    no anticipation", so an anticipation-enabled fit would render
    self-contradictory prose: the strict clause AND the relaxed one in
    the same paragraph. The helper now strips the strict phrasing
    before appending. These regressions cover every anticipation-
    capable estimator base description that previously carried such
    wording.
    """

    _STRICT_PATTERNS = (
        "plus no anticipation",
        "Also assumes no anticipation",
    )

    @staticmethod
    def _stub(class_name: str, **extras):
        stub_cls = type(class_name, (), {})
        stub = stub_cls()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 400
        stub.n_treated = 100
        stub.n_control = 300
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.anticipation = 2
        for k, v in extras.items():
            setattr(stub, k, v)
        return stub

    def _assert_no_strict_contract(self, description: str):
        assert isinstance(description, str) and description
        for pat in self._STRICT_PATTERNS:
            assert pat not in description, (
                f"Anticipation-enabled fit description must not carry "
                f"the strict phrase {pat!r}. Got: {description!r}"
            )
        # Must still say anticipation is allowed (relaxed contract).
        assert "Anticipation is allowed" in description
        assert "not strict no-anticipation" in description

    def test_generic_group_time_strips_strict_clause(self):
        # Generic CS/SA/Imputation/TwoStage/Wooldridge branch.
        stub = self._stub("CallawaySantAnnaResults")
        block = BusinessReport(stub, auto_diagnostics=False).to_dict()["assumption"]
        assert block["no_anticipation"] is False
        assert block["anticipation_periods"] == 2
        self._assert_no_strict_contract(block["description"])

    def test_efficient_did_pt_all_strips_strict_clause(self):
        stub = self._stub("EfficientDiDResults", pt_assumption="all")
        block = BusinessReport(stub, auto_diagnostics=False).to_dict()["assumption"]
        self._assert_no_strict_contract(block["description"])
        # PT-All identifying content should still be present.
        assert "PT-All" in block["description"]

    def test_efficient_did_pt_post_strips_strict_clause(self):
        stub = self._stub("EfficientDiDResults", pt_assumption="post")
        block = BusinessReport(stub, auto_diagnostics=False).to_dict()["assumption"]
        self._assert_no_strict_contract(block["description"])
        assert "PT-Post" in block["description"]

    def test_stacked_did_strips_strict_clause(self):
        stub = self._stub("StackedDiDResults", control_group="not_yet_treated")
        block = BusinessReport(stub, auto_diagnostics=False).to_dict()["assumption"]
        self._assert_no_strict_contract(block["description"])
        # Stacked sub-experiment identifying content preserved.
        assert "IC1" in block["description"] and "IC2" in block["description"]

    def test_rendered_full_report_has_no_strict_contract_for_anticipation(self):
        """Integration: the rendered markdown's Identifying Assumption
        section must also be free of the strict phrase on an
        anticipation-enabled fit.
        """
        stub = self._stub("CallawaySantAnnaResults")
        md = BusinessReport(stub, auto_diagnostics=False).full_report()
        assumption_section = md.split("## Identifying Assumption", 1)[1].split("\n## ", 1)[0]
        for pat in self._STRICT_PATTERNS:
            assert pat not in assumption_section, (
                f"Rendered assumption section must not carry the strict "
                f"phrase {pat!r} under anticipation > 0. Got: "
                f"{assumption_section!r}"
            )
        assert "Anticipation is allowed" in assumption_section


class TestAnticipationAwareAssumptionBlock:
    """Round-17 P1 regression: ``_describe_assumption`` must drop the
    strict "plus no anticipation" language when the fit allows
    ``anticipation > 0``. REGISTRY.md §CallawaySantAnna lines 355-395
    (and the matching SA / MultiPeriod / Wooldridge / EfficientDiD
    sections) treat anticipation as a relaxation of the strict no-
    anticipation assumption: no treatment effects earlier than ``k``
    periods before treatment, not none at all.
    """

    def test_cs_with_anticipation_sets_no_anticipation_false(self):
        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
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
        stub.anticipation = 2

        br = BusinessReport(stub, auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert (
            a["no_anticipation"] is False
        ), f"anticipation=2 must flip no_anticipation off; got {a}"
        assert a["anticipation_periods"] == 2
        assert "2 periods" in a["description"]
        assert "not strict no-anticipation" in a["description"]

    def test_efficient_did_with_anticipation_flips_no_anticipation_off(self):
        class EfficientDiDResults:
            pass

        stub = EfficientDiDResults()
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
        stub.pt_assumption = "all"
        stub.control_group = "never_treated"
        stub.anticipation = 1

        br = BusinessReport(stub, auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a["no_anticipation"] is False
        assert a["anticipation_periods"] == 1
        assert "1 period" in a["description"]

    def test_anticipation_zero_preserves_strict_no_anticipation(self):
        """Default (``anticipation=0``) keeps the strict text."""

        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
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
        stub.anticipation = 0

        br = BusinessReport(stub, auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a["no_anticipation"] is True
        assert "anticipation_periods" not in a
        assert "not strict no-anticipation" not in a["description"]


class TestContinuousDiDDynamicControlSample:
    """Round-18 P1 regression: ContinuousDiD with
    ``control_group="not_yet_treated"`` must take the dynamic-control
    path in ``to_dict()``, ``summary()``, and ``full_report()``. The
    stored ``n_control_units`` is only the fully-untreated ``D=0``
    tally; the actual comparison set includes future-treated cohorts
    beyond the anticipation window.
    """

    def test_continuous_did_not_yet_treated_surfaces_dynamic_mode(self):
        class ContinuousDiDResults:
            pass

        stub = ContinuousDiDResults()
        stub.overall_att = 1.0
        stub.overall_att_se = 0.2
        stub.overall_att_p_value = 0.001
        stub.overall_att_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 120
        stub.n_treated = 50
        stub.n_control = 70  # D=0 (never-treated) count only.
        stub.survey_metadata = None
        stub.control_group = "not_yet_treated"

        br = BusinessReport(stub, auto_diagnostics=False)
        sample = br.to_dict()["sample"]
        assert sample["n_control"] is None
        assert sample["n_never_treated"] == 70
        assert sample["dynamic_control"] is True

        summary = br.summary()
        assert " control)" not in summary
        assert "dynamic not-yet-treated" in summary

        full = br.full_report()
        assert "- Control: 70" not in full
        assert "dynamic not-yet-treated" in full


class TestStaggeredTripleDiffDynamicControlSample:
    """Round-18 P1 regression: StaggeredTripleDifference with
    ``control_group="notyettreated"`` (no underscore per the estimator
    contract) must also take the dynamic-control path. Its fixed
    subset is ``n_never_enabled`` (separate field) rather than a
    never-treated count.
    """

    def test_notyettreated_surfaces_n_never_enabled(self):
        class StaggeredTripleDiffResults:
            pass

        stub = StaggeredTripleDiffResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 200
        stub.n_treated = 80
        stub.n_control = 120  # Composite total (ignored in this mode).
        stub.n_never_enabled = 30  # Fixed subset exposed in this mode.
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.inference_method = "analytical"
        stub.control_group = "notyettreated"  # No underscore.

        br = BusinessReport(stub, auto_diagnostics=False)
        sample = br.to_dict()["sample"]
        assert sample["n_control"] is None
        assert sample["dynamic_control"] is True
        assert sample["n_never_enabled"] == 30
        assert sample["n_never_treated"] is None, (
            "StaggeredTripleDiff must expose n_never_enabled, not " "n_never_treated"
        )

        summary = br.summary()
        assert " control)" not in summary
        assert "dynamic not-yet-treated" in summary
        assert "30 never-enabled" in summary

        full = br.full_report()
        assert "- Control:" not in full
        assert "Never-enabled units present in the panel: 30" in full


class TestWooldridgeSampleNotYetTreatedSemantics:
    """Round-17 P1 regression: Wooldridge's ``n_control_units`` is the
    total eligible comparison set (never-treated plus future-treated
    units that contribute valid not-yet-treated comparisons). BR must
    NOT reinterpret that count as ``n_never_treated`` for Wooldridge,
    which would overstate never-treated availability. CS / SA /
    ImputationDiD / etc. retain the existing reinterpretation because
    their contracts define ``n_control_units`` as never-treated only.
    """

    def test_wooldridge_not_yet_treated_keeps_fixed_n_control(self):
        class WooldridgeDiDResults:
            pass

        stub = WooldridgeDiDResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60  # Total eligible, NOT never-treated only.
        stub.survey_metadata = None
        stub.control_group = "not_yet_treated"

        br = BusinessReport(stub, auto_diagnostics=False)
        sample = br.to_dict()["sample"]
        assert sample["n_control"] == 60, (
            "Wooldridge n_control_units is total eligible controls; "
            "must not be hidden behind not_yet_treated reinterpretation"
        )
        assert sample["n_never_treated"] is None

    def test_cs_not_yet_treated_still_reinterprets(self):
        """CS retains the existing behavior: the fixed ``n_control`` is
        suppressed and ``n_never_treated`` surfaces the never-treated
        count. Regression from round 13."""
        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        cs = CallawaySantAnna(base_period="universal", control_group="not_yet_treated").fit(
            sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        br = BusinessReport(cs, auto_diagnostics=False)
        sample = br.to_dict()["sample"]
        assert sample["n_control"] is None
        assert sample["n_never_treated"] == getattr(cs, "n_control_units", None)


class TestWooldridgeResultsRouting:
    """Round-16 P1 regression: the collectors must accept
    ``WooldridgeDiDResults`` payloads, which use ``att`` (not
    ``effect``). Without this, PT and heterogeneity silently skip on
    Wooldridge fits. Also, Wooldridge aggregation keeps ``t >= g`` and
    ignores the ``anticipation`` shift used by CS / SA / EfficientDiD
    (REGISTRY.md §Wooldridge lines 1351-1352).
    """

    def _wooldridge_stub(self, *, anticipation: int = 0):
        class WooldridgeDiDResults:
            pass

        stub = WooldridgeDiDResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.anticipation = anticipation
        # Event study: Wooldridge payloads use ``att`` not ``effect``.
        stub.event_study_effects = {
            -2: {"att": -0.05, "se": 0.1, "p_value": 0.62},
            -1: {"att": 0.04, "se": 0.1, "p_value": 0.69},
            0: {"att": 1.00, "se": 0.1, "p_value": 0.001},
            1: {"att": 1.20, "se": 0.1, "p_value": 0.001},
            2: {"att": 1.40, "se": 0.1, "p_value": 0.001},
        }
        return stub

    def test_pre_period_collector_reads_att_payload(self):
        from diff_diff.diagnostic_report import _collect_pre_period_coefs

        stub = self._wooldridge_stub()
        pre, _ = _collect_pre_period_coefs(stub)
        keys = sorted(row[0] for row in pre)
        assert keys == [
            -2,
            -1,
        ], f"pre-period collector must read Wooldridge ``att`` payloads; got {keys}"
        effects = {row[0]: row[1] for row in pre}
        assert effects[-2] == pytest.approx(-0.05)
        assert effects[-1] == pytest.approx(0.04)

    def test_heterogeneity_reads_att_payload(self):
        from diff_diff import DiagnosticReport

        stub = self._wooldridge_stub()
        dr = DiagnosticReport(stub)
        effects = sorted(dr._collect_effect_scalars())
        # Event-study post-only: rel >= 0 → {1.00, 1.20, 1.40}.
        assert effects == pytest.approx([1.00, 1.20, 1.40])

    def test_wooldridge_ignores_anticipation_shift_on_pre_periods(self):
        from diff_diff.diagnostic_report import _collect_pre_period_coefs

        stub = self._wooldridge_stub(anticipation=1)
        pre, _ = _collect_pre_period_coefs(stub)
        keys = sorted(row[0] for row in pre)
        # Wooldridge keeps rel < 0 regardless of anticipation.
        assert keys == [-2, -1]

    def test_wooldridge_ignores_anticipation_shift_on_heterogeneity(self):
        from diff_diff import DiagnosticReport

        stub = self._wooldridge_stub(anticipation=1)
        dr = DiagnosticReport(stub)
        effects = sorted(dr._collect_effect_scalars())
        # Anticipation window (rel=-1) must not leak into the post set
        # for Wooldridge even with anticipation=1.
        assert effects == pytest.approx([1.00, 1.20, 1.40])


class TestAnticipationAwareHorizonClassification:
    """Round-15 P1 regression: on anticipation-aware fits (CS / SA /
    EfficientDiD with ``anticipation > 0``), the report layer must
    classify horizons using the shifted boundary:

    - True pre-periods (PT + pre-trends power): ``rel < -anticipation``.
    - Treatment-affected horizons (heterogeneity dispersion):
      ``rel >= -anticipation`` (anticipation window is post-announcement).

    Prior code hard-coded ``rel < 0`` / ``rel >= 0`` and could include
    anticipation-window coefficients as "pre" in PT / power while
    excluding them as "post" in heterogeneity. REGISTRY.md
    §CallawaySantAnna lines 355-395 documents the shifted-boundary rule.
    """

    def _cs_stub_with_anticipation(self, *, anticipation: int = 1):
        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.base_period = "universal"
        stub.anticipation = anticipation
        stub.event_study_effects = {
            -3: {"effect": -0.05, "se": 0.1, "p_value": 0.62, "n_groups": 15},
            -2: {"effect": 0.04, "se": 0.1, "p_value": 0.69, "n_groups": 15},
            -1: {"effect": 0.80, "se": 0.1, "p_value": 0.01, "n_groups": 15},
            0: {"effect": 1.00, "se": 0.1, "p_value": 0.001, "n_groups": 15},
            1: {"effect": 1.20, "se": 0.1, "p_value": 0.001, "n_groups": 12},
            2: {"effect": 1.40, "se": 0.1, "p_value": 0.001, "n_groups": 10},
        }
        return stub

    def test_pre_period_collector_excludes_anticipation_window(self):
        from diff_diff.diagnostic_report import _collect_pre_period_coefs

        stub = self._cs_stub_with_anticipation(anticipation=1)
        pre, _ = _collect_pre_period_coefs(stub)
        keys = sorted(row[0] for row in pre)
        # Anticipation window (rel=-1) must be excluded; only -3, -2 remain.
        assert keys == [-3, -2], (
            f"pre-period collector must exclude the anticipation " f"window; got {keys}"
        )

    def test_heterogeneity_includes_anticipation_window(self):
        from diff_diff import DiagnosticReport

        stub = self._cs_stub_with_anticipation(anticipation=1)
        dr = DiagnosticReport(stub)
        effects = sorted(dr._collect_effect_scalars())
        # rel ∈ {-1, 0, 1, 2} → {0.80, 1.00, 1.20, 1.40}.
        assert effects == pytest.approx([0.80, 1.00, 1.20, 1.40])

    def test_anticipation_zero_preserves_old_behavior(self):
        from diff_diff import DiagnosticReport
        from diff_diff.diagnostic_report import _collect_pre_period_coefs

        stub = self._cs_stub_with_anticipation(anticipation=0)
        pre, _ = _collect_pre_period_coefs(stub)
        assert sorted(row[0] for row in pre) == [-3, -2, -1]

        dr = DiagnosticReport(stub)
        effects = sorted(dr._collect_effect_scalars())
        # Only non-negative horizons: 1.00, 1.20, 1.40.
        assert effects == pytest.approx([1.00, 1.20, 1.40])


class TestDiagFallbackDowngradeAppliedCentrally:
    """Round-14 regression: when ``compute_pretrends_power`` fell back to
    a diagonal-SE approximation while the full ``event_study_vcov`` was
    available, the ``well_powered`` tier must be downgraded to
    ``moderately_powered`` on **every** report surface (BR summary, BR
    full_report, BR schema, DR summary), not just inside one of them.
    Centralize the downgrade in ``_check_pretrends_power`` so every
    consumer reads the same adjusted tier. REPORTING.md lines 126-139.
    """

    def test_br_schema_tier_is_downgraded(self):
        """Smoke-check that the centralized downgrade lands in the DR
        schema when ``covariance_source`` is the flagged fallback value."""
        # Build a hand-crafted DR schema exactly as the centralized
        # downgrade would emit it — mdv ratio < 0.25 (so the pre-
        # downgrade tier is ``well_powered``), cov_source is the
        # diag-fallback-with-full-vcov-available sentinel.
        from diff_diff.diagnostic_report import DiagnosticReportResults

        schema = {
            "schema_version": "2.0",
            "estimator": "CallawaySantAnnaResults",
            "headline_metric": {"name": "overall_att", "value": 1.0},
            "parallel_trends": {
                "status": "ran",
                "method": "joint_wald_event_study",
                "joint_p_value": 0.40,
                "verdict": "no_detected_violation",
            },
            "pretrends_power": {
                "status": "ran",
                "method": "compute_pretrends_power",
                "mdv": 0.10,
                "mdv_share_of_att": 0.10,
                # Central downgrade: tier already reflects the cov-source.
                "tier": "moderately_powered",
                "covariance_source": "diag_fallback_available_full_vcov_unused",
            },
            "sensitivity": {"status": "not_applicable"},
            "placebo": {"status": "skipped", "reason": "opt-in"},
            "bacon": {"status": "not_applicable"},
            "design_effect": {"status": "not_applicable"},
            "heterogeneity": {"status": "not_applicable"},
            "epv": {"status": "not_applicable"},
            "estimator_native_diagnostics": {"status": "not_applicable"},
            "skipped": {},
            "warnings": [],
            "overall_interpretation": "",
            "next_steps": [],
        }

        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None

        dr_results = DiagnosticReportResults(
            schema=schema,
            interpretation="",
            applicable_checks=("parallel_trends", "pretrends_power"),
            skipped_checks={},
            warnings=(),
        )
        br = BusinessReport(stub, diagnostics=dr_results)
        br_schema = br.to_dict()
        pt_block = br_schema["pre_trends"]
        assert pt_block["power_tier"] == "moderately_powered"
        # All three prose surfaces must reflect the downgraded tier —
        # none should render the well-powered phrasing ("likely have
        # been detected" / well-powered adjective).
        summary = br.summary()
        full = br.full_report()
        for text in (summary, full):
            assert "well-powered" not in text.lower()
            assert "likely have" not in text
        # Positive check: moderately-informative phrasing appears in BR
        # prose and BR's overall-interpretation pass-through.
        assert (
            "moderately informative" in summary
            or "moderately informative" in full
            or "moderately-informative" in summary
        )

    def test_center_downgrade_fires_on_real_cs_fit(self, cs_fit):
        """On a real CS fit the central downgrade should land in the DR
        schema when the helper used the diagonal fallback — no separate
        BR-side downgrade is needed."""
        from diff_diff import DiagnosticReport

        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        pp = dr.to_dict()["pretrends_power"]
        if pp.get("status") != "ran":
            pytest.skip("pretrends_power did not run on this fixture")
        cov = pp.get("covariance_source")
        if cov != "diag_fallback_available_full_vcov_unused":
            pytest.skip(
                "fixture did not trigger the diag_fallback_available path; " "nothing to downgrade"
            )
        # When the flagged cov_source fires, tier must never be
        # ``well_powered`` — centralized downgrade guarantees this.
        assert pp["tier"] != "well_powered"

    def test_full_vcov_path_no_downgrade_on_real_cs_fit(self, cs_fit):
        """PR-B R4 regression: when ``compute_pretrends_power`` actually
        consumes the full ``event_study_vcov`` sub-block (PR-B Step 3),
        the DR / BR layer must NOT downgrade ``well_powered``.

        Exercises the live PR-B path on the deterministic ``cs_fit``
        fixture (analytical non-bootstrap CS, ``seed=7``,
        ``treatment_effect=1.5``). On this fixture the raw
        ``mdv / |att|`` ratio is well under the ``0.25`` well_powered
        threshold, so the expected tier is unconditionally
        ``well_powered`` — no skip-on-different-tier branch (R6 codex:
        previous version would silently bypass the key assertion if a
        regression reintroduced the downgrade).

        ``pretrends.py`` records
        ``covariance_source='full_pre_period_vcov'`` on the result, which
        the DR adapter consumes directly. The BR ``summary()`` prose
        (the actual surface the well-powered phrasing is rendered on)
        must contain the well-powered text and lack the conservative
        moderately-informative text.
        """
        from diff_diff import BusinessReport, DiagnosticReport
        from diff_diff.pretrends import compute_pretrends_power

        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        block = dr.to_dict()["pretrends_power"]
        assert block.get("status") == "ran", "pretrends_power should run on cs_fit"

        # Deterministic fixture pins (cs_fit at seed=7, treatment_effect=1.5):
        # cov_source = full_pre_period_vcov; max_abs_pre_violation ≈ 0.375
        # (γ * max(|t|) where pre-periods are [-4, -3, -2]); |att| ≈ 1.779;
        # mdv_share_of_att ≈ 0.211, well under 0.25 → tier = well_powered.
        # Codex R12 P1: this ratio is now `max_abs_pre_violation / |att|`,
        # the level-scale max pre-period violation under the MDV (post-PR-B
        # Step 4 linear MDV is in Roth's γ units, a slope; the level-scale
        # comparable is mdv * max(|violation_weights|)).
        assert block["covariance_source"] == "full_pre_period_vcov", (
            "cs_fit is analytical CS with event_study_vcov populated — "
            "PR-B routing must report full_pre_period_vcov"
        )
        # max_abs_pre_violation = mdv * max(|t|) = 0.0937 * 4 ≈ 0.375
        assert block.get("max_abs_pre_violation") is not None
        assert 0.35 < block["max_abs_pre_violation"] < 0.40, (
            f"cs_fit max_abs_pre_violation={block['max_abs_pre_violation']} "
            "should be ≈ 0.375 (γ ≈ 0.094 × max|t|=4)"
        )
        ratio = block["mdv_share_of_att"]
        assert ratio is not None and ratio < 0.25, (
            f"cs_fit mdv_share_of_att={ratio} (level-scale max_abs_pre_violation / "
            "|att|) must be in the well_powered range (<0.25) for this assertion "
            "to pin the no-downgrade contract"
        )
        assert (
            block["tier"] == "well_powered"
        ), "well-powered ratio must NOT be downgraded under the PR-B full-VCV path"

        # Architectural fix: the same provenance label appears on the
        # compute_pretrends_power output's persisted field, locking that
        # provenance is recorded at fit time and consumed at the report
        # layer (not re-inferred from the source-fit type).
        pp = compute_pretrends_power(fit, alpha=0.05, target_power=0.80)
        assert pp.covariance_source == "full_pre_period_vcov"

        # Positive prose contract on the rendered surfaces.
        br = BusinessReport(fit, data=sdf)
        summary = br.summary()
        full = br.full_report()
        # Primary surface: summary() renders the tier prose.
        assert "well-powered" in summary, (
            "BR.summary() should surface well-powered phrasing under the "
            "PR-B full-VCV no-downgrade path"
        )
        assert "moderately informative" not in summary
        assert "moderately-informative" not in summary
        # Secondary defensive check on full_report().
        assert "moderately informative" not in full.lower()
        assert "moderately-informative" not in full.lower()

        # PR-B R14 P2: max_abs_pre_violation must round-trip through the
        # BR schema lift AND render in full_report(). Pre-R14 the field
        # was emitted by DR, the renderer printed it, but the BR lift
        # boundary at `_lift_pre_trends` silently dropped it — so the
        # rendered line never fired even though the renderer had the
        # branch.
        br_schema = br.to_dict()
        pt_block = br_schema.get("pre_trends", {})
        assert "max_abs_pre_violation" in pt_block, (
            "BR.to_dict()['pre_trends'] must surface max_abs_pre_violation "
            "post-PR-B R14 — _lift_pre_trends regression"
        )
        assert pt_block["max_abs_pre_violation"] is not None
        assert np.isclose(pt_block["max_abs_pre_violation"], 0.375, atol=0.05)
        # full_report() must render the new "Max pre-period level
        # deviation at MDV" line.
        assert "Max pre-period level deviation at MDV:" in full, (
            "BR.full_report() must render the max_abs_pre_violation line "
            "(renderer wired in R12; lift boundary fixed in R14)"
        )


class TestCSNotYetTreatedControlGroupSemantics:
    """Round-13 P1 regression: ``BusinessReport`` must not relabel
    ``n_control_units`` as generic "control" for a
    ``CallawaySantAnna(control_group='not_yet_treated')`` fit — that
    field counts only never-treated units, while the actual comparison
    group is the dynamic not-yet-treated set at each (g, t) cell.
    """

    def test_not_yet_treated_fit_does_not_render_misleading_control_count(self):
        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        # Fit with the dynamic not-yet-treated comparison mode.
        cs = CallawaySantAnna(base_period="universal", control_group="not_yet_treated").fit(
            sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        br = BusinessReport(cs, auto_diagnostics=False)
        sample = br.to_dict()["sample"]

        # Fixed ``n_control`` must NOT be populated — the comparison set
        # is dynamic per (g, t), not a fixed unit tally.
        assert (
            sample["n_control"] is None
        ), f"n_control must be None for not_yet_treated; got {sample['n_control']}"
        # The new fields surface the real semantics.
        assert sample["control_group"] == "not_yet_treated"
        assert sample["n_never_treated"] == getattr(cs, "n_control_units", None)

        # Both summary and full_report must describe the dynamic
        # comparison group rather than asserting a misleading "control"
        # count.
        summary = br.summary()
        # No "(N treated, N control)" phrasing on this path.
        assert " control)" not in summary
        assert "not-yet-treated" in summary or "dynamic" in summary

        full = br.full_report()
        assert "- Control:" not in full or "not-yet-treated" in full
        assert "dynamic not-yet-treated" in full or "not-yet-treated" in full

    def test_never_treated_fit_still_shows_fixed_control_count(self, cs_fit):
        """Default path (``control_group='never_treated'``) keeps the
        fixed ``n_control`` tally so existing prose is unchanged."""
        fit, _ = cs_fit  # default is never_treated
        br = BusinessReport(fit, auto_diagnostics=False)
        sample = br.to_dict()["sample"]
        assert isinstance(sample["n_control"], int)
        assert sample["control_group"] == "never_treated"


class TestBRDataKwargsPassthroughToAutoDR:
    """Round-12 regression: ``BusinessReport`` now accepts
    ``data`` / ``outcome`` / ``treatment`` / ``unit`` / ``time`` /
    ``first_treat`` kwargs and forwards them to the auto-constructed
    ``DiagnosticReport``. Without this, data-dependent checks (2x2 PT,
    Bacon, EfficientDiD Hausman) are silently skipped on the zero-
    config auto path even though the README markets one-call
    diagnostics from a fitted result.
    """

    def test_did_fit_gets_2x2_pt_via_passthrough(self, did_fit):
        fit, df = did_fit
        br = BusinessReport(
            fit,
            data=df,
            outcome="outcome",
            treatment="treated",
            time="post",
        )
        # Auto-DR received the kwargs and ran the 2x2 PT check.
        dr_schema = br.to_dict()["diagnostics"]["schema"]
        assert dr_schema["parallel_trends"]["status"] == "ran"
        assert dr_schema["parallel_trends"]["method"] == "slope_difference"

    def test_cs_fit_gets_bacon_via_passthrough(self, cs_fit):
        fit, sdf = cs_fit
        br = BusinessReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        dr_schema = br.to_dict()["diagnostics"]["schema"]
        # Bacon needs data + outcome + time + unit + first_treat; before
        # the passthrough, the auto path skipped because only the
        # estimator result was available.
        assert dr_schema["bacon"]["status"] == "ran"

    def test_no_passthrough_still_works_and_skips_gracefully(self, did_fit):
        """Zero-config auto path must still produce a valid report; it
        just skips data-dependent checks."""
        fit, _ = did_fit
        br = BusinessReport(fit)  # no data kwargs
        dr_schema = br.to_dict()["diagnostics"]["schema"]
        # PT needs data for 2x2 and was gated out of applicable — section
        # is "skipped" rather than "ran".
        assert dr_schema["parallel_trends"]["status"] in {"skipped", "not_applicable"}


class TestSensitivityProseGuarding:
    """Round-12 regression: BR / DR summary prose must not promise a
    "sensitivity analysis below" sentence when no sensitivity block
    actually ran (e.g., SDiD / TROP routed to native diagnostics,
    single-M precomputed passthrough rendered separately, skipped
    sensitivity for varying-base CS).
    """

    def test_br_sdid_does_not_mention_sensitivity_below(self, sdid_fit):
        fit, _ = sdid_fit
        summary = BusinessReport(fit).summary()
        # SDiD routes to estimator-native diagnostics, not HonestDiD.
        # The PT verdict for SDiD is ``design_enforced_pt`` which does
        # not append any "see sensitivity below" clause, so the prose
        # should not mention it.
        assert "sensitivity analysis below" not in summary

    def test_dr_trop_does_not_mention_sensitivity_below(self, sdid_fit):
        # SDiD and TROP both skip HonestDiD. Use SDiD as proxy here
        # since it already has a fixture; the same guard covers TROP.
        from diff_diff import DiagnosticReport

        fit, _ = sdid_fit
        summary = DiagnosticReport(fit).summary()
        assert "sensitivity analysis below" not in summary


class TestSDiDTROPSkippedSensitivityCaveatSuppressed:
    """Round-20 P2 regression on PR #318: ``DiagnosticReport`` marks the
    HonestDiD sensitivity block ``status="skipped", method="estimator_native"``
    for SDiD / TROP because robustness is routed to the native diagnostics
    (``in_time_placebo``, ``sensitivity_to_zeta_omega``, factor-model
    metrics) under ``estimator_native_diagnostics``. ``BusinessReport``
    must not surface "HonestDiD sensitivity was not run" as a warning
    caveat when the native battery actually ran, because that contradicts
    the documented native-routing contract and misleads the reader into
    thinking robustness was skipped.
    """

    def test_sdid_native_routed_suppresses_skipped_caveat(self, sdid_fit):
        from diff_diff import DiagnosticReport

        fit, _ = sdid_fit
        br = BusinessReport(fit)
        schema = br.to_dict()

        # BR's lifted ``sensitivity`` block only carries status/reason; the
        # ``method`` field lives on the DR schema, which BR reads internally
        # to decide caveat suppression. Confirm the DR-side shape separately.
        assert schema["sensitivity"]["status"] == "skipped"
        dr_schema = DiagnosticReport(fit).to_dict()
        assert dr_schema["sensitivity"]["status"] == "skipped"
        assert dr_schema["sensitivity"]["method"] == "estimator_native"
        native_ran = dr_schema["estimator_native_diagnostics"].get("status") == "ran"

        caveat_topics = [c.get("topic") for c in schema.get("caveats", [])]
        if native_ran:
            # The fix: no "sensitivity_skipped" warning; instead an info
            # caveat pointing at the native block.
            assert "sensitivity_skipped" not in caveat_topics
            assert "sensitivity_native_routed" in caveat_topics
            native_msg = next(
                c for c in schema["caveats"] if c.get("topic") == "sensitivity_native_routed"
            )
            assert native_msg["severity"] == "info"
            assert "estimator-native" in native_msg["message"].lower()
        else:
            # When the native battery did not produce a ran block, the
            # legacy warning behavior is still correct — SDiD users should
            # know HonestDiD was not attempted.
            assert (
                "sensitivity_skipped" in caveat_topics
                or "sensitivity_native_routed" in caveat_topics
            )


class TestEfficientDiDHausmanStepTaggedAsParallelTrends:
    """Round-20 P2 regression on PR #318: the EfficientDiD practitioner
    workflow step "Run Hausman pretest (PT-All vs PT-Post)" must be
    tagged ``_step_name="parallel_trends"``, not ``"heterogeneity"``, so
    that ``DiagnosticReport._collect_next_steps()`` — which treats a ran
    Hausman block as parallel-trends completion — correctly suppresses the
    step from the "next steps" list when the report already executed it.
    REGISTRY.md §EfficientDiD (lines 895-908) classifies the Hausman
    pretest as a parallel-trends diagnostic, so the fix aligns the
    practitioner tag with the identification-layer classification.
    """

    def test_hausman_step_is_tagged_parallel_trends(self):
        """``practitioner_next_steps`` strips ``_step_name`` from the
        returned steps, so we exercise the tagging via the
        ``completed_steps=["parallel_trends"]`` filter contract: a
        correctly-tagged Hausman step is removed from the output; a
        mistagged step remains.
        """
        from diff_diff.practitioner import practitioner_next_steps

        class EfficientDiDResults:
            pass

        stub = EfficientDiDResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.3
        stub.overall_p_value = 0.01
        stub.overall_conf_int = (0.4, 1.6)
        stub.alpha = 0.05
        stub.n_obs = 500
        stub.n_treated = 200
        stub.n_control = 300
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.pt_assumption = "all"

        # Without any completed steps, the Hausman pretest is included.
        baseline = practitioner_next_steps(stub, verbose=False)["next_steps"]
        hausman_in_baseline = any("Hausman pretest" in s.get("label", "") for s in baseline)
        assert hausman_in_baseline, "EfficientDiD workflow must include the Hausman pretest step"

        # After marking ``parallel_trends`` complete (which DR does when
        # ``_check_pt_hausman`` runs), the Hausman step must be filtered
        # out. Before the round-20 retag it was tagged as
        # ``heterogeneity`` and survived this filter — that is the bug.
        filtered = practitioner_next_steps(
            stub, completed_steps=["parallel_trends"], verbose=False
        )["next_steps"]
        assert not any("Hausman pretest" in s.get("label", "") for s in filtered), (
            "Hausman step must be tagged as 'parallel_trends' (REGISTRY.md "
            "§EfficientDiD classifies it as a PT diagnostic) so that "
            "DR's _collect_next_steps() suppresses it after running the same "
            "check. Still present after completed_steps=['parallel_trends'] "
            "filter, meaning the tag is wrong."
        )


class TestSpecificationComparisonStepTagPersistsAfterSensitivityRuns:
    """Pre-emptive audit regression: several practitioner handlers
    previously tagged their "compare specifications" / "vary control
    group" step as ``_step_name="sensitivity"``. DR marks ``sensitivity``
    complete when HonestDiD runs — which is orthogonal to the
    specification-variation recommendation — so these steps were
    incorrectly suppressed from ``next_steps`` after a fit with
    HonestDiD sensitivity. Retag as ``specification_comparison`` so the
    recommendations persist alongside a completed HonestDiD block. Same
    class of mistag as the round-20 Hausman finding (which was about
    ``heterogeneity`` vs ``parallel_trends``).
    """

    @staticmethod
    def _build_stub(class_name: str, **extras):
        stub_cls = type(class_name, (), {})
        stub = stub_cls()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 400
        stub.n_treated = 100
        stub.n_control = 300
        stub.survey_metadata = None
        stub.event_study_effects = None
        for k, v in extras.items():
            setattr(stub, k, v)
        return stub

    @staticmethod
    def _step_labels_after_completed(stub, completed):
        from diff_diff.practitioner import practitioner_next_steps

        return [
            s.get("label", "")
            for s in practitioner_next_steps(stub, completed_steps=completed, verbose=False)[
                "next_steps"
            ]
        ]

    def test_sa_specification_falsification_persists_after_sensitivity_runs(self):
        stub = self._build_stub("SunAbrahamResults")
        labels = self._step_labels_after_completed(stub, completed=["sensitivity"])
        assert any("Specification-based falsification" in lab for lab in labels), (
            "SA's 'Specification-based falsification' step must persist "
            "after DR marks sensitivity complete — HonestDiD does not run "
            "control_group / anticipation variation."
        )

    def test_imputation_specification_falsification_persists_after_sensitivity_runs(self):
        stub = self._build_stub("ImputationDiDResults")
        labels = self._step_labels_after_completed(stub, completed=["sensitivity"])
        assert any("Specification-based falsification" in lab for lab in labels)

    def test_two_stage_specification_falsification_persists_after_sensitivity_runs(self):
        stub = self._build_stub("TwoStageDiDResults")
        labels = self._step_labels_after_completed(stub, completed=["sensitivity"])
        assert any("Specification-based falsification" in lab for lab in labels)

    def test_stacked_clean_control_variation_persists_after_sensitivity_runs(self):
        stub = self._build_stub("StackedDiDResults")
        labels = self._step_labels_after_completed(stub, completed=["sensitivity"])
        assert any("Vary clean control" in lab for lab in labels), (
            "StackedDiD's 'Vary clean control definition' step must "
            "persist after DR marks sensitivity complete — HonestDiD does "
            "not replay clean_control variation."
        )

    def test_efficient_compare_control_groups_persists_after_sensitivity_runs(self):
        stub = self._build_stub("EfficientDiDResults", pt_assumption="all")
        labels = self._step_labels_after_completed(stub, completed=["sensitivity"])
        assert any("Compare control group definitions" in lab for lab in labels), (
            "EfficientDiD's 'Compare control group definitions' step "
            "must persist after DR marks sensitivity complete — HonestDiD "
            "does not re-estimate with alternative control_group."
        )


class TestCSRepeatedCrossSectionCountLabels:
    """Round-28 P2 CI review on PR #318: ``CallawaySantAnna(panel=False)``
    stores treated / control counts as OBSERVATIONS, not units
    (``staggered_results.py L183-L184`` renders them as "obs:" in that
    mode). BR previously labeled them as "units" / "present in the
    panel", which misstates the sample composition on repeated-cross-
    section fits. The schema now carries a ``count_unit`` flag and the
    rendering branches on it.
    """

    @staticmethod
    def _stub(panel: bool):
        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 1000
        stub.n_treated_units = 200
        stub.n_control_units = 800
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.control_group = "not_yet_treated"
        stub.panel = panel
        return stub

    def test_schema_exposes_count_unit(self):
        for panel, expected in [(True, "units"), (False, "observations")]:
            sample = BusinessReport(self._stub(panel), auto_diagnostics=False).to_dict()["sample"]
            assert sample["count_unit"] == expected

    def test_panel_true_renders_unit_wording(self):
        br = BusinessReport(self._stub(panel=True), auto_diagnostics=False)
        summary = br.summary()
        md = br.full_report()
        assert "never-treated units" in summary
        assert "present in the panel" in md
        assert "repeated cross-section sample" not in md

    def test_panel_false_renders_rcs_wording(self):
        br = BusinessReport(self._stub(panel=False), auto_diagnostics=False)
        summary = br.summary()
        md = br.full_report()
        # RCS-specific wording in both surfaces.
        assert "never-treated observations" in summary
        assert "repeated cross-section sample" in md
        # No misleading "units" or "panel" claims.
        assert "never-treated units" not in summary
        assert "present in the panel" not in md


class TestTROPApplicableChecksExcludesParallelTrends:
    """Round-28 P2 CI review on PR #318: TROP identification is
    factor-model-based; its native PT handler returns
    ``status="not_applicable"``. Advertising ``parallel_trends`` in
    ``DiagnosticReport.applicable_checks`` for TROP was a contract
    mismatch for callers using that set to gate workflows or UI.
    """

    def test_trop_applicable_checks_omits_parallel_trends(self):
        from diff_diff import DiagnosticReport

        class TROPResults:
            pass

        stub = TROPResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.alpha = 0.05
        stub.n_obs = 100

        dr = DiagnosticReport(stub)
        assert "parallel_trends" not in dr.applicable_checks, (
            "TROP PT routes to factor-model diagnostics and is "
            "not_applicable; it must not appear in applicable_checks."
        )


class TestSurveyPTProsePropagation:
    """Round-28 P3 CI review on PR #318: the survey F-reference PT
    variants (``joint_wald_survey``, ``joint_wald_event_study_survey``)
    must carry through BR's method-aware label helpers so prose uses
    "joint p" (not the fall-through default) and preserves the
    ``df_denom`` provenance in the BR schema.
    """

    def test_lift_pre_trends_preserves_df_denom(self):
        from diff_diff.business_report import _lift_pre_trends

        fake_dr = {
            "parallel_trends": {
                "status": "ran",
                "method": "joint_wald_event_study_survey",
                "joint_p_value": 0.35,
                "df_denom": 30.0,
                "n_pre_periods": 3,
                "verdict": "no_detected_violation",
            },
            "pretrends_power": {"status": "not_applicable"},
        }
        lifted = _lift_pre_trends(fake_dr)
        assert lifted["method"] == "joint_wald_event_study_survey"
        assert lifted["df_denom"] == 30.0

    def test_lift_pre_trends_exposes_power_reason(self):
        """Round-29 P3 regression: when ``compute_pretrends_power`` cannot
        run, REPORTING.md lines 118-125 promise the fallback reason is
        recorded in the BR pre-trends block. Previously only the enum
        status surfaced and the reason was dropped at the lift
        boundary; the new ``power_reason`` field carries the
        plain-English explanation alongside the existing enum
        ``power_status``.
        """
        from diff_diff.business_report import _lift_pre_trends

        fake_dr = {
            "parallel_trends": {
                "status": "ran",
                "method": "joint_wald_event_study",
                "joint_p_value": 0.35,
                "n_pre_periods": 3,
                "verdict": "no_detected_violation",
            },
            "pretrends_power": {
                "status": "not_applicable",
                "reason": (
                    "StackedDiDResults does not yet have a " "compute_pretrends_power adapter."
                ),
            },
        }
        lifted = _lift_pre_trends(fake_dr)
        # Machine-readable status preserved.
        assert lifted["power_status"] == "not_applicable"
        # Plain-English reason now exposed on the schema.
        assert lifted["power_reason"] == (
            "StackedDiDResults does not yet have a " "compute_pretrends_power adapter."
        )

    def test_survey_pt_method_stat_label_uses_joint_p(self):
        from diff_diff.business_report import (
            _pt_method_stat_label,
            _pt_method_subject,
        )

        for method in ("joint_wald_survey", "joint_wald_event_study_survey"):
            assert _pt_method_stat_label(method) == "joint p", (
                f"Survey PT variant {method!r} must map to 'joint p' "
                f"(the joint test remains; only the reference "
                f"distribution changes)."
            )
            assert _pt_method_subject(method) == "Pre-treatment event-study coefficients", (
                f"Survey PT variant {method!r} must use the event-study "
                f"subject phrase, not the generic fall-through."
            )


class TestSDiDJackknifeStepPersistsAfterNativeSensitivity:
    """Round-24 P2 CI review on PR #318: the SyntheticDiD practitioner
    step "Leave-one-out influence (jackknife)" must persist after
    ``DiagnosticReport`` marks ``sensitivity`` complete via the SDiD
    native battery (pre-treatment fit, weight concentration,
    ``in_time_placebo``, ``sensitivity_to_zeta_omega``). DR does NOT
    run the jackknife LOO workflow — ``get_loo_effects_df`` requires a
    separate ``variance_method='jackknife'`` fit — so suppressing the
    recommendation when the native block fires overstates what the
    report has already executed. Same class as round-20 Hausman and
    pre-emptive TROP-placebo retags: step_name was coarser than DR's
    actual coverage.
    """

    def test_sdid_jackknife_step_persists_via_practitioner_filter(self):
        """Unit-level: ``practitioner_next_steps`` with
        ``completed_steps=["sensitivity"]`` still surfaces the jackknife
        recommendation because it is now tagged ``loo_jackknife``.
        """
        from diff_diff.practitioner import practitioner_next_steps

        class SyntheticDiDResults:
            pass

        stub = SyntheticDiDResults()
        stub.att = 1.0
        stub.se = 0.2
        stub.p_value = 0.001
        stub.conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 200
        stub.n_treated = 20
        stub.n_control = 180
        stub.survey_metadata = None
        stub.event_study_effects = None

        labels = [
            s.get("label", "")
            for s in practitioner_next_steps(stub, completed_steps=["sensitivity"], verbose=False)[
                "next_steps"
            ]
        ]
        assert any("Leave-one-out influence (jackknife)" in lab for lab in labels), (
            "SDiD jackknife recommendation must persist after DR marks "
            "sensitivity complete — the SDiD native battery does not run "
            "the jackknife LOO workflow (requires a separate "
            "variance_method='jackknife' fit)."
        )

    def test_sdid_jackknife_step_persists_in_dr_next_steps(self, sdid_fit):
        """Integration: ``DiagnosticReport(...).to_dict()["next_steps"]``
        preserves the jackknife recommendation when only the default
        native SDiD diagnostics ran.
        """
        from diff_diff import DiagnosticReport

        fit, _ = sdid_fit
        next_steps = DiagnosticReport(fit).to_dict()["next_steps"]
        labels = [s.get("label", "") for s in next_steps]
        assert any("Leave-one-out influence (jackknife)" in lab for lab in labels), (
            "DR next_steps must preserve the SDiD jackknife recommendation "
            "when the SDiD native battery ran but the jackknife workflow "
            f"did not. Got labels: {labels}"
        )


class TestTROPInTimePlaceboStepTaggedAsPlacebo:
    """Pre-emptive audit regression: the TROP practitioner workflow
    step "In-time or in-space placebo" was previously tagged
    ``_step_name="sensitivity"``. TROP's estimator-native diagnostics
    surface factor-model fit metrics (``effective_rank``, ``loocv_score``,
    selected lambdas) — not placebos — and
    ``DiagnosticReport._collect_next_steps`` marks ``sensitivity`` complete
    for SDiD / TROP when the native battery runs. That suppressed the
    TROP placebo recommendation unjustly. Retag as ``placebo`` so it
    persists.
    """

    def test_trop_placebo_step_persists_after_native_sensitivity_completion(self):
        from diff_diff.practitioner import practitioner_next_steps

        class TROPResults:
            pass

        stub = TROPResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 400
        stub.n_treated = 40
        stub.n_control = 360
        stub.survey_metadata = None
        stub.event_study_effects = None

        labels = [
            s.get("label", "")
            for s in practitioner_next_steps(stub, completed_steps=["sensitivity"], verbose=False)[
                "next_steps"
            ]
        ]
        assert any("In-time or in-space placebo" in lab for lab in labels), (
            "TROP's placebo recommendation must persist after DR marks "
            "sensitivity complete (SDiD/TROP native battery) — factor-"
            "model diagnostics are not a placebo substitute."
        )


class TestPrecomputedSensitivityHonoredOnAllCompatibleEstimators:
    """Round-31 P1 CI review on PR #318: ``DiagnosticReport(precomputed=
    {"sensitivity": ...})`` and ``BusinessReport(honest_did_results=...)``
    were silently dropped on estimator families whose ``_APPLICABILITY``
    row lacked ``"sensitivity"`` — SA, Imputation, TwoStage, Stacked,
    EfficientDiD, Wooldridge, TripleDifference, StaggeredTripleDiff,
    ContinuousDiD, and plain DiD. The applicability gate filtered the
    section out before the supplied object reached the runner, so the
    schema rendered ``sensitivity: {"status": "not_applicable"}`` and
    the user never learned their robustness result had been ignored.

    The gate now honors an explicit passthrough regardless of the
    default ``_APPLICABILITY`` matrix. SDiD / TROP are still rejected
    up front in ``__init__`` (round-21) because their native-routing
    contract is methodology-incompatible with HonestDiD.
    """

    @staticmethod
    def _fake_grid_sens():
        from types import SimpleNamespace

        return SimpleNamespace(
            M_values=[0.5, 1.0, 1.5],
            bounds=[(0.1, 2.0), (-0.2, 2.5), (-0.5, 3.0)],
            robust_cis=[(0.05, 2.1), (-0.3, 2.6), (-0.6, 3.1)],
            breakdown_M=1.25,
            method="relative_magnitude",
            original_estimate=1.0,
            original_se=0.2,
            alpha=0.05,
        )

    @staticmethod
    def _stub(class_name: str, **extras):

        # For estimator types that have fits, we'd use real fits; but
        # several of these need specific setup. Stub with minimal
        # required fields — the gate fix operates on the applicability
        # set and the sensitivity runner short-circuits on the
        # precomputed key without touching result internals.
        stub_cls = type(class_name, (), {})
        stub = stub_cls()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.att = 1.0
        stub.se = 0.2
        stub.p_value = 0.001
        stub.conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 500
        stub.n_treated = 200
        stub.n_control = 300
        stub.survey_metadata = None
        stub.event_study_effects = None
        for k, v in extras.items():
            setattr(stub, k, v)
        return stub

    def test_dr_precomputed_sensitivity_honored_on_sun_abraham(self):
        from diff_diff import DiagnosticReport

        stub = self._stub("SunAbrahamResults")
        dr = DiagnosticReport(stub, precomputed={"sensitivity": self._fake_grid_sens()})
        sens = dr.to_dict()["sensitivity"]
        assert sens["status"] == "ran", (
            f"precomputed sensitivity on SunAbrahamResults must be honored; " f"got {sens!r}"
        )
        assert sens.get("precomputed") is True
        assert sens["breakdown_M"] == 1.25

    def test_dr_precomputed_sensitivity_honored_on_efficient_did(self):
        from diff_diff import DiagnosticReport

        stub = self._stub("EfficientDiDResults", pt_assumption="all")
        dr = DiagnosticReport(stub, precomputed={"sensitivity": self._fake_grid_sens()})
        sens = dr.to_dict()["sensitivity"]
        assert sens["status"] == "ran"
        assert sens.get("precomputed") is True

    def test_dr_precomputed_sensitivity_honored_on_plain_did(self):
        from diff_diff import DiagnosticReport

        stub = self._stub("DiDResults")
        dr = DiagnosticReport(stub, precomputed={"sensitivity": self._fake_grid_sens()})
        sens = dr.to_dict()["sensitivity"]
        assert sens["status"] == "ran"

    def test_br_honest_did_results_honored_on_imputation(self):
        stub = self._stub("ImputationDiDResults")
        br = BusinessReport(stub, honest_did_results=self._fake_grid_sens())
        sens = br.to_dict()["sensitivity"]
        assert sens["status"] == "computed", (
            f"honest_did_results on ImputationDiDResults must be honored " f"by BR; got {sens!r}"
        )
        assert sens["breakdown_M"] == 1.25


class TestHeterogeneityLiftAlwaysReturnsDict:
    """Round-31 P2 CI review on PR #318: ``_lift_heterogeneity`` used to
    return ``None`` whenever the DR heterogeneity section didn't
    successfully run, so the BR schema stored a raw ``None`` at
    ``schema["heterogeneity"]``. The rest of the schema promises dict-
    shaped ``{"status": ..., "reason": ...}`` blocks on every top-
    level key; this one broke the contract and forced downstream
    consumers to special-case it.
    """

    def test_lift_none_dr_returns_dict(self):
        from diff_diff.business_report import _lift_heterogeneity

        block = _lift_heterogeneity(None)
        assert isinstance(block, dict)
        assert block["status"] == "skipped"
        assert "auto_diagnostics" in (block.get("reason") or "")

    def test_lift_skipped_dr_section_returns_dict_with_status(self):
        from diff_diff.business_report import _lift_heterogeneity

        block = _lift_heterogeneity(
            {
                "heterogeneity": {
                    "status": "skipped",
                    "reason": "No group_effects or event_study_effects on result.",
                }
            }
        )
        assert block["status"] == "skipped"
        assert "No group_effects" in block["reason"]

    def test_lift_not_applicable_dr_section_returns_dict(self):
        from diff_diff.business_report import _lift_heterogeneity

        block = _lift_heterogeneity(
            {
                "heterogeneity": {
                    "status": "not_applicable",
                    "reason": "TripleDifferenceResults is a 2-period design.",
                }
            }
        )
        assert block["status"] == "not_applicable"
        assert block["reason"]

    def test_br_schema_heterogeneity_is_always_dict(self):
        """End-to-end: a fit whose heterogeneity did not run still
        exposes a dict-shaped block at ``schema["heterogeneity"]``
        rather than a raw ``None``.
        """

        class DiDResults:
            pass

        stub = DiDResults()
        stub.att = 1.0
        stub.se = 0.2
        stub.p_value = 0.001
        stub.conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 200
        stub.n_treated = 100
        stub.n_control = 100
        stub.survey_metadata = None

        het = BusinessReport(stub, auto_diagnostics=True).to_dict()["heterogeneity"]
        assert isinstance(het, dict), (
            f"schema['heterogeneity'] must be a dict (the stable-schema "
            f"contract); got {type(het).__name__}: {het!r}"
        )
        assert "status" in het


class TestSDiDTROPRejectPrecomputedPretrendsPower:
    """Round-32 P1 CI review on PR #318: round-21 rejected
    ``precomputed["sensitivity"]`` / ``precomputed["parallel_trends"]``
    on SDiD / TROP because the native-routing contract makes those
    methodology-incompatible. Round-31's broadening of the
    applicability gate exposed a parallel hole — ``precomputed[
    "pretrends_power"]`` was not in the rejection set, so a Roth-
    style power verdict could surface on a report whose PT is
    design-enforced (SDiD) or factor-model (TROP). The guard now
    rejects all three precomputed keys uniformly on the native-
    routed estimator families.
    """

    @staticmethod
    def _dummy_power_object():
        from types import SimpleNamespace

        return SimpleNamespace(
            mdv=0.1,
            violation_type="linear",
            alpha=0.05,
            target_power=0.80,
            violation_magnitude=0.1,
            power=0.80,
            n_pre_periods=2,
        )

    def test_dr_rejects_precomputed_pretrends_power_on_sdid(self, sdid_fit):
        from diff_diff import DiagnosticReport

        fit, _ = sdid_fit
        with pytest.raises(ValueError, match="estimator_native_diagnostics"):
            DiagnosticReport(fit, precomputed={"pretrends_power": self._dummy_power_object()})

    def test_dr_rejects_precomputed_pretrends_power_on_trop(self):
        from diff_diff import DiagnosticReport

        class TROPResults:
            pass

        stub = TROPResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.alpha = 0.05
        stub.n_obs = 100
        with pytest.raises(ValueError, match="estimator_native_diagnostics"):
            DiagnosticReport(stub, precomputed={"pretrends_power": self._dummy_power_object()})


class TestHeterogeneityOmittedFromFullReportWhenNotRan:
    """Round-32 P2 CI review on PR #318: round-31 made
    ``_lift_heterogeneity`` always return a dict (stable schema
    contract), but the full-report renderer's ``if het:`` truthiness
    guard then entered the Heterogeneity section on every fit and
    printed ``Source: None`` / ``N effects: None`` / ``Sign
    consistent: None``. Renderer now gates on ``status == "ran"``.
    """

    def test_full_report_omits_heterogeneity_section_when_skipped(self):
        class DiDResults:
            pass

        stub = DiDResults()
        stub.att = 1.0
        stub.se = 0.2
        stub.p_value = 0.001
        stub.conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 200
        stub.n_treated = 100
        stub.n_control = 100
        stub.survey_metadata = None

        md = BusinessReport(stub, auto_diagnostics=True).full_report()
        # The section header is only emitted when status == "ran".
        # Plain DiD does not have heterogeneity in its applicability
        # row, so the section should NOT appear.
        assert "## Heterogeneity" not in md, (
            f"Heterogeneity section must be omitted when it did not "
            f"run; rendering ``Source: None`` / ``N effects: None`` "
            f"is worse than omitting. Got markdown:\n{md}"
        )
        # Specifically, none of the placeholder ``None`` lines may
        # appear anywhere in the rendered report.
        assert "Source: `None`" not in md
        assert "N effects: None" not in md
        assert "Sign consistent: None" not in md


class TestDesignEffectBandLabel:
    """Round-32 P2 CI review on PR #318: REPORTING.md promises a
    plain-English band label on the ``design_effect`` section, but the
    implementation only emitted numeric fields plus ``is_trivial``.
    Add a stable ``band_label`` enum aligned with the REPORTING.md
    threshold rule.
    """

    @staticmethod
    def _stub_with_deff(deff: float):
        from types import SimpleNamespace

        from diff_diff import DiagnosticReport

        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 500
        stub.n_treated = 100
        stub.n_control_units = 400
        stub.event_study_effects = None
        stub.survey_metadata = SimpleNamespace(
            design_effect=deff,
            effective_n=500.0 / max(deff, 1e-9),
            weight_type="pweight",
            n_strata=None,
            n_psu=None,
            df_survey=None,
            replicate_method=None,
        )
        return DiagnosticReport(stub).to_dict()["design_effect"]

    def test_trivial_band_under_1_05(self):
        assert self._stub_with_deff(1.01)["band_label"] == "trivial"

    def test_slightly_reduces_band_under_2(self):
        assert self._stub_with_deff(1.5)["band_label"] == "slightly_reduces"

    def test_materially_reduces_band_under_5(self):
        assert self._stub_with_deff(3.2)["band_label"] == "materially_reduces"

    def test_large_warning_band_at_or_above_5(self):
        assert self._stub_with_deff(7.5)["band_label"] == "large_warning"

    def test_deff_exactly_1_05_is_slightly_reduces_not_trivial(self):
        """Round-43 P2 regression: REPORTING.md defines the ``trivial``
        band as ``0.95 <= deff < 1.05`` (half-open) and
        ``slightly_reduces`` as starting at ``1.05``. The prior code used
        ``is_trivial = 0.95 <= deff <= 1.05`` (closed), producing a
        schema that labeled exactly ``deff == 1.05`` as
        ``band_label="slightly_reduces"`` AND ``is_trivial=True`` —
        internally inconsistent, and the ``is_trivial=True`` flag
        suppressed the non-trivial prose that the documented threshold
        says should fire.
        """
        # DR schema: band_label="slightly_reduces" and is_trivial=False.
        block = self._stub_with_deff(1.05)
        assert block["band_label"] == "slightly_reduces", (
            f"DEFF==1.05 must land in the ``slightly_reduces`` band per "
            f"REPORTING.md (half-open threshold). Got: {block!r}"
        )
        assert block["is_trivial"] is False, (
            f"DEFF==1.05 must NOT be classified as trivial (half-open "
            f"threshold). Got is_trivial={block['is_trivial']!r}"
        )

        # BR's sample-block ``is_trivial`` must match.
        from types import SimpleNamespace

        from diff_diff import BusinessReport

        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 500
        stub.n_treated = 100
        stub.n_control_units = 400
        stub.event_study_effects = None
        stub.survey_metadata = SimpleNamespace(
            design_effect=1.05,
            effective_n=500.0 / 1.05,
            weight_type="pweight",
            n_strata=None,
            n_psu=None,
            df_survey=None,
            replicate_method=None,
        )
        br = BusinessReport(stub, auto_diagnostics=False)
        sample = br.to_dict()["sample"]
        assert sample["survey"] is not None
        assert sample["survey"]["is_trivial"] is False, (
            f"BR sample-block ``is_trivial`` at DEFF==1.05 must match "
            f"DR's half-open threshold. Got: {sample['survey']!r}"
        )

    def test_deff_just_under_1_05_is_trivial(self):
        """Round-43 P2 regression: the lower-bound adjacent point
        ``deff == 1.049`` is still inside the half-open ``trivial``
        band ``[0.95, 1.05)``.
        """
        block = self._stub_with_deff(1.049)
        assert block["band_label"] == "trivial"
        assert block["is_trivial"] is True


class TestSDiDTROPRejectIncompatiblePrecomputedInputs:
    """Round-21 P1 CI review on PR #318: ``precomputed={"sensitivity":
    ...}`` and ``BusinessReport(honest_did_results=...)`` previously
    short-circuited the SDiD / TROP native-routing guards, letting the
    generic report sections surface methodology-incompatible HonestDiD
    or generic PT diagnostics on estimators that route robustness to
    ``estimator_native_diagnostics``. DR / BR must now reject those
    passthroughs with a clear error pointing users at the native
    diagnostics on the result object.
    """

    @staticmethod
    def _dummy_sens_object():
        from types import SimpleNamespace

        return SimpleNamespace(
            M_values=[0.5, 1.0],
            bounds=[(0.1, 2.0), (-0.2, 2.5)],
            robust_cis=[(0.05, 2.1), (-0.3, 2.6)],
            breakdown_M=0.75,
            method="relative_magnitude",
            original_estimate=1.0,
            original_se=0.2,
            alpha=0.05,
        )

    @staticmethod
    def _dummy_pt_object():
        from types import SimpleNamespace

        return SimpleNamespace(joint_p_value=0.2, n_pre_periods=3, method="event_study")

    def test_dr_rejects_precomputed_sensitivity_on_sdid(self, sdid_fit):
        from diff_diff import DiagnosticReport

        fit, _ = sdid_fit
        with pytest.raises(ValueError, match="estimator_native_diagnostics"):
            DiagnosticReport(fit, precomputed={"sensitivity": self._dummy_sens_object()})

    def test_dr_rejects_precomputed_parallel_trends_on_sdid(self, sdid_fit):
        from diff_diff import DiagnosticReport

        fit, _ = sdid_fit
        with pytest.raises(ValueError, match="estimator_native_diagnostics"):
            DiagnosticReport(fit, precomputed={"parallel_trends": self._dummy_pt_object()})

    def test_br_rejects_honest_did_results_on_sdid(self, sdid_fit):
        fit, _ = sdid_fit
        with pytest.raises(ValueError, match="estimator_native_diagnostics"):
            BusinessReport(fit, honest_did_results=self._dummy_sens_object())

    def test_dr_rejects_precomputed_sensitivity_on_trop(self):
        """TROP construction is expensive; use a stub with the right name."""
        from diff_diff import DiagnosticReport

        class TROPResults:
            pass

        stub = TROPResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.alpha = 0.05
        stub.n_obs = 100
        with pytest.raises(ValueError, match="estimator_native_diagnostics"):
            DiagnosticReport(stub, precomputed={"sensitivity": self._dummy_sens_object()})

    def test_dr_rejects_precomputed_parallel_trends_on_trop(self):
        from diff_diff import DiagnosticReport

        class TROPResults:
            pass

        stub = TROPResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.alpha = 0.05
        stub.n_obs = 100
        with pytest.raises(ValueError, match="estimator_native_diagnostics"):
            DiagnosticReport(stub, precomputed={"parallel_trends": self._dummy_pt_object()})

    def test_dr_still_accepts_precomputed_on_compatible_estimators(self, cs_fit):
        """CS remains a valid passthrough target — the guardrail is
        estimator-specific, not a blanket ban.
        """
        from diff_diff import DiagnosticReport

        fit, _ = cs_fit
        # Should not raise.
        DiagnosticReport(fit, precomputed={"sensitivity": self._dummy_sens_object()})

    def test_br_still_accepts_honest_did_results_on_compatible_estimators(self, cs_fit):
        fit, _ = cs_fit
        # Should not raise.
        BusinessReport(fit, honest_did_results=self._dummy_sens_object())


class TestBRLiftSensitivityPreservesMethodOnSkip:
    """Pre-emptive audit regression: ``_lift_sensitivity`` previously
    dropped the ``method`` field from BR's ``sensitivity`` block when
    ``status != "ran"``. That forced BR-schema consumers to re-consult
    the DR schema to distinguish native-routed skips
    (``method="estimator_native"`` for SDiD / TROP, where robustness is
    covered by the native battery) from methodology-blocked skips (e.g.,
    CS with ``base_period='varying'``). Preserving the field keeps BR
    self-describing.
    """

    def test_sdid_br_schema_exposes_native_method_on_sensitivity_skip(self, sdid_fit):
        fit, _ = sdid_fit
        sens_block = BusinessReport(fit).to_dict()["sensitivity"]
        assert sens_block["status"] == "skipped"
        # The round-20 DR fix set method="estimator_native"; BR must pass
        # it through so an agent consuming BR alone can tell this is a
        # native-routed skip.
        assert sens_block.get("method") == "estimator_native", (
            "BR's sensitivity block must preserve method='estimator_native' "
            "when DR emitted it; otherwise downstream agents cannot "
            f"distinguish native routing from methodology blocks. Got: {sens_block}"
        )


class TestHausmanTestStatisticPopulated:
    """Round-10 P3 regression: ``HausmanPretestResult`` exposes
    ``statistic`` (not ``test_statistic``); the DR schema was previously
    reading the wrong attribute and losing the H statistic."""

    def test_test_statistic_field_is_populated_on_success(self, edid_fit):
        from diff_diff import DiagnosticReport

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
        if pt["status"] == "ran":
            # Method-specific: only Hausman exposes a test_statistic.
            assert pt["method"] == "hausman"
            ts = pt["test_statistic"]
            assert (
                ts is not None and isinstance(ts, float) and np.isfinite(ts)
            ), f"Hausman H statistic must be populated on success; got {ts}"


class TestFullReportSingleM:
    """Regression: ``full_report()`` must not claim full-grid robustness for a
    single-M HonestDiDResults passthrough. The summary path was fixed earlier;
    the structured-markdown path had the same bug and now mirrors it."""

    @staticmethod
    def _fake_single_m(M=1.5, ci_lb=1.0, ci_ub=3.0):
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

    def test_full_report_does_not_claim_full_grid_for_single_m(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, honest_did_results=self._fake_single_m())
        md = br.full_report()
        assert "robust across full grid" not in md
        assert "Single point checked" in md or "single point" in md.lower()


# ---------------------------------------------------------------------------
# Summary + full_report work across estimators
# ---------------------------------------------------------------------------
class TestAcrossEstimators:
    def test_summary_nonempty_for_all(self, did_fit, event_study_fit, cs_fit, sdid_fit):
        for fit, _ in (did_fit, event_study_fit, cs_fit, sdid_fit):
            br = BusinessReport(fit, auto_diagnostics=False)
            s = br.summary()
            assert isinstance(s, str)
            assert len(s) > 0


# ---------------------------------------------------------------------------
# Public API exposure
# ---------------------------------------------------------------------------
def test_public_api_exports():
    for name in ("BusinessReport", "BusinessContext", "BUSINESS_REPORT_SCHEMA_VERSION"):
        assert hasattr(dd, name)


def test_repr_includes_estimator_and_effect(cs_fit):
    fit, _ = cs_fit
    r = repr(BusinessReport(fit, auto_diagnostics=False))
    assert "CallawaySantAnnaResults" in r


def test_str_equals_summary(cs_fit):
    fit, _ = cs_fit
    br = BusinessReport(fit, auto_diagnostics=False)
    assert str(br) == br.summary()


def test_business_context_is_frozen_dataclass():
    ctx = BusinessContext(
        outcome_label="x",
        outcome_unit=None,
        outcome_direction=None,
        business_question=None,
        treatment_label="y",
        alpha=0.05,
    )
    with pytest.raises((AttributeError, Exception)):
        ctx.alpha = 0.10  # type: ignore[misc]


class TestBootstrapResultsAndNBootstrapDetection:
    """Regression for the round-5 P0 finding that ``_extract_headline``
    only preserved native CI surfaces when a result advertised
    ``inference_method`` / ``bootstrap_distribution`` / ``variance_method``
    / ``df_survey``.

    Several staggered / continuous / dCDH result classes copy bootstrap-
    derived se/p/conf_int into their top-level fields at fit time and
    expose the bootstrap only via a ``bootstrap_results`` sub-object or
    an ``n_bootstrap > 0`` attribute. An ``alpha`` override on such a
    fit would silently swap a percentile/multiplier bootstrap CI for a
    normal-approximation one. BR must now detect either marker and
    preserve the fitted CI at its native level.
    """

    def _base_stub(self):
        stub = type("Stub", (), {})()
        stub.att = 1.0
        stub.se = 0.5
        stub.p_value = 0.04
        stub.conf_int = (0.05, 1.95)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        # Crucially NOT exposing inference_method / bootstrap_distribution
        # / variance_method / df_survey: exactly the surface the reviewer
        # flagged as silently falling through.
        return stub

    def test_bootstrap_results_object_alone_preserves_fit_ci(self):
        stub = self._base_stub()
        stub.bootstrap_results = type("BootSub", (), {"n_bootstrap": 199})()
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95, (
            "Result carrying bootstrap_results must preserve fitted CI "
            "level on alpha mismatch; got " + str(h["ci_level"])
        )
        assert h["ci_lower"] == pytest.approx(0.05)
        assert h["ci_upper"] == pytest.approx(1.95)
        topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in topics

    def test_n_bootstrap_positive_alone_preserves_fit_ci(self):
        """ContinuousDiDResults-style: ``n_bootstrap`` field, no bootstrap_results."""
        stub = self._base_stub()
        stub.n_bootstrap = 499
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(0.05)
        assert h["ci_upper"] == pytest.approx(1.95)
        topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in topics

    def test_n_bootstrap_zero_still_preserves_on_alpha_mismatch(self):
        """Analytic fits (``n_bootstrap = 0``) also preserve the fitted CI
        on alpha mismatch — BR cannot reproduce a ``DiDResults`` /
        ``MultiPeriodDiDResults`` / TROP t-quantile CI without the fit's
        finite ``df``, which is not exposed uniformly. Round-7 regression.
        """
        stub = self._base_stub()
        stub.n_bootstrap = 0
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        # Analytic fit's native 95% CI is preserved at 95% on 90% override.
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(0.05)
        assert h["ci_upper"] == pytest.approx(1.95)
        topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in topics

    def test_dcdh_shaped_bootstrap_stub_preserves_fit_ci(self):
        """dCDH copies bootstrap se/p/conf_int into top-level fields without
        ``inference_method``. The reviewer called this out specifically."""

        class ChaisemartinDHaultfoeuilleResults:  # name-keyed dispatch
            pass

        stub = ChaisemartinDHaultfoeuilleResults()
        stub.att = 1.5
        stub.se = 0.4
        stub.p_value = 0.02
        stub.conf_int = (0.72, 2.28)
        stub.alpha = 0.05
        stub.n_obs = 200
        stub.n_treated = 80
        stub.n_control = 120
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.placebo_event_study = None
        # dCDH carries bootstrap via a sub-object; top-level fields are
        # the bootstrap-derived values, not analytic.
        stub.bootstrap_results = type("DCDHBoot", (), {"n_bootstrap": 499})()

        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(0.72)
        assert h["ci_upper"] == pytest.approx(2.28)


class TestAnalyticalFiniteDfAlphaOverride:
    """Round-7 regressions for the P0 finding that
    ``_extract_headline`` was recomputing a normal-z CI on alpha
    mismatch for analytical fits whose native inference used a finite
    ``df`` (``DifferenceInDifferences`` / ``MultiPeriodDiD`` / TROP)
    that BR cannot reproduce from ``(att, se)`` alone. The fix is to
    always preserve the fitted CI on alpha mismatch.
    """

    def test_analytical_did_result_preserves_native_ci(self):
        from diff_diff import DifferenceInDifferences, generate_did_data

        df = generate_did_data(n_units=80, n_periods=4, treatment_effect=1.5, seed=7)
        fit = DifferenceInDifferences().fit(df, outcome="outcome", treatment="treated", post="post")
        native_lo, native_hi = fit.conf_int

        br = BusinessReport(fit, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        # Native 95% CI preserved — no z-based recomputation.
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(native_lo)
        assert h["ci_upper"] == pytest.approx(native_hi)
        topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in topics

    def test_multiperiod_preserves_native_ci_on_alpha_override(self):
        from diff_diff import MultiPeriodDiD, generate_did_data

        df = generate_did_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
        fit = MultiPeriodDiD().fit(
            df,
            outcome="outcome",
            treatment="treated",
            time="period",
            unit="unit",
            reference_period=3,
        )
        native_lo, native_hi = fit.avg_conf_int

        br = BusinessReport(fit, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(native_lo)
        assert h["ci_upper"] == pytest.approx(native_hi)

    def test_undefined_df_survey_stub_does_not_invent_finite_ci(self):
        """When the fit's native inference returned NaN (rank-deficient
        replicate design: ``df_survey = 0``), BR must not recompute a
        finite interval — the NaN signal must propagate through."""
        from types import SimpleNamespace

        class _UndefinedDfStub:
            pass

        stub = _UndefinedDfStub()
        stub.att = 1.0
        stub.se = float("nan")
        stub.p_value = float("nan")
        stub.conf_int = (float("nan"), float("nan"))
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.inference_method = "analytical"
        stub.survey_metadata = SimpleNamespace(
            weight_type="replicate",
            replicate_method="JK1",
            effective_n=80.0,
            design_effect=1.25,
            sum_weights=100.0,
            n_strata=None,
            n_psu=None,
            df_survey=0,
        )

        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        # NaN bounds must propagate — BR must not invent a finite CI.
        lo, hi = h["ci_lower"], h["ci_upper"]
        assert lo is None or not np.isfinite(lo), f"ci_lower should be NaN/None, got {lo}"
        assert hi is None or not np.isfinite(hi), f"ci_upper should be NaN/None, got {hi}"


class TestDCDHAssumptionTransitionBased:
    """Regression for the round-5 P1 finding that
    ``ChaisemartinDHaultfoeuilleResults`` was narrated with generic group-
    time PT text instead of source-backed transition-based identification.
    """

    def test_dcdh_uses_transition_based_language(self):
        class ChaisemartinDHaultfoeuilleResults:
            pass

        obj = ChaisemartinDHaultfoeuilleResults()
        obj.att = 1.0
        obj.se = 0.1
        obj.p_value = 0.001
        obj.conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 40
        obj.n_control = 60
        obj.survey_metadata = None
        obj.event_study_effects = None
        obj.placebo_event_study = None
        obj.inference_method = "analytical"

        br = BusinessReport(obj, auto_diagnostics=False)
        assumption = br.to_dict()["assumption"]
        assert assumption["parallel_trends_variant"] == "transition_based"
        desc = assumption["description"]
        # Source-faithful: joiners/leavers/stable-control, dCDH paper attribution.
        assert "joiner" in desc.lower()
        assert "leaver" in desc.lower()
        assert "Chaisemartin" in desc or "D'Haultfoeuille" in desc
        # Must NOT open with the generic group-time PT framing. The text
        # may reference it inside a contrast clause ("not a single
        # group-time ATT PT"), which is fine and intended.
        assert not desc.startswith("Identification relies on parallel trends")


class TestCSVaryingBaseSensitivitySkipped:
    """Regression for the round-5 P1 finding that DR would narrate HonestDiD
    bounds as robust sensitivity for a CallawaySantAnna fit with
    ``base_period='varying'`` (the CS default). The HonestDiD helper
    explicitly warns that those bounds are not valid for interpretation;
    DR must preemptively skip and surface the reason."""

    def test_cs_varying_base_skips_sensitivity_with_reason(self):
        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.3
        stub.overall_p_value = 0.01
        stub.overall_conf_int = (0.4, 1.6)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.event_study_vcov = None
        stub.event_study_vcov_index = None
        stub.vcov = None
        stub.interaction_indices = None
        stub.base_period = "varying"
        stub.inference_method = "analytical"

        from diff_diff import DiagnosticReport

        dr = DiagnosticReport(stub).run_all()
        sens = dr.schema["sensitivity"]
        assert sens["status"] == "skipped"
        reason = sens["reason"]
        assert "base_period" in reason and "universal" in reason
        # And BR must surface this as a warning-severity caveat.
        br = BusinessReport(stub, diagnostics=dr)
        caveats = br.caveats()
        topics = {c.get("topic") for c in caveats}
        assert "sensitivity_skipped" in topics, (
            "BR must surface varying-base sensitivity skip as a caveat; " f"got topics {topics}"
        )


class TestCSVaryingBaseSensitivityRejectsPrecomputed:
    """Round-44 P1 CI review on PR #318: ``precomputed['sensitivity']``
    (and BR's ``honest_did_results`` shorthand) previously bypassed the
    varying-base CS guard — the applicability gate's precomputed early-
    return unlocked the sensitivity section and BR/DR narrated the
    Rambachan-Roth bounds as ordinary robustness on a displayed fit
    whose consecutive-comparison pre-period surface has a different
    interpretation than the bounds' provenance (REGISTRY.md
    §CallawaySantAnna line 410, §HonestDiD line 2458). DR and BR must
    reject at construction."""

    def _cs_varying_base_stub(self):
        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.3
        stub.overall_p_value = 0.01
        stub.overall_conf_int = (0.4, 1.6)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.event_study_vcov = None
        stub.event_study_vcov_index = None
        stub.vcov = None
        stub.interaction_indices = None
        stub.base_period = "varying"
        stub.inference_method = "analytical"
        return stub

    def _dummy_sens_object(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            M_values=[0.5, 1.0],
            bounds=[(0.1, 2.0), (-0.2, 2.5)],
            robust_cis=[(0.05, 2.1), (-0.3, 2.6)],
            breakdown_M=0.75,
            method="relative_magnitude",
            original_estimate=1.0,
            original_se=0.2,
            alpha=0.05,
        )

    def test_dr_rejects_precomputed_sensitivity_on_varying_base_cs(self):
        from diff_diff import DiagnosticReport

        stub = self._cs_varying_base_stub()
        with pytest.raises(ValueError, match="base_period='universal'"):
            DiagnosticReport(stub, precomputed={"sensitivity": self._dummy_sens_object()})

    def test_dr_allows_precomputed_sensitivity_on_universal_base_cs(self):
        """Universal-base CS + precomputed sensitivity is the supported
        path — must not be rejected."""
        from diff_diff import DiagnosticReport

        stub = self._cs_varying_base_stub()
        stub.base_period = "universal"
        # Should not raise.
        DiagnosticReport(stub, precomputed={"sensitivity": self._dummy_sens_object()})

    def test_br_rejects_precomputed_sensitivity_on_varying_base_cs(self):
        stub = self._cs_varying_base_stub()
        with pytest.raises(ValueError, match="base_period='varying'"):
            BusinessReport(
                stub,
                precomputed={"sensitivity": self._dummy_sens_object()},
            )

    def test_br_rejects_honest_did_results_on_varying_base_cs(self):
        """BR's ``honest_did_results`` shorthand must hit the same
        rejection — it becomes ``precomputed['sensitivity']`` under the
        hood, and the methodology problem is identical."""
        stub = self._cs_varying_base_stub()
        with pytest.raises(ValueError, match="honest_did_results"):
            BusinessReport(
                stub,
                honest_did_results=self._dummy_sens_object(),
            )

    def test_br_rejects_both_passthrough_inputs_names_them(self):
        """When both passthrough inputs are supplied, the error must
        name both so the user knows every input that was rejected."""
        stub = self._cs_varying_base_stub()
        with pytest.raises(ValueError) as excinfo:
            BusinessReport(
                stub,
                honest_did_results=self._dummy_sens_object(),
                precomputed={"sensitivity": self._dummy_sens_object()},
            )
        msg = str(excinfo.value)
        assert "honest_did_results" in msg
        assert "precomputed['sensitivity']" in msg


class TestCanonicalValidationSurfaceFixes:
    """Regression coverage for issues surfaced by the first round of
    BR/DR canonical-dataset validation (``docs/validation/
    validate_br_dr_canonical.py``). Each test pins a wording bug
    observed on a real published-applied-work fit.
    """

    def _cs_like_stub_with_zero_breakdown(self):
        """CS-style result stub matching the Cheng-Hoekstra Castle
        Doctrine fit pattern."""

        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        stub.overall_att = 0.5608
        stub.overall_se = 0.1216
        stub.overall_p_value = 0.0
        stub.overall_conf_int = (0.323, 0.799)
        stub.alpha = 0.05
        stub.n_obs = 539
        stub.n_treated = 22
        stub.n_control_units = 27
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.base_period = "universal"
        stub.inference_method = "analytical"
        return stub

    def _fragile_dr_schema(self, breakdown_m: float, grid=None):
        """Build a fake DiagnosticReportResults whose ``sensitivity``
        block carries the given ``breakdown_M`` value and grid. Pass
        ``grid`` as a list of ``{"M": float, "robust_to_zero": bool}``
        dicts (other fields populated with plausible values).
        """
        from diff_diff.diagnostic_report import DiagnosticReportResults

        grid = grid if grid is not None else []
        # Populate optional CI / bound fields so grid entries match
        # the schema the BR/DR runners actually emit.
        grid_full = [
            {
                "M": row["M"],
                "ci_lower": row.get("ci_lower", 0.0),
                "ci_upper": row.get("ci_upper", 0.0),
                "bound_lower": row.get("bound_lower", 0.0),
                "bound_upper": row.get("bound_upper", 0.0),
                "robust_to_zero": row["robust_to_zero"],
            }
            for row in grid
        ]
        schema = {
            "schema_version": "2.0",
            "estimator": {"class_name": "CallawaySantAnnaResults", "display_name": "CS"},
            "headline_metric": {},
            "parallel_trends": {"status": "skipped", "reason": "stub"},
            "pretrends_power": {"status": "skipped", "reason": "stub"},
            "sensitivity": {
                "status": "ran",
                "method": "relative_magnitude",
                "breakdown_M": breakdown_m,
                "conclusion": "fragile",
                "grid": grid_full,
            },
            "placebo": {"status": "skipped", "reason": "stub"},
            "bacon": {"status": "skipped", "reason": "stub"},
            "design_effect": {"status": "skipped", "reason": "stub"},
            "heterogeneity": {"status": "skipped", "reason": "stub"},
            "epv": {"status": "skipped", "reason": "stub"},
            "estimator_native_diagnostics": {"status": "not_applicable"},
            "skipped": {},
            "warnings": [],
            "overall_interpretation": "",
            "next_steps": [],
        }
        return DiagnosticReportResults(
            schema=schema,
            interpretation="",
            applicable_checks=("sensitivity",),
            skipped_checks={},
            warnings=(),
        )

    def test_treatment_label_preserves_embedded_abbreviations(self):
        """Card-Krueger use case: ``treatment_label="the NJ minimum-wage
        increase"`` previously rendered as ``"The nj minimum-wage
        increase"`` because ``str.capitalize()`` lowercases every
        character after the first. The fix preserves user-supplied
        casing and only uppercases the first character.
        """

        class DiDResults:
            pass

        stub = DiDResults()
        stub.att = 1.47
        stub.se = 1.93
        stub.t_stat = 0.76
        stub.p_value = 0.45
        stub.conf_int = (-2.32, 5.27)
        stub.alpha = 0.05
        stub.n_obs = 620
        stub.n_treated = 462
        stub.n_control = 158
        stub.survey_metadata = None
        stub.inference_method = "analytical"
        br = BusinessReport(
            stub,
            outcome_label="FTE employment",
            treatment_label="the NJ minimum-wage increase",
            auto_diagnostics=False,
        )
        headline = br.headline()
        assert "The NJ minimum-wage increase" in headline, (
            "Embedded ``NJ`` abbreviation must survive the first-word "
            f"capitalization. Got headline: {headline!r}"
        )
        assert "The nj" not in headline, (
            "Previous capitalize() bug lowercased the NJ abbreviation. " f"Got: {headline!r}"
        )

    def test_treatment_label_preserves_proper_noun_case(self):
        """Castle Doctrine use case: ``treatment_label="Castle Doctrine
        law adoption"`` previously rendered as ``"Castle doctrine law
        adoption"`` because capitalize() lowercased the rest. Must
        preserve proper-noun casing.
        """
        stub = self._cs_like_stub_with_zero_breakdown()
        br = BusinessReport(
            stub,
            outcome_label="Homicide rate",
            treatment_label="Castle Doctrine law adoption",
            auto_diagnostics=False,
        )
        headline = br.headline()
        assert (
            "Castle Doctrine law adoption" in headline
        ), f"Proper-noun casing must be preserved. Got: {headline!r}"

    def test_smallest_grid_m_fails_uses_smallest_grid_point_wording(self):
        """When the smallest M actually evaluated on the grid has
        ``robust_to_zero == False``, the "smallest M evaluated" wording
        is semantically correct. This is the Cheng-Hoekstra Castle
        Doctrine pattern: default grid ``[0.5, 1.0, 1.5, 2.0]`` with
        M=0.5 already non-robust.
        """
        stub = self._cs_like_stub_with_zero_breakdown()
        dr = self._fragile_dr_schema(
            breakdown_m=0.0,
            grid=[
                {"M": 0.5, "robust_to_zero": False},
                {"M": 1.0, "robust_to_zero": False},
                {"M": 1.5, "robust_to_zero": False},
                {"M": 2.0, "robust_to_zero": False},
            ],
        )
        br = BusinessReport(stub, diagnostics=dr)
        summary = br.summary()
        # Must not render the degenerate multiplier form on the
        # zero-breakdown case.
        assert (
            "0x the pre-period variation" not in summary
        ), f"Summary must not quote ``0x`` multiplier. Got: {summary!r}"
        # New wording quotes the actual smallest evaluated M.
        assert "smallest M evaluated on the sensitivity grid" in summary, (
            f"Summary must use the smallest-M-evaluated wording when the "
            f"smallest grid point actually fails. Got: {summary!r}"
        )
        assert "M = 0.5" in summary

    def test_smallest_grid_m_robust_falls_through_to_multiplier_wording(self):
        """CI review on PR #341 R1: ``breakdown_M`` is the interpolated
        threshold between grid points, not a claim about any specific
        grid point. On a grid starting at M=0 where the smallest
        evaluated point is still robust, a small ``breakdown_M=0.03``
        does NOT mean the smallest grid point failed — it means
        fragility emerges between grid points. The correct wording is
        the numeric multiplier, not the smallest-grid-point claim.
        """
        stub = self._cs_like_stub_with_zero_breakdown()
        dr = self._fragile_dr_schema(
            breakdown_m=0.03,
            grid=[
                # Smallest evaluated M (0) is still robust: CI excludes
                # zero. Breakdown is interpolated somewhere between M=0
                # and M=0.25.
                {"M": 0.0, "robust_to_zero": True},
                {"M": 0.25, "robust_to_zero": False},
                {"M": 0.5, "robust_to_zero": False},
            ],
        )
        br = BusinessReport(stub, diagnostics=dr)
        summary = br.summary()
        # Must NOT claim the smallest grid point failed — it didn't.
        assert "smallest M evaluated on the sensitivity grid" not in summary, (
            f"Summary must not assert ``smallest M evaluated fails`` when the "
            f"smallest grid point is still robust. Got: {summary!r}"
        )
        # Correct wording quotes the numeric multiplier.
        assert "0.03x" in summary, (
            f"Fragile fit with robust smallest-M should quote the interpolated "
            f"breakdown multiplier. Got: {summary!r}"
        )

    def test_breakdown_m_normal_keeps_multiplier_wording(self):
        """Breakdown values at the usual fragile-but-nonzero range
        (e.g., 0.3) must still quote the ``0.3x`` multiplier — the
        smallest-M-evaluated wording is only for grids whose smallest
        actually-evaluated point is already non-robust.
        """
        stub = self._cs_like_stub_with_zero_breakdown()
        dr = self._fragile_dr_schema(breakdown_m=0.3)
        br = BusinessReport(stub, diagnostics=dr)
        summary = br.summary()
        assert "0.3x" in summary
        assert "smallest M evaluated on the sensitivity grid" not in summary


class TestBaconCaveatEstimatorAware:
    """Round-45 P1 CI review on PR #318: Goodman-Bacon decomposes TWFE
    weights. On fits already produced by a heterogeneity-robust
    estimator (CS / SA / BJS / Gardner / Wooldridge / EfficientDiD /
    Stacked / dCDH / TripleDifference / StaggeredTripleDiff / SDiD /
    TROP), a high forbidden-weight share says "TWFE would have been
    materially biased on this rollout", not "the displayed estimator
    needs to be replaced" — the displayed fit is already robust.

    BR's caveat must be estimator-aware: keep the "switch to a robust
    estimator" recommendation for TWFE-style fits only.
    """

    @staticmethod
    def _bacon_schema_with_high_forbidden_weight():
        """Build a fake ``DiagnosticReportResults`` whose schema carries
        a Bacon block with ``forbidden_weight > 0.10`` so BR's caveat
        builder fires on the Bacon branch."""
        from diff_diff.diagnostic_report import DiagnosticReportResults

        schema = {
            "schema_version": "2.0",
            "estimator": {"class_name": "Stub", "display_name": "Stub"},
            "headline_metric": {},
            "parallel_trends": {"status": "skipped", "reason": "stub"},
            "pretrends_power": {"status": "skipped", "reason": "stub"},
            "sensitivity": {"status": "skipped", "reason": "stub"},
            "placebo": {"status": "skipped", "reason": "stub"},
            "bacon": {
                "status": "ran",
                "twfe_estimate": 1.2,
                "weight_by_type": {
                    "treated_vs_never": 0.5,
                    "earlier_vs_later": 0.1,
                    "later_vs_earlier": 0.4,
                },
                "forbidden_weight": 0.4,  # > 0.10 threshold
                "verdict": "materially_contaminated",
                "n_timing_groups": 3,
            },
            "design_effect": {"status": "skipped", "reason": "stub"},
            "heterogeneity": {"status": "skipped", "reason": "stub"},
            "epv": {"status": "skipped", "reason": "stub"},
            "estimator_native_diagnostics": {"status": "not_applicable"},
            "skipped": {},
            "warnings": [],
            "overall_interpretation": "",
            "next_steps": [],
        }
        return DiagnosticReportResults(
            schema=schema,
            interpretation="",
            applicable_checks=("bacon",),
            skipped_checks={},
            warnings=(),
        )

    @staticmethod
    def _make_cs_like_stub(class_name: str):
        cls = type(class_name, (), {})
        obj = cls()
        obj.overall_att = 1.0
        obj.overall_se = 0.2
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.6, 1.4)
        obj.alpha = 0.05
        obj.n_obs = 500
        obj.n_treated = 100
        obj.n_control_units = 400
        obj.survey_metadata = None
        obj.event_study_effects = None
        obj.inference_method = "analytical"
        return obj

    @staticmethod
    def _make_twfe_style_stub():
        class MultiPeriodDiDResults:
            pass

        obj = MultiPeriodDiDResults()
        obj.avg_att = 1.0
        obj.avg_se = 0.2
        obj.avg_p_value = 0.001
        obj.avg_conf_int = (0.6, 1.4)
        obj.alpha = 0.05
        obj.n_obs = 500
        obj.n_treated = 100
        obj.n_control = 400
        obj.survey_metadata = None
        obj.pre_period_effects = None
        obj.inference_method = "analytical"
        return obj

    def test_cs_like_fit_does_not_recommend_switching_estimators(self):
        """On an already-robust CS-style fit with high forbidden
        Bacon weight, BR must not recommend switching to a robust
        estimator — the displayed fit IS already robust."""
        stub = self._make_cs_like_stub("CallawaySantAnnaResults")
        dr = self._bacon_schema_with_high_forbidden_weight()
        br = BusinessReport(stub, diagnostics=dr)
        caveats = br.caveats()
        bacon_caveats = [c for c in caveats if c.get("topic") == "bacon_contamination"]
        assert len(bacon_caveats) == 1, (
            f"High forbidden-weight Bacon must surface a caveat. " f"Got caveats: {caveats!r}"
        )
        msg = bacon_caveats[0]["message"].lower()
        # Must NOT tell the user to switch estimators.
        assert "re-estimate with a heterogeneity-robust" not in msg, (
            f"CS is already heterogeneity-robust; must not recommend "
            f"switching. Got message: {msg!r}"
        )
        # Must frame this as a TWFE benchmark problem / rollout-design
        # statement, not a displayed-fit-validity problem.
        assert (
            "heterogeneity-robust" in msg
            or "already" in msg
            or "twfe benchmark" in msg
            or "rollout design" in msg
        ), f"CS Bacon caveat must reframe as rollout/TWFE issue. Got: {msg!r}"
        # And the full-report rendering must reflect the softer wording.
        md = br.full_report()
        assert "Re-estimate with a heterogeneity-robust estimator" not in md, md

    def test_other_robust_estimators_also_avoid_switch_recommendation(self):
        """Spot-check the same rule holds for multiple
        heterogeneity-robust estimators on the Bacon path."""
        for class_name in (
            "SunAbrahamResults",
            "ImputationDiDResults",
            "TwoStageDiDResults",
            "StackedDiDResults",
            "WooldridgeDiDResults",
            "ChaisemartinDHaultfoeuilleResults",
            "EfficientDiDResults",
        ):
            stub = self._make_cs_like_stub(class_name)
            dr = self._bacon_schema_with_high_forbidden_weight()
            br = BusinessReport(stub, diagnostics=dr)
            msgs = [c["message"] for c in br.caveats() if c.get("topic") == "bacon_contamination"]
            assert msgs, f"{class_name}: Bacon caveat must fire"
            assert (
                "Re-estimate with a heterogeneity-robust estimator" not in msgs[0]
            ), f"{class_name} is already robust; must not recommend switching. Got: {msgs[0]!r}"

    def test_twfe_style_fit_keeps_switch_recommendation(self):
        """The switch-to-robust recommendation is load-bearing for
        genuinely TWFE-style fits and must be preserved there."""
        stub = self._make_twfe_style_stub()
        dr = self._bacon_schema_with_high_forbidden_weight()
        br = BusinessReport(stub, diagnostics=dr)
        caveats = br.caveats()
        bacon_caveats = [c for c in caveats if c.get("topic") == "bacon_contamination"]
        assert len(bacon_caveats) == 1
        msg = bacon_caveats[0]["message"]
        # TWFE-style fit: keep the explicit switch recommendation.
        assert "Re-estimate with a heterogeneity-robust estimator" in msg, (
            f"MultiPeriodDiDResults (TWFE event-study) must keep the "
            f"switch-to-robust recommendation. Got: {msg!r}"
        )


class TestBusinessReportSurveyDesignPassthrough:
    """Round-40 P1 CI review on PR #318: ``BusinessReport`` must accept
    ``survey_design`` and forward it to the auto-constructed
    ``DiagnosticReport``, so Bacon replay on survey-backed fits is
    fit-faithful and the simple 2x2 PT path skips with an explicit
    reason rather than reporting an unweighted verdict for a weighted
    estimate."""

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

    def test_survey_backed_did_br_rolls_up_pt_skip(self):
        """BR's auto-constructed DR must skip the 2x2 PT helper on a
        survey-backed DiDResults. BR's schema then surfaces the
        skipped PT block with the survey-design reason (no unweighted
        verdict leaks into the narrative)."""
        import pandas as pd

        panel = pd.DataFrame(
            {
                "outcome": [1.0, 2.0, 1.1, 2.2],
                "post": [0, 1, 0, 1],
                "treated": [0, 0, 1, 1],
            }
        )
        obj = self._did_with_survey()
        br = BusinessReport(
            obj,
            outcome_label="Revenue",
            outcome_unit="$",
            data=panel,
            outcome="outcome",
            time="post",
            treatment="treated",
        )
        schema = br.to_dict()
        diag = schema.get("diagnostics", {})
        dr_schema = diag.get("schema", {}) if isinstance(diag, dict) else {}
        pt_block = dr_schema.get("parallel_trends", {}) if isinstance(dr_schema, dict) else {}
        # Round-40 schema: parallel_trends skipped with a survey-design
        # reason rather than emitting an unweighted verdict. BR's auto
        # path must honor the skip.
        assert pt_block.get("status") == "skipped"
        reason = (pt_block.get("reason") or "").lower()
        assert "survey design" in reason

    def test_survey_backed_staggered_br_forwards_survey_design_to_bacon(self):
        """BR must forward ``survey_design`` to the auto-constructed
        DR, which in turn threads it to ``bacon_decompose``. Verify via
        ``unittest.mock.patch`` that the kwarg reaches the decomposer.
        """
        from unittest.mock import MagicMock, patch

        import pandas as pd

        panel = pd.DataFrame(
            {
                "outcome": [1.0, 2.0, 1.1, 2.2, 1.2, 2.3, 1.3, 2.4],
                "unit": [1, 1, 2, 2, 3, 3, 4, 4],
                "period": [1, 2, 1, 2, 1, 2, 1, 2],
                "first_treat": [0, 0, 0, 0, 2, 2, 2, 2],
            }
        )
        obj = self._staggered_stub_with_survey()
        sentinel_design = object()
        fake_decomp = MagicMock()
        fake_decomp.total_weight_treated_vs_never = 0.9
        fake_decomp.total_weight_earlier_vs_later = 0.05
        fake_decomp.total_weight_later_vs_earlier = 0.05
        fake_decomp.twfe_estimate = 1.1
        fake_decomp.n_timing_groups = 2
        with patch("diff_diff.bacon.bacon_decompose", return_value=fake_decomp) as m:
            br = BusinessReport(
                obj,
                data=panel,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                survey_design=sentinel_design,
            )
            br.to_dict()  # trigger DR build
            assert m.called, "bacon_decompose was not called"
            _, kwargs = m.call_args
            assert kwargs.get("survey_design") is sentinel_design

    def test_survey_backed_staggered_br_skips_bacon_without_survey_design(self):
        """Without ``survey_design``, BR's DR must skip Bacon with the
        survey-design reason (fit-faithful replay requires it)."""
        import pandas as pd

        panel = pd.DataFrame(
            {
                "outcome": [1.0, 2.0, 1.1, 2.2, 1.2, 2.3, 1.3, 2.4],
                "unit": [1, 1, 2, 2, 3, 3, 4, 4],
                "period": [1, 2, 1, 2, 1, 2, 1, 2],
                "first_treat": [0, 0, 0, 0, 2, 2, 2, 2],
            }
        )
        obj = self._staggered_stub_with_survey()
        br = BusinessReport(
            obj,
            data=panel,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            # survey_design intentionally omitted
        )
        schema = br.to_dict()
        diag = schema.get("diagnostics", {})
        dr_schema = diag.get("schema", {}) if isinstance(diag, dict) else {}
        bacon_block = dr_schema.get("bacon", {}) if isinstance(dr_schema, dict) else {}
        assert bacon_block.get("status") == "skipped"
        reason = (bacon_block.get("reason") or "").lower()
        assert "survey design" in reason


class TestBusinessReportPrecomputedPassthrough:
    """Round-43 P2 CI review on PR #318: ``BusinessReport`` must accept
    the ``precomputed=`` dict that its docs advertise and forward every
    key to the auto-constructed ``DiagnosticReport``. DR owns key
    validation (rejects unsupported keys and estimator-incompatible
    entries)."""

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

    def test_br_accepts_precomputed_parallel_trends_on_survey_did(self):
        """The BR docs advertise
        ``precomputed={'parallel_trends': ...}`` as the opt-in for
        survey-backed 2x2 PT. That contract must actually be
        reachable from BR's constructor, not just DR's."""
        obj = self._did_with_survey()
        precomputed_pt = {
            "p_value": 0.42,
            "treated_trend": 0.05,
            "control_trend": 0.04,
            "trend_difference": 0.01,
            "t_statistic": 0.8,
        }
        br = BusinessReport(
            obj,
            outcome_label="Revenue",
            outcome_unit="$",
            precomputed={"parallel_trends": precomputed_pt},
        )
        schema = br.to_dict()
        dr_schema = schema["diagnostics"]["schema"]
        pt = dr_schema["parallel_trends"]
        assert pt["status"] == "ran", (
            "BR precomputed PT passthrough must unlock the otherwise-"
            f"skipped survey-backed 2x2 path. Got PT block: {pt!r}"
        )
        assert pt["joint_p_value"] == pytest.approx(0.42)

    def test_br_rejects_unsupported_precomputed_key(self):
        """DR validates the precomputed key set; BR must raise the same
        error rather than silently dropping unsupported keys."""
        obj = self._did_with_survey()
        with pytest.raises(ValueError, match="precomputed="):
            BusinessReport(obj, precomputed={"unknown_check": object()})

    def test_br_explicit_precomputed_sensitivity_wins_over_honest_did_results(self):
        """When both ``honest_did_results`` (shorthand) and explicit
        ``precomputed['sensitivity']`` are supplied, the explicit
        passthrough wins. Documented contract: honest_did_results is a
        convenience that only kicks in when no explicit sensitivity is
        present."""
        from types import SimpleNamespace

        class MultiPeriodDiDResults:
            pass

        obj = MultiPeriodDiDResults()
        obj.avg_att = 1.0
        obj.avg_se = 0.1
        obj.avg_p_value = 0.001
        obj.avg_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 40
        obj.n_control = 60
        obj.survey_metadata = None
        obj.pre_period_effects = {}
        obj.vcov = None
        obj.interaction_indices = None
        obj.event_study_vcov = None
        obj.event_study_vcov_index = None

        explicit_sens = SimpleNamespace(
            M_values=[0.5, 1.0],
            bounds=[(0.1, 2.0), (-0.2, 2.5)],
            robust_cis=[(0.05, 2.1), (-0.3, 2.6)],
            breakdown_M=0.87,  # sentinel — the explicit one's breakdown
            method="relative_magnitude",
            original_estimate=1.0,
            original_se=0.1,
            alpha=0.05,
        )
        shorthand_sens = SimpleNamespace(
            M_values=[0.5, 1.0],
            bounds=[(0.1, 2.0), (-0.2, 2.5)],
            robust_cis=[(0.05, 2.1), (-0.3, 2.6)],
            breakdown_M=0.33,  # different sentinel
            method="relative_magnitude",
            original_estimate=1.0,
            original_se=0.1,
            alpha=0.05,
        )
        br = BusinessReport(
            obj,
            honest_did_results=shorthand_sens,
            precomputed={"sensitivity": explicit_sens},
        )
        dr_schema = br.to_dict()["diagnostics"]["schema"]
        sens = dr_schema["sensitivity"]
        assert sens.get("breakdown_M") == pytest.approx(0.87), (
            "Explicit precomputed['sensitivity'] must win over "
            "honest_did_results shorthand. Got sens block: " + repr(sens)
        )


class TestSyntheticDiDBootstrapInferenceLabel:
    """Cross-surface guard: ``variance_method='bootstrap'`` must be
    recognized by the BR inference-label allow-list at
    ``business_report.py:602``.

    Without the allow-list, BR's alpha-override fallback would emit
    ``"analytical (native degrees of freedom)"`` as the inference label
    for SDID bootstrap fits — a user-visible regression that silently
    mis-describes the fit's variance method.
    """

    def test_bootstrap_fit_uses_bootstrap_label_on_alpha_override(self):
        import warnings as _warnings

        from diff_diff import SyntheticDiD, generate_factor_data
        from diff_diff.business_report import BusinessReport

        fdf = generate_factor_data(n_units=25, n_pre=8, n_post=4, n_treated=4, seed=11)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            fit = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=1).fit(
                fdf,
                outcome="outcome",
                unit="unit",
                time="period",
                treatment="treat",
            )
        assert fit.variance_method == "bootstrap"

        # Alpha mismatch (fit at 0.05, BR requesting 0.10) activates the
        # inference-label branch at business_report.py:593-638.
        br = BusinessReport(fit, alpha=0.10, auto_diagnostics=False)
        caveats = br.caveats()
        override_caveats = [c for c in caveats if c.get("topic") == "alpha_override_preserved"]
        assert (
            len(override_caveats) == 1
        ), f"expected one alpha_override_preserved caveat, got {caveats}"
        message = override_caveats[0].get("message", "")
        assert "bootstrap variance" in message, (
            f"expected 'bootstrap variance' in caveat message, " f"got: {message!r}"
        )
        assert "analytical (native degrees of freedom)" not in message, (
            f"allow-list regression: bootstrap fits must not fall through to the "
            f"analytical label. caveat message: {message!r}"
        )
