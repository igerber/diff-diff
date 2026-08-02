"""Tests for the BR/DR ``target_parameter`` block (gap #6).

Covers:

- Per-estimator dispatch in ``describe_target_parameter`` emits the
  expected ``aggregation`` tag and non-empty prose for each of the 16
  result classes in ``_APPLICABILITY``.
- Fit-time config reads are honored:
  - ``EfficientDiDResults.pt_assumption`` branches the tag between
    ``pt_all_combined`` and ``pt_post_single_baseline``.
  - ``StackedDiDResults.clean_control`` varies the ``definition``
    clause (never_treated vs strict vs not_yet_treated).
  - ``ChaisemartinDHaultfoeuilleResults.L_max`` + ``covariate_residuals``
    + ``linear_trends_effects`` branches the dCDH estimand tag.
- BR and DR emit identical ``target_parameter`` blocks (cross-surface
  parity).
- Exhaustiveness: every result-class name in ``_APPLICABILITY`` gets
  a non-default, non-empty target-parameter block.
- Prose renders the target-parameter sentence in BR's summary + full
  report and DR's overall_interpretation + full report.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from diff_diff._reporting_helpers import describe_target_parameter
from diff_diff.diagnostic_report import _APPLICABILITY


def _minimal_result(class_name: str, **attrs):
    """Build a minimal stub result object with the given class name."""
    cls = type(class_name, (), {})
    obj = cls()
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


class TestTargetParameterPerEstimator:
    """Dispatch table lookup for each of the 16 result classes."""

    def test_did_results(self):
        tp = describe_target_parameter(_minimal_result("DiDResults"))
        assert tp["aggregation"] == "did_or_twfe"
        assert tp["headline_attribute"] == "att"
        assert "ATT" in tp["name"]

    def test_multi_period_did_results(self):
        tp = describe_target_parameter(_minimal_result("MultiPeriodDiDResults"))
        assert tp["aggregation"] == "event_study"
        assert tp["headline_attribute"] == "avg_att"
        assert "event-study" in tp["name"].lower()

    def test_did_results_mentions_twfe(self):
        """``TwoWayFixedEffects.fit()`` returns ``DiDResults`` (verified in
        PR #347 R1 review), so the DiDResults branch must cover both
        the 2x2 DiD and TWFE interpretations. There is no separate
        ``TwoWayFixedEffectsResults`` class.
        """
        tp = describe_target_parameter(_minimal_result("DiDResults"))
        assert tp["aggregation"] == "did_or_twfe"
        assert "TWFE" in tp["name"] or "TWFE" in tp["definition"]

    def test_callaway_santanna(self):
        tp = describe_target_parameter(_minimal_result("CallawaySantAnnaResults"))
        assert tp["aggregation"] == "simple"
        assert tp["headline_attribute"] == "overall_att"
        assert "ATT(g,t)" in tp["name"] or "ATT(g, t)" in tp["definition"]

    def test_sun_abraham(self):
        tp = describe_target_parameter(_minimal_result("SunAbrahamResults"))
        assert tp["aggregation"] == "iw"
        assert "interaction-weighted" in tp["name"].lower() or "IW" in tp["name"]

    def test_imputation(self):
        tp = describe_target_parameter(_minimal_result("ImputationDiDResults"))
        assert tp["aggregation"] == "simple"
        assert tp["headline_attribute"] == "overall_att"
        assert (
            "imputation" in tp["name"].lower()
            or "BJS" in tp["reference"]
            or "Borusyak" in tp["reference"]
        )

    def test_two_stage(self):
        tp = describe_target_parameter(_minimal_result("TwoStageDiDResults"))
        assert tp["aggregation"] == "simple"
        assert tp["headline_attribute"] == "overall_att"
        assert "Gardner" in tp["reference"] or "two-stage" in tp["name"].lower()

    def test_stacked(self):
        tp = describe_target_parameter(
            _minimal_result("StackedDiDResults", clean_control="not_yet_treated")
        )
        assert tp["aggregation"] == "stacked"
        assert tp["headline_attribute"] == "overall_att"
        assert "sub-experiment" in tp["definition"].lower()

    def test_wooldridge_ols(self):
        """PR #347 R4 P2: OLS Wooldridge ETWFE must not be labeled with
        ASF wording. The OLS path aggregates ATT(g,t) coefficients with
        observation-count weights; the ASF path is for nonlinear links.
        """
        tp = describe_target_parameter(_minimal_result("WooldridgeDiDResults", method="ols"))
        assert tp["aggregation"] == "simple"
        assert tp["headline_attribute"] == "overall_att"
        # OLS wording: mentions ATT(g,t) aggregation, not ASF.
        assert (
            "ATT(g,t)" in tp["name"] or "ATT(g,t)" in tp["definition"] or "OLS ETWFE" in tp["name"]
        )
        assert "ASF" not in tp["name"]

    def test_wooldridge_nonlinear(self):
        """Nonlinear (logit/Poisson) Wooldridge ETWFE uses the ASF-based
        ATT path — different wording, different REGISTRY reference.
        """
        for method in ("logit", "poisson"):
            tp = describe_target_parameter(_minimal_result("WooldridgeDiDResults", method=method))
            assert tp["aggregation"] == "simple"
            assert tp["headline_attribute"] == "overall_att"
            assert "ASF" in tp["name"]
            assert method in tp["name"] or method in tp["definition"]

    def test_efficient_did_pt_all(self):
        tp = describe_target_parameter(_minimal_result("EfficientDiDResults", pt_assumption="all"))
        assert tp["aggregation"] == "pt_all_combined"
        assert "PT-All" in tp["name"]
        # PR #347 R7 P1 regression: the definition must disambiguate
        # the library's cohort-size-weighted ``overall_att`` from the
        # paper's uniform-event-time ``ES_avg``.
        defn = tp["definition"]
        assert "cohort-size-weighted" in defn
        assert "ES_avg" in defn
        assert "post-treatment" in defn.lower()

    def test_efficient_did_pt_post(self):
        tp = describe_target_parameter(_minimal_result("EfficientDiDResults", pt_assumption="post"))
        assert tp["aggregation"] == "pt_post_single_baseline"
        assert "PT-Post" in tp["name"]
        defn = tp["definition"]
        assert "cohort-size-weighted" in defn
        assert "ES_avg" in defn

    def test_continuous_did(self):
        tp = describe_target_parameter(_minimal_result("ContinuousDiDResults"))
        assert tp["aggregation"] == "dose_overall"
        # Definition must name both PT and SPT readings per plan-review CRITICAL #2.
        defn = tp["definition"]
        assert "PT" in defn and "SPT" in defn

    def test_triple_difference(self):
        tp = describe_target_parameter(_minimal_result("TripleDifferenceResults"))
        assert tp["aggregation"] == "ddd"
        assert tp["headline_attribute"] == "att"

    def test_staggered_triple_difference(self):
        tp = describe_target_parameter(_minimal_result("StaggeredTripleDiffResults"))
        assert tp["aggregation"] == "staggered_ddd"
        assert tp["headline_attribute"] == "overall_att"

    def test_dcdh_m(self):
        tp = describe_target_parameter(
            _minimal_result("ChaisemartinDHaultfoeuilleResults", L_max=None)
        )
        assert tp["aggregation"] == "M"
        assert "DID_M" in tp["name"]
        # R1 PR #347 review: headline lives in ``overall_att``, not
        # ``att``. Machine-readable field must match the raw attribute
        # downstream consumers should read.
        assert tp["headline_attribute"] == "overall_att"

    def test_dcdh_did_1(self):
        """``L_max = 1`` → headline is ``DID_1`` (per-group single-
        horizon estimand, Equation 3 of the dCDH dynamic paper), NOT
        the generic ``DID_l``. R2 PR #347 review P1 regression:
        previously every ``L_max >= 1`` was flattened to ``DID_l``.
        """
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=1,
                covariate_residuals=None,
                linear_trends_effects=None,
            )
        )
        assert tp["aggregation"] == "DID_1"
        assert "DID_1" in tp["name"]
        assert tp["headline_attribute"] == "overall_att"

    def test_dcdh_delta(self):
        """``L_max >= 2`` (no trends) → headline is the cost-benefit
        ``delta`` aggregate (Lemma 4), NOT ``DID_l``. Mirrors
        ``chaisemartin_dhaultfoeuille.py:2602-2634``.
        """
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=2,
                covariate_residuals=None,
                linear_trends_effects=None,
            )
        )
        assert tp["aggregation"] == "delta"
        assert "delta" in tp["name"]
        assert tp["headline_attribute"] == "overall_att"

    def test_dcdh_did_1_with_controls(self):
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=1,
                covariate_residuals=SimpleNamespace(),
                linear_trends_effects=None,
            )
        )
        assert tp["aggregation"] == "DID_1_x"
        assert "DID^X_1" in tp["name"]

    def test_dcdh_delta_with_controls(self):
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=2,
                covariate_residuals=SimpleNamespace(),
                linear_trends_effects=None,
            )
        )
        assert tp["aggregation"] == "delta_x"
        assert "delta^X" in tp["name"]

    def test_dcdh_trends_linear_with_l_max_geq_2_emits_no_scalar_headline(self):
        """``trends_linear=True`` with ``L_max >= 2`` intentionally
        suppresses the scalar aggregate (``overall_att`` is NaN by
        design per ``chaisemartin_dhaultfoeuille.py:2828-2834``). The
        target-parameter block must reflect that — ``aggregation =
        "no_scalar_headline"`` and ``headline_attribute = None``.
        R2 PR #347 review P1 regression.
        """
        # PR #347 R9 P1: exercise BOTH routes to the no-scalar branch
        # — the explicit persisted flag (primary contract) and the
        # legacy ``linear_trends_effects is not None`` inference
        # (fallback for older fits).
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=2,
                covariate_residuals=None,
                linear_trends_effects={"foo": "bar"},
                trends_linear=True,
            )
        )
        assert tp["aggregation"] == "no_scalar_headline"
        assert tp["headline_attribute"] is None
        assert "linear_trends_effects" in tp["definition"]

    def test_dcdh_trends_linear_empty_surface_still_no_scalar(self):
        """PR #347 R9 P1 regression: the estimator sets
        ``linear_trends_effects=None`` when the cumulated-horizon dict
        is empty, but still unconditionally NaN-s ``overall_att`` for
        ``trends_linear=True`` with ``L_max >= 2`` (see
        ``chaisemartin_dhaultfoeuille.py:2828-2834``). The previous
        inference on ``linear_trends_effects is not None`` would have
        routed this case to the ``delta`` branch and narrated it as an
        estimation failure. With the persisted ``trends_linear`` flag
        the no-scalar branch fires correctly.
        """
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=2,
                covariate_residuals=None,
                linear_trends_effects=None,  # empty surface
                trends_linear=True,  # persisted fit-time flag
            )
        )
        assert tp["aggregation"] == "no_scalar_headline"
        assert tp["headline_attribute"] is None

    def test_dcdh_legacy_fit_without_persisted_flag_still_routes_correctly(self):
        """Older fits that do not carry ``trends_linear`` on the
        result still work via the legacy
        ``linear_trends_effects is not None`` inference. This
        preserves backwards compatibility for any cached fit objects
        predating the persisted-flag change.
        """
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=2,
                covariate_residuals=None,
                linear_trends_effects={"foo": "bar"},
                # trends_linear NOT set — legacy fallback kicks in
            )
        )
        assert tp["aggregation"] == "no_scalar_headline"

    def test_dcdh_trends_linear_with_l_max_1_still_has_scalar(self):
        """``trends_linear=True`` with ``L_max = 1`` still produces a
        scalar headline (``DID^{fd}_1``) — the no-scalar rule only
        applies when ``L_max >= 2``.
        """
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=1,
                covariate_residuals=None,
                linear_trends_effects={"foo": "bar"},
            )
        )
        assert tp["aggregation"] == "DID_1_fd"
        assert "DID^{fd}_1" in tp["name"]
        assert tp["headline_attribute"] == "overall_att"

    def test_sdid(self):
        tp = describe_target_parameter(_minimal_result("SyntheticDiDResults"))
        assert tp["aggregation"] == "synthetic"
        assert "synthetic" in tp["name"].lower()

    def test_trop(self):
        tp = describe_target_parameter(_minimal_result("TROPResults"))
        assert tp["aggregation"] == "factor_model"
        assert "factor" in tp["name"].lower()

    def test_spillover_did(self):
        tp = describe_target_parameter(_minimal_result("SpilloverDiDResults"))
        assert tp["aggregation"] == "spillover"
        assert tp["headline_attribute"] == "att"
        assert "Butts" in tp["reference"]
        assert "far-away" in tp["definition"].lower()

    def test_bacon_decomposition(self):
        """PR #347 R8 P3: BaconDecompositionResults is accepted by DR
        (as a diagnostic read-out) but is NOT in ``_APPLICABILITY``,
        so the exhaustiveness test does not exercise it. Cover the
        branch directly. Contract: the target parameter of a Bacon
        decomposition is the TWFE coefficient it decomposes.
        """
        tp = describe_target_parameter(_minimal_result("BaconDecompositionResults"))
        assert tp["aggregation"] == "twfe"
        assert tp["headline_attribute"] == "twfe_estimate"
        assert "TWFE" in tp["name"]
        assert "Goodman-Bacon" in tp["definition"] or "decomposition" in tp["definition"].lower()
        assert "Goodman-Bacon" in tp["reference"]


class TestTargetParameterBaconDRIntegration:
    """PR #347 R8 P3 follow-on: pass a real ``BaconDecompositionResults``
    through DR and assert the ``target_parameter`` block propagates
    into the DR schema. BR rejects ``BaconDecompositionResults`` with
    a ``TypeError`` (Bacon is a diagnostic, not an estimator), so this
    branch is DR-only.
    """

    def test_dr_with_bacon_result_emits_target_parameter(self):
        import warnings

        from diff_diff import DiagnosticReport, bacon_decompose, generate_staggered_data

        warnings.filterwarnings("ignore")
        df = generate_staggered_data(n_units=40, n_periods=5, seed=21)
        bacon = bacon_decompose(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        dr = DiagnosticReport(bacon).to_dict()
        tp = dr["target_parameter"]
        assert tp["aggregation"] == "twfe"
        assert tp["headline_attribute"] == "twfe_estimate"
        # Sanity: the named attribute exists on the real result object.
        assert hasattr(bacon, "twfe_estimate")


class TestTargetParameterFitConfigReads:
    """Parameterized fit-config-branching tests."""

    @pytest.mark.parametrize(
        "clean_control, expected_clause",
        [
            ("never_treated", "never-treated"),
            ("strict", "strictly untreated"),
            ("not_yet_treated", "not yet treated"),
        ],
    )
    def test_stacked_clean_control_branches_definition(self, clean_control, expected_clause):
        tp = describe_target_parameter(
            _minimal_result("StackedDiDResults", clean_control=clean_control)
        )
        assert tp["aggregation"] == "stacked"
        assert expected_clause in tp["definition"]

    @pytest.mark.parametrize(
        "pt_assumption, expected_tag",
        [("all", "pt_all_combined"), ("post", "pt_post_single_baseline")],
    )
    def test_efficient_did_pt_assumption_branches_tag(self, pt_assumption, expected_tag):
        tp = describe_target_parameter(
            _minimal_result("EfficientDiDResults", pt_assumption=pt_assumption)
        )
        assert tp["aggregation"] == expected_tag

    @pytest.mark.parametrize(
        "L_max, controls, trends, expected_tag",
        [
            # L_max=None -> DID_M (Phase 1 per-period aggregate).
            (None, None, None, "M"),
            (None, SimpleNamespace(), None, "M_x"),
            # L_max=1 -> DID_1 (single-horizon per-group).
            (1, None, None, "DID_1"),
            (1, SimpleNamespace(), None, "DID_1_x"),
            (1, None, {"foo": "bar"}, "DID_1_fd"),
            (1, SimpleNamespace(), {"foo": "bar"}, "DID_1_x_fd"),
            # L_max>=2 (no trends) -> cost-benefit delta aggregate.
            (2, None, None, "delta"),
            (2, SimpleNamespace(), None, "delta_x"),
            # L_max>=2 + trends_linear -> no scalar aggregate.
            (2, None, {"foo": "bar"}, "no_scalar_headline"),
            (2, SimpleNamespace(), {"foo": "bar"}, "no_scalar_headline"),
        ],
    )
    def test_dcdh_config_branches_tag(self, L_max, controls, trends, expected_tag):
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=L_max,
                covariate_residuals=controls,
                linear_trends_effects=trends,
            )
        )
        assert tp["aggregation"] == expected_tag
        # Contract: ``headline_attribute`` is "overall_att" whenever
        # there IS a scalar aggregate, and None when the
        # ``trends_linear + L_max>=2`` no-scalar rule fires.
        if expected_tag == "no_scalar_headline":
            assert tp["headline_attribute"] is None
        else:
            assert tp["headline_attribute"] == "overall_att"


class TestTargetParameterCoversEveryResultClass:
    """Exhaustiveness guard (per plan-review MEDIUM #5): every result-
    class name in DR's ``_APPLICABILITY`` dict must get a non-default,
    non-empty target-parameter block."""

    def test_every_applicability_class_has_explicit_branch(self):
        missing = []
        for class_name in _APPLICABILITY:
            tp = describe_target_parameter(_minimal_result(class_name))
            if tp["aggregation"] == "unknown":
                missing.append(class_name)
        assert not missing, (
            "Every result class in _APPLICABILITY must have an explicit "
            f"describe_target_parameter branch. Missing: {missing}"
        )

    def test_every_applicability_class_has_nonempty_fields(self):
        for class_name in _APPLICABILITY:
            tp = describe_target_parameter(_minimal_result(class_name))
            assert tp["name"], f"{class_name}: name is empty"
            assert tp["definition"], f"{class_name}: definition is empty"
            assert tp["aggregation"], f"{class_name}: aggregation is empty"
            assert tp["headline_attribute"], f"{class_name}: headline_attribute is empty"
            assert tp["reference"], f"{class_name}: reference is empty"


class TestTargetParameterCrossSurfaceParity:
    """BR and DR emit identical target_parameter blocks (the shared
    helper is the single source of truth)."""

    def test_br_and_dr_emit_identical_target_parameter(self):
        from diff_diff import BusinessReport, DiagnosticReport

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
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.base_period = "universal"
        stub.inference_method = "analytical"

        br_tp = BusinessReport(stub, auto_diagnostics=False).to_dict()["target_parameter"]
        dr_tp = DiagnosticReport(stub).to_dict()["target_parameter"]
        assert br_tp == dr_tp


class TestTargetParameterProseRendering:
    """The short ``name`` must render in BR summary + DR overall_interpretation;
    the ``definition`` must render in BR full_report and DR full report."""

    def _cs_stub(self):
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
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.base_period = "universal"
        stub.inference_method = "analytical"
        return stub

    def test_br_summary_emits_target_parameter_name(self):
        from diff_diff import BusinessReport

        summary = BusinessReport(self._cs_stub(), auto_diagnostics=False).summary()
        assert "Target parameter:" in summary
        assert "overall ATT" in summary
        # But summary must not embed the full verbose definition —
        # stakeholder paragraph stays within the 6-10 sentence target.
        assert "event_study_effects" not in summary

    def test_br_full_report_emits_target_parameter_section(self):
        from diff_diff import BusinessReport

        md = BusinessReport(self._cs_stub(), auto_diagnostics=False).full_report()
        assert "## Target Parameter" in md
        assert "cohort-size-weighted" in md
        # Full report DOES carry the definition (structural context for
        # readers who scan the markdown).
        assert "event_study_effects" in md or "event-study" in md.lower()

    def test_dr_overall_interpretation_emits_target_parameter_sentence(self):
        from diff_diff import DiagnosticReport

        dr = DiagnosticReport(self._cs_stub()).run_all()
        prose = dr.interpretation
        assert "Target parameter:" in prose
        assert "overall ATT" in prose

    def test_dr_full_report_emits_target_parameter_section(self):
        from diff_diff import DiagnosticReport

        md = DiagnosticReport(self._cs_stub()).full_report()
        assert "## Target Parameter" in md
        assert "Aggregation tag:" in md
        assert "Headline attribute:" in md


class TestTargetParameterRealFitIntegration:
    """PR #347 R1 review P2: stub-based dispatch tests missed (a) the
    TWFE-returns-DiDResults mismatch, (b) the dCDH headline_attribute
    bug. Exercise real fits so these contracts are enforced end-to-end.
    """

    def test_twfe_fit_returns_did_results_branch(self):
        """``TwoWayFixedEffects.fit()`` returns ``DiDResults``, so the
        target-parameter block must be the DiD/TWFE-covering branch.
        Guards against reintroducing a dead-code
        ``TwoWayFixedEffectsResults`` branch.
        """
        import warnings

        from diff_diff import TwoWayFixedEffects, generate_did_data

        warnings.filterwarnings("ignore")
        df = generate_did_data(n_units=40, n_periods=4, seed=7)
        fit = TwoWayFixedEffects().fit(
            df, outcome="outcome", treatment="treated", time="post", unit="unit"
        )
        # Real TWFE fit returns DiDResults (no separate TWFE result class).
        assert type(fit).__name__ == "DiDResults"
        tp = describe_target_parameter(fit)
        assert tp["aggregation"] == "did_or_twfe"
        # The DiDResults branch must name TWFE explicitly so the
        # description is source-faithful for both DiD and TWFE fits.
        assert "TWFE" in tp["name"] or "TWFE" in tp["definition"]

    def test_stacked_did_fit_headline_attribute_matches_real_estimand(self):
        """``StackedDiDResults.overall_att`` is the average of
        post-treatment event-study coefficients ``delta_h`` with
        delta-method SE (``stacked_did.py`` around line 541). Real-fit
        regression against the reviewer's R1 P1 wording catch.
        """
        import warnings

        from diff_diff import StackedDiD, generate_staggered_data

        warnings.filterwarnings("ignore")
        df = generate_staggered_data(n_units=60, n_periods=6, seed=13)
        fit = StackedDiD(kappa_pre=1, kappa_post=1).fit(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert type(fit).__name__ == "StackedDiDResults"
        tp = describe_target_parameter(fit)
        assert tp["headline_attribute"] == "overall_att"
        # R1 P1 fix: the definition must describe the actual estimand —
        # the average of post-treatment delta_h event-study coefficients.
        assert (
            "event-study" in tp["definition"].lower()
            or "delta_h" in tp["definition"]
            or "post-treatment" in tp["definition"].lower()
        )

    def _dcdh_reversible_panel(self, seed):
        """Build a minimal reversible-treatment panel that dCDH
        accepts (group / time / treatment columns, at least one
        switcher)."""
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        units = list(range(20))
        periods = list(range(6))
        rows = []
        for u in units:
            # Half of units switch from 0 -> 1 at period 3.
            for t in periods:
                d = 1 if (u < 10 and t >= 3) else 0
                y = d * 2.0 + rng.normal(0.0, 0.5)
                rows.append({"unit": u, "period": t, "treated": d, "outcome": y})
        return pd.DataFrame(rows)

    def test_dcdh_did_m_fit_headline_attribute_is_overall_att(self):
        """``ChaisemartinDHaultfoeuilleResults.overall_att`` holds the
        DID_M headline scalar (``chaisemartin_dhaultfoeuille_results.py``
        line ~357). R1 P1: previously ``headline_attribute="att"``
        pointed at a non-existent attribute.
        """
        import warnings

        from diff_diff import ChaisemartinDHaultfoeuille

        warnings.filterwarnings("ignore")
        df = self._dcdh_reversible_panel(seed=11)
        # DID_M regime: L_max not supplied.
        fit = ChaisemartinDHaultfoeuille().fit(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            treatment="treated",
        )
        assert type(fit).__name__ == "ChaisemartinDHaultfoeuilleResults"
        tp = describe_target_parameter(fit)
        assert tp["aggregation"] == "M"
        assert tp["headline_attribute"] == "overall_att"
        # Sanity: the attribute BR/DR points at actually exists on the
        # real fit object.
        assert hasattr(fit, tp["headline_attribute"]), (
            f"headline_attribute={tp['headline_attribute']!r} must name "
            "an attribute that actually exists on the real result object."
        )

    def test_dcdh_did_1_fit_overall_att_real(self):
        """Real ``L_max = 1`` fit: headline is ``DID_1`` (single-
        horizon per-group estimand), not the generic ``DID_l``.
        PR #347 R2 P1 regression on a live fit.
        """
        import warnings

        from diff_diff import ChaisemartinDHaultfoeuille

        warnings.filterwarnings("ignore")
        df = self._dcdh_reversible_panel(seed=13)
        fit = ChaisemartinDHaultfoeuille().fit(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            treatment="treated",
            L_max=1,
        )
        tp = describe_target_parameter(fit)
        assert tp["aggregation"] == "DID_1"
        assert "DID_1" in tp["name"]
        assert tp["headline_attribute"] == "overall_att"
        assert hasattr(fit, "overall_att")

    def test_dcdh_delta_fit_real(self):
        """Real ``L_max >= 2`` fit: headline is the cost-benefit
        ``delta`` aggregate. PR #347 R2 P1 regression on a live fit.
        """
        import warnings

        from diff_diff import ChaisemartinDHaultfoeuille

        warnings.filterwarnings("ignore")
        df = self._dcdh_reversible_panel(seed=14)
        fit = ChaisemartinDHaultfoeuille().fit(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            treatment="treated",
            L_max=2,
        )
        tp = describe_target_parameter(fit)
        assert tp["aggregation"] == "delta"
        assert "delta" in tp["name"]
        assert tp["headline_attribute"] == "overall_att"
        assert hasattr(fit, "overall_att")

    def test_dcdh_trends_linear_no_scalar_propagates_through_br(self):
        """PR #347 R4 P1 end-to-end: on the dCDH no-scalar
        configuration (``trends_linear=True`` + ``L_max>=2``), BR's
        ``to_dict()`` headline must carry ``status="no_scalar_by_design"``
        and BR's summary / full report must emit explicit no-scalar
        prose — NOT the generic "non-finite effect / inspect the fit
        for rank deficiency" estimation-failure messaging.
        """
        import warnings

        from diff_diff import BusinessReport, ChaisemartinDHaultfoeuille

        warnings.filterwarnings("ignore")
        df = self._dcdh_reversible_panel(seed=16)
        fit = ChaisemartinDHaultfoeuille().fit(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            treatment="treated",
            L_max=2,
            trends_linear=True,
        )
        br = BusinessReport(fit, outcome_label="the outcome", auto_diagnostics=False)
        schema = br.to_dict()
        assert schema["headline"]["status"] == "no_scalar_by_design"
        assert schema["headline"]["effect"] is None
        # BR's summary prose must be explicit no-scalar, not
        # "non-finite estimate / inspect rank deficiency".
        summary = br.summary()
        assert "no scalar" in summary.lower() or "does not produce a scalar" in summary.lower()
        assert "rank deficiency" not in summary.lower()
        assert "estimation failed" not in summary.lower()
        # Must NOT emit the "estimation_failure" caveat either.
        caveats = br.caveats()
        topics = {c.get("topic") for c in caveats}
        assert "estimation_failure" not in topics

    def test_dcdh_trends_linear_no_scalar_propagates_through_dr(self):
        """Same contract on the DR side: ``headline_metric`` carries
        ``status="no_scalar_by_design"`` and the overall-interpretation
        prose is explicit no-scalar, not an estimation-failure sentence.
        """
        import warnings

        from diff_diff import ChaisemartinDHaultfoeuille, DiagnosticReport

        warnings.filterwarnings("ignore")
        df = self._dcdh_reversible_panel(seed=17)
        fit = ChaisemartinDHaultfoeuille().fit(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            treatment="treated",
            L_max=2,
            trends_linear=True,
        )
        dr_report = DiagnosticReport(fit)
        dr = dr_report.run_all()
        schema = dr.schema
        assert schema["headline_metric"]["status"] == "no_scalar_by_design"
        # DR interpretation must not narrate estimation failure.
        prose = dr.interpretation
        assert "does not produce a scalar" in prose.lower() or "no scalar" in prose.lower()
        assert "rank deficiency" not in prose.lower()
        assert "zero effective sample" not in prose.lower()
        # PR #347 R5 P2 + P3: the DR markdown full_report must also
        # handle the no-scalar case — the top **Headline** line
        # previously formatted ``None`` values straight in
        # (``**Headline**: ... = None (SE None, p = None)``). The
        # fixed renderer should emit explicit no-scalar markdown
        # instead.
        md = dr_report.full_report()
        assert "**Headline**: no scalar aggregate by design" in md, (
            f"DR full_report must emit explicit no-scalar top headline on "
            f"the trends_linear + L_max>=2 dCDH branch. Got: {md!r}"
        )
        # And must NOT contain the raw `None`-interpolated line from
        # the generic headline path.
        assert "= None (SE None" not in md
        assert "p = None)" not in md

    def test_dcdh_empty_surface_target_parameter_definition_names_empty_state(self):
        """PR #347 R12 P1: when ``trends_linear=True, L_max>=2,
        linear_trends_effects=None`` (no horizons survived), the
        target-parameter ``definition`` must NOT tell users to "see
        linear_trends_effects" — that dict is empty. It must name the
        empty-surface state explicitly and point at a different
        remediation (re-fit).
        """
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=2,
                covariate_residuals=None,
                linear_trends_effects=None,
                trends_linear=True,
            )
        )
        defn = tp["definition"]
        # Must NOT claim the horizon dict has content.
        assert "``results.linear_trends_effects[l]``" not in defn
        # Must name the empty-surface state.
        assert (
            "no cumulated level effects" in defn.lower()
            or "horizon surface is empty" in defn.lower()
            or "survived estimation" in defn.lower()
        ), f"Empty-surface definition must name the state. Got: {defn!r}"

    def _empty_surface_stub(self):
        from diff_diff.chaisemartin_dhaultfoeuille_results import (
            ChaisemartinDHaultfoeuilleResults,
        )

        return ChaisemartinDHaultfoeuilleResults(
            overall_att=float("nan"),
            overall_se=float("nan"),
            overall_t_stat=float("nan"),
            overall_p_value=float("nan"),
            overall_conf_int=(float("nan"), float("nan")),
            joiners_att=float("nan"),
            joiners_se=float("nan"),
            joiners_t_stat=float("nan"),
            joiners_p_value=float("nan"),
            joiners_conf_int=(float("nan"), float("nan")),
            n_joiner_cells=0,
            n_joiner_obs=0,
            joiners_available=False,
            leavers_att=float("nan"),
            leavers_se=float("nan"),
            leavers_t_stat=float("nan"),
            leavers_p_value=float("nan"),
            leavers_conf_int=(float("nan"), float("nan")),
            n_leaver_cells=0,
            n_leaver_obs=0,
            leavers_available=False,
            placebo_effect=float("nan"),
            placebo_se=float("nan"),
            placebo_t_stat=float("nan"),
            placebo_p_value=float("nan"),
            placebo_conf_int=(float("nan"), float("nan")),
            placebo_available=False,
            per_period_effects={},
            units=[1, 2, 3],
            time_periods=[1, 2, 3, 4],
            n_obs=100,
            n_treated_obs=50,
            n_switcher_cells=10,
            n_cohorts=2,
            n_groups_dropped_crossers=0,
            n_groups_dropped_singleton_baseline=0,
            n_groups_dropped_never_switching=0,
            L_max=2,
            trends_linear=True,
            linear_trends_effects=None,
        )

    def test_dcdh_empty_surface_br_dr_reason_names_empty_state(self):
        """Mirror empty-surface contract through BR and DR headline
        ``reason``. Pointing users at ``linear_trends_effects`` is
        dead-end guidance when that surface is empty.
        """
        from diff_diff import BusinessReport, DiagnosticReport

        stub = self._empty_surface_stub()
        br_reason = BusinessReport(stub, auto_diagnostics=False).to_dict()["headline"]["reason"]
        assert "``results.linear_trends_effects[l]``" not in br_reason
        assert (
            "no cumulated level effects" in br_reason.lower()
            or "horizon surface is empty" in br_reason.lower()
            or "survived estimation" in br_reason.lower()
        )

        dr_reason = DiagnosticReport(stub).to_dict()["headline_metric"]["reason"]
        assert "``results.linear_trends_effects[l]``" not in dr_reason
        assert (
            "no cumulated level effects" in dr_reason.lower()
            or "horizon surface is empty" in dr_reason.lower()
            or "survived estimation" in dr_reason.lower()
        )

    def test_dcdh_empty_surface_to_dataframe_linear_trends_returns_empty_frame(self):
        """PR #347 R12 P1: ``to_dataframe("linear_trends")`` on an
        empty-surface trends-linear fit previously raised the wrong
        remediation ("Pass trends_linear=True to fit()") — which is
        dead-end when the user already did. Now returns an empty
        DataFrame.
        """
        stub = self._empty_surface_stub()
        df = stub.to_dataframe("linear_trends")
        assert df.empty
        assert "horizon" in df.columns
        assert "effect" in df.columns

    def test_dcdh_empty_surface_br_rendered_prose_no_stale_guidance(self):
        """PR #347 R13 P1: BR's rendered prose surfaces
        (``headline()``, ``summary()``, ``full_report()``) must not
        tell users to "see linear_trends_effects" on the empty-
        surface subcase. Previously the schema ``reason`` was
        correct but the renderers still hardcoded the populated-
        surface phrasing.
        """
        from diff_diff import BusinessReport

        stub = self._empty_surface_stub()
        br = BusinessReport(stub, auto_diagnostics=False)

        headline = br.headline()
        assert "see ``linear_trends_effects``" not in headline
        assert (
            "no cumulated level effects" in headline.lower()
            or "horizon surface is empty" in headline.lower()
            or "survived estimation" in headline.lower()
        )

        summary = br.summary()
        assert "see ``linear_trends_effects``" not in summary

        full = br.full_report()
        assert "see ``linear_trends_effects``" not in full
        assert (
            "no cumulated level effects" in full.lower()
            or "horizon surface is empty" in full.lower()
            or "survived estimation" in full.lower()
        )

    def test_dcdh_empty_surface_dr_rendered_prose_no_stale_guidance(self):
        """PR #347 R13 P1: DR's overall-interpretation prose must not
        tell users to "see linear_trends_effects" on the empty-
        surface subcase.
        """
        from diff_diff import DiagnosticReport

        stub = self._empty_surface_stub()
        dr = DiagnosticReport(stub).run_all()

        prose = dr.interpretation
        assert "see ``linear_trends_effects``" not in prose
        assert (
            "no cumulated level effects" in prose.lower()
            or "horizon surface is empty" in prose.lower()
            or "survived estimation" in prose.lower()
        )

    def _empty_surface_stub_with_controls(self):
        """Mirror of ``_empty_surface_stub`` with ``covariate_residuals``
        populated — the covariate-adjusted empty-surface subcase.
        """
        from diff_diff.chaisemartin_dhaultfoeuille_results import (
            ChaisemartinDHaultfoeuilleResults,
        )

        return ChaisemartinDHaultfoeuilleResults(
            overall_att=float("nan"),
            overall_se=float("nan"),
            overall_t_stat=float("nan"),
            overall_p_value=float("nan"),
            overall_conf_int=(float("nan"), float("nan")),
            joiners_att=float("nan"),
            joiners_se=float("nan"),
            joiners_t_stat=float("nan"),
            joiners_p_value=float("nan"),
            joiners_conf_int=(float("nan"), float("nan")),
            n_joiner_cells=0,
            n_joiner_obs=0,
            joiners_available=False,
            leavers_att=float("nan"),
            leavers_se=float("nan"),
            leavers_t_stat=float("nan"),
            leavers_p_value=float("nan"),
            leavers_conf_int=(float("nan"), float("nan")),
            n_leaver_cells=0,
            n_leaver_obs=0,
            leavers_available=False,
            placebo_effect=float("nan"),
            placebo_se=float("nan"),
            placebo_t_stat=float("nan"),
            placebo_p_value=float("nan"),
            placebo_conf_int=(float("nan"), float("nan")),
            placebo_available=False,
            per_period_effects={},
            units=[1, 2, 3],
            time_periods=[1, 2, 3, 4],
            n_obs=100,
            n_treated_obs=50,
            n_switcher_cells=10,
            n_cohorts=2,
            n_groups_dropped_crossers=0,
            n_groups_dropped_singleton_baseline=0,
            n_groups_dropped_never_switching=0,
            L_max=2,
            trends_linear=True,
            linear_trends_effects=None,
            # The load-bearing difference from ``_empty_surface_stub``:
            # covariates are active, so every label and reason surface
            # must flip to the covariate-adjusted form ``DID^{X,fd}_l``.
            covariate_residuals=SimpleNamespace(),
        )

    def test_dcdh_empty_surface_with_controls_target_parameter_uses_x_fd_label(self):
        """PR #347 R14 P1 regression: when ``trends_linear=True``,
        ``L_max >= 2``, ``covariate_residuals`` is populated, and
        ``linear_trends_effects is None`` (empty-surface subcase with
        covariates), the helper definition must name
        ``DID^{X,fd}_l`` — NOT bare ``DID^{fd}_l``. The R14 review
        flagged the hardcoded ``DID^{fd}_l`` label on this exact
        branch.
        """
        stub = self._empty_surface_stub_with_controls()
        tp = describe_target_parameter(stub)
        defn = tp["definition"]
        assert "DID^{X,fd}_l" in defn, (
            f"Covariate-adjusted empty-surface definition must use "
            f"``DID^{{X,fd}}_l``. Got: {defn!r}"
        )
        # The helper must NOT also emit the bare ``DID^{fd}_l`` form on
        # the covariate path (that label belongs to the no-covariate
        # branch only).
        assert "DID^{fd}_l" not in defn.replace("DID^{X,fd}_l", ""), (
            f"Covariate-adjusted empty-surface definition must not emit "
            f"the bare ``DID^{{fd}}_l`` label. Got: {defn!r}"
        )

    def test_dcdh_empty_surface_with_controls_br_dr_reason_uses_x_fd_label(self):
        """PR #347 R14 P1 regression: BR's ``headline.reason`` and
        DR's ``headline_metric.reason`` on the covariate-adjusted
        empty-surface subcase must both name ``DID^{X,fd}_l``, not
        bare ``DID^{fd}_l``.
        """
        from diff_diff import BusinessReport, DiagnosticReport

        stub = self._empty_surface_stub_with_controls()

        br_reason = BusinessReport(stub, auto_diagnostics=False).to_dict()["headline"]["reason"]
        assert "DID^{X,fd}_l" in br_reason, (
            f"BR headline.reason on covariate-adjusted empty-surface fit "
            f"must name ``DID^{{X,fd}}_l``. Got: {br_reason!r}"
        )
        assert "DID^{fd}_l" not in br_reason.replace("DID^{X,fd}_l", ""), (
            f"BR headline.reason must not emit the bare ``DID^{{fd}}_l`` "
            f"label when covariates are active. Got: {br_reason!r}"
        )

        dr_reason = DiagnosticReport(stub).to_dict()["headline_metric"]["reason"]
        assert "DID^{X,fd}_l" in dr_reason, (
            f"DR headline_metric.reason on covariate-adjusted empty-surface "
            f"fit must name ``DID^{{X,fd}}_l``. Got: {dr_reason!r}"
        )
        assert "DID^{fd}_l" not in dr_reason.replace("DID^{X,fd}_l", ""), (
            f"DR headline_metric.reason must not emit the bare "
            f"``DID^{{fd}}_l`` label when covariates are active. "
            f"Got: {dr_reason!r}"
        )

    def test_dcdh_empty_surface_with_controls_rendered_prose_uses_x_fd_label(self):
        """PR #347 R14 P1 regression: BR's rendered prose
        (``headline()``, ``full_report()``) and DR's
        ``interpretation`` / full report on the covariate-adjusted
        empty-surface subcase must all name ``DID^{X,fd}_l``.
        """
        from diff_diff import BusinessReport, DiagnosticReport

        stub = self._empty_surface_stub_with_controls()
        br = BusinessReport(stub, auto_diagnostics=False)
        br_headline = br.headline()
        assert "DID^{X,fd}_l" in br_headline, (
            f"BR headline() prose must name ``DID^{{X,fd}}_l`` on "
            f"covariate-adjusted empty-surface fit. Got: {br_headline!r}"
        )
        br_full = br.full_report()
        assert "DID^{X,fd}_l" in br_full, (
            "BR full_report() prose must name ``DID^{X,fd}_l`` on "
            "covariate-adjusted empty-surface fit."
        )

        dr = DiagnosticReport(stub).run_all()
        assert "DID^{X,fd}_l" in dr.interpretation, (
            f"DR interpretation prose must name ``DID^{{X,fd}}_l`` on "
            f"covariate-adjusted empty-surface fit. Got: "
            f"{dr.interpretation!r}"
        )

    def test_dcdh_empty_surface_native_label_no_stale_guidance(self):
        """PR #347 R13 P1: the dCDH result's own
        ``_estimand_label()`` (surfaced via
        ``to_dataframe("overall")``) must not direct users to
        ``linear_trends_effects`` on the empty-surface subcase.
        Instead it names the empty state.
        """
        stub = self._empty_surface_stub()
        label = stub._estimand_label()
        assert "see linear_trends_effects" not in label
        assert "no cumulated level effects survived" in label.lower()

        # to_dataframe("overall") pulls the label too.
        row = stub.to_dataframe("overall").iloc[0]
        assert "see linear_trends_effects" not in row["estimand"]

    def test_dcdh_empty_surface_propagates_to_assumption_and_native_label(self):
        """PR #347 R10 P1 regression: the persisted ``trends_linear``
        flag must drive every downstream consumer that previously
        inferred trends-linear from ``linear_trends_effects``. This
        test exercises two consumers the R9 patch missed:

        (a) ``ChaisemartinDHaultfoeuilleResults._estimand_label()``
            and its dependencies (``_horizon_label``,
            ``to_dataframe("overall")``) — must return a
            ``DID^{fd}_l`` / ``DID^{X,fd}_l`` label rather than
            ``delta`` on an empty-surface fit.
        (b) ``BusinessReport._describe_assumption`` — must emit the
            ``DID^{fd}_l`` identification clause rather than the
            non-trends default.

        Builds a stub ``ChaisemartinDHaultfoeuilleResults`` with
        ``trends_linear=True``, ``L_max=2``, and
        ``linear_trends_effects=None`` (the empty-surface case).
        """
        from diff_diff import BusinessReport
        from diff_diff.chaisemartin_dhaultfoeuille_results import (
            ChaisemartinDHaultfoeuilleResults,
        )

        stub = ChaisemartinDHaultfoeuilleResults(
            overall_att=float("nan"),
            overall_se=float("nan"),
            overall_t_stat=float("nan"),
            overall_p_value=float("nan"),
            overall_conf_int=(float("nan"), float("nan")),
            joiners_att=float("nan"),
            joiners_se=float("nan"),
            joiners_t_stat=float("nan"),
            joiners_p_value=float("nan"),
            joiners_conf_int=(float("nan"), float("nan")),
            n_joiner_cells=0,
            n_joiner_obs=0,
            joiners_available=False,
            leavers_att=float("nan"),
            leavers_se=float("nan"),
            leavers_t_stat=float("nan"),
            leavers_p_value=float("nan"),
            leavers_conf_int=(float("nan"), float("nan")),
            n_leaver_cells=0,
            n_leaver_obs=0,
            leavers_available=False,
            placebo_effect=float("nan"),
            placebo_se=float("nan"),
            placebo_t_stat=float("nan"),
            placebo_p_value=float("nan"),
            placebo_conf_int=(float("nan"), float("nan")),
            placebo_available=False,
            per_period_effects={},
            units=[1, 2, 3],
            time_periods=[1, 2, 3, 4],
            n_obs=100,
            n_treated_obs=50,
            n_switcher_cells=10,
            n_cohorts=2,
            n_groups_dropped_crossers=0,
            n_groups_dropped_singleton_baseline=0,
            n_groups_dropped_never_switching=0,
            L_max=2,
            # The load-bearing configuration: trends_linear with empty
            # horizon surface.
            trends_linear=True,
            linear_trends_effects=None,
        )

        # (a) Native result label: must NOT be "delta", must name DID^{fd}_l.
        label = stub._estimand_label()
        assert "delta" not in label, (
            f"Empty-surface trends_linear fit must not be labeled delta. "
            f"Got _estimand_label={label!r}"
        )
        assert "DID^{fd}_l" in label or "DID^{X,fd}_l" in label, (
            f"Empty-surface trends_linear fit must name DID^{{fd}}_l. " f"Got: {label!r}"
        )

        # to_dataframe("overall") uses _estimand_label() — same guarantee.
        row = stub.to_dataframe("overall").iloc[0]
        assert "delta" not in row["estimand"]

        # (b) BR assumption description: must mention linear trends /
        # first-differences clause.
        br = BusinessReport(stub, auto_diagnostics=False)
        desc = br.to_dict()["assumption"]["description"]
        assert (
            "first-differenced" in desc.lower()
            or "linear pre-trend" in desc.lower()
            or "linear trends" in desc.lower()
        ), (
            "BR's identifying-assumption description must include the "
            f"linear-trends identification clause on an empty-surface "
            f"trends_linear fit. Got: {desc!r}"
        )

    def test_dcdh_trends_linear_with_l_max_geq_2_fit_real(self):
        """Real ``trends_linear=True`` + ``L_max>=2`` fit: the library
        intentionally sets ``overall_att=NaN`` and populates the
        per-horizon effects on ``linear_trends_effects`` instead
        (``chaisemartin_dhaultfoeuille.py:2828-2834``). The target-
        parameter block must reflect that via
        ``aggregation="no_scalar_headline"`` and
        ``headline_attribute is None``. PR #347 R3 P3 regression on
        a live fit.
        """
        import math
        import warnings

        from diff_diff import ChaisemartinDHaultfoeuille

        warnings.filterwarnings("ignore")
        df = self._dcdh_reversible_panel(seed=15)
        fit = ChaisemartinDHaultfoeuille().fit(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            treatment="treated",
            L_max=2,
            trends_linear=True,
        )
        # Real fit must intentionally produce NaN overall_att.
        assert math.isnan(fit.overall_att), (
            "trends_linear + L_max>=2 must suppress overall_att (NaN by "
            "design). If this test fails, the library contract changed — "
            "update the ``no_scalar_headline`` branch of "
            "describe_target_parameter accordingly."
        )
        # linear_trends_effects must be populated (the per-horizon
        # cumulated level effects that replace the scalar aggregate).
        assert fit.linear_trends_effects is not None
        # Target-parameter block must route through the no-scalar branch.
        tp = describe_target_parameter(fit)
        assert tp["aggregation"] == "no_scalar_headline"
        assert tp["headline_attribute"] is None
        assert "linear_trends_effects" in tp["definition"]
