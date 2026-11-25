"""The ``design_reconciliation`` block of DiagnosticReport and BusinessReport."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    BusinessReport,
    CallawaySantAnna,
    DiagnosticReport,
    diagnose_design,
)
from diff_diff.business_report import BUSINESS_REPORT_SCHEMA_VERSION
from diff_diff.diagnostic_report import DIAGNOSTIC_REPORT_SCHEMA_VERSION

REPORTS = pytest.mark.parametrize("report", [DiagnosticReport, BusinessReport])
COLUMNS = dict(outcome="y", unit="unit", time="period", first_treat="g")


def history(seed=7, n_units=20, n_periods=8):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(1, n_periods + 1), n_units)
    y = rng.normal(size=n_units)[unit] + 0.1 * period + rng.normal(size=len(unit))
    return pd.DataFrame({"unit": unit, "period": period, "y": y})


def analysis(panel, n_new=3, n_treated=10, effect=1.0):
    """The diagnosed panel plus ``n_new`` periods, ``n_treated`` units treated in them."""
    last = int(panel["period"].max())
    units = np.sort(panel["unit"].unique())
    rng = np.random.default_rng(11)
    later = pd.DataFrame(
        {
            "unit": np.repeat(units, n_new),
            "period": np.tile(np.arange(last + 1, last + 1 + n_new), len(units)),
        }
    )
    later["y"] = rng.normal(size=len(later)) + effect * (later["unit"] < n_treated)
    frame = pd.concat([panel, later], ignore_index=True)
    frame["g"] = np.where(frame["unit"] < n_treated, last + 1, 0)
    return frame


@pytest.fixture(scope="module")
def setup():
    """A lock, the analysis frame, and reconciliations of each kind."""
    panel = history()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lock = diagnose_design(
            panel,
            outcome="y",
            unit="unit",
            time="period",
            estimators={"cs": CallawaySantAnna()},
            cohort_periods=[6],
            n_treated=10,
            n_simulations=4,
            seed=3,
            progress=False,
        ).lock(progress=False)
        frame = analysis(panel)
        fitted = lock.fit(frame, first_treat="g")

        estimator = lock.to_estimator()
        by_hand = estimator.fit(frame, **COLUMNS)
        declared = lock.reconcile(estimator, data=frame, results=by_hand, **COLUMNS)
        # No data: the realized-design checks are not run.
        partial = lock.reconcile(estimator, results=by_hand, **COLUMNS)

        other = CallawaySantAnna(control_group="not_yet_treated")
        other_fit = other.fit(frame, **COLUMNS)
        deviating = lock.reconcile(other, data=frame, results=other_fit, **COLUMNS)
        without_results = lock.reconcile(estimator, data=frame, **COLUMNS)
    return {
        "lock": lock,
        "frame": frame,
        "fitted": fitted,
        "declared": declared,
        "partial": partial,
        "deviating": deviating,
        "without_results": without_results,
    }


# The default battery's pre-trends power check takes seconds per report on the
# pure-Python backend and is not what these tests are about.
FAST = {
    DiagnosticReport: {"run_pretrends_power": False},
    BusinessReport: {"auto_diagnostics": False},
}


def build(report, reconciliation, **kwargs):
    options = {**FAST[report], **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report(reconciliation.results, design_reconciliation=reconciliation, **options)


class TestBlock:
    @REPORTS
    def test_absent_by_default(self, setup, report):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            built = report(setup["fitted"].results, **FAST[report])
            assert built.to_dict()["design_reconciliation"] is None
            assert "design lock" not in built.summary()

    @REPORTS
    def test_fitted_from_the_lock(self, setup, report):
        built = build(report, setup["fitted"])
        block = built.to_dict()["design_reconciliation"]
        n_checks = len(setup["fitted"].checks)
        assert block == {
            "status": "match",
            "declared": False,
            "label": "cs",
            "lock_sha256": setup["lock"].sha256,
            "n_checks": n_checks,
            "n_match": n_checks,
            "n_deviates": 0,
            "n_not_checked": 0,
            "deviations": [],
            "not_checked": [],
        }
        json.dumps(built.to_dict())

    @REPORTS
    def test_declared(self, setup, report):
        block = build(report, setup["declared"]).to_dict()["design_reconciliation"]
        assert block["status"] == "match" and block["declared"] is True
        assert block["n_not_checked"] == 0

    @REPORTS
    def test_checks_not_run_are_counted_and_named(self, setup, report):
        block = build(report, setup["partial"]).to_dict()["design_reconciliation"]
        assert block["status"] == "match"
        assert block["n_not_checked"] == len(block["not_checked"]) == 6
        assert "data_sha256" in block["not_checked"]
        assert block["n_match"] + block["n_not_checked"] == block["n_checks"]

    @REPORTS
    def test_deviation(self, setup, report):
        built = build(report, setup["deviating"])
        block = built.to_dict()["design_reconciliation"]
        assert block["status"] == "deviates" and block["n_deviates"] == 1
        assert block["deviations"] == [
            {
                "item": "params.control_group",
                "locked": "never_treated",
                "realized": "not_yet_treated",
            }
        ]
        json.dumps(built.to_dict())

    def test_schema_versions_are_unchanged(self):
        # An additive top-level key; no bump under the stability policy.
        assert DIAGNOSTIC_REPORT_SCHEMA_VERSION == "3.0"
        assert BUSINESS_REPORT_SCHEMA_VERSION == "4.0"

    def test_it_is_not_a_check_section(self, setup):
        schema = build(DiagnosticReport, setup["fitted"]).to_dict()
        assert "design_reconciliation" not in schema["skipped"]
        results = build(DiagnosticReport, setup["fitted"]).run_all()
        assert "design_reconciliation" not in results.applicable_checks
        assert "design_reconciliation" not in set(results.to_dataframe()["check"])


class TestRefusals:
    @REPORTS
    def test_wrong_type(self, setup, report):
        with pytest.raises(TypeError, match="must be a DesignReconciliationResults"):
            report(setup["fitted"].results, design_reconciliation=setup["lock"])
        with pytest.raises(TypeError, match="must be a DesignReconciliationResults"):
            report(setup["fitted"].results, design_reconciliation=setup["fitted"].to_dict())

    @REPORTS
    def test_made_for_other_results(self, setup, report):
        with pytest.raises(ValueError, match="was not made for the results"):
            report(setup["declared"].results, design_reconciliation=setup["fitted"])

    @REPORTS
    def test_made_without_results(self, setup, report):
        assert setup["without_results"].results is None
        with pytest.raises(ValueError, match="pass\\s+results= to DesignLock.reconcile"):
            report(setup["declared"].results, design_reconciliation=setup["without_results"])


class TestProse:
    @REPORTS
    def test_fitted_match(self, setup, report):
        built = build(report, setup["fitted"])
        n_checks = len(setup["fitted"].checks)
        sentence = (
            f"The analysis was fitted from design lock 'cs' and agrees with it on all "
            f"{n_checks} checks that were run. The lock does not record when it was made."
        )
        assert sentence in built.summary()
        full = built.full_report()
        assert "## Design Reconciliation" in full and sentence in full
        assert "- configuration: fitted from the lock" in full
        assert f"- checks: {n_checks} match, 0 deviate, 0 not checked" in full

    @REPORTS
    def test_declared_match_with_checks_not_run(self, setup, report):
        built = build(report, setup["partial"])
        n_run = len(setup["partial"].checks) - 6
        sentence = (
            f"As declared to reconcile(), the analysis agrees with design lock 'cs' on "
            f"all {n_run} checks that were run. 6 checks were not run. The lock does "
            f"not record when it was made."
        )
        assert sentence in built.summary()
        full = built.full_report()
        assert "- configuration: declared to reconcile()" in full
        assert "- not checked: `data_sha256`" in full

    @REPORTS
    def test_deviation_names_the_item_and_how_it_was_established(self, setup, report):
        built = build(report, setup["deviating"])
        n_run = len(setup["deviating"].checks)
        sentence = (
            f"As declared to reconcile(), the analysis deviates from design lock 'cs' "
            f"on 1 of {n_run} checks that were run: params.control_group."
        )
        assert sentence in built.summary()
        assert built.summary().count("deviates from design lock") == 1
        assert (
            "- deviates: `params.control_group` (locked `never_treated`, realized "
            "`not_yet_treated`)"
        ) in built.full_report()

    def test_a_fitted_analysis_that_deviates_says_it_was_fitted(self, setup):
        frame = setup["frame"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fewer = setup["lock"].fit(frame[frame["unit"] != frame["unit"].max()], first_treat="g")
            text = build(BusinessReport, fewer).summary()
        assert "The analysis was fitted from design lock 'cs' and deviates from it on" in text

    def test_more_than_three_deviations_are_counted(self):
        from diff_diff._reporting_helpers import render_design_reconciliation

        block = {
            "status": "deviates",
            "declared": True,
            "label": "cs",
            "n_match": 6,
            "n_deviates": 5,
            "n_not_checked": 1,
            "deviations": [{"item": name} for name in "abcde"],
        }
        assert render_design_reconciliation(block) == (
            "As declared to reconcile(), the analysis deviates from design lock 'cs' on "
            "5 of 11 checks that were run: a, b, c and 2 more. 1 check was not run. The "
            "lock does not record when it was made."
        )

    @REPORTS
    @pytest.mark.parametrize("kind", ["fitted", "declared", "partial", "deviating"])
    def test_never_claims_pre_specification(self, setup, report, kind):
        built = build(report, setup[kind])
        text = (built.summary() + built.full_report() + json.dumps(built.to_dict())).lower()
        for claim in ("pre-specified", "prespecified", "pre-registered", "preregistered"):
            assert claim not in text


class TestBusinessReport:
    def caveats(self, reconciliation, **kwargs):
        return {
            caveat["topic"]: caveat
            for caveat in build(BusinessReport, reconciliation, **kwargs).to_dict()["caveats"]
        }

    def test_deviation_caveat(self, setup):
        caveat = self.caveats(setup["deviating"])["design_deviation"]
        assert caveat["severity"] == "warning"
        assert "deviates from design lock 'cs'" in caveat["message"]
        assert "design_not_checked" not in self.caveats(setup["deviating"])

    def test_not_checked_caveat(self, setup):
        caveat = self.caveats(setup["partial"])["design_not_checked"]
        assert caveat["severity"] == "info"
        assert "6 checks were not run" in caveat["message"]

    def test_no_caveat_for_a_complete_match(self, setup):
        topics = set(self.caveats(setup["fitted"]))
        assert not topics & {"design_deviation", "design_not_checked"}

    def test_block_and_caveat_without_diagnostics(self, setup):
        built = build(BusinessReport, setup["deviating"], auto_diagnostics=False)
        schema = built.to_dict()
        assert schema["diagnostics"]["status"] == "skipped"
        assert schema["design_reconciliation"]["status"] == "deviates"
        assert "design_deviation" in {caveat["topic"] for caveat in schema["caveats"]}
        assert "deviates from design lock" in built.summary()

    def test_forwarded_to_the_diagnostics_it_builds(self, setup, monkeypatch):
        import diff_diff.business_report as module

        def without_the_slow_check(results, **kwargs):
            return DiagnosticReport(results, run_pretrends_power=False, **kwargs)

        monkeypatch.setattr(module, "DiagnosticReport", without_the_slow_check)
        schema = build(BusinessReport, setup["fitted"], auto_diagnostics=True).to_dict()
        assert schema["diagnostics"]["status"] == "ran"
        nested = schema["diagnostics"]["schema"]["design_reconciliation"]
        assert nested == schema["design_reconciliation"]

    @pytest.mark.parametrize("as_results", [True, False])
    def test_not_lifted_from_supplied_diagnostics(self, setup, as_results):
        fitted = setup["fitted"]
        diagnostics = build(DiagnosticReport, fitted)
        if as_results:
            diagnostics = diagnostics.run_all()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = BusinessReport(fitted.results, diagnostics=diagnostics)
            schema = report.to_dict()
        assert schema["design_reconciliation"] is None
        assert not {"design_deviation", "design_not_checked"} & {
            caveat["topic"] for caveat in schema["caveats"]
        }
        assert "design lock" not in report.summary()
        assert "## Design Reconciliation" not in report.full_report()
        # The supplied diagnostics' own content is passed through unchanged.
        assert schema["diagnostics"]["schema"]["design_reconciliation"]["status"] == "match"

    @pytest.mark.parametrize("as_results", [True, False])
    def test_same_reconciliation_in_both_is_accepted(self, setup, as_results):
        fitted = setup["fitted"]
        diagnostics = build(DiagnosticReport, fitted)
        if as_results:
            diagnostics = diagnostics.run_all()
        schema = build(BusinessReport, fitted, diagnostics=diagnostics).to_dict()
        assert schema["design_reconciliation"]["status"] == "match"

    @pytest.mark.parametrize("as_results", [True, False])
    def test_conflicting_reconciliations_are_refused(self, setup, as_results):
        declared, partial = setup["declared"], setup["partial"]
        assert declared.results is partial.results
        diagnostics = build(DiagnosticReport, partial)
        if as_results:
            diagnostics = diagnostics.run_all()
        with pytest.raises(ValueError, match="describe different reconciliations"):
            BusinessReport(
                declared.results, design_reconciliation=declared, diagnostics=diagnostics
            )
