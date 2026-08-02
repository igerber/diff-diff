"""Canonical-dataset regression guards for BusinessReport / DiagnosticReport.

Closes BR/DR foundation gap #4 (real-dataset validation): the risk was
that BR/DR's prose could silently diverge from canonical interpretations
of applied work without synthetic-DGP tests catching it. These tests
run BR on four canonical fits and assert direction / verdict / tier
properties that should hold regardless of small data-aggregation
differences between the bundled dataset and the published author
sample.

Assertions are property-level, not exact-match:
- Sign of the point estimate.
- Whether the CI includes zero.
- Pre-trends verdict bin (``no_detected_violation`` vs
  ``clear_violation``).
- HonestDiD sensitivity tier (robust vs fragile, via ``breakdown_M``).
- Cross-estimator consistency (CS and SA produce the same direction
  and verdict on the same data).

These tests use the ``_construct_*`` fallback data from
``diff_diff.datasets`` to avoid network dependency in CI. The
construction targets match the published summary statistics, so
canonical-direction / canonical-verdict properties hold.
"""

from __future__ import annotations

import warnings

import pytest

from diff_diff import (
    BusinessReport,
    CallawaySantAnna,
    DifferenceInDifferences,
    SunAbraham,
)
from diff_diff.datasets import (
    _construct_card_krueger_data,
    _construct_castle_doctrine_data,
    _construct_mpdta_data,
)


@pytest.fixture(scope="module")
def card_krueger_long():
    """Card-Krueger dataset reshaped wide -> long for DiD fitting."""
    warnings.filterwarnings("ignore")
    ck = _construct_card_krueger_data()
    ck_long = ck.melt(
        id_vars=["store_id", "state", "treated"],
        value_vars=["emp_pre", "emp_post"],
        var_name="period",
        value_name="employment",
    )
    ck_long["post"] = (ck_long["period"] == "emp_post").astype(int)
    return ck_long


@pytest.fixture(scope="module")
def mpdta_panel():
    """Callaway-Sant'Anna benchmark (mpdta) as constructed by the fallback."""
    warnings.filterwarnings("ignore")
    return _construct_mpdta_data()


@pytest.fixture(scope="module")
def castle_panel():
    """Cheng-Hoekstra Castle Doctrine dataset as constructed by the fallback."""
    warnings.filterwarnings("ignore")
    return _construct_castle_doctrine_data()


class TestCardKruegerCanonicalDirection:
    """Card & Krueger (1994): NJ minimum-wage increase vs PA control.

    Canonical finding: no significant disemployment effect; published
    ATT is positive (~+0.59 FTE per store) but the CI includes zero.
    """

    def test_no_significant_disemployment(self, card_krueger_long):
        did = DifferenceInDifferences().fit(
            card_krueger_long,
            outcome="employment",
            treatment="treated",
            post="post",
        )
        br = BusinessReport(
            did,
            outcome_label="FTE employment",
            outcome_unit="FTE",
            treatment_label="the NJ minimum-wage increase",
            outcome_direction="higher_is_better",
            auto_diagnostics=False,
        )
        h = br.to_dict()["headline"]
        # Canonical: positive sign (no disemployment, if anything a
        # small positive lift).
        assert h["sign"] == "positive", (
            f"Card-Krueger canonical finding is a positive ATT; got "
            f"sign={h['sign']!r}, effect={h['effect']!r}"
        )
        # Canonical: CI includes zero -> not statistically significant.
        assert h["ci_lower"] < 0 < h["ci_upper"], (
            f"Card-Krueger canonical finding is CI includes zero; got "
            f"[{h['ci_lower']}, {h['ci_upper']}]"
        )
        assert h["is_significant"] is False
        # BR prose must name this in stakeholder-readable language.
        summary = br.summary().lower()
        assert "consistent with no effect" in summary, (
            f"BR summary must report 'consistent with no effect' on "
            f"Card-Krueger. Got: {summary!r}"
        )

    def test_treatment_label_abbreviation_preserved(self, card_krueger_long):
        """The ``NJ`` abbreviation in the treatment label must survive
        BR's sentence capitalization (regression for the
        ``str.capitalize()`` bug surfaced by this dataset).
        """
        did = DifferenceInDifferences().fit(
            card_krueger_long,
            outcome="employment",
            treatment="treated",
            post="post",
        )
        br = BusinessReport(
            did,
            outcome_label="FTE employment",
            treatment_label="the NJ minimum-wage increase",
            auto_diagnostics=False,
        )
        assert "The NJ minimum-wage increase" in br.headline()


class TestMpdtaCanonicalDirection:
    """Callaway-Sant'Anna benchmark (mpdta): staggered minimum-wage
    increases, log employment outcome.

    Canonical finding: aggregate ATT is negative; the published fit is
    robust under HonestDiD sensitivity; pre-trends do not reject
    parallel trends.
    """

    def test_negative_att_robust_sensitivity_clean_pretrends(self, mpdta_panel):
        cs = CallawaySantAnna(base_period="universal").fit(
            mpdta_panel,
            outcome="lemp",
            unit="countyreal",
            time="year",
            first_treat="first_treat",
            aggregate="event_study",
        )
        br = BusinessReport(
            cs,
            outcome_label="Log employment",
            outcome_unit="log_points",
            treatment_label="the state-level minimum wage increase",
            outcome_direction="higher_is_better",
            data=mpdta_panel,
            outcome="lemp",
            unit="countyreal",
            time="year",
            first_treat="first_treat",
        )
        d = br.to_dict()
        h = d["headline"]
        # Canonical direction: ATT on log employment is negative.
        assert (
            h["sign"] == "negative"
        ), f"mpdta canonical finding is negative ATT; got sign={h['sign']!r}"
        # Canonical robustness: HonestDiD breakdown M > 1 means the
        # result survives violations at least as large as the observed
        # pre-period variation.
        bkd = h.get("breakdown_M")
        assert isinstance(bkd, (int, float)) and bkd > 1.0, (
            f"mpdta canonical finding is robust sensitivity "
            f"(breakdown_M > 1.0); got breakdown_M={bkd!r}"
        )
        # Canonical pre-trends: do not reject PT.
        pt = d["pre_trends"]
        assert pt.get("verdict") == "no_detected_violation", (
            f"mpdta canonical finding is clean pre-trends "
            f"(no_detected_violation); got verdict={pt.get('verdict')!r}"
        )


class TestCastleDoctrineCanonicalDirection:
    """Cheng & Hoekstra (2013): Castle Doctrine / Stand Your Ground
    laws staggered across U.S. states.

    Canonical finding: ~8% INCREASE in homicide rates (no deterrent
    effect; if anything, escalation). Pre-trends violation is a
    well-known issue with this dataset; HonestDiD sensitivity
    flags the headline as fragile.
    """

    def test_cs_positive_att_clear_violation_fragile_sensitivity(self, castle_panel):
        cs = CallawaySantAnna(base_period="universal", control_group="never_treated").fit(
            castle_panel,
            outcome="homicide_rate",
            unit="state",
            time="year",
            first_treat="first_treat",
            aggregate="event_study",
        )
        br = BusinessReport(
            cs,
            outcome_label="Homicide rate (per 100k)",
            treatment_label="Castle Doctrine law adoption",
            outcome_direction="lower_is_better",
            data=castle_panel,
            outcome="homicide_rate",
            unit="state",
            time="year",
            first_treat="first_treat",
        )
        d = br.to_dict()
        h = d["headline"]
        # Canonical direction: homicides went UP (positive ATT).
        assert h["sign"] == "positive", (
            f"Castle Doctrine canonical finding is positive ATT (homicide "
            f"escalation); got sign={h['sign']!r}"
        )
        # Canonical: clear PT violation on this dataset.
        pt = d["pre_trends"]
        assert pt.get("verdict") == "clear_violation", (
            f"Castle Doctrine canonical finding is clear PT violation; "
            f"got verdict={pt.get('verdict')!r}"
        )
        # Canonical: HonestDiD flags fragility given the PT violation.
        sens = d["sensitivity"]
        assert sens.get("status") == "computed"
        bkd = sens.get("breakdown_M")
        assert isinstance(bkd, (int, float)) and bkd < 0.5, (
            f"Castle Doctrine canonical finding is fragile sensitivity "
            f"(breakdown_M < 0.5); got breakdown_M={bkd!r}"
        )

    def test_treatment_label_proper_noun_preserved(self, castle_panel):
        """ "Castle Doctrine" must survive BR's sentence capitalization
        (regression for the ``str.capitalize()`` bug surfaced by this
        dataset).
        """
        cs = CallawaySantAnna(base_period="universal", control_group="never_treated").fit(
            castle_panel,
            outcome="homicide_rate",
            unit="state",
            time="year",
            first_treat="first_treat",
            aggregate="event_study",
        )
        br = BusinessReport(
            cs,
            outcome_label="Homicide rate (per 100k)",
            treatment_label="Castle Doctrine law adoption",
            outcome_direction="lower_is_better",
            auto_diagnostics=False,
        )
        assert "Castle Doctrine law adoption" in br.headline()

    def test_breakdown_m_zero_uses_smallest_m_evaluated_wording(self, castle_panel):
        """Castle Doctrine's fragile sensitivity surfaced a
        ``breakdown_M == 0`` edge case. The default HonestDiD grid
        starts at M=0.5, and every grid point has the robust CI
        including zero — so the smallest-M-evaluated wording is
        semantically accurate here. BR's summary must say ``smallest
        M evaluated on the sensitivity grid (M = 0.5)`` and must not
        quote the degenerate ``0x`` multiplier.
        """
        cs = CallawaySantAnna(base_period="universal", control_group="never_treated").fit(
            castle_panel,
            outcome="homicide_rate",
            unit="state",
            time="year",
            first_treat="first_treat",
            aggregate="event_study",
        )
        br = BusinessReport(
            cs,
            outcome_label="Homicide rate (per 100k)",
            treatment_label="Castle Doctrine law adoption",
            outcome_direction="lower_is_better",
            data=castle_panel,
            outcome="homicide_rate",
            unit="state",
            time="year",
            first_treat="first_treat",
        )
        summary = br.summary()
        bkd = br.to_dict()["headline"].get("breakdown_M")
        # Sanity: this dataset actually produces the edge case.
        assert isinstance(bkd, (int, float)) and bkd <= 0.05, (
            f"This test assumes Castle Doctrine + CS produces "
            f"breakdown_M <= 0.05; if not, the dataset or estimator "
            f"changed. Got breakdown_M={bkd!r}"
        )
        # Must not render the degenerate ``0x`` multiplier.
        assert "0x the pre-period variation" not in summary, (
            f"Summary must not quote ``0x`` multiplier on edge-case " f"breakdown. Got: {summary!r}"
        )
        # Must name the smallest evaluated grid point (0.5 for the
        # default grid).
        assert "smallest M evaluated on the sensitivity grid" in summary
        assert "M = 0.5" in summary


class TestCastleDoctrineCrossEstimatorConsistency:
    """Running the same Castle Doctrine dataset through CS and SA must
    produce consistent direction + PT verdict. SA is a natural
    cross-check on the CS finding.
    """

    def test_sa_agrees_with_cs_on_direction_and_pt(self, castle_panel):
        cs = CallawaySantAnna(base_period="universal", control_group="never_treated").fit(
            castle_panel,
            outcome="homicide_rate",
            unit="state",
            time="year",
            first_treat="first_treat",
            aggregate="event_study",
        )
        sa = SunAbraham().fit(
            castle_panel,
            outcome="homicide_rate",
            unit="state",
            time="year",
            first_treat="first_treat",
        )
        br_cs = BusinessReport(
            cs,
            outcome_label="Homicide rate",
            treatment_label="Castle Doctrine law adoption",
            outcome_direction="lower_is_better",
            data=castle_panel,
            outcome="homicide_rate",
            unit="state",
            time="year",
            first_treat="first_treat",
        )
        br_sa = BusinessReport(
            sa,
            outcome_label="Homicide rate",
            treatment_label="Castle Doctrine law adoption",
            outcome_direction="lower_is_better",
            data=castle_panel,
            outcome="homicide_rate",
            unit="state",
            time="year",
            first_treat="first_treat",
        )
        # Direction must agree: both positive (homicides up).
        assert br_cs.to_dict()["headline"]["sign"] == br_sa.to_dict()["headline"]["sign"]
        assert br_cs.to_dict()["headline"]["sign"] == "positive"
        # PT verdict must agree on the clear-violation bin (both
        # estimators read the same underlying pre-period coefficients).
        assert (
            br_cs.to_dict()["pre_trends"]["verdict"]
            == br_sa.to_dict()["pre_trends"]["verdict"]
            == "clear_violation"
        )
