"""Stata parity for WooldridgeDiD (ETWFE) and CallawaySantAnna ATT(g,t).

Anchors BOTH staggered estimators against their canonical Stata implementations
-- ``jwdid`` (Rios-Avila, Wooldridge ETWFE) and ``csdid`` (Rios-Avila /
Sant'Anna, Callaway-Sant'Anna) -- on the genuine ``mpdta`` panel.

WHY THIS ARM EXISTS
-------------------
``tests/test_wooldridge.py`` previously asserted that ETWFE ATT(g,t) EQUALS
CallawaySantAnna ATT(g,t) within ``5e-3``. That claim is FALSE on real data: at
``(g=2007, t=2007)`` the estimators differ by 0.0171 (-0.0431 vs -0.0261). The
assertion only ever passed because ``load_mpdta()`` was silently substituting a
synthetic, effect-homogeneous DGP when its source URL 404'd (issue #722), and on
that DGP the two estimators do coincide.

Stata's ``jwdid`` and ``csdid`` reproduce the SAME disagreement, which is what
establishes it as a property of the estimators rather than a bug in either
implementation. So instead of one self-referential cross-check that validated
nothing, each estimator is pinned to its own external reference, and the
ETWFE-vs-CS gap is RECORDED rather than asserted away.

The golden is committed (``benchmarks/data/etwfe_cs_stata_golden.json``) and the
panel is committed alongside it, so CI never needs Stata or network access --
network dependence being the exact failure mode that produced the false
assertion in the first place. Regenerate with
``benchmarks/stata/generate_etwfe_cs_golden.do``.
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, WooldridgeDiD

_GOLDEN = Path(__file__).parent.parent / "benchmarks" / "data" / "etwfe_cs_stata_golden.json"
_PANEL = Path(__file__).parent.parent / "benchmarks" / "data" / "mpdta_stata_panel.csv"

#: Upstream ``mpdta.csv`` digest. Pinned identically in the generator and in
#: ``diff_diff/datasets.py``; asserted here so a swapped panel cannot silently
#: retarget the parity.
_PANEL_SHA256 = "2283bea1221a152420f98dfa20f633c5d054ea51d881115c8cd702a97bcd3167"

pytestmark = pytest.mark.skipif(
    not (_GOLDEN.exists() and _PANEL.exists()),
    reason="Stata golden/panel not present (regenerate with benchmarks/stata/generate_etwfe_cs_golden.do)",
)


def _panel() -> pd.DataFrame:
    df = pd.read_csv(_PANEL)
    return df.rename(columns={"first.treat": "first_treat"})


def _golden() -> dict:
    with open(_GOLDEN) as fh:
        return json.load(fh)


def _stata_csdid_cells(golden: dict) -> dict:
    """``g2004:t_2003_2007`` -> ``(2004, 2007)``, post-treatment cells only."""
    out = {}
    for key, rec in golden["csdid"].items():
        m = re.match(r"g(\d+):t_(\d+)_(\d+)$", key)
        if not m:
            continue
        g, _base, t = (int(x) for x in m.groups())
        if t >= g:  # pre-treatment placebo cells are not compared here
            out[(g, t)] = rec
    return out


def _stata_jwdid_cells(golden: dict, block: str = "jwdid") -> dict:
    """``2004bn.first_treat#2005.year#c.__tr__`` -> ``(2004, 2005)``.

    ``block`` selects the arm: ``"jwdid"`` (not-yet-treated, the default) or
    ``"jwdid_never"`` (never-treated controls, the issue #724 anchor). Both use
    the same coefficient-name shape.
    """
    out = {}
    for key, rec in golden[block].items():
        m = re.match(r"(\d+)b?n?\.first_treat#(\d+)b?n?\.year", key)
        if not m:
            continue  # _cons and any non-cell terms
        g, t = (int(x) for x in m.groups())
        out[(g, t)] = rec
    return out


@pytest.fixture(scope="module")
def fits():
    df = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        etwfe = WooldridgeDiD(method="ols", control_group="not_yet_treated").fit(
            df, outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
        )
        cs = CallawaySantAnna(control_group="not_yet_treated").fit(
            df, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
    return etwfe, cs


def test_panel_matches_the_pinned_upstream_digest():
    """The parity is only meaningful against the authentic panel."""
    import hashlib

    got = hashlib.sha256(_PANEL.read_bytes()).hexdigest()
    assert got == _PANEL_SHA256, f"committed panel digest {got} != pinned {_PANEL_SHA256}"


def test_golden_records_every_reference_implementation():
    golden = _golden()
    assert _stata_csdid_cells(golden), "golden has no csdid cells"
    assert _stata_jwdid_cells(golden, "jwdid"), "golden has no jwdid cells"
    assert _stata_jwdid_cells(golden, "jwdid_never"), "golden has no jwdid_never cells"
    assert golden["meta"]["source_sha256"] == _PANEL_SHA256


def test_golden_records_a_usable_drift_identifier_for_every_ssc_dependency():
    """The SSC packages are UNPINNED, so this metadata is the only thing between
    a silent upstream change and an unexplained golden diff.

    ``"MISSING"`` means the package was absent at generation time; ``"unknown"``
    means its ``.ado`` carried no parseable ``*!`` version header, which
    silently disables drift detection for that dependency -- ``drdid`` was in
    exactly that state. The generator now falls back to a checksum of the
    ``.ado``, which is opaque but still changes when upstream does. Codex R6.
    """
    versions = _golden()["meta"]["ssc_versions"]
    assert set(versions) == {"drdid", "csdid", "jwdid", "hdfe"}
    for pkg, ident in versions.items():
        assert ident not in ("unknown", "MISSING", "", None), (
            f"SSC dependency {pkg!r} has no usable drift identifier ({ident!r}); "
            "regenerate the golden with benchmarks/stata/generate_etwfe_cs_golden.do"
        )


@pytest.fixture(scope="module")
def never_fit():
    """``never_treated`` OLS on the authentic panel - the issue #724 surface."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WooldridgeDiD(method="ols", control_group="never_treated").fit(
            _panel(), outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
        )


class TestNeverTreatedVsStataJwdid:
    """Issue #724: the library returned 5 of 7 post-treatment cells because the
    ETWFE reference period was left to QR rank detection instead of being
    omitted explicitly. ``jwdid ... never`` is the external anchor -- it omits
    ``g-1`` per cohort, matching Wooldridge (2025) Eq. 6.1/6.4.
    """

    def test_cell_set_matches_jwdid_exactly(self, never_fit):
        """The actual #724 regression: WHICH cells come back, not their values.

        Asserted as a SET, never against a column name -- which cell QR
        sacrificed was panel-dependent, so naming one would pin an artifact of
        this fixture rather than the contract.
        """
        stata = set(_stata_jwdid_cells(_golden(), "jwdid_never"))
        ours = set(never_fit.group_time_effects)
        assert ours == stata, (
            f"cell set differs from jwdid: only-ours={sorted(ours - stata)}, "
            f"only-stata={sorted(stata - ours)}"
        )

    def test_all_post_treatment_cells_present(self, never_fit):
        """Pre-fix this returned 5 of 7, with (2004, 2004) absent entirely."""
        post = {k for k in never_fit.group_time_effects if k[1] >= k[0]}
        assert post == {
            (2004, 2004),
            (2004, 2005),
            (2004, 2006),
            (2004, 2007),
            (2006, 2006),
            (2006, 2007),
            (2007, 2007),
        }

    def test_no_rank_deficiency(self):
        """The reference is now omitted by construction, so nothing is left for
        QR to drop -- the warning that accompanied #724 must be gone."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WooldridgeDiD(method="ols", control_group="never_treated").fit(
                _panel(), outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
            )
        rank = [w for w in caught if "Rank-deficient" in str(w.message)]
        assert rank == [], f"unexpected rank deficiency: {[str(w.message)[:120] for w in rank]}"

    def test_omitted_cell_is_the_g_minus_1_reference(self, never_fit):
        """W2025 Eq. 6.1 omits ``g-1``; jwdid does the same. Derived from the
        golden rather than hardcoded, so it tracks the reference implementation.
        """
        stata = _stata_jwdid_cells(_golden(), "jwdid_never")
        by_cohort: dict = {}
        for g, t in stata:
            by_cohort.setdefault(g, set()).add(t)
        all_times = {2003, 2004, 2005, 2006, 2007}
        for g, times in by_cohort.items():
            missing = all_times - times
            assert missing == {g - 1}, f"cohort {g} omits {missing}, expected {{{g - 1}}}"
            assert (g, g - 1) not in never_fit.group_time_effects

    def test_att_matches_jwdid(self, never_fit):
        stata = _stata_jwdid_cells(_golden(), "jwdid_never")
        for key, rec in stata.items():
            np.testing.assert_allclose(
                never_fit.group_time_effects[key]["att"],
                rec["att"],
                rtol=0,
                atol=1e-9,
                err_msg=f"never_treated ATT{key} != Stata jwdid never",
            )

    def test_se_gap_matches_the_not_yet_treated_arm(self, never_fit):
        """The SE gap is a control-group-INDEPENDENT finite-sample factor.

        The golden stores ``jwdid_never`` SEs, so the path this PR actually
        changed gets its own SE gate rather than inheriting the not-yet-treated
        one. Measured here across all 12 never-treated cells (7 post + 5
        placebo): the ratio is uniform to ~1e-14 -- two orders tighter than the
        1e-5 the sibling test allows -- and its mean, 1.00100634, is the SAME
        factor the not-yet-treated arm shows.

        That agreement is the point. Whatever produces the gap does NOT depend
        on the control group, on the cell being post-treatment vs placebo, or
        on the cell count (12 here vs 7 there), which constrains any mechanism
        later proposed for it (tracked in TODO.md). As with the sibling test,
        no closed form is asserted -- only the observed factor.
        """
        stata = _stata_jwdid_cells(_golden(), "jwdid_never")
        ratios = [rec["se"] / never_fit.group_time_effects[key]["se"] for key, rec in stata.items()]
        assert len(ratios) == 12, f"expected 12 never_treated cells, got {len(ratios)}"
        assert max(ratios) - min(ratios) < 1e-5, f"SE ratio is not uniform: {ratios}"
        mean_ratio = float(np.mean(ratios))
        assert mean_ratio > 1.0, "library SE should be below jwdid's, not above"
        np.testing.assert_allclose(mean_ratio, 1.001006, rtol=1e-4)


class TestETWFEvsStataJwdid:
    def test_att_matches_jwdid(self, fits):
        """Point estimates are the tight gate."""
        etwfe, _ = fits
        stata = _stata_jwdid_cells(_golden())
        compared = 0
        for key, rec in stata.items():
            assert key in etwfe.group_time_effects, f"library is missing cell {key}"
            np.testing.assert_allclose(
                etwfe.group_time_effects[key]["att"],
                rec["att"],
                rtol=0,
                atol=1e-6,
                err_msg=f"ETWFE ATT{key} != Stata jwdid",
            )
            compared += 1
        assert compared == 7, f"expected 7 ETWFE cells, compared {compared}"

    def test_se_is_uniformly_below_jwdid(self, fits):
        """MEASURED deviation, recorded without asserting a mechanism.

        Every ``hc1`` SE is smaller than ``jwdid``'s by a factor that is
        UNIFORM across cells (spread < 1e-6 within a fit), so it is a
        finite-sample convention difference rather than noise or a per-cell
        bug. The magnitude shrinks as the cluster count grows -- 1.0280 at
        G=20, 1.0132 at G=40, 1.0010 at G=500 -- so the library is
        systematically anti-conservative relative to the reference, negligibly
        with many clusters and materially with few.

        Deliberately NOT asserted: a closed form. The gap tracks
        ``sqrt(G/(G-1))`` closely but is consistently ABOVE it (by ~0.2% at
        G=20, ~0.001% at G=500), and ``solve_ols`` already applies
        ``(G/(G-1)) * ((n-1)/(n-k))`` -- so "the library omits G/(G-1)" is
        NOT the explanation, and no verified formula has been derived. Pinning
        a wrong mechanism here would be worse than pinning none: this test
        locks the OBSERVED gap at this panel's G, and the derivation is tracked
        in TODO.md. See the REGISTRY WooldridgeDiD note.

        SCOPE: this pins the FULL-PANEL ratio (G=500) only. The smaller-G
        figures quoted in the REGISTRY note were measured ad hoc on subsampled
        panels and are not gated here, so a regression in the few-cluster
        behavior -- where the gap is materially largest -- would not fail CI.
        Committing that ladder is listed as a required artifact on the
        derivation row in TODO.md.
        """
        etwfe, _ = fits
        stata = _stata_jwdid_cells(_golden())

        ratios = [rec["se"] / etwfe.group_time_effects[key]["se"] for key, rec in stata.items()]
        assert len(ratios) == 7
        # Uniform across every cell: a finite-sample factor, not noise.
        assert max(ratios) - min(ratios) < 1e-5, f"SE ratio is not uniform: {ratios}"
        # Stata is LARGER (we are anti-conservative), by the amount observed on
        # this G=500 panel. Regenerating against a different panel moves this.
        mean_ratio = float(np.mean(ratios))
        assert mean_ratio > 1.0, "library SE should be below jwdid's, not above"
        np.testing.assert_allclose(mean_ratio, 1.001006, rtol=1e-4)


class TestCallawaySantAnnaVsStataCsdid:
    def test_att_matches_csdid(self, fits):
        _, cs = fits
        stata = _stata_csdid_cells(_golden())
        compared = 0
        for key, rec in stata.items():
            assert key in cs.group_time_effects, f"library is missing cell {key}"
            np.testing.assert_allclose(
                cs.group_time_effects[key]["effect"],
                rec["att"],
                rtol=0,
                atol=1e-6,
                err_msg=f"CS ATT{key} != Stata csdid",
            )
            compared += 1
        assert compared == 7, f"expected 7 CS cells, compared {compared}"

    def test_se_matches_csdid(self, fits):
        """Unlike ETWFE, the CS SEs agree outright - which is what makes the
        ETWFE SE gap a real finding rather than a convention difference the
        whole library shares."""
        _, cs = fits
        stata = _stata_csdid_cells(_golden())
        for key, rec in stata.items():
            np.testing.assert_allclose(
                cs.group_time_effects[key]["se"],
                rec["se"],
                rtol=1e-5,
                atol=0,
                err_msg=f"CS SE{key} != Stata csdid",
            )


def test_etwfe_and_cs_genuinely_disagree_on_real_data(fits):
    """The claim the retired cross-check got wrong.

    Both estimators match their own reference implementation exactly, and Stata's
    two implementations disagree with each other by the same amount - so this
    divergence is a property of the estimands, not a defect. Pinned so nobody
    re-derives an equivalence assertion from a homogeneous synthetic DGP.
    """
    etwfe, cs = fits
    shared = set(etwfe.group_time_effects) & set(cs.group_time_effects)
    gaps = {
        k: abs(etwfe.group_time_effects[k]["att"] - cs.group_time_effects[k]["effect"])
        for k in shared
    }
    worst_key = max(gaps, key=lambda k: gaps[k])
    assert worst_key == (2007, 2007), f"expected the terminal cell to diverge most, got {worst_key}"
    # Far beyond the 5e-3 the retired test asserted.
    assert gaps[worst_key] > 5e-3
    np.testing.assert_allclose(gaps[worst_key], 0.017052, rtol=1e-3)
