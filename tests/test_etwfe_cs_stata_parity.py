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

    ``block`` selects the arm: ``"jwdid"`` (not-yet-treated, the default),
    ``"jwdid_never"`` (never-treated controls, the issue #724 anchor), or
    ``"jwdid_alltreated"`` (the W2025 Section 5.4 anchor). All use the same
    coefficient-name shape; the all-treated block nests its cells under
    ``"cells"`` because it also records ``n`` / ``n_units``.
    """
    raw = golden[block]
    if "cells" in raw:
        raw = raw["cells"]
    out = {}
    for key, rec in raw.items():
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
    assert _stata_jwdid_cells(golden, "jwdid_alltreated"), "golden has no jwdid_alltreated cells"
    # The all-treated arm's row count is part of its finding, not incidental.
    assert golden["jwdid_alltreated"]["n"] > 0
    assert golden["jwdid_alltreated"]["n_units"] > 0
    # Every arm records the exact command that produced it.
    for key in ("csdid_cmd", "jwdid_cmd", "jwdid_never_cmd", "jwdid_alltreated_cmd"):
        assert golden["meta"].get(key), f"golden meta is missing {key}"
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

    def test_se_matches_jwdid_never_treated_arm(self, never_fit):
        """SEs now reproduce Stata jwdid (reghdfe) at machine precision.

        The historical 1.00100634 SE ratio on this arm was the D2 defect:
        the clustered CR1 factor's ``k`` omitted the absorbed FE not nested
        in the unit cluster. Under the K_reference convergence
        (variance-conventions.md, 3.9 program) all 12 never-treated cells
        sit at ratio 1.0 with spread ~1e-14 — the same machine-precision
        agreement the not-yet-treated and all-eventually-treated arms show.
        Two-sided by design: a strict inequality on a quantity at
        1.0 ± float noise would be a coin flip.
        """
        stata = _stata_jwdid_cells(_golden(), "jwdid_never")
        ratios = [rec["se"] / never_fit.group_time_effects[key]["se"] for key, rec in stata.items()]
        assert len(ratios) == 12, f"expected 12 never_treated cells, got {len(ratios)}"
        assert max(ratios) - min(ratios) < 1e-9, f"SE ratio is not uniform: {ratios}"
        np.testing.assert_allclose(float(np.mean(ratios)), 1.0, rtol=1e-9)


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

    def test_se_matches_jwdid(self, fits):
        """SEs reproduce Stata jwdid (reghdfe 3.2.9) at machine precision.

        The historical uniform 1.001006 gap on this G=500 arm was defect D2:
        the clustered CR1 factor used only the visible treatment-cell count,
        omitting the absorbed unit/time FE not nested in the unit cluster.
        The mechanism was derived in closed form —
        ``K_reference = cells + T`` on this no-intercept within design — and
        shipped as the 3.9 K_reference convergence
        (docs/methodology/variance-conventions.md); all 7 cells now sit at
        ratio 1.0 with spread ~1e-14. Two-sided by design (a strict
        inequality at 1.0 ± float noise is a coin flip). The subsample
        LADDER (G≈20..500) gates the few-cluster behavior separately.
        """
        etwfe, _ = fits
        stata = _stata_jwdid_cells(_golden())

        ratios = [rec["se"] / etwfe.group_time_effects[key]["se"] for key, rec in stata.items()]
        assert len(ratios) == 7
        assert max(ratios) - min(ratios) < 1e-9, f"SE ratio is not uniform: {ratios}"
        np.testing.assert_allclose(float(np.mean(ratios)), 1.0, rtol=1e-9)


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
        """CS SEs agree with Stata csdid outright (IF variance, no CR1
        factor). Historically this contrast is what isolated the ETWFE SE
        gap as a CR1-k accounting defect rather than a library-wide
        convention difference; both estimators now match their references."""
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


@pytest.fixture(scope="module")
def alltreated_fit():
    """The pinned panel with never-treated counties dropped: 191 units, 955 rows.

    Mirrors the generator's `drop if first_treat == 0` exactly, so the library
    and Stata see the same frame.
    """
    df = _panel()
    sub = df[df["first_treat"] != 0].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WooldridgeDiD(method="ols", control_group="not_yet_treated").fit(
            sub, outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
        )


class TestAllEventuallyTreatedVsStataJwdid:
    """External anchor for W2025 Section 5.4 on a real all-eventually-treated panel.

    Stata `jwdid` succeeds here and silently estimates on a reduced sample: the
    2007 cohort becomes the reference, the fully-treated periods carry no
    identified ATT, and jwdid reports only a smaller N. The library computes the
    same cell set on the same rows -- and says so.

    Uses `control_group="not_yet_treated"`, the only option available: with no
    cohort-0 rows left, `never_treated` raises.
    """

    def test_cell_set_matches_jwdid(self, alltreated_fit):
        """The Section 5.4 cell set is an EXTERNAL result, not our convention."""
        stata = _stata_jwdid_cells(_golden(), "jwdid_alltreated")
        assert sorted(stata) == [(2004, 2004), (2004, 2005), (2004, 2006), (2006, 2006)]
        lib = {(int(g), int(t)) for g, t in alltreated_fit.group_time_effects}
        assert lib == set(stata)
        # The last cohort is the reference and receives nothing.
        assert 2007 not in {g for g, _t in lib}

    def test_att_matches_jwdid(self, alltreated_fit):
        """Point estimates are the tight gate: agreement is to machine precision."""
        stata = _stata_jwdid_cells(_golden(), "jwdid_alltreated")
        for key, rec in stata.items():
            np.testing.assert_allclose(
                alltreated_fit.group_time_effects[key]["att"], rec["att"], atol=1e-12
            )

    def test_estimation_sample_size_matches_jwdid(self, alltreated_fit):
        """The row count IS the finding -- 764 of 955, i.e. the fully-treated
        periods are gone from both implementations' estimation samples.

        Pinned because a silent divergence here would mean the two are
        estimating on different data while agreeing on coefficients by luck.
        """
        golden = _golden()["jwdid_alltreated"]
        assert golden["n_units"] == 191
        assert golden["n"] == 764
        assert alltreated_fit.n_obs == golden["n"]

    def test_se_matches_jwdid_at_this_cluster_count(self, alltreated_fit):
        """The G=191 arm matches jwdid at machine precision too.

        Pre-fix this arm showed the LARGEST pinned gap (1.00264201 at G=191
        vs 1.001006 at G=500 — the D2 defect is cluster-count dependent),
        which is exactly why it gets its own gate rather than inheriting the
        sibling constant. Under the K_reference convergence all 4 cells sit
        at ratio 1.0 with spread ~1e-15, confirming the fix holds at a
        materially smaller G than the full panel. Two-sided by design.
        """
        stata = _stata_jwdid_cells(_golden(), "jwdid_alltreated")
        ratios = [
            rec["se"] / alltreated_fit.group_time_effects[key]["se"] for key, rec in stata.items()
        ]
        assert len(ratios) == 4, f"expected 4 all-treated cells, got {len(ratios)}"
        assert max(ratios) - min(ratios) < 1e-9, f"SE ratio is not uniform: {ratios}"
        np.testing.assert_allclose(float(np.mean(ratios)), 1.0, rtol=1e-9)


_LADDER_RUNGS = [5, 10, 20, 40, 80, 200, 500]


@pytest.fixture(scope="module")
def ladder():
    """The golden's ladder block. Module-level: a class-scoped fixture
    defined as an instance method is deprecated (PytestRemovedIn10Warning),
    and the @staticmethod form breaks pytest COLLECTION on Python 3.9
    (staticmethod has no __name__ there) — the library floor's CI leg."""
    return _golden()["ladder"]


def _ladder_subsample(df: pd.DataFrame, n_per_cohort: int) -> pd.DataFrame:
    """The generator's roster rule, verbatim: the first ``n_per_cohort`` units
    per ``first_treat`` cohort by ascending ``countyreal``."""
    units = df.drop_duplicates("countyreal")[["countyreal", "first_treat"]]
    keep = set(
        units.sort_values("countyreal").groupby("first_treat").head(n_per_cohort)["countyreal"]
    )
    return df[df["countyreal"].isin(keep)].copy()


class TestSubsampleLadderVsStataJwdid:
    """The K_reference gate at every cluster count (the ``ladder`` block).

    The D2 defect was cluster-count dependent — measured SE ratios of 1.0280
    at G=20, 1.0132 at G=40, 1.00264 at G=191, 1.0010 at G=500 — so a
    full-panel gate alone could mask a partial fix that only converges at
    large G. Each rung refits the library on the identical roster and pins
    the ratio at 1.0 (observed spreads ~1e-15..1e-14 across G=20..500).

    The ladder doubles as the K-ACCOUNTING probe: reghdfe's own df
    decomposition is recorded per rung, and ``df_a`` must equal the library's
    ``absorbed_fe_cr1_k_increment`` minus the reported constant — the two
    implementations agreeing on WHY the factor is what it is, not just on
    the resulting number.
    """

    def test_rung_set_and_roster_rule_are_pinned(self, ladder):
        """A changed roster rule or panel would move G/n; pin both."""
        assert sorted(int(k) for k in ladder["rungs"]) == _LADDER_RUNGS
        expected_G = {5: 20, 10: 40, 20: 80, 40: 140, 80: 220, 200: 391, 500: 500}
        df = _panel()
        for n_per_cohort, rung in ((int(k), v) for k, v in ladder["rungs"].items()):
            sub = _ladder_subsample(df, n_per_cohort)
            assert rung["G"] == expected_G[n_per_cohort]
            assert sub["countyreal"].nunique() == rung["G"]
            assert len(sub) == rung["n"]

    @pytest.mark.parametrize("n_per_cohort", _LADDER_RUNGS)
    def test_se_and_att_match_jwdid_at_every_cluster_count(self, ladder, n_per_cohort):
        rung = ladder["rungs"][str(n_per_cohort)]
        sub = _ladder_subsample(_panel(), n_per_cohort)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = WooldridgeDiD(method="ols", control_group="not_yet_treated").fit(
                sub, outcome="lemp", unit="countyreal", time="year", cohort="first_treat"
            )
        stata = _stata_jwdid_cells({"rung": rung["cells"]}, "rung")
        assert len(stata) == 7, f"expected 7 cells at G={rung['G']}, got {len(stata)}"
        for key, rec in stata.items():
            np.testing.assert_allclose(
                fit.group_time_effects[key]["att"],
                rec["att"],
                rtol=0,
                atol=1e-9,
                err_msg=f"ladder G={rung['G']} ATT{key} != Stata jwdid",
            )
        ratios = [rec["se"] / fit.group_time_effects[key]["se"] for key, rec in stata.items()]
        assert max(ratios) - min(ratios) < 1e-9, f"G={rung['G']}: SE ratio is not uniform: {ratios}"
        np.testing.assert_allclose(float(np.mean(ratios)), 1.0, rtol=1e-9)

    @pytest.mark.parametrize("n_per_cohort", _LADDER_RUNGS)
    def test_stata_df_accounting_matches_the_library_increment(self, ladder, n_per_cohort):
        """reghdfe's df_a == absorbed_fe_cr1_k_increment - 1 at every rung.

        reghdfe reports the absorbed constant separately (``_cons`` in e(b),
        ``report_constant``), so its ``df_a`` covers only the FE ranks beyond
        it: df_a_initial (all FE levels) minus the cluster-nested unit FE and
        the shared constant. The library folds that constant into the
        increment (``has_intercept_col=False`` -> +1), hence the -1. On this
        panel: increment = 1 + (G + 5 - 1) - G = 5 = T at every G, i.e.
        K_reference = 7 cells + 5 = 12.
        """
        from diff_diff.utils import absorbed_fe_cr1_k_increment

        rung = ladder["rungs"][str(n_per_cohort)]
        sub = _ladder_subsample(_panel(), n_per_cohort)
        increment = absorbed_fe_cr1_k_increment(
            sub,
            ["countyreal", "year"],
            sub["countyreal"].to_numpy(),
            has_intercept_col=False,
        )
        assert rung["df_a"] == increment - 1
        # The decomposition pins WHICH FE were dropped: all G unit FE are
        # cluster-nested; the redundant count adds the shared constant.
        assert rung["df_a_nested"] == rung["G"]
        assert rung["df_a_redundant"] == rung["G"] + 1
        assert rung["df_a_initial"] == rung["G"] + 5
        # Visible rank = the 7 treatment cells; cluster df = G-1.
        assert rung["rank"] == 7
        assert rung["df_r"] == rung["G"] - 1
