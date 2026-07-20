"""External-reference parity: ImputationDiD leave-one-out SE vs Stata `did_imputation`.

`ImputationDiD(leave_one_out=True)` applies the Borusyak-Jaravel-Spiess (2024)
Supplementary Appendix A.9 finite-sample variance refinement. No R package computes it
(R `didimputation` omits LOO), so the library LOO SE was validated only by an internal
psi-identity + hand-calc + MC coverage. The authors' own Stata `did_imputation` ships
the same option (`leaveout`); this arm turns that into a measured anchor.

Second Stata parity arm, and the first SSC-dependent one. The golden is produced by
`benchmarks/stata/generate_imputation_loo_golden.do`, which runs `did_imputation ...,
leaveout avgeffectsby(Ei t) cluster(unit)` on the committed R-arm panel
`didimputation_test_panel.csv` (no clean-sample reconstruction - did_imputation takes
the raw panel; the only mapping is Ei = first_treat, missing for never-treated).

The library uses its own sparse IF solver while did_imputation goes through reghdfe, so
agreement is cross-implementation, not bit-identical: the SE agrees to ~1e-9 and the
point to ~2e-8. Gates use ``abs=1e-7, rel=0`` - the repo's imputation SE-parity tolerance
(`test_methodology_imputation.py::test_overall_se_matches_r` asserts 1e-7 despite
observing ~1e-10) - since the golden is committed and this test recomputes the library
on cross-platform CI.

Gates (overall + each event-study horizon):
1. **Point** - library ATET vs Stata ATET (same estimand; overall ~6.5e-11, per-horizon
   up to ~1.8e-8 locally - both well inside the abs=1e-7 gate).
2. **LOO SE anchor** - library LOO SE vs Stata `leaveout` SE (the point of this arm).
3. **Non-LOO SE** - library non-LOO SE vs Stata non-`leaveout` SE (corroboration).
4. **Non-LOO three-way** - Stata golden `se_nonloo` vs the committed R golden `se`
   (library == R == Stata on the same panel; committed-vs-committed).
5. **Warning-cleanliness** - the LOO fit emits no A.9-singleton `UserWarning`, proving
   the LOO rescale is genuinely exercised rather than silently falling back to non-LOO.

Guard per ``feedback_golden_file_pytest_skip``: CI isolated-install jobs copy ``tests/``
only, not ``benchmarks/data/``, so a missing fixture downgrades to pytest.skip rather
than fail. Regenerate (after `benchmarks/stata/requirements.do`) with::

    /Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do \
        benchmarks/stata/generate_imputation_loo_golden.do
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from diff_diff import ImputationDiD

_DATA = Path(__file__).parent.parent / "benchmarks" / "data"
STATA_GOLDEN_PATH = _DATA / "didimputation_loo_stata_golden.json"
R_GOLDEN_PATH = _DATA / "didimputation_golden.json"
PANEL_PATH = _DATA / "didimputation_test_panel.csv"

_FIXTURE_AVAILABLE = (
    STATA_GOLDEN_PATH.is_file() and R_GOLDEN_PATH.is_file() and PANEL_PATH.is_file()
)

HORIZONS = [0, 1, 2, 3, 4, 5]

# Cross-platform tolerance: the library IF variance vs did_imputation (reghdfe) agree
# ~1e-9..1e-10 same-machine; 1e-7 matches the repo's imputation SE-parity convention
# and absorbs cross-platform BLAS variation. rel=0 -> purely absolute bound.
ATOL = 1e-7


def _skip_if_missing() -> None:
    if not _FIXTURE_AVAILABLE:
        pytest.skip(
            "Stata ImputationDiD-LOO parity fixture not present. Regenerate via "
            "`stata-se -b do benchmarks/stata/generate_imputation_loo_golden.do` "
            "(after benchmarks/stata/requirements.do)."
        )


@pytest.fixture(scope="module")
def stata_golden() -> dict:
    _skip_if_missing()
    with STATA_GOLDEN_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def r_golden() -> dict:
    _skip_if_missing()
    with R_GOLDEN_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def library_fit() -> dict:
    """Fit the LOO and non-LOO paths once each; return per-horizon + overall + warnings.

    ImputationDiD exposes ``res.event_study_effects[h]["effect"]`` / ``["se"]`` (a
    dict-of-dicts, NOT a ``.event_study`` DataFrame) and scalar ``res.overall_att`` /
    ``res.overall_se``. ``aggregate="event_study"`` also computes the overall
    unconditionally, so one fit per variance mode suffices.
    """
    _skip_if_missing()
    panel = pd.read_csv(PANEL_PATH)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loo = ImputationDiD(leave_one_out=True).fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        loo_msgs = [str(w.message) for w in caught]
    nonloo = ImputationDiD(leave_one_out=False).fit(
        panel,
        outcome="y",
        unit="unit",
        time="time",
        first_treat="first_treat",
        aggregate="event_study",
    )

    def by_h(res):
        return {h: res.event_study_effects[h] for h in HORIZONS}

    return {
        "loo_overall": (float(loo.overall_att), float(loo.overall_se)),
        "loo_es": {h: (float(v["effect"]), float(v["se"])) for h, v in by_h(loo).items()},
        "nonloo_overall_se": float(nonloo.overall_se),
        "nonloo_es_se": {h: float(v["se"]) for h, v in by_h(nonloo).items()},
        "loo_warnings": loo_msgs,
    }


# ----- Gate 1: point (library vs Stata, same estimand) -----


def test_overall_point_matches_stata(stata_golden, library_fit):
    lib_att, _ = library_fit["loo_overall"]
    st_att = stata_golden["overall"]["att"]
    assert lib_att == pytest.approx(st_att, abs=ATOL, rel=0)


@pytest.mark.parametrize("h", HORIZONS)
def test_event_study_point_matches_stata(h, stata_golden, library_fit):
    lib_att, _ = library_fit["loo_es"][h]
    st_att = stata_golden["event_study"][str(h)]["att"]
    assert lib_att == pytest.approx(st_att, abs=ATOL, rel=0)


# ----- Gate 2: the LOO SE anchor (library vs Stata `leaveout`) -----


def test_overall_loo_se_matches_stata(stata_golden, library_fit):
    _, lib_se = library_fit["loo_overall"]
    st_se = stata_golden["overall"]["se"]
    assert lib_se == pytest.approx(
        st_se, abs=ATOL, rel=0
    ), f"overall LOO SE: library {lib_se} != Stata leaveout {st_se}"


@pytest.mark.parametrize("h", HORIZONS)
def test_event_study_loo_se_matches_stata(h, stata_golden, library_fit):
    _, lib_se = library_fit["loo_es"][h]
    st_se = stata_golden["event_study"][str(h)]["se"]
    assert lib_se == pytest.approx(
        st_se, abs=ATOL, rel=0
    ), f"h={h} LOO SE: library {lib_se} != Stata leaveout {st_se}"


# ----- Gate 3: non-LOO SE corroboration (library vs Stata non-`leaveout`) -----


def test_overall_nonloo_se_matches_stata(stata_golden, library_fit):
    lib_se = library_fit["nonloo_overall_se"]
    st_se = stata_golden["overall"]["se_nonloo"]
    assert lib_se == pytest.approx(st_se, abs=ATOL, rel=0)


@pytest.mark.parametrize("h", HORIZONS)
def test_event_study_nonloo_se_matches_stata(h, stata_golden, library_fit):
    lib_se = library_fit["nonloo_es_se"][h]
    st_se = stata_golden["event_study"][str(h)]["se_nonloo"]
    assert lib_se == pytest.approx(st_se, abs=ATOL, rel=0)


# ----- Gate 4: non-LOO three-way (Stata golden vs committed R golden) -----


def test_overall_nonloo_se_three_way_stata_vs_r(stata_golden, r_golden):
    """Committed-vs-committed: Stata non-LOO SE == R didimputation SE (same panel)."""
    st_se = stata_golden["overall"]["se_nonloo"]
    r_se = r_golden["overall"]["se"]
    assert st_se == pytest.approx(r_se, abs=ATOL, rel=0)


@pytest.mark.parametrize("h", HORIZONS)
def test_event_study_nonloo_se_three_way_stata_vs_r(h, stata_golden, r_golden):
    """Committed-vs-committed, per horizon: Stata non-LOO SE == R didimputation SE."""
    st_se = stata_golden["event_study"][str(h)]["se_nonloo"]
    r_by_h = dict(zip(r_golden["event_study"]["horizons"], r_golden["event_study"]["se"]))
    assert st_se == pytest.approx(r_by_h[h], abs=ATOL, rel=0)


# ----- Gate 5: the LOO rescale is genuinely exercised (no singleton fallback) -----


def test_loo_fit_emits_no_singleton_warning(library_fit):
    """App. A.9 fn. 51: a single-positive-weight group has an undefined LOO and falls
    back with a UserWarning. On this panel every group has >=2 units, so the LOO fit
    must be warning-clean - otherwise the LOO SE would silently equal the non-LOO SE.
    """
    singletons = [
        m
        for m in library_fit["loo_warnings"]
        if "leave-one-out" in m.lower() or "singleton" in m.lower() or "single" in m.lower()
    ]
    assert not singletons, f"unexpected LOO-singleton warning(s): {singletons}"
