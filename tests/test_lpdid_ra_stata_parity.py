"""External-reference parity: LPDiD regression-adjustment SE vs Stata ``teffects ra``.

The LP-DiD (Dube, Girardi, Jorda & Taylor 2025) regression-adjustment (RA) covariate
path reports an influence-function cluster variance with **no finite-sample factor**.
No R package computes it (``alexCardazzi/lpdid`` uses direct covariate inclusion, not
RA), so the canonical reference is Stata ``teffects ra ... atet vce(cluster)``. This is
the repo's FIRST Stata parity arm; ``teffects`` is native to Stata (no SSC dependency),
so the generator's ``version 19`` fully pins the numerical behavior.

The golden is produced by ``benchmarks/stata/generate_lpdid_ra_golden.do``, which
INDEPENDENTLY reconstructs each horizon's clean sample (porting the R
``generate_lpdid_golden.R`` prep/clean_h recipe) and runs ``teffects``. Three gates:

1. **Point** - Stata ATET vs the R-anchored ``ra_cov[h][0]`` (both committed) at
   ``atol=1e-10``, plus a *direct* library-ATET vs Stata-ATET check at the repo's
   cross-platform RA-point tolerance: strongly corroborates that the Stata
   reconstruction used the same clean sample as R and the library (three-way point
   agreement, no longer transitive through R).
2. **Sample shape** - Stata ``(e(N), e(N_clust))`` vs the library's realized
   ``(n_obs, n_clusters)`` (exact): narrows the "point matches on a *different* sample"
   hole - equal size + cluster count alongside the 1e-10 point and ~1e-16 SE agreement
   strongly corroborate the same realized clean sample (strong corroboration, not a
   formal membership proof).
3. **SE anchor** - the library RA IF SE vs the Stata ``teffects`` SE. Same-machine
   agreement is ~1e-16 (the two compute the identical estimator - IF cluster variance,
   no finite-sample factor), but the library side is recomputed on CI and its RA SE is
   held only to ``abs=1e-6`` cross-platform (see ``test_ra_covariate_se_regression_pin``
   / ``RA_SE_PIN``), so the assertion uses ``atol=1e-7`` - the repo's LPDiD R-parity SE
   tolerance - for BLAS/OS robustness.

Guard per ``feedback_golden_file_pytest_skip``: CI isolated-install jobs copy ``tests/``
only, not ``benchmarks/data/``, so a missing fixture downgrades to pytest.skip rather
than fail. Regenerate with::

    /Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do \
        benchmarks/stata/generate_lpdid_ra_golden.do
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from diff_diff import LPDiD

_DATA = Path(__file__).parent.parent / "benchmarks" / "data"
STATA_GOLDEN_PATH = _DATA / "lpdid_ra_stata_golden.json"
R_GOLDEN_PATH = _DATA / "lpdid_golden.json"
PANEL_PATH = _DATA / "lpdid_test_panel.csv"

# Both goldens + the shared panel must be present (the point gate reads the R golden).
_FIXTURE_AVAILABLE = (
    STATA_GOLDEN_PATH.is_file() and R_GOLDEN_PATH.is_file() and PANEL_PATH.is_file()
)

# Horizons pinned by the Stata golden: post {0..4} plus pre placebos {-2, -3};
# h = -1 is the omitted (zero) reference.
HORIZONS = [0, 1, 2, 3, 4, -2, -3]

POINT_ATOL = 1e-10  # committed Stata ATET vs committed R ra_cov point (R digits=12)
SE_ATOL = 1e-7  # library RA SE (recomputed) vs committed Stata SE; cross-platform
# library point (recomputed) vs committed Stata ATET; matches the repo's LPDiD
# RA-point cross-platform tolerance (test_methodology_lpdid.py::test_ra_covariate_point).
POINT_XPLATFORM_ATOL = 1e-6


def _skip_if_missing() -> None:
    if not _FIXTURE_AVAILABLE:
        pytest.skip(
            "Stata LPDiD-RA parity fixture not present. Regenerate via "
            "`stata-se -b do benchmarks/stata/generate_lpdid_ra_golden.do`."
        )


@pytest.fixture(scope="module")
def stata_golden() -> dict:
    _skip_if_missing()
    with STATA_GOLDEN_PATH.open() as f:
        return json.load(f)["ra_se"]


@pytest.fixture(scope="module")
def r_ra_points() -> dict:
    _skip_if_missing()
    with R_GOLDEN_PATH.open() as f:
        return json.load(f)["ra_cov"]


@pytest.fixture(scope="module")
def library_fit() -> dict:
    """Fit the RA path once; return {horizon: {coef, se, n_obs, n_clusters}} + warnings.

    Mirrors the generator's configuration exactly:
    ``LPDiD(pre_window=3, post_window=4, reweight=True, cluster="unit")`` with
    ``covariates=["x"]``.
    """
    _skip_if_missing()
    panel = pd.read_csv(PANEL_PATH)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = LPDiD(pre_window=3, post_window=4, reweight=True, cluster="unit").fit(
            panel,
            covariates=["x"],
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        messages = [str(w.message) for w in caught]
    es = res.event_study.set_index("horizon")
    by_h = {
        int(h): {
            "coef": float(es.loc[h, "coefficient"]),
            "se": float(es.loc[h, "se"]),
            "n_obs": int(es.loc[h, "n_obs"]),
            "n_clusters": int(es.loc[h, "n_clusters"]),
        }
        for h in HORIZONS
    }
    return {"by_h": by_h, "warnings": messages}


@pytest.mark.parametrize("h", HORIZONS)
def test_stata_point_matches_r_anchor(h, stata_golden, r_ra_points):
    """Gate 1: the independent Stata reconstruction reproduces the R-anchored ATET.

    ``rel=0`` makes this a purely absolute ``abs=POINT_ATOL`` bound, independent of
    pytest's default relative tolerance.
    """
    stata_att = stata_golden[str(h)]["att"]
    r_att = r_ra_points[str(h)][0]  # ra_cov[h] = [att, conditional_CR0_se_ref_only, null]
    assert stata_att == pytest.approx(r_att, abs=POINT_ATOL, rel=0), (
        f"h={h}: Stata ATET {stata_att} != R anchor {r_att} - the Stata clean-sample "
        "reconstruction diverged from R; the SE anchor below would be meaningless."
    )


@pytest.mark.parametrize("h", HORIZONS)
def test_stata_sample_shape_matches_library(h, stata_golden, library_fit):
    """Gate 2: Stata (e(N), e(N_clust)) == library (n_obs, n_clusters), exact.

    Equal size AND cluster count do not by themselves *prove* identical (unit, time)
    membership, but combined with the 1e-10 point gate and the ~1e-16 SE match they
    provide strong corroboration of the same realized clean sample.
    """
    stata_n = int(stata_golden[str(h)]["N"])
    stata_g = int(stata_golden[str(h)]["G"])
    lib = library_fit["by_h"][h]
    assert (stata_n, stata_g) == (lib["n_obs"], lib["n_clusters"]), (
        f"h={h}: Stata (N={stata_n}, G={stata_g}) != library "
        f"(n_obs={lib['n_obs']}, n_clusters={lib['n_clusters']}) - the clean samples differ."
    )


@pytest.mark.parametrize("h", HORIZONS)
def test_library_point_matches_stata_teffects(h, stata_golden, library_fit):
    """Direct Python<->Stata ATET agreement (not transitive through the R golden).

    Gate 1 pins Stata vs R (committed values); this closes the loop by comparing the
    recomputed library ATET directly to the Stata ATET, at the repo's cross-platform
    RA-point tolerance (``rel=0`` -> purely absolute).
    """
    stata_att = stata_golden[str(h)]["att"]
    lib_coef = library_fit["by_h"][h]["coef"]
    assert lib_coef == pytest.approx(
        stata_att, abs=POINT_XPLATFORM_ATOL, rel=0
    ), f"h={h}: library ATET {lib_coef} != Stata teffects ATET {stata_att}"


def test_library_fit_emits_no_ra_drop_warning(library_fit):
    """Gate 2 (companion): no treated observation was dropped for non-identification.

    A drop would mean the library and Stata clean samples could diverge silently; on
    the committed panel every horizon is identified (G=60, all treated cells have a
    clean control), so the fit must be warning-clean.
    """
    drops = [m for m in library_fit["warnings"] if "regression adjustment: dropped" in m]
    assert not drops, f"unexpected RA drop warning(s): {drops}"


@pytest.mark.parametrize("h", HORIZONS)
def test_library_ra_se_matches_stata_teffects(h, stata_golden, library_fit):
    """Gate 3 (the anchor): library RA IF SE == Stata teffects atet cluster SE.

    Confirms the no-finite-sample-factor RA convention (REGISTRY LPDiD Deviation 2)
    against an independent implementation. Local agreement is ~1e-16; SE_ATOL is the
    repo's cross-platform LPDiD SE tolerance (see module docstring). ``rel=0`` makes
    this a purely absolute ``abs=SE_ATOL`` bound, independent of pytest's default rel.
    """
    stata_se = stata_golden[str(h)]["se"]
    lib_se = library_fit["by_h"][h]["se"]
    assert lib_se == pytest.approx(
        stata_se, abs=SE_ATOL, rel=0
    ), f"h={h}: library RA SE {lib_se} != Stata teffects SE {stata_se}"
