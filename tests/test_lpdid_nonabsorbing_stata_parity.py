"""External-reference parity: LPDiD non-absorbing SEs vs the authors' Stata ``lpdid``.

The non-absorbing LP-DiD modes (Dube, Girardi, Jorda & Taylor 2025, JAE Eq. 12 /
Eq. 13) were R-parity-locked only on the event-study surfaces the R arm covers
(independent ``fixest::feols`` reconstruction; no Eq. 12 reweighted arm, no pooled
block). The canonical reference is the authors' Stata ``lpdid`` package (SSC), which
this arm runs END-TO-END - the package builds its own clean samples. This anchors,
for the first time externally: the non-absorbing REWEIGHTED SE (previously
pinned-only via ``RW_SE_PIN``), the non-absorbing POOLED windows (points and SEs),
and the Eq. 12 reweighted point.

Mapping (pinned in the golden ``meta`` and gated below)::

    lpdid, nonabsorbing(, firsttreat notyet)  ==  LPDiD(non_absorbing="first_entry")
    lpdid, nonabsorbing(L)                    ==  LPDiD(non_absorbing="effect_stabilization",
                                                        stabilization_window=L)

Parity scoping. Eq. 12 (pseudo-absorbing) agrees on EVERY surface of the full
committed panel - all event-study horizons including placebos, plus both pooled
windows, vw and rw. Eq. 13 agrees on POST horizons + pooled post only, and only on a
convention-neutral SUBSAMPLE, because the package's sample construction differs from
the library's on three measured conventions (REGISTRY ``## LPDiD`` Deviation 4):

1. *Boundary*: the package's missing-lag semantics admit always-treated units as
   controls at early t; the library's pre-panel-untreated clamp excludes them until
   t >= L+2 (a paper-silent surface - both are conventions).
2. *Exact-L respell*: the package's switch-free lag window effectively requires L+1
   untreated periods before re-entry; the library's levels reading of Eq. 13
   (untreated over [t-L, t-1]) admits re-entry after an exactly-L untreated spell
   (the package is stricter than the paper's stated levels condition).
3. *Placebo windows*: the package builds pre-horizon clean samples by recursive
   lagged intersection of its CCS_0 indicator; the library uses the backward window
   [t-max(L,-h), t-1] (paper-silent; both are conventions).

The subsample drops the units classes 1+2 admit differently (always-treated
{31..40}; exact-L respell {24, 25, 27}), making the Eq. 13 post-horizon and
pooled-post samples row-identical across implementations. Class 3 is NOT
neutralized by this subsample (a dedicated late-entry/never-treated subsample
could in principle align the two placebo constructions - a possible follow-up,
tracked in DEFERRED.md), so Eq. 13 placebo/pooled-pre rows are recorded as
measured DIVERGENCE documentation (att + obs only) and gated as such (gate 6), never
as parity. Note the per-class attribution is established by unit-set enumeration
(gate 5 asserts each sub-class list separately); gate 6(a) documents the JOINT
classes-1+2 divergence - no separately-measured per-class numeric effect is claimed.

Gates:

1. **Stata Eq. 12 vw vs the committed R golden** ``first_entry`` (committed vs
   committed, ``abs=1e-9``): the package's end-to-end run reproduces the independent
   feols reconstruction.
2. **Library ``first_entry`` vs Stata**, vw + rw: all ES horizons INCLUDING placebos
   + pooled pre/post (att ``abs=1e-6``, SE ``abs=1e-7``).
3. **Library ``effect_stabilization`` subsample vs Stata**, vw + rw: post horizons +
   pooled post (same tolerances).
4. **Sample shape**: Stata per-row obs == library ``n_obs``, exact, on every parity
   surface of gates 2-3. The package's single multi-horizon call returns no
   per-horizon cluster count (``e(results)`` columns are
   coefficient/se/t/p/ci_low/ci_high/obs), so - unlike the ``teffects`` arm - there
   is no per-horizon ``e(N_clust)`` to gate alongside obs; the obs equality plus the
   1e-14-class att/SE agreement carry the same-sample corroboration.
5. **Drop-rule consistency**: golden ``dropped_units``/``n_rows`` vs the pandas
   recompute (each convention sub-class asserted separately), and the free-text
   ``drop_rule``/``mapping`` meta strings vs hard-coded literals.
6. **Convention-divergence documentation**: every recorded divergence surface stays
   measurably divergent - (a) Eq. 13 full-panel att + obs vs the library, (b)
   Eq. 13 subsample placebo/pooled-pre att + obs vs the library, both at a 1e-3 att
   floor. Threshold rationale: 1e-3 sits 2.35x below the smallest measured
   divergence (2.35e-3 at full-panel h=0; every other gated surface clears it by
   >3.9x). Goldens regenerate only in deliberate local runs (Stata is not in CI);
   the regeneration protocol is to re-measure and refresh these floors if an SSC
   release narrows a convention gap.
7. **Warning cleanliness** on every library fit.
8. **Provenance**: ``meta.ssc_versions`` covers every fail-closed generator
   dependency with a real identifier that always embeds the ado file's checksum +
   length (SSC has no immutable archive, so a same-version-string upstream edit
   must still move the metadata); ``source_panel``/``source_sha256`` match the
   committed panel; prose meta fields are non-empty.

Guard per ``feedback_golden_file_pytest_skip``: CI isolated-install jobs copy
``tests/`` only, not ``benchmarks/data/``, so a missing fixture downgrades to
pytest.skip. The guards are SPLIT: the Stata golden + panel carry the module,
while the R golden gates only gate 1 - a missing R fixture cannot silence the
primary Stata-vs-library gates. A PRESENT golden missing the expected top-level
blocks HARD-FAILS
instead (a stale committed golden is a real inconsistency, not an absent fixture).
Regenerate with::

    /Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do \
        benchmarks/stata/generate_lpdid_nonabsorbing_golden.do
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

from diff_diff import LPDiD

_DATA = Path(__file__).parent.parent / "benchmarks" / "data"
STATA_GOLDEN_PATH = _DATA / "lpdid_nonabsorbing_stata_golden.json"
R_GOLDEN_PATH = _DATA / "lpdid_nonabsorbing_golden.json"
PANEL_PATH = _DATA / "lpdid_nonabsorbing_panel.csv"

# Split availability guards: the Stata golden + panel carry the whole module; the
# R golden is needed only by gate 1 (a missing R fixture must not silence the
# primary Stata-vs-library gates).
_STATA_FIXTURE_AVAILABLE = STATA_GOLDEN_PATH.is_file() and PANEL_PATH.is_file()
_R_FIXTURE_AVAILABLE = R_GOLDEN_PATH.is_file()

# Event-study horizons pinned by the golden; h = -1 is the omitted reference.
ES_HORIZONS = [-3, -2, 0, 1, 2, 3, 4]
POST_HORIZONS = [0, 1, 2, 3, 4]
PRE_HORIZONS = [-3, -2]
MODES = ["vw", "rw"]
L = 3

# committed Stata vs committed R golden (R digits=12); measured ~1e-13 locally.
R_ANCHOR_ATOL = 1e-9
ATT_ATOL = 1e-6  # library (recomputed) vs committed Stata, cross-platform
SE_ATOL = 1e-7  # the repo's LPDiD cross-platform SE tolerance
DIVERGENCE_FLOOR = 1e-3  # gate 6; see module docstring for the rationale

# Hard-coded twins of the golden's meta strings (gate 5 / gate 8 desync guards).
DROP_RULE = "min(treat)==1 | any(dD==1 & L3.dD==-1)"
MAPPING = (
    "nonabsorbing(, firsttreat notyet) == first_entry; nonabsorbing(L) == effect_stabilization"
)
ALWAYS_TREATED_UNITS = list(range(31, 41))  # convention class 1
EXACT_L_RESPELL_UNITS = [24, 25, 27]  # convention class 2
SUBSAMPLE_N_ROWS = 658
# Every fail-closed generator dependency (guard set == version set; both guarded
# egenmore files carry their own drift signal - _gfilter under "egenmore",
# _gclsst under "egenmore_gclsst").
SSC_VERSION_KEYS = {
    "lpdid",
    "reghdfe",
    "ftools",
    "require",
    "boottest",
    "listreg",
    "egenmore",
    "egenmore_gclsst",
}


def _skip_if_missing() -> None:
    if not _STATA_FIXTURE_AVAILABLE:
        pytest.skip(
            "Stata LPDiD non-absorbing parity fixture not present. Regenerate via "
            "`stata-se -b do benchmarks/stata/generate_lpdid_nonabsorbing_golden.do`."
        )


@pytest.fixture(scope="module")
def stata_golden() -> dict:
    _skip_if_missing()
    with STATA_GOLDEN_PATH.open() as f:
        golden = json.load(f)
    # Present-but-stale golden = a real inconsistency (forgotten regeneration);
    # hard-fail rather than silently skipping the external anchor.
    missing = {"meta", "first_entry", "effect_stab_sub", "effect_stab_full_vw"} - set(golden)
    assert not missing, (
        f"{STATA_GOLDEN_PATH.name} is present but missing block(s) {sorted(missing)} - "
        "stale golden; regenerate via generate_lpdid_nonabsorbing_golden.do."
    )
    return golden


@pytest.fixture(scope="module")
def r_first_entry() -> dict:
    _skip_if_missing()
    if not _R_FIXTURE_AVAILABLE:
        pytest.skip(
            "R non-absorbing golden not present (gate 1 only; the Stata-vs-library "
            "gates run without it). Regenerate via benchmarks/R/generate_lpdid_golden.R."
        )
    with R_GOLDEN_PATH.open() as f:
        return json.load(f)["first_entry"]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    _skip_if_missing()
    return pd.read_csv(PANEL_PATH)


@pytest.fixture(scope="module")
def subsample(panel: pd.DataFrame) -> pd.DataFrame:
    """Pandas recompute of the generator's drop rule, sub-class by sub-class."""
    df = panel.sort_values(["unit", "time"])
    grp = df.groupby("unit")
    unit_min_treat = grp["treat"].min()
    always = sorted(unit_min_treat[unit_min_treat == 1].index)
    d_d = grp["treat"].diff()
    d_d_lag_l = d_d.groupby(df["unit"]).shift(L)
    respell = sorted(df.loc[(d_d == 1) & (d_d_lag_l == -1), "unit"].unique())
    assert (
        always == ALWAYS_TREATED_UNITS
    ), f"always-treated recompute {always} != expected {ALWAYS_TREATED_UNITS}"
    assert (
        respell == EXACT_L_RESPELL_UNITS
    ), f"exact-L respell recompute {respell} != expected {EXACT_L_RESPELL_UNITS}"
    sub = panel[~panel["unit"].isin(always + respell)].reset_index(drop=True)
    assert len(sub) == SUBSAMPLE_N_ROWS
    return sub


def _fit(data: pd.DataFrame, mode: str, reweight: bool, only_event: bool = False) -> dict:
    """One library fit; returns per-horizon + pooled tables and captured warnings."""
    kwargs = {
        "pre_window": 3,
        "post_window": 4,
        "cluster": "unit",
        "non_absorbing": mode,
        "reweight": reweight,
    }
    if mode == "effect_stabilization":
        kwargs["stabilization_window"] = L
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = LPDiD(**kwargs).fit(
            data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            only_event=only_event,
        )
        messages = [str(w.message) for w in caught]
    es = res.event_study.set_index("horizon")
    by_h = {
        int(h): {
            "att": float(es.loc[h, "coefficient"]),
            "se": float(es.loc[h, "se"]),
            "n_obs": int(es.loc[h, "n_obs"]),
        }
        for h in ES_HORIZONS
    }
    pooled = {}
    if not only_event:
        for window in ("pre", "post"):
            row = res.pooled.loc[res.pooled["window"] == window].iloc[0]
            pooled[window] = {
                "att": float(row["coefficient"]),
                "se": float(row["se"]),
                "n_obs": int(row["n_obs"]),
            }
    return {"by_h": by_h, "pooled": pooled, "warnings": messages}


@pytest.fixture(scope="module")
def fits(panel: pd.DataFrame, subsample: pd.DataFrame) -> dict:
    """The five library fits (see module docstring): keyed ('fe'|'es'|'es_full', mode)."""
    return {
        ("fe", "vw"): _fit(panel, "first_entry", reweight=False),
        ("fe", "rw"): _fit(panel, "first_entry", reweight=True),
        ("es", "vw"): _fit(subsample, "effect_stabilization", reweight=False),
        ("es", "rw"): _fit(subsample, "effect_stabilization", reweight=True),
        ("es_full", "vw"): _fit(panel, "effect_stabilization", reweight=False, only_event=True),
    }


def _stata_block(golden: dict, arm: str, mode: str) -> dict:
    key = {"fe": "first_entry", "es": "effect_stab_sub"}[arm]
    return golden[key][mode]


# ---------------------------------------------------------------------------
# Gate 1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h", ES_HORIZONS)
def test_stata_first_entry_vw_matches_r_golden(h, stata_golden, r_first_entry):
    """Gate 1: the package's end-to-end Eq. 12 run reproduces the R feols anchor.

    Committed vs committed; ``rel=0`` makes the bound purely absolute.
    """
    stata = _stata_block(stata_golden, "fe", "vw")["es"][str(h)]
    r_att, r_se = r_first_entry[str(h)]
    assert stata["att"] == pytest.approx(
        r_att, abs=R_ANCHOR_ATOL, rel=0
    ), f"h={h}: Stata lpdid att {stata['att']} != R anchor {r_att}"
    assert stata["se"] == pytest.approx(
        r_se, abs=R_ANCHOR_ATOL, rel=0
    ), f"h={h}: Stata lpdid se {stata['se']} != R anchor {r_se}"


# ---------------------------------------------------------------------------
# Gates 2-3 (att + SE parity)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("h", ES_HORIZONS)
def test_library_first_entry_matches_stata(h, mode, stata_golden, fits):
    """Gate 2 (ES): library ``first_entry`` == Stata lpdid, every horizon incl. placebos."""
    stata = _stata_block(stata_golden, "fe", mode)["es"][str(h)]
    lib = fits[("fe", mode)]["by_h"][h]
    assert lib["att"] == pytest.approx(
        stata["att"], abs=ATT_ATOL, rel=0
    ), f"first_entry {mode} h={h}: library att {lib['att']} != Stata {stata['att']}"
    assert lib["se"] == pytest.approx(
        stata["se"], abs=SE_ATOL, rel=0
    ), f"first_entry {mode} h={h}: library se {lib['se']} != Stata {stata['se']}"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("window", ["pre", "post"])
def test_library_first_entry_pooled_matches_stata(window, mode, stata_golden, fits):
    """Gate 2 (pooled): both Eq. 12 pooled windows, att + SE - first external anchor."""
    stata = _stata_block(stata_golden, "fe", mode)["pooled"][window]
    lib = fits[("fe", mode)]["pooled"][window]
    assert lib["att"] == pytest.approx(
        stata["att"], abs=ATT_ATOL, rel=0
    ), f"first_entry {mode} pooled {window}: library att {lib['att']} != Stata {stata['att']}"
    assert lib["se"] == pytest.approx(
        stata["se"], abs=SE_ATOL, rel=0
    ), f"first_entry {mode} pooled {window}: library se {lib['se']} != Stata {stata['se']}"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("h", POST_HORIZONS)
def test_library_effect_stab_subsample_matches_stata(h, mode, stata_golden, fits):
    """Gate 3 (ES): library Eq. 13 == Stata lpdid on the convention-neutral subsample.

    Post horizons only - the placebo rows diverge by construction (class 3; gate 6b).
    The rw SEs here are the previously pinned-only surface, now externally anchored.
    """
    stata = _stata_block(stata_golden, "es", mode)["es"][str(h)]
    lib = fits[("es", mode)]["by_h"][h]
    assert lib["att"] == pytest.approx(
        stata["att"], abs=ATT_ATOL, rel=0
    ), f"effect_stab {mode} h={h}: library att {lib['att']} != Stata {stata['att']}"
    assert lib["se"] == pytest.approx(
        stata["se"], abs=SE_ATOL, rel=0
    ), f"effect_stab {mode} h={h}: library se {lib['se']} != Stata {stata['se']}"


@pytest.mark.parametrize("mode", MODES)
def test_library_effect_stab_pooled_post_matches_stata(mode, stata_golden, fits):
    """Gate 3 (pooled post): the Eq. 13 pooled-post window, att + SE."""
    stata = _stata_block(stata_golden, "es", mode)["pooled"]["post"]
    lib = fits[("es", mode)]["pooled"]["post"]
    assert lib["att"] == pytest.approx(
        stata["att"], abs=ATT_ATOL, rel=0
    ), f"effect_stab {mode} pooled post: library att {lib['att']} != Stata {stata['att']}"
    assert lib["se"] == pytest.approx(
        stata["se"], abs=SE_ATOL, rel=0
    ), f"effect_stab {mode} pooled post: library se {lib['se']} != Stata {stata['se']}"


# ---------------------------------------------------------------------------
# Gate 4 (sample shape - parity surfaces only; see module docstring on N_clust)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_sample_shape_first_entry(mode, stata_golden, fits):
    """Gate 4 (Eq. 12): Stata obs == library n_obs, all horizons + both pooled windows."""
    stata = _stata_block(stata_golden, "fe", mode)
    lib = fits[("fe", mode)]
    for h in ES_HORIZONS:
        assert int(stata["es"][str(h)]["N"]) == lib["by_h"][h]["n_obs"], (
            f"first_entry {mode} h={h}: Stata obs {stata['es'][str(h)]['N']} != "
            f"library n_obs {lib['by_h'][h]['n_obs']}"
        )
    for window in ("pre", "post"):
        assert (
            int(stata["pooled"][window]["N"]) == lib["pooled"][window]["n_obs"]
        ), f"first_entry {mode} pooled {window}: sample sizes differ"


@pytest.mark.parametrize("mode", MODES)
def test_sample_shape_effect_stab_post(mode, stata_golden, fits):
    """Gate 4 (Eq. 13): obs equality on post horizons + pooled post only.

    Pre horizons are excluded for BOTH modes: those samples differ by construction
    (class 3), and gate 6(b) asserts the mismatch instead.
    """
    stata = _stata_block(stata_golden, "es", mode)
    lib = fits[("es", mode)]
    for h in POST_HORIZONS:
        assert int(stata["es"][str(h)]["N"]) == lib["by_h"][h]["n_obs"], (
            f"effect_stab {mode} h={h}: Stata obs {stata['es'][str(h)]['N']} != "
            f"library n_obs {lib['by_h'][h]['n_obs']}"
        )
    assert (
        int(stata["pooled"]["post"]["N"]) == lib["pooled"]["post"]["n_obs"]
    ), f"effect_stab {mode} pooled post: sample sizes differ"


# ---------------------------------------------------------------------------
# Gate 5 (drop-rule consistency)
# ---------------------------------------------------------------------------
def test_drop_rule_consistency(stata_golden, subsample):
    """Gate 5: golden provenance == the pandas recompute, sub-class by sub-class.

    The ``subsample`` fixture already asserted the two sub-class unit lists (locking
    the per-class attribution) and the 658-row count; this test pins the golden's
    record of them plus the free-text rule string against hard-coded twins.
    """
    meta = stata_golden["meta"]
    assert meta["drop_rule"] == DROP_RULE
    assert meta["dropped_units"] == sorted(ALWAYS_TREATED_UNITS + EXACT_L_RESPELL_UNITS)
    assert meta["n_rows_sub"] == SUBSAMPLE_N_ROWS == len(subsample)
    assert stata_golden["effect_stab_sub"]["n_rows"] == SUBSAMPLE_N_ROWS


# ---------------------------------------------------------------------------
# Gate 6 (convention-divergence documentation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h", POST_HORIZONS)
def test_full_panel_divergence_documented(h, stata_golden, fits):
    """Gate 6(a): Eq. 13 FULL-panel Stata != library - classes 1+2, jointly.

    Asserts both the att divergence (floor 1e-3; measured 2.35e-3 at h=0 up to
    ~4e-2 at h=4) and the obs mismatch (the sample-admission mechanism itself:
    measured Stata 628/544/463/380/320 vs library 601/517/436/353/291).
    """
    stata = stata_golden["effect_stab_full_vw"]["es"][str(h)]
    lib = fits[("es_full", "vw")]["by_h"][h]
    att_diff = abs(stata["att"] - lib["att"])
    assert att_diff > DIVERGENCE_FLOOR, (
        f"full-panel h={h}: att divergence {att_diff} fell below the documented floor "
        "- the package's boundary/respell conventions may have changed; re-measure "
        "and update REGISTRY LPDiD Deviation 4."
    )
    assert int(stata["N"]) != lib["n_obs"], (
        f"full-panel h={h}: Stata obs == library n_obs ({lib['n_obs']}) - the "
        "classes-1+2 sample-admission difference disappeared; re-measure."
    )


@pytest.mark.parametrize("mode", MODES)
def test_subsample_placebo_divergence_documented(mode, stata_golden, fits):
    """Gate 6(b): Eq. 13 subsample placebo rows + pooled Pre diverge - class 3.

    Asserts att divergence (floor 1e-3; measured 3.9e-3..7.9e-2) and obs mismatch
    (measured ES vw 342/408 vs 390/437, rw 173/220 vs 282/329; pooled Pre vw 342 vs
    390, rw 173 vs 282) for BOTH modes.
    """
    stata = _stata_block(stata_golden, "es", mode)
    lib = fits[("es", mode)]
    for h in PRE_HORIZONS:
        s_row = stata["es"][str(h)]
        l_row = lib["by_h"][h]
        att_diff = abs(s_row["att"] - l_row["att"])
        assert att_diff > DIVERGENCE_FLOOR, (
            f"subsample {mode} h={h}: placebo att divergence {att_diff} fell below "
            "the documented floor - the package's placebo-window convention may have "
            "changed; re-measure and update REGISTRY LPDiD Deviation 4."
        )
        assert (
            int(s_row["N"]) != l_row["n_obs"]
        ), f"subsample {mode} h={h}: placebo samples coincide - class 3 disappeared."
    s_pre = stata["pooled"]["pre"]
    l_pre = lib["pooled"]["pre"]
    assert abs(s_pre["att"] - l_pre["att"]) > DIVERGENCE_FLOOR
    assert int(s_pre["N"]) != l_pre["n_obs"]


# ---------------------------------------------------------------------------
# Gate 7 (warning cleanliness)
# ---------------------------------------------------------------------------
def test_all_fits_warning_clean(fits):
    """Gate 7: none of the five library fits emits any warning."""
    noisy = {key: f["warnings"] for key, f in fits.items() if f["warnings"]}
    assert not noisy, f"unexpected warning(s): {noisy}"


# ---------------------------------------------------------------------------
# Gate 8 (provenance)
# ---------------------------------------------------------------------------
def test_provenance_contract(stata_golden):
    """Gate 8: drift signals + load-bearing meta strings.

    ``ssc_versions`` must cover exactly the generator's fail-closed dependency set
    with real identifiers - never empty/unknown/MISSING, and each embedding the ado
    file's ``checksum:`` marker (the generator always appends checksum + length to
    the version text: SSC has no immutable archive, so a same-version-string
    upstream edit must still move this metadata). ``mapping`` is the load-bearing
    option<->mode claim; the prose fields are asserted non-empty.
    """
    meta = stata_golden["meta"]
    versions = meta["ssc_versions"]
    assert set(versions) == SSC_VERSION_KEYS
    for pkg, ver in versions.items():
        assert isinstance(ver, str) and ver.strip(), f"{pkg}: empty version"
        assert ver not in {"unknown", "MISSING"}, f"{pkg}: no usable drift signal"
        assert re.search(
            r"checksum:\d+ len:\d+", ver
        ), f"{pkg}: version string lacks the numeric checksum+len drift signal: {ver!r}"
    assert meta["mapping"] == MAPPING
    assert meta["source_panel"] == "benchmarks/data/lpdid_nonabsorbing_panel.csv"
    sha = hashlib.sha256(PANEL_PATH.read_bytes()).hexdigest()
    assert meta["source_sha256"] == sha, (
        "golden source_sha256 does not match the committed panel - the panel changed "
        "without regenerating the Stata golden."
    )
    for field in ("cmd", "point_anchor", "se_convention", "convention_notes"):
        assert isinstance(meta[field], str) and meta[field].strip()
