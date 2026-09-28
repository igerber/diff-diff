"""External-reference parity: ImputationDiD leave-one-out SE vs Stata `did_imputation`.

`ImputationDiD(leave_one_out=True)` applies the Borusyak-Jaravel-Spiess (2024)
Supplementary Appendix A.9 finite-sample variance refinement. No R package computes it
(R `didimputation` omits LOO), so the library LOO SE was validated only by an internal
psi-identity + hand-calc + MC coverage. The authors' own Stata `did_imputation` ships
the same option (`leaveout`); this arm turns that into a measured anchor.

Second Stata parity arm, and the first SSC-dependent one. The golden is produced by
`benchmarks/stata/generate_imputation_loo_golden.do`, which runs `did_imputation ...,
[leaveout] avgeffectsby(...) cluster(unit)` on the committed R-arm panel
`didimputation_test_panel.csv` (no clean-sample reconstruction - did_imputation takes
the raw panel; the mappings are Ei = first_treat, missing for never-treated, and
K = time - Ei for the `avgeffectsby(K)` variant).

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

Coarser-`aux_partition` extension (Stata `avgeffectsby(Ei)` == `aux_partition="cohort"`,
`avgeffectsby(K)` == `"horizon"`, `K = t - Ei`), gates 6-10:
6. **Balanced `horizon` variant** - point + LOO SE + non-LOO SE at the overall AND every
   horizon h=0..5 (max observed deviation ~1.1e-8; per-horizon SEs genuinely differ from
   the default at h=0..3 and coincide at the single-cohort horizons h=4,5 - the TODO
   row's "no-op per-horizon" claim was wrong).
7. **Balanced `cohort` variant** - (a) library vs Stata at the overall; (b) the
   committed-vs-committed degeneracy identity: on this BALANCED panel the cohort
   partition is an arithmetic identity with the default (only v != 0 rows contribute per
   group), so Stata's `avgeffectsby(Ei)` overall matches the default-arm golden to 1e-9
   (observed ~1e-12; leg (a) is transitively implied by leg (b) + gates 1-3, kept as a
   direct anchor). The distinct-cohort measurement is gate 9's unbalanced block.
8. **Library-side identity pins** (library-only assertions, no golden values; shares this
   module's `_skip_if_missing` guard, so it skips in the isolated-install CI leg - the
   no-benchmarks-dir divergence coverage lives in tests/test_imputation.py::
   TestImputationVariance::test_coarser_partition_diverges_on_unbalanced_panel):
   `cohort` == default bit-tight
   (rtol=0, atol=1e-12) at overall + every horizon, both variance modes; `horizon`
   == default at h=4,5 only, differing at h=0..3.
9. **Unbalanced subsample** (drop_rule `mod(unit,4)==0 & time>=6`, 1305 rows,
   reconstructed identically on both sides; the row count alone does NOT identify the
   subsample - every residue class gives 1305 rows - so sample equivalence is pinned by
   the numeric gates, with `drop_rule`/`n_rows` consistency asserts riding along):
   point + LOO SE + non-LOO SE per partition, plus the mechanism gate that `cohort`
   LOO SE differs from the default by >10% (measured ratio 1.234).
10. **Warning-cleanliness for every new LOO fit** - mirrors gate 5 across the variant
    and unbalanced fixtures.

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
from numpy.testing import assert_allclose

from diff_diff import ImputationDiD

# ---------------------------------------------------------------------------
# Rows M-021/M-022 (+ M-118/M-119): ImputationDiD / TwoStageDiD
# ``fit(aggregate=, balance_e=)`` is deprecated (3.9, removed 4.0) and warns on
# ANY supplied value. The deprecated fit-time route is kept DELIBERATELY here:
# these tests pin FIT-TIME surface behaviour (bit-equality grids, bootstrap
# aggregation, R/Stata parity, replicate overrides, native effect dicts) that
# the post-fit ``results.aggregate(...)`` container route does not reproduce
# shape-for-shape. The shim warning is therefore filtered BY MESSAGE, scoped to
# these two estimators only - every other FutureWarning (including the other
# estimators' aggregate() shims) still surfaces.
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.filterwarnings(
    r"ignore:(ImputationDiD|TwoStageDiD)\.fit\((aggregate=|balance_e=|aggregate= / balance_e=)\):FutureWarning"
)

_DATA = Path(__file__).parent.parent / "benchmarks" / "data"
STATA_GOLDEN_PATH = _DATA / "didimputation_loo_stata_golden.json"
R_GOLDEN_PATH = _DATA / "didimputation_golden.json"
PANEL_PATH = _DATA / "didimputation_test_panel.csv"

_FIXTURE_AVAILABLE = (
    STATA_GOLDEN_PATH.is_file() and R_GOLDEN_PATH.is_file() and PANEL_PATH.is_file()
)

HORIZONS = [0, 1, 2, 3, 4, 5]

# Coarser-partition arm (gates 6-10): Stata avgeffectsby() <-> library aux_partition.
VARIANT_PARTITIONS = {"cohort": "Ei", "horizon": "K"}
# Deterministic unbalanced subsample rule; MUST match the golden's recorded drop_rule
# and the generator's `drop if` line (the numeric gates pin actual sample equivalence).
UNBALANCED_DROP_RULE = "mod(unit,4)==0 & time>=6"
UNBALANCED_N_ROWS = 1305

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


def _require_block(golden: dict, key: str) -> dict:
    """A PRESENT golden missing a new block is a stale/reverted regeneration - a real
    inconsistency that must FAIL, not skip (the skip convention covers absent files)."""
    assert key in golden, (
        f"golden {STATA_GOLDEN_PATH.name} lacks the '{key}' block - stale golden? "
        f"Regenerate via benchmarks/stata/generate_imputation_loo_golden.do"
    )
    return golden[key]


def _fit_partition(panel, aux_partition: str, leave_one_out: bool):
    """One event-study fit; returns (results, [warning messages]). The event_study
    aggregate populates the per-horizon surfaces gates 6/8 read. The DEPRECATED
    fit-time ``aggregate=`` route is deliberate, matching this module's convention
    (see the M-021/M-022 pytestmark comment above: parity tests pin the fit-time
    surface shape-for-shape; the shim warning is filtered by message)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = ImputationDiD(leave_one_out=leave_one_out, aux_partition=aux_partition).fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
    assert res.event_study_effects is not None
    return res, [str(w.message) for w in caught]


@pytest.fixture(scope="module")
def library_fit_variants() -> dict:
    """Balanced-panel fits for gates 6-8: {cohort, horizon, cohort_horizon} x
    {LOO, non-LOO} = 6 event-study fits (the default partition is refit as gate 8's
    identity reference)."""
    _skip_if_missing()
    panel = pd.read_csv(PANEL_PATH)
    out: dict = {"loo_warnings": []}
    for part in ("cohort", "horizon", "cohort_horizon"):
        for loo in (True, False):
            res, msgs = _fit_partition(panel, part, loo)
            if loo:
                out["loo_warnings"] += msgs
            es = res.event_study_effects
            assert es is not None
            mode = "loo" if loo else "nonloo"
            out[f"{part}_{mode}"] = {
                "att": float(res.overall_att),
                "se": float(res.overall_se),
                "es": {h: (float(es[h]["effect"]), float(es[h]["se"])) for h in HORIZONS},
            }
    return out


@pytest.fixture(scope="module")
def library_fit_unbalanced() -> dict:
    """Unbalanced-subsample fits for gates 9-10: 3 partitions x {LOO, non-LOO}, overall
    only. The pandas reconstruction mirrors the generator's `drop if` rule exactly."""
    _skip_if_missing()
    panel = pd.read_csv(PANEL_PATH)
    # keep rows NOT matching UNBALANCED_DROP_RULE (De Morgan of mod(unit,4)==0 & time>=6)
    sub = panel.loc[(panel["unit"] % 4 != 0) | (panel["time"] < 6)]
    assert len(sub) == UNBALANCED_N_ROWS
    out: dict = {"loo_warnings": []}
    for part in ("cohort_horizon", "cohort", "horizon"):
        for loo in (True, False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = ImputationDiD(leave_one_out=loo, aux_partition=part).fit(
                    sub,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                )
            if loo:
                out["loo_warnings"] += [str(w.message) for w in caught]
            mode = "loo" if loo else "nonloo"
            out[f"{part}_{mode}"] = (float(res.overall_att), float(res.overall_se))
    return out


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


# ----- Gate 6: balanced `horizon` variant (avgeffectsby(K)), overall + every horizon -----


def test_variant_horizon_overall_matches_stata(stata_golden, library_fit_variants):
    blk = _require_block(stata_golden, "variants")["horizon"]["overall"]
    loo = library_fit_variants["horizon_loo"]
    nonloo = library_fit_variants["horizon_nonloo"]
    assert loo["att"] == pytest.approx(blk["att"], abs=ATOL, rel=0)
    assert loo["se"] == pytest.approx(blk["se"], abs=ATOL, rel=0)
    assert nonloo["se"] == pytest.approx(blk["se_nonloo"], abs=ATOL, rel=0)


@pytest.mark.parametrize("h", HORIZONS)
def test_variant_horizon_event_study_matches_stata(h, stata_golden, library_fit_variants):
    blk = _require_block(stata_golden, "variants")["horizon"]["event_study"][str(h)]
    att, loo_se = library_fit_variants["horizon_loo"]["es"][h]
    _, nonloo_se = library_fit_variants["horizon_nonloo"]["es"][h]
    assert att == pytest.approx(blk["att"], abs=ATOL, rel=0)
    assert loo_se == pytest.approx(blk["se"], abs=ATOL, rel=0)
    assert nonloo_se == pytest.approx(blk["se_nonloo"], abs=ATOL, rel=0)


# ----- Gate 7: balanced `cohort` variant (avgeffectsby(Ei)) + degeneracy identity -----


def test_variant_cohort_overall_matches_stata_and_degenerates(stata_golden, library_fit_variants):
    """(a) library vs Stata avgeffectsby(Ei) at the overall; (b) committed-vs-committed:
    on this BALANCED panel the cohort partition is an arithmetic identity with the
    default, so Stata's cohort block matches the default-arm golden to 1e-9 (observed
    ~1e-12 - pure reghdfe accumulation noise). Leg (a) is transitively implied by (b) +
    gates 1-3; kept as a direct anchor. The distinct-cohort measurement is gate 9."""
    blk = _require_block(stata_golden, "variants")["cohort"]["overall"]
    loo = library_fit_variants["cohort_loo"]
    nonloo = library_fit_variants["cohort_nonloo"]
    assert loo["att"] == pytest.approx(blk["att"], abs=ATOL, rel=0)
    assert loo["se"] == pytest.approx(blk["se"], abs=ATOL, rel=0)
    assert nonloo["se"] == pytest.approx(blk["se_nonloo"], abs=ATOL, rel=0)
    # (b) degeneracy identity, committed vs committed. The SEs pin the identity at 1e-9
    # (observed ~1e-12); the att only at ATOL - did_imputation's POINT jitters ~4e-9
    # across runs (first-stage reghdfe noise), while the identity claim is about the
    # variance construction (the library att is bit-identical across partitions).
    default = stata_golden["overall"]
    assert blk["att"] == pytest.approx(default["att"], abs=ATOL, rel=0)
    assert blk["se"] == pytest.approx(default["se"], abs=1e-9, rel=0)
    assert blk["se_nonloo"] == pytest.approx(default["se_nonloo"], abs=1e-9, rel=0)


# ----- Gate 8: library-side identity pins (protect the documented partition facts) -----


def test_cohort_partition_is_identity_with_default_on_balanced_panel(library_fit_variants):
    """`aux_partition="cohort"` == the default at every surface on the balanced panel
    (bit-equal locally; rtol=0 so numpy's default rtol=1e-7 cannot mask the pin)."""
    for mode in ("loo", "nonloo"):
        co, ch = (
            library_fit_variants[f"cohort_{mode}"],
            library_fit_variants[f"cohort_horizon_{mode}"],
        )
        assert_allclose(co["att"], ch["att"], rtol=0, atol=1e-12)
        assert_allclose(co["se"], ch["se"], rtol=0, atol=1e-12)
        for h in HORIZONS:
            assert_allclose(co["es"][h], ch["es"][h], rtol=0, atol=1e-12)


def test_horizon_partition_differs_except_single_cohort_horizons(library_fit_variants):
    """`aux_partition="horizon"` per-horizon SEs differ from the default at h=0..3 and
    coincide exactly at h=4,5 - the horizons only cohort 3 reaches, where the K group
    equals the (Ei, h) cell (corrects the old "no-op per-horizon" claim)."""
    for mode in ("loo", "nonloo"):
        hz, ch = (
            library_fit_variants[f"horizon_{mode}"],
            library_fit_variants[f"cohort_horizon_{mode}"],
        )
        for h in (4, 5):
            assert_allclose(hz["es"][h][1], ch["es"][h][1], rtol=0, atol=1e-12)
        for h in (0, 1, 2, 3):
            assert (
                abs(hz["es"][h][1] - ch["es"][h][1]) > 10 * ATOL
            ), f"h={h} ({mode}): horizon-partition SE unexpectedly equals the default's"


# ----- Gate 9: unbalanced subsample - where the cohort partition genuinely bites -----


def test_unbalanced_block_consistency(stata_golden):
    """drop_rule/n_rows consistency (golden <-> test literals). The row count alone does
    NOT identify the subsample (every mod-4 residue class gives 1305 rows); actual
    sample equivalence is pinned by the numeric gates below."""
    blk = _require_block(stata_golden, "unbalanced")
    assert blk["drop_rule"] == UNBALANCED_DROP_RULE
    assert blk["n_rows"] == UNBALANCED_N_ROWS


def test_meta_records_variant_mapping(stata_golden):
    """The golden's meta must record the avgeffectsby <-> aux_partition mapping the
    variant gates rely on (drift detection for the generator's meta twin)."""
    meta_line = stata_golden["meta"]["avgeffectsby_variants"]
    for lib_partition, stata_var in VARIANT_PARTITIONS.items():
        # Assert the exact pairing ("Ei == cohort"), not mere co-occurrence - a
        # swapped mapping must fail here, not only in the numeric gates.
        assert f"{stata_var} == {lib_partition}" in meta_line


@pytest.mark.parametrize("partition", ["cohort_horizon", "cohort", "horizon"])
def test_unbalanced_matches_stata(partition, stata_golden, library_fit_unbalanced):
    blk = _require_block(stata_golden, "unbalanced")[partition]
    att, loo_se = library_fit_unbalanced[f"{partition}_loo"]
    _, nonloo_se = library_fit_unbalanced[f"{partition}_nonloo"]
    assert att == pytest.approx(blk["att"], abs=ATOL, rel=0)
    assert loo_se == pytest.approx(blk["se"], abs=ATOL, rel=0)
    assert nonloo_se == pytest.approx(blk["se_nonloo"], abs=ATOL, rel=0)


def test_unbalanced_cohort_genuinely_diverges(stata_golden):
    """Mechanism gate: on the unbalanced subsample the cohort partition must differ from
    the default by >10% (measured ratio ~1.234) - proving the block exercises the
    coarse-partition path rather than re-measuring the balanced degeneracy."""
    blk = _require_block(stata_golden, "unbalanced")
    assert blk["cohort"]["se"] > 1.10 * blk["cohort_horizon"]["se"]


# ----- Gate 10: warning-cleanliness for every new LOO fit (mirrors gate 5) -----


def test_variant_and_unbalanced_loo_fits_emit_no_singleton_warning(
    library_fit_variants, library_fit_unbalanced
):
    msgs = library_fit_variants["loo_warnings"] + library_fit_unbalanced["loo_warnings"]
    singletons = [
        m
        for m in msgs
        if "leave-one-out" in m.lower() or "singleton" in m.lower() or "single" in m.lower()
    ]
    assert not singletons, f"unexpected LOO-singleton warning(s): {singletons}"
