"""Audit matrix: which clustered-variance convention each estimator surface gets.

This is the CI-enforced inventory behind ``docs/methodology/variance-conventions.md``.
Each row pins, for one (estimator, fit configuration) cell:

- ``cr1_k``   — the sorted multiset of K_reference counts reaching the shared
  clustered CR1 denominator: visible columns + the signed
  ``cluster_k_adjustment`` (``linalg._compute_robust_vcov_numpy``; 3.9 D1/D2
  convergence), or ``()`` when the surface's contract is that it makes NO
  shared-CR1 call.
- ``tail_df`` — the sorted multiset of ``df`` values passed to
  ``safe_inference`` / ``safe_inference_batch`` (``None`` = normal theory).

The point is visibility, not endorsement: the clustered-CR1 ``k`` cells now
pin the converged K_reference accounting, while the remaining DOCUMENTED
DEFECT rows are the mixed tail-df conventions scheduled for PR C of the 3.9
consolidation program. Every row carries ``status`` and, for legitimate
differences, a ``reason``. When a later PR changes a convention, the
expected literal changes HERE, in one reviewable table.

Instrumentation notes (each guards against a failure mode that produced wrong
inventory numbers during planning):

- ``safe_inference`` is bound at import by ~20 modules AND imported dynamically
  inside ``LinearRegression.get_inference``. Binder modules are discovered by
  scanning ``diff_diff`` at runtime — the hand-maintained list was extended four
  times and twice wrongly declared exhaustive.
- The Rust backend resolves ``DIFF_DIFF_BACKEND`` once at import, so the Python
  lane is forced by nulling each module's own ``HAS_RUST_BACKEND`` binding —
  also discovered programmatically (10 modules hold one).
- Wrappers bind via ``inspect.signature`` so positional and keyword call styles
  both register; a naive wrapper silently captured zero calls.
- Rows whose contract is "no shared-CR1 call" assert EXACTLY zero CR1 captures
  while still capturing tail df (the zero/non-zero rule is about CR1 only).
"""

import importlib
import inspect
import pkgutil
import warnings

import numpy as np
import pandas as pd
import pytest

import diff_diff

# ---------------------------------------------------------------------------
# Shared DGP (documented in docs/methodology/variance-conventions.md — keep in
# sync; the doc's table is generated from THIS fixture via
# `python -m tests.test_variance_conventions`).
# ---------------------------------------------------------------------------


def make_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for unit in range(60):
        first_treat = [0, 3, 4, 5][unit % 4]
        for time in range(1, 7):
            treated = int(first_treat > 0 and time >= first_treat)
            grp = unit % 2
            post = int(time >= 4)
            y = rng.normal() + 0.3 * unit + 0.2 * time + 2.0 * treated
            rows.append(
                dict(
                    unit=unit,
                    time=time,
                    first_treat=first_treat,
                    treated=treated,
                    grp=grp,
                    post=post,
                    y=y,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Programmatic discovery of instrumentation targets.
# ---------------------------------------------------------------------------


def _diff_diff_modules():
    """Import and yield every diff_diff submodule (idempotent)."""
    for info in pkgutil.iter_modules(diff_diff.__path__, prefix="diff_diff."):
        try:
            yield importlib.import_module(info.name)
        except ImportError:
            continue


def _modules_binding(attr: str):
    """Modules holding their own module-level binding of ``attr``."""
    for mod in _diff_diff_modules():
        if attr in vars(mod):
            yield mod


class Capture:
    """Wrap the inference helpers + the shared CR1 kernel across every lane."""

    def __init__(self, monkeypatch):
        self.cr1_k: list = []
        self.tail_df: list = []
        # Per-call K_reference adjustments on the clustered hc1 lane, so a
        # row can pin the increment itself even where the exact k is a
        # contract detail.
        self.cr1_adjustments: list = []
        # Clustered calls in a NON-hc1 family: (vcov_type, k). The CR1 factor
        # this matrix audits is the clustered hc1 denominator, so a surface
        # that silently switches clustered family must fail its row even when
        # the design width is unchanged.
        self.unexpected_clustered: list = []
        import diff_diff.linalg as lmod
        import diff_diff.utils as umod

        # --- CR1 kernel (numpy lane; the Rust lane is disabled below) ---
        orig_vcov = lmod._compute_robust_vcov_numpy
        sig_vcov = inspect.signature(orig_vcov)

        def spy_vcov(*a, **k):
            b = sig_vcov.bind(*a, **k)
            b.apply_defaults()
            if b.arguments.get("cluster_ids") is not None:
                if b.arguments.get("vcov_type") == "hc1":
                    # cr1_k records the K_REFERENCE count reaching the CR1
                    # denominator: visible columns + the signed
                    # cluster_k_adjustment (variance-conventions.md D1/D2).
                    _adj = int(b.arguments.get("cluster_k_adjustment") or 0)
                    self.cr1_k.append(int(b.arguments["X"].shape[1]) + _adj)
                    self.cr1_adjustments.append(_adj)
                else:
                    self.unexpected_clustered.append(
                        (str(b.arguments.get("vcov_type")), int(b.arguments["X"].shape[1]))
                    )
            return orig_vcov(*a, **k)

        monkeypatch.setattr(lmod, "_compute_robust_vcov_numpy", spy_vcov)

        # --- inference helpers, canonical + every import-time binder ---
        for name in ("safe_inference", "safe_inference_batch"):
            orig = getattr(umod, name)
            sig = inspect.signature(orig)

            def make_spy(orig=orig, sig=sig):
                def spy(*a, **k):
                    b = sig.bind(*a, **k)
                    b.apply_defaults()
                    df = b.arguments.get("df")
                    self.tail_df.append(None if df is None else float(df))
                    return orig(*a, **k)

                return spy

            spy = make_spy()
            monkeypatch.setattr(umod, name, spy)  # dynamic-import lane
            for mod in _modules_binding(name):
                if mod is not umod:
                    monkeypatch.setattr(mod, name, spy)

        # --- force the canonical numpy backend in EVERY module holding a flag ---
        for mod in _modules_binding("HAS_RUST_BACKEND"):
            monkeypatch.setattr(mod, "HAS_RUST_BACKEND", False)
        for mod in _diff_diff_modules():
            for rust_name in [n for n in vars(mod) if n.startswith("_rust_")]:
                monkeypatch.setattr(mod, rust_name, None)

    def snapshot(self):
        return (
            tuple(sorted(self.cr1_k)),
            tuple(sorted(self.tail_df, key=lambda v: (v is None, v))),
        )


# ---------------------------------------------------------------------------
# The matrix. One entry per (surface, configuration) cell.
#
# status:  "defect"      — scheduled to change in the 3.9 consolidation program
#          "legitimate"  — a declared exception with its reason
# cr1_k = () means the row's CONTRACT is "no shared clustered-CR1 call".
# ---------------------------------------------------------------------------

ROWS = [
    dict(
        key="did_absorb_hc1_cluster_unit",
        fit=lambda df: diff_diff.DifferenceInDifferences(cluster="unit").fit(
            df, outcome="y", treatment="grp", time="post", absorb=["unit", "time"]
        ),
        cr1_k=(7,),
        tail_df=(294.0,),
        status="legitimate",
        reason=(
            "K_reference (D2 fixed): k = 2 visible + rank(time given unit) = 5; "
            "matches reghdfe/fixest ssc(K.fixef='nested')"
        ),
    ),
    dict(
        key="did_fixed_effects_hc1_cluster_unit",
        fit=lambda df: diff_diff.DifferenceInDifferences(cluster="unit").fit(
            df, outcome="y", treatment="grp", time="post", fixed_effects=["unit", "time"]
        ),
        cr1_k=(7,),
        tail_df=(294.0,),
        status="legitimate",
        reason=(
            "K_reference (D1 fixed): 66 visible minus the 59 cluster-nested "
            "unit dummies -> identical SE to did_absorb (documented deviation "
            "from a literal explicit-dummy R comparison, which counts all 66)"
        ),
    ),
    dict(
        key="did_plain_hc1_cluster_unit",
        fit=lambda df: diff_diff.DifferenceInDifferences(cluster="unit").fit(
            df, outcome="y", treatment="grp", time="post"
        ),
        cr1_k=(4,),
        tail_df=(356.0,),
        status="legitimate",
        reason="no absorbed FE: visible k is the whole design; nothing is omitted",
    ),
    dict(
        key="twfe_hc1_cluster_unit_time_post",
        fit=lambda df: diff_diff.TwoWayFixedEffects(vcov_type="hc1", cluster="unit").fit(
            df, outcome="y", treatment="grp", time="post", unit="unit"
        ),
        cr1_k=(3,),
        tail_df=(298.0,),
        status="legitimate",
        reason=(
            "K_reference (D2 fixed): 2 visible + rank(post given unit) = 1; "
            "matches fixest cluster arm at rel 0 (committed golden)"
        ),
    ),
    dict(
        key="wooldridge_hc1_within",
        fit=lambda df: diff_diff.WooldridgeDiD(method="ols").fit(
            df, outcome="y", unit="unit", time="time", cohort="first_treat"
        ),
        cr1_k=(15,),
        tail_df=(None,) * 10,
        status="defect",
        reason=(
            "CR1 k converged on K_reference (D2 fixed: 9 cells + T = 15, "
            "no intercept col -> +1 term; jwdid arms at ratio 1.0); tail df "
            "is still normal theory with no df_convention knob (PR C)"
        ),
    ),
    dict(
        key="sun_abraham_hc1",
        fit=lambda df: diff_diff.SunAbraham().fit(
            df, outcome="y", unit="unit", time="time", first_treat="first_treat"
        ),
        cr1_k=(21,),
        tail_df=(280.0,) * 15 + (None,) * 8,
        status="defect",
        reason=(
            "CR1 k converged on K_reference (D2 fixed: 15 cells + 6, no "
            "intercept col; fixest sunab parity ~5e-15); D4 remains: residual "
            "df per cohort-period cell but normal theory on aggregates (PR C)"
        ),
    ),
    dict(
        key="stacked_did_hc1",
        fit=lambda df: diff_diff.StackedDiD().fit(
            df, outcome="y", unit="unit", time="time", first_treat="first_treat"
        ),
        cr1_k=(6,),
        tail_df=(None,) * 1,
        status="legitimate",
        reason=(
            "L1: k_total is clubSandwich CR1S by construction (stacked_did.py "
            "pins vcovCR(type='CR1S') at atol=1e-10); normal-theory tail df is "
            "an open PR C question"
        ),
    ),
    dict(
        key="lpdid_pre2_post2",
        fit=lambda df: diff_diff.LPDiD(pre_window=2, post_window=2).fit(
            df, outcome="y", unit="unit", time="time", treatment="treated"
        ),
        cr1_k=(4, 4, 5, 5, 5, 6),
        tail_df=(59.0,) * 6,
        status="legitimate",
        reason="L2: G-1 tail df (Stata/fixest convention) — the convergence target",
    ),
    dict(
        key="mpd_absorb_hc1_cluster_unit",
        fit=lambda df: diff_diff.MultiPeriodDiD(cluster="unit").fit(
            df,
            outcome="y",
            treatment="grp",
            time="time",
            absorb=["unit", "time"],
            reference_period=1,
        ),
        cr1_k=(11,),
        tail_df=(290.0,) * 6,
        status="legitimate",
        reason=(
            "K_reference: 6 visible + rank(time given unit) = 5; equals the "
            "fixed_effects form's 70 - 59 (MPD absorb/fixed_effects equivalence)"
        ),
    ),
    dict(
        key="mpd_fixed_effects_hc1_cluster_unit",
        fit=lambda df: diff_diff.MultiPeriodDiD(cluster="unit").fit(
            df,
            outcome="y",
            treatment="grp",
            time="time",
            fixed_effects=["unit", "time"],
            reference_period=1,
        ),
        cr1_k=(11,),
        tail_df=(290.0,) * 6,
        status="legitimate",
        reason=(
            "K_reference: 70 visible (incl. built-in period dummies, MPD's "
            "time-FE block) minus the 59 cluster-nested unit dummies = 11 — "
            "identical to the absorb form"
        ),
    ),
    dict(
        key="mpd_plain_hc1_cluster_time",
        fit=lambda df: diff_diff.MultiPeriodDiD(cluster="time").fit(
            df, outcome="y", treatment="grp", time="time", reference_period=1
        ),
        cr1_k=(7,),
        tail_df=(348.0,) * 6,
        status="legitimate",
        reason=(
            "the NESTED orientation of the built-in period dummies: 12 "
            "visible minus their rank 5 under a time cluster (under-"
            "subtraction is caught here; the unit-cluster rows catch over-)"
        ),
    ),
    dict(
        key="lpdid_absorb_nested_cluster_grp",
        fit=lambda df: diff_diff.LPDiD(pre_window=2, post_window=2, cluster="grp").fit(
            df.assign(region=df["unit"] % 2),
            outcome="y",
            unit="unit",
            time="time",
            treatment="treated",
            absorb=["region"],
        ),
        cr1_k=(4, 4, 5, 5, 5, 6),
        tail_df=(1.0,) * 6,
        status="legitimate",
        reason=(
            "LPDiD absorb dummies nested in the cluster subtract their rank "
            "(adj -1 per horizon: region == grp here); _event_time stays "
            "counted (unit-level cluster does not nest time); G-1 tail df (L2)"
        ),
        expected_adjustment=-1,
    ),
    dict(
        key="imputation_default",
        fit=lambda df: diff_diff.ImputationDiD().fit(
            df, outcome="y", unit="unit", time="time", first_treat="first_treat"
        ),
        cr1_k=(),
        tail_df=(None,) * 1,
        status="legitimate",
        reason="L3: BJS imputation variance, not the shared CR1 sandwich",
    ),
    dict(
        key="imputation_pretrends_event_study",
        fit=lambda df: diff_diff.ImputationDiD(pretrends=True).fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        ),
        cr1_k=None,  # contract row: shared CR1 REACHED; exact k is config-detail
        tail_df=None,
        # The increment IS pinned even though k is not: every clustered-hc1
        # call on this surface must carry the [time, unit] no-intercept
        # increment (1 + 65 - 60 = 6 on the audit panel).
        expected_adjustment=6,
        status="defect",
        reason=(
            "pretrends lead regression: CR1 k converged on K_reference "
            "(D2 fixed); normal-theory tail df remains (PR C family)"
        ),
    ),
    dict(
        key="two_stage_default",
        fit=lambda df: diff_diff.TwoStageDiD().fit(
            df, outcome="y", unit="unit", time="time", first_treat="first_treat"
        ),
        cr1_k=(),
        tail_df=(None,) * 1,
        status="legitimate",
        reason="L3: Gardner two-stage variance, not the shared CR1 sandwich",
    ),
    dict(
        key="callaway_santanna_default",
        fit=lambda df: diff_diff.CallawaySantAnna().fit(
            df, outcome="y", unit="unit", time="time", first_treat="first_treat"
        ),
        cr1_k=(),
        tail_df=(None,) * 16,
        status="legitimate",
        reason="L3: influence-function variance anchored to Stata csdid",
    ),
]

_FAST_KEYS = {
    "did_absorb_hc1_cluster_unit",
    "did_fixed_effects_hc1_cluster_unit",
    "twfe_hc1_cluster_unit_time_post",
    "wooldridge_hc1_within",
    "lpdid_pre2_post2",
    "callaway_santanna_default",
}


def _run_row(row, monkeypatch):
    df = make_panel()
    cap = Capture(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row["fit"](df)
    return cap


def _assert_row(row, monkeypatch):
    cap = _run_row(row, monkeypatch)
    cr1_k, tail_df = cap.snapshot()
    assert cap.unexpected_clustered == [], (
        f"{row['key']}: clustered vcov calls in a non-hc1 family "
        f"{cap.unexpected_clustered} — the audit classifies only vcov_type='hc1' "
        "as the shared clustered CR1; this surface switched clustered family"
    )
    if row["cr1_k"] is None:
        # Contract row: the shared CR1 must be REACHED (the exact k is
        # configuration-detail); used where a literal would be brittle.
        assert len(cr1_k) > 0, f"{row['key']}: expected shared-CR1 calls, saw none"
    elif row["cr1_k"] == ():
        assert cr1_k == (), (
            f"{row['key']}: contract is NO shared-CR1 call, captured k={cr1_k} — "
            "a refactor routed this estimator through the shared sandwich"
        )
    else:
        assert cr1_k == row["cr1_k"], f"{row['key']}: cr1_k {cr1_k} != {row['cr1_k']}"
    if row.get("expected_adjustment") is not None:
        # Contract rows pin the K_reference increment itself even where the
        # exact k is configuration-detail: every clustered-hc1 call must
        # carry exactly the expected adjustment.
        assert cap.cr1_adjustments and all(
            a == row["expected_adjustment"] for a in cap.cr1_adjustments
        ), (
            f"{row['key']}: expected every cluster_k_adjustment == "
            f"{row['expected_adjustment']}, captured {cap.cr1_adjustments}"
        )
    if row["tail_df"] is not None:
        assert tail_df == row["tail_df"], f"{row['key']}: tail_df {tail_df} != {row['tail_df']}"
    # Self-check: a row claiming tail-df expectations must actually capture some
    # (guards the silent-zero-capture failure that produced wrong inventory
    # numbers during planning).
    if row["tail_df"] not in (None, ()):
        assert len(tail_df) > 0, f"{row['key']}: instrumentation captured nothing"


@pytest.mark.parametrize("row", [r for r in ROWS if r["key"] in _FAST_KEYS], ids=lambda r: r["key"])
def test_variance_convention_fast(row, monkeypatch):
    """One config per major surface — runs in the default suite."""
    _assert_row(row, monkeypatch)


@pytest.mark.slow
@pytest.mark.parametrize(
    "row", [r for r in ROWS if r["key"] not in _FAST_KEYS], ids=lambda r: r["key"]
)
def test_variance_convention_full(row, monkeypatch):
    """The remaining rows — excluded from the default suite via addopts."""
    _assert_row(row, monkeypatch)


def test_every_row_declares_status_and_reason():
    for row in ROWS:
        assert row["status"] in ("defect", "legitimate"), row["key"]
        assert row["reason"], row["key"]


def test_capture_flags_non_hc1_clustered_family(monkeypatch):
    """The CR1 classification is family-aware: a clustered fit in another
    vcov family lands in unexpected_clustered, never in cr1_k (measured: a
    clustered hc2_bm DiD reaches the same kernel with k=4 — the old
    cluster_ids-only spy recorded it indistinguishably from the plain-hc1
    row's literal)."""
    df = make_panel()
    cap = Capture(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diff_diff.DifferenceInDifferences(vcov_type="hc2_bm", cluster="unit").fit(
            df, outcome="y", treatment="grp", time="post"
        )
    assert cap.cr1_k == []
    assert cap.unexpected_clustered, "non-hc1 clustered call was not flagged"
    assert all(fam == "hc2_bm" for fam, _ in cap.unexpected_clustered)


def test_d1_convergence_is_pinned():
    """absorb= and fixed_effects= now return the IDENTICAL clustered SE.

    Before the K_reference fix the two documented-equivalent idioms differed
    by exactly sqrt((360-2)/(360-66)) = 10.35% (defect D1). Both lanes now
    land on K_reference = 7, and the common SE equals the old absorb-side
    value rescaled by the K change — the externally anchored prediction.
    """
    df = make_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = diff_diff.DifferenceInDifferences(cluster="unit").fit(
            df, outcome="y", treatment="grp", time="post", absorb=["unit", "time"]
        )
        f = diff_diff.DifferenceInDifferences(cluster="unit").fit(
            df, outcome="y", treatment="grp", time="post", fixed_effects=["unit", "time"]
        )
    np.testing.assert_allclose(a.att, f.att, rtol=0, atol=1e-10)
    np.testing.assert_allclose(f.se / a.se, 1.0, rtol=1e-9)
    # measured pre-fix absorb SE (k=2) rescaled to K_reference = 7:
    old_absorb_se = 0.2414226781
    predicted = old_absorb_se * np.sqrt((360 - 2) / (360 - 7))
    np.testing.assert_allclose(a.se, predicted, rtol=1e-8)


# ---------------------------------------------------------------------------
# Generator: emit the markdown table for docs/methodology/variance-conventions.md
# (run: python -m tests.test_variance_conventions)
# ---------------------------------------------------------------------------


def emit_markdown_table() -> str:
    """Render the committed inventory table from the rows' EXPECTED literals.

    Deliberately no fitting here: the parametrized tests above assert these
    literals against live instrumentation (fast subset in the default suite,
    full sweep under ``-m slow``), so the doc-sync gate stays O(formatting)
    and the slow split keeps bounding the default suite as the matrix grows.
    ``None`` literals (contract rows) render as ``unpinned``.
    """
    lines = [
        "| surface | CR1 `k` (multiset) | tail df (multiset) | status | reason |",
        "|---|---|---|---|---|",
    ]

    def fmt(t):
        if t is None:
            return "unpinned"
        if not t:
            return "—"
        return ", ".join("None" if v is None else f"{v:g}" for v in t)

    for row in ROWS:
        lines.append(
            f"| `{row['key']}` | {fmt(row['cr1_k'])} | {fmt(row['tail_df'])} "
            f"| **{row['status']}** | {row['reason']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(emit_markdown_table())


# ---------------------------------------------------------------------------
# absorbed_fe_rank (D3 fix): unit + end-to-end verification arms.
# ---------------------------------------------------------------------------


class TestClusterKIncrement:
    """The K_reference helpers: nested-dim classification + the CR1 k
    increment (variance-conventions.md D2), on the shared audit panel."""

    def test_nested_dim_classification(self):
        from diff_diff.utils import cluster_nested_fe_dims

        df = make_panel()
        u, t = df["unit"].values, df["time"].values
        assert cluster_nested_fe_dims(df, ["unit", "time"], u) == ["unit"]
        assert cluster_nested_fe_dims(df, ["unit", "time"], t) == ["time"]
        # coarser cluster: unit is a function of grp's preimage -> nested
        assert cluster_nested_fe_dims(df, ["unit", "time"], df["grp"].values) == ["unit"]
        # constructed crossing cluster: varies WITHIN units and periods
        assert cluster_nested_fe_dims(df, ["unit", "time"], (u + t) % 2) == []
        assert cluster_nested_fe_dims(df, [], u) == []

    def test_increment_arithmetic_checks(self):
        """The five checks that retrodict the externally-verified arms."""
        from diff_diff.utils import absorbed_fe_cr1_k_increment as incr

        df = make_panel()
        u, t = df["unit"].values, df["time"].values
        # audit panel absorb form: 0 + 65 - 60 = 5 -> K = 2 + 5 = 7
        assert incr(df, ["unit", "time"], u, has_intercept_col=True) == 5
        # Wooldridge within (no intercept col): 1 + 65 - 60 = 6 = T
        assert incr(df, ["unit", "time"], u, has_intercept_col=False) == 6
        # cluster=year inverts nesting: 0 + 65 - 6 = 59 -> K = 61 = 66 - 5
        assert incr(df, ["unit", "time"], t, has_intercept_col=True) == 59
        # ZERO nested dims (the max(...,1) floor case): 0 + 65 - 1 = 64
        assert incr(df, ["unit", "time"], (u + t) % 2, has_intercept_col=True) == 64
        # no absorbed FE -> 0 regardless of intercept flag (the +1 is the
        # ABSORBED constant's rank, absent without absorbed dims)
        assert incr(df, [], u, has_intercept_col=False) == 0

    def test_empty_effective_support_returns_zero(self):
        """All rows zero-weight: increment 0, never -1 via the floor."""
        from diff_diff.utils import absorbed_fe_cr1_k_increment as incr

        df = make_panel()
        u = df["unit"].values
        w = np.zeros(len(df))
        assert incr(df, ["unit", "time"], u, has_intercept_col=True, weights=w) == 0
        assert incr(df, ["unit", "time"], u, has_intercept_col=False, weights=w) == 0

    def test_positive_weight_rows_only(self):
        """Zero-weight padding is inference-invariant (REGISTRY guarantee)."""
        from diff_diff.utils import absorbed_fe_cr1_k_increment as incr

        df = make_panel()
        u = df["unit"].values
        base = incr(df, ["unit", "time"], u, has_intercept_col=True)
        padded = pd.concat(
            [df, pd.DataFrame({"unit": [999], "time": [0], "grp": [0]})],
            ignore_index=True,
        )
        w = np.r_[np.ones(len(df)), 0.0]
        cl = np.r_[u, 999]
        assert incr(padded, ["unit", "time"], cl, has_intercept_col=True, weights=w) == base

    def test_nan_keys_raise_actionable(self):
        from diff_diff.utils import absorbed_fe_cr1_k_increment as incr

        df = make_panel().astype({"unit": "float64"})
        u = df["unit"].values.copy()
        df.loc[0, "unit"] = np.nan
        with pytest.raises(ValueError, match="'unit' contains NaN group keys"):
            incr(df, ["unit", "time"], u, has_intercept_col=True)
        df2 = make_panel()
        cl = df2["unit"].values.astype("float64").copy()
        cl[0] = np.nan
        with pytest.raises(ValueError, match="cluster_ids contain missing values"):
            incr(df2, ["unit", "time"], cl, has_intercept_col=True)

    def test_hierarchical_nested_pair_rank(self):
        """Two nested dims (unit + state, cluster=state): the nested rank is
        the component-aware PAIR rank, not sum(levels-1) — the discriminating
        property the full-dummy negative adjustment relies on."""
        from diff_diff.utils import absorbed_fe_rank, cluster_nested_fe_dims

        df = make_panel()
        df["state"] = df["unit"] % 6
        state = df["state"].values
        nested = cluster_nested_fe_dims(df, ["unit", "state"], state)
        assert nested == ["unit", "state"]
        pair_rank = absorbed_fe_rank(df, ["unit", "state"], has_intercept_col=True)
        naive = (60 - 1) + (6 - 1)
        assert pair_rank == 59  # sum(levels) - C - 1 = 66 - 6 - 1, C = n_states
        assert pair_rank < naive

    def test_three_dim_increment_inherits_the_documented_d3_approximation(self):
        """LIMITATION PIN (documented deviation, not an endorsement): for
        THREE or more absorbed dims the increment inherits absorbed_fe_rank's
        D3 ``sum(levels) - N + 1`` approximation, which OVER-counts on
        duplicated/nested triples. On a(5) x b(4) with c == b, cluster = a:
        the true dummy-space rank is 5 + 4 - 1 = 8 (c adds nothing), so the
        true increment is 8 - 5 = 3; the formula returns
        (5 + 4 + 4) - 3 + 1 = 11 -> increment 6. Pinned so the day the exact
        N-way rank ships (TODO.md N-way row; REGISTRY absorbed-FE note) this
        test fails loudly and flips to the exact value. One- and two-dim
        increments are exact (the tests above)."""
        from diff_diff.utils import absorbed_fe_cr1_k_increment

        n = np.arange(80)
        df = pd.DataFrame({"a": n % 5, "b": (n // 5) % 4})
        df["c"] = df["b"]  # duplicated dim: adds NOTHING to the span
        inc3 = absorbed_fe_cr1_k_increment(
            df, ["a", "b", "c"], df["a"].to_numpy(), has_intercept_col=True
        )
        inc2_true = absorbed_fe_cr1_k_increment(
            df, ["a", "b"], df["a"].to_numpy(), has_intercept_col=True
        )
        assert inc2_true == 3  # exact two-dim: (5 + 4 - 1) - 5
        assert inc3 == 6  # the documented D3 over-count (true value: 3)


class TestAbsorbedFeRank:
    """Component-aware absorbed-FE rank: bit-identity where C=1, correctness
    where the old ``sum(levels - 1)`` count over-stated rank."""

    @staticmethod
    def _connected():
        return pd.DataFrame(
            {"unit": np.repeat(np.arange(60), 6), "time": np.tile(np.arange(6), 60)}
        )

    @staticmethod
    def _disconnected(seed=11, effect=0.0):
        """Two period-disjoint unit blocks: {1,2} and {3,4,5,6} (C=2).

        The block boundary is deliberately NOT the ``post`` cut — a symmetric
        3/3 split makes ``post`` unit-constant and the treatment column gets
        rank-dropped, leaving no finite SE to assert.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(30):
            for t in (1, 2):
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        grp=int(u % 2),
                        post=int(t == 2),
                        y=rng.normal() + 0.1 * u + effect * (u % 2) * (t == 2),
                    )
                )
        for u in range(30, 60):
            for t in (3, 4, 5, 6):
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        grp=int(u % 2),
                        post=int(t >= 5),
                        y=rng.normal() + 0.1 * u + effect * (u % 2) * (t >= 5),
                    )
                )
        return pd.DataFrame(rows)

    # ---- unit cases -------------------------------------------------------

    def test_connected_two_way_reproduces_historical_counts(self):
        from diff_diff.utils import absorbed_fe_rank

        d = self._connected()
        # intercept-bearing designs (TWFE, DiD/MPD absorb): n_units + n_times - 2
        assert absorbed_fe_rank(d, ["unit", "time"], has_intercept_col=True) == 64
        # no-intercept design (SunAbraham): n_units + n_times - 1
        assert absorbed_fe_rank(d, ["unit", "time"], has_intercept_col=False) == 65

    def test_hierarchical_dims_counted_as_components(self):
        """absorb=["state", "state_year"]: each state is its own component, so
        the absorbed rank is that of the finer dimension alone (measured true
        rank 29 vs the old count 34)."""
        from diff_diff.utils import absorbed_fe_rank

        h = pd.DataFrame(
            [
                {"state": s, "state_year": s * 100 + y}
                for s in range(6)
                for y in range(5)
                for _ in range(4)
            ]
        )
        assert absorbed_fe_rank(h, ["state", "state_year"], has_intercept_col=True) == 29

    def test_disconnected_two_way(self):
        from diff_diff.utils import absorbed_fe_rank

        d = self._disconnected()
        assert absorbed_fe_rank(d, ["unit", "time"], has_intercept_col=True) == 63

    def test_single_and_three_way_unchanged(self):
        from diff_diff.utils import absorbed_fe_rank

        d = self._connected()
        assert absorbed_fe_rank(d, ["unit"], has_intercept_col=True) == 59
        rng = np.random.default_rng(0)
        t3 = pd.DataFrame(
            {
                "a": rng.integers(0, 5, 400),
                "b": rng.integers(0, 4, 400),
                "c": rng.integers(0, 4, 400),
            }
        )
        assert absorbed_fe_rank(t3, ["a", "b", "c"], has_intercept_col=True) == 10

    def test_three_way_duplicated_dim_is_a_known_over_count(self):
        """Pinned LIMITATION, not correct behavior: for N>=3 the helper keeps
        sum(levels-1), which over-counts when dimensions are duplicated/nested
        (true rank beyond intercept here is 7). Tracked in TODO.md."""
        from diff_diff.utils import absorbed_fe_rank

        rng = np.random.default_rng(0)
        a = rng.integers(0, 5, 400)
        b = rng.integers(0, 4, 400)
        t3 = pd.DataFrame({"a": a, "b": b, "c": b.copy()})
        assert absorbed_fe_rank(t3, ["a", "b", "c"], has_intercept_col=True) == 10

    def test_zero_weight_rows_do_not_contribute_levels_or_edges(self):
        """REGISTRY guarantee: zero-weight padding is inference-invariant on the
        generic paths — a level carried only by zero-weight rows adds no df."""
        from diff_diff.utils import absorbed_fe_rank

        d = self._connected()
        padded = pd.concat(
            [d, pd.DataFrame({"unit": [999, 999], "time": [0, 1]})],
            ignore_index=True,
        )
        w = np.r_[np.ones(len(d)), np.zeros(2)]
        assert absorbed_fe_rank(padded, ["unit", "time"], has_intercept_col=True, weights=w) == 64

    # ---- end-to-end: the df change reaches user-visible inference ---------

    def test_disconnected_end_to_end_did_absorb(self):
        """On the C=2 panel the absorbed df is 63 (old count: 64). The
        non-clustered hc1 rescale uses it, so the SE discriminates between the
        two counts; the clustered lane and the full inference tuple stay
        finite."""
        df = self._disconnected(effect=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_un = diff_diff.DifferenceInDifferences(vcov_type="hc1").fit(
                df, outcome="y", treatment="grp", time="post", absorb=["unit", "time"]
            )
            r_cl = diff_diff.DifferenceInDifferences(cluster="unit").fit(
                df, outcome="y", treatment="grp", time="post", absorb=["unit", "time"]
            )
        # new count (adj=63): measured; old count (adj=64) would be ~0.44% larger
        np.testing.assert_allclose(r_un.se, 0.2811268249, rtol=1e-8)
        old_count_se = r_un.se * np.sqrt(
            ((180 - 2) / (180 - 2 - 64)) / ((180 - 2) / (180 - 2 - 63))
        )
        assert abs(r_un.se - old_count_se) / r_un.se > 0.003
        for v in (r_cl.se, r_cl.t_stat, r_cl.p_value, *r_cl.conf_int):
            assert np.isfinite(v)

    def test_disconnected_end_to_end_twfe_and_mpd(self):
        df = self._disconnected(effect=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tw = diff_diff.TwoWayFixedEffects(vcov_type="hc1", cluster="unit").fit(
                df, outcome="y", treatment="grp", time="time", unit="unit"
            )
            mpd = diff_diff.MultiPeriodDiD(cluster="unit").fit(
                df, outcome="y", treatment="grp", time="time", absorb=["unit", "time"]
            )
        assert np.isfinite(tw.se)
        # Discriminating df pins: the C=2 rank lowers the absorbed adjustment
        # by 1, so each caller's reported residual df moves off the legacy
        # value - reverting either caller to sum(levels-1) reads 114.0 (TWFE)
        # / 111.0 (MPD) on this fixture and fails here.
        assert tw.inference_df == 115.0
        assert mpd.inference_df == 112.0
        assert mpd.event_study_df is not None
        assert set(mpd.event_study_df.values()) == {112.0}
        # at least one period effect identified and finite on each component
        finite = [e for e in mpd.period_effects.values() if np.isfinite(e.se)]
        assert finite, "no finite MPD period effect on the disconnected panel"

    def test_fail_closed_boundary_moves_both_directions(self):
        """Lowering the absorbed df by C-1 moves the n - k - adj boundary.

        Newly-finite direction: 8 obs, 2 components of 2x2 -> old adj 6 gives
        n-k-adj = 0 (NaN vcov); new adj 5 gives 1 (finite SE). Still-NaN
        direction: drop one row -> n-k-adj = 0 under the NEW count too.
        """
        rng = np.random.default_rng(5)
        rows = []
        for us, ts in [((0, 1), (1, 2)), ((2, 3), (3, 4))]:
            for u in us:
                for t in ts:
                    rows.append(
                        dict(
                            unit=u,
                            time=t,
                            grp=int(u % 2),
                            post=int(t in (2, 4)),
                            y=rng.normal() + 0.3 * u,
                        )
                    )
        tiny = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = diff_diff.DifferenceInDifferences(vcov_type="hc1").fit(
                tiny, outcome="y", treatment="grp", time="post", absorb=["unit", "time"]
            )
        assert np.isfinite(r.se), "newly-finite direction: SE must be finite now"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = diff_diff.DifferenceInDifferences(vcov_type="hc1").fit(
                tiny.iloc[:-1],
                outcome="y",
                treatment="grp",
                time="post",
                absorb=["unit", "time"],
            )
        assert not np.isfinite(r2.se), "saturated design must stay fail-closed NaN"
        assert not np.isfinite(r2.p_value) and not np.isfinite(r2.conf_int[0])

    def test_disconnected_end_to_end_sun_abraham(self):
        """SunAbraham has no public df surface, so capture LinearRegression.df_
        at the fit boundary: on the C=2 panel the no-intercept raw rank is
        levels - C = 43, giving df_ = 100 - 3 - 43 = 54 (old count: 53)."""
        import diff_diff.linalg as lmod

        rng = np.random.default_rng(13)
        rows = []
        for u in range(20):
            ft = 0 if u < 10 else 2
            for t in (1, 2):
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        first_treat=ft,
                        y=rng.normal() + 0.2 * u + 1.0 * (ft > 0 and t >= ft),
                    )
                )
        for u in range(20, 40):
            ft = 0 if u < 30 else 5
            for t in (4, 5, 6):
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        first_treat=ft,
                        y=rng.normal() + 0.2 * u + 1.0 * (ft > 0 and t >= ft),
                    )
                )
        df = pd.DataFrame(rows)
        captured = []
        orig = lmod.LinearRegression.fit

        def spy(self, X, y, **kw):
            out = orig(self, X, y, **kw)
            captured.append(self.df_)
            return out

        lmod.LinearRegression.fit = spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = diff_diff.SunAbraham().fit(
                    df, outcome="y", unit="unit", time="time", first_treat="first_treat"
                )
        finally:
            lmod.LinearRegression.fit = orig
        assert (
            captured and captured[-1] == 54
        ), f"SA df_ {captured} != 54 (the old sum(levels-1) count gives 53)"
        assert np.isfinite(r.overall_se)

    def test_nan_group_key_raises_actionable_error(self):
        """A NaN absorb key must name the offending column, not surface as
        scipy's 'negative axis 0 index: -1'."""
        from diff_diff.utils import absorbed_fe_rank

        d = self._connected().astype({"unit": float})
        d.loc[3, "unit"] = np.nan
        with pytest.raises(ValueError, match="'unit' contains NaN group keys"):
            absorbed_fe_rank(d, ["unit", "time"], has_intercept_col=True)


def test_committed_doc_table_matches_generator():
    """The inventory doc's table is GENERATED — assert the committed markdown
    block is byte-equal to ``emit_markdown_table()`` so the doc cannot drift
    from the fixture (regenerate: ``python -m tests.test_variance_conventions``)."""
    import pathlib

    doc = pathlib.Path(__file__).parent.parent / "docs" / "methodology" / "variance-conventions.md"
    if not doc.exists():
        # CI's wheel-install legs run the test suite from a temp copy outside
        # the repo checkout, where docs/ does not exist (same convention as
        # the benchmarks/data golden skip guards). The checkout-based legs
        # (pure Python fallback, local runs) still enforce the byte-equality.
        pytest.skip("variance-conventions.md not present (run outside the repo checkout)")
    text = doc.read_text()
    generated = emit_markdown_table()
    header = generated.splitlines()[0]
    start = text.find(header)
    assert start >= 0, "generated table header not found in variance-conventions.md"
    committed = text[start : start + len(generated)]
    assert committed == generated, (
        "committed inventory table is stale — regenerate with "
        "`python -m tests.test_variance_conventions` and paste into the doc"
    )


# ---------------------------------------------------------------------------
# R parity: the component-aware rank matches fixest ssc(K.exact = TRUE).
# ---------------------------------------------------------------------------

_KEXACT_GOLDEN = (
    __import__("pathlib").Path(__file__).parent.parent
    / "benchmarks"
    / "data"
    / "fixest_kexact_golden.json"
)


@pytest.mark.skipif(
    not _KEXACT_GOLDEN.exists(),
    reason=(
        "fixest_kexact_golden.json not present; regenerate via "
        "`Rscript benchmarks/R/generate_fixest_kexact_golden.R`."
    ),
)
class TestFixestKExactParity:
    """On a hierarchical two-way design (state_year nested in state, C=6) the
    component-aware absorbed rank matches ``fixest::ssc(K.exact = TRUE)`` at
    machine precision — a documented deviation from the R DEFAULT
    (``K.exact = FALSE``), whose naive count reproduces the library's OLD
    ``sum(levels - 1)`` behavior."""

    @staticmethod
    def _load():
        import json

        with open(_KEXACT_GOLDEN) as fh:
            return json.load(fh)

    def test_classical_se_matches_k_exact_not_default(self):
        from diff_diff.linalg import LinearRegression
        from diff_diff.utils import absorbed_fe_rank, demean_by_groups

        g = self._load()
        d = pd.DataFrame(g["data"])
        # +1: the demeaned design carries no intercept column, but fixest's K
        # counts one; absorbed_fe_rank(has_intercept_col=True) returns the
        # rank BEYOND the intercept.
        adj = absorbed_fe_rank(d, ["state", "state_year"], has_intercept_col=True) + 1
        dm, _ = demean_by_groups(d.copy(), ["out", "x"], ["state", "state_year"])
        reg = LinearRegression(include_intercept=False, robust=False).fit(
            dm[["x"]].values, dm["out"].values, df_adjustment=adj
        )
        se = float(np.sqrt(reg.vcov_[0, 0]))
        np.testing.assert_allclose(reg.coefficients_[0], g["coef"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(se, g["iid_k_exact"]["se"], rtol=0, atol=1e-12)
        assert reg.df_ == g["n_obs"] - g["iid_k_exact"]["df_k"]
        # Discriminating: the R DEFAULT (naive count) must NOT match.
        assert abs(se - g["iid_default"]["se"]) / se > 0.02

    def test_rank_matches_fixest_exact_k(self):
        from diff_diff.utils import absorbed_fe_rank

        g = self._load()
        d = pd.DataFrame(g["data"])
        rank_beyond_intercept = absorbed_fe_rank(d, ["state", "state_year"], has_intercept_col=True)
        # fixest df.K = x (1) + intercept (1) + absorbed rank
        assert rank_beyond_intercept + 2 == g["iid_k_exact"]["df_k"]
        # and the naive count reproduces the R DEFAULT
        naive = (d["state"].nunique() - 1) + (d["state_year"].nunique() - 1)
        assert naive + 2 == g["iid_default"]["df_k"]


# ---------------------------------------------------------------------------
# R + Stata parity: the clustered CR1 non-nested RANK term (K_reference D2).
# ---------------------------------------------------------------------------

_CR1_NONNESTED_GOLDEN = (
    __import__("pathlib").Path(__file__).parent.parent
    / "benchmarks"
    / "data"
    / "fixest_cr1_nonnested_golden.json"
)

_REGHDFE_KREF_GOLDEN = (
    __import__("pathlib").Path(__file__).parent.parent
    / "benchmarks"
    / "data"
    / "reghdfe_kref_golden.json"
)


def _reghdfe_kref_frame() -> pd.DataFrame:
    """The generator's deterministic DGP, rebuilt from the same integer
    formulas (no RNG, no embedded data): a disconnected two-way panel --
    units 0-9 observed in periods 0-4, units 10-19 in periods 5-9 (C=2)."""
    n = np.arange(1, 101)
    unit = (n - 1) // 5
    time = (n - 1) % 5 + 5 * (unit >= 10)
    x = ((n * 7) % 13 - 6) / 13
    z = ((n * 11) % 17 - 8) / 17
    out = 0.5 * x + 0.2 * (unit % 3) + 0.1 * time + z
    return pd.DataFrame({"unit": unit, "time": time, "x": x, "out": out, "c5": (unit + time) % 5})


def _demeaned_cluster_fit(d: pd.DataFrame, cluster_col: str):
    """Absorb [unit, time], fit x on the demeaned design with the wired
    K_reference adjustment; returns (coef, se, increment)."""
    from diff_diff.linalg import LinearRegression
    from diff_diff.utils import absorbed_fe_cr1_k_increment, demean_by_groups

    inc = absorbed_fe_cr1_k_increment(
        d, ["unit", "time"], d[cluster_col].to_numpy(), has_intercept_col=False
    )
    dm, _ = demean_by_groups(d.copy(), ["out", "x"], ["unit", "time"])
    reg = LinearRegression(include_intercept=False, cluster_ids=d[cluster_col].to_numpy()).fit(
        dm[["x"]].values, dm["out"].values, cluster_k_adjustment=inc
    )
    return float(reg.coefficients_[0]), float(np.sqrt(reg.vcov_[0, 0])), inc


@pytest.mark.skipif(
    not _CR1_NONNESTED_GOLDEN.exists(),
    reason=(
        "fixest_cr1_nonnested_golden.json not present; regenerate via "
        "`Rscript benchmarks/R/generate_fixest_cr1_nonnested_golden.R`."
    ),
)
class TestFixestCr1NonNestedParity:
    """The non-nested absorbed-FE RANK term in the clustered CR1 k, anchored
    against fixest on a DISCONNECTED two-way panel (C=2, span rank
    U + T - C = 28).

    ``crossed_cluster`` (nothing nested) is the parity arm: the library's
    K = x + exact span = 29 matches ``ssc(K.fixef="full", K.exact=TRUE)`` at
    machine precision and DIFFERS from both the default approximate count
    (df.K=30) and the K.exact-under-nested-K.fixef quirk (df.K=28 -- the
    clustered nested path removes 1 df even with nothing nested). It also
    exercises the zero-nested-dim ``max(..., 1)`` floor end-to-end.

    ``unit_cluster`` is the documentation arm: NO fixest ssc implements
    nested-drop + exact-remainder (library K = 10); the default lands on 11
    (approximate remainder T-1) and the deviation is pinned as an exact
    one-df SE ratio."""

    @staticmethod
    def _load():
        import json

        with open(_CR1_NONNESTED_GOLDEN) as fh:
            return json.load(fh)

    def test_crossed_cluster_matches_k_exact_full(self):
        g = self._load()
        d = pd.DataFrame(g["disconnected"]["data"])
        arm = g["disconnected"]["crossed_cluster"]
        coef, se, inc = _demeaned_cluster_fit(d, "c5")
        assert inc == 28  # exact span U + T - C, nothing nested (the floor case)
        assert 1 + inc == arm["k_exact"]["df_k"] == 29
        np.testing.assert_allclose(coef, arm["coef"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(se, arm["k_exact"]["se"], rtol=0, atol=1e-12)
        # Discriminating: the default approximate count (30) and the
        # nested-K.fixef quirk (28) must NOT match.
        assert arm["default"]["df_k"] == 30
        assert arm["k_exact_nested"]["df_k"] == 28
        n = g["disconnected"]["n_obs"]
        for other in ("default", "k_exact_nested"):
            assert abs(se - arm[other]["se"]) > 1e-4
        np.testing.assert_allclose(
            arm["default"]["se"] / se, np.sqrt((n - 29) / (n - 30)), rtol=1e-12
        )

    def test_unit_cluster_deviation_is_exactly_one_df(self):
        """The nested COMPOSITION arm: library K=10 (exact remainder 8),
        fixest default df.K=11 (approximate remainder T-1=9). Pinned as the
        exact one-df ratio so any drift in either convention fails loudly."""
        g = self._load()
        d = pd.DataFrame(g["disconnected"]["data"])
        arm = g["disconnected"]["unit_cluster"]
        coef, se, inc = _demeaned_cluster_fit(d, "unit")
        assert inc == 9  # 1 (constant) + 28 (span) - 20 (nested unit rank)
        assert arm["default"]["df_k"] == 11
        assert arm["nested_k_exact"]["df_k"] == 28  # the incoherent composition
        np.testing.assert_allclose(coef, arm["coef"], rtol=0, atol=1e-12)
        n = g["disconnected"]["n_obs"]
        np.testing.assert_allclose(
            arm["default"]["se"] / se, np.sqrt((n - 10) / (n - 11)), rtol=1e-12
        )

    def test_connected_control_matches_default_ssc(self):
        """At C=1 with a nested unit cluster the conventions coincide
        (remainder T-1 exactly): parity against plain default ssc."""
        g = self._load()
        d = pd.DataFrame(g["connected"]["data"])
        coef, se, inc = _demeaned_cluster_fit(d, "unit")
        assert 1 + inc == g["connected"]["cluster_default"]["df_k"] == 11
        np.testing.assert_allclose(coef, g["connected"]["coef"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(se, g["connected"]["cluster_default"]["se"], rtol=0, atol=1e-12)


@pytest.mark.skipif(
    not _REGHDFE_KREF_GOLDEN.exists(),
    reason=(
        "reghdfe_kref_golden.json not present; regenerate via "
        "`stata-se -b do benchmarks/stata/generate_reghdfe_kref_golden.do`."
    ),
)
class TestReghdfeKReferenceParity:
    """The same disconnected design against Stata reghdfe (dof pairwise), the
    second reference. ``cross_cluster`` agrees with the library (and with
    fixest full+K.exact) at machine precision; ``unit_cluster`` documents
    that reghdfe TOO approximates the nested remainder (df_a = 9 = T-1,
    implied K = 11), so the library's exact-remainder K = 10 matches no
    external reference on disconnected nested designs -- pinned as the same
    exact one-df ratio. On CONNECTED designs the compositions coincide and
    the jwdid subsample ladder pins machine-precision agreement at every
    cluster count."""

    @staticmethod
    def _load():
        import json

        with open(_REGHDFE_KREF_GOLDEN) as fh:
            return json.load(fh)

    def test_cross_cluster_exact_span_parity(self):
        g = self._load()["cross_cluster"]
        d = _reghdfe_kref_frame()
        coef, se, inc = _demeaned_cluster_fit(d, "c5")
        # reghdfe's df_a IS the exact span here (constants included);
        # its denominator N - rank - df_a == N - (1 + inc) == N - 29.
        assert g["df_a"] == inc == 28
        assert g["df_a_nested"] == 0
        assert g["df_a_redundant"] == 2  # the C=2 pairwise correction
        assert g["df_r"] == g["G"] - 1
        np.testing.assert_allclose(coef, g["coef"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(se, g["se"], rtol=0, atol=1e-12)

    def test_unit_cluster_composition_deviation_is_exactly_one_df(self):
        g = self._load()["unit_cluster"]
        d = _reghdfe_kref_frame()
        coef, se, inc = _demeaned_cluster_fit(d, "unit")
        assert inc == 9  # library: 1 + 28 - 20 (exact remainder)
        # reghdfe: nested drop (20) but APPROXIMATE remainder -- redundant
        # counts only the global constant, so df_a = T - 1 = 9 and the
        # implied K = rank + df_a + constant = 11 vs the library's 10.
        assert g["df_a"] == 9
        assert g["df_a_nested"] == 20
        assert g["df_a_redundant"] == 21
        assert g["df_r"] == g["G"] - 1
        np.testing.assert_allclose(coef, g["coef"], rtol=0, atol=1e-12)
        n = g["n"]
        np.testing.assert_allclose(g["se"] / se, np.sqrt((n - 10) / (n - 11)), rtol=1e-12)


class TestKReferenceConvergence:
    """End-to-end properties of the D1/D2 K_reference convergence that the
    matrix rows cannot express."""

    def test_two_nested_dim_full_dummy_uses_rank_not_column_count(self):
        """The ONLY design that discriminates rank from raw dummy count.

        With fixed_effects=["unit","state"] and cluster="state" (state =
        unit % 6), BOTH dims are cluster-nested and hierarchical: the pair
        rank incl. the constant is 66 - 6 - 1 = 59, while the raw drop-first
        dummy count is 59 + 5 = 64. The state dummies are collinear with the
        unit dummies, so the kernel keeps 62 of 66 columns — and 62 - 59 = 3
        equals the absorb form's K exactly (the collinearity-drop equivalence
        two reviewers proved algebraically: k is the design RANK, so
        K_ref(full) == K_ref(absorb) identically). A raw-count adjustment
        (-64) would give k_inf = -2 and a NaN vcov instead.
        """
        df = make_panel()
        df = df.assign(state=df["unit"] % 6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = diff_diff.DifferenceInDifferences(cluster="state").fit(
                df, outcome="y", treatment="grp", time="post", absorb=["unit", "state"]
            )
            f = diff_diff.DifferenceInDifferences(cluster="state").fit(
                df, outcome="y", treatment="grp", time="post", fixed_effects=["unit", "state"]
            )
        assert np.isfinite(a.se) and a.se > 0
        np.testing.assert_allclose(f.se, a.se, rtol=1e-12)

    @staticmethod
    def _saturation_panel(n_units, n_periods, drop=0):
        """Deterministic (RNG-free) absorbed-fit fixture; the mod-arithmetic
        noise term is outside the unit+time+interaction span on the larger
        grids, so above-boundary fits carry genuine residuals."""
        rows = []
        for u in range(n_units):
            for t in range(n_periods):
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        grp=u % 2,
                        post=int(t >= n_periods - 1),
                        y=0.5 * u + 0.25 * t + 0.3 * ((u * 13 + t * 7 + (u * t) % 11) % 17),
                    )
                )
        df = pd.DataFrame(rows)
        return df.iloc[: len(df) - drop].reset_index(drop=True) if drop else df

    def _saturation_fit(self, df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return diff_diff.DifferenceInDifferences(cluster="unit").fit(
                df, outcome="y", treatment="grp", time="post", absorb=["unit", "time"]
            )

    def test_clustered_fail_closed_below_the_boundary(self):
        """Clustered absorbed fits whose effective dof are exhausted fail
        closed to the FULL all-NaN inference tuple — never a silent ~0 SE.

        Both fixtures sit at or below the composed boundary (n = 4 and n = 5
        against K_reference = 4 with absorbed rank 4 on the U=2, T=3 grid;
        verified by construction via absorbed_fe_cr1_k_increment == 2 with
        kernel k == 2). The exact two-sided n - k_inf boundary in isolation
        is pinned at the seam (test_linalg
        TestClusterKAdjustmentSeam::test_fail_closed_k_inf_exact_boundary),
        where the tail-df lane cannot fire first; here the assertion is the
        estimator-level composition: ALL-NaN, no mixed tuple, no finite ~0.
        """
        from diff_diff.utils import absorbed_fe_cr1_k_increment

        for drop in (2, 1):
            df = self._saturation_panel(2, 3, drop=drop)
            inc = absorbed_fe_cr1_k_increment(
                df, ["unit", "time"], df["unit"].to_numpy(), has_intercept_col=True
            )
            assert inc == 2  # K_reference = kernel k (2) + 2 = 4 >= n - 1
            r = self._saturation_fit(df)
            vals = (r.se, r.t_stat, r.p_value, *r.conf_int)
            assert all(np.isnan(v) for v in vals), f"n={len(df)}: expected all-NaN, got {vals}"

    def test_clustered_finite_above_the_boundary(self):
        """Clearly above the boundary (U=4, T=4: n=16, K_reference = 2 + 3,
        absorbed rank 7 -> 7 genuine residual dof) the same configuration
        returns a complete finite tuple with a NON-degenerate SE — the
        recovery side, guarding against an over-eager fail-closed."""
        r = self._saturation_fit(self._saturation_panel(4, 4))
        vals = (r.se, r.t_stat, r.p_value, *r.conf_int)
        assert all(np.isfinite(v) for v in vals)
        assert r.se > 1e-8, f"degenerate ~0 SE on a non-saturated fit: {r.se}"

    def test_wcb_identity_and_p_invariance(self):
        """WCB: se == sqrt(vcov[att, att]) exactly on adjusted fits across
        the absorb, fixed_effects, and TWFE hc2 full-dummy lanes; p-values
        are invariant to the corr change (the factor cancels in |t*| vs |t0|)
        and CI endpoints move only within the bisection tolerance."""
        df = make_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_ab = diff_diff.DifferenceInDifferences(
                cluster="unit", inference="wild_bootstrap", n_bootstrap=99, seed=42
            ).fit(df, outcome="y", treatment="grp", time="post", absorb=["unit", "time"])
            r_fe = diff_diff.DifferenceInDifferences(
                cluster="unit", inference="wild_bootstrap", n_bootstrap=99, seed=42
            ).fit(df, outcome="y", treatment="grp", time="post", fixed_effects=["unit", "time"])
            r_tw = diff_diff.TwoWayFixedEffects(
                vcov_type="hc2",
                cluster="unit",
                inference="wild_bootstrap",
                n_bootstrap=99,
                seed=42,
            ).fit(df, outcome="y", treatment="grp", time="post", unit="unit")
        assert r_ab.se == np.sqrt(r_ab.vcov[3, 3])
        assert r_fe.se == np.sqrt(r_fe.vcov[3, 3])
        assert r_tw.se == np.sqrt(r_tw.vcov[1, 1])
        # the two idioms agree on the corrected SE exactly (cross-design
        # bootstrap p-values are NOT compared: draws tie at |t0| within
        # float noise, so p can differ by one bootstrap step legitimately)
        np.testing.assert_allclose(r_ab.se, r_fe.se, rtol=1e-12)

    def test_wcb_p_invariant_to_the_adjustment(self):
        """The corr constant cancels in |t*| vs |t0|: for a FIXED design and
        seed, the bootstrap p-value is IDENTICAL under any adjustment, the
        reported SE scales by exactly the K_reference factor, and CI
        endpoints agree to the bisection tolerance (never exactly)."""
        from diff_diff.utils import wild_bootstrap_se

        rng = np.random.default_rng(9)
        n, G = 120, 12
        X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
        y = X @ np.array([0.5, 0.3, 0.0]) + rng.normal(size=n)
        cl = np.repeat(np.arange(G), n // G)
        r0 = wild_bootstrap_se(X, y, np.zeros(n), cl, 1, n_bootstrap=199, seed=7)
        r5 = wild_bootstrap_se(
            X, y, np.zeros(n), cl, 1, n_bootstrap=199, seed=7, cluster_k_adjustment=5
        )
        assert r5.p_value == r0.p_value
        expect = np.sqrt((n - 3) / (n - 3 - 5))
        np.testing.assert_allclose(r5.se / r0.se, expect, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            r5.t_stat_original * r5.se, r0.t_stat_original * r0.se, rtol=1e-12
        )
        np.testing.assert_allclose(
            (r5.ci_lower, r5.ci_upper), (r0.ci_lower, r0.ci_upper), rtol=1e-6
        )

    def test_lpdid_never_swallows_the_contract_raise(self, monkeypatch):
        """LPDiD's broad except around the clustered solve must RE-RAISE
        InvalidClusterKAdjustment rather than degrade to a silent unclustered
        se=NaN refit (the no-silent-failure rule)."""
        import diff_diff.lpdid as lp
        from diff_diff.linalg import InvalidClusterKAdjustment

        df = make_panel()
        orig = lp.solve_ols

        def inject_bad_adjustment(*a, **k):
            if k.get("cluster_ids") is not None:
                k["cluster_k_adjustment"] = 10**6  # forces n - k_inf <= 0... valid int
                k["vcov_type"] = "hc2_bm"  # nonzero adj + non-hc1 -> front door raises
            return orig(*a, **k)

        monkeypatch.setattr(lp, "solve_ols", inject_bad_adjustment)
        with pytest.raises(InvalidClusterKAdjustment):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diff_diff.LPDiD(pre_window=2, post_window=2).fit(
                    df, outcome="y", unit="unit", time="time", treatment="treated"
                )
