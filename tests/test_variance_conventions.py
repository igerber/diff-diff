"""Audit matrix: which clustered-variance convention each estimator surface gets.

This is the CI-enforced inventory behind ``docs/methodology/variance-conventions.md``.
Each row pins, for one (estimator, fit configuration) cell:

- ``cr1_k``   — the sorted multiset of visible column counts ``k`` reaching the
  shared clustered CR1 denominator (``linalg._compute_robust_vcov_numpy``), or
  ``()`` when the surface's contract is that it makes NO shared-CR1 call.
- ``tail_df`` — the sorted multiset of ``df`` values passed to
  ``safe_inference`` / ``safe_inference_batch`` (``None`` = normal theory).

The point is visibility, not endorsement: several pinned values are DOCUMENTED
DEFECTS (anti-conservative k accounting, mixed tail-df conventions) scheduled to
change in the 3.9 consolidation program. Every row carries ``status`` and, for
legitimate differences, a ``reason``. When a later PR changes a convention, the
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
                    self.cr1_k.append(int(b.arguments["X"].shape[1]))
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
        cr1_k=(2,),
        tail_df=(294.0,),
        status="defect",
        reason="D2: CR1 k omits absorbed FE not nested in the cluster (time)",
    ),
    dict(
        key="did_fixed_effects_hc1_cluster_unit",
        fit=lambda df: diff_diff.DifferenceInDifferences(cluster="unit").fit(
            df, outcome="y", treatment="grp", time="post", fixed_effects=["unit", "time"]
        ),
        cr1_k=(66,),
        tail_df=(294.0,),
        status="defect",
        reason=(
            "D1: same model as did_absorb yet k=66 vs 2 -> SEs differ 10.35%; "
            "full-dummy k also counts the cluster-nested unit FE the references drop"
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
        cr1_k=(2,),
        tail_df=(298.0,),
        status="defect",
        reason="D2 (within-transform k_visible); tail df is residual n-K_full",
    ),
    dict(
        key="wooldridge_hc1_within",
        fit=lambda df: diff_diff.WooldridgeDiD(method="ols").fit(
            df, outcome="y", unit="unit", time="time", cohort="first_treat"
        ),
        cr1_k=(9,),
        tail_df=(None,) * 10,
        status="defect",
        reason="D2 (k_visible=cells only) + normal-theory tail df with no df_convention knob",
    ),
    dict(
        key="sun_abraham_hc1",
        fit=lambda df: diff_diff.SunAbraham().fit(
            df, outcome="y", unit="unit", time="time", first_treat="first_treat"
        ),
        cr1_k=(15,),
        tail_df=(280.0,) * 15 + (None,) * 8,
        status="defect",
        reason="D2 + D4: residual df per cohort-period cell but normal theory on aggregates",
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
        cr1_k=None,  # resolved at collection: must be NON-empty (D2 applies here)
        tail_df=None,
        status="defect",
        reason="pretrends lead regression runs the shared clustered CR1 with k_visible",
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


def test_d1_divergence_is_pinned():
    """The absorb-vs-fixed_effects SE split: same model, same ATT, k=2 vs 66."""
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
    ratio = f.se / a.se
    predicted = np.sqrt((360 - 2) / (360 - 66))
    np.testing.assert_allclose(ratio, predicted, rtol=1e-9)


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
