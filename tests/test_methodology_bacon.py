"""
Methodology verification tests for BaconDecomposition.

Targets the Goodman-Bacon (2021) decomposition theorem:

    β̂^DD = Σ_{k≠U} s_{kU} · β̂_{kU}^{2x2}
         + Σ_{k≠U} Σ_{ℓ>k} [s_{kℓ}^k · β̂_{kℓ}^{2x2,k} + s_{kℓ}^ℓ · β̂_{kℓ}^{2x2,ℓ}]   (Eq. 10a)

with weights from Eqs. 7-9 + 10e-g. See:
- ``docs/methodology/papers/goodman-bacon-2021-review.md``  paper review
- ``docs/methodology/REGISTRY.md`` ``## BaconDecomposition`` block
- ``METHODOLOGY_REVIEW.md`` ``BaconDecomposition`` section

Test class breakdown:
- ``TestBaconHandCalculation`` — hand-calculable balanced panel; sum-to-1,
  TWFE-vs-weighted-sum identity, per-Eq variance hand-checks at atol=1e-10.
- ``TestBaconParityR`` — R parity at atol=1e-6 against the committed
  ``benchmarks/data/r_bacondecomp_golden.json`` (skip if missing).
- ``TestBaconAlwaysTreatedRemap`` — warn+remap of first_treat <= min(time)
  (excluding never-treated sentinels 0 and np.inf) to U; user's first_treat
  column preserved unchanged.
- ``TestBaconEdgeCases`` — no-untreated, single-cohort, boundary D̄_k,
  unbalanced panel, constant-ATT recovery.
- ``TestBaconWeightModes`` — exact-is-default, approximate-opt-in,
  exact-vs-approximate differ meaningfully.
- ``TestBaconSurveyDesignNarrowing`` — survey_design composes cleanly
  with the remap and the new exact-mode default.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    BaconDecomposition,
    bacon_decompose,
)


def _bacon_fit(
    data,
    outcome=None,
    unit=None,
    time=None,
    first_treat=None,
    *,
    weights="exact",
    survey_design=None,
    **kw,
):
    """Construct-and-fit via the canonical class API (2(d) PR-A, M-076).

    Migrated from the deprecated ``bacon_decompose()`` wrapper: ``weights``
    is a constructor kwarg, everything else goes to ``fit()``.
    """
    return BaconDecomposition(weights=weights).fit(
        data,
        outcome=outcome,
        unit=unit,
        time=time,
        first_treat=first_treat,
        survey_design=survey_design,
        **kw,
    )


# ---------------------------------------------------------------------------
# Hand-calculable DGP
# ---------------------------------------------------------------------------
#
# Balanced panel with two timing groups + never-treated, T = 4 periods,
# 3 units per group. The treatment shares are:
#
#   D̄_2 = (T - 2 + 1) / T = 3/4 = 0.75
#   D̄_3 = (T - 3 + 1) / T = 2/4 = 0.50
#   D̄_U = 0
#
# Sample shares (n_k = units in group / total units):
#
#   n_2 = 3/9 = 1/3
#   n_3 = 3/9 = 1/3
#   n_U = 3/9 = 1/3
#
# Theorem 1 weights (numerators of Eqs. 10e-g, before V̂^D normalization):
#
#   s_{2U}    ∝ (n_2 + n_U)^2 · V̂_{2U}^D    = (2/3)^2 · (1/2)(1/2)(3/4)(1/4)  = 0.020833...
#   s_{3U}    ∝ (n_3 + n_U)^2 · V̂_{3U}^D    = (2/3)^2 · (1/2)(1/2)(1/2)(1/2)  = 0.027778...
#   s_{23}^k  ∝ ((n_2+n_3)(1-D̄_3))^2 · V̂^{D,k}_{23}  = ((2/3)(1/2))^2 · 0.0625 = 0.006944...
#   s_{23}^ℓ  ∝ ((n_2+n_3)·D̄_2)^2 · V̂^{D,ℓ}_{23}    = ((2/3)(3/4))^2 · 0.05555 = 0.013889...
#
# V̂^D = Σ numerators = 0.069444... → normalized weights:
#
#   s_{2U}   = 0.3
#   s_{3U}   = 0.4
#   s_{23}^k = 0.1
#   s_{23}^ℓ = 0.2
#
# With constant ATT=5 across cohorts and periods (no noise), every 2x2 DD
# equals 5, so β̂^DD = 5 × (0.3 + 0.4 + 0.1 + 0.2) = 5 exactly.


def _hand_calc_panel(true_effect: float = 5.0) -> pd.DataFrame:
    """Build the hand-calculable 2-cohort + U panel described above.

    No noise. Outcome y = unit_fe + 0.5*time + true_effect*D, with unit
    fixed effects spaced so each unit's intercept is distinct.
    """
    rows: List[Tuple] = []
    uid = 1
    # Group 2 (first_treat=2): 3 units, 4 periods each
    for _ in range(3):
        for t in range(1, 5):
            D = 1 if t >= 2 else 0
            y = float(uid) * 10.0 + 0.5 * t + true_effect * D
            rows.append((uid, t, y, 2))
        uid += 1
    # Group 3 (first_treat=3): 3 units
    for _ in range(3):
        for t in range(1, 5):
            D = 1 if t >= 3 else 0
            y = float(uid) * 10.0 + 0.5 * t + true_effect * D
            rows.append((uid, t, y, 3))
        uid += 1
    # Never-treated U (first_treat=0): 3 units
    for _ in range(3):
        for t in range(1, 5):
            y = float(uid) * 10.0 + 0.5 * t
            rows.append((uid, t, y, 0))
        uid += 1
    return pd.DataFrame(rows, columns=["unit", "time", "y", "first_treat"])


def _staggered_data(seed: int = 42) -> pd.DataFrame:
    """Larger DGP for sum-to-1 / TWFE-equals-sum checks beyond the toy case."""
    rng = np.random.default_rng(seed)
    n_units = 60
    n_periods = 8
    cohort_periods = np.array([3, 5, 7])
    # 20% never-treated, evenly split across cohorts otherwise
    cohort_assign = rng.choice(
        np.concatenate([[0], cohort_periods]),
        size=n_units,
        p=[0.2, 0.27, 0.27, 0.26],
    )
    rows = []
    for u in range(n_units):
        ft = cohort_assign[u]
        unit_fe = rng.standard_normal() * 2.0
        for t in range(1, n_periods + 1):
            D = int(ft > 0 and t >= ft)
            y = unit_fe + 0.05 * t + 2.0 * D + rng.standard_normal() * 0.5
            rows.append((u + 1, t, y, int(ft)))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "first_treat"])


# ---------------------------------------------------------------------------
# 1. Hand-calculable Theorem 1 verification
# ---------------------------------------------------------------------------


class TestBaconHandCalculation:
    """Theorem 1 / Eqs. 7-9 + 10e-g verified on a minimal balanced panel.

    The hand-calculable DGP at module scope was hand-derived to produce
    normalized weights ``{0.3, 0.4, 0.1, 0.2}`` and TWFE = ATT = 5 exactly
    (constant treatment effect, no noise). Tolerances are tight (1e-10)
    because the identity is purely algebraic.
    """

    def test_weights_sum_to_one(self) -> None:
        df = _hand_calc_panel()
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        total = sum(c.weight for c in results.comparisons)
        assert abs(total - 1.0) < 1e-10

    def test_twfe_equals_weighted_sum(self) -> None:
        """Theorem 1's algebraic identity at machine precision."""
        df = _hand_calc_panel()
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        weighted_sum = sum(c.weight * c.estimate for c in results.comparisons)
        assert abs(results.twfe_estimate - weighted_sum) < 1e-10
        # Also check the dataclass-cached value.
        assert results.decomposition_error < 1e-10

    def test_three_comparison_types_present(self) -> None:
        df = _hand_calc_panel()
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        types = {c.comparison_type for c in results.comparisons}
        assert "treated_vs_never" in types
        assert "earlier_vs_later" in types
        assert "later_vs_earlier" in types

    def test_eq_10b_treated_vs_never_value(self) -> None:
        """β̂_{2U}^{2x2} = 5 on the hand-calc panel (constant ATT, no noise)."""
        df = _hand_calc_panel()
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        treated_vs_never = [
            c for c in results.comparisons if c.comparison_type == "treated_vs_never"
        ]
        # All treated_vs_never comparisons should yield ATT = 5
        for comp in treated_vs_never:
            assert (
                abs(comp.estimate - 5.0) < 1e-10
            ), f"{comp.treated_group} vs U: estimate={comp.estimate}, expected 5.0"

    def test_eq_7_treated_untreated_variance(self) -> None:
        """V̂_{kU}^D = n_kU(1-n_kU) · D̄_k(1-D̄_k) — verify via weight-share decomp.

        s_{2U} / s_{3U} = (n_2+n_U)^2 · V̂_{2U}^D / ((n_3+n_U)^2 · V̂_{3U}^D)
        With equal cohort sizes, this simplifies to:
        V̂_{2U}^D / V̂_{3U}^D = (D̄_2(1-D̄_2)) / (D̄_3(1-D̄_3))
                            = (0.75 · 0.25) / (0.5 · 0.5)
                            = 0.1875 / 0.25 = 0.75
        So s_{2U} / s_{3U} = 0.75 (= 0.3 / 0.4).
        """
        df = _hand_calc_panel()
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        s2U = next(
            c.weight
            for c in results.comparisons
            if c.comparison_type == "treated_vs_never" and c.treated_group == 2
        )
        s3U = next(
            c.weight
            for c in results.comparisons
            if c.comparison_type == "treated_vs_never" and c.treated_group == 3
        )
        # Expected normalized: s2U=0.3, s3U=0.4 → ratio = 0.75
        assert abs(s2U / s3U - 0.75) < 1e-10

    def test_eq_8_earlier_vs_later_variance(self) -> None:
        """V̂_{kℓ}^{D,k}: weight s_{23}^k normalized expected = 0.1.

        Per the hand calc: s_{23}^k = 0.006944... / 0.069444... = 0.1
        """
        df = _hand_calc_panel()
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        s_kl_k = next(
            c.weight
            for c in results.comparisons
            if c.comparison_type == "earlier_vs_later"
            and c.treated_group == 2
            and c.control_group == 3
        )
        assert abs(s_kl_k - 0.1) < 1e-10

    def test_eq_9_later_vs_earlier_variance(self) -> None:
        """V̂_{kℓ}^{D,ℓ}: weight s_{23}^ℓ normalized expected = 0.2.

        Per the hand calc: s_{23}^ℓ = 0.013889... / 0.069444... = 0.2
        """
        df = _hand_calc_panel()
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        s_kl_l = next(
            c.weight
            for c in results.comparisons
            if c.comparison_type == "later_vs_earlier"
            and c.treated_group == 3
            and c.control_group == 2
        )
        assert abs(s_kl_l - 0.2) < 1e-10


# ---------------------------------------------------------------------------
# 2. R parity (skip if golden JSON not committed)
# ---------------------------------------------------------------------------

_R_GOLDEN_PATH = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "r_bacondecomp_golden.json"
)


def _load_r_golden() -> dict:
    if not _R_GOLDEN_PATH.exists():
        pytest.skip(
            f"R parity goldens missing at {_R_GOLDEN_PATH}. To regenerate, "
            "install R + `install.packages('bacondecomp')` + "
            "`install.packages('jsonlite')` then `cd benchmarks/R && "
            "Rscript generate_bacon_golden.R`. The goldens are committed "
            "to the repo by default; this skip path covers partial-checkout "
            "or packaging scenarios where the JSON file is unavailable."
        )
    return json.loads(_R_GOLDEN_PATH.read_text())


class TestBaconParityR:
    """R `bacondecomp::bacon()` parity at atol=1e-6 (when goldens present)."""

    @pytest.fixture(scope="class")
    def golden(self) -> dict:
        return _load_r_golden()

    def test_twfe_coef_matches_r(self, golden) -> None:
        for fixture_name, fix in golden.items():
            if fixture_name == "meta":
                continue
            panel = pd.DataFrame(fix["panel"])
            results = _bacon_fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
            )
            assert abs(results.twfe_estimate - fix["r_twfe_coef"]) < 1e-6, (
                f"{fixture_name}: TWFE Python={results.twfe_estimate} " f"vs R={fix['r_twfe_coef']}"
            )

    def test_weights_sum_matches_r(self, golden) -> None:
        for fixture_name, fix in golden.items():
            if fixture_name == "meta":
                continue
            panel = pd.DataFrame(fix["panel"])
            results = _bacon_fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
            )
            py_sum = sum(c.weight for c in results.comparisons)
            assert abs(py_sum - fix["r_weights_sum"]) < 1e-6, (
                f"{fixture_name}: weight sum Python={py_sum} " f"vs R={fix['r_weights_sum']}"
            )

    def test_component_estimates_match_r(self, golden) -> None:
        """Match per-component estimates at atol=1e-6 across Python + R.

        Join key is ``(comparison_type, treated_group_float, control_canonical)``
        where ``control_canonical`` is the literal ``"U"`` for
        ``treated_vs_never`` comparisons (Python stores the string
        ``"never_treated"``; R may use ``Inf`` or its own string) and the
        float-coerced cohort timing for timing-vs-timing comparisons.
        Including ``comparison_type`` in the key disambiguates the two
        directions of a timing pair (earlier_vs_later vs later_vs_earlier).
        """

        def _canonical_control(ctype: str, group):
            if ctype == "treated_vs_never":
                return "U"
            return float(group)

        def _classify_r_type(c: dict, fixture_name: str) -> str:
            # R bacondecomp's `type` strings vary across versions
            # ("Treated vs Untreated", "Earlier vs Later Treated",
            # "Later vs Always Treated", ...). Fall back to inferring from
            # the control_group: U sentinel (0, np.inf, or "never"-containing
            # string) -> treated_vs_never; otherwise treated_group <
            # control_group is earlier-vs-later. Note: ``Later vs Always
            # Treated`` is canonicalized to ``treated_vs_never`` here because
            # Python's paper-footnote-11 convention folds always-treated
            # units into the U bucket — semantically these R rows belong
            # to the U comparison set even though R numbers them by the
            # always-treated cohort (typically first_treat=1).
            t = c.get("type") or ""
            if "never" in t.lower() or "untreated" in t.lower():
                return "treated_vs_never"
            if "always" in t.lower():
                return "treated_vs_never"
            ctrl = c["control_group"]
            if isinstance(ctrl, str) and "never" in ctrl.lower():
                return "treated_vs_never"
            if isinstance(ctrl, (int, float)) and (ctrl == 0 or np.isinf(ctrl)):
                return "treated_vs_never"
            try:
                if float(c["treated_group"]) < float(ctrl):
                    return "earlier_vs_later"
                return "later_vs_earlier"
            except (TypeError, ValueError):
                pytest.fail(
                    f"{fixture_name}: cannot classify R component "
                    f"treated={c.get('treated_group')} "
                    f"control={ctrl} type={t!r}"
                )

        for fixture_name, fix in golden.items():
            if fixture_name == "meta":
                continue
            panel = pd.DataFrame(fix["panel"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                results = _bacon_fit(
                    panel,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    weights="exact",
                )
            py_estimates = {}
            py_weights = {}
            for c in results.comparisons:
                key = (
                    c.comparison_type,
                    float(c.treated_group),
                    _canonical_control(c.comparison_type, c.control_group),
                )
                py_estimates[key] = c.estimate
                py_weights[key] = c.weight
            r_estimates: dict = {}
            r_weights: dict = {}
            for c in fix["r_components"]:
                ctype = _classify_r_type(c, fixture_name)
                key = (
                    ctype,
                    float(c["treated_group"]),
                    _canonical_control(ctype, c["control_group"]),
                )
                r_estimates[key] = c["estimate"]
                r_weights[key] = c["weight"]
            # ``always_treated_remapped`` carves out only the U-bucket rows,
            # which R and Python decompose under different conventions
            # (R: separate ``Later vs Always Treated`` + ``Treated vs
            # Untreated``; Python: single ``treated_vs_never`` per cohort
            # via paper-footnote-11 remap). The aggregated fold-back is
            # asserted in ``test_always_treated_remapped_fold_back_matches_r``.
            # The 6 timing-vs-timing rows in that fixture are NOT affected
            # by the convention split and must satisfy direct per-component
            # parity at atol=1e-6 — narrow the carve-out to U-bucket keys
            # only so regressions in timing-vs-timing decomposition are
            # caught directly, not just through aggregate parity.
            if fixture_name == "always_treated_remapped":
                # Drop only treated_vs_never keys from both sides; keep
                # earlier_vs_later + later_vs_earlier for direct parity.
                py_estimates = {k: v for k, v in py_estimates.items() if k[0] != "treated_vs_never"}
                py_weights = {k: v for k, v in py_weights.items() if k[0] != "treated_vs_never"}
                r_estimates = {k: v for k, v in r_estimates.items() if k[0] != "treated_vs_never"}
                r_weights = {k: v for k, v in r_weights.items() if k[0] != "treated_vs_never"}
            # Full-set equality: no Python component missing from R, no R
            # component missing from Python. A dropped β̂_{kU} term or an
            # extra spurious comparison would fail here.
            py_keys = set(py_estimates)
            r_keys = set(r_estimates)
            missing_in_r = py_keys - r_keys
            missing_in_py = r_keys - py_keys
            assert not missing_in_r and not missing_in_py, (
                f"{fixture_name}: component sets differ. "
                f"In Python but not R: {sorted(missing_in_r)}. "
                f"In R but not Python: {sorted(missing_in_py)}."
            )
            # Per-component estimate AND weight parity.
            for k in py_keys:
                assert abs(py_estimates[k] - r_estimates[k]) < 1e-6, (
                    f"{fixture_name} {k}: estimate Python={py_estimates[k]} "
                    f"vs R={r_estimates[k]}"
                )
                assert abs(py_weights[k] - r_weights[k]) < 1e-6, (
                    f"{fixture_name} {k}: weight Python={py_weights[k]} " f"vs R={r_weights[k]}"
                )

    def test_always_treated_remapped_fold_back_matches_r(self, golden) -> None:
        """Pin the documented R→Python fold-back for the always-treated U bucket.

        The per-component test above carves out **only the U-bucket rows**
        from ``always_treated_remapped`` (the 6 timing-vs-timing rows are
        still asserted directly at atol=1e-6); R and Python decompose the
        U bucket differently — but the documented REGISTRY claim is that
        **aggregating** R's `Later vs Always Treated` + `Treated vs
        Untreated` rows by treated cohort matches Python's single
        `treated_vs_never` cell for that cohort. Assert that fold-back
        directly so a cohort-level regression can't slip through under
        overall TWFE parity.

        For each treated cohort k:
        - R: combined weight w_R = w(k vs always-treated) + w(k vs untreated)
          and weight-weighted estimate e_R = Σ w_i * e_i / w_R
        - Python: single treated_vs_never component (w_Py, e_Py)
        - Assert |w_Py - w_R| < 1e-6 AND |e_Py - e_R| < 1e-6.
        """
        if "always_treated_remapped" not in golden:
            pytest.skip("always_treated_remapped fixture not in goldens")
        fix = golden["always_treated_remapped"]
        panel = pd.DataFrame(fix["panel"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            results = _bacon_fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
            )
        # Build Python's treated_vs_never lookup: cohort -> (weight, estimate)
        py_tvn = {
            float(c.treated_group): (c.weight, c.estimate)
            for c in results.comparisons
            if c.comparison_type == "treated_vs_never"
        }
        # Aggregate R's two U-bucket types per treated cohort.
        # R uses ctrl=99999 for untreated and ctrl=1 (the always-treated cohort)
        # for the `Later vs Always Treated` rows. Match on case-insensitive
        # semantic tokens so the selector survives `bacondecomp` label
        # variation across versions (same convention as the neighboring
        # ``_classify_r_type`` helper used by the per-component test).
        r_agg: dict = {}
        for c in fix["r_components"]:
            tlow = (c.get("type") or "").lower()
            is_untreated = "untreated" in tlow or "never" in tlow
            is_always_treated_compare = "always" in tlow
            if is_untreated or is_always_treated_compare:
                k = float(c["treated_group"])
                w = float(c["weight"])
                e = float(c["estimate"])
                if k not in r_agg:
                    r_agg[k] = [0.0, 0.0]  # [sum_w, sum_w_e]
                r_agg[k][0] += w
                r_agg[k][1] += w * e
        # Cohorts must match
        assert set(py_tvn.keys()) == set(r_agg.keys()), (
            f"always_treated_remapped: treated_vs_never cohorts differ. "
            f"Python: {sorted(py_tvn)}, R-aggregated: {sorted(r_agg)}"
        )
        for k, (py_w, py_e) in py_tvn.items():
            r_w, r_we = r_agg[k]
            r_e = r_we / r_w
            assert abs(py_w - r_w) < 1e-6, (
                f"always_treated_remapped cohort={k}: combined weight "
                f"Python={py_w:.10f} vs R-aggregated={r_w:.10f}"
            )
            assert abs(py_e - r_e) < 1e-6, (
                f"always_treated_remapped cohort={k}: weight-averaged estimate "
                f"Python={py_e:.10f} vs R-aggregated={r_e:.10f}"
            )


# ---------------------------------------------------------------------------
# 3. Always-treated warn+remap
# ---------------------------------------------------------------------------


def _panel_with_always_treated() -> pd.DataFrame:
    """Build a panel with a small always-treated cohort (first_treat=1)."""
    rng = np.random.default_rng(7)
    rows = []
    uid = 1
    # 4 always-treated units (first_treat=1; treated in every observable period)
    for _ in range(4):
        unit_fe = rng.standard_normal() * 2.0
        for t in range(1, 7):
            y = unit_fe + 0.05 * t + 2.0 + rng.standard_normal() * 0.3
            rows.append((uid, t, y, 1))
        uid += 1
    # 10 never-treated
    for _ in range(10):
        unit_fe = rng.standard_normal() * 2.0
        for t in range(1, 7):
            y = unit_fe + 0.05 * t + rng.standard_normal() * 0.3
            rows.append((uid, t, y, 0))
        uid += 1
    # 10 cohort-3 + 10 cohort-5
    for cohort in (3, 5):
        for _ in range(10):
            unit_fe = rng.standard_normal() * 2.0
            for t in range(1, 7):
                D = int(t >= cohort)
                y = unit_fe + 0.05 * t + 2.0 * D + rng.standard_normal() * 0.3
                rows.append((uid, t, y, cohort))
            uid += 1
    return pd.DataFrame(rows, columns=["unit", "time", "y", "first_treat"])


class TestBaconAlwaysTreatedRemap:
    """Goodman-Bacon (2021) footnote 11 with the library's first-period
    boundary convention.

    The paper's footnote 11 says units treated before the first observable
    period (``t_i < 1`` under the paper's 1-indexed convention) belong in
    ``U``. The library generalizes this to units whose
    ``first_treat <= min(time)`` (i.e. includes ``first_treat == min(time)``,
    which the paper's strict ``<`` shorthand excludes). The library
    convention is pragmatic: units treated at the first observable period
    have no untreated cell within the panel and cannot contribute to any
    valid 2x2 DD as a treated cohort, so folding them into ``U`` mirrors
    the always-treated handling. The never-treated sentinels
    (``first_treat ∈ {0, np.inf}``) are excluded from the remap.

    bacon.py applies the remap via an internal column
    (``__bacon_first_treat_internal__``), preserving the user's original
    ``first_treat`` column unchanged. Detection uses ordered-time logic
    on the **time axis**, so event-time-encoded panels
    (``time ∈ [-2,..,3]``) are handled correctly.
    """

    def test_warn_emitted_on_remap(self) -> None:
        df = _panel_with_always_treated()
        with pytest.warns(UserWarning, match="Remapping to U bucket"):
            _bacon_fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
            )

    def test_user_first_treat_column_unchanged(self) -> None:
        """Regression test: user's data should not be mutated by the remap."""
        df = _panel_with_always_treated()
        original = df["first_treat"].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            _bacon_fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
            )
        pd.testing.assert_series_equal(df["first_treat"], original)

    def test_n_always_treated_remapped_reported(self) -> None:
        df = _panel_with_always_treated()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            results = _bacon_fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
            )
        # 4 units were always-treated (first_treat=1)
        assert results.n_always_treated_remapped == 4

    def test_treated_vs_never_emitted_when_U_is_only_remapped_always_treated(
        self,
    ) -> None:
        """Regression test for the R1 P0 finding.

        When the user supplies NO ``first_treat ∈ {0, np.inf}`` units (so
        ``n_never_treated == 0``) but supplies always-treated units that
        the remap reclassifies into ``U``, the ``treated_vs_never``
        comparison loop must still fire — gated on the POST-remap U
        bucket, not the pre-remap never-treated count. Otherwise
        Theorem 1's algebraic identity breaks because all
        ``β̂_{kU}^{2x2}`` terms are silently dropped.
        """
        rng = np.random.default_rng(13)
        rows = []
        uid = 1
        # 8 always-treated units (first_treat=1) — these become the entire U bucket
        for _ in range(8):
            unit_fe = rng.standard_normal() * 2.0
            for t in range(1, 7):
                y = unit_fe + 0.05 * t + 2.0 + rng.standard_normal() * 0.3
                rows.append((uid, t, y, 1))
            uid += 1
        # 8 cohort-3 + 8 cohort-5 (no true never-treated)
        for cohort in (3, 5):
            for _ in range(8):
                unit_fe = rng.standard_normal() * 2.0
                for t in range(1, 7):
                    D = int(t >= cohort)
                    y = unit_fe + 0.05 * t + 2.0 * D + rng.standard_normal() * 0.3
                    rows.append((uid, t, y, cohort))
                uid += 1
        df = pd.DataFrame(rows, columns=["unit", "time", "y", "first_treat"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            results = _bacon_fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
            )
        # n_never_treated counts TRUE never-treated only (user column = 0 or inf).
        # All 8 U-bucket units came from the always-treated remap, so this is 0.
        assert results.n_never_treated == 0
        assert results.n_always_treated_remapped == 8
        # CRITICAL: treated_vs_never comparisons MUST be present because the
        # remapped U bucket is the paper's `U` per footnote 11.
        types = {c.comparison_type for c in results.comparisons}
        assert "treated_vs_never" in types, (
            "Theorem 1 violation: treated_vs_never comparisons were "
            "silently dropped when U is composed of remapped always-treated "
            "units only (pre-fix R1 P0 regression)."
        )
        # Decomposition identity must hold at machine precision.
        weighted_sum = sum(c.weight * c.estimate for c in results.comparisons)
        assert abs(results.twfe_estimate - weighted_sum) < 1e-10
        assert results.decomposition_error < 1e-10

    def test_negative_first_treat_as_valid_timing_group(self) -> None:
        """Regression for nonpositive-time encodings (event-time panels).

        On a panel with ``time = [-2, -1, 0, 1, 2, 3]``, a cohort with
        ``first_treat = -1`` is a valid timing group with one observable
        pre-period (t=-2). Detection must use ordered-time logic, not
        positive-sign restriction. Prior to the R3 P0 fix this cohort
        was silently dropped (both the remap mask and the timing_groups
        filter required ``first_treat > 0``), violating Theorem 1.
        """
        rng = np.random.default_rng(73)
        rows = []
        uid = 1
        # 10 never-treated U
        for _ in range(10):
            unit_fe = rng.standard_normal() * 2.0
            for t in range(-2, 4):
                y = unit_fe + 0.05 * t + rng.standard_normal() * 0.3
                rows.append((uid, t, y, 0))
            uid += 1
        # 8 cohort treated at t=-1 (valid timing group, one pre-period)
        for _ in range(8):
            unit_fe = rng.standard_normal() * 2.0
            for t in range(-2, 4):
                D = int(t >= -1)
                y = unit_fe + 0.05 * t + 2.0 * D + rng.standard_normal() * 0.3
                rows.append((uid, t, y, -1))
            uid += 1
        # 8 cohort treated at t=2
        for _ in range(8):
            unit_fe = rng.standard_normal() * 2.0
            for t in range(-2, 4):
                D = int(t >= 2)
                y = unit_fe + 0.05 * t + 2.0 * D + rng.standard_normal() * 0.3
                rows.append((uid, t, y, 2))
            uid += 1
        df = pd.DataFrame(rows, columns=["unit", "time", "y", "first_treat"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            results = _bacon_fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
            )
        # -1 cohort should appear as a timing group, NOT remapped to U
        assert -1 in results.timing_groups
        assert 2 in results.timing_groups
        assert results.n_always_treated_remapped == 0
        # Both cohorts should produce treated_vs_never comparisons
        types_per_treated = {(c.treated_group, c.comparison_type) for c in results.comparisons}
        assert (-1, "treated_vs_never") in types_per_treated
        assert (2, "treated_vs_never") in types_per_treated
        # Theorem 1 identity holds at machine precision.
        weighted_sum = sum(c.weight * c.estimate for c in results.comparisons)
        assert abs(results.twfe_estimate - weighted_sum) < 1e-10
        assert results.decomposition_error < 1e-10

    def test_negative_first_treat_below_min_time_remapped(self) -> None:
        """``first_treat=-3`` on a panel with ``min(time)=-2`` is always-treated.

        The unit's first treatment is before the first observable period,
        so per paper footnote 11 it goes in U. Detection must use
        ordered-time logic (``first_treat <= min(time)``), not positive-
        sign restriction.
        """
        rng = np.random.default_rng(74)
        rows = []
        uid = 1
        # 6 always-treated (first_treat=-3, pre-panel)
        for _ in range(6):
            unit_fe = rng.standard_normal() * 2.0
            for t in range(-2, 4):
                y = unit_fe + 0.05 * t + 2.0 + rng.standard_normal() * 0.3
                rows.append((uid, t, y, -3))
            uid += 1
        # 8 cohort treated at t=2
        for _ in range(8):
            unit_fe = rng.standard_normal() * 2.0
            for t in range(-2, 4):
                D = int(t >= 2)
                y = unit_fe + 0.05 * t + 2.0 * D + rng.standard_normal() * 0.3
                rows.append((uid, t, y, 2))
            uid += 1
        df = pd.DataFrame(rows, columns=["unit", "time", "y", "first_treat"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            results = _bacon_fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
            )
        assert results.n_always_treated_remapped == 6
        assert results.n_never_treated == 0
        # 2 should be the only timing group (always-treated are remapped to U).
        assert 2 in results.timing_groups
        assert -3 not in results.timing_groups
        # treated_vs_never must fire because U is non-empty post-remap.
        types = {c.comparison_type for c in results.comparisons}
        assert "treated_vs_never" in types
        # Theorem 1 identity holds at machine precision.
        weighted_sum = sum(c.weight * c.estimate for c in results.comparisons)
        assert abs(results.twfe_estimate - weighted_sum) < 1e-10
        assert results.decomposition_error < 1e-10

    def test_no_warning_on_sentinel_only_inputs(self) -> None:
        """When first_treat ∈ {0, np.inf} only, no remap warning fires."""
        df = _staggered_data(seed=11)
        # Replace any first_treat=1 (shouldn't be any in this DGP, but defensive)
        df = df[df["first_treat"] != 1]
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)
            try:
                _bacon_fit(
                    df,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    weights="exact",
                )
            except UserWarning as w:
                if "Remapping to U bucket" in str(w):
                    pytest.fail(f"Unexpected remap warning on sentinel-only inputs: {w}")


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------


class TestBaconEdgeCases:
    """Edge cases enumerated in REGISTRY.md ## BaconDecomposition."""

    def test_no_untreated_group(self) -> None:
        """Only timing groups, no U. Weights still sum to 1 after normalization."""
        df = _staggered_data(seed=22)
        # Drop never-treated
        df = df[df["first_treat"] > 0].copy()
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        total = sum(c.weight for c in results.comparisons)
        assert abs(total - 1.0) < 1e-10
        # No treated_vs_never terms — only timing-only
        type_set = {c.comparison_type for c in results.comparisons}
        assert "treated_vs_never" not in type_set

    def test_single_timing_group_with_never_treated(self) -> None:
        """K=1 (one timing group) + U: only treated_vs_never comparisons."""
        rng = np.random.default_rng(33)
        rows = []
        uid = 1
        for _ in range(10):
            for t in range(1, 6):
                D = int(t >= 3)
                y = float(uid) + 0.1 * t + 2.0 * D + rng.standard_normal() * 0.2
                rows.append((uid, t, y, 3))
            uid += 1
        for _ in range(10):
            for t in range(1, 6):
                y = float(uid) + 0.1 * t + rng.standard_normal() * 0.2
                rows.append((uid, t, y, 0))
            uid += 1
        df = pd.DataFrame(rows, columns=["unit", "time", "y", "first_treat"])
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        type_set = {c.comparison_type for c in results.comparisons}
        assert type_set == {"treated_vs_never"}
        total = sum(c.weight for c in results.comparisons)
        assert abs(total - 1.0) < 1e-10

    def test_unbalanced_panel_warns(self) -> None:
        df = _staggered_data(seed=44)
        # Drop a few rows to unbalance
        df = df.drop(df.sample(n=10, random_state=44).index).copy()
        with pytest.warns(UserWarning, match="Unbalanced panel"):
            _bacon_fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
            )

    def test_unbalanced_panel_finite_but_not_machine_precision(self) -> None:
        """Unbalanced panels are a library extension (Goodman-Bacon Appendix A
        proof assumes balanced panels). Decomposition still produces finite,
        well-defined output, but the Theorem 1 identity holds only
        approximately — do NOT assert ``decomposition_error < 1e-10``
        on unbalanced data; the REGISTRY Deviation block documents the
        deviation explicitly.
        """
        df = _staggered_data(seed=44)
        df = df.drop(df.sample(n=10, random_state=44).index).copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            results = _bacon_fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
            )
        # All outputs are finite
        assert np.isfinite(results.twfe_estimate)
        assert all(np.isfinite(c.weight) for c in results.comparisons)
        assert all(np.isfinite(c.estimate) for c in results.comparisons)
        # Weights still sum to ~1 after post-hoc normalization
        total = sum(c.weight for c in results.comparisons)
        assert abs(total - 1.0) < 1e-10
        # But the decomposition identity is NOT machine-precision on
        # unbalanced data — REGISTRY's machine-precision claim is scoped
        # to balanced panels (see Deviation note in REGISTRY entry).
        # Empirically the error is small (well under 0.01) but not 1e-10.

    def test_constant_att_recovers_effect(self) -> None:
        """ΔATT=0 + VWCT=0 → β̂^DD ≈ true ATT (sample noise only)."""
        df = _staggered_data(seed=55)
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        # True ATT = 2.0 in the DGP. With 60 units and clean DGP, the TWFE
        # estimate should be within a few hundredths of truth.
        assert abs(results.twfe_estimate - 2.0) < 0.2

    def test_weighted_sum_machine_precision(self) -> None:
        """The TWFE-vs-weighted-sum identity holds on noisy data too."""
        df = _staggered_data(seed=66)
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        weighted_sum = sum(c.weight * c.estimate for c in results.comparisons)
        assert abs(results.twfe_estimate - weighted_sum) < 1e-10


# ---------------------------------------------------------------------------
# 5. Weight modes
# ---------------------------------------------------------------------------


class TestBaconWeightModes:
    """``weights="exact"`` is the paper-faithful default; ``"approximate"`` opt-in."""

    def test_exact_is_default(self) -> None:
        est = BaconDecomposition()
        assert est.weights == "exact"

    def test_approximate_still_supported(self) -> None:
        est = BaconDecomposition(weights="approximate")
        assert est.weights == "approximate"
        df = _staggered_data(seed=77)
        results = est.fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        # Approximate mode produces valid output with weights summing to 1
        total = sum(c.weight for c in results.comparisons)
        assert abs(total - 1.0) < 0.01

    def test_exact_vs_approximate_differ_meaningfully(self) -> None:
        """The two modes produce different relative weights (the approximate
        path uses a simplified variance not matching Eqs. 7-9)."""
        df = _staggered_data(seed=88)
        r_exact = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
        )
        r_approx = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="approximate",
        )
        # Both have the same set of (treated, control) comparisons
        assert len(r_exact.comparisons) == len(r_approx.comparisons)
        # But at least one weight differs by more than 1e-6 (i.e., the modes
        # are not numerically equivalent on a non-trivial DGP)
        exact_map = {
            (c.treated_group, c.control_group, c.comparison_type): c.weight
            for c in r_exact.comparisons
        }
        approx_map = {
            (c.treated_group, c.control_group, c.comparison_type): c.weight
            for c in r_approx.comparisons
        }
        diffs = [abs(exact_map[k] - approx_map[k]) for k in exact_map if k in approx_map]
        assert max(diffs) > 1e-6


# ---------------------------------------------------------------------------
# 6. Survey-design narrowing
# ---------------------------------------------------------------------------


class TestBaconSurveyDesignNarrowing:
    """survey_design= composes cleanly with the warn+remap and the new exact-mode default."""

    def test_survey_design_compatible_with_exact_mode(self) -> None:
        """``weights="exact"`` (new default) accepts ``survey_design=`` without
        path-specific assertion failures.
        """
        from diff_diff import SurveyDesign

        df = _staggered_data(seed=99)
        # Constant-within-unit weights (required by exact-mode survey validator)
        unit_w = df.groupby("unit").ngroup() * 0.1 + 1.0
        df = df.assign(w=unit_w.values)
        sd = SurveyDesign(weights="w")
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="exact",
            survey_design=sd,
        )
        # Sum-to-1 contract still holds under survey weighting
        total = sum(c.weight for c in results.comparisons)
        assert abs(total - 1.0) < 1e-10

    def test_survey_design_propagates_through_remap(self) -> None:
        """When always-treated remap fires and survey_design is set, no crash
        and the survey metadata is preserved in the result."""
        from diff_diff import SurveyDesign

        df = _panel_with_always_treated()
        unit_w = df.groupby("unit").ngroup() * 0.1 + 1.0
        df = df.assign(w=unit_w.values)
        sd = SurveyDesign(weights="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            results = _bacon_fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                weights="exact",
                survey_design=sd,
            )
        assert results.n_always_treated_remapped == 4
        assert results.survey_metadata is not None

    def _time_varying_survey_panel(self) -> "tuple[pd.DataFrame, object]":
        """Build a panel with time-varying within-unit survey weights.

        Used to exercise the post-PR-B default flip: ``weights="exact"``
        (now the default) routes through ``_validate_unit_constant_survey``
        and rejects time-varying within-unit weights; ``weights="approximate"``
        accepts them via observation-level weighted means.
        """
        from diff_diff import SurveyDesign

        np.random.seed(42)
        n_u, n_t = 20, 4
        df = pd.DataFrame(
            {
                "unit": np.repeat(range(n_u), n_t),
                "time": np.tile(range(1, n_t + 1), n_u),
                "first_treat": np.repeat(np.where(np.arange(n_u) < 10, 3, 0), n_t),
                "y": np.random.randn(n_u * n_t),
                # Time-varying weights (different per period within unit)
                "w": np.random.uniform(0.5, 2.0, n_u * n_t),
            }
        )
        return df, SurveyDesign(weights="w")

    def test_default_bacon_decomposition_class_rejects_time_varying_weights(
        self,
    ) -> None:
        """The new ``weights="exact"`` default routes ``BaconDecomposition()``
        through the unit-constant-survey validator. Time-varying within-unit
        weights are rejected. Locks the public-default contract surfaced in
        the PR-B Changed CHANGELOG entry.
        """
        df, sd = self._time_varying_survey_panel()
        with pytest.raises(ValueError, match="varies within units"):
            BaconDecomposition().fit(df, "y", "unit", "time", "first_treat", survey_design=sd)

    def test_default_bacon_decompose_function_rejects_time_varying_weights(
        self,
    ) -> None:
        """KEEP (2(d) PR-A, M-076): the ``weights="exact"`` default flows
        through the DEPRECATED ``bacon_decompose(...)`` wrapper (no
        explicit weights= kwarg) - only the wrapper path exercises that
        default routing. Same rejection contract, now behind the
        wrapper-deprecation FutureWarning."""
        df, sd = self._time_varying_survey_panel()
        with pytest.warns(FutureWarning, match="bacon_decompose\\(\\) is deprecated"):
            with pytest.raises(ValueError, match="varies within units"):
                bacon_decompose(
                    df,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    survey_design=sd,
                )

    def test_explicit_approximate_accepts_time_varying_weights(self) -> None:
        """Users can opt back into the obs-level weighted-means path via
        explicit ``weights="approximate"``. This is the documented
        migration path in the PR-B Changed CHANGELOG entry.
        """
        df, sd = self._time_varying_survey_panel()
        # Should not raise; produces a valid decomposition via the legacy
        # approximate path that tolerates obs-level weighted means.
        results = _bacon_fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            weights="approximate",
            survey_design=sd,
        )
        total = sum(c.weight for c in results.comparisons)
        assert abs(total - 1.0) < 0.01
        assert np.isfinite(results.twfe_estimate)

    def test_diagnostic_report_skips_with_structured_reason_on_replicate_weights(
        self,
    ) -> None:
        """PR #454 R4 P3 regression: ``DiagnosticReport._check_bacon``
        emits ``status="skipped"`` (not ``"error"``) when the survey
        design uses replicate weights, which Bacon rejects with
        ``NotImplementedError`` upstream. The skip reason names the
        ``precomputed={'bacon': ...}`` escape hatch and points users at
        a TSL-based survey design as the supported alternative.
        """
        from diff_diff import DiagnosticReport, SurveyDesign

        df, _ = self._time_varying_survey_panel()
        df["rep_w1"] = 1.0
        df["rep_w2"] = 1.0
        sd_rep = SurveyDesign(
            weights="w",
            replicate_weights=["rep_w1", "rep_w2"],
            replicate_method="BRR",
        )

        class _Stub:
            pass

        dr = DiagnosticReport(
            _Stub(),
            data=df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd_rep,
        )
        block = dr._check_bacon()
        assert block["status"] == "skipped"
        reason = block["reason"]
        assert "replicate weights" in reason
        assert "precomputed" in reason

    def test_diagnostic_report_skips_with_structured_reason_on_time_varying_survey(
        self,
    ) -> None:
        """PR #454 R1 P1 regression: ``DiagnosticReport._check_bacon`` now
        emits ``status="skipped"`` (not ``"error"``) when the panel has
        within-unit-varying survey columns. The skip reason names the
        ``precomputed={'bacon': ...}`` + explicit ``weights="approximate"``
        escape hatch so users have a documented migration path.
        """
        from diff_diff import DiagnosticReport

        df, sd = self._time_varying_survey_panel()

        class _Stub:
            """Minimal results stub that does not carry survey_metadata,
            so the survey-metadata-without-survey_design early-skip at
            ``diagnostic_report.py:1723`` does not pre-empt this path."""

        dr = DiagnosticReport(
            _Stub(),
            data=df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        block = dr._check_bacon()
        assert block["status"] == "skipped"
        reason = block["reason"]
        assert "varies within units" not in reason or "approximate" in reason
        assert "precomputed" in reason
        assert 'weights="approximate"' in reason or "approximate" in reason
