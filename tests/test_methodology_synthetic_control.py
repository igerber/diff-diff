"""Methodology + R-parity tests for the classic Synthetic Control estimator.

Covers Abadie-Diamond-Hainmueller (2010) ``SyntheticControl``:

* **Validation gates** (10 baked-in checks): predictor-period leakage, absorbing
  post-period suffix + no-anticipation cross-check, post canonicalization, donor
  filtering, empty windows, poor-fit warning, duplicate predictor labels,
  inner-solve non-convergence warning, order-independent gap path, and the
  ``standardize="none"`` deviation.
* **custom_v cross-field** + degenerate ``J==1`` / ``T0`` paths + ``get_params``
  round-trip + the NaN-inference contract.
* **Two-tier R `Synth` parity** on the Basque dataset (Abadie-Gardeazabal 2003):
  Tier-1 feeds R's ``solution.v`` through ``custom_v`` and asserts the donor
  weights match deterministically (optimizer-independent); Tier-2 checks the
  data-driven nested fit lands in a tolerance band (the nested V legitimately
  differs because our outer objective uses all pre periods, not R's
  ``time.optimize.ssr`` window).

The Basque fixtures live in ``tests/data/`` (not ``benchmarks/data/``) so the
deterministic Tier-1 test runs in isolated-install CI without R; regenerate via
``Rscript benchmarks/R/generate_synth_basque_golden.R``.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DiagnosticReport,
    SyntheticControl,
    SyntheticControlResults,
    synthetic_control,
)
from diff_diff.conformal import (
    _block_collapse,
    _cwz_proxy_fit,
    _cwz_pvalue,
    _cwz_statistic,
    _iid_perms,
    _moving_block_perms,
)
from diff_diff.synthetic_control import (
    _constant_f_post,
    _floored_pre_mspe,
    _invert_sharp_null,
    _linear_f_post,
    _rmspe_f_ratio,
    _rmspe_ratio,
    _sharp_null_pvalue,
)
from tests.conftest import assert_nan_inference

DATA_DIR = Path(__file__).parent / "data"
GOLDEN_PATH = DATA_DIR / "synth_basque_golden.json"
PANEL_PATH = DATA_DIR / "synth_basque_panel.csv"

PREDICTORS = [
    "school.illit",
    "school.prim",
    "school.med",
    "school.high",
    "school.post.high",
    "invest",
]


# ---------------------------------------------------------------------------
# Cheap optimizer settings for behavior tests (pure-Python CI speed)
# ---------------------------------------------------------------------------
# Behavior tests only need a VALID, cleanly-converged fit, not data-driven V quality.
# The production nested defaults (n_starts=4, inner_max_iter=10000, inner_min_decrease=1e-5)
# cost 30-150s per *pure-Python* fit because the inner Frank-Wolfe solve grinds its slow
# sublinear tail to hit the tight tolerance on every objective evaluation. Loosening the
# inner tolerance + a single start + a small outer cap gives a clean ~0.1s fit without
# changing what these tests assert. Pure-Python coverage of the production-default nested
# path (n_starts=4 with the _v_starts heuristic candidates + the tight inner_min_decrease=1e-5)
# is kept by the dedicated non-slow ``test_nested_production_defaults_smoke`` (a 2-donor panel
# whose inner FW simplex is ~1-D, so defaults stay <0.1s). The @slow Tier-2 Basque test
# additionally covers the defaults in the Rust matrix, and the Rust<->numpy Frank-Wolfe kernel
# equivalence is locked by tests/test_rust_backend.py::test_sc_weight_fw_matches_numpy.
#
# NB: inner_max_iter is deliberately LEFT AT DEFAULT here — the speedup comes from the
# looser tolerance letting FW terminate on *convergence* (not on an iteration cap), so the
# solve stays clean (no non-convergence warning). Do NOT fold inner_max_iter into _FAST or
# the inner-non-convergence warning starts firing spuriously.
_FAST = dict(n_starts=1, optimizer_options={"maxiter": 50}, inner_min_decrease=1e-3)
# Churn tests deliberately force inner non-convergence (inner_max_iter=1); KEEP that and only
# cap the outer optimizer so it does not iterate to maxiter on the flat penalty landscape.
_FAST_CHURN = dict(n_starts=1, optimizer_options={"maxiter": 5})


# ---------------------------------------------------------------------------
# Synthetic panel builders (fast; no R needed)
# ---------------------------------------------------------------------------


def _make_panel(n_donors=4, T=8, T0=6, effect=3.0, seed=0):
    """Balanced panel where the treated unit is a convex mix of two donors."""
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + T))
    donors = {}
    for j in range(n_donors):
        base = rng.normal(10, 2)
        trend = rng.normal(0, 0.3)
        donors[j] = base + trend * np.arange(T) + rng.normal(0, 0.15, T)
    if n_donors >= 2:
        treated = 0.6 * donors[0] + 0.4 * donors[1] + rng.normal(0, 0.08, T)
    else:
        treated = donors[0] + rng.normal(0, 0.08, T)
    treated = treated.copy()
    treated[T0:] += effect
    rows = []
    for j in range(n_donors):
        for t in range(T):
            rows.append(
                {"unit": f"d{j}", "year": years[t], "y": donors[j][t], "treated": 0, "x": float(j)}
            )
    for t in range(T):
        rows.append(
            {
                "unit": "treated",
                "year": years[t],
                "y": treated[t],
                "treated": int(t >= T0),
                "x": 0.5,
            }
        )
    return pd.DataFrame(rows), years, T0


# ---------------------------------------------------------------------------
# Validation 1: predictor periods must be within the pre window (no leakage)
# ---------------------------------------------------------------------------


def test_predictor_window_outside_pre_rejected():
    df, years, T0 = _make_panel()
    post_year = years[T0]
    with pytest.raises(ValueError, match="outside the pre-treatment window"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            predictors=["y"],
            predictor_window=[years[0], post_year],
        )


def test_special_predictor_period_outside_pre_rejected():
    df, years, T0 = _make_panel()
    with pytest.raises(ValueError, match="outside the pre-treatment window"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", [years[T0]], "mean")],
        )


def test_pre_period_outcomes_outside_pre_rejected():
    df, years, T0 = _make_panel()
    with pytest.raises(ValueError, match="outside the pre-treatment window"):
        SyntheticControl(seed=0).fit(
            df, "y", "treated", "unit", "year", pre_period_outcomes=[years[T0]]
        )


# ---------------------------------------------------------------------------
# Validation 2: post must be a contiguous suffix + no-anticipation
# ---------------------------------------------------------------------------


def test_non_contiguous_post_rejected():
    df, years, T0 = _make_panel()
    # Drop a MIDDLE post period -> the remaining set is not a suffix of the axis.
    bad_post = [years[T0]] + years[T0 + 2 :]
    with pytest.raises(ValueError, match="contiguous suffix"):
        SyntheticControl(seed=0).fit(df, "y", "treated", "unit", "year", post_periods=bad_post)


def test_anticipation_in_pre_rejected():
    df, years, T0 = _make_panel()
    # Mark a pre period as treated for the treated unit, but declare the standard
    # post window -> D==1 appears inside the pre window (anticipation).
    df = df.copy()
    mask = (df["unit"] == "treated") & (df["year"] == years[T0 - 1])
    df.loc[mask, "treated"] = 1
    with pytest.raises(ValueError, match="no-anticipation"):
        SyntheticControl(seed=0).fit(df, "y", "treated", "unit", "year", post_periods=years[T0:])


def test_untreated_period_in_post_rejected():
    # Absorbing exposure: a D==0 period inside the (contiguous) post suffix must be
    # rejected, not averaged into the ATT (treated path 0,...,1,0 with post=[T0:]).
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "treated") & (df["year"] == years[-1]), "treated"] = 0
    with pytest.raises(ValueError, match="uninterrupted exposure|D==0 in post"):
        SyntheticControl(seed=0).fit(df, "y", "treated", "unit", "year", post_periods=years[T0:])


def test_non_binary_treatment_rejected():
    # A non-{0,1} treatment code must fail closed (else the unit is silently dropped
    # from both treated and donor sets, changing the donor pool / weights / ATT).
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"] == years[0]), "treated"] = 2
    with pytest.raises(ValueError, match="binary"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


def test_missing_treatment_value_rejected():
    # A donor with a missing treatment cell would be silently classified by
    # groupby(...).max() (NaN dropped) — must fail closed before classification.
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"] == years[0]), "treated"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


def test_estimators_module_reexport():
    # Backward-compat import surface (mirrors SyntheticDiD / TwoWayFixedEffects).
    from diff_diff.estimators import SyntheticControl as SC

    assert SC is SyntheticControl


# ---------------------------------------------------------------------------
# Validation 3 + 9: explicit post canonicalized; gap path order-independent
# ---------------------------------------------------------------------------


def test_post_periods_canonicalized_and_gap_order_independent():
    df, years, T0 = _make_panel()
    ordered = years[T0:]
    scrambled = list(reversed(ordered)) + [ordered[-1]]  # unsorted + duplicate
    r1 = synthetic_control(
        df, "y", "treated", "unit", "year", post_periods=ordered, seed=0, **_FAST
    )
    r2 = synthetic_control(
        df, "y", "treated", "unit", "year", post_periods=scrambled, seed=0, **_FAST
    )
    assert r1.post_periods == r2.post_periods == ordered
    assert abs(r1.att - r2.att) < 1e-12
    gdf = r2.get_gap_df()
    # Calendar-sorted regardless of input order.
    assert list(gdf["period"]) == sorted(gdf["period"])
    assert (gdf[gdf["phase"] == "post"]["period"].tolist()) == ordered


# ---------------------------------------------------------------------------
# Validation 4: donor pool filtering
# ---------------------------------------------------------------------------


def test_donor_pool_restricts_donors():
    df, years, T0 = _make_panel(n_donors=4)
    res = synthetic_control(
        df, "y", "treated", "unit", "year", donor_pool=["d0", "d1"], seed=0, **_FAST
    )
    assert res.n_donors == 2
    assert set(res.get_weights_df()["unit"]) <= {"d0", "d1"}


def test_contaminated_donor_pool_rejected():
    df, years, T0 = _make_panel()
    # The treated unit itself must never appear in the donor pool.
    with pytest.raises(ValueError, match="treated unit|ever-treated|never-treated"):
        synthetic_control(df, "y", "treated", "unit", "year", donor_pool=["d0", "treated"], seed=0)


def test_ever_treated_donor_rejected():
    # A second ever-treated unit (not the designated treated) cannot be a donor.
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"] >= years[T0]), "treated"] = 1
    with pytest.raises(ValueError, match="ever-treated|never-treated"):
        synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            treated_unit="treated",
            donor_pool=["d0", "d1"],
            seed=0,
        )


# ---------------------------------------------------------------------------
# Validation 5: empty windows rejected
# ---------------------------------------------------------------------------


def test_empty_predictor_window_rejected():
    df, _, _ = _make_panel()
    with pytest.raises(ValueError, match="must not be empty"):
        SyntheticControl(seed=0).fit(
            df, "y", "treated", "unit", "year", predictors=["y"], predictor_window=[]
        )


def test_empty_special_period_list_rejected():
    df, _, _ = _make_panel()
    with pytest.raises(ValueError, match="must not be empty"):
        SyntheticControl(seed=0).fit(
            df, "y", "treated", "unit", "year", special_predictors=[("y", [], "mean")]
        )


# ---------------------------------------------------------------------------
# Fail-closed on non-finite data entering the matching problem
# ---------------------------------------------------------------------------


def test_non_finite_predictor_rejected():
    # PARTIAL missingness in a predictor window: fail closed (deliberate deviation
    # from R Synth's na.rm=TRUE — see REGISTRY). All-NA windows behave identically.
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"] == years[0]), "x"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            predictors=["x"],
            predictor_window=[years[0], years[1]],
        )


def test_all_na_predictor_window_rejected():
    # FULLY-missing predictor window: same fail-closed contract as partial (no na.rm).
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"].isin([years[0], years[1]])), "x"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            predictors=["x"],
            predictor_window=[years[0], years[1]],
        )


def test_outer_v_nonconvergence_warning():
    # Outer V-search non-convergence must not be silent (optimizer capped at 1 iter).
    df, _, _ = _make_panel()
    with pytest.warns(UserWarning, match="Outer V-search"):
        # maxiter=1 forces the OUTER non-convergence; n_starts=1 + a loose inner tolerance
        # keep the (still-real) inner solves cheap. Loosening inner_min_decrease does not
        # affect whether the outer optimizer hits its 1-iteration cap.
        synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            seed=0,
            n_starts=1,
            optimizer_options={"maxiter": 1},
            inner_min_decrease=1e-3,
        )


def test_inner_v_search_nonconvergence_warning():
    # Intermediate inner solves during the nested V search must not be silent: forcing
    # inner_max_iter=1 makes them truncate, and the estimator emits an aggregated warning.
    df, _, _ = _make_panel()
    with pytest.warns(UserWarning, match="during nested V selection"):
        synthetic_control(
            df, "y", "treated", "unit", "year", seed=0, inner_max_iter=1, **_FAST_CHURN
        )


def test_single_inner_nonconvergence_excluded_from_v_ranking(monkeypatch):
    # A single LOW-RATE non-converged objective evaluation must be EXCLUDED from V
    # ranking (penalized out of the argmin), not merely warned about: force exactly one
    # objective eval (the uniform-start eval, max(v) < 0.9) to report conv=False and
    # assert (a) the any-occurrence warning fires, and (b) the selected V is a genuine
    # small-MSPE fit (mspe_v << penalty), i.e. the truncated candidate did not win.
    import importlib

    # NB: ``diff_diff.synthetic_control`` the attribute is the convenience *function*
    # (it shadows the submodule, same as ``diff_diff.trop``), so reach the module via
    # importlib to monkeypatch its module-global _inner_solve_W.
    sc = importlib.import_module("diff_diff.synthetic_control")

    df, _, _ = _make_panel()
    real_solve = sc._inner_solve_W
    state = {"failed": False}

    def patched(X1s, X0s, v, max_iter, min_decrease):
        w, conv = real_solve(X1s, X0s, v, max_iter, min_decrease)
        if not state["failed"] and float(np.max(v)) < 0.9:  # a spread V => an objective eval
            state["failed"] = True
            return w, False
        return w, conv

    monkeypatch.setattr(sc, "_inner_solve_W", patched)
    with pytest.warns(UserWarning, match="during nested V selection"):
        res = synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)

    assert state["failed"]  # the patch actually fired on an objective evaluation
    assert np.isfinite(res.att)
    # Exclusion proof: the chosen V's outer-objective MSPE is a real (small) value, not
    # the large penalty a truncated candidate would have carried.
    assert res.mspe_v is not None and res.mspe_v < 1.0


def test_n_starts_one_runs():
    # n_starts=1 uses only the uniform start (short-circuits the heuristic candidates)
    # and still produces a valid nested fit.
    df, _, _ = _make_panel()
    res = synthetic_control(
        df,
        "y",
        "treated",
        "unit",
        "year",
        seed=0,
        n_starts=1,
        optimizer_options={"maxiter": 50},
        inner_min_decrease=1e-3,
    )
    assert np.isfinite(res.att)
    assert abs(sum(res.donor_weights.values()) - 1.0) < 1e-6


def test_nested_production_defaults_smoke():
    # Coverage anchor: exercise the FULL production-default nested path end-to-end in
    # pure-Python — n_starts=4 (so the _v_starts heuristic candidates: inverse-variance,
    # univariate-fit and Dirichlet starts are generated, which the n_starts=1 _FAST tests
    # skip) and the tight inner_min_decrease=1e-5. A 2-donor panel keeps the inner
    # Frank-Wolfe simplex effectively 1-D, so the default settings still run in <0.1s and
    # this stays non-slow. The @slow Tier-2 Basque test covers the defaults only in the Rust
    # matrix; this is the pure-Python complement.
    df, _, _ = _make_panel(n_donors=2)
    res = synthetic_control(df, "y", "treated", "unit", "year", seed=0)  # production defaults
    assert np.isfinite(res.att)
    assert abs(sum(res.donor_weights.values()) - 1.0) < 1e-6
    assert res.n_donors == 2
    assert res.mspe_v is not None  # nested V was selected by minimizing pre-period MSPE


def test_non_finite_outcome_rejected():
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d1") & (df["year"] == years[2]), "y"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


def test_distinct_special_period_sets_not_duplicate():
    # Same var/op, same endpoints + length, different intermediate period -> distinct
    # predictors, must NOT be rejected as duplicates.
    df, years, T0 = _make_panel(T=8, T0=6)
    res = SyntheticControl(seed=0, **_FAST).fit(
        df,
        "y",
        "treated",
        "unit",
        "year",
        special_predictors=[
            ("y", [years[0], years[2], years[4]], "mean"),
            ("y", [years[0], years[3], years[4]], "mean"),
        ],
    )
    assert len(res.v_weights) == 2
    assert len(set(res.v_weights)) == 2  # two distinct labels


def test_reordered_special_periods_are_duplicates():
    # Same var/op with reordered periods canonicalize to the same spec -> duplicate.
    df, years, T0 = _make_panel(T=8, T0=6)
    with pytest.raises(ValueError, match="Duplicate predictor label"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[
                ("y", [years[0], years[1], years[2]], "mean"),
                ("y", [years[2], years[1], years[0]], "mean"),
            ],
        )


def test_duplicate_predictor_window_periods_deduped():
    # A repeated period in predictor_window must not re-weight the mean: the
    # deduped window [y0,y0,y1] matches the explicit [y0,y1].
    df, years, T0 = _make_panel()
    r_dup = synthetic_control(
        df,
        "y",
        "treated",
        "unit",
        "year",
        predictors=["y"],
        predictor_window=[years[0], years[0], years[1]],
        seed=0,
        **_FAST,
    )
    r_uniq = synthetic_control(
        df,
        "y",
        "treated",
        "unit",
        "year",
        predictors=["y"],
        predictor_window=[years[0], years[1]],
        seed=0,
        **_FAST,
    )
    assert abs(r_dup.att - r_uniq.att) < 1e-9


def test_median_op_rejected():
    # median is a non-linear aggregation, not an ADH linear combination.
    df, _, _ = _make_panel()
    with pytest.raises(ValueError, match="must be one of"):
        SyntheticControl(seed=0).fit(
            df, "y", "treated", "unit", "year", predictors=["x"], predictors_op="median"
        )


# ---------------------------------------------------------------------------
# Validation 6: poor pre-fit warning
# ---------------------------------------------------------------------------


def test_poor_fit_warning():
    # Donors are all ~constant near 10; treated is centred near 50 with a trend,
    # so no convex combination can reproduce it -> RMSPE >> SD(treated pre).
    rng = np.random.default_rng(1)
    years = list(range(2000, 2010))
    T0 = 7
    rows = []
    for j in range(4):
        for t, yr in enumerate(years):
            rows.append({"unit": f"d{j}", "year": yr, "y": 10 + rng.normal(0, 0.1), "treated": 0})
    for t, yr in enumerate(years):
        rows.append({"unit": "treated", "year": yr, "y": 50 + 2.0 * t, "treated": int(t >= T0)})
    df = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match="Pre-treatment fit is poor"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)


def test_poor_fit_warning_flat_treated_pre_path():
    # Flat treated pre-path (SD == 0) that donors near 10 cannot reproduce: RMSPE > 0
    # must still warn (the former `pre_sd > 0` gate suppressed this case).
    rng = np.random.default_rng(2)
    years = list(range(2000, 2010))
    T0 = 7
    rows = []
    for j in range(4):
        for yr in years:
            rows.append({"unit": f"d{j}", "year": yr, "y": 10 + rng.normal(0, 0.1), "treated": 0})
    for i, yr in enumerate(years):
        rows.append(
            {"unit": "treated", "year": yr, "y": (5.0 if i < T0 else 8.0), "treated": int(i >= T0)}
        )
    df = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match="Pre-treatment fit is poor"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)


# ---------------------------------------------------------------------------
# Validation 7: duplicate predictor labels rejected
# ---------------------------------------------------------------------------


def test_duplicate_predictor_label_rejected():
    df, years, T0 = _make_panel()
    pre = years[:T0]
    with pytest.raises(ValueError, match="Duplicate predictor label"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", pre, "mean"), ("y", pre, "mean")],
        )


def test_duplicate_regular_predictor_rejected():
    df, _, _ = _make_panel()
    with pytest.raises(ValueError, match="Duplicate predictor label"):
        SyntheticControl(seed=0).fit(df, "y", "treated", "unit", "year", predictors=["x", "x"])


# ---------------------------------------------------------------------------
# Validation 8: inner-solve non-convergence warning
# ---------------------------------------------------------------------------


def test_inner_nonconvergence_warning():
    df, _, _ = _make_panel(n_donors=4)
    with pytest.warns(UserWarning, match="did not converge"):
        SyntheticControl(seed=0, v_method="nested", inner_max_iter=1, **_FAST_CHURN).fit(
            df, "y", "treated", "unit", "year"
        )


# ---------------------------------------------------------------------------
# Validation 10: standardize="none" deviation runs
# ---------------------------------------------------------------------------


def test_standardize_none_runs():
    df, _, _ = _make_panel()
    res = synthetic_control(df, "y", "treated", "unit", "year", standardize="none", seed=0, **_FAST)
    assert res.standardize == "none"
    assert np.isfinite(res.att)


# ---------------------------------------------------------------------------
# custom_v cross-field rules (fail-closed)
# ---------------------------------------------------------------------------


def test_custom_v_required_when_method_custom():
    with pytest.raises(ValueError, match="custom_v is required"):
        SyntheticControl(v_method="custom")


def test_custom_v_rejected_when_method_nested():
    with pytest.raises(ValueError, match="must be None when v_method='nested'"):
        SyntheticControl(v_method="nested", custom_v=[1.0, 1.0])


def test_custom_v_negative_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        SyntheticControl(v_method="custom", custom_v=[1.0, -1.0])


def test_custom_v_wrong_length_rejected():
    df, _, _ = _make_panel()
    # 3 entries but the default (all-pre-outcomes) predictor count differs.
    with pytest.raises(ValueError, match="custom_v has length"):
        SyntheticControl(v_method="custom", custom_v=[1.0, 1.0, 1.0]).fit(
            df, "y", "treated", "unit", "year"
        )


# ---------------------------------------------------------------------------
# Degenerate paths: J==1, T0==0, T0==1
# ---------------------------------------------------------------------------


def test_single_donor_degenerate_warns():
    df, _, _ = _make_panel(n_donors=1)
    with pytest.warns(UserWarning, match="single donor"):
        res = synthetic_control(df, "y", "treated", "unit", "year", seed=0)
    assert res.n_donors == 1
    assert abs(sum(res.donor_weights.values()) - 1.0) < 1e-9


def test_no_pre_period_rejected():
    # All periods treated for the treated unit -> no pre period.
    rows = []
    years = [2000, 2001]
    for j in range(3):
        for yr in years:
            rows.append({"unit": f"d{j}", "year": yr, "y": 10.0 + yr, "treated": 0})
    for yr in years:
        rows.append({"unit": "treated", "year": yr, "y": 12.0 + yr, "treated": 1})
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="No pre-treatment periods|Cannot infer"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


def test_single_pre_period_nested_warns():
    rows = []
    years = [2000, 2001, 2002]
    rng = np.random.default_rng(0)
    for j in range(3):
        for yr in years:
            rows.append({"unit": f"d{j}", "year": yr, "y": 10.0 + rng.normal(), "treated": 0})
    for i, yr in enumerate(years):
        rows.append({"unit": "treated", "year": yr, "y": 11.0 + i, "treated": int(i >= 1)})
    df = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match="single pre period"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


def test_multiple_treated_units_rejected():
    df, _, _ = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"] >= 2006), "treated"] = 1
    with pytest.raises(ValueError, match="exactly one"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


# ---------------------------------------------------------------------------
# sklearn-like API: get_params round-trip + transactional set_params
# ---------------------------------------------------------------------------


def test_get_set_params_roundtrip():
    est = SyntheticControl(n_starts=3, standardize="none", alpha=0.1, seed=7)
    params = est.get_params()
    assert set(params) == {
        "v_method",
        "custom_v",
        "optimizer_options",
        "n_starts",
        "inner_max_iter",
        "inner_min_decrease",
        "standardize",
        "alpha",
        "seed",
        "v_cv_t0",
    }
    est2 = SyntheticControl().set_params(**params)
    assert est2.get_params() == params


def test_set_params_rolls_back_on_invalid():
    est = SyntheticControl(alpha=0.05)
    with pytest.raises(ValueError):
        est.set_params(alpha=1.5)
    assert est.alpha == 0.05  # unchanged after failed update


# ---------------------------------------------------------------------------
# NaN-inference contract + result accessors
# ---------------------------------------------------------------------------


def test_nan_inference_contract():
    df, _, _ = _make_panel()
    res = synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)
    assert_nan_inference(
        {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
    )
    assert np.isfinite(res.att)


def test_result_accessors_render():
    df, _, _ = _make_panel()
    res = synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)
    assert isinstance(res, SyntheticControlResults)
    assert isinstance(res.summary(), str) and "Synthetic Control" in res.summary()
    assert "att" in res.to_dict()
    assert res.to_dataframe().shape[0] == 1
    gdf = res.get_gap_df()
    assert set(gdf.columns) == {"period", "gap", "phase"}
    wdf = res.get_weights_df()
    assert list(wdf.columns) == ["unit", "weight"]
    # PR-2: fit() populates the placebo refit snapshot and the treated unit's
    # RMSPE ratio; the placebo reference distribution is not computed until
    # in_space_placebo() runs (placebo_p_value stays NaN, gaps/df unset).
    assert res._fit_snapshot is not None
    assert res._placebo_gaps is None and res._placebo_df is None
    assert np.isfinite(res.rmspe_ratio)
    assert np.isnan(res.placebo_p_value) and res.n_placebos == 0


def test_inferred_post_matches_explicit():
    df, years, T0 = _make_panel()
    r_inf = synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)
    r_exp = synthetic_control(
        df, "y", "treated", "unit", "year", post_periods=years[T0:], seed=0, **_FAST
    )
    assert r_inf.post_periods == r_exp.post_periods == years[T0:]
    assert abs(r_inf.att - r_exp.att) < 1e-12


# ---------------------------------------------------------------------------
# R-parity (Basque / Abadie-Gardeazabal 2003 via R `Synth`)
# ---------------------------------------------------------------------------


def _load_golden():
    if not GOLDEN_PATH.exists() or not PANEL_PATH.exists():
        pytest.skip(
            "Basque golden fixtures missing — regenerate via "
            "`Rscript benchmarks/R/generate_synth_basque_golden.R`."
        )
    return json.load(open(GOLDEN_PATH)), pd.read_csv(PANEL_PATH)


def _basque_kwargs(golden):
    special = [
        (
            s["var"],
            list(s["periods"]) if isinstance(s["periods"], list) else [s["periods"]],
            s["op"],
        )
        for s in golden["config"]["special"]
    ]
    return dict(
        treated_unit=golden["config"]["treated_regionno"],
        donor_pool=list(golden["config"]["controls"]),
        predictors=PREDICTORS,
        predictors_op="mean",
        predictor_window=list(range(1964, 1970)),
        special_predictors=special,
    )


def test_basque_tier1_custom_v_parity():
    """Tier-1 (hard gate): given R's solution.v, donor weights match R deterministically."""
    golden, df = _load_golden()
    custom_v = np.asarray(golden["solution_v"], dtype=float)
    res = SyntheticControl(v_method="custom", custom_v=custom_v).fit(
        df, "gdpcap", "treated", "regionno", "year", **_basque_kwargs(golden)
    )
    # Predictor matrix + ordering: X1 matches R's dataprep exactly.
    X1_py = res.predictor_balance["treated"].to_numpy(dtype=float)
    X1_r = np.asarray(golden["X1"], dtype=float)
    np.testing.assert_allclose(X1_py, X1_r, atol=1e-6)

    # Donor weights match R's solution.w (the published Cataluna/Madrid mix).
    controls = sorted(int(c) for c in golden["config"]["controls"])
    w_r = {int(k): v for k, v in golden["solution_w"].items()}
    w_py = {int(k): v for k, v in res.donor_weights.items()}
    wr = np.array([w_r.get(c, 0.0) for c in controls])
    wp = np.array([w_py.get(c, 0.0) for c in controls])
    np.testing.assert_allclose(wp, wr, atol=1e-3)
    # Published anchor: region 10 ~ 0.85, region 14 ~ 0.15.
    assert w_py.get(10, 0) > 0.80 and w_py.get(14, 0) > 0.10


@pytest.mark.slow
def test_basque_tier2_nested_band():
    """Tier-2 (band): the data-driven nested fit lands near R's solution.

    Loose by design — our outer objective minimizes MSPE over all pre periods,
    while R uses the ``time.optimize.ssr`` (1960-1969) window, so the nested V
    legitimately differs; multistart Nelder-Mead/Powell is also BLAS/platform
    sensitive. We check fit quality and that the dominant donors agree.
    """
    golden, df = _load_golden()
    res = SyntheticControl(v_method="nested", seed=0).fit(
        df, "gdpcap", "treated", "regionno", "year", **_basque_kwargs(golden)
    )
    r_sqrt_loss = golden["loss_v"] ** 0.5
    assert res.pre_rmspe <= r_sqrt_loss * 1.5  # comparable pre-fit quality

    years = np.asarray(golden["years"])
    r_att = float(np.asarray(golden["gap"])[years >= 1970].mean())
    assert abs(res.att - r_att) < 0.2  # avg post-gap within band

    # Dominant donors agree with R (Cataluna region 10, Madrid region 14).
    top2 = [u for u, _ in sorted(res.donor_weights.items(), key=lambda kv: -kv[1])[:2]]
    assert set(top2) == {10, 14}
    assert res.donor_weights.get(10, 0) + res.donor_weights.get(14, 0) > 0.7


def test_basque_tier1_leave_one_out_parity():
    """Tier-1 LOO (deterministic): dropping the dominant donor (region 10) with R's
    ``solution.v`` held fixed, the reduced-pool refit's ATT and gap path match R's
    drop-donor ``synth`` exactly (a direct R anchor on the reduced-pool W-solve;
    ``leave_one_out()`` on a custom-V fit reuses that fixed V on the donor pool minus
    the dropped unit). Region 10 carries ~85% of the full-pool weight, so dropping it
    swings the synthetic onto regions 7+14 — the single-donor-dependence signal LOO
    exists to surface."""
    golden, df = _load_golden()
    if "leave_one_out" not in golden:
        pytest.skip("LOO golden missing — regenerate via the R script.")
    loo_g = golden["leave_one_out"]
    dropped = int(loo_g["dropped_regionno"])
    custom_v = np.asarray(golden["solution_v"], dtype=float)
    res = SyntheticControl(v_method="custom", custom_v=custom_v).fit(
        df, "gdpcap", "treated", "regionno", "year", **_basque_kwargs(golden)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = res.leave_one_out()
    row = loo[(loo["status"] == "loo") & (loo["dropped_unit"] == dropped)]
    assert len(row) == 1
    assert float(row["att"].iloc[0]) == pytest.approx(float(loo_g["att"]), abs=1e-2)
    # Full reduced-pool gap trajectory (1955-1997) matches R's drop-donor synth.
    gaps = res.get_leave_one_out_gaps()
    gap_py = gaps[gaps["dropped_unit"] == dropped].sort_values("period")["gap"].to_numpy()
    np.testing.assert_allclose(gap_py, np.asarray(loo_g["gap"], dtype=float), atol=2e-2)


# ---------------------------------------------------------------------------
# In-space placebo permutation inference (Abadie-Diamond-Hainmueller 2010 §2.4)
# ---------------------------------------------------------------------------


def _fit_for_placebo(n_donors=4, effect=3.0, **kw):
    """Fit with cheap settings on a panel carrying a strong post-treatment effect."""
    df, _, _ = _make_panel(n_donors=n_donors, effect=effect)
    opts = dict(_FAST)
    opts.update(kw)
    with warnings.catch_warnings():  # single-donor / poor-fit fit warnings are not under test
        warnings.simplefilter("ignore")
        return synthetic_control(df, "y", "treated", "unit", "year", seed=0, **opts)


def test_in_space_placebo_strong_effect_ranks_treated_first():
    # A 3.0-unit post effect on a treated unit that is a clean convex mix of two
    # donors -> treated RMSPE ratio is the most extreme -> rank 1 -> p = 1/(J+1).
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdf = res.in_space_placebo()
    assert res.n_placebos == 4 and res.n_failed == 0
    treated_ratio = pdf.loc[pdf["is_treated"], "rmspe_ratio"].iloc[0]
    placebo_ratios = pdf.loc[~pdf["is_treated"], "rmspe_ratio"]
    assert (treated_ratio > placebo_ratios).all()  # treated is the most extreme unit
    assert res.placebo_p_value == pytest.approx(1 / (res.n_placebos + 1))
    # Exactly one treated row; the placebo rows are exactly the donor units.
    assert int(pdf["is_treated"].sum()) == 1
    assert pdf.loc[pdf["is_treated"], "unit"].iloc[0] == "treated"
    assert set(pdf.loc[~pdf["is_treated"], "unit"]) == {"d0", "d1", "d2", "d3"}


def test_in_space_placebo_excludes_real_treated_from_donor_pools():
    # The real treated unit is never in the donor universe, so it cannot serve as
    # a donor for any placebo (ADH 2010 contamination guard; SCtools convention).
    res = _fit_for_placebo(n_donors=4)
    snap = res._fit_snapshot
    assert snap.treated_id == "treated" and "treated" not in snap.donor_ids
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    # Each donor became a placebo exactly once; the treated unit is not a placebo.
    assert "treated" not in res._placebo_gaps
    assert set(res._placebo_gaps) == set(snap.donor_ids)


def test_in_space_placebo_p_in_valid_discrete_set():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    valid = {(k + 1) / (res.n_placebos + 1) for k in range(res.n_placebos + 1)}
    assert any(res.placebo_p_value == pytest.approx(v) for v in valid)


def test_in_space_placebo_does_not_touch_analytical_inference():
    # The permutation p-value is SEPARATE from the analytical fields, which stay
    # NaN; is_significant stays bound to the (NaN) p_value, not placebo_p_value.
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    assert np.isfinite(res.placebo_p_value)
    assert_nan_inference(
        {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
    )
    assert res.is_significant is False


def test_in_space_placebo_deterministic():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1 = res.in_space_placebo()
        first_p = res.placebo_p_value
        p2 = res.in_space_placebo()
    assert res.placebo_p_value == first_p  # bit-equal p-value across runs
    pd.testing.assert_frame_equal(p1, p2)  # identical rows AND row order


def test_in_space_placebo_requires_two_donors():
    res = _fit_for_placebo(n_donors=1)
    with pytest.warns(UserWarning, match="at least 2 donors"):
        pdf = res.in_space_placebo()
    assert np.isnan(res.placebo_p_value) and res.n_placebos == 0
    assert len(pdf) == 1 and bool(pdf["is_treated"].iloc[0])


def test_in_space_placebo_two_donors_warns_coarse():
    res = _fit_for_placebo(n_donors=2)
    with pytest.warns(UserWarning, match="coarse"):
        res.in_space_placebo()
    # 2 placebos -> reference set of 3 -> p in {1/3, 2/3, 1}.
    assert res.n_placebos == 2
    assert any(res.placebo_p_value == pytest.approx(v) for v in (1 / 3, 2 / 3, 1.0))


def test_in_space_placebo_fails_closed_on_nonconverged_treated_fit():
    # inner_max_iter=1 truncates the treated unit's own Frank-Wolfe solve, so its
    # RMSPE ratio is not a valid optimum. in_space_placebo() must fail closed
    # (NaN p-value + warning), NOT rank a truncated treated statistic.
    df, _, _ = _make_panel(n_donors=4, effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            seed=0,
            n_starts=1,
            inner_max_iter=1,
            optimizer_options={"maxiter": 5},
        )
    assert res._fit_converged is False  # treated fit was truncated
    with pytest.warns(UserWarning, match="did not converge at fit time"):
        pdf = res.in_space_placebo()
    assert np.isnan(res.placebo_p_value)
    assert res.n_placebos == 0 and res.n_failed == 0  # the placebo loop never ran
    assert len(pdf) == 1 and bool(pdf["is_treated"].iloc[0])  # treated row only


def test_in_space_placebo_pickle_drops_snapshot_keeps_scalars():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    restored = pickle.loads(pickle.dumps(res))
    # Scalars survive; panel-derived state is dropped.
    assert restored.placebo_p_value == res.placebo_p_value
    assert restored.rmspe_ratio == res.rmspe_ratio
    assert restored.n_placebos == res.n_placebos and restored.n_failed == res.n_failed
    assert restored.n_infeasible == res.n_infeasible
    assert restored._fit_snapshot is None and restored._placebo_gaps is None
    # The small aggregate table survives, so get_placebo_df still works...
    assert len(restored.get_placebo_df()) == len(res.get_placebo_df())
    # ...but a re-run of the refit raises (the snapshot is gone).
    with pytest.raises(ValueError, match="requires the fit snapshot"):
        restored.in_space_placebo()


def test_legacy_pickle_missing_infeasible_fields_backfills_to_zero():
    # A result pickled by a version predating n_infeasible / _loo_n_infeasible unpickles
    # (bypassing __init__ / __post_init__) WITHOUT those attributes. __setstate__ must
    # backfill them to 0 so the reporting paths that dereference them directly
    # (summary() / to_dict()) do not raise AttributeError on a legacy result.
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
        res.leave_one_out()
    legacy_state = res.__getstate__()  # what pickle would persist ...
    legacy_state.pop("n_infeasible", None)  # ... minus the fields an older version lacked
    legacy_state.pop("_loo_n_infeasible", None)
    restored = object.__new__(type(res))  # bypasses __init__/__post_init__, as pickle does
    restored.__setstate__(legacy_state)
    assert restored.n_infeasible == 0 and restored._loo_n_infeasible == 0
    # Reporting paths that dereference the new fields directly must not raise.
    assert isinstance(restored.summary(), str)
    assert restored.to_dict()["n_infeasible"] == 0
    native = DiagnosticReport(restored).to_dict()["estimator_native_diagnostics"]
    assert native["in_space_placebo"]["n_infeasible"] == 0
    assert native["leave_one_out"]["n_infeasible"] == 0


def test_in_space_placebo_custom_v_path():
    df, _, _ = _make_panel(n_donors=4)
    # Default predictors = all pre-period outcomes -> k = number of pre periods (T0).
    k = 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=np.ones(k),
            inner_min_decrease=1e-3,
        )
        pdf = res.in_space_placebo()
    assert res.n_placebos == 4 and np.isfinite(res.placebo_p_value)
    assert len(pdf) == 5


def test_get_placebo_df_before_run_returns_treated_row_only():
    res = _fit_for_placebo(n_donors=4)
    pdf = res.get_placebo_df()
    assert len(pdf) == 1
    assert bool(pdf["is_treated"].iloc[0]) and pdf["status"].iloc[0] == "treated"
    assert set(pdf.columns) == {
        "unit",
        "pre_mspe",
        "post_mspe",
        "rmspe_ratio",
        "is_treated",
        "status",
    }


def test_rmspe_ratio_floors_zero_pre_mspe():
    # Perfect pre-fit (pre-MSPE == 0) must yield a large-but-finite ratio, not
    # inf/nan (which would corrupt the permutation rank).
    from diff_diff.synthetic_control import _rmspe_ratio

    pre = np.zeros(5)
    assert np.isfinite(_rmspe_ratio(pre, np.array([1.0, 2.0, 3.0]), scale=10.0))
    # A zero-effect (post all zero) placebo has ratio 0 — the least extreme.
    assert _rmspe_ratio(pre, np.zeros(3), scale=10.0) == 0.0


def test_in_space_placebo_perfect_treated_fit_finite_ratio():
    # 2-donor panel where the treated unit EQUALS d0 in the pre-period -> the inner
    # FW solve lands on w=[1, 0], so the treated pre-MSPE is (bit-)exactly 0. The
    # RMSPE ratio must stay FINITE (scale-aware floor), never inf/nan.
    rng = np.random.default_rng(2)
    T, T0 = 8, 6
    years = list(range(2000, 2000 + T))
    d0 = rng.normal(10, 2, T)
    d1 = rng.normal(5, 2, T)
    treated = d0.copy()
    treated[T0:] += 5.0  # identical to d0 in the pre-period, clean post effect
    rows = []
    for name, series in (("d0", d0), ("d1", d1)):
        for t in range(T):
            rows.append({"unit": name, "year": years[t], "y": series[t], "treated": 0})
    for t in range(T):
        rows.append({"unit": "treated", "year": years[t], "y": treated[t], "treated": int(t >= T0)})
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=np.ones(T0),
            inner_min_decrease=1e-3,
        )
    assert res.pre_rmspe == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(res.rmspe_ratio) and res.rmspe_ratio > 0


def test_in_space_placebo_immune_to_post_fit_mutation():
    # The fit snapshot must COPY caller-owned mutable inputs (custom_v,
    # optimizer_options), so mutating them after fit() cannot silently change
    # in_space_placebo() output on an already-returned results object.
    df, _, _ = _make_panel(n_donors=4)
    cv = np.ones(6)  # k = 6 default pre-period-outcome predictors
    opts = {"maxiter": 50}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=cv,
            optimizer_options=opts,
            inner_min_decrease=1e-3,
        )
        p1 = res.in_space_placebo().copy()
        pval1 = res.placebo_p_value
    snap = res._fit_snapshot
    assert snap.custom_v is not cv and snap.optimizer_options is not opts
    # Mutate the caller-owned originals AFTER fit -> placebo output must not change.
    cv[:] = [1e6, 1.0, 1.0, 1.0, 1.0, 1.0]
    opts["maxiter"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p2 = res.in_space_placebo().copy()
    assert res.placebo_p_value == pval1
    pd.testing.assert_frame_equal(p1, p2)


def test_get_placebo_df_includes_failed_donors(monkeypatch):
    # When the treated fit IS valid but some per-donor placebo refits fail to
    # converge, get_placebo_df() must still list EVERY unit (treated + each donor)
    # so callers can tell which donors failed -> exactly n_donors + 1 rows.
    # (A truncated treated fit instead fails the whole placebo run closed, tested
    # separately; here we simulate isolated donor failures with a converged treated
    # fit by monkeypatching the per-donor refit to fail for the first two donors.)
    import importlib

    # diff_diff.synthetic_control the SUBMODULE is shadowed by the re-exported
    # synthetic_control FUNCTION on the package, so import the module explicitly.
    sc = importlib.import_module("diff_diff.synthetic_control")

    res = _fit_for_placebo(n_donors=4)  # treated fit converges (normal settings)
    real_fit_unit = sc._placebo_fit_unit
    calls = {"n": 0}

    def flaky_fit_unit(snap, unit, donor_pool, n_starts):
        calls["n"] += 1
        if calls["n"] <= 2:  # first two donor refits "fail" (solver non-convergence)
            return None, "failed"
        return real_fit_unit(snap, unit, donor_pool, n_starts)

    monkeypatch.setattr(sc, "_placebo_fit_unit", flaky_fit_unit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdf = res.in_space_placebo()
    assert len(pdf) == res.n_donors + 1  # treated + every donor, regardless of failures
    assert res.n_failed == 2 and res.n_placebos == res.n_donors - 2
    failed = pdf[pdf["status"] == "failed"]
    assert len(failed) == 2 and failed["rmspe_ratio"].isna().all()  # NaN metrics


def test_in_space_placebo_fails_closed_on_underoptimized_outer_v():
    # An under-optimized OUTER V search (optimizer maxiter=1) leaves the treated
    # fit's V non-optimal even though the inner solve converges. Its RMSPE ratio is
    # therefore not a valid optimum, so in_space_placebo() must FAIL CLOSED rather
    # than silently rank an anti-conservatively under-optimized statistic.
    df, _, _ = _make_panel(n_donors=4, effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            seed=0,
            n_starts=1,
            optimizer_options={"maxiter": 1},  # outer V search cannot converge
            inner_min_decrease=1e-3,  # inner still converges -> isolates the outer path
        )
    assert res._fit_converged is False  # outer V non-convergence -> invalid fit
    with pytest.warns(UserWarning, match="did not converge at fit time"):
        res.in_space_placebo()
    assert np.isnan(res.placebo_p_value)
    assert res.n_placebos == 0 and res.n_failed == 0  # placebo loop never ran


def test_outer_v_convergence_tracks_selected_incumbent(monkeypatch):
    # _outer_solve_V must report convergence of the SELECTED (lowest-objective)
    # incumbent, NOT "any start converged". Here the first multistart succeeds with a
    # HIGH objective while the winning (lowest-objective) start reports success=False;
    # the fit must be flagged non-converged so in_space_placebo() fails closed.
    import importlib

    from scipy.optimize import OptimizeResult

    sc = importlib.import_module("diff_diff.synthetic_control")
    calls = {"n": 0}

    def fake_minimize(fun, x0, **kwargs):
        calls["n"] += 1
        x0 = np.asarray(x0, dtype=float)
        if kwargs.get("method") == "Nelder-Mead":
            # 1st start: high objective but converged; later: low objective (wins) but NOT.
            if calls["n"] == 1:
                return OptimizeResult(x=x0, fun=10.0, success=True)
            return OptimizeResult(x=x0, fun=1.0, success=False)
        # Powell polish: neither improves on nor converges at the incumbent.
        return OptimizeResult(x=x0, fun=5.0, success=False)

    monkeypatch.setattr(sc, "minimize", fake_minimize)
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df, "y", "treated", "unit", "year", seed=0, n_starts=2, inner_min_decrease=1e-3
        )
    # The winning incumbent came from a success=False run -> selected V is not a
    # validated optimum, so the fit must not be marked converged.
    assert res._fit_converged is False
    with pytest.warns(UserWarning, match="did not converge at fit time"):
        res.in_space_placebo()
    assert np.isnan(res.placebo_p_value)


def test_outer_v_powell_success_at_worse_point_does_not_validate(monkeypatch):
    # The Powell polish must validate the SELECTED incumbent only when it converges
    # back AT the incumbent's objective level. Here the winning (lowest-objective)
    # Nelder-Mead start reports success=False, and Powell "succeeds" but at a STRICTLY
    # WORSE objective (it ended elsewhere). Powell's success says nothing about the
    # selected incumbent, so the fit must stay non-converged and fail closed -- a flag
    # of "converged" here would silently admit an under-optimized V into the placebo
    # ranking and produce wrong permutation inference.
    import importlib

    from scipy.optimize import OptimizeResult

    sc = importlib.import_module("diff_diff.synthetic_control")
    calls = {"n": 0}

    def fake_minimize(fun, x0, **kwargs):
        calls["n"] += 1
        x0 = np.asarray(x0, dtype=float)
        if kwargs.get("method") == "Nelder-Mead":
            # Single start: lowest objective wins but reports success=False.
            return OptimizeResult(x=x0, fun=1.0, success=False)
        # Powell polish: SUCCEEDS, but at a strictly worse objective than the incumbent.
        return OptimizeResult(x=x0, fun=5.0, success=True)

    monkeypatch.setattr(sc, "minimize", fake_minimize)
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df, "y", "treated", "unit", "year", seed=0, n_starts=1, inner_min_decrease=1e-3
        )
    # Powell's success at a worse point must NOT flip the selected incumbent to converged.
    assert res._fit_converged is False
    with pytest.warns(UserWarning, match="did not converge at fit time"):
        res.in_space_placebo()
    assert np.isnan(res.placebo_p_value)


def test_to_dict_includes_placebo_scalars():
    res = _fit_for_placebo(n_donors=4)
    d = res.to_dict()
    for key in ("placebo_p_value", "rmspe_ratio", "n_placebos", "n_failed"):
        assert key in d
    # Before the placebo run: rmspe_ratio is finite (fit-time), placebo_p_value NaN.
    assert np.isfinite(d["rmspe_ratio"]) and np.isnan(d["placebo_p_value"])
    assert d["n_placebos"] == 0 and d["n_failed"] == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    d2 = res.to_dict()
    assert np.isfinite(d2["placebo_p_value"]) and d2["n_placebos"] == 4


def test_summary_distinguishes_infeasible_placebo_from_not_run():
    # summary() must tell "placebo never run" apart from "placebo run but produced
    # no valid reference set" (J<2 here -> placebo_p_value NaN but it WAS attempted),
    # and name the SPECIFIC infeasibility reason (too few donors).
    df, _, _ = _make_panel(n_donors=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)
        before = res.summary()
        res.in_space_placebo()  # infeasible: single donor -> no placebo distribution
        after = res.summary()
    assert "Run in_space_placebo()" in before  # never run
    assert np.isnan(res.placebo_p_value) and res._placebo_df is not None  # attempted
    assert res._placebo_status == "too_few_donors"
    assert "requires at least 2 donors" in after  # specific reason, not "not run"
    assert "Run in_space_placebo()" not in after  # not mislabeled as "not run"


def test_summary_treated_fit_failure_names_specific_reason():
    # When the treated unit's OWN fit fails to converge, in_space_placebo() fails
    # closed with n_placebos=0, n_failed=0 -- the SAME counts as the J<2 case. The
    # CI codex P2: summary() must not reconstruct the reason from those counts and
    # narrate "too few donors or all donor refits failed" (false here); it must name
    # the treated-fit non-convergence recorded in _placebo_status.
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            seed=0,
            n_starts=1,
            optimizer_options={"maxiter": 1},  # outer V cannot converge -> fail closed
            inner_min_decrease=1e-3,
        )
        assert res._fit_converged is False
        with pytest.warns(UserWarning, match="did not converge at fit time"):
            res.in_space_placebo()
        after = res.summary()
    assert res._placebo_status == "treated_fit_nonconverged"
    assert res.n_placebos == 0 and res.n_failed == 0  # same counts as J<2
    assert "treated unit's own SCM fit" in after and "did not converge" in after
    # Must NOT misdiagnose as the donor-side reason.
    assert "too few" not in after.lower()
    assert "all donor refits" not in after.lower()


def test_in_space_placebo_rejects_invalid_n_starts():
    # CI codex P2: the n_starts override must fail fast on non-positive / non-integer
    # values (mirroring the estimator constructor) rather than silently coercing via
    # int(...) into a degenerate one-start (or invalid) permutation procedure.
    res = _fit_for_placebo(n_donors=4)
    for bad in (0, -1, -5):
        with pytest.raises(ValueError, match="n_starts override must be a positive integer"):
            res.in_space_placebo(n_starts=bad)
    for bad in (2.5, "3"):
        with pytest.raises(ValueError, match="n_starts override must be a positive integer"):
            res.in_space_placebo(n_starts=bad)  # type: ignore[arg-type]
    # The placebo state must be untouched by a rejected override.
    assert res._placebo_status is None and res._placebo_df is None


def test_rmspe_ratio_is_root_scale():
    # The reported statistic is the ROOT-scale ratio RMSPE_post/RMSPE_pre =
    # sqrt(MSPE_post/MSPE_pre), NOT the MSPE ratio. Hand-worked: pre-MSPE = 4,
    # post-MSPE = 9 -> sqrt(9/4) = 1.5 (the MSPE ratio would be 9/4 = 2.25).
    from diff_diff.synthetic_control import _rmspe_ratio

    pre = np.array([2.0, 2.0])  # MSPE = 4
    post = np.array([3.0, 3.0])  # MSPE = 9
    assert _rmspe_ratio(pre, post, scale=10.0) == pytest.approx(1.5)
    # Zero post-effect -> ratio 0; perfect pre-fit -> finite (floored), not inf.
    assert _rmspe_ratio(pre, np.zeros(2), scale=10.0) == pytest.approx(0.0)
    # Perfect pre-fit (zero pre-gaps) -> floored denominator -> finite, not inf.
    assert np.isfinite(_rmspe_ratio(np.zeros(2), post, scale=10.0))


# ---------------------------------------------------------------------------
# Leave-one-out donor robustness (ADH 2015 §4)
# ---------------------------------------------------------------------------


def _equal_mix_panel(n_donors=5, T=8, T0=6, effect=3.0, seed=1):
    """Near-identical donors -> equal-ish weights -> dropping any one barely moves
    the synthetic (the LOO-stable regime)."""
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + T))
    base = rng.normal(10, 0.4, n_donors)
    common = np.cumsum(rng.normal(0, 0.2, T))  # shared trend
    donors = {j: base[j] + common + rng.normal(0, 0.08, T) for j in range(n_donors)}
    treated = np.mean([donors[j] for j in range(n_donors)], axis=0) + rng.normal(0, 0.04, T)
    treated = treated.copy()
    treated[T0:] += effect
    rows = []
    for j in range(n_donors):
        for t in range(T):
            rows.append({"unit": f"d{j}", "year": years[t], "y": donors[j][t], "treated": 0})
    for t in range(T):
        rows.append({"unit": "treated", "year": years[t], "y": treated[t], "treated": int(t >= T0)})
    return pd.DataFrame(rows)


def _single_donor_panel(n_donors=4, T=8, T0=6, effect=3.0, seed=2):
    """One donor (d0) tracks the treated unit; the rest are far away -> weight
    concentrates on d0 -> dropping d0 swings the result (the LOO-fragile regime)."""
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + T))
    d0_path = 10 + np.cumsum(rng.normal(0, 0.3, T))
    donors = {0: d0_path + rng.normal(0, 0.03, T)}
    for j in range(1, n_donors):
        donors[j] = (25.0 + 6.0 * j) + np.cumsum(rng.normal(0, 0.3, T))  # far from treated
    treated = d0_path + rng.normal(0, 0.03, T)
    treated = treated.copy()
    treated[T0:] += effect
    rows = []
    for j in range(n_donors):
        for t in range(T):
            rows.append({"unit": f"d{j}", "year": years[t], "y": donors[j][t], "treated": 0})
    for t in range(T):
        rows.append({"unit": "treated", "year": years[t], "y": treated[t], "treated": int(t >= T0)})
    return pd.DataFrame(rows)


def _fit_cheap(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)


_LOO_COLS = ["dropped_unit", "att", "pre_rmspe", "post_rmspe", "rmspe_ratio", "delta_att", "status"]


def test_leave_one_out_baseline_row_and_structure():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = res.leave_one_out()
    assert list(loo.columns) == _LOO_COLS
    # Exactly one baseline row, first, reading directly from the full fit.
    base = loo.iloc[0]
    # dropped_unit is "not applicable" for the baseline row (pandas renders the
    # None as NA in the donor-id column).
    assert base["status"] == "baseline" and pd.isna(base["dropped_unit"])
    assert base["att"] == pytest.approx(res.att) and base["delta_att"] == 0.0
    assert base["pre_rmspe"] == pytest.approx(res.pre_rmspe)
    assert base["rmspe_ratio"] == pytest.approx(res.rmspe_ratio)
    # One LOO row per positively-weighted donor (no failures on this clean panel).
    pos = [d for d in res._fit_snapshot.donor_ids if d in res.donor_weights]
    loo_rows = loo[loo["status"] == "loo"]
    assert set(loo_rows["dropped_unit"]) == set(pos)
    assert res._loo_n_failed == 0 and res._loo_status == "ran"
    # delta_att == att - full att, exactly.
    for _, r in loo_rows.iterrows():
        assert r["delta_att"] == pytest.approx(r["att"] - res.att)
    # Sorted by |delta_att| descending.
    deltas = loo_rows["delta_att"].abs().to_numpy()
    assert np.all(np.diff(deltas) <= 1e-12)
    # att_range spans the LOO refits.
    lo, hi = res._loo_att_range
    assert lo <= hi and lo == pytest.approx(loo_rows["att"].min())
    assert hi == pytest.approx(loo_rows["att"].max())


def test_leave_one_out_stable_when_no_donor_dominates():
    res = _fit_cheap(_equal_mix_panel(n_donors=5, effect=3.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = res.leave_one_out()
    loo_rows = loo[loo["status"] == "loo"]
    # Near-identical donors -> dropping any one barely moves the ATT (well under the
    # 3.0 effect). att_range is correspondingly tight.
    assert loo_rows["delta_att"].abs().max() < 1.0
    lo, hi = res._loo_att_range
    assert (hi - lo) < 1.0


def test_leave_one_out_swings_when_one_donor_dominates():
    res = _fit_cheap(_single_donor_panel(n_donors=4, effect=3.0))
    # Weight concentrates on d0.
    assert res.donor_weights.get("d0", 0.0) > 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = res.leave_one_out()
    loo_rows = loo[loo["status"] == "loo"]
    # Dropping the dominant donor is the most influential drop (top finite row) and
    # moves the ATT by a non-trivial amount.
    top = loo_rows.iloc[0]
    assert top["dropped_unit"] == "d0"
    assert abs(top["delta_att"]) > 0.2


def test_leave_one_out_deterministic():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo1 = res.leave_one_out()
        loo2 = res.leave_one_out()
    pd.testing.assert_frame_equal(loo1, loo2)


def test_leave_one_out_requires_two_donors():
    res = _fit_for_placebo(n_donors=1)
    with pytest.warns(UserWarning, match="at least 2 donors"):
        loo = res.leave_one_out()
    assert len(loo) == 1 and loo.iloc[0]["status"] == "baseline"
    assert res._loo_status == "too_few_donors" and res._loo_att_range is None


def test_leave_one_out_fails_closed_on_nonconverged_treated_fit():
    df, _, _ = _make_panel(n_donors=4, effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df, "y", "treated", "unit", "year", seed=0, inner_max_iter=1, **_FAST_CHURN
        )
    assert res._fit_converged is False
    with pytest.warns(UserWarning, match="did not converge at fit time"):
        loo = res.leave_one_out()
    assert len(loo) == 1 and loo.iloc[0]["status"] == "baseline"
    assert res._loo_status == "treated_fit_nonconverged"


def test_leave_one_out_refit_failure_tallied(monkeypatch):
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    real_fit_unit = sc._placebo_fit_unit
    calls = {"n": 0}

    def flaky_fit_unit(snap, unit, donor_pool, n_starts):
        calls["n"] += 1
        if calls["n"] == 1:  # first leave-one-out refit "fails" (solver non-convergence)
            return None, "failed"
        return real_fit_unit(snap, unit, donor_pool, n_starts)

    monkeypatch.setattr(sc, "_placebo_fit_unit", flaky_fit_unit)
    with pytest.warns(UserWarning, match="did not reach a valid optimum"):
        loo = res.leave_one_out()
    assert res._loo_n_failed == 1
    failed = loo[loo["status"] == "failed"]
    assert len(failed) == 1
    assert failed[["att", "pre_rmspe", "rmspe_ratio", "delta_att"]].isna().all().all()
    # Failed rows sort last (after the baseline + the converged LOO rows).
    assert loo.iloc[-1]["status"] == "failed"


def test_leave_one_out_pickle_drops_gaps_keeps_table():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.leave_one_out()
    restored = pickle.loads(pickle.dumps(res))
    # The summary table + scalars survive; panel-derived gap paths do not.
    pd.testing.assert_frame_equal(restored.get_leave_one_out_df(), res.get_leave_one_out_df())
    assert restored._loo_gaps is None
    assert restored._loo_att_range == res._loo_att_range
    with pytest.raises(ValueError, match="not retained after pickling"):
        restored.get_leave_one_out_gaps()


def test_leave_one_out_gaps_long_form():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.leave_one_out()
    gaps = res.get_leave_one_out_gaps()
    assert list(gaps.columns) == ["dropped_unit", "period", "gap", "phase"]
    pos = [d for d in res._fit_snapshot.donor_ids if d in res.donor_weights]
    assert set(gaps["dropped_unit"]) == set(pos)
    # Every dropped donor has a full pre+post trajectory.
    n_periods = len(res.pre_periods) + len(res.post_periods)
    assert (gaps.groupby("dropped_unit").size() == n_periods).all()
    assert set(gaps["phase"]) == {"pre", "post"}


def test_leave_one_out_accessor_before_run_raises():
    res = _fit_for_placebo(n_donors=4)
    with pytest.raises(ValueError, match="call leave_one_out"):
        res.get_leave_one_out_df()
    with pytest.raises(ValueError, match="call leave_one_out"):
        res.get_leave_one_out_gaps()


def test_leave_one_out_does_not_touch_analytical_inference():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.leave_one_out()
    assert_nan_inference(
        {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
    )
    assert res.is_significant is False


def test_leave_one_out_requires_snapshot():
    res = _fit_for_placebo(n_donors=4)
    restored = pickle.loads(pickle.dumps(res))
    with pytest.raises(ValueError, match="requires the fit snapshot"):
        restored.leave_one_out()


# ---------------------------------------------------------------------------
# In-time placebo: snapshot-truncation helper (ADH 2015 §4)
# ---------------------------------------------------------------------------


def _snap_for_in_time(**kw):
    return _fit_for_placebo(n_donors=4, **kw)._fit_snapshot


def test_truncate_snapshot_positional_split():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    snap = _snap_for_in_time()
    assert list(snap.pre_periods) == [2000, 2001, 2002, 2003, 2004, 2005]
    mod, _ = _truncate_snapshot_in_time(snap, 2003)
    assert mod is not None
    assert mod.pre_periods == [2000, 2001, 2002]  # pre-fake = strictly before t_f
    assert mod.post_periods == [2003, 2004, 2005]  # post-fake = held-out pre, t_f first
    # all_periods EXCLUDES the true post periods (2006, 2007) -> airtight no-peeking.
    assert mod.all_periods == [2000, 2001, 2002, 2003, 2004, 2005]
    assert 2006 not in mod.all_periods and 2007 not in mod.all_periods


def test_truncate_snapshot_drops_specs_in_held_out_window():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    snap = _snap_for_in_time()  # default pre_period_outcomes="all": one lag per pre period
    mod, dropped = _truncate_snapshot_in_time(snap, 2003)
    for spec in mod.specs:  # surviving specs reference only pre-fake periods
        assert all(p < 2003 for p in spec.periods)
    assert len(dropped) == 3  # lags at 2003/2004/2005 dropped
    assert len(mod.specs) == len(snap.specs) - 3


def test_truncate_snapshot_custom_v_lockstep():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=np.arange(1.0, 7.0),  # distinct entries to verify the subset
            inner_min_decrease=1e-3,
        )
    snap = res._fit_snapshot
    mod, _ = _truncate_snapshot_in_time(snap, 2003)
    # custom_v subset IN LOCKSTEP with the surviving specs (the default lag specs are
    # ordered by ascending pre period, so the first three entries survive).
    assert mod.custom_v is not None and len(mod.custom_v) == len(mod.specs)
    np.testing.assert_array_equal(mod.custom_v, np.array([1.0, 2.0, 3.0]))


def test_truncate_snapshot_straddling_window_partial_keep():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", [2002, 2003, 2004], "mean")],
            pre_period_outcomes=[2000, 2001],
            inner_min_decrease=1e-3,
        )
    snap = res._fit_snapshot
    mod, _ = _truncate_snapshot_in_time(snap, 2003)
    # The special predictor straddles t_f -> truncated to its pre-fake part [2002].
    special = [s for s in mod.specs if s.kind == "special"]
    assert len(special) == 1 and special[0].periods == [2002]


def test_truncate_snapshot_infeasible_too_few_pre_fake():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    snap = _snap_for_in_time()
    # Fewer than 2 pre-fake periods -> infeasible (the deliberate >=2 rule; an
    # auto-swept single-pre-fake placebo is a non-credible pre-fit — documented Note).
    assert _truncate_snapshot_in_time(snap, 2000)[0] is None  # 0 pre-fake
    assert _truncate_snapshot_in_time(snap, 2001)[0] is None  # 1 pre-fake


def test_truncate_snapshot_infeasible_all_specs_dropped():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", [2004, 2005], "mean")],
            pre_period_outcomes=[2004, 2005],
            inner_min_decrease=1e-3,
        )
    snap = res._fit_snapshot
    # t_f=2003 leaves >=2 pre-fake periods, but every spec lives in [2004, 2005]
    # -> all dropped -> infeasible (cannot fit with zero predictors).
    mod, dropped = _truncate_snapshot_in_time(snap, 2003)
    assert mod is None and len(dropped) == len(snap.specs)


def test_truncate_snapshot_does_not_mutate_original():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    snap = _snap_for_in_time()
    before = [list(s.periods) for s in snap.specs]
    _truncate_snapshot_in_time(snap, 2003)
    after = [list(s.periods) for s in snap.specs]
    assert before == after  # shared spec objects are never mutated in place


# ---------------------------------------------------------------------------
# In-time placebo: end-to-end (ADH 2015 §4)
# ---------------------------------------------------------------------------

_IN_TIME_COLS = [
    "placebo_period",
    "placebo_att",
    "pre_fit_rmspe",
    "rmspe_ratio",
    "n_pre_fake",
    "n_post_fake",
    "n_dropped_specs",
    "status",
]


def test_in_time_placebo_near_zero_when_effect_post_only():
    # The effect is only in the TRUE post window (>=2006); every backdated placebo
    # falls in the clean pre window, so the placebo "effect" should be ~0.
    res = _fit_for_placebo(n_donors=4, effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo()
    assert list(itp.columns) == _IN_TIME_COLS
    ran = itp[itp["status"] == "ran"]
    assert len(ran) > 0
    assert ran["placebo_att"].abs().max() < 1.0  # well below the 3.0 true effect


def test_in_time_placebo_sweep_feasibility():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo()
    # pre = [2000..2005] -> feasible dates = pre[2:] = [2002, 2003, 2004, 2005]
    # (>=2 pre-fake periods — the deliberate Note-documented restriction).
    assert list(itp["placebo_period"]) == [2002, 2003, 2004, 2005]
    assert (itp["status"] == "ran").all()
    # n_pre_fake + n_post_fake == n_pre for every row, with >=2 pre-fake + >=1 post-fake.
    assert ((itp["n_pre_fake"] + itp["n_post_fake"]) == len(res.pre_periods)).all()
    assert (itp["n_pre_fake"] >= 2).all() and (itp["n_post_fake"] >= 1).all()


def test_in_time_placebo_explicit_post_date_raises():
    res = _fit_for_placebo(n_donors=4)
    with pytest.raises(ValueError, match="true post-treatment period"):
        res.in_time_placebo([2006])


def test_in_time_placebo_date_not_in_pre_raises():
    res = _fit_for_placebo(n_donors=4)
    with pytest.raises(ValueError, match="not a pre-treatment period"):
        res.in_time_placebo([1999])


def test_in_time_placebo_empty_explicit_input_raises():
    # An explicit but EMPTY container is malformed (NOT "every date infeasible") -> raise
    # (codex R6 P1). None still means "sweep all feasible dates".
    res = _fit_for_placebo(n_donors=4)
    for empty in ([], (), pd.Index([]), np.array([])):
        with pytest.raises(ValueError, match="placebo_periods is empty"):
            res.in_time_placebo(empty)
    # The malformed call must not leave any in-time state behind.
    assert res._in_time_df is None and res._in_time_status is None


def test_in_time_placebo_dedups_and_canonicalizes_explicit_dates():
    # Duplicate / unordered explicit dates -> de-duplicated + pre-period-ordered, so no
    # duplicate refits and n_dates is not inflated (codex R7 P3).
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo([2004, 2002, 2004])  # duplicate 2004, unordered
    assert list(itp["placebo_period"]) == [2002, 2004]  # unique, canonical pre-period order


def test_in_time_placebo_ran_block_reports_partial_coverage():
    # CI codex P2: a sweep where SOME dates ran and SOME were infeasible must surface
    # n_ran / n_infeasible on the status="ran" block so coverage is not overstated.
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_time_placebo([2001, 2003])  # 2001 infeasible (1 pre-fake), 2003 runs
    assert res._in_time_status == "ran"  # at least one date ran
    block = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]["in_time_placebo"]
    assert block["status"] == "ran"
    assert block["n_dates"] == 2 and block["n_ran"] == 1
    assert block["n_infeasible"] == 1 and block["n_failed"] == 0


def test_leave_one_out_immune_to_donor_weights_mutation():
    # Codex R8 P1: the LOO drop-set is FROZEN at fit time (snap.weighted_donor_ids =
    # the >1e-6 reportable support), NOT read from the mutable presentation-level
    # donor_weights dict. So mutating donor_weights after the fit must NOT change which
    # donors are dropped — the robustness result depends only on the fit.
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = set(res.leave_one_out()[lambda d: d["status"] != "baseline"]["dropped_unit"])
    assert before == set(res._fit_snapshot.weighted_donor_ids)  # drops the frozen support
    # Mutate the public dict: drop a real donor, inject a bogus one.
    victim = next(iter(res.donor_weights))
    res.donor_weights = {k: v for k, v in res.donor_weights.items() if k != victim}
    res.donor_weights["bogus_donor"] = 0.99
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        after = set(res.leave_one_out()[lambda d: d["status"] != "baseline"]["dropped_unit"])
    assert after == before  # unchanged by the mutation
    assert "bogus_donor" not in after  # a donor not in the fit is never dropped
    assert victim in after  # still dropped despite removal from donor_weights


def test_in_time_placebo_early_date_infeasible_no_raise():
    res = _fit_for_placebo(n_donors=4)
    # A valid pre-date with too few (<2) pre-fake periods -> NaN infeasible row +
    # warning, NOT a raise.
    with pytest.warns(UserWarning, match="infeasible"):
        itp = res.in_time_placebo([2001])  # 1 pre-fake period
    assert len(itp) == 1 and itp.iloc[0]["status"] == "infeasible"
    assert np.isnan(itp.iloc[0]["placebo_att"])


def test_in_time_placebo_custom_v_zero_mass_is_infeasible_not_failed():
    # A custom_v whose mass lies entirely on specs that TRUNCATE drops leaves a
    # zero-mass surviving V -> the date is INFEASIBLE under the supplied custom_v,
    # NOT a convergence failure (codex R2 P1b: v/v.sum() would be 0/0).
    df, _, _ = _make_panel(n_donors=4)  # default: 6 lag specs (2000..2005)
    v = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # all mass on the 2003/2004/2005 lags
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v,
            inner_min_decrease=1e-3,
        )
        itp = res.in_time_placebo([2003])  # keeps lags 2000/2001/2002 -> all zero weight
    row = itp[itp["placebo_period"] == 2003]
    assert len(row) == 1 and row.iloc[0]["status"] == "infeasible"  # NOT "failed"
    assert res._in_time_status == "all_dates_infeasible"


def test_leave_one_out_uniform_shift_surfaced_by_delta_not_range(monkeypatch):
    # Codex R3 P1b: when every donor-drop shifts the ATT the SAME way, the raw
    # att_range has ~zero width (looks stable) but the donor dependence is large.
    # The headline metric must be baseline-relative (max |delta_att|), not the range.
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    baseline = float(res.att)
    snap = res._fit_snapshot
    shift = 5.0  # same large shift for EVERY drop -> uniform

    def uniform_shift(snap_arg, unit, pool, n_starts):
        gp = {p: 0.0 for p in snap.pre_periods}
        gp.update({p: baseline + shift for p in snap.post_periods})
        return (gp, 1.0), "ran"

    monkeypatch.setattr(sc, "_placebo_fit_unit", uniform_shift)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.leave_one_out()
    lo, hi = res._loo_att_range
    assert (hi - lo) == pytest.approx(0.0, abs=1e-9)  # raw range would hide the shift
    assert res._loo_max_abs_delta_att == pytest.approx(shift, abs=1e-9)  # delta reveals it
    native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
    assert native["leave_one_out"]["max_abs_delta_att"] == pytest.approx(shift, abs=1e-9)


def test_in_time_placebo_windowed_covariate_dropped_and_warns():
    # A special predictor measured over [2004, 2005] falls entirely in the held-out
    # window for t_f=2003 -> dropped (TRUNCATE) + warning + n_dropped_specs reflects it.
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", [2004, 2005], "mean")],
            pre_period_outcomes=[2000, 2001, 2002, 2003],
            inner_min_decrease=1e-3,
        )
    with pytest.warns(UserWarning, match="dropped"):
        itp = res.in_time_placebo([2003])
    row = itp.iloc[0]
    # The special predictor (and the lag at 2003) lie in [2003, 2005] -> dropped.
    assert row["n_dropped_specs"] >= 1 and row["status"] == "ran"


def test_in_time_placebo_all_specs_dropped_infeasible():
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", [2004, 2005], "mean")],
            pre_period_outcomes=[2004, 2005],
            inner_min_decrease=1e-3,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo([2003])  # every predictor is at 2004/2005
    assert itp.iloc[0]["status"] == "infeasible"


def test_in_time_placebo_custom_v_runs_without_shape_error():
    # End-to-end guard for the custom_v lockstep subset: without it the custom path
    # would raise a shape mismatch once specs are dropped.
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=np.ones(6),
            inner_min_decrease=1e-3,
        )
        itp = res.in_time_placebo()
    assert (itp["status"] == "ran").any()


def test_in_time_placebo_accepts_2d_custom_v():
    # fit() accepts an array-like custom_v (e.g. a (1, k) row vector, raveled during
    # validation); the in-time TRUNCATE subset must ravel before indexing or a 2D
    # custom_v raises IndexError (codex R5 P1). Must match the 1D result exactly.
    df, _, _ = _make_panel(n_donors=4)
    v1d = np.arange(1.0, 7.0)
    v2d = v1d.reshape(1, 6)  # row-vector form accepted at fit time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res1 = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v1d,
            inner_min_decrease=1e-3,
        )
        res2 = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v2d,
            inner_min_decrease=1e-3,
        )
        itp1 = res1.in_time_placebo([2003])
        itp2 = res2.in_time_placebo([2003])  # would IndexError before the ravel fix
    pd.testing.assert_frame_equal(itp1, itp2)


def test_in_time_placebo_deterministic():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp1 = res.in_time_placebo()
        itp2 = res.in_time_placebo()
    pd.testing.assert_frame_equal(itp1, itp2)


def test_in_time_placebo_fails_closed_on_nonconverged_treated_fit():
    df, _, _ = _make_panel(n_donors=4, effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df, "y", "treated", "unit", "year", seed=0, inner_max_iter=1, **_FAST_CHURN
        )
    assert res._fit_converged is False
    with pytest.warns(UserWarning, match="did not converge"):
        itp = res.in_time_placebo()
    assert len(itp) == 0 and res._in_time_status == "treated_fit_nonconverged"


def test_in_time_placebo_pickle_drops_gaps_keeps_table():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_time_placebo()
    restored = pickle.loads(pickle.dumps(res))
    pd.testing.assert_frame_equal(restored.get_in_time_placebo_df(), res.get_in_time_placebo_df())
    assert restored._in_time_gaps is None
    with pytest.raises(ValueError, match="not retained after pickling"):
        restored.get_in_time_placebo_gaps()
    with pytest.raises(ValueError, match="requires the fit snapshot"):
        restored.in_time_placebo()


def test_in_time_placebo_gaps_long_form():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_time_placebo([2003])
    gaps = res.get_in_time_placebo_gaps()
    assert list(gaps.columns) == ["placebo_period", "period", "gap", "phase"]
    assert set(gaps["phase"]) == {"pre_fake", "post_fake"}
    # Periods before t_f=2003 are pre_fake; 2003+ are post_fake.
    assert set(gaps.loc[gaps["phase"] == "pre_fake", "period"]) == {2000, 2001, 2002}
    assert set(gaps.loc[gaps["phase"] == "post_fake", "period"]) == {2003, 2004, 2005}


def test_in_time_placebo_accessor_before_run_raises():
    res = _fit_for_placebo(n_donors=4)
    with pytest.raises(ValueError, match="call in_time_placebo"):
        res.get_in_time_placebo_df()
    with pytest.raises(ValueError, match="call in_time_placebo"):
        res.get_in_time_placebo_gaps()


def test_in_time_placebo_does_not_touch_analytical_inference():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_time_placebo()
    assert_nan_inference(
        {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
    )
    assert res.is_significant is False


# ---------------------------------------------------------------------------
# Self-consistency parity: the ADH-2015 diagnostics are EXACT re-runs of the
# validated solver on the equivalent sub-problem.
#
# R `Synth` has NO in-time-placebo or leave-one-out function (verified against its
# full CRAN function index), so there is no canonical R *output* to match for these
# diagnostics specifically. Instead we prove (deterministically, via a fixed custom
# V) that leave_one_out() equals a from-scratch fit on the reduced donor pool, and
# in_time_placebo() equals a from-scratch fit on the backdated/truncated panel.
# Because the custom-V solver is itself R-anchored on Basque
# (test_basque_tier1_custom_v_parity), this transitively anchors the diagnostics to
# R while directly validating that the re-run mechanism is exact (not approximate).
# ---------------------------------------------------------------------------


def test_leave_one_out_matches_fresh_reduced_pool_fit():
    df, _, _ = _make_panel(n_donors=4)
    v = np.arange(1.0, 7.0)  # k = 6 default lag predictors; fixed V -> deterministic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v,
            inner_min_decrease=1e-3,
        )
        loo = res.leave_one_out()
    donor_ids = list(res._fit_snapshot.donor_ids)
    d = [x for x in donor_ids if x in res.donor_weights][0]  # a positively-weighted donor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v,
            inner_min_decrease=1e-3,
            donor_pool=[x for x in donor_ids if x != d],
        )
    loo_att = loo.loc[loo["dropped_unit"] == d, "att"].iloc[0]
    assert loo_att == pytest.approx(fresh.att, abs=1e-7)


def test_in_time_placebo_matches_fresh_backdated_fit():
    df, _, _ = _make_panel(n_donors=4)  # years 2000-2007, T0=6 -> pre = 2000..2005
    v = np.arange(1.0, 7.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v,
            inner_min_decrease=1e-3,
        )
        itp = res.in_time_placebo([2003])
    placebo_att = itp.loc[itp["placebo_period"] == 2003, "placebo_att"].iloc[0]
    # Fresh backdated fit: drop the true post periods, treat 2003 as the intervention,
    # feed the pre-fake-subset V (lags at 2000/2001/2002 -> v[:3]).
    back = df[df["year"] <= 2005].copy()
    back["treated"] = ((back["unit"] == "treated") & (back["year"] >= 2003)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = synthetic_control(
            back,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v[:3],
            inner_min_decrease=1e-3,
        )
    assert placebo_att == pytest.approx(fresh.att, abs=1e-7)


# ---------------------------------------------------------------------------
# All-refits-failed branches (codex R1 P1): when EVERY refit fails to converge,
# the status must NOT be reported as "ran" / mislabeled as dimensional infeasibility.
# ---------------------------------------------------------------------------


def test_leave_one_out_all_refits_failed_status(monkeypatch):
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    # every drop fails to converge (solver, not structural)
    monkeypatch.setattr(sc, "_placebo_fit_unit", lambda *a, **k: (None, "failed"))
    with pytest.warns(UserWarning, match="did not reach a valid optimum"):
        loo = res.leave_one_out()
    # Distinct status (NOT "ran"); att_range is None; baseline + only failed rows.
    assert res._loo_status == "all_refits_failed"
    assert res._loo_att_range is None
    assert (loo["status"] != "loo").all()  # no successful drop
    assert (loo.iloc[1:]["status"] == "failed").all()
    # DiagnosticReport must surface it as NOT "ran", with the convergence reason.
    native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
    assert native["leave_one_out"]["status"] != "ran"
    # Machine-readable code distinguishes numerical failure from structural infeasibility.
    assert native["leave_one_out"]["reason_code"] == "all_refits_failed"
    assert "failed to converge" in native["leave_one_out"]["reason"]


def test_in_time_placebo_all_dates_failed_status(monkeypatch):
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    # every refit fails to converge (solver, not structural)
    monkeypatch.setattr(sc, "_placebo_fit_unit", lambda *a, **k: (None, "failed"))
    with pytest.warns(UserWarning, match="failed to converge"):
        itp = res.in_time_placebo()
    # Convergence failure must NOT be mislabeled as dimensional infeasibility.
    assert res._in_time_status == "all_dates_failed"
    assert (itp["status"] == "failed").all() and len(itp) > 0
    native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
    assert native["in_time_placebo"]["status"] != "ran"
    assert native["in_time_placebo"]["reason_code"] == "all_dates_failed"
    assert "failed to converge" in native["in_time_placebo"]["reason"]


def test_in_time_placebo_mixed_failed_and_infeasible_status(monkeypatch):
    # Codex R8 P2: a no-success run with BOTH a dimensionally-infeasible date AND a
    # convergence-failed date must report the mixed "all_dates_unusable" status with
    # both counts — NOT be mislabeled as exclusively failed (which would falsely claim
    # "none was dimensionally infeasible").
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    # Feasible dates "fail" to converge; 2001 (1 pre-fake) is dimensionally infeasible.
    monkeypatch.setattr(sc, "_placebo_fit_unit", lambda *a, **k: (None, "failed"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo([2001, 2003])  # 2001 infeasible, 2003 fails
    assert res._in_time_status == "all_dates_unusable"
    assert res._in_time_n_failed == 1 and res._in_time_n_infeasible == 1
    assert set(itp["status"]) == {"infeasible", "failed"}
    block = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]["in_time_placebo"]
    assert block["reason_code"] == "all_dates_unusable"
    assert block["n_failed"] == 1 and block["n_infeasible"] == 1


def test_in_space_placebo_all_infeasible_status(monkeypatch):
    # Every donor refit is STRUCTURALLY infeasible (cv donor-indistinguishability) ->
    # "all_placebos_infeasible", distinct from the solver "all_placebos_failed", with a
    # machine-readable reason_code + n_infeasible surfaced on DiagnosticReport.
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    monkeypatch.setattr(sc, "_placebo_fit_unit", lambda *a, **k: (None, "infeasible"))
    with pytest.warns(UserWarning, match="STRUCTURALLY infeasible"):
        pdf = res.in_space_placebo()
    assert res._placebo_status == "all_placebos_infeasible"
    assert res.n_placebos == 0 and res.n_failed == 0 and res.n_infeasible == res.n_donors
    assert (pdf["status"] == "infeasible").sum() == res.n_donors  # every donor row
    block = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]["in_space_placebo"]
    assert block["status"] == "infeasible"
    assert block["reason_code"] == "all_placebos_infeasible"
    assert block["n_infeasible"] == res.n_donors and block["n_failed"] == 0
    assert "structurally infeasible" in block["reason"]


def test_in_space_placebo_unusable_status(monkeypatch):
    # A mix of solver failures AND structural infeasibilities with no usable placebo ->
    # "all_placebos_unusable" (both counters surfaced), not mislabeled as exclusively one.
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    calls = {"n": 0}

    def mixed(*a, **k):
        calls["n"] += 1
        return (None, "infeasible") if calls["n"] % 2 else (None, "failed")

    monkeypatch.setattr(sc, "_placebo_fit_unit", mixed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    assert res._placebo_status == "all_placebos_unusable"
    assert res.n_placebos == 0 and res.n_failed > 0 and res.n_infeasible > 0
    block = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]["in_space_placebo"]
    assert block["reason_code"] == "all_placebos_unusable"
    assert block["n_failed"] == res.n_failed and block["n_infeasible"] == res.n_infeasible


def test_confidence_set_reason_names_all_placebos_infeasible(monkeypatch):
    # The test-inversion entrypoint (_require_placebo_reference, shared by confidence_set /
    # test_sharp_null) must NAME the new no-reference statuses: an all-infeasible in-space
    # run raises with the STRUCTURAL reason, not the generic "no valid reference set" fallback.
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    monkeypatch.setattr(sc, "_placebo_fit_unit", lambda *a, **k: (None, "infeasible"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    assert res._placebo_status == "all_placebos_infeasible"
    with pytest.raises(ValueError, match="structurally infeasible"):
        res.confidence_set(family="constant", gamma=0.25)


def test_cv_leave_one_out_infeasible_drop_counted_separately(monkeypatch):
    # A single structurally-infeasible drop is routed to _loo_n_infeasible (NOT
    # _loo_n_failed); the run still "ran" via the other drops, and the excluded row
    # carries status="infeasible".
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    real_fit_unit = sc._placebo_fit_unit
    calls = {"n": 0}

    def flaky(snap, unit, pool, n_starts):
        calls["n"] += 1
        if calls["n"] == 1:  # first drop is structurally infeasible
            return None, "infeasible"
        return real_fit_unit(snap, unit, pool, n_starts)

    monkeypatch.setattr(sc, "_placebo_fit_unit", flaky)
    with pytest.warns(UserWarning, match="STRUCTURALLY infeasible"):
        loo = res.leave_one_out()
    assert res._loo_status == "ran"
    assert res._loo_n_infeasible == 1 and res._loo_n_failed == 0
    assert (loo["status"] == "infeasible").sum() == 1
    native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]["leave_one_out"]
    assert native["status"] == "ran" and native["n_infeasible"] == 1


def test_cv_leave_one_out_flat_reduced_pool_infeasible():
    # LOO real-mechanism test (solve-level cv sentinel, mirroring the in-space flat-refit
    # test): the treated unit equals the linchpin donor d0 (=8) in the flat validation
    # window {2003,2004,2005}, so the cv fit loads all weight on d0 (the only donor at 8).
    # Dropping d0 — the sole reportably-weighted donor — leaves {d1,d2,d3} identical (=10)
    # there, so the reduced pool is donor-indistinguishable: a STRUCTURAL cv infeasibility
    # routed to _loo_n_infeasible (NOT the solver _loo_n_failed), with status="infeasible".
    rng = np.random.default_rng(3)
    years = list(range(2000, 2008))
    rows = []
    for j in range(4):
        for yr in years:
            if yr in (2003, 2004, 2005):
                y = 8.0 if j == 0 else 10.0  # d0 distinguishes the pool in the flat window
            elif yr <= 2005:
                y = 5.0 + j + rng.normal(0, 0.2)
            else:
                y = 10.0 + rng.normal(0, 0.2)
            rows.append({"unit": f"d{j}", "year": yr, "y": y, "treated": 0})
    for yr in years:
        # treated == d0 in the flat window -> the synthetic must weight d0 (only donor at 8)
        y = 8.0 if yr in (2003, 2004, 2005) else (5.5 + rng.normal(0, 0.2) if yr <= 2005 else 13.0)
        rows.append({"unit": "treated", "year": yr, "y": y, "treated": int(yr >= 2006)})
    df = pd.DataFrame(rows)
    res = _fit_cv(df, seed=0)
    assert np.isfinite(res.att)  # headline well-posed (full donor set distinguishable)
    assert list(res._fit_snapshot.weighted_donor_ids) == ["d0"]  # d0 is the sole weighted donor
    with pytest.warns(UserWarning, match="STRUCTURALLY infeasible"):
        loo = res.leave_one_out()
    # Real cv sentinel: the d0 drop is structural infeasibility, not solver failure.
    assert res._loo_status == "all_refits_infeasible"
    assert res._loo_n_infeasible == 1 and res._loo_n_failed == 0
    assert res._loo_att_range is None
    dropped = loo[loo["dropped_unit"] == "d0"]
    assert len(dropped) == 1 and dropped.iloc[0]["status"] == "infeasible"
    native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]["leave_one_out"]
    assert native["status"] != "ran"
    assert native["reason_code"] == "all_refits_infeasible"
    assert native["n_infeasible"] == 1 and native["n_failed"] == 0


def test_leave_one_out_unusable_status(monkeypatch):
    # A mix of failed + infeasible drops with none usable -> "all_refits_unusable".
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    assert len(res._fit_snapshot.weighted_donor_ids) >= 2  # need >=2 drops for a mix
    calls = {"n": 0}

    def mixed(*a, **k):
        calls["n"] += 1
        return (None, "infeasible") if calls["n"] == 1 else (None, "failed")

    monkeypatch.setattr(sc, "_placebo_fit_unit", mixed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.leave_one_out()
    assert res._loo_status == "all_refits_unusable"
    assert res._loo_n_failed > 0 and res._loo_n_infeasible > 0
    native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]["leave_one_out"]
    assert native["reason_code"] == "all_refits_unusable"
    assert (
        native["n_failed"] == res._loo_n_failed and native["n_infeasible"] == res._loo_n_infeasible
    )


# ===========================================================================
# V-selection menu: v_method="inverse_variance" and v_method="cv"
# (ADH 2015 § / Abadie 2021 Eq. 9; §3.2(a) inverse-variance). The CV per-window
# re-aggregation reproduces ADH 2015's manual two-dataprep CV re-run for our
# absolute-period spec aggregates (see REGISTRY §SyntheticControl).
# ===========================================================================


# --- config / validation (cheap; no fit) ----------------------------------


def test_inverse_variance_and_cv_methods_accepted():
    # Both new v_method values construct without error and round-trip v_cv_t0.
    SyntheticControl(v_method="inverse_variance")
    est = SyntheticControl(v_method="cv", v_cv_t0=3)
    assert est.get_params()["v_cv_t0"] == 3


def test_v_cv_t0_requires_cv_method():
    # Fail closed: v_cv_t0 is meaningless unless v_method="cv" (it would be silently
    # ignored otherwise), mirroring the custom_v cross-field rule.
    for method in ("nested", "custom", "inverse_variance"):
        kw = {"custom_v": [1.0, 1.0]} if method == "custom" else {}
        with pytest.raises(ValueError, match="v_cv_t0 is only valid when v_method='cv'"):
            SyntheticControl(v_method=method, v_cv_t0=2, **kw)


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True])
def test_v_cv_t0_type_and_positivity_rejected(bad):
    with pytest.raises(ValueError, match=r"v_cv_t0 must be"):
        SyntheticControl(v_method="cv", v_cv_t0=bad)


def test_custom_v_forbidden_for_cv_and_inverse_variance():
    for method in ("cv", "inverse_variance"):
        with pytest.raises(ValueError, match="custom_v must be None when v_method="):
            SyntheticControl(v_method=method, custom_v=[1.0, 1.0])


def test_set_params_cv_rollback():
    # A valid cv update sticks; an invalid combo (v_cv_t0 without cv) rolls back fully.
    est = SyntheticControl(v_method="cv", v_cv_t0=2)
    est.set_params(v_method="cv", v_cv_t0=3)
    assert est.v_cv_t0 == 3 and est.v_method == "cv"
    with pytest.raises(ValueError):
        est.set_params(v_method="nested")  # v_cv_t0=3 now invalid -> rollback
    assert est.v_method == "cv" and est.v_cv_t0 == 3  # unchanged


# --- inverse_variance behavior + parity ------------------------------------


def test_inverse_variance_fit_is_deterministic_and_searchless():
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = synthetic_control(df, "y", "treated", "unit", "year", v_method="inverse_variance")
        r2 = synthetic_control(df, "y", "treated", "unit", "year", v_method="inverse_variance")
    assert r1.mspe_v is None  # no outer search ran
    assert r1.att == r2.att  # fully deterministic (no rng)
    assert r1.donor_weights == r2.donor_weights


def test_inverse_variance_weights_equal_inverse_row_variance():
    # Closed-form anchor: the selected V equals trace-normalized 1/Var(X_row) computed
    # on the UNSTANDARDIZED predictors over donors+treated.
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(df, "y", "treated", "unit", "year", v_method="inverse_variance")
    snap = res._fit_snapshot
    X1, X0, labels = sc._build_predictor_matrix(
        snap.pivots, snap.specs, snap.treated_id, snap.donor_ids
    )
    expected = sc._inverse_variance_v(X1, X0)
    got = np.array([res.v_weights[lab] for lab in labels])
    assert np.allclose(got, expected, atol=1e-12)


def test_inverse_variance_exact_for_tiny_positive_variances():
    # Regression (local codex R6 P1): the inverse-variance V must be the EXACT 1/Var
    # selector for EVERY strictly-positive variance — no flooring of tiny-but-positive
    # variances. With two predictor rows of tiny-but-UNEQUAL variance (ratio 1:4 here), a
    # 1e-12 floor would clip both to the same value and equalize their V weights; the exact
    # selector preserves their 1/Var ratio. The oracle is built INLINE from raw row
    # variances (NOT via the production helper) so it genuinely cross-checks the code.
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    # 3 donors + treated (4 cols). Rows 0,1 have tiny positive variances; row 2 is normal.
    d0, d1 = 1e-8, 2e-8
    X0 = np.array(
        [
            [5.0 + d0, 5.0 - d0, 5.0 + d0],
            [2.0 + d1, 2.0 - d1, 2.0 + d1],
            [1.0, 3.0, 2.5],
        ]
    )
    X1 = np.array([5.0 - d0, 2.0 - d1, 2.0])
    row_var = np.var(np.column_stack([X0, X1.reshape(-1, 1)]), axis=1, ddof=1)
    assert np.all((row_var[:2] > 0) & (row_var[:2] < 1e-12))  # tiny-but-positive
    inv_oracle = 1.0 / row_var  # exact, all rows positive here
    v_oracle = inv_oracle / inv_oracle.sum()
    v = sc._inverse_variance_v(X1, X0)
    assert np.allclose(v, v_oracle, rtol=1e-12, atol=0.0)
    # Discrimination: the two tiny rows keep their exact 1/Var ratio (~4:1), which a
    # clipping implementation would collapse to 1:1.
    assert v[0] / v[1] == pytest.approx(row_var[1] / row_var[0], rel=1e-9)
    assert not np.isclose(v[0], v[1])  # NOT equalized by a floor


def test_inverse_variance_matches_paper_objective():
    # SOURCE-anchored: inverse_variance must realize Abadie 2021 §3.2(a)'s unit-variance
    # rescaled objective Σ_h diff_h²/Var_h, NOT the double-rescaled Σ_h diff_h²/Var_h²
    # that applying 1/Var on already-standardized predictors would produce. Two
    # independent encodings of the SAME paper objective (both R-anchored via the custom_v
    # path) must agree with the inverse_variance fit:
    #   (a) standardize="std" + custom_v = uniform  -> Σ_h (diff_h/SD_h)²·1 = Σ diff²/Var
    #   (b) standardize="none" + custom_v = 1/Var(X) -> Σ_h diff_h²·(1/Var_h) = Σ diff²/Var
    # The OLD self-equivalence (custom_v=1/Var at the default standardize="std") would
    # encode the BUGGY Σ diff²/Var² objective and is deliberately NOT used here.
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(df, "y", "treated", "unit", "year", v_method="inverse_variance")
        snap = res._fit_snapshot
        X1, X0, labels = sc._build_predictor_matrix(
            snap.pivots, snap.specs, snap.treated_id, snap.donor_ids
        )
        k = X1.shape[0]
        v_iv = sc._inverse_variance_v(X1, X0)
        res_uniform_std = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=np.ones(k),
            standardize="std",
        )
        res_invvar_none = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v_iv,
            standardize="none",
        )
    for ref in (res_uniform_std, res_invvar_none):
        assert res.att == pytest.approx(ref.att, abs=1e-9)
        assert res.donor_weights.keys() == ref.donor_weights.keys()
        for d in res.donor_weights:
            assert res.donor_weights[d] == pytest.approx(ref.donor_weights[d], abs=1e-9)
    # Guard: confirm the BUGGY double-rescale (custom_v=1/Var at standardize="std") gives
    # a DIFFERENT result, so this test actually discriminates the fix.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_double = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v_iv,
            standardize="std",
        )
    assert not np.allclose(
        [res.donor_weights.get(d, 0.0) for d in snap.donor_ids],
        [res_double.donor_weights.get(d, 0.0) for d in snap.donor_ids],
        atol=1e-6,
    )


def _panel_with_constant_lag(constant_years, n_donors=4, T=8, T0=6):
    """Panel where the outcome in ``constant_years`` is identical across ALL units
    (treated + donors), so those pre-period lag predictors have zero cross-unit
    variance. A post effect is added to the treated unit."""
    rng = np.random.default_rng(0)
    years = list(range(2000, 2000 + T))
    rows = []
    for j in range(n_donors):
        series = rng.normal(10, 2, T)
        for t in range(T):
            y = 7.0 if years[t] in constant_years else float(series[t])
            rows.append({"unit": f"d{j}", "year": years[t], "y": y, "treated": 0})
    for t in range(T):
        y = 7.0 if years[t] in constant_years else 10.0 + rng.normal(0, 1)
        rows.append(
            {
                "unit": "treated",
                "year": years[t],
                "y": y + (5.0 if t >= T0 else 0.0),
                "treated": int(t >= T0),
            }
        )
    return pd.DataFrame(rows), years


def test_inverse_variance_zero_variance_row_gets_zero_weight():
    # A single zero-variance predictor row (one pre-period constant across units) gets
    # 0 V weight; the others keep positive, trace-normalized weight.
    df, years = _panel_with_constant_lag(constant_years={2001})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(df, "y", "treated", "unit", "year", v_method="inverse_variance")
    assert res.v_weights["y_2001"] == pytest.approx(0.0, abs=1e-12)
    assert sum(res.v_weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert any(v > 0 for k, v in res.v_weights.items() if k != "y_2001")


def test_inverse_variance_all_zero_variance_falls_back_to_uniform():
    # EVERY pre-period constant across units -> no information to weight predictors ->
    # uniform V + ONE warning.
    df, years = _panel_with_constant_lag(constant_years={2000, 2001, 2002, 2003, 2004, 2005})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("always", UserWarning)
        with pytest.warns(UserWarning, match="no usable predictor variance"):
            res = synthetic_control(df, "y", "treated", "unit", "year", v_method="inverse_variance")
    vals = list(res.v_weights.values())
    assert np.allclose(vals, 1.0 / len(vals))


def test_inverse_variance_single_donor_returns_uniform_v():
    # Documented single-donor contract (NOT a skip-bug): with J==1, w=[1] is forced and V is
    # UNIDENTIFIED (every V yields the same synthetic), so v_weights is uniform and mspe_v is
    # None for EVERY v_method — inverse_variance included (its closed-form 1/Var would be
    # inert here). The fit warns rather than silently relabeling.
    df, _, _ = _make_panel(n_donors=1)
    with pytest.warns(UserWarning, match="uniform regardless of v_method"):
        res = synthetic_control(df, "y", "treated", "unit", "year", v_method="inverse_variance")
    assert res.n_donors == 1
    assert abs(sum(res.donor_weights.values()) - 1.0) < 1e-9
    vw = list(res.v_weights.values())
    assert np.allclose(vw, 1.0 / len(vw))  # uniform V (unidentified), NOT 1/Var
    assert res.mspe_v is None


def test_inverse_variance_leave_one_out_recomputes_per_unit():
    # LOO under inverse_variance recomputes the closed-form V on each reduced pool and
    # runs deterministically.
    res = _fit_for_placebo(n_donors=4, v_method="inverse_variance")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = res.leave_one_out()
    assert res._loo_status == "ran"
    assert (loo[loo["status"] == "loo"].shape[0]) >= 1


def test_inverse_variance_in_space_placebo_recomputes_v_and_enters_reference_set(monkeypatch):
    # The in-space placebo refits must take the inverse_variance branch (recompute the
    # closed-form V per pseudo-treated unit, NOT fall through to nested) AND enter the
    # permutation reference set.
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4, v_method="inverse_variance")
    real_iv = sc._inverse_variance_v
    state = {"calls": 0}

    def spy(*a, **k):
        state["calls"] += 1
        return real_iv(*a, **k)

    monkeypatch.setattr(sc, "_inverse_variance_v", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    assert state["calls"] >= res.n_donors  # the iv branch recomputed V for each placebo
    assert res.n_placebos >= 1 and res._placebo_status == "ran"


def test_inverse_variance_in_time_placebo_matches_fresh_backdated_fit():
    # Self-consistency: the in-time placebo under inverse_variance equals a fresh
    # inverse_variance fit on the backdated panel. inverse_variance is deterministic (no
    # search), so the match is exact; this anchors the in-time placebo's inverse_variance
    # branch to a direct fit on the equivalent sub-problem.
    df, _, _ = _make_panel(n_donors=4)  # pre = 2000..2005 (default per-period lags)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df, "y", "treated", "unit", "year", v_method="inverse_variance", **_FAST
        )
        itp = res.in_time_placebo([2004])  # pre-fake = {2000..2003}
    placebo_att = itp.loc[itp["placebo_period"] == 2004, "placebo_att"].iloc[0]
    back = df[df["year"] <= 2005].copy()
    back["treated"] = ((back["unit"] == "treated") & (back["year"] >= 2004)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = synthetic_control(
            back, "y", "treated", "unit", "year", v_method="inverse_variance", **_FAST
        )
    assert placebo_att == pytest.approx(fresh.att, abs=1e-7)


# --- cv behavior + parity + determinism ------------------------------------


# CV tests use SPANNING predictors — multi-period special predictors observed in BOTH the
# training and validation halves of the 6-period pre-window 2000-2005 (split at t0=3, and
# also at t0=2) — so cv's fully-spanning precondition is satisfied and each predictor can be
# re-aggregated on each window. The default per-period outcome lags are single-period (each
# lives in one window only) and rejected (see test_cv_rejects_non_spanning_predictors).
_CV_SPANNING = [("y", [2000, 2002, 2004], "mean"), ("y", [2001, 2003, 2005], "mean")]


def _fit_cv(df, *, specs=_CV_SPANNING, **kw):
    opts = dict(_FAST)
    opts.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return synthetic_control(
            df, "y", "treated", "unit", "year", v_method="cv", special_predictors=specs, **opts
        )


def test_cv_rejects_non_spanning_predictors():
    # Fully-spanning precondition (fail-closed): faithful per-window re-aggregation needs
    # every predictor measurable on BOTH windows. The default per-period outcome lags are
    # single-period (each lives in one window only) so they cannot span, and cv rejects them
    # with guidance to pass spanning predictors.
    df, _, _ = _make_panel(n_donors=4)
    with pytest.raises(ValueError, match="span BOTH the training"):
        synthetic_control(df, "y", "treated", "unit", "year", v_method="cv", seed=0, **_FAST)


def test_cv_runs_and_reports_validation_mspe():
    df, _, _ = _make_panel(n_donors=4)
    res = _fit_cv(df, seed=0)
    assert np.isfinite(res.att)
    assert res.mspe_v is not None and np.isfinite(res.mspe_v)  # validation MSPE
    assert abs(sum(res.donor_weights.values()) - 1.0) < 1e-6


def test_cv_default_t0_is_half_and_explicit_t0_changes_result():
    df, _, _ = _make_panel(n_donors=4)  # 6 pre periods -> default t0 = 3
    res_default = _fit_cv(df, seed=0)
    res_t0_3 = _fit_cv(df, v_cv_t0=3, seed=0)
    res_t0_2 = _fit_cv(df, v_cv_t0=2, seed=0)
    # Default == explicit t0=3 (len(pre)//2 == 3).
    assert res_default.mspe_v == pytest.approx(res_t0_3.mspe_v, abs=1e-12)
    # A different split (different validation window) yields a different criterion value.
    assert res_t0_2.mspe_v != pytest.approx(res_t0_3.mspe_v, abs=1e-9)


def test_cv_t0_out_of_range_raises():
    # The t0-range check fires before the predictor-precondition check.
    df, _, _ = _make_panel(n_donors=4)  # 6 pre periods -> valid 1..5
    with pytest.raises(ValueError, match="out of range"):
        synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="cv",
            special_predictors=_CV_SPANNING,
            v_cv_t0=6,
            **_FAST,
        )


def test_cv_requires_two_pre_periods():
    # A single pre period cannot form both a training and validation window (this n_pre<2
    # check fires before the predictor-precondition check).
    rows = []
    years = [2000, 2001, 2002]
    rng = np.random.default_rng(0)
    for j in range(3):
        for yr in years:
            rows.append({"unit": f"d{j}", "year": yr, "y": 10.0 + rng.normal(), "treated": 0})
    for i, yr in enumerate(years):  # pre = {2000}; post = {2001, 2002}
        rows.append({"unit": "treated", "year": yr, "y": 11.0 + i, "treated": int(i >= 1)})
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="requires at least 2 pre-treatment periods"):
        synthetic_control(df, "y", "treated", "unit", "year", v_method="cv", **_FAST)


def test_cv_single_donor_validates_and_surfaces_v_cv_t0():
    # The single-donor (J==1) fast path must NOT bypass cv's v_cv_t0 resolution/validation:
    # an out-of-range split still raises, and a valid split is resolved + surfaced on the
    # result, even though the single-donor synthetic is degenerate (w = 1). Spanning
    # predictors keep the cv precondition satisfied so we exercise the J==1 path itself.
    df, _, _ = _make_panel(n_donors=1)  # 6 pre periods 2000-2005
    with pytest.raises(ValueError, match="out of range"):
        synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="cv",
            special_predictors=_CV_SPANNING,
            v_cv_t0=99,
            **_FAST,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # degenerate single-donor warning
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="cv",
            special_predictors=_CV_SPANNING,
            v_cv_t0=2,
            **_FAST,
        )
    assert res.n_donors == 1
    assert res.v_cv_t0 == 2  # surfaced despite the degenerate single-donor fast path
    assert abs(sum(res.donor_weights.values()) - 1.0) < 1e-9
    # Documented degenerate contract: V unidentified with one donor -> uniform v_weights +
    # mspe_v None, same as every other v_method (the cv selection would be inert here).
    vw = list(res.v_weights.values())
    assert np.allclose(vw, 1.0 / len(vw))
    assert res.mspe_v is None


def test_cv_same_seed_reproducible_under_multistart():
    # Footnote-7 non-uniqueness: with a FIXED SEED the cv fit is reproducible under
    # multistart — the deterministic tie-break selects the same V* (closest-to-uniform
    # among ties) regardless of start-evaluation order. cv is seeded like nested (the
    # n_starts>=4 Dirichlet starts are seed-dependent), so this asserts same-seed
    # reproducibility at n_starts=3 (3 deterministic heuristic starts), NOT
    # seed-independence.
    df, _, _ = _make_panel(n_donors=4)
    r1 = _fit_cv(df, n_starts=3, seed=0)
    r2 = _fit_cv(df, n_starts=3, seed=0)
    assert r1.att == r2.att
    assert r1.donor_weights == r2.donor_weights


def test_cv_reaggregation_matches_custom_v_per_window_steps():
    # R-parity anchor: the cv fit's REPORTED weights come from the step-4 refit of V* on the
    # VALIDATION-window re-aggregated predictors, and mspe_v is the validation MSPE of the
    # step-2/3 fit of V* on the TRAINING-window re-aggregated predictors. Both per-window
    # fits are reproduced via the R-anchored custom_v path: re-aggregate each spec over its
    # window (intersect its periods) and fit custom_v=V* on those re-aggregated predictors.
    # R Synth has no built-in CV function (ADH 2015's CV is a manual two-dataprep re-run);
    # this self-consistency anchors both CV steps to the custom_v path (transitively
    # R-anchored by the PR-1 Basque custom_v parity).
    df, _, _ = _make_panel(n_donors=4)
    res = _fit_cv(df, seed=0)
    snap = res._fit_snapshot
    pre = snap.pre_periods
    t0 = len(pre) // 2  # default = 3
    tr_set, va_set = set(pre[:t0]), set(pre[t0:])
    v_star = np.array(list(res.v_weights.values()))  # selected V, in spec order
    # Re-aggregate each spec over each window (the faithful ADH-2015 per-window dataprep).
    train_reagg = [(s.var, [p for p in s.periods if p in tr_set], s.op) for s in snap.specs]
    val_reagg = [(s.var, [p for p in s.periods if p in va_set], s.op) for s in snap.specs]
    # Each spec genuinely re-aggregates to DIFFERENT periods per window (the two-window
    # distinction is real, not a no-op).
    assert all(tp != vp for (_, tp, _), (_, vp, _) in zip(train_reagg, val_reagg))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fin = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            special_predictors=val_reagg,
            custom_v=v_star,
            inner_min_decrease=1e-3,
        )
        tr = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            special_predictors=train_reagg,
            custom_v=v_star,
            inner_min_decrease=1e-3,
        )
    # Step 4: reported weights == V* refit on the VALIDATION-window re-aggregated predictors.
    for d in snap.donor_ids:
        assert res.donor_weights.get(d, 0.0) == pytest.approx(
            fin.donor_weights.get(d, 0.0), abs=1e-6
        )
    # Step 3: mspe_v == validation MSPE of V* fit on the TRAINING-window re-aggregated preds.
    Y = snap.pivots[snap.outcome]
    Z1 = Y.loc[pre, snap.treated_id].to_numpy(float)
    Z0 = Y.loc[pre, snap.donor_ids].to_numpy(float)
    w_tr = np.array([tr.donor_weights.get(d, 0.0) for d in snap.donor_ids])
    val_mspe = float(np.mean((Z1[t0:] - Z0[t0:] @ w_tr) ** 2))
    assert res.mspe_v == pytest.approx(val_mspe, abs=1e-7)


# --- cv placebo threading + in-time ----------------------------------------


def test_cv_placebo_refit_uses_cv_and_enters_reference_set(monkeypatch):
    # The in-space placebo refits must take the cv per-window re-aggregation branch (NOT
    # fall through to nested) AND actually enter the permutation reference set.
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4, v_method="cv", special_predictors=_CV_SPANNING)
    real_cv = sc._outer_solve_V_cv
    state = {"calls": 0}

    def spy(*a, **k):
        state["calls"] += 1
        return real_cv(*a, **k)

    monkeypatch.setattr(sc, "_outer_solve_V_cv", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    assert state["calls"] >= res.n_donors  # the cv branch ran for each placebo unit
    assert res.n_placebos >= 1 and res._placebo_status == "ran"  # placebos entered the set


def test_cv_in_time_placebo_pinned_t0_nulled_after_truncation():
    # An explicit v_cv_t0 that exceeds the truncated pre-fake window is nulled to the
    # //2 default for the placebo refit (not preserved across in-time truncation), so the
    # backdated date still runs. Backdate to 2004 keeps both spanning specs spanning.
    df, _, _ = _make_panel(n_donors=4)  # pre = 2000..2005
    res = _fit_cv(df, v_cv_t0=4, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo(
            [2004]
        )  # truncated pre-fake = {2000..2003}; t0=4 invalid -> nulled
    assert itp.loc[itp["placebo_period"] == 2004, "status"].iloc[0] == "ran"


def test_cv_in_time_placebo_matches_fresh_backdated_fit():
    # Self-consistency: the in-time placebo under cv equals a fresh cv fit on the
    # backdated panel (fixed seed + n_starts=1 -> deterministic). Backdate to 2004:
    # pre-fake = {2000..2003}, the spanning specs truncate to {2000,2002}/{2001,2003}
    # (still spanning the t0=2 split), so the date is feasible.
    df, _, _ = _make_panel(n_donors=4)  # pre = 2000..2005
    res = _fit_cv(df, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo([2004])
    placebo_att = itp.loc[itp["placebo_period"] == 2004, "placebo_att"].iloc[0]
    back = df[df["year"] <= 2005].copy()
    back["treated"] = ((back["unit"] == "treated") & (back["year"] >= 2004)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # backdated pre-fake = {2000..2003}; the spanning specs truncated to that window.
        fresh = synthetic_control(
            back,
            "y",
            "treated",
            "unit",
            "year",
            v_method="cv",
            special_predictors=[("y", [2000, 2002], "mean"), ("y", [2001, 2003], "mean")],
            seed=0,
            n_starts=1,
            optimizer_options={"maxiter": 50},
            inner_min_decrease=1e-3,
        )
    assert placebo_att == pytest.approx(fresh.att, abs=1e-7)


def test_cv_v_cv_t0_surfaced_on_results_and_serialized():
    # The new v_cv_t0 constructor param must appear on the public results surface
    # (downstream-propagation rule): a public field (the RESOLVED split), in to_dict()
    # /to_dataframe(), and surviving a pickle round-trip; None for non-cv methods.
    df, _, _ = _make_panel(n_donors=4)  # 6 pre periods -> default split 3
    res_default = _fit_cv(df, seed=0)
    res_explicit = _fit_cv(df, v_cv_t0=2, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_nested = synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)
    # Resolved value: None constructor -> len(pre)//2 = 3; explicit -> 2; non-cv -> None.
    assert res_default.v_cv_t0 == 3
    assert res_explicit.v_cv_t0 == 2
    assert res_nested.v_cv_t0 is None
    # to_dict() / to_dataframe() carry it.
    assert res_explicit.to_dict()["v_cv_t0"] == 2
    assert res_explicit.to_dataframe()["v_cv_t0"].iloc[0] == 2
    # Survives pickling (it is a public scalar, not snapshot-only).
    restored = pickle.loads(pickle.dumps(res_explicit))
    assert restored.v_cv_t0 == 2


def test_cv_in_time_placebo_empty_window_is_infeasible_not_failed():
    # A truncated cv date can keep a predictor overall yet leave it on only ONE side of
    # the split (the other window then has NO predictor) -> the fully-spanning precondition
    # is broken -> STRUCTURAL infeasibility, must report status="infeasible" (not "failed").
    # Use a single special predictor spanning both windows at full pre but only the training
    # side after backdating.
    # Panel: 4 donors, years 2000..2010 (10 pre 2000..2009, post 2010). One special
    # predictor on {2000, 2008}: at full pre (split t0=5) both windows hold it (feasible
    # headline fit); backdated to 2005 the pre-fake is {2000..2004}, the spec truncates
    # to {2000}, split t0=2 -> validation window {2002,2003,2004} has no predictor.
    rng = np.random.default_rng(0)
    years = list(range(2000, 2011))
    rows = []
    for j in range(4):
        series = rng.normal(10, 2) + rng.normal(0, 0.3) * np.arange(11) + rng.normal(0, 0.15, 11)
        for t in range(11):
            rows.append({"unit": f"d{j}", "year": years[t], "y": float(series[t]), "treated": 0})
    treated = rng.normal(10, 2) + rng.normal(0, 0.3) * np.arange(11) + rng.normal(0, 0.1, 11)
    treated = treated.copy()
    treated[10] += 3.0
    for t in range(11):
        rows.append(
            {"unit": "treated", "year": years[t], "y": float(treated[t]), "treated": int(t >= 10)}
        )
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="cv",
            special_predictors=[("y", [2000, 2008], "mean")],
            seed=0,
            **_FAST,
        )
        itp = res.in_time_placebo([2005])
    assert itp.loc[itp["placebo_period"] == 2005, "status"].iloc[0] == "infeasible"


# --- cv flat-window identification gate (local codex R9 P0) -----------------


def _cv_panel_flat_years(flat_years, treated_flat=True):
    # 4 donors + treated, years 2000-2007 (pre 2000-2005, post 2006-2007). In `flat_years`
    # every DONOR has the same outcome (=10) -> zero cross-DONOR variance for any predictor
    # re-aggregated onto those years (X0·W is constant in W -> unidentified); other pre
    # years vary across donors. treated_flat just sets whether the treated unit also equals
    # 10 there (True) or differs at 12 (False) — either way the DONORS are identical, so the
    # cv donor-identification gate fires regardless of the treated unit's value.
    rng = np.random.default_rng(0)
    years = list(range(2000, 2008))
    rows = []
    for j in range(4):
        for yr in years:
            if yr in flat_years:
                y = 10.0
            elif yr <= 2005:
                y = 5.0 + j + rng.normal(0, 0.2)
            else:
                y = 10.0 + rng.normal(0, 0.2)
            rows.append({"unit": f"d{j}", "year": yr, "y": y, "treated": 0})
    for yr in years:
        if yr in flat_years:
            y = 10.0 if treated_flat else 12.0
        elif yr <= 2005:
            y = 5.5 + rng.normal(0, 0.2)
        else:
            y = 13.0
        rows.append({"unit": "treated", "year": yr, "y": y, "treated": int(yr >= 2006)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("treated_flat", [True, False])
def test_cv_rejects_window_with_no_donor_variation(treated_flat):
    # Fail-closed identification gate: W is identified by the DONORS being distinguishable
    # (X0·W is a convex combination of donor columns). If every donor has identical
    # predictors in a window, X0·W is constant in W -> flat objective -> arbitrary weights
    # reported as converged. The headline fit must RAISE. The validation window
    # {2003,2004,2005} is constant across all donors, so _CV_SPANNING re-aggregates to
    # donor-indistinguishable validation predictors (t0=3). This must fail closed EVEN when
    # the treated unit differs (treated_flat=False) — treated-vs-donor variation does NOT
    # identify W (the gate that an earlier revision missed).
    df = _cv_panel_flat_years({2003, 2004, 2005}, treated_flat=treated_flat)
    with pytest.raises(ValueError, match="every donor has identical predictors"):
        _fit_cv(df, seed=0)


def test_cv_in_time_placebo_flat_window_is_infeasible_not_failed():
    # A backdated date can leave a cv window with no cross-DONOR variation even when the
    # headline fit is well-posed -> STRUCTURAL infeasibility (status="infeasible", not a
    # convergence "failed"). Donors are identical in 2002,2003 only, so the headline windows
    # still carry donor variation (via 2000/2001/2004/2005) but backdating to 2004 (pre-fake
    # {2000..2003}, t0=2) makes the validation window {2002,2003} donor-indistinguishable.
    df = _cv_panel_flat_years({2002, 2003}, treated_flat=True)
    res = _fit_cv(df, seed=0)
    assert np.isfinite(res.att)  # headline well-posed (donors vary in the other years)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo([2004])
    assert itp.loc[itp["placebo_period"] == 2004, "status"].iloc[0] == "infeasible"


def test_cv_in_space_placebo_excludes_donor_flat_refits():
    # In-space placebo path (solve-level sentinel): the FULL donor set is distinguishable so
    # the headline fit is well-posed, but dropping donor d0 (pseudo-treating it) leaves the
    # remaining donors {d1,d2,d3} identical in the validation window -> that placebo's pool
    # is donor-indistinguishable and must be EXCLUDED (not enter with arbitrary "converged"
    # weights). Placebos for d1/d2/d3 keep d0 in the pool, so they remain identified.
    rng = np.random.default_rng(3)
    years = list(range(2000, 2008))
    rows = []
    for j in range(4):
        for yr in years:
            if yr in (2003, 2004, 2005):
                y = 8.0 if j == 0 else 10.0  # d1=d2=d3 identical; d0 distinguishes the full set
            elif yr <= 2005:
                y = 5.0 + j + rng.normal(0, 0.2)  # training varies across all donors
            else:
                y = 10.0 + rng.normal(0, 0.2)
            rows.append({"unit": f"d{j}", "year": yr, "y": y, "treated": 0})
    for yr in years:
        y = 12.0 if yr in (2003, 2004, 2005) else (5.5 + rng.normal(0, 0.2) if yr <= 2005 else 13.0)
        rows.append({"unit": "treated", "year": yr, "y": y, "treated": int(yr >= 2006)})
    df = pd.DataFrame(rows)
    res = _fit_cv(df, seed=0)
    assert np.isfinite(res.att)  # headline well-posed: the full donor set is distinguishable
    # The d0 placebo (pool {d1,d2,d3} identical in val) is dropped as STRUCTURALLY
    # infeasible (cv donor-indistinguishability), NOT a solver "failed": the threading
    # from _outer_solve_V_cv's structural sentinel through _placebo_fit_unit routes it to
    # n_infeasible. The others keep d0 in the pool, so they remain identified and enter.
    with pytest.warns(UserWarning, match="STRUCTURALLY infeasible"):
        pdf = res.in_space_placebo()
    assert res._placebo_status == "ran"
    assert 1 <= res.n_placebos < res.n_donors
    assert res.n_infeasible >= 1 and res.n_failed == 0  # structural, not solver
    # The excluded donor row carries status="infeasible" (not "failed").
    excluded = pdf[pdf["rmspe_ratio"].isna()]
    assert len(excluded) == res.n_infeasible
    assert (excluded["status"] == "infeasible").all()
    # DiagnosticReport surfaces the split count on the ran block.
    block = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]["in_space_placebo"]
    assert block["status"] == "ran" and block["n_infeasible"] == res.n_infeasible


@pytest.mark.parametrize("specs", [_CV_SPANNING, [("y", [2000, 2002, 2004], "mean")]])
def test_cv_fails_closed_when_training_solve_truncates(specs):
    # The training-window solve defines mspe_v (Eq. 9's held-out criterion); if it truncates
    # (inner_max_iter too small) the fit must FAIL CLOSED — mspe_v=NaN and _fit_converged
    # False — so downstream placebo/LOO diagnostics never run off an invalid CV criterion.
    # Covers BOTH the general multi-predictor path and the single-predictor k==1 fast path.
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="cv",
            special_predictors=specs,
            seed=0,
            n_starts=1,
            inner_max_iter=1,  # truncate the inner Frank-Wolfe solve
            inner_min_decrease=1e-3,
        )
    assert np.isnan(res.mspe_v)
    assert res._fit_converged is False


def test_leave_one_out_cv_branch_and_fresh_reduced_pool_parity(monkeypatch):
    # leave_one_out() under v_method="cv" must (a) actually exercise the cv re-aggregation
    # branch (_outer_solve_V_cv), and (b) each drop's ATT must match a FRESH cv fit on the
    # reduced donor pool (self-consistency; deterministic at n_starts=1) — the LOO wrapper
    # was threaded for cv, so it needs direct coverage, not just in-space/in-time.
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    df, _, _ = _make_panel(n_donors=4)
    res = _fit_cv(df, seed=0)
    real_cv = sc._outer_solve_V_cv
    state = {"calls": 0}

    def spy(*a, **k):
        state["calls"] += 1
        return real_cv(*a, **k)

    monkeypatch.setattr(sc, "_outer_solve_V_cv", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = res.leave_one_out()
    assert res._loo_status == "ran"
    loo_rows = loo[loo["status"] == "loo"]
    assert state["calls"] >= len(loo_rows) >= 1  # the cv branch ran for each drop

    # Fresh-reduced-pool parity for the most influential drop (the spy forwards to the real
    # solver, so the fresh fit below is unaffected by the patch).
    dropped = loo_rows.iloc[0]["dropped_unit"]
    loo_att = loo_rows.iloc[0]["att"]
    reduced = df[df["unit"] != dropped].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = synthetic_control(
            reduced,
            "y",
            "treated",
            "unit",
            "year",
            v_method="cv",
            special_predictors=_CV_SPANNING,
            seed=0,
            n_starts=1,
            optimizer_options={"maxiter": 50},
            inner_min_decrease=1e-3,
        )
    assert loo_att == pytest.approx(fresh.att, abs=1e-7)


# ===========================================================================
# Confidence sets by test inversion (Firpo & Possebom 2018, Section 4)
# ===========================================================================
#
# Two opt-in SyntheticControlResults methods built ON TOP of the in-space placebo:
# test_sharp_null(effect) tests H_0^f: alpha_1t = f(t) by re-ranking the stored
# placebo gaps (Eqs 12-13, phi=0, v=(1,...,1)); confidence_set(family=...) inverts
# that test over a one-parameter family (Eqs 14/16/18, strict p^f > gamma). No SCM
# refits. The benchmark f==0 case is identically the existing placebo_p_value
# (Eq 5 = Eq 13 at f==0). No R anchor (Synth has no test inversion): validated by
# self-consistency to the placebo p-value, a numpy oracle, and a coverage MC.


def _gp(pre_vals, post_vals, pre_periods, post_periods):
    """Build a {period: gap} path from pre/post value lists."""
    d = {p: float(v) for p, v in zip(pre_periods, pre_vals)}
    d.update({p: float(v) for p, v in zip(post_periods, post_vals)})
    return d


# Hand-built scenario for the helper-level oracle tests: the treated unit has the
# best pre-fit and a constant +2 post effect; 4 placebos have worse pre-fit + scattered
# post gaps (so for large |c| the treated becomes the most-deviant unit -> bounded set).
_ORACLE_PRE = [0, 1, 2]
_ORACLE_POST = [3, 4, 5]
_ORACLE_TREATED = _gp([0.08, -0.06, 0.05], [2.0, 2.0, 2.0], _ORACLE_PRE, _ORACLE_POST)
_ORACLE_PLACEBOS = {
    "a": _gp([0.3, -0.3, 0.3], [0.6, -0.4, 0.2], _ORACLE_PRE, _ORACLE_POST),
    "b": _gp([-0.3, 0.3, 0.2], [-0.5, 0.4, 0.3], _ORACLE_PRE, _ORACLE_POST),
    "c": _gp([0.2, 0.3, -0.3], [0.3, 0.5, -0.4], _ORACLE_PRE, _ORACLE_POST),
    "d": _gp([-0.2, 0.3, 0.3], [0.4, -0.3, 0.5], _ORACLE_PRE, _ORACLE_POST),
}


def _oracle_pre_denoms(scale=1.0):
    units = {0: _ORACLE_TREATED, **_ORACLE_PLACEBOS}
    return {
        u: _floored_pre_mspe(np.array([gp[p] for p in _ORACLE_PRE]), scale)
        for u, gp in units.items()
    }


def test_sharp_null_pvalue_matches_independent_oracle():
    pre_denoms = _oracle_pre_denoms()
    for c in (0.0, 1.0, 2.0, 3.5, -1.0):
        f = _constant_f_post(c, len(_ORACLE_POST))
        p, r1, n_ref = _sharp_null_pvalue(
            _ORACLE_TREATED, _ORACLE_PLACEBOS, _ORACLE_POST, f, pre_denoms, 0
        )
        # Independent re-implementation of Eqs 12-13.
        resid_t = np.array([_ORACLE_TREATED[q] for q in _ORACLE_POST]) - f
        r1_o = float(np.sqrt(np.mean(resid_t**2) / pre_denoms[0]))
        rj = []
        for u, gp in _ORACLE_PLACEBOS.items():
            resid = np.array([gp[q] for q in _ORACLE_POST]) - f
            rj.append(float(np.sqrt(np.mean(resid**2) / pre_denoms[u])))
        p_o = (1 + sum(1 for r in rj if r >= r1_o)) / (len(rj) + 1)
        assert n_ref == 4
        assert r1 == pytest.approx(r1_o)
        assert p == pytest.approx(p_o), (c, p, p_o)


def test_sharp_null_zero_equals_rmspe_ratio_helper():
    # Eq 13 at f==0 reduces to the ADH RMSPE-ratio statistic (Eq 4).
    scale = 1.0
    pre_denoms = _oracle_pre_denoms(scale)
    f0 = _constant_f_post(0.0, len(_ORACLE_POST))
    _, r1, _ = _sharp_null_pvalue(
        _ORACLE_TREATED, _ORACLE_PLACEBOS, _ORACLE_POST, f0, pre_denoms, 0
    )
    pre = np.array([_ORACLE_TREATED[p] for p in _ORACLE_PRE])
    post = np.array([_ORACLE_TREATED[p] for p in _ORACLE_POST])
    assert r1 == pytest.approx(_rmspe_ratio(pre, post, scale))


def test_floored_pre_denominator_is_per_unit_not_global():
    # M1: the RMSPE floor scale is PER-UNIT max|Z1|. For a near-perfect pre-fit (the
    # floor bites), a wrong GLOBAL scale would change the denominator and break the
    # f==0 == placebo_p_value anchor. Assert the per-unit denom == the _rmspe_ratio
    # denom and differs from a global-scale denom.
    pre = np.array([1e-9, -1e-9, 5e-10])  # near-perfect pre-fit -> the floor dominates
    post = np.array([2.0, 2.0, 2.0])
    scale_unit = 10.0
    denom_unit = _floored_pre_mspe(pre, scale_unit)  # 1e-8 * 10**2 = 1e-6
    denom_global = _floored_pre_mspe(pre, 1.0)  # 1e-8
    assert denom_unit == pytest.approx(1e-6)
    assert not np.isclose(denom_unit, denom_global)
    gp = _gp(list(pre), list(post), _ORACLE_PRE, _ORACLE_POST)
    f0 = _constant_f_post(0.0, 3)
    assert _rmspe_f_ratio(gp, _ORACLE_POST, f0, denom_unit) == pytest.approx(
        _rmspe_ratio(pre, post, scale_unit)
    )
    # The wrong global denom yields a materially different statistic.
    assert not np.isclose(
        _rmspe_f_ratio(gp, _ORACLE_POST, f0, denom_unit),
        _rmspe_f_ratio(gp, _ORACLE_POST, f0, denom_global),
    )


def test_invert_constant_set_brackets_true_effect():
    pre_denoms = _oracle_pre_denoms()
    res = _invert_sharp_null(
        _ORACLE_TREATED, _ORACLE_PLACEBOS, _ORACLE_POST, pre_denoms, 0, "constant", 0.25, n_grid=120
    )
    assert res["status"] == "ran"
    assert res["point_estimate"] == pytest.approx(2.0)  # center = att = mean post gap
    assert res["lower"] <= 2.0 <= res["upper"]
    assert res["contiguous"]


def test_invert_strict_boundary_excludes_p_equals_gamma():
    # Eq 14 membership is STRICT (p^f > gamma). Use fixed wide bounds so the grid spans
    # rejected points too; with 4 placebos gamma=0.4 (=2/5) is attainable, so some grid
    # points have p == gamma and MUST be excluded.
    pre_denoms = _oracle_pre_denoms()
    res = _invert_sharp_null(
        _ORACLE_TREATED,
        _ORACLE_PLACEBOS,
        _ORACLE_POST,
        pre_denoms,
        0,
        "constant",
        0.4,
        bounds=(-6.0, 10.0),
        n_grid=401,
    )
    grid = res["grid"]
    assert all(in_set == (p > 0.4) for _, p, in_set in grid)  # strict operator
    at_gamma = [in_set for _, p, in_set in grid if np.isclose(p, 0.4)]
    assert at_gamma, "expected the grid to include a p == gamma point"
    assert not any(at_gamma)  # every p == gamma point excluded (strict)


def test_invert_unbounded_when_gamma_below_granularity():
    # p^f >= 1/(J+1) always (the treated ranks itself); gamma below that -> nothing is
    # rejected -> the set is all of R (Firpo & Possebom fn 8).
    pre_denoms = _oracle_pre_denoms()
    res = _invert_sharp_null(
        _ORACLE_TREATED, _ORACLE_PLACEBOS, _ORACLE_POST, pre_denoms, 0, "constant", 0.1, n_grid=10
    )
    assert res["status"] == "unbounded"
    assert res["lower"] == -np.inf and res["upper"] == np.inf


def test_invert_empty_set_when_family_cannot_fit():
    # A constant treated effect cannot be matched by the LINEAR family; with a near-
    # perfect pre-fit (tiny denom) the treated stays the most-deviant unit at every
    # slope, so p == 1/(J+1) everywhere -> rejected at gamma > 1/(J+1) -> empty.
    treated = _gp([1e-9, -1e-9, 1e-9], [2.0, 2.0, 2.0], _ORACLE_PRE, _ORACLE_POST)
    units = {0: treated, **_ORACLE_PLACEBOS}
    pre_denoms = {
        u: _floored_pre_mspe(np.array([gp[p] for p in _ORACLE_PRE]), 1.0) for u, gp in units.items()
    }
    res = _invert_sharp_null(
        treated, _ORACLE_PLACEBOS, _ORACLE_POST, pre_denoms, 0, "linear", 0.25, n_grid=60
    )
    assert res["status"] == "empty"
    assert np.isnan(res["lower"]) and np.isnan(res["upper"])


def test_invert_accepts_tails_when_center_is_rejected():
    # Regression for the centered-bracket bug (reviewer M1): when the treated unit has a
    # WORSE pre-fit than the placebos, its accepted region is in the TAILS, not around the
    # point estimate, and the central band is REJECTED. The exact breakpoint inversion must
    # return the unbounded, non-contiguous (two-tail) set — NOT "empty" (the old bug).
    # Treated post gaps [0, 2] with pre-MSPE 100 (poor fit); 4 placebos post [1, 1] pre-MSPE 1.
    pre, post = [0, 1], [2, 3]
    treated = {0: 10.0, 1: -10.0, 2: 0.0, 3: 2.0}  # pre-MSPE=100 -> D=100; post gaps [0, 2]
    placebos = {
        f"p{k}": {0: 1.0, 1: -1.0, 2: 1.0, 3: 1.0} for k in range(4)
    }  # pre-MSPE=1; post [1,1]
    pden = {
        u: _floored_pre_mspe(np.array([gp[p] for p in pre]), 1.0)
        for u, gp in {0: treated, **placebos}.items()
    }
    assert pden[0] == pytest.approx(100.0) and pden["p0"] == pytest.approx(1.0)
    res = _invert_sharp_null(treated, placebos, post, pden, 0, "constant", 0.25, n_grid=50)
    assert res["status"] == "unbounded"  # was wrongly "empty" under the centered bracket
    assert res["lower"] == -np.inf and res["upper"] == np.inf
    assert res["contiguous"] is False  # central rejected band -> two disjoint accepted tails
    # The point estimate (att = 1) is itself rejected; far-out values are accepted.
    p_center, _, _ = _sharp_null_pvalue(treated, placebos, post, _constant_f_post(1.0, 2), pden, 0)
    p_tail, _, _ = _sharp_null_pvalue(treated, placebos, post, _constant_f_post(50.0, 2), pden, 0)
    assert p_center <= 0.25 < p_tail
    # Under a user-bounded grid the same scenario is grid-limited "ran" but still flagged
    # non-contiguous (accepted at both edges, rejected through the middle).
    res_b = _invert_sharp_null(
        treated, placebos, post, pden, 0, "constant", 0.25, bounds=(-50.0, 50.0), n_grid=201
    )
    assert res_b["status"] == "ran" and res_b["contiguous"] is False


def test_invert_includes_accepted_breakpoint_singleton():
    # Reviewer round-2 (M1/DT1): strict p>gamma membership + tie-counting >= means a placebo
    # that EXACTLY ties the treated at a (tangent) breakpoint can push p above gamma THERE
    # while both neighbouring open intervals are rejected -> an isolated accepted singleton
    # the exact inversion must include (NOT report "empty"). Construct a placebo whose RMSPE
    # ratio touches the treated's only at c=0; 3 others stay strictly below.
    pre, post = [0, 1], [2, 3]
    treated = {0: 1.0, 1: -1.0, 2: 1.0, 3: -1.0}  # post [1,-1], pre-MSPE 1 -> D=1
    placebos = {"tie": {0: 2.0, 1: -2.0, 2: 2.0, 3: -2.0}}  # post [2,-2], pre-MSPE 4: ties at c=0
    for k in range(3):
        placebos[f"lo{k}"] = {0: 2.0, 1: -2.0, 2: 0.5, 3: -0.5}  # always strictly below treated
    pden = {
        u: _floored_pre_mspe(np.array([gp[p] for p in pre]), 1.0)
        for u, gp in {0: treated, **placebos}.items()
    }
    # At c=0 the tie places p at 2/5; just off c=0 only the treated ranks (p=1/5).
    p_at0, _, _ = _sharp_null_pvalue(treated, placebos, post, _constant_f_post(0.0, 2), pden, 0)
    p_near, _, _ = _sharp_null_pvalue(treated, placebos, post, _constant_f_post(0.5, 2), pden, 0)
    assert p_at0 == pytest.approx(0.4) and p_near == pytest.approx(0.2)
    res = _invert_sharp_null(treated, placebos, post, pden, 0, "constant", 0.25, n_grid=20)
    assert res["status"] == "ran"  # NOT "empty" -- the singleton is included
    assert res["lower"] == pytest.approx(0.0) and res["upper"] == pytest.approx(0.0)
    # the returned inspection grid reflects the non-empty singleton (one accepted row, not [])
    assert len(res["grid"]) == 1
    g_param, g_p, g_in = res["grid"][0]
    assert g_param == pytest.approx(0.0) and g_p > 0.25 and g_in is True


def test_linear_f_post_is_one_based_and_czero_equals_constant_zero():
    assert np.allclose(_linear_f_post(1.0, 4), [1.0, 2.0, 3.0, 4.0])  # (t - T0), 1-based
    pre_denoms = _oracle_pre_denoms()
    p_lin0, _, _ = _sharp_null_pvalue(
        _ORACLE_TREATED, _ORACLE_PLACEBOS, _ORACLE_POST, _linear_f_post(0.0, 3), pre_denoms, 0
    )
    p_con0, _, _ = _sharp_null_pvalue(
        _ORACLE_TREATED, _ORACLE_PLACEBOS, _ORACLE_POST, _constant_f_post(0.0, 3), pre_denoms, 0
    )
    assert p_lin0 == pytest.approx(p_con0)


def test_invert_monotone_in_gamma():
    # A larger gamma rejects more -> a narrower (or equal) confidence set.
    pre_denoms = _oracle_pre_denoms()

    def width(g):
        r = _invert_sharp_null(
            _ORACLE_TREATED,
            _ORACLE_PLACEBOS,
            _ORACLE_POST,
            pre_denoms,
            0,
            "constant",
            g,
            n_grid=120,
        )
        return (r["upper"] - r["lower"]) if r["status"] == "ran" else None

    w_lo, w_hi = width(0.25), width(0.45)
    assert w_lo is not None and w_hi is not None
    assert w_lo >= w_hi - 1e-9


# --- end-to-end (real fit): custom V skips the outer search for speed/determinism ---


def _fit_with_placebos(n_donors=6, T=10, T0=6, effect=3.0, seed=0, run_placebo=True):
    df, years, t0 = _make_panel(n_donors=n_donors, T=T, T0=T0, effect=effect, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SyntheticControl(v_method="custom", custom_v=np.ones(T0), seed=seed).fit(
            df, "y", "treated", "unit", "year", post_periods=years[T0:], treated_unit="treated"
        )
        if run_placebo:
            res.in_space_placebo()
    return res, years, t0


def _exact_combo_fit(effect=3.0, T=10, T0=6, n_donors=5):
    """Deterministic panel where the treated is an EXACT convex combo of two donors.

    The donors carry distinct sinusoidal idiosyncrasies, so no single donor is a convex
    combination of the others (placebos fit poorly -> larger pre-denominators), while the
    treated reproduces 0.5*d0 + 0.5*d1 (near-perfect pre-fit -> the smallest denominator).
    The treated is therefore uniquely the most-deviant unit in the tails, so the constant
    confidence set is BOUNDED ("ran") -- the end-to-end analogue of the helper oracle.
    """
    years = list(range(2000, 2000 + T))
    t = np.arange(T, dtype=float)
    donors = {
        j: 10.0 + 2.0 * j + (0.3 + 0.1 * j) * t + (0.5 + 0.3 * j) * np.sin(t)
        for j in range(n_donors)
    }
    treated = (0.5 * donors[0] + 0.5 * donors[1]).copy()
    treated[T0:] += effect
    rows = []
    for j in range(n_donors):
        for i in range(T):
            rows.append({"unit": f"d{j}", "year": years[i], "y": float(donors[j][i]), "treated": 0})
    for i in range(T):
        rows.append(
            {"unit": "treated", "year": years[i], "y": float(treated[i]), "treated": int(i >= T0)}
        )
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SyntheticControl(v_method="custom", custom_v=np.ones(T0), seed=0).fit(
            df, "y", "treated", "unit", "year", post_periods=years[T0:], treated_unit="treated"
        )
        res.in_space_placebo()
    return res


def test_test_sharp_null_zero_equals_placebo_p_value_end_to_end():
    res, _, _ = _fit_with_placebos()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s0 = res.test_sharp_null(0.0)
    assert s0["p_value"] == pytest.approx(res.placebo_p_value)
    assert s0["rmspe_f_treated"] == pytest.approx(res.rmspe_ratio)
    assert s0["n_placebos"] == res.n_placebos


def test_confidence_set_constant_contains_att_and_excludes_zero():
    res = _exact_combo_fit(effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = res.confidence_set(family="constant", gamma=0.25)
    ecs = res.effect_confidence_set
    assert ecs["status"] == "ran"
    assert ecs["lower"] <= res.att <= ecs["upper"]  # point estimate inside the set
    assert not (ecs["lower"] <= 0.0 <= ecs["upper"])  # a real +3 effect -> 0 excluded
    assert list(grid.columns) == ["param", "p_value", "in_set"]
    assert ecs["boundary"] == "strict" and ecs["parameter"] == "c"


def test_test_sharp_null_array_path_matches_scalar_and_validates():
    res, _, _ = _fit_with_placebos()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_scalar = res.test_sharp_null(3.0)
        s_array = res.test_sharp_null(np.array([3.0, 3.0, 3.0, 3.0]))
    assert s_array["p_value"] == pytest.approx(s_scalar["p_value"])
    for bad in (
        np.array([1.0, 2.0]),  # wrong length
        np.array([[1.0, 2.0, 3.0, 4.0]]),  # 2-D
        np.array([np.nan, 1.0, 1.0, 1.0]),  # non-finite
    ):
        with pytest.raises(ValueError):
            res.test_sharp_null(bad)


def test_confidence_set_conf_int_stays_nan():
    res, _, _ = _fit_with_placebos()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.confidence_set(family="constant", gamma=0.25)
    # The analytical fields stay NaN: this is a SEPARATE permutation object.
    assert np.isnan(res.se) and np.isnan(res.p_value) and np.isnan(res.t_stat)
    assert np.isnan(res.conf_int[0]) and np.isnan(res.conf_int[1])
    assert res.effect_confidence_set is not None


def test_confidence_set_to_dict_flattened_and_summary_renders():
    res = _exact_combo_fit(effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.confidence_set(family="constant", gamma=0.25)
    d = res.to_dict()
    assert d["effect_ci_status"] == "ran"
    assert d["effect_ci_family"] == "constant" and d["effect_ci_parameter"] == "c"
    assert np.isnan(d["conf_int_lower"])  # analytical interval still NaN
    assert np.isfinite(d["effect_ci_lower"]) and np.isfinite(d["effect_ci_upper"])
    row = res.to_dataframe()  # stays a single row of scalars
    assert len(row) == 1 and "effect_ci_lower" in row.columns
    assert "Confidence set by test inversion" in res.summary()


def test_confidence_set_linear_runs_and_sets_field():
    res, _, _ = _fit_with_placebos()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = res.confidence_set(family="linear", gamma=0.25)
    ecs = res.effect_confidence_set
    assert ecs["family"] == "linear" and ecs["parameter"] == "c_tilde"
    assert ecs["status"] in ("ran", "empty", "unbounded")
    assert list(grid.columns) == ["param", "p_value", "in_set"]


def test_confidence_set_lazy_runs_in_space_placebo():
    res, _, _ = _fit_with_placebos(run_placebo=False)
    assert res._placebo_gaps is None  # placebo NOT run yet
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.confidence_set(family="constant", gamma=0.25)
    assert res._placebo_gaps is not None  # lazily built
    assert res.effect_confidence_set is not None


def test_confidence_set_n_starts_ignored_with_warning_when_reference_exists():
    res, _, _ = _fit_with_placebos(run_placebo=True)  # reference already built
    with pytest.warns(UserWarning, match="n_starts is ignored"):
        res.confidence_set(family="constant", gamma=0.25, n_starts=3)


def test_get_confidence_set_df_requires_run():
    res, _, _ = _fit_with_placebos()
    with pytest.raises(ValueError, match="No confidence set"):
        res.get_confidence_set_df()


def test_in_space_placebo_rerun_invalidates_confidence_set():
    # CI-review P1: a confidence set is computed against the CURRENT placebo reference set,
    # so an explicit in_space_placebo() rebuild (which _require_placebo_reference even
    # suggests, via n_starts) must INVALIDATE the cached set rather than report a stale one.
    res = _exact_combo_fit(effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.confidence_set(family="constant", gamma=0.25)
    assert res.effect_confidence_set is not None
    native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
    assert native["confidence_set"]["status"] == "ran"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo(n_starts=2)  # rebuild the reference set
    assert res.effect_confidence_set is None
    with pytest.raises(ValueError, match="No confidence set"):
        res.get_confidence_set_df()
    native2 = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
    assert native2["confidence_set"]["status"] == "not_run"


def test_confidence_set_too_few_donors_raises():
    # One donor -> in_space_placebo cannot form a reference set -> CI / test raise.
    df, years, T0 = _make_panel(n_donors=1, T=10, T0=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SyntheticControl(v_method="custom", custom_v=np.ones(T0), seed=0).fit(
            df, "y", "treated", "unit", "year", post_periods=years[T0:], treated_unit="treated"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="reference set"):
            res.confidence_set(family="constant", gamma=0.25)
        with pytest.raises(ValueError, match="reference set"):
            res.test_sharp_null(0.0)


def test_confidence_set_unpickled_raises():
    res, _, _ = _fit_with_placebos()
    restored = pickle.loads(pickle.dumps(res))  # snapshot + placebo gaps dropped
    with pytest.raises(ValueError):
        restored.confidence_set(family="constant", gamma=0.25)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"family": "quadratic"},
        {"gamma": 0.0},
        {"gamma": 1.0},
        {"n_grid": 1},
        {"bounds": (1.0, 1.0)},
        {"bounds": (2.0, 1.0)},
        {"bounds": 5.0},  # scalar -> ValueError, not a bare TypeError from len()
        {"bounds": (1.0,)},  # wrong length
        {"bounds": (np.inf, 1.0)},  # non-finite
    ],
)
def test_confidence_set_input_validation(kwargs):
    res, _, _ = _fit_with_placebos()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            res.confidence_set(**kwargs)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_test_sharp_null_gamma_validation(bad):
    res, _, _ = _fit_with_placebos()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            res.test_sharp_null(0.0, gamma=bad)


@pytest.mark.slow
def test_confidence_set_coverage_simulation():
    # Behavioral coverage check: under a constant-effect DGP the (1 - gamma) confidence
    # set should cover the true effect at roughly 1 - gamma. A looser inner tolerance keeps
    # the refits converging cleanly; reps with ANY dropped placebo (n_failed > 0) are
    # EXCLUDED from the coverage count so dropped placebos cannot bias it (M5), and we
    # assert that the large majority of reps are clean (the settings are adequate).
    # J = 9 -> attainable p in multiples of 1/10, gamma = 0.1.
    gamma = 0.1
    c_true = 2.0
    reps = 100
    clean = 0
    covered = 0
    for s in range(reps):
        df, years, T0 = _make_panel(n_donors=9, T=10, T0=6, effect=c_true, seed=1000 + s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SyntheticControl(
                v_method="custom", custom_v=np.ones(T0), seed=s, inner_min_decrease=1e-3
            ).fit(
                df, "y", "treated", "unit", "year", post_periods=years[T0:], treated_unit="treated"
            )
            res.in_space_placebo()
            if res.n_failed != 0:
                continue  # exclude biased reps from the coverage count (M5)
            clean += 1
            res.confidence_set(family="constant", gamma=gamma)
        ecs = res.effect_confidence_set
        if ecs["status"] == "unbounded":
            covered += 1  # an unbounded set trivially covers the truth
        elif ecs["status"] == "ran" and ecs["lower"] <= c_true <= ecs["upper"]:
            covered += 1
    assert clean >= 0.8 * reps, f"only {clean}/{reps} reps converged cleanly"
    coverage = covered / clean
    # Permutation inference is finite-sample valid under exchangeability; allow a wide
    # band (the convex-combo treated is not perfectly exchangeable with single donors).
    assert coverage >= 0.70, f"coverage {coverage:.3f} too low (target ~{1 - gamma})"


# ===========================================================================
# CWZ conformal inference — pure-function oracle tests (Chernozhukov, Wüthrich
# & Zhu 2021). These exercise diff_diff/conformal.py primitives directly (no
# SyntheticControl fit), with hand-computed oracles + an independent brute-force
# permutation re-implementation. See chernozhukov-wuthrich-zhu-2021-review.md.
# ===========================================================================


def _bruteforce_Sq(u, post_mask, q):
    """Independent re-implementation of S_q(u) for oracle cross-checks (CWZ §2.2)."""
    post = [abs(v) for v, m in zip(u, post_mask) if m]
    tstar = len(post)
    if tstar == 0:
        return float("nan")
    if q == float("inf"):
        return max(post)
    return (sum(p**q for p in post) / (tstar**0.5)) ** (1.0 / q)


def test_cwz_statistic_matches_hand_computed_values():
    # post window = last two slots, values 5 and 6; T*=2.
    u = np.array([1.0, -2.0, 3.0, -4.0, 5.0, 6.0])
    post_mask = np.array([False, False, False, False, True, True])
    assert _cwz_statistic(u, post_mask, 1) == pytest.approx(11.0 / np.sqrt(2))
    assert _cwz_statistic(u, post_mask, 2) == pytest.approx(np.sqrt((25.0 + 36.0) / np.sqrt(2)))
    assert _cwz_statistic(u, post_mask, float("inf")) == pytest.approx(6.0)
    # empty post window -> NaN
    assert np.isnan(_cwz_statistic(u, np.zeros(6, dtype=bool), 1))


def test_moving_block_perms_are_cyclic_shifts():
    m = 5
    perms = _moving_block_perms(m)
    assert perms.shape == (m, m)
    # row 0 = identity
    assert np.array_equal(perms[0], np.arange(m))
    # each row is a valid permutation and the documented cyclic shift (i+j) mod m
    for j in range(m):
        assert sorted(perms[j].tolist()) == list(range(m))
        assert np.array_equal(perms[j], (np.arange(m) + j) % m)
    # applying a shift reads the residual vector wrapped around
    u = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert np.array_equal(u[perms[1]], np.array([20.0, 30.0, 40.0, 50.0, 10.0]))


def test_iid_perms_exact_enumeration_and_random_sampling():
    rng = np.random.default_rng(0)
    # m! <= n_draws -> exact enumeration (4! = 24), identity present, all unique perms
    exact = _iid_perms(4, n_draws=100, rng=rng)
    assert exact.shape == (24, 4)
    assert np.array_equal(exact[0], np.arange(4))
    rows = {tuple(r) for r in exact.tolist()}
    assert len(rows) == 24
    for r in exact:
        assert sorted(r.tolist()) == [0, 1, 2, 3]
    # m! > n_draws -> random sampling, identity prepended, every row a valid permutation
    sampled = _iid_perms(10, n_draws=50, rng=rng)
    assert sampled.shape == (50, 10)
    assert np.array_equal(sampled[0], np.arange(10))
    for r in sampled:
        assert sorted(r.tolist()) == list(range(10))


def test_cwz_pvalue_brute_force_equivalence_moving_block():
    # Independent brute-force permutation p-value must equal the production helper
    # bit-for-bit (catches the calendar-order / fixed-post-mask bug class).
    rng = np.random.default_rng(7)
    T, n_post = 24, 3
    u = rng.normal(size=T)
    post_mask = np.zeros(T, dtype=bool)
    post_mask[-n_post:] = True
    for q in (1, 2, float("inf")):
        perms = _moving_block_perms(T)
        p, s_obs, n = _cwz_pvalue(u, post_mask, perms, q)
        assert n == T
        # brute force: recompute S from scratch for every shift, count >=
        s_ref = _bruteforce_Sq(u.tolist(), post_mask.tolist(), q)
        cnt = 0
        for j in range(T):
            up = u[perms[j]]
            cnt += _bruteforce_Sq(up.tolist(), post_mask.tolist(), q) >= s_ref - 1e-12
        assert p == pytest.approx(cnt / T)
        assert s_obs == pytest.approx(s_ref)
        # identity is in Pi -> p >= 1/|Pi|
        assert p >= 1.0 / T - 1e-12


def test_cwz_pvalue_iid_includes_identity_floor():
    rng = np.random.default_rng(3)
    T = 8
    u = rng.normal(size=T)
    post_mask = np.zeros(T, dtype=bool)
    post_mask[-1] = True
    perms = _iid_perms(T, n_draws=500, rng=rng)
    p, _, n = _cwz_pvalue(u, post_mask, perms, 1)
    assert p >= 1.0 / n - 1e-12


def test_cwz_proxy_fit_matches_scipy_simplex_ls():
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(1)
    T, J = 30, 4
    Y0 = rng.normal(size=(T, J))
    y1 = Y0 @ np.array([0.5, 0.3, 0.2, 0.0]) + rng.normal(scale=0.05, size=T)
    # min_decrease is the ABSOLUTE tolerance; the caller scales by a theta0-invariant
    # outcome norm (mirrors how the Results methods pass it). A generous max_iter lets
    # the Frank-Wolfe simplex solve grind its documented slow tail to convergence so the
    # flag is True for this exact-comparison oracle (production uses warm-starts).
    md = 1e-6 * float(np.linalg.norm(y1))
    w, resid, conv = _cwz_proxy_fit(y1, Y0, max_iter=200000, min_decrease=md)
    assert conv
    # simplex constraint delivered by the solver (w >= 0, sum w = 1) — no extra normalization
    assert w.min() >= -1e-9
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(resid, y1 - Y0 @ w)

    def obj(v):
        r = y1 - Y0 @ v
        return float(r @ r)

    cons = [{"type": "eq", "fun": lambda v: v.sum() - 1.0}]
    best = None
    for _ in range(8):
        res = scipy_opt.minimize(
            obj,
            rng.dirichlet(np.ones(J)),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * J,
            constraints=cons,
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if best is None or res.fun < best.fun:
            best = res
    # FW reaches essentially the same convex optimum as SLSQP (the slow tail leaves a
    # tiny residual gap, well under the inference-relevant scale).
    assert obj(w) == pytest.approx(obj(best.x), abs=1e-3)
    assert np.allclose(w, best.x, atol=1e-2)


def test_cwz_proxy_fit_single_donor_is_degenerate():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    Y0 = np.array([[0.5], [1.0], [1.5], [2.0]])
    w, resid, conv = _cwz_proxy_fit(y1, Y0, max_iter=100, min_decrease=1e-6)
    assert conv
    assert np.array_equal(w, np.array([1.0]))
    assert np.allclose(resid, y1 - Y0[:, 0])


def test_block_collapse_averages_and_drops_leftover_pre_periods():
    # T=9, n_pre=6, n_post=3 -> drop=0; pre blocks [0,1,2],[3,4,5]; post [6,7,8]
    y1 = np.arange(9, dtype=float)
    Y0 = np.column_stack([np.arange(9, dtype=float), np.arange(9, dtype=float) * 2.0])
    y1b, Y0b, drop = _block_collapse(y1, Y0, n_pre=6, n_post=3)
    assert drop == 0
    assert np.allclose(y1b, [1.0, 4.0, 7.0])  # block means
    assert np.allclose(Y0b[:, 0], [1.0, 4.0, 7.0])
    assert np.allclose(Y0b[:, 1], [2.0, 8.0, 14.0])
    # T=10, n_pre=7, n_post=3 -> drop earliest 1 pre-period; pre blocks [1,2,3],[4,5,6]; post [7,8,9]
    y1 = np.arange(10, dtype=float)
    Y0 = np.arange(10, dtype=float).reshape(-1, 1)
    y1b, Y0b, drop = _block_collapse(y1, Y0, n_pre=7, n_post=3)
    assert drop == 1
    assert np.allclose(y1b, [2.0, 5.0, 8.0])


# ===========================================================================
# CWZ conformal inference — conformal_test (joint sharp-null) integration tests
# ===========================================================================


def _fit_for_conformal(n_donors=5, T=18, T0=14, effect=4.0, seed=2):
    df, years, T0 = _make_panel(n_donors=n_donors, T=T, T0=T0, effect=effect, seed=seed)
    res = SyntheticControl(**_FAST).fit(
        df, "y", "treated", "unit", "year", post_periods=years[T0:], treated_unit="treated"
    )
    return res


def test_conformal_test_p_value_shape_and_moving_block_size():
    res = _fit_for_conformal()
    s = res.conformal_test(res.att, scheme="moving_block")
    assert set(s.index) >= {"p_value", "S_observed", "q", "scheme", "n_perms", "n_post"}
    assert 0.0 < s["p_value"] <= 1.0
    assert s["p_value"] >= 1.0 / s["n_perms"] - 1e-12  # identity is in Pi
    assert int(s["n_perms"]) == res.n_pre_periods + res.n_post_periods  # |Pi_->| = T
    # near the point estimate the null is not rejected; far from it, p is small
    p_near = res.conformal_test(res.att)["p_value"]
    p_far = res.conformal_test(res.att + 1000.0)["p_value"]
    assert p_near > p_far
    assert p_far == pytest.approx(1.0 / int(s["n_perms"]))


def test_conformal_test_detects_strong_effect_with_iid():
    res = _fit_for_conformal(T=24, T0=18, effect=10.0)
    # H0: no effect, with a strong true effect -> small permutation p (iid is sharper)
    p0 = res.conformal_test(0.0, scheme="iid", n_iid=2000, seed=0)["p_value"]
    p_true = res.conformal_test(res.att, scheme="iid", n_iid=2000, seed=0)["p_value"]
    assert p0 < 0.1
    assert p_true > p0


def test_conformal_test_keeps_analytical_inference_nan():
    res = _fit_for_conformal()
    res.conformal_test(0.0)
    assert_nan_inference(
        {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
    )
    assert not res.is_significant  # bound to the NaN analytical p_value, not the conformal one
    assert res.conformal_inference["kind"] == "joint"


def test_conformal_test_q_variants_run_and_inf_is_sup():
    res = _fit_for_conformal()
    for q in (1, 2, float("inf"), "inf"):
        s = res.conformal_test(0.0, q=q)
        assert 0.0 < s["p_value"] <= 1.0
    # q=inf statistic is the sup of |residuals| over the post window
    s_inf = res.conformal_test(0.0, q="inf")
    assert s_inf["q"] == float("inf")


def test_conformal_test_iid_reproducible_for_fixed_seed():
    res = _fit_for_conformal()
    a = res.conformal_test(0.0, scheme="iid", n_iid=1000, seed=7)["p_value"]
    b = res.conformal_test(0.0, scheme="iid", n_iid=1000, seed=7)["p_value"]
    assert a == b


def test_conformal_test_validation_errors():
    res = _fit_for_conformal()
    with pytest.raises(ValueError, match="q must be"):
        res.conformal_test(0.0, q=3)
    with pytest.raises(ValueError, match="scheme must be"):
        res.conformal_test(0.0, scheme="bogus")
    with pytest.raises(ValueError, match="n_iid"):
        res.conformal_test(0.0, scheme="iid", n_iid=0)
    with pytest.raises(ValueError, match="post-treatment periods|effect path"):
        res.conformal_test(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))  # wrong length


def test_conformal_test_unpickled_fails_closed():
    res = _fit_for_conformal()
    restored = pickle.loads(pickle.dumps(res))
    with pytest.raises(ValueError, match="fit snapshot"):
        restored.conformal_test(0.0)


def test_conformal_test_single_donor_warns_degenerate():
    df, years, T0 = _make_panel(n_donors=1, T=12, T0=9, effect=2.0, seed=1)
    res = SyntheticControl(**_FAST).fit(
        df, "y", "treated", "unit", "year", post_periods=years[T0:], treated_unit="treated"
    )
    with pytest.warns(UserWarning, match="single donor"):
        s = res.conformal_test(0.0)
    assert 0.0 < s["p_value"] <= 1.0


def test_conformal_test_warns_when_post_not_short_relative_to_pre():
    # T* >= T0 -> validity caveat warning
    df, years, T0 = _make_panel(n_donors=4, T=8, T0=3, effect=2.0, seed=5)
    res = SyntheticControl(**_FAST).fit(
        df, "y", "treated", "unit", "year", post_periods=years[T0:], treated_unit="treated"
    )
    with pytest.warns(UserWarning, match="large pre-period"):
        res.conformal_test(0.0)


# ===========================================================================
# CWZ conformal inference — conformal_average_effect (block-collapse CI) tests
# ===========================================================================


def test_conformal_average_effect_basic_iid():
    res = _fit_for_conformal(n_donors=5, T=24, T0=20, effect=5.0, seed=3)  # T0=20, T*=4 -> 6 blocks
    s = res.conformal_average_effect(alpha=0.1, scheme="iid", n_iid=2000, seed=0)
    assert set(s.index) >= {"lower", "upper", "point_estimate", "status", "n_blocks"}
    assert int(s["n_blocks"]) == 6
    assert np.isfinite(s["point_estimate"])
    assert s["status"] in {"ran", "grid_limited", "empty"}
    if s["status"] != "empty":
        assert s["lower"] <= s["upper"]
    # separate permutation object — analytical inference stays NaN
    assert np.isnan(res.conf_int[0]) and np.isnan(res.p_value)
    assert res.conformal_inference["kind"] == "average"
    assert res.get_conformal_grid_df().shape[0] == 200


def test_conformal_average_effect_drops_leftover_pre_periods_with_warning():
    res = _fit_for_conformal(n_donors=4, T=23, T0=19, effect=4.0, seed=4)  # T0=19, T*=4 -> drop 3
    with pytest.warns(UserWarning, match="not a multiple of"):
        s = res.conformal_average_effect(alpha=0.2, scheme="iid", n_iid=1000, seed=0)
    assert int(s["n_dropped_pre"]) == 3
    assert int(s["n_blocks"]) == 5  # (19-3)//4 pre-blocks + 1 post-block


def test_conformal_average_effect_moving_block_granularity_warning():
    res = _fit_for_conformal(n_donors=4, T=24, T0=20, effect=4.0, seed=6)  # 6 blocks
    # moving-block: |Pi| = 6 blocks -> granularity 1/6 = 0.167 > alpha 0.1 -> unbounded
    with pytest.warns(UserWarning, match="granularity"):
        res.conformal_average_effect(alpha=0.1, scheme="moving_block")


def test_conformal_average_effect_validation_errors():
    res = _fit_for_conformal()
    with pytest.raises(ValueError, match="alpha must be"):
        res.conformal_average_effect(alpha=0.0)
    with pytest.raises(ValueError, match="scheme must be"):
        res.conformal_average_effect(scheme="x")
    with pytest.raises(ValueError, match="n_grid"):
        res.conformal_average_effect(n_grid=1)
    with pytest.raises(ValueError, match="bounds must"):
        res.conformal_average_effect(bounds=(5.0, 1.0))


def test_conformal_average_effect_requires_T0_at_least_Tstar():
    # T0 < T* -> cannot form a full pre-block
    df, years, T0 = _make_panel(n_donors=4, T=9, T0=3, effect=2.0, seed=7)  # T0=3 < T*=6
    res = SyntheticControl(**_FAST).fit(
        df, "y", "treated", "unit", "year", post_periods=years[T0:], treated_unit="treated"
    )
    with pytest.raises(ValueError, match="T0 >= T\\*"):
        res.conformal_average_effect()


def test_conformal_average_effect_explicit_bounds_grid_stays_within():
    res = _fit_for_conformal(n_donors=5, T=24, T0=20, effect=5.0, seed=3)
    res.conformal_average_effect(scheme="iid", n_iid=500, seed=0, bounds=(-2.0, 12.0), n_grid=50)
    grid = res.get_conformal_grid_df()
    assert grid["param"].min() >= -2.0 - 1e-9
    assert grid["param"].max() <= 12.0 + 1e-9
    assert len(grid) == 50


def test_conformal_average_effect_get_grid_before_run_raises():
    res = _fit_for_conformal()
    with pytest.raises(ValueError, match="No conformal inversion grid"):
        res.get_conformal_grid_df()


@pytest.mark.slow
def test_conformal_average_effect_coverage_simulation():
    # The (1-alpha) average-effect CI should cover a known constant effect at ~ 1-alpha.
    alpha = 0.2
    c_true = 3.0
    reps, covered, clean = 60, 0, 0
    for rep in range(reps):
        df, years, T0 = _make_panel(n_donors=5, T=20, T0=16, effect=c_true, seed=1000 + rep)
        res = SyntheticControl(**_FAST).fit(
            df, "y", "treated", "unit", "year", post_periods=years[T0:], treated_unit="treated"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = res.conformal_average_effect(alpha=alpha, scheme="iid", n_iid=400, seed=rep)
        clean += 1
        if s["status"] in {"grid_limited"}:
            covered += 1  # touches the grid edge -> conservatively counts as covering
        elif s["status"] == "ran" and s["lower"] <= c_true <= s["upper"]:
            covered += 1
    coverage = covered / clean
    assert coverage >= 0.70, f"coverage {coverage:.3f} below target ~{1 - alpha}"


# ===========================================================================
# CWZ conformal inference — conformal_confidence_intervals (pointwise) tests
# ===========================================================================


def test_conformal_confidence_intervals_shape_and_columns():
    res = _fit_for_conformal(n_donors=6, T=22, T0=16, effect=6.0, seed=4)
    ci = res.conformal_confidence_intervals(alpha=0.1, scheme="iid", n_iid=1000, seed=0)
    assert list(ci["period"]) == list(res.post_periods)
    assert set(ci.columns) >= {
        "period",
        "lower",
        "upper",
        "point_estimate",
        "status",
        "contiguous",
        "n_grid_in_set",
        "n_grid_nonconverged",
    }
    ok = ci["status"] != "empty"
    assert (ci.loc[ok, "lower"] <= ci.loc[ok, "upper"]).all()
    assert np.isnan(res.conf_int[0])  # analytical CI untouched
    assert res.conformal_inference["kind"] == "pointwise"
    grid = res.get_conformal_grid_df()
    assert "period" in grid.columns
    assert len(grid) == len(res.post_periods) * 100


def test_conformal_confidence_intervals_recover_true_constant_effect():
    # The pointwise CIs should bracket the known true effect for (almost) every period.
    c_true = 6.0
    res = _fit_for_conformal(n_donors=6, T=22, T0=16, effect=c_true, seed=4)
    ci = res.conformal_confidence_intervals(alpha=0.1, scheme="iid", n_iid=1500, seed=1)
    brackets = ((ci["lower"] <= c_true) & (c_true <= ci["upper"])).sum()
    assert brackets >= len(ci) - 1  # allow at most one period to miss from noise


def test_conformal_confidence_intervals_moving_block_is_usable():
    # moving-block granularity = 1/(T0+1); with T0=16 that is 1/17 < alpha=0.1 -> usable
    res = _fit_for_conformal(n_donors=6, T=22, T0=16, effect=6.0, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = res.conformal_confidence_intervals(alpha=0.1, scheme="moving_block")
    assert (ci["status"] == "ran").any()


def test_conformal_confidence_intervals_validation_and_pickle():
    res = _fit_for_conformal()
    with pytest.raises(ValueError, match="alpha must be"):
        res.conformal_confidence_intervals(alpha=1.5)
    with pytest.raises(ValueError, match="scheme must be"):
        res.conformal_confidence_intervals(scheme="x")
    with pytest.raises(ValueError, match="n_iid"):
        res.conformal_confidence_intervals(scheme="iid", n_iid=0)
    with pytest.raises(ValueError, match="n_grid"):
        res.conformal_confidence_intervals(n_grid=1)
    with pytest.raises(ValueError, match="bounds must"):
        res.conformal_confidence_intervals(bounds=(1.0,))
    restored = pickle.loads(pickle.dumps(res))
    with pytest.raises(ValueError, match="fit snapshot"):
        restored.conformal_confidence_intervals()


def test_conformal_confidence_intervals_explicit_bounds_applied_per_period():
    res = _fit_for_conformal(n_donors=6, T=22, T0=16, effect=6.0, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.conformal_confidence_intervals(
            scheme="iid", n_iid=500, seed=0, bounds=(0.0, 12.0), n_grid=40
        )
    grid = res.get_conformal_grid_df()
    assert grid["param"].min() >= -1e-9
    assert grid["param"].max() <= 12.0 + 1e-9
    # every period scanned the same fixed grid -> equal point counts
    assert (grid.groupby("period").size() == 40).all()


# ---- CWZ conformal: fail-closed edge cases (non-converged, unbounded, accessor) ----


def test_conformal_ci_nonconverged_points_are_indeterminate_not_rejected(monkeypatch):
    # A non-converged grid point must be treated as indeterminate (kept in the set),
    # NOT rejected — excluding it would understate the interval width.
    import diff_diff.conformal as cf

    res = _fit_for_conformal(n_donors=6, T=22, T0=16, effect=6.0, seed=4)
    real = cf._cwz_proxy_fit

    def never_converge(*a, **k):
        w, resid, _ = real(*a, **k)
        return w, resid, False

    monkeypatch.setattr(cf, "_cwz_proxy_fit", never_converge)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = res.conformal_confidence_intervals(
            alpha=0.1, scheme="iid", n_iid=400, seed=0, n_grid=40
        )
    # every grid point indeterminate -> none rejected -> hull spans the grid, never "empty"
    assert (ci["status"] != "empty").all()
    assert (ci["n_grid_nonconverged"] == 40).all()
    assert (ci["n_grid_in_set"] == 40).all()
    grid = res.get_conformal_grid_df()
    assert (~grid["converged"]).all()
    assert grid["in_set"].all()


def test_conformal_average_effect_unbounded_below_granularity():
    # moving-block on 6 blocks -> |Pi|=6, 1/6 ~= 0.167 > alpha=0.1 -> unbounded set
    res = _fit_for_conformal(n_donors=4, T=24, T0=20, effect=4.0, seed=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = res.conformal_average_effect(alpha=0.1, scheme="moving_block")
    assert s["status"] == "unbounded"
    assert s["lower"] == -np.inf and s["upper"] == np.inf
    assert res.conformal_inference["status"] == "unbounded"


def test_conformal_confidence_intervals_unbounded_below_granularity():
    # moving-block pointwise -> |Pi|=T0+1=17, alpha=0.01 < 1/17 -> every period unbounded
    res = _fit_for_conformal(n_donors=6, T=22, T0=16, effect=6.0, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = res.conformal_confidence_intervals(alpha=0.01, scheme="moving_block")
    assert (ci["status"] == "unbounded").all()
    assert (ci["lower"] == -np.inf).all() and (ci["upper"] == np.inf).all()
    assert res.conformal_inference["n_unbounded"] == len(ci)


def test_get_conformal_ci_df_accessor():
    res = _fit_for_conformal()
    with pytest.raises(ValueError, match="No pointwise conformal CIs"):
        res.get_conformal_ci_df()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.conformal_confidence_intervals(scheme="iid", n_iid=300, seed=0)
    df = res.get_conformal_ci_df()
    assert "period" in df.columns and len(df) == res.n_post_periods


def test_conformal_average_effect_warns_when_post_not_short_relative_to_pre():
    # T0 == T* (n_pre == n_post) -> large-T0 validity caveat warning
    df, years, T0 = _make_panel(n_donors=5, T=8, T0=4, effect=3.0, seed=8)  # 4 pre, 4 post
    res = SyntheticControl(**_FAST).fit(
        df, "y", "treated", "unit", "year", post_periods=years[T0:], treated_unit="treated"
    )
    with pytest.warns(UserWarning, match="large pre-period"):
        res.conformal_average_effect(alpha=0.2, scheme="iid", n_iid=200, seed=0)


def test_conformal_confidence_intervals_warn_small_pre_period():
    # T0 == 1 -> each pointwise sub-series has a 1-period pre window -> validity caveat
    df, years, T0 = _make_panel(n_donors=4, T=5, T0=1, effect=2.0, seed=9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SyntheticControl(**_FAST).fit(
            df, "y", "treated", "unit", "year", post_periods=years[T0:], treated_unit="treated"
        )
    with pytest.warns(UserWarning, match="large pre-period"):
        res.conformal_confidence_intervals(alpha=0.3, scheme="iid", n_iid=200, seed=0)


# ---------------------------------------------------------------------------
# ADH-2015 §4 tail diagnostics: regression-weight extrapolation + sparse-SC
# subset search. Both are opt-in, hold the analytical inference contract (NaN),
# and re-use the FIXED baseline V captured on the fit snapshot.
# ---------------------------------------------------------------------------


def _fit_iv(df, **kw):
    """Fast inverse-variance SC fit (closed-form V, no slow outer search)."""
    return SyntheticControl(v_method="inverse_variance", seed=0, **_FAST, **kw).fit(
        df, "y", "treated", "unit", "year"
    )


def test_regression_weights_matches_paper_formula_oracle():
    # k = T0 = 3 predictors, J = 6 donors -> k+1 = 4 <= J so W^reg is full row rank.
    df, _years, _T0 = _make_panel(n_donors=6, T=8, T0=3, seed=1)
    res = _fit_iv(df)
    assert res._fit_converged
    snap = res._fit_snapshot
    X1s, X0s = snap.fit_X1s, snap.fit_X0s
    _k, J = X0s.shape
    X0a = np.vstack([np.ones((1, J)), X0s])
    X1a = np.concatenate([[1.0], X1s.ravel()])
    # ADH's exact formula W^reg = X0a'(X0a X0a')^{-1} X1a (independent of the impl's lstsq).
    w_expected = X0a.T @ np.linalg.solve(X0a @ X0a.T, X1a)
    tab = res.regression_weights().set_index("donor_id")
    got = np.array([tab.loc[d, "w_reg"] for d in snap.donor_ids])
    np.testing.assert_allclose(got, w_expected, atol=1e-10)
    # Full row rank -> intercept forces the weights to sum to 1.
    assert res._regw_rank_deficient is False
    assert res._regw_status == "ran"
    assert abs(res._regw_weight_sum - 1.0) < 1e-8
    # Flag columns are internally consistent with w_reg.
    full = res.get_regression_weights_df()
    for _, row in full.iterrows():
        w = row["w_reg"]
        assert row["extrapolates"] == bool(w < 0.0 or w > 1.0)
        assert row["abs_extrapolation"] == pytest.approx(max(0.0, -w, w - 1.0))
    assert res._regw_n_extrapolating == int(full["extrapolates"].sum())
    # The FULL analytical inference contract is untouched by the diagnostic (all NaN; the
    # permutation-only significance stays off the analytical fields).
    res.sparse_synthetic_control(sizes=[2])
    assert np.isnan(res.se) and np.isnan(res.p_value) and np.isnan(res.t_stat)
    assert np.isnan(res.conf_int[0]) and np.isnan(res.conf_int[1])
    assert not res.is_significant


def test_regression_weights_invariant_to_predictor_row_scaling():
    # At FULL ROW RANK (k=T0=3 predictors < J=6 donors, exact fit) W^reg is invariant to
    # per-predictor row scaling: a custom (standardized) fit and an inverse_variance (raw) fit
    # on the same data give identical implied regression weights. (The invariance holds only
    # under full row rank — see test_regression_weights_rank_deficient_warns_and_min_norm and
    # the REGISTRY note; in the rank-deficient min-norm case row scaling can change W^reg.)
    df, _years, T0 = _make_panel(n_donors=6, T=8, T0=3, seed=2)
    r_std = SyntheticControl(v_method="custom", custom_v=np.ones(T0), seed=0, **_FAST).fit(
        df, "y", "treated", "unit", "year"
    )
    r_raw = _fit_iv(df)
    t_std = r_std.regression_weights().set_index("donor_id")["w_reg"]
    t_raw = r_raw.regression_weights().set_index("donor_id")["w_reg"]
    assert not r_std._regw_rank_deficient and not r_raw._regw_rank_deficient  # full-rank regime
    order = sorted(t_std.index)
    np.testing.assert_allclose(
        t_std.reindex(order).to_numpy(), t_raw.reindex(order).to_numpy(), atol=1e-10
    )


def test_regression_weights_rank_deficient_warns_and_min_norm():
    # k = T0 = 6 predictors, J = 3 donors -> k+1 = 7 > J: not full row rank.
    df, _years, _T0 = _make_panel(n_donors=3, T=10, T0=6, seed=1)
    res = _fit_iv(df)
    with pytest.warns(UserWarning, match="not full row rank"):
        tab = res.regression_weights()
    assert res._regw_rank_deficient is True
    assert res._regw_status == "ran"
    assert len(tab) == 3  # all donors still reported


def test_regression_weights_fail_closed_on_unpickled():
    df, _years, _T0 = _make_panel(n_donors=5, T=8, T0=4, seed=1)
    res = _fit_iv(df)
    res2 = pickle.loads(pickle.dumps(res))
    with pytest.raises(ValueError, match="fit snapshot"):
        res2.regression_weights()


def test_regression_weights_too_few_donors():
    df, _years, _T0 = _make_panel(n_donors=1, T=8, T0=4, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SyntheticControl(seed=0, **_FAST).fit(df, "y", "treated", "unit", "year")
    with pytest.warns(UserWarning, match="at least 2 donors"):
        tab = res.regression_weights()
    assert res._regw_status == "too_few_donors"
    assert tab.empty


def test_sparse_self_consistency_and_holds_v_fixed():
    from diff_diff.synthetic_control import _inner_solve_W

    df, _years, _T0 = _make_panel(n_donors=6, T=8, T0=4, seed=1)
    res = _fit_iv(df)
    snap = res._fit_snapshot
    tab = res.sparse_synthetic_control(sizes=[2])
    row = tab[tab["status"] == "ran"].iloc[0]
    won_ids = row["donor_ids"]
    won_weights = row["weights"]

    # Independent brute-force best size-2 subset, using the FIXED baseline V (snap.fit_v) —
    # if the diagnostic re-searched V instead of holding it fixed, this would diverge.
    import itertools

    Y = snap.pivots[snap.outcome]
    Z1_pre = Y.loc[snap.pre_periods, snap.treated_id].to_numpy(float)
    Z0_pre = Y.loc[snap.pre_periods, snap.donor_ids].to_numpy(float)
    best, best_mspe = None, np.inf
    for cols in itertools.combinations(range(len(snap.donor_ids)), 2):
        w, conv = _inner_solve_W(
            snap.fit_X1s,
            snap.fit_X0s[:, list(cols)],
            snap.fit_v,
            snap.inner_max_iter,
            snap.inner_min_decrease,
        )
        if not conv:
            continue
        m = float(np.mean((Z1_pre - Z0_pre[:, list(cols)] @ w) ** 2))
        if m < best_mspe:
            best_mspe, best = m, (cols, w)
    exp_ids = tuple(snap.donor_ids[c] for c in best[0])
    assert won_ids == exp_ids
    got_w = np.array([won_weights[i] for i in won_ids])
    np.testing.assert_allclose(got_w, best[1], atol=1e-10)
    # pre_rmspe reported equals sqrt of the winning subset's pre-MSPE.
    assert row["pre_rmspe"] == pytest.approx(np.sqrt(best_mspe), abs=1e-10)


def test_sparse_l1_picks_best_single_donor():
    df, _years, _T0 = _make_panel(n_donors=5, T=8, T0=4, seed=3)
    res = _fit_iv(df)
    snap = res._fit_snapshot
    tab = res.sparse_synthetic_control(sizes=[1])
    won = tab[tab["status"] == "ran"].iloc[0]["donor_ids"]
    assert len(won) == 1
    # l=1 forces w=[1], so the synthetic IS that donor's series; the winner is the donor
    # whose own pre-period outcomes best match the treated unit's.
    Y = snap.pivots[snap.outcome]
    Z1 = Y.loc[snap.pre_periods, snap.treated_id].to_numpy(float)
    mspes = {
        d: float(np.mean((Z1 - Y.loc[snap.pre_periods, d].to_numpy(float)) ** 2))
        for d in snap.donor_ids
    }
    assert won[0] == min(mspes, key=mspes.get)


def test_sparse_explicit_oversize_raises():
    df, _years, _T0 = _make_panel(n_donors=6, T=8, T0=4, seed=1)
    res = _fit_iv(df)
    with pytest.raises(ValueError, match="exceeding max_subsets"):
        res.sparse_synthetic_control(sizes=3, max_subsets=5)  # C(6,3)=20 > 5


def test_sparse_default_skips_over_cap_without_raising():
    # J=8: C(8,1)=8 (ok), C(8,2)=28 and C(8,3)=56 (> cap 10) -> the two large defaults skip.
    df, _years, _T0 = _make_panel(n_donors=8, T=8, T0=4, seed=1)
    res = _fit_iv(df)
    with pytest.warns(UserWarning, match="skipping default size"):
        tab = res.sparse_synthetic_control(max_subsets=10)
    assert set(tab[tab["status"] == "ran"]["size"]) == {1}
    assert res._sparse_status == "ran"  # skipped, NOT raised


def test_sparse_baseline_row_is_exact():
    df, _years, _T0 = _make_panel(n_donors=5, T=8, T0=4, seed=1)
    res = _fit_iv(df)
    tab = res.sparse_synthetic_control(sizes=[2])
    base = tab[tab["status"] == "baseline"].iloc[0]
    assert base["delta_att"] == 0.0
    assert base["att"] == pytest.approx(res.att, abs=1e-12)
    assert base["size"] == len(res._fit_snapshot.weighted_donor_ids)


def test_sparse_fail_closed_on_unpickled():
    df, _years, _T0 = _make_panel(n_donors=5, T=8, T0=4, seed=1)
    res = _fit_iv(df)
    res2 = pickle.loads(pickle.dumps(res))
    with pytest.raises(ValueError, match="fit snapshot"):
        res2.sparse_synthetic_control()
    # The summary table survives, but the panel-derived gap accessor fails closed.
    res.sparse_synthetic_control(sizes=[2])
    res3 = pickle.loads(pickle.dumps(res))
    assert res3.get_sparse_synthetic_control_df() is not None  # small table survives
    with pytest.raises(ValueError, match="not retained after pickling"):
        res3.get_sparse_synthetic_control_gaps()


def test_sparse_too_few_donors():
    df, _years, _T0 = _make_panel(n_donors=1, T=8, T0=4, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SyntheticControl(seed=0, **_FAST).fit(df, "y", "treated", "unit", "year")
    with pytest.warns(UserWarning, match="at least 2 donors"):
        tab = res.sparse_synthetic_control()
    assert res._sparse_status == "too_few_donors"
    assert list(tab["status"]) == ["baseline"]


def test_adh_tail_diagnostics_surface_in_diagnostic_report():
    df, _years, T = _make_panel(n_donors=5, T=8, T0=4, seed=1)
    res = _fit_iv(df)
    # not_run stubs before the opt-in methods are called.
    nat0 = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
    assert nat0["regression_weights"]["status"] == "not_run"
    assert nat0["sparse_synthetic_control"]["status"] == "not_run"
    res.regression_weights()
    res.sparse_synthetic_control(sizes=[2, 3])
    gaps = res.get_sparse_synthetic_control_gaps()
    assert set(gaps["size"]) == {2, 3}
    assert len(gaps) == 2 * 8  # 2 sizes x T periods
    nat = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
    assert nat["regression_weights"]["status"] == "ran"
    assert "n_extrapolating" in nat["regression_weights"]
    assert nat["sparse_synthetic_control"]["status"] == "ran"
    assert len(nat["sparse_synthetic_control"]["sizes"]) == 2


def test_sparse_max_subsets_invalid_raises():
    df, _years, _T0 = _make_panel(n_donors=5, T=8, T0=4, seed=1)
    res = _fit_iv(df)
    for bad in (0, -1, 2.5, np.nan):
        with pytest.raises(ValueError, match="max_subsets must be a positive integer"):
            res.sparse_synthetic_control(sizes=[2], max_subsets=bad)


def test_sparse_empty_sizes_raises():
    df, _years, _T0 = _make_panel(n_donors=5, T=8, T0=4, seed=1)
    res = _fit_iv(df)
    with pytest.raises(ValueError, match="non-empty"):
        res.sparse_synthetic_control(sizes=[])


def test_cv_tail_diagnostics_use_validation_window_capture():
    # Exercise the special v_method="cv" snapshot capture: the fixed (X1s, X0s, V) triple is
    # the VALIDATION-window standardized predictor matrices (re-aggregated per window), NOT the
    # full-pre matrices. Both tail diagnostics must operate on that captured cv space.
    from diff_diff.synthetic_control import _inner_solve_W

    df, _years, _T0 = _make_panel(n_donors=6, T=8, T0=6, seed=1)
    res = _fit_cv(df)  # v_method="cv", spanning special predictors (_CV_SPANNING, k=2)
    assert res._fit_converged
    snap = res._fit_snapshot
    # Captured matrices are the 2 spanning specs x 6 donors, in the cv validation-window space.
    assert snap.fit_X0s.shape == (2, 6)
    assert snap.fit_X1s.shape == (2,)
    assert snap.fit_v.shape == (2,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rw = res.regression_weights()
    assert res._regw_status == "ran"
    assert len(rw) == 6

    # Sparse winner (size 2) matches an independent brute-force over the CAPTURED cv-space
    # matrices with the fixed cv V — proving the cv path holds V fixed in the right space.
    import itertools

    tab = res.sparse_synthetic_control(sizes=[2])
    won = tab[tab["status"] == "ran"].iloc[0]["donor_ids"]
    Y = snap.pivots[snap.outcome]
    Z1_pre = Y.loc[snap.pre_periods, snap.treated_id].to_numpy(float)
    Z0_pre = Y.loc[snap.pre_periods, snap.donor_ids].to_numpy(float)
    best, best_mspe = None, np.inf
    for cols in itertools.combinations(range(len(snap.donor_ids)), 2):
        w, conv = _inner_solve_W(
            snap.fit_X1s,
            snap.fit_X0s[:, list(cols)],
            snap.fit_v,
            snap.inner_max_iter,
            snap.inner_min_decrease,
        )
        if not conv:
            continue
        m = float(np.mean((Z1_pre - Z0_pre[:, list(cols)] @ w) ** 2))
        if m < best_mspe:
            best_mspe, best = m, cols
    assert won == tuple(snap.donor_ids[c] for c in best)


def test_sparse_non_integer_sizes_raise():
    df, _years, _T0 = _make_panel(n_donors=6, T=8, T0=4, seed=1)
    res = _fit_iv(df)
    # int(2.9) would silently truncate to size 2 -> reject non-integral / bool sizes.
    for bad in ([2.9], [2, 3.0], True, [True], ["2"]):
        with pytest.raises(ValueError, match="must be integer|int or a sequence"):
            res.sparse_synthetic_control(sizes=bad)
    # A valid numpy-int size still works.
    tab = res.sparse_synthetic_control(sizes=[np.int64(2)])
    assert set(tab[tab["status"] == "ran"]["size"]) == {2}
