"""Drift detection for Tutorial 21 (`docs/tutorials/21_had_pretest_workflow.ipynb`).

The tutorial narrative quotes seed-specific numbers (overall verdict
substring, QUG / Stute / Yatchew p-values, joint pre-trends and homogeneity
horizon counts and p-values, Yatchew side-panel statistics under both null
modes). If library numerics drift (estimator changes, RNG path changes,
BLAS path changes), the prose can go stale silently while `pytest --nbmake`
still passes - it only checks that the cells execute without error.

These asserts re-derive the same numbers using the locked T21 DGP and seeds
the notebook uses, then check them against the values quoted in the
tutorial markdown. If a future change moves any number outside its
tolerance band, this test fails and a maintainer is forced to either
update the prose or investigate the methodology shift before merge.

T21 DGP differs from T20: dose distribution is `Uniform[$0.01K, $50K]`
(was `[$5K, $50K]` in T20). The true support is strictly positive but very
near zero. Two independent things follow from that small `D_(1)` and are
exercised in this drift file: (a) the QUG step fails-to-reject
`H0: d_lower = 0` in this finite sample, populating the workflow's verdict
with the "Assumption 7 deferred" substring used for the upgrade-arc
narrative; and (b) HAD's `design="auto"` selector - a separate min/median
heuristic that does NOT consume the QUG p-value - independently lands on
`continuous_at_zero` because `d.min() < 0.01 * median(|d|)` (per
`_detect_design()` in `had.py`). Both checks point to the same
identification path on this panel, but the rules are independent.
DGP and seed locked at `_scratch/t21_pretests/10_panel.py`.
Quoted numbers derived from `_scratch/t21_pretests/50_compose_narrative.py`.

Bootstrap p-value pins use **abs tolerance bands >= 0.15** per
`feedback_bootstrap_drift_tests_need_backend_tolerance` (Rust vs pure-Python
RNG paths can diverge by ~0.05-0.15 and flip rounding boundaries).
Deterministic statistics (QUG, Yatchew sigma2_*) get exact `round(..., 2)`
or `round(..., 4)` pins.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from diff_diff import HAD, did_had_pretest_workflow, generate_continuous_did_data, yatchew_hr_test

# Locked T21 DGP parameters (must stay in sync with the notebook).
MAIN_SEED = 87
N_UNITS = 60
N_PERIODS = 8
COHORT_PERIOD = 5
TRUE_SLOPE = 100.0
BASELINE_VISITS = 5000.0
DOSE_LOW = 0.01  # T21 change vs T20 (was 5.0): near-zero lower bound chosen so QUG fails-to-reject H0: d_lower = 0.
DOSE_HIGH = 50.0
WORKFLOW_SEED = 21


@pytest.fixture(scope="module")
def panel():
    raw = generate_continuous_did_data(
        n_units=N_UNITS,
        n_periods=N_PERIODS,
        cohort_periods=[COHORT_PERIOD],
        never_treated_frac=0.0,
        dose_distribution="uniform",
        dose_params={"low": DOSE_LOW, "high": DOSE_HIGH},
        att_function="linear",
        att_intercept=0.0,
        att_slope=TRUE_SLOPE,
        unit_fe_sd=8.0,
        time_trend=0.5,
        noise_sd=2.0,
        seed=MAIN_SEED,
    )
    p = raw.copy()
    p.loc[p["period"] < p["first_treat"], "dose"] = 0.0
    p = p.rename(
        columns={
            "unit": "dma_id",
            "period": "week",
            "outcome": "weekly_visits",
            "dose": "regional_spend_k",
        }
    )
    p["weekly_visits"] = p["weekly_visits"] + BASELINE_VISITS
    return p


@pytest.fixture(scope="module")
def two_period(panel):
    p = panel.copy()
    p["period"] = (p["week"] >= COHORT_PERIOD).astype(int) + 1
    collapsed = p.groupby(["dma_id", "period"], as_index=False).agg(
        weekly_visits=("weekly_visits", "mean"),
        regional_spend_k=("regional_spend_k", "mean"),
    )
    collapsed.loc[collapsed["period"] == 1, "regional_spend_k"] = 0.0
    collapsed["first_treat"] = 2
    return collapsed


@pytest.fixture(scope="module")
def overall_report(two_period):
    return did_had_pretest_workflow(
        data=two_period,
        outcome="weekly_visits",
        dose="regional_spend_k",
        time="period",
        unit="dma_id",
        first_treat="first_treat",
        alpha=0.05,
        n_bootstrap=999,
        seed=WORKFLOW_SEED,
        aggregate="overall",
    )


@pytest.fixture(scope="module")
def event_study_report(panel):
    return did_had_pretest_workflow(
        data=panel,
        outcome="weekly_visits",
        dose="regional_spend_k",
        time="week",
        unit="dma_id",
        first_treat="first_treat",
        alpha=0.05,
        n_bootstrap=999,
        seed=WORKFLOW_SEED,
        aggregate="event_study",
    )


@pytest.fixture(scope="module")
def yatchew_side_panel_inputs(panel):
    """Section 5's Yatchew side panel: post-period dose paired with the
    within-pre-period first-difference dy = Y[w4] - Y[w3]. Shared
    construction between the linearity-mode and mean_independence-mode
    tests below."""
    panel_sorted = panel.sort_values(["dma_id", "week"]).reset_index(drop=True)
    pre = panel_sorted[panel_sorted["week"].isin([3, 4])]
    pre_pivot = pre.pivot(index="dma_id", columns="week", values="weekly_visits")
    dy = (pre_pivot[4] - pre_pivot[3]).to_numpy(dtype=np.float64)
    post_dose = (
        panel_sorted[panel_sorted["week"] == 5]
        .set_index("dma_id")
        .sort_index()["regional_spend_k"]
        .to_numpy(dtype=np.float64)
    )
    return post_dose, dy


def test_panel_matches_t21_locked_dgp(panel):
    """Section 2 narrative claims 60 DMAs over 8 weeks, regional spend
    drawn from Uniform[$0.01K, $50K] - true support strictly positive
    but very near zero (so QUG can fail-to-reject in this finite
    sample). If the DGP drifts, this surfaces."""
    assert panel["dma_id"].nunique() == N_UNITS
    assert panel["week"].nunique() == N_PERIODS
    post_doses = (
        panel.loc[panel["week"] >= COHORT_PERIOD].groupby("dma_id")["regional_spend_k"].first()
    )
    assert post_doses.min() >= DOSE_LOW, post_doses.min()
    assert post_doses.max() <= DOSE_HIGH, post_doses.max()
    # T21 narrative says "starts from $10" - i.e. the smallest dose is
    # below $1K (~$180 from numbers.json: d_order_1 = 0.180569...).
    assert post_doses.min() < 1.0, post_doses.min()


def test_overall_verdict_flags_assumption_7_deferred(overall_report):
    """Load-bearing pivot for the upgrade-arc narrative. Sections 3-4
    of the notebook quote this verdict substring verbatim. If
    `_compose_verdict()` is refactored such that the substring changes
    or moves, this test surfaces it."""
    pivot = "Assumption 7 pre-trends test NOT run"
    assert pivot in overall_report.verdict, overall_report.verdict
    # Adjacent pivot the prose also quotes:
    assert (
        "paper step 2 deferred to Phase 3 follow-up" in overall_report.verdict
    ), overall_report.verdict


def test_overall_path_structural_anchors(overall_report):
    """Notebook Section 3 prose claims `pretrends_joint` and
    `homogeneity_joint` are both None on the overall path (they are
    not populated on the two-period dispatch). Sturdier than a
    verdict-string anchor against future verdict refactors."""
    assert overall_report.aggregate == "overall"
    assert overall_report.pretrends_joint is None
    assert overall_report.homogeneity_joint is None
    assert overall_report.all_pass is True


def test_overall_qug_fails_to_reject(overall_report):
    """Section 3 narrative claims QUG fails-to-reject H0: d_lower = 0
    (data are statistically consistent with continuous_at_zero design).
    QUG is fully deterministic; pin exact rounded values. The independent
    HAD `design="auto"` selector decision is locked separately by
    `test_had_design_auto_lands_on_continuous_at_zero`."""
    assert overall_report.qug.reject is False
    # T statistic = D_(1) / (D_(2) - D_(1)) is fully deterministic.
    assert round(overall_report.qug.t_stat, 2) == 3.86, overall_report.qug.t_stat
    assert round(overall_report.qug.critical_value, 1) == 19.0, overall_report.qug.critical_value
    # Closed-form p-value `1 / (1 + T)` under Theorem 4's Exp/Exp limit
    # law is equally deterministic; the notebook output quotes 0.2059
    # so pin it directly so prose drift surfaces.
    assert round(overall_report.qug.p_value, 4) == 0.2059, overall_report.qug.p_value


def test_overall_stute_fails_to_reject(overall_report):
    """Section 3 narrative quotes Stute p_value ~0.686. Stute uses
    Mammen wild bootstrap so the p-value is RNG-dependent; use a
    bounded abs tolerance band per
    `feedback_bootstrap_drift_tests_need_backend_tolerance` (>= 0.15
    width). Both bounds tight enough to catch methodology drift in
    either direction, loose enough for backend RNG path differences."""
    assert overall_report.stute.reject is False
    assert 0.53 <= overall_report.stute.p_value <= 0.84, overall_report.stute.p_value


def test_overall_yatchew_fails_to_reject(overall_report):
    """Section 3 narrative + cell 9 callout describe the very large
    negative Yatchew T_hr (~-35,000) under perfect linearity with
    heterogeneous doses. Pin sigma2_* (deterministic) and the
    rejection decision."""
    assert overall_report.yatchew.reject is False
    assert overall_report.yatchew.p_value > 0.99, overall_report.yatchew.p_value
    # sigma2_diff is deterministic given the panel.
    assert (
        round(overall_report.yatchew.sigma2_diff, 0) == 6250.0
    ), overall_report.yatchew.sigma2_diff


def test_event_study_verdict_says_admissible(event_study_report):
    """Sections 4-5 narrative claims the event-study verdict reads
    'TWFE admissible under Section 4 assumptions' (no `deferred`
    caveat). Locks the upgrade-arc closure pivot."""
    assert "TWFE admissible" in event_study_report.verdict, event_study_report.verdict
    assert "deferred" not in event_study_report.verdict, event_study_report.verdict


def test_event_study_path_structural_anchors(event_study_report):
    """Section 4 narrative claims `pretrends_joint` and
    `homogeneity_joint` are both populated on the event-study path
    (the upgrade arc closure). Mirror of the overall path's negative
    structural anchor."""
    assert event_study_report.aggregate == "event_study"
    assert event_study_report.pretrends_joint is not None
    assert event_study_report.homogeneity_joint is not None
    assert event_study_report.all_pass is True


def test_event_study_qug_matches_overall(event_study_report, overall_report):
    """Section 4 narrative claims QUG re-runs deterministically with
    the same numbers as the overall path (same dose distribution at
    F)."""
    assert event_study_report.qug.reject is overall_report.qug.reject
    assert round(event_study_report.qug.t_stat, 4) == round(overall_report.qug.t_stat, 4)
    # The QUG p-value is also deterministic and must match across paths.
    assert round(event_study_report.qug.p_value, 4) == round(overall_report.qug.p_value, 4)


def test_event_study_pretrends_horizons_correct(event_study_report):
    """Section 4 narrative claims `joint_pretrends_test` runs over 3
    horizons (pre-periods 1, 2, 3, with week 4 reserved as the base
    period). Locks the earlier-pre-period precondition closure
    (PR #402 R7) for T21's specific panel: F=5, t_pre={1,2,3,4},
    base=4, earlier pre-periods={1,2,3}."""
    pj = event_study_report.pretrends_joint
    assert pj is not None
    assert pj.n_horizons == 3, pj.n_horizons
    assert pj.horizon_labels == ["1", "2", "3"], pj.horizon_labels


def test_event_study_homogeneity_horizons_correct(event_study_report):
    """Section 4 narrative claims `joint_homogeneity_test` runs over 4
    post horizons (weeks 5, 6, 7, 8)."""
    hj = event_study_report.homogeneity_joint
    assert hj is not None
    assert hj.n_horizons == 4, hj.n_horizons
    assert hj.horizon_labels == ["5", "6", "7", "8"], hj.horizon_labels


def test_event_study_pretrends_fails_to_reject(event_study_report):
    """Section 4 narrative quotes the pre-trends p-value as 'close to
    alpha = 0.05 ... warrants scrutiny' (~0.07 from numbers.json) -
    non-rejection is not a clean pass at this margin. Use binary
    fail-to-reject + a wide abs tolerance band - bootstrap p-values
    near alpha are the most sensitive to RNG path differences."""
    pj = event_study_report.pretrends_joint
    assert pj is not None
    assert pj.reject is False
    # Tight upper bound to catch a real methodology shift; lower bound
    # would catch a regression that pushes pre-trends to look pristine
    # (which would belie the "close to alpha" narrative).
    assert 0.0 <= pj.p_value <= 0.25, pj.p_value


def test_event_study_homogeneity_fails_to_reject(event_study_report):
    """Section 4 narrative claims joint homogeneity strongly fails to
    reject and quotes p ~0.763 from numbers.json. Use a bounded abs
    tolerance band per
    `feedback_bootstrap_drift_tests_need_backend_tolerance` so that
    drift in either direction (toward rejection or toward an even
    cleaner pass) flags the prose as stale rather than silently
    passing."""
    hj = event_study_report.homogeneity_joint
    assert hj is not None
    assert hj.reject is False
    assert 0.61 <= hj.p_value <= 0.92, hj.p_value


def test_had_design_auto_lands_on_continuous_at_zero(two_period):
    """Section 2 narrative claims HAD's `design="auto"` selector
    independently lands on `continuous_at_zero` (target = `WAS`) on
    this panel because `d.min() < 0.01 * median(|d|)`. This is a
    separate decision rule from the QUG test (locked by
    `test_overall_qug_fails_to_reject`); the two happen to agree on
    this panel but the rules are independent. We fit HAD with
    `design="auto"` here just to verify the prose."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        est = HAD(design="auto")
        result = est.fit(
            two_period,
            outcome="weekly_visits",
            dose="regional_spend_k",
            time="period",
            unit="dma_id",
        )
    assert result.design == "continuous_at_zero", result.design
    assert result.target_parameter == "WAS", result.target_parameter


def test_yatchew_side_panel_linearity_passes(yatchew_side_panel_inputs):
    """Section 5 (Yatchew side panel) narrative claims `null="linearity"`
    does not reject on the within-pre-period first-difference paired
    with post-period dose. Pin the T_hr statistic (deterministic);
    Yatchew has no bootstrap component."""
    post_dose, dy = yatchew_side_panel_inputs
    res = yatchew_hr_test(d=post_dose, dy=dy, alpha=0.05, null="linearity")
    assert res.reject is False
    assert res.null_form == "linearity"
    assert round(res.t_stat_hr, 2) == 0.02, res.t_stat_hr
    assert round(res.sigma2_lin, 2) == 6.53, res.sigma2_lin
    # Closed-form 1-sided normal p-value is deterministic; the notebook
    # output quotes 0.4917, pin it so prose drift surfaces.
    assert round(res.p_value, 4) == 0.4917, res.p_value


def test_yatchew_side_panel_mean_independence_passes(yatchew_side_panel_inputs):
    """Section 5 narrative claims `null="mean_independence"` does not
    reject on the same input but with larger sigma2_lin (the stricter
    null has more residual variance to explain)."""
    post_dose, dy = yatchew_side_panel_inputs
    res_mi = yatchew_hr_test(d=post_dose, dy=dy, alpha=0.05, null="mean_independence")
    res_lin = yatchew_hr_test(d=post_dose, dy=dy, alpha=0.05, null="linearity")
    assert res_mi.reject is False
    assert res_mi.null_form == "mean_independence"
    assert round(res_mi.t_stat_hr, 2) == 0.55, res_mi.t_stat_hr
    assert round(res_mi.sigma2_lin, 2) == 7.01, res_mi.sigma2_lin
    # Closed-form 1-sided normal p-value is deterministic; the notebook
    # output quotes 0.2899, pin it so prose drift surfaces.
    assert round(res_mi.p_value, 4) == 0.2899, res_mi.p_value
    # Pedagogical claim from Section 5: stricter null -> larger sigma2_lin.
    assert res_mi.sigma2_lin > res_lin.sigma2_lin
    # And the differencing variance (sigma2_diff) is shared across modes.
    assert round(res_mi.sigma2_diff, 4) == round(res_lin.sigma2_diff, 4)


# =============================================================================
# Notebook-narrative cross-check
# =============================================================================
#
# The asserts above re-derive numbers from the locked DGP+seed but do NOT
# verify that the rendered tutorial actually quotes those same numbers.
# Without this layer, the notebook prose can drift independently of the
# library numerics (or vice versa) and CI stays green because
# `nbsphinx_execute = "never"` in `docs/conf.py` (CI doesn't re-execute
# notebooks during build). Use the shared tutorial-drift helper that
# parses the notebook JSON and checks both markdown prose AND executed
# output cells (since the load-bearing verdict strings appear in
# print()-rendered output blocks, not just markdown prose).


T21_NOTEBOOK = "docs/tutorials/21_had_pretest_workflow.ipynb"


def test_notebook_quotes_match_pinned_constants():
    """Every load-bearing verdict/value this file pins must appear
    verbatim in the rendered T21 notebook surface (markdown prose +
    executed output cells).

    Closes the gap the file-level docstring claims to cover ("check
    against the values quoted in the tutorial markdown") but the rest
    of the file did not actually exercise — every prior assert
    re-derives numbers from the DGP and compares them to a hardcoded
    constant, leaving the notebook completely uncross-checked.
    Without this test, the notebook can drift independently of the
    library numerics (or vice versa) and CI stays green because
    ``nbsphinx_execute = "never"`` in ``docs/conf.py``.
    """
    from tests._tutorial_drift import assert_quotes_in_rendered

    expected_quotes = [
        # ---- Verdict-string anchors ----
        # Overall verdict substring (also pinned in test_overall_workflow_*).
        # Appears in markdown prose AND in the verdict-print output cell.
        "paper step 2 deferred to Phase 3 follow-up",
        # Event-study verdict substring (rendered output of the
        # aggregate='event_study' workflow + markdown reading-cell).
        "TWFE admissible under Section 4 assumptions",
        # Event-study output cell anchor — full verdict header.
        "QUG, joint pre-trends, and joint linearity diagnostics fail-to-reject",
        # ---- Structural-field anchors ----
        "aggregate = 'event_study'",
        "pretrends_joint populated? True",
        "homogeneity_joint populated? True",
        "aggregate = 'overall'",
        "pretrends_joint populated? False",
        # ---- Verdict-reading markdown anchors (cell 6) ----
        "T = D_(1) / (D_(2) - D_(1)) ~ 3.86",
        "1/alpha - 1 = 19",
        # ---- Numeric anchors pinned analytically above ----
        # Every value pinned via round(..., 4) == 0.NNNN in this file
        # must also appear in the rendered notebook (otherwise the
        # tutorial prose / output is showing a different number than
        # the test claims to lock).
        "0.2059",  # QUG p-value (test_overall_workflow_*)
        "0.6860",  # Stute p-value tolerance band anchor
        "0.0720",  # joint-pretrends Stute p-value (event-study)
        "0.7630",  # joint-homogeneity Stute p-value (event-study)
        "0.4917",  # Yatchew side-panel null=linearity p-value
        "0.2899",  # Yatchew side-panel null=mean_independence p-value
        # Design auto-detect outcome (also pinned by overall-path tests).
        "continuous_at_zero",
        # Use the exact paper-step-1 phrasing with target=`WAS` so we
        # don't false-pass on the many incidental occurrences of "WAS"
        # elsewhere in the prose.
        "target = `WAS`",
        # Overall Yatchew p-value (analytical short-circuit on this DGP).
        "1.0000",
        # Overall Yatchew sigma2_lin in the rendered output.
        "6250.2569",
        # Side-panel Yatchew sigma2_lin under null='linearity'.
        "6.5340",
        # Side-panel Yatchew sigma2_lin under null='mean_independence'.
        "7.0076",
    ]
    assert_quotes_in_rendered(T21_NOTEBOOK, expected_quotes, surface="rendered")
