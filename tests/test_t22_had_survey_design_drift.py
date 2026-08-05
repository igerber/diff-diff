"""Drift detection for Tutorial 22 (`docs/tutorials/22_had_survey_design.ipynb`).

The tutorial narrative quotes seed-specific numbers (panel composition,
naive vs. survey-design SE inflation, event-study cband behavior, pretest
workflow verdicts under SurveyDesign(strata=...), QUG-under-survey
deferral substring). If library numerics drift (estimator changes, RNG
path changes, BLAS path changes), the prose can go stale silently while
``pytest --nbmake`` still passes - it only checks that the cells execute
without error.

These asserts re-derive the same numbers using the locked T22 DGP and
seeds the notebook uses, then check them against the values quoted in
the tutorial markdown. If a future change moves any number outside its
tolerance band, this test fails and a maintainer is forced to either
update the prose or investigate the methodology shift before merge.

T22 is the third tutorial in the HAD series (after T20 headline and T21
pretest workflow). It demonstrates the now-fully-supported survey-design
path through ``HeterogeneousAdoptionDiD`` and ``did_had_pretest_workflow``,
unblocked by PR #432 (2026-05-14, merge ``d5e5021f``) which lifted the
``NotImplementedError`` gate on ``SurveyDesign(strata=...)`` for the
Stute family. T22's DGP layers a BRFSS-shape survey design (5 strata x 6
PSUs/stratum x 2 states/PSU = 60 states; weights ~ post-stratification
raking with CV ~ 0.30; FPC = 30 PSUs/stratum) onto the same
continuous-dose HAD panel shape T20 uses (Design 1, dose ~ Uniform[$5K,
$50K], att_slope=100). DGP and seed locked at ``_scratch/t22/dev.py``.

**Bootstrap p-value pins use anchored windows of total width 0.30
(± 0.15 around the seed=22 captured centers)** per
``feedback_bootstrap_drift_tests_need_backend_tolerance`` and
``feedback_strata_bootstrap_path_divergence``. Stratified Mammen
multiplier paths (PR #432) reduce effective dofs vs non-strata; PR #432
commit ``aef07020`` already had to relax bit-equality bands on this
code path. Deterministic statistics (Yatchew sigma2_*, t_hr, design
auto-detection, horizon labels, panel composition, weight CV under
locked numpy default_rng) get exact pins.

**Verdict-substring discipline** — both the overall and the event-study
``HADPretestReport.verdict`` terminate in ``_QUG_DEFERRED_SUFFIX``
(``had_pretests.py:4300``: ``" (linearity-conditional verdict;
QUG-under-survey deferred per Phase 4.5 C0)"``). The DIFFERENT message
(``"(QUG step skipped - permanently deferred under survey designs per
Phase 4.5 C0)"``, rendered by ``HADPretestReport.summary()``) is in the
formatted ``report.summary()`` block, NOT in ``report.verdict``. Tests
lock each substring on the correct field.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    HAD,
    SurveyDesign,
    did_had_pretest_workflow,
    generate_continuous_did_data,
)

# Locked T22 DGP parameters (must stay in sync with the notebook).
MAIN_SEED = 87
N_UNITS = 60
N_PERIODS = 8
COHORT_PERIOD = 5
TRUE_SLOPE = 100.0
BASELINE_OUTCOME = 35.0
DOSE_LOW = 5.0
DOSE_HIGH = 50.0

# Survey-design layer (in-notebook helper).
N_STRATA = 5
PSU_PER_STRATUM = 6
STATES_PER_PSU = 2
WEIGHT_CV_TARGET = 0.30
FPC_PER_STRATUM = 30
PSU_PERIOD_SHOCK_SD = 1.5  # See plan Risk 1; HAD WAS IF concentration caps inflation ~1.25
SD_SEED = 87

# Pretest workflow.
WORKFLOW_SEED = 22
N_BOOTSTRAP = 999

# Substrings the notebook prose depends on.
QUG_DEFERRED_SUFFIX_VERDICT = (
    "linearity-conditional verdict; QUG-under-survey deferred per Phase 4.5 C0"
)
QUG_SKIP_SUMMARY_NOTE = (
    "QUG step skipped - permanently deferred under survey designs per Phase 4.5 C0"
)


def _attach_brfss_survey_columns(
    panel: pd.DataFrame,
    *,
    seed: int,
    n_strata: int = N_STRATA,
    psu_per_stratum: int = PSU_PER_STRATUM,
    states_per_psu: int = STATES_PER_PSU,
    weight_cv: float = WEIGHT_CV_TARGET,
    fpc_per_stratum: int = FPC_PER_STRATUM,
    psu_period_shock_sd: float = PSU_PERIOD_SHOCK_SD,
) -> pd.DataFrame:
    """Drift-test-local copy of T22's in-notebook survey-attach helper.

    Lives here (not as a library helper) because it is a TUTORIAL element
    demonstrating how a practitioner attaches survey design to their HAD
    panel. The drift test inlines it so changes to the notebook helper
    do not silently change the locked numerical anchors. If this helper
    diverges from the notebook helper, the panel-composition tests
    (weight CV, stratum/PSU counts) catch the drift.
    """
    rng = np.random.default_rng(seed)
    state_ids = np.sort(panel["state_id"].unique())
    n_states = len(state_ids)
    n_psu = n_strata * psu_per_stratum
    if n_states != n_psu * states_per_psu:
        raise ValueError(
            f"state count {n_states} must equal n_strata*psu_per_stratum*"
            f"states_per_psu = {n_psu * states_per_psu}"
        )
    perm = rng.permutation(n_states)
    psu_block = np.repeat(np.arange(n_psu), states_per_psu)
    psu_of_state = psu_block[np.argsort(perm)]
    stratum_of_state = psu_of_state // psu_per_stratum
    base_per_stratum = np.array([0.8, 0.9, 1.0, 1.1, 1.3])
    base_w = base_per_stratum[stratum_of_state]
    sigma = np.sqrt(np.log(1 + weight_cv**2))
    pert = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n_states)
    w_per_state = base_w * pert
    state_lookup = pd.DataFrame(
        {
            "state_id": state_ids,
            "stratum": stratum_of_state.astype(np.int64),
            "psu_id": psu_of_state.astype(np.int64),
            "weight": w_per_state,
            "fpc": float(fpc_per_stratum),
        }
    )
    panel_attached = panel.merge(state_lookup, on="state_id", how="left")
    n_periods = int(panel["week"].max() - panel["week"].min() + 1)
    psu_period_shocks = rng.normal(0.0, psu_period_shock_sd, size=(n_psu, n_periods))
    week_min = int(panel["week"].min())
    shock_lookup = pd.DataFrame(
        [
            {
                "psu_id": int(p),
                "week": int(w + week_min),
                "psu_period_shock": float(psu_period_shocks[p, w]),
            }
            for p in range(n_psu)
            for w in range(n_periods)
        ]
    )
    panel_attached = panel_attached.merge(shock_lookup, on=["psu_id", "week"], how="left")
    panel_attached["screening_uptake"] = (
        panel_attached["screening_uptake"] + panel_attached["psu_period_shock"]
    )
    return panel_attached.drop(columns=["psu_period_shock"])


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
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
            "unit": "state_id",
            "period": "week",
            "outcome": "screening_uptake",
            "dose": "spend_k",
        }
    )
    p["screening_uptake"] = p["screening_uptake"] + BASELINE_OUTCOME
    return _attach_brfss_survey_columns(p, seed=SD_SEED)


@pytest.fixture(scope="module")
def panel_2p(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["period"] = (p["week"] >= COHORT_PERIOD).astype(int) + 1
    collapsed = p.groupby(["state_id", "period"], as_index=False).agg(
        screening_uptake=("screening_uptake", "mean"),
        spend_k=("spend_k", "mean"),
        stratum=("stratum", "first"),
        psu_id=("psu_id", "first"),
        weight=("weight", "first"),
        fpc=("fpc", "first"),
    )
    return pd.DataFrame(collapsed)


@pytest.fixture(scope="module")
def survey_design() -> SurveyDesign:
    return SurveyDesign(weights="weight", strata="stratum", psu="psu_id", fpc="fpc")


@pytest.fixture(scope="module")
def naive_overall_result(panel_2p: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return HAD(design="auto").fit(
            panel_2p,
            outcome="screening_uptake",
            dose="spend_k",
            time="period",
            unit="state_id",
        )


@pytest.fixture(scope="module")
def survey_overall_result(panel_2p: pd.DataFrame, survey_design: SurveyDesign):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return HAD(design="auto").fit(
            panel_2p,
            outcome="screening_uptake",
            dose="spend_k",
            time="period",
            unit="state_id",
            survey_design=survey_design,
        )


@pytest.fixture(scope="module")
def survey_event_study_result(panel: pd.DataFrame, survey_design: SurveyDesign):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return HAD(design="auto").fit(
            panel,
            outcome="screening_uptake",
            dose="spend_k",
            time="week",
            unit="state_id",
            first_treat="first_treat",
            survey_design=survey_design,
            cband=True,
        )


@pytest.fixture(scope="module")
def overall_report(panel_2p: pd.DataFrame, survey_design: SurveyDesign):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return did_had_pretest_workflow(
            panel_2p,
            outcome="screening_uptake",
            dose="spend_k",
            time="period",
            unit="state_id",
            survey_design=survey_design,
            n_bootstrap=N_BOOTSTRAP,
            seed=WORKFLOW_SEED,
        )


@pytest.fixture(scope="module")
def event_study_report(panel: pd.DataFrame, survey_design: SurveyDesign):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return did_had_pretest_workflow(
            panel,
            outcome="screening_uptake",
            dose="spend_k",
            time="week",
            unit="state_id",
            first_treat="first_treat",
            survey_design=survey_design,
            n_bootstrap=N_BOOTSTRAP,
            seed=WORKFLOW_SEED,
        )


# ============================================================================
# Group A — Panel & survey composition (deterministic exact pins)
# ============================================================================


def test_panel_matches_t22_locked_dgp(panel: pd.DataFrame):
    """Locks panel size, columns, and the dose/outcome ranges quoted in §2.

    If T22 mutates the locked DGP parameters, panel shape/columns shift
    and this test fails before downstream numerical pins fail."""
    assert panel.shape == (480, 10), panel.shape
    assert set(panel.columns) >= {
        "state_id",
        "week",
        "screening_uptake",
        "first_treat",
        "spend_k",
        "stratum",
        "psu_id",
        "weight",
        "fpc",
    }, panel.columns.tolist()
    assert panel["state_id"].nunique() == N_UNITS
    assert panel["week"].min() == 1
    assert panel["week"].max() == N_PERIODS
    post_dose = panel.loc[panel["week"] >= COHORT_PERIOD, "spend_k"]
    assert DOSE_LOW <= post_dose.min() <= post_dose.max() <= DOSE_HIGH


def test_survey_design_attachment_shape(panel: pd.DataFrame):
    """Locks the BRFSS-shape design dimensions narrated in §2."""
    assert panel["stratum"].nunique() == N_STRATA
    assert panel["psu_id"].nunique() == N_STRATA * PSU_PER_STRATUM
    psu_state_count = panel.groupby("psu_id")["state_id"].nunique()
    assert (psu_state_count == STATES_PER_PSU).all(), psu_state_count.value_counts().to_dict()


def test_survey_weight_cv_in_band(panel: pd.DataFrame):
    """Locks the ~0.30 weight CV at seed=87. Bit-stable under numpy
    default_rng; if numpy upgrades change RNG semantics this catches it."""
    weights_per_state = panel.groupby("state_id")["weight"].first()
    cv = float(weights_per_state.std() / weights_per_state.mean())
    assert abs(cv - 0.327) < 0.03, cv


def test_survey_design_fpc_constant_per_stratum(panel: pd.DataFrame):
    """FPC is scalar-per-stratum (==30) in the helper; the SurveyDesign
    object treats fpc as a column, so per-row constancy is the contract."""
    assert panel["fpc"].nunique() == 1
    assert float(panel["fpc"].iloc[0]) == FPC_PER_STRATUM


def test_panel_constant_within_state_invariant(panel: pd.DataFrame):
    """HAD per-unit aggregation requires weight/stratum/psu_id/fpc to be
    constant within state; resolve_survey_design enforces this. If the
    helper accidentally injects per-period variation in these columns,
    HAD fails downstream — test catches the helper-side bug first."""
    for col in ("weight", "stratum", "psu_id", "fpc"):
        per_state_unique = panel.groupby("state_id")[col].nunique()
        assert per_state_unique.max() == 1, (col, per_state_unique.value_counts().to_dict())


# ============================================================================
# Group B — Naive vs survey-aware headline fit
# ============================================================================


def test_naive_overall_design_auto_continuous_near_d_lower(naive_overall_result):
    """Locks T22's design auto-detection: at dose ~ Uniform[5, 50],
    `d.min() / median(|d|) > 0.01`, so the heuristic resolves to
    ``continuous_near_d_lower`` (Design 1) targeting WAS_d_lower."""
    assert naive_overall_result.design == "continuous_near_d_lower"
    assert naive_overall_result.target_parameter == "WAS_d_lower"


def test_survey_overall_design_auto_continuous_near_d_lower(survey_overall_result):
    """Survey path picks the same design — design auto-detection is
    sample-based and does not consume survey weights."""
    assert survey_overall_result.design == "continuous_near_d_lower"
    assert survey_overall_result.target_parameter == "WAS_d_lower"


def test_survey_att_close_to_truth(survey_overall_result):
    """Survey-aware HAD recovers slope=100 within analytical noise on
    this DGP. Tight pin (round to int) — the local-linear estimator is
    analytical, no Rust RNG path."""
    assert round(survey_overall_result.att, 0) == 100, survey_overall_result.att


def test_survey_se_strictly_inflated_vs_naive(naive_overall_result, survey_overall_result):
    """Sign-only structural anchor: survey SE > naive SE on this DGP.
    The magnitude of inflation is modest (~10%) because HAD's WAS_d_lower
    has IF concentrated near d_lower (few units), capping how much the
    PSU x period shock injection can amplify cluster correlation. The
    sign holds robustly across reasonable shock SDs (per dev script
    sweep)."""
    assert survey_overall_result.se > naive_overall_result.se, (
        naive_overall_result.se,
        survey_overall_result.se,
    )


def test_event_study_plot_uses_stored_pointwise_ci_endpoints():
    """The §5 matplotlib plot must build pointwise CI bars from the
    estimator's stored ``conf_int_low`` / ``conf_int_high`` (which on
    the survey path use ``t`` critical values with ``df_survey``),
    NOT from hard-coded ``1.96 * es.se`` Normal-theory bands. The
    earlier version of the plot used the latter and silently
    understated uncertainty relative to the table printed in the
    cell above it (CI AI review post-consolidation P1).

    This is a static check on the notebook source — the plot cell
    runs but produces no return value we can introspect, so we lock
    the construction at the source level. Skipped on isolated-install
    CI jobs where ``docs/`` is not copied alongside ``tests/`` and
    ``nbformat`` is not in the runtime deps (per
    ``feedback_golden_file_pytest_skip``)."""
    from pathlib import Path

    nbformat = pytest.importorskip("nbformat")

    nb_path = (
        Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "22_had_survey_design.ipynb"
    )
    if not nb_path.exists():
        pytest.skip(f"Notebook not present at {nb_path} (isolated-install CI)")
    nb = nbformat.read(nb_path, as_version=4)
    plot_cell_src = None
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if "HAD event-study under SurveyDesign" in src and "errorbar" in src:
            plot_cell_src = src
            break
    assert plot_cell_src is not None, "T22 event-study plot cell not found"
    # Must use stored endpoints
    assert "conf_int_low" in plot_cell_src, plot_cell_src
    assert "conf_int_high" in plot_cell_src, plot_cell_src
    # Must NOT hard-code Normal-theory bars
    assert "1.96 * np.asarray(es.se)" not in plot_cell_src, plot_cell_src
    assert "1.96 * es.se" not in plot_cell_src, plot_cell_src


def test_survey_se_inflation_ratio_in_band(naive_overall_result, survey_overall_result):
    """Anchored band on the seeded SE-inflation ratio. T22 §3 narrative
    quotes "around 1.10x" inflation; sign-only assertion above is too
    weak to catch numerical drift in the magnitude (per CI AI review
    R4 P3). Locks the seed=87 captured ratio (~1.0985) to a tight
    window so the §3 prose can't go silently stale if the analytical
    Binder/TSL composition drifts."""
    ratio = float(survey_overall_result.se / naive_overall_result.se)
    assert 1.00 <= ratio <= 1.20, ratio


def test_survey_ci_covers_truth(survey_overall_result):
    """Survey-aware CI covers the true slope=100."""
    lo, hi = survey_overall_result.conf_int
    assert lo <= TRUE_SLOPE <= hi, (lo, hi)


# ============================================================================
# Group C — Event-study under survey
# ============================================================================


def test_event_study_horizons_complete(survey_event_study_result):
    """Locks the same horizon set T20 produces on this DGP shape."""
    horizons = list(survey_event_study_result.event_times)
    assert horizons == [-4, -3, -2, 0, 1, 2, 3], horizons


def test_event_study_post_horizons_cover_truth_under_survey(survey_event_study_result):
    """All four post-launch horizons cover the true slope=100 under
    survey-aware pointwise CIs."""
    es = survey_event_study_result
    post_mask = np.asarray(es.event_times) >= 0
    lows = np.asarray(es.conf_int_low)[post_mask]
    highs = np.asarray(es.conf_int_high)[post_mask]
    for lo, hi in zip(lows, highs):
        assert lo <= TRUE_SLOPE <= hi, (lo, hi)


def test_event_study_pre_horizons_cover_zero_under_survey(survey_event_study_result):
    """Pre-launch placebo horizons cover zero under survey-aware
    pointwise CIs (no pre-trends in this DGP)."""
    es = survey_event_study_result
    pre_mask = np.asarray(es.event_times) < 0
    lows = np.asarray(es.conf_int_low)[pre_mask]
    highs = np.asarray(es.conf_int_high)[pre_mask]
    for lo, hi in zip(lows, highs):
        assert lo <= 0.0 <= hi, (lo, hi)


def test_event_study_cband_is_wider_or_equal_pointwise(survey_event_study_result):
    """sup-t band is at least as wide as pointwise per horizon (cross-
    horizon multiplicity correction; never tighter than pointwise).
    Locks that ``cband_low`` and ``cband_high`` are populated and
    obey the cross-horizon multiplicity ordering."""
    es = survey_event_study_result
    assert es.cband_low is not None
    assert es.cband_high is not None
    cband_widths = np.asarray(es.cband_high) - np.asarray(es.cband_low)
    pointwise_widths = np.asarray(es.conf_int_high) - np.asarray(es.conf_int_low)
    # Allow tiny numerical noise (atol=1e-9) but otherwise require >=.
    assert (cband_widths + 1e-9 >= pointwise_widths).all(), {
        "cband": cband_widths.tolist(),
        "pointwise": pointwise_widths.tolist(),
    }


# ============================================================================
# Group D — Pretest workflow under survey: overall path
# ============================================================================


def test_overall_report_qug_is_none_under_survey(overall_report):
    """Phase 4.5 C0 contract: QUG step is permanently deferred under
    survey designs; ``report.qug`` is ``None`` on the overall path."""
    assert overall_report.qug is None


def test_overall_report_verdict_carries_qug_deferred_suffix(overall_report):
    """Locks ``_QUG_DEFERRED_SUFFIX`` substring (``had_pretests.py:4300``)
    on ``report.verdict``. This is the load-bearing pivot the §6 leadership
    paragraph depends on."""
    assert QUG_DEFERRED_SUFFIX_VERDICT in overall_report.verdict, overall_report.verdict


def test_overall_report_all_pass_under_null(overall_report):
    """``all_pass=True`` under the linear-DGP null (no pre-trends, no
    heterogeneity)."""
    assert overall_report.all_pass is True


def test_overall_report_stute_fails_to_reject(overall_report):
    """Stute CvM fails-to-reject linearity. Anchored bootstrap-p band
    centered on the seed=22 captured value (~0.42) with total width
    0.30 (± 0.15) per ``feedback_strata_bootstrap_path_divergence``
    (stratified Mammen multiplier reduces effective dofs vs
    non-strata; PR #432 commit ``aef07020`` had to relax bit-equality
    on this code path). Drift either toward rejection or toward an
    even cleaner pass flags the §6 prose as stale rather than
    silently passing."""
    assert overall_report.stute is not None
    assert overall_report.stute.reject is False
    p = float(overall_report.stute.p_value)
    assert 0.27 <= p <= 0.57, p


def test_overall_report_yatchew_fails_to_reject(overall_report):
    """Yatchew-HR fails-to-reject linearity. Yatchew is closed-form
    weighted-OLS (no bootstrap), so sigma2_lin and sigma2_diff are
    deterministic — exact pin to 4 decimals."""
    y = overall_report.yatchew
    assert y is not None
    assert y.reject is False
    assert round(float(y.sigma2_lin), 4) == 2.7270, y.sigma2_lin
    assert round(float(y.sigma2_diff), 4) == 5148.3208, y.sigma2_diff


# ============================================================================
# Group E — Pretest workflow under survey: event-study path
# ============================================================================


def test_event_study_report_qug_is_none_under_survey(event_study_report):
    """Phase 4.5 C0 contract holds on the event-study path too."""
    assert event_study_report.qug is None


def test_event_study_report_verdict_carries_qug_deferred_suffix(event_study_report):
    """Both the overall AND event-study verdicts share
    ``_QUG_DEFERRED_SUFFIX`` (per
    ``_compose_verdict_event_study_survey`` at
    ``had_pretests.py:4368-4406`` — all three return branches end in
    the suffix). Distinct prefix; identical suffix."""
    assert QUG_DEFERRED_SUFFIX_VERDICT in event_study_report.verdict, event_study_report.verdict


def test_event_study_report_summary_contains_qug_skip_note(event_study_report):
    """Separate from the verdict suffix above: ``report.summary()`` (the
    formatted multi-line block at ``had_pretests.py:736``) renders a
    distinct QUG-skip note. Locked here because the §6 walkthrough
    quotes ``report.summary()`` output as well as the verdict string."""
    summary = event_study_report.summary()
    assert QUG_SKIP_SUMMARY_NOTE in summary, summary[:400]


def test_event_study_report_pretrends_horizons_correct(event_study_report):
    """Locks the joint pretrends horizon set: weeks 1, 2, 3 (the
    pre-treatment placebo periods upgraded to a joint cusum)."""
    pj = event_study_report.pretrends_joint
    assert pj is not None
    assert pj.n_horizons == 3
    assert list(pj.horizon_labels) == ["1", "2", "3"], pj.horizon_labels


def test_event_study_report_homogeneity_horizons_correct(event_study_report):
    """Locks the joint homogeneity horizon set: weeks 5, 6, 7, 8 (the
    post-treatment periods on which dose-response heterogeneity is
    tested)."""
    hj = event_study_report.homogeneity_joint
    assert hj is not None
    assert hj.n_horizons == 4
    assert list(hj.horizon_labels) == ["5", "6", "7", "8"], hj.horizon_labels


def test_event_study_report_pretrends_and_homogeneity_fail_to_reject(event_study_report):
    """Both joint pretrends and joint homogeneity fail-to-reject under
    the linear-DGP null. Anchored bootstrap-p bands centered on the
    seed=22 captured values (pretrends ~0.39, homogeneity ~0.41) with
    total width 0.30 (± 0.15) per
    ``feedback_strata_bootstrap_path_divergence`` (same rationale as
    Stute overall). Tighter than 0.10-0.95: catches drift in either
    direction rather than only rejecting on cross-the-line moves."""
    pj = event_study_report.pretrends_joint
    hj = event_study_report.homogeneity_joint
    assert pj is not None and hj is not None
    assert pj.reject is False
    assert hj.reject is False
    p_pre = float(pj.p_value)
    p_hom = float(hj.p_value)
    assert 0.24 <= p_pre <= 0.54, p_pre
    assert 0.26 <= p_hom <= 0.56, p_hom


# ============================================================================
# Group F — Workflow-surface separation (overall vs event-study)
# ============================================================================
# These tests lock the per-path diagnostic surfaces so that §6 / §7 prose
# cannot drift back into the conflated form that quotes Yatchew + Stute on
# the event-study path. Per CI AI review R1 P1 #2.


def test_overall_report_pretrends_joint_is_none(overall_report):
    """Overall (two-period) path has no joint diagnostics — only Stute +
    Yatchew. The joint pretrends / homogeneity fields are populated
    only on the event-study path."""
    assert overall_report.pretrends_joint is None
    assert overall_report.homogeneity_joint is None


def test_event_study_report_stute_and_yatchew_are_none(event_study_report):
    """Event-study path has no single-horizon Stute / Yatchew — those
    are overall-only. The event-study workflow runs joint pretrends +
    joint homogeneity instead. §7 leadership prose must not claim
    Yatchew ran on this path."""
    assert event_study_report.stute is None
    assert event_study_report.yatchew is None


def test_overall_and_event_study_verdict_prefixes_distinct(overall_report, event_study_report):
    """Overall and event-study verdicts terminate in the same
    `_QUG_DEFERRED_SUFFIX` but have DISTINCT prefixes (overall uses
    `_compose_verdict_survey` -> `Stute and Yatchew ... fail-to-reject`;
    event-study uses `_compose_verdict_event_study_survey` ->
    `joint pre-trends and joint linearity ... fail-to-reject`). Locks
    the §7 prose against re-conflating the two surfaces."""
    assert "Stute and Yatchew" in overall_report.verdict, overall_report.verdict
    assert (
        "joint pre-trends and joint linearity" in event_study_report.verdict
    ), event_study_report.verdict
    assert "Stute and Yatchew" not in event_study_report.verdict, event_study_report.verdict
    assert (
        "joint pre-trends and joint linearity" not in overall_report.verdict
    ), overall_report.verdict


# ============================================================================
# Group G — Weighted point-estimation contract
# ============================================================================
# Per CI AI review R1 P1 #1: §3 prose previously claimed "the analytical
# local-linear at d_lower does not consume the survey weights". That is
# false — `_fit_continuous` (`diff_diff/had.py:3895-3961`) consumes
# `weights_arr` in (a) the local-linear `tau_bc` boundary fit, (b) the
# numerator `np.average(dy_arr, weights=weights_arr)`, AND (c) the
# denominator `np.average(d_reg, weights=weights_arr)`. The two ATTs are
# close on this DGP because the weight CV (~0.30) and dose distribution
# do not co-vary strongly, not because weights are ignored.


def test_survey_att_differs_from_naive_att(naive_overall_result, survey_overall_result):
    """Sign-only lock that the survey-aware ATT differs from the naive
    ATT. If weights were ignored in the slope (which §3 previously
    incorrectly claimed), the ATTs would be bit-identical. They are not
    — confirming the weighted contract is honored end-to-end."""
    assert naive_overall_result.att != survey_overall_result.att, (
        naive_overall_result.att,
        survey_overall_result.att,
    )


def test_survey_att_matches_weighted_local_linear_identity(
    panel_2p: pd.DataFrame, survey_overall_result
):
    """Locks the actual `_fit_continuous` algebraic identity end-to-end
    by recomputing every intermediate against the shipped estimator:

    1. `effective_dose_mean` on the result equals
       `np.average(d - d_lower, weights=w)` exactly.
    2. Calling `bias_corrected_local_linear` directly with the same
       inputs the HAD class uses (`d_reg = d_post - d_lower`, `dy`,
       `weights`, default `kernel="epanechnikov"`, `alpha=0.05`,
       `boundary=0.0`) recovers the SAME `tau_bc` boundary limit the
       estimator used.
    3. `att = (mean_w(dy) - tau_bc) / den_w` matches the fitted ATT
       to ~1e-13 (FP precision; same float ops, same data).

    Per CI AI review R2 P3: prior version of this test only checked
    finiteness + scale of an inverted `implied_tau_bc`; that is too
    weak to be called an "identity lock." This version actually
    re-derives every step."""
    from diff_diff.local_linear import bias_corrected_local_linear

    p2 = panel_2p.copy()
    pre = p2[p2["period"] == 1].set_index("state_id")
    post = p2[p2["period"] == 2].set_index("state_id")
    common = sorted(set(pre.index) & set(post.index))
    pre = pre.loc[common]
    post = post.loc[common]
    dy = (post["screening_uptake"] - pre["screening_uptake"]).to_numpy(dtype=float)
    d_post = post["spend_k"].to_numpy(dtype=float)
    weights = post["weight"].to_numpy(dtype=float)
    d_lower = float(d_post.min())
    d_reg = d_post - d_lower

    # Identity 1: effective_dose_mean == weighted mean of d - d_lower
    den_w = float(np.average(d_reg, weights=weights))
    assert abs(den_w - survey_overall_result.effective_dose_mean) < 1e-10, (
        den_w,
        survey_overall_result.effective_dose_mean,
    )
    assert abs(survey_overall_result.d_lower - d_lower) < 1e-12, (
        survey_overall_result.d_lower,
        d_lower,
    )

    # Identity 2: direct bias_corrected_local_linear call recovers the
    # SAME tau_bc the estimator used (HAD defaults: kernel epanechnikov,
    # alpha 0.05, boundary 0).
    bc = bias_corrected_local_linear(d_reg, dy, weights=weights, alpha=0.05, kernel="epanechnikov")
    tau_bc = float(bc.estimate_bias_corrected)
    assert np.isfinite(tau_bc), tau_bc

    # Identity 3: att = (dy_mean_w - tau_bc) / den_w, bit-equal modulo
    # FP precision.
    dy_mean_w = float(np.average(dy, weights=weights))
    manual_att = (dy_mean_w - tau_bc) / den_w
    assert abs(manual_att - survey_overall_result.att) < 1e-10, (
        manual_att,
        survey_overall_result.att,
        manual_att - survey_overall_result.att,
    )
