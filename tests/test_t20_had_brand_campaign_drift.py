"""Drift detection for Tutorial 20 (`docs/tutorials/20_had_brand_campaign.ipynb`).

The tutorial narrative quotes seed-specific numbers (overall WAS_d_lower,
design auto-detection, per-week event-study coverage). If library numerics
drift (estimator changes, RNG path changes, BLAS path changes), the prose
can go stale silently while `pytest --nbmake` still passes - it only
checks that the cells execute without error.

These asserts re-derive the same numbers using the locked DGP and seed
the notebook uses, then check them against the values quoted in the
tutorial markdown. If a future change moves any number outside its
tolerance band, this test fails and a maintainer is forced to either
update the prose or investigate the methodology shift before merge.

DGP and seed locked at `_scratch/had_tutorial/40_build_notebook.py`.
Quoted numbers derived from `_scratch/had_tutorial/20_assemble_outputs.py`.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from diff_diff import HAD, generate_continuous_did_data

# Locked DGP parameters (must stay in sync with the notebook).
MAIN_SEED = 87
N_UNITS = 60
N_PERIODS = 8
COHORT_PERIOD = 5
TRUE_SLOPE = 100.0
BASELINE_VISITS = 5000.0
DOSE_LOW = 5.0
DOSE_HIGH = 50.0


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
    panel = raw.copy()
    panel.loc[panel["period"] < panel["first_treat"], "dose"] = 0.0
    panel = panel.rename(
        columns={
            "unit": "dma_id",
            "period": "week",
            "outcome": "weekly_visits",
            "dose": "regional_spend_k",
        }
    )
    panel["weekly_visits"] = panel["weekly_visits"] + BASELINE_VISITS
    return panel


@pytest.fixture(scope="module")
def panel_2pd(panel):
    p = panel.copy()
    p["period"] = (p["week"] >= COHORT_PERIOD).astype(int) + 1
    pre_post = p.groupby(["dma_id", "period"], as_index=False).agg(
        weekly_visits=("weekly_visits", "mean"),
        regional_spend_k=("regional_spend_k", "mean"),
    )
    return pd.DataFrame(pre_post)


@pytest.fixture(scope="module")
def overall_result(panel_2pd):
    """HAD overall WAS_d_lower fit on the 2-period collapsed panel.
    Filters the documented Assumption 5/6 advisory at fit time so the
    test output is focused on the value pins, not warning noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*continuous_near_d_lower.*Assumption.*",
            category=UserWarning,
        )
        est = HAD(design="auto")
        return est.fit(
            panel_2pd,
            outcome="weekly_visits",
            dose="regional_spend_k",
            time="period",
            unit="dma_id",
        )


@pytest.fixture(scope="module")
def event_study_result(panel):
    """HAD event-study fit on the original 8-week panel."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*continuous_near_d_lower.*Assumption.*",
            category=UserWarning,
        )
        est = HAD(design="auto")
        return est.fit(
            panel,
            outcome="weekly_visits",
            dose="regional_spend_k",
            time="week",
            unit="dma_id",
            first_treat="first_treat",
            aggregate="event_study",
        )


def test_panel_composition(panel):
    """Section 2 narrative quotes 60 DMAs over 8 weeks, with regional
    spend ranging from a $5K floor to $50K and every DMA participating
    (no DMA at $0). The Section 5 stakeholder template additionally
    quotes 'median ~$25K' for the spend distribution. If the DGP
    drifts, this surfaces."""
    assert panel["dma_id"].nunique() == N_UNITS
    assert panel["week"].nunique() == N_PERIODS
    post_doses = (
        panel.loc[panel["week"] >= COHORT_PERIOD].groupby("dma_id")["regional_spend_k"].first()
    )
    assert post_doses.min() >= DOSE_LOW, post_doses.min()
    assert post_doses.max() <= DOSE_HIGH, post_doses.max()
    assert (post_doses == 0.0).sum() == 0, (
        "No DMA should be at exactly $0 - the DGP framing is 'every DMA "
        "got some regional spend'. If a DMA appears at zero the "
        "Section 1/3 narrative is wrong."
    )
    # Pin the sample median so the README/template "median ~$25K" prose
    # cannot drift unnoticed (PR #394 R2 P3 fix).
    assert round(post_doses.median(), 1) == 24.8, post_doses.median()


def test_overall_design_auto_detection(overall_result):
    """Section 3 narrative claims HAD picked the
    `continuous_near_d_lower` regime (Design 1) because all DMAs have
    positive spend, with target `WAS_d_lower`. If HAD's design auto-
    detection changes, this test surfaces it."""
    assert overall_result.design == "continuous_near_d_lower", overall_result.design
    assert overall_result.target_parameter == "WAS_d_lower", overall_result.target_parameter
    assert round(overall_result.d_lower, 1) == 5.2, overall_result.d_lower


def test_overall_att_close_to_truth(overall_result):
    """Section 3 narrative quotes 'about 100 weekly visits per DMA per
    $1K above the boundary' as the headline lift. Pin the one-decimal
    display exactly. HAD's analytical SE path is bit-identical
    regardless of backend env (no Rust kernel involved on HAD), so a
    tight pin is appropriate."""
    assert round(overall_result.att, 1) == 100.0, overall_result.att


def test_overall_se_matches_quoted(overall_result):
    """Section 3 implicitly quotes the CI [98.6, 101.4] which depends
    on the SE = 0.7. Pin both for any drift in the local-linear
    bandwidth selector or bias-correction path."""
    assert round(overall_result.se, 1) == 0.7, overall_result.se


def test_overall_ci_endpoints_match_quoted(overall_result):
    """Section 3 narrative quotes '95% CI: 98.6 to 101.4'. Pin the
    one-decimal display exactly."""
    ci_low, ci_high = overall_result.conf_int
    assert round(ci_low, 1) == 98.6, ci_low
    assert round(ci_high, 1) == 101.4, ci_high


def test_overall_ci_covers_truth(overall_result):
    """Section 3 narrative claims the CI brackets the true per-$1K
    slope of 100."""
    ci_low, ci_high = overall_result.conf_int
    assert ci_low <= TRUE_SLOPE <= ci_high, (ci_low, ci_high)


def test_overall_dose_mean_matches_quoted(overall_result):
    """Section 5 stakeholder template quotes 'median ~$25K' for the
    spend distribution. The dose mean (D-bar) tracks the median for a
    uniform distribution; pin to one decimal."""
    assert round(overall_result.dose_mean, 1) == 24.7, overall_result.dose_mean


def test_overall_n_units(overall_result):
    """60 DMAs total; 59 above d_lower (the lightest-touch DMA sits at
    d_lower itself)."""
    assert overall_result.n_obs == 60, overall_result.n_obs
    assert overall_result.n_treated == 59, overall_result.n_treated


def test_event_study_horizons_complete(event_study_result):
    """Verify the expected pre-launch placebo horizons (e=-4,-3,-2)
    AND post-launch horizons (e=0..3) are ALL present. Skipping the
    presence check upstream lets per-horizon assertions silently pass
    on truncated event_times - which would let the tutorial lose
    promised rows undetected (PR #394 R1 P2 fix)."""
    event_times = list(event_study_result.event_times)
    expected = [-4, -3, -2, 0, 1, 2, 3]
    for e in expected:
        assert e in event_times, (
            f"Expected event-time {e} missing from event_times="
            f"{event_times}; the tutorial narrative quotes this horizon."
        )


def test_event_study_post_horizons_cover_truth(event_study_result):
    """Section 4 narrative claims 'per-week post-launch effects all
    hover right around 100 visits per $1K with overlapping 95% CIs and
    lower bounds well above zero'. Verify each post-launch horizon's
    CI covers the true slope of 100. event_times is integer-indexed,
    not label-indexed - map labels to positions."""
    event_times = list(event_study_result.event_times)
    ci_lows = list(event_study_result.conf_int_low)
    ci_highs = list(event_study_result.conf_int_high)
    for e in (0, 1, 2, 3):
        i = event_times.index(e)
        assert ci_lows[i] <= TRUE_SLOPE <= ci_highs[i], (
            e,
            ci_lows[i],
            ci_highs[i],
        )


def test_event_study_post_horizons_remain_positive(event_study_result):
    """Section 4 narrative claims all post-launch CI lower bounds are
    'well above zero' and the per-dollar lift is 'stable across all
    four weeks of the campaign'."""
    event_times = list(event_study_result.event_times)
    ci_lows = list(event_study_result.conf_int_low)
    for e in (0, 1, 2, 3):
        i = event_times.index(e)
        assert ci_lows[i] > 0, (e, ci_lows[i])


def test_event_study_post_atts_close_to_truth(event_study_result):
    """All four post-launch per-week WAS_d_lower estimates should be
    within ±0.5 of the headline overall estimate (~100) under linear
    ATT."""
    event_times = list(event_study_result.event_times)
    atts = list(event_study_result.att)
    for e in (0, 1, 2, 3):
        i = event_times.index(e)
        assert abs(atts[i] - 100.0) < 0.5, (e, atts[i])


def test_event_study_pre_placebos_cover_zero(event_study_result):
    """Section 4 narrative claims pre-launch placebos (e=-2,-3,-4) sit
    at essentially zero (within ±0.06) with 95% CIs comfortably
    bracketing zero. Presence of these horizons is verified separately
    by `test_event_study_horizons_complete` so we can reach into the
    arrays without an `if e in event_times` guard that would silently
    skip a missing horizon (PR #394 R1 P2 fix). Magnitude pinned to
    < 0.1 to lock the prose claim of 'within ±0.06' with light slack
    (PR #394 R2 P3 fix)."""
    event_times = list(event_study_result.event_times)
    atts = list(event_study_result.att)
    ci_lows = list(event_study_result.conf_int_low)
    ci_highs = list(event_study_result.conf_int_high)
    for e in (-2, -3, -4):
        i = event_times.index(e)
        assert abs(atts[i]) < 0.1, (e, atts[i])
        assert ci_lows[i] <= 0.0 <= ci_highs[i], (e, ci_lows[i], ci_highs[i])


# =============================================================================
# Notebook-narrative cross-check
# =============================================================================
#
# Mirror of the T21 notebook cross-check (PR #409 holistic re-audit).
# The asserts above re-derive numbers from the locked DGP+seed but
# never verify that the rendered T20 tutorial actually quotes those
# same numbers. Because ``nbsphinx_execute = "never"`` in
# ``docs/conf.py``, CI cannot detect drift between the pinned drift
# constants and the committed tutorial via notebook re-execution.
# Use the shared helper from ``tests/_tutorial_drift.py`` to assert
# each load-bearing quote is present in the rendered notebook
# surface (markdown prose + executed output cells).


T20_NOTEBOOK = "docs/tutorials/20_had_brand_campaign.ipynb"


def test_notebook_quotes_match_pinned_constants():
    """Every load-bearing T20 verdict / number must appear verbatim
    in the rendered notebook (markdown prose + executed output cells).

    Closes the same gap fixed for T21 in the PR #409 holistic
    re-audit: the file-level docstring claimed "check against the
    values quoted in the tutorial markdown" but every prior assert
    re-derived numbers from the DGP and compared them to a hardcoded
    constant, leaving the notebook completely uncross-checked.
    """
    from tests._tutorial_drift import assert_quotes_in_rendered

    expected_quotes = [
        # Headline WAS estimate quoted in cell 10 narrative.
        "100 weekly visits",
        # CI quoted alongside the headline.
        "98.6 to 101.4",
        # Design auto-detect outcome.
        "continuous_near_d_lower",
        # Target parameter label used across cells 8 / 12 / 13.
        "WAS_d_lower",
        # Placebo-magnitude prose claim (locked analytically above by
        # test_event_study_pre_atts_near_zero with the ±0.1 envelope).
        "±0.06",
        # Sample-summary phrase in the design-fit narrative. Use the
        # exact tilde-prefixed form so a future drift in the sentence
        # (e.g. "median around $25K") would surface here.
        "median ~$25K",
    ]
    assert_quotes_in_rendered(T20_NOTEBOOK, expected_quotes, surface="rendered")
