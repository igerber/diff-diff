"""Drift detection for Tutorial 19 (`docs/tutorials/19_dcdh_marketing_pulse.ipynb`).

The tutorial narrative quotes seed-specific numbers (overall_att, joiners,
leavers, event-study horizons, placebos). If library numerics drift
(estimator changes, RNG path changes, BLAS path changes), the prose can
go stale silently while `pytest --nbmake` still passes - it only checks
that the cells execute without error.

These asserts re-derive the same numbers using the locked DGP and seed
the notebook uses, then check them against the tolerance bands quoted in
the tutorial markdown. If a future change moves any number outside its
band, this test fails and a maintainer is forced to either update the
prose or investigate the methodology shift before merge.

DGP and seed locked at `_scratch/dcdh_tutorial/40_build_notebook.py`.
Quoted numbers derived from `_scratch/dcdh_tutorial/lock_seed.py`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from diff_diff import DCDH, generate_reversible_did_data

# Locked DGP parameters (must stay in sync with the notebook)
MAIN_SEED = 46
N_GROUPS = 60
N_PERIODS = 8
TREATMENT_EFFECT = 12.0
EFFECT_SD = 1.5


@pytest.fixture(scope="module")
def panel():
    raw = generate_reversible_did_data(
        n_groups=N_GROUPS,
        n_periods=N_PERIODS,
        pattern="single_switch",
        initial_treat_frac=0.4,
        treatment_effect=TREATMENT_EFFECT,
        heterogeneous_effects=True,
        effect_sd=EFFECT_SD,
        group_fe_sd=8.0,
        time_trend=0.5,
        noise_sd=2.0,
        seed=MAIN_SEED,
    )
    df = raw.rename(
        columns={
            "group": "market_id",
            "period": "week",
            "treatment": "promo_on",
            "outcome": "sessions",
        }
    )
    df["sessions"] = df["sessions"] + 100.0
    return df


@pytest.fixture(scope="module")
def phase1_results(panel):
    """Phase 1 fit: gets joiners/leavers split. placebo=False to skip the
    documented NaN-SE warning on the single-lag placebo path."""
    model = DCDH(twfe_diagnostic=False, placebo=False, seed=42)
    return model.fit(
        panel,
        outcome="sessions",
        unit="market_id",
        time="week",
        treatment="promo_on",
    )


@pytest.fixture(scope="module")
def event_study_results(panel):
    """Event-study fit: L_max=2 + multiplier bootstrap. The A7
    UserWarning is intentionally muted here so the fixture is quiet
    for the value-checking tests below; the notebook's actual
    warning-policy contract (A7 visible, only matmul filtered) is
    validated separately by `test_event_study_warning_policy_matches_notebook`."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*encountered in matmul",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Assumption 7 .* is violated: leavers present",
            category=UserWarning,
        )
        model = DCDH(twfe_diagnostic=False, placebo=True, n_bootstrap=199, seed=42)
        return model.fit(
            panel,
            outcome="sessions",
            unit="market_id",
            time="week",
            treatment="promo_on",
            L_max=2,
        )


def test_panel_composition(panel):
    """The narrative quotes 38 joiners and 22 leavers in the stakeholder
    template. If the DGP drifts, those counts shift and the template
    text goes stale."""
    counts = panel.groupby("switcher_type").size().to_dict()
    assert counts.get("joiner") == 38, counts
    assert counts.get("leaver") == 22, counts


def test_overall_att_close_to_truth(phase1_results):
    """Section 3 quotes 'about 12 sessions' headline (true effect = 12)."""
    assert 11.7 <= phase1_results.overall_att <= 12.4, phase1_results.overall_att


def test_overall_ci_covers_truth(phase1_results):
    """Section 3 narrative claims the CI covers the true effect of 12."""
    ci_low, ci_high = phase1_results.overall_conf_int
    assert ci_low <= TREATMENT_EFFECT <= ci_high, (ci_low, ci_high)


def test_overall_ci_endpoints_match_quoted(phase1_results):
    """Section 3 narrative quotes '95% CI: 11.3 to 12.8'. Pin the
    one-decimal display exactly so any drift past the displayed
    rounding fails this test."""
    ci_low, ci_high = phase1_results.overall_conf_int
    assert round(ci_low, 1) == 11.3, ci_low
    assert round(ci_high, 1) == 12.8, ci_high


def test_joiners_leavers_consistent(phase1_results):
    """Section 3 narrative quotes joiners ~12.1 and leavers ~11.9, both
    positive and within sampling uncertainty of each other."""
    assert 11.5 <= phase1_results.joiners_att <= 12.7, phase1_results.joiners_att
    assert 11.4 <= phase1_results.leavers_att <= 12.5, phase1_results.leavers_att
    # Both positive and similar in magnitude (no big disagreement)
    assert abs(phase1_results.joiners_att - phase1_results.leavers_att) < 1.5


def test_event_study_horizons_cover_truth(event_study_results):
    """Section 4 narrative quotes l=1 ~12.4, l=2 ~12.6, both with CIs
    covering the true effect of 12."""
    es = event_study_results.event_study_effects
    for l in (1, 2):
        eff = es[l]["effect"]
        ci = es[l]["conf_int"]
        assert 11.5 <= eff <= 13.3, (l, eff)
        assert ci[0] <= TREATMENT_EFFECT <= ci[1], (l, ci)


def test_event_study_ci_endpoints_match_quoted(event_study_results):
    """Section 4 narrative quotes l=1 CI [11.4, 13.3] and l=2 CI
    [11.5, 13.6]. These are bootstrap-based CIs and the bootstrap RNG
    path differs between Rust and pure-Python backends (per the
    bit-identity-baseline-per-backend convention), so we use a 0.15
    tolerance band rather than `round(_, 1) ==` exact matching - tight
    enough to catch real prose drift, loose enough to absorb the
    documented backend variance."""
    es = event_study_results.event_study_effects
    # l=1 CI [11.4, 13.3]
    assert abs(es[1]["conf_int"][0] - 11.4) < 0.15, es[1]["conf_int"]
    assert abs(es[1]["conf_int"][1] - 13.3) < 0.15, es[1]["conf_int"]
    # l=2 CI [11.5, 13.6]
    assert abs(es[2]["conf_int"][0] - 11.5) < 0.15, es[2]["conf_int"]
    assert abs(es[2]["conf_int"][1] - 13.6) < 0.15, es[2]["conf_int"]


def test_event_study_significance(event_study_results):
    """Section 5 stakeholder template claims 'bootstrap p < 0.01 at both
    post-treatment horizons'. Lock that significance threshold."""
    es = event_study_results.event_study_effects
    assert es[1]["p_value"] < 0.01, es[1]["p_value"]
    assert es[2]["p_value"] < 0.01, es[2]["p_value"]


def test_placebo_horizons_cover_zero(event_study_results):
    """Section 4 narrative claims pre-treatment placebos sit on zero."""
    pl = event_study_results.placebo_event_study
    assert pl is not None
    for l in (-1, -2):
        eff = pl[l]["effect"]
        ci = pl[l]["conf_int"]
        assert abs(eff) < 0.7, (l, eff)
        assert ci[0] <= 0.0 <= ci[1], (l, ci)


def test_assumption7_warning_fires_as_expected(panel):
    """The notebook surfaces and explains the A7 warning. If the library
    stops firing it, the markdown explanation goes stale and we should
    notice."""
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            model = DCDH(twfe_diagnostic=False, placebo=True, n_bootstrap=49, seed=42)
            model.fit(
                panel,
                outcome="sessions",
                unit="market_id",
                time="week",
                treatment="promo_on",
                L_max=2,
            )
    a7_warnings = [
        w
        for w in ws
        if w.category is UserWarning
        and "Assumption 7" in str(w.message)
        and "leavers present" in str(w.message)
    ]
    assert len(a7_warnings) >= 1, [str(w.message)[:80] for w in ws]


def test_event_study_warning_policy_matches_notebook(panel):
    """Mirror the notebook's exact warning policy on the visible
    event-study fit and assert the resulting warning set matches the
    documented contract: exactly one UserWarning (the A7 leavers-present
    warning that the notebook's markdown explains), and zero
    RuntimeWarnings (matmul-pattern ones filtered; everything else
    surfaces). If the library starts emitting an unexpected warning on
    this code path, this test fails and the notebook prose may need to
    be updated."""
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        # MIRROR the notebook's narrow filter exactly (no np.errstate, no
        # blanket A7 suppression).
        warnings.filterwarnings(
            "ignore",
            message=r".*encountered in matmul",
            category=RuntimeWarning,
        )
        model = DCDH(twfe_diagnostic=False, placebo=True, n_bootstrap=199, seed=42)
        model.fit(
            panel,
            outcome="sessions",
            unit="market_id",
            time="week",
            treatment="promo_on",
            L_max=2,
        )
    user_warnings = [w for w in ws if w.category is UserWarning]
    runtime_warnings = [w for w in ws if w.category is RuntimeWarning]
    # Exactly one UserWarning, and it's the documented A7 warning.
    assert len(user_warnings) == 1, [str(w.message)[:120] for w in user_warnings]
    msg = str(user_warnings[0].message)
    assert "Assumption 7" in msg, msg
    assert "leavers present" in msg, msg
    # All RuntimeWarnings should be the matmul pattern (filtered) - so
    # zero remaining. If a new RuntimeWarning fires from somewhere else,
    # this fails.
    assert len(runtime_warnings) == 0, [str(w.message)[:120] for w in runtime_warnings]


def test_a11_warning_does_not_fire():
    """The notebook claims this seed/DGP is in the A11-clean regime
    (no warning fires). If a library change starts triggering A11 on
    this panel, the prose claim is wrong."""
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            raw = generate_reversible_did_data(
                n_groups=N_GROUPS,
                n_periods=N_PERIODS,
                pattern="single_switch",
                initial_treat_frac=0.4,
                treatment_effect=TREATMENT_EFFECT,
                heterogeneous_effects=True,
                effect_sd=EFFECT_SD,
                group_fe_sd=8.0,
                time_trend=0.5,
                noise_sd=2.0,
                seed=MAIN_SEED,
            )
            df = raw.rename(
                columns={
                    "group": "market_id",
                    "period": "week",
                    "treatment": "promo_on",
                    "outcome": "sessions",
                }
            )
            df["sessions"] = df["sessions"] + 100.0
            DCDH(twfe_diagnostic=False, placebo=False, seed=42).fit(
                df,
                outcome="sessions",
                unit="market_id",
                time="week",
                treatment="promo_on",
            )
    a11_warnings = [
        w for w in ws if w.category is UserWarning and "Assumption 11" in str(w.message)
    ]
    assert len(a11_warnings) == 0, [str(w.message)[:80] for w in a11_warnings]
