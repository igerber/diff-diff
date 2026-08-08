"""Drift detection for Tutorial 24
(``docs/tutorials/24_staggered_vs_collapsed_power.ipynb``).

The tutorial is a power-analysis decision guide whose narrative rests on two
quantitative claims (the ones the deferred-TODO row names explicitly):

1. **Monotonic dilution, fast -> slow.** Collapsing a staggered rollout to a 2x2
   counts not-yet-live states as "post", diluting the estimate. The dilution
   ``E2 / E1`` shrinks monotonically as the rollout staggers (the committed table
   reads ``93.5% / 80.9% / 61.8%`` of the truth for fast / moderate / slow), and
   the 2x2's CI coverage of the true effect-on-treated collapses with it
   (``0.80 / 0.18 / 0.004``) while Callaway-Sant'Anna (CS) stays near nominal.
2. **CS-vs-2x2 MDE crossover / near-parity at slow rollout.** For pure detection
   the 2x2 is more powerful at a fast rollout, but its MDE *climbs* (~0.37 ->
   ~0.60) as the rollout staggers while CS's barely moves (~0.55) -- so the power
   gap closes to near parity (CS even edges ahead at slow; the prose flags that
   exact reversal as simulation-sensitive, so we pin only the robust direction).

``pytest --nbmake`` only checks the cells *execute*; it does not check the prose.
Because ``nbsphinx_execute = "never"`` (``docs/conf.py``), the committed outputs
are what RTD renders, so the numbers can silently drift from the live library.
These asserts re-derive the load-bearing numbers from the same public generator
(``generate_staggered_data``) + estimators the tutorial uses and check them
against the committed surface. If library numerics drift, the re-derivation
fails; if the notebook is edited without updating these constants (or vice
versa), the rendered-surface quote cross-check fails.

Structure (mirrors T25's split):
- The **deterministic** claims (dilution is read off the noise-free
  ``true_effect`` column; single seed-0 fits) run unmarked -- cheap, exercised in
  every CI leg including the pure-Python fallback.
- The **Monte Carlo** sweeps (coverage collapse, the MDE crossover, the
  flat-vs-growing estimand targeting) are ``@pytest.mark.slow`` so they stay off
  the ~1h pure-Python budget and run in the Rust legs (``-m ''``). They assert
  robust *orderings* with wide margins, not exact pins.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from diff_diff import (
    CallawaySantAnna,
    DifferenceInDifferences,
    generate_staggered_data,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/24_staggered_vs_collapsed_power.ipynb"

# Locked design -- must stay in sync with the notebook's first code cell
# (cross-checked by ``test_notebook_kwargs_match``).
N_STATES, N_WEEKS, HOLDOUT, GROWTH, NOISE = 50, 24, 0.20, 0.05, 1.0
ROLLOUTS = {"fast": (8, 9, 10), "moderate": (8, 10, 12, 14), "slow": (8, 12, 16, 20)}


# --------------------------------------------------------------------------- #
# Helpers -- faithful copies of the notebook's, so the test re-derives exactly
# what the prose describes. ``test_notebook_kwargs_match`` guards the kwargs.
# --------------------------------------------------------------------------- #
def make_panel(cohorts, effect, seed, *, n_states=N_STATES, growth=GROWTH):
    return generate_staggered_data(
        n_units=n_states,
        n_periods=N_WEEKS,
        cohort_periods=list(cohorts),
        never_treated_frac=HOLDOUT,
        treatment_effect=effect,
        dynamic_effects=True,
        effect_growth=growth,
        unit_fe_sd=2.0,
        time_trend=0.1,
        noise_sd=NOISE,
        seed=seed,
        panel=True,
    )


def collapse_2x2(panel, rollout_start, tail_start=None):
    d = panel.copy()
    if tail_start is None:
        d["post"] = (d["period"] >= rollout_start).astype(int)
    else:
        d = d[(d["period"] < rollout_start) | (d["period"] >= tail_start)].copy()
        d["post"] = (d["period"] >= tail_start).astype(int)
    return d.groupby(["unit", "post"], as_index=False).agg(
        outcome=("outcome", "mean"), treated=("treat", "max")
    )


def fit_2x2(panel, rollout_start, tail_start=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = DifferenceInDifferences(cluster="unit").fit(
            collapse_2x2(panel, rollout_start, tail_start),
            outcome="outcome",
            treatment="treated",
            post="post",
        )
    return r.att, r.se, r.conf_int, bool(r.p_value < 0.05)


def fit_cs(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = CallawaySantAnna(control_group="never_treated").fit(
            panel,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
    return r.overall_att, r.overall_se, r.overall_conf_int, bool(r.overall_p_value < 0.05)


def estimands(panel, rollout_start):
    """E1 = effect-on-treated (CS's target); E2 = E1 diluted by not-yet-live
    states counted as 'post' (the 2x2's target). Read off the noise-free
    ``true_effect`` column -> deterministic, no fitting, no BLAS variation."""
    on = panel["treated"] == 1
    E1 = float(panel.loc[on, "true_effect"].mean())
    grp = panel[panel["treat"] == 1]
    post = grp["period"] >= rollout_start
    E2 = float(grp.loc[post, "true_effect"].mean())
    return E1, E2


def mde(lifts, powers, target=0.80):
    x, y = np.asarray(lifts, float), np.asarray(powers, float)
    if (y >= target).all():
        return float(x[0])
    if (y < target).all():
        return float("nan")
    i = int(np.argmax(y >= target))
    t = (target - y[i - 1]) / (y[i] - y[i - 1])
    return float(x[i - 1] + t * (x[i] - x[i - 1]))


def paired_power(cohorts, lift, n_sims, *, growth=GROWTH, tail=False, seed0=2000):
    """Fraction of paired panels where each estimator rejects H0: no effect.
    ``tail=True`` runs the clean-all-treated-tail 2x2 (drops the rollout window)."""
    rs, ts = min(cohorts), max(cohorts)
    rej2, rejc = [], []
    for i in range(n_sims):
        p = make_panel(cohorts, lift, seed0 + i, growth=growth)
        rej2.append(fit_2x2(p, rs, tail_start=ts if tail else None)[3])
        rejc.append(fit_cs(p)[3])
    return np.mean(rej2), np.mean(rejc)


# --------------------------------------------------------------------------- #
# Deterministic claims (unmarked -- cheap, run everywhere)
# --------------------------------------------------------------------------- #
def test_panel_composition():
    """Scenario cell: 50 states x 24 weeks = 1200 rows; 40 treated / 10 never."""
    panel = make_panel(ROLLOUTS["moderate"], effect=2.0, seed=0)
    assert len(panel) == N_STATES * N_WEEKS == 1200
    assert panel["unit"].nunique() == N_STATES == 50
    assert panel["period"].nunique() == N_WEEKS == 24
    # ``treat`` is the unit-constant ever-treated indicator (==1 for all of a
    # treated state's rows), so unique treated unit ids count the treated states.
    treated_units = int(panel.loc[panel["treat"] == 1, "unit"].nunique())
    assert treated_units == 40, "expected 40 treated states"
    assert panel["unit"].nunique() - treated_units == 10, "expected 10 never-treated states"


def test_section1_estimand_gap():
    """Section 1 (committed output): on the moderate seed-0 panel,
    E1=2.61, E2=2.10 (E2<E1, dilution); 2x2=2.17 < CS=2.67; '2x2 reports ~81%'.

    Estimands are exact (means of the noise-free column); the fitted ATTs get
    bands to absorb cross-platform BLAS variation."""
    panel = make_panel(ROLLOUTS["moderate"], effect=2.0, seed=0)
    rs = min(ROLLOUTS["moderate"])
    E1, E2 = estimands(panel, rs)
    assert 2.50 <= E1 <= 2.72, f"E1={E1:.3f} outside [2.50, 2.72]"
    assert 2.00 <= E2 <= 2.22, f"E2={E2:.3f} outside [2.00, 2.22]"
    assert E2 < E1, "the 2x2 target must be diluted below the effect-on-treated"

    a2, _, ci2, _ = fit_2x2(panel, rs)
    ac, _, cic, _ = fit_cs(panel)
    assert 2.00 <= a2 <= 2.35, f"2x2 att={a2:.3f} outside [2.00, 2.35]"
    assert 2.50 <= ac <= 2.85, f"CS att={ac:.3f} outside [2.50, 2.85]"
    assert a2 < ac, "the naive 2x2 must read below CS (it answers the diluted question)"
    assert ci2[0] <= a2 <= ci2[1] and cic[0] <= ac <= cic[1]
    ratio = 100 * a2 / ac
    assert 76 <= ratio <= 86, f"2x2 reports {ratio:.0f}% of CS (quoted ~81%)"


def test_dilution_monotonic_structural():
    """Section 1 dilution table (committed): '2x2 reports % of truth' falls
    monotonically 93.5 -> 80.9 -> 61.8 as the rollout staggers. These are
    structural (E2/E1 over the display seeds 900-904), so they reproduce
    deterministically; bands only guard against RNG-stream version shifts."""
    pct = {}
    e1s = {}
    for name, coh in ROLLOUTS.items():
        rs = min(coh)
        pairs = [estimands(make_panel(coh, 2.0, 900 + s), rs) for s in range(5)]
        E1 = float(np.mean([p[0] for p in pairs]))
        E2 = float(np.mean([p[1] for p in pairs]))
        pct[name] = 100 * E2 / E1
        e1s[name] = E1

    # The headline: dilution worsens monotonically with stagger, with clear gaps.
    assert pct["fast"] > pct["moderate"] > pct["slow"], f"non-monotonic dilution: {pct}"
    assert pct["fast"] - pct["moderate"] > 5, f"fast/moderate gap too small: {pct}"
    assert pct["moderate"] - pct["slow"] > 8, f"moderate/slow gap too small: {pct}"
    # Banded pins on each quoted percentage.
    assert 88 <= pct["fast"] <= 97, f"fast={pct['fast']:.1f}% (quoted 93.5)"
    assert 76 <= pct["moderate"] <= 86, f"moderate={pct['moderate']:.1f}% (quoted 80.9)"
    assert 55 <= pct["slow"] <= 68, f"slow={pct['slow']:.1f}% (quoted 61.8)"
    # E1 (effect-on-treated) is ~stable across speeds -- it's the *undiluted* target.
    for name, E1 in e1s.items():
        assert 2.45 <= E1 <= 2.75, f"E1[{name}]={E1:.3f} drifted from ~2.6"


def test_rendered_surface_quotes():
    """Cross-check: the load-bearing numbers these constants assume are rendered
    verbatim in the committed notebook (markdown prose + executed outputs). If
    the notebook is re-executed and a number moves, or the prose is hand-edited,
    this surfaces the divergence between the tutorial and the pinned values."""
    assert_quotes_in_rendered(
        NB,
        [
            # Section 1 dilution (table output + prose)
            "93.500",
            "80.876",
            "61.754",
            "94% / 80% / 61%",
            "~60% of the true lift",
            "0.796",  # 2x2 coverage, fast
            "0.004",  # 2x2 coverage, slow ("almost never")
            "0.924",  # CS coverage, fast (~95% regardless of speed)
            "~81%",  # "2x2 reports ~81% of CS's number"
            # Section 2 headline: paired power + the MDE crossover prose
            "2x2 power = 0.90",
            "CS power = 0.75",
            "~0.37",  # 2x2 MDE, fast
            "~0.60",  # 2x2 MDE, slow (climbs)
            "~0.55",  # CS MDE (barely moves)
            # Refinements: flat-vs-growing targeting + flat-effects power
            "naive-2x2= 1.62",
            "clean-tail-2x2= 1.98",
            "MDE clean-tail 2x2 = 0.52",
            "MDE CS = 0.73",
        ],
    )


def test_headline_claims_present_in_prose():
    """The two named claims must remain stated in the markdown narrative, so a
    prose rewrite can't quietly drop them while the numbers still render."""
    md = notebook_markdown(NB)
    assert "diluted" in md, "dilution framing dropped from the narrative"
    assert "near parity" in md, "MDE near-parity / crossover claim dropped"
    assert "MDE" in md and "staggered" in md


def test_notebook_kwargs_match():
    """Sync guard: the notebook's design constants + generator/fit kwargs must
    match this module's locked config, so a notebook-only edit can't silently
    invalidate the re-derived numbers. Skips cleanly when ``docs/`` is absent
    (the Rust / isolated-install CI jobs copy only ``tests/``)."""
    import json
    from pathlib import Path

    nb_path = Path(__file__).resolve().parents[1] / NB
    if not nb_path.exists():
        pytest.skip(f"Notebook not found at {nb_path}; sync guard is local-dev only.")
    nb = json.loads(nb_path.read_text())
    src = "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in nb["cells"]
        if c["cell_type"] == "code"
    )
    for needle in (
        "N_STATES, N_WEEKS, HOLDOUT, GROWTH, NOISE = 50, 24, 0.20, 0.05, 1.0",
        '"fast": (8, 9, 10)',
        '"moderate": (8, 10, 12, 14)',
        '"slow": (8, 12, 16, 20)',
        "never_treated_frac=HOLDOUT",
        "effect_growth=growth",
        'DifferenceInDifferences(cluster="unit")',
        'CallawaySantAnna(control_group="never_treated")',
    ):
        assert needle in src, f"notebook drifted from locked config: {needle!r}"


# --------------------------------------------------------------------------- #
# Monte Carlo claims (slow -- Rust legs at full count; off the pure-Python budget)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_dilution_coverage_collapse():
    """Section 1 (committed): the 2x2's 95% CI coverage of the true
    effect-on-treated collapses as the rollout staggers (0.80 -> 0.18 -> 0.004),
    while CS stays near nominal (~0.92-0.96) regardless of speed. We assert the
    robust ordering + wide margins (not the exact rates)."""
    n_sims = 60
    cov2, covc = {}, {}
    for name, coh in ROLLOUTS.items():
        rs = min(coh)
        c2, cc = [], []
        for i in range(n_sims):
            p = make_panel(coh, 2.0, 1000 + i)
            e1_i = estimands(p, rs)[0]  # this panel's own truth
            _, _, ci2, _ = fit_2x2(p, rs)
            c2.append(ci2[0] <= e1_i <= ci2[1])
            _, _, cic, _ = fit_cs(p)
            cc.append(cic[0] <= e1_i <= cic[1])
        cov2[name], covc[name] = float(np.mean(c2)), float(np.mean(cc))

    assert cov2["fast"] > cov2["moderate"] > cov2["slow"], f"2x2 coverage not collapsing: {cov2}"
    assert cov2["slow"] < 0.10, f"2x2 slow coverage {cov2['slow']:.3f} not ~0 (quoted .004)"
    assert cov2["fast"] > 0.40, f"2x2 fast coverage {cov2['fast']:.3f} unexpectedly low"
    # CS covers the truth ~95% of the time regardless of speed.
    for name in ROLLOUTS:
        assert covc[name] > 0.75, f"CS coverage at {name}={covc[name]:.3f} not near nominal"
    # Where dilution bites, CS is the clearly-better-covered estimator.
    assert covc["moderate"] > cov2["moderate"] and covc["slow"] > cov2["slow"]


@pytest.mark.slow
def test_mde_crossover():
    """Section 2 headline (committed plot + prose): as the rollout staggers the
    2x2's MDE *climbs* (~0.37 -> ~0.60) while CS's barely moves (~0.55), so the
    power gap closes from "2x2 clearly wins" (fast) to near-parity / a slight
    crossover (slow). Robust direction only -- the exact reversal is
    simulation-sensitive (the prose says so)."""
    lifts = [0.20, 0.35, 0.50, 0.65, 0.85]
    n_sims = 60
    res = {}
    for name in ("fast", "slow"):
        coh = ROLLOUTS[name]
        pw = np.array([paired_power(coh, L, n_sims) for L in lifts])  # cols: 2x2, CS
        res[name] = (mde(lifts, pw[:, 0]), mde(lifts, pw[:, 1]))
    (m2_fast, mc_fast), (m2_slow, mc_slow) = res["fast"], res["slow"]

    mdes = (m2_fast, mc_fast, m2_slow, mc_slow)
    assert all(np.isfinite(v) for v in mdes), f"an MDE did not bracket 0.80: {res}"
    # The 2x2's MDE climbs with stagger; CS's barely moves.
    assert m2_slow - m2_fast > 0.10, f"2x2 MDE didn't climb: {m2_fast:.3f}->{m2_slow:.3f}"
    assert abs(mc_slow - mc_fast) < 0.20, f"CS MDE not ~flat: {mc_fast:.3f}->{mc_slow:.3f}"
    # Fast rollout: the 2x2 is clearly the more sensitive detector.
    assert mc_fast - m2_fast > 0.04, f"2x2 not ahead at fast: {m2_fast:.3f} vs {mc_fast:.3f}"
    # Slow rollout: the gap has closed to parity (the 2x2's edge is gone).
    assert m2_slow > mc_slow - 0.10, f"2x2 still ahead at slow: {m2_slow:.3f} vs {mc_slow:.3f}"


@pytest.mark.slow
def test_estimand_targeting_flat_vs_growing():
    """Refinements (committed): with a clean all-treated tail, the 2x2 lands on
    the truth only when effects are *flat* (clean-tail-2x2 == E1 == CS, naive
    diluted below); when effects *grow*, only CS tracks E1 -- the naive 2x2 reads
    low and the clean-tail 2x2 overshoots (captures the grown tail)."""
    rs, ts = min(ROLLOUTS["moderate"]), max(ROLLOUTS["moderate"])
    centers = {}
    for label, growth in [("flat", 0.0), ("growing", 0.05)]:
        E1s, naive, tail, cs = [], [], [], []
        for s in range(20):
            p = make_panel(ROLLOUTS["moderate"], 2.0, 300 + s, growth=growth)
            E1s.append(estimands(p, rs)[0])
            naive.append(fit_2x2(p, rs)[0])
            tail.append(fit_2x2(p, rs, tail_start=ts)[0])
            cs.append(fit_cs(p)[0])
        centers[label] = tuple(float(np.mean(v)) for v in (E1s, naive, tail, cs))

    # Flat: clean-tail 2x2 and CS both land on the truth; the naive 2x2 is diluted.
    E1, naive, tail, cs = centers["flat"]
    assert abs(tail - E1) < 0.15, f"flat: clean-tail 2x2={tail:.2f} not ~E1={E1:.2f}"
    assert abs(cs - E1) < 0.15, f"flat: CS={cs:.2f} not ~E1={E1:.2f}"
    assert naive < E1 - 0.20, f"flat: naive 2x2={naive:.2f} not diluted below E1={E1:.2f}"

    # Growing: only CS tracks E1; naive reads low, clean-tail overshoots.
    E1, naive, tail, cs = centers["growing"]
    assert abs(cs - E1) < 0.15, f"growing: CS={cs:.2f} not tracking E1={E1:.2f}"
    assert naive < E1 - 0.20, f"growing: naive 2x2={naive:.2f} not below E1={E1:.2f}"
    assert tail > E1 + 0.05, f"growing: clean-tail 2x2={tail:.2f} not overshooting E1={E1:.2f}"
    assert naive < E1 < tail, "growing: expected naive < E1 < clean-tail ordering"
