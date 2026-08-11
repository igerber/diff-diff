"""Drift detection for Tutorial 26
(``docs/tutorials/26_composition_drift_calibration.ipynb``).

The tutorial narrative quotes seed-specific numbers (planted ATT -3.0pp
with realized population ATT ~-2.98pp; design-weight CS ~-4.1pp with
clean pre-trends; national per-wave rake ~-4.4pp as the "false fix";
state-year rake ~-3.2pp as the recovery; 2024 composition shares). If library numerics drift, the prose can go
stale silently while ``pytest --nbmake`` still passes - it only checks
that cells execute. These asserts re-derive the headline numbers using
the locked T26 DGP duplicated below (verbatim from the notebook SS2 code
cell) and check them against tolerance bands around the quoted values.

Requires the ``balance`` package (>=0.21) for the raking acts - the
whole module skips when balance is absent (main-suite CI legs). It DOES
run in the ``interop-notebooks`` CI job, where balance is installed;
that job is this guard's CI home.

The DGP-builder constants below MUST stay in sync with the notebook SS2
code cell; ``test_notebook_dgp_constants_match`` catches silent drift on
those values.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

balance = pytest.importorskip(
    "balance",
    minversion="0.21",
    reason="balance>=0.21 required (interop-notebooks CI job / local)",
)
from balance import Sample  # noqa: E402
from balance.interop import diff_diff as bd  # noqa: E402

from diff_diff import CallawaySantAnna, SurveyDesign, aggregate_survey  # noqa: E402

logging.getLogger("balance").setLevel(logging.ERROR)

TRUE_ATT_QUOTED = -3.0  # pp, planted coefficient (realized ATT ~-2.98pp after floor)

# ---------------------------------------------------------------------------
# Locked DGP - duplicated verbatim from the notebook SS2 cell. Keep in sync.
# ---------------------------------------------------------------------------

N_STATES = 50
YEARS = np.arange(2018, 2025)
N_INVITED = 1200
N_STRATA = 5
PSUS_PER_STATE = 8
FPC_PSUS_PER_STRATUM = 200.0

AGE_BANDS = ["18-34", "35-49", "50-64", "65+"]
AGE_SHARES = np.array([0.30, 0.25, 0.25, 0.20])
EDUC_CATS = ["hs_or_less", "some_college", "college_plus"]
EDUC_SHARES = np.array([0.35, 0.30, 0.35])

BASE_EDUC_PP = np.array([22.0, 15.0, 9.0])
AGE_ADJ_PP = np.array([2.0, 3.0, 1.0, -2.0])
TREND_COMMON_PP = 0.25
TREND_EDUC_PP = np.array([-0.15, 0.0, 0.10])
TRUE_ATT_PP = -3.0
STATE_RE_SD_PP = 1.5
PSU_SHOCK_SD_PP = 0.8
P_CLIP_PP = (1.0, 60.0)

R_BASE = 0.70
R_AGE_SHIFT = np.array([-0.10, -0.02, 0.03, 0.08])
R_EDUC_SHIFT = np.array([-0.09, 0.00, 0.06])
R_COMMON_DRIFT_EDUC = np.array([0.015, 0.0075, 0.0])
R_COMMON_DRIFT_YOUNG = 0.010
R_DIFF_DRIFT_PER_EVENT_YEAR = 0.07
R_CLIP = (0.10, 0.95)

TARGET_N = 20_000
SEED = 20260704

RAKE_VARS = ["age_band", "educ_cat"]


def simulate_brfss_smoking(differential, seed=SEED, drift_start_offset=0):
    """Duplicated from the notebook SS2 code cell. Keep in sync.

    No arm-specific trends by construction: population parallel trends hold
    in expectation (mean-zero PSU-year shocks add noise, not drift); the
    planted effect is -3.0pp (realized population ATT ~-2.98pp after the
    probability floor). All SYSTEMATIC estimator bias comes from sample
    composition.
    """
    rng = np.random.default_rng(seed)

    perm = rng.permutation(N_STATES)
    g_of_state = np.zeros(N_STATES, dtype=int)
    g_of_state[perm[:10]] = 2020
    g_of_state[perm[10:20]] = 2022
    stratum_of_state = rng.integers(0, N_STRATA, size=N_STATES)
    state_pop = rng.lognormal(mean=np.log(4e6), sigma=0.6, size=N_STATES)
    state_re = np.clip(rng.normal(0.0, STATE_RE_SD_PP, size=N_STATES), -3.0, 3.0)
    psu_shock = rng.normal(0.0, PSU_SHOCK_SD_PP, size=(N_STATES, PSUS_PER_STATE, len(YEARS)))

    n_inv = N_STATES * len(YEARS) * N_INVITED
    state = np.repeat(np.arange(N_STATES), len(YEARS) * N_INVITED)
    year = np.tile(np.repeat(YEARS, N_INVITED), N_STATES)
    age_idx = rng.choice(len(AGE_BANDS), size=n_inv, p=AGE_SHARES)
    educ_idx = rng.choice(len(EDUC_CATS), size=n_inv, p=EDUC_SHARES)
    psu_idx = rng.integers(0, PSUS_PER_STATE, size=n_inv)
    u_respond = rng.uniform(size=n_inv)
    u_smoker = rng.uniform(size=n_inv)
    weight_jitter = rng.uniform(0.85, 1.15, size=n_inv)

    k = year - YEARS[0]
    year_idx = year - YEARS[0]
    g = g_of_state[state]
    treated_post = (g > 0) & (year >= g)

    base_pp = (
        BASE_EDUC_PP[educ_idx]
        + AGE_ADJ_PP[age_idx]
        + state_re[state]
        + psu_shock[state, psu_idx, year_idx]
        - (TREND_COMMON_PP + TREND_EDUC_PP[educ_idx]) * k
    )
    p_pp = np.clip(base_pp + TRUE_ATT_PP * treated_post, *P_CLIP_PP)
    smoker = (u_smoker < p_pp / 100.0).astype(int)

    r = (
        R_BASE
        + R_AGE_SHIFT[age_idx]
        + R_EDUC_SHIFT[educ_idx]
        - R_COMMON_DRIFT_EDUC[educ_idx] * k
        - R_COMMON_DRIFT_YOUNG * k * (age_idx == 0)
    )
    if differential:
        event_time = year - g - drift_start_offset
        hit = (g > 0) & (event_time >= 0) & (educ_idx == 0)
        r = r - R_DIFF_DRIFT_PER_EVENT_YEAR * (event_time + 1) * hit
    r = np.clip(r, *R_CLIP)
    responded = u_respond < r

    micro = pd.DataFrame(
        {
            "id": np.arange(n_inv)[responded],
            "state": state[responded],
            "year": year[responded],
            "g": g[responded],
            "smoker": smoker[responded],
            "age_band": np.array(AGE_BANDS)[age_idx[responded]],
            "educ_cat": np.array(EDUC_CATS)[educ_idx[responded]],
            "stratum": stratum_of_state[state[responded]],
            "psu": state[responded] * 100 + psu_idx[responded],
            "fpc": FPC_PSUS_PER_STRATUM,
            "design_weight": (state_pop[state] / N_INVITED * weight_jitter)[responded],
        }
    )

    rng_t = np.random.default_rng(seed + 1)
    target_df = pd.DataFrame(
        {
            "id": np.arange(TARGET_N),
            "age_band": np.array(AGE_BANDS)[
                rng_t.choice(len(AGE_BANDS), size=TARGET_N, p=AGE_SHARES)
            ],
            "educ_cat": np.array(EDUC_CATS)[
                rng_t.choice(len(EDUC_CATS), size=TARGET_N, p=EDUC_SHARES)
            ],
        }
    )

    kk = YEARS - YEARS[0]
    cell = (
        BASE_EDUC_PP[None, :, None]
        + AGE_ADJ_PP[None, None, :]
        - (TREND_COMMON_PP + TREND_EDUC_PP[None, :, None]) * kk[:, None, None]
    )
    tp = (g_of_state[None, :] > 0) & (YEARS[:, None] >= g_of_state[None, :])
    base_prev = np.einsum("tea,e,a->t", cell, EDUC_SHARES, AGE_SHARES)
    w_s = state_pop / state_pop.sum()
    pop_prev = base_prev + TRUE_ATT_PP * (tp * w_s[None, :]).sum(axis=1)
    # Realized population ATT: the probability floor P_CLIP_PP[0] binds for
    # ~2% of treated-post person-years, attenuating the planted -3.0pp.
    y1 = np.clip(base_pp + TRUE_ATT_PP, *P_CLIP_PP)
    y0 = np.clip(base_pp, *P_CLIP_PP)
    w_pop = state_pop[state]
    realized_att_pp = ((y1 - y0) * w_pop)[treated_post].sum() / w_pop[treated_post].sum()
    truth = {
        "true_att_pp": TRUE_ATT_PP,
        "realized_att_pp": float(realized_att_pp),
        "floor_bind_share": float(((y1 - y0) > TRUE_ATT_PP + 1e-12)[treated_post].mean()),
        "pop_prevalence_by_year": dict(zip(YEARS.tolist(), pop_prev / 100.0)),
        "g_of_state": g_of_state,
        "state_pop": state_pop,
    }
    return micro, target_df, truth


def rake_to_population(micro, target_df, granularity, weight_name, cell_totals):
    """Duplicated verbatim from the notebook SS5 code cell. Keep in sync."""
    target_sample = Sample.from_frame(target_df, id_column="id")
    cols = ["id", *RAKE_VARS, "smoker", "design_weight"]
    w_new = pd.Series(np.nan, index=micro.index)
    adjusted = {}
    for key, cell in micro.groupby(granularity):
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        s = Sample.from_frame(
            cell[cols].copy(),
            id_column="id",
            weight_column="design_weight",
            outcome_columns=["smoker"],
        )
        adj = s.set_target(target_sample).adjust(method="rake", variables=RAKE_VARS)
        w = adj.df.set_index("id")[adj.weight_column]
        aligned = w.reindex(cell["id"].astype(str).values).to_numpy()
        assert not np.isnan(aligned).any(), f"NaN raked weights in cell {key}"
        aligned = aligned * (cell_totals[key] / aligned.sum())
        w_new.loc[cell.index] = aligned
        adjusted[key] = adj
    out = micro.copy()
    out[weight_name] = w_new
    return out, adjusted


def fit_survey_cs(micro, weights_col):
    """Duplicated from the notebook SS3 code cell (native seam). Keep in sync."""
    design = SurveyDesign(weights=weights_col, strata="stratum", psu="psu", fpc="fpc")
    panel, second_stage = aggregate_survey(
        micro, by=["state", "year"], outcomes="smoker", survey_design=design
    )
    panel = panel.merge(micro[["state", "g"]].drop_duplicates(), on="state", how="left")
    cs = CallawaySantAnna(
        estimation_method="reg",
        control_group="not_yet_treated",
        base_period="universal",
    )
    return cs.fit(
        panel,
        outcome="smoker_mean",
        unit="state",
        time="year",
        first_treat="g",
        survey_design=second_stage,
    )


# ---------------------------------------------------------------------------
# Shared pipeline run (module-scoped: the rakes dominate the ~10s runtime)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        micro_a, _, _ = simulate_brfss_smoking(differential=False)
        micro, target, truth = simulate_brfss_smoking(differential=True)
        state_pop = truth["state_pop"]
        res_a = fit_survey_cs(micro_a, "design_weight")
        res_design = fit_survey_cs(micro, "design_weight")
        micro, _ = rake_to_population(
            micro,
            target,
            ["year"],
            "w_national",
            cell_totals={int(y): state_pop.sum() for y in YEARS},
        )
        res_national = fit_survey_cs(micro, "w_national")
        micro, _ = rake_to_population(
            micro,
            target,
            ["state", "year"],
            "w_raked",
            cell_totals={(st, int(y)): state_pop[st] for st in range(N_STATES) for y in YEARS},
        )
        res_raked = fit_survey_cs(micro, "w_raked")
    return {
        "micro": micro,
        "truth": truth,
        "A_design": res_a,
        "B_design": res_design,
        "B_national": res_national,
        "B_raked": res_raked,
    }


def test_planted_and_realized_truth(pipeline):
    # Planted coefficient is exactly -3.0pp; the 1pp probability floor binds
    # for ~2% of treated-post person-years, so the realized population ATT
    # (quoted throughout the tutorial as the truth line) is ~-2.98pp.
    truth = pipeline["truth"]
    assert truth["true_att_pp"] == TRUE_ATT_QUOTED
    assert (
        -3.0 <= truth["realized_att_pp"] <= -2.95
    ), f"realized ATT drifted: {truth['realized_att_pp']:.4f}pp"
    assert 0.005 <= truth["floor_bind_share"] <= 0.05


def test_scenario_a_design_weights_robust(pipeline):
    att = pipeline["A_design"].overall_att * 100
    assert -3.5 <= att <= -2.7, f"scenario A design ATT drifted: {att:.2f}pp"


def test_scenario_b_design_weights_overstate(pipeline):
    res = pipeline["B_design"]
    att = res.overall_att * 100
    assert -4.6 <= att <= -3.7, f"scenario B design ATT drifted: {att:.2f}pp"
    # The quoted story: the design-weight CI EXCLUDES the realized truth
    # (~-2.98pp; -2.95 is the upper edge of the realized-truth band pinned
    # in test_planted_and_realized_truth).
    hi = res.overall_conf_int[1]
    assert hi * 100 < -2.95, (
        f"design-weight CI upper bound {hi*100:.2f}pp no longer excludes "
        "the realized truth (~-2.98pp)"
    )
    # Pre-trends stay clean (drift starts at adoption): |pre| below 1.5pp.
    # (Post-fit container, mirroring the notebook's migrated read.)
    agg = res.aggregate("event_study")
    max_pre = max(abs(a) * 100 for t, a in zip(agg.event_time, agg.att) if t < -1)
    assert max_pre < 1.5, f"pre-trend coefficient drifted: {max_pre:.2f}pp"


def test_national_rake_is_not_a_fix(pipeline):
    att_nat = pipeline["B_national"].overall_att * 100
    att_des = pipeline["B_design"].overall_att * 100
    realized = pipeline["truth"]["realized_att_pp"]
    assert -4.9 <= att_nat <= -4.0, f"national-rake ATT drifted: {att_nat:.2f}pp"
    # Quoted lesson: national raking does NOT move the estimate toward truth.
    assert abs(att_nat - realized) >= abs(att_des - realized) - 0.1


def test_state_rake_recovers_truth(pipeline):
    res = pipeline["B_raked"]
    att = res.overall_att * 100
    se = res.overall_se * 100
    realized = pipeline["truth"]["realized_att_pp"]
    assert -3.6 <= att <= -2.8, f"state-rake ATT drifted: {att:.2f}pp"
    assert (
        abs(att - realized) <= 2 * se
    ), f"state-rake ATT {att:.2f}pp not within 2 SE ({se:.2f}) of realized truth"


def test_composition_shares_2024(pipeline):
    micro = pipeline["micro"]
    m24 = micro[micro.year == 2024].assign(hs=lambda d: (d.educ_cat == "hs_or_less").astype(float))

    def share(sub, wcol):
        return (sub.hs * sub[wcol]).sum() / sub[wcol].sum()

    treated = m24[m24.g == 2020]
    never = m24[m24.g == 0]
    # Quoted: treated design-weighted hs share collapses (~0.11)...
    assert share(treated, "design_weight") < 0.15
    # ...and state-year raking restores BOTH arms to the 0.35 margin.
    assert abs(share(treated, "w_raked") - 0.35) < 0.02
    assert abs(share(never, "w_raked") - 0.35) < 0.02


def test_native_adapter_parity(pipeline):
    """The notebook asserts the native seam and bd.fit_did agree exactly."""
    micro = pipeline["micro"]
    keep = [
        "id",
        "state",
        "year",
        "smoker",
        "age_band",
        "educ_cat",
        "stratum",
        "psu",
        "fpc",
        "w_raked",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample = Sample.from_frame(
            micro[keep].copy(),
            id_column="id",
            weight_column="w_raked",
            outcome_columns=["smoker"],
        )
        panel_df, second_stage = bd.to_panel_for_did(
            sample, by=["state", "year"], outcomes="smoker"
        )
        panel_df = panel_df.merge(micro[["state", "g"]].drop_duplicates(), on="state", how="left")
        panel_df["panel_id"] = np.arange(len(panel_df))
        panel_sample = Sample.from_frame(
            panel_df,
            id_column="panel_id",
            weight_column=second_stage.weights,
            outcome_columns=["smoker_mean"],
        )
        res_adapter = bd.fit_did(
            panel_sample,
            estimator="CallawaySantAnna",
            outcome="smoker_mean",
            time="year",
            unit="state",
            treatment_first="g",
            design_columns={"psu": "state"},
            estimation_method="reg",
            control_group="not_yet_treated",
            base_period="universal",
        )
    np.testing.assert_allclose(res_adapter.overall_att, pipeline["B_raked"].overall_att, rtol=1e-12)
    assert hasattr(res_adapter, "_balance_adjustment")


def test_notebook_dgp_constants_match():
    """Sync guard: the notebook's DGP constants must match this module's
    locked copy, so a notebook-only edit can't silently invalidate the
    quoted numbers (t25 precedent).

    CI isolation note: CI legs that copy ``tests/`` without ``docs/`` skip
    gracefully here (nbmake separately verifies execution)."""
    import json
    from pathlib import Path

    nb_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "tutorials"
        / "26_composition_drift_calibration.ipynb"
    )
    if not nb_path.exists():
        pytest.skip(f"Notebook not found at {nb_path}; sync guard is local-dev only.")
    with nb_path.open() as f:
        nb = json.load(f)
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for needle in (
        # locked constants
        "SEED = 20260704",
        "N_INVITED = 1200",
        "TRUE_ATT_PP = -3.0",
        "BASE_EDUC_PP = np.array([22.0, 15.0, 9.0])",
        "TREND_EDUC_PP = np.array([-0.15, 0.0, 0.10])",
        "R_DIFF_DRIFT_PER_EVENT_YEAR = 0.07",
        "R_COMMON_DRIFT_EDUC = np.array([0.015, 0.0075, 0.0])",
        'RAKE_VARS = ["age_band", "educ_cat"]',
        "TARGET_N = 20_000",
        # load-bearing logic lines (mechanism, rescale, realized truth, fit)
        "r = r - R_DIFF_DRIFT_PER_EVENT_YEAR * (event_time + 1) * hit",
        "aligned = aligned * (cell_totals[key] / aligned.sum())",
        "y1 = np.clip(base_pp + TRUE_ATT_PP, *P_CLIP_PP)",
        'estimation_method="reg"',
        'control_group="not_yet_treated"',
        'base_period="universal"',
    ):
        assert needle in src, f"notebook SS2 missing locked constant: {needle!r}"
