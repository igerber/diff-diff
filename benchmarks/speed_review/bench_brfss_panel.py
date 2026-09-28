"""
Scenario 3: BRFSS-style microdata -> aggregate_survey -> CS panel.

Chains: aggregate_survey (microdata -> state-year panel) -> CS fit with
stage-2 SurveyDesign + bootstrap at PSU -> event-study pre-trends ->
HonestDiD grid -> SunAbraham robustness refit -> practitioner_next_steps.

Three scales, grounded in BRFSS 2024 (~458K records total):
  - small  (50K rows):  single-year / single-state substudy slice
  - medium (250K rows): multi-year multi-state analytic slice
  - large  (1M rows):   pooled 10-year BRFSS-scale panel
"""

import numpy as np
import pandas as pd

from diff_diff import (
    CallawaySantAnna,
    SunAbraham,
    SurveyDesign,
    aggregate_survey,
    compute_honest_did,
    practitioner_next_steps,
)

from bench_shared import run_scenario


SCALES = {
    "small":  {"n_states": 50, "n_years": 10, "n_per_cell": 100,
               "n_strata": 10, "n_psu": 200},
    "medium": {"n_states": 50, "n_years": 10, "n_per_cell": 500,
               "n_strata": 15, "n_psu": 600},
    "large":  {"n_states": 50, "n_years": 10, "n_per_cell": 2000,
               "n_strata": 20, "n_psu": 1000},
}


def build_microdata(n_states, n_years, n_per_cell, n_strata, n_psu, seed=42):
    rng = np.random.default_rng(seed)
    n_rows = n_states * n_years * n_per_cell
    state = np.repeat(np.arange(n_states), n_years * n_per_cell)
    year = np.tile(
        np.repeat(np.arange(2010, 2010 + n_years), n_per_cell),
        n_states,
    )
    stratum = rng.integers(0, n_strata, size=n_rows)
    psu = stratum * (n_psu // n_strata) + rng.integers(
        0, n_psu // n_strata, size=n_rows,
    )
    weight = rng.lognormal(0, 0.4, size=n_rows) * 50.0

    cohort_map = rng.choice(
        [0, 2013, 2014, 2015, 2016, 2017],
        size=n_states,
        p=[0.4, 0.12, 0.12, 0.12, 0.12, 0.12],
    )
    first_treat = cohort_map[state]
    treated = (first_treat > 0) & (year >= first_treat)
    y = (
        rng.normal(0, 1, size=n_rows)
        + 0.5 * (year - 2010)
        + 3.0 * treated.astype(float)
        + rng.normal(0, 0.2, size=n_rows) * state
    )
    return pd.DataFrame({
        "state": state, "year": year,
        "strata": stratum, "psu": psu, "finalwt": weight,
        "y": y, "first_treat": first_treat,
    })


def make_phases(micro, results):
    def aggregate():
        sd = SurveyDesign(
            weights="finalwt", strata="strata", psu="psu",
        )
        panel, stage2 = aggregate_survey(
            micro, by=["state", "year"], outcomes="y",
            survey_design=sd,
        )
        panel["first_treat"] = panel["state"].map(
            micro.groupby("state")["first_treat"].first(),
        )
        results["panel"] = panel
        results["stage2"] = stage2

    def cs_fit():
        cs = CallawaySantAnna(
            control_group="never_treated", estimation_method="reg",
            n_bootstrap=199, seed=123,
        )
        results["cs"] = cs.fit(
            results["panel"], outcome="y_mean",
            unit="state", time="year", first_treat="first_treat",
            survey_design=results["stage2"], aggregate="all",
        )

    def inspect_pretrends():
        es = results["cs"].event_study_effects or {}
        results["pretrends"] = {
            rel_t: eff for rel_t, eff in es.items() if rel_t < 0
        }

    def honest_grid():
        out = {}
        for M in (0.5, 1.0, 1.5):
            out[M] = compute_honest_did(
                results["cs"], method="relative_magnitude", M=M,
            )
        results["honest"] = out

    def sun_abraham():
        sa = SunAbraham(control_group="never_treated")
        results["sa"] = sa.fit(
            results["panel"], outcome="y_mean", unit="state",
            time="year", first_treat="first_treat",
            survey_design=results["stage2"],
        )

    def guidance():
        results["guidance"] = practitioner_next_steps(results["cs"])

    return [
        ("1_aggregate_survey_microdata_to_panel", aggregate),
        ("2_cs_fit_with_stage2_survey_design", cs_fit),
        ("3_inspect_pretrends", inspect_pretrends),
        ("4_honest_did_grid", honest_grid),
        ("5_sun_abraham_robustness", sun_abraham),
        ("6_practitioner_next_steps", guidance),
    ]


def run_scale(scale, config):
    micro = build_microdata(**config)
    results = {}
    phases = make_phases(micro, results)
    run_scenario(
        f"brfss_panel_{scale}",
        phases,
        metadata={
            "scale": scale,
            "n_microdata_rows": int(len(micro)),
            "n_states": int(micro["state"].nunique()),
            "n_years": int(micro["year"].nunique()),
            "n_strata": int(micro["strata"].nunique()),
            "n_psu": int(micro["psu"].nunique()),
            "n_bootstrap": 199,
        },
    )


def main():
    for scale, config in SCALES.items():
        n_rows = (config["n_states"] * config["n_years"]
                  * config["n_per_cell"])
        print(f"\n{'='*60}\n  brfss_panel / scale={scale} "
              f"({n_rows:,} microdata rows)\n{'='*60}")
        run_scale(scale, config)


if __name__ == "__main__":
    main()
