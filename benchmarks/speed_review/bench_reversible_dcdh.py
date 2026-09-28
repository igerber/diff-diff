"""
Scenario 5: Reversible treatment with dCDH, L_max multi-horizon, survey TSL.

Chains: dCDH fit with L_max=3 (multi-horizon DID_l + dynamic placebos +
sup-t bands + TWFE diagnostic + survey TSL) -> inspect placebo ->
compute_honest_did on placebo event study -> heterogeneity refit.

Data shape: 120 groups x 10 periods, single-switch reversible pattern,
survey-weighted with 8 strata and 24 PSUs.
"""

import numpy as np
import pandas as pd

from diff_diff import (
    ChaisemartinDHaultfoeuille,
    SurveyDesign,
    compute_honest_did,
)
from diff_diff.prep import generate_reversible_did_data

from bench_shared import run_scenario


def attach_survey_columns(df, seed=42, n_strata=8, psu_per_stratum=3):
    rng = np.random.default_rng(seed)
    groups = sorted(df["group"].unique())
    n_groups = len(groups)
    stratum_map = {g: i % n_strata for i, g in enumerate(groups)}
    psu_map = {
        g: stratum_map[g] * psu_per_stratum + (i // n_strata) % psu_per_stratum
        for i, g in enumerate(groups)
    }
    weight_map = {
        g: float(rng.lognormal(0, 0.3)) for g in groups
    }
    df = df.copy()
    df["stratum"] = df["group"].map(stratum_map)
    df["psu"] = df["group"].map(psu_map)
    df["pw"] = df["group"].map(weight_map)
    return df


def main():
    raw = generate_reversible_did_data(
        n_groups=120, n_periods=10, pattern="single_switch",
        initial_treat_frac=0.3, p_switch=0.15,
        treatment_effect=2.0, heterogeneous_effects=True,
        seed=42,
    )
    data = attach_survey_columns(raw)

    results = {}
    fit_kwargs = dict(
        data=data, outcome="outcome", group="group", time="period",
        treatment="treatment",
    )
    sd = SurveyDesign(
        weights="pw", strata="stratum", psu="psu",
    )

    def dcdh_fit_lmax3():
        est = ChaisemartinDHaultfoeuille(seed=123)
        results["dcdh"] = est.fit(
            **fit_kwargs, L_max=3, survey_design=sd,
        )

    def inspect_placebo():
        r = results["dcdh"]
        results["placebo_summary"] = {
            "placebo_effect": getattr(r, "placebo_effect", None),
            "overall_att": getattr(r, "overall_att", None),
            "joiners_att": getattr(r, "joiners_att", None),
            "leavers_att": getattr(r, "leavers_att", None),
        }

    def honest_placebo():
        out = {}
        for M in (0.5, 1.0, 1.5):
            out[M] = compute_honest_did(
                results["dcdh"], method="relative_magnitude", M=M,
            )
        results["honest"] = out

    def heterogeneity_refit():
        # Use the same SurveyDesign as the main fit; the scenario framing
        # is the survey-TSL workflow, and the TSL-sharing optimization
        # conclusion in performance-plan.md depends on both fits running
        # under the same survey design.
        est = ChaisemartinDHaultfoeuille(seed=123)
        results["het"] = est.fit(
            **fit_kwargs, L_max=3, survey_design=sd, heterogeneity="group",
        )

    phases = [
        ("1_dcdh_fit_Lmax3_survey_TSL", dcdh_fit_lmax3),
        ("2_inspect_placebo_and_summary", inspect_placebo),
        ("3_honest_did_on_placebo", honest_placebo),
        ("4_heterogeneity_refit", heterogeneity_refit),
    ]

    run_scenario(
        "reversible_dcdh",
        phases,
        metadata={
            "n_groups": 120, "n_periods": 10,
            "pattern": "single_switch", "L_max": 3,
            "n_strata": int(data["stratum"].nunique()),
            "n_psu": int(data["psu"].nunique()),
        },
    )


if __name__ == "__main__":
    main()
