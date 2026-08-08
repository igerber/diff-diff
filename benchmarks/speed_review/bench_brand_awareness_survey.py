"""
Scenario 2: Brand awareness survey DiD - 2x2 with survey design.

DifferenceInDifferences + SurveyDesign under two variance paths:
  (a) analytical Taylor-series linearization (strata + PSU + FPC)
  (b) replicate-weight variance (JK1 delete-one-PSU; count equals
      the number of PSUs, so 40/90/160 at small/medium/large).
      This is replicate-weight variance, not bootstrap resampling -
      see REGISTRY.md for the distinction.

Chains: naive fit (for SE-inflation comparison) -> TSL -> replicate -> multi-
outcome refit loop -> check_parallel_trends -> placebo -> HonestDiD grid.

Three scales:
  - small  (200 units x 12 periods): Tutorial 17 analog
  - medium (500 units x 12 periods): realistic CPG quarterly brand-tracking wave
  - large  (1000 units x 12 periods): multi-region brand tracking at scale
"""

import numpy as np

from diff_diff import (
    DifferenceInDifferences,
    MultiPeriodDiD,
    SurveyDesign,
    check_parallel_trends,
    compute_honest_did,
)
from diff_diff.prep import generate_survey_did_data

from bench_shared import run_scenario


SCALES = {
    "small":  {"n_units": 200,  "n_periods": 12, "n_strata": 10, "psu_per_stratum": 4},
    "medium": {"n_units": 500,  "n_periods": 12, "n_strata": 15, "psu_per_stratum": 6},
    "large":  {"n_units": 1000, "n_periods": 12, "n_strata": 20, "psu_per_stratum": 8},
}


def build_data(n_units, n_periods, n_strata, psu_per_stratum, seed=42):
    df = generate_survey_did_data(
        n_units=n_units, n_periods=n_periods, cohort_periods=[7],
        never_treated_frac=0.5, treatment_effect=2.0,
        dynamic_effects=True, effect_growth=0.2,
        n_strata=n_strata, psu_per_stratum=psu_per_stratum,
        weight_variation="high", psu_re_sd=1.5,
        include_replicate_weights=True, panel=True, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    df["consideration"] = df["outcome"] + rng.normal(0, 0.4, size=len(df))
    df["purchase_intent"] = df["outcome"] * 0.6 + rng.normal(0, 0.3, size=len(df))
    df["post"] = (df["period"] >= 7).astype(int)
    df["treat_unit"] = (df["first_treat"] > 0).astype(int)
    return df


def make_phases(data, results, rw_cols):
    # One analytical TSL SurveyDesign is reused across every analytical
    # survey phase (TSL, multi-outcome, placebo, HonestDiD event-study).
    # Keeping strata/PSU/FPC/nest constant is what the scenario spec and
    # Tutorial 17 declare, and what the finite-population variance
    # expressions require. The replicate-weight path (phase 3) is a
    # different variance surface (JK1) that does not take FPC.
    sd_tsl = SurveyDesign(
        weights="weight", strata="stratum", psu="psu",
        fpc="fpc", nest=True,
    )

    def naive_fit():
        # Truly naive comparison point: no survey design, no clustering -
        # matches Tutorial 17's first pass where an analyst has not yet
        # accounted for the sampling structure. The SE-inflation story
        # only shows up if this step is as untreated-for-design as
        # practitioners actually start.
        did = DifferenceInDifferences(robust=True)
        results["naive"] = did.fit(
            data, outcome="outcome", treatment="treat_unit", post="post",
        )

    def tsl_fit():
        did = DifferenceInDifferences(robust=True)
        results["tsl"] = did.fit(
            data, outcome="outcome", treatment="treat_unit", post="post",
            survey_design=sd_tsl,
        )

    def replicate_fit():
        if not rw_cols:
            raise RuntimeError("replicate weights not generated")
        sd = SurveyDesign(
            weights="weight", replicate_weights=rw_cols,
            replicate_method="JK1",
        )
        did = DifferenceInDifferences(robust=True)
        results["replicate"] = did.fit(
            data, outcome="outcome", treatment="treat_unit", post="post",
            survey_design=sd,
        )

    def multi_outcome_loop():
        out = {}
        for y in ("outcome", "consideration", "purchase_intent"):
            did = DifferenceInDifferences(robust=True)
            out[y] = did.fit(
                data, outcome=y, treatment="treat_unit", post="post",
                survey_design=sd_tsl,
            )
        results["multi_outcome"] = out

    def pretrends():
        results["pt"] = check_parallel_trends(
            data, outcome="outcome", time="period",
            treatment_group="treat_unit",
            pre_periods=list(range(1, 7)),
        )

    def placebo_refit():
        pre = data[data["period"] < 7].copy()
        pre["placebo_post"] = (pre["period"] >= 4).astype(int)
        did = DifferenceInDifferences(robust=True)
        results["placebo"] = did.fit(
            pre, outcome="outcome", treatment="treat_unit",
            time="placebo_post", survey_design=sd_tsl,
        )

    def honest_did_grid():
        es = MultiPeriodDiD()
        es_result = es.fit(
            data, outcome="outcome", treatment="treat_unit",
            time="period", unit="unit", reference_period=6,
            survey_design=sd_tsl,
        )
        results["event_study"] = es_result
        out = {}
        for M in (0.5, 1.0, 1.5):
            out[M] = compute_honest_did(
                es_result, method="relative_magnitude", M=M,
            )
        results["honest"] = out

    return [
        ("1_naive_fit_no_survey_design", naive_fit),
        ("2_tsl_strata_psu_fpc", tsl_fit),
        ("3_replicate_weights_jk1", replicate_fit),
        ("4_multi_outcome_loop_3_metrics", multi_outcome_loop),
        ("5_check_parallel_trends", pretrends),
        ("6_placebo_refit_pre_period", placebo_refit),
        ("7_event_study_plus_honest_did", honest_did_grid),
    ]


def run_scale(scale, config):
    data = build_data(**config)
    rw_cols = [c for c in data.columns if c.startswith("rep_")]
    results = {}
    phases = make_phases(data, results, rw_cols)

    run_scenario(
        f"brand_awareness_survey_{scale}",
        phases,
        metadata={
            "scale": scale,
            "n_units": config["n_units"],
            "n_periods": config["n_periods"],
            "n_obs": int(len(data)),
            "n_strata": config["n_strata"],
            "n_psu_per_stratum": config["psu_per_stratum"],
            "n_replicate_weights": len(rw_cols),
            "outcomes": ["outcome", "consideration", "purchase_intent"],
        },
    )


def main():
    for scale, config in SCALES.items():
        print(f"\n{'='*60}\n  brand_awareness_survey / scale={scale} "
              f"(n_units={config['n_units']})\n{'='*60}")
        run_scale(scale, config)


if __name__ == "__main__":
    main()
