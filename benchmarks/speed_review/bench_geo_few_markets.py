"""
Scenario 4: Geo-experiment with few treated markets (SyntheticDiD).

Chains: SDiD with jackknife variance (N LOO refits) -> SDiD with bootstrap
variance for SE comparison -> in_time_placebo -> get_loo_effects_df ->
sensitivity_to_zeta_omega -> weight-concentration diagnostic.

Three scales:
  - small  (80 units,  5 treated):  Tutorial 18 DMA panel
  - medium (200 units, 15 treated): zip-cluster or large geo-experiment
  - large  (500 units, 30 treated): zip-level or multi-market at scale
                                    (Python backend skipped at this scale;
                                    Python FW solver scales poorly)

The python backend is skipped at "large" because the pure-numpy Frank-Wolfe
solver plus jackknife (500 LOO refits x ~0.5s each) would take tens of
minutes without providing additional signal; the medium scale already
establishes the Python-vs-Rust gap.
"""

import os

from diff_diff import SyntheticDiD
from diff_diff.prep import generate_factor_data

from bench_shared import run_scenario


SCALES = {
    "small":  {"n_units": 80,  "n_pre": 6, "n_post": 6, "n_treated": 5},
    "medium": {"n_units": 200, "n_pre": 6, "n_post": 6, "n_treated": 15},
    "large":  {"n_units": 500, "n_pre": 6, "n_post": 6, "n_treated": 30},
}
SKIP_PYTHON_AT = {"large"}


def build_data(n_units, n_pre, n_post, n_treated, seed=42):
    return generate_factor_data(
        n_units=n_units, n_pre=n_pre, n_post=n_post, n_treated=n_treated,
        n_factors=2, treatment_effect=2.0,
        factor_strength=1.0, treated_loading_shift=0.5,
        seed=seed,
    )


def make_phases(data, post_periods, results):
    def sdid_jackknife():
        sdid = SyntheticDiD(variance_method="jackknife", seed=123)
        results["jk"] = sdid.fit(
            data, outcome="outcome", unit="unit", time="period",
            treatment="treat", post_periods=post_periods,
        )

    def sdid_bootstrap():
        sdid = SyntheticDiD(
            variance_method="bootstrap", n_bootstrap=200, seed=123,
        )
        results["bs"] = sdid.fit(
            data, outcome="outcome", unit="unit", time="period",
            treatment="treat", post_periods=post_periods,
        )

    def in_time_placebo():
        fn = getattr(results["jk"], "in_time_placebo", None)
        if fn is None:
            raise RuntimeError("in_time_placebo not available on results")
        results["in_time"] = fn()

    def loo_effects_df():
        fn = getattr(results["jk"], "get_loo_effects_df", None)
        if fn is None:
            raise RuntimeError("get_loo_effects_df not available")
        results["loo"] = fn()

    def sensitivity_zeta_omega():
        fn = getattr(results["jk"], "sensitivity_to_zeta_omega", None)
        if fn is None:
            raise RuntimeError("sensitivity_to_zeta_omega not available")
        results["zeta"] = fn()

    def weight_concentration():
        fn = getattr(results["jk"], "get_weight_concentration", None)
        if fn is None:
            raise RuntimeError("get_weight_concentration not available")
        results["wc"] = fn()

    return [
        ("1_sdid_jackknife_variance", sdid_jackknife),
        ("2_sdid_bootstrap_variance_200", sdid_bootstrap),
        ("3_in_time_placebo", in_time_placebo),
        ("4_get_loo_effects_df", loo_effects_df),
        ("5_sensitivity_to_zeta_omega", sensitivity_zeta_omega),
        ("6_weight_concentration", weight_concentration),
    ]


def run_scale(scale, config):
    backend_env = os.environ.get("DIFF_DIFF_BACKEND", "auto").lower()
    if scale in SKIP_PYTHON_AT and backend_env == "python":
        print(f"  [skip] geo_few_markets/{scale} backend=python "
              f"(Python FW solver scales poorly)")
        return

    data = build_data(**config)
    post_periods = sorted(
        data.loc[(data["treat"] == 1) & (data["treated"] == 1),
                 "period"].unique().tolist(),
    )
    results = {}
    phases = make_phases(data, post_periods, results)

    run_scenario(
        f"geo_few_markets_{scale}",
        phases,
        metadata={
            "scale": scale,
            "n_units": config["n_units"],
            "n_pre": config["n_pre"],
            "n_post": config["n_post"],
            "n_treated": config["n_treated"],
            "n_factors": 2,
        },
    )


def main():
    for scale, config in SCALES.items():
        print(f"\n{'='*60}\n  geo_few_markets / scale={scale} "
              f"(n_units={config['n_units']}, "
              f"n_treated={config['n_treated']})\n{'='*60}")
        run_scale(scale, config)


if __name__ == "__main__":
    main()
