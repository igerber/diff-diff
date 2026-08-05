"""
Scenario 6: Pricing dose-response with ContinuousDiD cubic spline.

Chains: CDiD fit (overall ATT + ACRT + dose-response curves are always
computed; bootstrap 199) -> dataframe extraction -> post-fit event-study
pre-trend (results.aggregate('event_study'), row M-025) -> binarized-DiD
comparison -> spline sensitivity (degree=1, num_knots=2).

Data shape: 500 stores x 6 quarterly periods, 1 cohort at period 3,
log-normal dose. Matches Tutorial 14 scaled from 200 to 500 units.
"""

import numpy as np
import pandas as pd

from diff_diff import ContinuousDiD, DifferenceInDifferences
from diff_diff.prep import generate_continuous_did_data

from bench_shared import run_scenario


def build_data(seed=42):
    # cohort_periods=[3] pins the single treated cohort to period 3 to
    # match the documented scenario shape. The generator default would
    # be period 2, which would desync this scenario from the spec in
    # docs/performance-scenarios.md and from the binarized DiD
    # comparison phase below.
    df = generate_continuous_did_data(
        n_units=500, n_periods=6, cohort_periods=[3], seed=seed,
    )
    positive_first_treat = sorted(
        v for v in df["first_treat"].unique() if v > 0
    )
    assert len(positive_first_treat) == 1, (
        f"dose-response scenario expects exactly one treated cohort; "
        f"got first_treat values {positive_first_treat}"
    )
    return df


def main():
    data = build_data()

    results = {}
    fit_kwargs = dict(
        data=data, outcome="outcome", unit="unit", time="period",
        first_treat="first_treat", dose="dose",
    )

    def cdid_cubic_fit():
        cdid = ContinuousDiD(
            degree=3, num_knots=1, n_bootstrap=199, seed=123,
        )
        results["cubic"] = cdid.fit(**fit_kwargs)

    def extract_curves():
        # Dose-response and group-time levels are always available on the
        # result (fit computes them unconditionally). The event study is
        # produced post-fit in the dedicated pretrend phase below via
        # results.aggregate('event_study') - the unified spelling; the
        # fit-time 'eventstudy' kwarg is deprecated (row M-025).
        r = results["cubic"]
        out = {}
        for level in ("dose_response", "group_time"):
            out[level] = r.to_dataframe(level=level)
        results["curves"] = out

    def cdid_event_study():
        # Post-fit route (M-025): the phase still times the event-study
        # computation - the analytical fit plus the kit recompute.
        cdid = ContinuousDiD(
            degree=3, num_knots=1, n_bootstrap=0, seed=123,
        )
        res = cdid.fit(**fit_kwargs)
        results["event_study"] = res.aggregate("event_study")

    def binarized_comparison():
        # Derive post from the actual first_treat cohort in the data so
        # this phase is aligned with the CDiD fits above. A hardcoded
        # period cutoff would silently desync if the DGP cohort moves.
        treated_cohort = int(
            sorted(v for v in data["first_treat"].unique() if v > 0)[0]
        )
        data_bin = data.copy()
        data_bin["treated_any"] = (data_bin["dose"] > 0).astype(int)
        data_bin["post"] = (data_bin["period"] >= treated_cohort).astype(int)
        did = DifferenceInDifferences(robust=True)
        results["binarized"] = did.fit(
            data_bin, outcome="outcome", treatment="treated_any", time="post",
        )

    def spline_sensitivity_linear():
        cdid = ContinuousDiD(
            degree=1, num_knots=0, n_bootstrap=199, seed=123,
        )
        results["linear"] = cdid.fit(**fit_kwargs)

    def spline_sensitivity_more_knots():
        cdid = ContinuousDiD(
            degree=3, num_knots=2, n_bootstrap=199, seed=123,
        )
        results["many_knots"] = cdid.fit(**fit_kwargs)

    phases = [
        ("1_cdid_cubic_spline_bootstrap199", cdid_cubic_fit),
        ("2_extract_dose_response_dataframes", extract_curves),
        ("3_cdid_event_study_pretrend", cdid_event_study),
        ("4_binarized_did_comparison", binarized_comparison),
        ("5_spline_sensitivity_degree1", spline_sensitivity_linear),
        ("6_spline_sensitivity_num_knots2", spline_sensitivity_more_knots),
    ]

    run_scenario(
        "dose_response",
        phases,
        metadata={
            "n_units": 500, "n_periods": 6, "n_bootstrap": 199,
            "spline_configs": ["degree=3,k=1", "degree=1,k=0", "degree=3,k=2"],
        },
    )


if __name__ == "__main__":
    main()
