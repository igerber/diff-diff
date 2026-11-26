"""Dump the pre-stacked StackedDiD panels read by the R golden generator.

`benchmarks/R/generate_stacked_did_golden.R` fits `lm(weights = Q)` on these
CSVs and records clubSandwich CR1S / CR2 standard errors, so the R side is
independent of the stacking logic but uses the Q-weights computed here. Rerun
this script, then the R script, whenever the stacked sample or the Q-weights
change for a variant:

    python benchmarks/python_helpers/dump_stacked_did_panel.py
    Rscript benchmarks/R/generate_stacked_did_golden.R

The panels mirror the fixtures in `tests/test_methodology_stacked_did.py`
(`panel`, `population_panel`), which is what the parity tests fit.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from diff_diff import StackedDiD
from diff_diff.prep_dgp import generate_staggered_data

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _base_panel():
    return generate_staggered_data(
        n_units=50,
        n_periods=8,
        cohort_periods=[3, 5, 7],
        never_treated_frac=0.3,
        treatment_effect=2.0,
        dynamic_effects=False,
        seed=20260521,
    )


def _population_panel():
    rng = np.random.default_rng(99)
    data = _base_panel()
    unit_pops = rng.uniform(1.0, 100.0, size=data["unit"].nunique())
    data = data.copy()
    data["pop"] = data["unit"].map(lambda u: unit_pops[int(u)])
    return data


# name -> (panel builder, StackedDiD kwargs, fit kwargs)
VARIANTS = {
    "test": (_base_panel, {}, {}),
    "population": (_population_panel, {"weighting": "population"}, {"population": "pop"}),
    "anticipation1": (_base_panel, {"anticipation": 1}, {}),
    "sample_share": (_base_panel, {"weighting": "sample_share"}, {}),
}


def dump(name: str) -> str:
    builder, est_kwargs, fit_kwargs = VARIANTS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = StackedDiD(kappa_pre=2, kappa_post=2, **est_kwargs).fit(
            builder(),
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            **fit_kwargs,
        )
    stacked = res.stacked_data.copy()
    stacked["unit_subexp"] = stacked["unit"].astype(str) + "_" + stacked["_sub_exp"].astype(str)
    path = os.path.join(_DATA_DIR, f"stacked_did_{name}_panel.csv")
    stacked.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    for variant in VARIANTS:
        print("wrote", os.path.relpath(dump(variant)))
