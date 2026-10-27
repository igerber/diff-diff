#!/usr/bin/env python3
"""Estimate the shared Appendix E.1 simulation fixture in Python."""

from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from diff_diff import DIDOVBSensitivity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--r-json")
    args = parser.parse_args()

    wide = pd.read_csv(args.csv)
    data = pd.concat(
        [
            wide[["unit", "treated", "x"]].assign(time=0, outcome=wide["pre"]),
            wide[["unit", "treated", "x"]].assign(time=1, outcome=wide["post"]),
        ],
        ignore_index=True,
    )
    propensity = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    outcome = LinearRegression()
    result = DIDOVBSensitivity(
        n_folds=10,
        seed=0,
        propensity_learner=propensity,
        outcome_learner=outcome,
        pscore_trim=0.01,
    ).fit(
        data,
        outcome="outcome",
        treatment="treated",
        time="time",
        unit="unit",
        covariates=["x"],
        fold_ids=(wide["unit"].to_numpy() - 1) % 10,
    )
    output = result.to_dict()
    print(json.dumps(output, indent=2))
    if args.r_json:
        reference = json.loads(open(args.r_json).read())
        print("R short_att:", reference["short_att"])
        print("R scale:", reference["scale"])
        print("Python-R short_att:", result.short_att - reference["short_att"])
        print("Python-R scale:", result.scale - reference["scale"])


if __name__ == "__main__":
    main()
