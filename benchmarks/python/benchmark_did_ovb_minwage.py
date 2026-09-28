#!/usr/bin/env python3
"""Run the Wang et al. minimum-wage OVB application on exported RDS data.

The authors distribute the source data as ``min_wage_CS.rds``.  This runner
expects a user-created CSV export so that the repository does not redistribute
the external replication data.  It intentionally exposes the learner settings
instead of claiming that sklearn's random forest is identical to R's ranger.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from diff_diff import DIDOVBSensitivity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV export of CS_RR/data/min_wage_CS.rds")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-features", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--folds-csv",
        help="Optional countyreal/fold_id CSV produced by the R application oracle",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="tune RF/ExtraTrees hyperparameters within each outer fold",
    )
    args = parser.parse_args()

    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:  # pragma: no cover - exercised by environments
        raise SystemExit("Install the optional scikit-learn dependency first") from exc

    data = pd.read_csv(args.csv)
    data = data.loc[data["year"].isin([2006, 2007]) & data["first.treat"].isin([0, 2007])].copy()
    data["treated"] = (data["first.treat"] == 2007).astype(int)

    forest_kwargs = {
        "n_estimators": args.n_estimators,
        "max_features": args.max_features,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.seed,
        "n_jobs": -1,
    }

    from sklearn.base import BaseEstimator
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    class _RegressionPropensity(BaseEstimator):
        def __init__(
            self,
            n_estimators=1000,
            max_features=2,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ):
            self.n_estimators = n_estimators
            self.max_features = max_features
            self.min_samples_leaf = min_samples_leaf
            self.random_state = random_state
            self.n_jobs = n_jobs

        def fit(self, X, y):
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ).fit(X, y)
            return self

        def predict_proba(self, X):
            probability = self.model_.predict(X)
            return np.column_stack([1 - probability, probability])

    propensity = _RegressionPropensity(**forest_kwargs)
    outcome = RandomForestRegressor(**forest_kwargs)
    if args.tune:
        from sklearn.model_selection import GridSearchCV

        # This is the sklearn analogue of the paper's ranger grid.  The
        # estimator is cloned independently inside every outer fold by the
        # existing cross-fitting machinery.
        prop_grid = [
            {
                "model": [_RegressionPropensity(**forest_kwargs)],
                "model__max_features": [2, 4, 6],
                "model__min_samples_leaf": [10, 15, 25, 50, 75, 100, 125, 150],
            },
            {
                "model": [
                    _RegressionPropensity(
                        n_estimators=forest_kwargs["n_estimators"],
                        min_samples_leaf=forest_kwargs["min_samples_leaf"],
                        random_state=forest_kwargs["random_state"],
                        n_jobs=forest_kwargs["n_jobs"],
                    )
                ],
                "model__max_features": [2, 4, 6],
                "model__min_samples_leaf": [10, 15, 25, 50, 75, 100, 125, 150],
            },
        ]
        out_grid = [
            {
                "model": [RandomForestRegressor(**forest_kwargs)],
                "model__max_features": [2, 4, 6],
                "model__min_samples_leaf": [10, 15, 25, 50, 75, 100, 125, 150],
            },
            {
                "model": [ExtraTreesRegressor(**forest_kwargs)],
                "model__max_features": [2, 4, 6],
                "model__min_samples_leaf": [10, 15, 25, 50, 75, 100, 125, 150],
            },
        ]
        # A one-step wrapper keeps the public estimator API unchanged while
        # allowing GridSearchCV to switch between ranger-like split rules.
        from sklearn.base import BaseEstimator
        from sklearn.pipeline import Pipeline

        class _GridLearner(BaseEstimator):
            def __init__(self, estimator=None, grid=None, scoring=None, cv=3):
                self.estimator = estimator
                self.grid = grid
                self.scoring = scoring
                self.cv = cv

            def fit(self, X, y):
                params = self.grid or {}
                self.search_ = GridSearchCV(
                    self.estimator,
                    params,
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=-1,
                    refit=True,
                ).fit(X, y)
                return self

            def predict(self, X):
                return self.search_.predict(X)

            def predict_proba(self, X):
                return self.search_.predict_proba(X)

        propensity = _GridLearner(
            estimator=Pipeline([("model", _RegressionPropensity(**forest_kwargs))]),
            grid=prop_grid,
            scoring="neg_root_mean_squared_error",
            cv=3,
        )
        outcome = _GridLearner(
            estimator=Pipeline([("model", RandomForestRegressor(**forest_kwargs))]),
            grid=out_grid,
            scoring="neg_root_mean_squared_error",
            cv=3,
        )

    fit_kwargs = {}
    if args.folds_csv:
        folds = pd.read_csv(args.folds_csv)
        if not {"countyreal", "fold_id"}.issubset(folds.columns):
            raise SystemExit("--folds-csv must contain countyreal and fold_id columns")
        unit_order = data.loc[data["year"] == 2006, "countyreal"].drop_duplicates()
        fold_map = dict(zip(folds["countyreal"], folds["fold_id"], strict=True))
        try:
            fit_kwargs["fold_ids"] = unit_order.map(fold_map).to_numpy(dtype=int)
        except (TypeError, ValueError) as exc:
            raise SystemExit("--folds-csv does not cover every application unit") from exc

    result = DIDOVBSensitivity(
        n_folds=args.n_folds,
        seed=args.seed,
        propensity_learner=propensity,
        outcome_learner=outcome,
    ).fit(
        data,
        outcome="lemp",
        treatment="treated",
        time="year",
        unit="countyreal",
        covariates=["region", "white", "hs", "pov", "lpop", "lmedinc"],
        **fit_kwargs,
    )

    print(result.summary())
    print("RV:", result.robustness_value().rv)
    print("XRV:", result.robustness_value().xrv)
    payload = {
        "n": result.n_obs,
        "treated": result.n_treated,
        "control": result.n_control,
        "n_folds": result.n_folds,
        "seed": result.seed,
        "short_att": result.short_att,
        "short_se": result.short_se,
        "sigma2_control": result.sigma2_control,
        "nu2_selection": result.nu2_selection,
        "scale": result.scale,
        "rv": result.robustness_value().rv,
        "xrv": result.robustness_value().xrv,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
