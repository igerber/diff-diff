#!/usr/bin/env python3
"""Random-forest diagnostic for the Appendix E.2 misspecification DGP.

This is intentionally a Python-only diagnostic because the current R runtime
does not have the paper's ``ranger`` package installed. It uses the same
cross-fitting score as the parametric runner and reports whether a flexible
learner reduces the misspecification bias.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def one_rep(n: int, p: float, seed: int, alpha: float, trees: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = rng.binomial(1, p, n)
    x = rng.normal(np.where(d == 0, 0.3, 0.0), np.where(d == 0, np.sqrt(6), np.sqrt(3)))
    u = rng.normal(np.where(d == 0, 0.3, 0.0), np.where(d == 0, np.sqrt(6), np.sqrt(3)))
    dy = 1 + x + u + 2 * d + rng.normal(0, np.sqrt(2), n)
    z = np.exp(x / 2).reshape(-1, 1)
    ps = np.empty(n)
    m = np.empty(n)
    folds = np.arange(n) % 10
    for fold in range(10):
        test = folds == fold
        train = ~test
        prop = RandomForestClassifier(
            n_estimators=trees,
            max_features=1,
            min_samples_leaf=10,
            random_state=seed + fold,
            n_jobs=-1,
        ).fit(z[train], d[train])
        outcome = RandomForestRegressor(
            n_estimators=trees,
            max_features=1,
            min_samples_leaf=10,
            random_state=seed + fold,
            n_jobs=-1,
        ).fit(z[train & (d == 0)], dy[train & (d == 0)])
        ps[test] = prop.predict_proba(z[test])[:, 1]
        m[test] = outcome.predict(z[test])
    ps = np.clip(ps, 0.01, 0.99)
    phat = float(d.mean())
    omega = (ps / (1 - ps)) / (phat / (1 - phat))
    score = (d / phat - (1 - d) / (1 - phat) * omega) * (dy - m)
    att = float(score.mean())
    se = float(np.sqrt(np.mean((score - att) ** 2) / n))
    from scipy.stats import norm

    zcrit = norm.ppf(1 - alpha / 2)
    return att, se, float(att - zcrit * se <= 1.7 <= att + zcrit * se)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output")
    args = parser.parse_args()
    draws = np.asarray(
        [
            one_rep(args.n, args.p, args.seed + i, args.alpha, args.trees)
            for i in range(args.reps)
        ]
    )
    result = {
        "n": args.n,
        "reps": args.reps,
        "p": args.p,
        "alpha": args.alpha,
        "trees": args.trees,
        "true_short_att": 1.7,
        "mean_att": float(draws[:, 0].mean()),
        "sd_att": float(draws[:, 0].std(ddof=1)),
        "mean_se": float(draws[:, 1].mean()),
        "bias": float(draws[:, 0].mean() - 1.7),
        "coverage": float(draws[:, 2].mean()),
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")


if __name__ == "__main__":
    main()
