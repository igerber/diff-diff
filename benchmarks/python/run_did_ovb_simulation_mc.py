#!/usr/bin/env python3
"""Monte Carlo diagnostic for the Appendix E.1 parametric DGP."""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.special import expit

from diff_diff.linalg import solve_logit, solve_ols


def one_rep(n: int, p: float, seed: int, alpha: float, misspecified: bool) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = rng.binomial(1, p, n)
    x = rng.normal(np.where(d == 0, 0.3, 0.0), np.where(d == 0, np.sqrt(6), np.sqrt(3)))
    u = rng.normal(np.where(d == 0, 0.3, 0.0), np.where(d == 0, np.sqrt(6), np.sqrt(3)))
    dy = 1 + x + u + 2 * d + rng.normal(0, np.sqrt(2), n)
    x_fit = np.exp(x / 2) if misspecified else x
    x_short = np.column_stack([x_fit])
    x_prop = np.column_stack([x_fit, x_fit**2])
    ps = np.empty(n)
    m = np.empty(n)
    folds = np.arange(n) % 10
    for fold in range(10):
        test = folds == fold
        train = ~test
        beta_p, _ = solve_logit(x_prop[train], d[train], max_iter=25, tol=1e-8)
        eta = np.column_stack([np.ones(test.sum()), x_prop[test]]) @ beta_p
        ps[test] = expit(eta)
        beta_m, _, _ = solve_ols(
            np.column_stack([np.ones((np.sum(train & (d == 0)), 1)), x_short[train & (d == 0)]]),
            dy[train & (d == 0)],
            return_vcov=False,
        )
        m[test] = np.column_stack([np.ones(test.sum()), x_short[test]]) @ beta_m
    ps = np.clip(ps, 0.01, 0.99)
    phat = float(d.mean())
    omega = (ps / (1 - ps)) / (phat / (1 - phat))
    residual = dy - m
    score = (d / phat - (1 - d) / (1 - phat) * omega) * residual
    att = float(score.mean())
    se = float(np.sqrt(np.mean((score - att) ** 2) / n))
    sigma2 = float(np.mean(residual[d == 0] ** 2))
    nu2 = float(np.mean(omega[d == 0] ** 2))
    scale = float(np.sqrt(sigma2 * nu2))
    theta_if = score - att
    sigma_if = (1 - d) / (1 - phat) * (residual**2 - sigma2)
    nu_if = (1 - d) / (1 - phat) * (omega**2 - nu2)
    scale_if = (nu2 * sigma_if + sigma2 * nu_if) / (2 * scale)
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    lo = att - z * se
    hi = att + z * se

    def contains(strength: float, kind: str) -> bool:
        if kind == "rv":
            multiplier = strength / np.sqrt(1 - strength)
        else:
            multiplier = np.sqrt(strength / (1 - strength))
        radius = multiplier * scale
        lower_if = theta_if - multiplier * scale_if
        upper_if = theta_if + multiplier * scale_if
        lower_se = np.sqrt(np.mean(lower_if**2) / n)
        upper_se = np.sqrt(np.mean(upper_if**2) / n)
        return att - radius - z * lower_se <= 0 <= att + radius + z * upper_se

    def rv(kind: str) -> float:
        if contains(1 - 1e-12, kind):
            lo_s, hi_s = 0.0, 1 - 1e-12
            for _ in range(70):
                mid = (lo_s + hi_s) / 2
                if contains(mid, kind):
                    hi_s = mid
                else:
                    lo_s = mid
            return hi_s
        return float("nan")

    return att, se, float(lo <= 1.7 <= hi), rv("rv"), rv("xrv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--misspecified",
        action="store_true",
        help="fit nuisance models using X*=exp(X/2), as in Appendix E.2",
    )
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        raise SystemExit("--alpha must be in (0, 1)")
    draws = np.asarray(
        [
            one_rep(args.n, args.p, args.seed + i, args.alpha, args.misspecified)
            for i in range(args.reps)
        ]
    )
    result = {
        "n": args.n,
        "reps": args.reps,
        "p": args.p,
        "alpha": args.alpha,
        "misspecified": args.misspecified,
        "seed": args.seed,
        "true_short_att": 1.7,
        "mean_att": float(draws[:, 0].mean()),
        "sd_att": float(draws[:, 0].std(ddof=1)),
        "mean_se": float(draws[:, 1].mean()),
        "bias": float(draws[:, 0].mean() - 1.7),
        "coverage": float(draws[:, 2].mean()),
        "mean_rv": float(np.nanmean(draws[:, 3])),
        "mean_xrv": float(np.nanmean(draws[:, 4])),
        "rv_sd": float(np.nanstd(draws[:, 3], ddof=1)),
        "xrv_sd": float(np.nanstd(draws[:, 4], ddof=1)),
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")


if __name__ == "__main__":
    main()
