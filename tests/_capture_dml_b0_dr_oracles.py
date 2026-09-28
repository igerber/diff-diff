"""Capture numeric oracles for the PR-B0 DR-score relocation.

Run this on the UNMODIFIED tree (before `ContinuousDiD._dr_cell_inf_func` is
lifted into ``diff_diff/_dr_scores.py``) to (re)generate the hardcoded literals
in ``tests/test_dr_scores.py``::

    DIFF_DIFF_BACKEND=python python tests/_capture_dml_b0_dr_oracles.py

The DGP helpers below are imported by the consuming test so the capture and
the assertion always run the same inputs. Two tiers are captured:

1. Function tier — a fixed synthetic ``(dY, D, X, gamma, ps)`` input and the
   full per-unit influence-function vector from ``_dr_cell_inf_func``.
2. Estimator tier — ``ContinuousDiD(estimation_method="dr")`` on a covariate
   panel (the DR path runs only when covariates are supplied to ``fit``),
   capturing ``overall_att`` / ``overall_att_se`` / ``overall_acrt`` /
   ``overall_acrt_se``.
"""

import numpy as np

from diff_diff import ContinuousDiD
from diff_diff.prep_dgp import generate_continuous_did_data


def function_tier_inputs():
    """Fixed synthetic inputs for the function-tier oracle (treated-then-control)."""
    rng = np.random.default_rng(20260822)
    n_t, n_c = 6, 10
    n = n_t + n_c
    D = np.concatenate([np.ones(n_t), np.zeros(n_c)])
    # X includes an intercept column (the estimator's convention at the call site).
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    gamma = np.array([0.5, 1.0, -0.25])
    dY = X @ gamma + 2.0 * D + rng.normal(scale=0.5, size=n)
    ps = np.clip(1.0 / (1.0 + np.exp(-(0.3 * X[:, 1] - 0.2 * X[:, 2]))), 0.05, 0.95)
    return dY, D, X, gamma, ps


def estimator_tier_data(seed=5, n_units=120):
    """Covariate panel for the estimator-tier oracle (mirrors tests/test_continuous_did.py::_cov_data)."""
    data = generate_continuous_did_data(n_units=n_units, n_periods=3, seed=seed, noise_sd=0.5)
    rng = np.random.default_rng(seed)
    uc = data.groupby("unit").ngroup().to_numpy()
    per_unit = rng.normal(size=data["unit"].nunique())
    data["x1"] = per_unit[uc]
    return data


def estimator_tier_fit():
    """The estimator-tier fit whose outputs are pinned as literals."""
    est = ContinuousDiD(estimation_method="dr", n_bootstrap=0)
    return est.fit(
        estimator_tier_data(),
        "outcome",
        "unit",
        "period",
        "first_treat",
        "dose",
        covariates=["x1"],
    )


def main():
    np.set_printoptions(precision=17, floatmode="unique")

    dY, D, X, gamma, ps = function_tier_inputs()
    est = ContinuousDiD(estimation_method="dr")
    inf = est._dr_cell_inf_func(dY, D, X, gamma, ps)
    print("# Function tier: per-unit DR influence function (treated-then-control)")
    print("FUNCTION_TIER_IF = np.array(")
    print(repr(inf.tolist()))
    print(")")

    res = estimator_tier_fit()
    print("# Estimator tier: ContinuousDiD DR covariate path")
    print(f"ESTIMATOR_TIER_OVERALL_ATT = {res.overall_att!r}")
    print(f"ESTIMATOR_TIER_OVERALL_ATT_SE = {res.overall_att_se!r}")
    print(f"ESTIMATOR_TIER_OVERALL_ACRT = {res.overall_acrt!r}")
    print(f"ESTIMATOR_TIER_OVERALL_ACRT_SE = {res.overall_acrt_se!r}")


if __name__ == "__main__":
    main()
