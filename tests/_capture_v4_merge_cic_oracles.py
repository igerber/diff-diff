"""Step 0: capture the pre-merge oracles for tests/test_v4_merge_cic.py.

MUST run on the UNMODIFIED tree (before ``method=`` lands on ChangesInChanges),
under DIFF_DIFF_BACKEND=python:

    DIFF_DIFF_BACKEND=python python3 tests/_capture_v4_merge_cic_oracles.py

Unlike the 3(b) DDD capture, this one does NOT guard a relocation - nothing
moves in 3(c). It guards the two things a same-process parity gate cannot see:
a regression shared by both callers of ``_fit_distributional`` (e.g. a broken
``_validate_all_params`` or a changed default), and a drift in the values the
``kind`` dispatch selects.

Scope decisions, both deliberate:

  * UNCONDITIONAL arm only. The covariate quantile-regression path is
    tie-selection-bounded with BLAS-dependent tie flips (see
    tests/test_changes_in_changes_parity.py: COV_ATT_ATOL = 0.04,
    COV_QTE_ATOL = 0.25) and CI runs ubuntu/macos/windows/arm, so committed
    literals at 1e-9 would be platform-fragile. Covariate coverage comes from
    the in-process bit-exact parity gate, which cannot be platform-dependent.

  * ``quantile_effects`` is NOT captured. benchmarks/data/qte_golden.json is
    git-tracked, so test_changes_in_changes_parity.py::test_point_parity never
    skips and already pins quantile_effects["qte"] against the R qte 1.3.1
    golden at atol=1e-10, rtol=0 for cic/qdid x panel/rcs. That is a stronger
    absolute cross-tree pin than anything this file would add.

The DGP lives HERE, not in the test module: this script runs at step 1 while
the test module is written at step 7, so the dependency must point this way.
tests/test_v4_merge_cic.py imports ``make_2x2`` from this module.

PROVENANCE of the literals committed in tests/test_v4_merge_cic.py:
    commit  c2941caa2a7865c9458c6092359e75238fdbabb1  (origin/main, pre-3(c))
    command DIFF_DIFF_BACKEND=python python3 tests/_capture_v4_merge_cic_oracles.py
    tree    clean apart from this file (no source edits had landed)

Re-running this on the merged tree must reproduce the same literals - that is
the gate. Note the qdid arms legitimately emit the Athey-Imbens footnote-21
non-monotonicity UserWarning on this DGP (that is the restriction QDiD places
on the data), so the capture is not warning-free and callers must not treat a
warning here as a failure.
"""

import json

import numpy as np
import pandas as pd

from diff_diff import ChangesInChanges, QDiD


def make_2x2(n_treated=60, n_control=80, seed=0, effect=1.0):
    """Full-overlap continuous 2x2 panel (long format, one row per unit-period).

    Mirrors tests/test_changes_in_changes.py's helper of the same name. The
    ``id`` column repeats across periods, so the same frame serves both
    ``panel=False`` (pooled row resample) and ``panel=True`` (unit-block
    resample, ``unit="id"``).
    """
    rng = np.random.default_rng(seed)
    n = n_treated + n_control
    treat = np.repeat([1, 0], [n_treated, n_control])
    u = rng.normal(0, 1, n)
    y_pre = u + rng.normal(0, 0.3, n)
    y_post = u + 0.5 + rng.normal(0, 0.3, n) + treat * effect
    return pd.DataFrame(
        {
            "id": np.tile(np.arange(n), 2),
            "post": np.repeat([0, 1], n),
            "treated": np.tile(treat, 2),
            "y": np.concatenate([y_pre, y_post]),
        }
    )


# Every knob is pinned explicitly - they ARE the oracle's meaning.
N_TREATED = 60
N_CONTROL = 80
DGP_SEED = 0
EFFECT = 1.0
QUANTILES = None  # the default 0.05-0.95 grid
ALPHA = 0.05
BOOT_N = 49
BOOT_SEED = 7


def _record(res):
    lo, hi = res.conf_int
    return {
        "att": float(res.att),
        "se": float(res.se),
        "t_stat": float(res.t_stat),
        "p_value": float(res.p_value),
        "conf_int_lower": float(lo),
        "conf_int_upper": float(hi),
        "q_lower": float(res.q_lower),
        "q_upper": float(res.q_upper),
        "sup_t_crit": float(res.sup_t_crit),
        "n_obs": int(res.n_obs),
        "n_bootstrap_valid": int(res.n_bootstrap_valid),
        "cell_sizes": {k: int(v) for k, v in res.cell_sizes.items()},
    }


def main():
    df = make_2x2(n_treated=N_TREATED, n_control=N_CONTROL, seed=DGP_SEED, effect=EFFECT)
    out = {}

    for label, cls in (("cic", ChangesInChanges), ("qdid", QDiD)):
        for panel in (False, True):
            est = cls(quantiles=QUANTILES, n_bootstrap=0, alpha=ALPHA, panel=panel)
            if panel:
                res = est.fit(df, outcome="y", treatment="treated", time="post", unit="id")
            else:
                res = est.fit(df, outcome="y", treatment="treated", time="post")
            out[f"{label}_panel{int(panel)}_nb0"] = _record(res)

        est = cls(quantiles=QUANTILES, n_bootstrap=BOOT_N, alpha=ALPHA, panel=False, seed=BOOT_SEED)
        out[f"{label}_panel0_nb{BOOT_N}"] = _record(
            est.fit(df, outcome="y", treatment="treated", time="post")
        )

    print(json.dumps(out, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
