"""DoubleML RCS CHARACTERIZATION spike for the DMLDiD panel=False lane.

CHARACTERIZATION, NOT PARITY. ``DoubleMLDIDCSBinary`` (doubleml 0.11.4;
``score="observational"``, ``in_sample_normalization=False``, trimming 0.01)
implements the Sant'Anna-Zhao repeated-cross-section score — FOUR
treatment-by-period outcome regressions — while Chang (2020) Equation 3.2
uses ONE control-only regression of ``(T - lam) * y`` on X, and
DoubleML's variance omits Chang's Theorem 2 ``G_2lambda * (T - lam)``
correction entirely. The two estimators are consistent for the same
estimand but carry different first-order influence functions and
finite-sample values: ATT gaps are O_p(N^{-1/2}) by construction and SE
gaps additionally carry the lambda term. This spike DOCUMENTS both gaps per cell
(and reports the lambda-omitted Chang SE beside them); the machine-precision
assertions here are SELF-parity — the public ``DMLDiD(panel=False)``
estimator vs the hand-rolled Chang pipeline under DMLDiD's own
reconstructed folds (Part 2), whose 12-decimal literals are the goldens
consumed by tests/test_methodology_dml_did.py::TestRCSGoldenCharacterization
(native reproduction swaps sklearn's lbfgs logit for the library IRLS
solver; tolerance atol 2e-4 ATT / 1e-5 SE, the B0/B1 precedent).

Environment (side venv; doubleml/sklearn are NEVER diff-diff dependencies):

    python -m venv .venv-doubleml
    .venv-doubleml/bin/pip install "doubleml==0.11.4" scikit-learn
    .venv-doubleml/bin/python benchmarks/doubleml/chang_rcs_characterization.py

Observed transcript (2026-08-26, doubleml 0.11.4, sklearn 1.9.0, macOS arm64):

    Part 1 — Chang Eq 3.2 vs DoubleMLDIDCSBinary (shared folds; gaps EXPECTED):
    cell (g=3, t=3, base=2)  CHANG ATT = 1.875325609339  SE(full) = 0.243971253081  SE(no-lam) = 0.249877072301 | DML ATT = 2.121300323064  SE = 0.109099376634 | ATT gap -2.460e-01  SE gap +1.349e-01  SE(no-lam) gap +1.408e-01
    cell (g=3, t=4, base=2)  CHANG ATT = 2.360914148301  SE(full) = 0.252067309584  SE(no-lam) = 0.258532386882 | DML ATT = 2.293055793734  SE = 0.106740749269 | ATT gap +6.786e-02  SE gap +1.453e-01  SE(no-lam) gap +1.518e-01
    cell (g=4, t=3, base=2)  CHANG ATT = -0.075262310611  SE(full) = 0.187206946073  SE(no-lam) = 0.187207274463 | DML ATT = -0.057963388953  SE = 0.106181054127 | ATT gap -1.730e-02  SE gap +8.103e-02  SE(no-lam) gap +8.103e-02
    cell (g=4, t=4, base=3)  CHANG ATT = 1.799124069195  SE(full) = 0.260544978066  SE(no-lam) = 0.265671872127 | DML ATT = 1.995045506760  SE = 0.105732614154 | ATT gap -1.959e-01  SE gap +1.548e-01  SE(no-lam) gap +1.599e-01
    CHARACTERIZATION OK (nonzero, bounded gaps — different scores, same estimand)

    Part 2 — public DMLDiD(panel=False) vs hand Chang (DMLDiD's own folds):
    cell (g=3, t=3, base=2)  DMLDiD ATT = 1.867260790882  SE = 0.244040011648 | diff vs hand ATT +0.000e+00  SE +0.000e+00
    cell (g=3, t=4, base=2)  DMLDiD ATT = 2.347579854514  SE = 0.251882814022 | diff vs hand ATT +0.000e+00  SE +0.000e+00
    cell (g=4, t=3, base=2)  DMLDiD ATT = -0.089752999600  SE = 0.187258740736 | diff vs hand ATT +0.000e+00  SE +0.000e+00
    cell (g=4, t=4, base=3)  DMLDiD ATT = 1.822066292980  SE = 0.260472487961 | diff vs hand ATT +0.000e+00  SE +0.000e+00
    SELF-PARITY OK (public DMLDiD cells within 1e-10 of the hand Chang pipeline)

Interpretation. The ATT gaps (1.7e-2 to 2.5e-1) are finite-sample
differences between two DIFFERENT orthogonal scores estimating the same
estimand — the pre-registered outcome for a characterization, and exactly
why the REGISTRY records DoubleMLDIDCSBinary as "not an oracle" for
Case 2. The SE columns are NOT directly comparable beyond direction: on
top of the missing lambda term, DoubleMLDIDCSBinary scatters its
per-cell score to the FULL frame with zero fill (did_cs_binary.py's
_set_id_positions), so its variance scaling factor is the full N rather
than the cell's pooled row count — the transcript records the observed
values rather than claiming a decomposition of the gap. The lambda
term's own effect is isolated in the SE(full)-vs-SE(no-lam) columns
(same pipeline, one term removed): up to ~2.6% relative on these cells,
and near-zero exactly where lam_hat ~ 0.5 (cell (4,3): the (1-2*lam)
factor vanishes) — matching the closed form. Part 2's zero diffs are the
real precision anchor: the shipped estimator IS the hand-rolled
Equation 3.2 / Theorem 2 pipeline under identical folds and learners.
"""

import numpy as np
import pandas as pd
from doubleml.data import DoubleMLPanelData
from doubleml.did import DoubleMLDIDCSBinary
from sklearn.linear_model import LinearRegression, LogisticRegression

SEED = 7
N_ROWS = 4000
PERIODS = [1, 2, 3, 4]
COHORTS = [0, 3, 4]  # never-treated + two staggered cohorts
K = 5
TRIM = 1e-2

rng = np.random.default_rng(SEED)
cohort = rng.choice(COHORTS, size=N_ROWS, p=[0.5, 0.25, 0.25])
tt = rng.choice(PERIODS, size=N_ROWS)
X_row = rng.standard_normal((N_ROWS, 2))
y = (
    0.8 * X_row[:, 0]
    - 0.4 * X_row[:, 1]
    + 0.3 * tt
    + 0.5 * X_row[:, 0] * tt / 4  # covariate-dependent trend (X matters)
    + rng.standard_normal(N_ROWS)
)
post = (cohort > 0) & (tt >= cohort)
y = y + post * (2.0 + 0.2 * (tt - cohort))  # heterogeneous dynamic effect

df = pd.DataFrame(
    {
        "id": np.arange(N_ROWS),
        "t": tt,
        "g": cohort,
        "y": y,
        "x1": X_row[:, 0],
        "x2": X_row[:, 1],
    }
)
panel = DoubleMLPanelData(df, y_col="y", d_cols="g", t_col="t", id_col="id", x_cols=["x1", "x2"])

CELLS = [(3, 3, 2), (3, 4, 2), (4, 3, 2), (4, 4, 3)]  # (g, t_eval, base)


def chang_cell(idx, t_eval, g_val, smpls):
    """Hand-rolled Chang Eq 3.2 cell under the given fold splits.

    Returns (theta, se_full, se_no_lambda) — se_full carries the Theorem 2
    G_2lambda correction, se_no_lambda deliberately omits it (the
    'plausible implementation bug' the review warns against), so the
    transcript records how much the lambda term moves the SE.
    """
    D = (cohort[idx] == g_val).astype(float)
    T = (tt[idx] == t_eval).astype(float)
    yv = y[idx]
    X = X_row[idx]
    n = idx.shape[0]
    p_hat = D.mean()
    lam = T.mean()
    oof_g = np.empty(n)
    oof_l = np.empty(n)
    r = (T - lam) * yv
    for tr, te in smpls:
        lg = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X[tr], D[tr])
        oof_g[te] = np.clip(lg.predict_proba(X[te])[:, 1], TRIM, 1 - TRIM)
        ctrl = tr[D[tr] == 0]
        oof_l[te] = LinearRegression().fit(X[ctrl], r[ctrl]).predict(X[te])
    w = (D - oof_g) / (p_hat * lam * (1 - lam) * (1 - oof_g))
    summand = w * ((T - lam) * yv - oof_l)
    theta = summand.mean()
    odds = (D - oof_g) / (1 - oof_g)
    g2 = np.mean(
        -((1 - 2 * lam) / (lam**2 * (1 - lam) ** 2)) * (odds / p_hat) * ((T - lam) * yv - oof_l)
        - (yv / (p_hat * lam * (1 - lam))) * odds
    )
    psi_full = summand - D * theta / p_hat + g2 * (T - lam)
    psi_no_lam = summand - D * theta / p_hat
    se_full = np.sqrt(np.mean(psi_full**2) / n)
    se_no_lam = np.sqrt(np.mean(psi_no_lam**2) / n)
    return float(theta), float(se_full), float(se_no_lam)


# ---------------------------------------------------------------------------
# Part 1 — characterization vs DoubleMLDIDCSBinary under SHARED folds.
# ---------------------------------------------------------------------------
print("Part 1 — Chang Eq 3.2 vs DoubleMLDIDCSBinary (shared folds; gaps EXPECTED):")
gaps_nonzero = True
for g_val, t_eval, base in CELLS:
    in_cell = ((cohort == g_val) | (cohort == 0)) & ((tt == t_eval) | (tt == base))
    idx = np.flatnonzero(in_cell)
    n = idx.shape[0]
    D = (cohort[idx] == g_val).astype(float)
    T = (tt[idx] == t_eval).astype(float)

    cell_rng = np.random.default_rng(1000 + g_val * 10 + t_eval)
    strata = (D + 2 * T).astype(int)
    test_folds = [[] for _ in range(K)]
    cursor = 0
    for s_val in np.unique(strata):
        members = np.flatnonzero(strata == s_val)
        members = cell_rng.permutation(members)
        for m_i in members:
            test_folds[cursor % K].append(m_i)
            cursor += 1
    test_folds = [np.sort(np.array(f, dtype=int)) for f in test_folds]
    smpls = [(np.setdiff1d(np.arange(n), te), te) for te in test_folds]

    m = DoubleMLDIDCSBinary(
        panel,
        g_value=g_val,
        t_value_pre=base,
        t_value_eval=t_eval,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000),
        control_group="never_treated",
        n_folds=K,
        n_rep=1,
        score="observational",
        in_sample_normalization=False,
        trimming_threshold=TRIM,
        draw_sample_splitting=False,
    )
    m.set_sample_splitting([smpls])
    m.fit()
    att_dml, se_dml = float(m.coef[0]), float(m.se[0])

    theta, se_full, se_no_lam = chang_cell(idx, t_eval, g_val, smpls)
    d_att = theta - att_dml
    print(
        f"cell (g={g_val}, t={t_eval}, base={base})  CHANG ATT = {theta:.12f}  "
        f"SE(full) = {se_full:.12f}  SE(no-lam) = {se_no_lam:.12f} | "
        f"DML ATT = {att_dml:.12f}  SE = {se_dml:.12f} | "
        f"ATT gap {d_att:+.3e}  SE gap {se_full - se_dml:+.3e}  "
        f"SE(no-lam) gap {se_no_lam - se_dml:+.3e}"
    )
    # Characterization honesty: the scores are DIFFERENT — a zero gap would
    # mean the two implementations coincide, which they must not.
    gaps_nonzero &= abs(d_att) > 1e-12
    # Sanity band: both estimate the same estimand.
    gaps_nonzero &= abs(d_att) < 0.5

print(
    "CHARACTERIZATION OK (nonzero, bounded gaps — different scores, same estimand)"
    if gaps_nonzero
    else "CHARACTERIZATION FAILED"
)

# ---------------------------------------------------------------------------
# Part 2 — SELF-parity: public DMLDiD(panel=False) vs the hand-rolled Chang
# pipeline under DMLDiD's OWN reconstructed folds (sklearn learners on both
# sides — machine precision expected). The printed 12-decimal CHANG-side
# numbers are the goldens for TestRCSGoldenCharacterization.
# ---------------------------------------------------------------------------
import sys  # noqa: E402

sys.path.insert(0, ".")
from diff_diff import DMLDiD  # noqa: E402
from diff_diff._crossfit import assign_folds  # noqa: E402

SEED_DML = 11
est = DMLDiD(
    propensity_learner=LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000),
    outcome_learner=LinearRegression(),
    seed=SEED_DML,
    n_folds=K,
    pscore_trim=TRIM,
    panel=False,
)
import warnings  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = est.fit(df, outcome="y", unit="id", time="t", first_treat="g", covariates=["x1", "x2"])

sorted_cohorts = [3, 4]
sorted_periods = PERIODS
ok2 = True
print("\nPart 2 — public DMLDiD(panel=False) vs hand Chang (DMLDiD's own folds):")
for g_val, t_eval, base in CELLS:
    entry = res.group_time_effects[(g_val, t_eval)]
    in_cell = ((cohort == g_val) | (cohort == 0)) & ((tt == t_eval) | (tt == base))
    idx = np.flatnonzero(in_cell)
    n = idx.shape[0]
    D = (cohort[idx] == g_val).astype(float)
    T = (tt[idx] == t_eval).astype(float)
    g_idx = sorted_cohorts.index(g_val)
    t_idx = sorted_periods.index(t_eval)
    rng2 = np.random.default_rng(np.random.SeedSequence(entropy=SEED_DML, spawn_key=(g_idx, t_idx)))
    folds = assign_folds(n, K, rng=rng2, stratify=D + 2.0 * T)
    smpls = [
        (np.flatnonzero(folds.fold_ids != k), np.flatnonzero(folds.fold_ids == k)) for k in range(K)
    ]
    theta, se_full, _ = chang_cell(idx, t_eval, g_val, smpls)
    d_att = float(entry["effect"]) - theta
    d_se = float(entry["se"]) - se_full
    print(
        f"cell (g={g_val}, t={t_eval}, base={base})  DMLDiD ATT = {entry['effect']:.12f}  "
        f"SE = {entry['se']:.12f} | diff vs hand ATT {d_att:+.3e}  SE {d_se:+.3e}"
    )
    ok2 &= abs(d_att) < 1e-10 and abs(d_se) < 1e-10

print(
    "SELF-PARITY OK (public DMLDiD cells within 1e-10 of the hand Chang pipeline)"
    if ok2
    else "SELF-PARITY FAILED"
)
raise SystemExit(0 if (gaps_nonzero and ok2) else 1)
