"""DoubleML STAGGERED parity spike for the DMLDiD per-(g,t) cell estimator.

Per cell, ``DoubleMLDIDBinary`` (the per-cell estimator ``DoubleMLDIDMulti``
orchestrates over staggered timing) under pinned config —
``score="observational"``, ``in_sample_normalization=False``, panel data,
``trimming_threshold=0.01`` — and SHARED fold assignments, vs the
hand-rolled Chang (2020) Case 1 cell: cross-fitted logit propensity +
control-only linear outcome-change regression, global p-hat, the
``chang_panel_score`` mean, and the augmented-score SE. This is the
staggered anchor cited by the REGISTRY "DMLDiD" section; the 2-period
anchor (``DoubleMLDID``) lives in ``chang_case1_parity.py`` and is kept.

Environment (side venv; doubleml/sklearn are NEVER diff-diff dependencies):

    python -m venv .venv-doubleml
    .venv-doubleml/bin/pip install "doubleml==0.11.4" scikit-learn
    .venv-doubleml/bin/python benchmarks/doubleml/chang_staggered_parity.py

Observed transcript (2026-08-25, doubleml 0.11.4, sklearn 1.9.0, macOS arm64):

    cell (g=3, t=3, base=2)  DML ATT = 2.030643126019  SE = 0.173107723124 | hand diff ATT +4.441e-16  SE +8.327e-17
    cell (g=3, t=4, base=2)  DML ATT = 2.268834143564  SE = 0.187095112140 | hand diff ATT +8.882e-16  SE -2.776e-17
    cell (g=4, t=3, base=2)  DML ATT = -0.177672235406  SE = 0.167181769564 | hand diff ATT +8.327e-17  SE -2.776e-17
    cell (g=4, t=4, base=3)  DML ATT = 2.139119845409  SE = 0.186007089606 | hand diff ATT -4.441e-16  SE +5.551e-17
    PARITY OK (every cell's ATT and SE within 1e-10 of DoubleMLDIDBinary)

    Part 2 — public DMLDiD.fit() vs DoubleMLDIDBinary (shared DMLDiD folds):
    cell (g=3, t=3, base=2)  DMLDiD ATT = 2.013640496612  SE = 0.173611909306 | diff vs DML ATT +8.882e-16  SE +8.327e-17
    cell (g=3, t=4, base=2)  DMLDiD ATT = 2.312506769326  SE = 0.182958385142 | diff vs DML ATT +4.441e-16  SE +0.000e+00
    cell (g=4, t=3, base=2)  DMLDiD ATT = -0.153758273010  SE = 0.164671337301 | diff vs DML ATT +5.551e-17  SE +0.000e+00
    cell (g=4, t=4, base=3)  DMLDiD ATT = 2.157511261169  SE = 0.191218867463 | diff vs DML ATT -4.441e-16  SE -2.776e-17
    END-TO-END PARITY OK (public DMLDiD cells within 1e-10 of DoubleMLDIDBinary)

The golden literals above are consumed (with a dependency-free native
reproduction) by tests/test_methodology_dml_did.py::TestDoubleMLGoldenParity;
the native reproduction swaps sklearn's lbfgs logit for the library IRLS
solver, so its tolerance is atol 2e-4 (ATT) / 1e-5 (SE) — the measured
IRLS-vs-lbfgs optimizer gap, the B0 precedent — while THIS spike asserts
machine precision against DoubleML under identical (sklearn) learners.
"""

import numpy as np
import pandas as pd
from doubleml.data import DoubleMLPanelData
from doubleml.did import DoubleMLDIDBinary
from sklearn.linear_model import LinearRegression, LogisticRegression

SEED = 7
N_UNITS = 400
PERIODS = [1, 2, 3, 4]
COHORTS = [0, 3, 4]  # never-treated + two staggered cohorts
K = 5
TRIM = 1e-2

rng = np.random.default_rng(SEED)
cohort = rng.choice(COHORTS, size=N_UNITS, p=[0.5, 0.25, 0.25])
X_unit = rng.standard_normal((N_UNITS, 2))

rows = []
for i in range(N_UNITS):
    for t in PERIODS:
        y = (
            0.8 * X_unit[i, 0]
            - 0.4 * X_unit[i, 1]
            + 0.3 * t
            + 0.5 * X_unit[i, 0] * t / 4  # covariate-dependent trend (X matters)
            + rng.standard_normal()
        )
        if cohort[i] > 0 and t >= cohort[i]:
            y += 2.0 + 0.2 * (t - cohort[i])  # heterogeneous dynamic effect
        rows.append((i, t, cohort[i], y, X_unit[i, 0], X_unit[i, 1]))
df = pd.DataFrame(rows, columns=["id", "t", "g", "y", "x1", "x2"])

panel = DoubleMLPanelData(df, y_col="y", d_cols="g", t_col="t", id_col="id", x_cols=["x1", "x2"])

# The DMLDiD cell grid (varying base, never-treated controls): post cells only
# here — DoubleMLDIDBinary evaluates g-vs-never cells with base < g.
CELLS = [(3, 3, 2), (3, 4, 2), (4, 3, 2), (4, 4, 3)]  # (g, t_eval, base)

ok = True
for g, t_eval, base in CELLS:
    # Cell arrays exactly as DMLDiD builds them: treated = cohort g,
    # controls = never-treated, dY = y(t_eval) - y(base), X from the cell.
    in_cell = (cohort == g) | (cohort == 0)
    idx = np.flatnonzero(in_cell)
    y_eval = df[df["t"] == t_eval].set_index("id").loc[idx, "y"].to_numpy()
    y_base = df[df["t"] == base].set_index("id").loc[idx, "y"].to_numpy()
    dY = y_eval - y_base
    D = (cohort[idx] == g).astype(float)
    X = X_unit[idx]
    n = idx.shape[0]

    # Shared fold assignment (same replayable partition for both sides).
    cell_rng = np.random.default_rng(1000 + g * 10 + t_eval)
    perm = cell_rng.permutation(n)
    test_folds = [np.sort(perm[k::K]) for k in range(K)]
    smpls = [(np.setdiff1d(np.arange(n), te), te) for te in test_folds]

    m = DoubleMLDIDBinary(
        panel,
        g_value=g,
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
    # DoubleMLDIDBinary's internal panel ordering must match our cell arrays
    # for the shared splitting to be meaningful; its data are id-sorted, as
    # are ours (idx ascending).
    m.set_sample_splitting([smpls])
    m.fit()
    att_dml, se_dml = float(m.coef[0]), float(m.se[0])

    # Hand-rolled per-cell Chang (global p-hat; augmented-score SE).
    p_glob = D.mean()
    oof_g = np.empty(n)
    oof_l = np.empty(n)
    for tr, te in smpls:
        lg = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X[tr], D[tr])
        oof_g[te] = np.clip(lg.predict_proba(X[te])[:, 1], TRIM, 1 - TRIM)
        ctrl = tr[D[tr] == 0]
        oof_l[te] = LinearRegression().fit(X[ctrl], dY[ctrl]).predict(X[te])
    summand = (D - oof_g * (1 - D) / (1 - oof_g)) * (dY - oof_l) / p_glob
    theta = summand.mean()
    psi_bar = summand - D * theta / p_glob
    se = np.sqrt(np.mean(psi_bar**2) / n)

    d_att, d_se = theta - att_dml, se - se_dml
    print(
        f"cell (g={g}, t={t_eval}, base={base})  DML ATT = {att_dml:.12f}  "
        f"SE = {se_dml:.12f} | hand diff ATT {d_att:+.3e}  SE {d_se:+.3e}"
    )
    ok &= abs(d_att) < 1e-10 and abs(d_se) < 1e-10

print(
    "PARITY OK (every cell's ATT and SE within 1e-10 of DoubleMLDIDBinary)"
    if ok
    else "PARITY FAILED"
)

# ---------------------------------------------------------------------------
# Part 2 — END-TO-END public-estimator parity: DMLDiD.fit() itself (sklearn
# learner objects via the duck-typed protocol) vs DoubleMLDIDBinary under
# DMLDiD's OWN seeded fold assignments (reconstructed via assign_folds), so
# public cell construction, base-period selection, IF scattering, and result
# wiring are all on the compared path — not just the score formula.
# ---------------------------------------------------------------------------
import sys  # noqa: E402

sys.path.insert(0, ".")
from diff_diff import DMLDiD  # noqa: E402
from diff_diff._crossfit import assign_folds  # noqa: E402

SEED_DML = 11
long_df = df.rename(columns={})  # same frame; DMLDiD takes long format directly

est = DMLDiD(
    propensity_learner=LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000),
    outcome_learner=LinearRegression(),
    seed=SEED_DML,
    n_folds=K,
    pscore_trim=TRIM,
)
res = est.fit(long_df, outcome="y", unit="id", time="t", first_treat="g", covariates=["x1", "x2"])

sorted_cohorts = [3, 4]
sorted_periods = PERIODS

ok2 = True
print("\nPart 2 — public DMLDiD.fit() vs DoubleMLDIDBinary (shared DMLDiD folds):")
for g, t_eval, base in CELLS:
    idx = np.flatnonzero((cohort == g) | (cohort == 0))
    D = (cohort[idx] == g).astype(float)
    n = idx.shape[0]
    g_idx = sorted_cohorts.index(g)
    t_idx = sorted_periods.index(t_eval)
    rng2 = np.random.default_rng(np.random.SeedSequence(entropy=SEED_DML, spawn_key=(g_idx, t_idx)))
    folds = assign_folds(n, K, rng=rng2, stratify=D)
    smpls = [
        (np.flatnonzero(folds.fold_ids != k), np.flatnonzero(folds.fold_ids == k)) for k in range(K)
    ]
    m2 = DoubleMLDIDBinary(
        panel,
        g_value=g,
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
    m2.set_sample_splitting([smpls])
    m2.fit()
    cell = res.group_time_effects[(g, t_eval)]
    d_att = cell["effect"] - float(m2.coef[0])
    d_se = cell["se"] - float(m2.se[0])
    print(
        f"cell (g={g}, t={t_eval}, base={base})  DMLDiD ATT = {cell['effect']:.12f}  "
        f"SE = {cell['se']:.12f} | diff vs DML ATT {d_att:+.3e}  SE {d_se:+.3e}"
    )
    ok2 &= abs(d_att) < 1e-10 and abs(d_se) < 1e-10

print(
    "END-TO-END PARITY OK (public DMLDiD cells within 1e-10 of DoubleMLDIDBinary)"
    if ok2
    else "END-TO-END PARITY FAILED"
)
raise SystemExit(0 if (ok and ok2) else 1)
