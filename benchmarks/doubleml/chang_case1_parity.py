"""DoubleML parity spike for the Chang (2020) Case 1 (repeated outcomes) score.

Hand-rolled DML2 cross-fitted Chang estimator vs DoubleMLDID under identical
folds, learners, and clipping. This is the reproducible anchor cited by the
REGISTRY "Cross-fitting, DR-score, and ridge infrastructure (DML)" section's
global-p-hat Note.

Environment (side venv; doubleml/sklearn are NEVER diff-diff dependencies):

    python -m venv .venv-doubleml
    .venv-doubleml/bin/pip install "doubleml==0.11.4" scikit-learn
    .venv-doubleml/bin/python benchmarks/doubleml/chang_case1_parity.py

Observed transcript (2026-08-22, doubleml 0.11.4, sklearn 1.9.0, macOS arm64):

    DoubleML              ATT = 3.260530717619   SE = 0.173029660048
    hand Chang (global p) ATT = 3.260530717619   SE = 0.173029660048
      diff vs DoubleML:   ATT -4.441e-16   SE +5.551e-17
    hand Chang (fold-mean p) ATT = 3.262788670382   (finite-sample gap +2.258e-03)
    PARITY OK (global-p ATT and SE within 1e-10 of DoubleML)

The global-p-hat convention (p-hat = full-sample treated share) matches
DoubleML exactly; the fold-mean convention from the paper's proofs differs
only in finite samples. The SE parity uses the augmented score
psi_bar_i = summand_i - D_i * theta / p_hat.
"""

import numpy as np
from doubleml import DoubleMLDID
from doubleml.data import DoubleMLDIDData
from sklearn.linear_model import LinearRegression, LogisticRegression

rng = np.random.default_rng(42)
N, d = 500, 5
X = rng.standard_normal((N, d))
g0 = 1 / (1 + np.exp(-(X[:, 0] - 0.5 * X[:, 1])))
D = (rng.uniform(size=N) < g0).astype(float)
# Delta Y = ell(X) + theta*D + noise, theta = 3
ell0 = X[:, 0] + 0.5 * X[:, 2] ** 2
dY = ell0 + 3.0 * D + rng.standard_normal(N)

K = 5
TRIM = 1e-2  # match DoubleML clipping default behavior

# Fixed fold assignment shared by both implementations.
perm = rng.permutation(N)
folds = [np.sort(perm[i::K]) for i in range(K)]
smpls = [(np.setdiff1d(np.arange(N), te), te) for te in folds]  # (train, test)


def _nuisances(tr, te):
    lg = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X[tr], D[tr])
    g_hat = np.clip(lg.predict_proba(X[te])[:, 1], TRIM, 1 - TRIM)
    ctrl = tr[D[tr] == 0]  # outcome nuisance fit on untreated complement only
    lr = LinearRegression().fit(X[ctrl], dY[ctrl])
    return g_hat, lr.predict(X[te])


# --- DoubleML reference ----------------------------------------------------
data = DoubleMLDIDData.from_arrays(X, dY, D)
m = DoubleMLDID(
    data,
    ml_g=LinearRegression(),
    ml_m=LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000),
    n_folds=K,
    n_rep=1,
    score="observational",
    in_sample_normalization=False,
    clipping_threshold=TRIM,
    draw_sample_splitting=False,
)
m.set_sample_splitting([smpls])
m.fit()
att_dml, se_dml = float(m.coef[0]), float(m.se[0])

# --- hand-rolled Chang Case 1, DML2 cross-fitting --------------------------
p_glob = D.mean()  # global treated share (the library convention)
theta_k_glob = np.empty(K)
theta_k_fold = np.empty(K)
for k, (tr, te) in enumerate(smpls):
    g_hat, ell_hat = _nuisances(tr, te)
    w_num = (D[te] - g_hat) / (1 - g_hat)
    resid = dY[te] - ell_hat
    theta_k_glob[k] = np.mean(w_num * resid / p_glob)
    theta_k_fold[k] = np.mean(w_num * resid / D[te].mean())
theta_glob = theta_k_glob.mean()
theta_fold = theta_k_fold.mean()

# Variance from the augmented score psi_bar = summand - D*theta/p (global p).
sig2_k = np.empty(K)
for k, (tr, te) in enumerate(smpls):
    g_hat, ell_hat = _nuisances(tr, te)
    summand = (D[te] - g_hat) / (p_glob * (1 - g_hat)) * (dY[te] - ell_hat)
    psi_bar = summand - D[te] * theta_glob / p_glob
    sig2_k[k] = np.mean(psi_bar**2)
se_glob = np.sqrt(sig2_k.mean() / N)

print(f"DoubleML              ATT = {att_dml:.12f}   SE = {se_dml:.12f}")
print(f"hand Chang (global p) ATT = {theta_glob:.12f}   SE = {se_glob:.12f}")
print(f"  diff vs DoubleML:   ATT {theta_glob - att_dml:+.3e}   SE {se_glob - se_dml:+.3e}")
print(
    f"hand Chang (fold-mean p) ATT = {theta_fold:.12f}   "
    f"(finite-sample gap {theta_fold - theta_glob:+.3e})"
)

assert abs(theta_glob - att_dml) < 1e-10, "global-p ATT parity broken"
assert abs(se_glob - se_dml) < 1e-10, "global-p SE parity broken"
print("PARITY OK (global-p ATT and SE within 1e-10 of DoubleML)")
