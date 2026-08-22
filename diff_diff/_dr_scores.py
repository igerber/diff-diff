"""Shared doubly-robust panel DiD scores (private DML infrastructure).

Two distinct score families live here — the repo's methodology review
(``docs/methodology/papers/chang-2020-review.md``) pins that they are NOT
interchangeable:

- :func:`drdid_panel_inf_func` — the Sant'Anna & Zhao (2020) locally efficient
  doubly-robust panel influence function (``DRDID::drdid_panel$att.inf.func``
  parity), including the first-stage nuisance-correction terms.
  Algebra-preserving relocation from ``ContinuousDiD._dr_cell_inf_func``
  (fail-closed input validation added; the mathematical body is unchanged); the relocation is pinned
  by committed oracles in ``tests/test_dr_scores.py``.
- :func:`chang_panel_score` / :func:`chang_panel_score_augmented` — the
  Chang (2020) Neyman-orthogonal Case 1 (repeated outcomes) score for
  cross-fitted (DML) nuisances, normalized by the unconditional treated share
  ``p_hat``, with the augmented variant carrying the finite-dimensional
  treated-share variance correction (``G_1p = -theta/p_hat``).

References
----------
Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences
estimators. Journal of Econometrics, 219(1), 101-122.

Chang, N.-C. (2020). Double/debiased machine learning for
difference-in-differences models. The Econometrics Journal, 23(2), 177-191.
"""

from typing import Tuple

import numpy as np

from diff_diff.linalg import _rank_guarded_inv

__all__ = [
    "drdid_panel_inf_func",
    "chang_panel_score",
    "chang_panel_score_augmented",
]


def drdid_panel_inf_func(
    dY: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    gamma: np.ndarray,
    ps: np.ndarray,
) -> np.ndarray:
    """DRDID ``drdid_panel`` doubly-robust per-unit influence function.

    Direct port of ``DRDID::drdid_panel$att.inf.func`` (unit weights = 1,
    propensity already clipped so no drop-trimming). Units are ordered
    treated-then-control. Validated to ~1e-13 against DRDID in the original
    spike; the algebra-preserving relocation from
    ``ContinuousDiD._dr_cell_inf_func`` (validation added, body unchanged) is pinned
    by the committed oracles in ``tests/test_dr_scores.py``.

    Parameters
    ----------
    dY : np.ndarray
        Outcome changes, shape (n,).
    D : np.ndarray
        Treatment indicator (1 treated, 0 control), shape (n,).
    X : np.ndarray
        Covariate matrix INCLUDING the intercept column, shape (n, p).
    gamma : np.ndarray
        Outcome-regression coefficients (fit on controls), shape (p,).
    ps : np.ndarray
        Propensity scores, already clipped away from 1, shape (n,).

    Returns
    -------
    np.ndarray
        Per-unit influence function, shape (n,).
    """
    dY = np.asarray(dY, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    context = "drdid_panel_inf_func"
    if dY.ndim != 1 or D.ndim != 1 or ps.ndim != 1 or X.ndim != 2 or gamma.ndim != 1:
        raise ValueError(
            f"{context}: expected dY/D/ps 1-dimensional and X 2-dimensional with "
            "1-dimensional gamma"
        )
    n = dY.shape[0]
    if n == 0:
        raise ValueError(f"{context}: inputs are empty (n=0)")
    if D.shape[0] != n or ps.shape[0] != n or X.shape[0] != n:
        raise ValueError(f"{context}: dY/D/ps/X row counts must all equal {n}")
    if X.shape[1] != gamma.shape[0]:
        raise ValueError(
            f"{context}: X has {X.shape[1]} columns but gamma has {gamma.shape[0]} entries"
        )
    for name, arr in (("dY", dY), ("D", D), ("X", X), ("gamma", gamma), ("ps", ps)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{context}: {name} contains non-finite values")
    if not np.all((D == 0.0) | (D == 1.0)):
        raise ValueError(f"{context}: D must be strictly binary 0/1")
    if not (np.any(D == 1.0) and np.any(D == 0.0)):
        raise ValueError(
            f"{context}: both treated and control units are required "
            "(the treated- and control-weight denominators would be zero)"
        )
    if np.any(ps < 0.0) or np.any(ps >= 1.0):
        raise ValueError(
            f"{context}: ps must lie in [0, 1) strictly; clip propensity scores " "before calling"
        )
    out_delta = X @ gamma
    w_treat = D
    w_cont = ps * (1 - D) / (1 - ps)
    dr_treat = w_treat * (dY - out_delta)
    dr_cont = w_cont * (dY - out_delta)
    w_cont_mean = w_cont.mean()
    if not np.isfinite(w_cont_mean) or w_cont_mean <= 0.0:
        raise ValueError(
            f"{context}: the control-odds mass is zero (every control unit has "
            "ps == 0); the comparison population is empty and the influence "
            "function is undefined"
        )
    eta_treat = dr_treat.mean() / w_treat.mean()
    eta_cont = dr_cont.mean() / w_cont_mean
    weights_ols = 1.0 - D
    wols_eX = (weights_ols * (dY - out_delta))[:, np.newaxis] * X
    XpX = ((weights_ols[:, np.newaxis] * X).T @ X) / n
    XpX_inv, _, _ = _rank_guarded_inv(XpX)
    asy_wols = wols_eX @ XpX_inv
    W = ps * (1 - ps)
    score_ps = (D - ps)[:, np.newaxis] * X
    Hess, _, _ = _rank_guarded_inv((X.T @ (W[:, np.newaxis] * X)) / n)
    asy_ps = score_ps @ Hess
    inf_treat_1 = dr_treat - w_treat * eta_treat
    M1 = (w_treat[:, np.newaxis] * X).mean(axis=0)
    inf_treat = (inf_treat_1 - asy_wols @ M1) / w_treat.mean()
    inf_cont_1 = dr_cont - w_cont * eta_cont
    M2 = (w_cont[:, np.newaxis] * (dY - out_delta - eta_cont)[:, np.newaxis] * X).mean(axis=0)
    M3 = (w_cont[:, np.newaxis] * X).mean(axis=0)
    inf_control = (inf_cont_1 + asy_ps @ M2 - asy_wols @ M3) / w_cont_mean
    return inf_treat - inf_control


def _validate_chang_inputs(
    dY: np.ndarray,
    D: np.ndarray,
    m_hat: np.ndarray,
    ps: np.ndarray,
    p_hat: float,
    context: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared validation for the Chang Case 1 score functions."""
    dY = np.asarray(dY, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    m_hat = np.asarray(m_hat, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    for name, arr in (("dY", dY), ("D", D), ("m_hat", m_hat), ("ps", ps)):
        if arr.ndim != 1:
            raise ValueError(f"{context}: {name} must be 1-dimensional, got ndim={arr.ndim}")
    n = dY.shape[0]
    if n == 0:
        raise ValueError(f"{context}: inputs are empty (n=0); the score is undefined")
    for name, arr in (("dY", dY), ("D", D), ("m_hat", m_hat), ("ps", ps)):
        if arr.shape[0] != n:
            raise ValueError(
                f"{context}: {name} has length {arr.shape[0]}, expected {n} (length of dY)"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{context}: {name} contains non-finite values")
    if not np.all((D == 0.0) | (D == 1.0)):
        raise ValueError(f"{context}: D must be strictly binary 0/1")
    if not np.isfinite(p_hat) or not (0.0 < p_hat < 1.0):
        raise ValueError(
            f"{context}: p_hat must satisfy 0 < p_hat < 1 strictly, got {p_hat!r} "
            "(p_hat = 1 means no comparison population; the estimand is unidentified)"
        )
    if np.any(ps < 0.0) or np.any(ps >= 1.0):
        raise ValueError(
            f"{context}: ps must lie in [0, 1) strictly; clip/trim propensity "
            "scores before calling (values >= 1 would divide by zero)"
        )
    return dY, D, m_hat, ps


def chang_panel_score(
    dY: np.ndarray,
    D: np.ndarray,
    m_hat: np.ndarray,
    ps: np.ndarray,
    p_hat: float,
) -> np.ndarray:
    """Chang (2020) Case 1 per-unit UNCENTERED score summand.

    Returns ``summand_i = (D_i - ps_i (1 - D_i) / (1 - ps_i)) * (dY_i -
    m_hat_i) / p_hat``, whose sample mean is the ATT (Equation 3.1's
    ``psi_1`` equals ``summand - theta``). Nuisances ``m_hat`` (control
    outcome-change regression, cross-fitted out-of-fold) and ``ps``
    (propensity, cross-fitted, trimmed by the CALLER's policy) are plug-ins;
    no first-stage correction terms appear — under DML2 cross-fitting the
    score is Neyman-orthogonal in the infinite-dimensional nuisances.

    ``p_hat`` is caller-supplied; the library convention (see the REGISTRY
    "Cross-fitting, DR-score, and ridge infrastructure (DML)" Note) is the
    FULL-SAMPLE treated share.
    """
    dY, D, m_hat, ps = _validate_chang_inputs(dY, D, m_hat, ps, p_hat, "chang_panel_score")
    weight = (D - ps * (1 - D) / (1 - ps)) / p_hat
    return weight * (dY - m_hat)


def chang_panel_score_augmented(
    summand: np.ndarray,
    D: np.ndarray,
    theta: float,
    p_hat: float,
) -> np.ndarray:
    """Chang (2020) augmented score ``psi_bar_i = summand_i - D_i * theta / p_hat``.

    The finite-dimensional treated-share correction Chang's Theorem 2 variance
    requires (``G_1p = -theta/p_hat`` folded into the score). The variance
    estimator is ``SE = sqrt(mean(psi_bar**2) / N)``; this exact object was
    matched to DoubleML's SE at ~6e-17 in the committed parity spike
    (``benchmarks/doubleml/chang_case1_parity.py``).
    """
    summand = np.asarray(summand, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    context = "chang_panel_score_augmented"
    if summand.ndim != 1 or D.ndim != 1:
        raise ValueError(f"{context}: summand and D must be 1-dimensional")
    if summand.shape[0] == 0:
        raise ValueError(f"{context}: inputs are empty (n=0); the score is undefined")
    if summand.shape[0] != D.shape[0]:
        raise ValueError(
            f"{context}: summand has length {summand.shape[0]}, D has length {D.shape[0]}"
        )
    if not np.all(np.isfinite(summand)):
        raise ValueError(f"{context}: summand contains non-finite values")
    if not np.all((D == 0.0) | (D == 1.0)):
        raise ValueError(f"{context}: D must be strictly binary 0/1")
    if not np.isfinite(theta):
        raise ValueError(f"{context}: theta must be finite, got {theta!r}")
    if not np.isfinite(p_hat) or not (0.0 < p_hat < 1.0):
        raise ValueError(
            f"{context}: p_hat must satisfy 0 < p_hat < 1 strictly, got {p_hat!r} "
            "(p_hat = 1 means no comparison population; the estimand is unidentified)"
        )
    return summand - D * theta / p_hat
